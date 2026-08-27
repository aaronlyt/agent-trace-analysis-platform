"""do-then-verify recovery -- DoVer, arXiv:2512.06749 (ICLR'26) §4.

Mechanism (pipeline from the paper):
* **Trial Segmenter** (Fig 5): classify the full log into initial/update planning
  and execution -- a trial = a planning step + the execution steps after it,
  with re-plans as split points;
* **Trial Summarizer + Failure Proposer** (Fig 6/7): each trial outputs
  {mistake_agent, mistake_step_index, mistake_reason} (an improvement over
  all-at-once, locating the earliest erroneous step);
* **Intervention Recommender** (Fig 8): strict JSON -- category∈
  {orchestrator_ledger (minimal FACTS/PLAN_REPLACEMENT fragment, does not
  rewrite the whole ledger), orchestrator_instruction (correct to a single
  atomic next step), subagent_instruction (rewrite the orchestrator's
  instruction to that sub-agent)} + replacement_text; "Keep changes minimal
  and targeted. Avoid global resets"; "Do not give any ground truth in the
  intervention message";
* **Intervention Execution**: load the checkpoint of that step, **replace the
  message in place**, resume from the intervention step to the end, repeating
  each intervention 3 times (Sec 4.2);
* **Milestone Extractor/Evaluator** (Fig 9/10): K≤5 milestones; progress
  Prog=(A(τ̃)−A(τ))/K; achieved/partial/missed + new_path evaluation;
* **Outcome Classifier** (Fig 11): Validated (≥2/3 runs succeed) /
  Partially (<2/3 success and ≥2/3 faithful execution and progress ≥20%) /
  Refuted (faithful execution yet no progress -- hypothesis invalid) /
  Inconclusive.

Essential difference from targeted_rerun (stated in this docstring, measured
in paper Sec 5.3): in-place message **replacement + outcome diff** (milestone
progress) vs appending a feedback message + external verification -- the
CRITIC-style append-feedback baseline flips 0% vs DoVer 17.6% (WW-GAIA).

[adaptation] Sandbox mapping -- orchestrator→planner, subagent→searcher/
reporter; split points = the planner's plan/re-plan messages (deterministically
identified by the pseudo-judge; sandbox trajectories have T=1, the re-plan
scenario is left for multi-plan trajectories); interventions go through
``env.replay_intervene(step, edit, n_repeats=3)`` (in-place message
replacement middleware); milestones are rule-generated from the task +
verifier criteria (the paper leaves the case without human solution steps as
future work -- the sandbox generates them by construction: search→read→
correct answer with citations, K=3). Further statements: (1) the failure
proposer input is **sliced per Fig 6**: it sees only the failing trial's
log (events inside the trial's exec_range, original [index] numbering
preserved) plus a ``previous_trial_summary`` field (empty for the first
trial; per-trial plan/execution text is not summarized separately -- one
proposer call does summary + attribution together [adaptation]; for sandbox
T=1 trajectories the slice is the whole session minus the env bookkeeping
TASK_START event, so behavior is unchanged), while the intervention
recommender input remains the full trajectory (the original Fig 8 context =
the two steps before the failing step + the failing step -- the
pseudo-judge needs the full symptom context to locate deterministically);
(2) the
Milestone Evaluator (the A(γ)/Prog values of Fig 10) is not implemented --
the verdict is reduced from the fault_removed environment response signal
(the "progress ≥20%" criterion text above does not apply on the offline
path; only the original wording of the criterion is retained); (3) the
recovery criterion adds runs[-1].success on top of the paper's "run succeeds
after intervention" and also admits Partially [adaptation: under
deterministic ×3 the three replays are equivalent, last success ⟺ majority
success]. LLM calls/trajectories: 1(segment)+T(proposer)+T(intervene)+
1(milestone)+1(classifier). anti-leak: the edit text is consumed by
environment-side middleware (contains fault naming, sandbox response
channel) and is stripped of fault names via ``_redact_fault_names`` before
being echoed into the classify prompt.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from atap.core.registry import register
from atap.core.render import judge_view
from atap.recover.base import Recoverer

# The edit text is consumed by environment-side middleware keyed on fault
# name (sandbox response channel), but fault names must be stripped before
# it is echoed into the judge prompt (anti-leak iron rule: the judge prompt
# never sees fault types). Separator-normalized matching: the middleware
# accepts space-separated variants ("info withholding"), so the redactor
# must strip every separator variant ([-_\s]+) of each fault name --
# otherwise a space-variant edit is consumed by the environment yet echoed
# verbatim into the classify prompt.
_FAULT_NAME_RE = re.compile(
    r"malformed[-_\s]+tool[-_\s]+call|step[-_\s]+repetition"
    r"|info[-_\s]+withholding|premature[-_\s]+termination"
    r"|ungrounded[-_\s]+citation|disobey[-_\s]+task[-_\s]+spec"
    r"|retrieval[-_\s]+detour|agent[-_\s]+deadlock",
    re.I,
)


def _redact_fault_names(text: str) -> str:
    return _FAULT_NAME_RE.sub("<edited>", text)


def _exec_range(trial: dict, last: int) -> tuple[int, int]:
    """The trial's [start, end] event-index range, clamped to the stream and
    defensive against malformed segmenter output (fallback: whole trial
    span)."""
    rng = trial.get("exec_range") or [0, last]
    try:
        lo, hi = int(rng[0]), int(rng[1])
    except (TypeError, ValueError, IndexError):
        lo, hi = 0, last
    return max(0, lo), min(hi, last)


def _slice_view(view: str, lo: int, hi: int) -> str:
    """Slice a rendered judge view to the [lo, hi] event-index range (Fig 6
    ``trial_logs_to_summarize``: only the trial's own steps reach the
    failure proposer). Original ``[index]`` line numbers are preserved so
    the proposer's ``mistake_step_index`` still references the full session
    stream; header lines (task/outcome/trace markers) carry no ``[n]``
    prefix and are kept."""
    out: list[str] = []
    for line in view.splitlines():
        m = re.match(r"\[(\d+)\]", line)
        if m and not (lo <= int(m.group(1)) <= hi):
            continue
        out.append(line)
    return "\n".join(out)


class TrialSplit(BaseModel):
    trials: list[dict] = Field(
        default_factory=list,
        description="[{trial_index, plan_step, exec_range: [start, end]}]",
    )


class TrialFailure(BaseModel):
    is_succeed: bool
    mistake_agent: str | None = None
    mistake_step_index: int | None = Field(default=None, ge=0)
    mistake_reason: str = ""


class Intervention(BaseModel):
    category: Literal[
        "orchestrator_ledger", "orchestrator_instruction", "subagent_instruction"
    ] = Field(
        description="orchestrator_ledger | orchestrator_instruction | "
                    "subagent_instruction"
    )
    replacement_text: str = Field(description="Minimal edit text that replaces the step's message in place")
    rationale: str = ""

    @field_validator("category", mode="before")
    @classmethod
    def _norm_category(cls, v):
        # Normalize common judge spellings (case-insensitive, hyphen/space
        # variants folded to the underscore token); any other value is an
        # explicit parse failure (LLMError upstream), never silently trusted
        # -- same policy as judge_eval's severity handling.
        token = re.sub(r"[\s\-]+", "_", str(v).strip().lower())
        return {
            "orchestrator_ledger": "orchestrator_ledger",
            "orchestrator_instruction": "orchestrator_instruction",
            "subagent_instruction": "subagent_instruction",
        }.get(token, v)


class Milestones(BaseModel):
    milestones: list[str] = Field(
        default_factory=list, description="K≤5 decidable milestones (in order)"
    )


# Normalize common judge synonyms (any other value is explicitly rejected by
# Literal validation -- same policy as judge_eval's severity handling)
_OUTCOME_LABEL_ALIAS = {
    "validated": "Validated",
    "partially": "Partially",
    "partially valid": "Partially",
    "partially validated": "Partially",
    "partially_valided": "Partially",
    "refuted": "Refuted",
    "inconclusive": "Inconclusive",
}


class OutcomeLabel(BaseModel):
    label: Literal["Validated", "Partially", "Refuted", "Inconclusive"] = Field(
        description="Validated | Partially | Refuted | Inconclusive"
    )
    reason: str

    @field_validator("label", mode="before")
    @classmethod
    def _norm_label(cls, v):
        return _OUTCOME_LABEL_ALIAS.get(str(v).strip().lower(), v)


_SEGMENT_SYSTEM = (
    "You are a trial segmenter. Split the trajectory into trials at planning "
    "events: each trial = one planning (or re-planning) step + the execution "
    "steps after it; the planning step is the split point. Output the list of "
    "trials (plan_step and the execution range). Output JSON."
)
_PROPOSER_SYSTEM = (
    "You are a trial failure proposer. Given the task, the summary of the "
    "previous trials in the same session (previous_trial_summary; an empty "
    "list for the first trial) and the log of one trial "
    "(trial_logs_to_summarize), decide whether the trial failed; when it "
    "failed, give the mistaken agent, the earliest erroneous step "
    "(mistake_step_index, line-start [index] as numbered in the session "
    "log) and the reason. Output JSON."
)
_INTERVENE_SYSTEM = (
    "You are an intervention recommender. Given the task, the failing-step "
    "diagnosis and context, design a **minimal** message intervention: "
    "category∈{orchestrator_ledger (replace a minimal fragment of the "
    "plan/facts ledger), orchestrator_instruction (correct to a single "
    "atomic next step), subagent_instruction (rewrite the instruction given "
    "to the sub-agent)} plus replacement_text (replaces that step's message "
    "in place). Keep changes minimal and avoid global resets; do not include "
    "any answer content in the intervention message. Output JSON."
)
_MILESTONE_SYSTEM = (
    "You are a milestone extractor. Given the task, produce K≤5 ordered "
    "milestones that can be decided from the trajectory (abstracted to the "
    "outcome level, without concrete tool operations). Output JSON."
)
# Template: the majority threshold and repeat count follow the configured
# n_repeats (strict majority > n_repeats//2; for the default n=3 this is the
# paper's >=2/3) -- never a hardcoded "3 repeats / >=2/3" when n_repeats is
# configurable. [adaptation] the input's milestone part is a list to check
# progress against, not measured progress values (the paper's Prog values are
# not implemented offline, see the module docstring).
_CLASSIFY_SYSTEM = (
    "You are an intervention outcome classifier. Given the original "
    "trajectory, intervention details and replay results (the success/"
    "failure list of the {n_repeats} repeats and the milestone list), "
    "classify: Validated (more than {maj} of the {n_repeats} repeats "
    "succeed) / Partially (faithful execution and progress >= 20%) / "
    "Refuted (faithful execution yet no measurable progress toward the "
    "milestones) / Inconclusive. Output JSON."
)


@register
class DoVerRecoverer(Recoverer):
    stage = "recover"
    name = "dover"

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        if not t.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure "
                f"canonical_events first"
            )
        if bundle.succeeded:
            # recovered stays False: nothing was broken, so nothing was
            # recovered (same semantics as targeted_rerun's silent skip)
            bundle.put("recover", self.name, {"status": "skipped_success",
                                              "recovered": False})
            return
        if ctx.llm is None:
            raise RuntimeError("dover requires an LLM client (RunContext.llm)")
        replay = getattr(ctx.env, "replay_intervene", None) if ctx.env else None
        if replay is None:
            bundle.put(
                "recover", self.name,
                {"status": "no_replay_environment", "recovered": False,
                 "note": "dover requires a message-intervention replay "
                         "environment (env.replay_intervene)"},
            )
            return

        view = judge_view(bundle)
        n_repeats = int(self.param("n_repeats", 3))
        if n_repeats < 1:
            raise ValueError(
                f"{bundle.trace_id}: dover n_repeats must be >= 1 "
                f"(got {n_repeats})"
            )

        # ---- ① Trial Segmenter ----
        r1 = ctx.llm.complete(
            [
                {"role": "system", "content": _SEGMENT_SYSTEM},
                {"role": "user", "content": f"Task: {t.task}\n{view}"},
            ],
            schema=TrialSplit,
            tag=f"{self.name}_segment",
        )
        split = r1.parsed
        assert isinstance(split, TrialSplit)
        trials = split.trials or [{"trial_index": 0, "plan_step": 0,
                                   "exec_range": [0, len(t.events) - 1]}]

        # ---- ⑤ Milestone Extractor (once per task, before the loop) ----
        r5 = ctx.llm.complete(
            [
                {"role": "system", "content": _MILESTONE_SYSTEM},
                {"role": "user", "content": f"Task: {t.task}"},
            ],
            schema=Milestones,
            tag=f"{self.name}_milestone",
        )
        miles = r5.parsed
        assert isinstance(miles, Milestones)
        milestones = miles.milestones[:5]

        attempts: list[dict] = []
        recovered = False
        # Fig 6 input structure: each proposer call sees previous_trial_summary
        # (the proposer records of all earlier trials; empty for the first
        # trial -- the paper's list carries trial_index/is_succeed/
        # trial_summary; the per-trial plan/execution text is not summarized
        # separately in this pipeline [adaptation: one proposer call does
        # summary + attribution together]) + this trial's sliced log.
        prev_summaries: list[dict] = []
        for trial in trials:
            # ---- ② Summarizer + Failure Proposer ----
            lo, hi = _exec_range(trial, len(t.events) - 1)
            trial_view = _slice_view(view, lo, hi)
            r2 = ctx.llm.complete(
                [
                    {"role": "system", "content": _PROPOSER_SYSTEM},
                    {"role": "user", "content": (
                        f"Task: {t.task}\ntrial {trial.get('trial_index', 0)} "
                        f"(range {trial.get('exec_range')}):\n"
                        f"previous_trial_summary: "
                        f"{json.dumps(prev_summaries, ensure_ascii=False)}\n"
                        f"trial_logs_to_summarize:\n{trial_view}"
                    )},
                ],
                schema=TrialFailure,
                tag=f"{self.name}_proposer",
            )
            failure = r2.parsed
            assert isinstance(failure, TrialFailure)
            # this trial's record for the next trial's proposer prompt
            # (redacted defensively: judge-visible text never carries fault
            # names, even via the summary chain)
            prev_summaries.append({
                "trial_index": trial.get("trial_index", 0),
                "is_succeed": bool(failure.is_succeed),
                "trial_summary": _redact_fault_names(
                    "succeeded"
                    if failure.is_succeed else
                    f"failed; mistake agent={failure.mistake_agent}, "
                    f"step={failure.mistake_step_index}, "
                    f"reason={failure.mistake_reason[:120]}"
                ),
            })
            if failure.is_succeed or failure.mistake_step_index is None:
                attempts.append({
                    "trial": trial.get("trial_index", 0), "verdict": "succeed_trial",
                })
                continue

            # ---- ③ Intervention Recommender ----
            r3 = ctx.llm.complete(
                [
                    {"role": "system", "content": _INTERVENE_SYSTEM},
                    {"role": "user", "content": (
                        f"Task: {t.task}\nFailure diagnosis: agent="
                        f"{failure.mistake_agent}, step="
                        f"{failure.mistake_step_index}, reason="
                        f"{failure.mistake_reason}\nFull trajectory:\n{view}"
                    )},
                ],
                schema=Intervention,
                tag=f"{self.name}_intervene",
            )
            itv = r3.parsed
            assert isinstance(itv, Intervention)

            # ---- ④ checkpoint replay x n_repeats (in-place message replacement, run to the end) ----
            runs = replay(
                t, int(failure.mistake_step_index), itv.replacement_text,
                n_repeats=n_repeats,
            )
            for r in runs:
                bundle.reruns.append(r)
            n_ok = sum(1 for r in runs if r.outcome.success)
            removed = bool(runs[0].meta.get("fault_removed"))

            # ---- ⑥ Outcome Classifier ----
            classify_system = _CLASSIFY_SYSTEM.format(
                n_repeats=n_repeats, maj=n_repeats // 2
            )
            r6 = ctx.llm.complete(
                [
                    {"role": "system", "content": classify_system},
                    {"role": "user", "content": (
                        f"Task: {t.task}\nOriginal trajectory outcome: "
                        f"{t.outcome.note[:120]}\nIntervention: category="
                        f"{itv.category}, replacement="
                        f"{_redact_fault_names(itv.replacement_text[:120])}\n"
                        f"{n_repeats} replay outcomes: "
                        f"{[r.outcome.success for r in runs]}; "
                        f"fault removed by edit: {removed}\nMilestones: {milestones}"
                    )},
                ],
                schema=OutcomeLabel,
                tag=f"{self.name}_classify",
            )
            label = r6.parsed
            assert isinstance(label, OutcomeLabel)
            attempts.append({
                "trial": trial.get("trial_index", 0),
                "mistake": {
                    "agent": failure.mistake_agent,
                    "step": failure.mistake_step_index,
                    "reason": failure.mistake_reason[:120],
                },
                "intervention": {
                    "category": itv.category,
                    "replacement": itv.replacement_text[:120],
                },
                "replay_success": n_ok,
                "fault_removed": removed,
                "verdict": label.label,
                "verdict_reason": label.reason[:160],
            })
            if label.label in ("Validated", "Partially") and runs and runs[-1].outcome.success:
                recovered = True

        bundle.put(
            "recover",
            self.name,
            {
                "status": "ok",
                "recovered": recovered,
                "attempts": attempts,
                "milestones": milestones,
                "n_llm_calls_per_trial": 3,
                "mechanism": "message_replace_inplace + outcome diff "
                             "(vs append-feedback of targeted_rerun)",
            },
        )
