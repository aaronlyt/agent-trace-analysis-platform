"""Tree-index diagnosis — CodeTracer, arXiv:2604.11641 §3.2 Diagnosis.

Mechanism (paper Appendix D three-step workflow): the judge first reads the
``tree.md`` compressed index to locate suspicious regions (loops/stagnation/
erroneous commitments) → maps them to precise step ranges via stage_ranges →
**inspects only the specific steps in that region on demand** ("Do NOT
iterate over all steps"). Paper ablation: tree index +18.3pt while tokens
actually drop (105.1k→56.8k).

[adaptation] The paper's judge is a terminal-command-style agent (INSPECT/
WRITE/FINALIZE command discipline, anti-cheating constraints: every marked
step must have been inspected, a suspicious stage must be probed with at
least 2 steps including a neighbor); the atap judge is a stateless JSON
call — the "tree first, then drill" workflow is approximated with two
structured calls: (1) tree-level localization (input tree.md, output a
ranking of suspicious stages) (2) range drill-down (render the chosen
stage's events + the tail of the previous stage as neighbor context, output
the attribution). What remains of the anti-cheating spirit: the drill-down
render must genuinely contain the attributed step (guaranteed by the module,
not by judge self-discipline — the verdict step is clamped into the rendered
interval [lo, stage_end], and every clamp of step/agent/failure_mode leaves
a note in the Hypothesis evidence — all_at_once's clamp-with-trace
discipline). Stage-span selection when one stage name maps to
several ranges (paper appendix A: a trajectory may revisit a stage, and each
contiguous block gets its own span): prefer the span **containing the
failure event index** — the failure index is derived deterministically from
judge-visible data only (the first event whose content is a structural
error observation, else the terminal event); when no same-name span contains
it (e.g. the failing VERIFIER carries no phase and falls outside every
span), take the **last** same-name span (the latest revisit of that stage is
closest to the failure) [inference]. Further disclaimers: (1) the paper's §B stage
ranking features (verification regression / diff magnitude / backtrack
frequency / exploration-to-action ratio) are not implemented —
suspiciousness is judged by the judge from in-line signals in tree.md
[adaptation]; (2) output contraction: the paper outputs suspicious stage ŝ +
step set P (anti-cheating requires covering ≥3 stages, allowing multiple
incorrect/unuseful steps), whereas this implementation drills into the top-1
stage and outputs a single-step Hypothesis (atap's unified output contract)
[weakened claim]; (3) when the judge outputs no valid stage name, the last
stage (the report segment) is the fallback [inference].

Consumes the represent/hierarchy_tree artifact (by name, no import). The
artifact records n_events_inspected (evidence of token savings versus a
full render).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.classify.taxonomy import MAST_MODES, mast_definitions_block
from atap.core.registry import register
from atap.core.render import (
    TRACE_BEGIN,
    TRACE_END,
    is_error_observation,
    render_event_line,
    render_trace,
)
from atap.core.schema import Hypothesis


class StagePick(BaseModel):
    suspicious_stages: list[str] = Field(
        default_factory=list, description="Suspicious stage names, in descending order of suspiciousness"
    )
    reason: str = ""


class TreeDrillVerdict(BaseModel):
    responsible_agent: str
    step: int = Field(ge=0, description="Decisive error step (the [index] at the start of rendered lines)")
    reason: str
    fix_suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str | None = None


_STAGE_SYSTEM = (
    "You are an agent trajectory diagnosis expert. Given the hierarchical tree "
    "compressed index of the task (tree.md: one step per line with its "
    "intent/result summary, indentation encodes hierarchy, == stage == marks "
    "stage boundaries), locate the most suspicious stages (error signals in "
    "result summaries, unproductive repetition, anomalous submissions, etc.) "
    "and list the stage names in descending order of suspiciousness. Judge "
    "only from the in-line information on the tree. Output JSON."
)
_DRILL_SYSTEM = (
    "You are an expert in multi-agent system failure attribution. Given the "
    "task and the full event lines of the chosen stage (with neighboring "
    "context), determine the decisive error step (the earliest decisive "
    "error, not the step where the symptom manifests), the responsible "
    "agent, the reason, and a fix suggestion. You may consult the MAST "
    "failure mode codes:\n{definitions}"
)


@register
class TreeDiagnosisAttributor(Attributor):
    stage = "attribute"
    name = "tree_diagnosis"

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        if not t.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        tree = bundle.get("represent", "hierarchy_tree")
        if not (isinstance(tree, dict) and tree.get("tree_md")):
            raise ValueError(
                f"{bundle.trace_id} is missing the represent/hierarchy_tree artifact: "
                "tree_diagnosis consumes the R4 hierarchy tree; configure hierarchy_tree first"
            )
        if bundle.succeeded and not self.param("include_success", False):
            bundle.put(
                "attribute", self.name,
                {"hypotheses": [], "status": "success_no_attribution"},
            )
            return
        if ctx.llm is None:
            raise RuntimeError("tree_diagnosis requires an LLM client (RunContext.llm)")

        # ---- call 1: tree-level localization ----
        result = ctx.llm.complete(
            [
                {"role": "system", "content": _STAGE_SYSTEM},
                {"role": "user", "content": (
                    f"task: {t.task}\n\n{tree['tree_md']}"
                )},
            ],
            schema=StagePick,
            tag=f"{self.name}_stage",
        )
        pick = result.parsed
        assert isinstance(pick, StagePick)

        stage_names = [r["stage"] for r in tree["stage_ranges"]]
        chosen = next(
            (s for s in pick.suspicious_stages if s in stage_names),
            stage_names[-1] if stage_names else None,
        )
        # A revisited stage yields several same-name spans (paper appendix A:
        # each contiguous block is its own span). Selection rule [inference]:
        # prefer the span containing the failure event index (first structural
        # error observation, else the terminal event — judge-visible data
        # only, no ground truth); fall back to the last same-name span.
        same_name = [r for r in tree["stage_ranges"] if r["stage"] == chosen]
        fail_idx = next(
            (
                ev.index
                for ev in t.events
                if is_error_observation(str(ev.payload.get("content", "")))
            ),
            t.events[-1].index,
        )
        rng = next(
            (r for r in same_name if r["start"] <= fail_idx <= r["end"]),
            same_name[-1] if same_name else None,
        )
        if rng is None:
            bundle.put(
                "attribute", self.name,
                {
                    "hypotheses": [],
                    "status": "stage_not_resolved",
                    "stage_pick": pick.model_dump(),
                },
            )
            return

        # ---- drill-down render: the chosen stage's events + the previous
        # stage's tail (neighbor context, aligned with the paper's "probe a
        # suspicious stage with at least 2 steps including a neighbor") ----
        lo = max(0, rng["start"] - self.param("context_events", 3))
        seg_events = [ev for ev in t.events if lo <= ev.index <= rng["end"]]
        seg_lines = [render_event_line(ev) for ev in seg_events]
        drill_view = f"{TRACE_BEGIN}\n" + "\n".join(seg_lines) + f"\n{TRACE_END}"

        result2 = ctx.llm.complete(
            [
                {"role": "system", "content": _DRILL_SYSTEM.format(
                    definitions=mast_definitions_block()
                )},
                {"role": "user", "content": (
                    f"Task: {t.task}\nChosen suspicious stage {chosen} (with "
                    f"neighboring context; agent roster: {', '.join(t.agents())}):\n{drill_view}"
                )},
            ],
            schema=TreeDrillVerdict,
            tag=f"{self.name}_drill",
        )
        v = result2.parsed
        assert isinstance(v, TreeDrillVerdict)
        # Anti-cheating guarantee (docstring): the attributed step must lie
        # inside the rendered drill view [lo, rng.end] — clamp the judge's
        # verdict into that interval instead of the whole trajectory. Every
        # clamp (step / agent / failure_mode) leaves a note in the evidence,
        # same discipline as all_at_once (nothing silently rewritten).
        step = min(max(v.step, lo), rng["end"])
        responsible = (
            v.responsible_agent
            if v.responsible_agent in t.agents() else t.agents()[0]
        )
        code = v.failure_mode if v.failure_mode in MAST_MODES else None
        clamp_notes: list[str] = []
        if v.step != step:
            clamp_notes.append(f"(judgement clamped: step {v.step}->{step})")
        if v.responsible_agent != responsible:
            clamp_notes.append(
                f"(judgement clamped: agent {v.responsible_agent!r}->"
                f"{responsible!r})"
            )
        if v.failure_mode is not None and code is None:
            clamp_notes.append(
                f"(judgement clamped: failure_mode {v.failure_mode!r}->None: "
                f"not a MAST code)"
            )
        ev = t.events[step]

        hyp = Hypothesis(
            agent=responsible,
            step=step,
            root_cause=v.reason,
            root_cause_code=code,
            responsible_side="model",
            evidence=[
                f"[{ev.index}] {ev.agent} {ev.kind} :: "
                f"{str(ev.payload.get('content', ev.payload))[:140]}",
                f"stage_pick={pick.suspicious_stages} chosen={chosen}",
            ],
            fix_suggestion=v.fix_suggestion,
            confidence=v.confidence,
        )
        hyp.evidence.extend(clamp_notes)
        full_lines = len(render_trace(t).splitlines())
        bundle.put(
            "attribute",
            self.name,
            {
                "hypotheses": [hyp.to_dict()],
                "stage_pick": pick.model_dump(),
                "chosen_stage": chosen,
                "stage_range": rng,
                "n_events_inspected": len(seg_events),
                "full_render_lines": full_lines,
                "inspected_render_lines": len(drill_view.splitlines()),
            },
        )
