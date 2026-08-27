"""Hierarchical causal graph attribution — CHIEF, arXiv:2602.23701 §4.2-4.3.

Mechanism (from the paper):
* **oracle synthesis** (§4.2.1, Eq. 2): no human annotation — an LLM
  synthesizes, in order, a virtual oracle for each subtask O_k=⟨G_sub goal,
  P_pre preconditions, E_key key evidence, C_acc acceptance criteria⟩ (later
  oracles depend on earlier ones, with a global self-check);
* **oracle-guided backtracking** (§4.2.2): top-down three-level pruning —
  subtask-level **reverse topological order** traversal (F_eval compares
  actual output against G_sub/C_acc; correct subtasks skip their whole branch)
  → agent level (OTAR checked against P_pre/E_key) → step level;
* **progressive causal filtering** (§4.3, four steps): Local Attribution
  (Eq. 6: if Bias→Anomaly(x) is non-empty in Pre(x), propagate upstream,
  otherwise the error was locally generated) / Planning-Control (repeated
  steps aggregated into a Loop Group: if the planner still issues the same
  token after receiving the repetition signal → planner; if the executor keeps
  misbehaving → executor) / Data-Flow (follow step edges to locate the step
  where a valid input was **first polluted**: the generator = root cause) /
  Deviation-Aware (reversible deviations that later self-heal bear no blame);
* outputs (i*, t*) = the temporally earliest decisive error (Eq. 1), following
  the Who&When protocol. The paper does no real replay — counterfactuals are
  audited approximately on the graph (§3 only defines the problem).

[adaptation] Toy-domain implementation: oracle synthesis / subtask F_eval /
localization filtering are each compressed into a single LLM call (3 in total;
the paper's decomposition + alignment + OTAR + edges + oracle + backtracking +
filtering ≈5+K calls — graph construction is already done deterministically by
represent/hcg); within the three-level backtracking, the agent level and step
level are merged into the localization call (sandbox has ≤3 agents); the
mechanism field carries the progressive-filtering conclusion
(local_error/upstream_propagation/executor_loop/planning_error/
dataflow_first_pollution — the vocabulary is **clamped**: a free-string
mechanism outside MECHANISMS is mapped to the closest vocabulary word
(difflib, else "unknown") with the raw value kept in evidence and the
artifact's ``mechanism_clamped`` entry — same discipline as all_at_once's
failure_mode handling — and **upstream_propagation is an extension beyond the
paper's four-stage screening**: §4.3 defines only Local / Planning-Control /
Data-Flow / Deviation-Aware four stages; when S_cause≠∅ the paper proceeds to
the Planning-Control/Data-Flow stages rather than emitting a separate
"upstream" conclusion [adaptation]). Further
disclaimers: (1) Deviation-Aware (§4.3.4) is not implemented — the sandbox's
six faults have no self-healing/reversible-deviation scenario, so there is no
corresponding branch; (2) the backtracking start point is a single "deepest
failing subtask" (the paper's Eq. 3 collects the full set of failing subtasks
C_sub and then traverses in reverse topological order); (3) oracle input =
task + decomposition + full trajectory (the paper's Eq. 2 only sees τ>k and
O<k; the oracle never sees the execution of the subtask being evaluated
itself); (4) none of the three prompts takes ground_truth as input (the
prompts in the paper's Fig.5/7/9-11 contain {ground_truth}, with w/G as the
main setting) — equivalent to the paper's w/o G setting [adaptation:
anti-leak]; (5) the merged localization call receives the **full trajectory
view** with the failing subtask as a one-line hint — M2's search-space
pruning is nominal here: nothing is removed from the judge's input (no token
saving either), and the paper's Eq.4/Eq.5 scoping (agent candidates
restricted to C_sub, step candidates to C_agt) is not reproduced; full-graph
input is defensible only for the paper's Fig.13 filtering prompt (M3), which
this call also subsumes; (6) **the HCG graph structure never enters any
judge prompt** [adaptation: mechanism gap] — none of the three prompts takes
a {graph} input: E_step/E_agt/OTAR slots/Φ produced by represent/hcg have
**zero consumers** in the whole pipeline, and HCG's actual consumption is
reduced to the subtask interval segmentation used to seed the failing
subtask; the paper's Fig.11 hierarchical-backtracking prompt explicitly
contains the causal-graph {graph} input (its subtask/agent/step traversal
follows the graph's edges), which is not reproduced. Reference from the
literature (w/o G column, matching this implementation's anti-leak setting):
algorithm-generated subset step **45.60** — the highest among the same-column
baselines in Table 1 (the headline **52.00** is the w/ G setting, which this
implementation does not reproduce); token 2.5-3× (w/ G measured, Table 2).
"""

from __future__ import annotations

import difflib

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.render import judge_view
from atap.core.schema import Hypothesis

MECHANISMS = (
    "local_error",              # Local Attribution: no upstream bias, the error was locally generated
    "upstream_propagation",     # upstream propagation (Bias→Anomaly)
    "executor_loop",            # Planning-Control: executor repeats the misbehavior
    "planning_error",           # Planning-Control: the plan itself deviates
    "dataflow_first_pollution",  # Data-Flow: first-pollution point
)


class SubtaskOracle(BaseModel):
    subtask_id: str
    goal: str
    preconditions: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    acceptance: str


class OracleSet(BaseModel):
    oracles: list[SubtaskOracle] = Field(default_factory=list)


class SubtaskEval(BaseModel):
    subtask_id: str
    passed: bool
    evidence: str = ""


class SubtaskEvals(BaseModel):
    evals: list[SubtaskEval] = Field(default_factory=list)


class ChiefVerdict(BaseModel):
    responsible_agent: str
    step: int = Field(ge=0, description="Decisive error step (the [index] at the start of rendered lines)")
    mechanism: str = Field(description=f"one of {MECHANISMS}")
    reason: str
    fix_suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)


_ORACLE_SYSTEM = (
    "You are a task oracle synthesizer. Given a task and its subtask "
    "decomposition, synthesize acceptance criteria for each subtask in order: "
    "goal (what this subtask should accomplish), preconditions (what must "
    "already hold before it starts), key_evidence (what information needs to "
    "be obtained/verified), acceptance (an acceptance condition that can decide "
    "success or failure, phrased only in terms of observable behavior in the "
    "trajectory). Later subtasks may reference outputs of earlier subtasks. "
    "Output JSON."
)
_EVAL_SYSTEM = (
    "You are a subtask evaluator. Given each subtask's oracle acceptance "
    "criteria and the full trajectory, judge subtask by subtask whether the "
    "actual execution satisfies the acceptance (passed). Rely only on "
    "in-trajectory line evidence. Output JSON."
)
_LOCALIZE_SYSTEM = (
    "You are a hierarchical causal graph localizer. Given the events of the "
    "failing subtask interval (with upstream context) and the task, apply "
    "progressive causal filtering and provide: (1) the responsible agent and "
    "the decisive error step (the earliest decisive error); (2) the mechanism -- "
    "local_error (the agent's own behavior is wrong, with no bias in upstream "
    "inputs) / upstream_propagation (the error came from upstream "
    "transmission; give the first upstream pollution step) / executor_loop "
    "(the agent unproductively repeats the same action; take the second "
    "occurrence, i.e., the first repetition) / planning_error (the planning "
    "step itself deviates from the task requirements) / "
    "dataflow_first_pollution (the step where the erroneous information first "
    "entered the data flow); (3) the reason and a fix suggestion. Output JSON."
)


@register
class ChiefAttributor(Attributor):
    stage = "attribute"
    name = "chief"
    requires = (("represent", "hcg"),)   # consumes the hierarchical causal graph

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        if not t.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        hcg = bundle.get("represent", "hcg")
        if not (isinstance(hcg, dict) and hcg.get("subtasks")):
            raise ValueError(
                f"{bundle.trace_id} is missing the represent/hcg artifact: chief "
                "consumes the hierarchical causal graph; configure hcg first"
            )
        if bundle.succeeded and not self.param("include_success", False):
            bundle.put(
                "attribute", self.name,
                {"hypotheses": [], "status": "success_no_attribution"},
            )
            return
        if ctx.llm is None:
            raise RuntimeError("chief requires an LLM client (RunContext.llm)")

        subtasks = hcg["subtasks"]
        view = judge_view(bundle)

        # ---- (1) oracle synthesis (sequential dependency: generated in order within one call) ----
        sub_json = "\n".join(
            f"{s['id']} (phase={s['phase']}, steps [{s['start']}..{s['end']}], "
            f"agents={s['agents']})" for s in subtasks
        )
        r1 = ctx.llm.complete(
            [
                {"role": "system", "content": _ORACLE_SYSTEM},
                {"role": "user", "content": (
                    f"Task: {t.task}\n\nSubtask decomposition:\n{sub_json}\n\nTrajectory:\n{view}"
                )},
            ],
            schema=OracleSet,
            tag=f"{self.name}_oracle",
        )
        oracles = r1.parsed
        assert isinstance(oracles, OracleSet)
        oracle_by_id = {o.subtask_id: o for o in oracles.oracles}

        # ---- (2) subtask F_eval → take the deepest failing subtask in reverse topological order ----
        oracle_json = "\n".join(
            f"{s['id']} (phase={s['phase']}, steps [{s['start']}..{s['end']}]): "
            + (f"goal={o.goal}; acceptance={o.acceptance}"
               if (o := oracle_by_id.get(s["id"])) else "(no oracle)")
            for s in subtasks
        )
        r2 = ctx.llm.complete(
            [
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user", "content": (
                    f"Subtask oracles:\n{oracle_json}\n\nTask and trajectory:\n{view}"
                )},
            ],
            schema=SubtaskEvals,
            tag=f"{self.name}_eval",
        )
        evals = r2.parsed
        assert isinstance(evals, SubtaskEvals)
        passed_by_id = {e.subtask_id: e.passed for e in evals.evals}

        # reverse topological order (subtask order is the topological order):
        # the deepest failing subtask = backtracking start point
        failing = next(
            (s for s in reversed(subtasks) if not passed_by_id.get(s["id"], True)),
            None,
        )
        if failing is None:
            # all subtasks "pass" yet the task fails: fall back to whole-trajectory
            # localization (a failure Deviation-Aware cannot explain; recorded
            # faithfully)
            failing = subtasks[-1]
            fallback_note = "All subtasks passed acceptance yet the task failed — falling back to localization on the last subtask"
        else:
            fallback_note = ""

        # ---- (3) localization (progressive filtering: failing subtask + full
        # trajectory context — Planning-Control's Loop Group aggregation and
        # Data-Flow backtracking can both cross subtask boundaries; the paper's
        # Fig.13 integrated filtering prompt also includes the full graph
        # view) ----
        r3 = ctx.llm.complete(
            [
                {"role": "system", "content": _LOCALIZE_SYSTEM},
                {"role": "user", "content": (
                    f"Task: {t.task}\nFailing subtask {failing['id']} (phase="
                    f"{failing['phase']}, steps [{failing['start']}.."
                    f"{failing['end']}]). Full trajectory (with upstream "
                    f"dependencies; agent roster: {', '.join(t.agents())}):\n{view}"
                )},
            ],
            schema=ChiefVerdict,
            tag=f"{self.name}_localize",
        )
        v = r3.parsed
        assert isinstance(v, ChiefVerdict)

        step = min(max(v.step, 0), len(t.events) - 1)
        responsible = (
            v.responsible_agent
            if v.responsible_agent in t.agents() else t.agents()[0]
        )
        mechanism = self._clamp_mechanism(v.mechanism)
        ev = t.events[step]
        fail_oracle = oracle_by_id.get(failing["id"])
        oracle_line = (
            f"oracle[{failing['id']}].acceptance={fail_oracle.acceptance}"
            if fail_oracle else f"oracle[{failing['id']}]=(none)"
        )
        hyp = Hypothesis(
            agent=responsible,
            step=step,
            root_cause=f"[{mechanism}] {v.reason}",
            root_cause_code=None,
            responsible_side="model",
            evidence=[
                f"[{ev.index}] {ev.agent} {ev.kind} :: "
                f"{str(ev.payload.get('content', ev.payload))[:140]}",
                f"backtrack: subtask_evals="
                f"{[(e.subtask_id, e.passed) for e in evals.evals]}",
                oracle_line,
            ],
            fix_suggestion=v.fix_suggestion,
            confidence=v.confidence,
        )
        # clamp-with-trace (same discipline as all_at_once's failure_mode):
        # an out-of-vocabulary mechanism is clamped, and the raw judge value
        # is preserved in evidence + artifact rather than silently rewritten
        if mechanism != v.mechanism:
            hyp.evidence.append(
                f"(judgement clamped: mechanism {v.mechanism!r}->{mechanism!r}: "
                f"not in {MECHANISMS})"
            )
        if fallback_note:
            hyp.evidence.append(fallback_note)
        artifact: dict = {
            "hypotheses": [hyp.to_dict()],
            "oracles": [o.model_dump() for o in oracles.oracles],
            "subtask_evals": [e.model_dump() for e in evals.evals],
            "failing_subtask": failing["id"],
            "mechanism": mechanism,
            "fallback_note": fallback_note,
            "n_llm_calls": 3,
        }
        if mechanism != v.mechanism:
            artifact["mechanism_clamped"] = {"from": v.mechanism, "to": mechanism}
        bundle.put("attribute", self.name, artifact)

    @staticmethod
    def _clamp_mechanism(raw: str) -> str:
        """Clamp a free-string mechanism into MECHANISMS: closest vocabulary
        word by difflib similarity, else "unknown" (never passes an
        out-of-vocabulary value into the artifact unchecked). The 0.75
        cutoff only catches near-typo distances -- semantically different
        words (e.g. "propagated_error") map to "unknown" rather than to a
        spuriously similar vocabulary entry."""
        if raw in MECHANISMS:
            return raw
        close = difflib.get_close_matches(
            raw.lower(), MECHANISMS, n=1, cutoff=0.75
        )
        return close[0] if close else "unknown"
