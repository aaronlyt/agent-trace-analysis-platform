"""L3 counterfactual replay final review — TraceElephant, arXiv:2604.22708 A.6.3 (Dynamic Agentic).

Mechanism (from the paper):
* static analysis first proposes ≤3 candidate (agent, step, reason) triples
  plus an **expected oracle** (the model's own inference of the correct output
  for that step, not a human annotation);
* the trajectory is replayed, and **when execution reaches the candidate error
  step, an LLM API middleware modifies that step's input request**, steering
  the agent away from the error;
* only the next **k=3 steps** are observed for whether they satisfy the
  expected oracle and whether the failure mode recurs — it does not run to the
  end of the task (a short window cannot reveal behavior change, a long window
  introduces downstream stochastic variance);
* effect: filters pseudo-causality (step 30.3%→33.3%, a relative +10%).

[adaptation] Candidates come from the Hypotheses of **other attribution
algorithms** in the bundle (deduplicated and kept ≤3 by confidence — the
paper's Static Agentic is a static attributor; its ranking rule is not
specified); oracle/edit text is one LLM call (tag=counterfactual_replay_oracle,
pseudo-judge = corrected text of the candidate step's symptom); replay goes
through ``env.replay_intervene(trajectory, step, edit_text, horizon=k)``
(the sandbox's in-place message replacement middleware — intervention-point
difference: the paper modifies that step's **input request** via an LLM API
middleware, whereas this implementation replaces that step's **message
content**; edits on TOOL_CALL steps do not take effect and are recorded
faithfully in meta). Verdict: validated = the edit changed the failure's
course (fault_removed within the window — the environment's response signal
to the intervention, replacing the paper's method-side criterion of "does the
observation window satisfy the expected oracle and does the failure mode
recur"; the middleware consumes an edit only when it **lands on the fault's
onset step** and names the fault, so a fault-naming edit at any other step
is also refuted — pseudo-causality is filtered by the replay mechanism
itself, not by the oracle's wording; oracle.expected only enters evidence
and takes no part in the verdict
[weakened claim]), refuted = the edit did not change the course (the
candidate is pseudo-causal/a symptom step). Validated candidates get
confidence +0.2 / refuted −0.3 [not specified in the paper: values chosen by
us] — the output is still the unified Hypothesis.

[adaptation] Window and decoding declarations (documentation only, no
behavior change): (a) the k-event observation window **includes the
intervened step itself** (events [step, step+k)) — the paper's A.6.3
observes the k steps *after* the intervention, a one-step offset kept here
so the window stays anchored exactly at the replayed suffix boundary
(window_events = len(events) − step); (b) the paper fixes the replay
decoding temperature at 0.3 (A.6.3, to retain controlled stochastic
diversity while re-running) — not implemented: the sandbox is fully
deterministic (replays are identical by construction), so there is no
sampling for a temperature to shape [not applicable offline].
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.render import judge_view
from atap.core.schema import Hypothesis


class CFOracle(BaseModel):
    expected: str = Field(description="The correct behavior this step should exhibit (the expected oracle)")
    edit_text: str = Field(
        description="The edit text replacing this step's message (steers away from the error; minimal change)"
    )


_SYSTEM = (
    "You are the oracle synthesizer for a counterfactual replay. Given a task, "
    "a trajectory, and a candidate error step, infer the correct behavior this "
    "step **should have exhibited** (expected), and give the minimal edit text "
    "(edit_text) that replaces this step's message — the edit should directly "
    "correct the step's erroneous behavior so the task moves toward success. "
    "Output JSON."
)


@register
class CounterfactualReplayAttributor(Attributor):
    stage = "attribute"
    name = "counterfactual_replay"

    #: paper parameters: candidates ≤3, window k=3
    MAX_CANDIDATES = 3
    HORIZON = 3

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        if not t.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        # candidates = Hypotheses from other attribution algorithms
        # (this algorithm's own prior artifacts are excluded)
        candidates: list[Hypothesis] = []
        seen: set[tuple[str, int]] = set()
        for name, art in bundle.artifacts.get("attribute", {}).items():
            if name == self.name:
                continue
            items = art.get("hypotheses") if isinstance(art, dict) else None
            for h in items or []:
                key = (h.get("agent"), int(h.get("step", 0)))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(Hypothesis.from_dict(h))
        if bundle.succeeded and not self.param("include_success", False):
            bundle.put(
                "attribute", self.name,
                {"hypotheses": [], "status": "success_no_attribution",
                 "candidates": [c.to_dict() for c in candidates]},
            )
            return
        if not candidates:
            bundle.put(
                "attribute", self.name,
                {
                    "hypotheses": [],
                    "status": "no_upstream_candidates",
                    "note": "L3 final review requires upstream attribution "
                            "candidates: counterfactual_replay should be "
                            "configured after other attribution algorithms",
                },
            )
            return
        replay = getattr(ctx.env, "replay_intervene", None) if ctx.env else None
        if replay is None:
            raise RuntimeError(
                "counterfactual_replay requires an environment that supports "
                "message-intervention replay (env.replay_intervene; sandbox: "
                "{type: toy})"
            )
        if ctx.llm is None:
            raise RuntimeError("counterfactual_replay requires an LLM client (RunContext.llm)")

        view = judge_view(bundle)
        candidates.sort(key=lambda h: (-h.confidence, h.step))
        candidates = candidates[: int(self.param("max_candidates", self.MAX_CANDIDATES))]
        k = int(self.param("horizon", self.HORIZON))

        verdicts: list[dict] = []
        adjusted: list[Hypothesis] = []
        for cand in candidates:
            ev = t.events[min(max(cand.step, 0), len(t.events) - 1)]
            result = ctx.llm.complete(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": (
                        f"Task: {t.task}\nCandidate error step [{ev.index}] {ev.agent} "
                        f"{ev.kind} :: {str(ev.payload.get('content', ev.payload))[:160]}\n"
                        f"Full trajectory:\n{view}"
                    )},
                ],
                schema=CFOracle,
                tag=f"{self.name}_oracle",
            )
            oracle = result.parsed
            assert isinstance(oracle, CFOracle)

            runs = replay(t, cand.step, oracle.edit_text, horizon=k)
            window = runs[0]
            removed = bool(window.meta.get("fault_removed"))
            validated = removed
            verdicts.append({
                "candidate": {"agent": cand.agent, "step": cand.step,
                              "source_confidence": cand.confidence},
                "oracle": oracle.expected,
                "edit_snippet": oracle.edit_text[:120],
                "window_events": len(window.events) - cand.step,
                "verdict": "validated" if validated else "refuted",
            })
            adj = Hypothesis(
                agent=cand.agent,
                step=cand.step,
                root_cause=(
                    f"[L3 counterfactual replay validated] {cand.root_cause}"
                    if validated else
                    f"[L3 counterfactual replay refuted (editing this step did not change the failure's course — pseudo-causal or a symptom step)] "
                    f"{cand.root_cause}"
                ),
                root_cause_code=cand.root_cause_code,
                responsible_side=cand.responsible_side,
                evidence=cand.evidence + [
                    f"cf_oracle={oracle.expected[:100]}",
                    f"window(k={k}) fault_removed={removed}",
                ],
                fix_suggestion=cand.fix_suggestion,
                confidence=(
                    min(cand.confidence + 0.2, 1.0) if validated
                    else max(cand.confidence - 0.3, 0.0)
                ),
            )
            adjusted.append(adj)

        bundle.put(
            "attribute",
            self.name,
            {
                "hypotheses": [h.to_dict() for h in adjusted],
                "verdicts": verdicts,
                "role": "L3_counterfactual_verifier",
                "horizon": k,
                "note": "validated = editing the candidate step changed the failure's course; refuted = pseudo-causal or a symptom step",
            },
        )
