"""Targeted rerun (Targeted Re-rollout) -- AgentDebug, arXiv:2509.25370 Algorithm 1.

Mechanism (paper Stage 3): after locating the earliest critical step t*,
keep the prefix [0, t*) and re-roll out from t* with **executable
feedback**; return on success, otherwise refine the feedback and retry, for
at most I rounds (paper N=5; GPT-4o-mini ALFWorld 21→55). Key design: fix
only the root-cause step, not surface symptoms; the feedback must "name the
error type + give executable guidance" so the agent can change course
during the rerun.

This implementation: t* and the feedback come from the top Hypothesis of
the attribution output (consumes the unified attribution contract, unaware
of the concrete attribution algorithm). Param ``attribution`` [inference]:
when several attribution algorithms are configured in the same pipeline,
``confidence`` has no global scale across algorithms, so an explicit
``attribution=<algorithm name>`` declares which algorithm's Hypotheses
(matched against ``Hypothesis.source``) this recoverer consumes; unset =
consume all. UpdateFeedback is a weakened
version: plain string concatenation (attempt number + failure note +
re-pointing at step t*), without any LLM re-analysis of the latest failed
trajectory -- rounds 2..5 may repeat the same failure note and the feedback
information gain does not grow (paper Fig.15 regenerates more specific
guidance with iterative context via an LLM) [declared weakening]. The t*
selection rule = highest confidence, ties broken by the earliest step --
the paper Stage 2's t*←min(T*) (earliest critical step) is mainly carried
by the attribution prompt's "earliest decisive error" instruction; here it
serves as the tie-break fallback [declared deviation: confidence takes
priority over earliest step]. Replay is executed by RunContext.env
(ReplayEnvironment protocol), each round replaying from t* of the
**original trajectory** (the paper's pseudocode chains τ⁽ᵏ⁻¹⁾; under a
deterministic sandbox both are equivalent, and the rerun trajectory's meta
has injected_fault stripped -- chaining would misjudge "no fault" and
yield spurious success) [declared deviation]. Literature warning (§7): a
responsible-agent granularity that is too coarse cannot be consumed by
enhancement, so this algorithm consumes the two fine-grained fields
(step, fix_suggestion).

Artifacts: ``{"origin", "t_star", "rounds", "attempts", "recovered"}``;
rerun trajectories are appended to bundle.reruns (new trace_id,
meta.rerun_of=original trajectory) and sent back by the orchestrator to
analyze to verify improvement (step 6→3 closed loop).
"""

from __future__ import annotations

from atap.core.registry import register
from atap.recover.base import Recoverer


@register
class TargetedRerunRecoverer(Recoverer):
    stage = "recover"
    name = "targeted_rerun"
    requires = (("attribute", "*"),)   # consumes bundle.hypotheses() from any attributor

    def run_one(self, bundle, ctx) -> None:
        if bundle.succeeded:
            # nothing was broken, so nothing was recovered (same semantics as
            # dover's skipped_success); the artifact still lands so downstream
            # can tell "ran and skipped" from "not configured at all"
            bundle.put(
                "recover", self.name,
                {"status": "skipped_success", "recovered": False},
            )
            return
        hyps = bundle.hypotheses()
        attribution = self.param("attribution", None)
        if attribution is not None:
            # [inference] confidence has no global scale when multiple
            # attribution algorithms are configured together: an explicit
            # ``attribution`` param declares which algorithm's output this
            # recoverer consumes (Hypothesis.source; defensive getattr --
            # the field may be absent on older Hypothesis payloads)
            hyps = [h for h in hyps if getattr(h, "source", "") == attribution]
        if not hyps:
            note = (
                "failed trajectory has no attribution output: recovery must "
                "consume attribution (broken-link warning in literature §7)"
            )
            if attribution is not None:
                note += f" (attribution filter: no hypothesis with source={attribution!r})"
            bundle.put(
                "recover", self.name,
                {"status": "skipped_no_hypothesis", "recovered": False,
                 "attribution": attribution, "note": note},
            )
            return
        if ctx.env is None:
            bundle.put(
                "recover", self.name,
                {"status": "no_replay_environment", "recovered": False,
                 "note": "RunContext.env is not configured (sandbox.type=toy provides one)"},
            )
            return

        # Highest confidence, ties broken by earliest step (aligned with the paper's earliest-critical-step principle t*←min(T*))
        top = max(hyps, key=lambda h: (h.confidence, -h.step))
        max_rounds = int(self.param("max_rounds", 5))
        feedback = top.fix_suggestion or top.root_cause
        attempts: list[dict] = []
        recovered = False
        for k in range(1, max_rounds + 1):
            new_traj = ctx.env.rerun_from(bundle.trajectory, top.step, feedback)
            bundle.reruns.append(new_traj)
            attempts.append(
                {
                    "round": k,
                    "trace_id": new_traj.trace_id,
                    "success": new_traj.outcome.success,
                    "note": new_traj.outcome.note[:120],
                }
            )
            if new_traj.outcome.success:
                recovered = True
                break
            # UpdateFeedback (weakened): refine the guidance with the failure note and retry
            feedback = (
                f"{feedback}\n(attempt {k} failed: {new_traj.outcome.note} -- "
                f"give a more specific correction targeting the earliest "
                f"decisive error at step {top.step}.)"
            )

        bundle.put(
            "recover",
            self.name,
            {
                "status": "done",
                "origin": bundle.trace_id,
                "t_star": top.step,
                "responsible_agent": top.agent,
                "attribution": attribution,
                "feedback_seed": (top.fix_suggestion or top.root_cause)[:200],
                "rounds": len(attempts),
                "attempts": attempts,
                "recovered": recovered,
            },
        )
