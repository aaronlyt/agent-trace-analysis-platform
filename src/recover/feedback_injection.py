"""Attribution feedback-injection re-solve -- AgenTracer, arXiv:2509.03312 §5.3 (ICLR'26).

Mechanism (from the paper): the MAS completes one solve round producing a
failed trajectory τ → the tracer generates reflective feedback on τ
(AgenTracer-8B takes its ⟨think⟩ reasoning segment) → the feedback is
injected into M's **next full solve round** (a brand-new episode, no prefix
retention -- orthogonal to targeted_rerun's prefix-preserving rerun) →
iterate for 3 rounds. Paper numbers: 3 rounds +4.8~14.2% (MaAS/OWL/MetaGPT ×
GAIA/MATH-500/HumanEval+), while the self-reflection baseline CRITIC drops
4.9~5.5%.

Differences from the paper:
* feedback source: the paper = the fine-tuned tracer's reasoning segment;
  this implementation takes the round-1 feedback from the **attribution
  Hypothesis** (the reflective text of root_cause + fix_suggestion --
  a reflection of the attribution output), and later rounds regenerate it
  via a judge reflection call on the latest failed trajectory [adaptation];
* injection point: the paper does not specify (prompt/history both work);
  this implementation hands the whole feedback text to
  ``RunContext.env.resolve(trajectory, feedback)``, leaving the injection
  method to the environment [declaration];
* recovery gate: none in the paper; AgentDebugX's suggest-only semantics are
  carried by the targeted_rerun side, and this algorithm likewise only
  produces rerun trajectories without auto-modifying the system.

Consumes the unified attribution contract (top Hypothesis, ordered by
(confidence, -step) same as targeted_rerun); param ``attribution``
[inference]: when several attribution algorithms are configured in the same
pipeline, ``confidence`` has no global scale across algorithms, so an
explicit ``attribution=<algorithm name>`` declares which algorithm's
Hypotheses (matched against ``Hypothesis.source``) this recoverer consumes
(unset = consume all); each round re-solves from the
**original trajectory's** fault state (rerun trajectory meta has
injected_fault stripped -- passing it along the chain would yield spurious
success; see the sandbox.resolve convention).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.core.registry import register
from atap.core.render import judge_view
from atap.recover.base import Recoverer


class Reflection(BaseModel):
    feedback: str = Field(description="Reflective feedback injected into the next solve round (concise, actionable)")


_REFLECT_SYSTEM = (
    "You are a reflective-feedback generator for failed trajectories "
    "(attribution feedback-injection style). Given a failed solve "
    "trajectory, produce a short piece of reflective feedback to inject "
    "before the next solve round: state which step contains the decisive "
    "error, what went wrong, and how to avoid it in the next round. Rely "
    "only on observable evidence in the trajectory; be concise and "
    "actionable."
)


@register
class FeedbackInjectionRecoverer(Recoverer):
    stage = "recover"
    name = "feedback_injection"

    def run_one(self, bundle, ctx) -> None:
        if bundle.succeeded:
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
        env = getattr(ctx, "env", None)
        if env is None or not hasattr(env, "resolve"):
            bundle.put(
                "recover", self.name,
                {"status": "no_replay_environment", "recovered": False,
                 "note": "RunContext.env is not configured or does not support resolve (full re-solve)"},
            )
            return

        top = max(hyps, key=lambda h: (h.confidence, -h.step))
        max_rounds = int(self.param("max_rounds", 3))   # paper: 3 rounds
        feedback = self._reflection_from_hypothesis(top)
        feedback_log: list[str] = [feedback[:300]]
        attempts: list[dict] = []
        recovered = False
        for k in range(1, max_rounds + 1):
            new_traj = env.resolve(bundle.trajectory, feedback)
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
            if k >= max_rounds:
                break  # last round failed: do not generate feedback that would never be injected (saves one LLM call)
            # AgenTracer: next-round feedback is regenerated by the tracer on the latest failed trajectory
            reflected = self._reflect(ctx, new_traj)
            if reflected:
                feedback = reflected
            else:
                feedback = (
                    f"{feedback}\n(attempt {k} failed: {new_traj.outcome.note})"
                )
            feedback_log.append(feedback[:300])

        bundle.put(
            "recover",
            self.name,
            {
                "status": "done",
                "origin": bundle.trace_id,
                "mode": "full_reresolve",
                "attribution": attribution,
                "seed_hypothesis": {
                    "agent": top.agent,
                    "step": top.step,
                    "confidence": top.confidence,
                },
                "feedback_seed": feedback_log[0],
                "feedback_rounds": feedback_log,
                "rounds": len(attempts),
                "attempts": attempts,
                "recovered": recovered,
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _reflection_from_hypothesis(top) -> str:
        parts = ["Attribution reflection on the previous failed solve:"]
        if top.root_cause:
            parts.append(top.root_cause)
        if top.fix_suggestion:
            parts.append(f"Fix suggestion: {top.fix_suggestion}")
        if top.agent and top.step is not None:
            parts.append(f"(responsible party {top.agent}, decisive error at step {top.step})")
        return "\n".join(parts)

    def _reflect(self, ctx, failed_traj) -> str | None:
        """Generate the next-round reflection on the latest failed trajectory
        (returns None without an LLM to fall back to degraded concatenation)."""
        if ctx.llm is None:
            return None
        from atap.core.bundle import TrajectoryBundle

        bundle = TrajectoryBundle(failed_traj)  # no SSF artifacts → full-view rendering
        messages = [
            {"role": "system", "content": _REFLECT_SYSTEM},
            {"role": "user", "content": f"Failed trajectory:\n{judge_view(bundle)}"},
        ]
        result = ctx.llm.complete(messages, schema=Reflection, tag="feedback_reflection")
        parsed = result.parsed
        assert isinstance(parsed, Reflection)
        return parsed.feedback
