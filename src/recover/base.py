"""recover -- recovery and enhancement layer (closed loop of layer ⑤ in the overall architecture, literature §7).

Consumes attribution output (bundle.hypotheses()), produces repair actions and rerun trajectories:
new rerun trajectories are written to bundle.reruns and sent back by the orchestrator to analyze for verification (step 6→3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from atap.core.base import StageAlgorithm

if TYPE_CHECKING:  # type references only, no runtime import
    from atap.core.schema import Trajectory


class Recoverer(StageAlgorithm):
    """Base class for recovery algorithms. Artifact contract: write the recovery
    conclusion to artifacts["recover"]; append rerun trajectories to
    bundle.reruns (new trace_id, meta["rerun_of"]=original trajectory).

    Recovery execution goes through ``ctx.env`` (core's
    :class:`~atap.core.context.ReplayEnvironment` plus the message-intervention
    extension documented below); recoverers must degrade explicitly when the
    configured environment lacks the method they need, never silently bypass.
    """


@runtime_checkable
class RecoveryEnvironment(Protocol):
    """Recovery-side replay environment surface: the three execution sides
    consumed by the recover algorithms (superset of core's ReplayEnvironment;
    ``replay_intervene`` is the stage-4C extension).

    * ``rerun_from(trajectory, step, feedback) -> Trajectory`` (AgentDebug
      2509.25370 targeted rerun): keep the prefix ``[0, step)`` and
      re-execute from ``step`` with the executable ``feedback``;
    * ``resolve(trajectory, feedback) -> Trajectory`` (AgenTracer
      2509.03312 feedback injection): no prefix retention -- a full
      re-solve of the task carrying only the reflection feedback;
    * ``replay_intervene(trajectory, step, edit_text, *, horizon=None,
      n_repeats=1) -> list[Trajectory]`` (DoVer 2512.06749 message
      intervention): checkpoint replay with the message at ``step``
      replaced in place by ``edit_text``, repeated ``n_repeats`` times
      (``horizon=k`` returns only the k events after the step).
    """

    def rerun_from(
        self, trajectory: "Trajectory", step: int, feedback: str
    ) -> "Trajectory": ...

    def resolve(
        self, trajectory: "Trajectory", feedback: str
    ) -> "Trajectory": ...

    def replay_intervene(
        self,
        trajectory: "Trajectory",
        step: int,
        edit_text: str,
        *,
        horizon: int | None = None,
        n_repeats: int = 1,
    ) -> "list[Trajectory]": ...
