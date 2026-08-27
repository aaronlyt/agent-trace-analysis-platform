"""RunContext -- dependency injection container for algorithm runtimes.

Algorithms never create their own LLM clients / storage / random sources;
they always take them from ctx (low-dependency principle: algorithm modules
depend only on core interfaces, concrete implementations are assembled from
config). Reruns in the recovery stage access the executable environment via
the :data:`ReplayEnvironment` protocol; core stays unaware of sandbox
implementations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atap.core.schema import Trajectory
    from atap.io.base import ArtifactStore
    from atap.llm.base import LLMClient


@runtime_checkable
class ReplayEnvironment(Protocol):
    """Replay environment contract (from stage three onward, two recovery
    execution sides).

    * ``rerun_from`` (AgentDebug 2509.25370 targeted rerun): keeps the
      trajectory prefix ``[0, step)`` and re-executes from ``step`` with
      feedback;
    * ``resolve`` (AgenTracer 2509.03312 feedback-injection re-solving): does
      not keep the prefix; re-solves the task from scratch, carrying only the
      reflection feedback text -- every round is a fresh episode.
    Both return a new trajectory (new trace_id, with origin recorded in meta).
    """

    def rerun_from(
        self, trajectory: "Trajectory", step: int, feedback: str
    ) -> "Trajectory": ...

    def resolve(self, trajectory: "Trajectory", feedback: str) -> "Trajectory": ...


@dataclass
class RunContext:
    """Shared context for one pipeline run."""

    llm: "LLMClient | None" = None            # LLMClient protocol (llm/base.py)
    store: "ArtifactStore | None" = None      # artifact persistence
    env: ReplayEnvironment | None = None      # replay environment (used by recover)
    rng: random.Random = field(default_factory=random.Random)
    run_dir: str = ""                         # output directory of this run (for reports)
