"""io -- collection/storage interface layer (abstractions for steps ①②).

Aligned with the overall architecture §2: the storage layer is the single
source of truth, and ③④⑤ read data only from here. The Langfuse v3 adapter
is left for the long term (phase four); phases one to three use the local
JSONL implementation (zero service dependencies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atap.core.bundle import TrajectoryBundle
    from atap.core.schema import Trajectory


@runtime_checkable
class TraceSource(Protocol):
    """Trajectory read source (collection artifacts → trajectory list)."""

    def load(self) -> "list[Trajectory]": ...


@runtime_checkable
class TraceStore(Protocol):
    """Trajectory write (new trajectories produced by targeted rerun are
    written back to the source of truth, for the closed loop)."""

    def save(self, trajectory: "Trajectory") -> None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Persistence of per-stage artifacts (one directory per trace)."""

    def save_artifact(self, trace_id: str, stage: str, name: str, artifact: object) -> None: ...

    def save_report(self, filename: str, payload: object) -> None: ...
