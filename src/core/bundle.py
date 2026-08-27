"""TrajectoryBundle -- artifact container that flows a trajectory across pipelines.

Design (aligned with the overall architecture §2 inter-layer contract: the
representation layer is the sole data interface for analysis/attribution):
* one bundle per trajectory; algorithms write artifacts keyed by ``(stage, name)``;
* algorithms are decoupled: downstream algorithms read artifacts by upstream
  **algorithm name** only, without importing upstream modules; if an upstream
  artifact is absent, downstream should degrade explicitly or raise, never
  silently bypass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from atap.core.schema import Hypothesis, Trajectory


def _to_jsonable(obj: Any) -> Any:
    """Convert an artifact to a JSON-compatible form (dataclasses are expanded
    recursively; everything else passes through as-is)."""
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return _to_jsonable(vars(obj))
    return obj


@dataclass
class TrajectoryBundle:
    """A single trajectory + per-stage artifacts."""

    trajectory: Trajectory
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    # New trajectories produced by the recover stage (targeted rerun results),
    # collected by the orchestrator for closed-loop verification.
    reruns: list[Trajectory] = field(default_factory=list)

    # -- artifact read/write ---------------------------------------------------

    def put(self, stage: str, name: str, artifact: Any) -> None:
        """Write an artifact (overwrites same-name entries)."""
        self.artifacts.setdefault(stage, {})[name] = _to_jsonable(artifact)

    def get(self, stage: str, name: str, default: Any = None) -> Any:
        return self.artifacts.get(stage, {}).get(name, default)

    def has(self, stage: str, name: str) -> bool:
        return name in self.artifacts.get(stage, {})

    # -- typed access to common artifacts --------------------------------------

    @property
    def trace_id(self) -> str:
        return self.trajectory.trace_id

    @property
    def succeeded(self) -> bool:
        return self.trajectory.outcome.success

    def hypotheses(self) -> list[Hypothesis]:
        """Aggregate Hypotheses from all attribution algorithms (flattened by
        (algorithm, order)).

        Cross-algorithm reads of attribution results all go through here; the
        recovery stage stays unaware of specific attribution algorithm names.
        """
        out: list[Hypothesis] = []
        for name, art in self.artifacts.get("attribute", {}).items():
            items = art.get("hypotheses") if isinstance(art, dict) else None
            if items is None and isinstance(art, dict) and "hypothesis" in art:
                items = [art["hypothesis"]]
            if items is None:
                items = []
            for h in items:
                out.append(Hypothesis.from_dict(h) if isinstance(h, dict) else h)
        return out

    # -- report ----------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"trace={self.trace_id} task={self.trajectory.task[:60]!r}",
            f"outcome=({'success' if self.succeeded else 'FAILURE'})"
            f" events={len(self.trajectory.events)}",
        ]
        for stage in ("represent", "analyze", "classify", "attribute", "recover"):
            for name, art in self.artifacts.get(stage, {}).items():
                lines.append(f"  artifact {stage}/{name}: {json.dumps(art, ensure_ascii=False)[:160]}")
        for t in self.reruns:
            lines.append(
                f"  rerun {t.trace_id}: {'success' if t.outcome.success else 'FAILURE'}"
            )
        return "\n".join(lines)
