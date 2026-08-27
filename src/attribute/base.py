"""attribute — the failure attribution layer (core of layer 5 in the overall architecture, paper §6).

Organized as a cost ladder: L0 rules → L1 judge → L2 deep → L3 replay; uniformly
outputs ranked hypotheses (core.schema.Hypothesis), written to
artifacts["attribute"][algorithm_name]["hypotheses"].
Trigger semantics: attribution algorithms filter on their own (e.g., only failed
trajectories are processed — detection ≠ attribution).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atap.core.base import StageAlgorithm
from atap.core.schema import Hypothesis

if TYPE_CHECKING:
    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext


class Attributor(StageAlgorithm):
    """Base class for attribution algorithms. Artifact contract: bundle.put("attribute", name, {"hypotheses": [...]})."""

    stage = "attribute"

    def emit(self, bundle: "TrajectoryBundle", hypotheses: list[Hypothesis]) -> None:
        bundle.put(
            "attribute",
            self.name,
            {"hypotheses": [h.to_dict() for h in hypotheses]},
        )
