"""represent -- representation layer (layer 3 of the overall architecture, paper section 3).

Derived views: R0 canonical events / R1 folding / R2 dependency graph /
R3 claim ledger / R4 hierarchy tree / R5 action signatures. Representation is
the sole data interface for analysis/attribution: this package's outputs are
written into bundle.artifacts["represent"] for downstream consumption by name.
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Representer(StageAlgorithm):
    """Base class for representation algorithms. Artifact contract: write at least one view/statistics artifact keyed by the algorithm name."""

    stage = "represent"
