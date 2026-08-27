"""analyze —— analysis and evaluation layer (overall architecture layer 4, literature §4).

Answers "Is it good? Are there problems? How many problems?": deterministic
metrics, loop detection, LLM-as-a-judge, failure clustering, drift detection.
Detection only, no attribution (detection ≠ attribution).
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Analyzer(StageAlgorithm):
    """Base class for analysis algorithms. Artifact contract: write evaluation conclusions (scores/findings/metrics) to artifacts["analyze"]."""

    stage = "analyze"
