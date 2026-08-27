"""recover -- recovery and enhancement layer (closed loop of layer ⑤ in the overall architecture, literature §7).

Consumes attribution output (bundle.hypotheses()), produces repair actions and rerun trajectories:
new rerun trajectories are written to bundle.reruns and sent back by the orchestrator to analyze for verification (step 6→3).
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Recoverer(StageAlgorithm):
    """Base class for recovery algorithms. Artifact contract: write the recovery
    conclusion to artifacts["recover"]; append rerun trajectories to
    bundle.reruns (new trace_id, meta["rerun_of"]=original trajectory).
    """

    stage = "recover"
