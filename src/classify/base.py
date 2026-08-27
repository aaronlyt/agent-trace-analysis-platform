"""classify —— error-classification labeling layer (literature §5).

Taxonomy service: fusion label structure (interaction=MAST × module=AgentError
× system-level=SysTax × responsibility side=Model or Harness). taxonomy.py is
the shared vocabulary (not an algorithm module); labeling algorithms
(judge / rule pack) write their output to artifacts["classify"].
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Classifier(StageAlgorithm):
    """Base class for classification algorithms. Artifact contract: write the label list to artifacts["classify"]."""

    stage = "classify"
