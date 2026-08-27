"""attribute package: failure attribution algorithms (L0~L3 ladder). Importing registers them."""

from atap.attribute.all_at_once import AllAtOnceAttributor, AttributionVerdict
from atap.attribute.base import Attributor
from atap.attribute.binary_search import BinaryRefine, BinarySearchAttributor
from atap.attribute.chief import ChiefAttributor
from atap.attribute.counterfactual_replay import CounterfactualReplayAttributor
from atap.attribute.claim_audit import ClaimAuditAttributor
from atap.attribute.tree_diagnosis import TreeDiagnosisAttributor
from atap.attribute.rg_ug import RGUGAttributor
from atap.attribute.sbfl import SBFLAttributor

__all__ = [
    "Attributor",
    "AllAtOnceAttributor",
    "AttributionVerdict",
    "BinarySearchAttributor",
    "BinaryRefine",
    "RGUGAttributor",
    "ChiefAttributor",
    "CounterfactualReplayAttributor",
    "ClaimAuditAttributor",
    "TreeDiagnosisAttributor",
    "SBFLAttributor",
]
