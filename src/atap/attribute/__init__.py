"""attribute 包：失败归因算法（L0~L3 阶梯）。导入即注册。"""

from atap.attribute.all_at_once import AllAtOnceAttributor, AttributionVerdict
from atap.attribute.base import Attributor
from atap.attribute.binary_search import BinaryRefine, BinarySearchAttributor
from atap.attribute.sbfl import SBFLAttributor

__all__ = [
    "Attributor",
    "AllAtOnceAttributor",
    "AttributionVerdict",
    "BinarySearchAttributor",
    "BinaryRefine",
    "SBFLAttributor",
]
