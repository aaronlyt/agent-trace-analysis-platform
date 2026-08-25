"""attribute —— 失败归因层（总体架构 ⑤层核心，文献 §6）。

按成本阶梯 L0 规则 → L1 judge → L2 深度 → L3 重放组织；统一输出
ranked hypotheses（core.schema.Hypothesis），写入
artifacts["attribute"][算法名]["hypotheses"]。
触发语义：归因算法自行过滤（如只处理失败轨迹——检测≠归因）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atap.core.base import StageAlgorithm
from atap.core.schema import Hypothesis

if TYPE_CHECKING:
    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext


class Attributor(StageAlgorithm):
    """归因算法基类。产物契约：bundle.put("attribute", name, {"hypotheses": [...]})。"""

    stage = "attribute"

    def emit(self, bundle: "TrajectoryBundle", hypotheses: list[Hypothesis]) -> None:
        bundle.put(
            "attribute",
            self.name,
            {"hypotheses": [h.to_dict() for h in hypotheses]},
        )
