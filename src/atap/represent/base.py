"""represent —— 表征层（总体架构 ③层，文献 §3）。

派生视图：R0 规范事件 / R1 折叠 / R2 依赖图 / R3 claim 台账 /
R4 层级树 / R5 动作签名。表征是分析/归因的唯一数据接口：本包产物
写入 bundle.artifacts["represent"]，供下游按名消费。
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Representer(StageAlgorithm):
    """表征算法基类。产物契约：至少写一个以算法名为键的视图/统计产物。"""

    stage = "represent"
