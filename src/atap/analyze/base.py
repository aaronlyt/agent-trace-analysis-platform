"""analyze —— 分析与评测层（总体架构 ④层，文献 §4）。

回答"好不好？有没有问题？问题多不多？"：确定性指标、循环检测、
LLM-as-judge、失败聚类、漂移检测。只检测不归因（检测 ≠ 归因）。
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Analyzer(StageAlgorithm):
    """分析算法基类。产物契约：写评测结论（分数/finding/指标）到 artifacts["analyze"]。"""

    stage = "analyze"
