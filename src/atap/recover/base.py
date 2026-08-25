"""recover —— 恢复与增强层（总体架构 ⑤层闭环，文献 §7）。

消费归因输出（bundle.hypotheses()），产出修复动作与重跑轨迹：
重跑新轨迹写入 bundle.reruns 并由编排器送回 analyze 验证（环节 6→3）。
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Recoverer(StageAlgorithm):
    """恢复算法基类。产物契约：写恢复结论到 artifacts["recover"]；
    重跑轨迹 append 到 bundle.reruns（新 trace_id，meta["rerun_of"]=原轨迹）。
    """

    stage = "recover"
