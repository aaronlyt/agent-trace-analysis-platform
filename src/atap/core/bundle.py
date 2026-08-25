"""TrajectoryBundle —— 轨迹在各流程间流转的产物容器。

设计（对齐总体架构 §2 层间契约：表征层是分析/归因的唯一数据接口）：
* 一条轨迹对应一个 bundle；算法把产物按 ``(stage, name)`` 写入；
* 算法间解耦：下游算法只按上游**算法名**读产物，不 import 上游模块；
  若上游缺席，下游应显式降级或报错，而不是静默绕过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from atap.core.schema import Hypothesis, Trajectory


def _to_jsonable(obj: Any) -> Any:
    """产物转 JSON 兼容形态（dataclass 递归展开；其余原样透传）。"""
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return _to_jsonable(vars(obj))
    return obj


@dataclass
class TrajectoryBundle:
    """单轨迹 + 各阶段产物。"""

    trajectory: Trajectory
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    # recover 阶段产出的新轨迹（定向重跑结果），由编排器收集用于闭环验证。
    reruns: list[Trajectory] = field(default_factory=list)

    # -- 产物读写 -----------------------------------------------------------

    def put(self, stage: str, name: str, artifact: Any) -> None:
        """写入产物（覆盖同名）。"""
        self.artifacts.setdefault(stage, {})[name] = _to_jsonable(artifact)

    def get(self, stage: str, name: str, default: Any = None) -> Any:
        return self.artifacts.get(stage, {}).get(name, default)

    def has(self, stage: str, name: str) -> bool:
        return name in self.artifacts.get(stage, {})

    # -- 常用产物的类型化访问 --------------------------------------------------

    @property
    def trace_id(self) -> str:
        return self.trajectory.trace_id

    @property
    def succeeded(self) -> bool:
        return self.trajectory.outcome.success

    def hypotheses(self) -> list[Hypothesis]:
        """汇总所有归因算法的 Hypothesis（按 (算法, 序) 展平）。

        跨算法读取归因结果统一走这里；恢复阶段不感知具体归因算法名。
        """
        out: list[Hypothesis] = []
        for name, art in self.artifacts.get("attribute", {}).items():
            items = art.get("hypotheses") if isinstance(art, dict) else None
            if items is None and isinstance(art, dict) and "hypothesis" in art:
                items = [art["hypothesis"]]
            if items is None:
                items = []
            for h in items:
                out.append(Hypothesis.from_dict(h) if isinstance(h, dict) else h)
        return out

    # -- 报告 ---------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"trace={self.trace_id} task={self.trajectory.task[:60]!r}",
            f"outcome=({'success' if self.succeeded else 'FAILURE'})"
            f" events={len(self.trajectory.events)}",
        ]
        for stage in ("represent", "analyze", "classify", "attribute", "recover"):
            for name, art in self.artifacts.get(stage, {}).items():
                lines.append(f"  artifact {stage}/{name}: {json.dumps(art, ensure_ascii=False)[:160]}")
        for t in self.reruns:
            lines.append(
                f"  rerun {t.trace_id}: {'success' if t.outcome.success else 'FAILURE'}"
            )
        return "\n".join(lines)
