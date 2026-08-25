"""io —— 采集/存储接口层（环节①②的抽象）。

对齐总体架构 §2：存储层是唯一事实源，③④⑤只从这里读数据。Langfuse
v3 适配器留作远期（阶段四），阶段一~三用本地 JSONL 实现（零服务依赖）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atap.core.bundle import TrajectoryBundle
    from atap.core.schema import Trajectory


@runtime_checkable
class TraceSource(Protocol):
    """轨迹读取源（采集产物 → 轨迹列表）。"""

    def load(self) -> "list[Trajectory]": ...


@runtime_checkable
class TraceStore(Protocol):
    """轨迹写入（定向重跑产出的新轨迹写回事实源，闭环用）。"""

    def save(self, trajectory: "Trajectory") -> None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """各阶段产物持久化（按 trace 分目录）。"""

    def save_artifact(self, trace_id: str, stage: str, name: str, artifact: object) -> None: ...

    def save_report(self, filename: str, payload: object) -> None: ...
