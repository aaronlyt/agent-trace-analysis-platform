"""RunContext —— 算法运行时的依赖注入容器。

算法不自己创建 LLM 客户端 / 存储 / 随机源，一律从 ctx 取（低依赖原则：
算法模块只依赖 core 接口，具体实现由配置装配）。恢复阶段的重跑通过
:data:`ReplayEnvironment` 协议访问可执行环境，core 不感知沙盒实现。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from atap.core.schema import Trajectory
    from atap.io.base import ArtifactStore


@runtime_checkable
class ReplayEnvironment(Protocol):
    """可重放环境契约（阶段三起含两种恢复执行侧）。

    * ``rerun_from``（AgentDebug 2509.25370 定向重跑）：保留轨迹前缀
      ``[0, step)``，从 ``step`` 开始带反馈重新执行；
    * ``resolve``（AgenTracer 2509.03312 反馈注入再求解）：不保留前缀，
      从头完整重解任务，唯一携带的是反思反馈文本——每轮都是全新 episode。
    两者都返回新轨迹（新 trace_id，meta 记录来源）。
    """

    def rerun_from(
        self, trajectory: "Trajectory", step: int, feedback: str
    ) -> "Trajectory": ...

    def resolve(self, trajectory: "Trajectory", feedback: str) -> "Trajectory": ...


@dataclass
class RunContext:
    """一次 pipeline 运行的共享上下文。"""

    llm: object | None = None                 # LLMClient 协议（llm/base.py）
    store: "ArtifactStore | None" = None      # 产物持久化
    env: ReplayEnvironment | None = None      # 可重放环境（recover 用）
    rng: random.Random = field(default_factory=random.Random)
    run_dir: str = ""                         # 本次运行输出目录（报告用）
    # 闭环收集：recover 产出的新轨迹 trace_id 会在 pipeline 编排时归档
    closed_loop_rounds: int = 0
