"""core —— 核心抽象层（零算法、零 I/O 实现）。

依赖方向约定（架构不变量，tests/test_invariants.py 强制）：
core 不 import 任何 stage 包 / llm 实现 / io 实现 / sandbox。
"""

from atap.core.base import STAGE_ORDER, StageAlgorithm
from atap.core.bundle import TrajectoryBundle
from atap.core.config import (
    AlgorithmSpec,
    ConfigError,
    PipelineConfig,
    config_from_dict,
    load_config,
    validate_against_registry,
)
from atap.core.context import ReplayEnvironment, RunContext
from atap.core.pipeline import Pipeline, PipelineReport
from atap.core.registry import (
    RegistryError,
    create,
    list_algorithms,
    register,
)
from atap.core.schema import (
    EVENT_KINDS,
    Hypothesis,
    Outcome,
    TraceEvent,
    Trajectory,
)

__all__ = [
    "STAGE_ORDER",
    "StageAlgorithm",
    "TrajectoryBundle",
    "AlgorithmSpec",
    "ConfigError",
    "PipelineConfig",
    "config_from_dict",
    "load_config",
    "validate_against_registry",
    "ReplayEnvironment",
    "RunContext",
    "Pipeline",
    "PipelineReport",
    "RegistryError",
    "create",
    "list_algorithms",
    "register",
    "EVENT_KINDS",
    "Hypothesis",
    "Outcome",
    "TraceEvent",
    "Trajectory",
]
