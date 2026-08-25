"""atap —— Agent 轨迹分析与错误归因平台（复现《整体流程架构与算法文献》）。

导入本包即完成算法注册（transformers 式：算法模块在导入时自注册到
core.registry；新增算法 = 新模块 + @register，零改核心）。
"""

__version__ = "0.1.0"

# 核心抽象（供下游 from atap import ... 使用）
from atap.core import (  # noqa: F401
    STAGE_ORDER,
    ConfigError,
    Hypothesis,
    Pipeline,
    PipelineConfig,
    PipelineReport,
    RegistryError,
    RunContext,
    StageAlgorithm,
    Trajectory,
    TrajectoryBundle,
    create,
    list_algorithms,
    load_config,
    register,
)

# 注册引导：导入各 stage 包触发算法注册（core 不反向依赖它们）
from atap import represent, analyze, classify, attribute, recover  # noqa: F401,E402
