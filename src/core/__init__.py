"""core -- core abstraction layer (zero algorithms, zero I/O implementations).

Dependency direction convention (architectural invariant, enforced by tests/test_invariants.py):
core must not import any stage package / llm implementation / io implementation / sandbox.
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
