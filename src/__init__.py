"""atap -- Agent trajectory analysis and error attribution platform (reproducing "Overall Pipeline Architecture and Algorithm Literature").

Importing this package completes algorithm registration (transformers-style:
algorithm modules self-register into core.registry at import time; adding an
algorithm = a new module + @register, zero core changes).
"""

__version__ = "0.1.0"

# core abstractions (for downstream from atap import ...)
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

# registration bootstrap: importing each stage package triggers algorithm
# registration (core does not depend back on them)
from atap import represent, analyze, classify, attribute, recover  # noqa: F401,E402
