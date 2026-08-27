"""StageAlgorithm abstract base class -- common ancestor of all pipeline algorithms
(transformers-style pluggable units).

Each algorithm = one class inside one module:
* declare two ClassVars: ``stage`` (owning pipeline) and ``name`` (registry name);
* once decorated with ``@register`` it can be composed by name in YAML config;
* depends only on core/llm/io interfaces and artifacts (bundle), never imports other algorithm modules.

Dual scope (literature principle "aggregation before singleton": agent-level 53.5% usable
vs step-level 14.2%, Who&When 2505.00212):
* :meth:`run_one` -- single-trajectory scope, must be implemented;
* :meth:`run_corpus` -- cross-trajectory aggregation scope, defaults to calling
  run_one per trajectory; cross-trajectory algorithms (stage-three SBFL /
  failure clustering) override it and ignore run_one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # avoid runtime circular imports, type references only
    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext

# Five pluggable pipelines among the six stages ("collection" is handled by the io layer, not listed here).
STAGE_ORDER: tuple[str, ...] = (
    "represent",  # representation: R0/R1/R5...
    "analyze",    # analysis and evaluation
    "classify",   # error classification labeling
    "attribute",  # failure attribution
    "recover",    # recovery and enhancement
)


class StageAlgorithm(ABC):
    """Base class for pipeline algorithms. Subclasses must set ``stage`` and ``name``."""

    stage: ClassVar[str]
    name: ClassVar[str]

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    # -- convenience for subclasses to read config parameters -----------------

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    # -- dual scope -----------------------------------------------------------

    @abstractmethod
    def run_one(self, bundle: "TrajectoryBundle", ctx: "RunContext") -> None:
        """Process a single trajectory and write artifacts into ``bundle.artifacts``.

        Contract: algorithms must not modify ``bundle.trajectory``'s outcome
        (detection/attribution does not rewrite history); representation
        algorithms are responsible for populating ``trajectory.events``.
        """

    def run_corpus(self, bundles: list["TrajectoryBundle"], ctx: "RunContext") -> None:
        """Cross-trajectory aggregation scope. Default implementation = run_one per trajectory."""
        for bundle in bundles:
            self.run_one(bundle, ctx)

    # -- description ----------------------------------------------------------

    def describe(self) -> str:
        return f"[{self.stage}/{self.name}] {type(self).__name__}(params={self.params})"
