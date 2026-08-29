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
    """Base class for pipeline algorithms. Subclasses must set ``stage`` and ``name``.

    ``requires`` declares hard artifact dependencies on other algorithms --
    (stage, name) pairs this algorithm crashes without (it reads
    ``bundle.get(stage, name)`` and raises when absent). The wildcard name
    ``"*"`` means "any algorithm of that stage" (e.g. recoverers consume
    ``bundle.hypotheses()`` from whichever attribution algorithm ran).
    validate_against_registry enforces it at config/assembly time -- missing
    or later-positioned dependencies fail the build instead of crashing
    (or silently no-op-ing) mid-run; trajectory-meta dependencies (e.g.
    rg_ug's qrels) stay runtime-checked by nature.
    """

    stage: ClassVar[str]
    name: ClassVar[str]
    requires: ClassVar[tuple[tuple[str, str], ...]] = ()

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

    def run_corpus(
        self, bundles: list["TrajectoryBundle"], ctx: "RunContext"
    ) -> list[tuple[str, str]]:
        """Cross-trajectory aggregation scope. Default implementation = run_one
        per trajectory, with per-trajectory error isolation (review
        2026-08-28): one crashing trajectory no longer fails its siblings --
        the failure is recorded as an error artifact on that bundle only and
        returned as ``(trace_id, error)`` pairs, which Pipeline folds into
        n_errors (never a silent success). Cross-trajectory algorithms that
        override run_corpus keep the pipeline-level algorithm isolation
        instead and may return None."""
        failures: list[tuple[str, str]] = []
        for bundle in bundles:
            try:
                self.run_one(bundle, ctx)
            except Exception as e:  # noqa: BLE001 - isolation is the point
                err = f"{type(e).__name__}: {e}"
                failures.append((bundle.trace_id, err))
                bundle.put(self.stage, self.name, {
                    "status": "error",
                    "error": err[:500],
                    "isolated": True,
                })
        return failures

    # -- description ----------------------------------------------------------

    def describe(self) -> str:
        return f"[{self.stage}/{self.name}] {type(self).__name__}(params={self.params})"
