"""Pipeline -- six-stage orchestrator.

Pipeline semantics align with two key facts from "Overall Pipeline Architecture
and Algorithm Literature" §1:

* **detection != attribution**: analyze only discovers "whether there is a
  problem", attribute answers "which error determined the failure"; the step
  where failure manifests is often not the causal step (81% of
  mislocalizations skew late). Attribution algorithms decide their own trigger
  conditions (e.g. all_at_once only processes failed trajectories).
* **closed loop**: alerts/low scores/failures from stage 4 trigger stage 5
  attribution and recovery; new trajectories produced by recover flow back
  into analyze to verify improvement (stage 6 -> stage 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atap.core.base import STAGE_ORDER

if TYPE_CHECKING:
    from atap.core.base import StageAlgorithm
    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext
    from atap.core.schema import Trajectory


@dataclass
class PipelineReport:
    """Human-readable report of one run (written to disk / CLI output)."""

    run_name: str
    n_traces: int = 0
    n_failures: int = 0
    n_attributed: int = 0
    n_reruns: int = 0
    n_rerun_success: int = 0
    stage_log: list[str] = field(default_factory=list)
    bundle_summaries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "n_traces": self.n_traces,
            "n_failures": self.n_failures,
            "n_attributed": self.n_attributed,
            "n_reruns": self.n_reruns,
            "n_rerun_success": self.n_rerun_success,
            "stage_log": self.stage_log,
            "bundle_summaries": self.bundle_summaries,
        }


class Pipeline:
    """Executes each algorithm in STAGE_ORDER (each algorithm runs the
    run_corpus aggregation scope first)."""

    def __init__(self, algorithms: list["StageAlgorithm"]) -> None:
        self.algorithms = algorithms

    def run(
        self, trajectories: list["Trajectory"], ctx: "RunContext"
    ) -> tuple[list["TrajectoryBundle"], PipelineReport]:
        from atap.core.bundle import TrajectoryBundle

        report = PipelineReport(run_name=ctx.run_dir or "run")
        bundles = [TrajectoryBundle(t) for t in trajectories]
        report.n_traces = len(bundles)
        report.n_failures = sum(0 if b.succeeded else 1 for b in bundles)

        for stage in STAGE_ORDER:
            for algo in self.algorithms:
                if algo.stage != stage:
                    continue
                t0 = time.time()
                algo.run_corpus(bundles, ctx)
                report.stage_log.append(
                    f"{stage}/{getattr(algo, 'name', type(algo).__name__)} "
                    f"-> {len(bundles)} bundles in {time.time() - t0:.3f}s"
                )

        report.n_attributed = sum(1 for b in bundles if b.hypotheses())
        rerun_traces: list[Trajectory] = []
        for b in bundles:
            report.bundle_summaries.append(b.summary())
            report.n_reruns += len(b.reruns)
            report.n_rerun_success += sum(1 for t in b.reruns if t.outcome.success)
            rerun_traces.extend(b.reruns)
        self.last_reruns = rerun_traces
        return bundles, report

    def run_closed_loop(
        self,
        trajectories: list["Trajectory"],
        ctx: "RunContext",
        *,
        max_rounds: int = 1,
    ) -> tuple[list["TrajectoryBundle"], list[PipelineReport]]:
        """Closed loop: after one full round, new trajectories produced by
        recover are fed back through the whole pipeline to verify improvement.

        Detail (granularity differs from AgentDebugX's "failure re-entry loop"
        wording): for each origin only its **last** rerun enters the
        verification round (mid-way failed attempts are excluded); the
        verification round runs recover again -- still-failing reruns get
        re-attributed and rerun again (nested recovery), limited to a single
        verification round by ``max_rounds=1``.

        Returns the **first-round bundles** (preserving full
        attribution/recovery artifacts), where each rerun trajectory
        additionally carries a ``recover/closed_loop`` artifact recording the
        verification verdict; the verification round's report is appended to
        reports (the stage 6 -> 3 loop).
        """
        origin_bundles, report = self.run(trajectories, ctx)
        reports = [report]
        reruns = getattr(self, "last_reruns", [])
        if reruns and max_rounds >= 1:
            rerun_by_origin: dict[str, Trajectory] = {}
            for t in reruns:
                origin = t.meta.get("rerun_of")
                if origin:
                    rerun_by_origin[origin] = t
            current = [rerun_by_origin.get(t.trace_id, t) for t in trajectories]
            _, verify_report = self.run(current, ctx)
            reports.append(verify_report)
            improved = set()  # trace_ids of successful trajectories in the verification round
            for t in current:
                if t.outcome.success:
                    improved.add(t.trace_id)
            for b in origin_bundles:
                repl = rerun_by_origin.get(b.trace_id)
                b.put(
                    "recover",
                    "closed_loop",
                    {
                        "rerun_trace_id": repl.trace_id if repl else None,
                        "verified_improved": bool(repl and repl.trace_id in improved),
                    },
                )
            ctx.closed_loop_rounds += 1
        return origin_bundles, reports
