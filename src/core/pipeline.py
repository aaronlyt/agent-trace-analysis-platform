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
from typing import TYPE_CHECKING, Any

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
    # algorithm-level failures isolated by Pipeline.run (a crashing
    # algorithm no longer aborts the run: its error is recorded here, an
    # error artifact marks the affected bundles, later stages continue)
    n_errors: int = 0
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
            "n_errors": self.n_errors,
            "stage_log": self.stage_log,
            "bundle_summaries": self.bundle_summaries,
        }


class Pipeline:
    """Executes each algorithm in STAGE_ORDER (each algorithm runs the
    run_corpus aggregation scope first)."""

    def __init__(self, algorithms: list["StageAlgorithm"]) -> None:
        self.algorithms = algorithms
        # reruns collected by the most recent run() (explicit state, no getattr;
        # overwritten on every run)
        self.last_reruns: list["Trajectory"] = []

    @staticmethod
    def _flush(bundles: list["TrajectoryBundle"], ctx: "RunContext") -> None:
        """Persist all in-memory artifacts to the store (no-op without one).

        Called after every algorithm so partial runs survive a later crash;
        the end-of-run persistence in runtime.run_config re-saves the same
        content (idempotent overwrite), not double-writing.
        """
        if ctx.store is None:
            return
        for b in bundles:
            for stage, arts in b.artifacts.items():
                for name, art in arts.items():
                    ctx.store.save_artifact(b.trace_id, stage, name, art)

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
                algo_name = getattr(algo, "name", type(algo).__name__)
                t0 = time.time()
                # per-algorithm error isolation (review 2026-08-27 P1): one
                # crashing algorithm (e.g. binary_search's LLMError when the
                # judge answers neither upper nor lower) must not discard the
                # whole run's completed work. Granularity is the algorithm,
                # not the bundle: cross-trajectory algorithms (sbfl & co.)
                # own an internal run_corpus loop the pipeline cannot see
                # into; a class of missing-dependency crashes is instead
                # caught at config time via StageAlgorithm.requires.
                try:
                    algo.run_corpus(bundles, ctx)
                except Exception as e:  # noqa: BLE001 - isolation is the point
                    report.n_errors += 1
                    err = f"{type(e).__name__}: {e}"
                    report.stage_log.append(
                        f"{stage}/{algo_name} -> FAILED after "
                        f"{time.time() - t0:.3f}s: {err}"
                    )
                    for b in bundles:
                        if b.artifacts.get(stage, {}).get(algo_name) is None:
                            b.put(stage, algo_name, {
                                "status": "error",
                                "error": err[:500],
                                "isolated": True,
                            })
                else:
                    report.stage_log.append(
                        f"{stage}/{algo_name} "
                        f"-> {len(bundles)} bundles in {time.time() - t0:.3f}s"
                    )
                # incremental persistence: flush after every algorithm
                # attempt (success or failure), so a later crash never loses
                # earlier stages' artifacts (save_artifact is an idempotent
                # overwrite of <trace>/<stage>__<name>.json)
                self._flush(bundles, ctx)

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
        re-attributed and rerun again (nested recovery). ``max_rounds``: the
        verification round is capped at one round -- any value >= 1 enables
        it, larger values do not add rounds.

        Returns the **first-round bundles** (preserving full
        attribution/recovery artifacts), where each rerun trajectory
        additionally carries a ``recover/closed_loop`` artifact recording the
        verification verdict; the verification round's report is appended to
        reports (the stage 6 -> 3 loop). The verification round's judge
        verdict is recorded alongside under ``verify`` (the
        ``analyze/judge_eval`` score/summary re-read from the
        verification-round bundle matched by the rerun's trace_id), so the
        closed_loop artifact keeps judge-vs-outcome evidence on record for
        cross-audit -- the DRIFT "successful trajectory with a hidden
        erroneous step" contradiction is discoverable exactly here.
        """
        origin_bundles, report = self.run(trajectories, ctx)
        reports = [report]
        if self.last_reruns and max_rounds >= 1:
            rerun_by_origin: dict[str, Trajectory] = {}
            for t in self.last_reruns:
                origin = t.meta.get("rerun_of")
                if origin:
                    rerun_by_origin[origin] = t
            # the verification round feeds ONLY the reruns: originals without
            # a rerun (successes, or failures no recoverer picked up) were
            # already judged/attribution-processed in the first round --
            # re-running them re-judged identical work at full cost
            # (review 2026-08-27: 7 of 7 trajectories re-entered round 1
            # although only 6 carried a rerun). No-rerun origins keep their
            # first-round artifacts and simply carry no verification
            # evidence (verify is None, as below).
            current = list(rerun_by_origin.values())
            verify_bundles, verify_report = self.run(current, ctx)
            reports.append(verify_report)
            verify_bundle_by_trace = {b.trace_id: b for b in verify_bundles}
            improved = set()  # trace_ids of successful trajectories in the verification round
            for t in current:
                if t.outcome.success:
                    improved.add(t.trace_id)
            for b in origin_bundles:
                repl = rerun_by_origin.get(b.trace_id)
                verify: dict[str, Any] | None = None
                if repl is not None:
                    vb = verify_bundle_by_trace.get(repl.trace_id)
                    judge_art = vb.get("analyze", "judge_eval") if vb is not None else None
                    judge, judge_available = _judge_evidence(judge_art)
                    verify = {
                        "outcome_success": repl.trace_id in improved,
                        "judge": judge,
                        "judge_available": judge_available,
                    }
                b.put(
                    "recover",
                    "closed_loop",
                    {
                        "rerun_trace_id": repl.trace_id if repl else None,
                        "verified_improved": bool(repl and repl.trace_id in improved),
                        "verify": verify,
                    },
                )
        return origin_bundles, reports


def _judge_evidence(judge_art: Any) -> tuple[dict[str, Any] | None, bool]:
    """Verification-round judge evidence for the closed_loop artifact.

    An error-isolated algorithm leaves ``{"status": "error", ...}`` in place
    of its artifact (Pipeline.run); that must NOT count as "the judge ran"
    -- the artifact exists as a dict, but it carries no verdict, and
    judge_available=True with score=null would mislead exactly the
    judge-vs-outcome cross-audit this field exists for (verify-round
    review agent B, P1)."""
    if not isinstance(judge_art, dict) or judge_art.get("status") == "error":
        return None, False
    return (
        {"score": judge_art.get("score"), "summary": judge_art.get("summary")},
        True,
    )
