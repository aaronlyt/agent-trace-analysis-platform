"""runtime -- composition root.

Assembles a PipelineConfig into runnable objects: source → trajectories,
registry → algorithms, llm/io config → RunContext. core imports no
implementation; assembly happens only here and in cli -- this is the
layering invariant (checked by tests/test_invariants.py).
"""

from __future__ import annotations

import random
from pathlib import Path

from atap.core.config import ConfigError, PipelineConfig, validate_against_registry
from atap.core.context import RunContext
from atap.core.pipeline import Pipeline, PipelineReport
from atap.core.registry import create
from atap.io import build_source, build_store
from atap.llm import build_llm
from atap.log import attach_run_log, get_logger

log = get_logger("runtime")


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    """Config → Pipeline (validated against the registry first; errors list the available algorithms)."""
    validate_against_registry(cfg)
    algorithms = [
        create(spec.stage, spec.name, **spec.params)
        for spec in cfg.algorithms_in_order()
    ]
    return Pipeline(algorithms)


def build_context(
    cfg: PipelineConfig,
    run_dir: str | Path,
    *,
    llm: object | None = None,
) -> RunContext:
    """Config → RunContext. ``llm`` can be injected to replace construction
    (used by the compare counting wrapper)."""
    run_dir = str(run_dir)
    ctx = RunContext(
        llm=llm if llm is not None else build_llm(cfg.llm),
        store=build_store(cfg.store, run_dir),
        rng=random.Random(cfg.seed),
        run_dir=run_dir,
    )
    if cfg.sandbox:
        kind = cfg.sandbox.get("type")
        if kind == "toy":
            from atap.sandbox import ToySandbox

            # inject the LLM: sandbox feedback consumption is upgraded to
            # "keywords first, LLM semantics as fallback" (fixes the known
            # limitation of 0/6 recovery with free-text feedback from a real
            # model, see plan.md)
            ctx.env = ToySandbox(llm=ctx.llm)
        else:
            raise ConfigError(f"unknown sandbox type: {kind!r} (available: toy)")
    return ctx


def run_config(
    cfg: PipelineConfig,
    run_dir: str | Path,
    *,
    trajectories=None,
    llm: object | None = None,
) -> tuple[list, list[PipelineReport]]:
    """Full execution: read trajectories → orchestrate (optional closed
    loop) → persist artifacts/reports.

    report.json: the persisted counts are round-inclusive totals (with
    ``closed_loop`` on, ``n_traces``/``n_failures`` sum the initial round
    and the verification round); ``n_traces_by_round`` /
    ``n_failures_by_round`` break them down per round.

    Unified logging: this function automatically attaches
    ``<run_dir>/run.log`` (process log) and ``<run_dir>/llm_calls.jsonl``
    (LLM call audit, when the client supports attach_call_log -- both Fake
    and OpenAI do; the compare counting wrapper passes it through to the
    inner client).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    attach_run_log(run_dir / "run.log")
    if trajectories is None:
        trajectories = build_source(cfg.source).load()
    pipeline = build_pipeline(cfg)
    ctx = build_context(cfg, run_dir, llm=llm)
    if ctx.llm is not None and hasattr(ctx.llm, "attach_call_log"):
        ctx.llm.attach_call_log(run_dir / "llm_calls.jsonl")
    log.info(
        "run start: run=%s traces=%d failures=%d closed_loop=%s out=%s",
        cfg.run_name,
        len(trajectories),
        sum(0 if t.outcome.success else 1 for t in trajectories),
        cfg.closed_loop,
        run_dir,
    )
    if cfg.closed_loop:
        bundles, reports = pipeline.run_closed_loop(trajectories, ctx, max_rounds=1)
    else:
        bundles, report = pipeline.run(trajectories, ctx)
        reports = [report]

    # persist artifacts (bundle summaries + report)
    if ctx.store is not None:
        for b in bundles:
            for stage, arts in b.artifacts.items():
                for name, art in arts.items():
                    ctx.store.save_artifact(b.trace_id, stage, name, art)
        merged = PipelineReport(run_name=cfg.run_name)
        for r in reports:
            merged.n_traces += r.n_traces
            merged.n_failures += r.n_failures
            merged.n_attributed += r.n_attributed
            merged.n_reruns += r.n_reruns
            merged.n_rerun_success += r.n_rerun_success
            merged.n_errors += r.n_errors
            merged.stage_log.extend(r.stage_log)
            merged.bundle_summaries.extend(r.bundle_summaries)
        payload = merged.to_dict()
        # n_traces/n_failures above are round-inclusive totals (initial +
        # verification round): the *_by_round lists break them down per round
        # so a closed-loop run is not double-counted as a bigger corpus
        payload["n_traces_by_round"] = [r.n_traces for r in reports]
        payload["n_failures_by_round"] = [r.n_failures for r in reports]
        ctx.store.save_report("report.json", payload)
    for line in merged_stage_log(reports):
        log.info("stage %s", line)
    log.info(
        "run finished: run=%s rounds=%d traces=%d failures=%d attributed=%d "
        "reruns=%d(ok=%d)",
        cfg.run_name,
        len(reports),
        sum(r.n_traces for r in reports),
        sum(r.n_failures for r in reports),
        sum(r.n_attributed for r in reports),
        sum(r.n_reruns for r in reports),
        sum(r.n_rerun_success for r in reports),
    )
    return bundles, reports


def merged_stage_log(reports: list[PipelineReport]) -> list[str]:
    return [line for r in reports for line in r.stage_log]
