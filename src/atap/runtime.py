"""runtime —— 装配根（composition root）。

把 PipelineConfig 装配为可运行对象：source → 轨迹、registry → 算法、
llm/io 配置 → RunContext。core 不 import 任何实现，装配只发生在这里与
cli——这是分层不变量（tests/test_invariants.py 校验）。
"""

from __future__ import annotations

import random
from pathlib import Path

from atap.core.config import PipelineConfig, validate_against_registry
from atap.core.context import RunContext
from atap.core.pipeline import Pipeline, PipelineReport
from atap.core.registry import create
from atap.io import build_source, build_store
from atap.llm import build_llm


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    """配置 → Pipeline（先对照注册表校验，报错带可用算法清单）。"""
    validate_against_registry(cfg)
    algorithms = [
        create(spec.stage, spec.name, **spec.params)
        for spec in cfg.algorithms_in_order()
    ]
    return Pipeline(algorithms)


def build_context(cfg: PipelineConfig, run_dir: str | Path) -> RunContext:
    run_dir = str(run_dir)
    ctx = RunContext(
        llm=build_llm(cfg.llm),
        store=build_store(cfg.store, run_dir),
        rng=random.Random(cfg.seed),
        run_dir=run_dir,
    )
    if cfg.sandbox:
        kind = cfg.sandbox.get("type")
        if kind == "toy":
            from atap.sandbox import ToySandbox

            ctx.env = ToySandbox()
        else:
            raise ValueError(f"未知 sandbox type：{kind!r}（可用：toy）")
    return ctx


def run_config(
    cfg: PipelineConfig,
    run_dir: str | Path,
    *,
    trajectories=None,
) -> tuple[list, list[PipelineReport]]:
    """完整执行：读轨迹 → 编排（可选闭环）→ 产物/报告落盘。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if trajectories is None:
        trajectories = build_source(cfg.source).load()
    pipeline = build_pipeline(cfg)
    ctx = build_context(cfg, run_dir)
    if cfg.closed_loop:
        bundles, reports = pipeline.run_closed_loop(trajectories, ctx, max_rounds=1)
    else:
        bundles, report = pipeline.run(trajectories, ctx)
        reports = [report]

    # 产物落盘（bundle 汇总 + 报告）
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
            merged.stage_log.extend(r.stage_log)
            merged.bundle_summaries.extend(r.bundle_summaries)
        ctx.store.save_report("report.json", merged)
    return bundles, reports
