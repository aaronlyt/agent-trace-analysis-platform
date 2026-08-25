"""算法对比跑法 —— 同一轨迹集上运行多组 YAML 组合并产出对比表。

用法（CLI）::

    atap corpus --out runs/corpus/traces.jsonl          # 先生成频谱语料
    atap compare --config A.yaml --config B.yaml \
                 --traces runs/corpus/traces.jsonl --out runs/compare

契约：各配置必须跑**同一轨迹集**——显式 ``--traces`` 覆盖所有配置的
source，否则校验各配置 source 完全一致（不一致即报错，不静默猜测）。
指标：对注入故障 ground truth 的 step/agent/MAST 命中数、恢复数、闭环
改善数、LLM 调用数（按 tag 分桶——LLM 包装在装配层，算法无感知）。

注意：归因算法混排会改变 t* 选择（``hypotheses()`` 跨算法取 max
confidence）——对比实验的每个配置应只配一个归因算法（或明确接受混排
语义）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atap.core.config import PipelineConfig, load_config
from atap.llm import build_llm
from atap.runtime import run_config


class CountingLLM:
    """LLMClient 计数包装（装配层注入，算法无感知）。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def complete(self, messages, *, schema=None, model=None, tag=""):
        self.calls.append(tag or "(untagged)")
        return self.inner.complete(messages, schema=schema, model=model, tag=tag)


def evaluate_against_gt(bundles) -> dict[str, Any]:
    """对注入故障 GT 的命中统计（与 atap demo 同口径）。"""
    n_failed = step_hits = agent_hits = code_hits = 0
    recovered = closed_improved = 0
    per_fault: dict[str, dict[str, Any]] = {}
    for b in bundles:
        gt = b.trajectory.meta.get("injected_fault")
        if not gt:
            continue
        n_failed += 1
        hyps = b.hypotheses()
        top = max(hyps, key=lambda h: h.confidence) if hyps else None
        labels: list[dict] = []
        for name in ("mast_judge", "rule_pack"):
            art = b.get("classify", name)
            if isinstance(art, dict) and art.get("labels"):
                labels = art["labels"]
                break
            if isinstance(art, dict) and art.get("findings"):
                labels = art["findings"]
                break
        code = labels[0].get("code") or labels[0].get("mast_code") if labels else None
        hit_step = bool(top and top.step == gt["step"])
        hit_agent = bool(top and top.agent == gt["agent"])
        hit_code = bool(code and code == gt["mast_code"])
        step_hits += hit_step
        agent_hits += hit_agent
        code_hits += hit_code
        rec = False
        for art in b.artifacts.get("recover", {}).values():
            if isinstance(art, dict) and art.get("recovered"):
                rec = True
        recovered += rec
        loop = b.get("recover", "closed_loop", {})
        closed_improved += bool(loop.get("verified_improved"))
        per_fault[gt["kind"]] = {
            "gt_step": gt["step"],
            "pred_step": top.step if top else None,
            "pred_agent": top.agent if top else None,
            "hit_step": hit_step,
            "hit_agent": hit_agent,
            "hit_code": hit_code,
            "recovered": rec,
        }
    return {
        "n_failed": n_failed,
        "step_hits": step_hits,
        "agent_hits": agent_hits,
        "code_hits": code_hits,
        "recovered": recovered,
        "closed_loop_improved": closed_improved,
        "per_fault": per_fault,
    }


def run_compare(
    config_paths: list[str | Path],
    out: str | Path = "runs/compare",
    *,
    traces: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs: list[tuple[str, PipelineConfig]] = []
    for p in config_paths:
        cfg = load_config(str(p))
        cfgs.append((str(p), cfg))

    if traces is not None:
        for _, cfg in cfgs:
            cfg.source = {**cfg.source, "type": "jsonl", "path": str(traces)}
    else:
        sources = {json.dumps(cfg.source, sort_keys=True) for _, cfg in cfgs}
        if len(sources) > 1:
            raise ValueError(
                "各配置 source 不一致且未提供 --traces：对比必须跑同一轨迹集"
            )

    rows: list[dict[str, Any]] = []
    for path, cfg in cfgs:
        inner = build_llm(cfg.llm) if cfg.llm else None
        wrapper = CountingLLM(inner) if inner is not None else None
        run_dir = out_dir / (cfg.run_name or Path(path).stem)
        bundles, reports = run_config(cfg, run_dir, llm=wrapper)
        ev = evaluate_against_gt(bundles)
        by_tag: dict[str, int] = {}
        if wrapper is not None:
            for t in wrapper.calls:
                by_tag[t] = by_tag.get(t, 0) + 1
        row = {
            "config": path,
            "run_name": cfg.run_name,
            **{k: v for k, v in ev.items() if k != "per_fault"},
            "llm_calls": sum(by_tag.values()),
            "llm_calls_by_tag": by_tag,
            "closed_loop": cfg.closed_loop,
            "per_fault": ev["per_fault"],
        }
        rows.append(row)

    comparison = {
        "n_configs": len(rows),
        "traces": str(traces) if traces else cfgs[0][1].source.get("path"),
        "rows": rows,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_table(rows)
    return comparison


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'config':<28}{'step':>8}{'agent':>9}{'MAST':>8}"
          f"{'recover':>10}{'loop':>7}{'llm_calls':>11}")
    for r in rows:
        n = r["n_failed"]
        cells = (
            f"{r['step_hits']}/{n}", f"{r['agent_hits']}/{n}",
            f"{r['code_hits']}/{n}", f"{r['recovered']}/{n}",
            str(r["closed_loop_improved"]), str(r["llm_calls"]),
        )
        print(f"{r['run_name']:<28}{cells[0]:>8}{cells[1]:>9}{cells[2]:>8}"
              f"{cells[3]:>10}{cells[4]:>7}{cells[5]:>11}")
    print("对比明细 -> comparison.json（含 per-fault 命中与 LLM 调用分桶）")
