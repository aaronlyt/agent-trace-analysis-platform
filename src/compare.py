"""Algorithm comparison runner -- runs multiple YAML combos on the same
trajectory set and produces a comparison table.

Usage (CLI)::

    atap corpus --out runs/corpus/traces.jsonl          # generate the spectrum corpus first
    atap compare --config A.yaml --config B.yaml \
                 --traces runs/corpus/traces.jsonl --out runs/compare

Contract: all configs must run on the **same trajectory set** -- an explicit
``--traces`` overrides every config's source; otherwise the configs'
sources are validated to be identical (a mismatch is an error, no silent
guessing). Metrics: step/agent/MAST hit counts against the injected-fault
ground truth, recovery count, closed-loop improvement count, LLM call count
(bucketed by tag -- the LLM wrapper sits in the assembly layer, invisible
to algorithms).

Note: mixing attribution algorithms changes t* selection
(``hypotheses()`` takes max confidence across algorithms) -- each config in
a comparison experiment should configure exactly one attribution algorithm
(or explicitly accept the mixing semantics).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atap.core.config import PipelineConfig, load_config
from atap.llm import build_llm
from atap.log import get_logger
from atap.runtime import run_config

log = get_logger("compare")


class CountingLLM:
    """Counting wrapper for LLMClient (injected at the assembly layer,
    invisible to algorithms).

    attach_call_log is passed through to the inner client (silently
    skipped when the inner client lacks it -- auditing is best-effort and
    does not change call semantics).
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []

    def complete(self, messages, *, schema=None, model=None, tag=""):
        self.calls.append(tag or "(untagged)")
        return self.inner.complete(messages, schema=schema, model=model, tag=tag)

    def attach_call_log(self, path) -> None:
        if hasattr(self.inner, "attach_call_log"):
            self.inner.attach_call_log(path)


def evaluate_against_gt(bundles) -> dict[str, Any]:
    """Hit statistics against the injected-fault GT (same criteria as atap demo).

    ``top`` selection ties break on the earliest step ``(confidence, -step)``
    -- the same rule as the recover consumers, so evaluation and recovery
    act on one hypothesis. ``per_fault`` holds per-trace details keyed by
    trace_id (the record carries the fault ``kind``); ``per_kind``
    aggregates hits/total by fault kind and is what backs per-fault-kind
    conclusions (multiple traces may share one kind).
    """
    n_failed = step_hits = agent_hits = code_hits = 0
    recovered = closed_improved = 0
    per_fault: dict[str, dict[str, Any]] = {}
    per_kind: dict[str, dict[str, int]] = {}
    for b in bundles:
        gt = b.trajectory.meta.get("injected_fault")
        if not gt:
            continue
        n_failed += 1
        hyps = b.hypotheses()
        # eval keeps the algorithm's own ranking: max() returns the first
        # maximal element, i.e. the artifact's primary hypothesis (sbfl emits
        # a fixed prior confidence, so earliest-step tie-breaking here would
        # discard its suspiciousness ordering). The recovery side selects
        # (confidence, -step) deliberately -- the two scopes differ by design.
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
        per_fault[b.trace_id] = {
            "kind": gt["kind"],
            "gt_step": gt["step"],
            "pred_step": top.step if top else None,
            "pred_agent": top.agent if top else None,
            "hit_step": hit_step,
            "hit_agent": hit_agent,
            "hit_code": hit_code,
            "recovered": rec,
        }
        agg = per_kind.setdefault(
            gt["kind"],
            {"total": 0, "step_hits": 0, "agent_hits": 0, "code_hits": 0,
             "recovered": 0},
        )
        agg["total"] += 1
        agg["step_hits"] += hit_step
        agg["agent_hits"] += hit_agent
        agg["code_hits"] += hit_code
        agg["recovered"] += rec
    return {
        "n_failed": n_failed,
        "step_hits": step_hits,
        "agent_hits": agent_hits,
        "code_hits": code_hits,
        "recovered": recovered,
        "closed_loop_improved": closed_improved,
        "per_fault": per_fault,
        "per_kind": per_kind,
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
                "configs have inconsistent sources and no --traces given: comparison must run on the same trajectory set"
            )

    rows: list[dict[str, Any]] = []
    for path, cfg in cfgs:
        log.info("compare run: %s -> %s", path, cfg.run_name)
        inner = build_llm(cfg.llm) if cfg.llm else None
        wrapper = CountingLLM(inner) if inner is not None else None
        # run_name defaults to "run" and is never empty, so cfg.run_name is
        # always the directory name (no path-stem fallback needed)
        run_dir = out_dir / cfg.run_name
        if run_dir.exists():
            raise ValueError(
                f"run directory already exists: {run_dir} "
                f"(previous results would be overwritten; use a distinct run_name)"
            )
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
    print("comparison details -> comparison.json (per-fault hits and LLM call buckets)")
