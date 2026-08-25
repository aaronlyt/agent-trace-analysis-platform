#!/usr/bin/env python
"""真实 LLM 全链路测试（带 ground truth 对照）——供配额重置后一键复跑。

用法（需要 OpenRouter 密钥；免费档每日 50 次，本脚本 ≤25 次）::

    OPENAI_API_KEY=sk-or-v1-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
        .venv/bin/python realtest_nemotron.py [--closed-loop]

默认关闭闭环验证轮（省 7+ 次调用）：核心路径 = 7×judge_eval + 6×mast_judge
+ 6×all_at_once ≈ 19 次（另留解析重试余量）。--closed-loop 开启后与
configs/pipeline_nemotron.yaml 等价（≈26+ 次）。
"""

from __future__ import annotations

import argparse
import time

from atap.core.config import config_from_dict
from atap.runtime import run_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-loop", action="store_true", help="开启闭环验证轮（多 ~7 次调用）")
    args = ap.parse_args()

    cfg = config_from_dict(
        {
            "run_name": "nemotron-realtest",
            "seed": 7,
            "source": {"type": "jsonl", "path": "runs/demo/traces.jsonl"},
            "llm": {
                "type": "openai",
                "model": "nvidia/nemotron-3.5-lightning:free",
                "temperature": 0.0,
                "request_interval": 3.0,
            },
            "sandbox": {"type": "toy"},
            "closed_loop": args.closed_loop,
            "stages": {
                "represent": ["canonical_events", "ssf"],
                "analyze": [{"name": "judge_eval", "params": {"only_failures": False}}],
                "classify": ["mast_judge"],
                "attribute": ["all_at_once"],
                "recover": [{"name": "targeted_rerun", "params": {"max_rounds": 5}}],
            },
        }
    )
    t0 = time.time()
    bundles, reports = run_config(cfg, "runs/nemotron")
    dt = time.time() - t0

    print("=" * 78)
    print(f"真实 LLM 全链路  model=nvidia/nemotron-3.5-lightning:free  "
          f"closed_loop={args.closed_loop}  用时 {dt:.1f}s")
    print("=" * 78)
    n_step = n_agent = n_code = n_rec = 0
    for b in bundles:
        t = b.trajectory
        gt = t.meta.get("injected_fault")
        if not gt:
            v = b.get("analyze", "judge_eval", {})
            print(f"[{b.trace_id}] OK   judge_score={v.get('score', '-')}")
            continue
        hyps = b.hypotheses()
        top = max(hyps, key=lambda h: (h.confidence, -h.step)) if hyps else None
        labels = b.get("classify", "mast_judge", {}).get("labels", [])
        rec = b.get("recover", "targeted_rerun", {})
        hs = bool(top and top.step == gt["step"])
        ha = bool(top and top.agent == gt["agent"])
        hc = bool(labels and any(l["code"] == gt["mast_code"] for l in labels))
        n_step += hs
        n_agent += ha
        n_code += hc
        n_rec += bool(rec.get("recovered"))
        print(f"[{b.trace_id}] FAIL")
        print(f"  gt    : {gt['kind']} @step{gt['step']} ({gt['mast_code']})")
        if top:
            print(f"  归因  : agent={top.agent} step={top.step} code={top.root_cause_code} "
                  f"conf={top.confidence}")
            print(f"         reason: {top.root_cause[:150]}")
            print(f"         fix   : {top.fix_suggestion[:150]}")
        else:
            print("  归因  : 无")
        codes = [l["code"] for l in labels]
        print(f"  MAST  : {codes}  (invalid={b.get('classify', 'mast_judge', {}).get('invalid_codes')})")
        print(f"  恢复  : rounds={rec.get('rounds')} recovered={rec.get('recovered')}")
        print(f"  命中  : step{'✓' if hs else '✗'} agent{'✓' if ha else '✗'} mast{'✓' if hc else '✗'}")
    print("-" * 78)
    print(f"归因命中: step {n_step}/6  agent {n_agent}/6  MAST(标签集合) {n_code}/6  恢复 {n_rec}/6")
    for i, r in enumerate(reports):
        print(f"round{i}: traces={r.n_traces} failures={r.n_failures} attributed={r.n_attributed} "
              f"reruns={r.n_reruns}(ok={r.n_rerun_success})")


if __name__ == "__main__":
    main()
