"""atap demo —— 离线全链路演示（FakeLLM 确定性判官，零网络、可复现）。

流程：沙盒生成轨迹群体（含六种注入故障）→ 写 JSONL（采集层产物）→
六阶段 pipeline（R0 → SSF → judge 评测 → MAST 打标 → All-at-Once 归因 →
定向重跑）→ 闭环验证 → 打印「ground truth vs 归因」对照与恢复结果。
"""

from __future__ import annotations

import json
from pathlib import Path


def run_demo(seed: int = 7, out: str = "runs/demo") -> None:
    from atap.core.config import config_from_dict
    from atap.runtime import run_config
    from atap.sandbox import ToySandbox

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces = ToySandbox().generate_population(seed)
    traces_jsonl = out_dir / "traces.jsonl"
    with traces_jsonl.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    cfg = config_from_dict(
        {
            "run_name": f"demo-offline-seed{seed}",
            "seed": seed,
            "source": {"type": "jsonl", "path": str(traces_jsonl)},
            "llm": {"type": "fake"},
            "sandbox": {"type": "toy"},
            "closed_loop": True,
            "stages": {
                "represent": ["canonical_events", "ssf"],
                "analyze": ["judge_eval"],
                "classify": ["mast_judge"],
                "attribute": ["all_at_once"],
                "recover": ["targeted_rerun"],
            },
        }
    )
    bundles, reports = run_config(cfg, out_dir)

    n_ok = sum(1 for b in bundles if b.succeeded)
    print("=" * 78)
    print(f"atap 离线全链路演示  seed={seed}  traces={len(bundles)}  "
          f"({n_ok} 成功 + {len(bundles) - n_ok} 故障注入)")
    print("=" * 78)
    n_hit_step = n_hit_agent = n_hit_code = n_recovered = n_failed = 0
    for b in bundles:
        t = b.trajectory
        gt = t.meta.get("injected_fault")
        head = f"[{b.trace_id}] {'OK ' if b.succeeded else 'FAIL'}"
        if b.succeeded:
            verdict = b.get("analyze", "judge_eval", {})
            print(f"{head}  judge_score={verdict.get('score', '-')}  ({t.outcome.note[:44]})")
            continue
        n_failed += 1
        hyps = b.hypotheses()
        top = max(hyps, key=lambda h: h.confidence) if hyps else None
        labels = b.get("classify", "mast_judge", {}).get("labels", [])
        rec = b.get("recover", "targeted_rerun", {})
        loop = b.get("recover", "closed_loop", {})
        hit_step = bool(top and gt and top.step == gt["step"])
        hit_agent = bool(top and gt and top.agent == gt["agent"])
        hit_code = bool(labels and gt and labels[0]["code"] == gt["mast_code"])
        n_hit_step += hit_step
        n_hit_agent += hit_agent
        n_hit_code += hit_code
        recovered = bool(rec.get("recovered"))
        n_recovered += recovered
        print(
            f"{head}  gt={gt['kind']}@step{gt['step']}({gt['mast_code']})\n"
            f"        归因: agent={top.agent if top else '-'} step={top.step if top else '-'} "
            f"code={top.root_cause_code if top else '-'} conf={top.confidence if top else '-'} "
            f"| step{'✓' if hit_step else '✗'} agent{'✓' if hit_agent else '✗'} "
            f"mast{'✓' if hit_code else '✗'}\n"
            f"        恢复: rounds={rec.get('rounds')} recovered={recovered} "
            f"闭环验证改善={loop.get('verified_improved')}"
        )
    print("-" * 78)
    print(
        f"归因命中: step {n_hit_step}/{n_failed}  agent {n_hit_agent}/{n_failed}  "
        f"MAST {n_hit_code}/{n_failed}  恢复 {n_recovered}/{n_failed}"
    )
    for i, r in enumerate(reports):
        print(f"round{i}: traces={r.n_traces} failures={r.n_failures} "
              f"attributed={r.n_attributed} reruns={r.n_reruns}(ok={r.n_rerun_success})")
    print(f"产物目录: {out_dir}/artifacts（report.json + 每轨迹各阶段 JSON）")
