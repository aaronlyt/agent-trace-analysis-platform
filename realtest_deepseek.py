#!/usr/bin/env python
"""真实 LLM 全链路测试（DeepSeek，带 ground truth 对照）——阶段三验收的
真模型补测。密钥走环境变量（不落盘）::

    OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com \
        .venv/bin/python realtest_deepseek.py [--model deepseek-v4-flash]

两组栈（同一 7 条轨迹群体，runs/demo/traces.jsonl，先 `atap demo` 生成）：
* stack-a 阶段二回归：all_at_once + targeted_rerun——验证沙盒 LLM 语义
  反馈匹配修复真模型恢复 0/6 的已知限制；
* stack-b 阶段三：binary_search + feedback_injection——二分 vs 单遍的
  step 级对比 + AgenTracer 式全量再求解恢复。
"""

from __future__ import annotations

import argparse
import time

from atap.core.config import config_from_dict
from atap.runtime import run_config


def _stack(name, attributor, recover, recover_params):
    return {
        "run_name": name,
        "seed": 7,
        "source": {"type": "jsonl", "path": "runs/demo/traces.jsonl"},
        "llm": {
            "type": "openai",
            "model": MODEL,
            "temperature": 0.0,
            "request_interval": 1.0,
            # 思考型输出可能耗尽默认 4096 上限导致 content 为空（实测教训）
            "max_completion_tokens": 8192,
        },
        "sandbox": {"type": "toy"},   # runtime 自动注入 LLM → 反馈语义匹配兜底
        "closed_loop": False,         # 省验证轮调用；恢复判定看 reruns
        "stages": {
            "represent": ["canonical_events", "ssf"],
            "analyze": [{"name": "judge_eval", "params": {"only_failures": False}}],
            "classify": ["mast_judge"],
            "attribute": [attributor],
            "recover": [{"name": recover, "params": recover_params}],
        },
    }


def _report(stack_label, bundles, reports, dt, recover_algo):
    print("=" * 78)
    print(f"[{stack_label}] 用时 {dt:.1f}s")
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
        rec = b.get("recover", recover_algo, {})
        hs = bool(top and top.step == gt["step"])
        ha = bool(top and top.agent == gt["agent"])
        hc = bool(labels and any(l["code"] == gt["mast_code"] for l in labels))
        n_step += hs
        n_agent += ha
        n_code += hc
        n_rec += bool(rec.get("recovered"))
        print(f"[{b.trace_id}] FAIL  gt={gt['kind']}@step{gt['step']}")
        if top:
            print(f"  归因: agent={top.agent} step={top.step} code={top.root_cause_code} "
                  f"conf={top.confidence} | step{'✓' if hs else '✗'} agent{'✓' if ha else '✗'}"
                  f" mast{'✓' if hc else '✗'}")
            print(f"        fix: {top.fix_suggestion[:140]}")
        else:
            print("  归因: 无")
        print(f"  恢复: rounds={rec.get('rounds')} recovered={rec.get('recovered')} "
              f"(fault_removed={b.reruns[0].meta.get('fault_removed') if b.reruns else '-'})")
    print("-" * 78)
    print(f"[{stack_label}] step {n_step}/6  agent {n_agent}/6  MAST {n_code}/6  恢复 {n_rec}/6")
    for i, r in enumerate(reports):
        print(f"  round{i}: failures={r.n_failures} attributed={r.n_attributed} "
              f"reruns={r.n_reruns}(ok={r.n_rerun_success})")
    return n_step, n_agent, n_code, n_rec


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--stack", choices=["a", "b", "both"], default="both",
                    help="a=阶段二回归栈，b=阶段三栈（可单独重跑）")
    args = ap.parse_args()
    MODEL = args.model

    res_a = res_b = None
    if args.stack in ("a", "both"):
        t0 = time.time()
        cfg_a = config_from_dict(_stack(
            "deepseek-stack-a", "all_at_once", "targeted_rerun", {"max_rounds": 3}))
        bundles_a, reports_a = run_config(cfg_a, "runs/deepseek/stack_a")
        res_a = _report(f"stack-a 阶段二回归 all_at_once+targeted_rerun ({MODEL})",
                        bundles_a, reports_a, time.time() - t0, "targeted_rerun")

    if args.stack in ("b", "both"):
        t1 = time.time()
        cfg_b = config_from_dict(_stack(
            "deepseek-stack-b", "binary_search", "feedback_injection", {"max_rounds": 3}))
        bundles_b, reports_b = run_config(cfg_b, "runs/deepseek/stack_b")
        res_b = _report(f"stack-b 阶段三 binary_search+feedback_injection ({MODEL})",
                        bundles_b, reports_b, time.time() - t1, "feedback_injection")

    if res_a and res_b:
        print("=" * 78)
        print(f"对比（{MODEL}，同一六故障群体）: "
              f"all_at_once step {res_a[0]}/6 agent {res_a[1]}/6 恢复 {res_a[3]}/6  vs  "
              f"binary_search step {res_b[0]}/6 agent {res_b[1]}/6 恢复 {res_b[3]}/6")
        print("=" * 78)


if __name__ == "__main__":
    main()
