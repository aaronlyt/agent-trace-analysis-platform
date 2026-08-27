#!/usr/bin/env python
"""Full-pipeline real-LLM test (DeepSeek, with ground truth comparison) -- the real-model
supplementary test for phase-three acceptance. API keys come from environment variables
(never written to disk)::

    OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com \
        .venv/bin/python docs/realtest_deepseek.py [--model deepseek-v4-flash]

Two stacks (same population of 7 trajectories, runs/demo/traces.jsonl, generated first
with `atap demo`):
* stack-a phase-two regression: all_at_once + targeted_rerun -- verifies that the
  sandbox's LLM semantic feedback matching fixes the known limitation of 0/6 real-model
  recovery;
* stack-b phase three: binary_search + feedback_injection -- step-level comparison of
  bisection vs single pass + AgenTracer-style full re-solving recovery.
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
            # Reasoning-style output can exhaust the default 4096 cap and leave content empty (lesson from field testing)
            "max_completion_tokens": 8192,
        },
        "sandbox": {"type": "toy"},   # runtime auto-injects the LLM → fallback to feedback semantic matching
        "closed_loop": False,         # saves verification-round calls; recovery is judged from reruns
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
    print(f"[{stack_label}] elapsed {dt:.1f}s")
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
            print(f"  attribution: agent={top.agent} step={top.step} code={top.root_cause_code} "
                  f"conf={top.confidence} | step{'✓' if hs else '✗'} agent{'✓' if ha else '✗'}"
                  f" mast{'✓' if hc else '✗'}")
            print(f"        fix: {top.fix_suggestion[:140]}")
        else:
            print("  attribution: none")
        print(f"  recovery: rounds={rec.get('rounds')} recovered={rec.get('recovered')} "
              f"(fault_removed={b.reruns[0].meta.get('fault_removed') if b.reruns else '-'})")
    print("-" * 78)
    print(f"[{stack_label}] step {n_step}/6  agent {n_agent}/6  MAST {n_code}/6  recovery {n_rec}/6")
    for i, r in enumerate(reports):
        print(f"  round{i}: failures={r.n_failures} attributed={r.n_attributed} "
              f"reruns={r.n_reruns}(ok={r.n_rerun_success})")
    return n_step, n_agent, n_code, n_rec


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--stack", choices=["a", "b", "both"], default="both",
                    help="a=phase-two regression stack, b=phase-three stack (each can be rerun alone)")
    args = ap.parse_args()
    MODEL = args.model

    res_a = res_b = None
    if args.stack in ("a", "both"):
        t0 = time.time()
        cfg_a = config_from_dict(_stack(
            "deepseek-stack-a", "all_at_once", "targeted_rerun", {"max_rounds": 3}))
        bundles_a, reports_a = run_config(cfg_a, "runs/deepseek/stack_a")
        res_a = _report(f"stack-a phase-two regression all_at_once+targeted_rerun ({MODEL})",
                        bundles_a, reports_a, time.time() - t0, "targeted_rerun")

    if args.stack in ("b", "both"):
        t1 = time.time()
        cfg_b = config_from_dict(_stack(
            "deepseek-stack-b", "binary_search", "feedback_injection", {"max_rounds": 3}))
        bundles_b, reports_b = run_config(cfg_b, "runs/deepseek/stack_b")
        res_b = _report(f"stack-b phase-three binary_search+feedback_injection ({MODEL})",
                        bundles_b, reports_b, time.time() - t1, "feedback_injection")

    if res_a and res_b:
        print("=" * 78)
        print(f"Comparison ({MODEL}, same six-fault population): "
              f"all_at_once step {res_a[0]}/6 agent {res_a[1]}/6 recovery {res_a[3]}/6  vs  "
              f"binary_search step {res_b[0]}/6 agent {res_b[1]}/6 recovery {res_b[3]}/6")
        print("=" * 78)


if __name__ == "__main__":
    main()
