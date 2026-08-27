"""atap demo -- offline end-to-end demo (FakeLLM deterministic judge, zero
network, reproducible).

Flow: the sandbox generates a trajectory population (with six injected
faults) → write JSONL (collection-layer artifact) → six-stage pipeline
(R0 → SSF → judge evaluation → MAST labeling → All-at-Once attribution →
targeted rerun) → closed-loop verification → print the "ground truth vs
attribution" comparison and recovery results.
"""

from __future__ import annotations

import json
from pathlib import Path

from atap.log import get_logger

log = get_logger("demo")


def run_demo(seed: int = 7, out: str = "runs/demo") -> None:
    from atap.core.config import config_from_dict
    from atap.runtime import run_config
    from atap.sandbox import ToySandbox

    log.info("demo start: seed=%s out=%s", seed, out)
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
    print(f"atap offline end-to-end demo  seed={seed}  traces={len(bundles)}  "
          f"({n_ok} successes + {len(bundles) - n_ok} fault injections)")
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
        if not gt:
            # failure without injected-fault GT: nothing to compare against
            # (same guard as compare.evaluate_against_gt)
            continue
        n_failed += 1
        hyps = b.hypotheses()
        # keep the algorithm's own ranking (see compare.evaluate_against_gt);
        # the recovery side's (confidence, -step) tie-break is a separate scope
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
            f"        attribution: agent={top.agent if top else '-'} step={top.step if top else '-'} "
            f"code={top.root_cause_code if top else '-'} conf={top.confidence if top else '-'} "
            f"| step{'✓' if hit_step else '✗'} agent{'✓' if hit_agent else '✗'} "
            f"mast{'✓' if hit_code else '✗'}\n"
            f"        recovery: rounds={rec.get('rounds')} recovered={recovered} "
            f"closed-loop verified improvement={loop.get('verified_improved')}"
        )
    print("-" * 78)
    print(
        f"attribution hits: step {n_hit_step}/{n_failed}  agent {n_hit_agent}/{n_failed}  "
        f"MAST {n_hit_code}/{n_failed}  recovery {n_recovered}/{n_failed}"
    )
    for i, r in enumerate(reports):
        print(f"round{i}: traces={r.n_traces} failures={r.n_failures} "
              f"attributed={r.n_attributed} reruns={r.n_reruns}(ok={r.n_rerun_success})")
    print(f"artifacts directory: {out_dir}/artifacts (report.json + per-trajectory per-stage JSON)")
    log.info(
        "demo finished: step=%d/%d agent=%d/%d mast=%d/%d recovered=%d/%d",
        n_hit_step, n_failed, n_hit_agent, n_failed, n_hit_code, n_failed,
        n_recovered, n_failed,
    )
