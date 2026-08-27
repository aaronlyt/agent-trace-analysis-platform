#!/usr/bin/env python
"""Pre-launch real-model test - automated audit (step 1 of §3 in docs/plan_pre_launch_realtest.md).

Reads one or more run directories (runs/final/<tier>, containing llm_calls.jsonl + artifacts/),
outputs a markdown audit report and scores it against P0/P2 thresholds; GT is matched
automatically by trace_id against traces.jsonl from runs/demo and runs/corpus (the two
sources are merged; unmatched entries skip scoring).

Usage (from the repo root)::

    .venv/bin/python docs/realtest_audit.py runs/final/final-smoke [runs/final/final-v3 ...]
    .venv/bin/python docs/realtest_audit.py runs/final/final-smoke --sample   # sample 3 responses per tag for manual spot-check

Exit code: 1 if any prompt-side leakage hit (P0-2 hard threshold), otherwise 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# P0-2/P0-3: the 8 fault-kind names + 3 ground-truth field names. This script is a
# docs/ audit tool and never enters any judge prompt (the anti-leakage iron rule
# constrains algorithm-side prompt content).
FAULT_WORDS = [
    "malformed_tool_call", "step_repetition", "info_withholding",
    "premature_termination", "ungrounded_citation", "disobey_task_spec",
    "retrieval_detour", "agent_deadlock",
]
GT_FIELDS = ["injected_fault", "ground_truth", "qrels"]
LEAK_WORDS = FAULT_WORDS + GT_FIELDS

# Per-tier call plan (offline wc -l baseline, ±15% drift threshold); keys are run
# directory name substrings. Only tags with a solid formula are listed; unknown tags
# are counted but not compared. F=feedback_match semantic fallback is a variable 0~n,
# so no drift threshold is set.
# 校准（2026-08-26 深度求索真实运行对表）：mast_judge 只打标失败轨迹且验证轮
# 复用结果——corpus 档按 18（失败数）计而非 24（轨迹数）；dover 的
# proposer/classify/intervene 按干预尝试数（27，个别轨迹多轮尝试）而非 18。
TAG_PLAN = {
    "smoke-corpus": {"judge_eval": 48, "mast_judge": 18, "all_at_once": 18},
    "smoke": {"judge_eval": 14, "mast_judge": 6, "all_at_once": 6},
    "v3": {"judge_eval": 48, "mast_judge": 18},
    "dover": {"all_at_once": 18, "dover_segment": 18, "dover_milestone": 18},
    "v4": {"judge_eval": 48, "mast_judge": 18},
    "tree": {"tree_diagnosis_stage": 18, "tree_diagnosis_drill": 18},
    "chief": {"chief_oracle": 18, "chief_eval": 18, "chief_localize": 18},
    "claim": {},
}
PLAN_TOTAL = {  # per-tier record-count baseline (for drift comparison; None=no comparison)
    "smoke-corpus": 102, "smoke": 26, "v3": 147, "dover": 108, "v4": 42,
    "tree": 36, "chief": 54, "claim": 57,
}


def load_gt() -> dict:
    gt = {}
    for p in ("runs/demo/traces.jsonl", "runs/corpus/traces.jsonl"):
        fp = os.path.join(REPO, p)
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            t = json.loads(line)
            tid = t.get("trace_id") or t.get("meta", {}).get("trace_id")
            f = t.get("meta", {}).get("injected_fault")
            if tid and f:
                gt[tid] = f
    return gt


def pct(vals, q):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def scan_text(text, words):
    hits = [w for w in words if w in text]
    return hits


def audit_run(run_dir, gt, sample=False):
    name = os.path.basename(os.path.normpath(run_dir))
    calls_path = os.path.join(run_dir, "llm_calls.jsonl")
    if not os.path.exists(calls_path):
        return f"## {name}\n\n(llm_calls.jsonl missing, skipped)\n", False
    recs = [json.loads(l) for l in open(calls_path) if l.strip()]

    lines = [f"## {name}", ""]

    # --- P0-2 / P0-3 leakage scan ---
    # The iron rule protects judge/attributor prompts; feedback_match is a sandbox
    # environment-side matcher whose prompt's fault specification comes from the
    # environment's own injection (argued explicitly in policy.py _feedback_addresses),
    # so it is listed separately for review and excluded from the hard threshold.
    # All other tags have zero tolerance.
    ENV_SIDE_TAGS = {"feedback_match"}
    prompt_hits, env_hits, resp_hits = [], [], []
    for i, r in enumerate(recs):
        tag = r.get("tag")
        msg_text = " ".join(m.get("content", "") or "" for m in r.get("messages", []))
        for w in scan_text(msg_text, LEAK_WORDS):
            (env_hits if tag in ENV_SIDE_TAGS else prompt_hits).append((i, tag, w))
        resp = r.get("response") or ""
        for w in scan_text(resp, LEAK_WORDS):
            resp_hits.append((i, tag, w))
    p0_pass = not prompt_hits
    lines.append(f"- **P0-2 prompt leakage (judge-type tags, hard threshold)**: {'✅ 0 hits' if p0_pass else '🔴 ' + str(len(prompt_hits)) + ' hits'}")
    if prompt_hits:
        lines.append(f"  - details (record no./tag/word): {prompt_hits[:10]}")
    if env_hits:
        lines.append(f"- Environment-side known source (feedback_match carries the fault specification, not judge leakage; see the rationale in policy.py): {len(env_hits)} records")
    if resp_hits:
        lines.append(f"- **P0-3 response hits (manual review list, not auto-fail)**: {len(resp_hits)} records")
        lines.append(f"  - details: {resp_hits[:10]}")
    lines.append("")

    # --- P2 call metrics ---
    n = len(recs)
    ok = sum(1 for r in recs if r.get("ok"))
    http = sum(r.get("http_requests") or 1 for r in recs)
    repair = sum(1 for r in recs if (r.get("http_requests") or 1) > 1)
    lat = [r.get("latency_ms") or 0 for r in recs]
    up = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in recs)
    uc = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in recs)
    ut = sum((r.get("usage") or {}).get("total_tokens") or 0 for r in recs)
    lines.append("| metric | value | threshold |")
    lines.append("|---|---|---|")
    lines.append(f"| call records / ok-rate | {n} / {ok / n:.1%} | ≥95% |")
    lines.append(f"| HTTP amplification (Σhttp/record) | {http / n:.2f} | ≤1.15 |")
    lines.append(f"| parse repair rate (share with http>1) | {repair / n:.1%} | ≤10% |")
    lines.append(f"| latency p50/p95/max | {pct(lat, .5) / 1000:.1f}s / {pct(lat, .95) / 1000:.1f}s / {max(lat) / 1000:.1f}s | — |")
    lines.append(f"| tokens (prompt/completion/total) | {up:,} / {uc:,} / {ut:,} | billing record |")
    lines.append("")

    # --- tag distribution vs plan ---
    tags = Counter(r.get("tag", "?") for r in recs)
    plan_key = next((k for k in TAG_PLAN if k in name), None)
    lines.append("| tag | actual | planned | drift |")
    lines.append("|---|---|---|---|")
    for t, c in sorted(tags.items()):
        plan = TAG_PLAN.get(plan_key, {}).get(t)
        if plan:
            drift = (c - plan) / plan
            mark = "✅" if abs(drift) <= 0.15 else f"⚠️ {drift:+.0%}"
            lines.append(f"| {t} | {c} | {plan} | {mark} |")
        else:
            lines.append(f"| {t} | {c} | — | — |")
    total_plan = PLAN_TOTAL.get(plan_key)
    if total_plan:
        lines.append(f"| **total** | {n} | {total_plan}+F | {(n - total_plan):+d} vs baseline |")
    lines.append("")

    # --- GT hit-rate scoring ---
    art_root = os.path.join(run_dir, "artifacts")
    rows = []  # (metric, hit, total)
    detail = []
    if os.path.isdir(art_root):
        for td in sorted(os.listdir(art_root)):
            g = gt.get(td)
            if not g:
                continue
            base = os.path.join(art_root, td)

            def load(stage):
                p = glob.glob(os.path.join(base, f"{stage}__*.json"))
                return json.load(open(p[0])) if p else None

            # Attribution: first hypothesis of any attribute artifact
            for p in sorted(glob.glob(os.path.join(base, "attribute__*.json"))):
                algo = os.path.basename(p)[len("attribute__"):-5]
                d = json.load(open(p))
                hyps = d.get("hypotheses") or []
                if not hyps:
                    continue
                h = hyps[0]
                try:
                    s = int(h.get("step"))
                except (TypeError, ValueError):
                    s = None
                rows.append((f"{algo} step=GT", s == g["step"], None))
                rows.append((f"{algo} agent=GT", h.get("agent") == g["agent"], None))
                if h.get("root_cause_code"):
                    rows.append((f"{algo} code=GT", h.get("root_cause_code") == g["mast_code"], None))
                break  # only one attribute algorithm is configured per tier

            # claim coverage: whether the GT step falls inside some claim's [introduced, first_effective] window
            led_p = os.path.join(base, "represent__claim_ledger.json")
            if os.path.exists(led_p):
                led = json.load(open(led_p))
                spans = [(c.get("introduced_step"), c.get("first_effective_step")) for c in led.get("claims", [])]
                cov = any(a is not None and a <= g["step"] <= (b if b is not None else g["step"]) for a, b in spans)
                rows.append(("claim coverage window contains GT step", cov, None))

            # MAST primary label
            cj = load("classify")
            if cj and "fusion" in cj:
                rows.append(("mast_judge primary label=GT", (cj["fusion"][0].get("mast") == g["mast_code"]), None))
                rows.append(("mast_judge any label=GT", any(x.get("mast") == g["mast_code"] for x in cj["fusion"]), None))

            # dover: first attempt's mistake and verdict
            dv = json.load(open(os.path.join(base, "recover__dover.json"))) if os.path.exists(os.path.join(base, "recover__dover.json")) else None
            if dv:
                att = (dv.get("attempts") or [{}])[0]
                mk = att.get("mistake") or {}
                try:
                    ms = int(mk.get("step"))
                except (TypeError, ValueError):
                    ms = None
                rows.append(("dover mistake step=GT", ms == g["step"], None))
                rows.append(("dover mistake agent=GT", mk.get("agent") == g["agent"], None))
                rows.append(("dover verdict=Validated", att.get("verdict") == "Validated", None))
                rows.append(("dover recovered", bool(dv.get("recovered")), None))

            # Recovery: closed_loop first, otherwise each recover artifact
            cl = json.load(open(os.path.join(base, "recover__closed_loop.json"))) if os.path.exists(os.path.join(base, "recover__closed_loop.json")) else None
            if cl is not None and "verified_improved" in cl:
                rows.append(("closed-loop verified_improved", bool(cl.get("verified_improved")), None))
            else:
                for p in glob.glob(os.path.join(base, "recover__*.json")):
                    d = json.load(open(p))
                    if "recovered" in d:
                        rows.append((f"{os.path.basename(p)[8:-5]} recovered", bool(d["recovered"]), None))

    if rows:
        agg = {}
        for m, hit, _ in rows:
            h, t = agg.get(m, (0, 0))
            agg[m] = (h + int(bool(hit)), t + 1)
        lines.append("| GT comparison metric | hits/total |")
        lines.append("|---|---|")
        for m, (h, t) in agg.items():
            lines.append(f"| {m} | **{h}/{t}** |")
    else:
        lines.append("(no artifacts with GT to compare against)")
    lines.append("")

    if sample:
        lines.append("### Manual spot-check samples (first 3 responses per tag, truncated to 500 chars)")
        by_tag = {}
        for r in recs:
            by_tag.setdefault(r.get("tag"), []).append(r)
        for t, rs in sorted(by_tag.items()):
            for j, x in enumerate(rs[:3]):
                lines.append(f"- `{t}` #{j}: {(x.get('response') or '')[:500]}")
        lines.append("")

    return "\n".join(lines), p0_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sample", action="store_true", help="sample 3 responses per tag for manual spot-check")
    args = ap.parse_args()
    gt = load_gt()
    print(f"# Automated audit report (GT sources: runs/demo + runs/corpus, {len(gt)} ground truths in total)\n")
    all_pass = True
    for rd in args.runs:
        text, ok = audit_run(rd, gt, sample=args.sample)
        print(text)
        all_pass = all_pass and ok
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
