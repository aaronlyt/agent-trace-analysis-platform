#!/usr/bin/env python3
"""Estimate the API cost of running atap's attributors over Who&When -- no calls.

Reuses atap's real judge view (core.render.render_trace, 400-char/event cap)
and the real all_at_once system prompt, so the token counts match what the
pipeline would actually send. Binary-search is estimated from its bisection
depth. Nothing here touches the network.

    python scripts/estimate_whoswhen_cost.py data/whoswhen/traces.jsonl
    python scripts/estimate_whoswhen_cost.py data/whoswhen/traces.jsonl --in-price 0.14 --out-price 0.28

Token counting uses tiktoken when installed, else a ~4 chars/token fallback
(the fallback is clearly labelled in the output).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from atap.core.render import render_trace
from atap.core.schema import Trajectory

# ---- token counter --------------------------------------------------------
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def ntok(s: str) -> int:
        return len(_ENC.encode(s))

    _TOK_BACKEND = "tiktoken/cl100k_base"
except Exception:  # tiktoken not installed -> conservative char heuristic
    def ntok(s: str) -> int:
        return max(1, len(s) // 4)

    _TOK_BACKEND = "~chars/4 (install tiktoken for exact counts)"


# ---- real all_at_once system preamble (fallback to a flat estimate) --------
def _system_preamble() -> str:
    try:
        from atap.attribute.all_at_once import _FEW_SHOT, _SYSTEM
        from atap.classify.taxonomy import mast_definitions_block

        return _SYSTEM.format(definitions=mast_definitions_block()) + "\n\n" + _FEW_SHOT
    except Exception:
        return "x" * 4800  # ~1.2k token stand-in if internals move


#: preset (input, output) USD prices per 1M tokens
PRESETS = {
    "cheap": (0.15, 0.60),      # gpt-4o-mini / DeepSeek-class
    "frontier": (2.50, 10.00),  # gpt-4o-class
}
OUT_PER_CALL_AAO = 150          # structured verdict (agent/step/reason/fix)
OUT_PER_CALL_BS = 10            # "upper half" / "lower half"
BS_PREAMBLE_TOK = 130           # binary-search per-segment instruction block


def load(path: str) -> list[Trajectory]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Trajectory.from_dict(json.loads(line)))
    return out


def estimate(traces: list[Trajectory]) -> dict:
    sys_tok = ntok(_system_preamble())
    aao_in = aao_out = aao_calls = 0
    bs_in = bs_out = bs_calls = 0
    for t in traces:
        roster = ", ".join(t.agents())
        body = render_trace(t)  # same cap (400 chars/event) the judge sees
        trace_tok = ntok(f"The task and failure trajectory are as follows (agent roster: {roster}):\n{body}")

        # all-at-once: one call, full system preamble + full trace view
        aao_calls += 1
        aao_in += sys_tok + trace_tok
        aao_out += OUT_PER_CALL_AAO

        # binary search: ~ceil(log2(n)) segment calls; segment content halves
        # each level, so total content ~= 2x the full trace (geometric sum)
        n = max(len(t.events), 1)
        calls = max(1, math.ceil(math.log2(n)) if n > 1 else 1)
        bs_calls += calls
        bs_in += int(trace_tok * 2) + calls * BS_PREAMBLE_TOK
        bs_out += calls * OUT_PER_CALL_BS

    return {
        "n_traces": len(traces),
        "all_at_once": {"calls": aao_calls, "in_tok": aao_in, "out_tok": aao_out},
        "binary_search": {"calls": bs_calls, "in_tok": bs_in, "out_tok": bs_out},
    }


def _usd(in_tok: int, out_tok: int, in_price: float, out_price: float) -> float:
    return in_tok / 1e6 * in_price + out_tok / 1e6 * out_price


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Who&When atap cost estimator (no API calls)")
    ap.add_argument("jsonl", help="converted trajectories (scripts/whoswhen_to_atap.py output)")
    ap.add_argument("--in-price", type=float, default=None, help="USD per 1M input tokens (overrides presets)")
    ap.add_argument("--out-price", type=float, default=None, help="USD per 1M output tokens")
    args = ap.parse_args(argv)

    est = estimate(load(args.jsonl))
    print(f"Who&When cost estimate  |  {est['n_traces']} trajectories  |  tokens via {_TOK_BACKEND}")
    print("=" * 74)

    if args.in_price is not None and args.out_price is not None:
        price_sets = {"custom": (args.in_price, args.out_price)}
    else:
        price_sets = PRESETS

    for method in ("all_at_once", "binary_search"):
        m = est[method]
        approx = "  (~est)" if method == "binary_search" else ""
        print(f"\n{method}{approx}")
        print(f"  calls={m['calls']:>5}   input≈{m['in_tok']:>9,} tok   output≈{m['out_tok']:>7,} tok")
        for label, (pin, pout) in price_sets.items():
            cost = _usd(m["in_tok"], m["out_tok"], pin, pout)
            print(f"    {label:<9} (${pin}/${pout} per 1M):  ${cost:,.2f}")

    # combined ladder
    print("\nfull ladder (all_at_once + binary_search)")
    tot_in = est["all_at_once"]["in_tok"] + est["binary_search"]["in_tok"]
    tot_out = est["all_at_once"]["out_tok"] + est["binary_search"]["out_tok"]
    for label, (pin, pout) in price_sets.items():
        print(f"    {label:<9} (${pin}/${pout} per 1M):  ${_usd(tot_in, tot_out, pin, pout):,.2f}")
    print("\nnote: prices are illustrative; check your provider. binary_search is an")
    print("upper-ish estimate (early-exit on a confident split can be cheaper).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
