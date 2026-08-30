#!/usr/bin/env python3
"""Convert the Who&When dataset into atap-native JSONL trajectories.

    # 1. get the data (once)
    git clone https://github.com/ag2ai/Agents_Failure_Attribution.git

    # 2. convert -> data/whoswhen/traces.jsonl (184 trajectories)
    python scripts/whoswhen_to_atap.py \
        --whoswhen-root Agents_Failure_Attribution/"Who&When" \
        --out data/whoswhen/traces.jsonl

Then estimate cost (no API):  python scripts/estimate_whoswhen_cost.py data/whoswhen/traces.jsonl
Then run + score:            atap compare --config configs/whoswhen_all_at_once.yaml \
                                          --config configs/whoswhen_binary_search.yaml \
                                          --traces data/whoswhen/traces.jsonl --out runs/whoswhen
"""

from __future__ import annotations

import argparse
from collections import Counter

from atap.io.whoswhen import SPLIT_AGENT_KEY, load_whoswhen, write_jsonl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Who&When -> atap JSONL converter")
    ap.add_argument(
        "--whoswhen-root", required=True,
        help='the "Who&When" directory (contains Algorithm-Generated / Hand-Crafted)',
    )
    ap.add_argument("--out", default="data/whoswhen/traces.jsonl", help="output JSONL path")
    ap.add_argument(
        "--splits", nargs="+", default=list(SPLIT_AGENT_KEY), choices=list(SPLIT_AGENT_KEY),
        help="which splits to include (default: both)",
    )
    args = ap.parse_args(argv)

    traces = load_whoswhen(args.whoswhen_root, splits=args.splits)
    n = write_jsonl(traces, args.out)

    by_split = Counter(t.meta["injected_fault"]["kind"] for t in traces)
    steps = [len(t.events) for t in traces]
    print(f"converted {n} trajectories -> {args.out}")
    for kind, c in sorted(by_split.items()):
        print(f"  {kind}: {c}")
    print(f"  events/trajectory: min={min(steps)} max={max(steps)} "
          f"mean={sum(steps) / len(steps):.1f}")
    print("gold in meta['injected_fault'] = {step (0-based event idx), agent, kind, mast_code=None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
