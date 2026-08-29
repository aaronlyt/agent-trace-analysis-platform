#!/usr/bin/env python3
"""Attribution regression gate: `atap compare` output vs a committed baseline.

Fails (exit 1) when any metric drops below the baseline row for the same
config; improvements pass (and should be committed up by regenerating the
baseline). The offline corpus + FakeLLM stack is deterministic, so identical
inputs must reproduce identical numbers -- a drop means a real regression.

Usage:
    python scripts/check_attribution_baseline.py \\
        .github/baselines/attribution.json runs/ci/cmp/comparison.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRICS = ("n_failed", "step_hits", "agent_hits", "code_hits",
           "recovered", "closed_loop_improved")


def main(baseline_path: str, comparison_path: str) -> int:
    baseline = json.loads(Path(baseline_path).read_text())
    comparison = json.loads(Path(comparison_path).read_text())
    base_rows = {r["config"]: r["metrics"] for r in baseline["rows"]}
    failures: list[str] = []
    for row in comparison.get("rows", []):
        cfg = row["config"]
        base = base_rows.get(cfg)
        if base is None:
            continue  # config not covered by the baseline (compare allows extra configs)
        for m in METRICS:
            if row.get(m) is not None and row[m] < base[m]:
                failures.append(f"{cfg}: {m} {row[m]} < baseline {base[m]}")
        # errors in a compared run must never read as clean either
        if row.get("errors"):
            failures.append(f"{cfg}: {row['errors']} isolated algorithm error(s)")
    if failures:
        print("ATTRIBUTION REGRESSION:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"attribution baseline held: {len(comparison.get('rows', []))} config(s) checked")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
