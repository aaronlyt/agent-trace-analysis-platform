"""Render the Who&When head-to-head social card (docs/assets/head_to_head.png).

Usage: .venv/bin/python scripts/make_social_card.py   (needs matplotlib)

Numbers mirror docs/benchmark_whoswhen_2026-08-30.md — update both together.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#0e1116"
FG = "#e8ecf3"
MUTED = "#9aa4b2"
PAPER = "#5b6472"
ATAP = "#34d399"
LOSE = "#f0b86e"

# (group label, metric, paper %, atap %, delta text, atap better?)
ROWS = [
    ("Algorithm-Generated", "which agent", 51.1, 60.3, "+9.2", True),
    ("Algorithm-Generated", "which step", 13.5, 38.9, "+25.4  (2.9\u00d7)", True),
    ("Hand-Crafted", "which agent", 53.4, 46.6, "\u22126.8 \u2020", False),
    ("Hand-Crafted", "which step", 3.5, 20.7, "+17.2  (5.9\u00d7)", True),
]

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.30, 0.16, 0.52, 0.62])
ax.set_facecolor(BG)
ax.set_xlim(0, 78)
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.invert_yaxis()
ax.axis("off")

title = fig.text(0.065, 0.925, "Which agent caused the failure \u2014 and when?",
                 fontsize=32, fontweight="bold", color=FG, va="center")
fig.text(0.065, 0.855,
         "atap vs the Who&When paper's own GPT-4o baseline (ICML 2025) \u00b7 184 real multi-agent failure traces \u00b7 gold hidden from the judge",
         fontsize=15.5, color=MUTED, va="center")

# legend (\\$ keeps matplotlib from parsing the dollar signs as mathtext)
lx = 0.705
for i, (label, color) in enumerate([("paper \u00b7 GPT-4o", PAPER),
                                    ("atap \u00b7 deepseek-v4-flash \u00b7 \\$0.16\u2013\\$1.43", ATAP)]):
    y = 0.930 - i * 0.050
    fig.patches.append(FancyBboxPatch((lx, y - 0.008), 0.014, 0.020,
                                      boxstyle="round,pad=0.002",
                                      fc=color, ec="none", transform=fig.transFigure))
    fig.text(lx + 0.024, y, label, fontsize=13.5, color=FG if i else MUTED,
             fontweight="bold" if i else "normal", va="center")

BAR_H = 0.26
for i, (group, metric, paper, atap, delta, better) in enumerate(ROWS):
    ax.barh(i - 0.17, paper, height=BAR_H, color=PAPER, zorder=3)
    ax.barh(i + 0.17, atap, height=BAR_H, color=ATAP, zorder=3)
    ax.text(paper + 1.2, i - 0.17, f"{paper:.1f}%", fontsize=13.5, color=MUTED,
            va="center", ha="left")
    ax.text(atap + 1.2, i + 0.17, f"{atap:.1f}%", fontsize=14.5, color=ATAP,
            fontweight="bold", va="center", ha="left")
    fig.text(0.065, 0.705 - i * 0.136, group, fontsize=15, color=FG,
             fontweight="bold", va="center")
    fig.text(0.065, 0.660 - i * 0.136, metric, fontsize=13.5, color=MUTED, va="center")
    fig.text(0.845, 0.683 - i * 0.136, delta, fontsize=16,
             color=ATAP if better else LOSE, fontweight="bold", va="center")

ax.text(-1.5, -0.62, "exact-match accuracy, higher is better", fontsize=11.5,
        color=MUTED, style="italic")

fig.text(0.065, 0.075, "github.com/aaronlyt/agent-trace-analysis-platform \u00b7 pip install atap \u00b7 MIT",
         fontsize=15, color=FG, fontweight="bold", va="center")
fig.text(0.065, 0.033,
         "single-pass All-at-Once judge, Without-GT setting \u00b7 judge model differs from the paper (deepseek-v4-flash vs GPT-4o) \u2014 full caveats + repro in the repo benchmark report",
         fontsize=12, color="#aeb8c4", va="center")
fig.text(0.065, 0.008,
         "\u2020 hand-crafted agent figures corrected for a Magentic-One routing-label scoring artifact (46.6% post-hoc; 43.1% on a native no-thinking re-run)",
         fontsize=12, color="#aeb8c4", va="center")

fig.savefig("docs/assets/head_to_head.png", facecolor=BG)
print("wrote docs/assets/head_to_head.png")
