<!-- PyPI landing page. Trimmed variant of README.md with absolute image URLs
     (PyPI does not render relative paths). Keep in sync with README.md. -->

<div align="center">

# Agent Trace Analysis Platform (atap)

**The attribution & recovery layer on top of your existing agent observability stack
(e.g. Langfuse) — find, explain, and fix LLM-agent failures, then write the answers
back as scores**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/blob/main/LICENSE)

<img src="https://raw.githubusercontent.com/aaronlyt/agent-trace-analysis-platform/main/docs/assets/langfuse_roundtrip.gif" alt="atap external evaluation on a live Langfuse instance" width="100%">

<sub>atap × Langfuse: pull live traces → analyze / classify / attribute →
root-cause + confidence + blamed-step written back as Scores</sub>

</div>

atap pulls traces from Langfuse (or JSONL / OTel / Phoenix exports), locates the
responsible agent and the decisive step of every failure, and writes the verdicts
back as Langfuse scores.

- **24 pluggable algorithms, 5 stages** — one module each, YAML-composed, artifact-coupled
- **Deterministic offline mode** — FakeLLM judge + fault-injected sandbox, zero network
- **Real LLMs** — any OpenAI-compatible API, per-call audit log
- **Langfuse integration** — `atap langfuse-eval` writes root-cause scores + blamed-step markers back onto your traces
- **Closed loop** — one shared `Hypothesis` contract; recovered reruns re-enter analysis for verification

## Results on real data

First external benchmark: **Who&When** (ICML 2025) — 184 real multi-agent failure
trajectories, gold hidden from the judge, scored with the stock evaluator:

| Stack (deepseek-v4-flash) | step hit | agent hit | cost |
|---|---|---|---|
| all_at_once | **33.2%** | **56.0%**† | $1.43 · 2.9h |
| all_at_once, thinking off | **33.2%** | 54.3% | $0.16 · 9min |
| binary_search | 11.4% | 34.2% | $0.26 · 32min |

vs the paper's GPT-4o baseline (Without-GT): step-level **2.9×** on algorithm-generated
(13.5% → 38.9%) and **5.9×** on hand-crafted (3.5% → 20.7%) — on a cheaper model.

## Quick start

```bash
pip install atap
atap demo    # offline end-to-end pipeline: FakeLLM judge, deterministic, zero network
```

Real-LLM runs, live-Langfuse round trips, the full head-to-head table, caveats and
reproduction commands:

**https://github.com/aaronlyt/agent-trace-analysis-platform**

> † hand-crafted agent figures corrected for a Magentic-One routing-label scoring
> artifact; the judge model differs from the paper's GPT-4o, so gains reflect prompt
> engineering + model — details and a controlled-ablation recipe in the repo's
> benchmark report.
