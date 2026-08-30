<div align="center">

# Agent Trace Analysis Platform (atap)

**The attribution & recovery layer on top of your existing agent observability stack
(e.g. Langfuse) — find, explain, and fix LLM-agent failures, then write the answers
back as scores**

[![CI](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/aaronlyt/agent-trace-analysis-platform/badges/coverage.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aaronlyt/agent-trace-analysis-platform)](https://github.com/aaronlyt/agent-trace-analysis-platform/releases)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md)

<img src="docs/assets/langfuse_roundtrip.gif" alt="atap external evaluation on a live Langfuse instance" width="100%">

<sub>atap × Langfuse: push a corpus → pull live traces → analyze / classify / attribute →
root-cause + confidence + blamed-step written back as Scores (terminal-only demo:
`docs/assets/demo.gif`)</sub>

<img src="docs/assets/integration.svg" alt="atap integration architecture: trace sources → adapters → six-stage pipeline → scores written back" width="100%">

```bash
git clone https://github.com/aaronlyt/agent-trace-analysis-platform && cd agent-trace-analysis-platform
pip install -e ".[dev,llm]"
atap demo    # offline end-to-end pipeline: FakeLLM judge, deterministic, zero network
```

<sub><b>STAGES &amp; METHODS — 24 ALGORITHMS</b></sub>

[![collection](https://img.shields.io/badge/collection-%E2%9C%85_1-brightgreen)](docs/algorithms.md) [![represent](https://img.shields.io/badge/represent-%E2%9C%85_7-brightgreen)](docs/algorithms.md) [![analyze](https://img.shields.io/badge/analyze-%E2%9C%85_3-brightgreen)](docs/algorithms.md)

[![classify](https://img.shields.io/badge/classify-%E2%9C%85_3-brightgreen)](docs/algorithms.md) [![attribute](https://img.shields.io/badge/attribute-%E2%9C%85_8-brightgreen)](docs/algorithms.md) [![recover](https://img.shields.io/badge/recover-%E2%9C%85_3-brightgreen)](docs/algorithms.md)

</div>

Agent Trace Analysis Platform (**atap**) is the **attribution & recovery layer on top
of your observability stack**: it pulls traces from Langfuse (or JSONL / OTel /
Phoenix exports), locates the responsible agent and the decisive step of every
failure, and writes the verdicts back as Langfuse scores.

- **24 pluggable algorithms, 5 stages** — one module each, YAML-composed, artifact-coupled ([docs/algorithms.md](docs/algorithms.md))
- **Deterministic offline mode** — FakeLLM judge + fault-injected sandbox, zero network
- **Real LLMs** — any OpenAI-compatible API, per-call audit log
- **Langfuse integration** — `atap langfuse-eval` writes root-cause scores + blamed-step markers back onto your traces ([below](#integration-with-langfuse))
- **Closed loop** — one shared `Hypothesis` contract; recovered reruns re-enter analysis for verification

> **Disclaimer** — this project is a learning- and research-oriented implementation of
> the agent error-analysis pipeline. Validation is **limited**: acceptance numbers come
> from a toy sandbox with constructed faults, plus a small number of real-model runs
> (see [docs/validation.md](docs/validation.md)); they are not benchmark results. Use it
> to learn the pipeline and the algorithms — not as production tooling or as evidence
> of real-world performance.

## Results on real data

First external benchmark: **Who&When** ([ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution),
ICML 2025) — 184 real multi-agent failure trajectories, gold hidden from the judge,
scored with the stock `compare` evaluator
([full report](docs/benchmark_whoswhen_2026-08-30.md)):

| Stack (deepseek-v4-flash) | step hit | agent hit | cost |
|---|---|---|---|
| all_at_once | **33.2%** | **56.0%**† | $1.43 · 2.9h |
| all_at_once, thinking off | **33.2%** | 54.3% | $0.16 · 9min |
| binary_search | 11.4% | 34.2% | $0.26 · 32min |

Model choice dominates — step accuracy is effort-insensitive (33.2% at both effort
levels) at 9× lower cost; on hand-crafted transcripts thinking adds only single
digits once the scoring artifact below is fixed.
The sandbox acceptance numbers ([docs/validation.md](docs/validation.md)) prove the
pipeline contract; these numbers prove it on real data.

**vs the paper's GPT-4o baseline (Without-GT).** atap runs the Who&When paper's own
methods under one contract and, on the harder **step-level** metric, lifts its
single-pass baseline **~2.9× on algorithm-generated (13.5% → 38.9%) and ~5.9× on
hand-crafted (3.5% → 20.7%)** — on a cheaper model — while leading agent-level on
algorithm-generated (51.1% → 60.3%). The gain traces to an engineered single-pass prompt
(few-shot + MAST vocabulary + *earliest-decisive-error* framing), not the model. Full
head-to-head, per-method table and caveats: [benchmark report](docs/benchmark_whoswhen_2026-08-30.md).

> † A Magentic-One routing-label artifact understated hand-crafted agent attribution
> above; fixed in `src/io/whoswhen.py`. The thinking-off row is a native re-run on the
> fixed adapter (hand-crafted agent 8.6% → **43.1%**, aggregate 44.0% → **54.3%**); the
> thinking row is post-hoc corrected (hand-crafted **46.6%**, aggregate **56.0%**).



## How it works

```
 ① collection                     ② storage
 io/  (JSONL · Langfuse v3 · OTel GenAI · live Langfuse API)  ──▶  traces.jsonl
                                                       │
                                                       ▼
 ③ representation — represent/
 R0 canonical events · SSF saliency folding · action signatures
 IDG dependency graph · hierarchy tree · claim ledger · HCG
                                                       │   artifacts only
                                                       ▼
 ④ analysis & classification — analyze/ + classify/
 judge_eval · loop_detect · drift_detect · mast_judge · rule_pack · inducer
                                                       │   failures trigger attribution
                                                       ▼
 ⑤ attribution — attribute/                            ──▶  Hypothesis
 all_at_once · binary_search · sbfl · rg_ug · chief ·
 claim_audit · tree_diagnosis · counterfactual_replay
                                                       │
                                                       ▼
 ⑥ recovery — recover/
 targeted_rerun · feedback_injection · dover
                                                       │
      rerun traces re-enter ④ for verification  ◀─────┘   (closed_loop: true)
```

The full algorithm table (24 methods, one paper each) and the companion
infrastructure notes live in **[docs/algorithms.md](docs/algorithms.md)**
([中文版](docs/算法清单.md)).

## Installation

atap is currently distributed as a **local install from source** (a PyPI package may come
later). Requires **Python ≥ 3.10** (developed and tested on 3.12).

```bash
git clone https://github.com/aaronlyt/agent-trace-analysis-platform
cd agent-trace-analysis-platform
```

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev,llm]"
```

Or with plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm]"
```

Extras: base install pulls `pyyaml` + `pydantic` only; `llm` adds the `openai` client
(needed for real-model runs), `dev` adds `pytest`. Everything offline (FakeLLM) works
without the `llm` extra.

## Quick start

```bash
# 1) list all registered algorithms
atap list

# 2) offline end-to-end demo — FakeLLM, deterministic, zero network (seed=7)
atap demo
```

`atap demo` runs the full pipeline on 7 sandbox trajectories (1 success + 6 injected
faults) and prints, for each fault, whether attribution hit the ground-truth step/agent
and whether recovery was verified:

```
attribution hits: step 6/6  agent 6/6  MAST 6/6  recovery 6/6
round0: traces=7 failures=6 attributed=6 reruns=6(ok=6)
round1: traces=6 failures=0 attributed=0 reruns=0(ok=0)
artifacts directory: runs/demo/artifacts (report.json + per-trajectory per-stage JSON)
```

Then explore the runnable configurations:

<details>
<summary><b>More runnable configs</b> — compare · v3/v4 stacks · drift · dover · counterfactual replay · export · verbose logs</summary>

```bash
# same stack as the demo, driven by a config file
atap run --config configs/pipeline_offline.yaml

# stage-3 stack: bisection + loop predicates + L0 rule pack + feedback injection
# on a 24-trace spectrum corpus
atap corpus --out runs/corpus/traces.jsonl
atap run --config configs/pipeline_offline_v3.yaml --out runs/v3

# compare two algorithm stacks on the same trajectory set
atap compare --config configs/pipeline_offline_v3.yaml \
             --config configs/pipeline_sbfl.yaml --out runs/compare

# stage-4 deterministic layer (IDG + hierarchy tree + RG/UG + drift + inducer)
atap run --config configs/pipeline_offline_v4.yaml --out runs/v4

# drift monitoring corpus (three constructed drift scenarios)
atap corpus --drift --out runs/drift/traces.jsonl
atap run --config configs/pipeline_drift.yaml --out runs/drift

# L3 recovery closed loop (DoVer) and counterfactual replay
atap run --config configs/pipeline_dover.yaml --out runs/dover
atap run --config configs/pipeline_cf_replay.yaml --out runs/cf_replay

# export traces for Langfuse (v3 ingestion) or OTel (GenAI semconv)
atap export --traces runs/corpus/traces.jsonl --format langfuse --out runs/export.json

# DEBUG-level process logs (default INFO)
atap -v run --config configs/pipeline_offline.yaml
```

</details>

### Real LLM runs

Any OpenAI-compatible endpoint works. Keys are read from environment variables only —
never written to disk.

```bash
export OPENAI_API_KEY=sk-...                     # required
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional (default: OpenAI)
atap run --config configs/pipeline_llm.yaml
```

`configs/` also contains the `final_*` (full pre-release test matrix) and `realtest_*`
(smoke variants for specific models) configurations used to produce the real-model
numbers below.

### Auditing and logs

Every `run` / `demo` / `compare` writes two records under `runs/<name>/`:

- `run.log` — process log (stage timings, acceptance numbers); `-v` switches to DEBUG;
- `llm_calls.jsonl` — **one audit record per LLM call**: timestamp, client, tag, model,
  schema, latency, full prompt messages, response, token usage, error info. Fake and real
  clients share the same auditor.

```bash
python -c "import json,collections; \
recs=[json.loads(l) for l in open('runs/demo/llm_calls.jsonl')]; \
print(len(recs), dict(collections.Counter(r['tag'] for r in recs)))"
```

## Integration with Langfuse

atap works as an **external evaluation pipeline** on a live Langfuse deployment:
pull traces by tag/time window, run your analysis/classify/attribute stack on them,
and write the results back — attribution shows up directly in your own Langfuse UI.

| Score | Placed on | Carries |
|---|---|---|
| `atap:root-cause` | the trace | root-cause code (categorical) |
| `atap:confidence` | the trace | top-hypothesis confidence (numeric) |
| `atap:blamed-step` | the responsible observation | agent @ step + root cause |
| score `metadata` | every score | full `Hypothesis` + run identity (`run_id` / `llm`) — evaluation batches stay distinguishable |

```bash
pip install -e ".[llm,langfuse]"
export LANGFUSE_BASE_URL=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
atap langfuse-eval --config configs/langfuse_eval.yaml --out runs/lf1 \
    --tags production --since 24h --dry-run    # dry-run first; drop the flag to write
```

`--dry-run` prints the scores without sending anything; traces whose earlier batch
completed are skipped (the trace-level `atap:root-cause` is written last and doubles
as the completion marker, so an interrupted batch is simply re-evaluated; `--force`
re-evaluates regardless). Credentials come from the environment only. Seeding a demo
instance (`atap langfuse-push`) and the full round-trip walkthrough live in
[docs/集成指南_Langfuse.md](docs/集成指南_Langfuse.md) and
`docker-compose.langfuse.yml`.

## Configuration — the pluggable core

Pipelines are plain YAML; algorithms are referenced by registered name and can take params:

```yaml
run_name: offline-full-pipeline
seed: 7
source: {type: jsonl, path: runs/demo/traces.jsonl}
llm: {type: fake}          # or {type: openai, model: ..., temperature: 0.0}
sandbox: {type: toy}
closed_loop: true          # recover outputs flow back to analyze for verification

stages:
  represent:
    - canonical_events
    - ssf                  # add/swap an algorithm = one line
  analyze:
    - judge_eval
  classify:
    - mast_judge
  attribute:
    - all_at_once
  recover:
    - name: targeted_rerun
      params: {max_rounds: 5}
```

### Adding your own algorithm

One module under the stage package, subclass the stage base, decorate with `@register`
— zero core changes, and it shows up in `atap list`. A worked example lives in
[docs/algorithms.md](docs/algorithms.md#adding-your-own-algorithm).

## Roadmap

- [x] Phase 4A — deterministic layer: `idg` / `hierarchy_tree` / `rg_ug` / `drift_detect` / `inducer` + taxonomy accept
- [x] Phase 4B — LLM representation & attribution: `claim_ledger`+`claim_audit` (DRIFT), `tree_diagnosis` (CodeTracer), `hcg`+`chief` (CHIEF)
- [x] Phase 4C — L3 counterfactual replay: sandbox `replay_intervene`, `counterfactual_replay` (TraceElephant), `dover` (DoVer)
- [x] Phase 4D — collection adapters: Langfuse v3 ingestion, OTel GenAI semconv, `atap export` + roundtrip
- [x] Phase 4E — live Langfuse bridge: `atap langfuse-eval` (pull → pipeline → Scores write-back) + `atap langfuse-push`
- [ ] fuse SBFL as an L2 prior (currently a standalone algorithm); AgenTracer-style GRPO fine-tuned tracer
- [ ] sandbox evolution — grow the toy research-QA sandbox into more realistic multi-scenario execution environments (richer task types, real tool calls, broader fault injection)
- [ ] real-dataset evaluation — first round done on the public Who&When benchmark
  (184 real failure trajectories,
  [docs/benchmark_whoswhen_2026-08-30.md](docs/benchmark_whoswhen_2026-08-30.md));
  broadening to more methods and datasets remains

Architecture & contracts: [docs/architecture.md](docs/architecture.md) ·
validation status: [docs/validation.md](docs/validation.md) ·
algorithm table: [docs/algorithms.md](docs/algorithms.md) ·
benchmark report: [docs/benchmark_whoswhen_2026-08-30.md](docs/benchmark_whoswhen_2026-08-30.md) ·
detailed plans: [docs/plan.md](docs/plan.md) · [docs/plan_阶段四.md](docs/plan_阶段四.md) ·
integration guide: [docs/集成指南_Langfuse.md](docs/集成指南_Langfuse.md) ·
development log: [docs/README_dev_log.md](docs/README_dev_log.md)
