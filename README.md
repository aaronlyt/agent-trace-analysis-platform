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

Agent Trace Analysis Platform (**atap**) is not another tracing platform — it is the
**attribution & recovery layer that sits on top of the observability stack you already
run**: a live Langfuse instance, or plain JSONL / OTel / Phoenix exports.

Pulled traces are flattened into one canonical event stream and run through the
six-stage pipeline shared by recent agent error-analysis research — **representation →
analysis/evaluation → error classification → failure attribution → recovery**.

The verdicts go back where your team already looks:

- **on the trace** — a Langfuse score carrying the root-cause code and confidence;
- **on the responsible observation** — a blamed-step marker;
- **in score metadata** — the full hypothesis plus run-batch identity.

The pipeline is **transformers-style pluggable**: one algorithm per module, every
algorithm inherits a stage base class, registers itself in a `Registry`, and composes
into pipelines via YAML. Algorithms talk to each other only through artifacts — never
through imports (enforced by import-invariant tests).

- **24 algorithms across 5 stages**, each faithful to a specific paper — full table in
  [docs/algorithms.md](docs/algorithms.md)
- **Deterministic offline mode** — FakeLLM pseudo-judge + toy sandbox with injected faults,
  fully reproducible, zero network
- **Real LLMs** through any OpenAI-compatible API, with a per-call audit log
- **One attribution contract** — every localizer (rules, judge, graph, replay) emits the same
  `Hypothesis` structure that recovery consumes
- **Closed loop** — recovered reruns automatically re-enter analysis for verification
  (`closed_loop: true`)
- **Ingest & export** — Langfuse v3 ingestion / OTel GenAI semantic conventions, with
  field-level roundtrip equivalence and ground-truth leak guards on export
- **Live external evaluation** — `atap langfuse-eval` pulls traces from a running
  Langfuse instance and writes attribution results back as Scores on the
  original traces (root-cause + confidence on the trace, a blamed-step marker
  on the responsible observation); re-evaluations are batch-distinguishable via
  score metadata (`run_id` / `llm` / full hypothesis)

> **Disclaimer** — this project is a learning- and research-oriented implementation of
> the agent error-analysis pipeline. Validation is **limited**: acceptance numbers come
> from a toy sandbox with constructed faults, plus a small number of real-model runs
> (see [Validation status](#validation-status)); they are not benchmark results. Use it
> to learn the pipeline and the algorithms — not as production tooling or as evidence
> of real-world performance.

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

### External evaluation: write attribution back to Langfuse

atap can act as an **external evaluation pipeline** over a live Langfuse
deployment: pull traces by tag/time window, run your analysis/classify/attribute
stack on them, and write the results back as Scores on the original traces —
the failure attribution shows up directly in your own Langfuse UI.

```bash
pip install -e ".[llm,langfuse]"
export LANGFUSE_BASE_URL=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
atap langfuse-eval --config configs/langfuse_eval.yaml --out runs/lf1 \
    --tags production --since 24h --dry-run    # dry-run first; drop the flag to write
```

`--dry-run` prints the scores without sending anything; traces whose earlier
batch completed are skipped (the trace-level `atap:root-cause` score is
written last and doubles as the completion marker, so an interrupted batch is
simply re-evaluated; `--force` re-evaluates regardless).
Credentials come from the environment only. A self-hosted demo instance and the
full round-trip walkthrough (seed with `atap langfuse-push`, evaluate, watch the
scores appear) live in
[docs/集成指南_Langfuse.md](docs/集成指南_Langfuse.md) and
`docker-compose.langfuse.yml`.

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

One module under the stage package, subclass the stage base, decorate with `@register` —
zero core changes, and it shows up in `atap list`:

```python
# src/attribute/my_attributor.py
from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.schema import Hypothesis


@register
class MyAttributor(Attributor):        # stage = "attribute" is set by the base class
    name = "my_attributor"

    def run_one(self, bundle, ctx):
        if bundle.succeeded:
            return                      # detection ≠ attribution
        self.emit(bundle, [Hypothesis(
            agent="reporter", step=3,
            root_cause="…", root_cause_code="FM-1.3",
            evidence=["event-3 …"], fix_suggestion="…", confidence=0.6,
        )])                             # writes artifacts["attribute"]["my_attributor"]
```

## Key contracts

- **R0 event model** (`core/schema.py`): `TraceEvent(kind/agent/action/payload/refs/phase/parent/index)` —
  the representation layer is the only data interface for analysis and attribution;
- **Unified attribution output**: `Hypothesis(agent, step, root_cause, root_cause_code,
  responsible_side, evidence, fix_suggestion, confidence)` — produced by every L0~L3
  attribution algorithm, consumed only by recovery;
- **Dual scope**: `run_one` (single trajectory) / `run_corpus` (cross-trajectory
  aggregation — used by spectrum and clustering algorithms);
- **Detection ≠ attribution**: analyze discovers symptoms; attribution runs on failures;
  recovery outputs automatically return to analyze for verification (`closed_loop: true`).

## Layered invariants (enforced by `tests/test_invariants.py`)

- `core/` contains zero algorithms and zero I/O — interfaces only;
- algorithm modules must not import other stage packages (sole exception: the shared
  `classify/taxonomy` vocabulary) nor `sandbox/runtime/cli`;
- `llm/` and `io/` depend on no stage package; `sandbox/` depends only on `core`;
- every registered class's `stage` must match its owning package.

## Validation status

**Offline (FakeLLM, deterministic).** 332 tests pass in under a second, including the
offline end-to-end pipeline, replay-integrity invariants, and leak-guard regressions.
Representative acceptance numbers on the sandbox corpora:

| Stack | Corpus | Offline result |
|---|---|---|
| demo (SSF + all-at-once + targeted rerun) | 7 traces, 6 injected faults | step 6/6 · agent 6/6 · MAST 6/6 · recovery 6/6; closed-loop round 1: 0 failures |
| v3 (bisection + rules + feedback injection) | 18-fault corpus | step 15/18 · 141 LLM calls (vs SBFL 12/18 · 42 calls on the same corpus) |
| rg_ug (deterministic, zero LLM) | 18-fault corpus | step 15/18 · agent 15/18 |
| chief | 18-fault corpus | step 18/18 · agent 18/18 |
| tree_diagnosis / claim_audit | 18-fault corpus | 18/18 · 36 calls / 12/18 (two misses are documented method boundaries) |
| dover | 18-fault corpus | recovery 18/18; closed-loop improvement 18/18 |
| counterfactual_replay | 18-fault corpus | 15 validated / 3 refuted — the refuted three are exactly bisection's known mislocalizations |

**Real LLMs** (pre-release full test: `deepseek-v4-flash` direct, 8 config tiers, 594
audited calls, zero judge-prompt leaks, human spot-check found no hallucination; report
in [docs/audit_上线前真实测试_2026-08-25.md](docs/audit_上线前真实测试_2026-08-25.md)):

- smoke stack: step 6/6 · agent 6/6 · recovery 6/6
- **chief: step 17/18 · agent 18/18 — the best real-model localizer**
- claim coverage 14/18; tree diagnosis 14/18; dover recovery 18/18; v3 closed loop 18/18
- binary_search 3/18 — well below its 15/18 offline baseline (judge lower-half bias on
  short trajectories; a judge-capability limit, not a pipeline defect — demoted to
  auxiliary use)

> **Read the offline numbers correctly.** The offline sandbox decides "fault removed" by
> keyword-matching the judge's fix suggestion against the injected fault name, so offline
> recovery and replay-verdict numbers are deterministic functions of attribution hits.
> They prove the pipeline contract (hypothesis → feedback → replay → verification), not
> judge capability. For the same reason, offline *cross-algorithm* comparisons measure how
> much information the deterministic pseudo-judge exposes per algorithm, not algorithm
> superiority. The real-model numbers above are the meaningful ones.

## Project layout

```
src/
  core/        # registry · pipeline · schema · config — zero algorithms, zero I/O
  represent/   # R0–R5 trajectory representations
  analyze/     # symptom discovery: judge, loop predicates, drift detection
  classify/    # MAST labeling · L0 rule pack · residual-mode inducer
  attribute/   # L0–L3 failure attribution (cost ladder)
  recover/     # targeted rerun · feedback injection · do-then-verify
  llm/         # FakeLLM pseudo-judge · OpenAI-compatible client · call auditor
  io/          # JSONL store · Langfuse / OTel adapters · live Langfuse bridge · export leak guard
  sandbox/     # toy research-QA environment with fault injection and drift corpora
configs/       # runnable pipeline configurations (offline · LLM · realtest · final)
tests/         # 332 tests: e2e, invariants, leak regressions, replay integrity
docs/          # plans · audit reports · dev log
```

## Roadmap

- [x] Phase 4A — deterministic layer: `idg` / `hierarchy_tree` / `rg_ug` / `drift_detect` / `inducer` + taxonomy accept
- [x] Phase 4B — LLM representation & attribution: `claim_ledger`+`claim_audit` (DRIFT), `tree_diagnosis` (CodeTracer), `hcg`+`chief` (CHIEF)
- [x] Phase 4C — L3 counterfactual replay: sandbox `replay_intervene`, `counterfactual_replay` (TraceElephant), `dover` (DoVer)
- [x] Phase 4D — collection adapters: Langfuse v3 ingestion, OTel GenAI semconv, `atap export` + roundtrip
- [x] Phase 4E — live Langfuse bridge: `atap langfuse-eval` (pull → pipeline → Scores write-back) + `atap langfuse-push`
- [ ] fuse SBFL as an L2 prior (currently a standalone algorithm); AgenTracer-style GRPO fine-tuned tracer
- [ ] sandbox evolution — grow the toy research-QA sandbox into more realistic multi-scenario execution environments (richer task types, real tool calls, broader fault injection)
- [ ] real-dataset evaluation — validate the pipeline on public real agent-trajectory datasets/benchmarks, replacing constructed-corpus acceptance numbers

Detailed plans: [docs/plan.md](docs/plan.md) · [docs/plan_阶段四.md](docs/plan_阶段四.md) ·
algorithm table: [docs/algorithms.md](docs/algorithms.md) ·
integration guide: [docs/集成指南_Langfuse.md](docs/集成指南_Langfuse.md) ·
development log: [docs/README_dev_log.md](docs/README_dev_log.md)
