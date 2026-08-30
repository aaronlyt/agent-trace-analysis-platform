# Architecture & contracts

> Project layout, the data contracts every stage honors, and the layered invariants
> that keep the framework pluggable. Companion to the [README overview](../README.md)
> and the [algorithm table](algorithms.md) · [简体中文 README](../README.zh-CN.md).

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
tests/         # 332+ tests: e2e, invariants, leak regressions, replay integrity
docs/          # plans · audit reports · benchmarks · dev log
```

## Contracts & invariants

- **R0 event model** (`core/schema.py`): `TraceEvent(kind/agent/action/payload/refs/phase/parent/index)` —
  the representation layer is the only data interface for analysis and attribution;
- **Unified attribution output**: `Hypothesis(agent, step, root_cause, root_cause_code,
  responsible_side, evidence, fix_suggestion, confidence)` — produced by every L0~L3
  attribution algorithm, consumed only by recovery;
- **Dual scope**: `run_one` (single trajectory) / `run_corpus` (cross-trajectory
  aggregation — used by spectrum and clustering algorithms);
- **Detection ≠ attribution**: analyze discovers symptoms; attribution runs on failures;
  recovery outputs automatically return to analyze for verification (`closed_loop: true`);
- **Layered invariants** (enforced by `tests/test_invariants.py`): `core/` holds zero
  algorithms and zero I/O; algorithm modules import no other stage package (sole
  exception: the shared `classify/taxonomy` vocabulary) and never `sandbox/runtime/cli`;
  `llm/` + `io/` depend on no stage package; every registered class's `stage` matches
  its owning package.
