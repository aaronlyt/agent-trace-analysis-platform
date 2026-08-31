# Validation status

> What has been measured, on what data, and how to read each kind of number.
> Real-data results live in the [Who&When benchmark report](benchmark_whoswhen_2026-08-30.md)
> (184 real failure trajectories) and are summarized in the README's
> *Results on real data* section.

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
in [docs/audit_prelaunch_realtest_2026-08-25.md](audit_prelaunch_realtest_2026-08-25.md)):

- smoke stack: step 6/6 · agent 6/6 · recovery 6/6
- **chief: step 17/18 · agent 18/18 — the best real-model localizer**
- claim coverage 14/18; tree diagnosis 14/18; dover recovery 18/18; v3 closed loop 18/18
- binary_search 3/18 — well below its 15/18 offline baseline (judge lower-half bias on
  short trajectories; a judge-capability limit, not a pipeline defect — demoted to
  auxiliary use)

**External benchmark (Who&When)** — 184 real multi-agent failure trajectories,
gold hidden from the judge; see the
[benchmark report](benchmark_whoswhen_2026-08-30.md) for the 2×2 method-by-effort
matrix, per-split numbers, and cost.

> **Read the offline numbers correctly.** The offline sandbox decides "fault removed" by
> keyword-matching the judge's fix suggestion against the injected fault name, so offline
> recovery and replay-verdict numbers are deterministic functions of attribution hits.
> They prove the pipeline contract (hypothesis → feedback → replay → verification), not
> judge capability. For the same reason, offline *cross-algorithm* comparisons measure how
> much information the deterministic pseudo-judge exposes per algorithm, not algorithm
> superiority. The real-model and real-data numbers are the meaningful ones.
