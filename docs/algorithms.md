# Implemented algorithms

> 24 algorithms across 5 stages (collection is handled by the `io/` adapters), each
> faithful to a specific paper. This table is the expanded companion to the
> [README overview](../README.md) · [简体中文版](算法清单.md).

| Stage | Module | Method | Paper |
|---|---|---|---|
| represent | `canonical_events` | R0: flatten the span tree into a unified event stream — the single data interface for all downstream stages | AgentDebugX [2607.18754](https://arxiv.org/abs/2607.18754) |
| represent | `ssf` | R1: saliency folding with reversible placeholders + summaries, for long trajectories | TrajAudit [2605.26563](https://arxiv.org/abs/2605.26563) |
| represent | `action_signature` | R5: nine action classes + argument fingerprints + seven effect labels + anchor sets / milestones / LCS | TraceProbe [2607.06184](https://arxiv.org/abs/2607.06184) |
| represent | `idg` | R2: information dependency graph — usage edges derived directly from `refs`, zero LLM | GraphTracer [2510.10581](https://arxiv.org/abs/2510.10581) *(retracted; ideas only)* |
| represent | `hierarchy_tree` | R4: exploration = siblings / state changes = children, stage index, compact `tree.md` rendering | CodeTracer [2604.11641](https://arxiv.org/abs/2604.11641) |
| represent | `claim_ledger` | R3: six-field claim ledger from one global LLM pass | DRIFT [2606.02060](https://arxiv.org/abs/2606.02060) |
| represent | `hcg` | hierarchical causal graph — three node tiers, sub/agt/step edges, deterministic construction | CHIEF [2602.23701](https://arxiv.org/abs/2602.23701) |
| analyze | `judge_eval` | LLM-as-judge quality score + findings, few-shot | MAST [2503.13657](https://arxiv.org/abs/2503.13657) / Agent-as-a-Judge [2410.10934](https://arxiv.org/abs/2410.10934) |
| analyze | `loop_detect` | loop predicates: search loop, re-read churn, tool oscillation, redundant search — deterministic, zero LLM | TraceProbe [2607.06184](https://arxiv.org/abs/2607.06184) |
| analyze | `drift_detect` | version/data/behavior drift between control groups: shared-support PSI + support mismatch + minimum group size + collinearity dedup | system-level taxonomy [2511.19933](https://arxiv.org/abs/2511.19933) |
| classify | `mast_judge` | MAST 3 categories / 14 failure modes, vocabulary-validated; `allow_novel` residual channel | MAST [2503.13657](https://arxiv.org/abs/2503.13657) |
| classify | `rule_pack` | L0 free rules: malformed calls, no-progress loops, premature success claims, invalid output | AgentDebugX [2607.18754](https://arxiv.org/abs/2607.18754) |
| classify | `inducer` | residual clustering → new failure-mode proposals; human gate `atap taxonomy accept`, never auto-effective | AgentDebugX §3.4 [2607.18754](https://arxiv.org/abs/2607.18754) |
| attribute | `rg_ug` | L0: retrieval-gain / usage-gain (4 subtypes) from qrels set algebra + episode utility, zero LLM | search-agent diagnosis [2608.01913](https://arxiv.org/abs/2608.01913) |
| attribute | `sbfl` | L0: spectrum-based prior — γ/β/α + λ-decay-weighted Kulczynski2, cross-trace aggregation | FAMAS [2509.13782](https://arxiv.org/abs/2509.13782) |
| attribute | `all_at_once` | L1: single-pass attribution over the (SSF-folded) full trajectory | Who&When [2505.00212](https://arxiv.org/abs/2505.00212) |
| attribute | `binary_search` | L2: bisection localization, ⌈log₂n⌉ rounds, judge sees only the lower half | Who&When [2505.00212](https://arxiv.org/abs/2505.00212) |
| attribute | `claim_audit` | R3 consumer: four-level support check → expert audit → conservative backtracking → first error span | DRIFT [2606.02060](https://arxiv.org/abs/2606.02060) |
| attribute | `tree_diagnosis` | R4 consumer: tree-level localization, then stage-interval drill-down — two LLM calls | CodeTracer [2604.11641](https://arxiv.org/abs/2604.11641) |
| attribute | `chief` | HCG consumer: oracle synthesis → reverse-topological F_eval backtracking → progressive causal filtering | CHIEF [2602.23701](https://arxiv.org/abs/2602.23701) |
| attribute | `counterfactual_replay` | L3 final review: candidate-step message-intervention replay (k=3 window) filters spurious causality; adjusted verdicts `supersede` upstream hypotheses | TraceElephant [2604.22708](https://arxiv.org/abs/2604.22708) |
| recover | `targeted_rerun` | keep the prefix, rerun from t* with the fix as feedback, ≤5 rounds | AgentDebug [2509.25370](https://arxiv.org/abs/2509.25370) |
| recover | `feedback_injection` | attribution reflection injected into a full re-solve, 3 rounds | AgenTracer [2509.03312](https://arxiv.org/abs/2509.03312) |
| recover | `dover` | do-then-verify: trial split → minimal message intervention → in-place replay ×3 → milestone diff, four verdicts | DoVer [2512.06749](https://arxiv.org/abs/2512.06749) |

Companion infrastructure:

- `sandbox/` — a toy research-QA environment (planner → searcher → reporter) with mock
  retrieval corpus, two-level qrels (E/G), a verifier, six injected faults + two extended
  faults, and a drift-corpus generator. Ground truth is known by construction
  (AgenTracer route-B idea).
- `llm/` — `FakeLLM`, a deterministic pseudo-judge that rules on judge-visible text only
  (never reads ground truth), plus an OpenAI-compatible client and the shared call auditor.
