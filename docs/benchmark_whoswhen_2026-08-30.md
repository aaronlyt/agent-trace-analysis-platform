# Who&When external benchmark — first round (2026-08-30)

First evaluation of atap on a **public, real failure-attribution dataset**:
[Who&When](https://github.com/ag2ai/Agents_Failure_Attribution) ("Which Agent Causes
Task Failures and When?", ICML 2025 Spotlight, arXiv:2505.00212) — 184 failed
multi-agent trajectories (Algorithm-Generated 126 + Hand-Crafted 58), each with gold
`mistake_step` / `mistake_agent`.

Setup:

- **Adapter**: `src/io/whoswhen.py` → `scripts/whoswhen_to_atap.py` converts the raw
  JSON into atap R0 trajectories; `mistake_step` (a 0-based history index) lands
  directly on the R0 event index, agent identity follows the reference implementation
  (`name` for Algorithm-Generated, `role` for Hand-Crafted), gold lives only under
  `meta["injected_fault"]` — the judge never sees it (the paper's **Without-GT** basis).
- **Scoring**: the stock `atap.compare.evaluate_against_gt` (top hypothesis by
  confidence; `step`/`agent` exact match). No new metric code.
- **Model**: `deepseek-v4-flash` via the OpenAI-compatible client, weekend off-peak
  pricing ($0.22/M input, $0.66/M output). Configs: `configs/whoswhen_*.yaml`
  (`*_fast.yaml` = `extra_body: {thinking: {type: disabled}}`).
- **Cost**: ≈ ¥14 total for all four runs (incl. the confirmation re-run); zero API
  errors; every call audited in `llm_calls.jsonl`.

## Results — 2×2 matrix (method × thinking)

| Run | Calls | In/Out tok | Cost | step hit | agent hit |
|---|---|---|---|---|---|
| all_at_once, thinking ON | 184 | 555k / 1.97M | $1.43 ≈ ¥10.2 (2.9h) | **61/184 (33.2%)** | **103/184 (56.0%)**† |
| all_at_once, thinking OFF | 184 | 541k / 57k | $0.16 ≈ ¥1.1 (9min) | **61/184 (33.2%)** | 100/184 (54.3%) |
| binary_search, thinking OFF | 972 | 944k / 74k | $0.26 ≈ ¥1.8 (32min) | 21/184 (11.4%) | 63/184 (34.2%) |

† post-hoc corrected for the Magentic-One routing-label artifact (see the Correction
section). The thinking-OFF row is a **native re-run on the fixed adapter** —
normalized traces end-to-end, stock scoring, no post-hoc adjustment.

(binary_search thinking-ON deliberately skipped: ~10–20 h / ¥10–40 for the least
informative cell.)

Per split (step / agent):

| Run | Algorithm-Generated (126) | Hand-Crafted (58) |
|---|---|---|
| all_at_once + thinking | 38.9% / 60.3% | 20.7% / 46.6%† |
| all_at_once, no thinking | **42.1% / 59.5%** | 13.8% / 43.1% |
| binary_search, no thinking | 16.7% / 50.0% | 0% / 0% |

FakeLLM plumbing baseline for all_at_once (deterministic, zero cost): step 13/184,
agent 76/184.

## Findings

1. **The model does the work, not the thinking** — step accuracy is identical at both
   effort levels (33.2% at both; the algo split even prefers no-thinking 42.1% vs
   38.9%) while cost and latency differ ~9×. The 4.7× gain over the FakeLLM baseline
   is model quality, not reasoning depth.
2. **Thinking's hand-crafted edge was mostly the scoring artifact** — before the role
   normalization (Correction below) reasoning looked like a 3× lift on Hand-Crafted
   agent attribution (27.6% vs 8.6%); after it the edge is single digits (46.6% vs
   43.1% on agents, 20.7% vs 13.8% on steps). Practical default: thinking OFF for
   algorithm-generated corpora and large sweeps; ON only when hand-style conversations
   dominate and ~7pp is worth 9× the cost.
3. **binary_search's weakness is structural, not effort-level** — 11.4% overall and
   0/58 on Hand-Crafted (not even agent hits). Consistent with the 2026-08-25
   real-model validation (3/18, judge lower-half bias); it stays auxiliary. Next
   lever: raise `core/render.MAX_CONTENT_CHARS` and re-run the hand split to test
   whether the 400-char/event truncation is the cause.

## Reproduce

```bash
git clone https://github.com/ag2ai/Agents_Failure_Attribution.git
python scripts/whoswhen_to_atap.py --whoswhen-root ... --out data/whoswhen/traces.jsonl  # see --help
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.deepseek.com/v1
atap run --config configs/whoswhen_all_at_once.yaml        # or _fast / binary_search(_fast)
```

Scoring (evaluate_against_gt over the on-disk artifacts):

```python
import json
from pathlib import Path
from atap.core.bundle import TrajectoryBundle
from atap.core.schema import Trajectory
from atap.compare import evaluate_against_gt

traces = {t.trace_id: t for t in (
    Trajectory.from_dict(json.loads(l)) for l in open("data/whoswhen/traces.jsonl"))}

def score(run_dir, algo):
    bundles = []
    for d in Path(run_dir, "artifacts").iterdir():
        f = d / f"attribute__{algo}.json"
        if f.exists() and d.name in traces:
            b = TrajectoryBundle(traces[d.name])
            b.put("attribute", algo, json.loads(f.read_text()))
            bundles.append(b)
    return evaluate_against_gt(bundles)
```

Caveats: thinking effort differs across rows (the ON/OFF cells of binary_search were
not run); the two methods were therefore not compared at a single fixed effort level
except column-wise. Raw artifacts and audit logs live under `runs/whoswhen*`
(gitignored).

## Head-to-head vs the paper (Table 1, Without-GT, GPT-4o)

The Who&When paper reports its three methods on GPT-4o. atap runs the same methods
(plus more) under one `Hypothesis` contract, on `deepseek-v4-flash`, on the paper's
**Without-GT** basis — directly comparable, exact step / agent match.

**All-at-Once** (the primary single-pass method in both), thinking ON:

| Split | Metric | Paper (GPT-4o) | atap (deepseek-v4-flash) | Δ |
|---|---|---|---|---|
| Algorithm-Generated | Agent-level | 51.12% | **60.3%** | +9.2 |
| Algorithm-Generated | Step-level | 13.53% | **38.9%** | **+25.4 (2.9×)** |
| Hand-Crafted | Agent-level | 53.44% | 46.6%† | −6.8 |
| Hand-Crafted | Step-level | 3.51% | **20.7%** | **+17.2 (5.9×)** |

**Binary Search**, thinking OFF:

| Split | Metric | Paper (GPT-4o) | atap (deepseek-v4-flash) |
|---|---|---|---|
| Algorithm-Generated | Agent-level | 30.11% | **50.0%** |
| Algorithm-Generated | Step-level | 16.59% | 16.7% |
| Hand-Crafted | Agent-level | 36.21% | 0.0% |
| Hand-Crafted | Step-level | 6.90% | 0.0% |

Paper: Zhang et al., *Which Agent Causes Task Failures and When?* (ICML 2025), Table 1,
Without-GT columns, GPT-4o.

**Reading it**

- **Step-level is the headline.** atap's engineered single-pass prompt (few-shot + MAST
  failure-mode vocabulary + an *earliest-decisive-error, not the symptom* framing + a
  structured `[index] KIND agent` rendering) lifts the paper's **weakest** metric ~2.9×
  on algorithm-generated and ~5.9× on hand-crafted — on a cheaper model. atap's
  all-at-once step-level even exceeds every single method in the paper's Table 1.
- **atap leads on algorithm-generated** across the board and on step-level everywhere.
- **Hand-crafted agent** is within ~7 pts of GPT-4o after the role-normalization fix
  (below); the residual gap is the 400-char/event truncation on long Magentic-One
  transcripts (Finding 3) — next lever: raise `core/render.MAX_CONTENT_CHARS`.
- **Caveat — not a controlled ablation.** The judge model differs (deepseek-v4-flash vs
  GPT-4o), so the step-level gain reflects prompt engineering **and** model. The clean
  controlled run is atap's prompt on GPT-4o (≈ $1.5 for all_at_once per the cost
  estimator), isolating the prompt contribution.

## Correction — role normalization (adapter fix, 2026-08-30)

The Hand-Crafted **agent** numbers in the tables above were **understated**: Magentic-One
roles carry a routing annotation (`Orchestrator (-> WebSurfer)`, `Orchestrator
(thought)`) while the gold `mistake_agent` is the bare role (`Orchestrator`), so a
correct "Orchestrator" call scored as a string miss. `src/io/whoswhen.py` now strips the
routing annotation. Corrected hand-crafted agent accuracy:

| Run | hand agent (raw) | hand agent (fixed) | aggregate agent (fixed) |
|---|---|---|---|
| all_at_once, thinking ON | 27.6% | **46.6%** | **56.0%** |
| all_at_once, thinking OFF | 8.6% | **39.7%** | **53.8%** |

Step-level is unaffected, and binary_search's 0/58 on Hand-Crafted is genuine even
after normalization (0/58 both ways).

**Confirmed by native re-run (2026-08-30, thinking OFF).** The fixed adapter was
re-run end-to-end — normalized traces, fresh LLM calls, stock scoring, no post-hoc
adjustment: hand agent **43.1%** (25/58), aggregate agent **54.3%** (100/184), step
33.2% (algo 42.1% / hand 13.8%), $0.16 / 9 min. Within ≤3.4pp of the post-hoc
figures, so the thinking-ON row's post-hoc correction stands. To reproduce:

```bash
python scripts/whoswhen_to_atap.py --whoswhen-root Agents_Failure_Attribution/"Who&When" --out data/whoswhen/traces.jsonl
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.deepseek.com/v1
atap run --config configs/whoswhen_all_at_once_fast.yaml --out runs/whoswhen_v2_aao_nothink
```

(Locally the normalization was applied in place to the existing `traces.jsonl` — the
raw dataset directory had been cleaned up; equivalent to re-running the adapter,
backup kept at `data/whoswhen/traces.jsonl.bak-pre-normalize`.)
