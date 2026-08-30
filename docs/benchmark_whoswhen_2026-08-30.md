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
- **Cost**: ≈ ¥13 total for all three runs; zero API errors; every call audited in
  `llm_calls.jsonl`.

## Results — 2×2 matrix (method × thinking)

| Run | Calls | In/Out tok | Cost | step hit | agent hit |
|---|---|---|---|---|---|
| all_at_once, thinking ON | 184 | 555k / 1.97M | $1.43 ≈ ¥10.2 (2.9h) | **61/184 (33.2%)** | **92/184 (50.0%)** |
| all_at_once, thinking OFF | 184 | 546k / 55k | $0.16 ≈ ¥1.1 (9min) | 60/184 (32.6%) | 81/184 (44.0%) |
| binary_search, thinking OFF | 972 | 944k / 74k | $0.26 ≈ ¥1.8 (32min) | 21/184 (11.4%) | 63/184 (34.2%) |

(binary_search thinking-ON deliberately skipped: ~10–20 h / ¥10–40 for the least
informative cell.)

Per split (step / agent):

| Run | Algorithm-Generated (126) | Hand-Crafted (58) |
|---|---|---|
| all_at_once + thinking | 38.9% / 60.3% | 20.7% / 27.6% |
| all_at_once, no thinking | **41.3% / 60.3%** | 13.8% / 8.6% |
| binary_search, no thinking | 16.7% / 50.0% | **0% / 0%** |

FakeLLM plumbing baseline for all_at_once (deterministic, zero cost): step 13/184,
agent 76/184.

## Findings

1. **The model does the work, not the thinking** — step accuracy is identical at both
   effort levels (33.2% vs 32.6%; the algo split even prefers no-thinking 41.3% vs
   38.9%) while cost and latency differ ~9×. The 4.7× gain over the FakeLLM baseline
   is model quality, not reasoning depth.
2. **Thinking pays only on messy transcripts** — Hand-Crafted agent attribution more
   than triples with reasoning (27.6% vs 8.6%). Practical default: thinking OFF for
   algorithm-generated corpora and large sweeps, ON when hand-style conversations
   dominate.
3. **binary_search's weakness is structural, not effort-level** — 11.4% overall and
   0/58 on Hand-Crafted (not even agent hits). Consistent with the 2026-08-25
   real-model validation (3/18, judge lower-half bias); it stays auxiliary. Next
   lever: raise `core/render.MAX_CONTENT_CHARS` and re-run the hand split to test
   whether the 400-char/event truncation is the cause.

## Reproduce

```bash
git clone https://github.com/ag2ai/Agents_Failure_Attribution.git
python scripts/whoswhen_to_atap.py --whenswhen-root ... --out data/whoswhen/traces.jsonl  # see --help
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
