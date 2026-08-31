<!-- Promotion drafts for the Langfuse community. Discord: paste the SHORT post first,
     offer the LONG one as a follow-up/thread. GitHub Discussions: use the LONG post.
     Attach docs/assets/langfuse_roundtrip.gif where the channel allows embeds. -->

# Langfuse community — warm-up post drafts

## SHORT — Discord first message (e.g. #showcase / #community-projects)

Built an external evaluation pipeline for Langfuse traces: **atap** — it pulls your
traces via the v3 API, runs failure attribution (which agent, which step, why — MAST
taxonomy), and writes the verdicts **back onto the trace as Scores**:

- `atap:root-cause` (trace, categorical)
- `atap:confidence` (trace, numeric)
- `atap:blamed-step` — placed on the *responsible observation*, so the failing step
  is marked right where it happened in the UI

The "why did this run fail" layer on top of tracing. Works with any OpenAI-compatible
judge model, has a deterministic offline mode (FakeLLM, zero network), and re-runs are
idempotent — an interrupted batch just resumes, traces that already have a verdict are
skipped.

Round-trip demo (terminal + Langfuse UI): **docs/assets/langfuse_roundtrip.gif**
Repo + 5-minute walkthrough (docker-compose for a local Langfuse included):
https://github.com/aaronlyt/agent-trace-analysis-platform

Happy to write up details if useful — benchmarked it on the Who&When ICML'25
failure-attribution dataset (184 real failed agent runs): step-level 2.9–5.9× the
paper's GPT-4o baseline.

---

## LONG — GitHub Discussions / follow-up comment

**atap: external failure-attribution evaluation that writes Scores back to Langfuse**

Hey 👋 — sharing a project that uses Langfuse as *both* input and output for LLM-agent
failure analysis. If you run multi-agent systems, the question after a failed run is
usually "which agent broke it, at which step, and why" — tracing shows you *what*
happened, but not the verdict. atap adds that verdict layer as an external pipeline:

```
Langfuse traces → R0 event model → analyze / classify / attribute → Scores written back
```

**What lands in your Langfuse project:**

| Score | Placed on | Carries |
|---|---|---|
| `atap:root-cause` | the trace | root-cause code (categorical) |
| `atap:confidence` | the trace | top-hypothesis confidence (numeric) |
| `atap:blamed-step` | the responsible observation | agent @ step + root cause |
| score `metadata` | every score | full hypothesis + run identity (`run_id` / `llm`) |

Because the blamed step is scored onto the specific observation, you can filter traces
by root cause and jump straight to the failing span in the UI. Metadata keeps
evaluation batches distinguishable when you re-run with a different judge model.

**Why external rather than an in-app evaluator:** the pipeline is pluggable — 24
algorithms across 5 stages (one module each, YAML-composed), so you can swap the
attribution method per run without touching your app. Any OpenAI-compatible model
works as the judge; there's also a fully deterministic offline mode (FakeLLM judge +
fault-injected sandbox) for CI.

**Ops details people asked about so far:**
- Credentials only from the environment (`LANGFUSE_PUBLIC_KEY` / `SECRET_KEY`); nothing
  stored in config files.
- `--dry-run` prints everything it *would* write; writes nothing.
- Idempotent batches: the trace-level `atap:root-cause` is written last and doubles as
  a completion marker — interrupted batches are safely re-run, completed traces are
  skipped (`--force` overrides).
- Lazy pagination, so large tag/time-window pulls don't balloon memory.

**Try it locally** (spins up Langfuse via docker-compose, seeds a demo corpus, then
reads it back — no hosted account needed):

```bash
git clone https://github.com/aaronlyt/agent-trace-analysis-platform && cd agent-trace-analysis-platform
pip install -e ".[llm,langfuse]"
docker compose -f docker-compose.langfuse.yml up -d
atap langfuse-push    # seed demo corpus
atap langfuse-eval --config configs/langfuse_eval.yaml --tags demo --dry-run
```

Full walkthrough: `docs/集成指南_Langfuse.md` in the repo.

**Does it actually work?** First external benchmark — Who&When (ICML 2025 Spotlight,
184 real failed multi-agent runs, gold hidden from the judge): atap's single-pass
attribution scores **38.9% / 20.7% exact-step accuracy** (algorithm-generated /
hand-crafted splits) vs the paper's GPT-4o 13.5% / 3.5%, and 60.3% / 46.6% on
agent-level. Judge was deepseek-v4-flash at $0.16–1.43 per full run. Report + repro
commands: `docs/benchmark_whoswhen_2026-08-30.md`.

Feedback very welcome — especially on what scores/annotations would make this most
useful inside Langfuse's UI (e.g. would a `mast_code` categorical score per trace be
worth adding?).
