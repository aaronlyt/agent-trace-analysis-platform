"""atap command-line entry point.

Usage::

    atap run --config configs/pipeline_offline.yaml [--out runs/demo]
    atap list                       # list registered algorithms
    atap demo [--seed 42] [--out runs/demo]   # phase two: offline end-to-end demo
    atap -v run ...                 # verbose (DEBUG-level process logs)
    atap langfuse-eval --config cfg.yaml [--tags ...] [--since 24h] [--dry-run]
                                    # external evaluation over a live Langfuse
                                    # instance (pull -> pipeline -> score write-back)
    atap langfuse-push --traces runs/demo/traces.jsonl [--tags corpus-x]  # seed a live instance

Logging convention (atap.log): stdout carries only command results; process
events go through the ``atap`` logger → stderr, run/compare/demo
automatically persist ``<out>/run.log`` via runtime, and the LLM call audit
lands in ``<out>/llm_calls.jsonl``.
"""

from __future__ import annotations

import argparse

from atap.log import get_logger, setup_logging

log = get_logger("cli")


def _cmd_list(_args: argparse.Namespace) -> int:
    import atap  # trigger the registration bootstrap

    grouped = atap.list_algorithms()
    for stage, names in grouped.items():
        print(f"{stage}: {', '.join(names)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from atap.core.config import load_config
    from atap.io import build_store
    from atap.runtime import run_config

    cfg = load_config(args.config)
    _, reports = run_config(cfg, args.out)
    print(f"run={cfg.run_name} rounds={len(reports)}")
    for i, r in enumerate(reports):
        print(
            f"  round{i}: traces={r.n_traces} failures={r.n_failures} "
            f"attributed={r.n_attributed} reruns={r.n_reruns}(ok={r.n_rerun_success})"
            + (f" errors={r.n_errors}" if r.n_errors else "")
        )
    # actual artifact location (cfg.store may redirect it away from <out>/artifacts)
    print(f"artifacts -> {build_store(cfg.store, args.out).dir}")
    n_errors = sum(r.n_errors for r in reports)
    if n_errors:
        # isolated algorithm failures must stay visible in the exit status:
        # previously a mid-run crash failed the whole command; silently
        # exiting 0 after isolation would be a regression of that contract
        print(
            f"ERROR: {n_errors} algorithm failure(s) were isolated and skipped "
            f"(details in the FAILED stage_log lines and the status=error "
            f"artifacts above)"
        )
        return 1
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from atap.demo import run_demo

    run_demo(seed=args.seed, out=args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from atap.compare import run_compare

    comparison = run_compare(args.config, args.out, traces=args.traces)
    # symmetric with `run`: isolated algorithm failures in any compared
    # config must surface in the exit status (the errors column shows it,
    # but a partially-failed comparison should not read as clean)
    n_errors = sum(row.get("errors", 0) for row in comparison.get("rows", []))
    if n_errors:
        print(
            f"ERROR: {n_errors} isolated algorithm failure(s) across the "
            f"compared runs (see the errors column / per-run stage_log)"
        )
        return 1
    return 0


def _cmd_corpus(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from atap.sandbox import ToySandbox

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.drift:
        traces = ToySandbox().generate_drift_corpus()
    else:
        traces = ToySandbox().generate_corpus(
            successes_per_task=args.successes_per_task
        )
    with out.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    print(f"corpus: {len(traces)} traces -> {out}")
    return 0


def _cmd_taxonomy(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    run_dir = Path(args.run_dir)
    artifacts = sorted(run_dir.glob("artifacts/*/classify__inducer.json"))
    if not artifacts:
        log.error(
            "no classify__inducer artifact found under %s (first run a "
            "pipeline containing the inducer)",
            run_dir,
        )
        return 1
    art = json.loads(artifacts[0].read_text(encoding="utf-8"))
    proposals = art.get("proposals", [])
    prop = next((p for p in proposals if p["mode_id"] == args.id), None)
    if prop is None:
        ids = [p["mode_id"] for p in proposals] or ["(no proposals)"]
        log.error("proposal %s does not exist; available: %s", args.id, ids)
        return 1

    out = Path(args.out)
    data: dict = {"modes": []}
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        data.setdefault("modes", [])
    if any(m.get("code") == prop["mode_id"] for m in data["modes"]):
        print(f"{prop['mode_id']} already in {out}, skipped")
        return 0
    data["modes"].append({
        "code": prop["mode_id"],
        "name": prop["name"],
        "definition": prop["definition"],
        "kinship": prop.get("kinship"),
        "accepted_from": str(artifacts[0]),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"accepted {prop['mode_id']} [{prop['name']}] -> {out}")
    print("once mast_judge loads this file via params.extra_modes_file, it can label with the new code")
    return 0


def _ensure_flattened(traces: list) -> int:
    """Flatten raw-span-only trajectories in place via represent/canonical_events.

    Sandbox-original JSONL carries ``events=[]`` with only a nested
    ``raw["spans"]`` tree; the exporters consume the flattened R0 event
    stream, so without this step such a trajectory would export as a bare
    trace-create (Langfuse) or zero spans (OTel) and silently lose every
    event [fix]. Returns the number of trajectories flattened (already-flat
    trajectories are left untouched; flattening is the same registry
    algorithm the pipeline uses, so ids/refs are assigned identically).
    """
    import atap  # noqa: F401  registration bootstrap (canonical_events self-registers)

    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext
    from atap.core.registry import create

    todo = [
        t for t in traces
        if not t.events and isinstance(t.raw, dict)
        and isinstance(t.raw.get("spans"), list) and t.raw["spans"]
    ]
    if not todo:
        return 0
    rep = create("represent", "canonical_events")
    ctx = RunContext()
    for t in todo:
        rep.run_one(TrajectoryBundle(t), ctx)
    return len(todo)


def _cmd_export(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from atap.io import export_langfuse, export_otel
    from atap.io.jsonl_store import JSONLTraceSource

    traces = JSONLTraceSource(args.traces).load()
    n_flattened = _ensure_flattened(traces)
    if n_flattened:
        # observable record: the flattening must not happen silently
        log.info(
            "export: flattened %d raw-span-only trace(s) (empty events, raw "
            "spans present) via represent/canonical_events before export",
            n_flattened,
        )
    # args.format is argparse-constrained to langfuse|otel, no dead branch here
    if args.format == "langfuse":
        payload = export_langfuse(traces)
    else:
        payload = export_otel(traces)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    note = f"; flattened {n_flattened} raw-span-only trace(s)" if n_flattened else ""
    print(f"export: {len(traces)} traces -> {out} ({args.format}{note})")
    return 0


def _cmd_langfuse_eval(args: argparse.Namespace) -> int:
    """External evaluation over a live Langfuse instance: pull -> pipeline -> scores.

    Mapping knobs that have no natural CLI spelling (``outcome_from`` /
    ``agent_keys``) live in the config's ``source`` block when it is a
    ``langfuse_api`` source; CLI flags override the pull-window knobs.
    """
    from atap.core.config import load_config
    from atap.io.langfuse_live import LangfuseAPISource, ScoreWriter
    from atap.runtime import run_config

    cfg = load_config(args.config)
    reps = [s.name for s in cfg.stages.get("represent", [])]
    if "canonical_events" not in reps:
        log.error(
            "the config must include represent/canonical_events: live traces arrive "
            "as a raw span tree and only R0-flattening produces the events every "
            "downstream stage reads"
        )
        return 1
    if cfg.stages.get("recover") and not cfg.sandbox:
        log.warning(
            "recover is configured without a sandbox: reruns of live traces cannot "
            "be re-executed/verified -- consider an analyze/classify/attribute-only "
            "stack for external evaluation"
        )
    src_spec = dict(cfg.source) if cfg.source.get("type") == "langfuse_api" else {}
    source = LangfuseAPISource(
        base_url=args.base_url or src_spec.get("base_url"),
        tags=(args.tags.split(",") if args.tags else src_spec.get("tags")),
        since=args.since or src_spec.get("since"),
        limit=(args.limit if args.limit is not None else src_spec.get("limit")),
        outcome_from=src_spec.get("outcome_from"),
        agent_keys=src_spec.get("agent_keys"),
    )
    traces = source.load()
    n_fail = sum(0 if t.outcome.success else 1 for t in traces)
    print(
        f"langfuse: pulled {len(traces)} trace(s)"
        + (f" (tags={source.tags})" if source.tags else "")
        + (f" since={source.since}" if source.since else "")
        + f"; outcome: failure={n_fail} success={len(traces) - n_fail}"
    )
    if not traces:
        return 0

    bundles, reports = run_config(cfg, args.out, trajectories=traces)
    for i, r in enumerate(reports):
        print(
            f"  round{i}: traces={r.n_traces} failures={r.n_failures} "
            f"attributed={r.n_attributed} reruns={r.n_reruns}(ok={r.n_rerun_success})"
            + (f" errors={r.n_errors}" if r.n_errors else "")
        )

    # Batch identity on every written score: Langfuse scores are append-only,
    # so repeated evaluations of one trace are indistinguishable without it.
    # run_id = the --out dir name (unique per run by the fresh-dir contract).
    llm_cfg = cfg.llm or {}
    llm_label = llm_cfg.get("type", "")
    if llm_cfg.get("model"):
        llm_label = f"{llm_label}:{llm_cfg['model']}"
    run_meta = {
        "run_id": str(args.out).rstrip("/").rsplit("/", 1)[-1],
        "run_name": cfg.run_name,
        "llm": llm_label,
        "seed": cfg.seed,
    }
    writer = ScoreWriter(source.client, dry_run=args.dry_run, run_meta=run_meta)
    tally = {"written": 0, "dry-run": 0, "skipped": 0, "no-hypotheses": 0}
    n_scores = 0
    for b in bundles:
        prior = source.scores_by_trace.get(b.trace_id)
        if prior is None:
            continue  # not a pulled trace (e.g. a closed-loop rerun trajectory)
        decision, n = writer.write_bundle(
            b, prior_scores=prior, force=args.force, emit=print
        )
        tally[decision] += 1
        n_scores += n
    label = " (dry-run, nothing written)" if args.dry_run else ""
    print(
        f"scores: {n_scores} written across {tally['written'] or tally['dry-run']} "
        f"trace(s){label}; skipped(already scored)={tally['skipped']} "
        f"no-hypotheses={tally['no-hypotheses']}"
    )
    n_errors = sum(r.n_errors for r in reports)
    if n_errors:
        print(f"ERROR: {n_errors} isolated algorithm failure(s) -- see {args.out}/run.log")
        return 1
    return 0


def _cmd_langfuse_push(args: argparse.Namespace) -> int:
    """Seed a live Langfuse instance with local JSONL trajectories (demo round-trip)."""
    from atap.io.jsonl_store import JSONLTraceSource
    from atap.io.langfuse_live import LangfuseClient, push_langfuse

    traces = JSONLTraceSource(args.traces).load()
    n_flattened = _ensure_flattened(traces)
    if n_flattened:
        log.info(
            "push: flattened %d raw-span-only trace(s) via represent/"
            "canonical_events before export", n_flattened,
        )
    client = LangfuseClient.from_env(args.base_url)
    tags = args.tags.split(",") if args.tags else None
    n_events = push_langfuse(traces, client, tags=tags)
    label = f" (tags={tags})" if tags else ""
    print(
        f"langfuse: pushed {n_events} ingestion event(s) covering "
        f"{len(traces)} trace(s){label}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atap", description="Agent Trace Analysis Platform (ATAP)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG-level process logs (default INFO → stderr + run.log)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a pipeline from a config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default="runs/run")

    sub.add_parser("list", help="list registered algorithms")

    p_demo = sub.add_parser("demo", help="offline end-to-end demo (FakeLLM deterministic judge)")
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.add_argument("--out", default="runs/demo")

    p_cmp = sub.add_parser("compare", help="compare multiple algorithm configs on the same trajectory set")
    p_cmp.add_argument("--config", action="append", required=True,
                       help="config file path, may be given multiple times")
    p_cmp.add_argument("--traces", default=None,
                       help="override the trace source for all configs (guarantees the same trajectory set)")
    p_cmp.add_argument("--out", default="runs/compare")

    p_corpus = sub.add_parser("corpus", help="generate a corpus (default: SBFL spectrum corpus; --drift for a drift corpus)")
    p_corpus.add_argument("--out", default="runs/corpus/traces.jsonl")
    p_corpus.add_argument("--successes-per-task", type=int, default=2)
    p_corpus.add_argument("--drift", action="store_true",
                          help="generate a drift-detection corpus (three constructed drift scenarios)")

    p_tax = sub.add_parser("taxonomy", help="manual acceptance gate for residual taxonomy extension")
    tax_sub = p_tax.add_subparsers(dest="taxonomy_cmd", required=True)
    p_acc = tax_sub.add_parser("accept", help="accept an inducer proposal as an extended mode")
    p_acc.add_argument("--run-dir", required=True, help="run directory containing inducer artifacts")
    p_acc.add_argument("--id", required=True, help="proposal mode_id, e.g. NM-1")
    p_acc.add_argument("--out", default="taxonomy_accepted.json",
                       help="extended modes file (consumed by mast_judge extra_modes_file)")

    p_exp = sub.add_parser("export", help="export JSONL trajectories to Langfuse/OTel format")
    p_exp.add_argument("--traces", required=True, help="input trajectory JSONL")
    p_exp.add_argument("--format", required=True, choices=("langfuse", "otel"))
    p_exp.add_argument("--out", required=True, help="output JSON file")

    p_lfe = sub.add_parser(
        "langfuse-eval",
        help="external evaluation: pull traces from a live Langfuse instance, run the "
             "pipeline, write attribution results back as Scores",
    )
    p_lfe.add_argument("--config", required=True, help="pipeline config (recommend an analyze/classify/attribute stack)")
    p_lfe.add_argument("--out", default="runs/langfuse-eval", help="run output directory (must be fresh per run)")
    p_lfe.add_argument("--base-url", default=None,
                       help="Langfuse base URL (default: LANGFUSE_BASE_URL / LANGFUSE_HOST env)")
    p_lfe.add_argument("--tags", default=None, help="comma-separated trace tags (AND semantics, client-side filter)")
    p_lfe.add_argument("--since", default=None, help="only traces newer than this: '24h'/'7d' or an ISO 8601 timestamp")
    p_lfe.add_argument("--limit", type=int, default=None, help="maximum number of accepted traces")
    p_lfe.add_argument("--dry-run", action="store_true", help="print the scores that would be written, send nothing")
    p_lfe.add_argument("--force", action="store_true", help="re-evaluate traces that already carry an atap:* score")

    p_lfp = sub.add_parser(
        "langfuse-push",
        help="seed a live Langfuse instance with local JSONL trajectories (v3 ingestion batch)",
    )
    p_lfp.add_argument("--traces", required=True, help="input trajectory JSONL (Trajectory.to_dict lines)")
    p_lfp.add_argument("--base-url", default=None,
                       help="Langfuse base URL (default: LANGFUSE_BASE_URL / LANGFUSE_HOST env)")
    p_lfp.add_argument("--tags", default=None,
                       help="comma-separated tags stamped on every pushed trace; pushed corpora carry "
                            "no usable timestamps, so tags are the scoping handle for a later "
                            "`langfuse-eval --tags ...` over exactly this batch")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)
    handlers = {
        "run": _cmd_run,
        "list": _cmd_list,
        "demo": _cmd_demo,
        "compare": _cmd_compare,
        "corpus": _cmd_corpus,
        "taxonomy": _cmd_taxonomy,
        "export": _cmd_export,
        "langfuse-eval": _cmd_langfuse_eval,
        "langfuse-push": _cmd_langfuse_push,
    }
    try:
        return handlers[args.command](args)
    except Exception as e:  # CLI boundary: fail with an explicit error
        log.exception("command %s failed: %s", args.command, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
