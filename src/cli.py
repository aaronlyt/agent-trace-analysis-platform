"""atap command-line entry point.

Usage::

    atap run --config configs/pipeline_offline.yaml [--out runs/demo]
    atap list                       # list registered algorithms
    atap demo [--seed 42] [--out runs/demo]   # phase two: offline end-to-end demo
    atap -v run ...                 # verbose (DEBUG-level process logs)

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
    from atap.runtime import run_config

    cfg = load_config(args.config)
    _, reports = run_config(cfg, args.out)
    print(f"run={cfg.run_name} rounds={len(reports)}")
    for i, r in enumerate(reports):
        print(
            f"  round{i}: traces={r.n_traces} failures={r.n_failures} "
            f"attributed={r.n_attributed} reruns={r.n_reruns}(ok={r.n_rerun_success})"
        )
    print(f"artifacts -> {args.out}/artifacts")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from atap.demo import run_demo

    run_demo(seed=args.seed, out=args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from atap.compare import run_compare

    run_compare(args.config, args.out, traces=args.traces)
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
    if args.format == "langfuse":
        payload = export_langfuse(traces)
    elif args.format == "otel":
        payload = export_otel(traces)
    else:
        log.error("unknown export format %r (langfuse | otel)", args.format)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    note = f"; flattened {n_flattened} raw-span-only trace(s)" if n_flattened else ""
    print(f"export: {len(traces)} traces -> {out} ({args.format}{note})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atap", description="Agent trajectory analysis and error attribution platform")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="DEBUG-level process logs (default INFO → stderr + run.log)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a pipeline from a config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default="runs/run")

    p_list = sub.add_parser("list", help="list registered algorithms")

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
    }
    try:
        return handlers[args.command](args)
    except Exception as e:  # CLI boundary: fail with an explicit error
        log.exception("command %s failed: %s", args.command, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
