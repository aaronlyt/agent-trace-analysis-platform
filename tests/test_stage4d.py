"""Stage 4D tests: Langfuse v3 / OTel GenAI ingestion adapters (roundtrip
fidelity; the Langfuse import accepts both the current v3
span-create/generation-create events and the legacy v2-style
observation-create + body.type form)."""

from __future__ import annotations

import json

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import TraceEvent
from atap.io import (
    LangfuseTraceSource,
    OTelTraceSource,
    export_langfuse,
    export_otel,
)
from atap.io.jsonl_store import build_source
from atap.sandbox import ToySandbox


def _r0_traces():
    sb = ToySandbox()
    return [
        sb.generate("q-trajaudit", None),
        sb.generate("q-who-when", "info_withholding"),
        sb.generate("q-drift", "step_repetition"),
    ]


def _flatten(traces):
    """Flatten input trajectories to R0 (the baseline form before roundtrip)."""
    ctx = RunContext()
    bundles = [TrajectoryBundle(t) for t in traces]
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    return [b.trajectory for b in bundles]


def _sig(t) -> list[tuple]:
    """Semantic signature of a trajectory (roundtrip equivalence criterion: tree structure + event content + ref counts)."""
    def walk(events, parent=None):
        out = []
        for ev in events:
            out.append((
                parent, ev.kind, ev.agent, ev.action,
                json.dumps(ev.payload, sort_keys=True, ensure_ascii=False),
                len(ev.refs), ev.phase,
            ))
        return out
    return walk(t.events)


def _refs_sig(t) -> list[tuple[str, list[str]]]:
    """Reference-edge signature (event id -> referenced target ids).

    Stronger than the ``len(ev.refs)`` inside :func:`_sig`: the referenced
    *target ids* must survive the roundtrip, not just the counts."""
    return [(ev.id, list(ev.refs)) for ev in t.events]


def _roundtrip(tmp_path, fmt, exporter, source_cls):
    traces = _flatten(_r0_traces())
    payload = exporter(traces)
    f = tmp_path / f"export.{fmt}.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    imported = source_cls(str(f)).load()
    # imported payload is a raw span tree -> flatten via canonical_events
    ctx = RunContext()
    bundles = [TrajectoryBundle(t) for t in imported]
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    return traces, [b.trajectory for b in bundles]


def test_langfuse_roundtrip_semantic_equality(tmp_path):
    orig, rt = _roundtrip(tmp_path, "langfuse", export_langfuse,
                          LangfuseTraceSource)
    assert len(rt) == len(orig) == 3
    by_id = {t.trace_id: t for t in rt}
    for o in orig:
        r = by_id[o.trace_id]
        assert r.task == o.task
        # outcome survives in full: success AND score AND note
        assert r.outcome.success == o.outcome.success
        assert r.outcome.score == o.outcome.score
        assert r.outcome.note == o.outcome.note
        # the outcome payload is consumed into Trajectory.outcome on import
        # -- no residue "outcome" key may linger in meta
        assert "outcome" not in r.meta
        assert _sig(r) == _sig(o), f"{o.trace_id}: events not semantically equivalent"
        assert _refs_sig(r) == _refs_sig(o), \
            f"{o.trace_id}: referenced target ids not preserved"
        assert r.meta.get("model_version") == o.meta.get("model_version")
        assert r.meta.get("qrels") == o.meta.get("qrels")
        # GT is not exported (leak prevention)
        assert "injected_fault" not in r.meta


def test_otel_roundtrip_semantic_equality(tmp_path):
    orig, rt = _roundtrip(tmp_path, "otel", export_otel, OTelTraceSource)
    by_id = {t.trace_id: t for t in rt}
    for o in orig:
        r = by_id[o.trace_id]
        assert _sig(r) == _sig(o), f"{o.trace_id}: events not semantically equivalent"
        # outcome survives in full: success AND score AND note
        assert r.outcome.success == o.outcome.success
        assert r.outcome.score == o.outcome.score
        assert r.outcome.note == o.outcome.note
        assert _refs_sig(r) == _refs_sig(o), \
            f"{o.trace_id}: referenced target ids not preserved"
        # trace-level semantics aligned with the langfuse-side assertions
        assert r.task == o.task
        assert r.meta.get("model_version") == o.meta.get("model_version")
        assert r.meta.get("qrels") == o.meta.get("qrels")
        # GT is not exported (leak prevention)
        assert "injected_fault" not in r.meta


def test_langfuse_export_uses_v3_ingestion_schema():
    payload = export_langfuse(_flatten(_r0_traces()))
    evt_types = {e["type"] for e in payload["batch"]}
    # current v3 ingestion event names: the observation type lives in the
    # event name, not in the body
    assert evt_types == {"trace-create", "span-create", "generation-create"}
    obs = next(e for e in payload["batch"] if e["type"] == "generation-create")
    body = obs["body"]
    assert {"id", "traceId", "name", "parentObservationId"} <= set(body)
    assert "type" not in body   # v3 bodies carry no type field
    # GENERATION events are used for LLM_CALL
    gens = [e for e in payload["batch"] if e["type"] == "generation-create"]
    assert gens


def test_otel_export_uses_genai_semconv():
    payload = export_otel(_flatten(_r0_traces()))
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ops = set()
    for kv_attr in (s["attributes"] for s in spans):
        for kv in kv_attr:
            if kv["key"] == "gen_ai.operation.name":
                raw = kv["value"]["stringValue"]
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    pass
                ops.add(raw)
    assert ops <= {"chat", "execute_tool", "invoke_agent", "other"}
    assert "chat" in ops and "execute_tool" in ops


def test_otel_ids_satisfy_otlp_hex_constraints():
    """OTLP hard constraints: traceId = 32 lowercase hex chars, spanId = 16 hex
    chars, parentSpanId omitted when there is no parent (the R0 readable id is
    derived via sha256, the original id survives under atap.*)."""
    import re

    payload = export_otel(_flatten(_r0_traces()))
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans
    for s in spans:
        assert re.fullmatch(r"[0-9a-f]{32}", s["traceId"]), s["traceId"]
        assert re.fullmatch(r"[0-9a-f]{16}", s["spanId"]), s["spanId"]
        if "parentSpanId" in s:
            assert re.fullmatch(r"[0-9a-f]{16}", s["parentSpanId"])
    assert any("parentSpanId" not in s for s in spans)  # root span has no parent key


def test_otel_export_dangling_parent_omits_parent_span_id():
    """A dangling parent (ev.parent points to an id absent from the trace's
    events) must be treated like parentless: the parentSpanId key is omitted
    entirely -- an empty string would violate the 16-hex constraint."""
    import re

    t = _flatten(_r0_traces())[0]
    # sandbox traces flatten to a single-level tree; set up one resolvable
    # parent link and one dangling parent link
    t.events[1].parent = t.events[0].id   # resolvable within the trace
    t.events[2].parent = "e999"           # id not among this trace's events
    payload = export_otel([t])
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == len(t.events)   # no span silently dropped
    by_ev = {}
    for s in spans:
        for kv in s["attributes"]:
            if kv["key"] == "atap.ev_id":
                by_ev[kv["value"]["stringValue"]] = s
    # resolvable parent -> valid 16-hex parentSpanId (keeps the test non-vacuous)
    assert re.fullmatch(r"[0-9a-f]{16}", by_ev[t.events[1].id]["parentSpanId"])
    # dangling parent -> key omitted entirely (never the empty-string form)
    assert "parentSpanId" not in by_ev["e002"]
    for s in spans:
        assert s.get("parentSpanId") != ""
        if "parentSpanId" in s:
            assert re.fullmatch(r"[0-9a-f]{16}", s["parentSpanId"])


def test_build_source_dispatches_new_types(tmp_path):
    traces = _flatten(_r0_traces())
    lf = tmp_path / "lf.json"
    lf.write_text(json.dumps(export_langfuse(traces), ensure_ascii=False),
                  encoding="utf-8")
    src = build_source({"type": "langfuse", "path": str(lf)})
    assert isinstance(src, LangfuseTraceSource)
    assert len(src.load()) == 3
    ot = tmp_path / "ot.json"
    ot.write_text(json.dumps(export_otel(traces), ensure_ascii=False),
                  encoding="utf-8")
    src2 = build_source({"type": "otel", "path": str(ot)})
    assert isinstance(src2, OTelTraceSource)
    assert len(src2.load()) == 3
    from atap.core.config import ConfigError
    with pytest.raises(ConfigError, match="langfuse"):
        build_source({"type": "nope", "path": "x"})


def test_cli_export_roundtrip(tmp_path):
    from atap.cli import main

    traces = _r0_traces()
    jl = tmp_path / "traces.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    out = tmp_path / "lf.json"
    assert main(["export", "--traces", str(jl), "--format", "langfuse",
                 "--out", str(out)]) == 0
    assert len(LangfuseTraceSource(str(out)).load()) == 3


def _write_raw_span_only_jsonl(tmp_path):
    """Sandbox-original JSONL form: events=[] with only a nested raw span tree."""
    jl = tmp_path / "raw.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for t in _r0_traces():
            assert t.events == [] and t.raw and t.raw["spans"]
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return jl


def _expected_event_counts():
    return {t.trace_id: len(t.events) for t in _flatten(_r0_traces())}


def _flatten_loaded(imported):
    ctx = RunContext()
    bundles = [TrajectoryBundle(t) for t in imported]
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    return [b.trajectory for b in bundles]


def test_cli_export_flattens_raw_span_only_traces_langfuse(tmp_path, capsys):
    """Raw-span-only trajectories (sandbox-original JSONL) must not lose their
    events on export: the CLI flattens them first and leaves an observable
    record (stderr log + stdout summary count)."""
    from atap.cli import main

    expected = _expected_event_counts()
    jl = _write_raw_span_only_jsonl(tmp_path)
    out = tmp_path / "lf.json"
    assert main(["export", "--traces", str(jl), "--format", "langfuse",
                 "--out", str(out)]) == 0
    # observable record: stderr process log and the stdout summary both
    # carry the flattening count
    captured = capsys.readouterr()
    assert "flattened 3 raw-span-only" in captured.err
    assert "flattened 3 raw-span-only" in captured.out
    payload = json.loads(out.read_text(encoding="utf-8"))
    observations = [e for e in payload["batch"]
                    if e["type"] in ("span-create", "generation-create")]
    assert len(observations) == sum(expected.values())   # events not dropped
    for t in _flatten_loaded(LangfuseTraceSource(str(out)).load()):
        assert len(t.events) == expected[t.trace_id]
        assert t.events   # explicit: no empty-event trace survives silently


def test_cli_export_flattens_raw_span_only_traces_otel(tmp_path):
    """Same guarantee for the OTel side: raw-span-only input exports one span
    per flattened event (previously zero spans, the whole trace lost)."""
    from atap.cli import main

    expected = _expected_event_counts()
    jl = _write_raw_span_only_jsonl(tmp_path)
    out = tmp_path / "ot.json"
    assert main(["export", "--traces", str(jl), "--format", "otel",
                 "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == sum(expected.values())   # events not dropped
    assert spans   # explicit: no zero-span trace survives silently
    for t in _flatten_loaded(OTelTraceSource(str(out)).load()):
        assert len(t.events) == expected[t.trace_id]


def test_roundtrip_pipeline_end_to_end(tmp_path):
    """Export -> import -> run the 4A deterministic stack: rg_ug still yields
    deterministic attribution on the imported trajectory."""
    traces = _flatten(_r0_traces())
    f = tmp_path / "lf.json"
    f.write_text(json.dumps(export_langfuse(traces), ensure_ascii=False),
                  encoding="utf-8")
    imported = LangfuseTraceSource(str(f)).load()
    ctx = RunContext()
    b = TrajectoryBundle(next(
        t for t in imported if "info_withholding" in t.trace_id
    ))
    create("represent", "canonical_events").run_one(b, ctx)
    create("attribute", "rg_ug").run_one(b, ctx)
    art = b.get("attribute", "rg_ug")
    assert art["label"] == "UG_true_extraction"
    assert b.hypotheses()[0].root_cause_code == "utilization_gap"


def test_otel_import_pretty_printed_documents(tmp_path):
    """Pretty-printed OTLP JSON must import: a whole-document json.loads
    runs first (the old line-based strategy crashed on array member lines
    that are not standalone JSON)."""
    traces = _flatten(_r0_traces())
    # pretty-printed single document
    f1 = tmp_path / "pretty_single.json"
    f1.write_text(json.dumps(export_otel(traces), indent=2), encoding="utf-8")
    imported = OTelTraceSource(str(f1)).load()
    assert {t.trace_id for t in imported} == {t.trace_id for t in traces}
    # pretty-printed array of two disjoint documents
    f2 = tmp_path / "pretty_array.json"
    f2.write_text(
        json.dumps(
            [export_otel(traces[:1]), export_otel(traces[1:])], indent=2
        ),
        encoding="utf-8",
    )
    imported2 = OTelTraceSource(str(f2)).load()
    assert {t.trace_id for t in imported2} == {t.trace_id for t in traces}
    # JSONL fallback (one document per line) still works
    f3 = tmp_path / "docs.jsonl"
    f3.write_text(
        "\n".join(
            json.dumps(export_otel([t]), ensure_ascii=False)
            for t in traces
        ),
        encoding="utf-8",
    )
    imported3 = OTelTraceSource(str(f3)).load()
    assert {t.trace_id for t in imported3} == {t.trace_id for t in traces}


def _legacy_langfuse_batch(payload):
    """Current v3 export -> legacy v2-style batch (observation-create with
    body.type=SPAN/GENERATION)."""
    out = []
    for evt in payload["batch"]:
        if evt["type"] == "trace-create":
            out.append(evt)
        else:
            body = dict(evt["body"])
            body["type"] = (
                "GENERATION" if evt["type"] == "generation-create" else "SPAN"
            )
            out.append({**evt, "type": "observation-create", "body": body})
    return {"batch": out}


def test_langfuse_import_accepts_legacy_observation_create(tmp_path):
    """Backward compatibility: legacy observation-create batches import and
    roundtrip identically to the current span-create/generation-create form
    (both directions of the roundtrip hold)."""
    traces = _flatten(_r0_traces())
    f = tmp_path / "legacy.json"
    f.write_text(
        json.dumps(_legacy_langfuse_batch(export_langfuse(traces)),
                   ensure_ascii=False),
        encoding="utf-8",
    )
    imported = LangfuseTraceSource(str(f)).load()
    ctx = RunContext()
    bundles = [TrajectoryBundle(t) for t in imported]
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    by_id = {b.trace_id: b.trajectory for b in bundles}
    for o in traces:
        r = by_id[o.trace_id]
        assert _sig(r) == _sig(o), f"{o.trace_id}: legacy import differs"
        assert _refs_sig(r) == _refs_sig(o)
        assert r.outcome.success == o.outcome.success
        assert "injected_fault" not in r.meta


def test_export_strips_rerun_gt_copies():
    """Rerun trajectories carry ``origin_fault`` (the full injected-fault
    copy kept for chain re-attribution) and ``fault_removed`` (the
    environment's construction-side removal signal) in meta -- the export
    deny-list must strip both, not just ``injected_fault`` (review
    2026-08-27 P0: the GT used to leave verbatim through both exporters
    under a different key)."""
    sb = ToySandbox()
    b = TrajectoryBundle(sb.generate("q-who-when", "info_withholding"))
    create("represent", "canonical_events").run_one(b, RunContext())
    t = b.trajectory
    gt = t.meta["injected_fault"]
    rerun = sb.rerun_from(t, gt["step"], "faithfully report the documents")
    assert "injected_fault" not in rerun.meta          # _copy_meta drops it
    assert "origin_fault" in rerun.meta and "fault_removed" in rerun.meta
    assert rerun.meta["origin_fault"] == gt            # the full GT copy
    for payload in (export_langfuse([rerun]), export_otel([rerun])):
        blob = json.dumps(payload)
        for key in ("injected_fault", "origin_fault", "fault_removed"):
            assert key not in blob, f"{key} leaked through export"
        # the GT payload itself (mast_code exists only in fault meta) never
        # leaves through either exporter
        assert gt["mast_code"] not in blob
