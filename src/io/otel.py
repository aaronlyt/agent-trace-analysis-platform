"""OTel GenAI collection adapter -- OTLP/HTTP JSON import/export (zero dependencies).

Semantic conventions (semantic-conventions-genai, surveyed 2026-08): spans
must carry ``gen_ai.operation.name`` (enum chat/invoke_agent/execute_tool/
...; events that hit no R0 kind use the custom value ``other`` -- the
registry has no "other" enum, handled under the "unmatched values may use
custom values" rule [adaptation]) and ``gen_ai.provider.name``
(atap-sandbox is also a custom value); content-type attributes are Opt-In
PII (``gen_ai.input.messages``/``gen_ai.output.messages``; the deprecated
gen_ai.prompt/completion are both unused) -- the payload goes wholesale
through the custom ``atap.payload`` attribute, bypassing the Opt-In privacy
intent, acceptable for local-file scenarios [adaptation]); span naming
SHOULD be "{operation} {model/tool}" (action/agent stands in for model, and
gen_ai.request.model is added when meta has model_version [adaptation]);
LLM_CALL span kind=CLIENT(2), others INTERNAL(1).

OTLP id hard constraints: traceId=32 lowercase hex, spanId=16 hex and not
all zeros, parentSpanId omitted when there is no parent (or when the parent
id dangles outside the trace -- an empty value would violate 16-hex) -- R0's
trace_id/ev.id are readable strings ('q-...'/'e003') that do not satisfy
hex, so the export side derives deterministic hex ids via sha256 and the
original ids survive via ``atap.trace_id``/``atap.ev_id`` (the import side
restores the original trace_id; roundtrip fidelity) [adaptation].
Timestamps are synthetic (end=ts+1s; R0 ts itself is a copy of the ordinal).

Mapping (R0 ↔ OTel span; unmapped attributes go wholesale into ``atap.*``
custom attributes to prevent loss):
* gen_ai.operation.name → R0 kind: chat→LLM_CALL, execute_tool→TOOL_CALL,
  invoke_agent→AGENT_MESSAGE, retrieval→TOOL_CALL; others fall back to the
  ``atap.kind`` custom attribute (roundtrip fidelity);
* span tree (parentSpanId) → nested span tree (Trajectory.raw) → flattened
  by canonical_events; reference edges refs survive via ``atap.refs`` (W3C
  has no notion of reference edges);
* tool_call request-response pairing key ``gen_ai.tool.call.id``: when
  present, the import side registers that id as a span alias so original
  event ids in refs can be remapped;
* ``meta["qrels"]`` (gold sufficient set) is kept with trace_meta per the
  project contract (rg_ug data dependency); strip it yourself if strict
  leak prevention is required when exporting to external backends.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from atap.core.schema import Outcome, Trajectory

from atap.io._leak_guard import export_safe_meta

_OP_TO_KIND = {
    "chat": "LLM_CALL",
    "execute_tool": "TOOL_CALL",
    "invoke_agent": "AGENT_MESSAGE",
    "retrieval": "TOOL_CALL",
}


def _hex_id(seed: str, n: int) -> str:
    """R0 readable id → deterministic hex (OTLP requires traceId 32 hex /
    spanId 16 hex, and not all zeros -- sha256 derivation satisfies this
    naturally)."""
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    """Span attributes -> flat dict. Only ``stringValue`` values are read:
    OTLP attribute values carry explicit type tags, and this project's own
    export side writes stringValue exclusively (``_attr_kv``); non-string
    typed values from foreign collectors are skipped rather than coerced
    [adaptation, declared simplification -- no such producer in the current
    pipeline]."""
    out: dict[str, Any] = {}
    for kv in span.get("attributes") or []:
        out[str(kv.get("key"))] = kv.get("value", {}).get("stringValue")
    return out


def _attr_kv(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": {"stringValue": value if isinstance(value, str)
                                  else json.dumps(value, ensure_ascii=False)}}


def export_otel(traces: list[Trajectory]) -> dict[str, Any]:
    """R0 trajectory list → OTLP/HTTP JSON dict.

    Consumes the flattened R0 event stream (``trajectory.events``); a
    raw-span-only trajectory (empty events + raw spans) would export as zero
    spans and lose every event -- flatten it first via
    represent/canonical_events (the CLI export path does this; see
    ``cli._ensure_flattened``).
    """
    spans_out: list[dict[str, Any]] = []
    for t in traces:
        otid = _hex_id(t.trace_id, 32)
        span_id_by_ev: dict[str, str] = {}
        for ev in t.events:
            span_id_by_ev[ev.id] = _hex_id(f"{t.trace_id}:{ev.id}", 16)
        for ev in t.events:
            op = {"LLM_CALL": "chat", "TOOL_CALL": "execute_tool",
                  "AGENT_MESSAGE": "invoke_agent"}.get(ev.kind, "other")
            attrs = [
                _attr_kv("gen_ai.operation.name", op),
                _attr_kv("gen_ai.provider.name", "atap-sandbox"),
                _attr_kv("atap.kind", ev.kind),
                _attr_kv("atap.agent", ev.agent),
                _attr_kv("atap.ev_id", ev.id),
                _attr_kv("atap.trace_id", t.trace_id),
                _attr_kv("atap.payload", ev.payload),
                _attr_kv("atap.refs", ev.refs),
                _attr_kv("atap.action", ev.action or ""),
                _attr_kv("atap.phase", ev.phase or ""),
            ]
            if ev.index == 0:   # trace-level info attaches only to the first event (merged on import)
                attrs += [
                    _attr_kv("atap.trace_meta", export_safe_meta(t.meta)),
                    _attr_kv("atap.task", t.task),
                    _attr_kv("atap.outcome", t.outcome.to_dict()),
                ]
            if op == "chat" and t.meta.get("model_version"):
                attrs.append(
                    _attr_kv("gen_ai.request.model", t.meta["model_version"])
                )
            span: dict[str, Any] = {
                "traceId": otid,
                "spanId": span_id_by_ev[ev.id],
                # span naming SHOULD be "{operation} {model/tool}" (no model
                # dimension; action/agent stands in [adaptation])
                "name": f"{op} {ev.action or ev.agent}",
                # inference SHOULD be CLIENT(2), execute_tool/others INTERNAL(1)
                "kind": 2 if op == "chat" else 1,
                "startTimeUnixNano": str(int(ev.ts * 1_000_000_000)),
                "endTimeUnixNano": str(int((ev.ts + 1) * 1_000_000_000)),
                "attributes": attrs,
            }
            if ev.parent is not None and ev.parent in span_id_by_ev:
                # parentSpanId omitted when parentless (OTLP requirement); a
                # dangling parent (id not among this trace's events) is
                # treated the same way -- an empty value would violate the
                # 16-hex constraint [fix]
                span["parentSpanId"] = span_id_by_ev[ev.parent]
            spans_out.append(span)
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr_kv("service.name", "atap")]},
        "scopeSpans": [{"scope": {"name": "atap.export"}, "spans": spans_out}],
    }]}


class OTelTraceSource:
    """Read trajectories from OTLP/HTTP JSON (single file or multi-document JSONL)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _docs(self) -> list[dict[str, Any]]:
        text = Path(self.path).read_text(encoding="utf-8").strip()
        if not text:
            return []
        try:
            # whole-document JSON first: covers compact and pretty-printed
            # arrays as well as single objects (a pretty array's member lines
            # are not standalone JSON, so a line-first strategy would crash)
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        # fallback: JSONL, one OTLP document per line
        docs = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                docs.append(json.loads(line))
        return docs

    def load(self) -> list[Trajectory]:
        by_trace: dict[str, list[dict[str, Any]]] = {}
        for doc in self._docs():
            for rs in doc.get("resourceSpans") or []:
                for ss in rs.get("scopeSpans") or []:
                    for span in ss.get("spans") or []:
                        by_trace.setdefault(span.get("traceId", ""), []).append(span)

        out: list[Trajectory] = []
        for tid, spans in by_trace.items():
            nodes: dict[str, dict[str, Any]] = {}
            span_id_by_ev: dict[str, str] = {}
            meta: dict[str, Any] = {}
            task = ""
            outcome = None
            orig_trace_id = ""
            for span in spans:
                a = _attrs(span)
                ev_id = a.get("atap.ev_id") or span["spanId"]
                span_id_by_ev[ev_id] = span["spanId"]
                # the request-response pairing key is registered as a span
                # alias (refs can be remapped through it)
                if a.get("gen_ai.tool.call.id"):
                    span_id_by_ev[a["gen_ai.tool.call.id"]] = span["spanId"]
                if a.get("atap.trace_id"):
                    orig_trace_id = a["atap.trace_id"]
                kind = a.get("atap.kind") or _OP_TO_KIND.get(
                    a.get("gen_ai.operation.name", ""), "SPAN"
                )
                if a.get("atap.trace_meta"):
                    try:
                        m = json.loads(a["atap.trace_meta"])
                        if isinstance(m, dict) and m:
                            meta.update(m)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if a.get("atap.task"):
                    task = a["atap.task"]
                if a.get("atap.outcome"):
                    try:
                        o = json.loads(a["atap.outcome"])
                        if isinstance(o, dict) and o:
                            outcome = o
                    except (json.JSONDecodeError, TypeError):
                        pass
                payload: dict[str, Any] = {}
                if a.get("atap.payload"):
                    try:
                        payload = json.loads(a["atap.payload"])
                    except (json.JSONDecodeError, TypeError):
                        payload = {"raw": a["atap.payload"]}
                refs: list[str] = []
                if a.get("atap.refs"):
                    try:
                        refs = json.loads(a["atap.refs"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                nodes[span["spanId"]] = {
                    "id": span["spanId"],
                    "logical": span.get("name") or kind,
                    "kind": kind,
                    "agent": a.get("atap.agent", "unknown"),
                    "action": a.get("atap.action") or None,
                    "payload": payload,
                    "refs": [],
                    "phase": a.get("atap.phase") or None,
                    "children": [],
                    "_parent": span.get("parentSpanId") or None,
                    "_refs": refs,
                }
            for node in nodes.values():
                node["refs"] = [
                    span_id_by_ev[r] for r in node.pop("_refs")
                    if r in span_id_by_ev
                ]
            roots: list[dict[str, Any]] = []
            for node in nodes.values():
                parent = node.pop("_parent")
                if parent in nodes:
                    nodes[parent]["children"].append(node)
                else:
                    roots.append(node)

            meta.pop("injected_fault", None)   # GT must not appear on the import side
            meta.setdefault("task_id", tid)
            out.append(Trajectory(
                # prefer restoring the original trace_id from before export
                # (the OTLP hex id is only a transport representation)
                trace_id=orig_trace_id or tid,
                task=task or str(meta.get("task_id") or ""),
                events=[],
                outcome=Outcome.from_dict(outcome) if outcome else Outcome(success=False),
                meta=meta,
                raw={"task_id": meta.get("task_id"), "spans": roots},
            ))
        return out
