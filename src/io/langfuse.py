"""Langfuse v3 collection adapter -- ingestion batch import/export (zero service dependencies).

Data model (Langfuse data-model, surveyed 2026-08): Trace is the top-level
unit, and Observations form a tree via ``parentObservationId``
(type=SPAN/GENERATION/EVENT); the classic ingestion endpoint
``POST /api/public/ingestion`` takes batch events
``{id, type: trace-create|observation-create, timestamp, body}``.
Note that v4 deprecates this endpoint (cloud shutdown 2026-11) in favor of
OTLP -- v4's OTel spans are mapped via ``langfuse.*`` attributes, and this
adapter's span-attribute parsing applies equally (see otel.py; the GT strip
list is maintained inline separately in both places).

Mapping (R0 ↔ Langfuse; unmapped information goes wholesale into
``metadata`` to prevent secondary loss):
* R0 event → observation: LLM_CALL→GENERATION, other kinds→SPAN
  (kind/agent/action/phase/refs/index/ts stored in ``metadata["atap"]`` --
  refs are reference edges absent from the Langfuse model and must survive
  via metadata; ts is a synthetic ordinal and is not mapped to a fake ISO
  startTime [adaptation]);
* event-level ``timestamp`` gets an epoch placeholder, and ingestion id /
  observation id use deterministic prefixes (the canonical form calls for
  real creation times / UUIDv4 -- deterministic roundtrip takes priority
  [adaptation]; times are not trustworthy when consumed by external UIs);
* parentObservationId ← R0 ``parent`` (span tree structure preserved);
* import direction: observation tree → nested span tree (Trajectory.raw),
  flattened by represent/canonical_events -- the inter-layer contract does
  not change with the collection format;
* trace level: input→task, metadata keys (model_version/prompt_version/
  time_window/qrels/...) → Trajectory.meta; the ``outcome`` payload is
  consumed into ``Trajectory.outcome`` (score/note fully restored via
  Outcome.from_dict) and leaves no residue key in meta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atap.core.schema import Outcome, Trajectory

if TYPE_CHECKING:
    pass

#: namespace for R0 event information under observation metadata
_ATAP_NS = "atap"
_KIND_TO_TYPE = {"LLM_CALL": "GENERATION"}


def export_langfuse(traces: list[Trajectory]) -> dict[str, Any]:
    """R0 trajectory list → v3 ingestion batch dict (used both for roundtrip and external consumption).

    Consumes the flattened R0 event stream (``trajectory.events``); a
    raw-span-only trajectory (empty events + raw spans) would export as a
    bare trace-create and lose every event -- flatten it first via
    represent/canonical_events (the CLI export path does this; see
    ``cli._ensure_flattened``).
    """
    batch: list[dict[str, Any]] = []
    for t in traces:
        batch.append({
            "id": f"evt-{t.trace_id}-trace",
            "type": "trace-create",
            "timestamp": "1970-01-01T00:00:00Z",
            "body": {
                "id": t.trace_id,
                "name": t.meta.get("task_id") or t.trace_id,
                "input": t.task,
                "metadata": {
                    **{k: v for k, v in t.meta.items()
                       if k not in ("injected_fault",)},   # GT not exported (leak prevention)
                    "outcome": t.outcome.to_dict(),
                },
            },
        })
        obs_parent: dict[str, str | None] = {}
        for ev in t.events:
            obs_id = f"obs-{t.trace_id}-{ev.id}"
            obs_parent[ev.id] = obs_id
            batch.append({
                "id": f"evt-{t.trace_id}-{ev.id}",
                "type": "observation-create",
                "timestamp": "1970-01-01T00:00:00Z",
                "body": {
                    "id": obs_id,
                    "traceId": t.trace_id,
                    "type": _KIND_TO_TYPE.get(ev.kind, "SPAN"),
                    "name": ev.action or ev.kind,
                    "parentObservationId": obs_parent.get(ev.parent),
                    "input": dict(ev.payload),
                    "metadata": {
                        _ATAP_NS: {
                            "id": ev.id,
                            "kind": ev.kind, "agent": ev.agent,
                            "action": ev.action, "phase": ev.phase,
                            "refs": list(ev.refs), "index": ev.index,
                            "ts": ev.ts,
                        },
                    },
                },
            })
    return {"batch": batch}


class LangfuseTraceSource:
    """Read trajectories from Langfuse v3 ingestion batch JSON (single file or multi-batch JSONL)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _events(self) -> list[dict[str, Any]]:
        p = Path(self.path)
        text = p.read_text(encoding="utf-8").strip()
        out: list[dict[str, Any]] = []
        if not text:
            return out
        first = text[0]
        if first == "{":
            data = json.loads(text)
            return data.get("batch") or []
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.extend(json.loads(line).get("batch") or [])
        return out

    def load(self) -> list[Trajectory]:
        traces: dict[str, dict[str, Any]] = {}
        observations: dict[str, list[dict[str, Any]]] = {}
        for evt in self._events():
            body = evt.get("body") or {}
            if evt.get("type") == "trace-create":
                traces[body.get("id")] = {
                    "name": body.get("name"),
                    "input": body.get("input"),
                    "metadata": body.get("metadata") or {},
                }
            elif evt.get("type") == "observation-create":
                tid = body.get("traceId")
                observations.setdefault(tid, []).append(body)

        out: list[Trajectory] = []
        for tid, tr in traces.items():
            obs_list = observations.get(tid, [])
            nodes: dict[str, dict[str, Any]] = {}
            obs_id_by_ev: dict[str, str] = {}   # original event id → observation id
            for o in obs_list:
                meta = (o.get("metadata") or {}).get(_ATAP_NS) or {}
                nodes[o["id"]] = {
                    "id": o["id"],
                    "logical": o.get("name") or meta.get("kind") or "step",
                    "kind": meta.get("kind")
                    or ("LLM_CALL" if o.get("type") == "GENERATION" else "SPAN"),
                    "agent": meta.get("agent", "unknown"),
                    "action": meta.get("action"),
                    "payload": dict(o.get("input") or {}),
                    "refs": [],
                    "phase": meta.get("phase"),
                    "ts": meta.get("ts"),
                    "children": [],
                    "_parent": o.get("parentObservationId"),
                    "_refs": list(meta.get("refs") or []),
                }
                if meta.get("id"):
                    obs_id_by_ev[str(meta["id"])] = o["id"]
            for node in nodes.values():
                # refs are remapped from original event ids to observation ids
                # (canonical_events maps reference edges by span id)
                node["refs"] = [
                    obs_id_by_ev[r] for r in node.pop("_refs")
                    if r in obs_id_by_ev
                ]
            roots: list[dict[str, Any]] = []
            for node in nodes.values():
                parent = node.pop("_parent")
                if parent in nodes:
                    nodes[parent]["children"].append(node)
                else:
                    roots.append(node)

            tmeta = dict(tr.get("metadata") or {})
            tmeta.setdefault("task_id", tr.get("name") or tid)
            # consume the export-side outcome payload (no residue key in meta:
            # the outcome lives in Trajectory.outcome, restored below)
            raw_outcome = tmeta.pop("outcome", None)
            out.append(Trajectory(
                trace_id=tid,
                task=str(tr.get("input") or tmeta.get("task_id") or ""),
                events=[],
                outcome=Outcome.from_dict(raw_outcome)
                if isinstance(raw_outcome, dict) else Outcome(success=False),
                meta=tmeta,
                raw={"task_id": tmeta.get("task_id"), "spans": roots},
            ))
        return out
