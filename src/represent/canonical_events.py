"""R0 canonical events -- flatten the span tree into a unified event stream (representation-layer base algorithm).

Mechanism aligns with the AgentTrajectory event representation (AgentDebugX,
arXiv:2607.18754) and the overall architecture section 3 R0: the input to all
analysis. Collection-layer outputs are nested span trees (for both the
sandbox/Langfuse adapters); this algorithm is responsible for:
* DFS flattening with global sequence numbers (index) and canonical event ids
  (``e000``...); ``ts`` prefers the node's own timestamp when the collection
  layer provides a numeric one (Langfuse metadata ``ts``; the node dict key is
  ``"ts"``) and falls back to ``float(idx)`` otherwise -- the framework still
  has no analytic consumer of ts (only io/otel export maps it to
  startTimeUnixNano), so idx-typed traces are unchanged [declared];
* preserving parent (span parent-child) and refs (semantic reference edges,
  span id -> event id mapping); refs pointing at a span id absent from the
  tree are dropped and counted (``dropped_refs``);
* kind admission: collection adapters may emit kinds outside EVENT_KINDS
  (io/otel.py and io/langfuse.py fall back to ``"SPAN"`` for unknown
  operations) -- flatten admits every kind into the vocabulary: known
  out-of-vocab values get an explicit alias (``SPAN`` → AGENT_MESSAGE,
  ``GENERATION`` → LLM_CALL), any other unknown kind falls back to
  AGENT_MESSAGE (the most neutral agent-side member: excluded from R5 action
  signatures, so spectra gain no fabricated action classes; still an acting
  event for binary_search walk-back); alias/fallback admissions are counted
  as ``remapped_kinds``, while a value that is in-vocab after **pure
  case/whitespace normalization** (``tool_call`` → TOOL_CALL) is not counted
  [adaptation];
* a span node missing the ``id`` key raises an explicit :class:`ValueError`
  (message carries the span kind and flattened position) instead of a bare
  KeyError -- refs are span-id anchored, so an id-less span cannot be
  referenced;
* duplicate span ids: the span id -> event mapping keeps the **first**
  occurrence (a later duplicate must not steal refs); occurrences beyond the
  first are counted as ``duplicate_span_ids`` (previously the last silently
  overwrote, mis-pointing refs) [adaptation];
* if the trajectory is already flat R0 (re-run trajectories / handwritten
  fixtures), only normalization is applied (fill in id/index).

Artifacts: ``{"n_events", "kinds", "agents", "n_refs", "dropped_refs",
"duplicate_span_ids", "remapped_kinds"}`` plus ``"source_span_ids"`` -- the
originating span id per event, parallel to the event order (emitted only when
a span tree was flattened; already-flat trajectories keep the legacy shape).
Consumers that must pin an event back to its collection-layer span (e.g. the
live-Langfuse blamed-step write-back) read this instead of replaying the
walk, so mapping and flattening can never drift apart. Events are written
back to ``trajectory.events`` (the representation layer is the sole data
interface for downstream consumers).
"""

from __future__ import annotations

from collections import Counter

from atap.core.registry import register
from atap.core.schema import (
    AGENT_MESSAGE,
    EVENT_KINDS,
    LLM_CALL,
    TraceEvent,
)
from atap.represent.base import Representer

#: collection-layer out-of-vocab kind values -> nearest EVENT_KINDS member
#: (SPAN: generic observation span from otel/langfuse unknown-op fallback;
#: GENERATION: Langfuse model-call observation type) [adaptation]
_KIND_ALIASES = {
    "SPAN": AGENT_MESSAGE,
    "GENERATION": LLM_CALL,
}
#: fallback for any other unknown kind -- the most neutral agent-side member
_KIND_FALLBACK = AGENT_MESSAGE


def _admit_kind(kind: object) -> str:
    """Admit a node kind into EVENT_KINDS (in-vocab passes through; known
    out-of-vocab values are aliased; anything else falls back)."""
    k = str(kind or "").strip().upper()
    if k in EVENT_KINDS:
        return k
    return _KIND_ALIASES.get(k, _KIND_FALLBACK)


@register
class CanonicalEventsRepresenter(Representer):
    stage = "represent"
    name = "canonical_events"

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        dropped = 0
        duplicates = 0
        remapped = 0
        if t.raw and isinstance(t.raw.get("spans"), list):
            t.events, dropped, duplicates, remapped, span_ids = self._flatten(t.raw["spans"])
        else:
            t.events = self._normalize(t.events)
            span_ids = None
        artifact = {
            "n_events": len(t.events),
            "kinds": dict(Counter(ev.kind for ev in t.events)),
            "agents": t.agents(),
            "n_refs": sum(len(ev.refs) for ev in t.events),
            "dropped_refs": dropped,
            "duplicate_span_ids": duplicates,
            "remapped_kinds": remapped,
        }
        if span_ids is not None:
            artifact["source_span_ids"] = span_ids
        bundle.put("represent", self.name, artifact)

    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(
        spans: list[dict],
    ) -> tuple[list[TraceEvent], int, int, int, list[str]]:
        events: list[TraceEvent] = []
        span_ids: list[str] = []  # originating span id per event, same order
        by_span: dict[str, TraceEvent] = {}  # span id -> event (first occurrence wins)
        raw_refs: dict[str, list[str]] = {}  # event id -> raw span references
        dropped = 0
        duplicates = 0
        remapped = 0

        def walk(nodes: list[dict], parent_eid: str | None) -> None:
            nonlocal dropped, duplicates, remapped
            for node in nodes:
                idx = len(events)
                eid = f"e{idx:03d}"
                raw_kind = str(node.get("kind") or "")
                kind = _admit_kind(raw_kind)
                if kind != raw_kind and kind != raw_kind.strip().upper():
                    # an in-vocab value after pure case/whitespace
                    # normalization ("tool_call") is not an out-of-vocab
                    # remap; only alias/fallback admissions count
                    remapped += 1
                # node's own timestamp wins when numeric (Langfuse metadata
                # ts); otherwise fall back to the sequence number [declared]
                ts = node.get("ts")
                if isinstance(ts, bool) or not isinstance(ts, (int, float)):
                    ts = idx
                ev = TraceEvent(
                    id=eid,
                    ts=float(ts),
                    kind=kind,
                    agent=node.get("agent", "unknown"),
                    action=node.get("action"),
                    payload=dict(node.get("payload") or {}),
                    refs=[],
                    phase=node.get("phase"),
                    parent=parent_eid,
                    index=idx,
                )
                events.append(ev)
                span_id = node.get("id")
                if span_id is None:
                    # explicit instead of a bare KeyError: refs are span-id
                    # anchored, so a span without an id cannot be referenced
                    raise ValueError(
                        f"span tree node at flattened index {idx} "
                        f"(kind={raw_kind or '?'}, agent={ev.agent}, "
                        f"parent={parent_eid or 'None'}) is missing the "
                        "required 'id' key: every span must carry an id"
                    )
                if span_id in by_span:
                    # duplicate span id: keep the first occurrence so refs
                    # stay anchored to the earliest event; count the trace
                    duplicates += 1
                else:
                    by_span[span_id] = ev
                # one entry per event, duplicates included: the artifact stays
                # parallel to ``events`` even when ids repeat
                span_ids.append(str(span_id))
                raw_refs[eid] = list(node.get("refs") or [])
                walk(node.get("children") or [], eid)

        walk(spans, None)

        for ev in events:
            refs = raw_refs[ev.id]
            mapped = [by_span[r].id for r in refs if r in by_span]
            dropped += len(refs) - len(mapped)
            ev.refs = mapped
        return events, dropped, duplicates, remapped, span_ids

    @staticmethod
    def _normalize(events: list[TraceEvent]) -> list[TraceEvent]:
        out: list[TraceEvent] = []
        for i, ev in enumerate(events):
            ev.index = i
            if not ev.id:
                ev.id = f"e{i:03d}"
            if ev.ts < 0:
                ev.ts = float(i)
            out.append(ev)
        return out
