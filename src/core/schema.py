"""R0 canonical event model -- the framework's single data contract.

Aligned with the representation layer R0 of "Overall Pipeline Architecture and
Algorithm Literature" §3: the span tree is flattened into a unified event
stream (kind/agent/action/effect/reference edges/phase), serving as the input
to all analyses; the mechanism aligns with the AgentTrajectory event
representation (AgentDebugX, arXiv:2607.18754).

Design constraints:
* Pure stdlib dataclasses, JSON-serializable (core has zero third-party deps).
* The ``refs`` field holds reference edges (which prior information artifacts
  this event consumed), reserved for the R2 information dependency graph (IDG)
  and root-cause backtracking; ``parent`` preserves the span tree's
  parent-child relations.
* Compared to AgentTrajectory's original 12 fields (type/agent/module/step
  index/parent/timestamp/inputs/outputs/error/duration/metadata/artifacts),
  R0 omits error/duration/metadata/artifacts **and module** (no module
  attribution consumers in this domain) [declared simplification];
  inputs/outputs are **merged** into a single ``payload`` (observation text
  conventionally goes in ``payload["content"]``); error semantics are carried
  by the error-prefix convention (consumed by render.is_error_observation);
  artifacts are left for future multimodal/GUI trajectory needs. The event
  type vocabulary has no dedicated kinds for memory operations / UI actions
  (UI depends on the omitted artifacts).
* ``ts`` on the span-flattening path is a copy of index (canonical_events
  fills it with float(idx); the collection layer may provide real timestamps)
  -- currently no consumer in the framework uses ts [declared].
* ``Trajectory.raw`` may carry the collection layer's raw form (nested span
  tree); represent/canonical_events is responsible for flattening it into
  ``events``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Event types (kind). Values align with the top-level categories of the OTel
# GenAI semantic conventions (see the OTel_GenAI vs OpenInference comparison
# notes), plus supplementary MAS collaboration events.
# ---------------------------------------------------------------------------

TASK_START = "TASK_START"
LLM_CALL = "LLM_CALL"          # model call: reasoning/decision/text generation
TOOL_CALL = "TOOL_CALL"        # tool call request issued by the model side
TOOL_RESULT = "TOOL_RESULT"    # tool observation returned by the environment
AGENT_MESSAGE = "AGENT_MESSAGE"  # inter-agent message (the sent content is itself an information artifact)
HANDOFF = "HANDOFF"            # control/responsibility transfer (A->B)
VERIFIER = "VERIFIER"          # verifier check (task-level or step-level)
TASK_END = "TASK_END"          # termination: normal end/submission/abandonment

EVENT_KINDS = (
    TASK_START,
    LLM_CALL,
    TOOL_CALL,
    TOOL_RESULT,
    AGENT_MESSAGE,
    HANDOFF,
    VERIFIER,
    TASK_END,
)


@dataclass
class TraceEvent:
    """A flattened canonical event.

    Attributes:
        id: unique event identifier (unique within the trace, e.g. ``e007``).
        ts: monotonically increasing timestamp/sequence number (float; the
            collection layer may use real time).
        kind: one of :data:`EVENT_KINDS`.
        agent: acting subject ("planner" / "searcher" / ... / "env" / "verifier").
        action: canonical action name (placeholder for the nine R5 action
            classes; may be None in stage two).
        payload: JSON-native dict; a TOOL_RESULT's observation text
            conventionally goes in ``payload["content"]``.
        refs: reference edges -- ids of prior events this event consumed
            (e.g. a TOOL_RESULT referencing its TOOL_CALL; referencing a
            TOOL_RESULT it read; a HANDOFF referencing the transferred
            message).
        phase: task phase label (e.g. "plan" / "search" / "report").
        parent: parent event id in the span tree (None means top level).
        index: global sequence number after flattening (0-based, assigned by
            canonical_events).
    """

    id: str
    ts: float
    kind: str
    agent: str
    action: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    phase: str | None = None
    parent: str | None = None
    index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "kind": self.kind,
            "agent": self.agent,
            "action": self.action,
            "payload": self.payload,
            "refs": list(self.refs),
            "phase": self.phase,
            "parent": self.parent,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceEvent":
        return cls(
            id=d["id"],
            ts=float(d.get("ts", 0.0)),
            kind=d["kind"],
            agent=d.get("agent", "unknown"),
            action=d.get("action"),
            payload=dict(d.get("payload") or {}),
            refs=list(d.get("refs") or []),
            phase=d.get("phase"),
            parent=d.get("parent"),
            index=int(d.get("index", -1)),
        )


@dataclass
class Outcome:
    """Trajectory outcome label.

    Note (literature constraint, DRIFT 2606.02060): outcome labels must not be
    treated as a process-monitoring gold standard -- 36.9% of successful
    trajectories contain hidden erroneous steps; hence ``success`` is only
    the outcome perspective.
    """

    success: bool
    score: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "score": self.score, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Outcome":
        return cls(
            success=bool(d.get("success", False)),
            score=d.get("score"),
            note=d.get("note", ""),
        )


@dataclass
class Trajectory:
    """A complete trajectory (R0 canonical form).

    Attributes:
        trace_id: unique identifier.
        task: task description (includes expected-answer requirements; excludes
            the gold answer itself -- gold belongs to the verifier only, to
            avoid breaking "the prior provides only failure signals",
            TrajAudit 2605.26563).
        events: R0 canonical event stream (output/input of canonical_events).
        outcome: outcome label.
        meta: collection metadata. Drift-detection grouping keys (model
            version x prompt version x time window, system-level taxonomy
            2511.19933) go in ``meta["model_version"]`` etc.; injected-fault
            ground truth (sandbox-specific) goes in ``meta["injected_fault"]``.
        raw: collection layer's raw form (nested span tree dict); present only
            while the event stream has not yet been flattened.
    """

    trace_id: str
    task: str
    events: list[TraceEvent] = field(default_factory=list)
    outcome: Outcome = field(default_factory=lambda: Outcome(success=False))
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    # -- query helpers --------------------------------------------------------

    def event_by_id(self, eid: str) -> TraceEvent | None:
        for ev in self.events:
            if ev.id == eid:
                return ev
        return None

    def agents(self) -> list[str]:
        """Return the deduplicated agent list in first-appearance order."""
        seen: list[str] = []
        for ev in self.events:
            if ev.agent not in seen:
                seen.append(ev.agent)
        return seen

    # -- serialization --------------------------------------------------------

    def to_dict(self, *, include_raw: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trace_id": self.trace_id,
            "task": self.task,
            "events": [ev.to_dict() for ev in self.events],
            "outcome": self.outcome.to_dict(),
            "meta": self.meta,
        }
        if include_raw and self.raw is not None:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            trace_id=d["trace_id"],
            task=d.get("task", ""),
            events=[TraceEvent.from_dict(e) for e in d.get("events") or []],
            outcome=Outcome.from_dict(d.get("outcome") or {}),
            meta=dict(d.get("meta") or {}),
            raw=d.get("raw"),
        )


@dataclass
class Hypothesis:
    """Unified attribution output (literature §6 contract).

    ranked hypotheses = responsible agent + responsible step + root cause
    label + responsible side + evidence citations + fix suggestion +
    confidence. Any attribution algorithm (L0 rules / L1 judge / L2 deep /
    L3 replay) must produce this structure, consumed by the recovery stage
    and the Error Hub.
    """

    agent: str
    step: int                      # responsible step (R0 event index)
    root_cause: str                # root cause description
    root_cause_code: str | None = None   # taxonomy code (e.g. "FM-1.3")
    responsible_side: str = "model"      # "model" | "harness"
    evidence: list[str] = field(default_factory=list)   # event ids + excerpt citations
    fix_suggestion: str = ""
    confidence: float = 0.0
    source: str = ""               # producing attribution algorithm name (filled
                                   # by bundle.hypotheses() when empty; lets
                                   # cross-algorithm arbitration distinguish origins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "step": self.step,
            "root_cause": self.root_cause,
            "root_cause_code": self.root_cause_code,
            "responsible_side": self.responsible_side,
            "evidence": list(self.evidence),
            "fix_suggestion": self.fix_suggestion,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hypothesis":
        return cls(
            agent=d["agent"],
            step=int(d["step"]),
            root_cause=d.get("root_cause", ""),
            root_cause_code=d.get("root_cause_code"),
            responsible_side=d.get("responsible_side", "model"),
            evidence=list(d.get("evidence") or []),
            fix_suggestion=d.get("fix_suggestion", ""),
            confidence=float(d.get("confidence", 0.0)),
            source=d.get("source", ""),
        )
