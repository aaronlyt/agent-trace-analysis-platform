"""R5 action signature (TraceProbe 2607.06184) unit and integration tests."""

from __future__ import annotations

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import TraceEvent
from atap.represent.action_signature import (
    ACTION_CLASSES,
    EFFECT_LABELS,
    ActionSignatureRepresenter,
    classify_event,
)
from atap.sandbox import ToySandbox


def _bundle(task="q-trajaudit", fault=None):
    t = ToySandbox().generate(task, fault)
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    return b, ctx


def _sig_map(b):
    return {s["index"]: s for s in b.get("represent", "action_signature")["signatures"]}


# ---------------------------------------------------------------- mapping --

def test_classify_event_mapping():
    def ev(kind, **kw):
        return TraceEvent(id="e0", ts=0.0, kind=kind, agent="a", **kw)

    assert classify_event(ev("TOOL_CALL", action="search", payload={"query": "Q"})) == ("SEARCH", "q")
    assert classify_event(ev("TOOL_CALL", action="read_doc", payload={"doc_id": "d1"})) == ("FILE_READ", "d1")
    assert classify_event(ev("TOOL_CALL", action="submit", payload={"answer": "x"})) == ("COMMAND", "submit")
    assert classify_event(ev("VERIFIER")) == ("COMMAND", "verify")
    assert classify_event(ev("HANDOFF", payload={"to": "searcher"}))[0] == "AGENT_SPAWN"
    assert classify_event(ev("LLM_CALL", phase="plan"))[0] == "PLAN"
    assert classify_event(ev("LLM_CALL", phase="search"))[0] == "REASON"
    assert classify_event(ev("TASK_START")) is None
    assert classify_event(ev("TASK_END")) is None
    assert classify_event(ev("TOOL_RESULT")) is None


def test_label_sets_match_paper():
    assert set(ACTION_CLASSES) == {
        "FILE_READ", "FILE_WRITE", "SEARCH", "COMMAND", "PLAN",
        "NAVIGATE", "FETCH", "AGENT_SPAWN", "REASON",
    }
    assert set(EFFECT_LABELS) == {
        "SURVIVED", "FAILED", "REVERTED", "JUSTIFIED",
        "RECORDED", "OFF-ANCHOR", "REASONING",
    }


# ------------------------------------------------------------ single trajectory --

def test_run_one_degrades_without_corpus_reference():
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    ActionSignatureRepresenter().run_one(b, ctx)
    art = b.get("represent", "action_signature")
    assert art["anchor"] is None and art["milestones"] is None
    assert "skipped" in art["note"]
    assert art["signatures"]


def test_effects_on_fault_traces():
    b, _ = _bundle("q-trajaudit", "malformed_tool_call")
    ActionSignatureRepresenter().run_one(b, ctx=RunContext())
    m = _sig_map(b)
    assert m[3]["action_class"] == "SEARCH" and m[3]["effect"] == "FAILED"  # malformed call

    b2, _ = _bundle("q-trajaudit", "step_repetition")
    ActionSignatureRepresenter().run_one(b2, ctx=RunContext())
    m2 = _sig_map(b2)
    assert m2[14]["effect"] == "FAILED"  # submit verification failed (budget exhausted)


# ------------------------------------------------------------- corpus --

def _corpus_bundles(fault="step_repetition"):
    sb = ToySandbox()
    bundles = []
    for t in sb.generate_corpus(successes_per_task=1):
        b = TrajectoryBundle(t)
        bundles.append(b)
    ctx = RunContext()
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    ActionSignatureRepresenter().run_corpus(bundles, ctx)
    return bundles


def test_corpus_anchor_from_success_reference():
    bundles = _corpus_bundles()
    by_id = {b.trace_id: b for b in bundles}
    anchor_b = by_id["q-trajaudit--malformed_tool_call"]
    art = anchor_b.get("represent", "action_signature")
    assert art["anchor"]["source"] == "success_reference"
    assert art["anchor"]["docs"] == ["d1"]  # q-trajaudit's gold document
    m = _sig_map(anchor_b)
    # the malformed call itself FAILED; the searcher's reasoning and handoff
    # afterwards keep their meta-actions
    assert m[3]["effect"] == "FAILED"
    assert m[5]["action_class"] == "REASON"
    assert m[6]["action_class"] == "AGENT_SPAWN"


def test_milestones_and_alignment():
    bundles = _corpus_bundles()
    by_id = {b.trace_id: b for b in bundles}
    rep = by_id["q-trajaudit--step_repetition"]
    art = rep.get("represent", "action_signature")
    ms = art["milestones"]
    assert ms["M1_first_anchor_read"]["reached"] is True
    assert ms["M2_first_anchor_search"]["reached"] is True
    assert ms["M3_all_anchors_read"]["reached"] is True
    assert ms["M4_first_passing_validation"]["reached"] is False  # this trajectory fails verification
    ali = art["alignment"]
    assert ali["lcs_len"] > 0 and 0.0 < ali["coverage"] < 1.0
    # the two extra searches vs the success reference (search#1/#2) fall
    # into the divergence span
    assert any(s["start_index"] == 5 for s in ali["divergence_spans"])
    # the success trajectory itself: all milestones reached
    ok = by_id["q-trajaudit--ok0"]
    ok_ms = ok.get("represent", "action_signature")["milestones"]
    assert ok_ms["M4_first_passing_validation"]["reached"] is True


def test_off_anchor_when_search_misses_anchor():
    """Construction: search results lack the anchor document -> OFF-ANCHOR."""
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", None)
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    ok = TrajectoryBundle(sb.generate("q-trajaudit", None))
    create("represent", "canonical_events").run_one(ok, ctx)
    # modify the failing trajectory: the search returns results without d1
    for ev in b.trajectory.events:
        if ev.kind == "TOOL_RESULT" and ev.action == "search":
            ev.payload["content"] = "search results for 'q': 1 docs [d2]"
    b.trajectory.outcome.success = False
    ActionSignatureRepresenter().run_corpus([b, ok], ctx)
    m = _sig_map(b)
    assert m[3]["effect"] == "OFF-ANCHOR"
    assert b.get("represent", "action_signature")["alignment"]["off_anchor_ratio"] > 0


def test_group_without_success_degrades_explicitly():
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", "malformed_tool_call")
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    ActionSignatureRepresenter().run_corpus([b], ctx)
    art = b.get("represent", "action_signature")
    assert art["anchor"] is None
    assert "no successful trajectory" in art["note"]


def test_no_task_id_falls_back_to_run_one():
    from tests.helpers import failure_trace_ungrounded

    b = TrajectoryBundle(failure_trace_ungrounded())
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    ActionSignatureRepresenter().run_corpus([b], ctx)
    art = b.get("represent", "action_signature")
    assert art["anchor"] is None  # no task_id grouping -> single-trajectory degradation path


# ------------------------------------------------------- anchor word boundary --

def test_touches_anchor_word_boundary():
    """Anchor ids must match on word boundaries: anchor d1 hits '[d1]' but
    NOT '[d10, d11]' (plain substrings like '[d1' would also hit '[d10').
    The sandbox corpus ids are d1-d6, so this guard is behavior-neutral on
    the current corpus and only narrows future false hits."""
    from atap.represent.action_signature import _touches_anchor

    def res(content):
        return TraceEvent(
            id="r0", ts=0.0, kind="TOOL_RESULT", agent="env",
            payload={"content": content},
        )

    # positive: sandbox search formats mention d1 with non-alnum neighbors
    assert _touches_anchor(
        "SEARCH", "q", res("search results for 'q': 2 docs [d1, d3]"), {"d1"})
    assert _touches_anchor(
        "SEARCH", "q", res("search results for 'q': 1 docs [d1]"), {"d1"})
    assert _touches_anchor(
        "SEARCH", "q", res("relevant hits: (d1) and d1 alone"), {"d1"})
    # negative: d1 must not hit longer ids / unrelated text
    assert not _touches_anchor(
        "SEARCH", "q", res("search results for 'q': 2 docs [d10, d11]"), {"d1"})
    assert not _touches_anchor(
        "SEARCH", "q", res("no documents [d2]"), {"d1"})
    assert not _touches_anchor("SEARCH", "q", res("no docs"), {"d1"})
    # longer anchors still hit their own ids inside lists
    assert _touches_anchor(
        "SEARCH", "q", res("docs [d10, d11]"), {"d10", "d11"})
    # FILE_READ stays an exact-target membership test (no text matching)
    assert _touches_anchor("FILE_READ", "d1", None, {"d1"})
    assert not _touches_anchor("FILE_READ", "d10", None, {"d1"})
