"""Loop detection predicates (TraceProbe Table II) tests."""

from __future__ import annotations

import pytest

from atap.analyze.loop_detect import LoopDetectAnalyzer
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox


def _bundle(task="q-trajaudit", fault=None):
    b = TrajectoryBundle(ToySandbox().generate(task, fault))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    return b, ctx


def _detected(b, **params):
    LoopDetectAnalyzer(**params).run_one(b, RunContext())
    return b.get("analyze", "loop_detect")["detected"]


def test_search_loop_fires_on_step_repetition():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    hits = _detected(b, min_consecutive=3)
    assert any(d["predicate"] == "search_loop" for d in hits)
    sl = next(d for d in hits if d["predicate"] == "search_loop")
    assert sl["length"] >= 3
    assert sl["repetition_onset_index"] == 5  # second search = first repetition (GT convention)


def test_search_loop_default_threshold_is_paper_value():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    # the paper's frozen threshold is 10: the toy trajectory (4 consecutive)
    # must not fire -- threshold auditing is left to configuration
    hits = _detected(b)  # min_consecutive defaults to 10
    assert not any(d["predicate"] == "search_loop" for d in hits)


def test_no_detection_on_success_trace():
    b, _ = _bundle("q-trajaudit")
    assert _detected(b, min_consecutive=3) == []


def test_redundant_search_fires():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    hits = _detected(b)
    assert any(d["predicate"] == "redundant_search" for d in hits)


def test_re_read_churn_synthetic():
    """Same document read 3 times within a 10-action window -> re_read_churn."""
    from atap.core.schema import (
        TASK_END,
        TASK_START,
        TOOL_CALL,
        TOOL_RESULT,
        Outcome,
        TraceEvent,
        Trajectory,
    )

    events = [_ev0 := TraceEvent(id="e000", ts=0, kind=TASK_START, agent="env", index=0)]
    idx = 1
    for _ in range(3):
        events.append(TraceEvent(
            id=f"e{idx:03d}", ts=float(idx), kind=TOOL_CALL, agent="searcher",
            action="read_doc", payload={"doc_id": "d1"}, index=idx))
        idx += 1
        events.append(TraceEvent(
            id=f"e{idx:03d}", ts=float(idx), kind=TOOL_RESULT, agent="env",
            action="read_doc", refs=[events[-1].id], payload={"content": "doc"}, index=idx))
        idx += 1
    events.append(TraceEvent(
        id=f"e{idx:03d}", ts=float(idx), kind=TASK_END, agent="env", index=idx))
    t = Trajectory(trace_id="syn", task="t", events=events,
                   outcome=Outcome(success=True))
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    hits = _detected(b)
    assert any(d["predicate"] == "re_read_churn" and d["repeats"] == 3 for d in hits)


def _osc_bundle(seq: list[str], write_effect: str):
    """Bundle whose R5 signatures are rewritten into a FILE_READ/FILE_WRITE
    (f.py) sequence: ``read`` entries become FILE_READ, ``edit`` entries
    become FILE_WRITE carrying ``write_effect`` (Table II's file actions have
    no raw-event mapping in this domain, so the signatures are rewritten in
    place -- same construction as test_tool_oscillation_synthetic)."""
    from atap.core.schema import (
        TASK_END,
        TASK_START,
        TOOL_CALL,
        Outcome,
        TraceEvent,
        Trajectory,
    )

    events = [TraceEvent(id="e000", ts=0, kind=TASK_START, agent="env", index=0)]
    idx = 1
    for act in seq:
        events.append(TraceEvent(
            id=f"e{idx:03d}", ts=float(idx), kind=TOOL_CALL, agent="coder",
            action=act, payload={"file": "f.py"}, index=idx))
        idx += 1
    events.append(TraceEvent(
        id=f"e{idx:03d}", ts=float(idx), kind=TASK_END, agent="env", index=idx))
    t = Trajectory(trace_id="syn2", task="t", events=events,
                   outcome=Outcome(success=False))
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    art = b.get("represent", "action_signature")
    for s in art["signatures"]:
        s["action_class"] = "FILE_WRITE" if s["target"] == "edit" else "FILE_READ"
        s["target"] = "f.py"
        s["signature"] = f"{s['action_class']}(f.py)"
        if s["action_class"] == "FILE_WRITE":
            s["effect"] = write_effect
    return b


def test_tool_oscillation_synthetic():
    """Two READ-WRITE-READ rounds (intervening write FAILED) -> tool_oscillation."""
    b = _osc_bundle(["read", "edit", "read", "edit", "read"], write_effect="FAILED")
    hits = _detected(b)
    osc = [d for d in hits if d["predicate"] == "tool_oscillation"]
    assert any(d["cycles"] >= 2 for d in osc)
    d = osc[0]
    # interval shape aligned with the other three predicates: end_index =
    # the closing read of the last counted cycle (events 1..5 are the five
    # file actions; TASK_START=0 / TASK_END=6)
    assert d["start_index"] == 1
    assert d["end_index"] == 5
    assert d["start_index"] <= d["end_index"]
    # all four predicates share the interval shape (start_index/end_index)
    for hit in hits:
        assert "start_index" in hit and "end_index" in hit


def test_tool_oscillation_negative_write_survived():
    """Negative: R-W-R-W-R with the middle writes SURVIVED (persisted, not
    FAILED/REVERTED) -- the cycles do not count, no hit (Table II requires
    the middle write to be deterministically labeled failed or reverted)."""
    b = _osc_bundle(["read", "edit", "read", "edit", "read"], write_effect="SURVIVED")
    hits = _detected(b)
    assert not any(d["predicate"] == "tool_oscillation" for d in hits)


def test_tool_oscillation_negative_single_cycle():
    """Negative: a single R-W-R (write FAILED) is 1 cycle < osc_cycles=2 --
    no hit (the paper's threshold is 2 cycles)."""
    b = _osc_bundle(["read", "edit", "read"], write_effect="FAILED")
    hits = _detected(b)
    assert not any(d["predicate"] == "tool_oscillation" for d in hits)


def test_search_loop_broken_by_write_or_validation_command():
    """Negative: Table II requires the consecutive run to contain "no
    FILE_WRITE and no validation COMMAND between them" -- either breaker
    splits the run, so 4+ read/search actions in total must not fire
    search_loop."""
    from atap.core.schema import (
        TASK_END,
        TASK_START,
        TOOL_CALL,
        VERIFIER,
        Outcome,
        TraceEvent,
        Trajectory,
    )

    # variant 1: a validation COMMAND (a VERIFIER event -> COMMAND(verify))
    # between the searches breaks the run: 2 + 2 searches, no run >= 3
    events = [TraceEvent(id="e000", ts=0, kind=TASK_START, agent="env", index=0)]
    idx = 1
    plan = ["search", "search", "verify", "search", "search"]
    for act in plan:
        if act == "verify":
            events.append(TraceEvent(
                id=f"e{idx:03d}", ts=float(idx), kind=VERIFIER, agent="verifier",
                payload={"content": "passed"}, index=idx))
        else:
            events.append(TraceEvent(
                id=f"e{idx:03d}", ts=float(idx), kind=TOOL_CALL, agent="searcher",
                action="search", payload={"query": "q"}, index=idx))
        idx += 1
    events.append(TraceEvent(
        id=f"e{idx:03d}", ts=float(idx), kind=TASK_END, agent="env", index=idx))
    t = Trajectory(trace_id="syn3", task="t", events=events,
                   outcome=Outcome(success=True))
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    hits = _detected(b, min_consecutive=3)
    assert not any(d["predicate"] == "search_loop" for d in hits)
    # control: the same 4 searches without the verifier form one run >= 3
    unbroken = Trajectory(
        trace_id="syn3b", task="t",
        events=[e for e in t.events if e.kind != VERIFIER],
        outcome=Outcome(success=True))
    b2 = TrajectoryBundle(unbroken)
    create("represent", "canonical_events").run_one(b2, RunContext())
    create("represent", "action_signature").run_one(b2, RunContext())
    hits2 = _detected(b2, min_consecutive=3)
    assert any(d["predicate"] == "search_loop" and d["length"] == 4 for d in hits2)

    # variant 2: a FILE_WRITE between the reads breaks the run (signatures
    # rewritten via the same _osc_bundle construction); control: the same 5
    # reads without writes fire
    b3 = _osc_bundle(["read", "edit", "read", "edit", "read"],
                     write_effect="RECORDED")
    hits3 = _detected(b3, min_consecutive=3)
    assert not any(d["predicate"] == "search_loop" for d in hits3)
    b4 = _osc_bundle(["read"] * 5, write_effect="RECORDED")
    hits4 = _detected(b4, min_consecutive=3)
    assert any(d["predicate"] == "search_loop" and d["length"] == 5 for d in hits4)


def test_missing_upstream_raises():
    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit"))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    with pytest.raises(ValueError, match="action_signature"):
        LoopDetectAnalyzer().run_one(b, ctx)
