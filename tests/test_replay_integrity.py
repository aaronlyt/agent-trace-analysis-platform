"""Replay-integrity regression tests: suffix alignment (no duplicated event
stream), semantic reference edges on replayed suffixes, no object/meta
aliasing between a rerun and its original trajectory, and extended-fault
(retrieval_detour / agent_deadlock) recovery via rerun_from/resolve."""

from __future__ import annotations

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox
from atap.sandbox.faults import EXTRA_FAULTS


def _prepared(task: str, fault: str | None):
    """Generate a trajectory and run the R0 flattening (replay needs the
    canonical event stream)."""
    sb = ToySandbox()
    t = sb.generate(task, fault)
    b = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(b, RunContext())
    return sb, t


def _count(events, kind: str) -> int:
    return sum(1 for e in events if e.kind == kind)


# ------------------------------------------------ suffix alignment (no spliced second rollout) --


def test_rerun_step_repetition_no_duplicate_rollout():
    """Regression: t* of step_repetition is the *repeated* search step, which
    exists only in the faulted script; after fault removal the exact-logical
    lookup misses and the old fallback-0 spliced a whole second rollout
    behind the prefix (18 events, double TASK_START). The aligned replay
    must keep a single rollout shape."""
    sb, t = _prepared("q-trajaudit", "step_repetition")
    step = t.meta["injected_fault"]["step"]
    assert step == 5 and t.events[step].action == "search"  # the search#1 call
    rr = sb.rerun_from(t, step, "Avoid step_repetition: do not repeat the same search call.")
    assert rr.meta["fault_removed"] is True
    assert rr.outcome.success
    assert rr.meta["suffix_alignment"] == "next_surviving"
    # no duplicated task envelope
    assert _count(rr.events, "TASK_START") == 1
    assert _count(rr.events, "TASK_END") == 1
    # indexes strictly monotonically increasing (contiguous 0..n-1)
    assert [e.index for e in rr.events] == list(range(len(rr.events)))
    # event count == step + clean suffix length: the merged stream has
    # exactly the clean rollout's shape (13 events; previously 18)
    _, t_ok = _prepared("q-trajaudit", None)
    assert len(rr.events) == len(t_ok.events)
    # the suffix starts where the faulted steps are gone: at search_reason
    assert rr.events[step].action is None
    assert "most relevant doc" in rr.events[step].payload["content"]


def test_align_suffix_start_ladder_unit():
    """Unit coverage of the alignment ladder's phase/empty rungs (not
    reachable through real rollouts, where verify/end always survive)."""
    orig = [
        {"logical": "start", "phase": None},
        {"logical": "search", "phase": "search"},
        {"logical": "search#1", "phase": "search"},  # fault-only logical step
        {"logical": "end", "phase": None},
    ]
    new = [
        {"logical": "start", "phase": None},
        {"logical": "search", "phase": "search"},
        {"logical": "end", "phase": None},
    ]
    assert ToySandbox._align_suffix_start(orig, 1, new) == (1, "exact")
    # search#1 vanished -> the next surviving original-suffix logical (end)
    assert ToySandbox._align_suffix_start(orig, 2, new) == (2, "next_surviving")
    # no original-suffix logical survives -> first same-phase step
    orig2 = [
        {"logical": "start", "phase": None},
        {"logical": "detour", "phase": "search"},
        {"logical": "end2", "phase": None},
    ]
    assert ToySandbox._align_suffix_start(orig2, 1, new) == (1, "phase")
    # nothing survives, no shared phase label (None never matches) -> empty
    orig3 = [{"logical": "start", "phase": None}, {"logical": "end", "phase": None}]
    new3 = [{"logical": "start", "phase": None}]
    assert ToySandbox._align_suffix_start(orig3, 1, new3) == (1, "empty")


def test_replay_intervene_alignment_on_vanished_step():
    """replay_intervene shares the alignment ladder: an agent_deadlock onset
    (clarify#1) vanishes after fault removal; the suffix must continue from
    the surviving search step, not splice a second rollout."""
    sb, t = _prepared("q-trajaudit", "agent_deadlock")
    step = t.meta["injected_fault"]["step"]
    assert t.events[step].kind == "HANDOFF"  # clarify#1
    runs = sb.replay_intervene(
        t, step, "Avoid agent_deadlock: stop the clarification loop and search."
    )
    assert runs[0].meta["fault_removed"] is True
    assert runs[0].outcome.success
    assert runs[0].meta["suffix_alignment"] == "next_surviving"
    assert _count(runs[0].events, "TASK_START") == 1
    assert [e.index for e in runs[0].events] == list(range(len(runs[0].events)))


# ------------------------------------------- extended faults removable by named feedback --


@pytest.mark.parametrize("kind", sorted(EXTRA_FAULTS))
def test_extended_faults_removable_by_named_feedback(kind):
    """retrieval_detour / agent_deadlock are registered in ALL_FAULTS: a
    feedback naming the fault removes it in rerun_from/resolve (capability
    parity with replay_intervene). Previously the lookup hit FAULTS only, so
    the fault read as None -> unexplained_failure -> outcome=FAILURE while
    the replayed event stream showed VERIFIER=passed (self-contradictory)."""
    sb, t = _prepared("q-who-when", kind)
    assert not t.outcome.success
    fb = f"Avoid {kind}: follow the planned retrieval course directly."
    rr = sb.rerun_from(t, t.meta["injected_fault"]["step"], fb)
    assert rr.meta["fault_removed"] is True
    assert rr.outcome.success is True
    # the streamed verifier note agrees with the outcome (no contradiction)
    verify = [e for e in rr.events if e.kind == "VERIFIER"][-1]
    assert "passed" in verify.payload["content"]
    # single clean rollout, contiguous indexes
    assert _count(rr.events, "TASK_START") == 1
    assert [e.index for e in rr.events] == list(range(len(rr.events)))

    rs = sb.resolve(t, fb)
    assert rs.meta["fault_removed"] is True
    assert rs.outcome.success is True


def test_extended_faults_kept_failing_on_vague_feedback():
    """The removal is feedback-gated, not unconditional: vague feedback keeps
    the extended fault present (recovery rate still measures feedback
    quality)."""
    sb, t = _prepared("q-who-when", "retrieval_detour")
    rr = sb.rerun_from(t, t.meta["injected_fault"]["step"],
                       "Please inspect the trajectory more carefully and retry.")
    assert rr.meta["fault_removed"] is False
    assert rr.outcome.success is False


# ------------------------------------------------------- no object / meta aliasing --


def test_rerun_prefix_and_meta_not_aliased():
    """The rerun's prefix events must be copies of the original trajectory's
    events (the closed-loop round normalizes rerun events in place, which
    must never rewrite the original), and meta.qrels must not be the same
    mutable dict."""
    sb, t = _prepared("q-trajaudit", "info_withholding")
    step = t.meta["injected_fault"]["step"]
    rr = sb.rerun_from(t, step, "Avoid info_withholding: report the documents.")
    assert 0 < step
    assert all(rr.events[i] is not t.events[i] for i in range(step))
    assert [e.id for e in rr.events[:step]] == [e.id for e in t.events[:step]]
    assert rr.meta["qrels"] is not t.meta["qrels"]
    assert rr.meta["qrels"] == t.meta["qrels"]
    rs = sb.resolve(t, "Avoid info_withholding: report the documents.")
    assert rs.meta["qrels"] is not t.meta["qrels"]


# ------------------------------------------------ replayed suffix semantic refs --


def test_rerun_suffix_keeps_semantic_refs():
    """The replayed suffix keeps the span-tree's semantic reference edges:
    in-suffix refs map onto the new event ids, refs into the retained prefix
    bridge onto the original prefix event ids (bridged through the logical
    step names); TOOL_RESULT refs point at their own TOOL_CALL."""
    sb, t = _prepared("q-trajaudit", "step_repetition")
    step = t.meta["injected_fault"]["step"]
    rr = sb.rerun_from(t, step, "Avoid step_repetition: use the existing result.")
    by_id = {e.id: e for e in rr.events}
    # every ref resolves to an event of the merged stream
    for e in rr.events:
        for r in e.refs:
            assert r in by_id, f"{e.id} refs unknown {r}"
    assert rr.meta["dropped_refs"] == 0
    # bridged prefix edge: search_reason (first suffix event) consumes the
    # retained prefix's first search_result
    assert rr.events[step].refs == [rr.events[step - 1].id]
    assert rr.events[step - 1].kind == "TOOL_RESULT"
    # in-suffix TOOL_RESULT -> its own TOOL_CALL
    read_res = [e for e in rr.events[step:]
                if e.kind == "TOOL_RESULT" and e.action == "read_doc"][-1]
    assert len(read_res.refs) == 1
    read_call = by_id[read_res.refs[0]]
    assert read_call.kind == "TOOL_CALL" and read_call.action == "read_doc"
    # compose/handoff semantic edges survive replay (previously only
    # TOOL_RESULT/VERIFIER got any refs at all)
    handoff = [e for e in rr.events[step:] if e.kind == "HANDOFF"][-1]
    assert handoff.refs == [read_res.id]
    compose = [e for e in rr.events[step:]
               if e.kind == "LLM_CALL" and e.agent == "reporter"][-1]
    assert compose.refs == [handoff.id]
    verify = [e for e in rr.events if e.kind == "VERIFIER"][-1]
    submit = [e for e in rr.events if e.kind == "TOOL_CALL" and e.action == "submit"][-1]
    assert verify.refs == [submit.id]


def test_rerun_meta_records_alignment_field():
    sb, t = _prepared("q-trajaudit", "info_withholding")
    rr = sb.rerun_from(t, t.meta["injected_fault"]["step"],
                       "Avoid info_withholding: report the documents.")
    assert rr.meta["suffix_alignment"] == "exact"
    assert rr.meta["dropped_refs"] == 0
    runs = sb.replay_intervene(t, t.meta["injected_fault"]["step"],
                               "Avoid info_withholding: report the documents.")
    assert runs[0].meta["suffix_alignment"] == "exact"
    assert runs[0].meta["dropped_refs"] == 0


# ------------------------------------------ verifier/outcome consistency & chain recovery --


@pytest.mark.parametrize("kind", ["step_repetition", "malformed_tool_call"])
def test_rerun_at_task_end_keeps_verifier_outcome_consistent(kind):
    """Regression: pointing t* at (or past) the original VERIFIER -- which
    out-of-range clamping also lands on -- made the rerun report
    success=True while the retained prefix still showed the faulted
    verifier line. The visible stream's last verifier verdict must win."""
    sb, t = _prepared("q-trajaudit", kind)
    rr = sb.rerun_from(t, len(t.events) - 1, f"Avoid {kind}: fix the failure.")
    assert rr.meta["verifier_conflict"] is True
    assert rr.outcome.success is False
    last_verifier = next(e for e in reversed(rr.events) if e.kind == "VERIFIER")
    assert not str(last_verifier.payload.get("content", "")).startswith("passed")

    # an out-of-range step clamps to the same last position
    rr2 = sb.rerun_from(t, 999, f"Avoid {kind}: fix the failure.")
    assert rr2.meta["step_clamped"] is True
    assert rr2.meta["verifier_conflict"] is True
    assert rr2.outcome.success is False


def test_failed_rerun_remains_recoverable_via_origin_fault():
    """Regression: reruns strip injected_fault, so re-replaying a *failed*
    rerun (the closed-loop second-round recovery shape) could never remove
    the fault -- the lookup found nothing and the unexplained-failure guard
    forced failure. The carried origin_fault restores chain recovery, and a
    genuinely unexplained failure records the guard in meta."""
    sb, t = _prepared("q-trajaudit", "malformed_tool_call")
    step = t.meta["injected_fault"]["step"]
    rr1 = sb.rerun_from(t, step, "vague feedback that names nothing")
    assert rr1.outcome.success is False
    assert "injected_fault" not in rr1.meta
    assert rr1.meta["origin_fault"]["kind"] == "malformed_tool_call"

    # chain: named feedback now recovers the failed rerun
    rr2 = sb.rerun_from(rr1, step, "Avoid malformed_tool_call: validate arguments.")
    assert rr2.outcome.success is True
    assert rr2.meta["fault_removed"] is True
    assert rr2.meta["unexplained_failure"] is False
    assert rr2.meta["origin_fault"]["kind"] == "malformed_tool_call"

    # genuinely unexplained (tampered meta, no origin): guard recorded
    tampered = type(t)(**{**t.__dict__})
    tampered.meta = {k: v for k, v in t.meta.items() if k != "injected_fault"}
    rr3 = sb.rerun_from(tampered, step, "Avoid malformed_tool_call: validate arguments.")
    assert rr3.outcome.success is False
    assert rr3.meta["unexplained_failure"] is True


def test_replay_intervene_rejects_degenerate_horizon_and_repeats():
    sb, t = _prepared("q-trajaudit", "malformed_tool_call")
    step = t.meta["injected_fault"]["step"]
    with pytest.raises(ValueError, match="horizon"):
        sb.replay_intervene(t, step, "edit", horizon=0)
    with pytest.raises(ValueError, match="horizon"):
        sb.replay_intervene(t, step, "edit", horizon=-1)
    with pytest.raises(ValueError, match="n_repeats"):
        sb.replay_intervene(t, step, "edit", n_repeats=0)


def test_rerun_payload_and_qrels_not_aliased():
    """One level deeper than object identity: mutating the rerun's payload
    or qrels lists must not leak back into the original trajectory."""
    sb, t = _prepared("q-trajaudit", "malformed_tool_call")
    step = t.meta["injected_fault"]["step"]
    rr = sb.rerun_from(t, step, "Avoid malformed_tool_call: validate arguments.")
    rr.events[0].payload["injected"] = True
    assert "injected" not in t.events[0].payload
    rr.meta["qrels"]["evidence"].append("dX")
    assert "dX" not in t.meta["qrels"]["evidence"]
