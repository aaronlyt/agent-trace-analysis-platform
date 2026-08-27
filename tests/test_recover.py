"""Recovery algorithm tests: the three states of targeted rerun (successful
recovery / skipped without attribution / no environment)."""

from __future__ import annotations

import pytest

from atap.attribute.all_at_once import AllAtOnceAttributor
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import Hypothesis
from atap.llm.fake_client import FakeLLMClient
from atap.recover.targeted_rerun import TargetedRerunRecoverer
from atap.sandbox import ToySandbox


def _failed_bundle(task="q-trajaudit", fault="info_withholding", with_attribution=True):
    t = ToySandbox().generate(task, fault)
    b = TrajectoryBundle(t)
    ctx = RunContext(llm=FakeLLMClient(), env=ToySandbox())
    create("represent", "canonical_events").run_one(b, ctx)
    if with_attribution:
        AllAtOnceAttributor().run_one(b, ctx)  # the pseudo-judge yields a fix that names the fault
    return b, ctx


def test_targeted_rerun_recovers_in_one_round():
    b, ctx = _failed_bundle()
    TargetedRerunRecoverer().run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["recovered"] is True
    assert art["rounds"] == 1
    assert b.reruns and b.reruns[-1].outcome.success
    assert b.reruns[-1].meta["rerun_of"] == b.trace_id
    assert art["t_star"] == b.trajectory.meta["injected_fault"]["step"]


def test_targeted_rerun_exhausts_rounds_on_vague_feedback():
    b, ctx = _failed_bundle()
    # overwrite the attribution: the feedback contains no fault-type word ->
    # the replay does not repair -> max_rounds is exhausted
    b.put(
        "attribute", "all_at_once",
        {"hypotheses": [Hypothesis(
            agent="searcher", step=8, root_cause="r",
            fix_suggestion="Please inspect the trajectory more carefully.", confidence=0.9,
        ).to_dict()]},
    )
    TargetedRerunRecoverer(max_rounds=3).run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["recovered"] is False
    assert art["rounds"] == 3 and len(art["attempts"]) == 3
    assert all(not t.outcome.success for t in b.reruns)
    # UpdateFeedback (declared weakening) pinned on its observable contract:
    # pure concatenation that consumes only the failed round's sandbox
    # outcome note, and the feedback only ever grows (never replaced). The
    # per-round feedback each rerun consumed is recorded in its meta
    # (feedback_snippet, first 200 chars).
    fb = [t.meta["feedback_snippet"] for t in b.reruns]
    assert fb[0] == art["feedback_seed"]            # round 1 consumes the seed feedback
    assert fb[1].startswith(fb[0])                  # round 2 keeps round 1's text verbatim (append-only)
    assert fb[2].startswith(fb[1])                  # and so does round 3
    assert f"(attempt 1 failed: {b.reruns[0].outcome.note}" in fb[1]  # consumes the round-1 outcome note
    # attempts[].note is the sandbox outcome note of each rerun (not a
    # feedback echo -- the "(attempt k failed" phrasing only ever appears in
    # the *next* round's feedback)
    for attempt, rerun in zip(art["attempts"], b.reruns):
        assert attempt["note"] == rerun.outcome.note[:120]
        assert "(attempt" not in attempt["note"]


def test_targeted_rerun_skips_without_hypothesis():
    b, ctx = _failed_bundle(with_attribution=False)
    TargetedRerunRecoverer().run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["status"] == "skipped_no_hypothesis"
    assert b.reruns == []


def test_targeted_rerun_needs_env():
    b, ctx = _failed_bundle()
    ctx.env = None
    TargetedRerunRecoverer().run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["status"] == "no_replay_environment"
    # the note is real prose, not an unformatted "(sandbox: {type: toy})"
    # literal-braces leftover
    assert "sandbox.type=toy" in art["note"]
    assert "{type" not in art["note"]


def test_targeted_rerun_skips_success_trace():
    t = ToySandbox().generate("q-trajaudit")
    b = TrajectoryBundle(t)
    TargetedRerunRecoverer().run_one(b, RunContext(env=ToySandbox()))
    assert not b.has("recover", "targeted_rerun")


def _seed_hypotheses(b, hyps):
    """Attach raw Hypothesis objects as an attribution artifact (bypasses
    bundle.put's to_dict expansion so source attributes survive
    bundle.hypotheses()'s pass-through for non-dict entries)."""
    b.artifacts.setdefault("attribute", {})["seed"] = {"hypotheses": hyps}


def _with_source(h, source):
    """Set Hypothesis.source defensively: the field is owned by the
    attribution layer; until it lands, dynamic assignment on the plain
    dataclass stands in for it (skip if it ever becomes slotted)."""
    try:
        h.source = source
        return h
    except AttributeError:  # pragma: no cover -- future slotted Hypothesis
        pytest.skip("Hypothesis does not accept a source attribute yet")


def test_targeted_rerun_filters_hypotheses_by_attribution_source():
    """Param attribution= declares which attribution algorithm's Hypotheses
    the recoverer consumes (confidence has no global scale when several
    attribution algorithms are configured together)."""
    b, ctx = _failed_bundle()
    gt = b.trajectory.meta["injected_fault"]
    keep = _with_source(
        Hypothesis(agent="searcher", step=gt["step"], root_cause="r1",
                   fix_suggestion="Avoid info_withholding: report the "
                                  "retrieved documents faithfully.",
                   confidence=0.6),
        "algo_keep",
    )
    drop = _with_source(
        Hypothesis(agent="searcher", step=1, root_cause="r2",
                   fix_suggestion="vague", confidence=0.99),
        "algo_drop",
    )
    _seed_hypotheses(b, [keep, drop])

    TargetedRerunRecoverer(attribution="algo_keep").run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["recovered"] is True
    assert art["t_star"] == gt["step"]  # the 0.99-confidence step from the
    assert art["attribution"] == "algo_keep"  # other algorithm is not consumed

    # a filter that matches nothing degrades explicitly (no silent fallthrough
    # to the unfiltered maximum)
    b2, ctx2 = _failed_bundle()
    _seed_hypotheses(b2, [keep, drop])
    TargetedRerunRecoverer(attribution="algo_missing").run_one(b2, ctx2)
    art2 = b2.get("recover", "targeted_rerun")
    assert art2["status"] == "skipped_no_hypothesis"
    assert art2["recovered"] is False
    assert "algo_missing" in art2["note"]
    assert b2.reruns == []

    # unset (default) keeps the consume-all behavior
    b3, ctx3 = _failed_bundle()
    _seed_hypotheses(b3, [keep, drop])
    TargetedRerunRecoverer().run_one(b3, ctx3)
    assert b3.get("recover", "targeted_rerun")["t_star"] == 1  # unfiltered max confidence


def test_recovery_environment_protocol_documents_three_sides():
    """recover/base.py's RecoveryEnvironment protocol pins the three replay
    execution sides (rerun_from / resolve / replay_intervene) consumed by
    the recover algorithms; the sandbox implements all three."""
    from atap.recover.base import RecoveryEnvironment

    for meth in ("rerun_from", "resolve", "replay_intervene"):
        assert hasattr(RecoveryEnvironment, meth)
    assert isinstance(ToySandbox(), RecoveryEnvironment)
