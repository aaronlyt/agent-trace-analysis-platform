"""L0 free rule pack (AgentDebugX 2607.18754) tests."""

from __future__ import annotations

from atap.classify.rule_pack import RulePackClassifier
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def _bundle(task="q-trajaudit", fault=None, with_loop=True):
    b = TrajectoryBundle(ToySandbox().generate(task, fault))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    if with_loop:
        create("analyze", "loop_detect", min_consecutive=3).run_one(b, ctx)
    return b, ctx


def _findings(b, **params):
    RulePackClassifier(**params).run_one(b, RunContext())
    return b.get("classify", "rule_pack")


def test_malformed_rule_hits_call_step():
    b, _ = _bundle("q-trajaudit", "malformed_tool_call")
    art = _findings(b)
    rules = {f["rule"] for f in art["findings"]}
    assert "malformed_tool_call" in rules
    f = next(f for f in art["findings"] if f["rule"] == "malformed_tool_call")
    assert f["step"] == 3 and f["agent"] == "searcher"
    assert art["cost"] == "free"


def test_no_progress_rule_consumes_loop_detect():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    art = _findings(b)
    f = next(f for f in art["findings"] if f["rule"] == "no_progress_loop")
    assert f["step"] == 5  # loop_detect's repetition_onset
    assert f["mast_code"] == "FM-1.3"


def test_no_progress_fallback_on_r5_only():
    b, _ = _bundle("q-trajaudit", "step_repetition", with_loop=False)
    art = _findings(b)
    assert any(f["rule"] == "no_progress_loop" for f in art["findings"])


def test_no_progress_reread_surface_not_dropped_when_artifact_present():
    """P2 regression (2026-08-27 re-review): the loop artifact exists but
    contains only a re_read_churn hit (a FILE_READ loop symptom; the paper's
    default min_consecutive=10 keeps search_loop silent on a 3-read run).
    The rule used to return empty -- no finding, no R5 fallback, no note --
    exactly disabling the fallback's FILE_READ coverage when the artifact
    existed. Now the re-read surface still goes through the R5 fallback and
    the consumption boundary is observable in notes."""
    from atap.core.schema import (
        TASK_END,
        TASK_START,
        TOOL_CALL,
        TOOL_RESULT,
        Outcome,
        TraceEvent,
        Trajectory,
    )

    events = [TraceEvent(id="e000", ts=0, kind=TASK_START, agent="env", index=0)]
    idx = 1
    for _ in range(3):
        events.append(TraceEvent(
            id=f"e{idx:03d}", ts=float(idx), kind=TOOL_CALL, agent="searcher",
            action="read_doc", payload={"doc_id": "d1"}, index=idx))
        idx += 1
        events.append(TraceEvent(
            id=f"e{idx:03d}", ts=float(idx), kind=TOOL_RESULT, agent="env",
            action="read_doc", refs=[events[-1].id], payload={"content": "doc"},
            index=idx))
        idx += 1
    events.append(TraceEvent(
        id=f"e{idx:03d}", ts=float(idx), kind=TASK_END, agent="env", index=idx))
    t = Trajectory(trace_id="syn-reread", task="t", events=events,
                   outcome=Outcome(success=False))
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    create("analyze", "loop_detect").run_one(b, ctx)   # paper-default thresholds
    # precondition: the artifact exists and holds only re_read_churn --
    # a consumed-predicate (search_loop/redundant_search) hit must NOT exist
    assert {d["predicate"] for d in b.get("analyze", "loop_detect")["detected"]} \
        == {"re_read_churn"}

    art = _findings(b)
    npl = [f for f in art["findings"] if f["rule"] == "no_progress_loop"]
    assert npl, "re-read loop silently dropped: artifact present, no finding"
    assert npl[0]["step"] == 1 and npl[0]["agent"] == "searcher"
    assert any("R5 signature self-check fallback" in e for e in npl[0]["evidence"])
    # observable consumption trace even when nothing is consumed
    assert art["notes"] and "re-read surface" in art["notes"][0]


def test_premature_success_rule_targets_decision_step():
    b, _ = _bundle("q-trajaudit", "premature_termination")
    art = _findings(b)
    f = next(f for f in art["findings"] if f["rule"] == "premature_success_claim")
    assert f["step"] == 1 and f["agent"] == "planner"  # Eq.5: the planning step, not the submit


def test_invalid_output_rule_on_verifier_rejection():
    b, _ = _bundle("q-trajaudit", "disobey_task_spec")
    art = _findings(b)
    f = next(f for f in art["findings"] if f["rule"] == "invalid_output")
    assert f["step"] == 9 and f["agent"] == "reporter"
    assert f["mast_code"] == "FM-1.1"


def test_success_trace_yields_no_findings():
    b, _ = _bundle("q-trajaudit")
    art = _findings(b)
    assert art["findings"] == []


def test_fusion_labels_filled():
    b, _ = _bundle("q-trajaudit", "malformed_tool_call")
    art = _findings(b)
    assert art["fusion"] and art["fusion"][0]["mast"]


def test_all_faults_get_at_least_one_rule_or_none():
    """Run all six faults: the targeted faults each get a hit; the rest
    (information withholding / ungrounded citation) may be empty -- the L0
    rule pack only covers mechanically verifiable failures, the rest goes
    to the L1 judge."""
    targets = {
        "malformed_tool_call": "malformed_tool_call",
        "step_repetition": "no_progress_loop",
        "premature_termination": "premature_success_claim",
        "disobey_task_spec": "invalid_output",
    }
    for kind, rule in targets.items():
        b, _ = _bundle("q-trajaudit", kind)
        art = _findings(b)
        assert any(f["rule"] == rule for f in art["findings"]), f"{kind} did not hit {rule}"
