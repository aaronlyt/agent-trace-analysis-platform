"""Tests for RG/UG deterministic attribution (2608.01913)."""

from __future__ import annotations

import pytest

from atap.attribute.rg_ug import RGUGAttributor
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox
from tests.helpers import _ev
from atap.core.schema import Outcome, Trajectory


def _bundle(trace):
    b = TrajectoryBundle(trace)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    return b, ctx


#: construction-expected labels of the six faults + detour under RG/UG judgement
EXPECTED_LABELS = {
    "premature_termination": "RG_directional",   # no retrieval at all
    "malformed_tool_call": "RG_directional",      # the retrieval call fails, C_M=empty set
    "retrieval_detour": "RG_last_hop",            # hits evidence, misses gold
    "info_withholding": "UG_true_extraction",     # gold was read yet falsely reported
    "ungrounded_citation": "UG_true_extraction",
    "disobey_task_spec": "UG_true_extraction",
    "step_repetition": "UG_true_extraction",      # gold is in the results, budget exhausted
}


def test_labels_match_construction_on_all_faults():
    ctx = RunContext()
    for kind, expected in EXPECTED_LABELS.items():
        b, _ = _bundle(ToySandbox().generate("q-trajaudit", kind))
        RGUGAttributor().run_one(b, ctx)
        art = b.get("attribute", "rg_ug")
        assert art["label"] == expected, f"{kind}: {art['label']} != {expected}"
        assert art["cost"] == "free" and art["role"] == "L0_deterministic"
        hyp = b.hypotheses()[0]
        assert hyp.confidence == 1.0
        assert hyp.root_cause_code == (
            "retrieval_gap" if expected.startswith("RG") else "utilization_gap"
        )


def test_correct_trace_has_no_attribution():
    b, ctx = _bundle(ToySandbox().generate("q-who-when", None))
    RGUGAttributor().run_one(b, ctx)
    art = b.get("attribute", "rg_ug")
    assert art["label"] == "correct"
    assert art["status"] == "success_no_attribution"
    assert b.hypotheses() == []
    assert art["first_gold_hit"] is not None
    assert art["episodes"][0]["utility"] == "productive"
    assert art["visit_precision"] == 1.0


def test_step_mapping_hits_known_ug_and_rg_points():
    """Step mapping: withholding -> the false-report handoff;
    ungrounded/disobey -> compose; premature -> the plan decision step;
    malformed/detour -> the first search call. step_repetition lands on
    compose (a mapping boundary, not the first repeated step)."""
    ctx = RunContext()
    cases = {
        "info_withholding": ("HANDOFF", "searcher"),
        "ungrounded_citation": ("LLM_CALL", "reporter"),
        "disobey_task_spec": ("LLM_CALL", "reporter"),
        "premature_termination": ("LLM_CALL", "planner"),
        "malformed_tool_call": ("TOOL_CALL", "searcher"),
        "retrieval_detour": ("TOOL_CALL", "searcher"),
    }
    for kind, (want_kind, want_agent) in cases.items():
        b, _ = _bundle(ToySandbox().generate("q-who-when", kind))
        RGUGAttributor().run_one(b, ctx)
        hyp = b.hypotheses()[0]
        ev = b.trajectory.events[hyp.step]
        assert (ev.kind, ev.agent) == (want_kind, want_agent), (
            f"{kind}: attributed to {ev.kind}/{ev.agent}, expected {want_kind}/{want_agent}"
        )


def test_episode_utilities_and_wasted_tail():
    b, ctx = _bundle(ToySandbox().generate("q-trajaudit", "step_repetition"))
    RGUGAttributor().run_one(b, ctx)
    art = b.get("attribute", "rg_ug")
    utils = [ep["utility"] for ep in art["episodes"]]
    assert utils == ["productive", "redundant", "redundant"]
    assert art["k_star"] == 0 and art["wasted_tail"] == 2
    assert art["M"] == 3


def test_ug_boundary_synthetic_fixture():
    """G has multiple documents, partially retrieved -> UG_boundary (the
    existing tasks' gold sets are all single-document, so build a fixture)."""
    events = [
        _ev(0, "TASK_START", "env", payload={"task": "compare two methods"}),
        _ev(1, "TOOL_CALL", "searcher", action="search", phase="search",
            payload={"query": "methods"}),
        _ev(2, "TOOL_RESULT", "env", action="search", refs=["e001"], phase="search",
            payload={"content": "search results for 'methods': 2 docs [d1, d2]"}),
        _ev(3, "TOOL_CALL", "searcher", action="read_doc", refs=["e002"], phase="search",
            payload={"doc_id": "d1"}),
        _ev(4, "TOOL_RESULT", "env", action="read_doc", refs=["e003"], phase="search",
            payload={"content": "doc d1 body"}),
        _ev(5, "LLM_CALL", "reporter", refs=["e004"], phase="report",
            payload={"content": "based on d1 only, answer partial (cited: d1)"}),
        _ev(6, "TOOL_CALL", "reporter", action="submit", refs=["e005"], phase="report",
            payload={"answer": "partial (d1)"}),
        _ev(7, "VERIFIER", "verifier", refs=["e006"],
            payload={"content": "failed: answer incomplete"}),
        _ev(8, "TASK_END", "env"),
    ]
    t = Trajectory(
        trace_id="syn-boundary",
        task="compare two methods",
        events=events,
        outcome=Outcome(success=False, note="incomplete"),
        meta={"task_id": "q-x", "qrels": {
            "evidence": ["d1", "d2", "d4"], "gold": ["d1", "d4"],
        }},
    )
    b, ctx = _bundle(t)
    RGUGAttributor().run_one(b, ctx)
    art = b.get("attribute", "rg_ug")
    assert art["label"] == "UG_boundary"
    assert art["G_star"] == ["d1"]           # proper subset of gold={d1,d4}
    hyp = b.hypotheses()[0]
    assert hyp.step == 5                     # first decision step (fallback with no contradicting symptom)


def test_missing_qrels_raises_explicitly():
    from tests.helpers import failure_trace_ungrounded

    trace = failure_trace_ungrounded()       # no meta["qrels"]
    b, ctx = _bundle(trace)
    with pytest.raises(ValueError, match="qrels"):
        RGUGAttributor().run_one(b, ctx)


def test_pre_search_read_enters_c_m_via_implicit_episode():
    """A read issued before the first search used to belong to no episode
    (excluded from every R_k, hence from C_M) while still counting toward
    visit_precision's denominator. Gold read pre-search + wrong answer
    submitted must therefore be judged UG (C_M contains gold via the implicit
    leading episode), not RG."""
    events = [
        _ev(0, "TASK_START", "env", payload={"task": "which tool does X propose?"}),
        _ev(1, "TOOL_CALL", "searcher", action="read_doc", phase="search",
            payload={"doc_id": "d1"}),
        _ev(2, "TOOL_RESULT", "env", action="read_doc", refs=["e001"], phase="search",
            payload={"content": "doc d1 body"}),
        _ev(3, "LLM_CALL", "reporter", refs=["e002"], phase="report",
            payload={"content": "the answer is a magic fixer"}),
        _ev(4, "TOOL_CALL", "reporter", action="submit", refs=["e003"], phase="report",
            payload={"answer": "a magic fixer"}),
        _ev(5, "VERIFIER", "verifier", refs=["e004"],
            payload={"content": "failed: answer wrong"}),
        _ev(6, "TASK_END", "env"),
    ]
    t = Trajectory(
        trace_id="pre-search-read",
        task="which tool does X propose?",
        events=events,
        outcome=Outcome(success=False, note="wrong answer"),
        meta={"task_id": "q-pre", "qrels": {
            "evidence": ["d1", "d2"], "gold": ["d1"],
        }},
    )
    b, ctx = _bundle(t)
    RGUGAttributor().run_one(b, ctx)
    art = b.get("attribute", "rg_ug")
    assert art["label"] == "UG_true_extraction"   # not RG: gold reached C_M
    assert art["C_M"] == ["d1"]
    assert art["episodes"][0]["implicit"] is True
    assert art["episodes"][0]["start_index"] == 1  # the opening read call
    assert art["M"] == 1
    assert art["visit_precision"] == 1.0           # numerator/denominator now agree
    assert art["first_gold_hit"] == 1
    hyp = b.hypotheses()[0]
    assert hyp.root_cause_code == "utilization_gap"
    assert (hyp.agent, hyp.step) == ("reporter", 3)


def test_ug_fallback_prefers_last_non_env_event():
    """Gold surfaced yet no LLM_CALL follows it (and no contradiction fired):
    the fallback target is the last non-env event (the submit call), not
    events[0] (env TASK_START)."""
    events = [
        _ev(0, "TASK_START", "env", payload={"task": "which tool does X propose?"}),
        _ev(1, "TOOL_CALL", "searcher", action="search", phase="search",
            payload={"query": "x"}),
        _ev(2, "TOOL_RESULT", "env", action="search", refs=["e001"], phase="search",
            payload={"content": "search results for 'x': 2 docs [d1, d3]"}),
        _ev(3, "TOOL_CALL", "searcher", action="read_doc", refs=["e002"], phase="search",
            payload={"doc_id": "d3"}),
        _ev(4, "TOOL_RESULT", "env", action="read_doc", refs=["e003"], phase="search",
            payload={"content": "doc d3 body"}),
        _ev(5, "TOOL_CALL", "reporter", action="submit", refs=["e004"], phase="report",
            payload={"answer": "a wrong answer"}),
        _ev(6, "TASK_END", "env"),
    ]
    t = Trajectory(
        trace_id="ug-fallback",
        task="which tool does X propose?",
        events=events,
        outcome=Outcome(success=False, note="wrong answer"),
        meta={"task_id": "q-fb", "qrels": {
            "evidence": ["d1", "d3"], "gold": ["d3"],
        }},
    )
    b, ctx = _bundle(t)
    RGUGAttributor().run_one(b, ctx)
    art = b.get("attribute", "rg_ug")
    assert art["label"] == "UG_true_extraction"
    hyp = b.hypotheses()[0]
    assert (hyp.agent, hyp.step) == ("reporter", 5)   # last non-env event


def test_ug_fallback_all_env_trace_uses_first_event():
    """Only when every event is env-owned does the fallback fall back to
    events[0] (declared boundary, mirroring binary_search's s*=0)."""
    events = [
        _ev(0, "TASK_START", "env", payload={"task": "t"}),
        _ev(1, "TOOL_CALL", "env", action="read_doc", payload={"doc_id": "d1"}),
        _ev(2, "TOOL_RESULT", "env", action="read_doc", refs=["e001"],
            payload={"content": "doc d1 body"}),
        _ev(3, "TASK_END", "env"),
    ]
    t = Trajectory(
        trace_id="all-env",
        task="t",
        events=events,
        outcome=Outcome(success=False),
        meta={"qrels": {"evidence": ["d1"], "gold": ["d1"]}},
    )
    b, ctx = _bundle(t)
    RGUGAttributor().run_one(b, ctx)
    hyp = b.hypotheses()[0]
    assert (hyp.agent, hyp.step) == ("env", 0)


def test_rg_no_search_evidence_renders_actual_kind():
    """The RG no-search fallback targets the first LLM_CALL decision step —
    the evidence line must render the event's actual kind, not a hardcoded
    'search' label."""
    b, ctx = _bundle(ToySandbox().generate("q-trajaudit", "premature_termination"))
    RGUGAttributor().run_one(b, ctx)
    hyp = b.hypotheses()[0]
    assert hyp.root_cause_code == "retrieval_gap"
    assert hyp.evidence[0].startswith("[1] planner LLM_CALL ")
    assert "planner search " not in hyp.evidence[0]
