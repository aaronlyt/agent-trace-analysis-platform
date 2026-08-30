"""Who&When adapter tests -- gold contract, split-aware agent identity, leak
freedom, and end-to-end scoring through compare.evaluate_against_gt."""

from __future__ import annotations

import json

import atap  # noqa: F401  registration bootstrap
from atap.compare import evaluate_against_gt
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import Hypothesis, Trajectory
from atap.io.whoswhen import (
    load_whoswhen,
    trajectory_from_record,
    write_jsonl,
)

# an Algorithm-Generated-shaped record: agent identity = "name",
# mistake_step is a 0-based index into history (entry index 2 == "Stats_Expert")
ALGO_RECORD = {
    "is_correct": False,
    "question": "What is the 2020 population?",
    "question_ID": "q-1",
    "level": "2",
    "ground_truth": "56000",                       # gold -- must NOT leak to the judge
    "history": [
        {"role": "user", "name": "user", "content": "solve it"},
        {"role": "assistant", "name": "Planner_Expert", "content": "let's search"},
        {"role": "assistant", "name": "Stats_Expert", "content": "the value is 56583"},
        {"role": "assistant", "name": "Verifier_Expert", "content": "looks fine"},
    ],
    "mistake_agent": "Stats_Expert",
    "mistake_step": "2",
    "mistake_reason": "used an unverified value",   # gold -- must NOT leak
}

HAND_RECORD = {
    "is_correct": False,
    "question": "browse and answer",
    "ground_truth": "42",
    "history": [
        {"role": "Orchestrator (thought)", "content": "plan"},
        {"role": "WebSurfer", "content": "opened the page"},
        {"role": "ComputerTerminal", "content": "ran code"},
    ],
    "mistake_agent": "WebSurfer",
    "mistake_step": "1",
    "mistake_reason": "misread the page",
}


def test_algo_record_maps_gold_and_agents():
    t = trajectory_from_record(ALGO_RECORD, "Algorithm-Generated", trace_id="ww-1")
    assert t.trace_id == "ww-1"
    assert t.task == "What is the 2020 population?"
    assert t.outcome.success is False
    # one event per history message, agent identity from "name"
    assert [e.agent for e in t.events] == ["user", "Planner_Expert", "Stats_Expert", "Verifier_Expert"]
    assert [e.index for e in t.events] == [0, 1, 2, 3]
    # gold: step is the 0-based history index; agent on the same vocabulary
    gt = t.meta["injected_fault"]
    assert gt["step"] == 2 and gt["agent"] == "Stats_Expert"
    assert gt["kind"] == "whoswhen:algo" and gt["mast_code"] is None
    # the blamed event's agent equals the gold agent -> agent-hit is well-defined
    assert t.events[gt["step"]].agent == gt["agent"]


def test_handcrafted_uses_role_and_normalizes_orchestrator():
    t = trajectory_from_record(HAND_RECORD, "Hand-Crafted", trace_id="ww-h")
    # Magentic-One routing annotations stripped so the identity matches the
    # bare-role gold ("Orchestrator (thought)" -> "Orchestrator")
    assert [e.agent for e in t.events] == ["Orchestrator", "WebSurfer", "ComputerTerminal"]
    assert t.meta["injected_fault"] == {
        "step": 1, "agent": "WebSurfer", "kind": "whoswhen:hand", "mast_code": None,
    }


def test_handcrafted_orchestrator_variants_all_normalize():
    rec = {
        "is_correct": False, "question": "q", "ground_truth": "x",
        "history": [
            {"role": "Orchestrator (-> WebSurfer)", "content": "route"},
            {"role": "Orchestrator (termination condition)", "content": "stop"},
            {"role": "WebSurfer", "content": "browse"},
        ],
        "mistake_agent": "Orchestrator", "mistake_step": "0",
    }
    t = trajectory_from_record(rec, "Hand-Crafted", trace_id="ww-o")
    assert [e.agent for e in t.events] == ["Orchestrator", "Orchestrator", "WebSurfer"]
    # the blamed step's agent now equals the bare-role gold -> a real hit
    gt = t.meta["injected_fault"]
    assert t.events[gt["step"]].agent == gt["agent"] == "Orchestrator"


def test_gold_never_leaks_into_the_judge_view():
    t = trajectory_from_record(ALGO_RECORD, "Algorithm-Generated", trace_id="ww-1")
    # everything the judge renders: task + event contents + outcome note
    visible = t.task + " " + t.outcome.note + " " + " ".join(
        str(e.payload.get("content", "")) for e in t.events
    )
    assert "56000" not in visible                 # ground_truth stays in meta
    assert "unverified value" not in visible      # mistake_reason stays in meta
    # ...and it IS retained in meta for reference
    assert t.meta["whoswhen"]["ground_truth"] == "56000"


def test_jsonl_roundtrip_preserves_gold(tmp_path):
    t = trajectory_from_record(ALGO_RECORD, "Algorithm-Generated", trace_id="ww-1")
    p = tmp_path / "w.jsonl"
    assert write_jsonl([t], p) == 1
    back = Trajectory.from_dict(json.loads(p.read_text().splitlines()[0]))
    assert back.meta["injected_fault"] == t.meta["injected_fault"]
    assert [e.agent for e in back.events] == [e.agent for e in t.events]


def test_scored_by_compare_gt_path():
    """A hypothesis matching gold must score a step-hit and an agent-hit through
    the same evaluate_against_gt path `atap compare` uses -- proves the adapter's
    gold shape is exactly what the scorer consumes."""
    t = trajectory_from_record(ALGO_RECORD, "Algorithm-Generated", trace_id="ww-1")
    b = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(b, RunContext())
    b.put("attribute", "all_at_once", {"hypotheses": [Hypothesis(
        agent="Stats_Expert", step=2, root_cause="unverified value",
        confidence=0.9, source="all_at_once",
    )]})
    ev = evaluate_against_gt([b])
    assert ev["n_failed"] == 1
    assert ev["step_hits"] == 1 and ev["agent_hits"] == 1
    # a wrong step would not score
    b2 = TrajectoryBundle(trajectory_from_record(ALGO_RECORD, "Algorithm-Generated", trace_id="ww-2"))
    create("represent", "canonical_events").run_one(b2, RunContext())
    b2.put("attribute", "all_at_once", {"hypotheses": [Hypothesis(
        agent="Planner_Expert", step=1, root_cause="x", confidence=0.9, source="all_at_once")]})
    ev2 = evaluate_against_gt([b2])
    assert ev2["step_hits"] == 0 and ev2["agent_hits"] == 0


def test_load_whoswhen_missing_root():
    try:
        load_whoswhen("/nonexistent/whoswhen/root")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing dataset root")
