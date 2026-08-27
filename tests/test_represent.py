"""Representation-layer tests: R0 flattening + R1 SSF folding."""

from __future__ import annotations

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.render import render_trace
from atap.core.schema import EVENT_KINDS, Outcome, Trajectory
from atap.represent.ssf import unfold
from atap.sandbox import ToySandbox

from helpers import success_trace


def _bundle(task="q-trajaudit", fault=None):
    t = ToySandbox().generate(task, fault)
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    return b, ctx


def test_canonical_flatten_from_spans():
    b, _ = _bundle()
    t = b.trajectory
    art = b.get("represent", "canonical_events")
    assert art["n_events"] == len(t.events) > 10
    assert [e.index for e in t.events] == list(range(len(t.events)))
    assert [e.id for e in t.events] == [f"e{i:03d}" for i in range(len(t.events))]
    # reference edges: TOOL_RESULT -> its TOOL_CALL; HANDOFF -> plan
    res = next(e for e in t.events if e.kind == "TOOL_RESULT" and e.action == "search")
    call = next(e for e in t.events if e.kind == "TOOL_CALL" and e.action == "search")
    assert res.refs == [call.id]
    assert art["n_refs"] >= 4 and art["dropped_refs"] == 0
    assert set(art["agents"]) >= {"planner", "searcher", "reporter", "env", "verifier"}


def test_canonical_normalize_flat_trajectory():
    t = success_trace("flat-1")
    b = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(b, RunContext())
    assert [e.index for e in t.events] == list(range(len(t.events)))


def _nested_spans() -> list[dict]:
    """3-level nested span tree (sandbox never nests; collection adapters do):
    root(d0) -> plan(d1) -> search(d2) -> res(d3); msg(d1) references res
    across levels and one dangling span id absent from the tree."""
    return [
        {
            "id": "s-root", "kind": "TASK_START", "agent": "user", "ts": 100.0,
            "children": [
                {
                    "id": "s-plan", "kind": "LLM_CALL", "agent": "planner",
                    "refs": ["s-root"],
                    "children": [
                        {
                            "id": "s-search", "kind": "TOOL_CALL", "agent": "searcher",
                            "children": [
                                {
                                    "id": "s-res", "kind": "TOOL_RESULT", "agent": "env",
                                    "refs": ["s-search"], "ts": 103.5,
                                },
                            ],
                        },
                    ],
                },
                {
                    "id": "s-msg", "kind": "AGENT_MESSAGE", "agent": "reporter",
                    "refs": ["s-res", "s-ghost"],   # cross-level + dangling
                },
            ],
        },
    ]


def _flatten_spans(spans):
    t = Trajectory(
        trace_id="nested-1", task="t", events=[],
        outcome=Outcome(success=False), raw={"spans": spans},
    )
    b = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(b, RunContext())
    return b, t


def test_canonical_flatten_nested_tree():
    """DFS preorder flattening of a nested span tree: order, parent mapping,
    continuous indices, cross-level refs, dangling-ref dropping, node-ts
    preference."""
    b, t = _flatten_spans(_nested_spans())
    evs = t.events
    # DFS preorder: root -> plan -> search -> res -> msg (children before siblings)
    assert [e.id for e in evs] == ["e000", "e001", "e002", "e003", "e004"]
    assert [e.index for e in evs] == list(range(5))          # indices continuous
    by_id = {e.id: e for e in evs}
    # parent mapping follows the span tree depth
    assert by_id["e000"].parent is None
    assert by_id["e001"].parent == "e000"                    # depth 1 under root
    assert by_id["e002"].parent == "e001"                    # depth 2
    assert by_id["e003"].parent == "e002"                    # depth 3
    assert by_id["e004"].parent == "e000"                    # depth-1 sibling
    # refs remapped span id -> event id; the cross-level ref survives,
    # the dangling s-ghost is dropped and counted
    assert by_id["e001"].refs == ["e000"]
    assert by_id["e003"].refs == ["e002"]
    assert by_id["e004"].refs == ["e003"]
    art = b.get("represent", "canonical_events")
    assert art["dropped_refs"] == 1
    assert art["n_refs"] == 3
    assert art["duplicate_span_ids"] == 0 and art["remapped_kinds"] == 0
    # ts: the node's own numeric ts wins; otherwise float(idx)
    assert by_id["e000"].ts == pytest.approx(100.0)
    assert by_id["e001"].ts == pytest.approx(1.0)
    assert by_id["e002"].ts == pytest.approx(2.0)
    assert by_id["e003"].ts == pytest.approx(103.5)
    assert by_id["e004"].ts == pytest.approx(4.0)


def test_canonical_kind_admission_out_of_vocab():
    """Collection adapters may emit kinds outside EVENT_KINDS (otel/langfuse
    fall back to "SPAN"); flatten output kinds are always in-vocab and the
    remaps are counted."""
    spans = [
        {"id": "a", "kind": "SPAN", "agent": "x"},          # unknown-op fallback
        {"id": "b", "kind": "GENERATION", "agent": "y"},    # langfuse model-call type
        {"id": "c", "kind": "MEMORY_WRITE", "agent": "z"},  # arbitrary external op
        {"id": "d", "kind": "TOOL_CALL", "agent": "w"},     # in-vocab passthrough
        {"id": "e", "kind": None, "agent": "v"},            # missing kind
    ]
    b, t = _flatten_spans(spans)
    assert [e.kind for e in t.events] == [
        "AGENT_MESSAGE", "LLM_CALL", "AGENT_MESSAGE", "TOOL_CALL", "AGENT_MESSAGE",
    ]
    assert all(e.kind in EVENT_KINDS for e in t.events)
    assert b.get("represent", "canonical_events")["remapped_kinds"] == 4


def test_canonical_duplicate_span_id_keeps_first():
    """A duplicate span id must not steal refs from the first occurrence;
    duplicates are counted instead of silently overwriting."""
    spans = [
        {"id": "s1", "kind": "TOOL_CALL", "agent": "searcher"},
        {"id": "s2", "kind": "LLM_CALL", "agent": "planner"},
        {"id": "s1", "kind": "LLM_CALL", "agent": "ghost"},   # duplicate id, later
        {"id": "s3", "kind": "TOOL_RESULT", "agent": "env", "refs": ["s1"]},
    ]
    b, t = _flatten_spans(spans)
    by_id = {e.id: e for e in t.events}
    # the ref to the duplicated span id anchors to the FIRST occurrence
    assert by_id["e003"].refs == ["e000"]
    assert by_id["e002"].kind == "LLM_CALL"                  # the duplicate still flattens
    art = b.get("represent", "canonical_events")
    assert art["duplicate_span_ids"] == 1
    assert art["dropped_refs"] == 0


def test_ssf_folds_long_prose_keeps_errors():
    # malformed path: the only observation is an error message -> kept, nothing to fold
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    create("represent", "ssf").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_tool_results"] == 1 and stats["n_kept_error"] == 1

    # normal read path: read_doc long prose folded, short search results
    # kept, no error observations
    b2, ctx2 = _bundle("q-trajaudit", "ungrounded_citation")
    create("represent", "ssf").run_one(b2, ctx2)
    art = b2.get("represent", "ssf")
    stats = art["stats"]
    assert stats["n_kept_error"] == 0
    assert stats["n_folded"] == 1  # read_doc long text
    assert stats["n_kept_short"] == 1  # the search result is short, kept
    assert stats["fold_ratio"] == 0.5
    # folding is reversible: the original text can be retrieved from the table
    fid = next(iter(art["table"]))
    assert "TrajAudit" in unfold(art, fid)
    # the placeholder carries a digest; the judge gets minimal evidence from
    # the folded view
    ph = next(iter(art["fold"].values()))
    assert ph.startswith("⟦folded:") and "TrajAudit" in ph


def test_ssf_does_not_treat_domain_prose_as_error():
    """The corpus prose contains dictionary words (error step / missing /
    failed) but is not an error observation."""
    b, ctx = _bundle("q-drift", "ungrounded_citation")  # d5's body contains "missing"
    create("represent", "ssf").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_kept_error"] == 0
    assert stats["n_folded"] >= 1  # read_doc long text gets folded


def test_ssf_loose_mode_keeps_keyword_prose():
    b, ctx = _bundle("q-drift", "ungrounded_citation")
    create("represent", "ssf", keyword_mode="loose").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_kept_error"] >= 1  # loose: lexical dictionary hits prose


def test_ssf_patch_kept():
    b, ctx = _bundle()
    t = b.trajectory
    # inject an observation with a diff header, verifying patch retention
    for e in t.events:
        if e.kind == "TOOL_RESULT" and e.action == "read_doc":
            e.payload["content"] = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,3 @@\n+fix\n" * 5
    create("represent", "ssf").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_kept_patch"] == 1 and stats["n_folded"] == 0


def test_judge_view_uses_fold():
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    create("represent", "ssf").run_one(b, ctx)
    from atap.core.render import judge_view

    view = judge_view(b)
    # the only TOOL_RESULT here is the error observation: nothing folds, no placeholder
    assert b.get("represent", "ssf")["stats"]["n_folded"] == 0
    assert "⟦folded:" not in view
    # the key evidence (error observation) is kept in the folded view
    assert "error: invalid arguments" in view


def test_canonical_missing_span_id_raises_value_error():
    """A span node without the 'id' key used to die with a bare KeyError
    deep inside the walk; it now raises an explicit ValueError naming the
    span kind and flattened position (refs are span-id anchored)."""
    spans = [
        {"id": "s1", "kind": "TOOL_CALL", "agent": "searcher"},
        {"kind": "LLM_CALL", "agent": "planner"},   # missing id
    ]
    with pytest.raises(ValueError, match=r"kind=LLM_CALL.*missing the required 'id' key"):
        _flatten_spans(spans)


def test_canonical_case_normalization_not_counted_as_remapped():
    """Pure case/whitespace normalization ("tool_call" -> TOOL_CALL) is
    in-vocabulary and must not inflate remapped_kinds; only alias/fallback
    admissions (SPAN alias, unknown kind, missing kind) count."""
    spans = [
        {"id": "a", "kind": "tool_call", "agent": "w"},     # case-only
        {"id": "b", "kind": " tool_result ", "agent": "x"},  # case + outer whitespace
        {"id": "c", "kind": "SPAN", "agent": "y"},          # alias admission
        {"id": "d", "kind": None, "agent": "z"},            # fallback admission
    ]
    b, t = _flatten_spans(spans)
    assert [e.kind for e in t.events] == [
        "TOOL_CALL", "TOOL_RESULT", "AGENT_MESSAGE", "AGENT_MESSAGE",
    ]
    assert b.get("represent", "canonical_events")["remapped_kinds"] == 2
