"""Tests for the R2 information dependency graph (idg) and the R4 hierarchy
tree (hierarchy_tree)."""

from __future__ import annotations

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.represent.hierarchy_tree import HierarchyTreeRepresenter
from atap.represent.idg import IDGRepresenter
from atap.sandbox import ToySandbox
from tests.helpers import failure_trace_ungrounded, success_trace


def _bundle(trace):
    b = TrajectoryBundle(trace)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    return b, ctx


def _ancestors(art: dict, start: str) -> set[str]:
    """Reverse closure along usage edges (a reference implementation of the
    consumer-side convention)."""
    incoming: dict[str, list[str]] = {}
    for e in art["edges"]:
        incoming.setdefault(e["dst"], []).append(e["src"])
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        v = frontier.pop()
        for u in incoming.get(v, []):
            if u not in seen:
                seen.add(u)
                frontier.append(u)
    return seen


# --------------------------------------------------------------------- idg --


def test_idg_builds_usage_edges_in_temporal_order():
    b, ctx = _bundle(ToySandbox().generate("q-trajaudit", None))
    IDGRepresenter().run_one(b, ctx)
    art = b.get("represent", "idg")
    index_by_id = {ev.id: ev.index for ev in b.trajectory.events}
    # TASK_END carries no payload and stays out of the graph: nodes =
    # information artifacts that carry a payload
    assert art["stats"]["n_nodes"] == art["stats"]["n_events"] - 1
    assert art["edges"]
    for e in art["edges"]:
        assert index_by_id[e["src"]] < index_by_id[e["dst"]]   # t_src < t_dst
    # a TOOL_RESULT references its TOOL_CALL: the usage edge exists
    assert any(e["src"].endswith("003") and e["dst"].endswith("004") for e in art["edges"])
    assert art["conflicts"] == []
    assert art["cost"] == "free"


def test_idg_gt_root_cause_in_ancestor_closure():
    """Four of the six faults: GT root cause is in the ancestor closure of
    the terminal information artifact (reachable by backtracking along
    dependency edges).

    Two honest misses (method boundary, recorded honestly rather than
    doctoring the graph to please acceptance):
    * step_repetition -- the repeated searches' results were never used
      downstream; IDG semantics is "information that was used", blind to
      unused waste (R5/loop_detect fills the gap);
    * premature_termination -- the ungrounded submit has empty refs, and the
      root-cause plan step is not on the dependency chain at all (an
      "information missing" fault; rule_pack/pseudo-judge rule 4 fills the
      gap).
    """
    from atap.sandbox.faults import FAULTS

    hits, misses = [], []
    ctx = RunContext()
    for kind in FAULTS:
        b = TrajectoryBundle(ToySandbox().generate("q-who-when", kind))
        create("represent", "canonical_events").run_one(b, ctx)
        IDGRepresenter().run_one(b, ctx)
        art = b.get("represent", "idg")
        gt = b.trajectory.meta["injected_fault"]
        gt_event = b.trajectory.events[gt["step"]]
        terminal = next(ev for ev in reversed(b.trajectory.events) if ev.payload)
        closure = _ancestors(art, terminal.id) | {terminal.id}
        (hits if gt_event.id in closure else misses).append(kind)
    assert set(hits) == {
        "malformed_tool_call", "info_withholding",
        "ungrounded_citation", "disobey_task_spec",
    }
    assert set(misses) == {"step_repetition", "premature_termination"}


def test_idg_synthetic_fixture_respects_refs():
    b, ctx = _bundle(success_trace())
    IDGRepresenter().run_one(b, ctx)
    art = b.get("represent", "idg")
    ids = [n["event_id"] for n in art["nodes"]]
    assert ids  # every event carries a payload -> all enter the graph
    # e004 (TOOL_RESULT search) references e003 (TOOL_CALL search)
    assert {"src": "e003", "dst": "e004"} in [
        {"src": e["src"], "dst": e["dst"]} for e in art["edges"]
    ]
    # determinism: repeated runs produce identical artifacts
    art2 = (IDGRepresenter().run_one(b, ctx), b.get("represent", "idg"))[1]
    assert art2["edges"] == art["edges"] and art2["nodes"] == art["nodes"]


def test_idg_requires_r0_events():
    from atap.core.schema import Outcome, Trajectory

    b = TrajectoryBundle(Trajectory(trace_id="empty", task="t"))
    with pytest.raises(ValueError, match="canonical_events"):
        IDGRepresenter().run_one(b, RunContext())


# ----------------------------------------------------------- hierarchy_tree --


def test_tree_sibling_and_child_rules():
    """exploration=sibling (same parent); state-changing=child of the
    previous step."""
    b, ctx = _bundle(success_trace())
    HierarchyTreeRepresenter().run_one(b, ctx)
    art = b.get("represent", "hierarchy_tree")
    nodes = {n["step"]: n for n in art["nodes"]}
    # helpers.success_trace step order: e001 LLM_CALL(plan) -> e002 HANDOFF
    # (state-changing) -> e003 search (exploration=sibling) -> e005 read_doc
    # (exploration=sibling) -> e007 HANDOFF -> e008 LLM_CALL -> e009 submit
    assert nodes["e001"]["parent"] is None
    assert nodes["e002"]["parent"] == "e001"      # state_changing: child of the previous step
    assert nodes["e003"]["parent"] == nodes["e002"]["parent"]   # exploration: sibling of e002
    assert nodes["e005"]["parent"] == nodes["e003"]["parent"] == "e001"
    assert nodes["e007"]["parent"] == "e005"
    assert nodes["e008"]["parent"] == "e007"
    assert nodes["e009"]["parent"] == "e008"
    assert nodes["e003"]["class"] == "exploration"
    assert nodes["e009"]["class"] == "state_changing"
    # environment observations do not become standalone nodes, but are
    # merged into the summary (the read result hangs on e005)
    assert "TrajAudit" in nodes["e005"]["summary"] or "->" in nodes["e005"]["summary"]


def test_tree_stage_ranges_and_md():
    b, ctx = _bundle(ToySandbox().generate("q-drift", "info_withholding"))
    HierarchyTreeRepresenter().run_one(b, ctx)
    art = b.get("represent", "hierarchy_tree")
    stages = [r["stage"] for r in art["stage_ranges"]]
    assert stages == ["plan", "search", "report"]
    for r in art["stage_ranges"]:
        assert r["start"] <= r["end"]
    md = art["tree_md"]
    assert md.startswith("# trace tree")
    assert "== stage: plan ==" in md and "== stage: search ==" in md
    assert "== terminal ==" in md
    # compressed index: exactly one line per step + header (title + n_stages
    # lines each of stage comment/marker + terminal line; count("\n") is one
    # less than the content lines -- the last line has no newline)
    stats = art["stats"]
    assert stats["tree_md_lines"] == stats["n_steps"] + 2 * stats["n_stages"] + 1
    assert stats["n_steps"] < stats["n_events"]   # environment observations merge into summaries, taking no lines
    assert stats["n_exploration"] >= 1   # search/read_doc


def test_tree_deterministic_and_configurable():
    b, ctx = _bundle(ToySandbox().generate("q-who-when", None))
    HierarchyTreeRepresenter().run_one(b, ctx)
    art1 = b.get("represent", "hierarchy_tree")
    HierarchyTreeRepresenter(exploration_actions=["search"]).run_one(b, ctx)
    art2 = b.get("represent", "hierarchy_tree")
    # read_doc re-classified as state_changing changes the structure -> the
    # configuration takes effect
    assert art1["stats"]["n_exploration"] != art2["stats"]["n_exploration"]
    HierarchyTreeRepresenter().run_one(b, ctx)
    art3 = b.get("represent", "hierarchy_tree")
    assert art3["nodes"] == art1["nodes"]        # determinism
