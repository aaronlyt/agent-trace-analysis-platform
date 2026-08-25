"""R5 动作签名（TraceProbe 2607.06184）单元与集成测试。"""

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


# ---------------------------------------------------------------- 映射 --

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


# ------------------------------------------------------------ 单轨迹 --

def test_run_one_degrades_without_corpus_reference():
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    ActionSignatureRepresenter().run_one(b, ctx)
    art = b.get("represent", "action_signature")
    assert art["anchor"] is None and art["milestones"] is None
    assert "跳过" in art["note"]
    assert art["signatures"]


def test_effects_on_fault_traces():
    b, _ = _bundle("q-trajaudit", "malformed_tool_call")
    ActionSignatureRepresenter().run_one(b, ctx=RunContext())
    m = _sig_map(b)
    assert m[3]["action_class"] == "SEARCH" and m[3]["effect"] == "FAILED"  # 畸形调用

    b2, _ = _bundle("q-trajaudit", "step_repetition")
    ActionSignatureRepresenter().run_one(b2, ctx=RunContext())
    m2 = _sig_map(b2)
    assert m2[14]["effect"] == "FAILED"  # submit 验证失败（预算耗尽）


# ------------------------------------------------------------- 语料 --

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
    assert art["anchor"]["docs"] == ["d1"]  # q-trajaudit 的 gold 文档
    m = _sig_map(anchor_b)
    # 畸形调用本身 FAILED；其后 searcher 的 reasoning 与 handoff 保持元动作
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
    assert ms["M4_first_passing_validation"]["reached"] is False  # 该轨迹验证失败
    ali = art["alignment"]
    assert ali["lcs_len"] > 0 and 0.0 < ali["coverage"] < 1.0
    # 与成功参照相比多出的两个 search（search#1/#2）落进分歧段
    assert any(s["start_index"] == 5 for s in ali["divergence_spans"])
    # 成功轨迹自身：里程碑全达
    ok = by_id["q-trajaudit--ok0"]
    ok_ms = ok.get("represent", "action_signature")["milestones"]
    assert ok_ms["M4_first_passing_validation"]["reached"] is True


def test_off_anchor_when_search_misses_anchor():
    """构造：检索结果不含锚文档 → OFF-ANCHOR。"""
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", None)
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    ok = TrajectoryBundle(sb.generate("q-trajaudit", None))
    create("represent", "canonical_events").run_one(ok, ctx)
    # 改造失败轨迹：检索返回不含 d1 的结果
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
    assert "无成功轨迹" in art["note"]


def test_no_task_id_falls_back_to_run_one():
    from tests.helpers import failure_trace_ungrounded

    b = TrajectoryBundle(failure_trace_ungrounded())
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    ActionSignatureRepresenter().run_corpus([b], ctx)
    art = b.get("represent", "action_signature")
    assert art["anchor"] is None  # 无 task_id 分组 → 单轨迹降级路径
