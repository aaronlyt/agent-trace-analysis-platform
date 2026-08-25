"""循环检测谓词（TraceProbe Table II）测试。"""

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
    assert sl["repetition_onset_index"] == 5  # 第二次 search = 首次重复（GT 约定）


def test_search_loop_default_threshold_is_paper_value():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    # 原文冻结阈值 10：玩具轨迹（连续 4）不应触发——阈值审计留给配置
    hits = _detected(b)  # min_consecutive 默认 10
    assert not any(d["predicate"] == "search_loop" for d in hits)


def test_no_detection_on_success_trace():
    b, _ = _bundle("q-trajaudit")
    assert _detected(b, min_consecutive=3) == []


def test_redundant_search_fires():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    hits = _detected(b)
    assert any(d["predicate"] == "redundant_search" for d in hits)


def test_re_read_churn_synthetic():
    """同文档 10 窗内读 3 次 → re_read_churn。"""
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


def test_tool_oscillation_synthetic():
    """两轮 READ-WRITE-READ（中间写 FAILED）→ tool_oscillation。"""
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
    seq = ["read", "edit", "read", "edit", "read"]  # R-W-R-W-R：两个环
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
    # 手工把签名改造成 FILE_READ/FILE_WRITE(f.py) 序列（edit 的写 FAILED）
    for s in art["signatures"]:
        s["action_class"] = "FILE_WRITE" if s["target"] == "edit" else "FILE_READ"
        s["target"] = "f.py"
        s["signature"] = f"{s['action_class']}(f.py)"
        if s["action_class"] == "FILE_WRITE":
            s["effect"] = "FAILED"
    hits = _detected(b)
    assert any(d["predicate"] == "tool_oscillation" and d["cycles"] >= 2
               for d in hits)


def test_missing_upstream_raises():
    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit"))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    with pytest.raises(ValueError, match="action_signature"):
        LoopDetectAnalyzer().run_one(b, ctx)
