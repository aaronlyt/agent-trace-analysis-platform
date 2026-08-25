"""沙盒单元测试：环境工具、七种 rollout、定向重放语义。"""

from __future__ import annotations

import pytest

from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.bundle import TrajectoryBundle
from atap.sandbox import ToySandbox
from atap.sandbox.env import CORPUS, TASKS, read_doc, search, verify
from atap.sandbox.faults import FAULTS, TOOL_BUDGET


def test_search_returns_two_hits_per_task_query():
    for task in TASKS.values():
        assert f"2 docs" in search(task["query"])


def test_read_doc_and_errors():
    assert read_doc("d1").startswith("TrajAudit")
    assert read_doc("nope").startswith("error: invalid")


def test_verify_branches():
    ok, note = verify("q-trajaudit", "semantic saliency folding (d1)", ["d1"])
    assert ok and "passed" in note
    _, note = verify("q-trajaudit", "semantic saliency folding (d1)", [])
    assert "without reading" in note
    _, note = verify("q-trajaudit", "semantic saliency folding (d2)", ["d1"])
    assert "never read" in note
    _, note = verify("q-trajaudit", "semantic saliency folding", ["d1"])
    assert "missing required citation" in note
    _, note = verify("q-trajaudit", "wrong (d1)", ["d1"])
    assert "does not contain" in note


def _prepared(trace_id_task: str, fault: str | None):
    """生成轨迹并跑 R0 拍平（重放需要事件流）。"""
    sb = ToySandbox()
    t = sb.generate(trace_id_task, fault)
    bundle = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(bundle, RunContext())
    return sb, t


@pytest.mark.parametrize("kind", list(FAULTS))
def test_fault_rollouts_fail_with_ground_truth(kind):
    sb, t = _prepared("q-trajaudit", kind)
    inj = t.meta["injected_fault"]
    assert t.outcome.success is False
    assert inj["kind"] == kind
    assert 0 <= inj["step"] < len(t.events)
    assert t.events[inj["step"]].agent == inj["agent"]
    assert inj["mast_code"] == FAULTS[kind].mast_code


def test_success_rollout():
    sb, t = _prepared("q-trajaudit", None)
    assert t.outcome.success
    assert "injected_fault" not in t.meta
    # 引用边已拍平：read 结果引用 read 调用
    reads = [e for e in t.events if e.kind == "TOOL_RESULT" and e.action == "read_doc"]
    assert reads and reads[0].refs == [t.events[reads[0].index - 1].id]


def test_repetition_bursts_budget():
    sb, t = _prepared("q-trajaudit", "step_repetition")
    n_calls = sum(1 for e in t.events if e.kind == "TOOL_CALL")
    assert n_calls > TOOL_BUDGET
    assert "budget exhausted" in t.outcome.note


def test_rerun_targeted_vs_vague_feedback():
    sb, t = _prepared("q-trajaudit", "info_withholding")
    step = t.meta["injected_fault"]["step"]
    targeted = sb.rerun_from(t, step, "避免 info_withholding：如实报告检索到的文档。")
    vague = sb.rerun_from(t, step, "请更仔细地检查轨迹并重试。")
    assert targeted.outcome.success and targeted.meta["fault_removed"] is True
    assert vague.outcome.success is False and vague.meta["fault_removed"] is False
    assert targeted.meta["rerun_of"] == t.trace_id
    # 前缀保留：重跑轨迹保留了 [0, step) 事件
    assert [e.id for e in targeted.events[:step]] == [e.id for e in t.events[:step]]
    # 重跑后 index 连续且唯一
    idx = [e.index for e in targeted.events]
    assert idx == list(range(len(idx)))


def test_rerun_success_trace_noop_prefix():
    sb, t = _prepared("q-trajaudit", None)
    rr = sb.rerun_from(t, 0, "任意反馈")
    assert rr.outcome.success  # 无故障重放仍成功
