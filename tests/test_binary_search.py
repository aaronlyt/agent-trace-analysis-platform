"""L2 二分定位（Who&When 2505.00212 Algorithm 2）测试。"""

from __future__ import annotations

import math

import pytest

from atap.attribute.binary_search import BinarySearchAttributor, _parse_half
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.llm.fake_client import FakeLLMClient
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def _bundle(task="q-trajaudit", fault=None):
    b = TrajectoryBundle(ToySandbox().generate(task, fault))
    ctx = RunContext(llm=FakeLLMClient())
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "ssf").run_one(b, ctx)
    return b, ctx


def test_parse_half():
    assert _parse_half("lower half") == "lower half"
    assert _parse_half("Upper Half") == "upper half"
    with pytest.raises(ValueError):
        _parse_half("maybe in the middle")


def test_rounds_bounded_by_log2n():
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    BinarySearchAttributor().run_one(b, ctx)
    art = b.get("attribute", "binary_search")
    n = len(b.trajectory.events)
    # 原文 App. D.3 的 ⌈log₂n⌉ 为上界（区间每次至少折半；非 2 幂时可更少）
    assert art["n_rounds_expected"] == math.ceil(math.log2(n))
    assert len(art["rounds"]) <= art["n_rounds_expected"]
    for r in art["rounds"]:
        assert r["answer"] in ("upper half", "lower half")


def test_agent_walkback_from_env_event():
    """s* 落在 env 侧事件时回退到最近 agent 行为事件。"""
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    BinarySearchAttributor().run_one(b, ctx)
    top = b.hypotheses()[0]
    assert top.agent == "searcher"  # 收敛点是 env 的 TOOL_RESULT → 回退到 search 调用


@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_agent_level_six_of_six(kind):
    b, ctx = _bundle("q-trajaudit", kind)
    BinarySearchAttributor().run_one(b, ctx)
    gt = b.trajectory.meta["injected_fault"]
    top = max(b.hypotheses(), key=lambda h: h.confidence)
    assert top.agent == gt["agent"], f"{kind}: agent 归因错位"


# 已知离线行为（伪判官 + 二分的固有收敛特性，如实断言）：
# step_repetition 收敛到最后一次重复（症状），step 偏晚——与 Who&When
# 报告的二分 step 级弱于逐步审查的方向一致
_STEP_HITS = {
    "malformed_tool_call", "premature_termination", "info_withholding",
    "ungrounded_citation", "disobey_task_spec",
}


@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_step_level_expected_hits(kind):
    b, ctx = _bundle("q-trajaudit", kind)
    BinarySearchAttributor().run_one(b, ctx)
    gt = b.trajectory.meta["injected_fault"]
    top = max(b.hypotheses(), key=lambda h: h.confidence)
    if kind in _STEP_HITS:
        assert top.step == gt["step"], f"{kind}: step={top.step} 期望 {gt['step']}"
    else:  # step_repetition：收敛到症状（第 3 次重复），偏晚 3 步
        assert top.step == gt["step"] + 3


def test_refine_disabled_uses_mechanical_fields():
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    BinarySearchAttributor(refine=False).run_one(b, ctx)
    art = b.get("attribute", "binary_search")
    hyp = b.hypotheses()[0]
    assert hyp.root_cause.startswith("二分定位收敛于")
    assert hyp.confidence == 0.5


def test_prompts_do_not_leak_ground_truth():
    gt_tokens = (*FAULTS, "injected_fault", "mast_code", "ground truth")
    for kind in FAULTS:
        b, ctx = _bundle("q-trajaudit", kind)
        fake = ctx.llm
        BinarySearchAttributor().run_one(b, ctx)
        assert fake.calls
        for call in fake.calls:
            blob = " ".join(str(m.get("content", "")) for m in call["messages"])
            for tok in gt_tokens:
                assert tok not in blob, f"{kind}: prompt 泄漏 {tok!r}（tag={call['tag']}）"


def test_success_trace_skipped_by_default():
    b, ctx = _bundle("q-trajaudit")
    BinarySearchAttributor().run_one(b, ctx)
    assert not b.has("attribute", "binary_search")
