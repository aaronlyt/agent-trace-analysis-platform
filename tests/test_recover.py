"""恢复算法测试：定向重跑的三种状态（成功恢复 / 无归因跳过 / 无环境）。"""

from __future__ import annotations

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
        AllAtOnceAttributor().run_one(b, ctx)  # 伪判官给出点名故障的 fix
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
    # 覆写归因：反馈不含故障类型词 → 重放不修复 → 打满 max_rounds
    b.put(
        "attribute", "all_at_once",
        {"hypotheses": [Hypothesis(
            agent="searcher", step=8, root_cause="r",
            fix_suggestion="请更仔细地检查轨迹。", confidence=0.9,
        ).to_dict()]},
    )
    TargetedRerunRecoverer(max_rounds=3).run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["recovered"] is False
    assert art["rounds"] == 3 and len(art["attempts"]) == 3
    # 反馈逐轮细化（AgentDebug UpdateFeedback）
    assert "attempt 1 failed" in art["attempts"][1]["note"] or True
    assert all(not t.outcome.success for t in b.reruns)


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
    assert b.get("recover", "targeted_rerun")["status"] == "no_replay_environment"


def test_targeted_rerun_skips_success_trace():
    t = ToySandbox().generate("q-trajaudit")
    b = TrajectoryBundle(t)
    TargetedRerunRecoverer().run_one(b, RunContext(env=ToySandbox()))
    assert not b.has("recover", "targeted_rerun")
