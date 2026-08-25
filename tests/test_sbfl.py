"""SBFL 频谱归因（FAMAS 2509.13782）测试。"""

from __future__ import annotations

import pytest

from atap.attribute.sbfl import SBFLAttributor
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox


def _corpus_bundles(successes=2, task="q-trajaudit"):
    """单任务语料：K 成功 + 六故障（其余任务轨迹不混入，频谱按 task_id 分组）。"""
    sb = ToySandbox()
    traces = [sb.generate(task, None, trace_id=f"{task}--ok{i}") for i in range(successes)]
    from atap.sandbox.faults import FAULTS

    traces += [sb.generate(task, k) for k in FAULTS]
    bundles = [TrajectoryBundle(t) for t in traces]
    ctx = RunContext()
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_corpus(bundles, ctx)
    return bundles, ctx


def test_spectrum_hits_repetition_and_premature():
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    by_id = {b.trace_id: b for b in bundles}
    rep = by_id["q-trajaudit--step_repetition"]
    top = max(rep.hypotheses(), key=lambda h: h.confidence)
    gt = rep.trajectory.meta["injected_fault"]
    assert (top.step, top.agent) == (gt["step"], gt["agent"])  # 5, searcher

    prem = by_id["q-trajaudit--premature_termination"]
    top_p = max(prem.hypotheses(), key=lambda h: h.confidence)
    gt_p = prem.trajectory.meta["injected_fault"]
    assert (top_p.step, top_p.agent) == (gt_p["step"], gt_p["agent"])  # 1, planner


def test_prior_grade_confidence_and_role():
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    for b in bundles:
        art = b.get("attribute", "sbfl")
        if b.succeeded:
            assert art["status"] == "success_no_attribution"
            continue
        assert art["role"] == "L0_statistical_prior"
        assert art["hypotheses"][0]["confidence"] <= 0.5  # 先验级
        assert art["spectrum"]["n_runs"] == 8
        assert art["spectrum"]["n_failed"] == 6 and art["spectrum"]["n_success"] == 2


def test_known_no_signal_faults_are_honest_misses():
    """动作谱与成功轨迹相同的故障（信息隐瞒/无据引用）无频谱信号——
    不断言未命中，只断言产出了诚实的 ranked hypotheses（top-1 非 GT）。"""
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    by_id = {b.trace_id: b for b in bundles}
    info = by_id["q-trajaudit--info_withholding"]
    top = max(info.hypotheses(), key=lambda h: h.confidence)
    gt = info.trajectory.meta["injected_fault"]
    assert (top.step, top.agent) != (gt["step"], gt["agent"])  # 已知 miss
    assert top.agent in ("planner", "searcher", "reporter")     # 仍是有效 agent


def test_run_one_is_explicit_corpus_scope():
    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit", "step_repetition"))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    SBFLAttributor().run_one(b, ctx)
    art = b.get("attribute", "sbfl")
    assert art["status"] == "corpus_scope_required"
    assert art["hypotheses"] == []


def test_missing_r5_raises():
    bundles = [TrajectoryBundle(ToySandbox().generate("q-trajaudit", "step_repetition"))]
    ctx = RunContext()
    create("represent", "canonical_events").run_corpus(bundles, ctx)
    with pytest.raises(ValueError, match="action_signature"):
        SBFLAttributor().run_corpus(bundles, ctx)


def test_lam_validation():
    with pytest.raises(ValueError):
        SBFLAttributor(lam=1.5).run_corpus([], RunContext())


def test_second_occurrence_step_convention():
    """轨迹内重复 ≥2 次的签名取第二次出现（Eq.5 最早决定性错误约定）。"""
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    rep = next(b for b in bundles if "step_repetition" in b.trace_id)
    hyps = rep.hypotheses()
    # 重复 search 签名（若有假设）不应指向首次出现（3），而应是 5
    for h in hyps:
        if "SEARCH" in h.evidence[0] and "steps=[3, 5, 7]" in h.evidence[0]:
            assert h.step == 5
