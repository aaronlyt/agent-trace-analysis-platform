"""反馈注入再求解（AgenTracer 2509.03312）与沙盒 resolve 测试。"""

from __future__ import annotations

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.llm.fake_client import FakeLLMClient
from atap.recover.feedback_injection import FeedbackInjectionRecoverer
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def _bundle(task="q-trajaudit", fault=None, llm=None):
    b = TrajectoryBundle(ToySandbox().generate(task, fault))
    ctx = RunContext(llm=llm or FakeLLMClient())
    ctx.env = ToySandbox()   # 无 LLM：反馈消费走关键词路径（离线确定性）
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "ssf").run_one(b, ctx)
    create("attribute", "all_at_once").run_one(b, ctx)
    return b, ctx


# ------------------------------------------------------------- 沙盒 --

def test_sandbox_resolve_keyword_feedback_removes_fault():
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", "step_repetition")
    ok = sb.resolve(t, "避免 step_repetition：不要重复相同检索。")
    assert ok.outcome.success and ok.meta["fault_removed"] is True
    assert ok.meta["rerun_of"] == t.trace_id
    assert "injected_fault" not in ok.meta


def test_sandbox_resolve_unrelated_feedback_keeps_fault():
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", "step_repetition")
    bad = sb.resolve(t, "请换一个查询词再检索。")   # 未点名故障类型
    assert not bad.outcome.success and bad.meta["fault_removed"] is False


def test_sandbox_resolve_llm_semantic_fallback():
    """关键词未命中时，注入的 LLM 判 yes/no（伪判官按故障词模拟）。"""
    llm = FakeLLMClient()
    sb = ToySandbox(llm=llm)
    t = sb.generate("q-trajaudit", "step_repetition")
    # 自由文本不含故障类型词 → 关键词不命中 → 交 LLM（伪判官读消息内的
    # 故障规格与反馈文本，看到反馈描述了"重复检索"语义即 yes）
    feedback = "上一轮无进展地重复了相同的检索调用直至预算耗尽；本轮应直接使用已有检索结果继续推进。"
    ok = sb.resolve(t, feedback)
    assert llm.calls and llm.calls[-1]["tag"] == "feedback_match"
    assert ok.outcome.success  # 伪判官从反馈里识别出故障语义


def test_runtime_injects_llm_into_sandbox():
    from atap.core.config import config_from_dict
    from atap.runtime import build_context

    cfg = config_from_dict({
        "stages": {"represent": ["canonical_events"]},
        "llm": {"type": "fake"},
        "sandbox": {"type": "toy"},
    })
    ctx = build_context(cfg, "runs/x")
    assert ctx.env._llm is ctx.llm


# ------------------------------------------------------- 恢复算法 --

@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_feedback_injection_recovers_all_faults(kind):
    b, ctx = _bundle("q-trajaudit", kind)
    FeedbackInjectionRecoverer().run_one(b, ctx)
    art = b.get("recover", "feedback_injection")
    assert art["recovered"] is True, f"{kind}: {art['attempts']}"
    assert art["mode"] == "full_reresolve"
    assert art["rounds"] <= 3                      # AgenTracer：3 轮
    assert len(b.reruns) == art["rounds"]
    assert all(r.meta.get("resolve_mode") == "full_reresolve" for r in b.reruns)
    assert art["feedback_rounds"]                  # 反思反馈留痕


def test_no_hypothesis_skips_explicitly():
    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit", "step_repetition"))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    FeedbackInjectionRecoverer().run_one(b, ctx)
    art = b.get("recover", "feedback_injection")
    assert art["status"] == "skipped_no_hypothesis" and not art["recovered"]


def test_no_env_degrades():
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    ctx.env = None
    FeedbackInjectionRecoverer().run_one(b, ctx)
    art = b.get("recover", "feedback_injection")
    assert art["status"] == "no_replay_environment"


def test_reflection_prompt_no_gt_leak():
    """弱反馈（无故障关键词）→ 第 1 轮失败 → 触发反思调用；反思 prompt
    不得含 ground truth 键/故障类型词。"""
    gt_tokens = (*FAULTS, "injected_fault", "mast_code", "ground truth")
    b, ctx = _bundle("q-trajaudit", "info_withholding")
    fake = ctx.llm
    # 改写 top 假设的 fix 为不含关键词的弱反馈，强制走失败→反思→再解路径
    b.get("attribute", "all_at_once")["hypotheses"][0]["fix_suggestion"] = (
        "请更仔细地检查轨迹后再提交。"
    )
    FeedbackInjectionRecoverer(max_rounds=2).run_one(b, ctx)
    reflect_calls = [c for c in fake.calls if c["tag"] == "feedback_reflection"]
    assert reflect_calls, "弱反馈未触发反思调用"
    for call in reflect_calls:
        blob = " ".join(str(m.get("content", "")) for m in call["messages"])
        for tok in gt_tokens:
            assert tok not in blob, f"反思 prompt 泄漏 {tok!r}"
    art = b.get("recover", "feedback_injection")
    assert art["recovered"] is True  # 第 2 轮反思反馈（含症状修正）后恢复


def test_closed_loop_with_feedback_injection(tmp_path):
    from atap.core.config import config_from_dict
    from atap.runtime import run_config
    from tests.helpers import write_traces_jsonl

    sb = ToySandbox()
    traces = [sb.generate("q-trajaudit", k) for k in FAULTS]
    src = write_traces_jsonl(tmp_path / "t.jsonl", traces)
    cfg = config_from_dict({
        "run_name": "fi-closed-loop",
        "source": {"type": "jsonl", "path": src},
        "llm": {"type": "fake"},
        "sandbox": {"type": "toy"},
        "closed_loop": True,
        "stages": {
            "represent": ["canonical_events", "ssf"],
            "analyze": ["judge_eval"],
            "classify": ["mast_judge"],
            "attribute": ["all_at_once"],
            "recover": [{"name": "feedback_injection", "params": {"max_rounds": 3}}],
        },
    })
    bundles, reports = run_config(cfg, tmp_path / "out")
    assert reports[-1].n_failures == 0  # 闭环验证轮全部通过
    for b in bundles:
        loop = b.get("recover", "closed_loop")
        assert loop["verified_improved"] is True
