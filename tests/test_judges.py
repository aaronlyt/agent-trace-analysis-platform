"""判官/归因算法测试：pseudo_judge 解析、FakeLLM 驱动三算法、契约校验。"""

from __future__ import annotations

import pytest

from atap.analyze.judge_eval import JudgeEvalAnalyzer, JudgeVerdict
from atap.attribute.all_at_once import AllAtOnceAttributor
from atap.classify.mast_judge import MastJudgeClassifier
from atap.classify.taxonomy import MAST_MODES
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.llm.fake_client import FakeLLMClient
from atap.llm.pseudo_judge import _parse_block
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def _bundle(task="q-trajaudit", fault=None):
    t = ToySandbox().generate(task, fault)
    b = TrajectoryBundle(t)
    ctx = RunContext(llm=FakeLLMClient())
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "ssf").run_one(b, ctx)
    return b, ctx


def test_parse_verifier_line_without_action():
    lines = _parse_block("[11] VERIFIER verifier :: failed: answer missing required citation")
    assert lines[0].kind == "VERIFIER"
    assert lines[0].content == "failed: answer missing required citation"
    lines = _parse_block("[3] TOOL_CALL searcher search {'query': 'x'}")
    assert lines[0].action == "search" and lines[0].payload.startswith("{")


def test_mast_vocabulary_complete():
    assert len(MAST_MODES) == 14
    cats = {m["category"] for m in MAST_MODES.values()}
    assert cats == {"FC1", "FC2", "FC3"}
    assert sum(1 for m in MAST_MODES.values() if m["category"] == "FC1") == 5
    assert sum(1 for m in MAST_MODES.values() if m["category"] == "FC2") == 6
    assert sum(1 for m in MAST_MODES.values() if m["category"] == "FC3") == 3


def test_judge_eval_success_vs_failure():
    b, ctx = _bundle("q-trajaudit")
    JudgeEvalAnalyzer().run_one(b, ctx)
    ok_art = b.get("analyze", "judge_eval")
    assert ok_art["score"] >= 8 and ok_art["findings"] == []

    b2, ctx2 = _bundle("q-trajaudit", "step_repetition")
    JudgeEvalAnalyzer().run_one(b2, ctx2)
    art = b2.get("analyze", "judge_eval")
    assert art["score"] <= 4
    assert any(f["step"] is not None for f in art["findings"])


def test_mast_judge_success_trace_empty_and_fault_labeled():
    b, ctx = _bundle("q-trajaudit")
    MastJudgeClassifier().run_one(b, ctx)
    assert b.get("classify", "mast_judge")["labels"] == []

    b2, ctx2 = _bundle("q-trajaudit", "info_withholding")
    MastJudgeClassifier().run_one(b2, ctx2)
    art = b2.get("classify", "mast_judge")
    gt = b2.trajectory.meta["injected_fault"]
    assert art["labels"][0]["code"] == gt["mast_code"]
    assert art["fusion"][0]["mast"] == gt["mast_code"]
    assert art["invalid_codes"] == []


def test_mast_judge_drops_unknown_codes():
    b, ctx = _bundle("q-trajaudit", "premature_termination")
    fake = FakeLLMClient(responses=['{"labels": [{"code": "FM-9.9", "reason": "x", "step": 1}, {"code": "FM-3.1", "reason": "y", "step": 2}]}'])
    ctx.llm = fake
    MastJudgeClassifier().run_one(b, ctx)
    art = b.get("classify", "mast_judge")
    assert [l["code"] for l in art["labels"]] == ["FM-3.1"]
    assert art["invalid_codes"] == ["FM-9.9"]


def test_all_at_once_emits_hypothesis_contract():
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    AllAtOnceAttributor().run_one(b, ctx)
    hyps = b.hypotheses()
    assert len(hyps) == 1
    gt = b.trajectory.meta["injected_fault"]
    h = hyps[0]
    assert h.agent == gt["agent"] and h.step == gt["step"]
    assert h.root_cause_code == gt["mast_code"]
    assert h.fix_suggestion and "malformed_tool_call" in h.fix_suggestion
    assert h.evidence and any("step" not in e for e in h.evidence)  # 引文存在
    assert 0.0 <= h.confidence <= 1.0


def test_all_at_once_skips_success():
    b, ctx = _bundle("q-trajaudit")
    AllAtOnceAttributor().run_one(b, ctx)
    assert not b.has("attribute", "all_at_once")


def test_all_at_once_clamps_bad_step_and_agent():
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    fake = FakeLLMClient(
        responses=['{"responsible_agent": "HAL-9000", "step": 999, "reason": "r", '
                   '"fix_suggestion": "f", "confidence": 0.5, "failure_mode": "FM-1.3"}']
    )
    ctx.llm = fake
    AllAtOnceAttributor().run_one(b, ctx)
    h = b.hypotheses()[0]
    assert h.step == len(b.trajectory.events) - 1
    assert h.agent in b.trajectory.agents()
    assert any("clamped" in e for e in h.evidence)


def test_judges_require_r0_events():
    t = ToySandbox().generate("q-trajaudit")  # 未拍平
    b = TrajectoryBundle(t)
    with pytest.raises(ValueError, match="canonical_events"):
        JudgeEvalAnalyzer().run_one(b, RunContext(llm=FakeLLMClient()))


def test_parse_structured_tolerates_reasoning_noise():
    """推理型模型加固：思考文本/围栏/单引号 dict 均不影响结构化解析。"""
    from atap.analyze.judge_eval import JudgeVerdict
    from atap.llm.base import parse_structured

    # 1) 思考文本在前（其中含伪花括号片段），真 JSON 在后
    noisy = (
        'Let me think... the schema needs {"score"} maybe { \'x\': 1 }.\n'
        "After analysis: {\"score\": 2.5, \"summary\": \"ok\", \"findings\": []}"
    )
    assert parse_structured(noisy, JudgeVerdict).score == 2.5
    # 2) markdown 围栏包裹
    fenced = "结论如下：\n```json\n{\"score\": 9.0, \"summary\": \"s\", \"findings\": []}\n```"
    assert parse_structured(fenced, JudgeVerdict).score == 9.0
    # 3) 单引号 Python 风格 dict（literal_eval 回退）
    single = "{'score': 5.0, 'summary': 's', 'findings': []}"
    assert parse_structured(single, JudgeVerdict).score == 5.0
    # 4) 字符串字面量内的花括号不干扰平衡扫描
    braces_in_str = '{"score": 1.0, "summary": "has } brace { inside", "findings": []}'
    assert parse_structured(braces_in_str, JudgeVerdict).score == 1.0


def test_judge_prompts_do_not_leak_ground_truth():
    """防泄漏回归：三个判官的输入 prompt 绝不含 ground truth 键或故障类型词。

    trace_id 含故障名（如 q-trajaudit--info_withholding）但渲染视图不含
    trace_id/meta；本断言固化该契约，防止未来 prompt 改动引入泄漏。
    """
    gt_tokens = (*FAULTS, "injected_fault", "mast_code", "ground truth")
    for kind in FAULTS:
        b, ctx = _bundle("q-trajaudit", kind)
        fake = FakeLLMClient()
        ctx.llm = fake
        JudgeEvalAnalyzer().run_one(b, ctx)
        MastJudgeClassifier().run_one(b, ctx)
        AllAtOnceAttributor().run_one(b, ctx)
        assert fake.calls, f"{kind}: 判官未被调用"
        for call in fake.calls:
            blob = " ".join(
                str(m.get("content", "")) for m in call["messages"]
            )
            for tok in gt_tokens:
                assert tok not in blob, f"{kind}: prompt 泄漏 {tok!r}（tag={call['tag']}）"
