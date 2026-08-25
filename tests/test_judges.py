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
