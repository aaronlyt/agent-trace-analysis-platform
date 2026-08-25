"""L0 免费规则包（AgentDebugX 2607.18754）测试。"""

from __future__ import annotations

from atap.classify.rule_pack import RulePackClassifier
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def _bundle(task="q-trajaudit", fault=None, with_loop=True):
    b = TrajectoryBundle(ToySandbox().generate(task, fault))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    if with_loop:
        create("analyze", "loop_detect", min_consecutive=3).run_one(b, ctx)
    return b, ctx


def _findings(b, **params):
    RulePackClassifier(**params).run_one(b, RunContext())
    return b.get("classify", "rule_pack")


def test_malformed_rule_hits_call_step():
    b, _ = _bundle("q-trajaudit", "malformed_tool_call")
    art = _findings(b)
    rules = {f["rule"] for f in art["findings"]}
    assert "malformed_tool_call" in rules
    f = next(f for f in art["findings"] if f["rule"] == "malformed_tool_call")
    assert f["step"] == 3 and f["agent"] == "searcher"
    assert art["cost"] == "free"


def test_no_progress_rule_consumes_loop_detect():
    b, _ = _bundle("q-trajaudit", "step_repetition")
    art = _findings(b)
    f = next(f for f in art["findings"] if f["rule"] == "no_progress_loop")
    assert f["step"] == 5  # loop_detect 的 repetition_onset
    assert f["mast_code"] == "FM-1.3"


def test_no_progress_fallback_on_r5_only():
    b, _ = _bundle("q-trajaudit", "step_repetition", with_loop=False)
    art = _findings(b)
    assert any(f["rule"] == "no_progress_loop" for f in art["findings"])


def test_premature_success_rule_targets_decision_step():
    b, _ = _bundle("q-trajaudit", "premature_termination")
    art = _findings(b)
    f = next(f for f in art["findings"] if f["rule"] == "premature_success_claim")
    assert f["step"] == 1 and f["agent"] == "planner"  # Eq.5：规划步而非 submit


def test_invalid_output_rule_on_verifier_rejection():
    b, _ = _bundle("q-trajaudit", "disobey_task_spec")
    art = _findings(b)
    f = next(f for f in art["findings"] if f["rule"] == "invalid_output")
    assert f["step"] == 9 and f["agent"] == "reporter"
    assert f["mast_code"] == "FM-1.1"


def test_success_trace_yields_no_findings():
    b, _ = _bundle("q-trajaudit")
    art = _findings(b)
    assert art["findings"] == []


def test_fusion_labels_filled():
    b, _ = _bundle("q-trajaudit", "malformed_tool_call")
    art = _findings(b)
    assert art["fusion"] and art["fusion"][0]["mast"]


def test_all_faults_get_at_least_one_rule_or_none():
    """六故障全跑：靶故障各有命中；其余（信息隐瞒/无据引用）允许空——
    L0 规则包只覆盖机械可验证失败，其余归 L1 判官。"""
    targets = {
        "malformed_tool_call": "malformed_tool_call",
        "step_repetition": "no_progress_loop",
        "premature_termination": "premature_success_claim",
        "disobey_task_spec": "invalid_output",
    }
    for kind, rule in targets.items():
        b, _ = _bundle("q-trajaudit", kind)
        art = _findings(b)
        assert any(f["rule"] == rule for f in art["findings"]), f"{kind} 未命中 {rule}"
