"""Judge/attribution algorithm tests: pseudo_judge parsing, FakeLLM-driven
three algorithms, contract checks."""

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
from atap.sandbox import ToySandbox, env
from atap.sandbox.faults import ALL_FAULTS


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
    assert h.evidence and any("step" not in e for e in h.evidence)  # citations exist
    assert 0.0 <= h.confidence <= 1.0


def test_all_at_once_skips_success():
    """Successful trajectories produce no attribution, but the artifact
    still lands (repository-wide contract: downstream must be able to tell
    "ran and skipped" from "not configured at all"; review 2026-08-27 --
    it used to be a bare return)."""
    b, ctx = _bundle("q-trajaudit")
    AllAtOnceAttributor().run_one(b, ctx)
    art = b.get("attribute", "all_at_once")
    assert art["status"] == "success_no_attribution"
    assert art["hypotheses"] == []


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
    t = ToySandbox().generate("q-trajaudit")  # not flattened
    b = TrajectoryBundle(t)
    with pytest.raises(ValueError, match="canonical_events"):
        JudgeEvalAnalyzer().run_one(b, RunContext(llm=FakeLLMClient()))


def test_parse_structured_tolerates_reasoning_noise():
    """Hardening for reasoning models: thinking text / fences / single-quoted
    dicts all leave structured parsing unaffected."""
    from atap.analyze.judge_eval import JudgeVerdict
    from atap.llm.base import parse_structured

    # 1) thinking text first (containing fake brace fragments), the real JSON after
    noisy = (
        'Let me think... the schema needs {"score"} maybe { \'x\': 1 }.\n'
        "After analysis: {\"score\": 2.5, \"summary\": \"ok\", \"findings\": []}"
    )
    assert parse_structured(noisy, JudgeVerdict).score == 2.5
    # 2) wrapped in a markdown fence
    fenced = "The conclusion is as follows:\n```json\n{\"score\": 9.0, \"summary\": \"s\", \"findings\": []}\n```"
    assert parse_structured(fenced, JudgeVerdict).score == 9.0
    # 3) a single-quoted Python-style dict (literal_eval fallback)
    single = "{'score': 5.0, 'summary': 's', 'findings': []}"
    assert parse_structured(single, JudgeVerdict).score == 5.0
    # 4) braces inside string literals do not disturb the balance scan
    braces_in_str = '{"score": 1.0, "summary": "has } brace { inside", "findings": []}'
    assert parse_structured(braces_in_str, JudgeVerdict).score == 1.0


def test_judge_prompts_do_not_leak_ground_truth():
    """Anti-leak regression: the input prompts of the three judges must
    never contain ground-truth keys or fault-type words.

    Covers the full fault roster -- the six FAULTS plus the phase-four
    EXTRA_FAULTS (retrieval_detour / agent_deadlock), which the third-round
    version of this test missed.

    The trace_id contains the fault name (e.g. q-trajaudit--
    info_withholding) but the rendered view contains neither trace_id nor
    meta; this assertion pins down that contract, preventing future prompt
    changes from introducing leaks.

    "matches gold" additionally guards the verifier-success-note leak
    (review 2026-08-27 P0): the VERIFIER event line is rendered into every
    judge view, so a success note naming the gold answer/doc would hand the
    oracle to the judge -- see test_success_verifier_line_carries_no_gold.
    """
    gt_tokens = (*ALL_FAULTS, "injected_fault", "mast_code", "ground truth",
                 "matches gold")
    for kind in ALL_FAULTS:
        b, ctx = _bundle("q-trajaudit", kind)
        fake = FakeLLMClient()
        ctx.llm = fake
        JudgeEvalAnalyzer().run_one(b, ctx)
        MastJudgeClassifier().run_one(b, ctx)
        AllAtOnceAttributor().run_one(b, ctx)
        assert fake.calls, f"{kind}: the judges were not called"
        for call in fake.calls:
            blob = " ".join(
                str(m.get("content", "")) for m in call["messages"]
            )
            for tok in gt_tokens:
                assert tok not in blob, f"{kind}: prompt leaks {tok!r} (tag={call['tag']})"


def test_success_verifier_line_carries_no_gold():
    """Anti-leak regression (review 2026-08-27 P0), success path: on a
    successful trajectory the VERIFIER event line enters every judge view
    (render_trace emits all events), so the verifier success note must not
    name the gold answer or the gold doc.

    History: env.verify used to return "passed: ... matches gold
    '<answer>' (<doc>)", which put the oracle straight into judge_eval
    prompts (8 of 14 judge_eval calls in a fresh `atap demo` carried it);
    the fault-roster test above never caught it because faulted trajectories
    only ever see "failed:" notes. The scan is the FULL prompt blob -- no
    TRACE-block stripping: the verifier line inside tau is exactly the leak
    channel, not an exemption."""
    for task, gold in (
        ("q-trajaudit", "semantic saliency folding"),
        ("q-drift", "claim ledger"),
        ("q-who-when", "all-at-once"),
    ):
        b, ctx = _bundle(task)
        assert b.trajectory.outcome.success, f"{task}: fixture must succeed"
        fake = FakeLLMClient()
        ctx.llm = fake
        JudgeEvalAnalyzer().run_one(b, ctx)
        MastJudgeClassifier().run_one(b, ctx)
        AllAtOnceAttributor().run_one(b, ctx)
        assert fake.calls, f"{task}: the judges were not called"
        for call in fake.calls:
            blob = " ".join(
                str(m.get("content", "")) for m in call["messages"]
            )
            assert "matches gold" not in blob, (
                f"{task}: verifier success note leaks the gold phrase "
                f"(tag={call['tag']})"
            )
            assert f"gold '{gold}'" not in blob, (
                f"{task}: verifier success note names the gold answer "
                f"(tag={call['tag']})"
            )


def test_fewshot_step_numbers_do_not_collide_with_gt_onsets():
    """Anti-leak regression (fourth review round, 2026-08-27): every step
    number cited in the built-in few-shot examples of judge_eval and
    mast_judge must be fictional -- not equal to any sandbox GT onset step
    (FAULTS + EXTRA_FAULTS, computed across all tasks) and not reproducing
    a GT step-run like step_repetition's "3/5/7".

    History: judge_eval's example cited "step 3"/"step 9", which are exactly
    the onsets of malformed_tool_call and ungrounded_citation/
    disobey_task_spec -- a judge that pattern-matches the demonstration
    onto the trajectory gets the answer key for free. The third-round
    check (test_audit_fixes.py) covered only mast_judge/all_at_once; this
    pins judge_eval with the same criterion.
    """
    import re

    from atap.analyze.judge_eval import _FEW_SHOT as JE_FEW_SHOT
    from atap.classify.mast_judge import _FEW_SHOT as MAST_FEW_SHOT

    onsets = {
        ToySandbox().generate(task, kind).meta["injected_fault"]["step"]
        for task in env.TASKS
        for kind in ALL_FAULTS
    }
    assert onsets  # sanity: the sandbox roster actually produced onsets
    for name, fs in (("judge_eval", JE_FEW_SHOT), ("mast_judge", MAST_FEW_SHOT)):
        cited = {int(n) for n in re.findall(r"step\D*(\d+)", fs)}
        assert cited, f"{name}: few-shot cites no step number (regex broken?)"
        assert not (cited & onsets), (
            f"{name}: few-shot cites GT onset step(s) {sorted(cited & onsets)}; "
            "move the fictional step numbers off the onset set"
        )
        assert "3/5/7" not in fs  # GT search-call run of step_repetition


def test_extra_modes_empty_definition_skipped_with_trace(tmp_path):
    """A mode with an empty/whitespace definition must never reach the judge
    prompt (a bare code+name line invites free-form guessing); the skip is
    recorded in the artifact's skipped_extra_modes instead of being silent."""
    import json

    f = tmp_path / "modes.json"
    f.write_text(
        json.dumps({"modes": [
            {"code": "NM-1", "name": "Blank mode", "definition": "   "},
            {"code": "NM-2", "name": "Usable mode",
             "definition": "repeats identical calls without progress"},
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )
    clf = MastJudgeClassifier(extra_modes_file=str(f))
    assert sorted(clf.extra_modes) == ["NM-2"]
    assert clf.skipped_extra_modes == [{"code": "NM-1", "reason": "empty definition"}]

    b, ctx = _bundle("q-trajaudit", "step_repetition")
    fake = FakeLLMClient(responses=[
        '{"labels": [{"code": "NM-2", "reason": "repeats calls", "step": 2}]}'
    ])
    ctx.llm = fake
    clf.run_one(b, ctx)
    art = b.get("classify", "mast_judge")
    assert [l["code"] for l in art["labels"]] == ["NM-2"]
    assert art["skipped_extra_modes"] == [{"code": "NM-1", "reason": "empty definition"}]
    blob = " ".join(str(m.get("content", "")) for m in fake.calls[0]["messages"])
    assert "NM-2" in blob and "NM-1" not in blob  # empty definition never enters the prompt


def test_judge_prompts_use_ascii_punctuation_only():
    """Judge-visible prompt text must be pure ASCII/English punctuation: the
    em dash (——) previously leaked from the definitions block into all three
    judge prompts (taxonomy definitions + mast_judge few-shot/extra modes +
    judge_eval few-shot); it is replaced with "--" everywhere the judge can
    see it."""
    from atap.analyze.judge_eval import _FEW_SHOT as JE_FEW_SHOT
    from atap.classify.mast_judge import _FEW_SHOT as MAST_FEW_SHOT
    from atap.classify.taxonomy import mast_definitions_block

    for text in (JE_FEW_SHOT, MAST_FEW_SHOT, mast_definitions_block()):
        assert "——" not in text
        assert "--" in text   # the separator survives as ASCII "--"


def test_taxonomy_definitions_align_with_paper_appendix_a():
    """Pins the wording restored in the fourth review round: the definitions
    below enter the judge prompt verbatim (mast_definitions_block), so drift
    from the paper's Appendix A (e.g. toward the sandbox fault symptoms)
    would effectively steer the judge toward the ground truth."""
    d = {k: v["definition"] for k, v in MAST_MODES.items()}
    # FM-2.6: the paper's consequence clause was accidentally dropped
    assert "unexpected or undesired behaviors" in d["FM-2.6"]
    # FM-1.1: the paper says "specified constraints", not "explicit"
    assert "specified constraints" in d["FM-1.1"] and "explicit" not in d["FM-1.1"]
    # FM-2.2: the paper's word is "unclear or incomplete data", not "ambiguous"
    assert "unclear or incomplete data" in d["FM-2.2"]
    # FM-2.3: the paper anchors to the "intended objective", not "established goal"
    assert "intended objective" in d["FM-2.3"] and "established goal" not in d["FM-2.3"]
    # FM-2.1: the paper's "Unexpected" is kept (via "Unexpectedly")
    assert "Unexpectedly" in d["FM-2.1"]
    # FM-3.1: the paper quantifies "all necessary information"
    assert "before all necessary information" in d["FM-3.1"]
    # FM-3.2: the paper lets "errors or inconsistencies" propagate undetected
    assert "errors or inconsistencies" in d["FM-3.2"]
