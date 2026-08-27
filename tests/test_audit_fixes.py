"""Targeted regression tests for the paper-consistency audit fixes (second
audit round, 2026-08-25; fourth audit round, 2026-08-27).

Coverage: the loop_detect action-sequence window, judge_eval J.1 outcome
stripping and severity validation, taxonomy definition decontamination,
mast_judge few-shot step numbers and truncation records, action_signature
M3/reference_trace, ssf unfold_line, rule_pack successful-read check,
binary_search AGENT_MESSAGE, feedback_injection last-round reflection;
fourth round: non-vacuous unfold_line expansion + render table semantics,
all_at_once invalid failure_mode clamp trace.
"""

from __future__ import annotations

import pytest

from atap.analyze.judge_eval import Finding, JudgeEvalAnalyzer
from atap.attribute.binary_search import BinarySearchAttributor
from atap.classify.mast_judge import MastJudgeClassifier, _FEW_SHOT as MAST_FEW_SHOT
from atap.classify.rule_pack import RulePackClassifier
from atap.classify.taxonomy import MAST_MODES
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.render import render_event_line
from atap.core.schema import (
    LLM_CALL,
    TASK_END,
    TASK_START,
    TOOL_CALL,
    TOOL_RESULT,
    Outcome,
    TraceEvent,
    Trajectory,
)
from atap.llm.base import LLMError, parse_structured
from atap.llm.fake_client import FakeLLMClient
from atap.llm.pseudo_judge import find_outcome
from atap.represent.action_signature import ActionSignatureRepresenter
from atap.represent.ssf import unfold_line
from atap.sandbox import ToySandbox

from helpers import success_trace


def _ev(i, kind, agent, action=None, payload=None, refs=None, phase=None):
    return TraceEvent(
        id=f"e{i:03d}", ts=float(i), kind=kind, agent=agent, action=action,
        payload=payload or {}, refs=refs or [], phase=phase, parent=None, index=i,
    )


# ------------------------------------------------- loop_detect window semantics --

def _read_loop_trace(spread: int) -> Trajectory:
    """Read the same document 3 times; spread controls how many REASON
    actions separate adjacent reads.

    The span of the three reads on the action sequence = 3 + 2*spread
    signed actions (read-gap-read-gap-read).
    """
    events = [_ev(0, TASK_START, "env")]
    idx = 1
    for _ in range(3):
        events.append(_ev(idx, TOOL_CALL, "searcher", action="read_doc",
                          payload={"doc_id": "d1"}))
        idx += 1
        events.append(_ev(idx, TOOL_RESULT, "env", action="read_doc",
                          refs=[events[-1].id], payload={"content": "doc"}))
        idx += 1
        for _ in range(spread):
            events.append(_ev(idx, LLM_CALL, "searcher", phase="search",
                              payload={"content": "thinking..."}))
            idx += 1
    events.append(_ev(idx, TASK_END, "env"))
    return Trajectory(trace_id="syn", task="t", events=events,
                      outcome=Outcome(success=True))


def _loop_hits(t: Trajectory, **params):
    from atap.analyze.loop_detect import LoopDetectAnalyzer

    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_one(b, ctx)
    LoopDetectAnalyzer(**params).run_one(b, RunContext())
    return b.get("analyze", "loop_detect")["detected"]


def test_re_read_churn_window_is_action_sequence():
    # span of 5 signed actions <= window=5 -> triggers
    hits = _loop_hits(_read_loop_trace(spread=1), window=5)
    assert any(d["predicate"] == "re_read_churn" and d["repeats"] == 3 for d in hits)
    # span of 7 signed actions > window=5 -> no trigger (the old read-subsequence
    # window would have triggered spuriously)
    hits = _loop_hits(_read_loop_trace(spread=2), window=5)
    assert not any(d["predicate"] == "re_read_churn" for d in hits)


def test_redundant_search_window_is_action_sequence():
    def search_spread_trace(spread: int) -> Trajectory:
        events = [_ev(0, TASK_START, "env")]
        idx = 1
        for _ in range(2):
            events.append(_ev(idx, TOOL_CALL, "searcher", action="search",
                              payload={"query": "q"}))
            idx += 1
            events.append(_ev(idx, TOOL_RESULT, "env", action="search",
                              refs=[events[-1].id],
                              payload={"content": "search results for 'q': 1 docs [d1]"}))
            idx += 1
            for _ in range(spread):
                events.append(_ev(idx, LLM_CALL, "searcher", phase="search",
                                  payload={"content": "thinking..."}))
                idx += 1
        events.append(_ev(idx, TASK_END, "env"))
        return Trajectory(trace_id="syn", task="t", events=events,
                          outcome=Outcome(success=True))

    hits = _loop_hits(search_spread_trace(1), window=3)
    assert any(d["predicate"] == "redundant_search" for d in hits)  # span 3 <= 3
    hits = _loop_hits(search_spread_trace(2), window=3)
    assert not any(d["predicate"] == "redundant_search" for d in hits)  # span 5 > 3


# ------------------------------------------------- judge_eval J.1 protocol --

def _sandbox_bundle(task="q-trajaudit", fault=None):
    b = TrajectoryBundle(ToySandbox().generate(task, fault))
    ctx = RunContext(llm=FakeLLMClient())
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "ssf").run_one(b, ctx)
    return b, ctx


def test_judge_eval_hides_outcome_by_default():
    b, ctx = _sandbox_bundle("q-trajaudit", "step_repetition")
    fake = FakeLLMClient()
    ctx.llm = fake
    JudgeEvalAnalyzer().run_one(b, ctx)
    blob = " ".join(str(m.get("content", "")) for c in fake.calls for m in c["messages"])
    assert "outcome:" not in blob          # J.1: the judge sees no success/failure result
    assert b.get("analyze", "judge_eval")["outcome_shown"] is False


def test_judge_eval_show_outcome_restores_line():
    b, ctx = _sandbox_bundle("q-trajaudit", "step_repetition")
    fake = FakeLLMClient()
    ctx.llm = fake
    JudgeEvalAnalyzer(show_outcome=True).run_one(b, ctx)
    blob = " ".join(str(m.get("content", "")) for c in fake.calls for m in c["messages"])
    assert "outcome: FAILURE" in blob


def test_find_outcome_none_when_absent_and_verifier_inference():
    assert find_outcome([{"role": "user", "content": "no header here"}]) is None
    # the pseudo-judge infers success/failure from VERIFIER lines (a success
    # trajectory still gets a high score -- the e2e tests depend on this)
    from atap.llm.pseudo_judge import pseudo_judge_handler

    b, ctx = _sandbox_bundle("q-trajaudit")  # success trajectory
    fake = FakeLLMClient()
    ctx.llm = fake
    JudgeEvalAnalyzer().run_one(b, ctx)
    assert b.get("analyze", "judge_eval")["score"] >= 8


def test_severity_alias_normalized_and_unknown_rejected():
    assert parse_structured(
        '{"severity": "high", "description": "d", "step": 1}', Finding
    ).severity == "critical"
    assert parse_structured(
        '{"severity": "low", "description": "d"}', Finding
    ).severity == "minor"
    with pytest.raises(LLMError):
        parse_structured('{"severity": "extreme", "description": "d"}', Finding)


# ------------------------------------------------- taxonomy decontamination --

def test_taxonomy_definitions_no_unfounded_extensions():
    d = {k: v["definition"] for k, v in MAST_MODES.items()}
    assert "false positive" not in d["FM-3.3"] and "misleading conclusion" not in d["FM-3.3"]
    assert "repeated downstream failures" not in d["FM-2.4"] and "(requirements, constraints, findings)" not in d["FM-2.4"]
    assert "budget exhausted" not in d["FM-1.3"] and "(format, required fields" not in d["FM-1.1"]
    assert "suspended" not in d["FM-1.5"]
    # regress the distinctive semantics of the paper's App. A
    assert "a different agent" in d["FM-1.2"]
    assert "Unexpectedly" in d["FM-2.1"]
    assert "could" in d["FM-2.4"]  # the paper's "could impact": the weak causal hedge is kept


# ------------------------------------------------- mast_judge few-shot --

def test_mast_fewshot_does_not_encode_eval_answers():
    """Anti-leak regression (third audit round, 2026-08-26): the few-shot
    examples must not be answer keys for the sandbox evaluation set.

    History: the original few-shot ("searcher repeats the search at steps
    3/5/7 ... step=5, FM-1.3") matched the step_repetition ground truth
    verbatim (agent, onset step, MAST code, and even the verifier symptom),
    and an earlier version of this test *required* that match — cementing
    the leak. Now the assertion is inverted: no few-shot may contain any
    GT (agent, code) pair, any GT onset step as a "step N" reference, or
    any GT step-run like "3/5/7".
    """
    import re

    from atap.attribute.all_at_once import _FEW_SHOT as AAO_FEW_SHOT

    # real R0 indices of sandbox step_repetition: three search calls 3/5/7,
    # GT onset=5 (kept as documentation of the trace facts)
    t = ToySandbox().generate("q-trajaudit", "step_repetition")
    b = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(b, RunContext())
    searches = [e.index for e in b.trajectory.events
                if e.kind == "TOOL_CALL" and e.action == "search"]
    gt = b.trajectory.meta["injected_fault"]["step"]
    assert searches == [3, 5, 7] and gt == 5

    gts = set()  # (agent, step, mast_code) across all sandbox faults
    for kind in ("step_repetition", "malformed_tool_call", "info_withholding",
                 "premature_termination", "ungrounded_citation",
                 "disobey_task_spec"):
        meta = ToySandbox().generate("q-trajaudit", kind).meta["injected_fault"]
        gts.add((meta["agent"], meta["step"], meta["mast_code"]))
    for fs in (MAST_FEW_SHOT, AAO_FEW_SHOT):
        assert "3/5/7" not in fs
        for agent, step, code in gts:
            assert not (agent in fs and code in fs), (
                f"few-shot encodes the eval answer pair ({agent}, {code})")
            assert not re.search(rf"step[= ]{step}\b", fs), (
                f"few-shot cites GT onset step {step}")


def test_mast_max_labels_validates_then_truncates_with_record():
    b, ctx = _sandbox_bundle("q-trajaudit", "premature_termination")
    fake = FakeLLMClient(responses=[
        '{"labels": [{"code": "FM-9.9", "reason": "x", "step": 1},'
        ' {"code": "FM-3.1", "reason": "y", "step": 2},'
        ' {"code": "FM-1.3", "reason": "z", "step": 3},'
        ' {"code": "FM-3.2", "reason": "w", "step": 4}]}'
    ])
    ctx.llm = fake
    MastJudgeClassifier(max_labels=2).run_one(b, ctx)
    art = b.get("classify", "mast_judge")
    # validate the full list first, then truncate: invalid codes do not take
    # slots, and the dropped excess is recorded
    assert [l["code"] for l in art["labels"]] == ["FM-3.1", "FM-1.3"]
    assert art["invalid_codes"] == ["FM-9.9"]
    assert art["truncated_codes"] == ["FM-3.2"]


# ------------------------------------------------- all_at_once clamping --

def test_all_at_once_invalid_failure_mode_clamps_with_trace():
    """Fourth audit round (2026-08-27): an invalid MAST code is clamped to
    None like the step/agent clamps -- and the clamp must leave a trace in
    the hypothesis evidence instead of silently disappearing."""
    from atap.attribute.all_at_once import AllAtOnceAttributor

    b, ctx = _sandbox_bundle("q-trajaudit", "step_repetition")
    ctx.llm = FakeLLMClient(responses=[
        '{"responsible_agent": "searcher", "step": 5, "reason": "repeats",'
        ' "fix_suggestion": "f", "confidence": 0.5, "failure_mode": "FM-9.9"}'
    ])
    AllAtOnceAttributor().run_one(b, ctx)
    hyps = b.hypotheses()
    assert len(hyps) == 1
    assert hyps[0].root_cause_code is None            # invalid code does not pass through
    assert any(
        "failure_mode" in e and "FM-9.9" in e for e in hyps[0].evidence
    )                                                  # ...but the clamp is recorded
    # a valid code still passes through untouched (pseudo-judge path)
    b2, ctx2 = _sandbox_bundle("q-trajaudit", "step_repetition")
    AllAtOnceAttributor().run_one(b2, ctx2)
    assert b2.hypotheses()[0].root_cause_code == "FM-1.3"
    assert not any("failure_mode" in e for e in b2.hypotheses()[0].evidence)


# ------------------------------------------------- action_signature --

def test_milestone_m3_step_is_first_read_max():
    sigs = [
        {"index": 1, "action_class": "FILE_READ", "target": "d1", "effect": "JUSTIFIED"},
        {"index": 3, "action_class": "FILE_READ", "target": "d2", "effect": "JUSTIFIED"},
        {"index": 5, "action_class": "FILE_READ", "target": "d1", "effect": "JUSTIFIED"},  # repeated read
    ]
    ms = ActionSignatureRepresenter._milestones(sigs, {"d1", "d2"})
    assert ms["M3_all_anchors_read"]["reached"] is True
    assert ms["M3_all_anchors_read"]["step"] == 3  # max of first reads per anchor, not the last read


def test_alignment_reference_trace_is_ref_bundle():
    sb = ToySandbox()
    bundles = []
    for t in sb.generate_corpus(successes_per_task=1):
        b = TrajectoryBundle(t)
        bundles.append(b)
    ctx = RunContext()
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    ActionSignatureRepresenter().run_corpus(bundles, ctx)
    rep = next(b for b in bundles if b.trace_id == "q-trajaudit--step_repetition")
    ali = rep.get("represent", "action_signature")["alignment"]
    assert ali["reference_trace"] == "q-trajaudit--ok0"  # reference trace id, not the compared task_id


# ------------------------------------------------- ssf unfold_line --

def test_unfold_line_expands_rendered_line():
    """Fourth audit round (2026-08-27): the previous version of this test was
    vacuous -- both assertions held on the *unexpanded* line (startswith is
    trivially true when unfold_line returns the line unchanged, and the
    "TrajAudit" hit came from the placeholder digest, not from the restored
    body). A synthetic folded observation whose tail marker sits beyond the
    100-char digest pins the real contract: the folded-away body comes back,
    the placeholder marker disappears, and the ``[n] `` line head survives.
    """
    tail_marker = "UNIQUE-FOLDED-TAIL-MARKER-zz9"
    body = "folded body payload. " * 12 + tail_marker  # >120 chars, tail beyond the digest
    events = [
        _ev(0, TASK_START, "env"),
        _ev(1, TOOL_CALL, "searcher", action="read_doc", payload={"doc_id": "d1"}),
        _ev(2, TOOL_RESULT, "env", action="read_doc", refs=["e001"],
            payload={"content": "doc head sentence. " + body}),
        _ev(3, TASK_END, "env"),
    ]
    t = Trajectory(trace_id="syn", task="t", events=events,
                   outcome=Outcome(success=False))
    b = TrajectoryBundle(t)
    create("represent", "ssf").run_one(b, RunContext())
    art = b.get("represent", "ssf")
    assert art["stats"]["n_folded"] == 1          # the long observation really got folded
    line = render_event_line(events[2], fold=art["fold"])
    assert "⟦folded:" in line
    assert tail_marker not in line                # ...and the folded-away body is absent pre-unfold
    expanded = unfold_line(line, art)
    assert tail_marker in expanded                # the folded original body is restored
    assert "⟦folded:" not in expanded             # the placeholder marker is gone
    assert expanded.startswith("[2] TOOL_RESULT env read_doc")  # the [n] line head is kept
    assert events[2].payload["content"] in expanded  # the full original text came back


def test_render_table_expansion_requires_whole_content_placeholder():
    """Guard for the other FOLD_PLACEHOLDER_RE call site (render_event_line
    with table=): the unanchored regex must not loosen it -- only a content
    that is *entirely* a placeholder is expanded; placeholder-shaped text
    with trailing content is rendered verbatim."""
    tail_marker = "UNIQUE-FOLDED-TAIL-MARKER-zz9"
    body = "folded body payload. " * 12 + tail_marker
    content = "doc head sentence. " + body
    ev = _ev(2, TOOL_RESULT, "env", action="read_doc", refs=["e001"],
             payload={"content": content})
    fold = {ev.id: "⟦folded:F1 | doc head sentence...⟧"}
    table = {"F1": content}
    # folded render + table -> expanded back to the original body
    expanded_line = render_event_line(ev, fold=fold, table=table)
    assert tail_marker in expanded_line and "⟦folded:" not in expanded_line
    # content that merely *starts* with a placeholder is not replaced
    ev2 = _ev(4, TOOL_RESULT, "env", action="read_doc", refs=["e003"],
              payload={"content": "⟦folded:F1 | doc head sentence...⟧ trailing words"})
    partial = render_event_line(ev2, table=table)
    assert "trailing words" in partial      # rendered verbatim, trailing text kept
    assert table["F1"] not in partial       # no whole-content replacement


# ------------------------------------------------- rule_pack successful read --

def test_premature_success_counts_only_successful_reads():
    events = [
        _ev(0, TASK_START, "env"),
        _ev(1, LLM_CALL, "planner", phase="plan", payload={"content": "submit directly"}),
        _ev(2, TOOL_CALL, "searcher", action="read_doc", payload={"doc_id": "d9"}),
        _ev(3, TOOL_RESULT, "env", action="read_doc", refs=["e002"],
            payload={"content": "error: invalid doc_id"}),   # failed read -- not evidence
        _ev(4, TOOL_CALL, "reporter", action="submit", payload={"answer": "x"}),
        _ev(5, TASK_END, "env"),
    ]
    t = Trajectory(trace_id="syn", task="t", events=events,
                   outcome=Outcome(success=False))
    b = TrajectoryBundle(t)
    RulePackClassifier().run_one(b, RunContext())
    rules = {f["rule"] for f in b.get("classify", "rule_pack")["findings"]}
    assert "premature_success_claim" in rules


# ------------------------------------------------- binary_search A* --


def test_responsible_agent_reads_agent_message():
    events = [
        _ev(0, TASK_START, "env"),
        _ev(1, TOOL_CALL, "agentA", action="search", payload={"query": "q"}),
        _ev(2, TOOL_RESULT, "env", refs=["e001"], payload={"content": "ok"}),
        _ev(3, "AGENT_MESSAGE", "agentB", payload={"content": "message from B"}),
    ]
    assert BinarySearchAttributor._responsible_agent(events, 3) == "agentB"


# ------------------------------------------------- feedback_injection --

def test_feedback_injection_no_reflect_after_final_round():
    from atap.recover.feedback_injection import FeedbackInjectionRecoverer
    from atap.attribute.sbfl import SBFLAttributor  # noqa: F401  (registry)
    from atap.sandbox.faults import FAULTS

    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit", "step_repetition"))
    ctx = RunContext(env=ToySandbox())  # no llm: feedback consumption takes the keyword path (no hit)
    create("represent", "canonical_events").run_one(b, ctx)
    # inject an attribution that does not name the fault (feedback never
    # hits -> all 3 rounds fail)
    from atap.core.schema import Hypothesis

    b.put("attribute", "seed", {"hypotheses": [
        Hypothesis(agent="searcher", step=5, root_cause="r",
                   fix_suggestion="Re-examine the retrieval strategy", confidence=0.5).to_dict()
    ]})
    FeedbackInjectionRecoverer(max_rounds=3).run_one(b, ctx)
    art = b.get("recover", "feedback_injection")
    assert art["recovered"] is False and art["rounds"] == 3
    # after the final round fails, no more feedback is generated: one
    # injected feedback per round, no extra final entry
    assert len(art["feedback_rounds"]) == 3
