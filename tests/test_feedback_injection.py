"""Feedback-injection re-solving (AgenTracer 2509.03312) and sandbox resolve
tests."""

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
    ctx.env = ToySandbox()   # no LLM: feedback consumption takes the keyword path (offline, deterministic)
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "ssf").run_one(b, ctx)
    create("attribute", "all_at_once").run_one(b, ctx)
    return b, ctx


# ------------------------------------------------------------- sandbox --

def test_sandbox_resolve_keyword_feedback_removes_fault():
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", "step_repetition")
    ok = sb.resolve(t, "Avoid step_repetition: do not repeat the same search.")
    assert ok.outcome.success and ok.meta["fault_removed"] is True
    assert ok.meta["rerun_of"] == t.trace_id
    assert "injected_fault" not in ok.meta


def test_sandbox_resolve_unrelated_feedback_keeps_fault():
    sb = ToySandbox()
    t = sb.generate("q-trajaudit", "step_repetition")
    bad = sb.resolve(t, "Please search again with a different query.")   # does not name the fault type
    assert not bad.outcome.success and bad.meta["fault_removed"] is False


def test_sandbox_resolve_llm_semantic_fallback():
    """On a keyword miss, the injected LLM judges yes/no (the pseudo-judge
    simulates via the fault-spec words)."""
    llm = FakeLLMClient()
    sb = ToySandbox(llm=llm)
    t = sb.generate("q-trajaudit", "step_repetition")
    # the free text contains no fault-type word -> no keyword hit -> handed
    # to the LLM (the pseudo-judge reads the fault spec and the feedback
    # text inside the message; seeing that the feedback describes the
    # "repeated search" semantics, it answers yes)
    feedback = ("Last round repeated the same search call with no progress "
                "until the budget was exhausted; this round should use the "
                "existing results and move forward directly.")
    ok = sb.resolve(t, feedback)
    assert llm.calls and llm.calls[-1]["tag"] == "feedback_match"
    assert ok.outcome.success  # the pseudo-judge recognized the fault semantics from the feedback


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


# ------------------------------------------------------- recovery algorithm --

@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_feedback_injection_recovers_all_faults(kind):
    b, ctx = _bundle("q-trajaudit", kind)
    FeedbackInjectionRecoverer().run_one(b, ctx)
    art = b.get("recover", "feedback_injection")
    assert art["recovered"] is True, f"{kind}: {art['attempts']}"
    assert art["mode"] == "full_reresolve"
    assert art["rounds"] <= 3                      # AgenTracer: 3 rounds
    assert len(b.reruns) == art["rounds"]
    assert all(r.meta.get("resolve_mode") == "full_reresolve" for r in b.reruns)
    assert art["feedback_rounds"]                  # reflection feedback recorded


@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_first_round_recovery_stops_loop_without_reflection(kind):
    """Fourth audit round (2026-08-27): rounds<=3 above also passes when the
    loop runs to exhaustion -- success must *stop* the loop. With the default
    attribution pipeline the pseudo-judge's fix names the fault type, so the
    round-1 keyword path already removes the fault: rounds==1, exactly one
    re-solve, and the reflect call (which regenerates next-round feedback)
    never fires (FakeLLM.calls records every LLM call)."""
    b, ctx = _bundle("q-trajaudit", kind)
    fake = ctx.llm
    FeedbackInjectionRecoverer(max_rounds=3).run_one(b, ctx)
    art = b.get("recover", "feedback_injection")
    assert art["recovered"] is True, f"{kind}: {art['attempts']}"
    assert art["rounds"] == 1
    assert len(b.reruns) == 1
    reflect_calls = [c for c in fake.calls if c["tag"] == "feedback_reflection"]
    assert reflect_calls == []   # success stops the loop: zero reflection calls
    assert len(art["feedback_rounds"]) == 1   # only the seeded round-1 feedback exists


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


@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_reflection_prompt_no_gt_leak(kind):
    """Weak feedback (no fault keywords) -> round 1 fails -> the reflection
    call fires; the reflection prompt must contain no ground-truth keys /
    fault-type words. Parametrized over all six standard faults (fourth audit
    round 2026-08-27: it previously covered info_withholding only)."""
    gt_tokens = (*FAULTS, "injected_fault", "mast_code", "ground truth")
    b, ctx = _bundle("q-trajaudit", kind)
    fake = ctx.llm
    # rewrite the top hypothesis into a weak feedback without keywords,
    # forcing the fail -> reflect -> re-solve path. Both fields must be
    # neutralized: the round-1 feedback is built from root_cause as well
    # (e.g. the pseudo-judge's premature_termination reason literally
    # contains the phrase "premature termination", which would hit the
    # keyword path and recover in round 1 without ever reflecting)
    hyp = b.get("attribute", "all_at_once")["hypotheses"][0]
    hyp["root_cause"] = "The decisive error was not corrected."
    hyp["fix_suggestion"] = "Please examine the trajectory more carefully before submitting."
    FeedbackInjectionRecoverer(max_rounds=2).run_one(b, ctx)
    reflect_calls = [c for c in fake.calls if c["tag"] == "feedback_reflection"]
    assert reflect_calls, "weak feedback did not trigger the reflection call"
    for call in reflect_calls:
        blob = " ".join(str(m.get("content", "")) for m in call["messages"])
        for tok in gt_tokens:
            assert tok not in blob, f"reflection prompt leaks {tok!r}"
    art = b.get("recover", "feedback_injection")
    assert art["recovered"] is True  # recovered after the round-2 reflection feedback (with symptom correction)


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
    assert reports[-1].n_failures == 0  # all closed-loop verification rounds pass
    for b in bundles:
        loop = b.get("recover", "closed_loop")
        assert loop["verified_improved"] is True
