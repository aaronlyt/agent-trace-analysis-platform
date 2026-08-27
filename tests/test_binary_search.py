"""L2 binary-search localization (Who&When 2505.00212 Algorithm 2) tests."""

from __future__ import annotations

import math

import pytest

from atap.attribute.binary_search import BinarySearchAttributor, _parse_half
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
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
from atap.llm.base import LLMError
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
    assert _parse_half("the error is in the lower half") == "lower half"
    assert _parse_half("The error is in the Upper half.") == "upper half"
    # neither/both halves named -> unparseable, LLM-parse-failure taxonomy
    with pytest.raises(LLMError, match="Unparseable"):
        _parse_half("maybe in the middle")
    with pytest.raises(LLMError, match="Unparseable"):
        _parse_half("somewhere between the lower half and the upper half")


def test_parse_half_negated_answers():
    """Real judges sometimes negate: a negated half means the opposite half."""
    # negated lower -> upper
    assert _parse_half("not in the lower half") == "upper half"
    assert _parse_half("No error in the lower half") == "upper half"
    assert _parse_half("the lower half looks clean") == "upper half"
    assert _parse_half("the lower half is correct") == "upper half"
    assert _parse_half("nothing wrong in the lower half") == "upper half"
    # negated upper -> lower
    assert _parse_half("not in the upper half") == "lower half"
    assert _parse_half("no error in the upper half") == "lower half"
    assert _parse_half("the upper half doesn't contain the error") == "lower half"


def test_unparseable_answer_mid_run_raises_llm_error():
    """A judge answer that names no half aborts the attribution with LLMError
    (no silent coercion, no client retry: the round call is schema-less)."""
    b, _ = _bundle("q-trajaudit", "step_repetition")
    scripted = FakeLLMClient(responses=["I think it is somewhere in the middle"] + ["lower half"] * 8)
    with pytest.raises(LLMError, match="Unparseable binary-search answer"):
        BinarySearchAttributor(refine=False).run_one(b, RunContext(llm=scripted))


def test_s_star_zero_reads_first_event_agent():
    """All-lower convergence pins s*=0 on the first event (TASK_START): with
    no earlier event to walk back to, A* is read directly from
    events[0].agent (here 'user'), never defaulted to 'env'."""
    events = [
        TraceEvent(id="e000", ts=0.0, kind=TASK_START, agent="user", index=0,
                   payload={"task": "t"}),
        TraceEvent(id="e001", ts=1.0, kind=LLM_CALL, agent="planner", index=1),
        TraceEvent(id="e002", ts=2.0, kind=TOOL_CALL, agent="searcher", index=2),
        TraceEvent(id="e003", ts=3.0, kind=TOOL_RESULT, agent="env", index=3),
        TraceEvent(id="e004", ts=4.0, kind=TASK_END, agent="env", index=4),
    ]
    t = Trajectory("all-lower", "t", events=events, outcome=Outcome(success=False))
    b = TrajectoryBundle(t)
    ctx = RunContext(llm=FakeLLMClient(responses=["lower half"] * 6))
    BinarySearchAttributor(refine=False).run_one(b, ctx)
    art = b.get("attribute", "binary_search")
    assert art["s_star"] == 0
    assert all(r["answer"] == "lower half" for r in art["rounds"])
    top = b.hypotheses()[0]
    assert top.step == 0
    assert top.agent == "user"  # events[0].agent read directly


def test_rounds_bounded_by_log2n():
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    BinarySearchAttributor().run_one(b, ctx)
    art = b.get("attribute", "binary_search")
    n = len(b.trajectory.events)
    # the paper's App. D.3 bound of ceil(log2 n) is an upper bound (the
    # interval at least halves each round; can be fewer for non-powers of 2)
    assert art["n_rounds_expected"] == math.ceil(math.log2(n))
    assert len(art["rounds"]) <= art["n_rounds_expected"]
    for r in art["rounds"]:
        assert r["answer"] in ("upper half", "lower half")


def test_agent_walkback_from_env_event():
    """When s* lands on an env-side event, walk back to the nearest agent action event."""
    b, ctx = _bundle("q-trajaudit", "step_repetition")
    BinarySearchAttributor().run_one(b, ctx)
    top = b.hypotheses()[0]
    assert top.agent == "searcher"  # the convergence point is the env TOOL_RESULT -> walk back to the search call


@pytest.mark.parametrize("kind", sorted(FAULTS))
def test_agent_level_six_of_six(kind):
    b, ctx = _bundle("q-trajaudit", kind)
    BinarySearchAttributor().run_one(b, ctx)
    gt = b.trajectory.meta["injected_fault"]
    top = max(b.hypotheses(), key=lambda h: h.confidence)
    assert top.agent == gt["agent"], f"{kind}: agent attribution mismatch"


# Known offline behavior (inherent convergence properties of the pseudo-judge
# + binary search, asserted as-is):
# step_repetition converges to the last repetition (the symptom), so the step
# is late -- consistent with the Who&When finding that binary-search
# step-level accuracy is weaker than step-by-step review
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
        assert top.step == gt["step"], f"{kind}: step={top.step}, expected {gt['step']}"
    else:  # step_repetition: converges to the symptom (the 3rd repetition), 3 steps late
        assert top.step == gt["step"] + 3


def test_refine_disabled_uses_mechanical_fields():
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    BinarySearchAttributor(refine=False).run_one(b, ctx)
    art = b.get("attribute", "binary_search")
    hyp = b.hypotheses()[0]
    assert hyp.root_cause.startswith("Binary-search localization converged on step ")
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
                assert tok not in blob, f"{kind}: prompt leaks {tok!r} (tag={call['tag']})"


def test_success_trace_skipped_by_default():
    b, ctx = _bundle("q-trajaudit")
    BinarySearchAttributor().run_one(b, ctx)
    assert not b.has("attribute", "binary_search")
