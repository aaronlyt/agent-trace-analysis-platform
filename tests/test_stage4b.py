"""Stage 4B tests: claim ledger/audit (DRIFT), tree-indexed diagnosis
(CodeTracer), hierarchical causal-graph attribution (CHIEF). All
deterministic acceptance via the pseudo-judge (FakeLLM)."""

from __future__ import annotations

import json

import pytest

from atap.attribute.chief import (
    ChiefAttributor,
    ChiefVerdict,
    OracleSet,
    SubtaskEval,
    SubtaskEvals,
)
from atap.attribute.claim_audit import (
    ClaimAuditAttributor,
    SupportRecord,
    SupportRecords,
    TraceVerdict,
)
from atap.attribute.tree_diagnosis import (
    StagePick,
    TreeDiagnosisAttributor,
)
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.render import TRACE_BEGIN, TRACE_END
from atap.core.schema import Hypothesis
from atap.llm import FakeLLMClient
from atap.represent.claim_ledger import Ledger, LedgerClaim
from atap.sandbox import ToySandbox
from atap.sandbox.faults import EXTRA_FAULTS, FAULTS

DRIFT_HOME = ("info_withholding", "premature_termination",
              "ungrounded_citation", "disobey_task_spec")
#: mechanism mapping (progressive-filtering conclusions; expected by the six-fault construction)
CHIEF_MECH = {
    "step_repetition": "executor_loop",
    "malformed_tool_call": "local_error",
    "info_withholding": "dataflow_first_pollution",
    "premature_termination": "planning_error",
    "ungrounded_citation": "local_error",
    "disobey_task_spec": "local_error",
}
#: expected tree-level localization stage
TREE_STAGE = {
    "step_repetition": "search",
    "malformed_tool_call": "search",
    "info_withholding": "report",
    "premature_termination": "plan",
    "ungrounded_citation": "report",
    "disobey_task_spec": "report",
}


def _bundle(trace, reps=("ssf",)):
    b = TrajectoryBundle(trace)
    ctx = RunContext(llm=FakeLLMClient())
    create("represent", "canonical_events").run_one(b, ctx)
    for r in reps:
        create("represent", r).run_one(b, ctx)
    return b, ctx


def _hyp(b, name):
    arts = b.artifacts.get("attribute", {})
    h = arts.get(name, {}).get("hypotheses") or []
    return Hypothesis.from_dict(h[0]) if h else None


class _Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def complete(self, messages, *, schema=None, model=None, tag=""):
        self.calls.append(tag)
        return self.inner.complete(messages, schema=schema, model=model, tag=tag)


def _strip_trace_blocks(text: str) -> str:
    """Remove the tau blocks (TRACE_BEGIN..TRACE_END) from a prompt blob.

    Event-line content is the trajectory itself and is judge-visible by
    construction in every view (e.g. a successful run's VERIFIER line embeds
    the env pass note "matches gold ..."); the leak scan targets the
    judge-side wrapping text around tau, so the blocks are stripped first."""
    while TRACE_BEGIN in text and TRACE_END in text:
        lo = text.index(TRACE_BEGIN)
        hi = text.index(TRACE_END, lo) + len(TRACE_END)
        text = text[:lo] + text[hi:]
    return text


# ------------------------------------------------------------ claim ledger --


def test_ledger_structure_and_compactness():
    b, ctx = _bundle(ToySandbox().generate("q-trajaudit", None))
    create("represent", "claim_ledger").run_one(b, ctx)
    art = b.get("represent", "claim_ledger")
    assert 1 <= art["n_claims"] <= 6          # compact ledger (paper prefers 3-5)
    types = {c["type"] for c in art["claims"]}
    assert types <= {"entity", "constraint", "evidence",
                     "retrieval", "compute", "process"}
    # the task spec contains a citation requirement -> a hard-constraint claim enters the ledger
    assert any(c["type"] == "constraint" for c in art["claims"])
    assert art["hard_constraints"]


def test_ledger_clamps_dependency_steps():
    """P3 regression (audit 2026-08-27 §4): b_k/U_k used to be stored
    unvalidated (they could precede i_k or leave the trajectory), breaking
    the i_k <= b_k invariant claim_audit relies on. They are now clamped
    like i_k: never below i_k, at most the last event index."""
    scripted = Ledger(
        task_goal="g",
        claims=[
            LedgerClaim(   # everything out of range -> clamps to the last event
                id="c1", text="t1", type="entity", status="finalized",
                introduced_step=99, first_effective_step=150,
                reuse_steps=[200, -3],
            ),
            LedgerClaim(   # b_k/U_k earlier than i_k -> clamped up to i_k
                id="c2", text="t2", type="evidence", status="consequential",
                introduced_step=5, first_effective_step=0, reuse_steps=[2, 6],
            ),
            LedgerClaim(   # no dependency info -> stays None / empty
                id="c3", text="t3", type="process", status="consequential",
                introduced_step=2,
            ),
        ],
    )
    b = TrajectoryBundle(ToySandbox().generate("q-who-when", None))
    ctx = RunContext(llm=FakeLLMClient(responses=[scripted]))
    create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "claim_ledger").run_one(b, ctx)
    art = b.get("represent", "claim_ledger")
    last = len(b.trajectory.events) - 1
    by_id = {c["id"]: c for c in art["claims"]}
    assert (by_id["c1"]["introduced_step"], by_id["c1"]["first_effective_step"],
            by_id["c1"]["reuse_steps"]) == (last, last, [last])
    assert (by_id["c2"]["first_effective_step"],
            by_id["c2"]["reuse_steps"]) == (5, [5, 6])
    assert by_id["c3"]["first_effective_step"] is None
    assert by_id["c3"]["reuse_steps"] == []
    for c in art["claims"]:   # the invariant, after clamping
        i = c["introduced_step"]
        assert 0 <= i <= last
        if c["first_effective_step"] is not None:
            assert i <= c["first_effective_step"] <= last
        assert all(i <= s <= last for s in c["reuse_steps"])


# ---------------------------------------------------------- claim audit --


def test_claim_audit_hits_all_drift_home_faults():
    ctx = RunContext(llm=FakeLLMClient())
    for kind in DRIFT_HOME:
        b, _ = _bundle(ToySandbox().generate("q-who-when", kind))
        create("represent", "claim_ledger").run_one(b, ctx)
        ClaimAuditAttributor().run_one(b, ctx)
        gt = b.trajectory.meta["injected_fault"]
        h = _hyp(b, "claim_audit")
        assert h is not None, f"{kind}: no attribution produced"
        assert (h.agent, h.step) == (gt["agent"], gt["step"]), (
            f"{kind}: ({h.agent},{h.step}) != GT ({gt['agent']},{gt['step']})"
        )
        art = b.get("attribute", "claim_audit")
        assert art["support_records"]          # the four support levels are recorded
        levels = {r["support_status"] for r in art["support_records"]}
        assert levels & {"MISSING", "CONFLICTING"}   # origin levels of harmful claims


def test_claim_audit_boundary_misses_are_honest():
    """Honest recording of the method boundary for non-claim-level faults:
    in step_repetition all claims are true (budget exhaustion is not a
    claim-level failure -> no harmful claim); in malformed the 'no usable
    results' claim is true (the retrieval really did fail), the root cause is
    at the tool-call layer (DRIFT's Tracer explicitly forbids selecting pure
    tool/query spans)."""
    ctx = RunContext(llm=FakeLLMClient())
    b, _ = _bundle(ToySandbox().generate("q-who-when", "step_repetition"))
    create("represent", "claim_ledger").run_one(b, ctx)
    ClaimAuditAttributor().run_one(b, ctx)
    art = b.get("attribute", "claim_audit")
    assert art["status"] == "no_harmful_claim"

    b2, _ = _bundle(ToySandbox().generate("q-who-who", "malformed_tool_call")
                    ) if False else _bundle(
        ToySandbox().generate("q-who-when", "malformed_tool_call"))
    create("represent", "claim_ledger").run_one(b2, ctx)
    ClaimAuditAttributor().run_one(b2, ctx)
    gt = b2.trajectory.meta["injected_fault"]
    h = _hyp(b2, "claim_audit")
    assert h is not None and (h.agent, h.step) != (gt["agent"], gt["step"])


def test_claim_audit_success_and_contract():
    ctx = RunContext(llm=FakeLLMClient())
    b, _ = _bundle(ToySandbox().generate("q-who-when", None))
    create("represent", "claim_ledger").run_one(b, ctx)
    ClaimAuditAttributor().run_one(b, ctx)
    assert b.get("attribute", "claim_audit")["status"] == "success_no_attribution"

    b2, _ = _bundle(ToySandbox().generate("q-who-when", "info_withholding"))
    with pytest.raises(ValueError, match="claim_ledger"):
        ClaimAuditAttributor().run_one(b2, ctx)   # missing ledger artifact raises explicitly


def test_claim_audit_include_success_no_gold_leak():
    """P1 regression (audit 2026-08-27 §2-1): with include_success=True on a
    successful trajectory, the outcome line used to carry the env verifier
    note "matches gold 'all-at-once' (d3)" straight into the judge prompt.
    The success path now renders include_outcome=False (the paper's input
    protocol is question+span only), so no message contains an outcome line
    or the gold phrase outside the tau block. Scripted responses drive all
    three call sites (ledger + support + trace) on the success trajectory;
    the VERIFIER event line inside tau still embeds the env pass note -- that
    is the trajectory itself (judge-visible by construction in every view),
    so the scan strips TRACE blocks before looking for "matches gold"."""
    ledger = Ledger(task_goal="g", claims=[LedgerClaim(
        id="c1", text="the answer is known from memory; no search needed",
        type="entity", status="finalized", introduced_step=1,
        first_effective_step=2, reuse_steps=[3],
    )])
    support = SupportRecords(records=[SupportRecord(
        claim_id="c1", support_status="MISSING", support_steps=[],
        missing_support="no search/read support at all",
        verdict="harmful_unsupported_commitment", responsible_step=1,
        reason="asserts the answer from memory with no evidence shown",
    )])
    trace = TraceVerdict(
        first_error_step=1, error_steps=[1],
        reason="keep the introduction span (conservative backtracking)",
    )
    b, ctx = _bundle(ToySandbox().generate("q-who-when", None))
    ctx.llm = FakeLLMClient(responses=[ledger, support, trace])
    create("represent", "claim_ledger").run_one(b, ctx)
    ClaimAuditAttributor(include_success=True).run_one(b, ctx)
    # the include_success path still produces an attribution end to end
    assert _hyp(b, "claim_audit") is not None
    assert ctx.llm.calls, "FakeLLM recorded no calls"
    assert {c["tag"] for c in ctx.llm.calls} == {
        "claim_ledger", "claim_audit_support", "claim_audit_trace"}
    for call in ctx.llm.calls:
        blob = "".join(str(m.get("content", "")) for m in call["messages"])
        assert "outcome:" not in blob, (
            f"success path must render without the outcome line "
            f"(tag={call['tag']})"
        )
        assert "matches gold" not in _strip_trace_blocks(blob), (
            f"gold phrase leaked outside the tau block (tag={call['tag']}): "
            f"{_strip_trace_blocks(blob)[:120]}..."
        )
        low = blob.lower()
        for key in ("injected_fault", "ground_truth", "gold_doc", "gold_answer"):
            assert key not in low, f"GT key {key!r} leaked (tag={call['tag']})"


def test_claim_chain_llm_call_count():
    ctx = RunContext(llm=_Counting(FakeLLMClient()))
    b, _ = _bundle(ToySandbox().generate("q-drift", "ungrounded_citation"))
    create("represent", "claim_ledger").run_one(b, ctx)
    ClaimAuditAttributor().run_one(b, ctx)
    # ledger 1 (single global pass) + support audit 1 + dependency backtracking 1 = 3 calls
    assert ctx.llm.calls.count("claim_ledger") == 1
    assert ctx.llm.calls.count("claim_audit_support") == 1
    assert ctx.llm.calls.count("claim_audit_trace") == 1


# ---------------------------------------------------------- tree-indexed diagnosis --


def test_tree_diagnosis_six_of_six_with_token_saving():
    ctx = RunContext(llm=FakeLLMClient())
    for kind, want_stage in TREE_STAGE.items():
        b, _ = _bundle(ToySandbox().generate("q-trajaudit", kind))
        create("represent", "hierarchy_tree").run_one(b, ctx)
        TreeDiagnosisAttributor().run_one(b, ctx)
        gt = b.trajectory.meta["injected_fault"]
        art = b.get("attribute", "tree_diagnosis")
        h = _hyp(b, "tree_diagnosis")
        assert h is not None and (h.agent, h.step) == (gt["agent"], gt["step"]), (
            f"{kind}: ({h.agent},{h.step}) != GT"
        )
        assert art["chosen_stage"] == want_stage
        # tree first, then drill down: drill-down render lines < full render
        # lines (the token-saving evidence of compressed-index navigation)
        assert art["inspected_render_lines"] < art["full_render_lines"]


def test_tree_diagnosis_requires_tree_artifact():
    ctx = RunContext(llm=FakeLLMClient())
    b, _ = _bundle(ToySandbox().generate("q-trajaudit", "info_withholding"),
                   reps=())
    with pytest.raises(ValueError, match="hierarchy_tree"):
        TreeDiagnosisAttributor().run_one(b, ctx)


# ---------------------------------------------------------------- CHIEF --


def test_hcg_graph_structure():
    b, ctx = _bundle(ToySandbox().generate("q-who-when", None))
    create("represent", "hcg").run_one(b, ctx)
    art = b.get("represent", "hcg")
    assert [s["phase"] for s in art["subtasks"]] == ["plan", "search", "report"]
    assert art["stats"]["n_sub_edges"] == 2       # adjacent subtask progression
    assert art["stats"]["n_agt_edges"] >= 2       # handoff collaboration edges
    agents = {n["agent"] for n in art["agent_nodes"]}
    assert {"planner", "searcher", "reporter"} <= agents
    # OTAR tuple: searcher has thought/action/observation
    searcher = next(n for n in art["agent_nodes"] if n["agent"] == "searcher")
    assert searcher["otar"]["action"] is not None
    assert searcher["otar"]["observation"] is not None


def test_hcg_premature_single_subtask():
    b, ctx = _bundle(ToySandbox().generate("q-who-when", "premature_termination"))
    create("represent", "hcg").run_one(b, ctx)
    art = b.get("represent", "hcg")
    assert [s["phase"] for s in art["subtasks"]] == ["plan"]   # no search/report


def test_chief_six_of_six_step_agent_with_mechanisms():
    ctx = RunContext(llm=FakeLLMClient())
    for kind, want_mech in CHIEF_MECH.items():
        b, _ = _bundle(ToySandbox().generate("q-who-when", kind))
        create("represent", "hcg").run_one(b, ctx)
        ChiefAttributor().run_one(b, ctx)
        gt = b.trajectory.meta["injected_fault"]
        art = b.get("attribute", "chief")
        h = _hyp(b, "chief")
        assert h is not None and (h.agent, h.step) == (gt["agent"], gt["step"]), (
            f"{kind}: ({h.agent},{h.step}) != GT ({gt['agent']},{gt['step']})"
        )
        assert art["mechanism"] == want_mech
        assert art["n_llm_calls"] == 3           # oracle + eval + localize
        assert art["subtask_evals"]              # reverse-topology backtracking record


def test_chief_success_and_contract():
    ctx = RunContext(llm=FakeLLMClient())
    b, _ = _bundle(ToySandbox().generate("q-who-when", None))
    create("represent", "hcg").run_one(b, ctx)
    ChiefAttributor().run_one(b, ctx)
    assert b.get("attribute", "chief")["status"] == "success_no_attribution"

    b2, _ = _bundle(ToySandbox().generate("q-who-when", "info_withholding"),
                    reps=())
    with pytest.raises(ValueError, match="hcg"):
        ChiefAttributor().run_one(b2, ctx)


# ------------------------------------------------------------- anti-leak --


def test_stage4b_prompts_no_gt_leakage():
    """No 4B judge prompt may contain fault-type words / GT keys
    (anti-leak regression extension)."""
    from atap.attribute import chief as chief_mod
    from atap.attribute import claim_audit as audit_mod
    from atap.attribute import tree_diagnosis as tree_mod
    from atap.represent import claim_ledger as ledger_mod

    forbidden = (*FAULTS.keys(), "injected_fault", "ground_truth", "gold_doc",
                 "ground truth")
    prompts = [
        ledger_mod._SYSTEM, audit_mod._B_SYSTEM, audit_mod._T_SYSTEM,
        tree_mod._STAGE_SYSTEM, tree_mod._DRILL_SYSTEM,
        chief_mod._ORACLE_SYSTEM, chief_mod._EVAL_SYSTEM, chief_mod._LOCALIZE_SYSTEM,
    ]
    for p in prompts:
        low = p.lower()
        for word in forbidden:
            assert word.lower() not in low, f"prompt leaks {word!r}: {p[:60]}..."


def test_stage4b_runtime_prompts_no_fault_leakage():
    """Runtime anti-leak scan for the 4B judges (mirror of stage 4c's
    test_stage4c_runtime_prompts_no_fault_leakage): every message actually
    sent by claim_ledger/claim_audit, tree_diagnosis and chief across the six
    standard faults must be free of fault names (including the EXTRA_FAULTS
    pair) and GT keys. The pseudo-judge's symptom texts do name faults, but
    those are responses -- request messages only ever wrap task text, ledger
    claim lines, tau views, tree.md and subtask summaries, all derived from
    the trajectory itself."""
    forbidden = (
        *FAULTS.keys(), *EXTRA_FAULTS.keys(),
        "injected_fault", "ground_truth", "gold_doc", "gold_answer",
        "ground truth",
    )
    for kind in FAULTS:
        chains = []
        ctx = RunContext(llm=FakeLLMClient())   # claim chain
        b, _ = _bundle(ToySandbox().generate("q-who-when", kind))
        create("represent", "claim_ledger").run_one(b, ctx)
        ClaimAuditAttributor().run_one(b, ctx)
        chains.append(ctx)
        ctx = RunContext(llm=FakeLLMClient())   # tree chain
        b, _ = _bundle(ToySandbox().generate("q-trajaudit", kind))
        create("represent", "hierarchy_tree").run_one(b, ctx)
        TreeDiagnosisAttributor().run_one(b, ctx)
        chains.append(ctx)
        ctx = RunContext(llm=FakeLLMClient())   # chief chain
        b, _ = _bundle(ToySandbox().generate("q-who-when", kind))
        create("represent", "hcg").run_one(b, ctx)
        ChiefAttributor().run_one(b, ctx)
        chains.append(ctx)
        for c_ctx in chains:
            assert c_ctx.llm.calls, f"{kind}: FakeLLM recorded no calls"
            for call in c_ctx.llm.calls:
                blob = "".join(
                    str(m.get("content", "")) for m in call["messages"]
                ).lower()
                for word in forbidden:
                    assert word.lower() not in blob, (
                        f"{kind}: runtime prompt leaks {word!r} "
                        f"(tag={call['tag']}): {blob[:120]}..."
                    )


# ------------------------------------------------- clamp-with-trace (4B) --


def _chief_scripted(mechanism: str):
    """Bundle + context with hcg configured and the three chief calls
    scripted (empty oracles; S1 fails acceptance; localize verdict with the
    given mechanism)."""
    b, _ = _bundle(ToySandbox().generate("q-who-when", "malformed_tool_call"),
                   reps=())
    ctx = RunContext(llm=FakeLLMClient(responses=[
        OracleSet(oracles=[]),
        SubtaskEvals(evals=[SubtaskEval(subtask_id="S1", passed=False)]),
        ChiefVerdict(responsible_agent="searcher", step=3,
                     mechanism=mechanism, reason="r",
                     fix_suggestion="f", confidence=0.6),
    ]))
    create("represent", "hcg").run_one(b, ctx)
    return b, ctx


def test_chief_clamps_out_of_vocab_mechanism():
    """ChiefVerdict.mechanism is a free string; an out-of-vocabulary value
    used to enter the artifact verbatim. It is now clamped to the closest
    MECHANISMS word (difflib) or "unknown", with the raw value preserved in
    the evidence and the artifact's mechanism_clamped entry (all_at_once
    failure_mode discipline)."""
    b, ctx = _chief_scripted("local_err")          # close to local_error
    ChiefAttributor().run_one(b, ctx)
    art = b.get("attribute", "chief")
    assert art["mechanism"] == "local_error"
    assert art["mechanism_clamped"] == {"from": "local_err", "to": "local_error"}
    h = _hyp(b, "chief")
    assert h.root_cause.startswith("[local_error]")
    assert any("local_err" in e for e in h.evidence)   # raw value on record

    b2, ctx2 = _chief_scripted(" vibes ")          # no close vocabulary word
    ChiefAttributor().run_one(b2, ctx2)
    art2 = b2.get("attribute", "chief")
    assert art2["mechanism"] == "unknown"
    assert art2["mechanism_clamped"] == {"from": " vibes ", "to": "unknown"}
    assert _hyp(b2, "chief").root_cause.startswith("[unknown]")


def test_chief_valid_mechanism_passes_without_trace():
    """A vocabulary mechanism passes through untouched (no clamp record)."""
    b, ctx = _chief_scripted("executor_loop")
    ChiefAttributor().run_one(b, ctx)
    art = b.get("attribute", "chief")
    assert art["mechanism"] == "executor_loop"
    assert "mechanism_clamped" not in art
    assert not any("clamped" in e for e in _hyp(b, "chief").evidence)


def test_tree_diagnosis_clamps_leave_notes():
    """tree_diagnosis step/agent/failure_mode clamps must leave a note in
    the Hypothesis evidence (all_at_once discipline) instead of silently
    rewriting the judge's verdict."""
    b, _ = _bundle(ToySandbox().generate("q-trajaudit", "malformed_tool_call"),
                   reps=())
    ctx = RunContext(llm=FakeLLMClient(responses=[
        StagePick(suspicious_stages=["search"]),
        '{"responsible_agent": "ghost", "step": 999, "reason": "r",'
        ' "fix_suggestion": "f", "confidence": 0.5, "failure_mode": "FM-9.9"}',
    ]))
    create("represent", "hierarchy_tree").run_one(b, ctx)
    TreeDiagnosisAttributor().run_one(b, ctx)
    h = _hyp(b, "tree_diagnosis")
    assert h is not None
    assert h.agent != "ghost" and h.agent in b.trajectory.agents()
    assert h.root_cause_code is None               # FM-9.9 is not a MAST code
    joined = " ".join(h.evidence)
    assert f"step 999->{h.step}" in joined         # step clamped into [lo, end]
    assert "'ghost'" in joined                     # agent clamped into roster
    assert "FM-9.9" in joined and "not a MAST code" in joined


def test_claim_audit_unknown_claim_id_recorded_in_invalid_channel():
    """A B-audit record about a claim the ledger never saw (hallucinated
    claim_id) must land in the invalid channel with a reason — not silently
    vanish, and never reach the harmful-claim ranking or the Tracer
    prompt."""
    support = SupportRecords(records=[
        SupportRecord(
            claim_id="cX", support_status="MISSING",
            verdict="harmful_unsupported_commitment", responsible_step=2,
            reason="no support",
        ),
        SupportRecord(claim_id="c1", support_status="DIRECT",
                      verdict="supported"),
    ])
    ctx = RunContext(llm=FakeLLMClient())
    b, _ = _bundle(ToySandbox().generate("q-who-when", "info_withholding"))
    create("represent", "claim_ledger").run_one(b, ctx)
    ctx.llm = FakeLLMClient(responses=[support])
    ClaimAuditAttributor().run_one(b, ctx)
    art = b.get("attribute", "claim_audit")
    # cX is the only harmful verdict, but it does not exist in the ledger:
    # it neither becomes a hypothesis nor reaches the Tracer
    assert art["status"] == "no_harmful_claim"
    assert any(
        r["claim_id"] == "cX" and r.get("reason") == "claim_id not in ledger"
        for r in art["invalid_support_records"]
    )
    assert all(r["claim_id"] != "cX" for r in art["support_records"])
