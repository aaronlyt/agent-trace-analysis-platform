"""Stage 4C tests: L3 counterfactual replay (TraceElephant
counterfactual_replay + DoVer dover) and the sandbox message-intervention
replay infrastructure. All deterministic acceptance via the pseudo-judge."""

from __future__ import annotations

import json

import pytest

from atap.attribute.counterfactual_replay import CounterfactualReplayAttributor
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import Hypothesis, TraceEvent, Trajectory
from atap.llm import FakeLLMClient
from atap.llm.pseudo_judge import pseudo_judge_handler
from atap.recover.dover import DoVerRecoverer
from atap.sandbox import ToySandbox
from atap.sandbox.faults import EXTRA_FAULTS, FAULTS


def _bundle(trace, env=None):
    b = TrajectoryBundle(trace)
    ctx = RunContext(llm=FakeLLMClient(), env=env)
    create("represent", "canonical_events").run_one(b, ctx)
    return b, ctx


# ------------------------------------------------- sandbox message-intervention replay infra --


def test_replay_invariance_without_effective_edit():
    """Replay with no effective edit (empty text) == the original trajectory
    suffix (kind/agent/payload equal event by event) -- the prefix-consistency
    invariant of the replay middleware."""
    sb = ToySandbox()
    for kind in FAULTS:
        b, _ = _bundle(sb.generate("q-trajaudit", kind))
        runs = sb.replay_intervene(b.trajectory, 3, "")
        suffix = runs[0].events[3:]
        orig = b.trajectory.events[3:]
        assert [(e.kind, e.agent, e.payload) for e in suffix] == [
            (e.kind, e.agent, e.payload) for e in orig
        ], f"{kind}: no-edit replay changed the suffix"


def test_replay_edit_replaces_message_in_place_and_fixes():
    sb = ToySandbox()
    b, _ = _bundle(sb.generate("q-who-when", "info_withholding"))
    t = b.trajectory
    edit = "Avoid info_withholding: faithfully report the retrieved documents at step 8"
    runs = sb.replay_intervene(t, 8, edit)
    assert runs[0].outcome.success and runs[0].meta["fault_removed"]
    assert runs[0].events[8].payload["content"] == edit   # replaced in place
    assert runs[0].events[:8] == t.events[:8]              # prefix preserved
    # ineffective edit: the failure course is unchanged
    runs2 = sb.replay_intervene(t, 8, "please review carefully")
    assert not runs2[0].outcome.success
    assert not runs2[0].meta["fault_removed"]
    # window mode (TraceElephant's k-step reading)
    runs3 = sb.replay_intervene(t, 8, edit, horizon=3)
    assert len(runs3[0].events) == 8 + 3
    # x3 repeats (DoVer reading): multiple returns, incrementing ids
    runs4 = sb.replay_intervene(t, 8, edit, n_repeats=3)
    assert len(runs4) == 3 and len({r.trace_id for r in runs4}) == 3


def test_replay_fault_removal_is_step_sensitive():
    """Regression (step-independence hole): a fault-naming edit removes the
    fault ONLY when applied at the fault's onset step. The same edit applied
    at a non-onset (symptom) step must leave fault_removed=False -- otherwise
    any candidate would be validated by keyword luck alone and the
    pseudo-causality filter would rest on the oracle's wording instead of on
    the replay mechanism (the attributor turns fault_removed=False into a
    refuted verdict with lowered confidence)."""
    sb = ToySandbox()
    b, _ = _bundle(sb.generate("q-who-when", "info_withholding"))
    t = b.trajectory
    gt = t.meta["injected_fault"]
    edit = ("Avoid info_withholding: faithfully report the retrieved "
            "documents at that step")
    # at the onset step: consumed -> fault-free suffix
    onset = sb.replay_intervene(t, gt["step"], edit)[0]
    assert onset.meta["intervention_on_onset_step"] is True
    assert onset.meta["fault_removed"] is True
    assert onset.outcome.success
    # the same fault-naming edit at the symptom step (GT+1, the compose step
    # echoing the withheld "no documents" claim): NOT consumed
    symptom = sb.replay_intervene(t, gt["step"] + 1, edit)[0]
    assert symptom.meta["intervention_on_onset_step"] is False
    assert symptom.meta["fault_removed"] is False
    assert not symptom.outcome.success


def test_replay_intervention_applied_recorded_faithfully():
    """intervention_applied is a faithful record, not a constant: True only
    when the intervened step is a message event (LLM_CALL/HANDOFF/
    AGENT_MESSAGE); a TOOL_CALL step gets no in-place message replacement
    (recorded False with a reason) even when the middleware consumed the
    fault-naming edit (fault_removed True)."""
    sb = ToySandbox()
    b, _ = _bundle(sb.generate("q-who-when", "malformed_tool_call"))
    t = b.trajectory
    gt = t.meta["injected_fault"]
    assert t.events[gt["step"]].kind == "TOOL_CALL"   # GT step = the malformed call
    edit = ("Avoid malformed_tool_call: validate argument completeness "
            "before issuing the tool call.")
    runs = sb.replay_intervene(t, gt["step"], edit)
    assert runs[0].meta["fault_removed"]                 # onset + fault-naming edit
    assert runs[0].meta["intervention_applied"] is False  # but no message to replace
    assert "TOOL_CALL" in runs[0].meta["intervention_applied_note"]
    # the tool call's payload was not overwritten by the edit text
    assert "content" not in runs[0].events[gt["step"]].payload
    # message step: applied True
    assert t.events[1].kind == "LLM_CALL"
    msg = sb.replay_intervene(
        t, 1, "Avoid malformed_tool_call: fix the call at planning.",
    )
    assert msg[0].meta["intervention_applied"] is True


# ------------------------------------------------ counterfactual_replay --


def test_cf_replay_validates_gt_candidates_and_refutes_symptom():
    """L3 final review: the GT root-cause candidate is validated (confidence
    raised); the symptom-step candidate is refuted (confidence lowered,
    filtering pseudo-causes)."""
    sb = ToySandbox()
    b, ctx = _bundle(sb.generate("q-drift", "info_withholding"),
                     env=sb)
    create("attribute", "all_at_once").run_one(b, ctx)
    gt = b.trajectory.meta["injected_fault"]
    # inject a symptom-step candidate (compose=GT+1; editing that step does
    # not change the failure course)
    b.put("attribute", "dummy_upstream", {"hypotheses": [
        Hypothesis(agent="reporter", step=gt["step"] + 1,
                   root_cause="symptom step", confidence=0.9).to_dict()
    ]})
    CounterfactualReplayAttributor().run_one(b, ctx)
    art = b.get("attribute", "counterfactual_replay")
    by_step = {v["candidate"]["step"]: v for v in art["verdicts"]}
    assert by_step[gt["step"]]["verdict"] == "validated"
    assert by_step[gt["step"] + 1]["verdict"] == "refuted"
    hyps = {h.step: h for h in b.hypotheses()
            if h.root_cause.startswith("[L3")}
    assert hyps[gt["step"]].confidence > 0.7    # validated: raised
    assert hyps[gt["step"] + 1].confidence < 0.9  # refuted: lowered


def test_cf_replay_six_faults_all_validated():
    sb = ToySandbox()
    for kind in FAULTS:
        b, ctx = _bundle(sb.generate("q-who-when", kind), env=sb)
        create("attribute", "all_at_once").run_one(b, ctx)
        CounterfactualReplayAttributor().run_one(b, ctx)
        art = b.get("attribute", "counterfactual_replay")
        gt = b.trajectory.meta["injected_fault"]
        v = art["verdicts"][0]
        assert v["candidate"]["step"] == gt["step"]
        assert v["verdict"] == "validated"
        assert art["horizon"] == 3            # TraceElephant k=3


def test_cf_replay_degrades_explicitly():
    # no upstream candidates
    b, ctx = _bundle(ToySandbox().generate("q-who-when", "info_withholding"),
                     env=ToySandbox())
    CounterfactualReplayAttributor().run_one(b, ctx)
    assert b.get("attribute", "counterfactual_replay")["status"] == \
        "no_upstream_candidates"
    # no intervention replay environment
    b2, ctx2 = _bundle(ToySandbox().generate("q-who-when", "info_withholding"))
    create("attribute", "all_at_once").run_one(b2, ctx2)
    with pytest.raises(RuntimeError, match="replay_intervene"):
        CounterfactualReplayAttributor().run_one(b2, ctx2)


def test_cf_replay_supersedes_upstream_in_merged_hypotheses():
    """The reviewed copy replaces the upstream original in
    bundle.hypotheses() (same agent+step). Without supersede semantics the
    refuted copy (confidence -0.3) coexists with its un-reviewed original
    and every downstream max(confidence) selection keeps the original --
    the L3 review would have no effect at all (review 2026-08-27 P1)."""
    sb = ToySandbox()
    b, ctx = _bundle(sb.generate("q-who-when", "info_withholding"), env=sb)
    create("attribute", "all_at_once").run_one(b, ctx)
    upstream = b.hypotheses()
    assert len(upstream) == 1
    CounterfactualReplayAttributor().run_one(b, ctx)
    merged = b.hypotheses()
    # exactly one hypothesis per reviewed (agent, step) -- not two
    assert len(merged) == 1
    assert merged[0].source == "counterfactual_replay"
    assert merged[0].root_cause.startswith("[L3 counterfactual replay validated]")
    assert merged[0].confidence == min(upstream[0].confidence + 0.2, 1.0)
    # supersede is read-time only: the upstream artifact itself stays intact
    art_orig = b.get("attribute", "all_at_once")["hypotheses"][0]
    assert art_orig["step"] == upstream[0].step
    assert art_orig["confidence"] == upstream[0].confidence


def test_cf_replay_refuted_candidate_no_longer_wins_t_star():
    """End-to-end supersede effect: a symptom-step candidate that upstream
    ranks highest must lose t* selection once cf_replay refutes it
    (confidence -0.3), so targeted_rerun re-rolls out from the validated
    GT-step candidate instead of the refuted symptom step."""
    sb = ToySandbox()
    b, ctx = _bundle(sb.generate("q-who-when", "info_withholding"), env=sb)
    t = b.trajectory
    gt = t.meta["injected_fault"]
    symptom = (gt["step"] + 1) % len(t.events)
    b.put("attribute", "upstream_probe", {"hypotheses": [
        Hypothesis(agent=gt["agent"], step=symptom,
                   root_cause="symptom step (pseudo-causal)",
                   fix_suggestion="re-examine this step",
                   confidence=0.95).to_dict(),
        Hypothesis(agent=gt["agent"], step=gt["step"],
                   root_cause="withholds the retrieved information",
                   fix_suggestion="faithfully report the retrieved documents "
                                  "and avoid info_withholding",
                   confidence=0.9).to_dict(),
    ]})
    CounterfactualReplayAttributor().run_one(b, ctx)
    verdicts = {v["candidate"]["step"]: v["verdict"]
                for v in b.get("attribute", "counterfactual_replay")["verdicts"]}
    assert verdicts[symptom] == "refuted"
    assert verdicts[gt["step"]] == "validated"
    create("recover", "targeted_rerun").run_one(b, ctx)
    art = b.get("recover", "targeted_rerun")
    assert art["t_star"] == gt["step"]
    assert art["recovered"]


# ------------------------------------------------------------------ dover --


def test_dover_recovers_six_of_six_with_gt_mistake_steps():
    sb = ToySandbox()
    for kind in FAULTS:
        b, ctx = _bundle(sb.generate("q-who-when", kind), env=sb)
        DoVerRecoverer().run_one(b, ctx)
        art = b.get("recover", "dover")
        gt = b.trajectory.meta["injected_fault"]
        a = art["attempts"][0]
        assert a["mistake"]["step"] == gt["step"], (
            f"{kind}: mistake {a['mistake']['step']} != GT {gt['step']}"
        )
        assert a["verdict"] == "Validated"
        assert art["recovered"]
        # x3 replays per intervention, but only the representative (the last
        # repeat) enters bundle.reruns: report.n_reruns counts recovery
        # attempts, not deterministic repeats -- all repeats stay recorded in
        # the attempt (review 2026-08-27 P2)
        assert len(b.reruns) == 1
        assert len(a["replay_trace_ids"]) == 3
        assert b.reruns[-1].trace_id == a["replay_trace_ids"][-1]
        assert b.reruns[0].meta["replay_mode"] == "message_intervention"
        assert art["milestones"] and len(art["milestones"]) == 3
        # in-place message replacement semantics (vs targeted_rerun's appended
        # feedback): intervention_applied is recorded faithfully -- True only
        # when the intervened step is a message event (LLM_CALL/HANDOFF/
        # AGENT_MESSAGE); the six faults' GT steps split across both kinds
        step0 = a["mistake"]["step"]
        is_message = b.reruns[0].events[step0].kind in (
            "LLM_CALL", "HANDOFF", "AGENT_MESSAGE"
        )
        assert b.reruns[0].meta["intervention_applied"] == is_message, (
            f"{kind}: intervention_applied must match the intervened step's kind"
        )
        assert b.reruns[0].meta["intervention_applied_note"]


def test_dover_segmenter_and_trial_structure():
    sb = ToySandbox()
    b, ctx = _bundle(sb.generate("q-trajaudit", None))
    # success trajectory: skipped explicitly, and recovered stays False --
    # nothing was broken, so nothing was recovered (no inflated recovery
    # counts from trajectories that never needed recovery)
    DoVerRecoverer().run_one(b, ctx)
    art = b.get("recover", "dover")
    assert art["status"] == "skipped_success"
    assert art["recovered"] is False

    b2, ctx2 = _bundle(sb.generate("q-trajaudit", "step_repetition"), env=sb)
    DoVerRecoverer().run_one(b2, ctx2)
    art = b2.get("recover", "dover")
    # sandbox trajectories are single-plan -> a single trial; split point = the plan step
    assert len(art["attempts"]) == 1


def test_dover_proposer_input_sliced_per_trial():
    """DoVer Fig 6 input structure: each failure-proposer call sees ONLY its
    trial's log (sliced at the trial's exec_range, original [index] numbering
    preserved) plus previous_trial_summary (empty list for the first trial,
    the earlier trials' records afterwards) -- never the full session view.
    Regression for the unsliced-proposer hole (T=1 sandbox trajectories used
    to pass the whole view through)."""
    sb = ToySandbox()
    b0, ctx0 = _bundle(sb.generate("q-who-when", "info_withholding"), env=sb)
    t = b0.trajectory
    gt = t.meta["injected_fault"]

    # synthesize a 3-trial session: splice two re-plan LLM_CALLs into the
    # canonical stream (before the read step and before the failing handoff),
    # renumbering indexes
    def _plan(text: str) -> TraceEvent:
        return TraceEvent(
            id="plan-x", ts=0.0, kind="LLM_CALL", agent="planner", action=None,
            payload={"content": text}, refs=[], phase="plan", parent=None, index=0,
        )

    evs = list(t.events)
    read_idx = next(i for i, e in enumerate(evs) if e.action == "read_doc")
    evs.insert(read_idx, _plan("plan: read the located gold document next"))
    evs.insert(gt["step"] + 1, _plan("plan: report the found answer with a citation"))
    for i, e in enumerate(evs):
        e.index, e.id = i, f"e{i:03d}"
    syn = Trajectory(
        trace_id=t.trace_id + "-3plan", task=t.task, events=evs,
        outcome=t.outcome, meta=dict(t.meta), raw=dict(t.raw or {}),
    )
    b = TrajectoryBundle(syn)
    ctx = RunContext(llm=FakeLLMClient(), env=sb)
    DoVerRecoverer().run_one(b, ctx)

    prop = [c for c in ctx.llm.calls if c["tag"] == "dover_proposer"]
    assert len(prop) == 3
    p0, p1, p2 = (c["messages"][1]["content"] for c in prop)
    # every proposer prompt declares the Fig 6 field; the first trial has none
    for p in (p0, p1, p2):
        assert "previous_trial_summary" in p
        low = p.lower()
        for word in (*FAULTS.keys(), *EXTRA_FAULTS.keys(),
                     "injected_fault", "gold_doc"):
            assert word.lower() not in low, f"proposer prompt leaks {word!r}"
    assert "previous_trial_summary: []" in p0
    assert '"trial_index": 0' in p1                # trial 1 sees trial 0's record
    assert '"trial_index": 0' in p2 and '"trial_index": 1' in p2
    # trial 2 (the failing trial): only its own exec_range events are visible
    assert "[10] HANDOFF" in p2                    # original numbering preserved
    assert "no relevant documents found" in p2     # its symptom event
    assert "plan: report the found answer" in p2   # its own planning step
    assert "plan: search" not in p2                # trial 0's plan not leaked
    assert "search results for" not in p2          # trial 0's search evidence
    assert "read the located gold document next" not in p2  # trial 1's plan
    assert "WhoDunitAndWhen" not in p2             # trial 1's read evidence
    # earlier trials must not see later events either (no reverse leak)
    assert "no relevant documents found" not in p0
    assert "no relevant documents found" not in p1
    assert "read the located gold document next" not in p0


def test_dover_classify_handler_rules():
    """Deterministic unit verification of the four decision rules (DoVer Sec 4.2)."""
    def classify(runs: str, removed: bool) -> str:
        msg = [{"role": "user", "content": (
            f"3 replay outcomes: {runs}; fault removed by edit: {removed}\nMilestones: x"
        )}]
        import json as _json
        return _json.loads(pseudo_judge_handler("dover_classify", msg))["label"]

    assert classify("[True, True, True]", True) == "Validated"
    assert classify("[True, True, False]", True) == "Validated"
    assert classify("[True, False, False]", True) == "Partially"
    assert classify("[False, False, False]", False) == "Refuted"


def test_dover_classify_handler_majority_vote_follows_n():
    """The Validated rule is a strict majority 2*n_ok > n_repeats (paper-fixed
    n=3 keeps the >=2/3 behavior); configurable repeat counts must be
    honored: 2/4 successes is NOT a majority, 1/1 is."""
    def classify(runs: str) -> str:
        msg = [{"role": "user", "content": (
            f"{len(runs.split(','))} replay outcomes: {runs}; "
            f"fault removed by edit: True\nMilestones: x"
        )}]
        import json as _json
        return _json.loads(pseudo_judge_handler("dover_classify", msg))["label"]

    assert classify("[True, True, False, False]") == "Partially"  # 2/4: tie, not majority
    assert classify("[True, True, True, False]") == "Validated"   # 3/4: strict majority
    assert classify("[True]") == "Validated"                      # 1/1
    assert classify("[False]") == "Partially"                     # 0/1 but fault removed


def test_dover_vocab_aliases_normalize_and_reject():
    """OutcomeLabel/Intervention are vocabulary-validated (Literal + before-
    validator alias normalization, same policy as judge_eval's severity):
    common judge synonyms fold to the canonical token; anything still
    illegal is an explicit parse failure (LLMError), never a silently
    downgraded verdict."""
    from atap.llm.base import LLMError, parse_structured
    from atap.recover.dover import Intervention, OutcomeLabel

    # label aliases (case-insensitive)
    for raw, want in (
        ("Validated", "Validated"),
        ("VALIDATED", "Validated"),
        ("partially", "Partially"),
        ("partially validated", "Partially"),
        ("partially_valided", "Partially"),
        ("Partially Valid", "Partially"),
        ("refuted", "Refuted"),
        ("inconclusive", "Inconclusive"),
    ):
        got = parse_structured(
            json.dumps({"label": raw, "reason": "r"}), OutcomeLabel
        ).label
        assert got == want, raw
    with pytest.raises(LLMError):
        parse_structured('{"label": "mostly works", "reason": "r"}', OutcomeLabel)

    # category aliases (case/space/hyphen fold to the underscore token)
    for raw, want in (
        ("orchestrator_ledger", "orchestrator_ledger"),
        ("Orchestrator Instruction", "orchestrator_instruction"),
        ("subagent-instruction", "subagent_instruction"),
        ("SUBAGENT_INSTRUCTION", "subagent_instruction"),
    ):
        got = parse_structured(
            json.dumps({"category": raw, "replacement_text": "x"}), Intervention
        ).category
        assert got == want, raw
    with pytest.raises(LLMError):
        parse_structured(
            '{"category": "global_reset", "replacement_text": "x"}', Intervention
        )


def test_dover_prompts_follow_configured_n_repeats():
    """The classify prompt must state the actual n_repeats and the majority
    threshold derived from it (> n_repeats//2), never a hardcoded "3
    repeats / >=2/3"; n_repeats < 1 is rejected (empty replay list would
    otherwise IndexError on runs[0])."""
    from atap.recover import dover as dover_mod

    sb = ToySandbox()
    b, ctx = _bundle(sb.generate("q-who-when", "info_withholding"), env=sb)
    DoVerRecoverer(n_repeats=1).run_one(b, ctx)
    art = b.get("recover", "dover")
    assert art["recovered"] is True
    assert len(b.reruns) == 1
    classify_calls = [c for c in ctx.llm.calls if c["tag"] == "dover_classify"]
    assert len(classify_calls) == 1
    system = classify_calls[0]["messages"][0]["content"]
    user = classify_calls[0]["messages"][1]["content"]
    assert "1 repeats" in system and "more than 0 of the 1 repeats" in system
    assert "1 replay outcomes: [True]" in user
    assert "3 replay outcomes" not in user

    # default n=3 renders the paper's >=2/3 threshold
    rendered = dover_mod._CLASSIFY_SYSTEM.format(n_repeats=3, maj=1)
    assert "3 repeats" in rendered and "more than 1 of the 3 repeats" in rendered
    assert "milestone progress" not in rendered      # input carries a milestone list, not measured progress
    assert "no measurable progress toward the milestones" in rendered  # paper-neutral Refuted wording

    # n_repeats < 1: explicit rejection
    b2, ctx2 = _bundle(sb.generate("q-who-when", "info_withholding"), env=sb)
    with pytest.raises(ValueError, match="n_repeats"):
        DoVerRecoverer(n_repeats=0).run_one(b2, ctx2)


def test_dover_requires_env():
    b, ctx = _bundle(ToySandbox().generate("q-who-when", "info_withholding"))
    DoVerRecoverer().run_one(b, ctx)
    art = b.get("recover", "dover")
    assert art["status"] == "no_replay_environment"


# ------------------------------------------------------------- anti-leak --


def test_stage4c_prompts_no_gt_leakage():
    from atap.attribute import counterfactual_replay as cf_mod
    from atap.recover import dover as dover_mod

    forbidden = (*FAULTS.keys(), "injected_fault", "ground_truth", "gold_doc",
                 "ground truth")
    prompts = [
        cf_mod._SYSTEM, dover_mod._SEGMENT_SYSTEM, dover_mod._PROPOSER_SYSTEM,
        dover_mod._INTERVENE_SYSTEM, dover_mod._MILESTONE_SYSTEM,
        dover_mod._CLASSIFY_SYSTEM,
    ]
    for p in prompts:
        low = p.lower()
        for word in forbidden:
            assert word.lower() not in low, f"prompt leaks {word!r}: {p[:60]}..."


def test_stage4c_runtime_prompts_no_fault_leakage():
    """Full runtime scan: **every** prompt actually produced by dover /
    counterfactual_replay (including intervention text echoed in user
    messages) must contain no fault names / GT fields -- the pseudo-judge's
    edit text carries fault names as an environment-side middleware channel,
    which must be stripped before echoing."""
    sb = ToySandbox()
    b, ctx = _bundle(sb.generate("q-who-when", "info_withholding"), env=sb)
    # seed an attribution artifact first, for counterfactual_replay to take candidates from
    b.put("attribute", "seed", {
        "hypotheses": [Hypothesis(
            agent="searcher", step=8, root_cause="withholding",
            root_cause_code="FM-2.2", responsible_side="agent",
            evidence=[], fix_suggestion="", confidence=0.6,
        )],
    })
    CounterfactualReplayAttributor().run_one(b, ctx)
    DoVerRecoverer().run_one(b, ctx)
    forbidden = (*FAULTS.keys(), *EXTRA_FAULTS.keys(), "injected_fault",
                 "ground_truth", "gold_doc")
    assert ctx.llm.calls, "FakeLLM recorded no calls"
    for call in ctx.llm.calls:
        blob = "".join(
            m.get("content", "") for m in call["messages"]
        ).lower()
        for word in forbidden:
            assert word.lower() not in blob, (
                f"runtime prompt leaks {word!r} (tag={call['tag']}): "
                f"{blob[:120]}..."
            )


def test_redact_fault_names_covers_separator_variants():
    """_redact_fault_names must strip every separator variant of each fault
    name (underscore / hyphen / space, any case): the environment middleware
    accepts space variants ("info withholding"), so such an edit is
    environment-consumable and must not be echoed verbatim into judge-visible
    prompts (previously only the underscore form was stripped)."""
    from atap.recover.dover import _redact_fault_names

    for name in (*FAULTS.keys(), *EXTRA_FAULTS.keys()):
        variants = (
            name,
            name.replace("_", " "),
            name.replace("_", "-"),
            name.replace("_", " ").upper(),
        )
        for variant in variants:
            redacted = _redact_fault_names(f"please avoid {variant} here")
            assert variant.lower() not in redacted.lower(), (
                f"separator variant not stripped: {variant!r}"
            )
            assert redacted == "please avoid <edited> here", variant
