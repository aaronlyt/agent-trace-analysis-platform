"""SBFL spectrum attribution (FAMAS 2509.13782) tests."""

from __future__ import annotations

import math

import pytest

from atap.attribute.sbfl import SBFLAttributor
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import Outcome, Trajectory
from atap.sandbox import ToySandbox


def _corpus_bundles(successes=2, task="q-trajaudit"):
    """Single-task corpus: K successes + the six faults (other tasks' traces
    are not mixed in; the spectrum groups by task_id)."""
    sb = ToySandbox()
    traces = [sb.generate(task, None, trace_id=f"{task}--ok{i}") for i in range(successes)]
    from atap.sandbox.faults import FAULTS

    traces += [sb.generate(task, k) for k in FAULTS]
    bundles = [TrajectoryBundle(t) for t in traces]
    ctx = RunContext()
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    create("represent", "action_signature").run_corpus(bundles, ctx)
    return bundles, ctx


def test_spectrum_hits_repetition_and_premature():
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    by_id = {b.trace_id: b for b in bundles}
    rep = by_id["q-trajaudit--step_repetition"]
    top = rep.hypotheses()[0]  # ranked.sort is stable: hypotheses[0] is top-1
    gt = rep.trajectory.meta["injected_fault"]
    assert (top.step, top.agent) == (gt["step"], gt["agent"])  # 5, searcher

    prem = by_id["q-trajaudit--premature_termination"]
    top_p = prem.hypotheses()[0]
    gt_p = prem.trajectory.meta["injected_fault"]
    assert (top_p.step, top_p.agent) == (gt_p["step"], gt_p["agent"])  # 1, planner


def test_prior_grade_confidence_and_role():
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    for b in bundles:
        art = b.get("attribute", "sbfl")
        if b.succeeded:
            assert art["status"] == "success_no_attribution"
            continue
        assert art["role"] == "L0_statistical_prior"
        assert art["hypotheses"][0]["confidence"] <= 0.5  # prior grade
        assert art["spectrum"]["n_runs"] == 8
        assert art["spectrum"]["n_failed"] == 6 and art["spectrum"]["n_success"] == 2


def test_known_no_signal_faults_are_honest_misses():
    """Faults whose action spectrum equals the successful trajectories
    (information withholding / ungrounded citation) leave no spectrum
    signal -- do not assert a hit, only that honest ranked hypotheses were
    produced (top-1 is not the GT)."""
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    by_id = {b.trace_id: b for b in bundles}
    info = by_id["q-trajaudit--info_withholding"]
    top = info.hypotheses()[0]  # hypotheses[0] is top-1 (stable sort)
    gt = info.trajectory.meta["injected_fault"]
    assert (top.step, top.agent) != (gt["step"], gt["agent"])  # known miss
    assert top.agent in ("planner", "searcher", "reporter")     # still a valid agent


def test_run_one_is_explicit_corpus_scope():
    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit", "step_repetition"))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    SBFLAttributor().run_one(b, ctx)
    art = b.get("attribute", "sbfl")
    assert art["status"] == "corpus_scope_required"
    assert art["hypotheses"] == []


def test_missing_r5_raises():
    bundles = [TrajectoryBundle(ToySandbox().generate("q-trajaudit", "step_repetition"))]
    ctx = RunContext()
    create("represent", "canonical_events").run_corpus(bundles, ctx)
    with pytest.raises(ValueError, match="action_signature"):
        SBFLAttributor().run_corpus(bundles, ctx)


def test_lam_validation():
    with pytest.raises(ValueError):
        SBFLAttributor(lam=1.5).run_corpus([], RunContext())


def test_second_occurrence_step_convention():
    """Signatures repeated >= 2 times within the trajectory take the second
    occurrence (the Eq.5 earliest-decisive-error convention)."""
    bundles, ctx = _corpus_bundles()
    SBFLAttributor().run_corpus(bundles, ctx)
    rep = next(b for b in bundles if "step_repetition" in b.trace_id)
    hyps = rep.hypotheses()
    # the repeated search signature (if hypothesized) must not point at the
    # first occurrence (3) but at 5
    for h in hyps:
        if "SEARCH" in h.evidence[0] and "steps=[3, 5, 7]" in h.evidence[0]:
            assert h.step == 5


# ---------------------------------------------------------------------------
# Formula-level numeric regression (Eqs. 2-7): a controlled corpus whose
# spectrum metrics are recomputed by hand.  Signatures (agent coder):
#   eta  = (coder, SEARCH, docs)   plan = (coder, PLAN, "")
#   write = (coder, WRITE, "")
# Corpus (task q-sbfl-math, lam=0.9), coder units per trajectory:
#   fail1: eta@0 plan@1            (fail)
#   fail2: eta@0 plan@1 eta@2      (fail, the attributed tau_0)
#   ok1:   eta@0 plan@1  ok2: plan@0 write@1  ok3: plan@0 write@1  ok4: plan@0
# Hand recomputation:
#   eta:  n_cf^lam = 0.9^0 + 0.9^1 = 1.9;  n_cs^lam = 0.9^0 = 1;  n_uf = 0
#         Kul2^lam = 0.5*(1.9/1.9 + 1.9/2.9) = 0.8275862
#         gamma = nc_eta/nc_agent = 3/6 = 0.5    (eta covered by 3 of the 6
#                                                  trajectories that run coder)
#         beta  = f_eta/f_agent   = 4/12 = 1/3   (4 of the 12 coder units)
#         base  = Kul2^lam*(1+beta)(1+gamma) = 1.6551724
#   tau_0 (fail2): alpha = 1 + ln(2)/ln(1/0.9) = 7.5788135
#                  S(eta) = 7.5788135 * 1.6551724 = 12.5442430
#   plan: n_cf^lam=2, n_cs^lam=4, kul=2/3, gamma=1, beta=0.5 -> base=2.0
# ---------------------------------------------------------------------------

_ETA = ("coder", "SEARCH", "docs")
_PLAN = ("coder", "PLAN", "")
_WRITE = ("coder", "WRITE", "")


def _sig(u, index):
    return {"agent": u[0], "action_class": u[1], "target": u[2], "index": index}


def _numeric_corpus():
    layout = [
        ("fail1", False, [(_ETA, 0), (_PLAN, 1)]),
        ("fail2", False, [(_ETA, 0), (_PLAN, 1), (_ETA, 2)]),
        ("ok1", True, [(_ETA, 0), (_PLAN, 1)]),
        ("ok2", True, [(_PLAN, 0), (_WRITE, 1)]),
        ("ok3", True, [(_PLAN, 0), (_WRITE, 1)]),
        ("ok4", True, [(_PLAN, 0)]),
    ]
    bundles = []
    for tid, ok, units in layout:
        t = Trajectory(
            trace_id=tid, task="q-sbfl-math", events=[],
            outcome=Outcome(success=ok), meta={"task_id": "q-sbfl-math"},
        )
        b = TrajectoryBundle(t)
        b.put("represent", "action_signature", {"signatures": [_sig(u, i) for u, i in units]})
        bundles.append(b)
    return bundles


def test_sbfl_formula_numeric_regression():
    bundles = _numeric_corpus()
    SBFLAttributor(lam=0.9).run_corpus(bundles, RunContext())
    by_id = {b.trace_id: b for b in bundles}
    art = by_id["fail2"].get("attribute", "sbfl")

    # spectrum metrics of eta (rounded to 4 dp in the artifact)
    top = art["spectrum"]["top"][0]
    assert top["signature"] == "SEARCH(docs)" and top["agent"] == "coder"
    assert top["n_cf_lambda"] == pytest.approx(1.9, abs=1e-4)
    assert top["n_cs_lambda"] == pytest.approx(1.0, abs=1e-4)
    assert top["n_uf"] == 0
    assert top["kulczynski2_lambda"] == pytest.approx(0.8275862, abs=1e-4)
    assert top["gamma"] == pytest.approx(0.5, abs=1e-4)
    assert top["beta"] == pytest.approx(1 / 3, abs=1e-4)
    # S(eta) = alpha * Kul2^lam * (1+beta)(1+gamma), alpha = 1+ln2/ln(1/0.9)
    expected_s = (
        (1 + math.log(2) / math.log(1 / 0.9))
        * 0.5 * (1.9 / 1.9 + 1.9 / 2.9)
        * (1 + 1 / 3) * (1 + 0.5)
    )
    assert expected_s == pytest.approx(12.5442430, abs=1e-5)  # guard the arithmetic itself
    assert top["score"] == pytest.approx(expected_s, abs=1e-4)

    # the ranked list is descending; hypotheses[0] is top-1 = eta, at its
    # second occurrence (index 2) per the step convention
    scores = [row["score"] for row in art["spectrum"]["top"]]
    assert scores == sorted(scores, reverse=True)
    hyps = by_id["fail2"].hypotheses()
    assert hyps[0].agent == "coder"
    assert hyps[0].step == 2
    assert hyps[0].evidence[0].startswith("signature=SEARCH(docs)")

    # tau with f_tau=1 gets alpha=1: S(eta) = base = 1.6551724 (fail1), and
    # plan outranks eta there (2.0 > 1.6552) -- alpha is what lifts eta to
    # top-1 in fail2
    art1 = by_id["fail1"].get("attribute", "sbfl")
    rows1 = {r["signature"]: r for r in art1["spectrum"]["top"]}
    assert rows1["SEARCH(docs)"]["score"] == pytest.approx(
        0.5 * (1.9 / 1.9 + 1.9 / 2.9) * (1 + 1 / 3) * (1 + 0.5), abs=1e-4
    )
    assert art1["spectrum"]["top"][0]["signature"] == "PLAN(-)"
    assert art1["spectrum"]["top"][0]["score"] == pytest.approx(
        0.5 * (2 / 2 + 2 / 6) * (1 + 0.5) * (1 + 1), abs=1e-4
    )

    # successes produce no attribution; group bookkeeping
    assert by_id["ok1"].get("attribute", "sbfl")["status"] == "success_no_attribution"
    assert art["spectrum"]["n_runs"] == 6
    assert art["spectrum"]["n_failed"] == 2 and art["spectrum"]["n_success"] == 4
    assert art["spectrum"]["lam"] == 0.9
