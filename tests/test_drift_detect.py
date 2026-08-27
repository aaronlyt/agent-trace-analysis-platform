"""Distribution drift detection tests (2511.19933 constructed scenarios for the three drift families)."""

from __future__ import annotations

import math

from atap.analyze.drift_detect import DriftDetectAnalyzer
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.sandbox import ToySandbox, env


def _run(traces, **params):
    bundles = [TrajectoryBundle(t) for t in traces]
    ctx = RunContext()
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    DriftDetectAnalyzer(**params).run_corpus(bundles, ctx)
    return bundles[0].get("analyze", "drift_detect")


def test_drift_corpus_fires_all_three_families():
    art = _run(ToySandbox().generate_drift_corpus())
    families = {a["family"] for a in art["alerts"]}
    assert families == {"version", "behavior", "data"}, art["alerts"]


def test_stable_corpus_zero_false_alarms():
    """Same-distribution stable corpus (default corpus, same meta grouping)
    raises no alerts. Note: single group -> zero pairwise comparisons, so
    this is an empty contrast, not specificity evidence; the multi-window
    test below is the real false-alarm check."""
    art = _run(ToySandbox().generate_corpus())
    assert art["alerts"] == []
    assert art["status"] == "ok"
    # single group: no bucket pairs within any contrast family
    assert len(art["groups"]) == 1
    assert art["groups"][0]["n_traces"] == 24


def test_stable_multiwindow_zero_false_alarms():
    """Two windows with identical composition: a non-vacuous false-alarm
    check (real pairwise comparisons, same distribution -> 0 PSI, 0
    mismatch, no alert)."""
    sb = ToySandbox()
    traces = []
    for w in ("w1", "w5"):
        for task in env.TASKS:
            for i in range(2):
                traces.append(sb.generate(
                    task, None, meta={"time_window": w}, trace_id=f"stable-{w}-{task}-ok{i}",
                ))
    art = _run(traces)
    assert len(art["groups"]) == 2
    assert len(art["pairwise"]) == 2          # behavior + data, one pair each
    for e in art["pairwise"]:
        assert e["max_psi"] == 0.0 and e["max_mismatch"] == 0.0, e
        assert not e["alert"]
    assert art["alerts"] == []


def test_w1_vs_w3_is_data_only():
    """w3 only changes task composition, behavior unchanged: data alerts
    while behavior does not."""
    sb = ToySandbox()
    traces = sb.generate_drift_corpus()
    art = _run(traces)
    b103 = [
        e for e in art["pairwise"]
        if e["family"] == "behavior" and {e["a"], e["b"]} == {"w1", "w3"}
    ]
    assert b103 and not b103[0]["alert"]
    d103 = [
        e for e in art["pairwise"]
        if e["family"] == "data" and {e["a"], e["b"]} == {"w1", "w3"}
    ]
    assert d103 and d103[0]["alert"]


def test_psi_has_no_epsilon_artifact():
    """Empty bins must not inflate PSI (regression: ε-smoothing made the
    version-contrast length_bin PSI 11.46, 8.67 of which was one ε-filled
    empty bin; ε=1e-6 would give 15.8). PSI now covers the shared support
    only (≈2.79 here, independent of any smoothing constant) and the
    disjoint-bin mass is reported as support_mismatch."""
    art = _run(ToySandbox().generate_drift_corpus())
    version = [e for e in art["pairwise"] if e["family"] == "version"][0]
    assert version["a"] == "scripted-1.0" and version["b"] == "scripted-2.0"
    # shared support = {len[14-17]} only: PSI = (1-1/19)·ln(19) ≈ 2.7895
    assert abs(version["psi"]["length_bin"] - 0.9474 * math.log(19)) < 1e-3
    assert version["psi"]["length_bin"] < 3.0
    # 18/19 of scripted-1.0's mass (len[10-13]) sits in bins scripted-2.0 never hits
    assert abs(version["mismatch"]["length_bin"]["a"] - 18 / 19) < 1e-3
    assert version["mismatch"]["length_bin"]["b"] == 0.0
    assert version["alert"] and "length_bin" in version["firing_features"]


def test_psi_threshold_param_and_determinism():
    sb = ToySandbox()
    art1 = _run(sb.generate_drift_corpus())
    art2 = _run(sb.generate_drift_corpus())
    assert art1["pairwise"] == art2["pairwise"]      # deterministic
    assert art1["psi_alert"] == 0.2

    relaxed = _run(sb.generate_drift_corpus(), psi_alert=50.0, mismatch_alert=2.0)
    assert relaxed["alerts"] == []                   # no alerts with relaxed thresholds


def test_min_group_size_skips_small_groups():
    """A 6-vs-2 window contrast must not be evaluated as drift evidence:
    the small side is skipped and recorded."""
    sb = ToySandbox()
    traces = [
        sb.generate(task, None, meta={"time_window": "w1"}, trace_id=f"w1-{task}")
        for task in env.TASKS for _ in range(2)
    ] + [
        sb.generate(task, None, meta={"time_window": "w2"}, trace_id=f"w2-{task}")
        for task in list(env.TASKS)[:2]
    ]
    art = _run(traces)
    assert art["skipped"], art["skipped"]
    assert all(s["group"] == "w2" and s["n_traces"] == 2 for s in art["skipped"])
    assert all("w2" not in (e["a"], e["b"]) for e in art["pairwise"])


def test_collinear_features_are_aliases_not_two_signals():
    """In the drift corpus trajectory length and success/failure are
    perfectly collinear (13 events -> success, 17 -> failure), so
    length_bin and failure produce identical comparisons; the artifact
    must report the alias and dedupe the firing features."""
    art = _run(ToySandbox().generate_drift_corpus())
    alias_sets = [set(a["features"]) for a in art["feature_aliases"]]
    assert alias_sets == [{"length_bin", "failure"}]   # task_id spans 3 tasks per bin: NOT an alias
    version = [e for e in art["pairwise"] if e["family"] == "version"][0]
    assert {"length_bin", "failure"} <= set(version["firing_features"])
    assert version["firing_features_independent"] == [
        f for f in version["firing_features"] if f != "failure"
    ]


def test_single_scope_note():
    b = TrajectoryBundle(ToySandbox().generate("q-trajaudit", None))
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    DriftDetectAnalyzer().run_one(b, ctx)
    assert b.get("analyze", "drift_detect")["status"] == "corpus_scope_required"
