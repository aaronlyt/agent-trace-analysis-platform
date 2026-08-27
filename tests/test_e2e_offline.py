"""Stage two acceptance: offline full-chain e2e (FakeLLM deterministic judge).

For each of the six injected faults, assert: attribution step/agent/MAST
aligned with the ground truth, targeted rerun recovers successfully,
closed-loop verification improves -- a reproducible miniature of the
literature pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atap.core.config import config_from_dict
from atap.runtime import run_config
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def offline_cfg(traces_path: str, closed_loop: bool = True) -> dict:
    return {
        "run_name": "e2e-offline",
        "seed": 7,
        "source": {"type": "jsonl", "path": traces_path},
        "llm": {"type": "fake"},
        "sandbox": {"type": "toy"},
        "closed_loop": closed_loop,
        "stages": {
            "represent": ["canonical_events", "ssf"],
            "analyze": ["judge_eval"],
            "classify": ["mast_judge"],
            "attribute": ["all_at_once"],
            "recover": [{"name": "targeted_rerun", "params": {"max_rounds": 5}}],
        },
    }


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    traces = ToySandbox().generate_population(seed=7)
    path = out / "traces.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    bundles, reports = run_config(config_from_dict(offline_cfg(str(path))), out)
    return bundles, reports, out


def test_population_shape(e2e):
    bundles, _, _ = e2e
    faults = [b for b in bundles if not b.succeeded]
    assert len(bundles) == 7 and len(faults) == 6
    assert {b.trajectory.meta["injected_fault"]["kind"] for b in faults} == set(FAULTS)


@pytest.mark.parametrize("kind", list(FAULTS))
def test_attribution_matches_ground_truth(e2e, kind):
    bundles, _, _ = e2e
    b = next(x for x in bundles if x.trajectory.meta.get("injected_fault", {}).get("kind") == kind)
    gt = b.trajectory.meta["injected_fault"]
    hyps = b.hypotheses()
    assert hyps, f"{kind}: no attribution output"
    top = max(hyps, key=lambda h: h.confidence)
    assert top.step == gt["step"], f"{kind}: attributed step {top.step} != gt {gt['step']}"
    assert top.agent == gt["agent"]
    assert top.root_cause_code == gt["mast_code"]


@pytest.mark.parametrize("kind", list(FAULTS))
def test_classify_and_recover(e2e, kind):
    bundles, _, _ = e2e
    b = next(x for x in bundles if x.trajectory.meta.get("injected_fault", {}).get("kind") == kind)
    gt = b.trajectory.meta["injected_fault"]
    labels = b.get("classify", "mast_judge")["labels"]
    assert labels and labels[0]["code"] == gt["mast_code"]

    rec = b.get("recover", "targeted_rerun")
    assert rec["recovered"] is True and rec["rounds"] == 1
    assert b.reruns[-1].outcome.success

    loop = b.get("recover", "closed_loop")
    assert loop["verified_improved"] is True
    # verification-round judge evidence is kept in the closed_loop artifact
    # (cross-audit: judge vs outcome contradictions stay on record)
    assert loop["verify"]["outcome_success"] is True
    assert loop["verify"]["judge_available"] is True
    assert isinstance(loop["verify"]["judge"]["score"], (int, float))
    assert loop["verify"]["judge"]["summary"]


def test_success_traces_scored_high_not_attributed(e2e):
    bundles, _, _ = e2e
    for b in bundles:
        if b.succeeded:
            assert b.get("analyze", "judge_eval")["score"] >= 8
            assert not b.hypotheses()
            assert b.get("classify", "mast_judge")["labels"] == []


def test_closed_loop_second_round_all_pass(e2e):
    bundles, reports, _ = e2e
    assert len(reports) == 2  # initial round + closed-loop verification round
    assert reports[0].n_failures == 6
    assert reports[1].n_failures == 0  # all rerun trajectories pass the full-pipeline verification
    assert reports[1].n_attributed == 0
    # origins without a rerun carry no verification evidence (verify is None)
    ok = next(b for b in bundles if b.succeeded)
    loop = ok.get("recover", "closed_loop")
    assert loop["rerun_trace_id"] is None and loop["verify"] is None


def test_artifacts_persisted(e2e):
    _, _, out = e2e
    arts = out / "artifacts"
    assert (arts / "report.json").exists()
    files = list(arts.rglob("*.json"))
    # at least 5 stage artifacts per trajectory x 7 traces + report
    assert len(files) >= 7 * 5 + 1
    report = json.loads((arts / "report.json").read_text(encoding="utf-8"))
    assert report["n_reruns"] == 6 and report["n_rerun_success"] == 6
    # per-round breakdown: n_traces is the round-inclusive total (7 + 7),
    # the by-round lists expose the initial vs verification round split
    assert report["n_traces_by_round"] == [7, 7]
    assert report["n_failures_by_round"] == [6, 0]


def test_demo_skips_failures_without_gt(tmp_path, monkeypatch, capsys):
    """A failure trajectory without injected-fault GT must be skipped by the
    demo report (same guard as compare.evaluate_against_gt), not crash it."""
    from atap.demo import run_demo
    from atap.sandbox import ToySandbox

    real = ToySandbox.generate_population

    def patched(self, seed):
        traces = real(self, seed)
        for t in traces:
            if not t.outcome.success:
                t.meta.pop("injected_fault", None)   # simulate missing GT
        return traces

    monkeypatch.setattr(ToySandbox, "generate_population", patched)
    run_demo(seed=7, out=tmp_path / "demo")   # must not raise
    out = capsys.readouterr().out
    assert "gt=None@stepNone" not in out
