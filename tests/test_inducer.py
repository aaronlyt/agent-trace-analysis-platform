"""Failure clustering + residual vocabulary expansion (AgentDebugX inducer)
tests.

Scenario: the agent_deadlock fault (no counterpart among the MAST 14) x3
trajectories + the regular six faults as contrast, verifying the full loop
of novel channel -> cluster nomination -> human acceptance -> the new code
becomes usable.
"""

from __future__ import annotations

import json

from atap.classify.inducer import InducerClassifier
from atap.classify.mast_judge import MastJudgeClassifier
from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.llm import FakeLLMClient
from atap.sandbox import ToySandbox


def _deadlock_bundles(n=3):
    sb = ToySandbox()
    traces = [sb.generate("q-trajaudit", "agent_deadlock",
                          trace_id=f"dead-{i}") for i in range(n)]
    traces.append(sb.generate("q-trajaudit", None, trace_id="dead-ok"))
    traces.append(sb.generate("q-trajaudit", "step_repetition",
                              trace_id="dead-rep"))
    bundles = [TrajectoryBundle(t) for t in traces]
    ctx = RunContext(llm=FakeLLMClient())
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    return bundles, ctx


def test_novel_channel_emits_symptom_for_deadlock():
    bundles, ctx = _deadlock_bundles()
    MastJudgeClassifier(allow_novel=True).run_corpus(bundles, ctx)
    art = bundles[0].get("classify", "mast_judge")
    labels = art["labels"]
    assert labels and labels[0]["code"] == "novel"
    assert labels[0]["symptom"], "a novel label must carry a symptom phrase"
    # known faults take the regular code (step_repetition), emitting no novel
    rep_art = next(b.get("classify", "mast_judge")
                   for b in bundles if b.trace_id == "dead-rep")
    assert rep_art["labels"][0]["code"] == "FM-1.3"
    # success trajectories get no labels
    ok = next(b for b in bundles if b.trace_id == "dead-ok")
    assert ok.get("classify", "mast_judge")["labels"] == []


def test_default_mast_judge_unchanged_without_novel_flag():
    bundles, ctx = _deadlock_bundles(n=1)
    MastJudgeClassifier().run_corpus(bundles, ctx)
    art = bundles[0].get("classify", "mast_judge")
    # default closed vocabulary: deadlock has no known code -> the
    # pseudo-judge falls back to FM-2.6 (old behavior unchanged)
    assert art["labels"] and art["labels"][0]["code"] != "novel"
    assert "novel_channel" in art and art["novel_channel"] is False


def test_inducer_proposes_exactly_one_cluster():
    bundles, ctx = _deadlock_bundles()
    MastJudgeClassifier(allow_novel=True).run_corpus(bundles, ctx)
    InducerClassifier().run_corpus(bundles, ctx)
    art = bundles[0].get("classify", "inducer")
    assert art["status"] == "ok"
    assert len(art["proposals"]) == 1
    prop = art["proposals"][0]
    assert prop["mode_id"] == "NM-1"
    assert prop["status"] == "proposed"          # never auto-effective
    assert prop["support"] == 3                   # exactly the 3 deadlock trajectories
    assert prop["evidence_trace_ids"] == ["dead-0", "dead-1", "dead-2"]
    # the name comes from symptom content words (no fault-type words, anti-leak)
    assert "deadlock" not in prop["name"].lower()
    assert prop["name"]
    assert art["stats"]["n_candidates"] == 3


def test_inducer_proposal_name_pinned_value():
    """Regression (nondeterministic tie-break): the deadlock corpus has TWO
    equal-count recurring inter-agent messages (the clarify handoff and the
    re-answer, 3 each); the novel-symptom pick must be deterministic, so the
    proposal name equals this pinned value in every process, independent of
    hash-seed randomization (previously max() over a set let the name drift
    across processes)."""
    bundles, ctx = _deadlock_bundles()
    MastJudgeClassifier(allow_novel=True).run_corpus(bundles, ctx)
    InducerClassifier().run_corpus(bundles, ctx)
    art = bundles[0].get("classify", "inducer")
    # symptom = "please clarify which document I should prioritize before
    # searching" (alphabetically first among the count-tied pair) -> content
    # tokens clarify/document/prioritize/searching -> top-3 by frequency
    assert art["proposals"][0]["name"] == "clarify document prioritize"


def test_novel_symptom_tiebreak_is_deterministic():
    """Unit: with two equal-count recurring messages the picked symptom is
    the lexicographically smallest text (sorted by (-count, text)), not the
    first-seen one and not whatever set iteration order yields."""
    from atap.llm.pseudo_judge import _novel_symptom, _parse_block

    block = "\n".join([
        "[0] TASK_START env :: q",
        "[1] AGENT_MESSAGE planner :: zeta message repeated twice",
        "[2] HANDOFF searcher {'to': 'planner'} :: alpha message repeated twice",
        "[3] AGENT_MESSAGE planner :: zeta message repeated twice",
        "[4] HANDOFF searcher {'to': 'planner'} :: alpha message repeated twice",
        "[5] TOOL_CALL searcher search {'query': 'x'}",
    ])
    step, agent, text = _novel_symptom(_parse_block(block))
    # the lexicographically smaller text is seen LATER in the trace yet wins
    assert text == "alpha message repeated twice"
    assert (step, agent) == (2, "searcher")


def test_inducer_no_proposals_on_fully_labeled_corpus():
    """All six faults are labelable -> no residual -> zero proposals
    (support gating + no novel candidates)."""
    sb = ToySandbox()
    traces = [sb.generate("q-drift", k) for k in
              ("step_repetition", "info_withholding", "ungrounded_citation")]
    bundles = [TrajectoryBundle(t) for t in traces]
    ctx = RunContext(llm=FakeLLMClient())
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    MastJudgeClassifier(allow_novel=True).run_corpus(bundles, ctx)
    InducerClassifier().run_corpus(bundles, ctx)
    art = bundles[0].get("classify", "inducer")
    assert art["proposals"] == []
    assert art["stats"]["n_candidates"] == 0


def test_novel_candidate_agent_resolved_from_trajectory():
    """MastLabel carries no agent field, so a novel candidate's agent is
    resolved from the R0 event at the label's evidence step (previously a
    constant "unknown"); absent/out-of-range steps still yield "unknown"."""
    from atap.classify.inducer import _agent_at

    bundles, ctx = _deadlock_bundles()
    MastJudgeClassifier(allow_novel=True).run_corpus(bundles, ctx)
    b = bundles[0]
    step = b.get("classify", "mast_judge")["labels"][0]["step"]
    events = b.trajectory.events
    assert 0 <= step < len(events)          # novel labels carry a usable step
    assert _agent_at(b, step) == events[step].agent
    assert _agent_at(b, step) != "unknown"
    assert _agent_at(b, None) == "unknown"          # no evidence step
    assert _agent_at(b, len(events) + 5) == "unknown"  # out of range
    assert _agent_at(b, -1) == "unknown"            # negative index not trusted


def test_inducer_requires_mast_judge_artifact():
    bundles, ctx = _deadlock_bundles(n=1)
    import pytest

    with pytest.raises(ValueError, match="mast_judge"):
        InducerClassifier().run_corpus(bundles, ctx)


def test_acceptance_roundtrip_via_extra_modes(tmp_path):
    """Proposal -> acceptance lands in a file -> after mast_judge loads it,
    the new code is labelable (human-gate closed loop)."""
    from atap.cli import main

    bundles, ctx = _deadlock_bundles()
    MastJudgeClassifier(allow_novel=True).run_corpus(bundles, ctx)
    InducerClassifier().run_corpus(bundles, ctx)

    # persist the artifact into the run dir (the CLI accept reads it from here)
    art_dir = tmp_path / "artifacts" / "dead-0"
    art_dir.mkdir(parents=True)
    (art_dir / "classify__inducer.json").write_text(
        json.dumps(bundles[0].get("classify", "inducer"), ensure_ascii=False),
        encoding="utf-8",
    )
    modes_file = tmp_path / "modes.json"
    rc = main(["taxonomy", "accept", "--run-dir", str(tmp_path),
               "--id", "NM-1", "--out", str(modes_file)])
    assert rc == 0
    modes = json.loads(modes_file.read_text(encoding="utf-8"))["modes"]
    assert len(modes) == 1 and modes[0]["code"] == "NM-1"
    assert modes[0]["name"]   # from symptom content words

    # idempotent: repeated acceptance skips
    rc = main(["taxonomy", "accept", "--run-dir", str(tmp_path),
               "--id", "NM-1", "--out", str(modes_file)])
    assert rc == 0
    assert len(json.loads(modes_file.read_text(encoding="utf-8"))["modes"]) == 1

    # after loading the extended modes, the pseudo-judge labels new
    # trajectories with NM-1
    bundles2, ctx2 = _deadlock_bundles(n=1)
    MastJudgeClassifier(
        allow_novel=True, extra_modes_file=str(modes_file)
    ).run_corpus(bundles2, ctx2)
    art = bundles2[0].get("classify", "mast_judge")
    assert art["labels"][0]["code"] == "NM-1"
    assert art["extra_modes"] == ["NM-1"]


def test_extra_modes_conflict_rejected(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(
        {"modes": [{"code": "FM-1.3", "name": "clash", "definition": ""}]}),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="conflict"):
        MastJudgeClassifier(extra_modes_file=str(f))
