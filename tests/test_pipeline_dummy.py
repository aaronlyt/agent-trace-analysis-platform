"""Stage one acceptance: Dummy algorithms end to end -- read JSONL -> five-stage orchestration -> artifacts persisted.

Validates the pluggable mechanism itself: register 5 Dummies inside the test
(one per stage) + 1 cross-trajectory Dummy (run_corpus aggregation scope),
compose them via YAML config, run the full pipeline and verify artifact
persistence and execution order. Also guards the artifact-store surface
(trace_id sanitization, non-JSON artifact summaries) and the Pipeline's
explicit ``last_reruns`` state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import atap  # noqa: F401  import bootstraps registration
from atap.core.base import STAGE_ORDER, StageAlgorithm
from atap.core.bundle import TrajectoryBundle
from atap.core.registry import register
from atap.io.jsonl_store import JSONLArtifactStore
from atap.runtime import run_config

from helpers import failure_trace_ungrounded, success_trace, write_traces_jsonl

_SEQ = {"n": 0}


def _mk(stage_name: str):
    @register
    class _Dummy(StageAlgorithm):  # deliberately inherits the base directly: proves any stage is pluggable
        stage = stage_name
        name = f"dummy_{stage_name}"
        corpus_calls = 0

        def run_one(self, bundle, ctx):
            _SEQ["n"] += 1
            bundle.put(stage_name, self.name, {"seq": _SEQ["n"], "trace": bundle.trace_id})

        def run_corpus(self, bundles, ctx):
            type(self).corpus_calls += 1
            for b in bundles:
                self.run_one(b, ctx)

    return _Dummy


DUMMIES = [_mk(s) for s in STAGE_ORDER]


def test_dummy_pipeline_e2e(tmp_path):
    traces_jsonl = write_traces_jsonl(
        tmp_path / "traces.jsonl", [success_trace(), failure_trace_ungrounded()]
    )
    cfg_dict = {
        "run_name": "dummy-e2e",
        "source": {"type": "jsonl", "path": str(traces_jsonl)},
        "stages": {s: [f"dummy_{s}"] for s in STAGE_ORDER},
    }
    from atap.core.config import config_from_dict

    cfg = config_from_dict(cfg_dict)
    out = tmp_path / "out"
    bundles, reports = run_config(cfg, out)

    # both trajectories were processed
    assert len(bundles) == 2
    assert reports[-1].n_traces == 2
    assert reports[-1].n_failures == 1
    assert reports[-1].stage_log, "orchestration log should not be empty"

    # every bundle has artifacts for all five stages, and seq strictly increases along STAGE_ORDER
    for b in bundles:
        seqs = []
        for s in STAGE_ORDER:
            art = b.get(s, f"dummy_{s}")
            assert art is not None, f"{b.trace_id} missing {s} artifact"
            seqs.append(art["seq"])
        assert seqs == sorted(seqs), "stages must execute in represent->...->recover order"

    # artifacts persisted: report.json + one file per trajectory per stage
    artifacts_dir = out / "artifacts"
    assert (artifacts_dir / "report.json").exists()
    for b in bundles:
        for s in STAGE_ORDER:
            assert (artifacts_dir / b.trace_id / f"{s}__dummy_{s}.json").exists()


def test_jsonl_source_reads_traces(tmp_path):
    from atap.io.jsonl_store import JSONLTraceSource

    p = write_traces_jsonl(tmp_path / "t.jsonl", [success_trace("s1")])
    loaded = JSONLTraceSource(p).load()
    assert [t.trace_id for t in loaded] == ["s1"]
    assert loaded[0].events[0].kind == "TASK_START"


def test_artifact_store_roundtrip(tmp_path):
    store = JSONLArtifactStore(tmp_path / "arts")
    store.save_artifact("t1", "analyze", "judge_eval", {"score": 3.2})
    store.save_report("r.json", {"ok": True})
    assert json_load(tmp_path / "arts" / "t1" / "analyze__judge_eval.json") == {"score": 3.2}
    assert json_load(tmp_path / "arts" / "r.json") == {"ok": True}


def test_artifact_store_rejects_path_like_trace_id(tmp_path):
    """A trace_id becomes a directory name: path components ('/' / '\\' /
    '..') must be rejected up front, not silently split into directories."""
    store = JSONLArtifactStore(tmp_path / "arts")
    for bad in ("a/b", "a\\b", "..", ".", "", "a/../b"):
        with pytest.raises(ValueError, match="single path component"):
            store.save_artifact(bad, "analyze", "x", {"ok": True})
    assert not (tmp_path / "arts" / "a").exists()   # nothing leaked out


def test_bundle_summary_tolerates_non_json_artifacts():
    """summary() must not blow up on artifacts json.dumps cannot serialize
    (default=str fallback, matching the artifact-store write path)."""
    b = TrajectoryBundle(success_trace())
    b.put("analyze", "weird", {"s": {1, 2, 3}})   # a set: not JSON-serializable
    text = b.summary()
    assert "artifact analyze/weird" in text


def test_pipeline_last_reruns_initialized():
    from atap.core.pipeline import Pipeline

    assert Pipeline([]).last_reruns == []   # explicit state, no getattr


def test_pipeline_isolates_algorithm_crash_and_flushes_earlier_stages(tmp_path):
    """Per-algorithm error isolation + incremental persistence (review
    2026-08-27 P1): an algorithm crashing mid-corpus (historically e.g.
    binary_search raising LLMError when the judge answered neither upper
    nor lower) must not abort the run -- earlier stages' artifacts survive
    on disk, the failed algorithm leaves an error artifact, later stages
    still execute, and the report counts the error."""
    from atap.core.context import RunContext
    from atap.core.pipeline import Pipeline, PipelineReport

    class _Good(StageAlgorithm):
        stage = "represent"
        name = "good"

        def run_one(self, bundle, ctx):
            bundle.put("represent", "good", {"trace": bundle.trace_id})

    class _Boom(StageAlgorithm):
        stage = "attribute"
        name = "boom"
        fired = False

        def run_one(self, bundle, ctx):
            if _Boom.fired:
                raise RuntimeError("LLMError: judge answered neither upper nor lower")
            _Boom.fired = True   # first bundle succeeds, second crashes

        def run_corpus(self, bundles, ctx):
            for b in bundles:
                self.run_one(b, ctx)

    class _After(StageAlgorithm):
        stage = "recover"
        name = "after"
        ran_on = []

        def run_one(self, bundle, ctx):
            _After.ran_on.append(bundle.trace_id)
            bundle.put("recover", "after", {"ok": True})

    _Boom.fired = False   # reset class-level state (test may rerun in-session)
    _After.ran_on = []
    traces = [success_trace("t0"), success_trace("t1")]
    store = JSONLArtifactStore(tmp_path / "arts")
    ctx = RunContext(run_dir=str(tmp_path), llm=None, store=store)
    pipe = Pipeline([_Good(), _Boom(), _After()])
    bundles, report = pipe.run(traces, ctx)

    # the run completed; the crash was recorded, not propagated
    assert isinstance(report, PipelineReport)
    assert report.n_errors == 1
    assert any("attribute/boom -> FAILED" in line for line in report.stage_log)
    # later stages still executed on every bundle
    assert sorted(_After.ran_on) == ["t0", "t1"]
    # the failed algorithm left an explicit error artifact (no silent gap)
    for b in bundles:
        assert b.get("attribute", "boom") == {
            "status": "error",
            "error": "RuntimeError: LLMError: judge answered neither upper nor lower",
            "isolated": True,
        }
    # earlier + later stages' artifacts were already persisted (incremental
    # flush), not only at the end of the run
    for tid in ("t0", "t1"):
        assert (tmp_path / "arts" / tid / "represent__good.json").exists()
        assert (tmp_path / "arts" / tid / "attribute__boom.json").exists()
        assert (tmp_path / "arts" / tid / "recover__after.json").exists()
    # persisted report carries the error count
    store.save_report("report.json", report.to_dict())
    assert json_load(tmp_path / "arts" / "report.json")["n_errors"] == 1


def test_cli_run_exit_code_reflects_isolated_errors(tmp_path, monkeypatch, capsys):
    """Isolated algorithm failures must surface in the CLI exit status
    (independent-verify follow-up): before isolation a mid-run crash failed
    the whole command; after isolation, silently exiting 0 on n_errors > 0
    would regress that contract. The stdout summary also shows errors=N."""
    import atap.runtime
    from atap.cli import main
    from atap.core.pipeline import PipelineReport

    def fake_run_config(cfg, run_dir, **kwargs):
        reports = [PipelineReport(run_name=cfg.run_name, n_traces=3, n_errors=2)]
        return [], reports

    monkeypatch.setattr(atap.runtime, "run_config", fake_run_config)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(
        {"stages": {"represent": ["canonical_events"]}}), encoding="utf-8")
    rc = main(["run", "--config", str(cfg), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert rc == 1, "isolated failures must fail the command (exit 1)"
    assert "errors=2" in out and "algorithm failure(s) were isolated" in out


def test_cli_run_prints_actual_artifacts_dir(tmp_path, capsys):
    """The run command prints the artifact location resolved from the actual
    store config (a redirected store.dir must be reflected, not the
    hard-coded <out>/artifacts guess)."""
    from atap.cli import main

    traces_jsonl = write_traces_jsonl(tmp_path / "traces.jsonl", [success_trace()])
    base = {
        "run_name": "cli-run",
        "source": {"type": "jsonl", "path": str(traces_jsonl)},
        "stages": {"represent": ["canonical_events"]},
    }
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(base), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["run", "--config", str(cfg), "--out", str(out)]) == 0
    assert f"artifacts -> {out / 'artifacts'}" in capsys.readouterr().out

    redirected = dict(base, store={"type": "jsonl", "dir": str(tmp_path / "elsewhere")})
    cfg2 = tmp_path / "cfg2.json"
    cfg2.write_text(json.dumps(redirected), encoding="utf-8")
    out2 = tmp_path / "out2"
    assert main(["run", "--config", str(cfg2), "--out", str(out2)]) == 0
    assert f"artifacts -> {tmp_path / 'elsewhere'}" in capsys.readouterr().out


def json_load(p: Path):
    import json

    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_judge_evidence_rejects_error_artifacts():
    """closed_loop's judge_available must not count an error-isolated
    judge_eval artifact as "the judge ran" (independent verify agent B, P1):
    Pipeline.run leaves {"status": "error", ...} in place of the artifact --
    a dict, but carrying no verdict. judge_available=True with score=null
    would mislead the judge-vs-outcome cross-audit this field exists for."""
    from atap.core.pipeline import _judge_evidence

    # real verdict artifact -> evidence extracted, available
    judge, ok = _judge_evidence({"score": 8.5, "summary": "succeeded"})
    assert ok is True and judge == {"score": 8.5, "summary": "succeeded"}
    # error-isolated artifact -> NOT available, no judge payload
    judge, ok = _judge_evidence(
        {"status": "error", "error": "LLMError: ...", "isolated": True})
    assert ok is False and judge is None
    # absent artifact / non-dict -> not available
    assert _judge_evidence(None) == (None, False)
    assert _judge_evidence([1, 2]) == (None, False)
