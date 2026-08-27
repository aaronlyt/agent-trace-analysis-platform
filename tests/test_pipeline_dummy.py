"""Stage one acceptance: Dummy algorithms end to end -- read JSONL -> five-stage orchestration -> artifacts persisted.

Validates the pluggable mechanism itself: register 5 Dummies inside the test
(one per stage) + 1 cross-trajectory Dummy (run_corpus aggregation scope),
compose them via YAML config, run the full pipeline and verify artifact
persistence and execution order.
"""

from __future__ import annotations

from pathlib import Path

import atap  # noqa: F401  import bootstraps registration
from atap.core.base import STAGE_ORDER, StageAlgorithm
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


def json_load(p: Path):
    import json

    return json.loads(Path(p).read_text(encoding="utf-8"))
