"""算法对比跑法（atap compare）与频谱语料生成测试。"""

from __future__ import annotations

import json

import pytest

from atap.compare import CountingLLM, evaluate_against_gt, run_compare
from atap.core.bundle import TrajectoryBundle
from atap.core.registry import create
from atap.core.context import RunContext
from atap.sandbox import ToySandbox


def _write_cfg(path, name, attributor, recover="targeted_rerun"):
    stages = {
        "represent": ["canonical_events", "ssf", "action_signature"],
        "analyze": ["judge_eval"],
        "classify": ["mast_judge"],
        "attribute": [attributor],
        "recover": [recover],
    }
    path.write_text(
        json.dumps({
            "run_name": name,
            "source": {"type": "jsonl", "path": "ignored"},
            "llm": {"type": "fake"},
            "sandbox": {"type": "toy"},
            "closed_loop": False,
            "stages": stages,
        }),
        encoding="utf-8",
    )
    return str(path)


def test_generate_corpus_shape():
    traces = ToySandbox().generate_corpus(successes_per_task=2)
    assert len(traces) == 3 * (2 + 6)  # 3 任务 ×（2 成功 + 6 故障）
    per_task = {}
    for t in traces:
        task = t.meta["task_id"]
        per_task.setdefault(task, []).append(t.outcome.success)
    for task, oks in per_task.items():
        assert sum(oks) == 2 and len(oks) == 8


def test_run_compare_two_configs(tmp_path):
    from tests.helpers import write_traces_jsonl

    sb = ToySandbox()
    traces = sb.generate_corpus(successes_per_task=1)
    src = write_traces_jsonl(tmp_path / "traces.jsonl", traces)
    a = _write_cfg(tmp_path / "a.json", "cmp-all-at-once", "all_at_once",
                   recover="targeted_rerun")
    b = _write_cfg(tmp_path / "b.json", "cmp-binary", "binary_search",
                   recover="feedback_injection")
    comparison = run_compare([a, b], tmp_path / "cmp", traces=src)
    assert comparison["n_configs"] == 2
    rows = {r["run_name"]: r for r in comparison["rows"]}
    assert set(rows) == {"cmp-all-at-once", "cmp-binary"}
    for r in rows.values():
        assert r["n_failed"] == 18
        assert "llm_calls_by_tag" in r and r["llm_calls"] > 0
        assert len(r["per_fault"]) == 6
    # 两个配置都恢复了全部 18 条（伪判官反馈闭环）
    assert rows["cmp-all-at-once"]["recovered"] == 18
    assert rows["cmp-binary"]["recovered"] == 18
    out_json = tmp_path / "cmp" / "comparison.json"
    assert json.loads(out_json.read_text())["n_configs"] == 2


def test_run_compare_rejects_mismatched_sources(tmp_path):
    from tests.helpers import write_traces_jsonl

    src = write_traces_jsonl(tmp_path / "t.jsonl", ToySandbox().generate_corpus(1))
    a = _write_cfg(tmp_path / "a.json", "cfg-a", "all_at_once")
    (tmp_path / "b.json").write_text(
        json.dumps({
            "run_name": "cfg-b",
            "source": {"type": "jsonl", "path": "different.jsonl"},
            "llm": {"type": "fake"},
            "stages": {"represent": ["canonical_events"], "attribute": ["all_at_once"]},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同一轨迹集"):
        run_compare([a, str(tmp_path / "b.json")], tmp_path / "cmp")


def test_counting_llm_passes_through(tmp_path):
    from atap.llm.fake_client import FakeLLMClient

    inner = FakeLLMClient()
    wrapper = CountingLLM(inner)
    b, ctx = None, None
    from atap.core.config import config_from_dict
    from atap.runtime import run_config
    from tests.helpers import write_traces_jsonl

    src = write_traces_jsonl(
        tmp_path / "t.jsonl", [ToySandbox().generate("q-trajaudit", "malformed_tool_call")]
    )
    cfg = config_from_dict({
        "run_name": "count",
        "source": {"type": "jsonl", "path": src},
        "llm": {"type": "fake"},
        "stages": {
            "represent": ["canonical_events", "ssf"],
            "analyze": ["judge_eval"],
            "classify": ["mast_judge"],
            "attribute": ["all_at_once"],
        },
    })
    run_config(cfg, tmp_path / "out", llm=wrapper)
    assert wrapper.calls.count("judge_eval") == 1
    assert wrapper.calls.count("mast_judge") == 1
    assert wrapper.calls.count("all_at_once") == 1
