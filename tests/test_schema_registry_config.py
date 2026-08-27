"""schema / registry / config unit tests (part of stage one acceptance)."""

from __future__ import annotations

import json

import pytest

from atap.core.config import ConfigError, config_from_dict
from atap.core.registry import RegistryError, create, list_algorithms, register
from atap.core.schema import Hypothesis, Outcome, TraceEvent, Trajectory

from helpers import failure_trace_ungrounded, success_trace


# ---------------------------------------------------------------- schema ----


def test_trajectory_roundtrip():
    t = success_trace()
    d = t.to_dict()
    t2 = Trajectory.from_dict(json.loads(json.dumps(d)))
    assert t2.trace_id == t.trace_id
    assert len(t2.events) == len(t.events)
    assert t2.events[8].refs == ["e007"]
    assert t2.outcome.success is True


def test_hypothesis_roundtrip():
    h = Hypothesis(agent="reporter", step=7, root_cause="ungrounded citation",
                   root_cause_code="FM-3.3", confidence=0.8)
    h2 = Hypothesis.from_dict(json.loads(json.dumps(h.to_dict())))
    assert h2 == h


# -------------------------------------------------------------- registry ----


def test_registry_register_and_create():
    from atap.represent.base import Representer

    @register
    class _Tmp(Representer):
        stage = "represent"
        name = "tmp_test_repr"

        def run_one(self, bundle, ctx):
            bundle.put("represent", self.name, {"ok": True})

    try:
        algo = create("represent", "tmp_test_repr", flag=1)
        assert algo.param("flag") == 1
        assert "tmp_test_repr" in list_algorithms("represent")["represent"]
    finally:
        from atap.core.registry import _REGISTRY
        _REGISTRY.pop(("represent", "tmp_test_repr"), None)


def test_registry_conflict():
    from atap.represent.base import Representer

    @register
    class _A(Representer):
        stage = "represent"
        name = "conflict_test"

        def run_one(self, bundle, ctx): ...

    try:
        with pytest.raises(RegistryError, match="registry conflict"):
            @register
            class _B(Representer):
                stage = "represent"
                name = "conflict_test"

                def run_one(self, bundle, ctx): ...
    finally:
        from atap.core.registry import _REGISTRY
        _REGISTRY.pop(("represent", "conflict_test"), None)


def test_registry_bad_stage():
    from atap.represent.base import Representer

    with pytest.raises(RegistryError, match="is invalid"):
        @register
        class _Bad(Representer):
            stage = "nonexistent"
            name = "x"

            def run_one(self, bundle, ctx): ...


def test_create_unknown_lists_available():
    with pytest.raises(RegistryError, match="available algorithms"):
        create("analyze", "no_such_algo")


# ---------------------------------------------------------------- config ----


def test_config_from_dict_minimal():
    cfg = config_from_dict(
        {"stages": {"represent": ["canonical_events"]}, "source": {"type": "jsonl", "path": "x"}}
    )
    assert cfg.stages["represent"][0].name == "canonical_events"
    assert cfg.stages["represent"][0].params == {}
    assert cfg.source == {"type": "jsonl", "path": "x"}


def test_config_unknown_stage_and_keys():
    with pytest.raises(ConfigError, match="unknown stages"):
        config_from_dict({"stages": {"nope": ["x"]}})
    with pytest.raises(ConfigError, match="unknown top-level config keys"):
        config_from_dict({"stages": {"analyze": ["x"]}, "wrong_key": 1})
    with pytest.raises(ConfigError, match="must not be empty"):
        config_from_dict({"stages": {}})


def test_config_params_dict_form():
    cfg = config_from_dict(
        {"stages": {"recover": [{"name": "x", "params": {"max_rounds": 3}}]}}
    )
    assert cfg.stages["recover"][0].params == {"max_rounds": 3}
    with pytest.raises(ConfigError, match="name"):
        config_from_dict({"stages": {"recover": [{"params": {}}]}})


def test_config_yaml_load(tmp_path):
    from atap.core.config import load_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        """
source: {type: jsonl, path: traces/*.jsonl}
llm: {type: fake}
stages:
  analyze:
    - judge_eval
""",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.llm == {"type": "fake"}
    assert cfg.stages["analyze"][0].name == "judge_eval"


def test_event_kinds_valid():
    for t in (success_trace(), failure_trace_ungrounded()):
        for ev in t.events:
            assert ev.kind in {
                "TASK_START", "LLM_CALL", "TOOL_CALL", "TOOL_RESULT",
                "AGENT_MESSAGE", "HANDOFF", "VERIFIER", "TASK_END",
            }
