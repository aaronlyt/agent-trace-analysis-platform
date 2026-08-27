"""PipelineConfig -- parsing and validation of YAML/dict config into algorithm specs.

Config is the single entry point for pipeline composability (the transformers
config role)::

    source: {type: jsonl, path: runs/traces/*.jsonl}
    store:  {type: jsonl, dir: runs/out}
    llm:    {type: fake | openai, ...}
    stages:
      represent:
        - {name: canonical_events}
        - {name: ssf, params: {extra_keywords: [denied]}}
      attribute:
        - {name: all_at_once}

This module only parses/validates, it does not assemble (assembly lives in
atap.runtime, keeping core free of implementation dependencies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atap.core.base import STAGE_ORDER


class ConfigError(Exception):
    """Invalid config structure or value."""


@dataclass
class AlgorithmSpec:
    stage: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    stages: dict[str, list[AlgorithmSpec]]
    source: dict[str, Any] = field(default_factory=lambda: {"type": "jsonl"})
    store: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    sandbox: dict[str, Any] | None = None
    seed: int = 0
    closed_loop: bool = False          # whether recover outputs loop back into analyze for verification
    run_name: str = "run"

    def algorithms_in_order(self) -> list[AlgorithmSpec]:
        out: list[AlgorithmSpec] = []
        for stage in STAGE_ORDER:
            out.extend(self.stages.get(stage, []))
        return out


def config_from_dict(d: dict[str, Any]) -> PipelineConfig:
    """dict -> PipelineConfig with strict validation (unknown stage / unknown
    field structures raise immediately)."""
    if not isinstance(d, dict):
        raise ConfigError(f"config must be a dict, got {type(d).__name__}")
    known_top = {
        "stages", "source", "store", "llm", "sandbox", "seed",
        "closed_loop", "run_name",
    }
    unknown = set(d) - known_top
    if unknown:
        raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")

    raw_stages = d.get("stages") or {}
    if not isinstance(raw_stages, dict) or not raw_stages:
        raise ConfigError("stages must not be empty: configure at least one pipeline")
    bad = set(raw_stages) - set(STAGE_ORDER)
    if bad:
        raise ConfigError(
            f"unknown stages: {sorted(bad)}; valid values are {list(STAGE_ORDER)}"
        )

    stages: dict[str, list[AlgorithmSpec]] = {}
    for stage, entries in raw_stages.items():
        if not isinstance(entries, list) or not entries:
            raise ConfigError(f"stages.{stage} must be a non-empty list")
        specs: list[AlgorithmSpec] = []
        for i, ent in enumerate(entries):
            if isinstance(ent, str):
                specs.append(AlgorithmSpec(stage=stage, name=ent))
                continue
            if not isinstance(ent, dict) or "name" not in ent:
                raise ConfigError(
                    f"stages.{stage}[{i}] must be 'name' or a {{name, params}} structure"
                )
            params = ent.get("params") or {}
            if not isinstance(params, dict):
                raise ConfigError(f"stages.{stage}[{i}].params must be a dict")
            specs.append(AlgorithmSpec(stage=stage, name=ent["name"], params=params))
        stages[stage] = specs

    return PipelineConfig(
        stages=stages,
        source=dict(d.get("source") or {"type": "jsonl"}),
        store=d.get("store"),
        llm=d.get("llm"),
        sandbox=d.get("sandbox"),
        seed=int(d.get("seed", 0)),
        closed_loop=bool(d.get("closed_loop", False)),
        run_name=str(d.get("run_name", "run")),
    )


def load_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # lazy import: core's logic path does not hard-depend on it
        except ImportError as e:  # pragma: no cover
            raise ConfigError("pyyaml is required to read YAML config") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        raise ConfigError(f"config file is empty: {p}")
    return config_from_dict(data)


def validate_against_registry(cfg: PipelineConfig) -> list[str]:
    """Validate that algorithms resolve against the registry; returns the list
    of algorithm descriptions (missing entries raise RegistryError directly)."""
    from atap.core.registry import create, RegistryError

    descriptions: list[str] = []
    for spec in cfg.algorithms_in_order():
        try:
            algo = create(spec.stage, spec.name, **spec.params)
        except RegistryError as e:
            raise e
        descriptions.append(algo.describe())
    return descriptions
