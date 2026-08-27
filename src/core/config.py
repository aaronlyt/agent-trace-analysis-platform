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
            illegal = set(ent) - {"name", "params"}
            if illegal:
                # typo'd keys (e.g. paramz) must fail loudly, not be ignored
                raise ConfigError(
                    f"stages.{stage}[{i}] has unknown keys {sorted(illegal)}; "
                    f"allowed keys are 'name' and 'params'"
                )
            params = ent.get("params") or {}
            if not isinstance(params, dict):
                raise ConfigError(f"stages.{stage}[{i}].params must be a dict")
            specs.append(AlgorithmSpec(stage=stage, name=ent["name"], params=params))
        stages[stage] = specs

    closed_loop = d.get("closed_loop", False)
    if not isinstance(closed_loop, bool):
        # YAML "false"/"true" strings (or numbers) must not coerce to a truthy bool
        raise ConfigError(
            f"closed_loop must be a real boolean (true/false), got {closed_loop!r}"
        )
    try:
        seed = int(d.get("seed", 0))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"seed must be an integer, got {d.get('seed')!r}") from e

    return PipelineConfig(
        stages=stages,
        source=dict(d.get("source") or {"type": "jsonl"}),
        store=d.get("store"),
        llm=d.get("llm"),
        sandbox=d.get("sandbox"),
        seed=seed,
        closed_loop=closed_loop,
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
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in config file {p}: {e}") from e
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(f"invalid JSON in config file {p}: {e}") from e
    if data is None:
        raise ConfigError(f"config file is empty: {p}")
    return config_from_dict(data)


def validate_against_registry(cfg: PipelineConfig) -> list[str]:
    """Validate that algorithms resolve against the registry and their hard
    artifact dependencies (``StageAlgorithm.requires``) are satisfied by the
    SAME config, earlier in execution order; returns the list of algorithm
    descriptions (missing entries raise RegistryError, dependency violations
    ConfigError).

    Dependency violations used to surface only mid-run (sbfl raising
    "missing the represent/action_signature artifact" after earlier stages
    had already paid for their work) -- this check moves that failure class
    to config time, before any LLM call or artifact is produced."""
    from atap.core.registry import RegistryError, create

    specs = cfg.algorithms_in_order()
    # first-occurrence position of each configured (stage, name)
    positions: dict[tuple[str, str], int] = {}
    for i, spec in enumerate(specs):
        positions.setdefault((spec.stage, spec.name), i)

    descriptions: list[str] = []
    for i, spec in enumerate(specs):
        try:
            algo = create(spec.stage, spec.name, **spec.params)
        except RegistryError as e:
            raise e
        for rstage, rname in getattr(algo, "requires", ()):
            if rname == "*":
                satisfied = any(
                    pos < i for (st, _n), pos in positions.items() if st == rstage
                )
                if not satisfied:
                    raise ConfigError(
                        f"stages.{spec.stage}[{spec.name}] requires at least one "
                        f"{rstage}-stage algorithm configured before it (it "
                        f"consumes that stage's artifacts); add one, or drop "
                        f"{spec.name}"
                    )
            else:
                pos = positions.get((rstage, rname))
                if pos is None or pos >= i:
                    raise ConfigError(
                        f"stages.{spec.stage}[{spec.name}] requires "
                        f"{rstage}/{rname} to be configured before it (it "
                        f"consumes that artifact); add it to stages.{rstage} "
                        f"or reorder it ahead of {spec.name}"
                    )
        descriptions.append(algo.describe())
    return descriptions
