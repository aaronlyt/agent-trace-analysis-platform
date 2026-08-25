"""PipelineConfig —— YAML/dict 配置到算法规格的解析与校验。

配置是流程可组合的唯一入口（transformers 的 config 角色）::

    source: {type: jsonl, path: runs/traces/*.jsonl}
    store:  {type: jsonl, dir: runs/out}
    llm:    {type: fake | openai, ...}
    stages:
      represent:
        - {name: canonical_events}
        - {name: ssf, params: {extra_keywords: [denied]}}
      attribute:
        - {name: all_at_once}

本模块只做解析/校验，不做装配（装配在 atap.runtime，保持 core 零实现依赖）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atap.core.base import STAGE_ORDER


class ConfigError(Exception):
    """配置结构/取值不合法。"""


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
    closed_loop: bool = False          # recover 产物是否自动回到 analyze 验证
    run_name: str = "run"

    def algorithms_in_order(self) -> list[AlgorithmSpec]:
        out: list[AlgorithmSpec] = []
        for stage in STAGE_ORDER:
            out.extend(self.stages.get(stage, []))
        return out


def config_from_dict(d: dict[str, Any]) -> PipelineConfig:
    """dict → PipelineConfig，严格校验（未知 stage / 未知字段结构直接报错）。"""
    if not isinstance(d, dict):
        raise ConfigError(f"配置必须是 dict，得到 {type(d).__name__}")
    known_top = {
        "stages", "source", "store", "llm", "sandbox", "seed",
        "closed_loop", "run_name",
    }
    unknown = set(d) - known_top
    if unknown:
        raise ConfigError(f"未知顶层配置键：{sorted(unknown)}")

    raw_stages = d.get("stages") or {}
    if not isinstance(raw_stages, dict) or not raw_stages:
        raise ConfigError("stages 不能为空：至少配置一个流程")
    bad = set(raw_stages) - set(STAGE_ORDER)
    if bad:
        raise ConfigError(
            f"未知 stage：{sorted(bad)}；合法值为 {list(STAGE_ORDER)}"
        )

    stages: dict[str, list[AlgorithmSpec]] = {}
    for stage, entries in raw_stages.items():
        if not isinstance(entries, list) or not entries:
            raise ConfigError(f"stages.{stage} 必须是非空列表")
        specs: list[AlgorithmSpec] = []
        for i, ent in enumerate(entries):
            if isinstance(ent, str):
                specs.append(AlgorithmSpec(stage=stage, name=ent))
                continue
            if not isinstance(ent, dict) or "name" not in ent:
                raise ConfigError(
                    f"stages.{stage}[{i}] 必须是 'name' 或 {{name, params}} 结构"
                )
            params = ent.get("params") or {}
            if not isinstance(params, dict):
                raise ConfigError(f"stages.{stage}[{i}].params 必须是 dict")
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
        raise ConfigError(f"配置文件不存在：{p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # 延迟导入：core 逻辑路径不强制依赖
        except ImportError as e:  # pragma: no cover
            raise ConfigError("读取 YAML 配置需要 pyyaml") from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        raise ConfigError(f"配置文件为空：{p}")
    return config_from_dict(data)


def validate_against_registry(cfg: PipelineConfig) -> list[str]:
    """对照注册表校验算法可解析；返回算法描述列表（缺失直接抛 RegistryError）。"""
    from atap.core.registry import create, RegistryError

    descriptions: list[str] = []
    for spec in cfg.algorithms_in_order():
        try:
            algo = create(spec.stage, spec.name, **spec.params)
        except RegistryError as e:
            raise e
        descriptions.append(algo.describe())
    return descriptions
