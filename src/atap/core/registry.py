"""算法注册器 —— transformers 式可插拔机制的核心。

用法（算法模块内）::

    @register
    class SSFRepresenter(Representer):
        stage = "represent"
        name = "ssf"
        ...

配置侧按名组合（config.py 消费）::

    algo = create("represent", "ssf", keep_diff_head=True)

约束：core 不 import 任何算法模块；注册表由 ``atap/__init__.py``
导入各 stage 包时填充（算法模块只在导入时自注册，零改动核心）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from atap.core.base import StageAlgorithm

_T = TypeVar("_T", bound="StageAlgorithm")


class RegistryError(Exception):
    """注册契约违反（重名 / 未注册 / stage 不合法）。"""


_REGISTRY: dict[tuple[str, str], type["StageAlgorithm"]] = {}

STAGES = ("represent", "analyze", "classify", "attribute", "recover")


def register(cls: type[_T]) -> type[_T]:
    """类装饰器：按 ``(cls.stage, cls.name)`` 注册算法类。"""
    from atap.core.base import StageAlgorithm  # 局部导入避免环

    if not issubclass(cls, StageAlgorithm):
        raise RegistryError(f"{cls.__name__} 必须继承 StageAlgorithm")
    stage = getattr(cls, "stage", None)
    name = getattr(cls, "name", None)
    if stage not in STAGES:
        raise RegistryError(
            f"{cls.__name__}.stage={stage!r} 不合法，必须是 {STAGES} 之一"
        )
    if not name or not isinstance(name, str):
        raise RegistryError(f"{cls.__name__}.name 必须是非空字符串")
    key = (stage, name)
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise RegistryError(
            f"注册冲突：({stage!r}, {name!r}) 已被 "
            f"{_REGISTRY[key].__name__} 占用，无法注册 {cls.__name__}"
        )
    _REGISTRY[key] = cls
    return cls


def create(stage: str, name: str, **params: object) -> "StageAlgorithm":
    """AutoModel 式工厂：按注册名实例化算法。"""
    cls = _REGISTRY.get((stage, name))
    if cls is None:
        available = sorted(n for (s, n) in _REGISTRY if s == stage)
        raise RegistryError(
            f"未注册的算法 stage={stage!r} name={name!r}；"
            f"该 stage 可用算法：{available}"
        )
    return cls(**params)


def lookup(stage: str, name: str) -> type["StageAlgorithm"] | None:
    return _REGISTRY.get((stage, name))


def list_algorithms(stage: str | None = None) -> dict[str, list[str]]:
    """列出已注册算法（调试与配置校验错误提示用）。"""
    out: dict[str, list[str]] = {}
    for (s, n) in sorted(_REGISTRY):
        if stage is None or s == stage:
            out.setdefault(s, []).append(n)
    return out


def clear_registry() -> None:
    """仅供测试使用：清空注册表。"""
    _REGISTRY.clear()
