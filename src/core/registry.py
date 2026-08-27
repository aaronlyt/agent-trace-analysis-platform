"""Algorithm registry -- the heart of the transformers-style pluggable mechanism.

Usage (inside an algorithm module)::

    @register
    class SSFRepresenter(Representer):
        stage = "represent"
        name = "ssf"
        ...

The config side composes by name (consumed by config.py)::

    algo = create("represent", "ssf", keep_diff_head=True)

Constraint: core imports no algorithm modules; the registry is populated when
``atap/__init__.py`` imports the stage packages (algorithm modules only
self-register at import time, zero changes to the core).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from atap.core.base import StageAlgorithm

_T = TypeVar("_T", bound="StageAlgorithm")


class RegistryError(Exception):
    """Registry contract violation (duplicate name / not registered / invalid stage)."""


_REGISTRY: dict[tuple[str, str], type["StageAlgorithm"]] = {}

STAGES = ("represent", "analyze", "classify", "attribute", "recover")


def register(cls: type[_T]) -> type[_T]:
    """Class decorator: registers an algorithm class by ``(cls.stage, cls.name)``."""
    from atap.core.base import StageAlgorithm  # local import avoids a cycle

    if not issubclass(cls, StageAlgorithm):
        raise RegistryError(f"{cls.__name__} must subclass StageAlgorithm")
    stage = getattr(cls, "stage", None)
    name = getattr(cls, "name", None)
    if stage not in STAGES:
        raise RegistryError(
            f"{cls.__name__}.stage={stage!r} is invalid, must be one of {STAGES}"
        )
    if not name or not isinstance(name, str):
        raise RegistryError(f"{cls.__name__}.name must be a non-empty string")
    key = (stage, name)
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise RegistryError(
            f"registry conflict: ({stage!r}, {name!r}) is already taken by "
            f"{_REGISTRY[key].__name__}, cannot register {cls.__name__}"
        )
    _REGISTRY[key] = cls
    return cls


def create(stage: str, name: str, **params: object) -> "StageAlgorithm":
    """AutoModel-style factory: instantiate an algorithm by its registry name."""
    cls = _REGISTRY.get((stage, name))
    if cls is None:
        available = sorted(n for (s, n) in _REGISTRY if s == stage)
        raise RegistryError(
            f"unregistered algorithm stage={stage!r} name={name!r}; "
            f"available algorithms for this stage: {available}"
        )
    return cls(**params)


def lookup(stage: str, name: str) -> type["StageAlgorithm"] | None:
    return _REGISTRY.get((stage, name))


def list_algorithms(stage: str | None = None) -> dict[str, list[str]]:
    """List registered algorithms (for debugging and config validation error hints)."""
    out: dict[str, list[str]] = {}
    for (s, n) in sorted(_REGISTRY):
        if stage is None or s == stage:
            out.setdefault(s, []).append(n)
    return out


def clear_registry() -> None:
    """For tests only: empties the registry."""
    _REGISTRY.clear()
