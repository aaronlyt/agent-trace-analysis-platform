"""Architecture invariant tests -- enforce "low coupling, one-way flow"
via the import graph.

Rules (matching the README architecture conventions):
1. core/** may only import atap.core (+ stdlib/pydantic/lazy yaml);
2. algorithm modules (files in stage packages other than base/__init__/
   taxonomy) must not import: other stage packages (sole exception:
   classify.taxonomy as the shared vocabulary), atap.sandbox, atap.runtime,
   atap.cli, atap.demo -- algorithms decouple only through bundle
   artifacts;
3. llm/**, io/** must not import stage packages / sandbox / runtime;
4. sandbox/** may only import atap.core;
5. every class in the registry must have a stage attribute matching its
   package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import atap  # noqa: F401  registration bootstrap
from atap.core.registry import _REGISTRY

SRC = Path(atap.__file__).parent
STAGE_PKGS = ("represent", "analyze", "classify", "attribute", "recover")
# "vocabulary" files shareable by algorithm modules (no algorithm
# registration, pure definitions)
SHARED_VOCAB = {"classify/taxonomy.py"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def atap_imports(mods: set[str]) -> set[str]:
    return {m for m in mods if m == "atap" or m.startswith("atap.")}


def all_py(rel_glob: str) -> list[Path]:
    return sorted(SRC.glob(rel_glob))


def rel(path: Path) -> str:
    return str(path.relative_to(SRC))


# ------------------------------------------------------------------ rule 1 --

def test_core_is_self_contained():
    # pure-interface modules are allowed (llm/io base hold only Protocols, no implementations)
    allowed = ("atap.core", "atap.llm.base", "atap.io.base")
    violations = []
    for p in all_py("core/*.py"):
        for m in sorted(atap_imports(imports_of(p))):
            if m != "atap" and not m.startswith(allowed):
                violations.append(f"{rel(p)} imports {m}")
    assert not violations, f"core has out-of-bounds dependencies: {violations}"


# ------------------------------------------------------------------ rule 2 --

def test_algorithm_modules_decoupled():
    violations = []
    for pkg in STAGE_PKGS:
        for p in all_py(f"{pkg}/*.py"):
            r = rel(p)
            if r in SHARED_VOCAB or r.endswith(("base.py", "__init__.py")):
                continue
            for m in sorted(atap_imports(imports_of(p))):
                if m == "atap":
                    continue
                other = m.split(".")[1] if m.startswith("atap.") else None
                if other in STAGE_PKGS and not (
                    other == pkg or m == "atap.classify.taxonomy"
                ):
                    violations.append(f"{r} imports {m} (cross-algorithm dependency)")
                if other in ("sandbox", "runtime", "cli", "demo"):
                    violations.append(f"{r} imports {m} (algorithms must not depend on assembly/sandbox)")
    assert not violations, f"cross-algorithm coupling violations: {violations}"


# ------------------------------------------------------------------ rule 3 --

def test_llm_io_not_depend_on_stages():
    violations = []
    for glob in ("llm/*.py", "io/*.py"):
        for p in all_py(glob):
            for m in sorted(atap_imports(imports_of(p))):
                if m == "atap":
                    continue
                other = m.split(".")[1]
                if other in (*STAGE_PKGS, "sandbox", "runtime", "cli", "demo"):
                    violations.append(f"{rel(p)} imports {m}")
    assert not violations, f"llm/io out-of-bounds dependencies: {violations}"


# ------------------------------------------------------------------ rule 4 --

def test_sandbox_only_depends_on_core():
    violations = []
    for p in all_py("sandbox/*.py"):
        for m in sorted(atap_imports(imports_of(p))):
            if m == "atap":
                continue
            if not (m.startswith("atap.core") or m.startswith("atap.sandbox")):
                violations.append(f"{rel(p)} imports {m}")
    assert not violations, f"sandbox out-of-bounds dependencies: {violations}"


# ------------------------------------------------------------------ rule 5 --

def test_registered_class_stage_matches_package():
    mismatches = []
    for (stage, name), cls in _REGISTRY.items():
        mod = cls.__module__
        if not mod.startswith("atap."):
            continue  # Dummies registered inside tests are exempt from package-position constraints
        pkg = mod.split(".")[1]
        if pkg in STAGE_PKGS and pkg != stage:
            mismatches.append(
                f"{mod}:{cls.__name__} registered as stage={stage}, but located in the {pkg} package"
            )
    assert not mismatches, f"stage/package mismatch: {mismatches}"


def test_every_registered_class_is_instantiable_via_factory():
    from atap.core.registry import create

    for (stage, name), cls in list(_REGISTRY.items()):
        if cls.__module__.startswith("tests."):
            continue
        algo = create(stage, name)
        assert algo.stage == stage and algo.name == name
