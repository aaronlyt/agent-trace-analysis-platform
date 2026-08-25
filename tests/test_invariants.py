"""架构不变量测试 —— 用 import 图强制"低依赖、单向流动"。

规则（对应 README 架构约定）：
1. core/** 只能 import atap.core（+ stdlib/pydantic/延迟 yaml）；
2. 算法模块（stage 包内除 base/__init__/taxonomy 外的文件）不得 import：
   其它 stage 包（唯一例外：classify.taxonomy 共享词表）、atap.sandbox、
   atap.runtime、atap.cli、atap.demo —— 算法间只通过 bundle 产物解耦；
3. llm/**、io/** 不得 import stage 包 / sandbox / runtime；
4. sandbox/** 只能 import atap.core；
5. 注册表内所有类的 stage 属性必须与其所在包一致。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import atap  # noqa: F401  注册引导
from atap.core.registry import _REGISTRY

SRC = Path(atap.__file__).parent
STAGE_PKGS = ("represent", "analyze", "classify", "attribute", "recover")
# 算法模块可共享的"词表"文件（无算法注册、纯定义）
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


# ------------------------------------------------------------------ 规则 1 --

def test_core_is_self_contained():
    # 允许依赖纯接口模块（llm/io 的 base 只含 Protocol，无实现）
    allowed = ("atap.core", "atap.llm.base", "atap.io.base")
    violations = []
    for p in all_py("core/*.py"):
        for m in sorted(atap_imports(imports_of(p))):
            if m != "atap" and not m.startswith(allowed):
                violations.append(f"{rel(p)} imports {m}")
    assert not violations, f"core 出现越界依赖：{violations}"


# ------------------------------------------------------------------ 规则 2 --

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
                    violations.append(f"{r} imports {m}（跨算法依赖）")
                if other in ("sandbox", "runtime", "cli", "demo"):
                    violations.append(f"{r} imports {m}（算法不得依赖装配/沙盒）")
    assert not violations, f"算法间耦合违规：{violations}"


# ------------------------------------------------------------------ 规则 3 --

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
    assert not violations, f"llm/io 越界依赖：{violations}"


# ------------------------------------------------------------------ 规则 4 --

def test_sandbox_only_depends_on_core():
    violations = []
    for p in all_py("sandbox/*.py"):
        for m in sorted(atap_imports(imports_of(p))):
            if m == "atap":
                continue
            if not (m.startswith("atap.core") or m.startswith("atap.sandbox")):
                violations.append(f"{rel(p)} imports {m}")
    assert not violations, f"sandbox 越界依赖：{violations}"


# ------------------------------------------------------------------ 规则 5 --

def test_registered_class_stage_matches_package():
    mismatches = []
    for (stage, name), cls in _REGISTRY.items():
        mod = cls.__module__
        if not mod.startswith("atap."):
            continue  # 测试内注册的 Dummy 不受包位置约束
        pkg = mod.split(".")[1]
        if pkg in STAGE_PKGS and pkg != stage:
            mismatches.append(f"{mod}:{cls.__name__} 注册为 stage={stage}，但位于 {pkg} 包")
    assert not mismatches, f"stage/包不一致：{mismatches}"


def test_every_registered_class_is_instantiable_via_factory():
    from atap.core.registry import create

    for (stage, name), cls in list(_REGISTRY.items()):
        if cls.__module__.startswith("tests."):
            continue
        algo = create(stage, name)
        assert algo.stage == stage and algo.name == name
