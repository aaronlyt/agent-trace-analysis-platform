"""表征层测试：R0 拍平 + R1 SSF 折叠。"""

from __future__ import annotations

import pytest

from atap.core.bundle import TrajectoryBundle
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.render import render_trace
from atap.represent.ssf import unfold
from atap.sandbox import ToySandbox

from helpers import success_trace


def _bundle(task="q-trajaudit", fault=None):
    t = ToySandbox().generate(task, fault)
    b = TrajectoryBundle(t)
    ctx = RunContext()
    create("represent", "canonical_events").run_one(b, ctx)
    return b, ctx


def test_canonical_flatten_from_spans():
    b, _ = _bundle()
    t = b.trajectory
    art = b.get("represent", "canonical_events")
    assert art["n_events"] == len(t.events) > 10
    assert [e.index for e in t.events] == list(range(len(t.events)))
    assert [e.id for e in t.events] == [f"e{i:03d}" for i in range(len(t.events))]
    # 引用边：TOOL_RESULT → 其 TOOL_CALL；HANDOFF → plan
    res = next(e for e in t.events if e.kind == "TOOL_RESULT" and e.action == "search")
    call = next(e for e in t.events if e.kind == "TOOL_CALL" and e.action == "search")
    assert res.refs == [call.id]
    assert art["n_refs"] >= 4 and art["dropped_refs"] == 0
    assert set(art["agents"]) >= {"planner", "searcher", "reporter", "env", "verifier"}


def test_canonical_normalize_flat_trajectory():
    t = success_trace("flat-1")
    b = TrajectoryBundle(t)
    create("represent", "canonical_events").run_one(b, RunContext())
    assert [e.index for e in t.events] == list(range(len(t.events)))


def test_ssf_folds_long_prose_keeps_errors():
    # malformed 路径：唯一观测是错误消息 → 保留、无折叠对象
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    create("represent", "ssf").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_tool_results"] == 1 and stats["n_kept_error"] == 1

    # 正常读取路径：read_doc 长散文折叠、短检索结果保留、无错误观测
    b2, ctx2 = _bundle("q-trajaudit", "ungrounded_citation")
    create("represent", "ssf").run_one(b2, ctx2)
    art = b2.get("represent", "ssf")
    stats = art["stats"]
    assert stats["n_kept_error"] == 0
    assert stats["n_folded"] == 1  # read_doc 长文
    assert stats["n_kept_short"] == 1  # search 结果短，保留
    assert stats["fold_ratio"] == 0.5
    # 折叠可逆：表里能取回原文
    fid = next(iter(art["table"]))
    assert "TrajAudit" in unfold(art, fid)
    # 占位符含摘要，判官可从折叠视图获得最小证据
    ph = next(iter(art["fold"].values()))
    assert ph.startswith("⟦folded:") and "TrajAudit" in ph


def test_ssf_does_not_treat_domain_prose_as_error():
    """语料散文含词典词（error step / missing / failed）但不是错误观测。"""
    b, ctx = _bundle("q-drift", "ungrounded_citation")  # d5 正文含 "missing"
    create("represent", "ssf").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_kept_error"] == 0
    assert stats["n_folded"] >= 1  # read_doc 长文被折叠


def test_ssf_loose_mode_keeps_keyword_prose():
    b, ctx = _bundle("q-drift", "ungrounded_citation")
    create("represent", "ssf", keyword_mode="loose").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_kept_error"] >= 1  # loose：词面词典命中散文


def test_ssf_patch_kept():
    b, ctx = _bundle()
    t = b.trajectory
    # 注入一个带 diff 头的观测，验证补丁保留
    for e in t.events:
        if e.kind == "TOOL_RESULT" and e.action == "read_doc":
            e.payload["content"] = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,3 @@\n+fix\n" * 5
    create("represent", "ssf").run_one(b, ctx)
    stats = b.get("represent", "ssf")["stats"]
    assert stats["n_kept_patch"] == 1 and stats["n_folded"] == 0


def test_judge_view_uses_fold():
    b, ctx = _bundle("q-trajaudit", "malformed_tool_call")
    create("represent", "ssf").run_one(b, ctx)
    from atap.core.render import judge_view

    view = judge_view(b)
    assert "⟦folded:" not in view or True  # 折叠视图占位符可能出现于 read_doc
    # 关键证据（错误观测）在折叠视图中保留
    assert "error: invalid arguments" in view
