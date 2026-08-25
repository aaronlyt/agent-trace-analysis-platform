"""测试辅助 —— 程序化构造 R0 轨迹（研究问答玩具域，与 sandbox 同构）。"""

from __future__ import annotations

from typing import Any

from atap.core.schema import (
    AGENT_MESSAGE,
    HANDOFF,
    LLM_CALL,
    TASK_END,
    TASK_START,
    TOOL_CALL,
    TOOL_RESULT,
    VERIFIER,
    Outcome,
    TraceEvent,
    Trajectory,
)


def _ev(i: int, kind: str, agent: str, action: str | None = None,
        payload: dict | None = None, refs: list[str] | None = None,
        phase: str | None = None) -> TraceEvent:
    return TraceEvent(
        id=f"e{i:03d}", ts=float(i), kind=kind, agent=agent, action=action,
        payload=payload or {}, refs=refs or [], phase=phase, parent=None, index=i,
    )


DOC_TEXT = (
    "TrajAudit proposes semantic saliency folding. " * 20
)  # 长观测，供 SSF 折叠测试


def success_trace(trace_id: str = "t-ok-1") -> Trajectory:
    events = [
        _ev(0, TASK_START, "env", payload={"task": "which tool does TrajAudit propose?"}),
        _ev(1, LLM_CALL, "planner", phase="plan",
            payload={"content": "plan: search first, then read the doc, then report"}),
        _ev(2, HANDOFF, "planner", refs=["e001"], phase="plan",
            payload={"to": "searcher", "content": "please find the TrajAudit paper"}),
        _ev(3, TOOL_CALL, "searcher", action="search", phase="search",
            payload={"query": "trajaudit folding"}),
        _ev(4, TOOL_RESULT, "env", action="search", refs=["e003"], phase="search",
            payload={"content": "search results for 'trajaudit folding': 2 docs [d1, d3]"}),
        _ev(5, TOOL_CALL, "searcher", action="read_doc", refs=["e004"], phase="search",
            payload={"doc_id": "d1"}),
        _ev(6, TOOL_RESULT, "env", action="read_doc", refs=["e005"], phase="search",
            payload={"content": DOC_TEXT}),
        _ev(7, HANDOFF, "searcher", refs=["e006"], phase="report",
            payload={"to": "reporter", "content": "the answer is in d1"}),
        _ev(8, LLM_CALL, "reporter", refs=["e007"], phase="report",
            payload={"content": "based on d1, TrajAudit proposes semantic saliency folding (cited: d1)"}),
        _ev(9, TOOL_CALL, "reporter", action="submit", refs=["e008"], phase="report",
            payload={"answer": "semantic saliency folding (d1)"}),
        _ev(10, VERIFIER, "verifier", refs=["e009"],
            payload={"content": "passed: answer matches gold and cites a read document"}),
        _ev(11, TASK_END, "env"),
    ]
    return Trajectory(
        trace_id=trace_id,
        task="which tool does TrajAudit propose? answer must cite a read doc id",
        events=events,
        outcome=Outcome(success=True, score=1.0, note="passed"),
        meta={"model_version": "gpt4o-2026", "prompt_version": "p1"},
    )


def failure_trace_ungrounded(trace_id: str = "t-fail-1") -> Trajectory:
    """失败轨迹：reporter 引用了检索到但从未 read 过的 d3（无据引用）。"""
    events = [
        _ev(0, TASK_START, "env", payload={"task": "which tool does TrajAudit propose?"}),
        _ev(1, HANDOFF, "planner", phase="plan",
            payload={"to": "searcher", "content": "find the TrajAudit paper"}),
        _ev(2, TOOL_CALL, "searcher", action="search", phase="search",
            payload={"query": "trajaudit folding"}),
        _ev(3, TOOL_RESULT, "env", action="search", refs=["e002"], phase="search",
            payload={"content": "search results for 'trajaudit folding': 2 docs [d1, d3]"}),
        _ev(4, TOOL_CALL, "searcher", action="read_doc", refs=["e003"], phase="search",
            payload={"doc_id": "d1"}),
        _ev(5, TOOL_RESULT, "env", action="read_doc", refs=["e004"], phase="search",
            payload={"content": DOC_TEXT}),
        _ev(6, HANDOFF, "searcher", refs=["e005"], phase="report",
            payload={"to": "reporter", "content": "here is d1"}),
        _ev(7, LLM_CALL, "reporter", refs=["e006"], phase="report",
            payload={"content": "based on d3, the answer is a magic fixer (cited: d3)"}),
        _ev(8, TOOL_CALL, "reporter", action="submit", refs=["e007"], phase="report",
            payload={"answer": "a magic fixer (d3)"}),
        _ev(9, VERIFIER, "verifier", refs=["e008"],
            payload={"content": "failed: cited document d3 was never read; answer wrong"}),
        _ev(10, TASK_END, "env"),
    ]
    return Trajectory(
        trace_id=trace_id,
        task="which tool does TrajAudit propose? answer must cite a read doc id",
        events=events,
        outcome=Outcome(success=False, score=0.0, note="citation not grounded"),
        meta={"model_version": "gpt4o-2026", "prompt_version": "p1"},
    )


def write_traces_jsonl(path, trajectories: list[Trajectory]) -> str:
    import json

    with open(path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return str(path)
