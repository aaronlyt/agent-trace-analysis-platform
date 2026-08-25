"""R0 规范事件化 —— span 树拍平为统一事件流（表征层基础算法）。

机制对齐 AgentTrajectory 事件表示（AgentDebugX, arXiv:2607.18754）与
总体架构 §3 R0：所有分析的输入。采集层产物是嵌套 span 树（沙盒/Langfuse
适配器皆然），本算法负责：
* DFS 拍平并赋全局序号（index）与规范事件 id（``e000``…）；
* 保留 parent（span 父子）与 refs（语义引用边，span id → 事件 id 映射）；
* 若轨迹已是扁平 R0（重跑轨迹/手写 fixtures），只做归一化（补 id/index）。

产物：``{"n_events", "kinds", "agents", "n_refs", "dropped_refs"}``；
事件写回 ``trajectory.events``（表征层是下游唯一数据接口）。
"""

from __future__ import annotations

from collections import Counter

from atap.core.registry import register
from atap.core.schema import TraceEvent
from atap.represent.base import Representer


@register
class CanonicalEventsRepresenter(Representer):
    stage = "represent"
    name = "canonical_events"

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        dropped = 0
        if t.raw and isinstance(t.raw.get("spans"), list):
            t.events, dropped = self._flatten(t.raw["spans"])
        else:
            t.events = self._normalize(t.events)
        bundle.put(
            "represent",
            self.name,
            {
                "n_events": len(t.events),
                "kinds": dict(Counter(ev.kind for ev in t.events)),
                "agents": t.agents(),
                "n_refs": sum(len(ev.refs) for ev in t.events),
                "dropped_refs": dropped,
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(spans: list[dict]) -> tuple[list[TraceEvent], int]:
        events: list[TraceEvent] = []
        by_span: dict[str, TraceEvent] = {}  # span id -> 事件
        raw_refs: dict[str, list[str]] = {}  # 事件 id -> 原始 span 引用

        def walk(nodes: list[dict], parent_eid: str | None) -> None:
            for node in nodes:
                idx = len(events)
                eid = f"e{idx:03d}"
                ev = TraceEvent(
                    id=eid,
                    ts=float(idx),
                    kind=node["kind"],
                    agent=node.get("agent", "unknown"),
                    action=node.get("action"),
                    payload=dict(node.get("payload") or {}),
                    refs=[],
                    phase=node.get("phase"),
                    parent=parent_eid,
                    index=idx,
                )
                events.append(ev)
                by_span[node["id"]] = ev
                raw_refs[eid] = list(node.get("refs") or [])
                walk(node.get("children") or [], eid)

        walk(spans, None)

        dropped = 0
        for ev in events:
            refs = raw_refs[ev.id]
            mapped = [by_span[r].id for r in refs if r in by_span]
            dropped += len(refs) - len(mapped)
            ev.refs = mapped
        return events, dropped

    @staticmethod
    def _normalize(events: list[TraceEvent]) -> list[TraceEvent]:
        out: list[TraceEvent] = []
        for i, ev in enumerate(events):
            ev.index = i
            if not ev.id:
                ev.id = f"e{i:03d}"
            if ev.ts < 0:
                ev.ts = float(i)
            out.append(ev)
        return out
