"""R0 规范事件模型 —— 全框架唯一的数据契约。

对齐《整体流程架构与算法文献》§3 表征层 R0：span 树拍平为统一事件流
（kind/agent/动作/效果/引用边/阶段），作为所有分析的输入；机制对齐
AgentTrajectory 事件表示（AgentDebugX, arXiv:2607.18754）。

设计约束：
* 纯 stdlib dataclass，JSON 可序列化（core 零三方依赖）。
* ``refs`` 字段是引用边（本事件消费了哪些先前信息产物），为 R2 信息依赖图
  （IDG）与根因回溯预留；``parent`` 保留 span 树的父子关系。
* ``Trajectory.raw`` 允许携带采集层原始形态（嵌套 span 树），由
  represent/canonical_events 负责拍平为 ``events``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 事件类型（kind）。取值对齐 OTel GenAI 语义约定的一级分类
# （OTel_GenAI与OpenInference对比说明.md），并补充 MAS 协作事件。
# ---------------------------------------------------------------------------

TASK_START = "TASK_START"
LLM_CALL = "LLM_CALL"          # 模型调用：推理/决策/生成文本
TOOL_CALL = "TOOL_CALL"        # 模型侧发出的工具调用请求
TOOL_RESULT = "TOOL_RESULT"    # 环境侧返回的工具观测（observation）
AGENT_MESSAGE = "AGENT_MESSAGE"  # agent 间消息（发送内容即信息产物）
HANDOFF = "HANDOFF"            # 控制/职责转移（A→B）
VERIFIER = "VERIFIER"          # 验证器检查（任务级或步级）
TASK_END = "TASK_END"          # 终止：正常结束/提交/放弃

EVENT_KINDS = (
    TASK_START,
    LLM_CALL,
    TOOL_CALL,
    TOOL_RESULT,
    AGENT_MESSAGE,
    HANDOFF,
    VERIFIER,
    TASK_END,
)


@dataclass
class TraceEvent:
    """拍平后的规范事件。

    Attributes:
        id: 事件唯一标识（trace 内唯一，如 ``e007``）。
        ts: 单调递增时间戳/序号（float，采集层可用真实时间）。
        kind: :data:`EVENT_KINDS` 之一。
        agent: 行为主体（"planner" / "searcher" / ... / "env" / "verifier"）。
        action: 规范动作名（R5 九类动作的占位；阶段二可为 None）。
        payload: JSON 原生 dict；TOOL_RESULT 的观测文本约定放在 ``payload["content"]``。
        refs: 引用边——本事件消费的先前事件 id 列表（如 TOOL_RESULT 引用其
            TOOL_CALL；引用其读取过的 TOOL_RESULT；HANDOFF 引用被转移的消息）。
        phase: 任务阶段标签（如 "plan" / "search" / "report"）。
        parent: span 树中的父事件 id（None 表示顶层）。
        index: 拍平后的全局序号（0 起，canonical_events 负责赋值）。
    """

    id: str
    ts: float
    kind: str
    agent: str
    action: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    phase: str | None = None
    parent: str | None = None
    index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "kind": self.kind,
            "agent": self.agent,
            "action": self.action,
            "payload": self.payload,
            "refs": list(self.refs),
            "phase": self.phase,
            "parent": self.parent,
            "index": self.index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceEvent":
        return cls(
            id=d["id"],
            ts=float(d.get("ts", 0.0)),
            kind=d["kind"],
            agent=d.get("agent", "unknown"),
            action=d.get("action"),
            payload=dict(d.get("payload") or {}),
            refs=list(d.get("refs") or []),
            phase=d.get("phase"),
            parent=d.get("parent"),
            index=int(d.get("index", -1)),
        )


@dataclass
class Outcome:
    """轨迹结果标签。

    注意（文献约束，DRIFT 2606.02060）：结果标签不能当过程监控金标准——
    36.9% 成功轨迹含隐藏错误步；故 ``success`` 仅是 outcome 视角。
    """

    success: bool
    score: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "score": self.score, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Outcome":
        return cls(
            success=bool(d.get("success", False)),
            score=d.get("score"),
            note=d.get("note", ""),
        )


@dataclass
class Trajectory:
    """一条完整轨迹（R0 规范形态）。

    Attributes:
        trace_id: 唯一标识。
        task: 任务描述（含期望答案要求；不含 gold 答案本体——gold 只属于
            验证器，避免"先验只给失败信号"被破坏，TrajAudit 2605.26563）。
        events: R0 规范事件流（canonical_events 的输出/输入）。
        outcome: 结果标签。
        meta: 采集元信息。漂移检测分组键（模型版本×prompt 版本×时间窗，
            系统级 taxonomy 2511.19933）放 ``meta["model_version"]`` 等；
            故障注入 ground truth（沙盒专用）放 ``meta["injected_fault"]``。
        raw: 采集层原始形态（嵌套 span 树 dict）；仅当事件流尚未拍平时存在。
    """

    trace_id: str
    task: str
    events: list[TraceEvent] = field(default_factory=list)
    outcome: Outcome = field(default_factory=lambda: Outcome(success=False))
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    # -- 查询辅助 -----------------------------------------------------------

    def event_by_id(self, eid: str) -> TraceEvent | None:
        for ev in self.events:
            if ev.id == eid:
                return ev
        return None

    def agents(self) -> list[str]:
        """按首次出现顺序返回去重后的 agent 名单。"""
        seen: list[str] = []
        for ev in self.events:
            if ev.agent not in seen:
                seen.append(ev.agent)
        return seen

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self, *, include_raw: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trace_id": self.trace_id,
            "task": self.task,
            "events": [ev.to_dict() for ev in self.events],
            "outcome": self.outcome.to_dict(),
            "meta": self.meta,
        }
        if include_raw and self.raw is not None:
            d["raw"] = self.raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            trace_id=d["trace_id"],
            task=d.get("task", ""),
            events=[TraceEvent.from_dict(e) for e in d.get("events") or []],
            outcome=Outcome.from_dict(d.get("outcome") or {}),
            meta=dict(d.get("meta") or {}),
            raw=d.get("raw"),
        )


@dataclass
class Hypothesis:
    """统一归因输出（文献 §6 契约）。

    ranked hypotheses = 责任 agent + 责任步 + 根因标签 + 责任侧 +
    证据引文 + 修复建议 + 置信度。任何归因算法（L0 规则 / L1 judge /
    L2 深度 / L3 重放）都必须产出本结构，供恢复阶段与 Error Hub 消费。
    """

    agent: str
    step: int                      # 责任步（R0 事件 index）
    root_cause: str                # 根因描述
    root_cause_code: str | None = None   # 分类体系代码（如 "FM-1.3"）
    responsible_side: str = "model"      # "model" | "harness"
    evidence: list[str] = field(default_factory=list)   # 事件 id + 摘录引文
    fix_suggestion: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "step": self.step,
            "root_cause": self.root_cause,
            "root_cause_code": self.root_cause_code,
            "responsible_side": self.responsible_side,
            "evidence": list(self.evidence),
            "fix_suggestion": self.fix_suggestion,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hypothesis":
        return cls(
            agent=d["agent"],
            step=int(d["step"]),
            root_cause=d.get("root_cause", ""),
            root_cause_code=d.get("root_cause_code"),
            responsible_side=d.get("responsible_side", "model"),
            evidence=list(d.get("evidence") or []),
            fix_suggestion=d.get("fix_suggestion", ""),
            confidence=float(d.get("confidence", 0.0)),
        )
