"""L2 二分定位（Binary Search）—— Who&When, arXiv:2505.00212 §4.1 / App. A.3。

机制（原文 Algorithm 2，逐行对齐）::

    low, high = 1, n
    while low < high:
        mid = floor((low+high)/2)
        展示片段 L' = {l_low ... l_mid}（只展示下半段）
        若 LLM 判错在 L' → high = mid；否则 → low = mid + 1
    s* = low；A* = l_{s*} 中的行动 agent（判官全程不被问 agent）

每轮判官只输出 ``'upper half'`` 或 ``'lower half'``（原文 G.2：无推理、
无 JSON——本实现按原文做**裸文本解析**而非结构化调用）；轮数
⌈log₂n⌉（App. D.3）；原文 token 口径 34,659（表 3，手工系统 GPT-4o）。
原文结论：step 级优于 All-at-Once（23.98 vs 12.50）、agent 级次之——
与 all_at_once 互补的 L2 深度归因。

与原文的差异：
* **0 基 index**【适配】：本框架判官可见行号与 Hypothesis.step 都是 R0
  事件 index（0 起），区间逻辑不变；
* **收尾 refine 调用**【工程增强】：原文二分只产出 (A*, s*)，不产根因
  文本；Hypothesis 契约需要 root_cause/fix_suggestion——定位后追加单次
  结构化调用生成之（DeepDebug Refine 风格："根因已定位，不要二猜"），
  ``refine=false`` 可关（关闭则用事件行机械填充）；
* **agent 回退**【适配】：s* 落在环境侧事件（env/verifier）时，责任
  agent 取 s* 之前最近的 agent 行为事件（原文每个 log 条目都是 agent
  行动，无此问题）；
* prompt 的区间措辞按 G.2 模板重构（原文占位符未给出例填充）。

触发语义：默认只归因失败轨迹（同 all_at_once）。
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.classify.taxonomy import MAST_MODES  # 共享词表（不变量例外）
from atap.core.registry import register
from atap.core.render import (
    TRACE_BEGIN,
    TRACE_END,
    judge_view,
    render_event_line,
)
from atap.core.schema import Hypothesis


class BinaryRefine(BaseModel):
    reason: str = Field(description="该步为何是决定性错误的根因解释")
    fix_suggestion: str = Field(description="可执行的修复建议（供恢复注入）")
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str | None = Field(
        default=None, description="最匹配的 MAST 代码（如 FM-1.3），不确定可为空"
    )


_ROUND_SYSTEM = (
    "你是分析多智能体协作日志的助手。给你任务与失败日志的一个片段，"
    "请判断最关键的错误更可能位于当前区间的上半还是下半。"
    "只输出 'upper half' 或 'lower half'，不要输出任何其它内容。"
)
_REFINE_SYSTEM = (
    "你是多智能体系统失败归因专家。决定性错误步已由二分定位锁定"
    "（step {step}，agent {agent}）——不要更改或质疑该定位。请基于完整"
    "轨迹解释该步为何是决定性错误（最早的可翻盘错误，而非症状显现步），"
    "并给出可执行的修复建议。可参考 MAST 失败模式代码辅助归类。"
)


def _parse_half(text: str) -> str:
    """裸文本解析（原文 G.2：只应输出 'upper half'/'lower half'）。"""
    low = text.lower()
    has_upper = "upper" in low
    has_lower = "lower" in low
    if has_lower and not has_upper:
        return "lower half"
    if has_upper and not has_lower:
        return "upper half"
    raise ValueError(f"二分回答不可解析（需 upper/lower half）：{text[:120]!r}")


@register
class BinarySearchAttributor(Attributor):
    stage = "attribute"
    name = "binary_search"

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} 无 R0 事件流：请先配置 canonical_events"
            )
        if bundle.succeeded and not self.param("include_success", False):
            return
        if ctx.llm is None:
            raise RuntimeError("binary_search 需要 LLM 客户端（RunContext.llm）")

        ssf = bundle.get("represent", "ssf")
        fold = ssf.get("fold") if isinstance(ssf, dict) else None
        n = len(events)
        low, high = 0, n - 1
        rounds: list[dict] = []
        while low < high:
            mid = (low + high) // 2
            segment = self._segment_text(events, low, mid, fold)
            messages = [
                {"role": "system", "content": _ROUND_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"任务：{bundle.trajectory.task}\n"
                        f"以下是失败日志的片段（step {low}–{mid}，完整日志共 {n} 步）：\n"
                        f"{TRACE_BEGIN}\n{segment}\n{TRACE_END}\n"
                        f"当前搜索区间为 step {low}–{high}：lower half 指 step {low}–{mid}，"
                        f"upper half 指 step {mid + 1}–{high}。"
                        "错误更可能位于上半还是下半？只输出 'upper half' 或 'lower half'。"
                    ),
                },
            ]
            result = ctx.llm.complete(messages, tag=self.name)
            answer = _parse_half(result.text)
            rounds.append(
                {"interval": [low, high], "shown": [low, mid], "answer": answer}
            )
            if answer == "lower half":
                high = mid
            else:
                low = mid + 1

        s_star = low
        responsible = self._responsible_agent(events, s_star)
        refine = self._refine(bundle, ctx, s_star, responsible)
        if refine is not None:
            reason = refine.reason
            fix = refine.fix_suggestion
            confidence = refine.confidence
            code = refine.failure_mode if refine.failure_mode in MAST_MODES else None
        else:
            ev = events[s_star]
            reason = f"二分定位收敛于 step {s_star}（{responsible} 的 {ev.kind} 事件）"
            fix = f"复核 step {s_star} 的决策依据。"
            confidence = float(self.param("default_confidence", 0.5))
            code = None

        ev = events[s_star]
        evidence = [
            f"[{ev.index}] {ev.agent} {ev.kind} :: {str(ev.payload.get('content', ev.payload))[:160]}"
        ]
        if refine is not None:
            evidence.append(f"(refine: {reason[:160]})")
        self.emit_with_log(
            bundle,
            [
                Hypothesis(
                    agent=responsible,
                    step=s_star,
                    root_cause=reason,
                    root_cause_code=code,
                    responsible_side=self.param("responsible_side", "model"),
                    evidence=evidence,
                    fix_suggestion=fix,
                    confidence=confidence,
                )
            ],
            rounds=rounds,
            n_rounds_expected=math.ceil(math.log2(n)) if n > 1 else 0,
            s_star=s_star,
            method="who_when_binary_search",
        )

    # ------------------------------------------------------------------

    def _refine(self, bundle, ctx, step: int, agent: str) -> BinaryRefine | None:
        if not self.param("refine", True):
            return None
        if ctx.llm is None:
            return None
        messages = [
            {
                "role": "system",
                "content": _REFINE_SYSTEM.format(step=step, agent=agent),
            },
            {
                "role": "user",
                "content": (
                    f"任务与完整失败轨迹如下：\n{judge_view(bundle)}"
                ),
            },
        ]
        result = ctx.llm.complete(messages, schema=BinaryRefine, tag=f"{self.name}_refine")
        parsed = result.parsed
        assert isinstance(parsed, BinaryRefine)
        return parsed

    @staticmethod
    def _segment_text(events, low: int, mid: int, fold) -> str:
        return "\n".join(
            render_event_line(ev, fold=fold) for ev in events[low: mid + 1]
        )

    @staticmethod
    def _responsible_agent(events, s_star: int) -> str:
        """A* = l_{s*} 的行动 agent；落在环境侧事件时回退到之前最近的
        agent 行为事件【适配】。"""
        acting = {"LLM_CALL", "TOOL_CALL", "HANDOFF"}
        ev = events[s_star]
        if ev.kind in acting:
            return ev.agent
        for e in reversed(events[:s_star]):
            if e.kind in acting:
                return e.agent
        return ev.agent

    def emit_with_log(self, bundle, hypotheses: list[Hypothesis], **extra) -> None:
        bundle.put(
            "attribute",
            self.name,
            {"hypotheses": [h.to_dict() for h in hypotheses], **extra},
        )
