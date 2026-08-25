"""All-at-Once 单遍归因 —— Who&When, arXiv:2505.00212（ICML'25）§4.1。

机制：LLM 单窗口读取 query + 完整失败日志（本实现消费 SSF 折叠视图以
抗长轨迹噪声），一次输出责任 agent + 决定性错误步 + 原因。原文结论：
agent 级最佳（GPT-4o 主表 54.33——注意该数为 With-GT 列数字；本实现
不注入 gold，对应 Without-GT 列 51.12）、~17K token，step 级偏弱
（12.5）——故 agent 级结论为主、step 为辅（阶段三用二分定位补 step 级）。
prompt 中的 MAST 定义块与 few-shot 示例是论文 G.1 之外的工程增强
（``few_shot=False`` 可关）。

统一输出契约（文献 §6）：结果转为 core.schema.Hypothesis 的 ranked list
（本算法单假设；证据引文 = 责任步渲染行 + verifier 行）。

触发语义：默认只归因失败轨迹（检测≠归因：成功轨迹不进归因）；
``include_success=True`` 可覆盖（用于伪成功审查场景，DRIFT 2606.02060）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.classify.taxonomy import MAST_MODES, mast_definitions_block  # 共享词表
from atap.core.registry import register
from atap.core.render import judge_view, render_event_line
from atap.core.schema import Hypothesis


class AttributionVerdict(BaseModel):
    responsible_agent: str = Field(description="责任 agent 名（轨迹中出现）")
    step: int = Field(ge=0, description="决定性错误步的 R0 index")
    reason: str
    fix_suggestion: str = Field(description="可执行的修复建议（供定向重跑注入）")
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str | None = Field(
        default=None, description="最匹配的 MAST 代码（如 FM-1.3），不确定可为空"
    )


_SYSTEM = (
    "你是多智能体系统失败归因专家。给定任务与完整失败轨迹，请判断："
    "(1) 哪个 agent 负主要责任；(2) 决定性错误发生在哪一步（最早的决定性"
    "错误，不是症状显现步——症状常晚于根因）；(3) 原因与修复建议。"
    "可参考 MAST 失败模式代码：\n{definitions}"
)
_FEW_SHOT = (
    "示例：searcher 的同一 search 调用出现在 step 5/6/7（无进展重复）直至"
    "预算耗尽、最终提交失败——责任 agent=searcher、step=6（第二次调用即"
    "首次重复，最早的重复才是决定性错误），failure_mode=FM-1.3。"
)


@register
class AllAtOnceAttributor(Attributor):
    stage = "attribute"
    name = "all_at_once"

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} 无 R0 事件流：请先配置 canonical_events"
            )
        if bundle.succeeded and not self.param("include_success", False):
            return  # 成功轨迹不产出归因（无 artifacts 记录）
        if ctx.llm is None:
            raise RuntimeError("all_at_once 需要 LLM 客户端（RunContext.llm）")

        system = _SYSTEM.format(definitions=mast_definitions_block())
        if self.param("few_shot", True):
            system += "\n\n" + _FEW_SHOT
        agents = bundle.trajectory.agents()
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"任务与失败轨迹如下（agent 名单：{', '.join(agents)}）：\n"
                    f"{judge_view(bundle)}"
                ),
            },
        ]
        result = ctx.llm.complete(messages, schema=AttributionVerdict, tag=self.name)
        verdict = result.parsed
        assert isinstance(verdict, AttributionVerdict)

        events = bundle.trajectory.events
        step = min(max(verdict.step, 0), len(events) - 1)
        responsible = (
            verdict.responsible_agent
            if verdict.responsible_agent in agents
            else agents[0]
        )
        code = verdict.failure_mode if verdict.failure_mode in MAST_MODES else None

        ev = events[step]
        verifier = next((e for e in reversed(events) if e.kind == "VERIFIER"), None)
        evidence = [
            f"[{ev.index}] {ev.agent} {ev.kind} :: {str(ev.payload.get('content', ''))[:160]}"
        ]
        if verifier is not None:
            evidence.append(
                f"[{verifier.index}] verifier :: {str(verifier.payload.get('content', ''))[:160]}"
            )
        if verdict.step != step or verdict.responsible_agent != responsible:
            evidence.append(
                f"(judgement clamped: step {verdict.step}->{step}, "
                f"agent {verdict.responsible_agent!r}->{responsible!r})"
            )

        self.emit(
            bundle,
            [
                Hypothesis(
                    agent=responsible,
                    step=step,
                    root_cause=verdict.reason,
                    root_cause_code=code,
                    responsible_side=self.param("responsible_side", "model"),
                    evidence=evidence,
                    fix_suggestion=verdict.fix_suggestion,
                    confidence=verdict.confidence,
                )
            ],
        )
