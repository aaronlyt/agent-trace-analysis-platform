"""LLM-as-judge 评测 —— MAST 判官管线风格（arXiv:2503.13657 §3.3；
Agent-as-a-Judge, arXiv:2410.10934）。

机制：few-shot 判官读取轨迹（消费 SSF 折叠视图，降低长轨迹噪声），
输出质量分 + 类型化 finding。可靠性红线（原文 Table 2）：zero-shot
κ=0.58 → few-shot κ=0.77，故默认 few_shot=True 且 prompt 内置示例。

本算法只回答"好不好/哪里看起来有问题"（检测），不做因果归因——
检测 ≠ 归因（文献 §1 核心分工原则）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.core.registry import register
from atap.core.render import judge_view
from atap.analyze.base import Analyzer


class Finding(BaseModel):
    severity: str = Field(description="minor | major | critical")
    description: str
    step: int | None = Field(default=None, description="R0 事件 index；无法定位时为空")


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0, le=10, description="整体质量分 0-10")
    summary: str
    findings: list[Finding] = Field(default_factory=list)


_SYSTEM = (
    "你是严谨的 agent 轨迹评测判官。给定任务与执行轨迹（含每步的 index），"
    "评估任务完成质量并指出问题。只报告有证据支撑的问题，引用具体 step。"
)
_FEW_SHOT = (
    "示例（节选）：轨迹中 step 4 的工具结果含 error、step 9 提交的答案与任务"
    "要求不符——输出 {\"score\": 2.5, \"summary\": \"工具调用失败后仍提交无证据答案\", "
    "\"findings\": [{\"severity\": \"critical\", \"description\": \"畸形工具调用\", \"step\": 3}]}。"
)


@register
class JudgeEvalAnalyzer(Analyzer):
    stage = "analyze"
    name = "judge_eval"

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} 无 R0 事件流：请先在 represent 阶段配置 canonical_events"
            )
        if self.param("only_failures", False) and bundle.succeeded:
            return
        if ctx.llm is None:
            raise RuntimeError("judge_eval 需要 LLM 客户端（RunContext.llm）")

        messages = [
            {"role": "system", "content": _SYSTEM + ("\n" + _FEW_SHOT if self.param("few_shot", True) else "")},
            {"role": "user", "content": f"请评测以下轨迹：\n{judge_view(bundle)}"},
        ]
        result = ctx.llm.complete(messages, schema=JudgeVerdict, tag=self.name)
        verdict = result.parsed
        assert isinstance(verdict, JudgeVerdict)
        bundle.put(
            "analyze",
            self.name,
            {
                **verdict.model_dump(),
                "view": "ssf_folded" if bundle.has("represent", "ssf") else "full",
            },
        )
