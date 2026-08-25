"""MAST 判官打标 —— LLM-as-a-judge 按 14 失败模式分类（arXiv:2503.13657）。

机制（原文 §3.3 判官管线）：prompt = MAST 全部定义 + few-shot 示例 +
轨迹（折叠视图）→ 判官输出命中的失败模式代码 + 理由 + 证据步。
校验：代码必须存在于 MAST 词表，未知代码被丢弃并记录（不静默采信）。

默认只打失败轨迹（MAST 标注的是失败模式；``include_success=True`` 可覆盖）。
产物：``{"labels": [...], "fusion": [...], "invalid_codes": [...]}``。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.classify.base import Classifier
from atap.classify.taxonomy import MAST_MODES, FusionLabel, mast_definitions_block
from atap.core.registry import register
from atap.core.render import judge_view


class MastLabel(BaseModel):
    code: str = Field(description="MAST 失败模式代码，如 FM-1.3")
    reason: str
    step: int | None = Field(default=None, description="证据步（R0 index）")


class MastLabels(BaseModel):
    labels: list[MastLabel] = Field(default_factory=list)


_SYSTEM = (
    "你是 MAST 多智能体系统失败模式标注判官。下面给出 MAST 的 3 类 14 种失败"
    "模式定义与一条执行轨迹。请选出轨迹中实际发生的失败模式（可多个），"
    "每个给出代码、理由与证据步。不确定的不要选。只从给定代码中选择。\n\n"
    "MAST 定义：\n{definitions}"
)
_FEW_SHOT = (
    "示例：轨迹中 searcher 连续三次发起完全相同的 search 调用直至预算耗尽——"
    "输出 {\"labels\": [{\"code\": \"FM-1.3\", \"reason\": \"无进展重复检索\", \"step\": 5}]}。"
)


@register
class MastJudgeClassifier(Classifier):
    stage = "classify"
    name = "mast_judge"

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} 无 R0 事件流：请先配置 canonical_events"
            )
        if bundle.succeeded and not self.param("include_success", False):
            bundle.put("classify", self.name, {"labels": [], "fusion": [], "invalid_codes": []})
            return
        if ctx.llm is None:
            raise RuntimeError("mast_judge 需要 LLM 客户端（RunContext.llm）")

        system = _SYSTEM.format(definitions=mast_definitions_block())
        if self.param("few_shot", True):
            system += "\n\n" + _FEW_SHOT
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"请标注以下轨迹的失败模式：\n{judge_view(bundle)}"},
        ]
        result = ctx.llm.complete(messages, schema=MastLabels, tag=self.name)
        parsed = result.parsed
        assert isinstance(parsed, MastLabels)

        valid, invalid = [], []
        for lab in parsed.labels[: int(self.param("max_labels", 3))]:
            if lab.code in MAST_MODES:
                valid.append(lab)
            else:
                invalid.append(lab.code)
        fusion = [
            FusionLabel(mast=lab.code, evidence_step=lab.step, reason=lab.reason)
            for lab in valid
        ]
        bundle.put(
            "classify",
            self.name,
            {
                "labels": [lab.model_dump() for lab in valid],
                "fusion": [f.to_dict() for f in fusion],
                "invalid_codes": invalid,
            },
        )
