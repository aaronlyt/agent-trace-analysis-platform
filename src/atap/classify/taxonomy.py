"""分类词表 —— MAST 14 失败模式 + 融合标签结构（共享词表，非算法模块）。

MAST（arXiv:2503.13657，Figure 1 / Appendix A）：3 类 14 种，1,642 条轨迹
人工 κ=0.88、判官 κ=0.77。定义对齐 refs/2503.13657_mast 原文（App. A），
保持论文语义不扩写——本词表经 mast_definitions_block() 直接进入判官
prompt，任何项目适配都必须放在沙盒映射层（sandbox/faults.py，标注【适配】），
否则等于用改写的定义引导判官命中 ground truth。

融合标签结构（架构文档）：(交互=MAST) × (模块=AgentError) × (系统级=SysTax)
× (责任侧=Model or Harness)。阶段二只填 MAST 维；其余维度留给后续的
AgentErrorTaxonomy / 系统级 taxonomy / 责任侧判官增量填充。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAST_CATEGORIES: dict[str, str] = {
    "FC1": "System Design Issues（系统设计问题）",
    "FC2": "Inter-Agent Misalignment（智能体间失配）",
    "FC3": "Task Verification（任务验证问题）",
}

MAST_MODES: dict[str, dict[str, str]] = {
    # ---- FC1 系统设计问题 ----
    "FM-1.1": {
        "category": "FC1",
        "name": "Disobey task specification",
        "definition": "未遵守任务的明确约束或要求（格式、必填项、指定流程等），导致偏离任务目标。",
    },
    "FM-1.2": {
        "category": "FC1",
        "name": "Disobey role specification",
        "definition": "未遵守所指派角色的职责与约束，做出越权或失职行为。",
    },
    "FM-1.3": {
        "category": "FC1",
        "name": "Step repetition",
        "definition": "无必要地重复已完成的步骤，造成延迟、冗余或预算耗尽。",
    },
    "FM-1.4": {
        "category": "FC1",
        "name": "Loss of conversation history",
        "definition": "意外的上下文截断，忽略近期交互历史而回退到旧状态。",
    },
    "FM-1.5": {
        "category": "FC1",
        "name": "Unaware of termination conditions",
        "definition": "不理解或未识别任务应终止的条件，导致流程不必要地延续或悬置。",
    },
    # ---- FC2 智能体间失配 ----
    "FM-2.1": {
        "category": "FC2",
        "name": "Conversation reset",
        "definition": "不当或无理由地重启对话，丢失已建立的上下文与进展。",
    },
    "FM-2.2": {
        "category": "FC2",
        "name": "Fail to ask for clarification",
        "definition": "面对模糊或不完整的输入时未请求澄清，基于臆测继续执行。",
    },
    "FM-2.3": {
        "category": "FC2",
        "name": "Task derailment",
        "definition": "agent 交互偏离任务目标，逐渐滑向无关子问题。",
    },
    "FM-2.4": {
        "category": "FC2",
        "name": "Information withholding",
        "definition": "未向协作方传达其需要的关键信息（需求、约束、发现），导致下游重复失败。",
    },
    "FM-2.5": {
        "category": "FC2",
        "name": "Ignored other agent's input",
        "definition": "忽视或未充分考虑其他 agent 提供的输入或建议，导致次优决策或错失协作机会。",
    },
    "FM-2.6": {
        "category": "FC2",
        "name": "Reasoning-action mismatch",
        "definition": "陈述的推理过程与实际执行的行动不一致。",
    },
    # ---- FC3 任务验证问题 ----
    "FM-3.1": {
        "category": "FC3",
        "name": "Premature termination",
        "definition": "在必要信息尚未交换或目标尚未达成时即终止任务，导致不完整或不正确的结果。",
    },
    "FM-3.2": {
        "category": "FC3",
        "name": "No or incomplete verification",
        "definition": "对产出不做验证，或验证不完整就采信结果。",
    },
    "FM-3.3": {
        "category": "FC3",
        "name": "Incorrect verification",
        "definition": "对关键信息或决策的验证方式本身错误（假阳性通过、把未核实信息当作已验证依据），给出误导性结论。",
    },
}


def mast_definitions_block() -> str:
    """判官 prompt 用的 MAST 定义清单。"""
    lines = [f"{cat}: {name}" for cat, name in MAST_CATEGORIES.items()]
    for code, m in MAST_MODES.items():
        lines.append(f"{code} [{MAST_CATEGORIES[m['category']].split('（')[0]}] "
                     f"{m['name']} —— {m['definition']}")
    return "\n".join(lines)


@dataclass
class FusionLabel:
    """融合标签：四个正交维度，允许部分为空（按需增量填充）。"""

    mast: str | None = None      # 交互维度（MAST FM-x.y）
    module: str | None = None    # 模块维度（AgentErrorTaxonomy：memory/reflection/planning/action/system）
    system: str | None = None    # 系统级维度（SysTax 15 种 + 漂移）
    side: str | None = None      # 责任侧（model / harness 组件）
    evidence_step: int | None = None
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mast": self.mast,
            "module": self.module,
            "system": self.system,
            "side": self.side,
            "evidence_step": self.evidence_step,
            "reason": self.reason,
            "extra": self.extra,
        }
