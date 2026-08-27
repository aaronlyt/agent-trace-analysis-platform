"""Classification vocabulary —— MAST 14 failure modes + fusion label structure (shared vocabulary, not an algorithm module).

MAST (arXiv:2503.13657, Figure 1 / Appendix A): 3 categories, 14 modes,
1,642 trajectories; human κ=0.88, judge κ=0.77. Definitions are aligned with
the original text in refs/2503.13657_mast (App. A) and keep the paper's
semantics without elaboration —— this vocabulary enters the judge prompt
directly via mast_definitions_block(); any project adaptation must live in
the sandbox mapping layer (sandbox/faults.py, marked [adaptation]),
otherwise a rewritten definition would effectively steer the judge toward
the ground truth.

Fusion label structure (architecture doc): (interaction=MAST) ×
(module=AgentError) × (system-level=SysTax) × (responsibility side=Model or
Harness). Phase two fills only the MAST dimension; the remaining dimensions
are left for later incremental filling by the AgentErrorTaxonomy /
system-level taxonomy / responsibility-side judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAST_CATEGORIES: dict[str, str] = {
    "FC1": "System Design Issues",
    "FC2": "Inter-Agent Misalignment",
    "FC3": "Task Verification",
}

MAST_MODES: dict[str, dict[str, str]] = {
    # ---- FC1 System Design Issues ----
    "FM-1.1": {
        "category": "FC1",
        "name": "Disobey task specification",
        "definition": "Fails to adhere to the specified constraints or requirements of the given task, leading to suboptimal or incorrect outcomes.",
    },
    "FM-1.2": {
        "category": "FC1",
        "name": "Disobey role specification",
        "definition": "Fails to adhere to the duties and constraints of the assigned role, potentially causing the agent to behave like a different agent.",
    },
    "FM-1.3": {
        "category": "FC1",
        "name": "Step repetition",
        "definition": "Unnecessarily repeats steps that have already been completed, potentially causing delays or errors in task completion.",
    },
    "FM-1.4": {
        "category": "FC1",
        "name": "Loss of conversation history",
        "definition": "Unexpected truncation of context, ignoring recent interaction history and reverting to a previous conversation state.",
    },
    "FM-1.5": {
        "category": "FC1",
        "name": "Unaware of termination conditions",
        "definition": "Fails to recognize or understand the conditions that should trigger the end of the interaction, potentially causing unnecessary continuation.",
    },
    # ---- FC2 Inter-Agent Misalignment ----
    "FM-2.1": {
        "category": "FC2",
        "name": "Conversation reset",
        "definition": "Unexpectedly or without reason restarts the conversation, potentially losing established context and progress.",
    },
    "FM-2.2": {
        "category": "FC2",
        "name": "Fail to ask for clarification",
        "definition": "Fails to request additional information when faced with unclear or incomplete data, potentially leading to incorrect actions.",
    },
    "FM-2.3": {
        "category": "FC2",
        "name": "Task derailment",
        "definition": "Deviates from the intended objective or focus of the task, potentially leading to irrelevant or unproductive actions.",
    },
    "FM-2.4": {
        "category": "FC2",
        "name": "Information withholding",
        "definition": "Fails to share or communicate important data or insights in its possession that could influence other agents' decisions if shared.",
    },
    "FM-2.5": {
        "category": "FC2",
        "name": "Ignored other agent's input",
        "definition": "Overlooks or fails to adequately consider the input or advice provided by other agents, potentially causing suboptimal decisions or missed collaboration opportunities.",
    },
    "FM-2.6": {
        "category": "FC2",
        "name": "Reasoning-action mismatch",
        "definition": "Stated reasoning is inconsistent with the actions actually executed, potentially resulting in unexpected or undesired behaviors.",
    },
    # ---- FC3 Task Verification ----
    "FM-3.1": {
        "category": "FC3",
        "name": "Premature termination",
        "definition": "Terminates the task before all necessary information has been exchanged or the goal has been achieved, potentially causing incomplete or incorrect outcomes.",
    },
    "FM-3.2": {
        "category": "FC3",
        "name": "No or incomplete verification",
        "definition": "Performs no (or only partial) appropriate checks and confirmations of task outputs or system outputs, potentially allowing errors or inconsistencies to propagate undetected.",
    },
    "FM-3.3": {
        "category": "FC3",
        "name": "Incorrect verification",
        "definition": "Fails to adequately verify or cross-check key information or decisions during iteration, potentially leading to system errors or vulnerabilities.",
    },
}


def mast_definitions_block() -> str:
    """MAST definitions listing for the judge prompt."""
    lines = [f"{cat}: {name}" for cat, name in MAST_CATEGORIES.items()]
    for code, m in MAST_MODES.items():
        lines.append(f"{code} [{MAST_CATEGORIES[m['category']].split(' (')[0]}] "
                     f"{m['name']} -- {m['definition']}")
    return "\n".join(lines)


@dataclass
class FusionLabel:
    """Fusion label: four orthogonal dimensions, any of which may be empty (filled incrementally as needed)."""

    mast: str | None = None      # interaction dimension (MAST FM-x.y)
    module: str | None = None    # module dimension (AgentErrorTaxonomy: memory/reflection/planning/action/system)
    system: str | None = None    # system-level dimension (SysTax 15 kinds + drift)
    side: str | None = None      # responsibility side (model / harness component)
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
