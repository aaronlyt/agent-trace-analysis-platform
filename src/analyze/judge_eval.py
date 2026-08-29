"""LLM-as-judge evaluation —— in the style of the MAST judge pipeline (arXiv:2503.13657 §3.3;
Agent-as-a-Judge, arXiv:2410.10934).

Mechanism: a few-shot judge reads the trajectory (consuming the SSF folded
view to reduce noise on long trajectories) and outputs a quality score +
typed findings. Reliability red line (paper Table 2): zero-shot κ=0.58 →
few-shot κ=0.77, hence few_shot=True by default with a built-in prompt
example.

Differences from the paper (engineering adaptations):
* **the judge does not see the success/failure outcome** (aligned with the
  MAST Appendix J.1 judge setup: "we do not provide the success or failure
  result to the LLM Annotator"): by default the rendered view contains no
  ``outcome:`` line (the pseudo-judge infers success/failure from VERIFIER
  event lines inside the trajectory); ``show_outcome=true`` restores it;
* **the few-shot is one self-constructed output-format demonstration** ——
  the examples in the paper's §3.3 come from the human-annotated data in
  Appendix N (the κ gain in Table 2 is driven by real annotated examples);
  matching that number requires swapping in real annotated examples;
* the 0-10 quality score and the three-level minor/major/critical
  severity are a self-chosen output contract (the paper's judge outputs
  failure-mode hits + reasons; classification is mast_judge's
  responsibility; severity is vocabulary-validated —— common synonyms are
  normalized and anything still illegal after normalization is an explicit
  parse failure, never silently trusted).

This algorithm only answers "is it good / where does it look wrong"
(detection), not causal attribution —— detection ≠ attribution (the core
division of labor in literature §1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from atap.core.registry import register
from atap.core.render import judge_view
from atap.analyze.base import Analyzer

# Normalize common severity synonyms (any other value is explicitly rejected by Literal validation)
_SEVERITY_ALIAS = {
    "low": "minor", "medium": "major", "high": "critical",
    "severe": "critical",
}


class Finding(BaseModel):
    severity: Literal["minor", "major", "critical"] = Field(
        description="minor | major | critical"
    )
    description: str
    step: int | None = Field(default=None, description="R0 event index; empty when it cannot be localized")

    @field_validator("severity", mode="before")
    @classmethod
    def _norm_severity(cls, v):
        return _SEVERITY_ALIAS.get(str(v).strip().lower(), v)


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0, le=10, description="Overall quality score 0-10")
    summary: str
    findings: list[Finding] = Field(default_factory=list)


_SYSTEM = (
    "You are a rigorous evaluation judge for agent trajectories. Given the "
    "task and the execution trajectory (with the index of each step), "
    "assess the quality of task completion and point out problems. Report "
    "only problems supported by evidence, citing specific steps."
)
# Anti-leak constraint: the step numbers in the example (10/11/14) are
# fictional and must not coincide with any sandbox GT onset step
# (FAULTS/EXTRA_FAULTS onsets: 1/3/5/8/9) nor reproduce a GT step-run like
# "3/5/7" -- pinned by
# tests.test_judges.test_fewshot_step_numbers_do_not_collide_with_gt_onsets.
_FEW_SHOT = (
    "Example (excerpt): in the trajectory, the tool call at step 10 returns "
    "an error at step 11, and the answer submitted at step 14 does not match "
    "the task requirements -- output {\"score\": 2.5, \"summary\": "
    "\"submitted an unsupported answer after a failed tool call\", "
    "\"findings\": [{\"severity\": \"critical\", "
    "\"description\": \"tool call at step 10 failed\", \"step\": 10}, "
    "{\"severity\": \"major\", \"description\": \"answer submitted at step 14 "
    "does not match the task requirements\", \"step\": 14}]}."
)


@register
class JudgeEvalAnalyzer(Analyzer):
    stage = "analyze"
    name = "judge_eval"
    requires = (("represent", "canonical_events"),)   # judges the flattened R0 event stream

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events in the represent stage first"
            )
        if self.param("only_failures", False) and bundle.succeeded:
            return
        if ctx.llm is None:
            raise RuntimeError("judge_eval requires an LLM client (RunContext.llm)")

        show_outcome = bool(self.param("show_outcome", False))
        messages = [
            {"role": "system", "content": _SYSTEM + ("\n" + _FEW_SHOT if self.param("few_shot", True) else "")},
            {
                "role": "user",
                "content": (
                    f"Task: Evaluate the following trajectory:\n"
                    f"{judge_view(bundle, include_outcome=show_outcome)}"
                ),
            },
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
                "outcome_shown": show_outcome,
            },
        )
