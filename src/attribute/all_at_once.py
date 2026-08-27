"""All-at-Once single-pass attribution — Who&When, arXiv:2505.00212 (ICML'25) §4.1.

Mechanism: an LLM reads the query + the full failure log in a single window
(this implementation consumes the SSF folding view for robustness against long
trajectories) and outputs the responsible agent + decisive error step + reason
in one shot. Findings from the paper: agent-level works best (GPT-4o main table
54.33 — note that number is from the With-GT column; this implementation does
not inject gold, corresponding to the Without-GT column 51.12), ~17K tokens,
while step-level is weak (12.5 — likewise a With-GT number, from the
Algorithm-Generated sub-column; under this implementation's Without-GT basis
the step-level counterpart is 13.53) — hence agent-level conclusions are
primary and step-level secondary (stage three uses binary-search localization
to fill in step-level). The MAST definition block and few-shot example in the
prompt are engineering enhancements beyond the paper's G.1 (disable with
``few_shot=False``). The judge view also keeps the task-header ``outcome:``
line by default (core.render.judge_view) — neither of the paper's two G.1
prompt variants carries an outcome line; keeping it lets the single-pass judge
anchor on the verifier-explained failure [adaptation].

Unified output contract (repository convention — the framework's shared
Hypothesis schema, not a paper element; Who&When §6 is its conclusion):
results are converted to a ranked list of core.schema.Hypothesis (single
hypothesis for this algorithm; evidence citations = responsible step rendered
line + verifier line).

Trigger semantics: by default only failed trajectories are attributed
(detection ≠ attribution: successful trajectories do not enter attribution);
``include_success=True`` overrides this (for pseudo-success audit scenarios,
DRIFT 2606.02060).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.classify.taxonomy import MAST_MODES, mast_definitions_block  # shared vocabulary
from atap.core.registry import register
from atap.core.render import judge_view, render_event_line
from atap.core.schema import Hypothesis


class AttributionVerdict(BaseModel):
    responsible_agent: str = Field(description="Name of the responsible agent (as it appears in the trajectory)")
    step: int = Field(ge=0, description="R0 index of the decisive error step")
    reason: str
    fix_suggestion: str = Field(description="An actionable fix suggestion (injected into the targeted rerun)")
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str | None = Field(
        default=None, description="Best-matching MAST code (e.g., FM-1.3); may be empty if uncertain"
    )


_SYSTEM = (
    "You are an expert in multi-agent system failure attribution. Given a task and "
    "the full failure trajectory, determine: (1) which agent bears primary "
    "responsibility; (2) at which step the decisive error occurred (the earliest "
    "decisive error, not the step where the symptom manifests — symptoms often "
    "appear later than the root cause); (3) the reason and a fix suggestion. "
    "You may consult the MAST failure mode codes:\n{definitions}"
)
# Anti-leak constraint: the example must not be an answer key for any sandbox
# fault — no (GT agent, GT code) pair, no GT onset step, no GT step-run like
# "3/5/7". The agent here is fictional and outside the sandbox roster.
_FEW_SHOT = (
    "Example: the editor receives an ambiguous change request at step 6, "
    "never asks for clarification, and at step 10 delivers a rewrite of the "
    "wrong section; the failure only becomes visible at the later review "
    "step — responsible agent=editor, step=10 (proceeding on ambiguous input "
    "without clarification is the earliest decisive error, not the later "
    "review where the symptom manifests), failure_mode=FM-2.2."
)


@register
class AllAtOnceAttributor(Attributor):
    stage = "attribute"
    name = "all_at_once"

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        if bundle.succeeded and not self.param("include_success", False):
            return  # successful trajectories produce no attribution (no artifact recorded)
        if ctx.llm is None:
            raise RuntimeError("all_at_once requires an LLM client (RunContext.llm)")

        system = _SYSTEM.format(definitions=mast_definitions_block())
        if self.param("few_shot", True):
            system += "\n\n" + _FEW_SHOT
        agents = bundle.trajectory.agents()
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"The task and failure trajectory are as follows (agent roster: {', '.join(agents)}):\n"
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
        # an invalid failure_mode code is clamped to None like step/agent --
        # but must leave a trace instead of silently disappearing
        if verdict.failure_mode is not None and code is None:
            evidence.append(
                f"(judgement clamped: failure_mode {verdict.failure_mode!r}->None: "
                f"not a MAST code)"
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
