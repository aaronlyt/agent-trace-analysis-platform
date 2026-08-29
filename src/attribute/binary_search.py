"""L2 binary-search localization — Who&When, arXiv:2505.00212 §4.1 / App. A.3.

Mechanism (paper's Algorithm 2, line-by-line aligned)::

    low, high = 1, n
    while low < high:
        mid = floor((low+high)/2)
        show segment L' = {l_low ... l_mid} (only the lower half is shown)
        if the LLM judges the error to be in L' → high = mid; otherwise → low = mid + 1
    s* = low; A* = the acting agent in l_{s*} (the judge is never asked about agents)

Each round the judge outputs only ``'upper half'`` or ``'lower half'`` (paper
G.2: no reasoning, no JSON — this implementation follows the paper with
**plain-text parsing** instead of a structured call); number of rounds is
⌈log₂n⌉ (App. D.3); the paper's token count is 34,659 (Table 3, manual system
GPT-4o). Findings from the paper: step-level beats All-at-Once (23.98 vs
12.50, **With-GT column** — this implementation does not inject gold,
corresponding to the Without-GT column 16.59 vs 13.53), agent-level is second
— an L2 deep attribution complementary to all_at_once.

Differences from the paper:
* **0-based index** [adaptation]: judge-visible line numbers and
  Hypothesis.step in this framework are both R0 event indices (0-based); the
  interval logic is unchanged;
* **the With-GT prompt variant of G.2 is not implemented** [adaptation]: the
  framework stipulates that tasks contain no gold (core/schema.py; judge-input
  leak prevention is locked in by regression tests), so the Without-GT setting
  is fixed; the With-GT numbers quoted above serve only as reference for the
  paper's findings;
* **final refine call** [engineering enhancement]: the paper's binary search
  yields only (A*, s*) without root-cause text; the Hypothesis contract
  requires root_cause/fix_suggestion — after localization a single structured
  call is appended to generate them (DeepDebug Refine style: "the root cause is
  localized, do not second-guess it"); disable with ``refine=false`` (when
  disabled, fields are filled mechanically from event lines);
* **agent fallback** [adaptation]: when s* lands on an environment-side event
  (env/verifier), the responsible agent is taken from the nearest agent action
  event before s* (in the paper every log entry is an agent action, so this
  issue does not arise); AGENT_MESSAGE counts as an agent-side action — read
  its agent directly. Boundary [declared]: an all-lower convergence pins
  s*=0 on the first event (TASK_START; agent "env" in the sandbox) — with no
  earlier event to walk back to, A* is read directly from ``events[0].agent``,
  which may legitimately be an environment-side agent;
* **answer parsing** [adaptation]: the paper assumes bare answers, but real
  judges occasionally negate ("no error in the lower half") — negated halves
  are flipped by a **clause-scoped** negation-token check (not/no/nothing/
  none/without/cannot/clean/correct plus n't contractions): the answer is
  split into clauses on ``[,;:.!?\\n]`` and the negation tokens are matched
  only inside the clause that names the half, so a qualifier in another
  clause ("I cannot be fully certain, but the error is in the lower half")
  does not flip the answer; a flip is recorded in the round log as
  ``negated: true``; an answer naming neither or both halves raises
  :class:`LLMError` (LLM-parse-failure taxonomy, consistent with
  llm.base.parse_structured; not client-retryable — the round call is
  schema-less, so the OpenAI client's repair loop does not apply and the
  attribution run aborts with an explicit error, no silent degradation);
* the interval wording of the prompt is rebuilt following the G.2 template
  (the paper's placeholders are not shown with a filled example); round
  segment rendering consumes the SSF folding view (full view when SSF is not
  configured).

Trigger semantics: by default only failed trajectories are attributed (same
as all_at_once).
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.classify.taxonomy import MAST_MODES  # shared vocabulary (invariant exception)
from atap.core.registry import register
from atap.core.render import (
    TRACE_BEGIN,
    TRACE_END,
    judge_view,
    render_event_line,
)
from atap.core.schema import Hypothesis
from atap.llm.base import LLMError


class BinaryRefine(BaseModel):
    reason: str = Field(description="Root-cause explanation of why this step is the decisive error")
    fix_suggestion: str = Field(description="An actionable fix suggestion (injected into the recovery)")
    confidence: float = Field(ge=0.0, le=1.0)
    failure_mode: str | None = Field(
        default=None, description="Best-matching MAST code (e.g., FM-1.3); may be empty if uncertain"
    )


_ROUND_SYSTEM = (
    "You are an assistant analyzing multi-agent collaboration logs. Given a task "
    "and a segment of the failure log, decide whether the most critical error is "
    "more likely located in the upper half or the lower half of the current "
    "interval. Output only 'upper half' or 'lower half', nothing else."
)
_REFINE_SYSTEM = (
    "You are an expert in multi-agent system failure attribution. The decisive "
    "error step has already been locked in by binary-search localization "
    "(step {step}, agent {agent}) -- do not change or question that "
    "localization. Based on the full trajectory, explain why this step is the "
    "decisive error (the earliest failure-flipping error, not the step where "
    "the symptom manifests), and give an actionable fix suggestion. You may "
    "consult the MAST failure mode codes to help categorize."
)


#: negation tokens in a judge answer (word-bounded; \w+n't catches isn't /
#: doesn't / wasn't ...) — "no error in the lower half" means the upper half;
#: matched only inside the clause that names the half (see _CLAUSE_SPLIT_RE)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|nothing|none|without|cannot|clean|correct)\b|\w+n't",
    re.IGNORECASE,
)

#: clause boundaries for negation scoping — a negation token flips the half
#: only when it sits in the same clause ("no error in the lower half"), not in
#: a qualifying clause of its own ("I cannot be fully certain, but the error
#: is in the lower half" stays a plain lower-half answer)
_CLAUSE_SPLIT_RE = re.compile(r"[,;:.!?\n]")


def _parse_half(text: str) -> tuple[str, bool]:
    """Plain-text parsing (paper G.2: output should only be 'upper half'/'lower half').

    A negated half flips to the opposite half ("not in the lower half" /
    "the lower half looks clean" → upper) [adaptation: the paper assumes bare
    answers]. Negation is **clause-scoped**: the text is split on
    ``[,;:.!?\\n]`` and the negation tokens are matched only inside the clause
    naming the half — "There is no doubt: ... upper half" / "Nothing
    conclusive; lower half" are not flipped. Returns ``(half, negated)`` so
    the caller can leave a ``negated: true`` trail in the round log; an
    answer naming neither or both halves is unparseable and raises LLMError
    (parse-failure taxonomy; not silently coerced).
    """
    low = text.lower()
    has_upper = "upper" in low
    has_lower = "lower" in low
    if has_lower == has_upper:
        raise LLMError(
            f"Unparseable binary-search answer (expected upper/lower half): {text[:120]!r}"
        )
    word = "lower" if has_lower else "upper"
    clause = next(
        (c for c in _CLAUSE_SPLIT_RE.split(low) if word in c),
        low,
    )
    negated = _NEGATION_RE.search(clause) is not None
    if has_lower:
        return ("upper half" if negated else "lower half"), negated
    return ("lower half" if negated else "upper half"), negated


@register
class BinarySearchAttributor(Attributor):
    stage = "attribute"
    name = "binary_search"
    requires = (("represent", "canonical_events"),)   # bisection walks the R0 event stream

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        if bundle.succeeded and not self.param("include_success", False):
            return
        if ctx.llm is None:
            raise RuntimeError("binary_search requires an LLM client (RunContext.llm)")

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
                        f"Task: {bundle.trajectory.task}\n"
                        f"Below is a segment of the failure log (steps {low}–{mid}; the full log has {n} steps):\n"
                        f"{TRACE_BEGIN}\n{segment}\n{TRACE_END}\n"
                        f"The current search interval is steps {low}–{high}: 'lower half' means steps {low}–{mid}, "
                        f"'upper half' means steps {mid + 1}–{high}. "
                        "Is the error more likely in the upper half or the lower half? Output only 'upper half' or 'lower half'."
                    ),
                },
            ]
            result = ctx.llm.complete(messages, tag=self.name)
            answer, negated = _parse_half(result.text)
            round_rec: dict = {
                "interval": [low, high], "shown": [low, mid], "answer": answer
            }
            if negated:
                # negation-flip audit trail (clause-scoped detection)
                round_rec["negated"] = True
            rounds.append(round_rec)
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
            reason = f"Binary-search localization converged on step {s_star} ({ev.kind} event from {responsible})"
            fix = f"Re-examine the reasoning behind step {s_star}."
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
                    f"The task and the full failure trajectory are as follows:\n{judge_view(bundle)}"
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
        """A* = the acting agent of l_{s*}; when it lands on an environment-side
        event (TOOL_RESULT/TASK_*), fall back to the nearest agent action event
        before it [adaptation]. AGENT_MESSAGE is an agent-side action (the sent
        content is itself an information artifact) — read its agent directly.

        Boundary [declared]: an all-lower convergence pins s*=0 on the first
        event (typically TASK_START) — there is no earlier event to walk back
        to, so A* is read directly from ``events[0].agent`` (in the sandbox
        TASK_START is env-owned, so this may legitimately attribute to "env";
        traces whose first event carries a real owner get that agent)."""
        acting = {"LLM_CALL", "TOOL_CALL", "HANDOFF", "AGENT_MESSAGE"}
        ev = events[s_star]
        if ev.kind in acting:
            return ev.agent
        if s_star == 0:
            # no walk-back target exists: read the first event's own agent
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
