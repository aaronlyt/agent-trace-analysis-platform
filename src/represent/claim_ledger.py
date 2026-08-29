"""R3 claim ledger -- DRIFT, arXiv:2606.02060 section 4-A (the 6-tuple of Eq. 2).

Mechanism (original Prompt 2, appendix G): an LLM performs a **global
single pass** over the complete ordered trajectory for extraction (not
incremental), keeping only decision-critical claims (choosing the answer
path / a unique candidate / verifying hard constraints / supporting the
final answer); queries and tool calls themselves do not count as
commitments; the ledger stays compact (the original prefers 3-5 entries);
``finalized`` is used only for submissions/the final answer.

The 6-tuple c_k=(a_k, i_k, b_k, U_k, tau_k, sigma_k):
* a=claim text; i=introducing span; b=the span where it first becomes
  consequential; U=set of later reuse spans;
  tau in {entity, constraint, evidence, retrieval, compute, process};
  sigma in {exploratory, tentative, consequential, finalized}.

[adaptation] span=R0 event (DRIFT's span boundaries rely on boundary-signal
segmentation; the sandbox's event boundaries come for free); the output
attaches task_goal and hard_constraints (isomorphic to the original's
Prompt 2). Two narrowings of the original Prompt 2 wording in the
implemented system prompt (declared per audit 2026-08-27 P3): (a)
``finalized`` is scoped to the **finally submitted answer** only; (b) no
dedicated no-answer tracking rule -- declarations such as "no answer /
nothing found" enter the ledger only through the generic decision-critical
wording (the Tracer side keeps "a final answer or no-answer" as a span
class; see attribute/claim_audit). b_k/U_k are validated like i_k: clamped
into the trajectory and never earlier than i_k (the i_k <= b_k invariant).
Artifacts (``represent/claim_ledger``) are consumed by name by
attribute/claim_audit.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.core.registry import register
from atap.core.render import judge_view
from atap.represent.base import Representer

CLAIM_TYPES = ("entity", "constraint", "evidence", "retrieval", "compute", "process")
CLAIM_STATUSES = ("exploratory", "tentative", "consequential", "finalized")


class LedgerClaim(BaseModel):
    id: str = Field(description="claim identifier, e.g. c1")
    text: str = Field(description="claim text (the asserted content)")
    type: str = Field(description=f"claim type, one of {CLAIM_TYPES}")
    status: str = Field(description=f"commitment status, one of {CLAIM_STATUSES}")
    introduced_step: int = Field(ge=0, description="R0 index of the introducing span")
    first_effective_step: int | None = Field(
        default=None, description="R0 index where it first becomes consequential"
    )
    reuse_steps: list[int] = Field(
        default_factory=list, description="list of R0 indexes of later reuse spans"
    )


class Ledger(BaseModel):
    task_goal: str
    hard_constraints: list[str] = Field(default_factory=list)
    claims: list[LedgerClaim] = Field(
        default_factory=list, description="3-5 decision-critical claims"
    )
    notes: str = ""


_SYSTEM = (
    "You are a claim ledger builder for deep-research agent trajectories."
    " Given a task and the complete trajectory, do one global scan and"
    " extract decision-critical claims -- keep only the claims that affect"
    " answer-path selection (selecting a candidate / asserting a key fact /"
    " establishing a hard constraint / supporting the final answer); query"
    " text and tool calls themselves do not count as commitments. Keep the"
    " ledger compact (3-5 entries). The finalized status is used only for"
    " the finally submitted answer. Output JSON."
)


@register
class ClaimLedgerRepresenter(Representer):
    stage = "represent"
    name = "claim_ledger"
    requires = (("represent", "canonical_events"),)   # consumes the flattened R0 event stream

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        if ctx.llm is None:
            raise RuntimeError("claim_ledger requires an LLM client (RunContext.llm)")

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    "The task and trajectory follow; build the claim ledger"
                    f" (claim types {CLAIM_TYPES}; statuses {CLAIM_STATUSES};"
                    " step numbers refer to the [index] at the start of each"
                    " trajectory line):\n"
                    f"{judge_view(bundle, include_outcome=False)}"
                ),
            },
        ]
        result = ctx.llm.complete(messages, schema=Ledger, tag=self.name)
        parsed = result.parsed
        assert isinstance(parsed, Ledger)

        n_events = len(bundle.trajectory.events)
        claims: list[dict] = []
        invalid: list[str] = []
        for c in parsed.claims:
            if c.type not in CLAIM_TYPES or c.status not in CLAIM_STATUSES:
                invalid.append(f"{c.id}:type/status")
                continue
            c.introduced_step = min(max(c.introduced_step, 0), n_events - 1)
            # b_k/U_k are validated the same way as i_k (audit 2026-08-27
            # P3): clamped into the trajectory, and never earlier than i_k
            # (a claim cannot become consequential or be reused before it is
            # introduced -- the i_k <= b_k invariant claim_audit relies on);
            # out-of-range values clamp to the introducing/last event.
            if c.first_effective_step is not None:
                c.first_effective_step = min(
                    max(c.first_effective_step, c.introduced_step), n_events - 1
                )
            c.reuse_steps = sorted({
                min(max(s, c.introduced_step), n_events - 1) for s in c.reuse_steps
            })
            claims.append(c.model_dump())
        if len(claims) > 6:
            claims = claims[:6]   # compact ledger cap (original prefers 3-5, 1 slack allowed)

        bundle.put(
            "represent",
            self.name,
            {
                "task_goal": parsed.task_goal,
                "hard_constraints": parsed.hard_constraints,
                "claims": claims,
                "invalid": invalid,
                "n_claims": len(claims),
                "source": "llm_global_pass",
            },
        )
