"""claim audit attribution — DRIFT, arXiv:2606.02060 §4-B/C + Dependency Tracer.

Mechanism (paper Appendix G pipeline):
* **B Support Seeker**: for every consequential claim, judge its support level
  as DIRECT/WEAK/MISSING/CONFLICTING — high recall, deliberately over-routing
  ("allowed to over-route because C will later verify/filter");
* **C Specialist Auditor**: route specialist auditors by claim type, output
  verdict ∈ {supported, harmful_unsupported_commitment, conflicting_support,
  insufficient_but_nonharmful} and responsible_span (must not pick a pure
  query/tool call/failed search — unless that span itself asserts the claim is
  settled);
* **Dependency Tracer**: **conservatively backtrack** from the introduction
  span of unsupported claims — keep first_error_span unless it is clearly just
  a query/tool/retry; outputs {first_error_span, error_span_ids}.

Final prediction (Eq. 3): Ê={s | h(s)=1} — the set of spans that
commit/reuse/amplify/finalize a harmful claim; the first error = the earliest
span that commits to a harmful claim. By the paper's definitions i_k ≤ b_k
always holds (a claim cannot become consequential before it is introduced),
so "i_k or b_k, whichever is earlier" would be vacuous — the Tracer is
instructed to keep the introduction position unless it is merely a
query/tool/retry, and the run side additionally reconciles the LLM's
first_error_step down to the earliest error_steps member and orders harmful
claims by earliest responsible/introduced step (the paper: harm begins at
commitment, not at exploratory introduction).

[adaptation] B+C are merged into a single structured call (the paper's C
routes claim×auditor individually; merged as an engineering simplification;
C's decisive_defect/failure_mechanism/chain_action/commitment_status fields
are trimmed down to verdict+responsible_step — plan stage four once promised
to introduce them; the trimming is recorded faithfully), and the Tracer is a
second call — 2 LLM calls in total + 1 for the ledger. Other disclaimers:
(1) outcome-line handling (revised per audit 2026-08-27 P1): failed
trajectories keep the outcome line (the paper's input protocol is
question+span text only; this follows the framework's judge_view convention
— the marker is always FAILURE, and env failure notes name neither the gold
answer nor the gold doc), while successful trajectories under
include_success=True render **without** the outcome line: since the review
fix of 2026-08-27 env.verify's success note names neither the gold answer
nor the gold doc ("matches the expected method name"), so neither the
outcome line nor the VERIFIER event line inside tau carries the oracle
(the old wording "matches gold '<answer>' (<doc>)" leaked it into every
judge view); the pseudo-judge's claim_* handlers never read the outcome
line, so offline acceptance is unaffected; (2)
successful trajectories
are skipped by default (the paper reports 36.9% of successful trajectories
still contain process errors, and Case Study 2 is exactly "the answer is right
but the evidence chain is unsupported" — include_success can turn this on; it
defaults to off, following the framework's "attribute only failed
trajectories" convention); (3) the B audit submits all claims rather than
only consequential/finalized ones (the system wording constrains the judge's
focus); (4) Hypothesis confidence=0.7 is a fixed scalar [inference: the
paper's C only outputs low/medium/high tiers, with no scalar confidence];
(5) the ledger claims submitted to both the audit and the Tracer carry
i_k **plus the dependency structure b_k/U_k** (first_effective_step /
reuse_steps from the ledger artifact) — the paper's Appendix G chain
"later spans depend on it" gets its input (previously only introduced_step
was submitted, audit 2026-08-27 P2); the Tracer's exception list (a final
answer or no-answer / the first explicit commitment / an explicit fake
verification / a completed computation/count/source claim) follows the
paper's Prompt 5 span classes; the deterministic pseudo-judge's claim-line
parser does not consume the appended dependency bracket, so offline
acceptance results are unchanged.
Artifacts consume represent/claim_ledger (by name, no import). Hypothesis
mapping: step=first_error_span, agent=that event's agent, root_cause=claim
text + verdict.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.render import judge_view
from atap.core.schema import Hypothesis

SUPPORT_LEVELS = ("DIRECT", "WEAK", "MISSING", "CONFLICTING")
VERDICTS = (
    "supported", "harmful_unsupported_commitment",
    "conflicting_support", "insufficient_but_nonharmful",
)


class SupportRecord(BaseModel):
    claim_id: str
    support_status: str = Field(description=f"one of {SUPPORT_LEVELS}")
    support_steps: list[int] = Field(default_factory=list)
    missing_support: str = ""
    verdict: str = Field(description=f"one of {VERDICTS}")
    responsible_step: int | None = Field(
        default=None,
        description="Responsible span when harmful (must not be a pure query/tool call/failed search)",
    )
    reason: str = ""


class SupportRecords(BaseModel):
    records: list[SupportRecord] = Field(default_factory=list)


class TraceVerdict(BaseModel):
    first_error_step: int = Field(ge=0, description="R0 index of the earliest error span")
    error_steps: list[int] = Field(default_factory=list)
    reason: str


def _claim_line(c: dict) -> str:
    """Render one ledger claim as a judge-visible line.

    The parenthesized block (id/text/type/status/i_k) keeps the format the
    pseudo-judge's claim-line parser expects; the appended bracket carries
    the dependency structure b_k/U_k from the ledger artifact -- the paper's
    Appendix G Tracer needs "later spans depend on it" as input (the
    pseudo-judge does not consume the bracket, so offline acceptance is
    insensitive to it).
    """
    line = (
        f"{c['id']}: {c['text']} (type={c['type']}, status={c['status']}, "
        f"introduced_step={c['introduced_step']})"
    )
    dep = []
    if c.get("first_effective_step") is not None:
        dep.append(f"first_effective_step={c['first_effective_step']}")
    if c.get("reuse_steps"):
        dep.append(f"reuse_steps={c['reuse_steps']}")
    return f"{line} [{', '.join(dep)}]" if dep else line


_B_SYSTEM = (
    "You are a claim support auditor. For every consequential/finalized claim "
    "in the ledger, judge its support level from the trajectory: DIRECT (the "
    "trajectory directly establishes the decisive chain) / WEAK (relevant "
    "evidence exists but the decisive chain is partial/implicit/unverified) / "
    "MISSING (no support shown at all) / CONFLICTING (the shown evidence "
    "contradicts the claim). Also give an audit verdict: supported / "
    "harmful_unsupported_commitment (asserted as fact without support) / "
    "conflicting_support / insufficient_but_nonharmful. responsible_step must "
    "not be a pure query or tool-call line, unless that line itself asserts "
    "the claim is settled. Output JSON."
)
_T_SYSTEM = (
    "You are a dependency backtracer. Given the list of unsupported claims and "
    "the trajectory, conservatively locate the earliest error span: keep the "
    "introduction position of the earliest unsupported claim, unless it is "
    "clearly just a query/tool call/retry/discarded candidate; do not add new "
    "spans unless you are confident and the new span is a final answer or "
    "no-answer, the first explicit commitment, an explicit fake verification, "
    "or a completed computation/count/source claim, and adding it improves "
    "the earliest-error localization. Output JSON."
)


@register
class ClaimAuditAttributor(Attributor):
    stage = "attribute"
    name = "claim_audit"
    requires = (
        ("represent", "canonical_events"),   # audit calls view the R0 stream
        ("represent", "claim_ledger"),   # audits the R3 ledger
    )

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        if not t.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        ledger = bundle.get("represent", "claim_ledger")
        claims = ledger.get("claims") if isinstance(ledger, dict) else None
        if not isinstance(claims, list) or not claims:
            raise ValueError(
                f"{bundle.trace_id} is missing the represent/claim_ledger artifact: "
                "claim_audit consumes the R3 ledger; configure claim_ledger first"
            )
        if bundle.succeeded and not self.param("include_success", False):
            bundle.put(
                "attribute", self.name,
                {"hypotheses": [], "status": "success_no_attribution"},
            )
            return
        if ctx.llm is None:
            raise RuntimeError("claim_audit requires an LLM client (RunContext.llm)")

        n_events = len(t.events)
        # [P1 leak fix, audit 2026-08-27; verifier note itself fixed in the
        # 2026-08-27 review round] A successful trajectory's outcome line
        # used to carry the env verifier note "matches gold '<answer>'
        # (<doc>)" -- under include_success=True that would put the gold
        # answer into the judge prompt. The success path renders without
        # the outcome line; failed trajectories keep it (always FAILURE,
        # and env notes -- success or failure -- name neither the gold
        # answer nor the gold doc).
        view = judge_view(bundle, include_outcome=not bundle.succeeded)
        claims_json = "\n".join(_claim_line(c) for c in claims)
        result = ctx.llm.complete(
            [
                {"role": "system", "content": _B_SYSTEM},
                {"role": "user", "content": (
                    "Claims to audit:\n" + claims_json + "\n\nTask and trajectory:\n" + view
                )},
            ],
            schema=SupportRecords,
            tag=f"{self.name}_support",
        )
        records = result.parsed
        assert isinstance(records, SupportRecords)

        valid: list[dict] = []
        invalid_records: list[dict] = []
        # clamp-with-trace notes (all_at_once / tree_diagnosis discipline):
        # an out-of-range judge step is clamped, but the raw value stays on
        # record instead of being silently rewritten
        clamp_notes: list[str] = []
        ledger_ids = {c["id"] for c in claims}
        for r in records.records:
            if r.support_status not in SUPPORT_LEVELS or r.verdict not in VERDICTS:
                # kept on record rather than silently dropped (same
                # convention as mast_judge's invalid_codes)
                invalid_records.append(
                    {
                        "claim_id": r.claim_id,
                        "support_status": r.support_status,
                        "verdict": r.verdict,
                    }
                )
                continue
            if r.claim_id not in ledger_ids:
                # the audit answered about a claim the ledger never saw —
                # also kept on record in the invalid channel instead of
                # silently dropping (a hallucinated id must not reach the
                # Tracer prompt or the harmful-claim ranking)
                invalid_records.append(
                    {
                        "claim_id": r.claim_id,
                        "support_status": r.support_status,
                        "verdict": r.verdict,
                        "reason": "claim_id not in ledger",
                    }
                )
                continue
            if r.responsible_step is not None:
                clamped_step = min(max(r.responsible_step, 0), n_events - 1)
                if clamped_step != r.responsible_step:
                    clamp_notes.append(
                        f"claim {r.claim_id}: responsible_step "
                        f"{r.responsible_step}->{clamped_step} (judgement clamped)"
                    )
                r.responsible_step = clamped_step
            valid.append(r.model_dump())

        harmful = [
            r for r in valid
            if r["verdict"] in ("harmful_unsupported_commitment", "conflicting_support")
        ]
        if not harmful:
            bundle.put(
                "attribute", self.name,
                {
                    "hypotheses": [],
                    "status": "no_harmful_claim",
                    "support_records": valid,
                    "invalid_support_records": invalid_records,
                    "clamp_notes": clamp_notes,
                },
            )
            return

        harmful_json = "\n".join(
            f"{r['claim_id']}: support={r['support_status']}, verdict={r['verdict']}, "
            f"responsible_step={r['responsible_step']}"
            for r in harmful
        )
        result2 = ctx.llm.complete(
            [
                {"role": "system", "content": _T_SYSTEM},
                {"role": "user", "content": (
                    "Unsupported claims:\n" + harmful_json + "\n\nLedger claims:\n"
                    + claims_json + "\n\nTask and trajectory:\n" + view
                )},
            ],
            schema=TraceVerdict,
            tag=f"{self.name}_trace",
        )
        trace = result2.parsed
        assert isinstance(trace, TraceVerdict)

        step = min(max(trace.first_error_step, 0), n_events - 1)
        if step != trace.first_error_step:
            clamp_notes.append(
                f"first_error_step {trace.first_error_step}->{step} (judgement clamped)"
            )
        # Eq.3 earliest semantics: the reported first error cannot be later
        # than the earliest member of error_steps — reconcile instead of
        # trusting the LLM's field in isolation (e.g. first_error_step=9 with
        # error_steps=[5,9] is self-contradictory and resolves to 5)
        error_steps = []
        for s in trace.error_steps:
            cs = min(max(s, 0), n_events - 1)
            if cs != s:
                clamp_notes.append(f"error_steps {s}->{cs} (judgement clamped)")
            error_steps.append(cs)
        first_error_reconciled = False
        if error_steps and min(error_steps) < step:
            step = min(error_steps)
            first_error_reconciled = True
        ev = t.events[step]
        # earliest harmful claim (by responsible step, falling back to the
        # ledger's introduced step) — not the LLM's list order
        by_id = {c["id"]: c for c in claims}

        def _earliest(r: dict) -> int:
            if r["responsible_step"] is not None:
                return r["responsible_step"]
            return by_id.get(r["claim_id"], {}).get("introduced_step", n_events)

        top_claim = min(harmful, key=_earliest)
        claim_text = by_id.get(top_claim["claim_id"], {}).get("text", "")

        hyp = Hypothesis(
            agent=ev.agent,
            step=step,
            root_cause=(
                f"claim audit: support for '{claim_text[:80]}' is "
                f"{top_claim['support_status']} ({top_claim['verdict']}); "
                "the unsupported claim was subsequently used as fact"
            ),
            root_cause_code=None,
            responsible_side="model",
            evidence=[
                f"[{ev.index}] {ev.agent} {ev.kind} :: "
                f"{str(ev.payload.get('content', ev.payload))[:140]}",
                f"support_records={[(r['claim_id'], r['support_status']) for r in valid]}",
                f"tracer_reason={trace.reason[:120]}",
            ],
            fix_suggestion=(
                "Search for and verify support before introducing the claim, or re-verify it before downstream use"
            ),
            confidence=0.7,
        )
        bundle.put(
            "attribute",
            self.name,
            {
                "hypotheses": [hyp.to_dict()],
                "support_records": valid,
                "invalid_support_records": invalid_records,
                "clamp_notes": clamp_notes,
                "first_error_step": step,
                "first_error_reconciled": first_error_reconciled,
                "error_steps": error_steps,
                "tracer_reason": trace.reason,
            },
        )
