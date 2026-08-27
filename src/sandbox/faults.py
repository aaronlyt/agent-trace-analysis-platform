"""Fault injection library -- injectable implementations of six literature
fault modes inside the sandbox.

Aligned with the fault-injection data-generation idea (AgenTracer 2509.03312
route B / Aegis-Kong 2509.14295: labels known by construction). Each fault:
* changes agent behavior at some **logical step** of the deterministic
  scripted rollout;
* produces an observable symptom (visible to the judge) and maps to one MAST
  failure mode;
* meta["injected_fault"] records the ground truth (kind / onset event index /
  agent / MAST code) -- for evaluation assertions only; judge-style
  algorithms never read it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    agent: str          # responsible agent
    mast_code: str      # corresponding MAST failure mode (classification ground truth)
    onset_logical: str  # logical step name where the deviation starts in the script
    description: str


FAULTS: dict[str, FaultSpec] = {
    f.kind: f
    for f in [
        FaultSpec(
            kind="step_repetition", agent="searcher", mast_code="FM-1.3",
            onset_logical="search#1",
            description="searcher repeats the same search call with no progress until the budget is exhausted (MAST FM-1.3 / target symptom of TraceProbe loop predicates)",
        ),
        FaultSpec(
            kind="malformed_tool_call", agent="searcher", mast_code="FM-2.6",
            onset_logical="search",
            description=(
                "searcher issues a malformed tool call missing arguments and the "
                "environment returns an error (primary target symptom of the "
                "AgentDebugX free rule pack; the 14 MAST modes have no dedicated "
                "tool-format class, closest match mapped to FM-2.6 reasoning-action "
                "mismatch [adaptation])"
            ),
        ),
        FaultSpec(
            kind="info_withholding", agent="searcher", mast_code="FM-2.4",
            onset_logical="handoff_report",
            description="searcher retrieves documents but falsely reports finding nothing to the reporter (MAST FM-2.4)",
        ),
        FaultSpec(
            kind="premature_termination", agent="planner", mast_code="FM-3.1",
            onset_logical="plan",
            description=(
                "planner submits the answer from memory without searching "
                "(MAST FM-3.1; onset = the planning step that decides to skip "
                "retrieval -- the earliest decisive error per Who&When Eq.5, one "
                "step before the submit termination action)"
            ),
        ),
        FaultSpec(
            kind="ungrounded_citation", agent="reporter", mast_code="FM-3.3",
            onset_logical="compose",
            description=(
                "reporter cites a document that was retrieved but never read "
                "(target symptom of unsupported claims in DRIFT; verification was "
                "done yet an unread document is treated as verified evidence, "
                "MAST FM-3.3)"
            ),
        ),
        FaultSpec(
            kind="disobey_task_spec", agent="reporter", mast_code="FM-1.1",
            onset_logical="compose",
            description="reporter answer content is correct but violates the task spec (the answer is missing the required citation of a read document id, MAST FM-1.1)",
        ),
    ]
}

TOOL_BUDGET = 4  # a normal rollout needs only 3 tool calls; the repetition fault breaks the budget

# ---------------------------------------------------------------------------
# Experimental extended faults (phase four). Registered independently of the
# FAULTS registry: generate_corpus iterates only the six FAULTS kinds; this
# group leaves the existing spectrum corpus and acceptance numbers unchanged
# and is injected by name in dedicated scenarios.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtraFaultSpec(FaultSpec):
    """Extended fault. mast_code may be None = no counterpart among the 14
    MAST modes (residual scenario, inducer target)."""


EXTRA_FAULTS: dict[str, ExtraFaultSpec] = {
    f.kind: f
    for f in [
        ExtraFaultSpec(
            kind="retrieval_detour", agent="searcher", mast_code="FM-2.3",
            onset_logical="search",
            description=(
                "searcher drifts off the task topic with a generic query: it hits "
                "evidence documents yet never retrieves the gold doc (target "
                "scenario of RG last-hop, 2608.01913; closest MAST match mapped "
                "to FM-2.3 deviation from task focus [adaptation])"
            ),
        ),
        ExtraFaultSpec(
            kind="agent_deadlock", agent="searcher", mast_code=None,
            onset_logical="clarify#1",
            description=(
                "searcher and planner repeatedly clarify and wait on each other "
                "(3 rounds of identical back-and-forth) until the task stalls -- "
                "no counterpart among the 14 MAST modes; serves as the target "
                "scenario for AgentDebugX inducer residual vocabulary expansion "
                "(the paper lists the same example: multi-agent deadlock)"
            ),
        ),
    ]
}

#: Full fault table (for generate's by-name lookup; corpus generation still iterates FAULTS only)
ALL_FAULTS: dict[str, FaultSpec] = {**FAULTS, **EXTRA_FAULTS}
