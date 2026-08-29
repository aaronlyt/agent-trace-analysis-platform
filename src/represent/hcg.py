"""R2+ hierarchical causal graph (HCG) -- CHIEF, arXiv:2602.23701 section 4.1 (deterministic construction).

Mechanism (original paper): G=(V,E), V=V_sub union V_agt:
* **subtask nodes**: high-level logical abstractions of the task (in the
  original, RAG retrieval of 2 exemplars for few-shot decomposition +
  Trajectory-Aligned Reflection correction to prevent hallucination);
* **agent nodes**: atomic execution units within a subtask, with the OTAR
  tuple <Observation, Thought, Action, Result> as attributes (extending the
  TAR schema, section 4.1.1);
* Edges E=E_sub union E_agt union E_step: subtask edges (logical progression
  between adjacent subtasks), agent edges (collaboration), step edges (data
  flow snapshots: upstream output <-> downstream input, section 4.1.2);
  subtask/agent edges are bound to the counterfactual pattern
  Phi: Bias(u) -> Phi Anomaly(v).

[adaptation] The toy domain's three-layer graph can be built
**deterministically** (the original is fully LLM-driven, with construction
folded into its 2.5-3x token cost): subtasks come directly from R0 ``phase``
(the sandbox's plan/search/report is itself the task decomposition -- the
original's Reflection correction aims to align with logs, and phase already
comes from log structure); OTAR comes from rule-based event mapping
(TOOL_RESULT->Observation, LLM_CALL->Thought, TOOL_CALL->Action,
VERIFIER->Result); E_step reuses R0 refs (same basis as represent/idg);
E_sub=adjacent phase progression; E_agt=HANDOFF edges. Two deviations of the
OTAR mapping must be declared: (a) the original's Result is the agent's own
output (Fig.7 "Summarize each agent's behavior into O/T/A/R"), while this
domain records environment-side VERIFIER feedback into the result slot, and
the agent's own output (AGENT_MESSAGE/HANDOFF content) is not merged in --
so most agent nodes have result=None; (b) each slot stores only the index of
the first matching event (the original stores LLM-summarized behavior text).
E_step needs its own named declaration (not covered by the blanket
"deterministic construction" disclaimer): the original's Fig.9 prompt has an
LLM select only **"meaningful data passing"** step edges (a filtered subset;
zero/one/multiple edges per step are all allowed), while this implementation
takes the **entire set** of R0 refs -- an unfiltered superset that treats
every reference edge as meaningful [adaptation: E_step superset], with only
the represent/idg-basis mechanics applied (time-order guard: forward refs
dropped; per-pair deduplication). None of
these edges is pruned afterwards (see the consumption note below).
Phi pattern annotation is left as a placeholder (the original uses an LLM to
judge Bias->Anomaly): ``phi_patterns`` stays empty and **nothing currently
annotates it** — chief's screening never reads the placeholder (see the
consumption note below). The real LLM construction path
(few-shot decomposition + OTAR extraction) will be added later during
validation with a real model [declared].

Artifacts (``represent/hcg``): **only the ``subtasks`` intervals are
actually consumed** (by name) — attribute/chief uses them as the subtask
segmentation that seeds the failing-subtask backtracking start point;
``agent_nodes`` (OTAR slots), ``edges`` (E_sub/E_agt/E_step) and
``phi_patterns`` are currently **consumed by no prompt and no algorithm**
[adaptation: mechanism gap — the paper's Fig.11 backtracking prompt takes the
causal graph as the {graph} input; this reproduction never feeds the graph
into any judge, see the chief module docstring].
"""

from __future__ import annotations

from typing import Any

from atap.core.registry import register
from atap.represent.base import Representer

_OTAR_KIND = {
    "TOOL_RESULT": "observation",
    "LLM_CALL": "thought",
    "TOOL_CALL": "action",
    "VERIFIER": "result",
}


@register
class HCGRepresenter(Representer):
    stage = "represent"
    name = "hcg"
    requires = (("represent", "canonical_events"),)   # consumes the flattened R0 event stream

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )

        # ---- subtask layer: contiguous R0 phase intervals (TASK_*/VERIFIER
        # events without a phase are merged into the current subtask; the
        # leading TASK_START etc. are backfilled after the first phase appears) ----
        subtasks: list[dict[str, Any]] = []
        for ev in events:
            if ev.phase is not None:
                if not subtasks or subtasks[-1]["phase"] != ev.phase:
                    subtasks.append({
                        "id": f"S{len(subtasks)}",
                        "phase": ev.phase,
                        "start": (subtasks[-1]["end"] + 1) if subtasks else 0,
                        "end": ev.index,
                        "agents": [],
                    })
                else:
                    subtasks[-1]["end"] = ev.index
            elif subtasks:
                subtasks[-1]["end"] = ev.index
        if not subtasks:
            subtasks.append({
                "id": "S0", "phase": "task", "start": 0,
                "end": events[-1].index, "agents": [],
            })

        # ---- agent layer: subtask x agent OTAR tuples ----
        # TOOL_RESULT is an env event in R0 -- the observation is attributed
        # along refs to the agent that initiated the call (CHIEF's OTAR is
        # the agent's own observation)
        agent_nodes: list[dict[str, Any]] = []
        sub_by_idx: dict[int, str] = {}
        for s in subtasks:
            for i in range(s["start"], s["end"] + 1):
                sub_by_idx[i] = s["id"]
        agent_by_id = {ev.id: ev.agent for ev in events}

        def _node(sid: str, agent: str) -> dict[str, Any]:
            node = next(
                (n for n in agent_nodes
                 if n["subtask"] == sid and n["agent"] == agent),
                None,
            )
            if node is None:
                node = {
                    "subtask": sid, "agent": agent,
                    "otar": {"observation": None, "thought": None,
                             "action": None, "result": None},
                    "steps": [],
                }
                agent_nodes.append(node)
                s = next(x for x in subtasks if x["id"] == sid)
                if agent not in s["agents"] and agent != "env":
                    s["agents"].append(agent)
            return node

        for ev in events:
            sid = sub_by_idx.get(ev.index, subtasks[-1]["id"])
            otar_role = _OTAR_KIND.get(ev.kind)
            if otar_role is None:
                continue    # TASK_*/HANDOFF are not agent execution units
            if ev.kind == "TOOL_RESULT":
                owner = (
                    agent_by_id.get(ev.refs[0]) if ev.refs else "env"
                )
            elif ev.kind == "VERIFIER":
                owner = (
                    agent_by_id.get(ev.refs[0]) if ev.refs else "verifier"
                )
            else:
                owner = ev.agent
            if owner in ("env", "verifier", None):
                continue    # environment-side observations do not form agent execution units (same exclusion scope as SBFL)
            node = _node(sid, owner)
            if node["otar"][otar_role] is None:
                node["otar"][otar_role] = ev.index
            node["steps"].append(ev.index)

        # ---- the three edge kinds ----
        sub_edges = [
            {"src": subtasks[i]["id"], "dst": subtasks[i + 1]["id"]}
            for i in range(len(subtasks) - 1)
        ]
        agt_edges: list[dict[str, Any]] = []
        for ev in events:
            if ev.kind == "HANDOFF":
                to = str(ev.payload.get("to", ""))
                sid = sub_by_idx.get(ev.index, subtasks[-1]["id"])
                agt_edges.append({
                    "src": ev.agent, "dst": to or "?",
                    "subtask": sid, "step": ev.index,
                })
        # Step edges reuse R0 refs on the same basis as represent/idg: the
        # time-order guard (index[src] < index[dst]) plus deduplication --
        # a forward-pointing ref (if a collection layer ever emits one) is
        # dropped instead of becoming a backward data-flow edge, and a
        # repeated ref id yields one edge
        index_by_id = {ev.id: ev.index for ev in events}
        step_edges: list[dict[str, Any]] = []
        seen_step: set[tuple[str, str]] = set()
        for ev in events:
            for ref in ev.refs:
                if ref in index_by_id and index_by_id[ref] < ev.index:
                    key = (ref, ev.id)
                    if key not in seen_step:
                        seen_step.add(key)
                        step_edges.append({"src": ref, "dst": ev.id})

        bundle.put(
            "represent",
            self.name,
            {
                "subtasks": subtasks,
                "agent_nodes": agent_nodes,
                "edges": {"sub": sub_edges, "agt": agt_edges, "step": step_edges},
                "phi_patterns": [],   # Bias->Phi Anomaly annotation: approximated during chief screening
                "stats": {
                    "n_subtasks": len(subtasks),
                    "n_agent_nodes": len(agent_nodes),
                    "n_sub_edges": len(sub_edges),
                    "n_agt_edges": len(agt_edges),
                    "n_step_edges": len(step_edges),
                },
                "construction": "deterministic_r0 [adaptation: original is fully LLM-driven]",
                "cost": "free",
            },
        )
