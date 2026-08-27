"""R2 information dependency graph (IDG) -- a trajectory-level usage graph of information artifacts (deterministic, zero LLM).

Idea source: GraphTracer, arXiv:2510.10581 (retracted 2025-12; **absorb only
the graph-model idea and trust none of its numbers** -- TraceElephant A.6.3
also reports being unable to reproduce its results):

* Node = one information artifact produced during execution
  v=(t_v, mu_v, o_v) (time step / generating agent / the observation or
  conclusion itself); source nodes (in-degree 0) = initial observations such
  as search results and tool outputs; derived nodes = conclusions synthesizing
  upstream artifacts;
* Edge (v_i, v_j) iff o_vj **explicitly references** o_vi and t_vi < t_vj (a
  usage relation);
* Root-cause backtracking **follows dependency edges rather than time steps**
  (to solve "symptom at step 40, root cause at step 3"): take the reverse
  ancestor closure along in-edges from the failing terminal event; candidate
  ranking Impact(v)=alpha*deg+(v)+(1-alpha)*Betweenness(v)
  ([paper unspecified: the value of alpha -- the original only defines alpha,
  and the section 5.7 hyperparameter analysis reports only lambda~0.5 and
  sigma~1.0-1.5] -- ranking belongs to the consumer side, see the end of the
  docstring; this module only builds the graph).

[adaptation] GraphTracer's citation extraction is an after-the-fact
reconstruction via "structured-output pattern matching / unstructured
auxiliary LLM"; atap's R0 collection layer already lands semantic reference
edges in ``TraceEvent.refs`` (span-id -> event-id mapped by
canonical_events), so graph construction is purely deterministic O(V+E) set
operations with no LLM. The original paper's empirical scale is
|V|~0.5T and |E|~2.5|V| (most raw actions are mere
coordination/formatting and their outputs are never referenced) -- R0 events
almost all carry a payload (coordination messages are also information
artifacts), so node_ratio in this domain will be higher than that empirical
value; it is recorded faithfully in stats and is not an acceptance criterion.
Conflict edges c_ij (two sources sharing a descendant contradict each other):
the original section 4.2 describes the mechanism (pairwise consistency checks
over nodes sharing a descendant, with entity/attribute claim contradictions as
the criterion; the checker implementation is unspecified) -- semantic
contradiction judgment exceeds what an offline deterministic framework can do,
so it is left as an empty placeholder, [weakened declaration].

Artifacts (``represent/idg``): ``{nodes, edges, conflicts, stats}``;
nodes are **pruned triples** — each node stores only
``{event_id, index, agent, kind}``, i.e. t_v and mu_v of the original Eq(5)
triple v=(t_v, mu_v, o_v) plus a kind tag; **o_v (the observation/conclusion
itself) is not duplicated into the graph** (payloads live on the trajectory
events; a consumer that needs o_v must look the payload up via ``event_id``
[adaptation: o_v pruning — keeps the artifact JSON-serializable and free of
duplicated content]) [declared]. Attribution-side consumption convention
(for attribute-side algorithms to consume via artifacts, without importing
this module):
``ancestors(v)`` = reverse BFS along edges; ``impact`` ranking uses the
formula above.

Consumption honesty [declared, same style as hcg's mechanism-gap note]:
**no algorithm or prompt in the current repository actually consumes the
``ancestors(v)`` backtracking or the ``impact`` ranking convention** — the
attribution layer localizes through its own algorithms and never walks
these dependency edges; the artifact's ``edges`` are read only by
structural tests. The convention above is a placeholder for future
consumers, not an exercised mechanism. Malformed forward references (a ref
whose target event exists but is not earlier) are dropped from the graph
and counted in ``stats.dropped_forward_refs`` rather than silently
vanishing.
"""

from __future__ import annotations

from typing import Any

from atap.core.registry import register
from atap.represent.base import Representer


@register
class IDGRepresenter(Representer):
    stage = "represent"
    name = "idg"

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )

        # Information-artifact nodes: events carrying a payload
        # (observations/conclusions/messages/call parameters -- a zero-argument
        # call is still an "intent artifact", likewise a malformed call) or
        # having reference edges. Pure marker events with neither (e.g.
        # TASK_END) do not enter the graph.
        # [adaptation: node admission] The original Eq(11) admits only
        # "information artifacts referenced later" (only information that
        # influences downstream reasoning counts as a node); this domain
        # relaxes it to payload OR refs (isolated source nodes also enter the
        # graph), otherwise malformed calls/zero-argument events would be
        # silently dropped -- the consequence is node_ratio higher than the
        # original's |V|~0.5T empirical value, see stats (not an acceptance
        # criterion).
        nodes = [
            {"event_id": ev.id, "index": ev.index, "agent": ev.agent, "kind": ev.kind}
            for ev in events
            if ev.payload or ev.refs
        ]
        node_ids = {n["event_id"] for n in nodes}

        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        dropped_forward = 0
        index_by_id = {ev.id: ev.index for ev in events}
        for ev in events:
            if ev.id not in node_ids:
                continue
            for ref in ev.refs:
                # usage edge direction: referenced (upstream) -> referencing
                # (downstream); time-order guard (refs semantically point to
                # earlier artifacts; the original's t_vi<t_vj condition is
                # kept defensively). A ref that points at a non-earlier event
                # (a malformed forward reference) is dropped and **counted**
                # (stats.dropped_forward_refs) instead of silently vanishing
                if ref in node_ids and index_by_id[ref] < ev.index:
                    key = (ref, ev.id)
                    if key not in seen:
                        seen.add(key)
                        edges.append({"src": ref, "dst": ev.id})
                elif ref in index_by_id and index_by_id[ref] >= ev.index:
                    dropped_forward += 1

        in_deg = {n["event_id"]: 0 for n in nodes}
        for e in edges:
            in_deg[e["dst"]] += 1
        n_sources = sum(1 for d in in_deg.values() if d == 0)

        bundle.put(
            "represent",
            self.name,
            {
                "nodes": nodes,
                "edges": edges,
                "conflicts": [],
                "stats": {
                    "n_events": len(events),
                    "n_nodes": len(nodes),
                    "n_edges": len(edges),
                    "dropped_forward_refs": dropped_forward,
                    "n_sources": n_sources,
                    "node_ratio": round(len(nodes) / len(events), 4) if events else 0.0,
                    "edges_per_node": round(len(edges) / len(nodes), 4) if nodes else 0.0,
                    "reference_scale": "|V|~0.5T, |E|~2.5|V| (GraphTracer empirical values, for reference only, not an acceptance criterion)",
                },
                "cost": "free",
                "source": "R0 TraceEvent.refs",
            },
        )
