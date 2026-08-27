"""R4 hierarchy tree representation -- rebuild the flat event stream into a nested scope tree (deterministic, zero LLM).

Mechanism (CodeTracer, arXiv:2604.11641 Tree Indexing, §3.1 + appendix D):
* Tree nodes = normalized steps: **exploration** steps (only inspect the
  environment, do not modify execution state) are attached as **siblings** of
  the previous step (same parent); **state-changing** steps (induce state
  transitions) are attached as **children** of the previous step -- the
  layering criterion is "state transition" semantics, not agent/task layers.
  Source note: the paper states this rule in two forms -- §3.1 prose and the
  Fig.4 caption say exploration steps remain "under the current state node"
  while state-changing steps "induce transitions to child states", whereas
  the appendix diagnosis prompt operationalizes it as "change steps are
  CHILDREN of the previous step; explore steps are SIBLINGS of the previous
  step (same parent)". This implementation follows the appendix rule (the
  convention the original paper's own judge is taught to navigate by); the
  two forms coincide for homogeneous runs and diverge when exploration and
  state-changing steps alternate (a state change after an exploration nests
  under the exploration node, deepening the tree roughly once per
  alternation). The compression in tree.md (one line per step) comes from
  folding environment observations into step summaries, not from the
  hierarchy itself;
* Nodes carry intent+outcome summaries; stage is an orthogonal dimension
  (ordered stage intervals);
* ``tree.md``: one step_id + short summary per line, indentation encodes
  parent-child, stage boundary markers; used as a compressed navigational
  index (the judge first scans the tree to lock onto suspicious stages, then
  drills down; full scans are forbidden). Ablation in the original paper:
  tree index +18.3pt (average over three backbones), tokens actually drop
  (105.1k -> 56.8k).

Domain mapping [adaptation: the original domain is SWE (edit/install/verify
commands); for the research-QA sandbox, "state change" is defined as --
producing a new task-level information artifact or a submission: LLM_CALL
(plan/conclusion assertions), HANDOFF (control + message transfer), submit
commit; "exploration" = read-only tool calls on the environment such as
search/read_doc (configurable via exploration_actions). TOOL_RESULT/
VERIFIER/TASK_* are environment-side observations, attached to the summary of
the step they belong to, not made into independent nodes (corresponding to
the outcome dimension of the original's steps; several observations on one
step are joined with ``" | "`` — previously only the first survived). Steps
outside every stage interval (no R0 phase) render an explicit
``== stage: ? ==`` marker rather than ``== stage: None ==``. Stage
segmentation is also
[adapted]: the original's five stages are for the SWE domain, so this domain
uses the R0 ``phase`` field directly (sandbox plan/search/report)]. Node
summaries use deterministic templates (the idea of truncating the first line
of the command) -- the original does not specify how summaries are generated;
the pseudo-judge/offline path must be deterministic, and real LLM summaries
are left to be made configurable when needed. The trailing ``== terminal ==``
line of tree.md is new in this implementation (aligned with judge_view's
terminal-state marker; the tree.md format in the original's appendix D has no
such item) [adaptation].

Artifacts (``represent/hierarchy_tree``):
``{nodes, stage_ranges, tree_md, stats}``. Diagnostic consumption
(attribute/tree_diagnosis, 4b): locate the suspicious stage via tree_md ->
map the interval via stage_ranges -> render only that interval's events for
drill-down.
"""

from __future__ import annotations

from typing import Any

from atap.core.registry import register
from atap.represent.base import Representer

# Agent-initiated kinds that take part in the tree (other events are environment observations, merged into step summaries)
_STEP_KINDS = ("LLM_CALL", "TOOL_CALL", "HANDOFF")
_MAX_SUMMARY = 60


def _summary(ev) -> str:
    """Deterministic intent summary: agent + action + first payload item, truncated."""
    bits = []
    for key in ("query", "doc_id", "answer", "to", "content"):
        if ev.payload.get(key):
            bits.append(f"{key}={str(ev.payload[key])[:30]}")
            break
    head = f"{ev.agent} {ev.action or ev.kind.lower()}"
    return (head + (" " + " ".join(bits) if bits else ""))[:_MAX_SUMMARY]


@register
class HierarchyTreeRepresenter(Representer):
    stage = "represent"
    name = "hierarchy_tree"

    #: Actions that only read the environment without changing task state -> exploration (sibling nodes) [adaptation]
    DEFAULT_EXPLORATION_ACTIONS = ("search", "read_doc")

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        exploration_actions = set(
            self.param("exploration_actions", self.DEFAULT_EXPLORATION_ACTIONS)
        )

        nodes: list[dict[str, Any]] = []
        parent_of: dict[str, str | None] = {}
        prev_step: str | None = None
        outcome_by_step: dict[str, list[str]] = {}

        # Environment observations attach to the nearest initiating step (outcome dimension)
        for ev in events:
            if ev.kind in _STEP_KINDS:
                is_exploration = (
                    ev.kind == "TOOL_CALL" and (ev.action or "") in exploration_actions
                )
                parent = parent_of.get(prev_step) if is_exploration else prev_step
                parent_of[ev.id] = parent
                nodes.append(
                    {
                        "step": ev.id,
                        "index": ev.index,
                        "parent": parent,
                        "class": "exploration" if is_exploration else "state_changing",
                        "summary": _summary(ev),
                    }
                )
                prev_step = ev.id
            elif prev_step is not None:
                content = str(ev.payload.get("content", ""))[:40]
                if content:
                    outcome_by_step.setdefault(prev_step, []).append(content)

        for n in nodes:
            outs = outcome_by_step.get(n["step"]) or []
            if outs:
                # several observations may attach to one step: join them
                # (previously only the first survived)
                n["summary"] = (n["summary"] + " -> " + " | ".join(outs))[
                    :_MAX_SUMMARY + 40
                ]

        stage_ranges = self._stage_ranges(events)
        tree_md = self._render_md(nodes, stage_ranges, events)

        bundle.put(
            "represent",
            self.name,
            {
                "nodes": nodes,
                "stage_ranges": stage_ranges,
                "tree_md": tree_md,
                "stats": {
                    "n_events": len(events),
                    "n_steps": len(nodes),
                    "n_exploration": sum(
                        1 for n in nodes if n["class"] == "exploration"
                    ),
                    "n_state_changing": sum(
                        1 for n in nodes if n["class"] == "state_changing"
                    ),
                    "n_stages": len(stage_ranges),
                    "tree_md_lines": tree_md.count("\n"),
                    "exploration_actions": sorted(exploration_actions),
                },
                "cost": "free",
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _stage_ranges(events) -> list[dict[str, Any]]:
        """Split contiguous intervals by the R0 phase field (events without a
        phase fall outside adjacent intervals; no interval is forced --
        sandbox TASK_*/VERIFIER have no phase)."""
        ranges: list[dict[str, Any]] = []
        for ev in events:
            if ev.phase is None:
                continue
            if ranges and ranges[-1]["stage"] == ev.phase:
                ranges[-1]["end"] = ev.index
            else:
                ranges.append({"stage": ev.phase, "start": ev.index, "end": ev.index})
        return ranges

    @staticmethod
    def _render_md(
        nodes: list[dict[str, Any]],
        stage_ranges: list[dict[str, Any]],
        events,
    ) -> str:
        """tree.md: indentation encodes hierarchy + stage boundary markers (one step + summary per line)."""
        depth_of: dict[str, int] = {}

        def depth(step: str | None) -> int:
            if step is None:
                return 0
            if step not in depth_of:
                p = next(n["parent"] for n in nodes if n["step"] == step)
                depth_of[step] = depth(p) + 1
            return depth_of[step]

        stage_at = {}
        for r in stage_ranges:
            for i in range(r["start"], r["end"] + 1):
                stage_at[i] = r["stage"]
        index_by_step = {n["index"]: n for n in nodes}

        lines: list[str] = ["# trace tree"]
        for r in stage_ranges:
            lines.append(f"# stage {r['stage']}: steps [{r['start']}..{r['end']}]")
        current_stage: str | None = None
        for i in sorted(index_by_step):
            st = stage_at.get(i)
            if st != current_stage:
                current_stage = st
                # a step outside every stage interval (no R0 phase) renders
                # an explicit unknown marker instead of "== stage: None =="
                lines.append(f"== stage: {st if st is not None else '?'} ==")
            n = index_by_step[i]
            lines.append(f"{'  ' * depth(n['step'])}{n['step']} {n['summary']}")
        # Terminal-state line (the judge needs to know how the task ended; aligned with judge_view's outcome line)
        last = events[-1]
        lines.append(f"== terminal == index {last.index} {last.kind}")
        return "\n".join(lines)
