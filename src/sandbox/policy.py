"""Scripted policy + toy sandbox -- deterministic rollout, fault injection and
targeted replay.

The normal rollout is a fixed sequence of logical steps (the
planner→searcher→reporter research QA pipeline); the six faults change
behavior at their respective onset logical steps (see faults.py).

Output shape: a **raw span tree** (nested dicts, with semantic refs and
logical step names), flattened into the R0 event stream by
represent/canonical_events -- the sandbox never emits R0 directly, so the
collection→representation cross-layer contract is genuinely exercised.

Targeted replay (AgentDebug 2509.25370 Algorithm 1 Stage 3): keep the prefix
[0, step) and re-execute from step; if the feedback names the fault type
(e.g. the pseudo-judge's fix_suggestion contains "step_repetition"), the
policy takes the corrected branch (equivalent to removing the fault);
otherwise the fault remains and the rerun keeps failing -- feedback quality
determines the recovery success rate, which is deliberate.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from atap.core.schema import Outcome, TraceEvent, Trajectory
from atap.sandbox import env
from atap.sandbox.faults import ALL_FAULTS, FAULTS, TOOL_BUDGET, FaultSpec


class _Recorder:
    """Records the span tree; also maintains (span_id → event index) in DFS
    order for ground truth."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._n = 0
        self._order: dict[str, int] = {}  # span_id -> DFS ordinal
        self._pending_children: dict[str | None, list[dict]] = {}

    def add(
        self,
        logical: str,
        kind: str,
        agent: str,
        *,
        action: str | None = None,
        payload: dict | None = None,
        refs: list[str] | None = None,
        phase: str | None = None,
        parent: str | None = None,
    ) -> str:
        sid = f"s{self._n:03d}"
        self._n += 1
        span = {
            "id": sid,
            "logical": logical,
            "kind": kind,
            "agent": agent,
            "action": action,
            "payload": payload or {},
            "refs": refs or [],
            "phase": phase,
            "children": [],
        }
        self.spans.append(span)
        self._pending_children.setdefault(parent, []).append(span)
        return sid

    def finalize(self) -> list[dict[str, Any]]:
        by_id = {s["id"]: s for s in self.spans}
        for parent_id, kids in self._pending_children.items():
            if parent_id is not None:
                by_id[parent_id]["children"].extend(kids)
        roots = self._pending_children.get(None, [])
        order = 0
        stack = list(roots)
        # DFS ordinal (matches the flattening order of canonical_events)
        def walk(nodes: list[dict]) -> None:
            nonlocal order
            for node in nodes:
                self._order[node["id"]] = order
                order += 1
                walk(node["children"])

        walk(roots)
        return roots

    def ordinal(self, sid: str) -> int:
        return self._order[sid]


def execute(
    task_id: str,
    fault: FaultSpec | None,
    meta_overrides: dict | None = None,
) -> dict[str, Any]:
    """Execute one rollout, returning {spans, outcome, meta}. Deterministic:
    no randomness source.

    ``meta_overrides`` overrides meta grouping keys (model_version /
    prompt_version / time_window -- used by the drift-detection corpus) but
    not qrels/injected_fault.
    """
    task = env.TASKS[task_id]
    rec = _Recorder()
    onset_sid: str | None = None
    read_docs: list[str] = []
    n_tool_calls = 0

    rec.add("start", "TASK_START", "env", payload={"task": task["text"]})
    if fault and fault.kind == "premature_termination":
        plan_sid = rec.add(
            "plan", "LLM_CALL", "planner", phase="plan",
            payload={"content": f"plan: I recall the answer to '{task_id}' from memory; submit directly."},
        )
        answer = f"{task['gold_answer']}"
        call = rec.add(
            "submit", "TOOL_CALL", "planner", action="submit", phase="plan",
            payload={"answer": answer}, refs=[],  # no evidence to cite
        )
        n_tool_calls += 1
        ok, note = env.verify(task_id, answer, read_docs)
        rec.add("verify", "VERIFIER", "verifier", refs=[call], payload={"content": note})
        rec.add("end", "TASK_END", "env")
        # onset = the planning step (the decision to skip retrieval), not the
        # submit termination action -- Who&When Eq.5: the earliest step whose
        # correction flips the outcome is plan (fixing it → normal flow)
        onset_sid = plan_sid
        return _finish(rec, task_id, fault, onset_sid, ok, note, meta_overrides)

    rec.add(
        "plan", "LLM_CALL", "planner", phase="plan",
        payload={"content": f"plan: search '{task['query']}', read the most relevant doc, report with citation."},
    )
    hs = rec.add(
        "handoff_search", "HANDOFF", "planner", phase="plan",
        payload={"to": "searcher", "content": f"please find docs about '{task['query']}' and read the best one"},
        refs=[rec.spans[-2]["id"]],
    )

    # ---- search phase (three faults branch here) ----
    if fault and fault.kind == "malformed_tool_call":
        call = rec.add(
            "search", "TOOL_CALL", "searcher", action="search", phase="search",
            payload={},  # malformed call: missing the query argument
            refs=[hs],
        )
        # onset: the malformed search call (saved explicitly -- the ``call``
        # name is reused by the submit call below, so a spans index would be
        # the only alternative and breaks if any span is inserted earlier)
        malformed_call_sid = call
        n_tool_calls += 1
        res = rec.add(
            "search_result", "TOOL_RESULT", "env", action="search", phase="search",
            payload={"content": "error: invalid arguments for search: missing required parameter 'query'"},
            refs=[call],
        )
        rec.add(
            "search_reason", "LLM_CALL", "searcher", phase="search",
            payload={"content": "the search tool rejected my call; I cannot retrieve any document"},
            refs=[res],
        )
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={"to": "reporter", "content": "no usable search result obtained"},
            refs=[res],
        )
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={"content": "without any retrieved document I can only answer: unknown"},
            refs=[hr],
        )
        answer = "unknown"
        call = rec.add(
            "submit", "TOOL_CALL", "reporter", action="submit", phase="report",
            payload={"answer": answer}, refs=[compose],
        )
        n_tool_calls += 1
        ok, note = env.verify(task_id, answer, read_docs)
        rec.add("verify", "VERIFIER", "verifier", refs=[call], payload={"content": note})
        rec.add("end", "TASK_END", "env")
        onset_sid = malformed_call_sid  # first deviation: the malformed search call
        return _finish(rec, task_id, fault, onset_sid, ok, note, meta_overrides)

    # ---- retrieval detour (RG last-hop target scenario, phase-four extended fault) ----
    if fault and fault.kind == "retrieval_detour":
        detour_query = "failure taxonomy"   # hits evidence(d2), never hits gold
        detour_doc = env._search_hits(detour_query)[0]
        call = rec.add(
            "search", "TOOL_CALL", "searcher", action="search", phase="search",
            payload={"query": detour_query}, refs=[hs],
        )
        n_tool_calls += 1
        res = rec.add(
            "search_result", "TOOL_RESULT", "env", action="search", phase="search",
            payload={"content": env.search(detour_query)}, refs=[call],
        )
        rec.add(
            "search_reason", "LLM_CALL", "searcher", phase="search",
            payload={"content": f"the most relevant doc is {detour_doc}; I will read {detour_doc}"},
            refs=[res],
        )
        rec.add(
            "read", "TOOL_CALL", "searcher", action="read_doc", phase="search",
            payload={"doc_id": detour_doc}, refs=[res],
        )
        n_tool_calls += 1
        read_docs.append(detour_doc)
        read_res = rec.add(
            "read_result", "TOOL_RESULT", "env", action="read_doc", phase="search",
            payload={"content": env.read_doc(detour_doc)},
            refs=[rec.spans[-1]["id"]],
        )
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={"to": "reporter", "content": f"the answer is in {detour_doc}"},
            refs=[read_res],
        )
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={"content": f"based on {detour_doc}, the paper proposes a failure taxonomy survey (cited: {detour_doc})"},
            refs=[hr],
        )
        onset_sid = call   # earliest deviation: the off-target query
        answer = f"a failure taxonomy survey ({detour_doc})"
        call = rec.add(
            "submit", "TOOL_CALL", "reporter", action="submit", phase="report",
            payload={"answer": answer}, refs=[compose],
        )
        n_tool_calls += 1
        ok, note = env.verify(task_id, answer, read_docs)
        rec.add("verify", "VERIFIER", "verifier", refs=[call], payload={"content": note})
        rec.add("end", "TASK_END", "env")
        return _finish(rec, task_id, fault, onset_sid, ok, note, meta_overrides)

    # ---- two agents waiting on each other (inducer residual target scenario, phase-four extended fault) ----
    if fault and fault.kind == "agent_deadlock":
        prev_sid = hs
        for i in range(1, 4):
            clarify = rec.add(
                f"clarify#{i}", "HANDOFF", "searcher", phase="search",
                payload={
                    "to": "planner",
                    "content": "please clarify which document I should prioritize before searching",
                },
                refs=[prev_sid],
            )
            prev_sid = rec.add(
                f"reanswer#{i}", "AGENT_MESSAGE", "planner", phase="search",
                payload={
                    "to": "searcher",
                    "content": "prioritize the most relevant document; proceed with the search",
                },
                refs=[clarify],
            )
            if i == 1:
                onset_sid = clarify
        # after the clarification deadlock, searching and reading proceed as
        # usual (preserving the residual shape of "evidence exists yet the
        # task stalls", which dodges the known symptom rules of
        # premature/withholding)
        call = rec.add(
            "search", "TOOL_CALL", "searcher", action="search", phase="search",
            payload={"query": task["query"]}, refs=[prev_sid],
        )
        n_tool_calls += 1
        res = rec.add(
            "search_result", "TOOL_RESULT", "env", action="search", phase="search",
            payload={"content": env.search(task["query"])}, refs=[call],
        )
        rec.add(
            "read", "TOOL_CALL", "searcher", action="read_doc", phase="search",
            payload={"doc_id": task["gold_doc"]}, refs=[res],
        )
        n_tool_calls += 1
        read_docs.append(task["gold_doc"])
        read_res = rec.add(
            "read_result", "TOOL_RESULT", "env", action="read_doc", phase="search",
            payload={"content": env.read_doc(task["gold_doc"])},
            refs=[rec.spans[-1]["id"]],
        )
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={
                "to": "reporter",
                "content": f"the clarification rounds stalled the task; the answer is in {task['gold_doc']}",
            },
            refs=[read_res],
        )
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={
                "content": (
                    "after repeated clarification rounds the task stalled; "
                    f"answer: unknown (cited: {task['gold_doc']})"
                )
            },
            refs=[hr],
        )
        answer = f"unknown ({task['gold_doc']})"
        call = rec.add(
            "submit", "TOOL_CALL", "reporter", action="submit", phase="report",
            payload={"answer": answer}, refs=[compose],
        )
        n_tool_calls += 1
        ok, note = env.verify(task_id, answer, read_docs)
        rec.add("verify", "VERIFIER", "verifier", refs=[call], payload={"content": note})
        rec.add("end", "TASK_END", "env")
        return _finish(rec, task_id, fault, onset_sid, ok, note, meta_overrides)

    n_repeats = 3 if (fault and fault.kind == "step_repetition") else 1
    first_result_sid: str | None = None
    for i in range(n_repeats):
        logical = "search" if i == 0 else f"search#{i}"
        call = rec.add(
            logical, "TOOL_CALL", "searcher", action="search", phase="search",
            payload={"query": task["query"]}, refs=[hs],
        )
        n_tool_calls += 1
        if i == 1 and fault and fault.kind == "step_repetition":
            onset_sid = call  # the first repetition is the decisive error step
        res = rec.add(
            logical + "_result", "TOOL_RESULT", "env", action="search", phase="search",
            payload={"content": env.search(task["query"])}, refs=[call],
        )
        if first_result_sid is None:
            first_result_sid = res
    rec.add(
        "search_reason", "LLM_CALL", "searcher", phase="search",
        payload={
            "content": (
                f"the most relevant doc is {task['gold_doc']}; "
                + (f"repeating search did not help; " if n_repeats > 1 else "")
                + f"I will read {task['gold_doc']}"
            )
        },
        refs=[first_result_sid],
    )
    rec.add(
        "read", "TOOL_CALL", "searcher", action="read_doc", phase="search",
        payload={"doc_id": task["gold_doc"]}, refs=[first_result_sid],
    )
    n_tool_calls += 1
    read_docs.append(task["gold_doc"])
    read_res = rec.add(
        "read_result", "TOOL_RESULT", "env", action="read_doc", phase="search",
        payload={"content": env.read_doc(task["gold_doc"])},
        refs=[rec.spans[-1]["id"]],
    )

    # ---- report phase (three faults branch here) ----
    if fault and fault.kind == "info_withholding":
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={"to": "reporter", "content": "no relevant documents found for the query"},
            refs=[read_res],
        )
        onset_sid = hr
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={"content": "based on the searcher's report, no document addresses the question; answer: unknown"},
            refs=[hr],
        )
        answer = "unknown"
    elif fault and fault.kind == "ungrounded_citation":
        other = next(
            d for d in re.findall(r"d\d", env.search(task["query"])) if d != task["gold_doc"]
        )
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={"to": "reporter", "content": f"the answer is in {task['gold_doc']}"},
            refs=[read_res],
        )
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={
                "content": (
                    f"based on {other}, the paper proposes {task['gold_answer']} "
                    f"(cited: {other})"
                )
            },
            refs=[hr],
        )
        onset_sid = compose
        answer = f"{task['gold_answer']} ({other})"
    elif fault and fault.kind == "disobey_task_spec":
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={"to": "reporter", "content": f"the answer is in {task['gold_doc']}"},
            refs=[read_res],
        )
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={
                "content": (
                    f"based on {task['gold_doc']}, the paper proposes "
                    f"{task['gold_answer']}"  # content correct, but the doc id is not attached as required
                )
            },
            refs=[hr],
        )
        onset_sid = compose
        answer = task["gold_answer"]
    else:
        hr = rec.add(
            "handoff_report", "HANDOFF", "searcher", phase="report",
            payload={
                "to": "reporter",
                "content": f"the paper proposes {task['gold_answer']}; see {task['gold_doc']}",
            },
            refs=[read_res],
        )
        compose = rec.add(
            "compose", "LLM_CALL", "reporter", phase="report",
            payload={
                "content": (
                    f"based on {task['gold_doc']}, the paper proposes "
                    f"{task['gold_answer']} (cited: {task['gold_doc']})"
                )
            },
            refs=[hr],
        )
        answer = f"{task['gold_answer']} ({task['gold_doc']})"

    call = rec.add(
        "submit", "TOOL_CALL", "reporter", action="submit", phase="report",
        payload={"answer": answer}, refs=[compose],
    )
    n_tool_calls += 1
    ok, note = env.verify(task_id, answer, read_docs)
    if n_tool_calls > TOOL_BUDGET:
        ok = False
        note = f"failed: tool-call budget exhausted by repeated search calls ({n_tool_calls} > {TOOL_BUDGET})"
    rec.add("verify", "VERIFIER", "verifier", refs=[call], payload={"content": note})
    rec.add("end", "TASK_END", "env")
    return _finish(rec, task_id, fault, onset_sid, ok, note, meta_overrides)


def _finish(
    rec: _Recorder,
    task_id: str,
    fault: FaultSpec | None,
    onset_sid: str | None,
    ok: bool,
    note: str,
    meta_overrides: dict | None = None,
) -> dict[str, Any]:
    roots = rec.finalize()
    meta: dict[str, Any] = {
        "task_id": task_id,
        "model_version": "scripted-1.0",
        "prompt_version": "v1",
        "time_window": "w1",
        # qrels two-level annotation (E/G) -- the data dependency of rg_ug attribution (2608.01913)
        "qrels": env.qrels(task_id),
    }
    if meta_overrides:
        meta.update(meta_overrides)
    if fault is not None and onset_sid is not None:
        meta["injected_fault"] = {
            "kind": fault.kind,
            "agent": fault.agent,
            "mast_code": fault.mast_code,
            "step": rec.ordinal(onset_sid),  # DFS ordinal == canonical index
        }
    return {
        "spans": roots,
        "outcome": {"success": ok, "note": note},
        "meta": meta,
    }


class ToySandbox:
    """Implements the ReplayEnvironment protocol: trajectory generation +
    targeted replay + full re-solve.

    Once ``llm`` is injected, feedback consumption upgrades to "keyword
    first, LLM semantic fallback" (phase three fixes the known limitation
    that real-model recovery was 0/6): the environment knows which fault it
    injected (a construction-side fact, not judge-visible GT) and uses the
    LLM to decide whether free-text feedback targets that fault.
    """

    def __init__(self, llm: object | None = None) -> None:
        self._rr_counter = 0
        self._llm = llm

    # -- generation -----------------------------------------------------------

    def generate(
        self,
        task_id: str,
        fault_kind: str | None = None,
        trace_id: str | None = None,
        meta: dict | None = None,
    ) -> Trajectory:
        fault = ALL_FAULTS[fault_kind] if fault_kind else None
        result = execute(task_id, fault, meta_overrides=meta)
        tid = trace_id or (
            f"{task_id}--{fault_kind}" if fault_kind else f"{task_id}--ok"
        )
        return Trajectory(
            trace_id=tid,
            task=env.TASKS[task_id]["text"],
            events=[],
            outcome=Outcome(
                success=result["outcome"]["success"],
                score=1.0 if result["outcome"]["success"] else 0.0,
                note=result["outcome"]["note"],
            ),
            meta=result["meta"],
            raw={"task_id": task_id, "spans": result["spans"]},
        )

    def generate_population(self, seed: int = 0) -> list[Trajectory]:
        """Demo population: 1 success trace + 1 trace for each of the six
        faults (7 traces in total, rotated across tasks)."""
        import random

        rng = random.Random(seed)
        traces: list[Trajectory] = []
        task_ids = list(env.TASKS)
        for i, kind in enumerate(["__ok__", *FAULTS]):
            task_id = task_ids[i % len(task_ids)]
            traces.append(
                self.generate(task_id, None if kind == "__ok__" else kind)
            )
        rng.shuffle(traces)
        return traces

    def generate_corpus(self, successes_per_task: int = 2) -> list[Trajectory]:
        """SBFL spectrum corpus (a deterministic version of FAMAS's repeated
        execution idea): K successes per task + 1 trace for each of the six
        faults -- the same-task success/failure contrast gives the spectrum
        variation. "Repeated execution" in a deterministic sandbox has no
        random variation [adaptation: FAMAS samples via non-deterministic
        replays; here it is replaced by the full fault × task cross product],
        and the coverage-matrix semantics are unchanged."""
        traces: list[Trajectory] = []
        for task_id in env.TASKS:
            for i in range(successes_per_task):
                traces.append(
                    self.generate(task_id, None, trace_id=f"{task_id}--ok{i}")
                )
            for kind in FAULTS:
                traces.append(self.generate(task_id, kind))
        return traces

    def generate_drift_corpus(self) -> list[Trajectory]:
        """Drift-detection corpus (constructed scenarios for the three drift
        types of the system-level taxonomy 2511.19933).

        Four time windows built from meta grouping keys (version = model
        version bucket, data = task-composition bucket, behavior =
        behavioral-feature bucket; see analyze/drift_detect):
        * w1: v1, 2 successes per task for all three tasks -- baseline;
        * w2: v2, 2 step_repetition traces per task for all three tasks --
          **version drift** (behavioral distribution difference across model
          buckets: repeated search raises the action histogram and trajectory
          length; n=6 per group meets drift_detect's min_group_size floor --
          a 19-vs-3 contrast says more about the tiny side's sample size
          than about drift);
        * w3: v1, only q-who-when success ×6 -- **data drift** (task
          composition shift, behavioral features unchanged);
        * w4: v1, task composition as in w1 but with 1 step_repetition trace
          -- **behavior drift** (behavior changes across windows under the
          same model and prompt, task composition nearly unchanged).

        [adaptation: in the paper, version drift stems from real provider
        model updates and behavior drift from stochastic sampling -- the
        deterministic sandbox simulates both signals via fault injection
        (changing the behavior distribution) + meta version labels; data
        drift's "input distribution shift" is proxied by cross-window task
        composition shift [inference].]
        """
        traces: list[Trajectory] = []
        for task in env.TASKS:
            for i in range(2):
                traces.append(self.generate(
                    task, None, meta={"time_window": "w1"},
                    trace_id=f"drift-w1-{task}-ok{i}",
                ))
        for task in env.TASKS:
            for i in range(2):
                traces.append(self.generate(
                    task, "step_repetition",
                    meta={"model_version": "scripted-2.0", "time_window": "w2"},
                    trace_id=f"drift-w2-{task}-rep{i}",
                ))
        for i in range(6):
            traces.append(self.generate(
                "q-who-when", None, meta={"time_window": "w3"},
                trace_id=f"drift-w3-ok{i}",
            ))
        for task in env.TASKS:
            for i in range(2):
                traces.append(self.generate(
                    task, None, meta={"time_window": "w4"},
                    trace_id=f"drift-w4-{task}-ok{i}",
                ))
        traces.append(self.generate(
            "q-trajaudit", "step_repetition", meta={"time_window": "w4"},
            trace_id="drift-w4-trajaudit-rep",
        ))
        return traces

    # -- full re-solve (AgenTracer 2509.03312 §5.3 feedback injection) -------

    def resolve(self, trajectory: Trajectory, feedback: str) -> Trajectory:
        """Full re-solve from scratch with reflection feedback (a new episode,
        no prefix retained).

        Fault state is taken from the **original trajectory** meta (rerun
        trajectories have injected_fault stripped from meta; chaining it back
        in would misjudge "no fault" and fake success -- the same convention
        as rerun_from). Feedback consumption: a keyword hit removes the
        fault; on a miss with an injected LLM, ask the LLM (semantic
        matching of free-text feedback); with neither, the fault remains and
        the re-solve keeps failing.
        """
        task_id = (trajectory.raw or {}).get("task_id") or trajectory.meta.get("task_id")
        if task_id is None:
            raise ValueError(f"trajectory {trajectory.trace_id} has no task_id; cannot re-solve")
        inj = trajectory.meta.get("injected_fault") or {}
        fault_kind = inj.get("kind")
        # ALL_FAULTS: feedback lookup covers the extended registry too
        # (retrieval_detour / agent_deadlock), so a fault-naming feedback can
        # remove them instead of falling into unexplained_failure
        fault = ALL_FAULTS.get(fault_kind) if fault_kind else None
        # guard: if the original failure was not caused by an injected fault
        # (e.g. a real failure cause was wired in), a clean rerun necessarily
        # succeeds -- that must not be claimed as "recovery"; treat the fault
        # as still present
        unexplained_failure = fault is None and not trajectory.outcome.success

        removed = fault is not None and self._feedback_addresses(fault_kind, feedback)
        new_run = execute(task_id, None if removed else fault)
        self._rr_counter += 1
        fault_active = (fault is not None and not removed) or unexplained_failure
        return Trajectory(
            trace_id=f"{trajectory.trace_id}-rs{self._rr_counter}",
            task=trajectory.task,
            events=self._flatten_to_events(new_run["spans"]),
            outcome=Outcome(
                success=new_run["outcome"]["success"] if not fault_active else False,
                score=new_run["outcome"]["success"] and not fault_active,
                note=new_run["outcome"]["note"] if not fault_active else trajectory.outcome.note,
            ),
            meta={
                # qrels shallow-copied: the re-solve meta must not share the
                # original trajectory's mutable dict
                **{k: dict(v) if k == "qrels" else v
                   for k, v in trajectory.meta.items() if k != "injected_fault"},
                "rerun_of": trajectory.trace_id,
                "resolve_mode": "full_reresolve",
                "fault_removed": removed,
                "feedback_snippet": feedback[:200],
            },
            raw={"task_id": task_id, "spans": new_run["spans"]},
        )

    @staticmethod
    def _flatten_to_events(roots: list[dict]) -> list:
        """Span tree of the new rollout → R0 event stream (aligned with
        rerun_from's merged shape, so reflection calls inside the recovery
        round can render it directly; the closed-loop verification round
        normalizes again via canonical_events, so repeated flattening is
        idempotent)."""
        from atap.core.schema import TraceEvent

        out: list[TraceEvent] = []

        def walk(nodes: list[dict], parent: str | None) -> None:
            for n in nodes:
                idx = len(out)
                out.append(TraceEvent(
                    id=f"e{idx:03d}", ts=float(idx), kind=n["kind"],
                    agent=n.get("agent", "unknown"), action=n.get("action"),
                    payload=dict(n.get("payload") or {}), refs=[],
                    phase=n.get("phase"), parent=parent, index=idx,
                ))
                walk(n.get("children") or [], out[-1].id)

        walk(roots, None)
        return out

    # -- targeted replay (AgentDebug Algorithm 1 Stage 3) ---------------------

    def rerun_from(self, trajectory: Trajectory, step: int, feedback: str) -> Trajectory:
        task_id = (trajectory.raw or {}).get("task_id") or trajectory.meta.get("task_id")
        if task_id is None:
            raise ValueError(f"trajectory {trajectory.trace_id} has no task_id; cannot replay")
        inj = trajectory.meta.get("injected_fault") or {}
        fault_kind = inj.get("kind")
        # ALL_FAULTS: same extended-registry coverage as resolve/replay_intervene
        fault = ALL_FAULTS.get(fault_kind) if fault_kind else None
        unexplained_failure = fault is None and not trajectory.outcome.success

        removed = fault is not None and self._feedback_addresses(fault_kind, feedback)

        # deterministically replay the original rollout to find the logical
        # step name corresponding to step
        orig = execute(task_id, fault)
        flat = self._flatten_spans(orig["spans"])
        # out-of-range clamping: when the attribution step exceeds the event
        # stream, take the last step (prevents duplicate events / index
        # misalignment from concatenating prefix + full events), leaving a
        # trace
        clamped = not (0 <= step < len(flat))
        if clamped:
            step = max(0, min(int(step), len(flat) - 1, len(trajectory.events) - 1))

        new_run = execute(task_id, None if removed else fault)
        new_flat = self._flatten_spans(new_run["spans"])
        suffix_start, alignment = self._align_suffix_start(flat, step, new_flat)

        # copy the retained prefix events: the rerun must not alias the
        # original trajectory's event objects (the closed-loop verification
        # round normalizes rerun events in place via canonical_events
        # _normalize, which would rewrite the original's id/index otherwise)
        prefix = [replace(ev) for ev in trajectory.events[:step]]
        merged = list(prefix)
        bridge = self._prefix_bridge(flat, trajectory, step)
        new_pos = {s["id"]: i for i, s in enumerate(new_flat)}
        suffix_eid = {
            s["id"]: f"e{step + k:03d}"
            for k, s in enumerate(new_flat[suffix_start:])
        }
        dropped_refs = 0
        last_call_id: str | None = prefix[-1].id if prefix else None
        for k, span in enumerate(new_flat[suffix_start:]):
            idx = step + k
            eid = f"e{idx:03d}"
            refs, dropped = self._suffix_refs(
                span, new_flat=new_flat, new_pos=new_pos,
                suffix_start=suffix_start, suffix_eid=suffix_eid,
                bridge=bridge, last_call_id=last_call_id,
            )
            dropped_refs += dropped
            event = TraceEvent(
                id=eid, ts=float(idx), kind=span["kind"], agent=span["agent"],
                action=span["action"], payload=span["payload"], refs=refs,
                phase=span["phase"], parent=None, index=idx,
            )
            merged.append(event)
            if span["kind"] == "TOOL_CALL":
                last_call_id = eid

        self._rr_counter += 1
        # fault lookup spans ALL_FAULTS (six standard + two extended), so a
        # feedback naming an extended fault (retrieval_detour /
        # agent_deadlock) removes it; the fault remains only when it exists
        # but the feedback does not name it. unexplained_failure applies only
        # to failures caused by no injected fault at all -- a clean rerun of
        # those would succeed and must not be claimed as recovery (same guard
        # as resolve)
        fault_active = (fault is not None and not removed) or unexplained_failure
        note = new_run["outcome"]["note"] if not fault_active else trajectory.outcome.note
        return Trajectory(
            trace_id=f"{trajectory.trace_id}-rr{self._rr_counter}",
            task=trajectory.task,
            events=merged,
            outcome=Outcome(
                success=new_run["outcome"]["success"] if not fault_active else False,
                score=new_run["outcome"]["success"] and not fault_active,
                note=note,
            ),
            meta={
                # qrels shallow-copied: the rerun meta must not share the
                # original trajectory's mutable dict
                **{k: dict(v) if k == "qrels" else v
                   for k, v in trajectory.meta.items() if k != "injected_fault"},
                "rerun_of": trajectory.trace_id,
                "rerun_from_step": step,
                "step_clamped": clamped,
                "suffix_alignment": alignment,
                "dropped_refs": dropped_refs,
                "fault_removed": removed,
                "feedback_snippet": feedback[:200],
            },
        )

    # -- message intervention replay (L3 counterfactual replay: TraceElephant
    #    A.6.3 / DoVer M1) ----------------------------------------------------

    def replay_intervene(
        self,
        trajectory: Trajectory,
        step: int,
        edit_text: str,
        *,
        horizon: int | None = None,
        n_repeats: int = 1,
    ) -> list[Trajectory]:
        """Checkpoint replay + **in-place replacement** of the candidate
        step's message (as opposed to appending feedback).

        Mechanism [adaptation: neither paper gives caching/de-randomization
        details -- the deterministic sandbox implements it equivalently as
        "re-execute + consume the edit conditioned on the fault"]: locate
        the logical step at step; the edit is consumed by the fault
        middleware only when **both** hold -- (a) the intervention step is
        the injected fault's **onset step** (meta.injected_fault.step, a
        construction-side fact the executor knows; removing a fault by
        editing a *symptom* step downstream is causally impossible), and
        (b) the edit text names/targets that fault (keyword first, LLM
        semantic fallback, same as _feedback_addresses). On a hit the suffix
        from that step replays in fault-free form and the edited message's
        payload content is replaced in place with edit_text (DoVer's
        in-place replacement semantics); on a miss the suffix keeps the
        original failure course (the edit does not take effect) -- so
        pseudo-causal (symptom-step) candidates are refuted by the replay
        mechanism itself, not by the oracle's wording.

        * ``horizon=k``: return only k events from the candidate step
          (TraceElephant's "verify only the following k steps" reading --
          the outcome is inferred from in-window evidence);
        * ``horizon=None``: run to the end (DoVer's reading);
        * ``n_repeats``: repetition count (DoVer ×3; under the deterministic
          sandbox the results are identical and are faithfully returned
          multiple times).

        Meta keeps faithful traces: ``intervention_applied`` is True only
        when the in-place message replacement actually happened (the
        intervened step is a message event: LLM_CALL/HANDOFF/AGENT_MESSAGE);
        edits applied at TOOL_CALL/other events replace no message and are
        recorded as ``intervention_applied=False`` with a reason; the
        step-sensitivity gate is recorded as ``intervention_on_onset_step``.
        """
        task_id = (trajectory.raw or {}).get("task_id") or trajectory.meta.get("task_id")
        if task_id is None:
            raise ValueError(f"trajectory {trajectory.trace_id} has no task_id; cannot replay")
        inj = trajectory.meta.get("injected_fault") or {}
        fault_kind = inj.get("kind")
        fault = ALL_FAULTS.get(fault_kind) if fault_kind else None
        unexplained_failure = fault is None and not trajectory.outcome.success

        orig = execute(task_id, fault)
        flat = self._flatten_spans(orig["spans"])
        clamped = not (0 <= step < len(flat))
        if clamped:
            step = max(0, min(int(step), len(flat) - 1, len(trajectory.events) - 1))

        # step-sensitivity gate: the middleware consumes the edit only when
        # the intervention lands on the fault's onset step (computed after
        # clamping so the gate sees the actually-applied step). A
        # fault-naming edit applied at any other (symptom) step does NOT
        # remove the fault. [adaptation: injected_fault.step is a
        # construction-side fact held by the executor, not judge-visible GT;
        # with the step unknown (tampered meta) the gate fails closed]
        onset_targeted = (
            inj.get("step") is not None and int(inj["step"]) == step
        )
        removed = (
            fault is not None
            and onset_targeted
            and self._feedback_addresses(fault_kind, edit_text)
        )

        new_run = execute(task_id, None if removed else fault)
        new_flat = self._flatten_spans(new_run["spans"])
        suffix_start, alignment = self._align_suffix_start(flat, step, new_flat)
        suffix = new_flat[suffix_start:]
        if horizon is not None:
            suffix = suffix[: int(horizon)]
        # faithful in-place-replacement record: the replacement only applies
        # when the intervened step is a message event
        first_span = suffix[0] if suffix else None
        applied = bool(
            first_span is not None
            and first_span["kind"] in ("LLM_CALL", "HANDOFF", "AGENT_MESSAGE")
        )
        applied_note = (
            "message content replaced in place with the edit text"
            if applied else
            (
                f"intervened step is a {first_span['kind']} event; in-place "
                "replacement applies only to message events "
                "(LLM_CALL/HANDOFF/AGENT_MESSAGE)"
                if first_span is not None
                else "empty replay window: no event to replace"
            )
        )

        bridge = self._prefix_bridge(flat, trajectory, step)
        new_pos = {s["id"]: i for i, s in enumerate(new_flat)}
        suffix_eid = {s["id"]: f"e{step + k:03d}" for k, s in enumerate(suffix)}
        dropped_refs = 0
        results: list[Trajectory] = []
        for r in range(n_repeats):
            self._rr_counter += 1
            merged = list(trajectory.events[:step])
            last_call_id = merged[-1].id if merged else None
            for k, span in enumerate(suffix):
                idx = step + k
                eid = f"e{idx:03d}"
                payload = dict(span.get("payload") or {})
                # in-place message replacement: the edited step's message
                # content is exactly the edit text
                if k == 0 and span["kind"] in ("LLM_CALL", "HANDOFF", "AGENT_MESSAGE"):
                    payload["content"] = edit_text[:400]
                refs, dropped = self._suffix_refs(
                    span, new_flat=new_flat, new_pos=new_pos,
                    suffix_start=suffix_start, suffix_eid=suffix_eid,
                    bridge=bridge, last_call_id=last_call_id,
                )
                dropped_refs += dropped
                ev = TraceEvent(
                    id=eid, ts=float(idx), kind=span["kind"], agent=span["agent"],
                    action=span.get("action"), payload=payload, refs=refs,
                    phase=span.get("phase"), parent=None, index=idx,
                )
                merged.append(ev)
                if span["kind"] == "TOOL_CALL":
                    last_call_id = eid
            fault_active = (fault is not None and not removed) or unexplained_failure
            results.append(Trajectory(
                trace_id=f"{trajectory.trace_id}-iv{self._rr_counter}",
                task=trajectory.task,
                events=merged,
                outcome=Outcome(
                    success=(new_run["outcome"]["success"] if not fault_active else False)
                    if horizon is None else (not fault_active),
                    score=1.0 if (not fault_active and horizon is None) else 0.0,
                    note=(
                        new_run["outcome"]["note"] if not fault_active
                        else trajectory.outcome.note
                    ) if horizon is None else (
                        f"windowed replay (k={horizon}): "
                        + ("fault removed in window" if not fault_active
                           else "failure course unchanged in window")
                    ),
                ),
                meta={
                    **{kk: vv for kk, vv in trajectory.meta.items() if kk != "injected_fault"},
                    "rerun_of": trajectory.trace_id,
                    "replay_mode": "message_intervention",
                    "intervened_step": step,
                    "step_clamped": clamped,
                    "suffix_alignment": alignment,
                    "dropped_refs": dropped_refs,
                    "intervention_applied": applied,
                    "intervention_applied_note": applied_note,
                    "intervention_on_onset_step": onset_targeted,
                    "fault_removed": removed,
                    "edit_snippet": edit_text[:200],
                    "horizon": horizon,
                },
            ))
        return results

    # -- internals ------------------------------------------------------------

    def _feedback_addresses(self, fault_kind: str, feedback: str) -> bool:
        """Whether the feedback names/targets this fault. Keyword first
        (offline, deterministic); on a miss with an injected LLM, semantic
        fallback (the environment itself knows the fault spec -- not a
        judge-GT leak: the leakage constraint protects the judge/attributor
        and does not bind the execution environment)."""
        low = feedback.lower()
        if fault_kind in low or fault_kind.replace("_", " ") in low:
            return True
        if self._llm is None:
            return False
        fault = ALL_FAULTS.get(fault_kind)
        messages = [
            {
                "role": "user",
                "content": (
                    "Fault specification injected by the execution environment:\n"
                    f"fault type: {fault_kind}\ndescription: {fault.description if fault else ''}\n\n"
                    f"Correction feedback received by the solver system in the next round:\n{feedback[:1500]}\n\n"
                    "Does this feedback provide corrective guidance targeting this fault? Answer only yes or no."
                ),
            },
        ]
        result = self._llm.complete(messages, tag="feedback_match")
        ans = str(result.text).strip().lower()
        first = ans.split()[0] if ans.split() else ""
        if first in ("yes", "no"):
            return first == "yes"
        return "yes" in ans

    # -- replay internals ------------------------------------------------------

    @staticmethod
    def _align_suffix_start(
        orig_flat: list[dict], step: int, new_flat: list[dict]
    ) -> tuple[int, str]:
        """Where the re-executed rollout's suffix begins, aligned with the
        original logical step at ``step``. Returns (suffix_start, alignment).

        The logical step at ``step`` may exist only in the faulted script
        (e.g. the repeated search of step_repetition): after fault removal
        the new rollout has no such step, and blindly falling back to 0
        splices a whole second rollout behind the retained prefix (duplicate
        TASK_START / doubled event stream). Alignment ladder [adaptation:
        the replay papers do not specify re-alignment when the replayed-from
        step vanishes in the corrected run]:
        (a) "exact": first new step with the same logical name;
        (b) "next_surviving": the first logical of the original suffix
            (orig_flat[step:]) that still exists in the new rollout;
        (c) "phase": first new step in the same phase as orig_flat[step]
            (only when that phase is a real label -- TASK_START/TASK_END
            carry None and would match each other);
        (d) "empty": empty suffix -- never splice a duplicate rollout.
        """
        if not (0 <= step < len(orig_flat)) or not new_flat:
            return len(new_flat), "empty"
        logical = orig_flat[step]["logical"]
        for i, s in enumerate(new_flat):
            if s["logical"] == logical:
                return i, "exact"
        for cand in orig_flat[step:]:
            hit = next(
                (i for i, s in enumerate(new_flat) if s["logical"] == cand["logical"]),
                None,
            )
            if hit is not None:
                return hit, "next_surviving"
        phase = orig_flat[step].get("phase")
        if phase is not None:
            for i, s in enumerate(new_flat):
                if s.get("phase") == phase:
                    return i, "phase"
        return len(new_flat), "empty"

    @staticmethod
    def _prefix_bridge(
        orig_flat: list[dict], trajectory: Trajectory, step: int
    ) -> dict[str, str]:
        """logical step name -> retained-prefix event id. Both the original
        trajectory's event stream and the re-executed ``orig_flat`` are
        pre-order flattenings of the same deterministic rollout, so positions
        i < step carry the same logical steps and their ids can be bridged."""
        return {
            orig_flat[i]["logical"]: trajectory.events[i].id
            for i in range(min(step, len(orig_flat), len(trajectory.events)))
        }

    @staticmethod
    def _suffix_refs(
        span: dict,
        *,
        new_flat: list[dict],
        new_pos: dict[str, int],
        suffix_start: int,
        suffix_eid: dict[str, str],
        bridge: dict[str, str],
        last_call_id: str | None,
    ) -> tuple[list[str], int]:
        """Semantic refs (span-id references on the span itself) -> new event
        ids for a replayed suffix event. A ref target inside the suffix maps
        directly via ``suffix_eid``; a target before the suffix maps through
        the logical-name ``bridge`` onto the retained prefix's event ids
        (replay keeps the original prefix, not the new rollout's prefix);
        unmappable targets are dropped and counted. The legacy
        TOOL_RESULT/VERIFIER -> last tool-call fallback applies only when the
        span carries no refs of its own. Returns (refs, n_dropped)."""
        refs: list[str] = []
        dropped = 0
        for r in span.get("refs") or []:
            pos = new_pos.get(r)
            if pos is None:
                dropped += 1
                continue
            if pos >= suffix_start:
                eid = suffix_eid.get(r)
            else:
                eid = bridge.get(new_flat[pos]["logical"])
            if eid is None:
                dropped += 1
                continue
            refs.append(eid)
        if not refs and span["kind"] in ("TOOL_RESULT", "VERIFIER") and last_call_id:
            refs = [last_call_id]
        return refs, dropped

    @staticmethod
    def _flatten_spans(roots: list[dict]) -> list[dict]:
        out: list[dict] = []

        def walk(nodes: list[dict]) -> None:
            for n in nodes:
                out.append(n)
                walk(n["children"])

        walk(roots)
        return out
