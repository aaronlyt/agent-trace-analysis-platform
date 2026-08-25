"""脚本化策略 + 玩具沙盒 —— 确定性 rollout、故障注入与定向重放。

正常 rollout 是固定的逻辑步序列（planner→searcher→reporter 研究问答
流水线）；六种故障在各自 onset 逻辑步改变行为（见 faults.py）。

输出形态：**raw span 树**（嵌套 dict，带语义 refs 与逻辑步名），由
represent/canonical_events 拍平为 R0 事件流——沙盒不直接产出 R0，
保证采集→表征的层间契约真实生效。

定向重放（AgentDebug 2509.25370 Algorithm 1 Stage 3）：保留前缀
[0, step)，从 step 重执行；feedback 若点名故障类型（如伪判官的
fix_suggestion 含 "step_repetition"），策略走修正分支（等效移除故障），
否则故障仍在、重跑继续失败——反馈质量决定恢复成功率，这是刻意的。
"""

from __future__ import annotations

import re
from typing import Any

from atap.core.schema import Outcome, Trajectory
from atap.sandbox import env
from atap.sandbox.faults import FAULTS, TOOL_BUDGET, FaultSpec


class _Recorder:
    """记录 span 树；同时按 DFS 序维护 (span_id → 事件序号) 供 ground truth。"""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._n = 0
        self._order: dict[str, int] = {}  # span_id -> DFS 序号
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
        # DFS 序号（与 canonical_events 的拍平顺序一致）
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


def execute(task_id: str, fault: FaultSpec | None) -> dict[str, Any]:
    """执行一次 rollout，返回 {spans, outcome, meta}。确定性：无随机源。"""
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
            payload={"answer": answer}, refs=[],  # 无证据可引用
        )
        n_tool_calls += 1
        ok, note = env.verify(task_id, answer, read_docs)
        rec.add("verify", "VERIFIER", "verifier", refs=[call], payload={"content": note})
        rec.add("end", "TASK_END", "env")
        # onset=规划步（决定跳过检索的决策），非 submit 终止动作——
        # Who&When Eq.5：最早"修正即可翻盘"的步是 plan（修正它→正常流程）
        onset_sid = plan_sid
        return _finish(rec, task_id, fault, onset_sid, ok, note)

    rec.add(
        "plan", "LLM_CALL", "planner", phase="plan",
        payload={"content": f"plan: search '{task['query']}', read the most relevant doc, report with citation."},
    )
    hs = rec.add(
        "handoff_search", "HANDOFF", "planner", phase="plan",
        payload={"to": "searcher", "content": f"please find docs about '{task['query']}' and read the best one"},
        refs=[rec.spans[-2]["id"]],
    )

    # ---- search 阶段（三种故障在此分支）----
    if fault and fault.kind == "malformed_tool_call":
        call = rec.add(
            "search", "TOOL_CALL", "searcher", action="search", phase="search",
            payload={},  # 畸形调用：缺 query 参数
            refs=[hs],
        )
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
        onset_sid = rec.spans[3]["id"]  # 第一处偏离：畸形 search 调用
        return _finish(rec, task_id, fault, onset_sid, ok, note)

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
            onset_sid = call  # 首次重复才是决定性错误步
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

    # ---- report 阶段（三种故障在此分支）----
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
                    f"{task['gold_answer']}"  # 内容正确，但没按要求附文档编号
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
    return _finish(rec, task_id, fault, onset_sid, ok, note)


def _finish(
    rec: _Recorder,
    task_id: str,
    fault: FaultSpec | None,
    onset_sid: str | None,
    ok: bool,
    note: str,
) -> dict[str, Any]:
    roots = rec.finalize()
    meta: dict[str, Any] = {
        "task_id": task_id,
        "model_version": "scripted-1.0",
        "prompt_version": "v1",
        "time_window": "w1",
    }
    if fault is not None and onset_sid is not None:
        meta["injected_fault"] = {
            "kind": fault.kind,
            "agent": fault.agent,
            "mast_code": fault.mast_code,
            "step": rec.ordinal(onset_sid),  # DFS 序号 == canonical index
        }
    return {
        "spans": roots,
        "outcome": {"success": ok, "note": note},
        "meta": meta,
    }


class ToySandbox:
    """实现 ReplayEnvironment 协议：生成轨迹 + 定向重放 + 全量再求解。

    ``llm`` 注入后，反馈消费升级为"关键词优先、LLM 语义兜底"（阶段三
    修复真模型恢复 0/6 的已知限制）：环境知道自己注入的故障（构造侧
    事实，非判官可见 GT），用 LLM 判断自由文本反馈是否针对该故障。
    """

    def __init__(self, llm: object | None = None) -> None:
        self._rr_counter = 0
        self._llm = llm

    # -- 生成 ---------------------------------------------------------------

    def generate(self, task_id: str, fault_kind: str | None = None, trace_id: str | None = None) -> Trajectory:
        fault = FAULTS[fault_kind] if fault_kind else None
        result = execute(task_id, fault)
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
        """演示群体：每任务 1 条成功 + 六种故障各 1 条（跨任务轮转）。"""
        import random

        rng = random.Random(seed)
        traces: list[Trajectory] = []
        task_ids = list(env.TASKS)
        for i, kind in enumerate(["__ok__", *FAULTS]):
            task_id = task_ids[(i if kind == "__ok__" else i) % len(task_ids)]
            traces.append(
                self.generate(task_id, None if kind == "__ok__" else kind)
            )
        rng.shuffle(traces)
        return traces

    def generate_corpus(self, successes_per_task: int = 2) -> list[Trajectory]:
        """SBFL 频谱语料（FAMAS 重复执行思想的确定性版）：每任务 K 条
        成功 + 全部六种故障各 1 条——同任务的成败对照给频谱以变异。
        确定性沙盒的"重复执行"无随机变异【适配：FAMAS 靠非确定性重放
        采样，这里以故障×任务的完全交叉替代】，覆盖矩阵语义不变。"""
        traces: list[Trajectory] = []
        for task_id in env.TASKS:
            for i in range(successes_per_task):
                traces.append(
                    self.generate(task_id, None, trace_id=f"{task_id}--ok{i}")
                )
            for kind in FAULTS:
                traces.append(self.generate(task_id, kind))
        return traces

    # -- 全量再求解（AgenTracer 2509.03312 §5.3 反馈注入）--------------------

    def resolve(self, trajectory: Trajectory, feedback: str) -> Trajectory:
        """带反思反馈从头完整重解（新 episode，不保留前缀）。

        故障状态取自**原轨迹** meta（重跑轨迹的 meta 已剥离 injected_fault，
        链式传入会误判"无故障"而虚假成功——与 rerun_from 同一约定）。
        反馈消费：关键词命中即移除故障；未命中且有注入 LLM 时问 LLM
        （自由文本反馈的语义匹配）；都无则故障仍在、重解继续失败。
        """
        task_id = (trajectory.raw or {}).get("task_id") or trajectory.meta.get("task_id")
        if task_id is None:
            raise ValueError(f"轨迹 {trajectory.trace_id} 缺 task_id，无法重解")
        inj = trajectory.meta.get("injected_fault") or {}
        fault_kind = inj.get("kind")
        fault = FAULTS.get(fault_kind) if fault_kind else None

        removed = fault is not None and self._feedback_addresses(fault_kind, feedback)
        new_run = execute(task_id, None if removed else fault)
        self._rr_counter += 1
        fault_active = fault is not None and not removed
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
                **{k: v for k, v in trajectory.meta.items() if k != "injected_fault"},
                "rerun_of": trajectory.trace_id,
                "resolve_mode": "full_reresolve",
                "fault_removed": removed,
                "feedback_snippet": feedback[:200],
            },
            raw={"task_id": task_id, "spans": new_run["spans"]},
        )

    @staticmethod
    def _flatten_to_events(roots: list[dict]) -> list:
        """新 rollout 的 span 树 → R0 事件流（与 rerun_from 的合并形态对齐，
        供恢复轮内的反思调用直接渲染；闭环验证轮会再走 canonical_events
        归一化，重复拍平幂等）。"""
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

    # -- 定向重放（AgentDebug Algorithm 1 Stage 3）---------------------------

    def rerun_from(self, trajectory: Trajectory, step: int, feedback: str) -> Trajectory:
        task_id = (trajectory.raw or {}).get("task_id") or trajectory.meta.get("task_id")
        if task_id is None:
            raise ValueError(f"轨迹 {trajectory.trace_id} 缺 task_id，无法重放")
        inj = trajectory.meta.get("injected_fault") or {}
        fault_kind = inj.get("kind")
        fault = FAULTS.get(fault_kind) if fault_kind else None

        removed = fault is not None and self._feedback_addresses(fault_kind, feedback)

        # 确定性重放原 rollout，找到 step 对应的逻辑步名
        orig = execute(task_id, fault)
        flat = self._flatten_spans(orig["spans"])
        logical = flat[step]["logical"] if 0 <= step < len(flat) else None

        new_run = execute(task_id, None if removed else fault)
        new_flat = self._flatten_spans(new_run["spans"])

        prefix = [ev for ev in trajectory.events[:step]]
        suffix_start = next(
            (i for i, s in enumerate(new_flat) if s["logical"] == logical), 0
        )
        merged = list(prefix)
        last_call_id: str | None = prefix[-1].id if prefix else None
        for k, span in enumerate(new_flat[suffix_start:]):
            idx = step + k
            eid = f"e{idx:03d}"
            ev = span  # 下面复用 schema 构造
            from atap.core.schema import TraceEvent

            refs: list[str] = []
            if ev["kind"] == "TOOL_RESULT" and last_call_id:
                refs = [last_call_id]
            elif ev["kind"] == "VERIFIER" and last_call_id:
                refs = [last_call_id]
            event = TraceEvent(
                id=eid, ts=float(idx), kind=ev["kind"], agent=ev["agent"],
                action=ev["action"], payload=ev["payload"], refs=refs,
                phase=ev["phase"], parent=None, index=idx,
            )
            merged.append(event)
            if ev["kind"] == "TOOL_CALL":
                last_call_id = eid

        self._rr_counter += 1
        # 无故障（fault is None）时新跑即最终形态；有故障且未被反馈点名则故障仍在
        fault_active = fault is not None and not removed
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
                **{k: v for k, v in trajectory.meta.items() if k != "injected_fault"},
                "rerun_of": trajectory.trace_id,
                "rerun_from_step": step,
                "fault_removed": removed,
                "feedback_snippet": feedback[:200],
            },
        )

    # -- 内部 ---------------------------------------------------------------

    def _feedback_addresses(self, fault_kind: str, feedback: str) -> bool:
        """feedback 是否点名/针对该故障。关键词优先（离线确定性）；
        未命中且有注入 LLM 时语义兜底（环境侧自知故障规格，非判官 GT
        泄漏——泄漏约束保护的是判官/归因器，不限制执行环境）。"""
        low = feedback.lower()
        if fault_kind in low or fault_kind.replace("_", " ") in low:
            return True
        if self._llm is None:
            return False
        fault = FAULTS.get(fault_kind)
        messages = [
            {
                "role": "user",
                "content": (
                    "执行环境注入的故障规格：\n"
                    f"故障类型：{fault_kind}\n描述：{fault.description if fault else ''}\n\n"
                    f"求解系统在下一轮收到的修正反馈：\n{feedback[:1500]}\n\n"
                    "该反馈是否针对此故障给出了修正指导？只回答 yes 或 no。"
                ),
            },
        ]
        result = self._llm.complete(messages, tag="feedback_match")
        ans = str(result.text).strip().lower()
        first = ans.split()[0] if ans.split() else ""
        if first in ("yes", "no"):
            return first == "yes"
        return "yes" in ans

    @staticmethod
    def _flatten_spans(roots: list[dict]) -> list[dict]:
        out: list[dict] = []

        def walk(nodes: list[dict]) -> None:
            for n in nodes:
                out.append(n)
                walk(n["children"])

        walk(roots)
        return out
