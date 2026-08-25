"""L0 免费规则包 —— AgentDebugX, arXiv:2607.18754 §3.2（Detect 的确定性层）。

原文定义仅一句话："Deterministic rule packs first target mechanically
verifiable failures — malformed tool calls, no-progress loops, invalid
outputs, premature success — with no model call."（精确触发条件在官方
仓库、未随论文发表）——本实现的触发条件为自设【适配】，全部只依据
R0 可观测事件，绝不读 meta["injected_fault"]：

* ``malformed_tool_call``：TOOL_CALL 参数缺失（payload 为空）或其
  TOOL_RESULT 为结构化错误观测（error:/exception 前缀）；步 = 调用步；
* ``no_progress_loop``：消费 ``analyze/loop_detect`` 产物（search_loop/
  redundant_search 命中）；上游缺席时回退到 R5 签名自查（同一签名整条
  轨迹出现 ≥ ``min_repeats`` 次），两者皆无则显式 skip 并留痕；
* ``premature_success_claim``：submit（终态提交）之前没有任何成功的
  FILE_READ——无证据即宣称完成；步 = submit 前最后一个 LLM_CALL
  （决定跳过检索的决策步，对齐 Who&When Eq.5 最早决定性错误约定）；
* ``invalid_output``：VERIFIER 结构化驳回（失败说明含缺失/必填/格式
  类词）；步 = 驳回前最后一个 LLM_CALL（答案生成步）。

与判官打标的关系（原文）：规则命中的是"机械可验证失败"，检测定位
症状而非根因——findings 是归因的种子；规则未命中不等于无错（L1 判官
覆盖其余模式）。MAST 维度映射沿用 sandbox/faults.py 的【适配】口径。

产物：``{"findings": [...], "fusion": [...], "cost": "free", ...}``；
免费全量：成功轨迹同样过规则（成功轨迹中的反模式仍是过程信号）。
"""

from __future__ import annotations

import re
from typing import Any

from atap.classify.base import Classifier
from atap.classify.taxonomy import FusionLabel
from atap.core.registry import register
from atap.core.render import is_error_observation

# 规则 → MAST 代码【适配】：MAST 14 模式无工具格式/输出格式专门类
_RULE_MAST = {
    "malformed_tool_call": "FM-2.6",
    "no_progress_loop": "FM-1.3",
    "premature_success_claim": "FM-3.1",
    "invalid_output": "FM-1.1",
}

_VERIFIER_REJECT_RE = re.compile(
    r"missing|required|format|格式|必填|citation|invalid", re.I
)


@register
class RulePackClassifier(Classifier):
    stage = "classify"
    name = "rule_pack"

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} 无 R0 事件流：请先配置 canonical_events"
            )
        findings: list[dict[str, Any]] = []
        notes: list[str] = []
        findings += self._malformed_tool_call(events)
        findings += self._premature_success(events)
        findings += self._invalid_output(events)
        loop_findings, note = self._no_progress_loop(bundle)
        findings += loop_findings
        if note:
            notes.append(note)
        fusion = [
            FusionLabel(
                mast=f["mast_code"],
                evidence_step=f["step"],
                reason=f["rule"],
            ).to_dict()
            for f in findings
        ]
        bundle.put(
            "classify",
            self.name,
            {
                "findings": findings,
                "fusion": fusion,
                "cost": "free",
                "rules": sorted(_RULE_MAST),
                "notes": notes,
            },
        )

    # ------------------------------------------------------------------

    def _finding(
        self, rule: str, step: int, agent: str, evidence: list[str]
    ) -> dict[str, Any]:
        return {
            "rule": rule,
            "step": step,
            "agent": agent,
            "mast_code": _RULE_MAST[rule],
            "evidence": [e[:160] for e in evidence],
            "confidence": 0.9,   # 确定性规则命中即高置信【工程选择】
        }

    def _malformed_tool_call(self, events) -> list[dict[str, Any]]:
        result_by_call: dict[str, Any] = {}
        for ev in events:
            if ev.kind == "TOOL_RESULT" and ev.refs:
                result_by_call.setdefault(ev.refs[-1], ev)
        out: list[dict[str, Any]] = []
        for ev in events:
            if ev.kind != "TOOL_CALL":
                continue
            res = result_by_call.get(ev.id)
            empty_args = not ev.payload
            err = res is not None and is_error_observation(
                str(res.payload.get("content", ""))
            )
            if empty_args or err:
                out.append(
                    self._finding(
                        "malformed_tool_call",
                        ev.index,
                        ev.agent,
                        [
                            f"[{ev.index}] {ev.agent} TOOL_CALL {ev.action} {dict(ev.payload)}",
                            (
                                f"[{res.index}] TOOL_RESULT :: {str(res.payload.get('content', ''))[:120]}"
                                if res is not None
                                else "(参数集为空)"
                            ),
                        ],
                    )
                )
        return out

    def _premature_success(self, events) -> list[dict[str, Any]]:
        """submit 前无任何 read 型调用（无证据即提交）。"""
        read_actions = {"read_doc"}
        submits = [e for e in events if e.kind == "TOOL_CALL" and e.action == "submit"]
        out: list[dict[str, Any]] = []
        for s in submits:
            prior_reads = [
                e for e in events
                if e.index < s.index and e.kind == "TOOL_CALL"
                and (e.action or "") in read_actions
            ]
            if prior_reads:
                continue
            decisions = [
                e for e in events
                if e.index < s.index and e.kind == "LLM_CALL"
            ]
            target = decisions[-1] if decisions else s
            out.append(
                self._finding(
                    "premature_success_claim",
                    target.index,
                    target.agent,
                    [
                        f"[{target.index}] {target.agent} LLM_CALL :: {str(target.payload.get('content', ''))[:120]}",
                        f"[{s.index}] submit 于未读取任何文档时提交",
                    ],
                )
            )
        return out

    def _invalid_output(self, events) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ev in events:
            if ev.kind != "VERIFIER":
                continue
            content = str(ev.payload.get("content", ""))
            if not content.startswith("failed"):
                continue
            if not _VERIFIER_REJECT_RE.search(content):
                continue
            decisions = [
                e for e in events if e.index < ev.index and e.kind == "LLM_CALL"
            ]
            if not decisions:
                continue
            last = decisions[-1]
            out.append(
                self._finding(
                    "invalid_output",
                    last.index,
                    last.agent,
                    [
                        f"[{last.index}] {last.agent} LLM_CALL :: {str(last.payload.get('content', ''))[:120]}",
                        f"[{ev.index}] VERIFIER :: {content[:120]}",
                    ],
                )
            )
            break   # 一条轨迹记一次结构化驳回
        return out

    def _no_progress_loop(self, bundle) -> tuple[list[dict[str, Any]], str | None]:
        min_repeats = int(self.param("min_repeats", 3))
        loops = bundle.get("analyze", "loop_detect")
        hits: list[dict[str, Any]] = []
        if isinstance(loops, dict) and isinstance(loops.get("detected"), list):
            for d in loops["detected"]:
                if d.get("predicate") not in ("search_loop", "redundant_search"):
                    continue
                step = d.get("repetition_onset_index") or d.get("start_index")
                agent = self._agent_at(bundle, step)
                hits.append(
                    self._finding(
                        "no_progress_loop",
                        int(step),
                        agent,
                        [f"{d['predicate']}@{d['start_index']}..{d['end_index']}",
                         *d.get("evidence", [])[:2]],
                    )
                )
            return hits, None
        sigs_art = bundle.get("represent", "action_signature")
        sigs = sigs_art.get("signatures") if isinstance(sigs_art, dict) else None
        if isinstance(sigs, list) and sigs:
            counts: dict[str, dict[str, Any]] = {}
            for s in sigs:
                if s["action_class"] in ("SEARCH", "FILE_READ"):
                    counts.setdefault(s["signature"], {"n": 0, "first": s})["n"] += 1
            for sig, c in counts.items():
                if c["n"] >= min_repeats:
                    s = c["first"]
                    hits.append(
                        self._finding(
                            "no_progress_loop",
                            s["index"],
                            s["agent"],
                            [f"{sig} x{c['n']}（R5 签名自查回退）"],
                        )
                    )
            return hits, None
        return [], (
            "no_progress_loop：analyze/loop_detect 与 represent/action_signature"
            " 均缺席，该规则显式跳过（bundle 契约：不静默绕过）"
        )

    @staticmethod
    def _agent_at(bundle, step: int | None) -> str:
        if step is None:
            return "unknown"
        for ev in bundle.trajectory.events:
            if ev.index == step:
                return ev.agent
        return "unknown"
