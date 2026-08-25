"""R5 动作签名 + 效果标签 + 锚集/里程碑 —— TraceProbe, arXiv:2607.06184。

机制（原文 §III-B/§III-E，Table I/II）：
* **九类规范动作**：FILE_READ / FILE_WRITE / SEARCH / COMMAND / PLAN /
  NAVIGATE / FETCH / AGENT_SPAWN / REASON，附**参数指纹**（target：路径/
  命令/查询/计划摘要的可比对象）；
* **七个效果标签**（Table I；比综述文档多出 FAILED/RECORDED/REASONING）：
  SURVIVED（写持久化到终态）/ REVERTED（后续同目标写覆盖）/ FAILED
  （动作返回错误状态）/ JUSTIFIED（读到任务相关产物或跑了验证命令）/
  RECORDED（成功的非工作区元动作：计划更新/导航/取数/子 agent 派生）/
  OFF-ANCHOR（成功的读/检索落在锚集外）/ REASONING（纯推理步）；
* **锚集**：原文用 gold-patch 文件集（oracle）；非基准轨迹"可从存活写与
  测试/导入引用推导任务相关性"——本实现取同任务**成功轨迹读过的文档**
  为锚集（oracle-free 的同义替换，不读 meta["injected_fault"]）；
* **里程碑** M1..M5（原文：首个锚读/首个锚写/全部锚已写/首个通过验证/
  首个 justified 动作）与成功参照签名的**单调 LCS 对齐**（CONVERGE）。

域适配（研究问答沙盒，无文件写）【适配】：
* search→SEARCH(query)、read_doc→FILE_READ(doc_id)、submit→COMMAND、
  VERIFIER→COMMAND(验证命令)、HANDOFF→AGENT_SPAWN(委托)、phase=plan 的
  LLM_CALL→PLAN、其余 LLM_CALL→REASON；NAVIGATE/FETCH 无对应事件。
  TOOL_RESULT/TASK_START/TASK_END 不入签名序列（环境侧观测/簿记），
  TOOL_RESULT 经 refs 并回所属 TOOL_CALL 参与效果判定；
* 无 FILE_WRITE ⇒ REVERTED 恒不出现、submit 作为终态"写"（SURVIVED=验证
  通过，否则 FAILED）；里程碑 M2/M3 的"写"语义就近映射为"锚文档全部
  读毕/首个命中锚的检索"；
* LCS 对齐为 CONVERGE 的缩减版：匹配 = 动作类+指纹+效果相容（成功类
  互通、失败/元动作同标签、OFF-ANCHOR 宽容），分歧记为未匹配连续段，
  不做三层分类（file selection/edit stability/completion）。

工程决策：规范动作类**不写回** ``TraceEvent.action``（该字段承载采集层
原始工具名，判官渲染行与伪判官规则依赖它；改写等于变更判官可见视图），
全部派生数据放本产物。产物键 ``represent/action_signature``。

阈值声明：原文循环谓词阈值冻结于 SWE-Bench（如 search loop ≥10 连续），
原文明示"阈值应在目标基准上审计后复用"——本模块只做表征，不设阈值；
消费方（loop_detect/rule_pack）以参数显式给阈值。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from atap.core.registry import register
from atap.core.render import is_error_observation
from atap.core.schema import TraceEvent
from atap.represent.base import Representer

# 九类规范动作（Table I Canonical Action 标签集）
ACTION_CLASSES = (
    "FILE_READ", "FILE_WRITE", "SEARCH", "COMMAND", "PLAN",
    "NAVIGATE", "FETCH", "AGENT_SPAWN", "REASON",
)

# 七个效果标签（Table I Effect Label 标签集）
EFFECT_LABELS = (
    "SURVIVED", "FAILED", "REVERTED", "JUSTIFIED",
    "RECORDED", "OFF-ANCHOR", "REASONING",
)

# 签名序列纳入的 R0 事件 kind（排除环境侧观测与簿记）
_SIGNED_KINDS = ("LLM_CALL", "TOOL_CALL", "HANDOFF", "VERIFIER")


def classify_event(ev: TraceEvent) -> tuple[str, str | None] | None:
    """R0 事件 → (规范动作类, 参数指纹)。不入签名序列的事件返回 None。

    映射表【适配】：见模块 docstring。
    """
    if ev.kind == "TOOL_CALL":
        act = ev.action or ""
        if act == "search":
            return "SEARCH", _norm_query(str(ev.payload.get("query", "")))
        if act == "read_doc":
            return "FILE_READ", str(ev.payload.get("doc_id", ""))
        if act == "submit":
            return "COMMAND", "submit"
        return "COMMAND", act or "tool"
    if ev.kind == "VERIFIER":
        return "COMMAND", "verify"          # 验证命令
    if ev.kind == "HANDOFF":
        return "AGENT_SPAWN", str(ev.payload.get("to", ""))  # 委派对象为指纹
    if ev.kind == "LLM_CALL":
        if ev.phase == "plan":
            return "PLAN", _norm_query(str(ev.payload.get("content", "")))[:60]
        return "REASON", None
    return None


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _effect_compatible(a: str, b: str) -> bool:
    """CONVERGE 效果相容规则（缩减版）：成功工作区效果互通；失败/元动作
    同标签；OFF-ANCHOR 宽容（不确定探索不计分歧）。"""
    ws = {"SURVIVED", "JUSTIFIED"}
    if a in ws and b in ws:
        return True
    if "OFF-ANCHOR" in (a, b):
        return True
    return a == b


def _lcs_matches(
    ref: list[dict[str, Any]], cmp_: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """单调 LCS 对齐（O(nm)）：返回 (参照序号, 被比序号) 匹配对。"""
    n, m = len(ref), len(cmp_)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            ok = (
                ref[i]["action_class"] == cmp_[j]["action_class"]
                and ref[i]["target"] == cmp_[j]["target"]
                and _effect_compatible(ref[i]["effect"], cmp_[j]["effect"])
            )
            dp[i][j] = dp[i + 1][j + 1] + 1 if ok else max(dp[i + 1][j], dp[i][j + 1])
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        ok = (
            ref[i]["action_class"] == cmp_[j]["action_class"]
            and ref[i]["target"] == cmp_[j]["target"]
            and _effect_compatible(ref[i]["effect"], cmp_[j]["effect"])
        )
        if ok:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


@register
class ActionSignatureRepresenter(Representer):
    stage = "represent"
    name = "action_signature"

    def run_one(self, bundle, ctx) -> None:
        """单轨迹作用域：签名 + 锚无关效果标签；锚集/LCS/里程碑显式降级。"""
        sigs = self._signatures(bundle.trajectory, anchor=None)
        bundle.put(
            "represent",
            self.name,
            {
                "signatures": sigs,
                "anchor": None,
                "alignment": None,
                "milestones": None,
                "stats": self._stats(sigs),
                "note": "单轨迹作用域：锚集/里程碑/LCS 需要跨轨迹成功参照"
                        "（run_corpus），此处显式跳过",
            },
        )

    def run_corpus(self, bundles, ctx) -> None:
        """跨轨迹作用域：按 task_id 分组，成功轨迹给锚集与参照签名，
        再逐轨迹做锚相关效果标签、里程碑与 LCS 对齐（聚合先于单例的
        第二个用例——第一个是 attribute/sbfl）。"""
        groups: dict[str, list] = {}
        for b in bundles:
            key = str(b.trajectory.meta.get("task_id") or "")
            groups.setdefault(key, []).append(b)

        for key, grp in groups.items():
            if not key:
                for b in grp:
                    self.run_one(b, ctx)
                continue
            ok_bundles = [b for b in grp if b.succeeded]
            anchor: set[str] | None = None
            ref_bundle = None
            if ok_bundles:
                anchor = set()
                for b in ok_bundles:  # 锚集 = 成功轨迹读过的文档（oracle-free）
                    anchor.update(
                        s["target"]
                        for s in self._signatures(b.trajectory, anchor=None)
                        if s["action_class"] == "FILE_READ"
                    )
                # 参照 = 步数最少的成功轨迹（原文 per-task most-efficient）
                ref_bundle = min(
                    ok_bundles, key=lambda b: len(b.trajectory.events)
                )
            ref_sigs = (
                self._signatures(ref_bundle.trajectory, anchor)
                if ref_bundle is not None
                else None
            )
            for b in grp:
                sigs = self._signatures(b.trajectory, anchor)
                milestones = self._milestones(sigs, anchor)
                alignment = None
                if ref_sigs is not None and b is not ref_bundle:
                    alignment = self._alignment(b, ref_sigs, sigs)
                bundle_artifact: dict[str, Any] = {
                    "signatures": sigs,
                    "anchor": (
                        {
                            "source": "success_reference",
                            "reference_trace": ref_bundle.trace_id,
                            "docs": sorted(anchor),
                        }
                        if anchor is not None
                        else None
                    ),
                    "alignment": alignment,
                    "milestones": milestones,
                    "stats": self._stats(sigs),
                }
                if anchor is None:
                    bundle_artifact["note"] = (
                        f"任务 {key} 组内无成功轨迹：锚集/里程碑/LCS 不可用"
                        "（TraceProbe 锚集需要任务相关参照）"
                    )
                b.put("represent", self.name, bundle_artifact)

    # ------------------------------------------------------------------

    def _signatures(
        self, trajectory, anchor: set[str] | None
    ) -> list[dict[str, Any]]:
        events = trajectory.events
        # TOOL_RESULT.refs 指向其 TOOL_CALL（方向：结果→调用），
        # 由此建 调用→结果 映射供效果判定
        result_by_call: dict[str, TraceEvent] = {}
        for ev in events:
            if ev.kind == "TOOL_RESULT" and ev.refs:
                result_by_call.setdefault(ev.refs[-1], ev)
        verify_by_call: dict[str, TraceEvent] = {}
        for ev in events:
            if ev.kind == "VERIFIER" and ev.refs:
                verify_by_call.setdefault(ev.refs[-1], ev)

        sigs: list[dict[str, Any]] = []
        for ev in events:
            cls = classify_event(ev)
            if cls is None:
                continue
            action_class, target = cls
            effect = self._effect(
                ev, action_class, target, anchor, result_by_call, verify_by_call
            )
            sig: dict[str, Any] = {
                "event_id": ev.id,
                "index": ev.index,
                "agent": ev.agent,
                "action_class": action_class,
                "target": target,
                "effect": effect,
                "signature": (
                    f"{action_class}({target})"
                    if target is not None
                    else action_class
                ),
            }
            if ev.kind == "VERIFIER":
                # M4（首个通过的验证）需要通过与否，效果标签不足以承载
                sig["passed"] = str(ev.payload.get("content", "")).startswith("passed")
            sigs.append(sig)
        return sigs

    @staticmethod
    def _effect(
        ev: TraceEvent,
        action_class: str,
        target: str | None,
        anchor: set[str] | None,
        result_by_call: dict[str, TraceEvent],
        verify_by_call: dict[str, TraceEvent],
    ) -> str:
        if action_class == "REASON":
            return "REASONING"
        if action_class == "PLAN" or action_class == "AGENT_SPAWN":
            return "RECORDED"            # 非工作区元动作，成功即 RECORDED
        if ev.kind == "VERIFIER":
            return "JUSTIFIED"           # 验证命令
        # TOOL_CALL 三类：submit / read_doc / search
        if ev.action == "submit":
            v = verify_by_call.get(ev.id)
            if v is not None:
                return "SURVIVED" if v.payload.get("content", "").startswith("passed") else "FAILED"
            return "FAILED"              # 无验证观测的提交视为失败【适配】
        res = result_by_call.get(ev.id)
        failed = res is not None and is_error_observation(str(res.payload.get("content", "")))
        if failed:
            return "FAILED"
        # 成功的 read/search：锚相关判定（JUSTIFIED / OFF-ANCHOR）
        if anchor is not None and action_class in ("FILE_READ", "SEARCH"):
            return "JUSTIFIED" if _touches_anchor(action_class, target, res, anchor) else "OFF-ANCHOR"
        if action_class == "FILE_READ":
            return "JUSTIFIED" if anchor is None else "OFF-ANCHOR"
        return "RECORDED"

    @staticmethod
    def _milestones(
        sigs: list[dict[str, Any]], anchor: set[str] | None
    ) -> dict[str, Any] | None:
        """M1 首个锚读 / M2 首个命中锚的检索 / M3 全部锚读毕 / M4 首个
        通过验证 / M5 首个 justified 动作（原文 M2/M3 为"写"，本域无写，
        就近映射见 docstring【适配】）。未达成的里程碑右删失（reached=False）。"""
        if anchor is None:
            return None
        anchor_reads = [
            s for s in sigs
            if s["action_class"] == "FILE_READ" and s["target"] in anchor
        ]
        all_read_step = (
            anchor_reads[-1]["index"] if anchor and {s["target"] for s in anchor_reads} == set(anchor)
            else None
        )
        first_pass_verify = next(
            (s["index"] for s in sigs if s.get("passed")), None
        )
        first_justified = next(
            (s["index"] for s in sigs if s["effect"] == "JUSTIFIED"), None
        )
        first_anchor_search = next(
            (s["index"] for s in sigs
             if s["action_class"] == "SEARCH" and s["effect"] == "JUSTIFIED"),
            None,
        )
        return {
            "M1_first_anchor_read": {
                "reached": bool(anchor_reads),
                "step": anchor_reads[0]["index"] if anchor_reads else None,
            },
            "M2_first_anchor_search": {
                "reached": first_anchor_search is not None,
                "step": first_anchor_search,
            },
            "M3_all_anchors_read": {
                "reached": all_read_step is not None,
                "step": all_read_step,
            },
            "M4_first_passing_validation": {
                "reached": first_pass_verify is not None,
                "step": first_pass_verify,
            },
            "M5_first_justified": {
                "reached": first_justified is not None,
                "step": first_justified,
            },
        }

    def _alignment(
        self, bundle, ref_sigs: list[dict[str, Any]], sigs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        matches = _lcs_matches(ref_sigs, sigs)
        matched_cmp = {j for _, j in matches}
        spans: list[dict[str, Any]] = []
        run: list[int] = []
        for j in range(len(sigs)):
            if j in matched_cmp:
                if run:
                    spans.append(
                        {
                            "start_index": sigs[run[0]]["index"],
                            "end_index": sigs[run[-1]]["index"],
                            "length": len(run),
                            "actions": [sigs[k]["signature"] for k in run],
                        }
                    )
                    run = []
            else:
                run.append(j)
        if run:
            spans.append(
                {
                    "start_index": sigs[run[0]]["index"],
                    "end_index": sigs[run[-1]]["index"],
                    "length": len(run),
                    "actions": [sigs[k]["signature"] for k in run],
                }
            )
        n_off = sum(1 for s in sigs if s["effect"] == "OFF-ANCHOR")
        return {
            "reference_trace": bundle.trajectory.meta.get("task_id"),
            "lcs_len": len(matches),
            "coverage": round(len(matches) / len(sigs), 4) if sigs else 0.0,
            "n_added": len(sigs) - len(matches),
            "off_anchor_ratio": round(n_off / len(sigs), 4) if sigs else 0.0,
            "divergence_spans": spans,
        }

    @staticmethod
    def _stats(sigs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_signed": len(sigs),
            "action_classes": dict(Counter(s["action_class"] for s in sigs)),
            "effects": dict(Counter(s["effect"] for s in sigs)),
        }


def _touches_anchor(
    action_class: str, target: str | None, res: TraceEvent | None, anchor: set[str]
) -> bool:
    if action_class == "FILE_READ":
        return target in anchor
    # SEARCH：检索结果文本提及任一锚文档即视为命中锚
    if res is None:
        return False
    content = str(res.payload.get("content", ""))
    return any(
        f"[{d}" in content or f" {d}]" in content or f", {d}" in content
        for d in anchor
    )
