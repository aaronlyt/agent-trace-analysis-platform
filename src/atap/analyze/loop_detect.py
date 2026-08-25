"""循环检测谓词 —— TraceProbe, arXiv:2607.06184 Table II（结构单轨迹检测器）。

机制：动作签名序列上的确定性谓词（无 LLM、零成本、全量轨迹——含成功
轨迹：原文 search loop 在成功轨迹中也有 41.1% 出现率，它是过程信号而
非失败判据）。本模块实现四个结构谓词：

* **search_loop**（最稳，原文 Primary failure-associated clue）：至少
  ``min_consecutive`` 个连续 SEARCH/FILE_READ 动作，期间无 FILE_WRITE、
  无验证 COMMAND（原文冻结阈值 10，为 SWE-Bench 口径；原文明示阈值应在
  目标基准审计后复用——玩具域配置取 3【适配】）；
* **re_read_churn**（次稳）：同一指纹的 FILE_READ 在 ``window`` 动作窗内
  出现 ≥ ``read_repeat`` 次，且其间无对该目标的写（原文 3 次/10 窗）；
* **tool_oscillation**（弱）：同一文件目标 ≥ ``osc_cycles`` 个
  READ-WRITE-READ 环，且中间的写 FAILED/REVERTED（原文 2 环）；
* **redundant_search**：同一规范化查询的 SEARCH 在 ``window`` 窗内重复
  ≥ ``search_repeat`` 次（原文 2 次/10 窗）。

消费 R5 产物（``represent/action_signature`` 的 signatures 列表）；
缺失则显式报错（bundle 契约：不静默绕过）。检测≠归因：产物只报告
命中的谓词、区间与重复 onset（区间内第二个同签名动作——"第二次出现
即首次重复"），供 L0 规则包与归因做种子，不判定根因。
"""

from __future__ import annotations

from typing import Any

from atap.analyze.base import Analyzer
from atap.core.registry import register

_PREDICATES = ("search_loop", "re_read_churn", "tool_oscillation", "redundant_search")


def _is_validation_command(sig: dict[str, Any]) -> bool:
    return sig["action_class"] == "COMMAND" and sig.get("target") == "verify"


@register
class LoopDetectAnalyzer(Analyzer):
    stage = "analyze"
    name = "loop_detect"

    def run_one(self, bundle, ctx) -> None:
        art = bundle.get("represent", "action_signature")
        sigs = art.get("signatures") if isinstance(art, dict) else None
        if not isinstance(sigs, list):
            raise ValueError(
                f"{bundle.trace_id} 缺少 represent/action_signature 产物："
                "loop_detect 消费 R5 动作签名，请先配置 action_signature"
            )
        thresholds = {
            "min_consecutive": int(self.param("min_consecutive", 10)),
            "read_repeat": int(self.param("read_repeat", 3)),
            "search_repeat": int(self.param("search_repeat", 2)),
            "window": int(self.param("window", 10)),
            "osc_cycles": int(self.param("osc_cycles", 2)),
        }
        detected: list[dict[str, Any]] = []
        detected += self._search_loop(sigs, thresholds["min_consecutive"])
        detected += self._re_read_churn(sigs, thresholds)
        detected += self._redundant_search(sigs, thresholds)
        detected += self._tool_oscillation(sigs, thresholds["osc_cycles"])
        bundle.put(
            "analyze",
            self.name,
            {
                "detected": detected,
                "predicates_evaluated": list(_PREDICATES),
                "thresholds": thresholds,
                "source": "represent/action_signature",
                "cost": "free",
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _search_loop(
        sigs: list[dict[str, Any]], min_consecutive: int
    ) -> list[dict[str, Any]]:
        """连续 SEARCH|FILE_READ（FILE_WRITE/验证命令断开；REASON 等不断开
        ——原文显式只列这两类断开条件）。"""
        out: list[dict[str, Any]] = []
        run: list[dict[str, Any]] = []
        runs: list[list[dict[str, Any]]] = []
        for s in sigs:
            if s["action_class"] in ("SEARCH", "FILE_READ"):
                run.append(s)
            elif s["action_class"] == "FILE_WRITE" or _is_validation_command(s):
                if run:
                    runs.append(run)
                run = []
        if run:
            runs.append(run)
        for r in runs:
            if len(r) >= min_consecutive:
                counts: dict[str, int] = {}
                onset = None
                for s in r:
                    counts[s["signature"]] = counts.get(s["signature"], 0) + 1
                    if counts[s["signature"]] == 2 and onset is None:
                        onset = s["index"]  # 第二次出现 = 首次重复
                out.append(
                    {
                        "predicate": "search_loop",
                        "start_index": r[0]["index"],
                        "end_index": r[-1]["index"],
                        "length": len(r),
                        "repetition_onset_index": onset,
                        "repeated_signatures": sorted(
                            {s["signature"] for s in r if counts[s["signature"]] >= 2}
                        ),
                        "evidence": [s["signature"] for s in r][: 6],
                    }
                )
        return out

    @staticmethod
    def _re_read_churn(
        sigs: list[dict[str, Any]], th: dict[str, int]
    ) -> list[dict[str, Any]]:
        reads = [s for s in sigs if s["action_class"] == "FILE_READ"]
        out: list[dict[str, Any]] = []
        reported: set[str] = set()
        for i, s in enumerate(reads):
            tgt = s["target"]
            if tgt in reported:
                continue
            window = [
                r for r in reads[i: i + th["window"]]
                if r["target"] == tgt
                and not any(
                    w["action_class"] == "FILE_WRITE" and w["target"] == tgt
                    for w in sigs
                    if s["index"] < w["index"] < r["index"]
                )
            ]
            if len(window) >= th["read_repeat"]:
                reported.add(tgt)
                out.append(
                    {
                        "predicate": "re_read_churn",
                        "start_index": window[0]["index"],
                        "end_index": window[-1]["index"],
                        "repeats": len(window),
                        "target": tgt,
                        "evidence": [w["signature"] for w in window],
                    }
                )
        return out

    @staticmethod
    def _redundant_search(
        sigs: list[dict[str, Any]], th: dict[str, int]
    ) -> list[dict[str, Any]]:
        searches = [s for s in sigs if s["action_class"] == "SEARCH"]
        out: list[dict[str, Any]] = []
        reported: set[str] = set()
        for i, s in enumerate(searches):
            q = s["target"]
            if q in reported or q is None:
                continue
            window = [
                r for r in searches[i: i + th["window"]] if r["target"] == q
            ]
            if len(window) >= th["search_repeat"]:
                reported.add(q)
                out.append(
                    {
                        "predicate": "redundant_search",
                        "start_index": window[0]["index"],
                        "end_index": window[-1]["index"],
                        "repeats": len(window),
                        "target": q,
                        "evidence": [w["signature"] for w in window],
                    }
                )
        return out

    @staticmethod
    def _tool_oscillation(
        sigs: list[dict[str, Any]], osc_cycles: int
    ) -> list[dict[str, Any]]:
        """同文件 READ-WRITE-READ 环，中间写 FAILED/REVERTED。"""
        targets = {
            s["target"]
            for s in sigs
            if s["action_class"] in ("FILE_READ", "FILE_WRITE")
        }
        out: list[dict[str, Any]] = []
        for tgt in targets:
            seq = [
                s for s in sigs
                if s["action_class"] in ("FILE_READ", "FILE_WRITE")
                and s["target"] == tgt
            ]
            cycles = 0
            first_idx = None
            for i in range(len(seq) - 2):
                a, w, b = seq[i], seq[i + 1], seq[i + 2]
                if (
                    a["action_class"] == "FILE_READ"
                    and w["action_class"] == "FILE_WRITE"
                    and b["action_class"] == "FILE_READ"
                    and w["effect"] in ("FAILED", "REVERTED")
                ):
                    cycles += 1
                    if first_idx is None:
                        first_idx = a["index"]
            if cycles >= osc_cycles:
                out.append(
                    {
                        "predicate": "tool_oscillation",
                        "start_index": first_idx,
                        "cycles": cycles,
                        "target": tgt,
                        "evidence": [s["signature"] for s in seq][: 6],
                    }
                )
        return out
