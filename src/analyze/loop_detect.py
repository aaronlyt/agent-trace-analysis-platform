"""Loop detection predicates —— TraceProbe, arXiv:2607.06184 Table II (structural single-trajectory detectors).

Mechanism: deterministic predicates over the action-signature sequence (no
LLM, zero cost, over all trajectories —— including successful ones: in the
paper the search loop still occurs in 41.1% of successful trajectories; it
is a process signal, not a failure verdict). This module implements the
four structural predicates:

* **search_loop** (most robust; the paper's primary failure-associated
  clue): at least ``min_consecutive`` consecutive SEARCH/FILE_READ
  actions with no FILE_WRITE and no validation COMMAND in between (the
  paper froze the threshold at 10 for the SWE-Bench setting, and it
  explicitly says thresholds should be reused only after auditing the
  target benchmark —— the toy-domain configuration uses 3 [adaptation].
  Audit disclosure for this corpus: normal and non-repetition fault traces
  max out at 2 consecutive read actions (search→read_doc) while the
  injected step_repetition fault produces 3 identical searches (a run of 4
  with the trailing read_doc), so **any threshold in {3,4} separates all
  corpus traces** —— the offline detection rate is determined by this
  threshold choice, not measured; 3 also equals the fault's repetition
  count);
* **re_read_churn** (second most robust): FILE_READs of the same
  fingerprint appearing >= ``read_repeat`` times within a ``window``
  action window of the action sequence (``window`` consecutive signature
  actions) with no write to that target in between (paper: 3 times /
  10-window);
* **tool_oscillation** (weak): >= ``osc_cycles`` READ-WRITE-READ cycles
  on the same file target, where the intervening write FAILED/REVERTED
  (paper: 2 cycles); the triple is sought on the per-target
  read/write subsequence, so its members may be separated by other
  actions [inference: the paper's Table II does not state whether the
  R-W-R triple must be immediately consecutive in the full action
  sequence];
* **redundant_search**: SEARCHes with the same normalized query repeated
  >= ``search_repeat`` times within a ``window`` action window of the
  action sequence (paper: 2 times / 10-window).

Consumes the R5 artifact (the signatures list of
``represent/action_signature``); if it is missing, an explicit error is
raised (bundle contract: no silent bypass). Detection ≠ attribution: the
artifact only reports the predicates hit, their intervals, and the
repetition onset (the second same-signature action within the interval
—— "the second occurrence is the first repetition"), serving as seeds
for the L0 rule pack and attribution; it does not decide root causes.
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
    requires = (("represent", "action_signature"),)   # R5 signatures are the input unit

    def run_one(self, bundle, ctx) -> None:
        art = bundle.get("represent", "action_signature")
        sigs = art.get("signatures") if isinstance(art, dict) else None
        if not isinstance(sigs, list):
            raise ValueError(
                f"{bundle.trace_id} is missing the represent/action_signature "
                "artifact: loop_detect consumes R5 action signatures; "
                "configure action_signature first"
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
        """Consecutive SEARCH|FILE_READ (FILE_WRITE/validation commands break the
        run; REASON etc. do not —— the paper explicitly lists only those two
        break conditions)."""
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
                        onset = s["index"]  # second occurrence = first repetition
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
        """FILE_READs of the same target >= read_repeat times within a window
        window of the action sequence, with no write to that target in
        between (a write splits same-target reads into non-adjacent
        groups). The window is counted by **position in the action
        sequence** (the paper's 10-action window = 10 consecutive actions,
        including other actions), not 10 reads over the read subsequence."""
        by_target: dict[str, list[dict[str, Any]]] = {}
        for pos, s in enumerate(sigs):
            if s["action_class"] in ("FILE_READ", "FILE_WRITE") and s["target"] is not None:
                by_target.setdefault(s["target"], []).append({"pos": pos, "sig": s})
        out: list[dict[str, Any]] = []
        reported: set[str] = set()
        for tgt, seq in by_target.items():
            if tgt in reported:
                continue
            groups: list[list[dict[str, Any]]] = []
            group: list[dict[str, Any]] = []
            for item in seq:
                if item["sig"]["action_class"] == "FILE_WRITE":
                    if group:
                        groups.append(group)
                    group = []
                else:
                    group.append(item)
            if group:
                groups.append(group)
            for g in groups:
                for i in range(len(g)):
                    run = [g[i]]
                    for j in range(i + 1, len(g)):
                        if g[j]["pos"] - g[i]["pos"] + 1 <= th["window"]:
                            run.append(g[j])
                        else:
                            break
                    if len(run) >= th["read_repeat"]:
                        reported.add(tgt)
                        out.append(
                            {
                                "predicate": "re_read_churn",
                                "start_index": run[0]["sig"]["index"],
                                "end_index": run[-1]["sig"]["index"],
                                "repeats": len(run),
                                "target": tgt,
                                "evidence": [r["sig"]["signature"] for r in run],
                            }
                        )
                        break
                if tgt in reported:
                    break
        return out

    @staticmethod
    def _redundant_search(
        sigs: list[dict[str, Any]], th: dict[str, int]
    ) -> list[dict[str, Any]]:
        """SEARCHes with the same normalized query >= search_repeat times
        within a window window of the action sequence (window counted by
        action-sequence position, same as _re_read_churn)."""
        positions: dict[str, list[dict[str, Any]]] = {}
        for pos, s in enumerate(sigs):
            if s["action_class"] == "SEARCH" and s["target"] is not None:
                positions.setdefault(s["target"], []).append({"pos": pos, "sig": s})
        out: list[dict[str, Any]] = []
        reported: set[str] = set()
        for q, items in positions.items():
            if q in reported:
                continue
            for i in range(len(items)):
                run = [items[i]]
                for j in range(i + 1, len(items)):
                    if items[j]["pos"] - items[i]["pos"] + 1 <= th["window"]:
                        run.append(items[j])
                    else:
                        break
                if len(run) >= th["search_repeat"]:
                    reported.add(q)
                    out.append(
                        {
                            "predicate": "redundant_search",
                            "start_index": run[0]["sig"]["index"],
                            "end_index": run[-1]["sig"]["index"],
                            "repeats": len(run),
                            "target": q,
                            "evidence": [r["sig"]["signature"] for r in run],
                        }
                    )
                    break
        return out

    @staticmethod
    def _tool_oscillation(
        sigs: list[dict[str, Any]], osc_cycles: int
    ) -> list[dict[str, Any]]:
        """READ-WRITE-READ cycles on the same file, with the intervening write FAILED/REVERTED.

        The triple window slides over the **per-target** FILE_READ/FILE_WRITE
        subsequence: the R/W/R members need only be adjacent among that
        target's file actions and may be separated by other actions (or
        actions on other targets) in the full sequence; adjacent cycles may
        share their boundary read (R-W-R-W-R = 2 cycles) [inference: Table II
        fixes only ">= 2 READ-WRITE-READ cycles where the middle write is
        deterministically labeled failed or reverted" and does not state
        whether the triple must be consecutive in the full action sequence].
        Targets with a None fingerprint are skipped (same filter as
        _re_read_churn)."""
        targets = {
            s["target"]
            for s in sigs
            if s["action_class"] in ("FILE_READ", "FILE_WRITE")
            and s["target"] is not None
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
            last_idx = None
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
                    last_idx = b["index"]   # closing read of the latest cycle
            if cycles >= osc_cycles:
                out.append(
                    {
                        "predicate": "tool_oscillation",
                        "start_index": first_idx,
                        "end_index": last_idx,
                        "cycles": cycles,
                        "target": tgt,
                        "evidence": [s["signature"] for s in seq][: 6],
                    }
                )
        return out
