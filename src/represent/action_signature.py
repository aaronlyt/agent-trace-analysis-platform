"""R5 action signatures + effect tags + anchor set/milestones -- TraceProbe, arXiv:2607.06184.

Mechanism (original sections III-B/III-E, Table I/II):
* **Nine canonical action classes**: FILE_READ / FILE_WRITE / SEARCH /
  COMMAND / PLAN / NAVIGATE / FETCH / AGENT_SPAWN / REASON, each with an
  **argument fingerprint** (target: a comparable object such as path/command/
  query/plan digest);
* **Seven effect tags** (Table I; three more than the survey doc:
  FAILED/RECORDED/REASONING): SURVIVED (write persists to the terminal
  state) / REVERTED (a later same-target write overwrites it) / FAILED (the
  action returns an error status) / JUSTIFIED (reads a task-relevant
  artifact or runs a verification command) / RECORDED (successful
  non-workspace meta-actions: plan updates/navigation/fetching/sub-agent
  spawn) / OFF-ANCHOR (successful read/search falls outside the anchor set)
  / REASONING (pure reasoning step);
* **Anchor set**: the original uses the gold-patch file set (oracle); the
  oracle-free fallback is "the trajectory's own surviving writes and
  test/import references" (this domain has no FILE_WRITE, so that route is
  unavailable) -- this implementation instead uses the **documents read by
  successful trajectories of the same task** as the anchor set
  [adaptation: a new construction outside the original's fallback route --
  it depends on success/failure outcome labels (i.e. on the very label the
  diagnostic pipeline diagnoses) and would fall in the paper's
  oracle-*grounded* territory (its own milestones block is explicitly marked
  oracle-grounded); it reads neither the gold patch nor
  meta["injected_fault"]; broader than the gold-patch "modified files"
  semantics (a successful trajectory's whole read list, not just
  gold-deriving documents, counts), so OFF-ANCHOR decisions are
  correspondingly conservative — with the declared exclusion that **failed
  reads never count**: a read whose observation is an error (effect=FAILED)
  contributes neither to the anchor set nor to milestone anchor reads];
  when the success reference reads nothing at all, the anchor degrades to
  None (an empty anchor set would mark every read OFF-ANCHOR — declared in
  the artifact note);
* **Milestones** M1..M5 (original: first anchor read / first anchor write /
  all anchors written / first passing validation / first justified action)
  and **monotonic LCS alignment** against the success-reference signature
  (CONVERGE).

Domain adaptation (research-QA sandbox, no file writes) [adaptation]:
* search->SEARCH(query), read_doc->FILE_READ(doc_id), submit->COMMAND,
  VERIFIER->COMMAND (verification command), HANDOFF->AGENT_SPAWN
  (delegation), LLM_CALL with phase=plan->PLAN, other LLM_CALL->REASON;
  NAVIGATE/FETCH have no corresponding events. TOOL_RESULT/TASK_START/
  TASK_END/AGENT_MESSAGE do not enter the signature sequence
  (environment-side observations/bookkeeping/message artifacts -- this
  domain's three-agent pipeline only passes messages via HANDOFF);
  TOOL_RESULT participates in effect determination via refs back to its
  owning TOOL_CALL;
* No FILE_WRITE => REVERTED never occurs; submit acts as the terminal
  "write" (SURVIVED=validation passed, otherwise FAILED); the "write"
  semantics of milestones M2/M3 are mapped to the nearest equivalents
  "first anchor-hitting search (M2) / all anchor documents read (M3)";
* Default effect tags when there is no anchor set (single-trajectory scope /
  no successful trajectory in the group): successful read_doc->JUSTIFIED,
  successful search->RECORDED (an engineering default when task relevance
  cannot be determined; the two are asymmetric);
* PLAN/AGENT_SPAWN are tagged RECORDED **unconditionally, without checking
  success** — the original ties the tag to success ("a nonworkspace
  meta-action ... is recorded when it succeeds"); this domain's LLM_CALL
  (plan/reason) and HANDOFF (spawn) events carry no environment-side result
  observation to verify against, so a success check is unreachable in-domain
  [adaptation: recorded-unchecked];
* Anchor-hit for SEARCH is a format heuristic: search result text
  containing the doc id with **word boundaries** (``[d1]`` / ``(d1)`` /
  ``, d1`` / `` d1`` …) is treated as having read an anchor document (the
  sandbox search output format; format changes would cause misses). The
  match is boundary-guarded — anchor ``d1`` does **not** match ``d10``
  (plain substrings like ``[d1`` would hit ``[d10``);
* The LCS alignment is a reduced version of CONVERGE: a match = action
  class + fingerprint + compatible effect (successful classes are mutually
  compatible, failures/meta-actions share the label, OFF-ANCHOR is treated
  leniently); divergences are recorded as unmatched contiguous spans,
  without the three-layer classification (file selection/edit
  stability/completion).

Engineering decision: canonical action classes are **not written back** to
``TraceEvent.action`` (that field carries the raw tool name from the
collection layer, which judge rendering lines and pseudo-judge rules depend
on; rewriting it would change the judge-visible view); all derived data goes
into this artifact. Artifact key ``represent/action_signature``.

Threshold declaration: the original's loop-predicate thresholds were frozen
on SWE-Bench (e.g. search loop >=10 consecutive); the original explicitly
states "thresholds should be audited on the target benchmark before reuse" --
this module only does representation and sets no thresholds; consumers
(loop_detect/rule_pack) pass thresholds explicitly as parameters.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from atap.core.registry import register
from atap.core.render import is_error_observation
from atap.core.schema import TraceEvent
from atap.represent.base import Representer

# Nine canonical action classes (Table I Canonical Action label set)
ACTION_CLASSES = (
    "FILE_READ", "FILE_WRITE", "SEARCH", "COMMAND", "PLAN",
    "NAVIGATE", "FETCH", "AGENT_SPAWN", "REASON",
)

# Seven effect tags (Table I Effect Label label set)
EFFECT_LABELS = (
    "SURVIVED", "FAILED", "REVERTED", "JUSTIFIED",
    "RECORDED", "OFF-ANCHOR", "REASONING",
)

# R0 event kinds included in the signature sequence (environment-side observations and bookkeeping excluded)
_SIGNED_KINDS = ("LLM_CALL", "TOOL_CALL", "HANDOFF", "VERIFIER")


def classify_event(ev: TraceEvent) -> tuple[str, str | None] | None:
    """R0 event -> (canonical action class, argument fingerprint). Returns None
    for events that do not enter the signature sequence.

    Mapping table [adaptation]: see the module docstring.
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
        return "COMMAND", "verify"          # verification command
    if ev.kind == "HANDOFF":
        return "AGENT_SPAWN", str(ev.payload.get("to", ""))  # delegatee as fingerprint
    if ev.kind == "LLM_CALL":
        if ev.phase == "plan":
            return "PLAN", _norm_query(str(ev.payload.get("content", "")))[:60]
        return "REASON", None
    return None


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _effect_compatible(a: str, b: str) -> bool:
    """CONVERGE effect-compatibility rules (reduced version): successful
    workspace effects are mutually compatible; failures/meta-actions share
    the same label; OFF-ANCHOR is lenient (uncertain exploration does not
    count as divergence)."""
    ws = {"SURVIVED", "JUSTIFIED"}
    if a in ws and b in ws:
        return True
    if "OFF-ANCHOR" in (a, b):
        return True
    return a == b


def _lcs_matches(
    ref: list[dict[str, Any]], cmp_: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Monotonic LCS alignment (O(nm)): returns (reference index, compared
    index) match pairs."""
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
        """Single-trajectory scope: signatures + anchor-independent effect
        tags; anchor set/LCS/milestones explicitly degraded."""
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
                "note": "single-trajectory scope: the anchor set/milestones/LCS"
                        " require a cross-trajectory success reference"
                        " (run_corpus); explicitly skipped here",
            },
        )

    def run_corpus(self, bundles, ctx) -> None:
        """Cross-trajectory scope: group by task_id; successful trajectories
        provide the anchor set and reference signatures, then each trajectory
        gets anchor-dependent effect tags, milestones, and LCS alignment
        (the second use case where aggregation precedes per-instance -- the
        first is attribute/sbfl)."""
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
                for b in ok_bundles:  # anchor set = documents read by successful trajectories (outcome-label-dependent; no gold/injected-fault access)
                    anchor.update(
                        s["target"]
                        for s in self._signatures(b.trajectory, anchor=None)
                        if s["action_class"] == "FILE_READ"
                        and s["effect"] != "FAILED"   # failed reads contribute no read content
                    )
                if not anchor:
                    # the success reference read nothing: an empty anchor set
                    # would mark every read OFF-ANCHOR and make M3 vacuously
                    # false — degrade to anchor=None (declared in the note)
                    anchor = None
                # reference = the successful trajectory with the fewest steps (original: per-task most-efficient)
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
                    alignment = self._alignment(
                        ref_bundle.trace_id, ref_sigs, sigs
                    )
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
                        f"task {key}: success reference read nothing; anchor"
                        " unavailable (an empty anchor set would mark every"
                        " read OFF-ANCHOR, so the anchor-dependent tags and"
                        " milestones are skipped)"
                        if ok_bundles else
                        f"task {key} has no successful trajectory in its group:"
                        " anchor set/milestones/LCS unavailable"
                        " (TraceProbe's anchor set requires a task-relevant reference)"
                    )
                b.put("represent", self.name, bundle_artifact)

    # ------------------------------------------------------------------

    def _signatures(
        self, trajectory, anchor: set[str] | None
    ) -> list[dict[str, Any]]:
        events = trajectory.events
        # TOOL_RESULT.refs point to their TOOL_CALL (direction: result -> call);
        # build a call -> result mapping for effect determination
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
                # M4 (first passing validation) needs pass/fail info; the effect tag cannot carry it
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
            return "RECORDED"            # non-workspace meta-action; success means RECORDED
        if ev.kind == "VERIFIER":
            return "JUSTIFIED"           # verification command
        # The three TOOL_CALL kinds: submit / read_doc / search
        if ev.action == "submit":
            v = verify_by_call.get(ev.id)
            if v is not None:
                return (
                    "SURVIVED"
                    if str(v.payload.get("content", "")).startswith("passed")
                    else "FAILED"
                )
            return "FAILED"              # a submission without a verification observation is treated as failed [adaptation]
        res = result_by_call.get(ev.id)
        failed = res is not None and is_error_observation(str(res.payload.get("content", "")))
        if failed:
            return "FAILED"
        # Successful read/search: anchor-dependent decision (JUSTIFIED / OFF-ANCHOR)
        if anchor is not None and action_class in ("FILE_READ", "SEARCH"):
            return "JUSTIFIED" if _touches_anchor(action_class, target, res, anchor) else "OFF-ANCHOR"
        if action_class == "FILE_READ":
            # unreachable with an anchor set (handled above): an anchor-less
            # read cannot be checked against anything, so it is justified
            return "JUSTIFIED"
        return "RECORDED"

    @staticmethod
    def _milestones(
        sigs: list[dict[str, Any]], anchor: set[str] | None
    ) -> dict[str, Any] | None:
        """M1 first anchor read / M2 first anchor-hitting search / M3 all
        anchors read / M4 first passing validation / M5 first justified
        action (in the original M2/M3 are "writes"; this domain has no
        writes, nearest mapping per the docstring [adaptation]). Milestones
        not reached are right-censored (reached=False)."""
        if anchor is None:
            return None
        anchor_reads = [
            s for s in sigs
            if s["action_class"] == "FILE_READ" and s["target"] in anchor
            and s["effect"] != "FAILED"   # a failed read delivered no content: not an anchor read
        ]
        # The earliest moment of "all anchors read" = the max over the index
        # of each anchor's **first** read (repeated reads do not rewrite the
        # milestone moment)
        first_read_by_anchor: dict[str, int] = {}
        for s in anchor_reads:
            first_read_by_anchor.setdefault(s["target"], s["index"])
        all_read_step = (
            max(first_read_by_anchor.values())
            if anchor and set(first_read_by_anchor) == set(anchor)
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
        self, ref_trace_id: str, ref_sigs: list[dict[str, Any]], sigs: list[dict[str, Any]]
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
            "reference_trace": ref_trace_id,  # reference (successful) trace id, same basis as anchor.reference_trace
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
    # SEARCH: a hit counts if the search result text mentions any anchor
    # document. Word-boundary match: anchor "d1" must not sit inside a longer
    # id (plain substrings like "[d1" would also hit "[d10"); the sandbox
    # corpus ids d1-d6 never extend, so this guard changes no current
    # behavior — it only narrows future false hits.
    if res is None:
        return False
    content = str(res.payload.get("content", ""))
    return any(
        re.search(rf"(?<![0-9A-Za-z]){re.escape(d)}(?![0-9A-Za-z])", content)
        for d in anchor
    )
