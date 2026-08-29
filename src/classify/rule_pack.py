"""L0 free rule pack —— AgentDebugX, arXiv:2607.18754 §3.2 (the deterministic layer of Detect).

The paper's definition is a single sentence: "Deterministic rule packs
first target mechanically verifiable failures — malformed tool calls,
no-progress loops, invalid outputs, premature success — with no model
call." (exact trigger conditions live in the official repo and were not
published with the paper) —— this implementation's trigger conditions are
self-defined [adaptation], based solely on R0 observable events and never
reading meta["injected_fault"]:

* ``malformed_tool_call``: a TOOL_CALL with missing parameters (empty
  payload) or whose TOOL_RESULT is a structured error observation
  (error:/exception prefixes —— the latter also covers environment-side
  errors, a slight extension of the literal "malformed" [adaptation]);
  step = the call step;
* ``no_progress_loop``: consumes the ``analyze/loop_detect`` artifact
  only for its search-surface predicates (search_loop/redundant_search
  hits); re_read_churn/tool_oscillation hits are not consumed
  [adaptation: the paper names the rule target only ("no-progress
  loops") and leaves the predicate mapping open]. When the artifact is
  present it settles only the search surface — none of the consumed
  predicates reports repeated FILE_READs — so the re-read surface still
  goes through an R5 FILE_READ signature self-check fallback instead of
  being silently dropped [adaptation: 2026-08-27 re-review fix — an
  artifact containing only re_read_churn/tool_oscillation hits used to
  yield no finding, no fallback and no note, exactly disabling the R5
  fallback (which covers FILE_READ) when the artifact existed]; when
  the whole artifact is absent the fallback scans SEARCH and FILE_READ
  signatures (the same signature appearing >= ``min_repeats`` times
  across the whole trajectory); when neither exists it explicitly skips
  and leaves a record; the consumption boundary is recorded in
  ``notes`` whenever the artifact is present;
* ``premature_success_claim``: no successful FILE_READ before a submit
  (terminal submission) —— claiming completion without evidence (failed
  reads or reads without observations do not count as evidence; the
  ``read_doc`` action vocabulary is sandbox-bound [adaptation]); step =
  the last LLM_CALL before the submit (the decision step that decided to
  skip retrieval, aligned with the Who&When Eq.5 earliest-decisive-error
  convention);
* ``invalid_output``: structured VERIFIER rejection (failure description
  contains missing/required/format-type words); step = the last LLM_CALL
  before the rejection (the answer-generation step).

Relation to judge labeling (per the paper): the rules hit "mechanically
verifiable failures" and detect symptoms rather than root causes ——
findings are seeds for attribution; a rule miss does not mean no error
(the L1 judge covers the remaining modes). The MAST dimension mapping
follows the [adaptation] mapping in sandbox/faults.py.

Artifact: ``{"findings": [...], "fusion": [...], "cost": "free", ...}``;
free and exhaustive: successful trajectories also pass through the rules
(anti-patterns in successful trajectories are still process signals).
"""

from __future__ import annotations

import re
from typing import Any

from atap.classify.base import Classifier
from atap.classify.taxonomy import FusionLabel
from atap.core.registry import register
from atap.core.render import is_error_observation

# Rule → MAST code [adaptation]: MAST's 14 modes have no dedicated class for
# tool-format/output-format issues
_RULE_MAST = {
    "malformed_tool_call": "FM-2.6",
    "no_progress_loop": "FM-1.3",
    "premature_success_claim": "FM-3.1",
    "invalid_output": "FM-1.1",
}

_VERIFIER_REJECT_RE = re.compile(
    r"missing|required|format|citation|invalid", re.I
)


@register
class RulePackClassifier(Classifier):
    stage = "classify"
    name = "rule_pack"
    requires = (("represent", "canonical_events"),)   # predicates run over the R0 event stream

    def run_one(self, bundle, ctx) -> None:
        events = bundle.trajectory.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
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
            "confidence": 0.9,   # deterministic rule hit implies high confidence [engineering choice]
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
                                else "(argument set is empty)"
                            ),
                        ],
                    )
                )
        return out

    def _premature_success(self, events) -> list[dict[str, Any]]:
        """No successful FILE_READ before a submit (failed reads are not evidence —— submitting without evidence)."""
        read_actions = {"read_doc"}  # sandbox action vocabulary [adaptation]
        result_by_call: dict[str, Any] = {}
        for ev in events:
            if ev.kind == "TOOL_RESULT" and ev.refs:
                result_by_call.setdefault(ev.refs[-1], ev)
        submits = [e for e in events if e.kind == "TOOL_CALL" and e.action == "submit"]
        out: list[dict[str, Any]] = []
        for s in submits:
            prior_ok_reads = [
                e for e in events
                if e.index < s.index and e.kind == "TOOL_CALL"
                and (e.action or "") in read_actions
                and self._read_succeeded(e, result_by_call)
            ]
            if prior_ok_reads:
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
                        f"[{s.index}] submit issued without any successful document read",
                    ],
                )
            )
        return out

    @staticmethod
    def _read_succeeded(call, result_by_call: dict[str, Any]) -> bool:
        """Whether a read call has a successful observation (no observation / an error observation is not evidence)."""
        res = result_by_call.get(call.id)
        if res is None:
            return False
        return not is_error_observation(str(res.payload.get("content", "")))

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
            break   # record one structured rejection per trajectory
        return out

    def _no_progress_loop(self, bundle) -> tuple[list[dict[str, Any]], str | None]:
        """No-progress-loop rule; the consumption boundary of the loop
        artifact is: search-surface predicates (search_loop/
        redundant_search) are consumed, re_read_churn/tool_oscillation are
        not. Therefore, when the artifact is present, only the search surface
        is settled by it — the re-read surface (repeated FILE_READ
        signatures) still goes through the R5 fallback so a re-read loop is
        not silently dropped, and the consumed/fallback counts are recorded
        in the returned note (observable trace)."""
        min_repeats = int(self.param("min_repeats", 3))
        loops = bundle.get("analyze", "loop_detect")
        hits: list[dict[str, Any]] = []
        if isinstance(loops, dict) and isinstance(loops.get("detected"), list):
            consumed = 0
            unconsumed = 0
            for d in loops["detected"]:
                if d.get("predicate") not in ("search_loop", "redundant_search"):
                    unconsumed += 1
                    continue
                # onset=0 must not fall back to start (`or` would treat 0 as falsy)
                step = (
                    d["repetition_onset_index"]
                    if d.get("repetition_onset_index") is not None
                    else d.get("start_index")
                )
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
                consumed += 1
            # the artifact settles only the search surface: the re-read
            # surface (repeated FILE_READ signatures, the R5 fallback's own
            # coverage) is re-checked via R5 rather than silently dropped
            fallback = self._r5_fallback(bundle, min_repeats, ("FILE_READ",))
            note = (
                f"no_progress_loop: consumed {consumed} hit(s) from "
                "analyze/loop_detect (search_loop/redundant_search only; "
                f"{unconsumed} re_read_churn/tool_oscillation hit(s) not "
                "consumed by this rule); re-read surface covered by the R5 "
                "FILE_READ signature fallback"
            )
            if fallback is None:
                note += (
                    ": represent/action_signature absent -- the re-read "
                    "surface is explicitly skipped (bundle contract: no "
                    "silent bypass)"
                )
            else:
                note += f": {len(fallback)} hit(s)"
                hits += fallback
            return hits, note
        fallback = self._r5_fallback(bundle, min_repeats, ("SEARCH", "FILE_READ"))
        if fallback is not None:
            return fallback, None
        return [], (
            "no_progress_loop: both analyze/loop_detect and "
            "represent/action_signature are absent; this rule is explicitly "
            "skipped (bundle contract: no silent bypass)"
        )

    def _r5_fallback(
        self, bundle, min_repeats: int, classes: tuple[str, ...]
    ) -> list[dict[str, Any]] | None:
        """R5 signature self-check over the given action classes: any same
        signature appearing >= ``min_repeats`` times across the whole
        trajectory. Returns None when the R5 artifact is absent (the caller
        leaves an explicit skip record; bundle contract: no silent
        bypass)."""
        sigs_art = bundle.get("represent", "action_signature")
        sigs = sigs_art.get("signatures") if isinstance(sigs_art, dict) else None
        if not (isinstance(sigs, list) and sigs):
            return None
        counts: dict[str, dict[str, Any]] = {}
        for s in sigs:
            if s["action_class"] in classes:
                counts.setdefault(s["signature"], {"n": 0, "first": s})["n"] += 1
        out: list[dict[str, Any]] = []
        for sig, c in counts.items():
            if c["n"] >= min_repeats:
                s = c["first"]
                out.append(
                    self._finding(
                        "no_progress_loop",
                        s["index"],
                        s["agent"],
                        [f"{sig} x{c['n']} (R5 signature self-check fallback)"],
                    )
                )
        return out

    @staticmethod
    def _agent_at(bundle, step: int | None) -> str:
        if step is None:
            return "unknown"
        for ev in bundle.trajectory.events:
            if ev.index == step:
                return ev.agent
        return "unknown"
