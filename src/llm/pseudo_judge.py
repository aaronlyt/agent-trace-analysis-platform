"""Deterministic pseudo-judge -- the default FakeLLMClient handler, letting the LLM-judge pipeline run offline.

It performs rule-based judgments solely from the **judge-visible trajectory
text** (the folded view rendered by core/render.py) and never reads ground
truth such as meta["injected_fault"] (that would leak the answer and make
acceptance tests meaningless). The rules correspond one-to-one with the
**observable symptoms** of the sandbox faults:

=======================  ==============================  ========  ==========
Fault (sandbox)          Observable symptom              Rule #    MAST code
=======================  ==============================  ========  ==========
malformed tool call      TOOL_RESULT carries failure     1        FM-2.6
                         indicator words
step repetition          3 adjacent identical            2        FM-1.3
                         TOOL_CALLs
information withholding  claims no results although      3        FM-2.4
                         search results were non-empty
premature termination    submit without any read_doc     4        FM-3.1
ungrounded citation      cites a doc id never read       5        FM-3.3
task-spec violation      VERIFIER reports missing        6        FM-1.1
                         format/required fields
(fallback)               the last LLM_CALL               7        FM-2.6
=======================  ==============================  ========  ==========
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from atap.llm.base import ChatMessage
from atap.core.render import TRACE_BEGIN, TRACE_END, is_error_observation

MAST_CODES = {
    "malformed_tool_call": "FM-2.6",
    "step_repetition": "FM-1.3",
    "info_withholding": "FM-2.4",
    "premature_termination": "FM-3.1",
    "ungrounded_citation": "FM-3.3",
    "disobey_task_spec": "FM-1.1",
}


@dataclass
class Line:
    idx: int
    kind: str
    agent: str
    action: str | None
    payload: str
    content: str


@dataclass
class Signature:
    """A detected anomalous symptom."""

    fault: str
    step: int
    agent: str
    code: str
    reason: str
    fix: str
    detail: str = ""


_LINE_RE = re.compile(r"^\[(\d+)\]\s+(\S+)\s+(\S+)(?:\s(.*))?$")
_DOC_ID_RE = re.compile(r"\bd(\d+)\b")
_SEARCH_DOCS_RE = re.compile(r"search results[^:]*:\s*(\d+)\s*docs\s*\[([^\]]*)\]")


def _parse_block(block: str) -> list[Line]:
    lines: list[Line] = []
    for raw in block.splitlines():
        raw = raw.strip()
        if not raw.startswith("["):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            continue
        idx, kind, agent, rest = int(m.group(1)), m.group(2), m.group(3), m.group(4) or ""
        # An event line without an action looks like "[11] VERIFIER verifier :: failed: ...";
        # the regex already consumed the spaces before "::", so split on "::" rather than " :: "
        if "::" in rest:
            left, _, content = rest.partition("::")
            content = content.strip()
        else:
            left, content = rest, ""
        tokens = left.split()
        action: str | None = None
        payload_parts: list[str] = []
        for tok in tokens:
            if tok.startswith("{"):
                payload_parts.append(tok)
            elif action is None:
                action = tok
            else:
                payload_parts.append(tok)
        lines.append(
            Line(
                idx=idx,
                kind=kind,
                agent=agent,
                action=action,
                payload=" ".join(payload_parts),
                content=content.strip(),
            )
        )
    return lines


def find_trace_block(messages: list[ChatMessage]) -> str | None:
    """Take the last trajectory block (TRACE_BEGIN..TRACE_END) from the message list."""
    for msg in reversed(messages):
        content = str(msg.get("content", ""))
        if TRACE_BEGIN in content and TRACE_END in content:
            start = content.index(TRACE_BEGIN) + len(TRACE_BEGIN)
            end = content.index(TRACE_END)
            return content[start:end]
    return None


def find_outcome(messages: list[ChatMessage]) -> bool | None:
    """Take the outcome from the task header (SUCCESS counts as a successful trajectory);
    returns None when the view carries no outcome line (the J.1 protocol of
    judge_eval), leaving the caller to infer from trajectory evidence."""
    for msg in reversed(messages):
        content = str(msg.get("content", ""))
        m = re.search(r"outcome:\s*(SUCCESS|FAILURE)", content)
        if m:
            return m.group(1) == "SUCCESS"
    return None


def _verifier_passed(lines: list[Line]) -> bool | None:
    """Infer success/failure from VERIFIER event lines inside the trajectory
    (passed:/failed: prefixes).

    Environment-side verifier feedback is part of the trajectory (already
    visible to the judge), so it does not violate J.1's constraint of
    "providing no external success/failure result"; returns None when there
    is no VERIFIER line.
    """
    verdicts = [ln for ln in lines if ln.kind == "VERIFIER"]
    if not verdicts:
        return None
    return verdicts[-1].content.startswith("passed")


# ---------------------------------------------------------------------------
# Symptom-detection rules (short-circuit in order; a trajectory usually hits
# the symptom of just one injected fault)
# ---------------------------------------------------------------------------


def _sig(fault: str, step: int, agent: str, detail: str) -> Signature:
    code = MAST_CODES[fault]
    reasons = {
        "malformed_tool_call": f"step {step}: the tool call issued by {agent} returned an error; the action does not match the intent (reasoning-action mismatch)",
        "step_repetition": f"step {step}: {agent} repeats the same tool call without making progress",
        "info_withholding": f"step {step}: {agent} claims that no documents were found although earlier search results were non-empty -- key information was withheld from downstream",
        "premature_termination": f"step {step}: {agent} submitted the answer without reading any evidence document (premature termination)",
        "ungrounded_citation": f"step {step}: {agent} cited a document never read via read_doc (incorrect verification)",
        "disobey_task_spec": f"step {step}: the final answer of {agent} violates the task spec (missing required fields/format)",
    }
    fixes = {
        "malformed_tool_call": f"Avoid malformed_tool_call: validate argument completeness before issuing the tool call at step {step}.",
        "step_repetition": f"Avoid step_repetition: do not repeat the same tool call at step {step}; use the existing results to move forward.",
        "info_withholding": f"Avoid info_withholding: faithfully report the retrieved documents at step {step} and pass the results downstream.",
        "premature_termination": f"Avoid premature_termination: the decision at step {step} was preparing to submit without searching or reading the evidence; search and read_doc before submitting the answer.",
        "ungrounded_citation": f"Avoid ungrounded_citation: at step {step} cite only documents actually read via read_doc.",
        "disobey_task_spec": f"Avoid disobey_task_spec: at step {step} give the answer in the format required by the task (including the required document number).",
    }
    return Signature(
        fault=fault, step=step, agent=agent, code=code,
        reason=reasons[fault], fix=fixes[fault], detail=detail,
    )


def detect_signatures(lines: list[Line]) -> list[Signature]:
    sigs: list[Signature] = []

    # Rule 1: malformed tool call -- the first structured error observation
    # (error:/exception prefix), attributed to the immediately preceding
    # TOOL_CALL (dictionary words in running prose do not count as error
    # observations)
    calls = [ln for ln in lines if ln.kind == "TOOL_CALL"]
    results = [ln for ln in lines if ln.kind == "TOOL_RESULT"]
    for res in results:
        if is_error_observation(res.content):
            prev_calls = [c for c in calls if c.idx < res.idx]
            if prev_calls:
                call = prev_calls[-1]
                sigs.append(_sig("malformed_tool_call", call.idx, call.agent, res.content[:120]))
            break

    # Rule 2: step repetition -- 3 adjacent TOOL_CALLs with fully identical
    # signatures, attributed to the 2nd one
    for i in range(len(calls) - 2):
        a, b, c3 = calls[i], calls[i + 1], calls[i + 2]
        if (a.agent, a.action, a.payload) == (b.agent, b.action, b.payload) == (c3.agent, c3.action, c3.payload):
            sigs.append(_sig("step_repetition", b.idx, b.agent, f"repeated {a.action}"))
            break

    # Rule 3: information withholding -- claims no results although an
    # earlier search did return documents
    search_hits: list[int] = []  # event idx of non-empty search hits
    for ln in lines:
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "search":
            m = _SEARCH_DOCS_RE.search(ln.content)
            if m and int(m.group(1)) > 0:
                search_hits.append(ln.idx)
        if ln.kind in ("AGENT_MESSAGE", "LLM_CALL", "HANDOFF") and re.search(
            r"no (relevant|results|documents)|nothing found", ln.content, re.I
        ):
            if any(h < ln.idx for h in search_hits):
                sigs.append(_sig("info_withholding", ln.idx, ln.agent, ln.content[:120]))
                break

    # Rule 4: premature termination -- never read_doc before submit.
    # Attributed to the last decision (LLM_CALL) before submit rather than
    # the submit action itself: the planning step that decided to skip
    # retrieval precedes the termination action (Who&When Eq.5 earliest),
    # aligned with onset=plan in the sandbox ground truth
    read_doc_calls = [c for c in calls if (c.action or "") == "read_doc"]
    decisions = [ln for ln in lines if ln.kind == "LLM_CALL"]
    for c in calls:
        if (c.action or "") == "submit" and not [r for r in read_doc_calls if r.idx < c.idx]:
            target = next((d for d in reversed(decisions) if d.idx < c.idx), c)
            sigs.append(_sig("premature_termination", target.idx, target.agent, c.payload[:120]))
            break

    # Rule 5: ungrounded citation -- the answer/message assertively cites a
    # doc id never read as of that step
    read_ids: set[str] = set()
    for ln in lines:
        if ln.kind == "TOOL_CALL" and (ln.action or "") == "read_doc":
            m = _DOC_ID_RE.search(ln.payload)
            if m:
                read_ids.add(f"d{m.group(1)}")
            continue
        if ln.kind not in ("LLM_CALL", "AGENT_MESSAGE") or (ln.action or "") == "search":
            continue
        mentions = {f"d{m.group(1)}" for m in _DOC_ID_RE.finditer(ln.content)}
        if not mentions:
            continue
        is_assertion = bool(
            re.search(r"cite|cited|according to|based on", ln.content, re.I)
        )
        for did in mentions:
            cited_form = f"[{did}]" in ln.content or f"({did})" in ln.content
            if did not in read_ids and (is_assertion or cited_form):
                sigs.append(_sig("ungrounded_citation", ln.idx, ln.agent, f"cited {did} unread"))
                break
        if sigs and sigs[-1].fault == "ungrounded_citation":
            break

    # Rule 6: task-spec violation -- VERIFIER reports missing format/required
    # fields, attributed to the last answer generation
    verifier_issues = [
        ln for ln in lines
        if ln.kind == "VERIFIER" and re.search(r"missing|required|format", ln.content, re.I)
    ]
    if verifier_issues:
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        if llm_calls:
            last = llm_calls[-1]
            sigs.append(_sig("disobey_task_spec", last.idx, last.agent, verifier_issues[-1].content[:120]))

    # Rule 7: fallback
    if not sigs:
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        if llm_calls:
            last = llm_calls[-1]
            sigs.append(
                Signature(
                    fault="unknown", step=last.idx, agent=last.agent, code="FM-2.6",
                    reason=f"step {last.idx}: no explicit symptom observed; conservatively attributed to the last model decision",
                    fix=f"Re-examine the reasoning behind step {last.idx}.", detail="",
                )
            )
    return sigs


# ---------------------------------------------------------------------------
# Emit three kinds of structured JSON by tag (field names match the pydantic
# models of the algorithm modules)
# ---------------------------------------------------------------------------


def _segment_local_signatures(lines: list[Line]) -> list[Signature]:
    """**In-segment** symptom detection for binary-search localization rounds.

    Beyond reusing the whole-trajectory rules, three segment-visible criteria
    are added (the judge only sees the slice and must answer from in-slice
    evidence; the whole-trajectory rules themselves stay untouched -- they
    serve the all_at_once semantics):
    * a TOOL_CALL with an empty payload (a malformed call is visible on the
      call line itself, no need to wait for an error observation);
    * an LLM_CALL that self-reports "submit directly from memory" (the
      decision-step text of premature termination);
    * documents already read / a non-empty search within the segment,
      followed by a claim that "nothing was found" (an in-segment
      contradiction of information withholding).
    """
    sigs = detect_signatures(lines)
    # The fallback signature (fault="unknown") carries no symptom evidence:
    # a real judge shown a benign segment answers upper half, so the
    # pseudo-judge likewise must not treat the conservative-attribution
    # fallback as a symptom
    sigs = [s for s in sigs if s.fault != "unknown"]
    calls = [ln for ln in lines if ln.kind == "TOOL_CALL"]
    for c in calls:
        # A tool call with an empty payload (malformed): an empty dict renders
        # as an argument-less line, visible on the call line itself
        if c.action and not c.payload.strip() and not any(
            s.fault == "malformed_tool_call" and s.step == c.idx for s in sigs
        ):
            sigs.append(
                _sig("malformed_tool_call", c.idx, c.agent, "empty tool arguments")
            )
    evidence_idx: list[int] = []
    for ln in lines:
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "search":
            m = _SEARCH_DOCS_RE.search(ln.content)
            if m and int(m.group(1)) > 0:
                evidence_idx.append(ln.idx)
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "read_doc" \
                and not is_error_observation(ln.content):
            evidence_idx.append(ln.idx)
    for ln in lines:
        if ln.kind in ("AGENT_MESSAGE", "LLM_CALL", "HANDOFF") and re.search(
            r"no (relevant|results|documents)|nothing found", ln.content, re.I
        ):
            if any(h < ln.idx for h in evidence_idx) and not any(
                s.fault == "info_withholding" and s.step == ln.idx for s in sigs
            ):
                sigs.append(_sig("info_withholding", ln.idx, ln.agent, ln.content[:120]))
    for ln in lines:
        if ln.kind == "LLM_CALL" and re.search(
            r"recall|from memory|submit directly", ln.content, re.I
        ):
            if not any(s.fault == "premature_termination" for s in sigs):
                sigs.append(_sig("premature_termination", ln.idx, ln.agent, ln.content[:120]))
    return sigs


def _novel_symptom(lines: list[Line]) -> tuple[int, str, str]:
    """The symptom phrase for a novel candidate: the most frequently recurring
    inter-agent message (same text >= 2 times); if no message recurs, take a
    content snippet of the last LLM_CALL. Returns (step, agent, text)."""
    from collections import Counter

    msgs = [
        ln for ln in lines
        if ln.kind in ("HANDOFF", "AGENT_MESSAGE")
    ]
    counts = Counter(ln.content for ln in msgs)
    recurring = {c for c, n in counts.items() if n >= 2}
    if recurring:
        firsts = [ln for ln in msgs if ln.content in recurring]
        # deterministic tie-break: max() over a set iterates in hash order,
        # so equal-count candidates would drift across processes (and the
        # inducer's proposal name with them); sort by (-count, text) and take
        # the head instead
        best = sorted(recurring, key=lambda c: (-counts[c], c))[0]
        ln = next(l for l in firsts if l.content == best)
        return ln.idx, ln.agent, ln.content
    llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
    if llm_calls:
        last = llm_calls[-1]
        return last.idx, last.agent, last.content[:100]
    return (lines[-1].idx if lines else 0), "unknown", ""


def _match_extra_mode(system: str, symptom: str) -> str:
    """Word-level match between the symptom phrase and the accepted extended modes
    (the [extended] lines of the system definition block): >= 2 overlapping
    content words counts as a hit (the pseudo-judge's minimal simulation of
    the accept loop)."""
    extra = re.findall(r"^(\S+)\s*\[extended\]\s*(.+)$", system, re.M)
    toks = {t for t in re.findall(r"[a-z]{3,}", symptom.lower())}
    best_code, best_n = "novel", 0
    for code, text in extra:
        n = len(toks & {t for t in re.findall(r"[a-z]{3,}", text.lower())})
        if n > best_n:
            best_code, best_n = code, n
    return best_code if best_n >= 2 else "novel"


# ---------------------------------------------------------------------------
# Deterministic handlers for stage 4B (claim ledger / claim audit / tree
# diagnosis / CHIEF)
# ---------------------------------------------------------------------------

_CLAIM_LINE_RE = re.compile(
    r"^(\w+):\s+(.+?)\s+\(type=(\w+),\s*status=(\w+),\s+introduced_step=(\d+)\)",
    re.M,
)
_HARMFUL_LINE_RE = re.compile(
    r"^(\w+):\s+support=(\w+),\s+verdict=(\w+),\s+responsible_step=(\d+)", re.M
)


def _task_text(messages: list[ChatMessage]) -> str:
    for msg in messages:
        content = str(msg.get("content", ""))
        m = re.search(r"^task:\s*(.+)$", content, re.M)
        if m:
            return m.group(1)
    return ""


def _ledger_json(lines: list[Line], task: str) -> str:
    """Pseudo-extraction of the DRIFT ledger: decision-critical claims =
    plan commitments / handoff assertions / final-answer assertions (queries
    and tool calls do not count as commitments)."""
    claims: list[dict] = []

    def add(text: str, ctype: str, status: str, idx: int) -> None:
        claims.append({
            "id": f"c{len(claims) + 1}", "text": text[:120], "type": ctype,
            "status": status, "introduced_step": idx,
            "first_effective_step": None, "reuse_steps": [],
        })

    for ln in lines:
        c = ln.content
        if ln.kind == "LLM_CALL" and c.startswith("plan:"):
            if re.search(r"recall|from memory|submit directly", c, re.I):
                add("the answer is already known from memory", "entity",
                    "finalized", ln.idx)
            else:
                add("the task will be solved by search-read-report", "process",
                    "consequential", ln.idx)
        elif ln.kind == "HANDOFF" and "reporter" in (
            ln.payload + " " + (ln.action or "")
        ):
            # The {'to': 'reporter'} on a handoff line gets split by the line
            # parser into the action/payload fields; check both combined
            if re.search(r"no (relevant|usable|documents)|nothing found", c, re.I):
                add("no relevant documents were found", "evidence",
                    "consequential", ln.idx)
            elif c:
                add(c[:120], "evidence", "consequential", ln.idx)
        elif ln.kind == "LLM_CALL" and re.search(r"based on|answer:|proposes", c, re.I):
            add(c[:120], "evidence", "finalized", ln.idx)
    # Hard-constraint claims (the DRIFT Constraint family: the acceptance
    # constraints of the task spec also enter the ledger)
    submits = [ln for ln in lines
               if ln.kind == "TOOL_CALL" and (ln.action or "") == "submit"]
    if submits and re.search(r"must (end with|cite)", task, re.I):
        add("the final answer must cite a document actually read",
            "constraint", "consequential", submits[0].idx)
    claims = claims[:5]
    # first_effective / reuse: the first downstream step referencing the
    # claim's content after introduction
    for cl in claims:
        for ln in lines:
            if ln.idx > cl["introduced_step"] and ln.kind in (
                "HANDOFF", "LLM_CALL", "TOOL_CALL"
            ):
                cl["first_effective_step"] = ln.idx
                cl["reuse_steps"] = [ln.idx]
                break
    hard = []
    if re.search(r"must (end with|cite)", task, re.I):
        hard.append("the answer must cite the number of a document actually read")
    return json.dumps({
        "task_goal": task[:100],
        "hard_constraints": hard,
        "claims": claims,
        "notes": "",
    }, ensure_ascii=False)


def _support_json(messages: list[ChatMessage], lines: list[Line]) -> str:
    """Pseudo-verdict on the four support levels: whether the cited document
    was read / shown by search before the introducing step."""
    user = str(messages[-1].get("content", ""))
    reads: dict[str, int] = {}
    shown: set[str] = set()
    for ln in lines:
        if ln.kind == "TOOL_CALL" and (ln.action or "") == "read_doc":
            m = _DOC_ID_RE.search(ln.payload)
            if m:
                reads.setdefault(f"d{m.group(1)}", ln.idx)
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "search":
            shown |= {f"d{m.group(1)}" for m in _DOC_ID_RE.finditer(ln.content)}
    records = []
    submits = [ln for ln in lines
               if ln.kind == "TOOL_CALL" and (ln.action or "") == "submit"]
    submit_cited: set[str] = set()
    if submits:
        submit_cited = {
            f"d{m.group(1)}" for m in _DOC_ID_RE.finditer(submits[0].payload)
        }
    for cid, text, ctype, status, step in _CLAIM_LINE_RE.findall(user):
        step = int(step)
        if ctype == "process":
            records.append({
                "claim_id": cid, "support_status": "DIRECT",
                "support_steps": [step], "missing_support": "",
                "verdict": "supported", "responsible_step": None,
                "reason": "process-type claims are established directly by the trajectory's own behavior",
            })
            continue
        if ctype == "constraint":
            # Support for a constraint claim = whether the final answer
            # satisfies the constraint (DRIFT Constraint family: Constraint
            # Check Omission / Answer Format Error)
            if not submits:
                records.append({
                    "claim_id": cid, "support_status": "MISSING",
                    "support_steps": [], "missing_support": "no submission to check",
                    "verdict": "insufficient_but_nonharmful",
                    "responsible_step": None, "reason": "the trajectory has no submit step",
                })
                continue
            if submit_cited and submit_cited <= set(reads):
                records.append({
                    "claim_id": cid, "support_status": "DIRECT",
                    "support_steps": [submits[0].idx], "missing_support": "",
                    "verdict": "supported", "responsible_step": None,
                    "reason": "the final answer cites documents actually read",
                })
            else:
                decisions = [ln for ln in lines
                             if ln.kind == "LLM_CALL" and ln.idx < submits[0].idx]
                target = decisions[-1] if decisions else submits[0]
                records.append({
                    "claim_id": cid, "support_status": "CONFLICTING",
                    "support_steps": [],
                    "missing_support": "the final answer does not satisfy the citation constraint",
                    "verdict": "conflicting_support",
                    "responsible_step": target.idx,
                    "reason": "the answer-generation step did not follow the citation constraint of the task spec",
                })
            continue
        mentioned = {f"d{m.group(1)}" for m in _DOC_ID_RE.finditer(text)}
        read_before = {d for d, at in reads.items() if at < step}
        if re.search(r"no (relevant|usable|documents)|nothing found", text, re.I):
            if shown:
                records.append({
                    "claim_id": cid, "support_status": "CONFLICTING",
                    "support_steps": [], "missing_support": "search results are non-empty",
                    "verdict": "conflicting_support", "responsible_step": step,
                    "reason": "the no-results claim contradicts the search results already shown",
                })
            else:
                records.append({
                    "claim_id": cid, "support_status": "DIRECT",
                    "support_steps": [], "missing_support": "",
                    "verdict": "supported", "responsible_step": None,
                    "reason": "the search was indeed empty",
                })
            continue
        if mentioned and not (mentioned & read_before):
            records.append({
                "claim_id": cid, "support_status": "MISSING",
                "support_steps": [],
                "missing_support": f"cited {sorted(mentioned)} never read",
                "verdict": "harmful_unsupported_commitment",
                "responsible_step": step,
                "reason": "the claim relies on documents never read or verified",
            })
        elif not mentioned and re.search(r"memory|recall|unknown", text, re.I):
            records.append({
                "claim_id": cid, "support_status": "MISSING",
                "support_steps": [],
                "missing_support": "no search/read support at all",
                "verdict": "harmful_unsupported_commitment",
                "responsible_step": step,
                "reason": "asserts the answer from memory with no evidence shown",
            })
        else:
            records.append({
                "claim_id": cid, "support_status": "DIRECT",
                "support_steps": [step], "missing_support": "",
                "verdict": "supported", "responsible_step": None,
                "reason": "the cited document was read before introduction",
            })
    return json.dumps({"records": records}, ensure_ascii=False)


def _trace_json(messages: list[ChatMessage]) -> str:
    """Pseudo-implementation of dependency backtracking: the introduction
    point of the earliest unsupported claim (conservative, adds no new
    span)."""
    user = str(messages[-1].get("content", ""))
    steps = [int(m.group(4)) for m in _HARMFUL_LINE_RE.finditer(user)]
    first = min(steps) if steps else 0
    return json.dumps({
        "first_error_step": first,
        "error_steps": sorted(steps),
        "reason": "the introduction span of the earliest unsupported claim (conservative backtracking: no new span added)",
    }, ensure_ascii=False)


def _tree_stage_json(messages: list[ChatMessage]) -> str:
    """Tree-level localization: a suspicious stage = one containing a loop
    signal (>= 2 steps with the same prefix) or an error-result summary;
    loop signals take priority (CodeTracer workflow:
    loops/stalls/error commits)."""
    user = str(messages[-1].get("content", ""))
    sections: list[tuple[str, list[str]]] = []
    current = ("__header__", [])
    for line in user.splitlines():
        m = re.match(r"^== stage: (\S+) ==", line)
        if m:
            sections.append(current)
            current = (m.group(1), [])
        else:
            current[1].append(line)
    sections.append(current)
    loop_stages: list[str] = []
    error_stages: list[str] = []
    for name, body in sections:
        if name == "__header__":
            continue
        step_lines = [l.strip() for l in body if re.match(r"\s*e\d{3}\b", l)]
        # The prefix drops the event id (e003...): a loop signal looks at the
        # same agent and the same action
        prefixes = [" ".join(l.split()[1:4]) for l in step_lines]
        if any(prefixes.count(p) >= 2 for p in set(prefixes)):
            loop_stages.append(name)
        if any(re.search(r"\berror|failed|exhausted\b", l, re.I) for l in body):
            error_stages.append(name)
    suspicious = list(dict.fromkeys(loop_stages + error_stages))
    reason = (
        f"loop_signal={loop_stages or 'none'}; error_signal={error_stages or 'none'}"
        " (visible in the in-tree result summaries)"
    )
    return json.dumps({
        "suspicious_stages": suspicious[:3], "reason": reason,
    }, ensure_ascii=False)


_CHIEF_MECHANISM = {
    "malformed_tool_call": "local_error",
    "step_repetition": "executor_loop",
    "info_withholding": "dataflow_first_pollution",
    "premature_termination": "planning_error",
    "ungrounded_citation": "local_error",
    "disobey_task_spec": "local_error",
}


def _chief_localize_json(lines: list[Line]) -> str:
    """Pseudo-implementation of progressive filtering: symptom type → mechanism mapping (local/loop/dataflow/planning)."""
    sigs = [s for s in detect_signatures(lines) if s.fault != "unknown"]
    if not sigs:
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        last = llm_calls[-1] if llm_calls else lines[-1]
        return json.dumps({
            "responsible_agent": last.agent, "step": last.idx,
            "mechanism": "local_error",
            "reason": f"step {last.idx}: no sign of cross-step propagation observed; the error was generated locally",
            "fix_suggestion": f"Re-examine and correct the reasoning behind step {last.idx}.",
            "confidence": 0.4,
        }, ensure_ascii=False)
    s = sigs[0]
    mech = _CHIEF_MECHANISM.get(s.fault, "local_error")
    return json.dumps({
        "responsible_agent": s.agent, "step": s.step, "mechanism": mech,
        "reason": s.reason, "fix_suggestion": s.fix, "confidence": 0.7,
    }, ensure_ascii=False)


_CHIEF_ORACLE_TMPL = {
    "plan": {
        "goal": "formulate a search-read-report task plan",
        "preconditions": ["task received"],
        "key_evidence": ["task requirements"],
        "acceptance": "the plan includes search, read_doc, and a cited report, and does not skip retrieval",
    },
    "search": {
        "goal": "retrieve and read the documents that support the answer",
        "preconditions": ["the plan requires retrieval"],
        "key_evidence": ["search results and the content of read documents"],
        "acceptance": "the searcher performs the search, successfully reads at least one document, and faithfully hands the findings downstream",
    },
    "report": {
        "goal": "give a correct cited answer based on the documents read",
        "preconditions": ["upstream handed off the search findings"],
        "key_evidence": ["content of read documents and the upstream handoff message"],
        "acceptance": "the final answer cites documents actually read and the content is correct (verifier passes)",
    },
}


def _chief_oracle_json(messages: list[ChatMessage]) -> str:
    user = str(messages[-1].get("content", ""))
    oracles = []
    for m in re.finditer(r"^(S\d+) \(phase=(\w+)", user, re.M):
        sid, phase = m.group(1), m.group(2)
        tmpl = _CHIEF_ORACLE_TMPL.get(phase, {
            "goal": "complete the task", "preconditions": [], "key_evidence": [],
            "acceptance": "verifier passes",
        })
        oracles.append({"subtask_id": sid, **tmpl})
    return json.dumps({"oracles": oracles}, ensure_ascii=False)


def _chief_eval_json(messages: list[ChatMessage], lines: list[Line]) -> str:
    """Pseudo-implementation of subtask F_eval (based only on trajectory
    lines): report = verifier; search = successful reads or loops; plan =
    whether retrieval is skipped."""
    user = str(messages[-1].get("content", ""))
    verdicts = [ln for ln in lines if ln.kind == "VERIFIER"]
    passed = (
        verdicts[-1].content.startswith("passed") if verdicts else False
    )
    evals = []
    for m in re.finditer(
        r"^(S\d+) \(phase=(\w+), steps \[(\d+)\.\.(\d+)\]\)", user, re.M
    ):
        sid, phase, lo, hi = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        in_range = [ln for ln in lines if lo <= ln.idx <= hi]
        if phase == "report":
            ok = passed
            ev = (verdicts[-1].content[:60] if verdicts else "(no verifier)")
        elif phase == "search":
            ok_reads = [
                ln for ln in in_range
                if ln.kind == "TOOL_RESULT" and (ln.action or "") == "read_doc"
                and not is_error_observation(ln.content)
            ]
            calls = [ln for ln in in_range if ln.kind == "TOOL_CALL"]
            loop = any(
                (a.agent, a.action, a.payload) == (b.agent, b.action, b.payload)
                for a, b in zip(calls, calls[1:])
            )
            ok = bool(ok_reads) and not loop
            ev = f"reads={len(ok_reads)}; adjacent_repeat={loop}"
        elif phase == "plan":
            plans = [ln for ln in in_range if ln.kind == "LLM_CALL"]
            ok = not any(
                re.search(r"recall|from memory|submit directly", ln.content, re.I)
                for ln in plans
            )
            ev = "the plan includes retrieval" if ok else "the plan skips retrieval and submits directly"
        else:
            ok, ev = passed, ""
        evals.append({"subtask_id": sid, "passed": bool(ok), "evidence": ev})
    return json.dumps({"evals": evals}, ensure_ascii=False)


def _cf_oracle_json(messages: list[ChatMessage], lines: list[Line]) -> str:
    """Pseudo-implementation of the counterfactual oracle: if the candidate
    step hits a detectable symptom → the edit text = that symptom's
    corrective guidance (the named problem lets the sandbox middleware
    consume it); otherwise a minimal re-examination text (no correction)."""
    user = str(messages[-1].get("content", ""))
    m = re.search(r"[Cc]andidate error step \[(\d+)\]", user)
    step = int(m.group(1)) if m else -1
    sigs = [s for s in detect_signatures(lines) if s.fault != "unknown"]
    hit = next((s for s in sigs if s.step == step), None)
    if hit is None and sigs:
        hit = sigs[0] if sigs[0].step == step else None
    if hit is not None:
        return json.dumps({
            "expected": hit.reason,
            "edit_text": hit.fix,
        }, ensure_ascii=False)
    return json.dumps({
        "expected": f"step {step}: this step executed correctly per the task requirements",
        "edit_text": f"Re-examine the reasoning behind step {step} before continuing.",
    }, ensure_ascii=False)


def _dover_segment_json(lines: list[Line]) -> str:
    """Pseudo-implementation of trial segmentation: LLM_CALLs for
    plan/re-plan (content starting with plan:) are the split points; with no
    planning message the whole trajectory is a single trial."""
    plans = [ln for ln in lines
             if ln.kind == "LLM_CALL" and ln.content.startswith("plan:")]
    if not plans:
        last = lines[-1].idx if lines else 0
        return json.dumps({"trials": [
            {"trial_index": 0, "plan_step": 0, "exec_range": [0, last]}
        ]}, ensure_ascii=False)
    trials = []
    for i, p in enumerate(plans):
        if i + 1 < len(plans):
            # The next planning step is the split point, so this trial's
            # execution range ends one step before it (Table 6 of the paper:
            # non-overlapping trial ranges 0-38 / 39-65)
            end = plans[i + 1].idx - 1
        else:
            end = lines[-1].idx if lines else p.idx
        trials.append({
            "trial_index": i, "plan_step": p.idx,
            "exec_range": [p.idx, end],
        })
    return json.dumps({"trials": trials}, ensure_ascii=False)


def _dover_proposer_json(lines: list[Line], succeeded: bool) -> str:
    """Pseudo-implementation of the failed-trial proposal: the first symptom
    from whole-trajectory symptom detection is the mistake."""
    if succeeded:
        return json.dumps({"is_succeed": True}, ensure_ascii=False)
    sigs = [s for s in detect_signatures(lines) if s.fault != "unknown"]
    if not sigs:
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        last = llm_calls[-1] if llm_calls else (lines[-1] if lines else None)
        if last is None:
            return json.dumps({"is_succeed": True}, ensure_ascii=False)
        return json.dumps({
            "is_succeed": False, "mistake_agent": last.agent,
            "mistake_step_index": last.idx,
            "mistake_reason": "no explicit symptom observed; conservatively take the last decision step",
        }, ensure_ascii=False)
    s = sigs[0]
    return json.dumps({
        "is_succeed": False, "mistake_agent": s.agent,
        "mistake_step_index": s.step, "mistake_reason": s.reason,
    }, ensure_ascii=False)


def _dover_intervene_json(messages: list[ChatMessage], lines: list[Line]) -> str:
    """Pseudo-implementation of intervention recommendation: the mistake
    step's symptom → a minimal message edit (the fix guidance text); the
    category is mapped from the responsible agent (planner→orchestrator_*,
    others→subagent_*)."""
    user = str(messages[-1].get("content", ""))
    m_step = re.search(r"step=(\d+)", user)
    m_agent = re.search(r"agent=([\w\-]+)", user)
    step = int(m_step.group(1)) if m_step else -1
    agent = m_agent.group(1) if m_agent else "unknown"
    sigs = [s for s in detect_signatures(lines) if s.fault != "unknown"]
    hit = next((s for s in sigs if s.step == step), None)
    category = (
        "orchestrator_instruction" if agent == "planner"
        else "subagent_instruction"
    )
    if hit is not None:
        return json.dumps({
            "category": category,
            "replacement_text": hit.fix,
            "rationale": hit.reason[:100],
        }, ensure_ascii=False)
    return json.dumps({
        "category": category,
        "replacement_text": f"Re-examine the reasoning behind step {step}, then continue per the task requirements.",
        "rationale": "no explicit symptom observed; minimal re-examination intervention",
    }, ensure_ascii=False)


def _dover_milestone_json(messages: list[ChatMessage]) -> str:
    """Pseudo-implementation of milestone extraction: a three-stage form when
    the task text carries a citation requirement (search→read→a correct
    cited answer), K=3 [adaptation: no manual solution steps, so milestones
    are generated from the task spec]."""
    user = str(messages[-1].get("content", ""))
    ms = [
        "documents relevant to the task retrieved",
        "documents supporting the answer read",
    ]
    if re.search(r"must (end with|cite)", user, re.I):
        ms.append("the final answer contains correct content and cites the number of a document actually read")
    return json.dumps({"milestones": ms}, ensure_ascii=False)


def _dover_classify_json(messages: list[ChatMessage]) -> str:
    """Pseudo-implementation of outcome classification: deterministic rules --
    at least 2/3 successes = Validated; fault removed but not majority
    successful = Partially; faithfully executed (intervention_applied) yet
    no progress = Refuted; otherwise Inconclusive (DoVer Section 4.2
    decision rules)."""
    user = str(messages[-1].get("content", ""))
    m_runs = re.search(r"replay outcomes:\s*\[([^\]]*)\]", user)
    outcomes = (
        [x.strip() == "True" for x in m_runs.group(1).split(",") if x.strip()]
        if m_runs else []
    )
    removed = "fault removed by edit: True" in user
    if outcomes:
        n_ok = sum(outcomes)
        majority = n_ok * 2 >= len(outcomes) and n_ok >= 2
        if majority:
            label, reason = "Validated", "at least 2/3 replays succeeded"
        elif removed:
            label, reason = "Partially", "the edit removed the fault but majority success was not reached"
        else:
            label, reason = "Refuted", "the intervention was faithfully executed but did not change the failure course"
    elif removed:
        label, reason = "Partially", "the edit removed the fault (no replay outcome record)"
    else:
        label, reason = "Inconclusive", "insufficient evidence to classify"
    return json.dumps({"label": label, "reason": reason}, ensure_ascii=False)


def pseudo_judge_handler(tag: str, messages: list[ChatMessage]) -> "str | None":
    if tag == "feedback_match":
        # Environment-side call with no trajectory block -- must be handled
        # before find_trace_block. A deterministic simulation of fault spec
        # vs free-text feedback: a fault-type word hit means yes; otherwise
        # semantic understanding is simulated via content-word overlap
        # between the fault spec and the feedback.
        content = str(messages[0].get("content", "")) if messages else ""
        m_kind = re.search(r"fault type:\s*([\w_]+)", content)
        m_desc = re.search(r"description:\s*(.*)", content)
        m_fb = re.search(r"[Cc]orrection feedback[^\n]*:\n(.*?)(?:\n\n|$)", content, re.S)
        if not (m_kind and m_fb):
            return "no"
        kind = m_kind.group(1)
        fb = m_fb.group(1).lower()
        if kind in fb or kind.replace("_", " ") in fb:
            return "yes"
        desc = m_desc.group(1) if m_desc else ""

        def _words(text: str) -> set[str]:
            return set(re.findall(r"[a-z]{3,}", text.lower()))

        return "yes" if len(_words(desc) & _words(fb)) >= 2 else "no"

    # The tree-level localization prompt contains only tree.md (no
    # trajectory block); handled before find_trace_block
    if tag == "tree_diagnosis_stage":
        return _tree_stage_json(messages)

    # The dover milestone/classify prompts contain only the task and outcome
    # summaries (no trajectory block); likewise handled up front
    if tag == "dover_milestone":
        return _dover_milestone_json(messages)
    if tag == "dover_classify":
        return _dover_classify_json(messages)

    block = find_trace_block(messages)
    if block is None:
        return None
    lines = _parse_block(block)
    succeeded = find_outcome(messages)
    if succeeded is None:
        # The judge view carries no outcome line (J.1 protocol): infer from
        # the verifier event lines inside the trajectory; with no verifier
        # line, treat as failed (better to over-report problems)
        inferred = _verifier_passed(lines)
        succeeded = bool(inferred) if inferred is not None else False
    sigs = [] if succeeded else detect_signatures(lines)

    if tag == "judge_eval":
        if succeeded:
            return json.dumps(
                {"score": 9.0, "summary": "task succeeded; no anomalous symptoms observed", "findings": []},
                ensure_ascii=False,
            )
        findings = [
            {"severity": "critical", "description": s.reason, "step": s.step}
            for s in sigs
        ]
        return json.dumps(
            {
                "score": 2.5 if sigs else 4.0,
                "summary": "task failed: " + ("; ".join(s.detail or s.reason for s in sigs[:2]) or "no explicit symptom observed"),
                "findings": findings,
            },
            ensure_ascii=False,
        )

    if tag == "mast_judge":
        system = str(messages[0].get("content", "")) if messages else ""
        novel_allowed = 'code="novel"' in system
        known = [s for s in sigs if s.fault != "unknown"]
        if known or not novel_allowed:
            labels = [
                {"code": s.code, "reason": s.reason, "step": s.step} for s in sigs
            ]
        else:
            # Novel channel (allow_novel): no known symptom rule hit -- the
            # novel-mode candidate is summarized via the "recurring symptom
            # phrase" (AgentDebugX Sec. 3.4 judge behavior)
            sym_step, sym_agent, sym_text = _novel_symptom(lines)
            code = _match_extra_mode(system, sym_text)
            labels = [
                {
                    "code": code,
                    "reason": (
                        f"step {sym_step}: the symptom belongs to no known failure mode: {sym_text[:80]}"
                        if code == "novel"
                        else f"step {sym_step}: hit accepted extended mode {code}: {sym_text[:80]}"
                    ),
                    "step": sym_step,
                    "symptom": sym_text,
                }
            ]
        return json.dumps({"labels": labels}, ensure_ascii=False)

    if tag == "all_at_once":
        if not sigs:
            return None
        s = sigs[0]
        return json.dumps(
            {
                "responsible_agent": s.agent,
                "step": s.step,
                "reason": s.reason,
                "fix_suggestion": s.fix,
                "confidence": 0.7 if s.fault != "unknown" else 0.3,
                "failure_mode": s.code,
            },
            ensure_ascii=False,
        )

    if tag == "binary_search":
        # In-segment symptom detection: any symptom → the error is in the
        # shown lower half
        seg_sigs = [] if succeeded else _segment_local_signatures(lines)
        return "lower half" if seg_sigs else "upper half"

    if tag == "binary_search_refine":
        # Parse the step/agent pinned by binary search from the system
        # prompt (localization is settled; the reflection does not guess
        # twice)
        system = str(messages[0].get("content", "")) if messages else ""
        m_step = re.search(r"step (\d+)", system)
        m_agent = re.search(r"agent ([A-Za-z0-9_\-]+)", system)
        step = int(m_step.group(1)) if m_step else (sigs[0].step if sigs else 0)
        agent = m_agent.group(1) if m_agent else (sigs[0].agent if sigs else "unknown")
        hit = next((s for s in sigs if s.step == step), None)
        if hit is None and sigs:
            hit = sigs[0]
        if hit is not None:
            return json.dumps(
                {
                    "reason": hit.reason,
                    "fix_suggestion": hit.fix,
                    "confidence": 0.65,
                    "failure_mode": hit.code,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "reason": f"the behavior at step {step} (agent {agent}) constitutes the "
                "earliest decisive error: correcting that step can flip the failure.",
                "fix_suggestion": f"Re-examine and correct the reasoning behind step {step} (agent {agent}).",
                "confidence": 0.4,
                "failure_mode": None,
            },
            ensure_ascii=False,
        )

    if tag == "feedback_reflection":
        s = sigs[0] if sigs else None
        if s is not None:
            feedback = f"{s.reason}. Next round, please avoid: {s.fix}"
        else:
            feedback = "no explicit symptom observed: next round, please verify the task requirements step by step before submitting."
        return json.dumps({"feedback": feedback}, ensure_ascii=False)

    # ---- Stage 4B: deterministic branches for the new tags (trajectory
    # block/task header already parsed) ----
    if tag == "claim_ledger":
        return _ledger_json(lines, _task_text(messages))

    if tag == "claim_audit_support":
        return _support_json(messages, lines)

    if tag == "claim_audit_trace":
        return _trace_json(messages)

    if tag == "tree_diagnosis_drill":
        seg_sigs = [] if succeeded else _segment_local_signatures(lines)
        if seg_sigs:
            s = seg_sigs[0]
            return json.dumps({
                "responsible_agent": s.agent, "step": s.step,
                "reason": s.reason, "fix_suggestion": s.fix,
                "confidence": 0.65, "failure_mode": s.code,
            }, ensure_ascii=False)
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        last = llm_calls[-1] if llm_calls else (lines[-1] if lines else None)
        if last is None:
            return None
        return json.dumps({
            "responsible_agent": last.agent, "step": last.idx,
            "reason": f"step {last.idx}: no explicit symptom observed within the range; conservatively attributed to the last decision",
            "fix_suggestion": f"Re-examine the reasoning behind step {last.idx}.",
            "confidence": 0.4, "failure_mode": None,
        }, ensure_ascii=False)

    if tag == "chief_oracle":
        return _chief_oracle_json(messages)

    if tag == "chief_eval":
        return _chief_eval_json(messages, lines)

    if tag == "chief_localize":
        return _chief_localize_json(lines)

    # ---- Round 4C: L3 counterfactual replay ----
    if tag == "counterfactual_replay_oracle":
        return _cf_oracle_json(messages, lines)

    if tag == "dover_segment":
        return _dover_segment_json(lines)

    if tag == "dover_proposer":
        return _dover_proposer_json(lines, bool(succeeded))

    if tag == "dover_intervene":
        return _dover_intervene_json(messages, lines)

    return None
