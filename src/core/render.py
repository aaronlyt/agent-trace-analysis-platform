"""Trajectory rendering -- renders the R0 event stream into canonical text
lines readable by judges.

The three LLM-based algorithms (judge_eval / mast_judge / all_at_once) all use
this renderer to build the trajectory view inside their prompts, guaranteeing
"one view across judges"; FakeLLM's deterministic pseudo-judge
(llm/pseudo_judge.py) also parses this format, so any format change must be
synced in both places.

Line format (index aligns with event.index; the judge's output step references
it directly)::

    [7] TOOL_CALL searcher search {'query': 'x'}
    [8] TOOL_RESULT env search :: search results for 'x': 2 docs [d1, d3] ...

Folded view: ``fold`` replaces specified events' content with placeholders
(SSF artifact); ``table`` can expand placeholders on demand (debugging/audit).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atap.core.schema import Trajectory

MAX_CONTENT_CHARS = 400

TRACE_BEGIN = "=== TRACE BEGIN ==="
TRACE_END = "=== TRACE END ==="

# Failure-indicating vocabulary (the K of TrajAudit 2605.26563 Algorithm 1:
# LLM-generated + manually refined). Shared by SSF folding and the
# pseudo-judge -- the "symptom words" visible to the judge must match the
# folding keep-words, otherwise folding would hide evidence the judge needs.
FAILURE_KEYWORDS: tuple[str, ...] = (
    "error", "exception", "traceback", "invalid", "denied", "failed", "fail",
    "missing", "exhausted", "timeout",
)

# Word-boundary matching: the corpus text itself discusses failure/error, so
# plain substring matching would produce widespread false hits.
_FAILURE_KW_RE = re.compile(
    r"\b(" + "|".join(FAILURE_KEYWORDS) + r")\b", re.IGNORECASE
)


def matches_failure_keyword(text: str) -> bool:
    return bool(_FAILURE_KW_RE.search(text))


# Structured error-observation test: error messages returned by tools start
# with prefixes like error:/exception:/traceback. The toy corpus itself is
# "prose about error analysis", so a lexical dictionary would treat body text
# as errors -- separating error observations from domain prose must rely on
# structural prefixes (shared by the SSF keep rule and the pseudo-judge).
_ERROR_OBS_RE = re.compile(r"^\s*(error|exception|traceback|fatal|failed)\b[:\s]?", re.I)


def is_error_observation(text: str) -> bool:
    return bool(_ERROR_OBS_RE.match(text))

# SSF placeholder: ⟦folded:F3 | digest...⟧ Deliberately **unanchored**: a
# rendered line carries the placeholder after its "[n] KIND agent action :: "
# head, so in-line consumers (ssf.unfold_line) must locate it with search();
# consumers that need "the whole content is a placeholder" semantics
# (render_event_line's table expansion) must use fullmatch().
FOLD_PLACEHOLDER_RE = re.compile(r"⟦folded:(?P<fid>\w+) \| (?P<digest>.*)⟧")


def _short(obj: Any, limit: int = MAX_CONTENT_CHARS) -> str:
    s = obj if isinstance(obj, str) else repr(obj)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def render_event_line(
    ev: Any,
    fold: dict[str, str] | None = None,
    table: dict[str, str] | None = None,
) -> str:
    """Render a single event as one line.

    fold: {event_id: placeholder text} (SSF folded view).
    table: {fold_id: original text}; if given and this line's content is a
    placeholder, expand it back to the original text.
    """
    payload = dict(ev.payload or {})
    content = payload.pop("content", None)
    head = f"[{ev.index}] {ev.kind} {ev.agent}"
    if ev.action:
        head += f" {ev.action}"
    extra = ""
    if payload:
        extra = " " + _short(payload)
    if content is None:
        return head + extra

    text = str(content)
    if fold and ev.id in fold:
        text = fold[ev.id]
    if table:
        m = FOLD_PLACEHOLDER_RE.fullmatch(text.strip())
        if m and m.group("fid") in table:
            text = table[m.group("fid")]
    return f"{head}{extra} :: {_short(text)}"


def render_trace(
    trajectory: "Trajectory",
    *,
    fold: dict[str, str] | None = None,
    table: dict[str, str] | None = None,
    include_task: bool = True,
    include_outcome: bool = True,
) -> str:
    """Render a whole trajectory (task header + event lines).

    With include_outcome=False the task header omits the ``outcome:`` line --
    the MAST J.1 judge protocol does not show the success/failure result to
    the judge (default for judge_eval).
    """
    lines: list[str] = []
    if include_task:
        lines.append(f"task: {trajectory.task}")
        if include_outcome:
            out = "SUCCESS" if trajectory.outcome.success else "FAILURE"
            lines.append(
                f"outcome: {out}"
                + (f" ({trajectory.outcome.note})" if trajectory.outcome.note else "")
            )
        lines.append(TRACE_BEGIN)
    for ev in trajectory.events:
        lines.append(render_event_line(ev, fold, table))
    if include_task:
        lines.append(TRACE_END)
    return "\n".join(lines)


def judge_view(bundle, *, include_outcome: bool = True) -> str:
    """Judge view: use the SSF folded view if the folding artifact exists,
    otherwise the full view.

    The artifact key ``represent/ssf`` is the data contract between algorithms
    (downstream consumes by name, without importing the SSF module); when SSF
    is not configured, degrade explicitly to full rendering. include_outcome
    is passed through to render_trace (under the J.1 protocol the judge does
    not see the success/failure result).
    """
    ssf = bundle.get("represent", "ssf")
    if isinstance(ssf, dict) and ssf.get("fold"):
        return render_trace(
            bundle.trajectory, fold=ssf["fold"], include_outcome=include_outcome
        )
    return render_trace(bundle.trajectory, include_outcome=include_outcome)
