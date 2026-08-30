"""Who&When failure-attribution benchmark adapter (ag2ai/Agents_Failure_Attribution,
"Which Agent Causes Task Failures and When?", ICML 2025 Spotlight, arXiv:2505.00212).

Turns the public Who&When dataset into atap's native R0 trajectories so the
existing attribution ladder (all_at_once / binary_search / ...) can be scored
against an *external* gold standard with `atap compare` -- the same
`evaluate_against_gt` path the sandbox uses, no new metric code.

Dataset shape (per file, one failed multi-agent run):
* ``question`` -> Trajectory.task (the gold ``ground_truth`` is deliberately
  kept OUT of task/events -- atap's judges run on the paper's *Without-GT*
  basis, matching the number all_at_once.py cites);
* ``history`` -> one R0 event per message, in order; ``mistake_step`` is a
  **0-based index into history**, so it becomes the R0 event index directly;
* agent identity follows the reference implementation exactly:
  ``name`` for the Algorithm-Generated split, ``role`` for Hand-Crafted
  (Automated_FA/Lib/utils.py: ``index_agent = "role" if is_handcrafted else
  "name"``) -- so predicted ``event.agent`` and gold ``mistake_agent`` compare
  on the same vocabulary;
* ``mistake_agent`` + ``mistake_step`` -> ``meta["injected_fault"]``
  ``{step, agent, kind, mast_code}`` -- the exact contract compare.py reads
  (``kind`` labels the split for per-kind rollups; ``mast_code`` is None:
  Who&When has no failure-taxonomy gold, so code-level hits do not apply).

Leak discipline: gold (``ground_truth`` / ``mistake_reason`` / the fault dict)
lives only under ``meta`` -- never in ``task``, an event payload, or
``outcome.note`` -- so nothing the judge view renders (task + events + outcome)
can leak the answer or the labelled step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from atap.core.schema import (
    AGENT_MESSAGE,
    HANDOFF,
    LLM_CALL,
    TASK_START,
    TOOL_RESULT,
    Outcome,
    TraceEvent,
    Trajectory,
)

#: the two dataset splits and the message key that carries the acting agent
SPLIT_AGENT_KEY = {"Algorithm-Generated": "name", "Hand-Crafted": "role"}
_SPLIT_SHORT = {"Algorithm-Generated": "algo", "Hand-Crafted": "hand"}


def _kind_for(agent_id: str, idx: int) -> str:
    """Best-effort R0 kind (readability / SSF folding only -- attribution
    step/agent hits do not depend on it)."""
    a = agent_id.lower()
    if "terminal" in a or "computer" in a:
        return TOOL_RESULT          # code / tool execution turn
    if "->" in agent_id:
        return HANDOFF              # Magentic-One "Orchestrator (-> WebSurfer)"
    if a in ("human", "user"):
        return TASK_START if idx == 0 else AGENT_MESSAGE
    return LLM_CALL                 # an expert / assistant turn


def trajectory_from_record(
    record: dict[str, Any], split: str, *, trace_id: str
) -> Trajectory:
    """One Who&When JSON object -> an R0 Trajectory (flat event stream).

    ``split`` must be a key of :data:`SPLIT_AGENT_KEY`; it decides whether the
    acting agent is read from ``name`` or ``role``, matching the reference
    implementation's ``index_agent``.
    """
    if split not in SPLIT_AGENT_KEY:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_AGENT_KEY)}")
    agent_key = SPLIT_AGENT_KEY[split]

    history = record.get("history") or []
    events: list[TraceEvent] = []
    for idx, entry in enumerate(history):
        agent_id = str(
            entry.get(agent_key) or entry.get("name") or entry.get("role") or "unknown"
        )
        content = entry.get("content", "")
        events.append(TraceEvent(
            id=f"e{idx:03d}",
            ts=float(idx),
            kind=_kind_for(agent_id, idx),
            agent=agent_id,
            action=None,
            payload={"content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)},
            index=idx,
        ))

    # gold -> the exact shape compare.evaluate_against_gt consumes. step is the
    # 0-based history index == R0 event index; agent is on the same vocabulary
    # as event.agent; mast_code is None (no taxonomy gold in Who&When).
    injected_fault = {
        "step": int(record["mistake_step"]),
        "agent": str(record.get("mistake_agent", "")),
        "kind": f"whoswhen:{_SPLIT_SHORT[split]}",
        "mast_code": None,
    }
    # reference-only gold, kept out of everything the judge sees
    meta = {
        "injected_fault": injected_fault,
        "task_id": trace_id,
        "whoswhen": {
            "split": split,
            "question_id": record.get("question_ID"),
            "level": record.get("level"),
            "ground_truth": record.get("ground_truth"),
            "mistake_reason": record.get("mistake_reason"),
        },
    }
    return Trajectory(
        trace_id=trace_id,
        task=str(record.get("question", "")),
        events=events,
        outcome=Outcome(success=bool(record.get("is_correct", False))),
        meta=meta,
        raw=None,
    )


def _sorted_json_files(directory: Path) -> list[Path]:
    """Numeric sort by filename stem (matches the reference's file ordering)."""
    def _key(p: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return (int(digits) if digits else 0, p.name)
    return sorted((p for p in directory.glob("*.json")), key=_key)


def iter_split(root: str | Path, split: str) -> Iterator[Trajectory]:
    """Yield every trajectory of one split under the Who&When ``root`` dir
    (the directory that contains ``Algorithm-Generated`` / ``Hand-Crafted``)."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"split directory not found: {split_dir} (pass the Who&When root that "
            f"contains {list(SPLIT_AGENT_KEY)})"
        )
    for f in _sorted_json_files(split_dir):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{f}: invalid JSON ({e})") from e
        if record.get("mistake_step") is None or record.get("mistake_agent") is None:
            # a Who&When failure record without gold cannot be scored; skip loudly
            continue
        yield trajectory_from_record(
            record, split, trace_id=f"whoswhen-{_SPLIT_SHORT[split]}-{f.stem}"
        )


def load_whoswhen(root: str | Path, splits: list[str] | None = None) -> list[Trajectory]:
    """Load the whole benchmark (both splits by default) as R0 trajectories."""
    splits = splits or list(SPLIT_AGENT_KEY)
    out: list[Trajectory] = []
    for split in splits:
        out.extend(iter_split(root, split))
    return out


def write_jsonl(traces: list[Trajectory], out_path: str | Path) -> int:
    """Write trajectories as atap-native JSONL (one ``Trajectory.to_dict`` per
    line -- the format ``source: {type: jsonl}`` reads). Returns the count."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return len(traces)
