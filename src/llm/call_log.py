"""LLM call auditing -- complete()-level prompt/response persistence (JSONL append).

Recorded fields: ts / client / tag / model / schema / latency_ms / messages
(full prompt) / response (truncated to 20k) / ok / error / usage (real-model
token usage, when available). Mounting: ``client.attach_call_log(path)``;
runtime.run_config automatically attaches ``<run_dir>/llm_calls.jsonl`` to
ctx.llm (compare's counting wrapper passes it through to the inner client).

Note: this file is a **local audit artifact** (peer of artifacts, never fed
back to the judge) -- the leak-prevention iron rule constrains the prompt
content itself (judge inputs must not contain fault names/ground-truth
fields), not the audit records; the response text carries fix suggestions
(including fault names), consistent with fix_suggestion in artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Truncation cap for a persisted response (auditing; keeps oversized
#: replies from bloating the file)
RESPONSE_CAP = 20_000


class CallLogMixin:
    """LLM client call-audit mixin (shared by the Fake / OpenAI clients)."""

    _call_log_path: Path | None = None

    def attach_call_log(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._call_log_path = p

    def _emit_call_record(self, record: dict[str, Any]) -> None:
        if self._call_log_path is None:
            return
        record.setdefault(
            "ts", datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        )
        with self._call_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
