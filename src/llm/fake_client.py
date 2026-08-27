"""FakeLLMClient -- offline deterministic LLM stand-in (tests / CI / atap demo).

Two driving modes:
* ``responses``: a scripted queue popped in call order (for precise unit
  tests);
* ``handler``: a callback dispatched by ``tag`` (by default wired to the
  deterministic pseudo-judge in pseudo_judge, so the whole LLM-judge
  pipeline can run offline end to end).

Every call is recorded in ``calls`` (so tests can assert prompt content and
call counts); after ``attach_call_log(path)`` each call additionally appends
an audit record to a JSONL file (see llm/call_log.py; mounted automatically
by runtime.run_config).
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from atap.llm.base import (
    ChatMessage,
    Handler,
    LLMClient,
    LLMError,
    LLMResult,
    parse_structured,
)
from atap.llm.call_log import RESPONSE_CAP, CallLogMixin
from atap.llm.pseudo_judge import pseudo_judge_handler


class FakeLLMClient(CallLogMixin):
    """Implements the LLMClient protocol; no network, no randomness."""

    def __init__(
        self,
        responses: list["str | BaseModel"] | None = None,
        handler: Handler | None = None,
    ) -> None:
        self.responses: list["str | BaseModel"] = list(responses or [])
        self.handler: Handler = handler or pseudo_judge_handler
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> LLMResult:
        t0 = time.perf_counter()
        base = {
            "client": "fake",
            "tag": tag,
            "model": model,
            "schema": schema.__name__ if schema else None,
            "messages": [dict(m) for m in messages],
        }
        try:
            result = self._complete(messages, schema=schema, model=model, tag=tag)
        except Exception as e:
            self._emit_call_record({
                **base, "ok": False,
                "error": f"{type(e).__name__}: {e}"[:500],
            })
            raise
        self._emit_call_record({
            **base, "ok": True,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            "response": result.text[:RESPONSE_CAP],
            "usage": result.usage,
        })
        return result

    def _complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> LLMResult:
        self.calls.append(
            {"tag": tag, "messages": list(messages), "schema": schema.__name__ if schema else None}
        )
        if self.responses:
            raw = self.responses.pop(0)
        else:
            raw = self.handler(tag, messages)
            if raw is None:
                raise LLMError(f"pseudo-judge cannot handle the request with tag={tag!r} (missing handler branch)")
        if isinstance(raw, BaseModel):
            return LLMResult(text=raw.model_dump_json(), parsed=raw)
        if schema is not None:
            return LLMResult(text=raw, parsed=parse_structured(raw, schema))
        return LLMResult(text=raw)
