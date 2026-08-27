"""OpenAI-compatible client -- real LLM judge/attribution calls.

Credentials and base_url always come from environment variables (local
engineering paradigm: nothing persisted, nothing hardcoded):
* ``base_url_env`` (default ``OPENAI_BASE_URL``)
* ``api_key_env`` (default ``OPENAI_API_KEY``)
* ``model`` must be given explicitly in the configuration

Robustness: exponential-backoff retries on 429 rate limiting / 5xx /
network timeouts (shared-pool rate limiting on aggregating gateways such as
OpenRouter is the norm); raise an explicit error only after retries are
exhausted; a minimum inter-call interval can be configured for throttling.

Structured-output strategy [inferred]: to stay compatible with arbitrary
OpenAI-compatible backends (including GLM), avoid the beta path of
``chat.completions.parse``; instead embed the JSON Schema into the first
system message as a constraint + require a pure JSON reply, and retry
twice with the parse error appended when client-side parsing fails (the
reply at hand is always parsed before any new request is pulled).
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atap.llm.base import ChatMessage

from atap.llm.base import LLMClient, LLMError, LLMResult, parse_structured
from atap.llm.call_log import RESPONSE_CAP, CallLogMixin


class OpenAICompatibleLLMClient(CallLogMixin):
    """Implements the LLMClient protocol; use as needed after the optional extra ``pip install atap[llm]``.

    After ``attach_call_log(path)`` every complete appends an audit record
    (prompt/response/latency/usage tokens, retry count, plus error on
    failure); runtime.run_config mounts ``<run_dir>/llm_calls.jsonl``
    automatically.
    """

    def __init__(
        self,
        model: str,
        base_url_env: str = "OPENAI_BASE_URL",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_completion_tokens: int = 4096,
        max_retries: int = 5,
        retry_base_delay: float = 4.0,
        request_interval: float = 1.5,
    ) -> None:
        base_url = os.environ.get(base_url_env, "")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise LLMError(f"environment variable {api_key_env} is not set: a real LLM run requires an API key")
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise LLMError("missing openai dependency: pip install 'atap[llm]'") from e
        self._openai = openai
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_interval = request_interval
        self._last_call_ts = 0.0
        self.calls: list[dict] = []
        self.retries: list[dict] = []
        # structured-output parse-repair retries (peer of ``retries``, which
        # records transport-level backoff; both feed the audit's ``retries``
        # count)
        self.parse_retries: list[dict] = []
        self.http_requests = 0   # HTTP requests actually issued (including rate-limit/parse-repair retries)

    # ------------------------------------------------------------------

    def _schema_instruction(self, schema: type) -> str:
        return (
            "The output must be one and only one JSON object conforming to the "
            "following JSON Schema (do not output any other text or markdown "
            "fences):\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )

    def _retryable(self, e: Exception) -> bool:
        openai = self._openai
        if isinstance(e, openai.RateLimitError):
            return True
        if isinstance(e, openai.APIStatusError) and getattr(e, "status_code", 0) >= 500:
            return True
        if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
            return True
        return False

    def _throttle(self) -> None:
        if self.request_interval <= 0:
            return
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

    def _create(self, payload_messages: list, use_model: str, tag: str):
        """chat.completions.create with backoff retries (raises LLMError once retries are exhausted).

        OpenRouter occasionally returns HTTP 200 with ``choices=null``
        (empty provider response / error embedded in the error field) --
        treated as a retryable error and retried with backoff like 429/5xx.
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self._last_call_ts = time.time()
            self.http_requests += 1
            try:
                resp = self._client.chat.completions.create(
                    model=use_model,
                    messages=payload_messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_completion_tokens,
                )
            except Exception as e:
                last_err = e
                if not self._retryable(e) or attempt == self.max_retries:
                    raise LLMError(f"OpenAI call failed (tag={tag}): {e}") from e
                delay = self.retry_base_delay * (2**attempt)
                self.retries.append(
                    {"tag": tag, "attempt": attempt + 1, "wait": delay,
                     "error": f"{type(e).__name__}: {str(e)[:120]}"}
                )
                time.sleep(delay)
                continue
            if getattr(resp, "choices", None):
                return resp
            detail = self._resp_error_detail(resp)
            if attempt == self.max_retries:
                raise LLMError(f"OpenAI returned empty choices (tag={tag}): {detail}")
            delay = self.retry_base_delay * (2**attempt)
            self.retries.append(
                {"tag": tag, "attempt": attempt + 1, "wait": delay,
                 "error": f"empty choices: {detail}"}
            )
            time.sleep(delay)
        raise LLMError(f"OpenAI call failed (tag={tag}): {last_err}")  # unreachable, defensive

    @staticmethod
    def _resp_error_detail(resp) -> str:
        err = getattr(resp, "error", None)
        if err is None:
            return "choices empty and the response carries no error field"
        if isinstance(err, dict):
            return str(err.get("message") or err)[:200]
        return str(getattr(err, "message", err))[:200]

    # ------------------------------------------------------------------

    @staticmethod
    def _usage_dict(resp) -> dict[str, int] | None:
        u = getattr(resp, "usage", None)
        if u is None:
            return None
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }

    @staticmethod
    def _merge_usage(a: dict[str, int] | None,
                     b: dict[str, int] | None) -> dict[str, int] | None:
        """Sum two per-response usage dicts (per key; a missing side keeps
        the present side, both-missing stays None) -- a logical call that
        needed parse-repair retries paid for every reply, so its recorded
        usage must be the sum, not the first response's numbers."""
        if a is None:
            return b
        if b is None:
            return a
        out: dict[str, int] = {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            x, y = a.get(k), b.get(k)
            if x is None and y is None:
                out[k] = None
            elif x is None:
                out[k] = y
            elif y is None:
                out[k] = x
            else:
                out[k] = x + y
        return out

    def complete(
        self,
        messages: list["ChatMessage"],
        *,
        schema: type | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> LLMResult:
        t0 = time.perf_counter()
        n_http0 = self.http_requests
        n_retry0 = len(self.retries)
        n_parse0 = len(self.parse_retries)
        base = {
            "client": "openai",
            "tag": tag,
            "model": model or self.model,
            "schema": schema.__name__ if schema else None,
            "messages": [dict(m) for m in messages],
        }
        # http_requests = HTTP requests actually issued by this complete
        # (including rate-limit retries and parse-repair retries) -- the
        # upper bound for quota accounting; only successful requests
        # actually consume the free-tier quota. retries = retry events of
        # this complete (transport backoff + structured parse repairs).
        try:
            result = self._complete(messages, schema=schema, model=model, tag=tag)
        except Exception as e:   # non-LLMError exceptions also go into the
            # audit log (e.g. TypeError-like defects from empty choices) --
            # audit completeness takes priority over exception taxonomy;
            # afterwards re-raise unchanged
            self._emit_call_record({
                **base, "ok": False,
                "http_requests": self.http_requests - n_http0,
                "retries": (len(self.retries) - n_retry0)
                           + (len(self.parse_retries) - n_parse0),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "error": f"{type(e).__name__}: {e}"[:500],
            })
            raise
        self._emit_call_record({
            **base, "ok": True,
            "http_requests": self.http_requests - n_http0,
            "retries": (len(self.retries) - n_retry0)
                       + (len(self.parse_retries) - n_parse0),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            "response": result.text[:RESPONSE_CAP],
            "usage": result.usage,
        })
        return result

    def _complete(
        self,
        messages: list["ChatMessage"],
        *,
        schema: type | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> LLMResult:
        use_model = model or self.model
        payload_messages = [dict(m) for m in messages]
        if schema is not None:
            # Merge the schema constraint into the FIRST system message
            # instead of appending a second trailing system message: the
            # message order stays [system, user] and backends that only
            # honor leading system turns still see the constraint (with no
            # leading system message, one is prepended).
            instruction = self._schema_instruction(schema)
            if payload_messages and payload_messages[0].get("role") == "system":
                payload_messages[0] = {
                    **payload_messages[0],
                    "content": (
                        f"{payload_messages[0].get('content', '')}\n\n"
                        f"{instruction}"
                    ),
                }
            else:
                payload_messages.insert(
                    0, {"role": "system", "content": instruction}
                )
        self.calls.append({"tag": tag, "n_messages": len(payload_messages)})
        resp = self._create(payload_messages, use_model, tag)
        text = resp.choices[0].message.content or ""
        usage = self._usage_dict(resp)

        if schema is None:
            return LLMResult(text=text, usage=usage)
        last_err: Exception | None = None
        retry_messages = list(payload_messages)
        # Parse the already-fetched reply FIRST; only pull a new reply when
        # parsing failed and retry budget remains (2 retries in total). The
        # final failure therefore costs exactly 1 + 2 = 3 HTTP calls -- the
        # last fetched reply is never left unparsed (previously the 3rd
        # failure pulled a 4th reply that was discarded unexamined).
        for attempt in range(3):
            try:
                return LLMResult(
                    text=text, parsed=parse_structured(text, schema), usage=usage
                )
            except LLMError as e:
                last_err = e
                if attempt == 2:
                    break   # no retry budget left: fail without paying for an unused reply
                self.parse_retries.append(
                    {"tag": tag, "attempt": attempt + 1,
                     "error": str(e)[:160]}
                )
                retry_messages.append({"role": "assistant", "content": text[:2000]})
                retry_messages.append(
                    {"role": "user",
                     "content": (
                         f"Your previous reply could not be parsed as JSON: {e}\n"
                         "Ignore all previous output and answer again now: output "
                         "only one JSON object conforming to the Schema above -- "
                         "starting with { and ending with }, without any "
                         "explanation, reasoning process, or markdown fences."
                     )}
                )
                resp = self._create(retry_messages, use_model, tag)
                text = resp.choices[0].message.content or ""
                # token accounting must cover every HTTP round-trip of this
                # logical call (the initial reply + each parse-repair reply),
                # not just the first response (previously the audit recorded
                # the first reply's usage, undercounting by up to 2/3)
                usage = self._merge_usage(usage, self._usage_dict(resp))
        raise LLMError(f"structured parsing still failing after retries (tag={tag}): {last_err}")
