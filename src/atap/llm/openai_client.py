"""OpenAI 兼容客户端 —— 真实 LLM 判官/归因调用。

密钥与 base_url 一律走环境变量（本地工程范式：不落盘、不硬编码）：
* ``base_url_env``（默认 ``OPENAI_BASE_URL``）
* ``api_key_env``（默认 ``OPENAI_API_KEY``）
* ``model`` 必须在配置中显式给出

健壮性：对 429 限流 / 5xx / 网络超时做指数退避重试（OpenRouter 等聚合
网关的共享池限流是常态），重试耗尽才显式抛错；调用间可配最小间隔节流。

结构化输出策略【推断】：为兼容任意 OpenAI 兼容后端（含 GLM 等），
不走 ``chat.completions.parse`` 的 beta 路径，而是把 JSON Schema 嵌入
prompt 约束 + 要求纯 JSON 回复，客户端解析失败时带错误重试一次。
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atap.llm.base import ChatMessage

from atap.llm.base import LLMClient, LLMError, LLMResult, parse_structured


class OpenAICompatibleLLMClient:
    """实现 LLMClient 协议；按需在 extra ``pip install atap[llm]`` 后使用。"""

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
            raise LLMError(f"环境变量 {api_key_env} 未设置：真实 LLM 运行需要密钥")
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise LLMError("缺少 openai 依赖：pip install 'atap[llm]'") from e
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

    # ------------------------------------------------------------------

    def _schema_instruction(self, schema: type) -> str:
        return (
            "输出必须是且仅是一个 JSON 对象，符合如下 JSON Schema（不要输出"
            "任何其它文本或 markdown 围栏）：\n"
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
        """带退避重试的 chat.completions.create（重试耗尽抛 LLMError）。"""
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self._last_call_ts = time.time()
            try:
                return self._client.chat.completions.create(
                    model=use_model,
                    messages=payload_messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_completion_tokens,
                )
            except Exception as e:
                last_err = e
                if not self._retryable(e) or attempt == self.max_retries:
                    raise LLMError(f"OpenAI 调用失败（tag={tag}）：{e}") from e
                delay = self.retry_base_delay * (2**attempt)
                self.retries.append(
                    {"tag": tag, "attempt": attempt + 1, "wait": delay,
                     "error": f"{type(e).__name__}: {str(e)[:120]}"}
                )
                time.sleep(delay)
        raise LLMError(f"OpenAI 调用失败（tag={tag}）：{last_err}")  # 不可达，防御

    # ------------------------------------------------------------------

    def complete(
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
            payload_messages.append(
                {"role": "system", "content": self._schema_instruction(schema)}
            )
        self.calls.append({"tag": tag, "n_messages": len(payload_messages)})
        resp = self._create(payload_messages, use_model, tag)
        text = resp.choices[0].message.content or ""

        if schema is None:
            return LLMResult(text=text)
        last_err: Exception | None = None
        retry_messages = list(payload_messages)
        for _ in range(3):  # 解析失败带错误重试（共 2 次重试机会）
            try:
                return LLMResult(text=text, parsed=parse_structured(text, schema))
            except LLMError as e:
                last_err = e
                retry_messages.append({"role": "assistant", "content": text[:2000]})
                retry_messages.append(
                    {"role": "user",
                     "content": (
                         f"你上一条回复无法解析为 JSON：{e}\n"
                         "忽略之前的全部输出，现在重新回答：只输出一个符合上述 "
                         "Schema 的 JSON 对象——以 { 开头、以 } 结尾，不要任何"
                         "解释、推理过程或 markdown 围栏。"
                     )}
                )
                resp = self._create(retry_messages, use_model, tag)
                text = resp.choices[0].message.content or ""
        raise LLMError(f"结构化解析重试仍失败（tag={tag}）：{last_err}")
