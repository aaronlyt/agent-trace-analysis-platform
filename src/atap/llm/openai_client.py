"""OpenAI 兼容客户端 —— 真实 LLM 判官/归因调用。

密钥与 base_url 一律走环境变量（本地工程范式：不落盘、不硬编码）：
* ``base_url_env``（默认 ``OPENAI_BASE_URL``）
* ``api_key_env``（默认 ``OPENAI_API_KEY``）
* ``model`` 必须在配置中显式给出

结构化输出策略【推断】：为兼容任意 OpenAI 兼容后端（含 GLM 等），
不走 ``chat.completions.parse`` 的 beta 路径，而是把 JSON Schema 嵌入
prompt 约束 + 要求纯 JSON 回复，客户端解析失败时带错误重试一次。
"""

from __future__ import annotations

import json
import os
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
    ) -> None:
        base_url = os.environ.get(base_url_env, "")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise LLMError(f"环境变量 {api_key_env} 未设置：真实 LLM 运行需要密钥")
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMError("缺少 openai 依赖：pip install 'atap[llm]'") from e
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.calls: list[dict] = []

    def _schema_instruction(self, schema: type) -> str:
        return (
            "输出必须是且仅是一个 JSON 对象，符合如下 JSON Schema（不要输出"
            "任何其它文本或 markdown 围栏）：\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )

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
        last_err: Exception | None = None
        for attempt in range(2):  # 解析失败带错误重试一次
            try:
                resp = self._client.chat.completions.create(
                    model=use_model,
                    messages=payload_messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_completion_tokens,
                )
            except Exception as e:  # 网络/配额等：显式上抛
                raise LLMError(f"OpenAI 调用失败（tag={tag}）：{e}") from e
            text = resp.choices[0].message.content or ""
            if schema is None:
                return LLMResult(text=text)
            try:
                return LLMResult(text=text, parsed=parse_structured(text, schema))
            except LLMError as e:
                last_err = e
                payload_messages.append({"role": "assistant", "content": text})
                payload_messages.append(
                    {"role": "user", "content": f"你的回复无法解析：{e}\n请重新只输出符合 Schema 的 JSON 对象。"}
                )
        raise LLMError(f"结构化解析重试仍失败（tag={tag}）：{last_err}")
