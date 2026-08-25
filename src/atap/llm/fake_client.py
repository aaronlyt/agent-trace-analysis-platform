"""FakeLLMClient —— 离线确定性 LLM 替身（测试 / CI / atap demo）。

两种驱动方式：
* ``responses``：脚本化队列，按调用顺序弹出（精确单测用）；
* ``handler``：按 ``tag`` 分发的回调（默认挂 pseudo_judge 确定性伪判官，
  使整条 LLM-judge 链路可离线端到端跑通）。

所有调用都被记录在 ``calls``，供测试断言 prompt 内容与调用次数。
"""

from __future__ import annotations

from pydantic import BaseModel

from atap.llm.base import (
    ChatMessage,
    Handler,
    LLMClient,
    LLMError,
    LLMResult,
    parse_structured,
)
from atap.llm.pseudo_judge import pseudo_judge_handler


class FakeLLMClient:
    """实现 LLMClient 协议；不联网、无随机性。"""

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
        self.calls.append(
            {"tag": tag, "messages": list(messages), "schema": schema.__name__ if schema else None}
        )
        if self.responses:
            raw = self.responses.pop(0)
        else:
            raw = self.handler(tag, messages)
            if raw is None:
                raise LLMError(f"伪判官无法处理 tag={tag!r} 的请求（缺 handler 分支）")
        if isinstance(raw, BaseModel):
            return LLMResult(text=raw.model_dump_json(), parsed=raw)
        if schema is not None:
            return LLMResult(text=raw, parsed=parse_structured(raw, schema))
        return LLMResult(text=raw)
