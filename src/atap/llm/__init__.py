"""llm —— LLM 客户端接口层（协议 + Fake + OpenAI 兼容实现）。"""

from atap.llm.base import (
    ChatMessage,
    Handler,
    LLMClient,
    LLMError,
    LLMResult,
    extract_json_block,
    parse_structured,
)
from atap.llm.fake_client import FakeLLMClient

__all__ = [
    "ChatMessage",
    "Handler",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "extract_json_block",
    "parse_structured",
    "FakeLLMClient",
]


def build_llm(spec: dict | None):
    """按配置装配 LLM 客户端（runtime 组装入口；openai 实现延迟导入）。"""
    if spec is None:
        return None
    kind = spec.get("type")
    if kind == "fake":
        return FakeLLMClient()
    if kind == "openai":
        from atap.llm.openai_client import OpenAICompatibleLLMClient

        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return OpenAICompatibleLLMClient(**kwargs)
    raise ValueError(f"未知 llm type：{kind!r}（可用：fake / openai）")
