"""llm -- LLM client interface layer (protocol + Fake + OpenAI-compatible implementations)."""

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
    """Assemble an LLM client from the config (runtime assembly entry; the openai implementation is imported lazily)."""
    if spec is None:
        return None
    kind = spec.get("type")
    if kind == "fake":
        return FakeLLMClient()
    if kind == "openai":
        from atap.llm.openai_client import OpenAICompatibleLLMClient

        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return OpenAICompatibleLLMClient(**kwargs)
    raise ValueError(f"unknown llm type: {kind!r} (available: fake / openai)")
