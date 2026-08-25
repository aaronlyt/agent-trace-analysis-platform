"""LLMClient 协议 —— 判官/归因类算法的唯一外部效应出口。

设计对齐本地工程范式（k2-agentic / kimi-k3）：算法依赖协议而非具体
客户端；OpenAI 兼容实现走环境变量取密钥；Fake 实现提供离线确定性
判官供测试与 demo。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel


class ChatMessage(TypedDict, total=False):
    role: str
    content: str


@dataclass
class LLMResult:
    text: str
    parsed: BaseModel | None = None
    usage: dict[str, int] | None = None  # prompt/completion/total tokens（可用时）


@runtime_checkable
class LLMClient(Protocol):
    """聊天补全协议；schema 给定时客户端负责结构化解析。"""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> LLMResult: ...


class LLMError(Exception):
    """LLM 调用/解析失败（不静默降级——本地工程范式：显式异常）。"""


def extract_json_block(text: str) -> str:
    """从模型回复中提取第一个 JSON 对象文本（容忍 markdown 围栏与前后噪声）。"""
    fenced = "```"
    if fenced in text:
        parts = text.split(fenced)
        for i in range(1, len(parts), 2):
            body = parts[i]
            if body.lstrip().startswith("json"):
                body = body.lstrip()[4:]
            if body.strip().startswith("{"):
                return body.strip()
    start = text.find("{")
    if start == -1:
        raise LLMError(f"回复中不含 JSON 对象：{text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMError(f"JSON 对象未闭合：{text[:200]!r}")


def parse_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    """把回复文本解析为 pydantic 模型（失败抛 LLMError）。"""
    import json

    block = extract_json_block(text)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON 解析失败：{e}；原文：{text[:200]!r}") from e
    try:
        return schema.model_validate(data)
    except Exception as e:  # pydantic ValidationError
        raise LLMError(f"{schema.__name__} 校验失败：{e}") from e


# Fake 客户端的可注入处理器：按 tag（算法标记）返回脚本化结果。
Handler = Callable[[str, list[ChatMessage]], "str | BaseModel | None"]
