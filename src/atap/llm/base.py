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


def _balanced_blocks(text: str) -> list[str]:
    """扫描全部平衡的花括号块（跳过 JSON 字符串字面量内部的花括号）。"""
    blocks: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    blocks.append(text[start : i + 1])
    return blocks


def json_block_candidates(text: str) -> list[str]:
    """按优先级给出候选 JSON 文本（容忍围栏、<think> 块与前后噪声）。

    推理型模型（如 nemotron）常在正文前后包裹思考文本，思考里可能包含
    花括号片段（甚至是单引号的 Python 风格 dict）——单一"第一个块"策略
    会误选，这里返回全部候选，由 parse_structured 逐个尝试。
    """
    candidates: list[str] = []
    fenced = "```"
    if fenced in text:
        for i, part in enumerate(text.split(fenced)):
            if i % 2 == 1:  # 围栏内
                body = part
                if body.lstrip().startswith("json"):
                    body = body.lstrip()[4:]
                candidates.extend(_balanced_blocks(body))
    candidates.extend(_balanced_blocks(text))
    # 去重保序
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def extract_json_block(text: str) -> str:
    """从模型回复中提取第一个 JSON 对象文本（容忍 markdown 围栏与前后噪声）。"""
    candidates = json_block_candidates(text)
    if candidates:
        return candidates[0]
    if "{" in text:
        raise LLMError(f"JSON 对象未闭合：{text[:200]!r}")
    raise LLMError(f"回复中不含 JSON 对象：{text[:200]!r}")


def parse_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    """把回复文本解析为 pydantic 模型（失败抛 LLMError）。

    逐候选尝试 JSON 解析（跳过思考文本里的伪块）；全部失败时回退
    ``ast.literal_eval``（捕获单引号 Python 风格 dict——部分模型不改
    引号习惯）。Schema 校验取第一个通过 ``model_validate`` 的候选。
    """
    import ast
    import json

    candidates = json_block_candidates(text)
    if not candidates:
        raise LLMError(f"回复中不含 JSON 对象：{text[:200]!r}")
    last_json_err: Exception | None = None
    parsed_any: list[dict] = []
    for block in candidates:
        try:
            parsed_any.append(json.loads(block))
        except json.JSONDecodeError as e:
            last_json_err = e
            try:
                parsed_any.append(ast.literal_eval(block))  # 单引号 dict 回退
            except (ValueError, SyntaxError):
                pass
    if not parsed_any:
        raise LLMError(
            f"JSON 解析失败：{last_json_err}；原文：{text[:200]!r}"
        ) from last_json_err
    last_val_err: Exception | None = None
    for data in parsed_any:
        if not isinstance(data, dict):
            continue
        try:
            return schema.model_validate(data)
        except Exception as e:  # pydantic ValidationError
            last_val_err = e
    raise LLMError(
        f"{schema.__name__} 校验失败：{last_val_err}；候选 {len(parsed_any)} 个"
    ) from last_val_err


# Fake 客户端的可注入处理器：按 tag（算法标记）返回脚本化结果。
Handler = Callable[[str, list[ChatMessage]], "str | BaseModel | None"]
