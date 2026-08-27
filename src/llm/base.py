"""LLMClient protocol -- the only external-effect outlet for judge/attribution-style algorithms.

The design follows the local engineering paradigm (k2-agentic / kimi-k3):
algorithms depend on the protocol rather than a concrete client; the
OpenAI-compatible implementation reads its credentials from environment
variables; the Fake implementation provides an offline deterministic judge
for tests and demos.
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
    usage: dict[str, int] | None = None  # prompt/completion/total tokens (when available)


@runtime_checkable
class LLMClient(Protocol):
    """Chat-completion protocol; when a schema is given, the client is responsible for structured parsing."""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        tag: str = "",
    ) -> LLMResult: ...


class LLMError(Exception):
    """LLM call/parse failure (no silent degradation -- local engineering paradigm: explicit exceptions)."""


def _balanced_blocks(text: str) -> list[str]:
    """Scan all balanced curly-brace blocks (skipping braces inside JSON string literals)."""
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
    """Return candidate JSON texts in priority order (tolerates fences, <think> blocks, and surrounding noise).

    Reasoning-style models (e.g. nemotron) often wrap the answer with
    thinking text, and the thinking may contain curly-brace fragments (even
    single-quoted Python-style dicts) -- a single "first block" strategy
    would pick the wrong one, so all candidates are returned here and
    parse_structured tries them one by one.
    """
    candidates: list[str] = []
    fenced = "```"
    if fenced in text:
        for i, part in enumerate(text.split(fenced)):
            if i % 2 == 1:  # inside a fence
                body = part
                if body.lstrip().startswith("json"):
                    body = body.lstrip()[4:]
                candidates.extend(_balanced_blocks(body))
    candidates.extend(_balanced_blocks(text))
    # deduplicate, preserving order
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def extract_json_block(text: str) -> str:
    """Extract the first JSON object text from a model reply (tolerates markdown fences and surrounding noise)."""
    candidates = json_block_candidates(text)
    if candidates:
        return candidates[0]
    if "{" in text:
        raise LLMError(f"unclosed JSON object: {text[:200]!r}")
    raise LLMError(f"no JSON object found in reply: {text[:200]!r}")


def parse_structured(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse a reply text into a pydantic model (raises LLMError on failure).

    Each candidate is tried with JSON parsing (skipping pseudo-blocks inside
    thinking text); when all fail, fall back to ``ast.literal_eval`` (catches
    single-quoted Python-style dicts -- some models keep their quoting
    habits). Schema validation takes the first candidate that passes
    ``model_validate``.
    """
    import ast
    import json

    candidates = json_block_candidates(text)
    if not candidates:
        raise LLMError(f"no JSON object found in reply: {text[:200]!r}")
    last_json_err: Exception | None = None
    parsed_any: list[dict] = []
    for block in candidates:
        try:
            parsed_any.append(json.loads(block))
        except json.JSONDecodeError as e:
            last_json_err = e
            try:
                parsed_any.append(ast.literal_eval(block))  # single-quoted dict fallback
            except (ValueError, SyntaxError):
                pass
    if not parsed_any:
        raise LLMError(
            f"JSON parsing failed: {last_json_err}; source text: {text[:200]!r}"
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
        f"{schema.__name__} validation failed: {last_val_err}; {len(parsed_any)} candidate(s)"
    ) from last_val_err


# Injectable handler for the Fake client: returns scripted results by tag (algorithm marker).
Handler = Callable[[str, list[ChatMessage]], "str | BaseModel | None"]
