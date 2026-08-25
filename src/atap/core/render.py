"""轨迹渲染 —— 把 R0 事件流渲染为判官可读的规范文本行。

三个 LLM 类算法（judge_eval / mast_judge / all_at_once）统一用本渲染器
构造 prompt 中的轨迹视图，保证"同一视图跨判官"；FakeLLM 的确定性伪判官
（llm/pseudo_judge.py）也解析本格式，因此格式一旦变更两处同步。

行格式（index 对齐 event.index，判官输出的 step 直接引用它）::

    [7] TOOL_CALL searcher search {'query': 'x'}
    [8] TOOL_RESULT env search :: search results for 'x': 2 docs [d1, d3] ...

折叠视图：``fold`` 把指定事件的 content 替换为占位符（SSF 产物）；
``table`` 可把占位符按需展开（调试/审计）。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atap.core.schema import Trajectory

MAX_CONTENT_CHARS = 400

TRACE_BEGIN = "=== TRACE BEGIN ==="
TRACE_END = "=== TRACE END ==="

# 失败指示词典（TrajAudit 2605.26563 Algorithm 1 的 K：LLM 生成 + 人工精修）。
# SSF 折叠与伪判官共用——判官可见的"症状词"必须与折叠保留词一致，
# 否则折叠会藏掉判官需要的证据。
FAILURE_KEYWORDS: tuple[str, ...] = (
    "error", "exception", "traceback", "invalid", "denied", "failed", "fail",
    "missing", "exhausted", "timeout",
)

# 词边界匹配：语料文本本身讨论 failure/error，子串匹配会大面积误命中。
_FAILURE_KW_RE = re.compile(
    r"\b(" + "|".join(FAILURE_KEYWORDS) + r")\b", re.IGNORECASE
)


def matches_failure_keyword(text: str) -> bool:
    return bool(_FAILURE_KW_RE.search(text))


# 结构化错误观测判定：工具返回的错误消息以 error:/exception:/traceback 等
# 开头。玩具语料本身是"关于错误分析的散文"，词面词典会把正文当错误——
# 错误观测与领域散文的区分必须靠结构前缀（SSF 保留规则与伪判官共用）。
_ERROR_OBS_RE = re.compile(r"^\s*(error|exception|traceback|fatal|failed)\b[:：\s]?", re.I)


def is_error_observation(text: str) -> bool:
    return bool(_ERROR_OBS_RE.match(text))

# SSF 占位符：⟦folded:F3 | 摘要...⟧
FOLD_PLACEHOLDER_RE = re.compile(r"^⟦folded:(?P<fid>\w+) \| (?P<digest>.*)⟧$")


def _short(obj: Any, limit: int = MAX_CONTENT_CHARS) -> str:
    s = obj if isinstance(obj, str) else repr(obj)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def render_event_line(
    ev: Any,
    fold: dict[str, str] | None = None,
    table: dict[str, str] | None = None,
) -> str:
    """渲染单个事件为一行。

    fold: {event_id: 占位符文本}（SSF 折叠视图）。
    table: {fold_id: 原文}；给定且本行内容是占位符时展开为原文。
    """
    payload = dict(ev.payload or {})
    content = payload.pop("content", None)
    head = f"[{ev.index}] {ev.kind} {ev.agent}"
    if ev.action:
        head += f" {ev.action}"
    extra = ""
    if payload:
        extra = " " + _short(payload)
    if content is None:
        return head + extra

    text = str(content)
    if fold and ev.id in fold:
        text = fold[ev.id]
    if table:
        m = FOLD_PLACEHOLDER_RE.match(text.strip())
        if m and m.group("fid") in table:
            text = table[m.group("fid")]
    return f"{head}{extra} :: {_short(text)}"


def render_trace(
    trajectory: "Trajectory",
    *,
    fold: dict[str, str] | None = None,
    table: dict[str, str] | None = None,
    include_task: bool = True,
) -> str:
    """渲染整条轨迹（任务头 + 事件行）。"""
    lines: list[str] = []
    if include_task:
        out = "SUCCESS" if trajectory.outcome.success else "FAILURE"
        lines.append(f"task: {trajectory.task}")
        lines.append(
            f"outcome: {out}"
            + (f" ({trajectory.outcome.note})" if trajectory.outcome.note else "")
        )
        lines.append(TRACE_BEGIN)
    for ev in trajectory.events:
        lines.append(render_event_line(ev, fold, table))
    if include_task:
        lines.append(TRACE_END)
    return "\n".join(lines)


def judge_view(bundle) -> str:
    """判官视图：SSF 折叠产物存在则用折叠视图，否则全量视图。

    产物键 ``represent/ssf`` 是算法间的数据契约（下游按名消费，不 import
    SSF 模块）；SSF 未配置时显式降级为全量渲染。
    """
    ssf = bundle.get("represent", "ssf")
    if isinstance(ssf, dict) and ssf.get("fold"):
        return render_trace(bundle.trajectory, fold=ssf["fold"])
    return render_trace(bundle.trajectory)
