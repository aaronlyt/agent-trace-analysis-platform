"""R1 显著性折叠（SSF, Semantic Saliency Folding）—— TrajAudit, arXiv:2605.26563。

机制（原文 Algorithm 1，按本框架域适配）：
* **补丁保留**：内容命中代码 diff 结构模式（``--- a`` / ``+++ b`` /
  ``@@ -N,M +P,Q @@``）→ 保留全文（明确体现 agent 改了什么）；
* **失败观测保留**：结构化错误消息（``error:`` / ``exception`` 等前缀，
  :func:`is_error_observation`）→ 保留。原文用词面词典 K（词面子串、
  观测内任意位置），但本框架玩具语料本身是"关于错误分析的散文"，词面
  匹配会把正文误判为错误——故默认 ``keyword_mode=strict``（结构前缀），
  这是相对原文 K 的**收窄**（``invalid/denied/missing/exhausted/timeout``
  默认不参与保留判定；``fatal`` 为新增）。``keyword_mode=loose`` 并非
  恢复原文行为，而是 strict 前缀 ∪ **词边界**词典的叠加（原文为纯子串
  匹配，"errors"/"failover" 等形态原文命中、词边界不命中）——适用于
  代码/工具日志域，即原文场景【推断】；
* **短观测豁免**：长度 ≤ ``min_fold_len``（默认 120 字符）的观测不折叠。
  原文 Algorithm 1 无长度条件（无信号观测一律折叠），此为工程适配
  （短观测折叠省 token 有限、占位符+摘要可能不省反增）【推断】；
* 其余长观测 → 替换为**可逆占位符** ``⟦folded:F3 | 摘要…⟧``，原文存入
  side table，可按需展开（investigator 的 inspection tool 雏形；阶段二
  的单遍判官只读折叠视图，按需展开留作阶段三 L2 下钻）。

与原文的差异【推断】：占位符附带首行摘要（≤100 字符），让单遍判官仍能
从折叠视图获得最小证据（如检索命中列表），这是对本框架"无下钻工具的
单遍判官"场景的工程适配；原文占位符为纯提示文本。原文 investigator
以多轮下钻消费折叠视图（折叠步中 20.2% 会被按需展开），本切片无下钻，
信息损失由摘要兜底。

产物：``{"fold", "table", "stats"}``；不修改原始事件（视图与数据分离）。
原文数字参照（口径不同，仅作参照）：平均 94.6% 的步可折叠、折叠步中仅
20.2% 需按需展开——原文分母为全部 trajectory steps，本实现 ``fold_ratio``
分母为 TOOL_RESULT 事件数，两者不可直接对比。
"""

from __future__ import annotations

import re

from atap.core.registry import register
from atap.core.render import (
    FOLD_PLACEHOLDER_RE,
    is_error_observation,
    matches_failure_keyword,
)
from atap.represent.base import Representer

# 统一 diff 的三种结构标记（原文的 patch pattern P）
PATCH_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^--- a/", re.MULTILINE),
    re.compile(r"^\+\+\+ b/", re.MULTILINE),
    re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", re.MULTILINE),
)

DIGEST_CHARS = 100


def matches_patch(text: str) -> bool:
    return any(p.search(text) for p in PATCH_PATTERNS)


@register
class SSFRepresenter(Representer):
    stage = "represent"
    name = "ssf"

    def run_one(self, bundle, ctx) -> None:
        extra = [str(k).lower() for k in (self.param("extra_keywords") or [])]
        loose = self.param("keyword_mode", "strict") == "loose"
        min_len = int(self.param("min_fold_len", 120))
        fold: dict[str, str] = {}   # event_id -> 占位符
        table: dict[str, str] = {}  # fid -> 原文
        stats = {
            "n_tool_results": 0,
            "n_folded": 0,
            "n_kept_error": 0,
            "n_kept_patch": 0,
            "n_kept_short": 0,
        }
        for ev in bundle.trajectory.events:
            if ev.kind != "TOOL_RESULT":
                continue
            stats["n_tool_results"] += 1
            content = str(ev.payload.get("content", ""))
            if not content:
                continue
            if is_error_observation(content) or any(k in content.lower() for k in extra):
                stats["n_kept_error"] += 1
                continue
            if loose and matches_failure_keyword(content):
                stats["n_kept_error"] += 1
                continue
            if matches_patch(content):
                stats["n_kept_patch"] += 1
                continue
            if len(content) <= min_len:
                stats["n_kept_short"] += 1
                continue
            fid = f"F{len(table) + 1}"
            digest = " ".join(content.split())[:DIGEST_CHARS]
            fold[ev.id] = f"⟦folded:{fid} | {digest}⟧"
            table[fid] = content
            stats["n_folded"] += 1

        stats["fold_ratio"] = (
            round(stats["n_folded"] / stats["n_tool_results"], 4)
            if stats["n_tool_results"]
            else 0.0
        )
        bundle.put("represent", self.name, {"fold": fold, "table": table, "stats": stats})


def unfold(artifact: dict, fid: str) -> str:
    """按需展开（investigator 的 inspection tool 雏形）。"""
    content = artifact.get("table", {}).get(fid)
    if content is None:
        raise KeyError(f"折叠表里没有 {fid}；可用：{sorted(artifact.get('table', {}))}")
    return content


def unfold_line(line: str, artifact: dict) -> str:
    """把渲染行中的占位符展开为原文（调试/审计用）。"""
    m = FOLD_PLACEHOLDER_RE.match(line.strip())
    if m:
        return unfold(artifact, m.group("fid"))
    return line
