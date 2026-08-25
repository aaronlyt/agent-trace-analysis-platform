"""确定性伪判官 —— FakeLLMClient 的默认 handler，使 LLM-judge 链路可离线跑通。

只依据**判官可见的轨迹文本**（core/render.py 渲染的折叠视图）做规则化
判定，绝不读取 meta["injected_fault"] 之类的 ground truth（那会泄漏答案、
让验收测试失去意义）。规则与沙盒故障的**可观测症状**一一对应：

=============  ============================  ========  ==========
故障（沙盒）    可观测症状                     规则序    MAST 代码
=============  ============================  ========  ==========
畸形工具调用    TOOL_RESULT 含失败指示词        1        FM-2.6
无进展循环      相邻 3 次相同 TOOL_CALL         2        FM-1.3
信息隐瞒        声称无结果但检索结果非空         3        FM-2.4
过早终止        未见 read_doc 即 submit         4        FM-3.1
无据引用        引用了未 read 过的文档 id        5        FM-3.3
违反任务规格    VERIFIER 报告格式/必填缺失      6        FM-1.1
（兜底）        最后一个 LLM_CALL               7        FM-2.6
=============  ============================  ========  ==========
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from atap.llm.base import ChatMessage
from atap.core.render import TRACE_BEGIN, TRACE_END, is_error_observation

MAST_CODES = {
    "malformed_tool_call": "FM-2.6",
    "step_repetition": "FM-1.3",
    "info_withholding": "FM-2.4",
    "premature_termination": "FM-3.1",
    "ungrounded_citation": "FM-3.3",
    "disobey_task_spec": "FM-1.1",
}


@dataclass
class Line:
    idx: int
    kind: str
    agent: str
    action: str | None
    payload: str
    content: str


@dataclass
class Signature:
    """一条被检出的异常症状。"""

    fault: str
    step: int
    agent: str
    code: str
    reason: str
    fix: str
    detail: str = ""


_LINE_RE = re.compile(r"^\[(\d+)\]\s+(\S+)\s+(\S+)(?:\s(.*))?$")
_DOC_ID_RE = re.compile(r"\bd(\d+)\b")
_SEARCH_DOCS_RE = re.compile(r"search results[^:]*:\s*(\d+)\s*docs\s*\[([^\]]*)\]")


def _parse_block(block: str) -> list[Line]:
    lines: list[Line] = []
    for raw in block.splitlines():
        raw = raw.strip()
        if not raw.startswith("["):
            continue
        m = _LINE_RE.match(raw)
        if not m:
            continue
        idx, kind, agent, rest = int(m.group(1)), m.group(2), m.group(3), m.group(4) or ""
        # 无 action 的事件行形如 "[11] VERIFIER verifier :: failed: ..."，
        # 正则已吃掉 "::" 前的空格，必须按 "::" 而非 " :: " 切分
        if "::" in rest:
            left, _, content = rest.partition("::")
            content = content.strip()
        else:
            left, content = rest, ""
        tokens = left.split()
        action: str | None = None
        payload_parts: list[str] = []
        for tok in tokens:
            if tok.startswith("{"):
                payload_parts.append(tok)
            elif action is None:
                action = tok
            else:
                payload_parts.append(tok)
        lines.append(
            Line(
                idx=idx,
                kind=kind,
                agent=agent,
                action=action,
                payload=" ".join(payload_parts),
                content=content.strip(),
            )
        )
    return lines


def find_trace_block(messages: list[ChatMessage]) -> str | None:
    """从消息列表中取出最后一个轨迹块（TRACE_BEGIN..TRACE_END）。"""
    for msg in reversed(messages):
        content = str(msg.get("content", ""))
        if TRACE_BEGIN in content and TRACE_END in content:
            start = content.index(TRACE_BEGIN) + len(TRACE_BEGIN)
            end = content.index(TRACE_END)
            return content[start:end]
    return None


def find_outcome(messages: list[ChatMessage]) -> bool:
    """从任务头取 outcome（SUCCESS 视为成功轨迹）。"""
    for msg in reversed(messages):
        content = str(msg.get("content", ""))
        m = re.search(r"outcome:\s*(SUCCESS|FAILURE)", content)
        if m:
            return m.group(1) == "SUCCESS"
    return False


# ---------------------------------------------------------------------------
# 症状检测规则（按序短路；一条轨迹通常只命中一个注入故障的症状）
# ---------------------------------------------------------------------------


def _sig(fault: str, step: int, agent: str, detail: str) -> Signature:
    code = MAST_CODES[fault]
    reasons = {
        "malformed_tool_call": f"step {step}: {agent} 发起的工具调用返回错误，行动与意图不符（推理-行动失配）",
        "step_repetition": f"step {step}: {agent} 无进展地重复同一工具调用",
        "info_withholding": f"step {step}: {agent} 声称没有找到文档，但此前的检索结果非空——向下游隐瞒了关键信息",
        "premature_termination": f"step {step}: {agent} 在未读取任何证据文档的情况下就提交了答案（过早终止）",
        "ungrounded_citation": f"step {step}: {agent} 引用了从未用 read_doc 读过的文档（验证不正确）",
        "disobey_task_spec": f"step {step}: {agent} 的最终答案违反任务规格（缺失必填字段/格式）",
    }
    fixes = {
        "malformed_tool_call": f"避免 malformed_tool_call：在 step {step} 调用工具前校验参数完整性再发起调用。",
        "step_repetition": f"避免 step_repetition：不要在 step {step} 重复相同的工具调用，应使用已有结果继续推进。",
        "info_withholding": f"避免 info_withholding：在 step {step} 如实报告检索到的文档，把结果传递给下游。",
        "premature_termination": f"避免 premature_termination：step {step} 的决策在未检索阅读证据的情况下就准备提交，应先 search 并 read_doc 再提交答案。",
        "ungrounded_citation": f"避免 ungrounded_citation：在 step {step} 只引用实际用 read_doc 读过的文档。",
        "disobey_task_spec": f"避免 disobey_task_spec：在 step {step} 按任务要求的格式给出答案（含必填的文档编号）。",
    }
    return Signature(
        fault=fault, step=step, agent=agent, code=code,
        reason=reasons[fault], fix=fixes[fault], detail=detail,
    )


def detect_signatures(lines: list[Line]) -> list[Signature]:
    sigs: list[Signature] = []

    # 规则 1：畸形工具调用 —— 首个结构化错误观测（error:/exception 前缀），
    # 归到其紧邻的 TOOL_CALL（散文中的词典词不算错误观测）
    calls = [ln for ln in lines if ln.kind == "TOOL_CALL"]
    results = [ln for ln in lines if ln.kind == "TOOL_RESULT"]
    for res in results:
        if is_error_observation(res.content):
            prev_calls = [c for c in calls if c.idx < res.idx]
            if prev_calls:
                call = prev_calls[-1]
                sigs.append(_sig("malformed_tool_call", call.idx, call.agent, res.content[:120]))
            break

    # 规则 2：无进展循环 —— 相邻 3 次签名完全相同的 TOOL_CALL，归到第 2 次
    for i in range(len(calls) - 2):
        a, b, c3 = calls[i], calls[i + 1], calls[i + 2]
        if (a.agent, a.action, a.payload) == (b.agent, b.action, b.payload) == (c3.agent, c3.action, c3.payload):
            sigs.append(_sig("step_repetition", b.idx, b.agent, f"repeated {a.action}"))
            break

    # 规则 3：信息隐瞒 —— 声称无结果，但此前某次 search 明明返回了文档
    search_hits: list[int] = []  # 命中过非空检索的事件 idx
    for ln in lines:
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "search":
            m = _SEARCH_DOCS_RE.search(ln.content)
            if m and int(m.group(1)) > 0:
                search_hits.append(ln.idx)
        if ln.kind in ("AGENT_MESSAGE", "LLM_CALL", "HANDOFF") and re.search(
            r"no (relevant|results|documents)|nothing found|未找到|没有找到", ln.content, re.I
        ):
            if any(h < ln.idx for h in search_hits):
                sigs.append(_sig("info_withholding", ln.idx, ln.agent, ln.content[:120]))
                break

    # 规则 4：过早终止 —— submit 前从未 read_doc。归因到 submit 前最后一次
    # 决策（LLM_CALL）而非 submit 动作本身：决定跳过检索的规划步早于终止
    # 动作（Who&When Eq.5 earliest），与沙盒 ground truth 的 onset=plan 对齐
    read_doc_calls = [c for c in calls if (c.action or "") == "read_doc"]
    decisions = [ln for ln in lines if ln.kind == "LLM_CALL"]
    for c in calls:
        if (c.action or "") == "submit" and not [r for r in read_doc_calls if r.idx < c.idx]:
            target = next((d for d in reversed(decisions) if d.idx < c.idx), c)
            sigs.append(_sig("premature_termination", target.idx, target.agent, c.payload[:120]))
            break

    # 规则 5：无据引用 —— 答案/消息断言式引用了到该步为止从未 read 过的 doc id
    read_ids: set[str] = set()
    for ln in lines:
        if ln.kind == "TOOL_CALL" and (ln.action or "") == "read_doc":
            m = _DOC_ID_RE.search(ln.payload)
            if m:
                read_ids.add(f"d{m.group(1)}")
            continue
        if ln.kind not in ("LLM_CALL", "AGENT_MESSAGE") or (ln.action or "") == "search":
            continue
        mentions = {f"d{m.group(1)}" for m in _DOC_ID_RE.finditer(ln.content)}
        if not mentions:
            continue
        is_assertion = bool(
            re.search(r"cite|cited|引用|依据|according to|based on", ln.content, re.I)
        )
        for did in mentions:
            cited_form = f"[{did}]" in ln.content or f"({did})" in ln.content
            if did not in read_ids and (is_assertion or cited_form):
                sigs.append(_sig("ungrounded_citation", ln.idx, ln.agent, f"cited {did} unread"))
                break
        if sigs and sigs[-1].fault == "ungrounded_citation":
            break

    # 规则 6：违反任务规格 —— VERIFIER 报告格式/必填缺失，归到最后一次答案生成
    verifier_issues = [
        ln for ln in lines
        if ln.kind == "VERIFIER" and re.search(r"missing|required|format|格式|必填", ln.content, re.I)
    ]
    if verifier_issues:
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        if llm_calls:
            last = llm_calls[-1]
            sigs.append(_sig("disobey_task_spec", last.idx, last.agent, verifier_issues[-1].content[:120]))

    # 规则 7：兜底
    if not sigs:
        llm_calls = [ln for ln in lines if ln.kind == "LLM_CALL"]
        if llm_calls:
            last = llm_calls[-1]
            sigs.append(
                Signature(
                    fault="unknown", step=last.idx, agent=last.agent, code="FM-2.6",
                    reason=f"step {last.idx}: 未检出显式症状，保守归因到最后的模型决策",
                    fix=f"复核 step {last.idx} 的决策依据。", detail="",
                )
            )
    return sigs


# ---------------------------------------------------------------------------
# 按 tag 输出三种结构化 JSON（字段名与算法模块的 pydantic 模型一致）
# ---------------------------------------------------------------------------


def _segment_local_signatures(lines: list[Line]) -> list[Signature]:
    """二分定位轮次的**片段内**症状检测。

    复用整轨规则之外补三条片段可见的判据（判官只能看到切片，必须靠
    切片内证据作答；整轨规则本身不动——它服务于 all_at_once 的口径）：
    * 空 payload 的 TOOL_CALL（畸形调用在调用行即可见，无需等错误观测）；
    * LLM_CALL 自述"凭记忆直接提交"（过早终止的决策步文本）；
    * 片段内已读到文档/非空检索，其后却声称"没有找到"（信息隐瞒的
      片段内矛盾）。
    """
    sigs = detect_signatures(lines)
    # 兜底签名（fault="unknown"）不携带症状证据：真实判官看到良性片段
    # 会答 upper half，伪判官同样不能把"保守归因兜底"当作症状
    sigs = [s for s in sigs if s.fault != "unknown"]
    calls = [ln for ln in lines if ln.kind == "TOOL_CALL"]
    for c in calls:
        # 空 payload 的工具调用（畸形）：空 dict 渲染为无参行，调用行即可见
        if c.action and not c.payload.strip() and not any(
            s.fault == "malformed_tool_call" and s.step == c.idx for s in sigs
        ):
            sigs.append(
                _sig("malformed_tool_call", c.idx, c.agent, "empty tool arguments")
            )
    evidence_idx: list[int] = []
    for ln in lines:
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "search":
            m = _SEARCH_DOCS_RE.search(ln.content)
            if m and int(m.group(1)) > 0:
                evidence_idx.append(ln.idx)
        if ln.kind == "TOOL_RESULT" and (ln.action or "") == "read_doc" \
                and not is_error_observation(ln.content):
            evidence_idx.append(ln.idx)
    for ln in lines:
        if ln.kind in ("AGENT_MESSAGE", "LLM_CALL", "HANDOFF") and re.search(
            r"no (relevant|results|documents)|nothing found|未找到|没有找到", ln.content, re.I
        ):
            if any(h < ln.idx for h in evidence_idx) and not any(
                s.fault == "info_withholding" and s.step == ln.idx for s in sigs
            ):
                sigs.append(_sig("info_withholding", ln.idx, ln.agent, ln.content[:120]))
    for ln in lines:
        if ln.kind == "LLM_CALL" and re.search(
            r"recall|from memory|submit directly|凭记忆", ln.content, re.I
        ):
            if not any(s.fault == "premature_termination" for s in sigs):
                sigs.append(_sig("premature_termination", ln.idx, ln.agent, ln.content[:120]))
    return sigs


def pseudo_judge_handler(tag: str, messages: list[ChatMessage]) -> "str | None":
    if tag == "feedback_match":
        # 环境侧调用，无轨迹块——必须在 find_trace_block 之前处理
        # 故障规格 vs 自由文本反馈的确定性模拟：故障类型词命中即 yes；
        # 否则按"故障规格与反馈的 CJK bigram 重叠"模拟语义理解
        content = str(messages[0].get("content", "")) if messages else ""
        m_kind = re.search(r"故障类型：([\w_]+)", content)
        m_desc = re.search(r"描述：(.*)", content)
        m_fb = re.search(r"修正反馈：\n(.*?)(?:\n\n|$)", content, re.S)
        if not (m_kind and m_fb):
            return "no"
        kind = m_kind.group(1)
        fb = m_fb.group(1).lower()
        if kind in fb or kind.replace("_", " ") in fb:
            return "yes"
        desc = m_desc.group(1) if m_desc else ""
        cjk = re.compile(r"[\u4e00-\u9fff]+")

        def _bigrams(text: str) -> set[str]:
            out: set[str] = set()
            for run in cjk.findall(text):
                out.update(run[i: i + 2] for i in range(len(run) - 1))
            return out

        return "yes" if len(_bigrams(desc) & _bigrams(fb)) >= 3 else "no"

    block = find_trace_block(messages)
    if block is None:
        return None
    lines = _parse_block(block)
    succeeded = find_outcome(messages)
    sigs = [] if succeeded else detect_signatures(lines)

    if tag == "judge_eval":
        if succeeded:
            return json.dumps(
                {"score": 9.0, "summary": "任务成功，未见异常症状", "findings": []},
                ensure_ascii=False,
            )
        findings = [
            {"severity": "critical", "description": s.reason, "step": s.step}
            for s in sigs
        ]
        return json.dumps(
            {
                "score": 2.5 if sigs else 4.0,
                "summary": "任务失败：" + ("；".join(s.detail or s.reason for s in sigs[:2]) or "未见显式症状"),
                "findings": findings,
            },
            ensure_ascii=False,
        )

    if tag == "mast_judge":
        labels = [
            {"code": s.code, "reason": s.reason, "step": s.step} for s in sigs
        ]
        return json.dumps({"labels": labels}, ensure_ascii=False)

    if tag == "all_at_once":
        if not sigs:
            return None
        s = sigs[0]
        return json.dumps(
            {
                "responsible_agent": s.agent,
                "step": s.step,
                "reason": s.reason,
                "fix_suggestion": s.fix,
                "confidence": 0.7 if s.fault != "unknown" else 0.3,
                "failure_mode": s.code,
            },
            ensure_ascii=False,
        )

    if tag == "binary_search":
        # 片段内症状检测：有任何症状 → 错误在Shown下半段
        seg_sigs = [] if succeeded else _segment_local_signatures(lines)
        return "lower half" if seg_sigs else "upper half"

    if tag == "binary_search_refine":
        # 从 system 提示解析二分锁定的 step/agent（定位已定，反思不二猜）
        system = str(messages[0].get("content", "")) if messages else ""
        m_step = re.search(r"step (\d+)", system)
        m_agent = re.search(r"agent ([A-Za-z0-9_\-]+)", system)
        step = int(m_step.group(1)) if m_step else (sigs[0].step if sigs else 0)
        agent = m_agent.group(1) if m_agent else (sigs[0].agent if sigs else "unknown")
        hit = next((s for s in sigs if s.step == step), None)
        if hit is None and sigs:
            hit = sigs[0]
        if hit is not None:
            return json.dumps(
                {
                    "reason": hit.reason,
                    "fix_suggestion": hit.fix,
                    "confidence": 0.65,
                    "failure_mode": hit.code,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "reason": f"step {step}（agent {agent}）的行为构成最早的"
                "决定性错误：修正该步即可改变失败走向。",
                "fix_suggestion": f"复核 step {step}（agent {agent}）的决策依据并修正。",
                "confidence": 0.4,
                "failure_mode": None,
            },
            ensure_ascii=False,
        )

    if tag == "feedback_reflection":
        s = sigs[0] if sigs else None
        if s is not None:
            feedback = f"{s.reason}。下一轮请避免：{s.fix}"
        else:
            feedback = "未见显式症状：下一轮请逐步核对任务要求后再提交。"
        return json.dumps({"feedback": feedback}, ensure_ascii=False)

    return None
