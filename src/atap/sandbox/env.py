"""玩具研究问答环境 —— mock 检索语料 + 工具 + 验证器。

三 agent 流水线（planner→searcher→reporter）的可执行沙盒：
* ``search(query)``：关键词命中 mock 语料（综述里的真实论文条目），返回
  机器友好首行 ``search results for 'q': N docs [d1, d3]``；
* ``read_doc(doc_id)``：返回长文档内容（SSF 折叠的主要对象）；
* ``submit(answer)`` + ``verify``：答案必须同时含 gold 关键词与**实际读过**
  的文档编号（任务规格），验证器给出可观测的失败说明。

确定性：无随机源，同一 (task, fault) 永远产出同一轨迹（种子只影响
demo 的任务/故障抽样，不改变单条轨迹内容）。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Mock 语料：错误分析综述里的六篇论文条目（title / 关键词 / 含答案的正文）
# ---------------------------------------------------------------------------

CORPUS: dict[str, dict] = {
    "d1": {
        "title": "TrajAudit: trajectory failure diagnosis for coding agents",
        "keywords": {"trajaudit", "folding", "diagnosis", "coding"},
        "content": (
            "TrajAudit is a failure-diagnosis framework for repository-level coding agents. "
            "Its investigator agent is supported by two modules: semantic saliency folding (SSF), "
            "which compresses failure-irrelevant observations into recoverable placeholders while "
            "retaining code patches and failure-indicative keywords, and prior failure reasoning "
            "(PFR), which derives preliminary diagnostic guidance from test failure reports. "
            + "The investigator can expand folded observations on demand through inspection tools. " * 8
        ),
        "answer": "semantic saliency folding",
    },
    "d2": {
        "title": "MAST: Multi-Agent System Failure Taxonomy",
        "keywords": {"mast", "taxonomy", "multi-agent", "failure"},
        "content": (
            "MAST identifies 14 fine-grained failure modes in three categories: system design "
            "issues, inter-agent misalignment, and task-verification issues. It was built from "
            "1642 annotated execution traces using grounded theory, achieving inter-annotator "
            "agreement kappa 0.88, and ships an LLM-as-a-judge annotation pipeline calibrated "
            "to kappa 0.77 with few-shot examples. "
            + "The most frequent category is system design issues. " * 10
        ),
        "answer": "14 failure modes",
    },
    "d3": {
        "title": "Who&When: automated failure attribution",
        "keywords": {"who", "when", "attribution", "judgement"},
        "content": (
            "WhoDunitAndWhen introduces the failure attribution task and dataset for multi-agent "
            "systems, and compares three automated judgement methods: all-at-once, step-by-step, "
            "and binary search. All-at-once gives the LLM the complete failure log in a single "
            "window and asks for the failure-responsible agent and the decisive error step; it "
            "achieves the best agent-level accuracy, 54.33 with GPT-4o. "
            + "Step-level accuracy remains below 15 percent for all methods. " * 10
        ),
        "answer": "all-at-once",
    },
    "d4": {
        "title": "TraceProbe: action signatures for trajectory monitoring",
        "keywords": {"traceprobe", "signature", "monitoring", "loop"},
        "content": (
            "TraceProbe canonicalizes agent actions into nine action classes with parameter "
            "fingerprints, labels each step with effect tags such as survived, reverted, "
            "justified and off-anchor, and monitors progress by aligning a trajectory with "
            "success signatures via longest common subsequence. Loop predicates over action "
            "n-grams detect search loops reliably. "
            + "The reference-solution dependency stays below one percent. " * 10
        ),
        "answer": "action signatures",
    },
    "d5": {
        "title": "DRIFT: claim ledger for deep research agents",
        "keywords": {"drift", "claim", "ledger", "research"},
        "content": (
            "DRIFT maintains a claim ledger for deep-research agent trajectories: every key claim "
            "records its content, introduction position, first-effective location, reuse set, "
            "claim type and commitment status, with support levels direct, weak, missing and "
            "conflicting. Auditing unsupported claims that are later reused as facts improves "
            "macro F1 by up to 33 points on TELBench. "
            + "A third of successful trajectories contain hidden erroneous steps. " * 10
        ),
        "answer": "claim ledger",
    },
    "d6": {
        "title": "AgentDebugX: full-stack agent debugging",
        "keywords": {"agentdebugx", "debugging", "recovery", "vocabulary"},
        "content": (
            "AgentDebugX provides an engineering failure vocabulary of 19 patterns across "
            "planning, memory, tool, verification and collaboration domains, a free rule pack "
            "for malformed tool calls and premature success claims, and a policy-gated recovery "
            "loop that re-runs failed trajectories with suggested fixes. Residual clusters of "
            "unrecognized failures propose new vocabulary entries for human adjudication. "
            + "Closed-loop repair fixes 13 of 73 failing runs in one shot. " * 10
        ),
        "answer": "19 failure patterns",
    },
}

# 任务族：问某篇论文提出的方法；答案须以 "(dK)" 形式引用实际读过的文档。
TASKS: dict[str, dict] = {
    "q-trajaudit": {
        "text": (
            "Which compression technique does TrajAudit propose for trajectory observations? "
            "Your answer must end with the id of one document you actually read, like '(d1)'."
        ),
        "query": "trajaudit folding failure taxonomy",
        "gold_doc": "d1",
        "gold_answer": "semantic saliency folding",
    },
    "q-who-when": {
        "text": (
            "Which judgement method does Who&When identify as best for agent-level failure "
            "attribution? Your answer must end with the id of one document you actually read, "
            "like '(d3)'."
        ),
        "query": "who when attribution failure",
        "gold_doc": "d3",
        "gold_answer": "all-at-once",
    },
    "q-drift": {
        "text": (
            "Which bookkeeping structure does DRIFT maintain for deep research agent "
            "trajectories? Your answer must end with the id of one document you actually read, "
            "like '(d5)'."
        ),
        "query": "drift claim ledger failure",
        "gold_doc": "d5",
        "gold_answer": "claim ledger",
    },
}

_TOKEN_RE = re.compile(r"[a-z]+")


def search(query: str) -> str:
    """确定性检索：token 重叠命中；首行机器可解析（伪判官依赖该格式）。"""
    tokens = set(_TOKEN_RE.findall(query.lower()))
    hits: list[str] = []
    for did, doc in CORPUS.items():
        if tokens & doc["keywords"]:
            hits.append(did)
    hits.sort()
    return f"search results for '{query}': {len(hits)} docs [{', '.join(hits)}]"


def read_doc(doc_id: str) -> str:
    if doc_id not in CORPUS:
        return f"error: invalid doc_id '{doc_id}' (available: {sorted(CORPUS)})"
    return f"{CORPUS[doc_id]['title']}\n{CORPUS[doc_id]['content']}"


def verify(task_id: str, answer: str, read_docs: list[str]) -> tuple[bool, str]:
    """验证器：可观测失败说明（伪判官/判官 prompt 都只能看到这个文本）。

    检查顺序刻意区分故障：未读任何文档 → 无据引用 → 缺失引用格式 →
    答案错误，保证不同故障产生可区分的 verifier 说明。
    """
    task = TASKS[task_id]
    gold_doc, gold_answer = task["gold_doc"], task["gold_answer"]
    if not read_docs:
        return False, "failed: answer submitted without reading any document"
    cited = re.findall(r"\bd\d\b", answer)
    if any(c not in read_docs for c in cited):
        bad = next(c for c in cited if c not in read_docs)
        return False, f"failed: cited document {bad} was never read"
    if not cited:
        return False, "failed: answer missing required citation of a read doc id"
    if gold_answer.lower() not in answer.lower():
        return False, "failed: answer does not contain the correct method name"
    return True, f"passed: answer cites read document and matches gold '{gold_answer}' ({gold_doc})"
