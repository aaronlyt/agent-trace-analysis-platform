"""Toy research QA environment -- mock retrieval corpus + tools + verifier.

An executable sandbox for the three-agent pipeline (planner→searcher→reporter):
* ``search(query)``: keyword hits against the mock corpus (real paper entries
  from the survey), returning a machine-friendly first line
  ``search results for 'q': N docs [d1, d3]``;
* ``read_doc(doc_id)``: returns long document content (the main target of SSF
  folding);
* ``submit(answer)`` + ``verify``: the answer must contain both the gold
  keyword and the id of a document **actually read** (task spec); the verifier
  emits an observable failure explanation.

Determinism: no randomness source; the same (task, fault) always yields the
same trajectory (the seed only affects demo task/fault sampling, never the
content of an individual trajectory).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Mock corpus: six paper entries from the error-analysis survey (title /
# keywords / body containing the answer)
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

# Task family: ask which method a given paper proposes; the answer must cite a document actually read in the form "(dK)".
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


def _search_hits(query: str) -> list[str]:
    """Deterministic retrieval hits (doc ids ascending) -- shared by search and qrels."""
    tokens = set(_TOKEN_RE.findall(query.lower()))
    hits: list[str] = []
    for did, doc in CORPUS.items():
        if tokens & doc["keywords"]:
            hits.append(did)
    hits.sort()
    return hits


def search(query: str) -> str:
    """Deterministic retrieval: token-overlap hits; machine-parseable first line (the pseudo-judge relies on this format)."""
    hits = _search_hits(query)
    return f"search results for '{query}': {len(hits)} docs [{', '.join(hits)}]"


def qrels(task_id: str) -> dict[str, list[str]]:
    """Two-level qrels annotation required by RG/UG attribution (search-agent
    diagnosis 2608.01913 §4.3).

    E(q) = topically relevant documents (retrieval hits of the task's
    canonical query); G(q) ⊆ E = the sufficient set from which the gold
    answer can be derived. Known by construction: E is taken from the
    deterministic retrieval hits of the task query, G from the gold document
    (each of the three tasks has its gold doc inside its canonical query's
    hit set, so G ⊆ E holds). Enters the trajectory via meta["qrels"]; the
    attribution side only reads the data and never imports the sandbox.
    """
    task = TASKS[task_id]
    evidence = _search_hits(task["query"])
    if task["gold_doc"] not in evidence:
        raise AssertionError(
            f"{task_id} construction error: gold {task['gold_doc']} not in evidence"
        )
    return {"evidence": evidence, "gold": [task["gold_doc"]]}


def read_doc(doc_id: str) -> str:
    if doc_id not in CORPUS:
        return f"error: invalid doc_id '{doc_id}' (available: {sorted(CORPUS)})"
    return f"{CORPUS[doc_id]['title']}\n{CORPUS[doc_id]['content']}"


def verify(task_id: str, answer: str, read_docs: list[str]) -> tuple[bool, str]:
    """Verifier: observable failure explanation (both the pseudo-judge and
    the judge prompts only ever see this text).

    The check order deliberately separates faults: no document read →
    ungrounded citation → missing citation format → wrong answer. The
    explanation text separates the six faults only at "fault group"
    granularity (the first two groups are each shared by two faults); the
    six-way fine-grained separation is completed by trajectory symptoms (the
    event lines visible to the judge).
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
    # [leak fix, review 2026-08-27] the success note must not name the gold
    # answer/doc: the VERIFIER event line is rendered into every judge view
    # (render_trace emits all events), so the old wording leaked the oracle
    # into judge_eval prompts on successful trajectories. "matches the
    # expected method name" keeps the same observable information content
    # for the agent (the verifier's public feedback) without the oracle.
    return True, "passed: answer cites a read document and matches the expected method name"
