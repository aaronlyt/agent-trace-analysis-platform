"""Failure clustering + residual vocabulary extension (inducer) —— AgentDebugX, arXiv:2607.18754 §3.4.

Paper mechanism (a single paragraph, no pseudocode / no dedicated prompt /
not evaluated —— Appendix E: "implemented but not yet evaluated, and
induction requires human acceptance"):
* the judge records a novel-mode candidate for failures outside the seed
  vocabulary (the paper restricts this to "recurring"; in this
  implementation the judge records novel for any out-of-seed symptom, and
  the "recurring" semantics are delivered by a corpus-level support gate
  support>=threshold [adaptation]);
* clustering = "label, then lexical or embedding similarity, **gated by a
  support threshold**";
* each cluster nominates one candidate mode, deduplicated against the
  seeds;
* **proposals never overwrite the curated vocabulary** —— they take effect
  only after human acceptance.

This implementation [adaptation/inference declared]: the residual entry =
the ``novel`` labels of classify/mast_judge (with ``allow_novel=True`` the
judge outputs a symptom phrase for out-of-seed symptoms); clustering =
character 3-gram Jaccard over symptom phrases (the lexical path,
deterministic and usable offline; for the embedding path the paper
specifies no model, so an interface is left open [not specified in paper]);
the support/similarity/dedup thresholds are all self-chosen parameters
(dedup against the seeds is decided by the content-token overlap of
"cluster tokens vs MAST mode names" —— mode names are only 2-3 words, so a
long symptom cluster can almost never reach the threshold, the dedup
branch nearly never fires, honestly declared as a weakened mechanism);
naming = top-3 most frequent content tokens of the symptoms (the paper
does not specify the namer); kinship = content-token overlap with the MAST
seeds (the paper's example only says "notes its kinship to the existing
lost-handoff category"). The human gate = proposals land with
``status="proposed"`` and, after adjudication via ``atap taxonomy
accept``, are written to the extended modes file, which mast_judge loads
via ``extra_modes_file`` —— the seeds are never modified automatically.

Artifact (``classify/inducer``, corpus-level, same content written back to
every bundle): ``{proposals: [...], stats, thresholds}``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from atap.classify.base import Classifier
from atap.classify.taxonomy import MAST_MODES
from atap.core.registry import register

_STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "before", "after",
    "is", "are", "was", "were", "be", "been", "i", "will", "would", "shall",
    "should", "which", "please", "with", "that", "this", "these", "those",
    "it", "its", "task", "answer", "not", "any", "all", "into", "from",
}


def _trigrams(text: str) -> set[str]:
    norm = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return {norm[i:i + 3] for i in range(len(norm) - 2)} if len(norm) >= 3 else {norm}


def _similarity(a: str, b: str) -> float:
    ga, gb = _trigrams(a), _trigrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _content_tokens(texts: list[str]) -> list[str]:
    """Content tokens of the symptom texts (stopwords removed), preserving first-appearance order."""
    seen: list[str] = []
    for text in texts:
        for tok in re.findall(r"[a-z]{3,}", text.lower()):
            if tok not in _STOPWORDS and tok not in seen:
                seen.append(tok)
    return seen


def _nearest_mast(tokens: list[str]) -> tuple[str | None, float]:
    """Maximum overlap ratio against the MAST seeds (content tokens of the English mode names)."""
    best_code, best_ratio = None, 0.0
    for code, m in MAST_MODES.items():
        mode_toks = {t for t in re.findall(r"[a-z]{3,}", m["name"].lower())
                     if t not in _STOPWORDS}
        if not mode_toks or not tokens:
            continue
        ratio = len(set(tokens) & mode_toks) / len(set(tokens))
        if ratio > best_ratio:
            best_code, best_ratio = code, ratio
    return best_code, round(best_ratio, 4)


def _agent_at(bundle, step) -> str:
    """Resolve a novel candidate's agent from the R0 stream: MastLabel
    carries no agent field, so the agent is taken from the event at the
    label's evidence step (``unknown`` when the step is absent or outside
    the stream)."""
    events = bundle.trajectory.events
    if isinstance(step, int) and 0 <= step < len(events):
        return events[step].agent
    return "unknown"


@register
class InducerClassifier(Classifier):
    stage = "classify"
    name = "inducer"

    def run_one(self, bundle, ctx) -> None:
        bundle.put(
            "classify",
            self.name,
            {
                "status": "corpus_scope_required",
                "note": "inducer is a cross-trajectory residual clustering "
                        "algorithm: a single trajectory has no recurrence to "
                        "speak of; run it via the Pipeline (run_corpus is "
                        "automatic)",
            },
        )

    def run_corpus(self, bundles, ctx) -> None:
        support_threshold = int(self.param("support_threshold", 3))
        sim_threshold = float(self.param("sim_threshold", 0.35))
        dedup_threshold = float(self.param("dedup_threshold", 0.8))
        max_proposals = int(self.param("max_proposals", 5))

        candidates: list[dict[str, Any]] = []
        for b in bundles:
            art = b.get("classify", "mast_judge")
            if not isinstance(art, dict):
                raise ValueError(
                    f"{b.trace_id} is missing the classify/mast_judge "
                    "artifact: inducer consumes the judge's novel residual "
                    "labels; configure mast_judge(allow_novel=true) first"
                )
            for lab in art.get("labels", []):
                if lab.get("code") != "novel":
                    continue
                symptom = str(lab.get("symptom") or "").strip()
                if not symptom:
                    continue   # symptom-less novel labels were already judged invalid by mast_judge; defensive skip
                candidates.append({
                    "trace_id": b.trace_id,
                    "step": lab.get("step"),
                    "agent": _agent_at(b, lab.get("step")),
                    "symptom": symptom,
                })

        # ---- Clustering: representative-based greedy aggregation (lexical-similarity gated) ----
        clusters: list[dict[str, Any]] = []
        for cand in candidates:
            for cl in clusters:
                if _similarity(cand["symptom"], cl["representative"]) >= sim_threshold:
                    cl["members"].append(cand)
                    break
            else:
                clusters.append({
                    "representative": cand["symptom"],
                    "members": [cand],
                })

        # ---- One proposal per cluster (support gate + seed dedup) ----
        proposals: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        n = 0
        for cl in clusters:
            if len(cl["members"]) < support_threshold:
                continue
            symptoms = [m["symptom"] for m in cl["members"]]
            tokens = _content_tokens(symptoms)
            freq = Counter(
                tok for s in symptoms for tok in re.findall(r"[a-z]{3,}", s.lower())
                if tok not in _STOPWORDS
            )
            top = [t for t in sorted(tokens, key=lambda t: -freq[t])[:3]]
            kinship_code, kinship_ratio = _nearest_mast(tokens)
            if kinship_ratio >= dedup_threshold:
                dropped.append({
                    "symptom": cl["representative"],
                    "support": len(cl["members"]),
                    "reason": f"lexically too close to seed {kinship_code} ({kinship_ratio})",
                })
                continue
            if len(proposals) >= max_proposals:
                break
            n += 1
            proposals.append({
                "mode_id": f"NM-{n}",
                "name": " ".join(top) or "unclassified residual",
                "definition": (
                    "recurrent unclassified failure mode: "
                    + cl["representative"][:140]
                ),
                "kinship": (
                    {"code": kinship_code, "token_overlap": kinship_ratio}
                    if kinship_code else None
                ),
                "support": len(cl["members"]),
                "status": "proposed",   # never takes effect automatically: human adjudication via atap taxonomy accept
                "evidence_trace_ids": [m["trace_id"] for m in cl["members"]],
                "sample_steps": [m["step"] for m in cl["members"][:6]],
            })

        artifact = {
            "status": "ok",
            "proposals": proposals,
            "dropped": dropped,
            "stats": {
                "n_bundles": len(bundles),
                "n_candidates": len(candidates),
                "n_clusters": len(clusters),
                "cluster_sizes": sorted(
                    (len(c["members"]) for c in clusters), reverse=True
                ),
            },
            "thresholds": {
                "support_threshold": support_threshold,
                "sim_threshold": sim_threshold,
                "dedup_threshold": dedup_threshold,
            },
            "corpus_artifact": True,
            "cost": "free",
            "acceptance": "atap taxonomy accept (human gate; proposals never overwrite the seed vocabulary)",
        }
        for b in bundles:
            b.put("classify", self.name, artifact)
