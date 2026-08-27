"""L0 RG/UG deterministic attribution — search-agent diagnostics, arXiv:2608.01913 §4.3.

Mechanism (aligned equation-by-equation with the paper; attribution entirely
LLM-free — "decided by qrels rather than an LLM judge"):
* **two-tier qrels**: E(q) = topic-relevant documents, G(q)⊆E = a sufficient
  set for deriving gold (known by construction in this sandbox, carried via
  ``meta["qrels"]`` — a data dependency, not a code dependency);
* **episode segmentation**: each episode runs from one search call up to the
  next search; reads issued **before the first search** open an implicit
  leading episode with no search anchor (flagged ``implicit`` in the
  artifact, ``start_index`` = the opening read call) [adaptation] —
  otherwise such reads would enter visit_precision's denominator yet belong
  to no R_k, i.e. they would never reach C_M (two inconsistent scopes);
* **set operations**: R_k = doc ids returned by search/visit within episode k
  (exact match against E, "regardless of surface relatedness"); C_k=∪_{j≤k}R_j;
  Δ_k=C_k∖C_{k−1}; G*=C_M∩G(q); first gold hit = min{k:G_k*≠∅} (artifact
  note: the ``first_gold_hit`` field stores the R0 **event index** of the
  search call that opens the first gold-hitting episode — or, when that
  episode is the implicit pre-search one, the index of its opening read call
  — not the episode number; ``k_star`` is likewise 0-based, while
  ``wasted_tail`` is already converted to the paper's 1-based M−k*);
* **decision rules**: success → correct; failure ∧ G*=∅ → **RG**
  (C_M∩E=∅ → directional, otherwise last-hop); failure ∧ G*≠∅ → **UG**
  (G*=G → true-extraction, otherwise boundary);
* **episode utility**: productive (Δ≠∅) / redundant (R≠∅∧Δ=∅) /
  unproductive (R=∅); k* = last productive step, wasted tail = M−k*; visit
  precision = share of reads that are ∈E;
* **prescriptions** (from the paper): RG → switch query formulation; UG →
  stop at saturation.

Hypothesis mapping [adaptation: the paper's output unit is a trajectory-level
binary label + episode utilities, with no agent/step localization] — RG →
(searcher, the first search call step; if there is no search at all, the first
LLM_CALL decision step), root_cause_code=retrieval_gap; UG → (the first
"contradictory decision" event after gold surfaces: falsely claiming no
results / assertively citing unread documents; if no contradiction, the last
LLM_CALL before submit), code=utilization_gap. "Contradictory decision"
detection uses keyword patterns (_NO_RESULT_RE/_ASSERTION_RE) — [inference:
the paper has no contradictory-claim detection rule (wrong-answer
classification uses text similarity only); the keyword patterns are a sandbox
customization]. The RG attribution step takes the first search rather than the
design-contract's "start of the first consecutive unproductive episode" — in
detour scenarios (productive episodes exist yet gold is missing) the latter is
not executable, so taking the first search is more robust [adaptation:
deviation from the contract in plan_stage_four.md]. A step_repetition-type UG
maps onto the compose step rather than the first repeated step (the repeated
calls were never "utilized") — recorded as a mapping boundary; the
trajectory-level label is unaffected. confidence=1.0 carries two layers
[declared]: the trajectory-level RG/UG label is qrels-decided and
deterministic — the 1.0 expresses **that** layer; the agent/step mapping
onto R0 events is a keyword heuristic (first-search / contradiction
patterns) and may misplace, so the scalar must not be read as
step-localization certainty. UG fallback boundary [declared]: when gold has
surfaced but no LLM_CALL follows it (and no contradiction fired), the
fallback target is the last event whose agent != "env" (such a trace's
utilization failure still has a non-environment actor); only an all-env
trace falls back to events[0] — TASK_START is env-owned, mirroring the
declared s*=0 boundary of binary_search's walk-back. The success decision uses the sandbox
verifier result (trajectory outcome) in place of the paper's GPT-4o binary
answer judgment [adaptation: determinism].
"""

from __future__ import annotations

import re
from typing import Any

from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.schema import Hypothesis

_SEARCH_DOCS_RE = re.compile(r"docs\s*\[([^\]]*)\]")
_DOC_ID_RE = re.compile(r"\bd\d+\b")
_NO_RESULT_RE = re.compile(
    r"no (relevant|results|documents)|nothing found", re.I
)
_ASSERTION_RE = re.compile(r"cite|cited|according to|based on", re.I)


def _parse_docs(content: str) -> list[str]:
    m = _SEARCH_DOCS_RE.search(content)
    if not m:
        return []
    return [d.strip() for d in m.group(1).split(",") if d.strip()]


@register
class RGUGAttributor(Attributor):
    stage = "attribute"
    name = "rg_ug"

    def run_one(self, bundle, ctx) -> None:
        t = bundle.trajectory
        events = t.events
        if not events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        qrels = t.meta.get("qrels")
        if not (isinstance(qrels, dict) and qrels.get("evidence") is not None
                and qrels.get("gold") is not None):
            raise ValueError(
                f"{bundle.trace_id} is missing meta['qrels'] (two-tier "
                "evidence/gold document annotations): rg_ug is qrels-driven "
                "deterministic attribution, provided natively by sandbox "
                "trajectories; for external trajectories, inject the "
                "annotation at the collection layer"
            )
        evidence = set(qrels["evidence"])
        gold = set(qrels["gold"])

        # ---- episode segmentation and R_k collection (search boundaries) ----
        episodes: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        reads: list[tuple[int, str]] = []          # (index, doc_id)
        gold_hit_index: int | None = None
        for ev in events:
            if ev.kind == "TOOL_CALL" and (ev.action or "") == "search":
                current = {
                    "k": len(episodes), "start_index": ev.index,
                    "agent": ev.agent, "docs": [], "events": [ev],
                }
                episodes.append(current)
                continue
            if (
                current is None and not episodes
                and ev.kind == "TOOL_CALL" and (ev.action or "") == "read_doc"
                and _DOC_ID_RE.fullmatch(str(ev.payload.get("doc_id", "")))
            ):
                # implicit pre-search episode: a read before the first search
                # belongs to no search-anchored episode, yet it still enters
                # visit_precision's denominator — open a leading episode with
                # no search anchor so the read also enters C_M [adaptation,
                # declared in the module docstring]
                current = {
                    "k": len(episodes), "start_index": ev.index,
                    "agent": ev.agent, "docs": [], "events": [],
                    "implicit": True,
                }
                episodes.append(current)
            if current is not None:
                current["events"].append(ev)
            if ev.kind == "TOOL_RESULT" and (ev.action or "") == "search":
                if current is not None:
                    current["docs"] += _parse_docs(str(ev.payload.get("content", "")))
            elif ev.kind == "TOOL_CALL" and (ev.action or "") == "read_doc":
                did = str(ev.payload.get("doc_id", ""))
                if _DOC_ID_RE.fullmatch(did):
                    reads.append((ev.index, did))
                    if current is not None:
                        current["docs"].append(did)

        for ep in episodes:
            ep["R"] = sorted(set(ep["docs"]) & evidence)
            del ep["docs"]
            del ep["events"]

        cumulative: set[str] = set()
        first_gold_hit: int | None = None
        for ep in episodes:
            delta = set(ep["R"]) - cumulative
            ep["delta"] = sorted(delta)
            ep["utility"] = (
                "productive" if delta else
                ("redundant" if ep["R"] else "unproductive")
            )
            cumulative |= set(ep["R"])
            if first_gold_hit is None and (set(ep["R"]) & gold):
                first_gold_hit = ep["start_index"]
        c_m = cumulative
        g_star = c_m & gold

        # ---- decision rules (from the paper) ----
        if t.outcome.success:
            label = "correct"
        elif not g_star:
            label = "RG_directional" if not (c_m & evidence) else "RG_last_hop"
        else:
            label = "UG_true_extraction" if g_star == gold else "UG_boundary"

        productive_ks = [ep["k"] for ep in episodes if ep["delta"]]
        k_star = productive_ks[-1] if productive_ks else None
        wasted_tail = (len(episodes) - 1 - k_star) if k_star is not None else len(episodes)
        visit_precision = (
            round(sum(1 for _, d in reads if d in evidence) / len(reads), 4)
            if reads else None
        )

        artifact: dict[str, Any] = {
            "label": label,
            "M": len(episodes),
            "C_M": sorted(c_m),
            "G_star": sorted(g_star),
            "qrels": {"evidence": sorted(evidence), "gold": sorted(gold)},
            "episodes": episodes,
            "first_gold_hit": first_gold_hit,
            "k_star": k_star,
            "wasted_tail": wasted_tail,
            "visit_precision": visit_precision,
            "prescription": (
                "switch query formulation" if label.startswith("RG")
                else "stop at saturation / enforce grounded citation"
                if label.startswith("UG") else None
            ),
            "cost": "free",
            "role": "L0_deterministic",
        }

        if label == "correct":
            artifact["hypotheses"] = []
            artifact["status"] = "success_no_attribution"
            bundle.put("attribute", self.name, artifact)
            return

        hyp = self._hypothesis(events, label, c_m, g_star, gold, first_gold_hit, reads)
        artifact["hypotheses"] = [hyp.to_dict()]
        bundle.put("attribute", self.name, artifact)

    # ------------------------------------------------------------------

    def _hypothesis(
        self, events, label: str, c_m: set, g_star: set, gold: set,
        first_gold_hit: int | None, reads: list[tuple[int, str]],
    ) -> Hypothesis:
        if label.startswith("RG"):
            searches = [
                e for e in events
                if e.kind == "TOOL_CALL" and (e.action or "") == "search"
            ]
            if searches:
                target = searches[0]
            else:
                # no search at all (premature-termination-like): attribute to
                # the first decision step that chose to skip search
                decisions = [e for e in events if e.kind == "LLM_CALL"]
                target = decisions[0] if decisions else events[0]
            sub = "directional" if label == "RG_directional" else "last-hop"
            return Hypothesis(
                agent=target.agent,
                step=target.index,
                root_cause=(
                    f"Retrieval gap (RG-{sub}): gold evidence {sorted(gold)} never "
                    f"surfaced in any search or read (C_M={sorted(c_m)}) — the "
                    "necessary evidence was never found"
                ),
                root_cause_code="retrieval_gap",
                responsible_side="model",
                evidence=[
                    # render the target event's actual kind — the no-search
                    # fallback targets an LLM_CALL decision, not a search
                    f"[{target.index}] {target.agent} {target.kind} "
                    f"{dict(target.payload)}",
                    f"C_M={sorted(c_m)} G*={sorted(g_star)}",
                ],
                fix_suggestion="switch query formulation (reformulate the query and retry the search)",
                confidence=1.0,
            )

        # ---- UG: the first contradictory decision after gold surfaces ----
        read_by_then: set[str] = set()
        fallback: Any = None
        for ev in events:
            if ev.kind == "TOOL_CALL" and (ev.action or "") == "read_doc":
                read_by_then.add(str(ev.payload.get("doc_id", "")))
                continue
            if first_gold_hit is not None and ev.index <= first_gold_hit:
                continue
            if ev.kind == "LLM_CALL" and (ev.action or "") != "search":
                fallback = ev   # track the last decision step before submit
            if ev.kind not in ("HANDOFF", "AGENT_MESSAGE", "LLM_CALL"):
                continue
            if (ev.action or "") == "search":
                continue
            content = str(ev.payload.get("content", ""))
            # contradiction 1: gold has surfaced yet the agent claims no
            # results (information-withholding-type UG)
            if _NO_RESULT_RE.search(content):
                return self._ug_hypothesis(
                    ev, g_star, gold, "falsely claimed no results were found", content
                )
            # contradiction 2: assertively citing documents never read
            # (ungrounded-citation-type UG)
            mentions = set(_DOC_ID_RE.findall(content))
            if mentions and _ASSERTION_RE.search(content):
                unread = {d for d in mentions if d not in read_by_then}
                if unread:
                    return self._ug_hypothesis(
                        ev, g_star, gold,
                        f"assertively cited unread documents {sorted(unread)}", content,
                    )
        if fallback is None:
            # no decision step after gold surfaced: prefer the last
            # non-environment *acting* event (declared boundary, module
            # docstring) — VERIFIER rows are environment-side feedback even
            # when their agent column is populated; only an all-env trace
            # falls back to events[0] (TASK_START is env-owned — mirrors
            # binary_search's declared s*=0 boundary)
            fallback = next(
                (
                    e for e in reversed(events)
                    if e.agent != "env" and e.kind != "VERIFIER"
                ),
                events[0],
            )
        return self._ug_hypothesis(
            fallback, g_star, gold,
            "gold evidence was retrieved but never properly utilized", "",
        )

    @staticmethod
    def _ug_hypothesis(ev, g_star, gold, symptom: str, content: str) -> Hypothesis:
        sub = "true-extraction" if g_star == gold else "boundary"
        return Hypothesis(
            agent=ev.agent,
            step=ev.index,
            root_cause=(
                f"Utilization gap (UG-{sub}): gold evidence {sorted(g_star)} was "
                f"retrieved but never properly used — {symptom}"
            ),
            root_cause_code="utilization_gap",
            responsible_side="model",
            evidence=[
                f"[{ev.index}] {ev.agent} {ev.kind} :: {content[:120]}",
                f"G*={sorted(g_star)} ⊆ gold {sorted(gold)}",
            ],
            fix_suggestion="stop at saturation / enforce grounded citation"
                           " (stop once saturated; enforce citations grounded in retrieved evidence)",
            confidence=1.0,
        )
