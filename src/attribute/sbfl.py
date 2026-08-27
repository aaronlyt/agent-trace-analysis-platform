"""L0 SBFL spectrum attribution — FAMAS, arXiv:2509.13782 §4 (the first cross-trajectory-scope algorithm).

Mechanism (aligned formula-by-formula with the paper):
* **spectrum construction** (§4.1): the trajectory suite L is collected by
  repeatedly executing the failing task (the paper replays k=20 times); this
  implementation does not replay — it directly consumes existing trajectories
  of the same task population (sandbox ``generate_corpus`` or external
  multi-run data), grouped by ``meta["task_id"]``; the paper uses LLM
  hierarchical clustering to abstract logs into ⟨agent, action, state⟩
  triples — this implementation's abstraction layer is the R5 deterministic
  action signature ``(agent, action_class, target)`` [adaptation: LLM
  clustering → R5 deterministic signature; mechanism-equivalent with no LLM
  cost];
* **matrices** (§4.2.1): coverage matrix C (trajectories × unique
  signatures), frequency matrix F, outcome vector O, plus agent-level
  counterparts;
* **suspiciousness** (Eqs. 2-7): γ=nc_η/nc_agent (action coverage ratio),
  β=f_η/f_agent (action frequency share), α_τ=1+log_{1/λ}(f_τ,η) (local
  frequency boost), λ-decay coverage counts n_cf^λ/n_cs^λ=Σ λ^(f-1), base
  formula Kulczynski2: ``S(η) = [α·Kul2^λ(η)]·(1+β)·(1+γ)``; λ∈(0.5,1)
  (paper: 0.9);
* **output** (§4.2.4): ranked by S in descending order — the paper takes
  **top-1 as the final attribution** (strict evaluation); only signatures
  appearing in the attributed failing trajectory τ₀ are ranked
  [adaptation: this implementation outputs the top-k ranked list
  (``top_k``=5 by default) as low-confidence hypotheses rather than a single
  top-1 verdict];

Positioning (survey principle): **an L2 prior, not a final verdict** — SBFL
is a statistical signal (the paper reports 57.61 agent / 29.35 action on
Who&When); this implementation outputs low-confidence ranked hypotheses as
reference for L2 deep attribution. Known boundary (inherent to the method):
faults whose action spectrum is identical to successful trajectories (e.g.,
information withholding / ungrounded citation / spec violations — the
difference lies in message content, not in the action sequence) leave no
spectrum signal; groups with a single trajectory / no success reference have a
degenerate spectrum, and the artifact records this explicitly. Further
engineering choices declared only inline (the paper has no counterpart): the
responsible step of a signature repeated within τ₀ is taken at its second
occurrence; environment-side agents (verifier/env) are excluded from the
spectrum; Hypothesis confidence is a fixed prior scalar.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.schema import Hypothesis


def _spectrum_units(artifact: Any) -> list[dict[str, Any]] | None:
    """R5 artifact → sequence of spectrum units (in-trajectory order preserved)."""
    if not isinstance(artifact, dict):
        return None
    sigs = artifact.get("signatures")
    return sigs if isinstance(sigs, list) else None


def _unit_key(sig: dict[str, Any]) -> tuple[str, str, str]:
    return (sig["agent"], sig["action_class"], sig.get("target") or "")


@register
class SBFLAttributor(Attributor):
    stage = "attribute"
    name = "sbfl"

    #: prior-level confidence (below the L1 judge's 0.7 tier) [engineering choice]
    PRIOR_CONFIDENCE = 0.35

    def run_one(self, bundle, ctx) -> None:
        bundle.put(
            "attribute",
            self.name,
            {
                "hypotheses": [],
                "status": "corpus_scope_required",
                "note": "sbfl is a cross-trajectory spectrum algorithm: the "
                        "single-trajectory scope produces no attribution; run "
                        "it via the Pipeline (automatic run_corpus)",
            },
        )

    def run_corpus(self, bundles, ctx) -> None:
        lam = float(self.param("lam", 0.9))
        if not 0.5 < lam < 1.0:
            raise ValueError(f"λ must be in (0.5, 1), got {lam}")
        top_k = int(self.param("top_k", 5))
        # FAMAS attributes agent behavior (agent-action-state triples) —
        # environment-side events (verifier/env) are not agent behavior and
        # are excluded from spectrum units
        exclude_agents = set(
            self.param("exclude_agents", ["verifier", "env"])
        )

        groups: dict[str, list] = {}
        for b in bundles:
            key = str(b.trajectory.meta.get("task_id") or "")
            groups.setdefault(key, []).append(b)

        for key, grp in groups.items():
            if not key:
                for b in grp:
                    self.run_one(b, ctx)
                continue
            units: dict[str, list[dict[str, Any]] | None] = {}
            for b in grp:
                art = b.get("represent", "action_signature")
                sigs = _spectrum_units(art)
                if sigs is None:
                    raise ValueError(
                        f"{b.trace_id} is missing the represent/action_signature "
                        "artifact: sbfl uses R5 action signatures as spectrum "
                        "units; configure action_signature first"
                    )
                units[b.trace_id] = [
                    s for s in sigs if s["agent"] not in exclude_agents
                ]

            coverage: dict[str, set] = {}
            freq: dict[str, Counter] = {}
            agent_cov: dict[str, set] = {}
            agent_freq: dict[str, Counter] = {}
            for b in grp:
                sigs = units[b.trace_id] or []
                cov: set = set()
                fc: Counter = Counter()
                afc: Counter = Counter()
                for s in sigs:
                    k = _unit_key(s)
                    cov.add(k)
                    fc[k] += 1
                    afc[s["agent"]] += 1
                coverage[b.trace_id] = cov
                freq[b.trace_id] = fc
                agent_cov[b.trace_id] = {s["agent"] for s in sigs}
                agent_freq[b.trace_id] = afc

            failed = [b for b in grp if not b.succeeded]
            passed = [b for b in grp if b.succeeded]
            universe: set = set()
            for b in grp:
                universe |= coverage[b.trace_id]

            n_uf_map = {u: 0 for u in universe}
            for u in universe:
                n_uf_map[u] = sum(1 for b in failed if u not in coverage[b.trace_id])

            def decay_count(u, group_bundles) -> float:
                total = 0.0
                for b in group_bundles:
                    f = freq[b.trace_id].get(u, 0)
                    if f > 0:
                        total += lam ** (f - 1)
                return total

            scores: dict[tuple, float] = {}
            details: dict[tuple, dict[str, float]] = {}
            for u in universe:
                n_cf = decay_count(u, failed)
                n_cs = decay_count(u, passed)
                n_uf = n_uf_map[u]
                kul = 0.5 * (
                    (n_cf / (n_cf + n_uf) if n_cf + n_uf > 0 else 0.0)
                    + (n_cf / (n_cf + n_cs) if n_cf + n_cs > 0 else 0.0)
                )
                nc_eta = sum(1 for b in grp if u in coverage[b.trace_id])
                nc_agent = sum(
                    1 for b in grp if u[0] in agent_cov[b.trace_id]
                ) or 1
                f_eta = sum(freq[b.trace_id].get(u, 0) for b in grp)
                f_agent = sum(
                    agent_freq[b.trace_id].get(u[0], 0) for b in grp
                ) or 1
                gamma = nc_eta / nc_agent
                beta = f_eta / f_agent
                scores[u] = kul * (1 + beta) * (1 + gamma)   # α is multiplied per trajectory
                details[u] = {
                    "n_cf_lambda": round(n_cf, 4),
                    "n_cs_lambda": round(n_cs, 4),
                    "n_uf": n_uf,
                    "kulczynski2_lambda": round(kul, 4),
                    "gamma": round(gamma, 4),
                    "beta": round(beta, 4),
                }

            notes: list[str] = []
            if not passed:
                notes.append("no successful trajectory in the group: n_cs^λ=0, the spectrum is degenerate (results are for reference only)")
            if len(grp) == 1:
                notes.append("only 1 trajectory in the group: no repeated-execution variance, the spectrum is degenerate")

            for b in grp:
                if b.succeeded:
                    b.put(
                        "attribute", self.name,
                        {"hypotheses": [], "status": "success_no_attribution",
                         "spectrum_group": key},
                    )
                    continue
                sigs = units[b.trace_id] or []
                ranked: list[tuple[float, tuple, list[int]]] = []
                seen: dict[tuple, list[int]] = {}
                for s in sigs:
                    seen.setdefault(_unit_key(s), []).append(s["index"])
                for u, idxs in seen.items():
                    f_tau = len(idxs)
                    alpha = 1 + math.log(f_tau) / math.log(1 / lam)
                    s_score = alpha * scores[u]
                    ranked.append((s_score, u, idxs))
                ranked.sort(key=lambda r: -r[0])
                # responsible step: when the signature repeats ≥2 times within
                # the trajectory, take the second occurrence — for
                # repetition-type faults the earliest decisive error is the
                # **first repetition** (the second occurrence); the first
                # execution itself is legitimate, and correcting it would not
                # flip the failure (Who&When Eq.5 counterfactual semantics);
                # when it occurs only once, take the first occurrence
                # [engineering choice: the paper has no such convention]
                hyps = [
                    Hypothesis(
                        agent=u[0],
                        step=idxs[1] if len(idxs) >= 2 else idxs[0],
                        root_cause=(
                            f"SBFL spectrum prior: the action {u[1]}({u[2] or '-'}) "
                            f"concentrates in failing runs ({len(idxs)} times in this "
                            "trajectory) and is rare in successful runs -- statistically "
                            "suspicious; treat as an L2 prior, not a final verdict"
                        ),
                        root_cause_code=None,
                        responsible_side="model",
                        evidence=[
                            f"signature={u[1]}({u[2] or '-'}) agent={u[0]} "
                            f"steps={idxs[:6]}",
                            f"metrics={details[u]}",
                        ],
                        fix_suggestion=(
                            f"Re-examine whether {u[0]}'s {u[1]}({u[2] or '-'}) action "
                            "sequence constitutes the decisive error (handing off to "
                            "L2 deep attribution for confirmation is recommended)."
                        ),
                        confidence=self.PRIOR_CONFIDENCE,
                    )
                    for s_score, u, idxs in ranked[:top_k]
                ]
                b.put(
                    "attribute",
                    self.name,
                    {
                        "hypotheses": [h.to_dict() for h in hyps],
                        "role": "L0_statistical_prior",
                        "spectrum": {
                            "group": key,
                            "n_runs": len(grp),
                            "n_failed": len(failed),
                            "n_success": len(passed),
                            "lam": lam,
                            "top": [
                                {
                                    "signature": f"{u[1]}({u[2] or '-'})",
                                    "agent": u[0],
                                    "score": round(sc, 4),
                                    **details[u],
                                }
                                for sc, u, _ in ranked[:top_k]
                            ],
                            "notes": notes,
                        },
                    },
                )
