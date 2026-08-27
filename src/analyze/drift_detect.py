"""Distribution drift detection —— system-level taxonomy, arXiv:2511.19933 §V definitions / §VI signals.

**Fidelity boundary**: that paper is a position paper —— it gives the
definitions of three drift types (§V: version drift = workflow changes
caused by model updates; data drift = input-distribution deviation;
behavior drift = output drift for the same prompt across time windows)
and a checklist of monitoring signals (§VI: output variance / format
changes / longitudinal sampling of behavioral metrics), but **no
algorithm, formulas, or thresholds** (detection is listed as §VII future
work; PSI is never mentioned in the paper). The statistical part of this
implementation is entirely
[adaptation: engineering choice —— the paper provides no detection
algorithm; PSI is this project's self-selected implementation]:

* grouping key = (model_version, prompt_version, time_window) from meta
  (the schema reserves this convention);
* behavioral features (distributions aggregated across trajectories):
  event-kind histogram, agent×action histogram, trajectory-length bins,
  failure rate, adjacent-duplicate-call rate (R0-observable, zero LLM);
* test = PSI over discrete distributions restricted to the **shared
  support** (bins with nonzero mass in both groups). ε-smoothing is
  deliberately NOT applied to empty bins: an ε-smoothed empty bin
  contributes ≈ ln(1/ε)·mass to PSI —— an artifact of the smoothing
  constant, not an effect size (with ε=1e-4 the demo's version-contrast
  PSI was 11.46, of which 8.67 came from one empty bin; ε=1e-6 would
  have made it 15.8). Bins empty on one side are instead reported as
  ``support_mismatch``: each side's probability mass sitting in bins the
  other group never visits —— ε-free and directly interpretable
  ("94.7% of group A's mass is in bins group B never hits"). An alert
  fires if any feature's PSI > ``psi_alert`` (default 0.2) or mismatch >
  ``mismatch_alert`` (default 0.1);
* guards against small-sample / duplicate-signal inflation: pairs where
  either group has < ``min_group_size`` (default 5) traces are skipped
  (recorded in ``skipped``); per-trajectory categorical features that
  partition the corpus identically (e.g. length_bin ≡ failure when every
  long trace fails and every short one succeeds) are reported as
  ``feature_aliases`` and deduplicated in ``firing_features_independent``
  —— they are one signal measured twice, not two signals;
* three contrast families (confounder control): **version** = bucket by
  model version under the same prompt and compare behavior; **behavior**
  = bucket by time window under the same (model,prompt) and compare
  behavior; **data** = bucket by time window under the same
  (model,prompt) and compare task composition.

Artifact (``analyze/drift_detect``, corpus-level, same content written
back to every bundle): ``{groups, pairwise, alerts, skipped,
feature_aliases, family_definitions}``. Deterministic: same corpus,
same result.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from atap.analyze.base import Analyzer
from atap.core.registry import register

_LENGTH_BINS = ((0, 9), (10, 13), (14, 17), (18, float("inf")))

# per-trajectory categorical features whose histograms feed the PSI
# comparisons; bijective pairs among these are aliases (one signal)
_CATEGORICAL_FEATURES = ("length_bin", "failure", "task_id")


def _length_bin(n: int) -> str:
    for lo, hi in _LENGTH_BINS:
        if lo <= n <= hi:
            return f"len[{lo}-{hi if hi != float('inf') else '+'}]"
    return f"len[{_LENGTH_BINS[-1][0]}-+]"


def _positive(d: dict[str, float]) -> dict[str, float]:
    """Drop zero-mass entries (binary-feature dicts keep their keys even
    when a side's rate is exactly 0 or 1)."""
    return {k: v for k, v in d.items() if v > 0}


def _psi(p: dict[str, float], q: dict[str, float]) -> float:
    """PSI over the shared support only: Σ(p_i−q_i)·ln(p_i/q_i) for bins
    with nonzero mass in both distributions.

    Empty bins are excluded here (an ε-smoothed empty bin would add
    ≈ ln(1/ε)·mass —— a smoothing-constant artifact); their mass is
    quantified separately by :func:`_support_mismatch`. With no empty
    bins this equals classic PSI exactly.
    """
    p, q = _positive(p), _positive(q)
    total = 0.0
    for k in sorted(set(p) & set(q)):
        total += (q[k] - p[k]) * math.log(q[k] / p[k])
    return total


def _support_mismatch(
    p: dict[str, float], q: dict[str, float]
) -> tuple[float, float]:
    """(p-side, q-side) probability mass outside the shared support."""
    p, q = _positive(p), _positive(q)
    shared = set(p) & set(q)
    return (
        sum(v for k, v in p.items() if k not in shared),
        sum(v for k, v in q.items() if k not in shared),
    )


def _to_probs(counts: Counter) -> dict[str, float]:
    n = sum(counts.values())
    return {k: v / n for k, v in counts.items()} if n else {}


def _features(t) -> dict[str, Any]:
    """R0-observable features of a single trajectory (zero LLM)."""
    events = t.events
    kind_hist = Counter(ev.kind for ev in events)
    action_hist = Counter(f"{ev.agent}:{ev.action or ev.kind}" for ev in events)
    calls = [ev for ev in events if ev.kind == "TOOL_CALL"]
    n_repeat = sum(
        1 for a, b in zip(calls, calls[1:])
        if (a.agent, a.action, dict(a.payload)) == (b.agent, b.action, dict(b.payload))
    )
    return {
        "kind_hist": kind_hist,
        "action_hist": action_hist,
        "length_bin": _length_bin(len(events)),
        "n_events": len(events),
        "failed": 0 if t.outcome.success else 1,
        "n_calls": len(calls),
        "n_repeat_calls": n_repeat,
        "task_id": str(t.meta.get("task_id") or "unknown"),
    }


_BEHAVIORAL_DIMS = ("kind_hist", "action_hist", "length_bin", "failure", "repeat")


def _alias_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """Categorical feature pairs that induce identical trace partitions.

    If the value map is bijective in both directions (every value of one
    feature co-occurs with exactly one value of the other and vice
    versa), the two features' histograms are renamings of each other, so
    every PSI / mismatch comparison on them is guaranteed identical ——
    counting both would double-count one signal (in the drift corpus,
    trajectory length and success/failure are perfectly collinear by
    construction).
    """
    values = {
        f: [("1" if r["failed"] else "0") if f == "failure" else str(r[f])
            for r in rows]
        for f in _CATEGORICAL_FEATURES
    }

    def bijective(f: str, g: str) -> bool:
        fg: dict[str, set[str]] = {}
        gf: dict[str, set[str]] = {}
        for fv, gv in zip(values[f], values[g]):
            fg.setdefault(fv, set()).add(gv)
            gf.setdefault(gv, set()).add(fv)
        return all(len(s) == 1 for s in fg.values()) and \
            all(len(s) == 1 for s in gf.values())

    return [
        (f, g)
        for i, f in enumerate(_CATEGORICAL_FEATURES)
        for g in _CATEGORICAL_FEATURES[i + 1:]
        if bijective(f, g)
    ]


@register
class DriftDetectAnalyzer(Analyzer):
    stage = "analyze"
    name = "drift_detect"

    def run_one(self, bundle, ctx) -> None:
        bundle.put(
            "analyze",
            self.name,
            {
                "status": "corpus_scope_required",
                "note": "drift_detect is a cross-trajectory distribution "
                        "algorithm: a single trajectory offers no "
                        "distribution to compare; run it via the Pipeline "
                        "(run_corpus is automatic)",
            },
        )

    def run_corpus(self, bundles, ctx) -> None:
        psi_alert = float(self.param("psi_alert", 0.2))
        mismatch_alert = float(self.param("mismatch_alert", 0.1))
        min_group_size = int(self.param("min_group_size", 5))

        rows = []
        for b in bundles:
            t = b.trajectory
            rows.append({
                "trace_id": b.trace_id,
                "model_version": str(t.meta.get("model_version", "unknown")),
                "prompt_version": str(t.meta.get("prompt_version", "unknown")),
                "time_window": str(t.meta.get("time_window", "unknown")),
                **_features(t),
            })

        aliases = _alias_pairs(rows)

        def bucket_agg(items: list[dict]) -> dict[str, Counter | float]:
            agg: dict[str, Any] = {
                "kind_hist": Counter(),
                "action_hist": Counter(),
                "length_bin": Counter(),
                "task_id": Counter(),
                "n_failed": 0,
                "n_repeat": 0,
                "n_calls": 0,
            }
            for r in items:
                agg["kind_hist"] += r["kind_hist"]
                agg["action_hist"] += r["action_hist"]
                agg["length_bin"][r["length_bin"]] += 1
                agg["task_id"][r["task_id"]] += 1
                agg["n_failed"] += r["failed"]
                agg["n_repeat"] += r["n_repeat_calls"]
                agg["n_calls"] += r["n_calls"]
            return agg

        def behavioral_dists(a: dict, b: dict) -> dict[str, tuple[dict, dict]]:
            n_a = sum(a["length_bin"].values())
            n_b = sum(b["length_bin"].values())
            return {
                "kind_hist": (_to_probs(a["kind_hist"]), _to_probs(b["kind_hist"])),
                "action_hist": (_to_probs(a["action_hist"]), _to_probs(b["action_hist"])),
                "length_bin": (_to_probs(a["length_bin"]), _to_probs(b["length_bin"])),
                "failure": (
                    {"fail": a["n_failed"] / n_a, "ok": 1 - a["n_failed"] / n_a},
                    {"fail": b["n_failed"] / n_b, "ok": 1 - b["n_failed"] / n_b},
                ),
                "repeat": (
                    {"repeat": a["n_repeat"] / max(a["n_calls"], 1),
                     "clean": 1 - a["n_repeat"] / max(a["n_calls"], 1)},
                    {"repeat": b["n_repeat"] / max(b["n_calls"], 1),
                     "clean": 1 - b["n_repeat"] / max(b["n_calls"], 1)},
                ),
            }

        pairwise: list[dict[str, Any]] = []
        alerts: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        def compare_family(
            family: str,
            partitions: list[tuple[str, list[dict]]],
            dists_fn,
        ) -> None:
            too_small = {k: len(g) for k, g in partitions if len(g) < min_group_size}
            for k, n in sorted(too_small.items()):
                skipped.append({
                    "family": family,
                    "group": k,
                    "n_traces": n,
                    "reason": f"below min_group_size={min_group_size}",
                })
            eligible = [(k, g) for k, g in partitions if len(g) >= min_group_size]
            for i in range(len(eligible)):
                for j in range(i + 1, len(eligible)):
                    (ka, ga), (kb, gb) = eligible[i], eligible[j]
                    psis: dict[str, float] = {}
                    mismatches: dict[str, dict[str, float]] = {}
                    firing: list[str] = []
                    for feat, (p, q) in dists_fn(bucket_agg(ga), bucket_agg(gb)).items():
                        m_a, m_b = _support_mismatch(p, q)
                        psi = _psi(p, q)
                        psis[feat] = round(psi, 4)
                        mismatches[feat] = {"a": round(m_a, 4), "b": round(m_b, 4)}
                        if psi > psi_alert or max(m_a, m_b) > mismatch_alert:
                            firing.append(feat)
                    firing_indep = list(firing)
                    for f, g in aliases:
                        # drop the later alias of any pair that fired twice
                        if f in firing_indep and g in firing_indep:
                            firing_indep.remove(g)
                    entry = {
                        "family": family,
                        "a": ka,
                        "b": kb,
                        "n_a": len(ga),
                        "n_b": len(gb),
                        "psi": psis,
                        "mismatch": mismatches,
                        "max_psi": round(max(psis.values()), 4),
                        "max_mismatch": round(
                            max(max(m.values()) for m in mismatches.values()), 4
                        ),
                        "firing_features": firing,
                        "firing_features_independent": firing_indep,
                        "drift_type": family,
                        "alert": bool(firing),
                    }
                    pairwise.append(entry)
                    if entry["alert"]:
                        alerts.append(entry)

        def partition(key: str, fixed: dict[str, set]) -> list[tuple[str, list[dict]]]:
            buckets: dict[str, list[dict]] = {}
            for r in rows:
                if any(r[k] not in allowed for k, allowed in fixed.items()):
                    continue
                buckets.setdefault(r[key], []).append(r)
            return sorted(buckets.items())

        prompts = {r["prompt_version"] for r in rows}
        for p in sorted(prompts):
            compare_family(
                "version",
                partition("model_version", {"prompt_version": {p}}),
                behavioral_dists,
            )
        for mp in sorted({(r["model_version"], r["prompt_version"]) for r in rows}):
            fixed = {"model_version": {mp[0]}, "prompt_version": {mp[1]}}
            compare_family("behavior", partition("time_window", fixed), behavioral_dists)
            compare_family(
                "data",
                partition("time_window", fixed),
                lambda a, b: {
                    "task_composition": (
                        _to_probs(a["task_id"]), _to_probs(b["task_id"])
                    )
                },
            )

        groups = [
            {
                "model_version": mv,
                "prompt_version": pv,
                "time_window": tw,
                "n_traces": n,
            }
            for (mv, pv, tw), n in sorted(Counter(
                (r["model_version"], r["prompt_version"], r["time_window"])
                for r in rows
            ).items())
        ]
        artifact = {
            "status": "ok",
            "groups": groups,
            "n_traces": len(rows),
            "pairwise": pairwise,
            "alerts": alerts,
            "skipped": skipped,
            "psi_alert": psi_alert,
            "mismatch_alert": mismatch_alert,
            "min_group_size": min_group_size,
            "feature_aliases": [
                {"features": [f, g],
                 "note": "identical trace partitions: one signal measured twice"}
                for f, g in aliases
            ],
            "cost": "free",
            "corpus_artifact": True,
            "family_definitions": {
                "version": "bucket by model version under the same prompt and compare behavioral distributions",
                "behavior": "bucket by time window under the same (model,prompt) and compare behavioral distributions",
                "data": "bucket by time window under the same (model,prompt) and compare task composition"
                        " ([inference: the paper's data drift = inputs deviating from the training/validation"
                        " distribution, proxied here by task_id composition])",
            },
        }
        for b in bundles:
            b.put("analyze", self.name, artifact)
