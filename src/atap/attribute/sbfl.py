"""L0 SBFL 频谱归因 —— FAMAS, arXiv:2509.13782 §4（首个跨轨迹作用域算法）。

机制（原文公式逐条对齐）：
* **频谱构造**（§4.1）：对失败任务重复执行收集轨迹套件 L（原文重放
  k=20 次）；本实现不重放——直接消费同任务群体的既有轨迹（沙盒
  ``generate_corpus`` 或外部多次执行数据），按 ``meta["task_id"]`` 分组；
  原文用 LLM 层次聚类把日志抽象为 ⟨agent, action, state⟩ 三元组——本
  实现的抽象层是 R5 确定性动作签名 ``(agent, action_class, target)``
  【适配：LLM 聚类 → R5 确定性签名，机制等价、无 LLM 成本】；
* **矩阵**（§4.2.1）：覆盖矩阵 C（轨迹 × 唯一签名）、频次矩阵 F、
  结果向量 O，及 agent 级同类矩阵；
* **可疑度**（式 2-7）：γ=nc_η/nc_agent（动作覆盖比）、β=f_η/f_agent
  （动作频率占比）、α_τ=1+log_{1/λ}(f_τ,η)（局部频率增强）、
  λ-decay 覆盖计数 n_cf^λ/n_cs^λ=Σ λ^(f-1)，基公式 Kulczynski2：
  ``S(η) = [α·Kul2^λ(η)]·(1+β)·(1+γ)``；λ∈(0.5,1)（原文 0.9）；
* **输出**（§4.2.4）：按 S 降序排名，**top-1 为最终归因**（严格评测）；
  只排名出现在被归因失败轨迹 τ₀ 中的签名。

定位（综述原则）：**作 L2 先验而非终判**——SBFL 是统计信号（原文在
Who&When 上 agent 57.61/action 29.35），本实现输出低置信度 ranked
hypotheses 供 L2 深度归因参考。已知边界（方法本性）：与成功轨迹动作谱
完全相同的故障（如信息隐瞒/无据引用/违反规格——差异在消息内容而非
动作序列）无频谱信号；单轨迹/无成功参照的分组频谱退化，产物显式留痕。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.schema import Hypothesis


def _spectrum_units(artifact: Any) -> list[dict[str, Any]] | None:
    """R5 产物 → 频谱单元序列（保持轨迹内次序）。"""
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

    #: 先验级置信度（低于 L1 判官的 0.7 量级）【工程选择】
    PRIOR_CONFIDENCE = 0.35

    def run_one(self, bundle, ctx) -> None:
        bundle.put(
            "attribute",
            self.name,
            {
                "hypotheses": [],
                "status": "corpus_scope_required",
                "note": "sbfl 是跨轨迹频谱算法：单轨迹作用域不产出归因，"
                        "请经 Pipeline（自动 run_corpus）运行",
            },
        )

    def run_corpus(self, bundles, ctx) -> None:
        lam = float(self.param("lam", 0.9))
        if not 0.5 < lam < 1.0:
            raise ValueError(f"λ 必须在 (0.5, 1)，得到 {lam}")
        top_k = int(self.param("top_k", 5))
        # FAMAS 归因的是 agent 行为（agent-action-state 三元组）——环境侧
        # 事件（verifier/env）不是 agent 行为，排除出频谱单元
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
                        f"{b.trace_id} 缺少 represent/action_signature 产物："
                        "sbfl 以 R5 动作签名为频谱单元，请先配置 action_signature"
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
                scores[u] = kul * (1 + beta) * (1 + gamma)   # α 逐轨迹乘
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
                notes.append("组内无成功轨迹：n_cs^λ=0，频谱退化（结果仅供参考）")
            if len(grp) == 1:
                notes.append("组内仅 1 条轨迹：无重复执行变异，频谱退化")

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
                # 责任步：签名在轨迹内重复 ≥2 次时取第二次出现（"第二次
                # 出现即首次重复"，对齐 Who&When Eq.5 最早决定性错误约定），
                # 否则取首次出现【工程选择】
                hyps = [
                    Hypothesis(
                        agent=u[0],
                        step=idxs[1] if len(idxs) >= 2 else idxs[0],
                        root_cause=(
                            f"SBFL 频谱先验：动作 {u[1]}({u[2] or '-'}) 在失败运行中"
                            f"集中出现（本轨迹 {len(idxs)} 次）、成功运行中罕见——"
                            "统计可疑度高，作 L2 先验而非终判"
                        ),
                        root_cause_code=None,
                        responsible_side="model",
                        evidence=[
                            f"signature={u[1]}({u[2] or '-'}) agent={u[0]} "
                            f"steps={idxs[:6]}",
                            f"metrics={details[u]}",
                        ],
                        fix_suggestion=(
                            f"复核 {u[0]} 的 {u[1]}({u[2] or '-'}) 动作序列是否"
                            "构成决定性错误（建议交 L2 深度归因确认）。"
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
