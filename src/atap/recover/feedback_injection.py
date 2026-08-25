"""归因反馈注入再求解 —— AgenTracer, arXiv:2509.03312 §5.3（ICLR'26）。

机制（原文）：MAS 完成一轮求解产出失败轨迹 τ → tracer 对 τ 生成反思
反馈（AgenTracer-8B 取其 ⟨think⟩ 推理段）→ 反馈注入 M 的**下一轮完整
求解**（全新 episode，不保留前缀——与 targeted_rerun 的前缀保留重跑
正交）→ 迭代 3 轮。原文数字：3 轮 +4.8~14.2%（MaAS/OWL/MetaGPT ×
GAIA/MATH-500/HumanEval+），自反思基线 CRITIC 反降 4.9~5.5%。

与原文的差异：
* 反馈来源：原文 = 微调 tracer 的推理段；本实现第 1 轮反馈取**归因
  Hypothesis**（root_cause + fix_suggestion 的反思文本——归因输出的
  反思化），后续轮用判官反思调用对最新失败轨迹再生成【适配】；
* 注入位置：原文未指明（prompt/历史皆可）；本实现反馈文本整体交给
  ``RunContext.env.resolve(trajectory, feedback)``，由环境决定注入方式
  【声明】；
* 恢复闸门：原文无；AgentDebugX 的 suggest-only 语义由 targeted_rerun
  一侧承载，本算法同样只产重跑轨迹不自动改系统。

消费统一归因契约（top Hypothesis，(confidence, -step) 与 targeted_rerun
同序）；每轮从**原轨迹**的故障状态重解（rerun 轨迹 meta 已剥离
injected_fault，链式传入会虚假成功——见 sandbox.resolve 约定）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from atap.core.registry import register
from atap.core.render import judge_view
from atap.recover.base import Recoverer


class Reflection(BaseModel):
    feedback: str = Field(description="注入下一轮求解的反思反馈（简明、可执行）")


_REFLECT_SYSTEM = (
    "你是失败轨迹的反思反馈生成器（归因反馈注入风格）。给定一条失败的"
    "求解轨迹，产出一小段反思反馈供下一轮求解前注入：指明决定性错误在"
    "哪一步、错在哪、下一轮应如何避免。只基于轨迹可观测证据，简明可执行。"
)


@register
class FeedbackInjectionRecoverer(Recoverer):
    stage = "recover"
    name = "feedback_injection"

    def run_one(self, bundle, ctx) -> None:
        if bundle.succeeded:
            return
        hyps = bundle.hypotheses()
        if not hyps:
            bundle.put(
                "recover", self.name,
                {"status": "skipped_no_hypothesis", "recovered": False,
                 "note": "失败轨迹无归因输出：恢复必须消费归因（文献 §7 断裂警示）"},
            )
            return
        env = getattr(ctx, "env", None)
        if env is None or not hasattr(env, "resolve"):
            bundle.put(
                "recover", self.name,
                {"status": "no_replay_environment", "recovered": False,
                 "note": "RunContext.env 未配置或不支持 resolve（全量再求解）"},
            )
            return

        top = max(hyps, key=lambda h: (h.confidence, -h.step))
        max_rounds = int(self.param("max_rounds", 3))   # 原文：3 轮
        feedback = self._reflection_from_hypothesis(top)
        feedback_log: list[str] = [feedback[:300]]
        attempts: list[dict] = []
        recovered = False
        for k in range(1, max_rounds + 1):
            new_traj = env.resolve(bundle.trajectory, feedback)
            bundle.reruns.append(new_traj)
            attempts.append(
                {
                    "round": k,
                    "trace_id": new_traj.trace_id,
                    "success": new_traj.outcome.success,
                    "note": new_traj.outcome.note[:120],
                }
            )
            if new_traj.outcome.success:
                recovered = True
                break
            # AgenTracer：下一轮反馈由 tracer 对最新失败轨迹重新生成
            reflected = self._reflect(ctx, new_traj)
            if reflected:
                feedback = reflected
            else:
                feedback = (
                    f"{feedback}\n(attempt {k} failed: {new_traj.outcome.note})"
                )
            feedback_log.append(feedback[:300])

        bundle.put(
            "recover",
            self.name,
            {
                "status": "done",
                "origin": bundle.trace_id,
                "mode": "full_reresolve",
                "seed_hypothesis": {
                    "agent": top.agent,
                    "step": top.step,
                    "confidence": top.confidence,
                },
                "feedback_seed": feedback_log[0],
                "feedback_rounds": feedback_log,
                "rounds": len(attempts),
                "attempts": attempts,
                "recovered": recovered,
            },
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _reflection_from_hypothesis(top) -> str:
        parts = ["对上一轮失败求解的归因反思："]
        if top.root_cause:
            parts.append(top.root_cause)
        if top.fix_suggestion:
            parts.append(f"修正建议：{top.fix_suggestion}")
        if top.agent and top.step is not None:
            parts.append(f"（责任方 {top.agent}，决定性错误步 step {top.step}）")
        return "\n".join(parts)

    def _reflect(self, ctx, failed_traj) -> str | None:
        """对最新失败轨迹生成下一轮反思（无 LLM 时返回 None 走降级拼接）。"""
        if ctx.llm is None:
            return None
        from atap.core.bundle import TrajectoryBundle

        bundle = TrajectoryBundle(failed_traj)  # 无 SSF 产物 → 全量视图渲染
        messages = [
            {"role": "system", "content": _REFLECT_SYSTEM},
            {"role": "user", "content": f"失败轨迹如下：\n{judge_view(bundle)}"},
        ]
        result = ctx.llm.complete(messages, schema=Reflection, tag="feedback_reflection")
        parsed = result.parsed
        assert isinstance(parsed, Reflection)
        return parsed.feedback
