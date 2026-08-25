"""定向重跑（Targeted Re-rollout）—— AgentDebug, arXiv:2509.25370 Algorithm 1。

机制（原文 Stage 3）：定位最早关键步 t* 后，保留前缀 [0, t*)、从 t* 起
带**可执行反馈**重跑；成功即返回，失败则细化反馈再来，至多 I 轮
（原文 N=5；GPT-4o-mini ALFWorld 21→55）。关键设计：只修根因步，不修
表面症状；反馈要"指明错误类型 + 可执行指导"，供 agent 在重跑时改道。

本实现：t* 与反馈来自归因输出的 top Hypothesis（消费统一归因契约，
不感知具体归因算法）。t* 选择规则 = 置信度最高、并列取最早步——原文
Stage 2 的 t*←min(T*)（最早关键步）主要由归因 prompt 的"最早的决定性
错误"指令承载，此处做并列兜底【声明偏离：置信度优先于最早步】。重放
由 RunContext.env（ReplayEnvironment 协议）执行，每轮从**原始轨迹**的
t* 重放（原文伪代码链式传 τ⁽ᵏ⁻¹⁾；确定性沙盒下两者等价，且重跑轨迹
的 meta 已剥离 injected_fault，链式传递会误判"无故障"而虚假成功）
【声明偏离】。文献警示（§7）：responsible agent 粒度过粗无法被增强
消费，故本算法消费的是 (step, fix_suggestion) 两个细粒度字段。

产物：``{"origin", "t_star", "rounds", "attempts", "recovered"}``；
重跑轨迹 append 到 bundle.reruns（新 trace_id，meta.rerun_of=原轨迹），
由编排器送回 analyze 验证改善（环节 6→3 闭环）。
"""

from __future__ import annotations

from atap.core.registry import register
from atap.recover.base import Recoverer


@register
class TargetedRerunRecoverer(Recoverer):
    stage = "recover"
    name = "targeted_rerun"

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
        if ctx.env is None:
            bundle.put(
                "recover", self.name,
                {"status": "no_replay_environment", "recovered": False,
                 "note": "RunContext.env 未配置（sandbox: {type: toy}）"},
            )
            return

        # 置信度最高、并列取最早步（对齐原文 t*←min(T*) 的最早关键步原则）
        top = max(hyps, key=lambda h: (h.confidence, -h.step))
        max_rounds = int(self.param("max_rounds", 5))
        feedback = top.fix_suggestion or top.root_cause
        attempts: list[dict] = []
        recovered = False
        for k in range(1, max_rounds + 1):
            new_traj = ctx.env.rerun_from(bundle.trajectory, top.step, feedback)
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
            # UpdateFeedback（弱化版）：带上失败说明细化指导再试
            feedback = (
                f"{feedback}\n(attempt {k} failed: {new_traj.outcome.note} —— "
                f"请给出更具体、针对最早决定性错误 step {top.step} 的修正。)"
            )

        bundle.put(
            "recover",
            self.name,
            {
                "status": "done",
                "origin": bundle.trace_id,
                "t_star": top.step,
                "responsible_agent": top.agent,
                "feedback_seed": (top.fix_suggestion or top.root_cause)[:200],
                "rounds": len(attempts),
                "attempts": attempts,
                "recovered": recovered,
            },
        )
