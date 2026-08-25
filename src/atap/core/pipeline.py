"""Pipeline —— 六阶段编排器。

流程语义对齐《整体流程架构与算法文献》§1 的两个关键事实：

* **检测 ≠ 归因**：analyze 只发现"有没有问题"，attribute 才回答"哪个
  错误决定了失败"；失败显现步往往不是致因步（误定位 81% 偏晚）。
  归因算法自行决定触发条件（如 all_at_once 只处理失败轨迹）。
* **闭环**：④的告警/低分/失败触发⑤归因与恢复；recover 产出的新轨迹
  回到 analyze 验证改善（环节 6 → 环节 3）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atap.core.base import STAGE_ORDER

if TYPE_CHECKING:
    from atap.core.base import StageAlgorithm
    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext
    from atap.core.schema import Trajectory


@dataclass
class PipelineReport:
    """一次运行的人读报告（写盘 / CLI 输出）。"""

    run_name: str
    n_traces: int = 0
    n_failures: int = 0
    n_attributed: int = 0
    n_reruns: int = 0
    n_rerun_success: int = 0
    stage_log: list[str] = field(default_factory=list)
    bundle_summaries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "n_traces": self.n_traces,
            "n_failures": self.n_failures,
            "n_attributed": self.n_attributed,
            "n_reruns": self.n_reruns,
            "n_rerun_success": self.n_rerun_success,
            "stage_log": self.stage_log,
            "bundle_summaries": self.bundle_summaries,
        }


class Pipeline:
    """按 STAGE_ORDER 依次执行各算法（每算法先 run_corpus 聚合作用域）。"""

    def __init__(self, algorithms: list["StageAlgorithm"]) -> None:
        self.algorithms = algorithms

    def run(
        self, trajectories: list["Trajectory"], ctx: "RunContext"
    ) -> tuple[list["TrajectoryBundle"], PipelineReport]:
        from atap.core.bundle import TrajectoryBundle

        report = PipelineReport(run_name=ctx.run_dir or "run")
        bundles = [TrajectoryBundle(t) for t in trajectories]
        report.n_traces = len(bundles)
        report.n_failures = sum(0 if b.succeeded else 1 for b in bundles)

        for stage in STAGE_ORDER:
            for algo in self.algorithms:
                if algo.stage != stage:
                    continue
                t0 = time.time()
                algo.run_corpus(bundles, ctx)
                report.stage_log.append(
                    f"{stage}/{getattr(algo, 'name', type(algo).__name__)} "
                    f"-> {len(bundles)} bundles in {time.time() - t0:.3f}s"
                )

        report.n_attributed = sum(1 for b in bundles if b.hypotheses())
        rerun_traces: list[Trajectory] = []
        for b in bundles:
            report.bundle_summaries.append(b.summary())
            report.n_reruns += len(b.reruns)
            report.n_rerun_success += sum(1 for t in b.reruns if t.outcome.success)
            rerun_traces.extend(b.reruns)
        self.last_reruns = rerun_traces
        return bundles, report

    def run_closed_loop(
        self,
        trajectories: list["Trajectory"],
        ctx: "RunContext",
        *,
        max_rounds: int = 1,
    ) -> tuple[list["TrajectoryBundle"], list[PipelineReport]]:
        """闭环：跑完一轮后，把 recover 产出的新轨迹送回全流程验证改善。

        返回**第一轮的 bundles**（保留完整归因/恢复产物），其中每条被重跑
        的轨迹额外挂 ``recover/closed_loop`` 产物记录验证结论；验证轮的
        报告追加在 reports 里（环节 6→3 回路）。
        """
        origin_bundles, report = self.run(trajectories, ctx)
        reports = [report]
        reruns = getattr(self, "last_reruns", [])
        if reruns and max_rounds >= 1:
            rerun_by_origin: dict[str, Trajectory] = {}
            for t in reruns:
                origin = t.meta.get("rerun_of")
                if origin:
                    rerun_by_origin[origin] = t
            current = [rerun_by_origin.get(t.trace_id, t) for t in trajectories]
            _, verify_report = self.run(current, ctx)
            reports.append(verify_report)
            improved = set()  # 验证轮中成功轨迹的 trace_id
            for t in current:
                if t.outcome.success:
                    improved.add(t.trace_id)
            for b in origin_bundles:
                repl = rerun_by_origin.get(b.trace_id)
                b.put(
                    "recover",
                    "closed_loop",
                    {
                        "rerun_trace_id": repl.trace_id if repl else None,
                        "verified_improved": bool(repl and repl.trace_id in improved),
                    },
                )
            ctx.closed_loop_rounds += 1
        return origin_bundles, reports
