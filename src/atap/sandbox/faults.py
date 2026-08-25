"""故障注入库 —— 六种文献故障模式在沙盒中的可注入实现。

对齐故障注入造数思想（AgenTracer 2509.03312 路线 B / Aegis-Kong
2509.14295：标签按构造已知）。每种故障：
* 在确定性的脚本化 rollout 中的某个**逻辑步**改变 agent 行为；
* 产生可观测症状（判官可见），并对应一个 MAST 失败模式；
* meta["injected_fault"] 记录 ground truth（kind/onset 事件序号/agent/
  MAST 代码）——仅供评测断言，判官类算法绝不读取。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    agent: str          # 责任 agent
    mast_code: str      # 对应 MAST 失败模式（分类 ground truth）
    onset_logical: str  # 脚本中偏离开始的逻辑步名
    description: str


FAULTS: dict[str, FaultSpec] = {
    f.kind: f
    for f in [
        FaultSpec(
            kind="step_repetition", agent="searcher", mast_code="FM-1.3",
            onset_logical="search#1",
            description="searcher 无进展地重复同一 search 调用直至预算耗尽（MAST FM-1.3 / TraceProbe 循环谓词的靶症状）",
        ),
        FaultSpec(
            kind="malformed_tool_call", agent="searcher", mast_code="FM-2.6",
            onset_logical="search",
            description=(
                "searcher 发起缺参的畸形工具调用，环境返回错误（AgentDebugX 免费"
                "规则包的首要靶症状；MAST 14 模式无工具格式专门类，最近似归入"
                "FM-2.6 推理-行动失配【适配】）"
            ),
        ),
        FaultSpec(
            kind="info_withholding", agent="searcher", mast_code="FM-2.4",
            onset_logical="handoff_report",
            description="searcher 检索到文档却向 reporter 谎报'没有找到'（MAST FM-2.4）",
        ),
        FaultSpec(
            kind="premature_termination", agent="planner", mast_code="FM-3.1",
            onset_logical="plan",
            description=(
                "planner 跳过检索直接凭记忆提交答案（MAST FM-3.1；onset=决定"
                "跳过检索的规划步——Who&When Eq.5 最早决定性错误，早于 submit "
                "终止动作一步）"
            ),
        ),
        FaultSpec(
            kind="ungrounded_citation", agent="reporter", mast_code="FM-3.3",
            onset_logical="compose",
            description=(
                "reporter 引用检索到但从未 read 过的文档（DRIFT 无支撑主张的靶"
                "症状；做了验证但把未读文档当作已核实依据，MAST FM-3.3）"
            ),
        ),
        FaultSpec(
            kind="disobey_task_spec", agent="reporter", mast_code="FM-1.1",
            onset_logical="compose",
            description="reporter 答案内容正确但违反任务规格（缺必填的已读文档编号引用，MAST FM-1.1）",
        ),
    ]
}

TOOL_BUDGET = 4  # 正常 rollout 只需 3 次工具调用；重复故障会击穿预算
