# atap — Agent 轨迹分析与错误归因平台

复现 [《Agent 轨迹分析与错误归因：整体流程·架构·算法与文献》](../../paper_research/general/papers/research_surveys/agents/error_analysis/整体流程架构与算法文献.md)
的六环节流程（采集 → 表征 → 分析评测 → 错误分类打标 → 失败归因 → 恢复闭环），
做成 **transformers 式可插拔框架**：每个算法一个模块、继承阶段基类、注册进
Registry、YAML 配置组合 pipeline；算法之间只通过产物（artifact）解耦，
不互相 import（由 import 图不变量测试强制）。

```
①采集 io/         ②存储 JSONL     ③表征 represent/    ④分析 analyze/ + classify/
                                  R0 事件化, R1 SSF     LLM-as-judge, MAST 判官
⑤归因 attribute/                  ⑥恢复 recover/
   All-at-Once (Who&When)            定向重跑 (AgentDebug) → 新轨迹回到 ④ 验证（闭环）
```

## 快速开始

```bash
uv venv .venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/          # 70 个测试（含离线全链路 e2e）
.venv/bin/atap demo                        # 离线全链路演示（FakeLLM，确定性）
.venv/bin/atap run --config configs/pipeline_offline.yaml
# 真实 LLM：设置 OPENAI_API_KEY（可选 OPENAI_BASE_URL）后
.venv/bin/atap run --config configs/pipeline_llm.yaml
```

`atap demo` 的离线验收结果（seed=7，六种注入故障）：

```
归因命中: step 6/6  agent 6/6  MAST 6/6  恢复 6/6
round0: traces=7 failures=6 attributed=6 reruns=6(ok=6)
round1: traces=7 failures=0   ← 重跑轨迹全部通过全流程闭环验证
```

## 已实现算法（轮次一：阶段一 + 阶段二）

| 流程 | 算法 | 文献 |
|---|---|---|
| 表征 | `canonical_events`（R0：span 树→统一事件流） | AgentDebugX 2607.18754 |
| 表征 | `ssf`（R1：显著性折叠，可逆占位符+摘要） | TrajAudit 2605.26563 |
| 分析 | `judge_eval`（LLM-as-judge 质量分+finding，few-shot） | MAST 2503.13657 / Agent-as-a-Judge 2410.10934 |
| 分类 | `mast_judge`（MAST 3 类 14 模式打标，词表校验） | MAST 2503.13657 |
| 归因 | `all_at_once`（单遍全轨迹 → 统一 Hypothesis 输出） | Who&When 2505.00212 |
| 恢复 | `targeted_rerun`（保留前缀、从 t* 带反馈重跑 ≤5 轮） | AgentDebug 2509.25370 |

配套：`sandbox/` 玩具研究问答沙盒（planner→searcher→reporter，mock 检索
语料 + 验证器 + 六种故障注入，标签按构造已知——AgenTracer 路线 B 思想）；
`llm/` FakeLLM 确定性伪判官（只看判官可见文本做规则化症状判定，不读
ground truth）。

## 配置组合（可插拔的核心）

```yaml
stages:
  represent:
    - canonical_events
    - ssf                    # 换成/追加新算法只需写一行
  attribute:
    - all_at_once
  recover:
    - name: targeted_rerun
      params: {max_rounds: 5}
```

新增算法 = 在对应 stage 包下写一个模块：继承 `Representer/Analyzer/
Classifier/Attributor/Recoverer`，声明 `stage`/`name`，加 `@register`——
零改核心，`atap list` 即见。

## 关键契约

- **R0 事件模型**（`core/schema.py`）：`TraceEvent(kind/agent/action/payload/
  refs/phase/parent/index)`——表征层是分析/归因的唯一数据接口；
- **统一归因输出**：`Hypothesis(agent, step, root_cause, root_cause_code,
  responsible_side, evidence, fix_suggestion, confidence)`——L0~L3 任何
  归因算法都产出此结构，恢复阶段只消费它；
- **双作用域**：`run_one`（单轨迹）/ `run_corpus`（跨轨迹聚合，为阶段三
  SBFL/聚类预留）；
- **检测 ≠ 归因 / 闭环**：analyze 只发现症状，attribute 由失败触发，
  recover 产物自动回到 analyze 验证（`closed_loop: true`）。

## 分层不变量（tests/test_invariants.py 强制）

- `core/` 零算法、零 I/O 实现（只允许接口协议）；
- 算法模块不得 import 其它 stage 包（唯一例外：`classify/taxonomy` 共享
  词表）、不得 import sandbox/runtime/cli；
- `llm/ io/` 不依赖 stage 包；`sandbox/` 只依赖 core；
- 注册表内所有类的 stage 必须与其所在包一致。

## 路线图（阶段三，下一轮）

+R5 动作签名（TraceProbe 2607.06184）、+循环检测谓词、+L0 免费规则包
（AgentDebugX）、+二分定位（Who&When）、+SBFL 频谱归因（FAMAS 2509.13782，
run_corpus 聚合）、+反馈注入再求解（AgenTracer）；远期：R2 依赖图、R3 claim
台账（DRIFT）、失败聚类+残差词表扩展、分布漂移检测、L3 动态重放
（TraceElephant）、Langfuse/OTel 采集适配器。

详细进度见 [plan.md](plan.md)。
