# atap — Agent 轨迹分析与错误归因平台

模型设置 增加nvidia/nemotron-3.5-lightning:free，[REDACTED OpenRouter key]  run real test


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
.venv/bin/python -m pytest tests/          # 139 个测试（含离线全链路 e2e）
.venv/bin/atap demo                        # 离线全链路演示（FakeLLM，确定性）
.venv/bin/atap run --config configs/pipeline_offline.yaml
# 阶段三全栈（R5+循环谓词+L0 规则包+二分+反馈注入，频谱语料 24 条）：
.venv/bin/atap corpus --out runs/corpus/traces.jsonl
.venv/bin/atap run --config configs/pipeline_offline_v3.yaml --out runs/v3
# 同一轨迹集上的算法组合对比：
.venv/bin/atap compare --config configs/pipeline_offline_v3.yaml \
                        --config configs/pipeline_sbfl.yaml --out runs/compare
# 真实 LLM：设置 OPENAI_API_KEY（可选 OPENAI_BASE_URL）后
.venv/bin/atap run --config configs/pipeline_llm.yaml
```

`atap demo` 的离线验收结果（seed=7，六种注入故障）：

```
归因命中: step 6/6  agent 6/6  MAST 6/6  恢复 6/6
round0: traces=7 failures=6 attributed=6 reruns=6(ok=6)
round1: traces=7 failures=0   ← 重跑轨迹全部通过全流程闭环验证
```

阶段三全栈（24 条语料）离线验收：二分定位 **step 5/6、agent 6/6**；
SBFL 频谱 4/6（miss 两例为方法边界：一次性动作被 γ 压制、内容级故障在
动作谱不可见）；反馈注入恢复 6/6、闭环 failures=0；对比表
`v3 15/18·147 calls vs SBFL 12/18·42 calls`。

真实 LLM 验证（两轮）。**DeepSeek `deepseek-v4-flash`**（temperature=0）：
阶段二栈 **step 6/6、agent 6/6、MAST 主标签 3/6、恢复 6/6**——"恢复 0/6"
的已知限制已由沙盒"关键词优先、LLM 语义兜底"反馈消费修复确认（六故障
全部 1 轮恢复、fault_removed=True）；阶段三栈 binary_search step 2/6（与
Who&When 文献"二分弱于单遍"方向一致：短轨迹下判官 lower-half 偏置会把
区间塌缩到早期步）、feedback_injection 恢复同样 6/6。此前 OpenRouter
`stealth/ox-alpha`：step 4/6、agent 5/6、恢复 0/6。离线 6/6 证明的是框架
管路与契约正确，真实数字衡量判官能力，两者互补。复跑：
`realtest_deepseek.py` / `realtest_nemotron.py`（密钥走环境变量，不落盘）。

## 已实现算法（轮次一：阶段一+二；轮次二：阶段三）

| 流程 | 算法 | 文献 |
|---|---|---|
| 表征 | `canonical_events`（R0：span 树→统一事件流） | AgentDebugX 2607.18754 |
| 表征 | `ssf`（R1：显著性折叠，可逆占位符+摘要） | TrajAudit 2605.26563 |
| 表征 | `action_signature`（R5：九类动作+参数指纹+七效果标签+锚集/里程碑/LCS） | TraceProbe 2607.06184 |
| 分析 | `judge_eval`（LLM-as-judge 质量分+finding，few-shot） | MAST 2503.13657 / Agent-as-a-Judge 2410.10934 |
| 分析 | `loop_detect`（循环检测谓词：search loop/re-read churn/tool oscillation/redundant search，确定性免 LLM） | TraceProbe 2607.06184 |
| 分类 | `mast_judge`（MAST 3 类 14 模式打标，词表校验） | MAST 2503.13657 |
| 分类 | `rule_pack`（L0 免费规则包：畸形调用/无进展循环/过早成功声明/无效输出，确定性免 LLM） | AgentDebugX 2607.18754 |
| 归因 | `all_at_once`（L1 单遍全轨迹 → 统一 Hypothesis 输出） | Who&When 2505.00212 |
| 归因 | `binary_search`（L2 二分定位，⌈log₂n⌉ 轮，只展示下半段） | Who&When 2505.00212 |
| 归因 | `sbfl`（L0 频谱先验：γ/β/α/λ-decay+Kulczynski2^λ，`run_corpus` 跨轨迹聚合） | FAMAS 2509.13782 |
| 恢复 | `targeted_rerun`（保留前缀、从 t* 带反馈重跑 ≤5 轮） | AgentDebug 2509.25370 |
| 恢复 | `feedback_injection`（归因反思注入下一轮完整重解，3 轮） | AgenTracer 2509.03312 |

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
- **双作用域**：`run_one`（单轨迹）/ `run_corpus`（跨轨迹聚合）——阶段三起由
  `action_signature`（同任务成功参照锚集/LCS）与 `sbfl`（失败/成功频谱）使用；
- **检测 ≠ 归因 / 闭环**：analyze 只发现症状，attribute 由失败触发，
  recover 产物自动回到 analyze 验证（`closed_loop: true`）。

## 分层不变量（tests/test_invariants.py 强制）

- `core/` 零算法、零 I/O 实现（只允许接口协议）；
- 算法模块不得 import 其它 stage 包（唯一例外：`classify/taxonomy` 共享
  词表）、不得 import sandbox/runtime/cli；
- `llm/ io/` 不依赖 stage 包；`sandbox/` 只依赖 core；
- 注册表内所有类的 stage 必须与其所在包一致。

## 路线图（阶段四候选，下一轮）

R2 信息依赖图（含 CHIEF 层次因果图，step 级 52% 迄今最高）、R3 claim 台账
（DRIFT）、R4 层级树（CodeTracer，树索引 +18.3pt）、RG/UG 确定性归因
（搜索智能体诊断 2608.01913）、失败聚类+残差词表扩展（AgentDebugX inducer）、
分布漂移检测（2511.19933）、L3 动态重放（TraceElephant 2604.22708 / DoVer）、
Langfuse/OTel 采集适配器、SBFL 作为 L2 先验的实际融合（当前为独立算法）、
AgenTracer GRPO 微调 tracer 路线。

详细进度见 [plan.md](plan.md)。
