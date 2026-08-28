<div align="center">

# Agent Trace Analysis Platform（atap）

**定位、解释并修复 LLM Agent 失败 —— 从原始轨迹到验证恢复的一条可插拔流水线**

[![CI](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/aaronlyt/agent-trace-analysis-platform/badges/coverage.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aaronlyt/agent-trace-analysis-platform)](https://github.com/aaronlyt/agent-trace-analysis-platform/releases)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | **简体中文**

<img src="docs/assets/demo.gif" alt="atap 终端演示" width="100%">

```bash
git clone https://github.com/aaronlyt/agent-trace-analysis-platform && cd agent-trace-analysis-platform
pip install -e ".[dev,llm]"
atap demo    # 离线端到端流水线：FakeLLM 伪判官、确定性、零网络
```

<sub><b>阶段与方法 —— 24 个算法</b></sub>

[![采集](https://img.shields.io/badge/%E9%87%87%E9%9B%86-%E2%9C%85_1-brightgreen)](#已实现算法) [![表征](https://img.shields.io/badge/%E8%A1%A8%E5%BE%81-%E2%9C%85_7-brightgreen)](#已实现算法) [![分析](https://img.shields.io/badge/%E5%88%86%E6%9E%90-%E2%9C%85_3-brightgreen)](#已实现算法)

[![分类](https://img.shields.io/badge/%E5%88%86%E7%B1%BB-%E2%9C%85_3-brightgreen)](#已实现算法) [![归因](https://img.shields.io/badge/%E5%BD%92%E5%9B%A0-%E2%9C%85_8-brightgreen)](#已实现算法) [![恢复](https://img.shields.io/badge/%E6%81%A2%E5%A4%8D-%E2%9C%85_3-brightgreen)](#已实现算法)

<sub><b>交付状态 —— 阶段 1 → 4D</b></sub>

[![阶段一](https://img.shields.io/badge/%E4%B8%80-%E6%9E%B6%E6%9E%84%E9%AA%A8%E6%9E%B6_%E2%9C%85-brightgreen)](#路线图) [![阶段二](https://img.shields.io/badge/%E4%BA%8C-%E5%9E%82%E7%9B%B4%E5%88%87%E7%89%87_%E2%9C%85-brightgreen)](#路线图) [![阶段三](https://img.shields.io/badge/%E4%B8%89-L0%2FL2%E4%BB%A3%E4%BB%B7%E9%98%B6%E6%A2%AF_%E2%9C%85-brightgreen)](#路线图) [![阶段四A](https://img.shields.io/badge/%E5%9B%9BA-%E7%A1%AE%E5%AE%9A%E6%80%A7%E5%B1%82_%E2%9C%85-brightgreen)](#路线图)

[![阶段四B](https://img.shields.io/badge/%E5%9B%9BB-LLM%E8%A1%A8%E5%BE%81%2B%E5%BD%92%E5%9B%A0_%E2%9C%85-brightgreen)](#路线图) [![阶段四C](https://img.shields.io/badge/%E5%9B%9BC-%E5%8F%8D%E4%BA%8B%E5%AE%9E%E9%87%8D%E6%94%BE_%E2%9C%85-brightgreen)](#路线图) [![阶段四D](https://img.shields.io/badge/%E5%9B%9BD-Langfuse%2FOTel%E9%80%82%E9%85%8D_%E2%9C%85-brightgreen)](#路线图) [![计划中](https://img.shields.io/badge/%E8%AE%A1%E5%88%92%E4%B8%AD-%E7%9C%9F%E5%AE%9E%E6%95%B0%E6%8D%AE%E9%9B%86_%C2%B7_%E6%B2%99%E7%9B%92_%F0%9F%93%8B-blue)](#路线图)

</div>

Agent Trace Analysis Platform（**atap**，Agent 轨迹分析平台）是一个面向
LLM Agent 执行轨迹分析的可插拔框架：读入原始轨迹，完成**表征**、
**分析评测与错误分类打标**、把每条失败轨迹**归因**到根因——责任 agent 与
致因步——再**恢复**并在闭环中验证修复效果。它实现近年 agent 错误分析研究中通行的六环节流程
（采集 → 表征 → 分析评测 → 错误分类打标 → 失败归因 → 恢复闭环），
做成 **transformers 式可插拔框架**：每个算法一个模块、继承阶段基类、注册进
Registry、YAML 配置组合 pipeline；算法之间只通过产物（artifact）解耦，不互相
import（由 import 图不变量测试强制）。

- **5 个阶段 24 个算法**，每个都忠实对应一篇文献（见下表）
- **确定性离线模式** —— FakeLLM 伪判官 + 注入故障的玩具沙盒，完全可复现、零网络
- **真实 LLM** —— 任意 OpenAI 兼容 API，逐调用审计日志
- **统一归因契约** —— 规则/判官/图/重放等一切定位算法产出同一个 `Hypothesis` 结构，恢复阶段只消费它
- **恢复闭环** —— 恢复产物自动回到分析阶段验证（`closed_loop: true`）
- **采集适配** —— Langfuse v3 ingestion / OTel GenAI 语义约定导入导出，roundtrip 字段级等价，导出侧防 GT 泄漏

> **声明** —— 本项目是对 agent 错误分析流程的学习/研究性质实现。**测试有限**：
> 验收数字来自构造故障的玩具沙盒语料与少量真实模型轮次（见[验证状态](#验证状态)），
> 不构成基准测试结论。请将其用于学习管线与算法机制，请勿用于生产环境，也不宜
> 作为真实场景性能的依据。

## 整体流程

```
 ①采集                             ②存储
 io/（JSONL · Langfuse v3 · OTel GenAI） ──▶  traces.jsonl
                                                   │
                                                   ▼
 ③表征 — represent/
 R0 事件化 · SSF 显著性折叠 · 动作签名
 IDG 依赖图 · 层级树 · 主张台账 · 层次因果图
                                                   │   只经产物传递
                                                   ▼
 ④分析评测 + 分类打标 — analyze/ + classify/
 judge_eval · loop_detect · drift_detect · mast_judge · rule_pack · inducer
                                                   │   失败触发归因
                                                   ▼
 ⑤归因 — attribute/                                ──▶  Hypothesis
 all_at_once · binary_search · sbfl · rg_ug ·
 chief · claim_audit · tree_diagnosis · counterfactual_replay
                                                   │
                                                   ▼
 ⑥恢复 — recover/
 targeted_rerun · feedback_injection · dover
                                                   │
      重跑轨迹回到 ④ 闭环验证              ◀─────┘   （closed_loop: true）
```

## 已实现算法

| 流程 | 模块 | 机制 | 文献 |
|---|---|---|---|
| 表征 | `canonical_events` | R0：span 树拍平为统一事件流——下游一切阶段唯一的数据接口 | AgentDebugX [2607.18754](https://arxiv.org/abs/2607.18754) |
| 表征 | `ssf` | R1：显著性折叠，可逆占位符+摘要，抗长轨迹 | TrajAudit [2605.26563](https://arxiv.org/abs/2605.26563) |
| 表征 | `action_signature` | R5：九类动作+参数指纹+七效果标签+锚集/里程碑/LCS | TraceProbe [2607.06184](https://arxiv.org/abs/2607.06184) |
| 表征 | `idg` | R2：信息依赖图，refs 直出 usage 边，零 LLM | GraphTracer [2510.10581](https://arxiv.org/abs/2510.10581)（撤稿，仅思想） |
| 表征 | `hierarchy_tree` | R4：探索=兄弟/改状态=子节点 + stage 索引 + tree.md 压缩渲染 | CodeTracer [2604.11641](https://arxiv.org/abs/2604.11641) |
| 表征 | `claim_ledger` | R3：claim 六元组台账，LLM 全局单遍 | DRIFT [2606.02060](https://arxiv.org/abs/2606.02060) |
| 表征 | `hcg` | 层次因果图三层节点 + sub/agt/step 三类边，确定性构建 | CHIEF [2602.23701](https://arxiv.org/abs/2602.23701) |
| 分析 | `judge_eval` | LLM-as-judge 质量分+finding，few-shot | MAST [2503.13657](https://arxiv.org/abs/2503.13657) / Agent-as-a-Judge [2410.10934](https://arxiv.org/abs/2410.10934) |
| 分析 | `loop_detect` | 循环检测谓词：search loop/re-read churn/tool oscillation/redundant search，确定性免 LLM | TraceProbe [2607.06184](https://arxiv.org/abs/2607.06184) |
| 分析 | `drift_detect` | version/data/behavior 三族漂移对照：共享支撑 PSI + 支撑失配 + 最小分组门槛 + 共线特征去重 | 系统级 taxonomy [2511.19933](https://arxiv.org/abs/2511.19933) |
| 分类 | `mast_judge` | MAST 3 类 14 模式打标，词表校验；`allow_novel` 残差通道 | MAST [2503.13657](https://arxiv.org/abs/2503.13657) |
| 分类 | `rule_pack` | L0 免费规则包：畸形调用/无进展循环/过早成功声明/无效输出 | AgentDebugX [2607.18754](https://arxiv.org/abs/2607.18754) |
| 分类 | `inducer` | 残差聚类→新错误模式提案；人工闸门 `atap taxonomy accept`，永不自动生效 | AgentDebugX §3.4 [2607.18754](https://arxiv.org/abs/2607.18754) |
| 归因 | `rg_ug` | L0 确定性：qrels 集合运算判 RG/UG 四子类 + episode 效用，零 LLM | 搜索智能体诊断 [2608.01913](https://arxiv.org/abs/2608.01913) |
| 归因 | `sbfl` | L0 频谱先验：γ/β/α/λ-decay+Kulczynski2^λ，跨轨迹聚合 | FAMAS [2509.13782](https://arxiv.org/abs/2509.13782) |
| 归因 | `all_at_once` | L1 单遍全轨迹（消费 SSF 折叠视图） | Who&When [2505.00212](https://arxiv.org/abs/2505.00212) |
| 归因 | `binary_search` | L2 二分定位，⌈log₂n⌉ 轮，判官只看下半段 | Who&When [2505.00212](https://arxiv.org/abs/2505.00212) |
| 归因 | `claim_audit` | R3 消费：支撑四级→专家审计→保守回溯→first_error_span | DRIFT [2606.02060](https://arxiv.org/abs/2606.02060) |
| 归因 | `tree_diagnosis` | R4 消费：树级定位→stage 区间下钻，两次调用 | CodeTracer [2604.11641](https://arxiv.org/abs/2604.11641) |
| 归因 | `chief` | HCG 消费：oracle 合成→逆拓扑 F_eval 回溯→渐进因果筛选 | CHIEF [2602.23701](https://arxiv.org/abs/2602.23701) |
| 归因 | `counterfactual_replay` | L3 终审：候选步消息干预重放（k=3 窗口）滤伪因果；调整后的判决 `supersede` 上游假设 | TraceElephant [2604.22708](https://arxiv.org/abs/2604.22708) |
| 恢复 | `targeted_rerun` | 保留前缀、从 t* 带反馈重跑 ≤5 轮 | AgentDebug [2509.25370](https://arxiv.org/abs/2509.25370) |
| 恢复 | `feedback_injection` | 归因反思注入下一轮完整重解，3 轮 | AgenTracer [2509.03312](https://arxiv.org/abs/2509.03312) |
| 恢复 | `dover` | do-then-verify：trial 切分→最小消息干预→原位替换重放 ×3→里程碑差分四判定 | DoVer [2512.06749](https://arxiv.org/abs/2512.06749) |

配套基础设施：

- `sandbox/` —— 玩具研究问答沙盒（planner→searcher→reporter），mock 检索语料 +
  qrels 两层标注（E/G）+ 验证器 + 六种注入故障 + 两种扩展故障 + 漂移语料生成器，
  标签按构造已知（AgenTracer 路线 B 思想）；
- `llm/` —— FakeLLM 确定性伪判官（只看判官可见文本做规则化症状判定，不读
  ground truth），以及 OpenAI 兼容客户端与共用调用审计挂件。

## 安装

当前发布方式为**本地源码安装**（后续可能提供 PyPI 包）。要求 **Python ≥ 3.10**
（在 3.12 上开发与测试）。

```bash
git clone https://github.com/aaronlyt/agent-trace-analysis-platform
cd agent-trace-analysis-platform
```

使用 [uv](https://docs.astral.sh/uv/)（推荐）：

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev,llm]"
```

或使用 pip：

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm]"
```

依赖说明：基础安装只拉取 `pyyaml` + `pydantic`；`llm` 额外安装 `openai` 客户端
（真实模型运行需要）；`dev` 额外安装 `pytest`。离线（FakeLLM）功能不依赖 `llm`。

## 快速开始

```bash
# 1) 列出全部注册算法
atap list

# 2) 离线全链路演示 —— FakeLLM，确定性，零网络（seed=7）
atap demo
```

`atap demo` 在 7 条沙盒轨迹（1 成功 + 6 注入故障）上跑完整管线，逐故障打印
归因是否命中 GT 的 step/agent、恢复是否通过闭环验证：

```
attribution hits: step 6/6  agent 6/6  MAST 6/6  recovery 6/6
round0: traces=7 failures=6 attributed=6 reruns=6(ok=6)
round1: traces=6 failures=0 attributed=0 reruns=0(ok=0)
artifacts directory: runs/demo/artifacts (report.json + per-trajectory per-stage JSON)
```

再试各档可运行配置：

```bash
# 与 demo 同栈，由配置文件驱动
atap run --config configs/pipeline_offline.yaml

# 阶段三栈：二分定位 + 循环谓词 + L0 规则包 + 反馈注入（频谱语料 24 条）
atap corpus --out runs/corpus/traces.jsonl
atap run --config configs/pipeline_offline_v3.yaml --out runs/v3

# 同一轨迹集上对比两套算法组合
atap compare --config configs/pipeline_offline_v3.yaml \
             --config configs/pipeline_sbfl.yaml --out runs/compare

# 阶段四确定性层（IDG + 层级树 + RG/UG + 漂移 + inducer）
atap run --config configs/pipeline_offline_v4.yaml --out runs/v4

# 漂移监控语料（三类构造漂移场景）
atap corpus --drift --out runs/drift/traces.jsonl
atap run --config configs/pipeline_drift.yaml --out runs/drift

# L3 恢复闭环（DoVer）与反事实重放
atap run --config configs/pipeline_dover.yaml --out runs/dover
atap run --config configs/pipeline_cf_replay.yaml --out runs/cf_replay

# 导出为 Langfuse（v3 ingestion）或 OTel（GenAI semconv）格式
atap export --traces runs/corpus/traces.jsonl --format langfuse --out runs/export.json

# 过程日志 DEBUG 级（默认 INFO）
atap -v run --config configs/pipeline_offline.yaml
```

### 真实 LLM 运行

任意 OpenAI 兼容端点均可。密钥只从环境变量读取，绝不落盘：

```bash
export OPENAI_API_KEY=sk-...                     # 必需
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # 可选（默认 OpenAI）
atap run --config configs/pipeline_llm.yaml
```

`configs/` 中另有产出下述真实模型数字所用的 `final_*`（上线前全量测试八档）与
`realtest_*`（特定模型冒烟档）配置。

### 日志与调用审计

每次 `run / demo / compare` 在 `runs/<name>/` 下自动落两份记录：

- `run.log` —— 过程日志（各阶段耗时、验收数字）；`-v` 开 DEBUG；
- `llm_calls.jsonl` —— **每次 LLM 调用一条审计记录**：时间戳、client、tag、
  model、schema、延迟、完整 prompt messages、response、token 用量、错误信息。
  Fake 与真实客户端共用同一审计挂件。

```bash
python -c "import json,collections; \
recs=[json.loads(l) for l in open('runs/demo/llm_calls.jsonl')]; \
print(len(recs), dict(collections.Counter(r['tag'] for r in recs)))"
```

## 配置组合 —— 可插拔的核心

Pipeline 即普通 YAML；算法按注册名引用，可带参数：

```yaml
run_name: offline-full-pipeline
seed: 7
source: {type: jsonl, path: runs/demo/traces.jsonl}
llm: {type: fake}          # 或 {type: openai, model: ..., temperature: 0.0}
sandbox: {type: toy}
closed_loop: true          # 恢复产物自动回到分析阶段验证

stages:
  represent:
    - canonical_events
    - ssf                  # 换成/追加新算法只需写一行
  analyze:
    - judge_eval
  classify:
    - mast_judge
  attribute:
    - all_at_once
  recover:
    - name: targeted_rerun
      params: {max_rounds: 5}
```

### 新增自己的算法

在对应 stage 包下写一个模块：继承阶段基类，加 `@register`——零改核心，
`atap list` 即见：

```python
# src/attribute/my_attributor.py
from atap.attribute.base import Attributor
from atap.core.registry import register
from atap.core.schema import Hypothesis


@register
class MyAttributor(Attributor):        # stage = "attribute" 由基类声明
    name = "my_attributor"

    def run_one(self, bundle, ctx):
        if bundle.succeeded:
            return                      # 检测 ≠ 归因
        self.emit(bundle, [Hypothesis(
            agent="reporter", step=3,
            root_cause="…", root_cause_code="FM-1.3",
            evidence=["event-3 …"], fix_suggestion="…", confidence=0.6,
        )])                             # 写入 artifacts["attribute"]["my_attributor"]
```

## 关键契约

- **R0 事件模型**（`core/schema.py`）：`TraceEvent(kind/agent/action/payload/refs/
  phase/parent/index)` —— 表征层是分析/归因的唯一数据接口；
- **统一归因输出**：`Hypothesis(agent, step, root_cause, root_cause_code,
  responsible_side, evidence, fix_suggestion, confidence)` —— L0~L3 任何归因算法
  都产出此结构，恢复阶段只消费它；
- **双作用域**：`run_one`（单轨迹）/ `run_corpus`（跨轨迹聚合——频谱与聚类类
  算法使用）；
- **检测 ≠ 归因 / 闭环**：analyze 只发现症状，attribute 由失败触发，recover
  产物自动回到 analyze 验证（`closed_loop: true`）。

## 分层不变量（tests/test_invariants.py 强制）

- `core/` 零算法、零 I/O 实现（只允许接口协议）；
- 算法模块不得 import 其它 stage 包（唯一例外：`classify/taxonomy` 共享词表）、
  不得 import sandbox/runtime/cli；
- `llm/ io/` 不依赖 stage 包；`sandbox/` 只依赖 core；
- 注册表内所有类的 stage 必须与其所在包一致。

## 验证状态

**离线（FakeLLM，确定性）。** 332 个测试一秒内全绿，含离线全链路 e2e、重放
完整性不变量与防泄漏回归。沙盒语料上的代表性验收数字：

| 栈 | 语料 | 离线结果 |
|---|---|---|
| demo（SSF + 单遍归因 + 定向重跑） | 7 条轨迹、6 注入故障 | step 6/6 · agent 6/6 · MAST 6/6 · 恢复 6/6；闭环 round1 失败 0 |
| v3（二分 + 规则包 + 反馈注入） | 18 故障语料 | step 15/18 · 141 次调用（同语料 SBFL 12/18 · 42 次调用） |
| rg_ug（确定性，零 LLM） | 18 故障语料 | step 15/18 · agent 15/18 |
| chief | 18 故障语料 | step 18/18 · agent 18/18 |
| tree_diagnosis / claim_audit | 18 故障语料 | 18/18 · 36 次调用 / 12/18（两例 honest miss 为已记录的方法边界） |
| dover | 18 故障语料 | 恢复 18/18；闭环改善 18/18 |
| counterfactual_replay | 18 故障语料 | 15 validated / 3 refuted——被标伪的 3 例恰为二分的已知错步 |

**真实 LLM**（上线前全量测试：deepseek-v4-flash 直连、8 档配置、594 次审计调用、
判官 prompt 零泄漏、人工抽检无幻觉；报告见
[docs/audit_上线前真实测试_2026-08-25.md](docs/audit_上线前真实测试_2026-08-25.md)）：

- 冒烟栈：step 6/6 · agent 6/6 · 恢复 6/6；
- **chief：step 17/18 · agent 18/18——真实模型最佳定位器**；
- claim 覆盖 14/18；tree 14/18；dover 恢复 18/18；v3 闭环 18/18；
- binary_search 3/18——远低于其离线基线 15/18（短轨迹下判官 lower-half 偏置，
  属判官能力上限而非管线缺陷，建议降级辅助使用）。

> **离线数字请正确解读。** 离线沙盒按"判官修复文案是否命中注入故障名"判定
> "故障已移除"，因此离线恢复率与重放判决是归因命中的确定性函数：它们证明的是
> 管线契约（假设 → 反馈 → 重放 → 验证）正确，而非判官能力。同理，离线的
> **跨算法对比**量的是确定性伪判官对各算法暴露的信息量差异，不构成算法优劣
> 证据。真实模型数字才是衡量能力的口径。

## 项目结构

```
src/
  core/        # 注册表 · pipeline · schema · 配置 —— 零算法、零 I/O
  represent/   # R0–R5 轨迹表征
  analyze/     # 症状发现：判官评测、循环谓词、漂移检测
  classify/    # MAST 打标 · L0 规则包 · 残差模式 inducer
  attribute/   # L0–L3 失败归因（成本阶梯）
  recover/     # 定向重跑 · 反馈注入 · do-then-verify
  llm/         # FakeLLM 伪判官 · OpenAI 兼容客户端 · 调用审计
  io/          # JSONL 存储 · Langfuse/OTel 适配器 · 导出防泄漏
  sandbox/     # 玩具研究问答沙盒（故障注入 + 漂移语料）
configs/       # 可运行配置（offline · LLM · realtest · final）
tests/         # 332 个测试：e2e · 不变量 · 防泄漏回归 · 重放完整性
docs/          # 计划 · 审计报告 · 开发日志
```

## 路线图

- [x] 阶段四A 确定性层：`idg` / `hierarchy_tree` / `rg_ug` / `drift_detect` / `inducer` + taxonomy accept
- [x] 阶段四B LLM 表征与归因升级：`claim_ledger`+`claim_audit`（DRIFT）、`tree_diagnosis`（CodeTracer）、`hcg`+`chief`（CHIEF）
- [x] 阶段四C L3 反事实重放：沙盒 `replay_intervene` 基建、`counterfactual_replay`（TraceElephant）、`dover`（DoVer）
- [x] 阶段四D 采集适配器：Langfuse v3 ingestion、OTel GenAI semconv、`atap export` + roundtrip
- [ ] SBFL 作为 L2 先验的实际融合（当前为独立算法）；AgenTracer 式 GRPO 微调 tracer
- [ ] 沙盒演进 —— 从玩具研究问答沙盒走向更真实的多场景执行环境（更丰富的任务类型、真实工具调用、更多故障注入）
- [ ] 真实数据集评测 —— 在公开的真实 agent 轨迹数据集/基准上验证管线，替代构造语料的验收数字

详细计划：[docs/plan.md](docs/plan.md) · [docs/plan_阶段四.md](docs/plan_阶段四.md) ·
开发日志：[docs/README_dev_log.md](docs/README_dev_log.md)
