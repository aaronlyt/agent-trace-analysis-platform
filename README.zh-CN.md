<div align="center">

# Agent Trace Analysis Platform（atap）

**架在你现有 Agent 可观测性栈（如 Langfuse）之上的归因与恢复层 —— 定位、解释并修复 LLM Agent 失败，把结论作为 Score 写回原处**

[![CI](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/aaronlyt/agent-trace-analysis-platform/badges/coverage.svg)](https://github.com/aaronlyt/agent-trace-analysis-platform/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aaronlyt/agent-trace-analysis-platform)](https://github.com/aaronlyt/agent-trace-analysis-platform/releases)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.md) | **简体中文**

<img src="docs/assets/langfuse_roundtrip.gif" alt="atap 在活 Langfuse 实例上的外部评估" width="100%">

<sub>atap × Langfuse：推送语料 → 拉取活轨迹 → 分析/分类/归因 → 根因码 + 置信度 + 被归因步标记作为 Score 写回（纯终端演示见 `docs/assets/demo.gif`）</sub>

<img src="docs/assets/integration.svg" alt="atap 集成架构：轨迹来源 → 适配器 → 六段流水线 → Score 回写" width="100%">

```bash
git clone https://github.com/aaronlyt/agent-trace-analysis-platform && cd agent-trace-analysis-platform
pip install -e ".[dev,llm]"
atap demo    # 离线端到端流水线：FakeLLM 伪判官、确定性、零网络
```

<sub><b>阶段与方法 —— 24 个算法</b></sub>

[![采集](https://img.shields.io/badge/%E9%87%87%E9%9B%86-%E2%9C%85_1-brightgreen)](docs/算法清单.md) [![表征](https://img.shields.io/badge/%E8%A1%A8%E5%BE%81-%E2%9C%85_7-brightgreen)](docs/算法清单.md) [![分析](https://img.shields.io/badge/%E5%88%86%E6%9E%90-%E2%9C%85_3-brightgreen)](docs/算法清单.md)

[![分类](https://img.shields.io/badge/%E5%88%86%E7%B1%BB-%E2%9C%85_3-brightgreen)](docs/算法清单.md) [![归因](https://img.shields.io/badge/%E5%BD%92%E5%9B%A0-%E2%9C%85_8-brightgreen)](docs/算法清单.md) [![恢复](https://img.shields.io/badge/%E6%81%A2%E5%A4%8D-%E2%9C%85_3-brightgreen)](docs/算法清单.md)

</div>

Agent Trace Analysis Platform（**atap**）是**架在你现有可观测性栈之上的归因与
恢复层**：从 Langfuse（或 JSONL / OTel / Phoenix 导出）拉取轨迹，定位每条失败的
**责任 agent 与致因步**，把结论作为 Langfuse score 写回原处。

- **24 个可插拔算法、5 个阶段**——一个算法一个模块，YAML 组合，产物解耦（[docs/算法清单.md](docs/算法清单.md)）
- **确定性离线模式**——FakeLLM 判官 + 注入故障沙盒，零网络
- **真实 LLM**——任意 OpenAI 兼容 API，逐调用审计
- **Langfuse 集成**——`atap langfuse-eval` 把根因 score 与 blamed-step 标记写回你的 trace（[见下文](#与-langfuse-集成)）
- **闭环**——统一 `Hypothesis` 契约，恢复重跑自动回到分析验证

> **声明** —— 本项目是对 agent 错误分析流程的学习/研究性质实现。**测试有限**：
> 验收数字来自构造故障的玩具沙盒语料与少量真实模型轮次（见[docs/validation.md](docs/validation.md)），
> 不构成基准测试结论。请将其用于学习管线与算法机制，请勿用于生产环境，也不宜
> 作为真实场景性能的依据。

## 真实数据上的成绩

首个外部基准：**Who&When**（[ag2ai/Agents_Failure_Attribution](https://github.com/ag2ai/Agents_Failure_Attribution)，
ICML 2025）——184 条真实多智能体失败轨迹，gold 对判官不可见，用现成的
`compare` 评测器打分（[完整报告](docs/benchmark_whoswhen_2026-08-30.md)）：

| 栈（deepseek-v4-flash） | step 命中 | agent 命中 | 花费 |
|---|---|---|---|
| all_at_once | **33.2%** | **50.0%** | $1.43 · 2.9h |
| all_at_once，关思考 | 32.6% | 44.0% | $0.16 · 9min |
| binary_search | 11.4% | 34.2% | $0.26 · 32min |

模型选择主导成败——算法生成轨迹上 step 命中对思考档不敏感、成本低 9 倍，
思考只在手写转写上见效。沙盒验收数字（[docs/validation.md](docs/validation.md)）
证明管线契约，这一组数字证明它在真实数据上成立。

## 整体流程

```
 ①采集                             ②存储
 io/（JSONL · Langfuse v3 · OTel GenAI · 活实例 API） ──▶  traces.jsonl
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

完整算法表格（24 个算法、每个对应一篇文献）与配套基础设施说明移至
**[docs/算法清单.md](docs/算法清单.md)**（[English](docs/algorithms.md)）。

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

<details>
<summary><b>更多可运行配置</b>——compare · v3/v4 栈 · 漂移 · dover · 反事实重放 · 导出 · DEBUG 日志</summary>

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

</details>

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

## 与 Langfuse 集成

atap 可以作为**活 Langfuse 实例的外部评估管线**：按标签/时间窗拉取 trace，跑你的
分析/分类/归因栈，把结果写回原处——失败归因直接出现在你自己的 Langfuse 面板上。

| Score | 落点 | 内容 |
|---|---|---|
| `atap:root-cause` | trace | 根因码（categorical） |
| `atap:confidence` | trace | 置信度（numeric） |
| `atap:blamed-step` | 被归因的 observation | agent @ step + 根因 |
| score `metadata` | 每条 score | 完整 `Hypothesis` + 批次标识（`run_id` / `llm`），多次评估可区分 |

```bash
pip install -e ".[llm,langfuse]"
export LANGFUSE_BASE_URL=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
atap langfuse-eval --config configs/langfuse_eval.yaml --out runs/lf1 \
    --tags production --since 24h --dry-run    # 先空跑确认，去掉该旗标即真写
```

`--dry-run` 只打印不发请求；此前批次已完整评估的 trace 自动跳过（trace 级
`atap:root-cause` 最后写入、充当完成标记，被打断的半批次下次自动重评；`--force`
无条件重评）。凭据只从环境变量读取。自建演示实例（`atap langfuse-push` 播种）与
完整 round-trip 见 [docs/集成指南_Langfuse.md](docs/集成指南_Langfuse.md) 和
`docker-compose.langfuse.yml`。

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
`atap list` 即见。完整示例见
[docs/算法清单.md](docs/算法清单.md#新增自己的算法)。

## 路线图

- [x] 阶段四A 确定性层：`idg` / `hierarchy_tree` / `rg_ug` / `drift_detect` / `inducer` + taxonomy accept
- [x] 阶段四B LLM 表征与归因升级：`claim_ledger`+`claim_audit`（DRIFT）、`tree_diagnosis`（CodeTracer）、`hcg`+`chief`（CHIEF）
- [x] 阶段四C L3 反事实重放：沙盒 `replay_intervene` 基建、`counterfactual_replay`（TraceElephant）、`dover`（DoVer）
- [x] 阶段四D 采集适配器：Langfuse v3 ingestion、OTel GenAI semconv、`atap export` + roundtrip
- [x] 阶段四E 活实例桥接：`atap langfuse-eval`（拉取 → 流水线 → Score 回写）+ `atap langfuse-push`
- [ ] SBFL 作为 L2 先验的实际融合（当前为独立算法）；AgenTracer 式 GRPO 微调 tracer
- [ ] 沙盒演进 —— 从玩具研究问答沙盒走向更真实的多场景执行环境（更丰富的任务类型、真实工具调用、更多故障注入）
- [ ] 真实数据集评测 —— 公开 Who&When 基准第一轮已完成（184 条真实失败轨迹，见
  [docs/benchmark_whoswhen_2026-08-30.md](docs/benchmark_whoswhen_2026-08-30.md)）；
  扩展到更多方法与数据集的工作仍在继续

架构与契约：[docs/architecture.md](docs/architecture.md) ·
验证状态：[docs/validation.md](docs/validation.md) ·
算法清单：[docs/算法清单.md](docs/算法清单.md) ·
基准报告：[docs/benchmark_whoswhen_2026-08-30.md](docs/benchmark_whoswhen_2026-08-30.md) ·
详细计划：[docs/plan.md](docs/plan.md) · [docs/plan_阶段四.md](docs/plan_阶段四.md) ·
集成指南：[docs/集成指南_Langfuse.md](docs/集成指南_Langfuse.md) ·
开发日志：[docs/README_dev_log.md](docs/README_dev_log.md)
