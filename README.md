# atap — Agent 轨迹分析与错误归因平台

模型设置：nvidia/nemotron-3.5-lightning:free（密钥走环境变量 `OPENAI_API_KEY`，绝不落盘。历史版本曾在这一行误贴明文 OpenRouter key——该 key 应视为已泄露并**立即在 OpenRouter 后台吊销**（此事代码层无法代办）；本地 git 历史已于 2026-08-27 用 `git filter-repo --replace-text` 全量清洗并复验无残留，清洗前完整备份在仓库外 `../atap-pre-cleanup-backup.bundle`，清洗时仓库无任何远端、未曾外推。）


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
.venv/bin/python -m pytest tests/          # 317 个测试（含离线全链路 e2e 与审计回归）
.venv/bin/atap demo                        # 离线全链路演示（FakeLLM，确定性）
.venv/bin/atap run --config configs/pipeline_offline.yaml
# 阶段三全栈（R5+循环谓词+L0 规则包+二分+反馈注入，频谱语料 24 条）：
.venv/bin/atap corpus --out runs/corpus/traces.jsonl
.venv/bin/atap run --config configs/pipeline_offline_v3.yaml --out runs/v3
# 同一轨迹集上的算法组合对比：
.venv/bin/atap compare --config configs/pipeline_offline_v3.yaml \
                        --config configs/pipeline_sbfl.yaml --out runs/compare
# 阶段四确定性层全栈（R2 依赖图+R4 层级树+RG/UG+漂移+inducer）：
.venv/bin/atap run --config configs/pipeline_offline_v4.yaml --out runs/v4
# 分布漂移监控（三类漂移构造语料 22 条）：
.venv/bin/atap corpus --drift --out runs/drift/traces.jsonl
.venv/bin/atap run --config configs/pipeline_drift.yaml --out runs/drift
# L3 do-then-verify 恢复闭环（DoVer：消息原位替换重放 ×3 + 里程碑差分）：
.venv/bin/atap run --config configs/pipeline_dover.yaml --out runs/dover
# 采集格式导出（Langfuse v3 ingestion / OTel GenAI）与回导 roundtrip：
.venv/bin/atap export --traces runs/corpus/traces.jsonl --format langfuse --out runs/export.json
# 真实 LLM：设置 OPENAI_API_KEY（可选 OPENAI_BASE_URL）后
.venv/bin/atap run --config configs/pipeline_llm.yaml
# 过程日志 DEBUG 级（默认 INFO）：
.venv/bin/atap -v run --config configs/pipeline_offline.yaml
```

### 统一日志与调用审计

每次 `run / demo / compare` 自动落两份记录（`atap.log` + `llm/call_log.py`）：

* `runs/<name>/run.log` —— 过程日志（`atap` logger：run 起止/各阶段耗时/
  验收数字），同时输出到 stderr；stdout 只保留命令结果；`-v` 开 DEBUG；
* `runs/<name>/llm_calls.jsonl` —— **每次 LLM 调用一条审计记录**：ts /
  client / tag / model / schema / latency_ms / messages（完整 prompt）/
  response / usage（真模型 token 用量）/ ok+error（失败含错误）。Fake 与
  OpenAI 客户端共用同一挂件（`attach_call_log`），compare 的计数包装向
  内层透传；库形态默认不挂载（行为不变）。

```bash
.venv/bin/python -c "import json,collections; \
recs=[json.loads(l) for l in open('runs/demo/llm_calls.jsonl')]; \
print(len(recs), dict(collections.Counter(r['tag'] for r in recs)))"
# 25 {'judge_eval': 13, 'mast_judge': 6, 'all_at_once': 6}
# （重跑同目录，审计文件被截断重置，仍是 25 —— attach_call_log 一次 run 一次挂载）
```

`atap demo` 的离线验收结果（seed=7，六种注入故障）：

```
归因命中: step 6/6  agent 6/6  MAST 6/6  恢复 6/6
round0: traces=7 failures=6 attributed=6 reruns=6(ok=6)
round1: traces=6 failures=0   ← 只有 6 条重跑轨迹进入闭环验证轮，全部通过
```

> 说明（评审修复 2026-08-27）：验证轮**只喂重跑轨迹**——无 rerun 的原始
> 轨迹（成功轨迹、未被恢复接管的失败轨迹）第一轮已经判过，重跑它们等于
> 为相同工作再付一遍调用费（旧版 round1 traces=7、共 26 次调用；现
> round1 traces=6、25 次调用，judge_eval 14→13）。

> ⚠️ **离线"恢复 N/N"数字是构造性回环，不衡量判官/恢复能力**（与上文 CHIEF
> 18/18 的 ⚠️ 同级别的 caveat，适用于 demo/阶段三/阶段四C 各处出现的"恢复
> 6/6"与闭环 failures=0）：伪判官的 fix 文案在 `llm/pseudo_judge.py` 的
> fixes 字典中固定内嵌故障名，targeted_rerun/feedback_injection 将其原样
> 作为反馈注入，沙盒按"反馈文本是否含故障名关键词子串"判定故障移除。因此
> 离线恢复率是**归因命中的确定性函数**——只要 step 定位命中，恢复必然命中；
> 它衡量的是管路契约（Hypothesis→反馈→重放→验证的链路正确性），相对
> step/agent 命中率不含独立信息。真实模型的恢复数字见下方各轮 realtest
> 记录（曾出现 恢复 0/6 → 修复消费机制后 6/6 的真实差异）。

阶段三全栈（24 条语料）离线验收：二分定位 **step 5/6、agent 6/6**；
SBFL 频谱 4/6（miss 两例为方法边界：一次性动作被 γ 压制、内容级故障在
动作谱不可见）；反馈注入恢复 6/6、闭环 failures=0；对比表
`v3 15/18·141 calls vs SBFL 12/18·42 calls`（v3 的调用数为验证轮收窄后
的口径：只重跑 rerun 轨迹；147 为旧口径——验证轮整轮重跑）。

阶段四A 确定性层离线验收（轮次三，2026-08-25）：**179 测试全绿**
（154 旧 + 25 新，v3/sbfl/demo 数字全部回归不变）——
* `rg_ug`：六故障 + detour 构造标签 7/7 全中（RG_directional×2、
  RG_last_hop×1、UG_true_extraction×4；UG_boundary 由合成 fixture 覆盖），
  18 条故障轨迹 step **15/18**、agent 15/18（miss 3 例均为 step_repetition
  的已知映射边界：重复调用未被"利用"，UG 步映射落到 compose 而非首次
  重复步——轨迹级标签不受影响），零 LLM 调用；
* `idg`：四类故障 GT 根因 ∈ 终态祖先闭包；两例 honest miss（step_
  repetition 重复结果从未被引用、premature 提交无引用边——"信息缺失/
  浪费"型故障天然不在依赖链上，由 R5/rule_pack 补位）；
* `hierarchy_tree`：兄弟/子节点规则 golden 通过，tree.md 每步一行压缩
  渲染（13 事件 → 8 行）；
* `drift_detect`：漂移语料三类对照族全检出（version：共享支撑 PSI=2.79
  + 支撑失配 0.95；behavior：支撑失配 0.14；data：PSI=0.73 + 支撑失配
  0.67）。统计口径已修正：PSI 只算两侧非空桶（ε 平滑空桶曾把 11.46 里的
  8.67 变成平滑常数的产物，换 ε 数值即漂移），空桶质量单列为可解释的
  support_mismatch；分组 n<5 跳过（对照组过小不作漂移证据）；语料中长度
  与成败完全共线，artifact 以 feature_aliases 报告并在告警特征里去重
  （同一信号不重复计数）。稳定语料零误报（单分组空对照；双窗口同分布
  的真实零误报见 test_stable_multiwindow_zero_false_alarms）；
* `inducer`：deadlock 残差 ×3 轨迹 → 恰好 1 提案（命名来自症状实词、
  不含故障类型词），`atap taxonomy accept` 人工接受后 mast_judge 经
  extra_modes_file 用新码打标（闭环验证）；全可标语料 0 提案。

阶段四B（LLM 表征与归因升级）离线验收（轮次四，2026-08-25）：**191 测试
全绿**（179 旧 + 12 新）——
* `chief`（CHIEF）：六故障 **step 6/6、agent 6/6**，机制映射全对
  （executor_loop/dataflow_first_pollution/planning_error/local_error）；
  18 条故障语料 step **18/18**（vs v3 二分 15/18。⚠️ 该对比在离线 FakeLLM
  下量的是**伪判官 handler 的信息量差异**——`chief_localize` 与
  `all_at_once` 的 handler 都返回全局首个症状签名的 step，而 `binary_search`
  的 handler 只看当前区间，因此 18/18 vs 15/18 不构成算法间比较；"CHIEF
  step 级迄今最高"的文献结论需真实 LLM 复跑验证）；每轨迹 3 次 LLM 调用（oracle+eval+localize）；
* `tree_diagnosis`（CodeTracer）：六故障 step 6/6，树级 stage 定位 6/6，
  下钻渲染行数 < 全量渲染（先树后钻的省 token 证据）；18 条语料 step
  18/18·36 calls；
* `claim_audit`（DRIFT）：主场故障 4/4（withholding/premature/ungrounded/
  disobey step+agent 全中）；两例 honest miss 为方法边界（step_repetition
  全部主张为真、malformed 的"无结果"主张为真且 Tracer 禁选工具步——
  非 claim 级故障）；约束主张覆盖 Answer Format Error 家族；18 条语料
  12/18·57 calls；
* 防泄漏回归扩展：4B 全部 8 个 prompt 无故障类型词/GT 键。

阶段四D（采集适配器）离线验收（轮次六，2026-08-25）：**208 测试
全绿**（201+7）——`io/langfuse`（v3 ingestion batch 导入/导出，v4 OTLP
迁移窗口双支持）与 `io/otel`（OTLP JSON + gen_ai 语义约定，operation.name
映射 + 未映射属性入 atap.* 防丢失）；**roundtrip 字段级语义等价**（事件
树/kind/agent/action/payload/引用边数/phase/outcome/qrels；GT 不导出防
泄漏）；`atap export` CLI + build_source 双新类型分发；导入轨迹经
canonical_events 拍平后跑 4A 确定性栈结果不变（rg_ug 仍 UG_true_extraction）。

阶段四论文一致性审计（2026-08-25）：11 路只读 subagent 逐模块对照
refs/ 原文（报告 `docs/audit_阶段四论文一致性_2026-08-25.md`，与阶段一~三
的 `docs/audit_论文一致性_2026-08-25.md` 互补）。结论：机制层高保真
（rg_ug 逐式一致、CodeTracer 挂载规则对齐、CHIEF/DoVer/DRIFT 主干
忠实）；2 处 🔴 已修复（dover classify 运行期回显故障名 →
`_redact_fault_names` + 运行期防泄漏回归；otel traceId/spanId 违反
OTLP hex → sha256 派生 + 原id 走 atap.* 生存），约 30 处标注失实/
缺口已修正（idg α 无据、drift PSI 归因、claim_audit b_k 措辞等）。
**210 测试全绿，demo 验收数字不变。**

阶段四C（L3 反事实重放）离线验收（轮次五，2026-08-25）：**201 测试
全绿**（191+10）——沙盒新增 `replay_intervene`（检查点重放+消息**原位
替换**，horizon 窗口/n_repeats 参数化；无编辑效应时后缀逐事件不变的不
变量测试）；`counterfactual_replay`：六故障 GT 候选全 validated、症状步
候选 refuted（置信下调滤伪因果）；`dover`：六故障 mistake=GT、Validated、
恢复 6/6（每干预 ×3 重放）；e2e `pipeline_dover.yaml` 闭环——18/18·
18/18·108 calls，闭环改善 18/18。

真实 LLM 验证（两轮）。**DeepSeek `deepseek-v4-flash`**（temperature=0）：
阶段二栈 **step 6/6、agent 6/6、MAST 主标签 3/6、恢复 6/6**——"恢复 0/6"
的已知限制已由沙盒"关键词优先、LLM 语义兜底"反馈消费修复确认（六故障
全部 1 轮恢复、fault_removed=True）；阶段三栈 binary_search step 2/6——
**与文献方向相反**（Who&When Table 1 step 级四列全部是二分 > 单遍，如
With-GT 23.98 vs 12.50、w/o GT 16.59 vs 13.53；此前 README 写"二分弱于
单遍方向一致"系文献方向错引，模块 docstring 的记载才是对的）。候选解释：
短轨迹下判官 lower-half 偏置会把区间塌缩到早期步）、feedback_injection
恢复同样 6/6。此前 OpenRouter
`stealth/ox-alpha`：step 4/6、agent 5/6、恢复 0/6。离线 6/6 证明的是框架
管路与契约正确，真实数字衡量判官能力，两者互补。复跑：
`docs/realtest_deepseek.py` / `docs/realtest_nemotron.py`（密钥走环境变量，不落盘）。

真实 LLM 验证（第三轮，2026-08-26，OpenRouter `realtest_*.yaml` 四档）：
**ox-alpha** 冒烟栈 step 5/6·agent 6/6·MAST 3/6·**恢复 6/6**（闭环二轮
失败 0；自由文本修复建议触发 `feedback_match` 语义兜底 +6 次）；**ox-alpha
chief** 定位 **step 6/6·agent 6/6**（超过 all_at_once；code=None 为该算法
设计，与离线一致）；**nemotron-3-ultra** 冒烟栈 **step 6/6·agent 6/6·
code 3/6·MAST 3/6·恢复 6/6**（追平 FakeLLM 定位基线，中位时延 18s）；ox
claim 覆盖 5/6、step 3/5。本轮工程产出：配额记账 `http_requests` 入
llm_calls.jsonl；**200+choices=null 偶发** 视作可重试错误 + 非 LLMError
异常也入审计（216 测试）；claim 栈 `max_completion_tokens: 8192`
（4096 截断 JSON 实测）。配额事实：OpenRouter 50 次/日为**账户级
free-models-per-day**（所有 `:free` 共享、非每模型），`stealth/ox-alpha`
独立池（96+ 次 HTTP 无 429）。

上线前真实全量测试（2026-08-25，deepseek-v4-flash 直连，`configs/final_*.yaml`
八档 → `runs/final/`，审计报告 `docs/audit_上线前真实测试_2026-08-25.md`）：
**7/8 档 exit 0、594 条调用零业务失败、判官 prompt 泄漏 0 命中、人工抽检
无幻觉**。P1 过线 6/7——smoke 6/6·6/6·恢复 6/6；**chief 17/18·18/18**（真实
模型最佳）；claim 覆盖 14/18；tree 14/18；dover 恢复 18/18；v3 闭环 18/18；
binary_search 3/18 未达（同配置 FakeLLM 基线 15/18 → 判官 lower-half 偏置，
非管线缺陷，建议降级辅助）。思考型模型实测：8192 预算下 mast_judge
(allow_novel)/长语料偶发思考链爆炸截断（自愈或档级 16384）。⑧ smoke-corpus
因账户 402 余额耗尽差 ~60 次调用，充值后一条命令收尾。一键复审计：
`docs/realtest_audit.py <run 目录...>`（P0 泄漏扫描/P2 指标/GT 命中率）。

## 已实现算法（轮次一：阶段一+二；轮次二：阶段三；轮次三：阶段四A 确定性层）

| 流程 | 算法 | 文献 |
|---|---|---|
| 表征 | `canonical_events`（R0：span 树→统一事件流） | AgentDebugX 2607.18754 |
| 表征 | `ssf`（R1：显著性折叠，可逆占位符+摘要） | TrajAudit 2605.26563 |
| 表征 | `action_signature`（R5：九类动作+参数指纹+七效果标签+锚集/里程碑/LCS） | TraceProbe 2607.06184 |
| 表征 | `idg`（R2：信息依赖图，refs 直出 usage 边，零 LLM） | GraphTracer 2510.10581（撤稿，仅思想） |
| 表征 | `hierarchy_tree`（R4：探索=兄弟/改状态=子节点 + stage 索引 + tree.md 压缩渲染） | CodeTracer 2604.11641 |
| 表征 | `claim_ledger`（R3：claim 六元组台账，LLM 全局单遍，3-5 条紧凑） | DRIFT 2606.02060 |
| 表征 | `hcg`（R2+：层次因果图三层节点 + sub/agt/step 三类边，确定性构建） | CHIEF 2602.23701 |
| 分析 | `judge_eval`（LLM-as-judge 质量分+finding，few-shot） | MAST 2503.13657 / Agent-as-a-Judge 2410.10934 |
| 分析 | `loop_detect`（循环检测谓词：search loop/re-read churn/tool oscillation/redundant search，确定性免 LLM） | TraceProbe 2607.06184 |
| 分析 | `drift_detect`（version/data/behavior 三族漂移对照，共享支撑 PSI + 支撑失配 + 最小分组门槛 + 共线特征去重，工程选型，确定性免 LLM） | 系统级 taxonomy 2511.19933 |
| 分类 | `mast_judge`（MAST 3 类 14 模式打标，词表校验；`allow_novel` 残差通道） | MAST 2503.13657 |
| 分类 | `rule_pack`（L0 免费规则包：畸形调用/无进展循环/过早成功声明/无效输出，确定性免 LLM） | AgentDebugX 2607.18754 |
| 分类 | `inducer`（残差聚类→新错误模式提案，人工闸门 `atap taxonomy accept`，永不自动生效） | AgentDebugX 2607.18754 §3.4 |
| 归因 | `all_at_once`（L1 单遍全轨迹 → 统一 Hypothesis 输出） | Who&When 2505.00212 |
| 归因 | `binary_search`（L2 二分定位，⌈log₂n⌉ 轮，只展示下半段） | Who&When 2505.00212 |
| 归因 | `sbfl`（L0 频谱先验：γ/β/α/λ-decay+Kulczynski2^λ，`run_corpus` 跨轨迹聚合） | FAMAS 2509.13782 |
| 归因 | `rg_ug`（L0 确定性：qrels 集合运算判 RG/UG 四子类+episode 效用，零 LLM） | 搜索智能体诊断 2608.01913 |
| 归因 | `claim_audit`（R3 消费：支撑四级→专家审计→保守回溯→first_error_span） | DRIFT 2606.02060 |
| 归因 | `tree_diagnosis`（R4 消费：tree.md 树级定位→stage 区间下钻，两次调用） | CodeTracer 2604.11641 |
| 归因 | `chief`（HCG 消费仅 subtask 区间切分：oracle 合成→逆拓扑 F_eval 回溯→渐进因果筛选；图结构 E_step/E_agt/OTAR/Φ 未入任何判官 prompt） | CHIEF 2602.23701 |
| 归因 | `counterfactual_replay`（L3 终审：候选步消息干预重放 k=3 窗口，滤伪因果） | TraceElephant 2604.22708 |
| 恢复 | `targeted_rerun`（保留前缀、从 t* 带反馈重跑 ≤5 轮） | AgentDebug 2509.25370 |
| 恢复 | `dover`（do-then-verify：trial 切分→最小消息干预→原位替换重放 ×3→里程碑差分四判定） | DoVer 2512.06749 |
| 恢复 | `feedback_injection`（归因反思注入下一轮完整重解，3 轮） | AgenTracer 2509.03312 |

配套：`sandbox/` 玩具研究问答沙盒（planner→searcher→reporter，
mock 检索语料 + qrels 两层标注（E/G）+ 验证器 + 六种故障注入 + 两种
扩展故障（retrieval_detour=RG last-hop 靶、agent_deadlock=inducer 残差
靶）+ 漂移语料生成器，标签按构造已知——AgenTracer 路线 B 思想）；
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

## 路线图（阶段四后续轮次）

- [x] 轮次三（阶段四A 确定性层）：idg / hierarchy_tree / rg_ug /
      drift_detect / inducer + taxonomy accept ✅ 2026-08-25
- [x] 轮次四（阶段四B LLM 表征与归因升级）：claim_ledger+claim_audit
      （DRIFT）、tree_diagnosis（CodeTracer）、hcg+chief（CHIEF）✅ 2026-08-25
- [x] 轮次五（阶段四C L3 反事实重放）：沙盒 replay_intervene 基建、
      counterfactual_replay（TraceElephant）、dover（DoVer）✅ 2026-08-25
- [x] 轮次六（阶段四D 采集适配器）：io/langfuse（v3 ingestion 导入/
      导出）、io/otel（gen_ai semconv）、atap export + roundtrip ✅ 2026-08-25
- SBFL 作为 L2 先验的实际融合（当前为独立算法）、AgenTracer GRPO
      微调 tracer 路线

详细计划见 [docs/plan_阶段四.md](docs/plan_阶段四.md)，进度见 [docs/plan.md](docs/plan.md)。
