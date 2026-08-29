# 计划:集成层功能性调整(Langfuse 活实例闭环 → OpenInference → 分发形态)

> 状态:草案(2026-08-29);**Phase 0 + Phase 1(M1)已于同日实现**,详见文末
> "实现记录"。目标:把 atap 从"文件进、文件出的离线批处理"升级为
> "现有可观测性栈之上的归因 + 恢复层",同时不破坏分层不变量与离线确定性测试。

## 0. 现状盘点(代码事实)

- `io/` 是纯文件适配:`LangfuseTraceSource` / `OTelTraceSource` 读的是导出的
  ingestion-batch / OTLP JSON 文件,`export_langfuse` / `export_otel` 写 JSON 文件
  (`src/io/langfuse.py`、`src/io/otel.py`)。**没有任何网络代码,也没有 Langfuse SDK 依赖**;
  基础依赖只有 pyyaml + pydantic。
- `build_source`(`src/io/jsonl_store.py`)只认 `jsonl / langfuse / otel` 三种 source,
  且无条件要求 `path` 字段——API 型 source 无法接入。
- 归因输出统一为 `Hypothesis`(`src/core/schema.py`),经
  `TrajectoryBundle.hypotheses()` 聚合(`src/core/bundle.py`),含
  agent / step / root_cause / root_cause_code / evidence / fix_suggestion / confidence——
  正好一一对应 Langfuse score + comment 所需的全部信息。
- `run_config`(`src/runtime.py`)已经返回 bundles,CLI 层即可拿到每条 trace 的
  hypotheses 做回写——**写回闭环不需要动 core/,不需要动任何算法模块**。
- 分层不变量(`tests/test_invariants.py`):`io/` 不得依赖任何 stage 包;
  `core/` 零 I/O。新集成代码只能放在 `io/`(或新的顶层模块)且只 import `core.schema`。
- 测试纪律:332 个测试全部离线确定性、秒级完成——所有联网功能必须可用
  mock transport + 录制 fixture 离线测试。
- README 双语(`README.md` / `README.zh-CN.md`),docs/ 内计划文档为中文。

## 1. 目标与排序原则

1. **最高 ROI**:Langfuse 活实例 round-trip——按 tag/时间拉 trace → 跑现有
   流水线 → 把归因结果作为 Score + comment 写回原 trace。这是"atap 不是孤岛"
   的直接证明,也是最好的演示素材。
2. **去厂商绑定**:OTel GenAI 已支持,补 OpenInference(Phoenix / LangChain
   instrumentation 的事实标准)入口,即可宣称"只要 trace 符合
   OTel / OpenInference,atap 就能分析"。
3. **分发形态**:Python evaluator 函数(API 消费者)、MCP server(IDE/agent 现场
   调用)、GitHub Action(CI 回归门禁)。按 ROI 排后。
4. **叙事调整**:README 从 "platform" 改述为 "layer",补集成架构图与 round-trip
   演示 GIF。

## 2. Phase 0 — 决策与准备(约 0.5 天)

- **依赖策略**(已定,可复议):新增 optional extra `langfuse = ["httpx>=0.27"]`,
  只走公开 REST:
  - 拉取:`GET /api/public/traces`(分页)、`GET /api/public/observations?traceId=`
  - 回写:`POST /api/public/scores`(v3/v4 稳定)
  - 演示播种:`POST /api/public/ingestion`(v3 batch;云 v4 已宣布弃用、
    2026-11 关停,但 self-hosted v3 长期可用)
  不引官方 `langfuse` SDK(重、v3→v4 版本漂移大);云 v4 的 OTLP 推送列为
  后续迁移项,不进本轮。
- **演示环境**:`docker-compose.langfuse.yml` 起 self-hosted Langfuse(v3 兼容)
  + demo project。演示不依赖任何云账号。
- CI 依赖矩阵:base / dev / llm / langfuse 各跑一遍(离线,不含真实端点)。

## 3. Phase 1 — Langfuse 活实例回写闭环(核心,约 2–3 天)

### 3.1 新模块 `src/io/langfuse_live.py`

只 import `core.schema` + `io/_leak_guard`(满足不变量),包含三块:

**(a) `LangfuseAPISource`(实现 `TraceSource` 协议)**

- `__init__(base_url, auth, tags=None, since=None, limit=None, outcome_from=None)`
  ——凭据只从环境变量读(`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
  `LANGFUSE_BASE_URL`),不落盘(与 llm/ 的 key 纪律一致)。
- `load()`:分页拉 trace + observations → 复用现有 observation 树 → span 树
  映射逻辑,落到 `Trajectory(raw={"spans": ...})`,由 `represent/canonical_events`
  展平——**层间契约不变**。
- **通用映射(真正的增量)**:真实用户的 observation 没有 `metadata["atap"]`
  命名空间,需要一套无 atap 元数据的推导规则,全部可配置:
  - `GENERATION → LLM_CALL`,`SPAN → TOOL_CALL/TOOL_RESULT`(按 name/属性启发),
    `EVENT → VERIFIER/TASK_END`;
  - `name → action`;agent 从可配置键链取(如 `metadata.agent` →
    `attributes["gen_ai.system"]` → 兜底 `"unknown"`);
  - `outcome_from`:从某条 score 推导成功与否(如
    `{score: "user_feedback", op: ">=", value: 1}`);取不到时保守默认
    `success=False`——全部 trace 进入 analyze,由 judge 判定,不静默漏分析。
- 保留 **event-id ↔ observation-id 映射表**,挂在 source 实例上,供 observation 级
  score 定点回写。

**(b) `ScoreWriter`**

- `Hypothesis → score payload`:
  - trace 级 categorical:`atap:root-cause` = `root_cause_code`(或 MAST 码),
    comment = `root_cause + fix_suggestion + evidence 摘要`;
  - trace 级 numeric:`atap:confidence`;
  - observation 级 categorical:`atap:blamed-step` 写到归因 step 对应的
    observation 上(用 (a) 的映射表)。
- **防泄漏**:comment 组装走 `_leak_guard` 同一纪律——
  `injected_fault / origin_fault / fault_removed / qrels` 绝不进 score;
- **幂等**:默认跳过已带 `atap:*` score 的 trace(拉取时顺带取回的 score 列表
  复用,不额外发请求);`--force` 才重写。
  [实现时变更] 原计划的 `langfuse_cursor.json` 水位游标**取消**——skip-scored +
  `--since` 已覆盖增量语义,且无需状态文件(每次 eval 的 `--out` 本来就必须是
  新目录,游标文件反而要额外安置)。

**(c) demo 播种 `langfuse_push`**

- 把 `export_langfuse` 的 batch POST 到 `/api/public/ingestion`,
  用于把 atap 沙箱语料推上自建实例,让 round-trip 演示自包含。

### 3.2 对现有代码的最小改动

| 文件 | 改动 |
|---|---|
| `src/io/jsonl_store.py` | `build_source` 改为按 type 校验字段:`langfuse_api` 不要求 `path`(现在是无条件 `if "path" not in spec: raise`) |
| `src/cli.py` | 新增 `atap langfuse-eval`(拉 → `run_config` → 遍历 `b.hypotheses()` 写回;`--dry-run` 只打印;`--config` 复用现有栈,如 chief)与 `atap langfuse-push` |
| `pyproject.toml` | 增加 `langfuse` extra;packages 列表不变(无新子包) |
| `core/`、各算法模块 | **零改动**(闭环所需数据 `run_config` 已返回) |

### 3.3 测试(全部离线确定性)

- `httpx.MockTransport` + `tests/fixtures/langfuse_live/` 录制响应:
  - 通用映射单测(带 / 不带 atap ns 的 observation 树;agent 键链兜底;
    outcome_from 推导与保守默认);
  - score 格式化、防泄漏(qrels/GT 键不出现)、游标幂等、dry-run 零请求;
  - 回写后跳过逻辑(第二次 eval 不重复写)。
- 不变量测试扩展:`langfuse_live` 的 import 集合只含 `core.schema` / `io._leak_guard`。
- 真实端点测试标记 `@pytest.mark.live`,默认 deselect(CI 与本地默认都不跑)。

### 3.4 交付物

- `docker-compose.langfuse.yml` + `docs/` 集成指南(拉 → 分析 → 回写全流程);
- **60–90s GIF**:归因 score / confidence / blamed-step 实时出现在真实 Langfuse
  trace 页面上——同时用作 README 头图与社区推广素材;
- README 双语更新(见 Phase 4 的叙事调整可提前借这次落地一小部分:头图 + 一个
  "External evaluation" 小节)。

## 4. Phase 2 — OpenInference 入口 + Python evaluator API(约 1–2 天)

- `src/io/openinference.py`:`openinference.*` 属性(span.kind、
  `llm.*` / `tool.*` / `retrieval.*` 属性族)→ R0,对齐 `otel.py` 的文件式适配与
  roundtrip 测试纪律(Phoenix 导出 / OTLP JSON 文件进)。
  - 映射要点:`SPAN_KIND → kind`(LLM/TOOL/RETRIEVAL/AGENT/CHAIN),
    `openinference.span.kind` 与 message 属性 → payload/content,
    input/output value → payload;refs 无对应物则空(与 Langfuse 路径同策略)。
- `src/evaluator.py`(函数式门面,零 CLI 依赖):
  `attribute_trajectory(trace: Trajectory, config: PipelineConfig | None) -> list[Hypothesis]`
  ——默认装配一套离线栈;供 Langfuse 自定义 evaluator、pytest、Notebook 直接调用。
  只 import `core` + `runtime`,不违反分层。
- 测试:openinference 映射 + roundtrip;evaluator 在沙箱 trace 上返回非空
  hypotheses 且与 `atap run` 同结果。

## 5. Phase 3 — MCP server(约 2 天)

- extra:`mcp = ["mcp>=1.0"]`;入口 `atap mcp`(stdio,FastMCP)。
- 工具面(先窄后宽):
  - `atap_analyze_trace(trace: dict | 路径, config?: 路径)` → 运行
    represent→analyze→classify→attribute(recover 可选),返回结构化归因报告
    (hypotheses + MAST 码 + fix_suggestion);
  - `atap_list_algorithms()`。
- 约束:不写文件到别处、凭据环境变量;README 给 Cursor / Claude 接入示例。
  ("失败归因即 MCP 工具"是本轮最强的求职叙事点之一。)

## 6. Phase 4 — 叙事与分发(2026-08-29 完成)

- [x] README 双语改写:tagline 改为 "the attribution & recovery layer on top of
  your existing agent observability stack";头图换成活实例 roundtrip GIF
  (docs/assets/langfuse_roundtrip.gif,41.6s:Scores 总表 → trace 徽章 →
  blamed-step span),并列集成架构图 docs/assets/integration.svg
  (来源 → 适配器 → 六段流水线 → Score 回写);旧终端 demo.gif 降级为
  quickstart 下方引用。
- [x] GitHub Action:ci.yml 新增 `attribution` job——确定性离线语料上跑
  `atap compare`,对照提交基线 `.github/baselines/attribution.json`
  (18/18 全满),step/agent/MAST 命中或恢复数回退即 fail;
  `scripts/check_attribution_baseline.py` 校验(改进通过,应回写基线)。
- [ ] (可选)FastAPI sidecar:`POST /traces → scores`,供 Langfuse/Phoenix
  webhook(明确不做本轮)。

## 7. 明确不做(本轮)

- LangSmith / AgentOps / 框架埋点(LangGraph、CrewAI…)adapter——等 OpenInference
  验证映射模式后横向复制;
- Langfuse 云 v4 的 OTLP 推送路径——记一个迁移 ticket,2026-11 云关停前评估;
- SBFL 融合为 L2 先验、沙箱演进、真实数据集评估——维持原 roadmap 排期,
  不与集成线混排。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 真实 trace schema 长尾(agent 名取不到、无 outcome) | 映射键链全部可配置 + 保守默认(无 outcome → 全量进 analyze 由 judge 判定),并在 dry-run 输出映射命中率 |
| score 值域 / categorical config 限制 | v3 自由 categorical 可直接写;comment 过长截断并提示完整信息在 artifacts |
| 写错对象(重复回写污染) | 幂等跳过 + 水位游标 + `--dry-run` 默认建议 |
| GT 泄漏进外部系统 | 复用 `_leak_guard` deny-list,并新增回归测试覆盖 score 路径 |
| 破坏离线测试纪律 | 全部 live 功能走 MockTransport + fixture;真实端点测试 `live` 标记默认不跑 |
| httpx 引入供应链面 | 仅 `langfuse` extra,base 安装零新增依赖 |

## 9. 里程碑与工作量

| 里程碑 | 内容 | 预估 |
|---|---|---|
| M1 | Phase 0+1:Langfuse 活实例闭环 + 自建实例演示 + GIF | 3 天 |
| M2 | Phase 2:OpenInference adapter + evaluator API | 1.5 天 |
| M3 | Phase 3:MCP server | 2 天 |
| M4 | Phase 4:README 叙事 + 架构图 + Action | 1.5 天 |

总计约 8 天;M1 单独成立即可发布"集成层"叙事,M2–M4 可按反馈插拔。

## 10. 实现记录(2026-08-29,M1 完成)

| 项 | 落点 |
|---|---|
| 核心模块 | `src/io/langfuse_live.py`:`LangfuseClient`(lazy httpx、Basic auth、分页信封)/ `LangfuseAPISource`(通用映射 + outcome_from)/ `ScoreWriter`(三种 score、防泄漏、dry-run/skip/force)/ `push_langfuse` |
| source 扩展 | `build_source` 支持 `langfuse_api`(按 type 校验字段,不再无条件要求 path) |
| CLI | `atap langfuse-eval` / `atap langfuse-push`;config 的 `source` 块(`outcome_from`/`agent_keys`)与 CLI 旗标合并 |
| 依赖 | `langfuse` extra = httpx;dev 也带 httpx(离线 mock 测试用) |
| 配置 | `configs/langfuse_eval.yaml`(openai 栈;换 `{type: fake}` 即零网络彩排) |
| 演示环境 | `docker-compose.langfuse.yml`(上游 v3.0.0 compose 派生,镜像钉 `:3`,内置 demo org/project/pk/sk) |
| 文档 | `docs/集成指南_Langfuse.md`;README 双语新增 "External evaluation" 小节 + 阶段四E 路线图项 |
| 测试 | `tests/test_langfuse_live.py` 12 个用例:映射/过滤/outcome/score 格式/防泄漏/幂等/push↔离线适配互通/push 拒绝 raw-span-only(MockTransport)+ CLI 端到端 写入→跳过→force→dry-run(线程 stub 服务器);全仓 344 测试通过 |
| 不变量 | `io/*.py` 自动被 rule 3 覆盖,零新增例外;`core/` 与算法模块零改动 |
| 顺带修复 | `tests/test_schema_registry_config.py` 两段配置补 `represent/canonical_events`——用户在途的 `requires` 声明批次(judge_eval/mast_judge 等新增依赖)与该存量测试冲突,与本集成无关,按其意图更新了测试 |
| 独立验证 | 独立子代理核查(9 项全 PASS,无 blocker/major)后修复其发现:push_langfuse 对 raw-span-only 显式抛错(原为静默丢事件,守卫+测试)、outcome latest-wins 改按 score timestamp、兄弟排序改 epoch 键、缺 id observation 显式报错、blamed-step 去重按置信度取强、一句恒真测试断言修正 |
| 真实实例验证(同日) | 自建 Langfuse v3.225.5(colima/docker,compose 文件实测可用):push 318 事件/24 trace 秒级入库 → eval(fake 栈)18 失败全归因、54 个 score 回写 → UI 可见 `atap:root-cause: FM-1.3`、`atap:confidence: 0.70`、被归因 observation 上的 `atap:blamed-step: FM-1.3` → 重跑 18 skipped/0 written(幂等)、`--force --dry-run` 54 条零写入。截图 docs/assets/langfuse_roundtrip.png。真实环境发现并修复:①部分 self-hosted 构建忽略 per-trace 过滤参数 → 客户端强制按 traceId 过滤(+回归测试);②atap 自产 trace 增加 metadata.atap 快路径与 outcome 元数据恢复(push→live-pull 无损回环,+等价测试)。环境要点(VM ≥4GB、demo 邮箱需合法格式等)记入集成指南 §7。测试总数 346 |

未做(留给真实实例联调):对着活实例录 60–90s GIF;真实 trace 的 agent 键链
预设值打磨。

### 真实 LLM 实测(2026-08-29,deepseek-v4-flash)

用真实 OpenAI 兼容端点(DeepSeek 官方,key 仅经 `OPENAI_API_KEY`/`OPENAI_BASE_URL`
环境变量注入,未落盘)对同一活实例全量重跑:

- 冒烟(2 trace)→ 全量(24 trace,`--force`):**96 次 LLM 调用全部成功、
  0 传输重试、0 解析重试**;prompt 101,930 + completion 244,079 ≈ 34.6 万
  tokens;单调用 p50 8.2s / p95 96s;全链路 37 分钟(judge_eval 134s、
  mast_judge 1276s、chief 823s,限速间隔 1.5s/调用)。
- 写回:18 失败 trace × 3 = **54 个 score 全部落库**(API 分页全量核对);
  6 条成功轨迹正确零假设零写入;`atap:confidence` 区间 0.95–1.00。
- 质量交叉验证:3 篇论文语料 × 6 种注入故障,chief 对同一故障族的定位
  **完全一致**(如 step_repetition 全部 `searcher@5 + executor_loop`,
  premature_termination 全部 `planner@1 + planning_error`,
  info_withholding 全部 `searcher@8`),且与 fake 伪判官定位一致;真实模型的
  证据句明显更具体(如"step 3 已返回 [d1,d2] 但 step 5/7 重复同样查询")。
- 已知语义(非故障):本配置 attribute 用 `chief`,其假设 `root_cause_code`
  设计上为空(mechanism 词表写在 root_cause 文本前缀),故 `atap:root-cause`
  score 值落 `unlabeled` 兜底、mechanism 与证据在 comment;fake 批次显示
  `FM-1.3` 是因为 eval_fake 配置用的是 `all_at_once`(它才产出 MAST 码)。
  这正是 §11 第 2 条 Score Config backlog 的动机。
- UI 同屏可见两代 score 对照(trace 节点 `atap:confidence: 0.98, 0.70`、
  `atap:root-cause: unlabeled, FM-1.3`,被归因 span 上 `atap:blamed-step:
  blamed, FM-1.3`)。截图 docs/assets/langfuse_real_llm.png;运行产物
  `runs/lf_real_smoke/`、`runs/lf_real_full/`(含 llm_calls.jsonl 全量审计)。
- score metadata 结构化 + 批次标识(同日晚些,§11 第 1 项提前实现):
  `ScoreWriter` 增 `run_meta`,每个 score 的 `metadata` = 完整 Hypothesis
  平铺 + `run_id`(`--out` 目录名)/`run_name`/`llm`/`seed`,comment 头带
  run 标签;CLI `_cmd_langfuse_eval` 组装 run_meta。活实例复跑 2 trace
  验证新旧批次 API 可分;测试 +2(格式/防泄漏扩展、批次区分),全仓 347 绿。
- 全新语料真实评测 + tags 分批工作流(同日):`atap corpus --drift` 生成
  25 条全新轨迹(18 ok / 7 step_repetition,时间窗 w1–w4),`push --tags
  corpus-drift` 入库(378 事件)→ `langfuse-eval --tags corpus-drift` 精确
  圈定、老 24 条零重跑;53 次真实调用零失败(14.8 万 tokens,13 分钟),
  7 条失败全部定位 `searcher@5 + executor_loop + FM-1.3`(与旧语料同故障族
  结论一致),21 个 score 全带 `run_id=lf_drift_eval` 批次标识。项目级批次
  视图:lf_drift_eval(21)/ lf_real_meta(6)/ 无标识旧批(114)。
  `push_langfuse` 增 `tags=` 参数 + CLI `langfuse-push --tags`(tags 是
  分批评测把手;指南 §5 记入工作流)。测试 +1(tags 注入),全仓 348 绿。
- **外发 GT 泄漏修复(同日,截 GIF 时发现)**:活实例 trace metadata 里暴露了
  `qrels`(gold 集合)与 `outcome.note`(沙箱故障机制描述)——根因是
  `_GT_META_KEYS` 只含三个注入故障键,qrels 按离线契约保留但活推路径没剥。
  修复:`export_langfuse(..., external=True)` 严格模式(push_langfuse 采用)
  额外剥 `qrels`、中性化 `outcome.note`;离线文件导出行为不变(rg_ug 依赖),
  push 测试补泄漏断言 + 离线契约断言,348 绿。**存量残留**:活实例上已推的
  49 条 trace 的 metadata 仍带 qrels(PATCH 405、重推只合并不替换、DELETE 会
  连带毁掉 4 批 score)——本机演示实例,接受残留并记录;今后推送全部干净。
  演示 GIF 中含 qrels 的右侧面板已裁掉。
- push 时间戳修复(同日,用户报"traces 列表看不到数据"定位):exporter 为
  离线确定性把事件时间戳钉在 epoch 0,活实例 UI 轨迹列表默认时间窗
  (近 7 天)内查得 0 条——49 条全在但"隐形"。`push_langfuse` 现在在推送时
  给全部事件重打真实墙钟(单次共享 now,保序);存量 49 条仍为 1970,
  UI 里用自定义时间范围(1970 起)或经 Scores 页(真实时间戳)跳转可达。
  测试补时间戳断言,348 绿。

## 11. TODO:score 写回增强 backlog(2026-08-29 真实实例使用后提出)

均在 `ScoreWriter` 内小改动,不触碰管线契约;按价值排序:

- [x] **score `metadata` 塞结构化字段**(2026-08-29 实现):每个 score 的
  `metadata` 平铺写入完整 `Hypothesis`(agent / step / root_cause /
  root_cause_code / responsible_side / evidence / fix_suggestion /
  confidence / source——TODO 所列 6 项的超集,comment 内容全部可机器读取)
  **外加批次标识** `run_id`(= `--out` 目录名,每次运行唯一)/ `run_name` /
  `llm`(如 `openai:deepseek-v4-flash`,与 `fake` 批天然可分)/ `seed`,
  comment 头部同步带 `(by chief, run lf_real_meta)` 标签——Langfuse score
  只追加不更新,同一 trace 多批评测没有批次标识就无法区分(实测同一 trace
  两批真实 LLM 定位 step 9 / step 10 各异,更需要可追溯)。活实例已验证:
  新旧批次 API 一眼可分;
- [ ] **预创建 Score Config**:为 `atap:root-cause` / `atap:blamed-step` 定义
  MAST 14 码(加 `unlabeled`)的类别集合(`POST /api/public/score-configs` 或
  UI 手动),面板即可按类别上色/过滤/做分布仪表盘,并挡住自由字符串漂移;
- [ ] **BOOLEAN score `atap:recovered`**:闭环(closed_loop)复验结果回写——
  恢复重跑通过与否,语义天然是布尔;当前 eval 栈不含 recover,待闭环能在
  活实例语境下执行时启用;
- [ ] (可选)成功轨迹写 `atap:root-cause=clean` 标记,便于面板上区分
  "评过且干净" 与 "没评过";
- [ ] (可选)低于置信阈值不写的 `--min-confidence` 门槛(默认关,保持透明
  自曝语义)。
