# 集成指南:Langfuse 活实例外部评估闭环

> 对应计划:[plan_集成.md](plan_集成.md) Phase 1(M1)。
> 一句话:**从你的 Langfuse 里拉 trace → 跑 atap 六段流水线 → 把归因结果作为 Score
> 写回原 trace**,失败归因直接出现在你自己的 Langfuse 面板上。

这与 `io/langfuse.py`(离线文件适配:ingestion-batch JSON 进出)不同——
`io/langfuse_live.py` 通过公开 REST API 与**运行中**的 Langfuse 交互,扮演的是
Langfuse 官方为第三方评估器预留的 "external evaluation pipeline" 角色。

## 0. 安装

```bash
pip install -e ".[llm,langfuse]"     # langfuse extra 只增加 httpx
```

凭据只从环境变量读取、绝不落盘(与 `llm/` 的 key 纪律一致):

```bash
export LANGFUSE_BASE_URL=http://localhost:3000     # 或你的实例地址
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
```

## 1. 演示 round-trip(自建实例,约 5 分钟)

```bash
# 1) 起一个自带 demo org/project/密钥的 Langfuse v3(端口 3000)
docker compose -f docker-compose.langfuse.yml up -d
export LANGFUSE_BASE_URL=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-demo
export LANGFUSE_SECRET_KEY=sk-lf-demo

# 2) 把 atap 沙箱语料推上去(v3 ingestion batch;GT 字段已剥离)
atap corpus --out runs/lf/traces.jsonl
atap langfuse-push --traces runs/lf/traces.jsonl

# 3) 先空跑看会写什么(不发任何请求)
atap langfuse-eval --config configs/langfuse_eval.yaml --out runs/lf/eval1 --force --dry-run

# 4) 真跑:拉取 → 分析/分类/归因 → 写回 Score
atap langfuse-eval --config configs/langfuse_eval.yaml --out runs/lf/eval1 --force

# 5) 打开 http://localhost:3000(demo@local / demo-demo-demo),
#    任意 trace 上应出现 atap:root-cause / atap:confidence,
#    被归因的那一步 observation 上出现 atap:blamed-step
```

注意:每次 eval 的 `--out` 必须是**新目录**(run 目录携带上一次运行的证据,
重跑会破坏审计记录,这是 `ensure_fresh_run_dir` 的既有契约)。

真实 LLM 实测(2026-08-29,deepseek-v4-flash):24 trace 全量 96 次调用
零失败零重试、约 34.6 万 tokens、37 分钟,54 个 score 全部落库;定位质量
与成本明细见 `docs/plan_集成.md` §10,UI 对照截图
`docs/assets/langfuse_real_llm.png`。key 只经 `OPENAI_API_KEY` /
`OPENAI_BASE_URL` 环境变量注入,不写入任何文件。

用假裁判零成本彩排:把 `configs/langfuse_eval.yaml` 的 `llm` 换成
`{type: fake}` 即可,全流程离线确定。

## 2. 对自己的 Langfuse 实例做外部评估

```bash
export OPENAI_API_KEY=sk-...                       # 评估用的模型
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # 可选
atap langfuse-eval --config configs/langfuse_eval.yaml \
    --out runs/prod-eval-$(date +%m%d-%H%M) \
    --tags production --since 24h --limit 50
```

- `--tags a,b`:AND 语义的客户端过滤;
- `--since 24h`:支持 `30m/24h/7d` 或 ISO 时间戳(服务端 `fromTimestamp` +
  客户端双重强制);
- `--limit N`:接受的最大 trace 数;
- `--dry-run`:只打印将写入的 score,不发请求;
- `--force`:对已有 `atap:*` score 的 trace 重新评估。

## 3. 写回的 Score 语义

| Score 名 | 层级 | 类型 | 含义 |
|---|---|---|---|
| `atap:root-cause` | trace | categorical | 首要假设的 MAST/taxonomy 码(如 `FM-1.3`);comment 含完整归因:agent、step、root cause、fix suggestion、evidence 摘要 |
| `atap:confidence` | trace | numeric | 首要假设的置信度(按 confidence 取 top) |
| `atap:blamed-step` | observation | categorical | 写在被归因步骤对应的 observation 上;comment 为 `agent @ step: root cause` |

多条 hypothesis 时:trace 级取置信度最高的一条;每条 hypothesis 各得一个
`atap:blamed-step`(同一 observation 去重)。

每个 score 的 `metadata` 都携带两组机器可读字段(下游不必解析 comment):

- **假设字段**(平铺,完整 `Hypothesis` 原样):`agent` / `step` /
  `root_cause` / `root_cause_code` / `responsible_side` / `evidence`(数组)/
  `fix_suggestion` / `confidence` / `source`(产出算法,如 `chief`);
- **批次标识**:`run_id`(= 本次 `--out` 目录名,每次运行唯一)、`run_name`
  (配置)、`llm`(如 `openai:deepseek-v4-flash` 或 `fake`)、`seed`。

批次标识的用途:Langfuse score **只追加不更新**,同一 trace 反复评测会堆叠
多批结果——按 `metadata.run_id`(或 comment 头部的 `run <id>` 标签)即可分清
每条 score 来自哪次评测、用的哪个模型栈;UI 上点开 score 抽屉可直接看到
metadata 表。

## 4. 映射配置(真实 trace 与沙箱 trace 的差异)

真实 Langfuse trace 不携带 `metadata["atap"]`,拉取时按以下规则映射到 R0:

- observation type → kind:`GENERATION→LLM_CALL`、`EVENT→AGENT_MESSAGE`、
  `SPAN→TOOL_CALL`(叶子)/`AGENT_MESSAGE`(有子节点,视为编排容器);
- observation name 关键词覆盖任意 type:`handoff/transfer/delegate` →
  `HANDOFF`,`verif/evaluat/judge/guardrail/critic/assert` → `VERIFIER`;
- `agent` 从可配置的 metadata 键链取(默认 `agent / agent_name /
  gen_ai.agent.name / llm.app`),取不到记 `unknown`(不用 trace 名兜底,
  避免多 agent 归因坍缩到同一个名字);
- `input` 并入 payload,`output` → `payload["content"]`;
  `level=ERROR` 会给 content 加 `error: ` 前缀,保住下游错误观察约定;
- refs 置空(Langfuse 模型没有引用边,与离线适配同一声明);
- **outcome**:由 `outcome_from` 从某个 Langfuse score 推导
  (`{score, op, value}`,数值比较);无配置或无匹配 score 时保守置为
  失败——所有 trace 都进 analyze 由裁判判定,而不是被静默跳过。

这些键放在配置文件的 `source` 块(`atap langfuse-eval` 会读取
`type: langfuse_api` 的 source 与 CLI 参数合并,CLI 优先):

```yaml
source:
  type: langfuse_api
  outcome_from: {score: user_feedback, op: ">=", value: 1}
  agent_keys: [agent, agent_name]
```

`source: {type: langfuse_api}` 同样可用于 `atap run`(纯拉取+跑流水线,
不写回 score)。

## 5. 幂等与增量

- 默认跳过已带 `atap:*` score 的 trace(拉取时顺带取回的 score 列表复用,
  不额外发请求);
- 配合 `--since 24h` 即"每天评前一天的新 trace",无需游标文件;
  注意 `push_langfuse` 推入的轨迹时间戳是**推送时刻**(exporter 的 epoch-0
  仅用于离线确定性,push 时会重打真实时间),`--since` 对刚推的批次可用,
  但历史重推的老批次仍是老时间——分批请用下面的 tags;
- **按语料分批**(推送轨迹的标准工作流):
  `atap corpus --drift --out runs/d/traces.jsonl && atap langfuse-push
  --traces runs/d/traces.jsonl --tags corpus-drift`,之后
  `atap langfuse-eval ... --tags corpus-drift` 就只评这一批——tags 同时也是
  老轨迹不重跑的隔离手段;
- `--force` 强制重评。

## 6. 安全与防泄漏

- score comment 只由 `Hypothesis` 字段组装,**绝不包含** `trajectory.meta`;
  score `metadata` 同样只含假设字段 + 运行批次标识(run_id/run_name/llm/seed);
- push 路径走 `export_langfuse(..., external=True)` 严格模式:
  `injected_fault / origin_fault / fault_removed / qrels` 全部剥离,
  `outcome.note`(沙箱故障机制描述)中性化(有回归测试覆盖);
  **离线文件导出按契约保留 qrels**(rg_ug 数据依赖),只在外发时剥离;
- comment 超 4000 字符截断(完整信息在本地 artifacts)。

## 7. 已知限制与环境要点(2026-08-29 真实实例验证所得)

- **recover 不适用**:活 trace 没有可重放的沙箱,评估栈建议只到
  attribute(CLI 会在 recover + 无 sandbox 时告警);
- 云 v4 的 OTLP 推送不在本版(云 batch ingestion 2026-11 关停;self-hosted
  v3 长期可用,`docker-compose.langfuse.yml` 已钉在 `:3`,实测 v3.225.5);
- 分页参数使用 `page`/`limit` 信封(`{data, meta}`),对老版本回退为裸列表;
- **个别 self-hosted 构建忽略 `scores?traceId=` / `observations?traceId=` 过滤
  参数、返回全项目列表**(实测发现)——客户端已强制按 `traceId` 二次过滤
  (`LangfuseClient.scores/observations`),否则 skip 判定与 outcome 推导会被
  其它 trace 的数据污染;
- categorical score 的字符串值在 API 的 `stringValue` 字段(`value` 数字列
  显示为 0 属正常,UI 渲染 stringValue);
- Docker 环境:VM 内存 ≥ 4GB(colima 默认 2GB 会让 web 容器被 OOM kill,
  `colima start --memory 6 --cpu 4`);镜像拉取不通时给 VM 内 docker daemon
  配代理或 registry mirror;`LANGFUSE_INIT_USER_EMAIL` 必须是合法邮箱格式
  (缺 TLD 会导致 web 容器启动失败,默认值已是 demo@example.com)。

![atap scores on a live Langfuse trace](assets/langfuse_roundtrip.png)

## 8. 相关测试

`tests/test_langfuse_live.py`——全部离线确定性:MockTransport 单测
(映射/过滤/outcome/score 格式/防泄漏/幂等/push 与离线适配互通)+
线程 stub 服务器上的 CLI 端到端(写入 → 跳过 → force → dry-run)。
