# 上线前真实 API 全量测试方案（2026-08-26 制定）

> 目标：用真实 LLM 跑通全部评测档，对**输出内容**做系统审计，作为上线前
> 的最终验收。所有数字基于 2026-08-26 真实运行实测校准（见 plan.md 轮次九）。

## 0. 已校准的事实基础（不再估算）

| 事实 | 数值/结论 | 来源 |
|---|---|---|
| 冒烟栈真实调用公式 | 26 + F 次（F=失败轨迹数，`feedback_match` 语义兜底） | ox/nemotron 双实测 |
| 各档调用计划 | 离线 `wc -l llm_calls.jsonl` 即精确底数（8 档实测 512 条） | /tmp/atap-est |
| HTTP 放大系数 | 记录数 ×1.08（空响应重试/限流退避） | 四档实测 3%–12% |
| 单次 token | ~2.2k（ox 冒烟 2.16k / chief 2.4k / claim 3.4k） | usage 实测 |
| ox-alpha（stealth） | 独立配额池，96+ HTTP 无 429；中位 30–60s/次 | 本轮实测 |
| 免费档 `:free` | **50 次/日、账户级共享**（非每模型）；429 不耗额度 | 429 响应头 |
| claim 栈输出预算 | 必须 `max_completion_tokens: 8192`（4096 截断 JSON） | ox claim 首跑 |
| 历史真实命中率 | smoke step 5–6/6 · agent 6/6 · MAST 3/6 · 恢复 6/6；chief step 6/6 | 两模型四档 |

**总量**：8 档 ≈ 640 条调用 / ~690 次 HTTP / ~120 万 tokens / ox 串行 ≈ 8–9 小时。

## 1. 分层验收门槛（Gates）

**P0 功能与安全（全部必须通过，任一失败即阻塞上线）**

| # | 检查 | 判据 |
|---|---|---|
| P0-1 | 全部 8 档跑完无崩溃 | 每档 exit 0、report.json 完整、artifacts 齐全 |
| P0-2 | 防泄漏铁律（真实运行时） | 判官类 tag（judge_eval/mast_judge/all_at_once/binary_search/chief/claim/tree 等）的 **prompt** 中：8 个故障名关键词与 injected_fault/ground_truth/qrels 字段名零出现；`feedback_match` 为沙盒环境侧匹配器、prompt 持故障规格属既有设计（policy.py `_feedback_addresses` 论证），单列复核不计硬门槛 |
| P0-3 | 响应侧泄漏扫描 | 同上关键词在 **response** 中出现 → 列入人工复核清单（prompt 干净时模型不应知道这些词；不自动判死） |
| P0-4 | 闭环恢复 | 每档 closed_loop 验证轮失败数 < 首轮失败数；冒烟/v3/dover 改善 ≥5/6 |
| P0-5 | 采集回环（零 LLM） | langfuse/otel 导出→导入→字段级等价、4A 栈结果不变（离线跑，上线前复验一遍） |

**P1 质量底线（真实模型能力，低于底线需归因分析后再决定）**

| 档 | 门槛（demo 7 条 GT 对照） |
|---|---|
| 冒烟栈 | step ≥5/6 · agent ≥5/6 · MAST ≥2/6 · 恢复 ≥5/6 |
| chief | step ≥5/6 · agent ≥5/6 |
| claim | 覆盖 ≥4/6 · step ≥3/5（覆盖不到的轨迹单独分析台账质量） |
| v3（corpus 24 条） | binary_search step ≥8/18 · feedback_injection 恢复 ≥14/18 · 闭环改善 ≥14/18 |
| dover | mistake=GT ≥14/18 · Validated ≥12/18 · 恢复 ≥14/18 |
| tree | step ≥8/18 |

**P2 输出内容审计（本方案核心增量，见 §3）**

| 指标 | 判据 |
|---|---|
| ok-rate | ≥95%（失败调用/总调用） |
| 解析修复率 | schema 调用中 http>1 占比 ≤10% |
| HTTP 放大 | Σhttp/Σ记录 ≤1.15 |
| 调用拓扑漂移 | 每 tag 实计数 vs §2 计划 ±15% 内（超出说明分支异常） |
| token 结算 | Σusage 与 120 万估算偏差 <2 倍，逐档记录 |
| 人工抽检 | 每 tag 抽 3 条 response：引用的 step 序号存在、引文与轨迹 payload 一致、无幻觉事实 |

## 2. 执行排程（两日，密钥只走环境变量）

**Day 1（ox-alpha，~300 次 / ~4.5h）**

```bash
export OPENAI_API_KEY=sk-or-v1-... OPENAI_BASE_URL=https://openrouter.ai/api/v1
# ① 冒烟（32 次，~25min）：P0 全链路 + P1 命中率基线
.venv/bin/atap run --config configs/realtest_ox_smoke.yaml --out runs/final/ox-smoke
# ② v3 全栈（165–201 次，~2.5–3h）：二分 + 反馈注入 + 闭环
.venv/bin/atap run --config configs/pipeline_offline_v3.yaml --out runs/final/v3   # llm.type 需换 openai
# ③ chief（54 次，~50min）  ④ tree（36 次，~35min）——同样换 llm 块
```

**Day 2（ox-alpha ~310 次 + 免费档跨模型抽查 ~50 次）**

```bash
# ⑤ dover（108 次，~1.6h）  ⑥ claim（57 次，~1h，8192 已配）
# ⑦ 冒烟栈上 24 条语料（~102 次，~1.5h）  ⑧ v4 或 sbfl 任一（42 次）
# 跨模型抽查（免费档池，验证结论不依赖单一模型）：
.venv/bin/atap run --config configs/realtest_nemotron_ultra_smoke.yaml --out runs/final/nemotron-smoke   # 32
.venv/bin/atap run --config configs/realtest_ox_chief.yaml --out runs/final/nemotron-chief-替代档        # 若额度余 18
```

说明：v3/dover/chief/claim/tree/v4 现有 config 为 `llm: fake`，执行前复制为
`configs/final_*.yaml` 只改 llm 块（openai + stealth/ox-alpha + interval 3.0，
claim 档加 8192）；drift 与 sbfl-vs-v4 的另一档零 LLM 差异，离线验收即可。
顺序上冒烟最先（最快暴露环境问题），失败即停、修复后从该档重跑。

## 3. 输出内容审计流程（每档跑完立即执行）

数据源：`runs/final/<档>/llm_calls.jsonl`（messages 完整 prompt / response /
usage / http_requests / latency / ok）+ `artifacts/`（Hypothesis、MAST 标签、
闭环结论）+ `runs/demo|corpus/traces.jsonl`（GT）。

1. **自动审计**（建议做成 `docs/realtest_audit.py`，读一个或多个 run 目录输出报告）：
   - P0-2/P0-3 关键词扫描（8 故障名 + 3 真值字段名，prompt=硬门槛 / response=复核清单）；
   - P2 指标：ok-rate、http 放大、解析修复率、tag 分布 vs 计划、latency 分位数、token 结算；
   - GT 命中率自动打分（对照 injected_fault 的 mast_code/step/agent，输出 §1 表格数值）。
2. **人工抽检**：每 tag 抽 3 条 response，核对三件事——引用步号真实存在、
   引用内容与轨迹 payload 逐字一致、结论不包含轨迹外的世界知识（幻觉）。
   judge_eval 的 findings、mast_judge 的 reason、claim_audit 的六元组为重点。
3. **归档**：审计报告写 `docs/audit_上线前真实测试_<日期>.md`，附每档
   run.log 摘要；runs/final/ 全量保留（llm_calls.jsonl 即计费复核单）。

## 4. 风险与预案

| 风险 | 预案 |
|---|---|
| OpenRouter 偶发 200+choices=null / 429 | 已修复：退避重试 + 审计记录；ok-rate 跌破 95% 时查 retries 列表 |
| 输出截断（长 records 数组） | claim 已 8192；新档先冒烟 1 条轨迹看 completion_tokens 分布再放量 |
| 免费档额度误耗 | 只在抽查步用 `:free`；主流程全 ox-alpha（独立池） |
| 单模型结论偏置 | Day 2 跨模型抽查对照；两模型同语料同 seed |
| 结果不可复现 | temperature=0 + seed=7 固定语料；同档重跑差异记入审计（模型侧非确定性如实报告） |
| 墙钟超预期 | ox p95 ~100s；单日超 6h 可把 dover/claim 挪 Day 3，互不依赖 |

## 5. 上线判据（一页结论）

P0 五项全过 + P1 各档达底线 + P2 审计报告归档（ok-rate ≥95%、泄漏零、
人工抽检无幻觉实锤）→ **可上线**；任一 P1 未达 → 产出归因分析
（判官能力 vs 管线缺陷，用 FakeLLM 6/6 基线区分）后再决策。
