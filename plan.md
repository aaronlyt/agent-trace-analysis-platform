# atap 实施进度

> 复现《Agent 轨迹分析与错误归因：整体流程·架构·算法与文献》(2026-08-25)
> 六环节：采集 → 表征 → 分析评测 → 错误分类打标 → 失败归因 → 恢复与增强（闭环）

## 轮次一：阶段一（架构骨架）+ 阶段二（每流程 1 算法全链路）—— ✅ 完成 2026-08-25

### 阶段一：架构骨架 ✅

- [x] `core/`：R0 事件模型 + Hypothesis 统一归因输出 + bundle 产物容器 + StageAlgorithm 基类
      （run_one/run_corpus 双作用域）+ Registry + PipelineConfig + Pipeline 编排（含闭环）+ RunContext
- [x] `llm/`：LLMClient 协议 + OpenAI 兼容实现 + Fake 实现（确定性伪判官 pseudo_judge）
- [x] `io/`：TraceSource/TraceStore/ArtifactStore 协议 + JSONL 实现
- [x] 五个 stage 包 base（represent/analyze/classify/attribute/recover）
- [x] 验收：Dummy 算法 e2e + 架构不变量测试（import 图 / registry 契约）

### 阶段二：全链路 LLM-judge 垂直切片 ✅

- [x] `sandbox/`：玩具多 agent 系统（planner→searcher→reporter，mock 检索语料 + 验证器）
      + 六种故障注入（step_repetition/malformed_tool_call/info_withholding/
      premature_termination/ungrounded_citation/disobey_task_spec，各映射 MAST 模式，
      meta 记录 ground truth）
- [x] 表征：`canonical_events`（span 树拍平，引用边 span-id→event-id 映射）+ `ssf`
      （TrajAudit Algorithm 1；工程适配：结构化错误前缀判定 + 占位符带摘要，见模块 docstring【推断】）
- [x] 分析：`judge_eval`（few-shot 判官，质量分+类型化 finding）
- [x] 分类：`classify/taxonomy`（MAST 3 类 14 模式 + FusionLabel 融合标签结构）+ `mast_judge`
      （判官打标 + 未知代码丢弃）
- [x] 归因：`all_at_once`（Who&When 单遍，输出统一 Hypothesis，step/agent 越界钳制并留痕）
- [x] 恢复：`targeted_rerun`（AgentDebug Algorithm 1 Stage 3：t* 前缀保留、反馈注入 ≤5 轮、
      UpdateFeedback 弱化版；无归因/无环境显式降级不静默）
- [x] 闭环：`Pipeline.run_closed_loop`——重跑轨迹替换原失败轨迹回全流程验证，
      结论写回 origin bundle 的 recover/closed_loop 产物
- [x] CLI：`atap demo`（离线全链路）/ `atap run --config` / `atap list`；
      configs/pipeline_offline.yaml + pipeline_llm.yaml

**验收结果（离线 e2e，FakeLLM，seed=7）：70 个测试全绿；六种注入故障
归因命中 step 6/6、agent 6/6、MAST 6/6；定向重跑恢复 6/6；闭环验证轮
failures=0。**

**真实 LLM 验证（OpenRouter stealth/ox-alpha，temperature=0，2026-08-25）：**
18 次结构化调用全部成功（含 10 次上游限流退避重试）。六故障结果：
**step 4/6、agent 5/6、MAST 主标签 3/6（标签集合命中 5/6）、恢复 0/6**。

- 与文献一致的现象：agent 级（5/6）明显好于 step 级（4/6）；两处 step
  未命中其一为"早一步"（把 plan 步判为决定性错误——追溯式归因的合理
  因果故事），其二把 reporter 违规格归因到 planner 未传达约束（该轨迹
  上同样成立）；MAST 主标签偏差多为标签排序差异（如 info_withholding
  判 FM-2.6 主/FM-2.4 次）。
- **恢复 0/6 的根因是已知限制**：真 LLM 的自由文本 fix 建议质量很高
  （均正确指明约束与修正动作），但脚本化沙盒策略靠故障类型关键词匹配
  消费反馈，匹配不上 → 5 轮重跑全部失败。要闭环真模型反馈，需要
  LLM 驱动的沙盒策略（阶段三可做），或沙盒反馈解析增强。
- 伪判官的 6/6 证明的是框架管路与契约正确；真实分数如上，两者互补。

### 论文一致性审计与修复（六路 subagent，2026-08-25）

对照 refs/ 原文逐算法独立审计（SSF/MAST/All-at-Once/targeted_rerun/R0/
故障注入六路），总体结论：机制复现忠实、无语义偏离；修复了审计发现的
实质问题，并补齐全部声明缺口：

- **few-shot off-by-one**（all_at_once/mast_judge）：示例改为"第二次调用
  即首次重复"（step 5/6/7 重复 → 决定性错误在 6）——原示例会让真 LLM
  在重复类故障上系统性早一步；
- **FM-2.6 定义去污染**（taxonomy）：删除论文零支持的"（含畸形工具
  调用）"扩写——该定义直接进判官 prompt，等于用适配定义引导判官命中
  ground truth；适配移至 sandbox/faults.py 映射注释【适配】。FM-1.3/
  1.5/2.5/3.1/3.3 定义同步回归 App. A 原文语义；FC3 改名 "Task
  Verification"；
- **t\* 选择规则**（targeted_rerun）：max(confidence) → (confidence,
  -step)——并列取最早步，对齐 AgentDebug t\*←min(T\*)；产物 schema
  统一（skipped 分支补 recovered/status）；
- **premature_termination onset=plan 步**（原为 submit）：对齐 Who&When
  Eq.5 最早决定性错误（修正规划步即可翻盘，早于终止动作一步）；伪判官
  规则 4 同步归因到 submit 前的决策步；
- 声明补齐：SSF（min_fold_len 豁免、loose=strict∪词边界叠加而非恢复
  原文子串、strict 词表收窄、fold_ratio 分母口径）、mast_judge
  （few-shot 来源、max_labels 截断、include_success 与 J.1 协议）、
  all_at_once（54.33 为 With-GT 列口径）、targeted_rerun（每轮从原始
  轨迹重放而非 τ⁽ᵏ⁻¹⁾）、R0（省略 AgentTrajectory 的 error/duration/
  metadata/artifacts 四字段）、env（verifier 说明区分到故障组粒度）、
  pipeline（闭环只验证最后一条 rerun、验证轮含嵌套恢复）；
- 新增防泄漏回归测试：三判官 prompt 全文不得含故障类型词/GT 键。

**验证：71 测试全绿（70+1）；离线 e2e 六故障 step/agent/MAST/恢复仍
6/6，闭环验证轮 failures=0；七项针对性断言（定义去污染、t\* 规则、
onset 对齐、Eq.5 反事实重放、few-shot 语义、六故障全对齐）通过。**

### 阶段三：L0/L2 阶梯 + 跨轨迹聚合 + AgenTracer 恢复 ✅（2026-08-25）

- [x] 表征：`action_signature`（R5：九类规范动作+参数指纹+七效果标签+锚集/里程碑
      /LCS，TraceProbe 2607.06184——已用 paper-fetch 补入 refs/ 并对照原文实现；
      效果标签为论文全集 7 个而非综述的 4 个；锚集按原文 oracle-free 回退路线取
      "同任务成功轨迹读过的文档"；规范动作类不写回 `TraceEvent.action`——该字段
      承载采集层工具名，判官渲染行与伪判官规则依赖它，改写等于变更判官视图）
- [x] 分析：`loop_detect`（TraceProbe Table II 四谓词：search_loop/re_read_churn/
      tool_oscillation/redundant_search；阈值参数化，默认冻结值 10 为 SWE-Bench
      口径，玩具域配置审计为 3——原文明示阈值需按目标基准审计）
- [x] 分类：`rule_pack`（L0 免费规则包，AgentDebugX：malformed/no-progress/
      premature-success/invalid-output 四规则，触发条件自设【适配】——原文只有
      一句话定义，精确规则在官方仓库未随论文发表）
- [x] 归因：`binary_search`（Who&When Algorithm 2 逐行复现：只展示 [low,mid]、
      裸文本 upper/lower 应答、⌈log₂n⌉ 轮、A\* 从事件读；收尾 refine 调用为
      DeepDebug 风格工程增强）+ `sbfl`（FAMAS 式 2-7 逐式复现：γ/β/α/λ-decay
      + Kulczynski2^λ，λ=0.9，`run_corpus` 跨轨迹作用域首用；LLM 层次聚类抽象
      → R5 确定性签名【适配】；频谱单元排除 verifier/env 环境侧事件）
- [x] 恢复：`feedback_injection`（AgenTracer §5.3：3 轮全量再求解、第 1 轮反馈
      取归因 Hypothesis、后续轮判官反思再生成；`ReplayEnvironment` 协议 +resolve；
      沙盒反馈消费升级"关键词优先、LLM 语义兜底"——修复真模型恢复 0/6 的已知
      限制，环境侧自知故障规格不违反判官 GT 泄漏约束）
- [x] 对比跑法：`atap compare`（同一轨迹集多配置对比：step/agent/MAST 命中、
      恢复、闭环改善、LLM 调用分桶计数）+ `atap corpus`（频谱语料生成：
      每任务 K 成功 + 6 故障交叉）

**验收（离线，FakeLLM，seed=7，语料=每任务 2 成功+6 故障×3 任务=24 条）：**
139 测试全绿（74 旧+65 新，含不变量/防泄漏回归）；阶段二栈回归 6/6×4 不变；
阶段三全栈（configs/pipeline_offline_v3.yaml）——二分定位 **step 5/6、agent 6/6**
（唯一偏差 step_repetition 5→8：二分在片段失去三次重复上下文后收敛到症状末次
重复，与 Who&When 报告的二分 step 级弱于逐步审查方向一致，如实保留）；
loop/rule 各命中靶故障；SBFL **4/6**（超预期 3/6：step_repetition 被 α 局部
频率增强放大、premature/disobey/ungrounded 亦命中；miss 两例均可解释——
malformed 一次性异常动作被 γ 覆盖比压制、info_withholding 内容级故障在动作谱
不可见而牵连下游 report 步）；feedback_injection 恢复 **6/6**、闭环验证轮
failures=0；compare 表：v3 全栈 15/18·18/18·18/18·147 calls vs SBFL 12/18·
42 calls（L0 免判官调用）。

**遗留：真实 LLM 复跑（nemotron 免费档）待 OPENAI_API_KEY 可用时执行
`realtest_nemotron.py`（阶段二栈回归）+ v3 最小组合（二分 vs all_at_once 的
step 级对比、`feedback_matching=llm` 的真模型恢复验证）——离线验收为准入门槛，
此项非阻塞。**

### 远期（阶段四候选）

R2 信息依赖图（含 CHIEF 层次因果图）、R3 claim 台账（DRIFT）、R4 层级树
（CodeTracer）、RG/UG 确定性归因（搜索智能体诊断 2608.01913）、失败聚类+
残差词表扩展（AgentDebugX inducer）、分布漂移检测（2511.19933）、L3 动态
重放（TraceElephant 2604.22708 / DoVer）、Langfuse v3 / OTel GenAI 采集适配器、
AgenTracer GRPO 微调 tracer 路线。

## 设计要点（对齐文献的两条核心原则）

- 检测 ≠ 归因：analyze 只回答"有没有问题"，attribute 才回答"哪个错误决定了失败"
  （误定位 81% 偏晚、平均滞后 14.5 步——TrajAudit/RootSE）。
- 聚合先于单例：StageAlgorithm 提供 run_corpus 作用域，为跨轨迹算法（SBFL/聚类）预留
  （agent 级 53.5% 可用 vs step 级 14.2%——Who&When）。
