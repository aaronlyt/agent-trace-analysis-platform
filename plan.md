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

### 阶段三（下一轮）

- [ ] 表征：+R5 动作签名+效果标签+LCS（TraceProbe 2607.06184）
- [ ] 分析：+循环检测谓词（TraceProbe，确定性，消费 R5 签名）
- [ ] 分类：+L0 免费规则包（AgentDebugX 2607.18754：畸形调用/无进展循环/过早成功声明）
- [ ] 归因：+L2 二分定位（Who&When ⌈log₂n⌉ 轮）+ SBFL 频谱（FAMAS 2509.13782，
      run_corpus 跨轨迹聚合，作 L2 先验）
- [ ] 恢复：+归因反馈注入再求解（AgenTracer 2509.03312 多轮反馈）
- [ ] 同一轨迹集不同 YAML 组合的算法对比跑法

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
