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

**真实 LLM 验证（DeepSeek `deepseek-v4-flash`，temperature=0，2026-08-25，
`docs/realtest_deepseek.py`，同一六故障群体）：**

- **stack-a 阶段二回归（all_at_once + targeted_rerun）：step 6/6、agent 6/6、
  MAST 主标签 3/6、恢复 6/6（全部 fault_removed=True，1 轮即恢复）**——
  显著优于此前 OpenRouter ox-alpha（step 4/6、agent 5/6、恢复 0/6）；
  **"恢复 0/6"已知限制确认修复**：沙盒 LLM 语义反馈匹配把真模型自由文本
  fix 建议正确判为"针对该故障"，六故障首轮全部移除故障恢复成功。
- stack-b 阶段三（binary_search + feedback_injection）：step 2/6、agent 2/6、
  恢复 6/6。二分在真判官上明显弱于单遍（MAST 3/6 持平）——与 Who&When
  文献方向一致（二分 step 级弱于逐步审查、agent 级弱于 all-at-once）；
  玩具轨迹短（11–17 事件，3–4 轮），判官连续偏置回答"lower half"即把
  区间塌缩到 step 0/早期步（ungrounded/malformed 收敛到 TASK_START env
  事件）。premature/step_repetition 两例二分正确命中。AgenTracer 式
  反馈注入恢复与 targeted_rerun 同样 6/6。
- 工程教训：思考型输出可耗尽 4096 max_tokens 导致 content 为空——真测
  配置上限提至 8192（`docs/realtest_deepseek.py` 内注明）。

### 论文一致性审计二轮（12 路全模块 subagent）与修复 ✅（2026-08-25）

对照 refs/ 原文对全部 12 个算法模块逐个独立审计（canonical_events/ssf/
action_signature/judge_eval/loop_detect/mast_judge+taxonomy/rule_pack/
all_at_once/binary_search/sbfl/targeted_rerun/feedback_injection，报告见
`audit_论文一致性_2026-08-25.md`）。总评：硬数值（公式/轮数/阈值）全部
一致，多数偏离已声明；修复发现的全部实质问题：

- **loop_detect 窗口口径**（唯一实质算法语义偏离）：re_read_churn/
  redundant_search 的 "10-action window" 从"同类动作子序列上 10 个"改回
  动作序列口径（连续 window 个签名动作）——旧实现相隔任意多动作也触发；
- **judge_eval 对齐 MAST J.1**（唯一协议相悖）：判官视图默认剥离
  outcome 行（`show_outcome=true` 恢复），伪判官从轨迹内 VERIFIER 行推断
  成败；few-shot 改为格式演示声明 + severity 加 Literal 词表校验
  （同义词归一、未知值显式解析失败）；
- **taxonomy 定义二轮去污染**：FM-3.3（假阳性括号+误导性结论）、
  FM-2.4（导致下游重复失败）、FM-1.1/1.2/1.3/1.5/2.1/2.2/2.3/3.1/3.2
  全部回归附录 A 原文语义（保留"可能"限定语、恢复 FM-1.2"表现得像
  另一个 agent"等特有后果）；
- **few-shot 步号对齐真实轨迹**（mast_judge + all_at_once）：示例改为
  step 3/5/7、决定步 5（与沙盒 R0 索引及 GT onset 一致；旧示例的连续
  编号 5/6/7 在含结果事件的真实轨迹中不可能出现）；
- mast_judge `max_labels` 改为先全量校验再截断（无效代码不挤占名额），
  超出标签记入 `truncated_codes` 不静默丢弃；
- 小 bug 清单：action_signature M3 步数改为每锚首次读的最大值（原取
  最后一次重复读偏晚）、`alignment.reference_trace` 改填参照轨迹 id、
  锚集声明改为"oracle-free 新构造"如实措辞；binary_search A\* 补
  AGENT_MESSAGE；ssf `unfold_line` 改非锚定 search（原正则锚定行首、
  渲染行必然不匹配）+ 空观测/extra_keywords 声明；rule_pack
  premature-success 只认成功读取（读取失败不算证据）+ onset=0 边界；
  sbfl 第二次出现的选择理由改写为"首次重复才是决定性错误"；R0 省略
  声明补齐（module 省略、inputs/outputs 合并 payload、ts=index 副本）；
  feedback_injection 末轮失败后不再多调一次永不会注入的反思；沙盒
  rerun_from step 越界钳制留痕 + "无注入故障的失败轨迹"不再恒判恢复。

**验证：154 测试全绿（139 旧 + 15 新增审计回归 test_audit_fixes.py）；
离线 e2e 验收数字与修复前完全一致——demo 六故障 step/agent/MAST/恢复
6/6、闭环 failures=0；v3 全栈 15/18·18/18·18/18·147 calls（二分 per-fault
step 5/6，唯一 miss 仍为已记录的 step_repetition 5→8）；SBFL 12/18·
42 calls（4/6，miss 仍为已解释的 malformed/info_withholding 两例）。**

### 远期（阶段四候选）

R2 信息依赖图（含 CHIEF 层次因果图）、R3 claim 台账（DRIFT）、R4 层级树
（CodeTracer）、RG/UG 确定性归因（搜索智能体诊断 2608.01913）、失败聚类+
残差词表扩展（AgentDebugX inducer）、分布漂移检测（2511.19933）、L3 动态
重放（TraceElephant 2604.22708 / DoVer）、Langfuse v3 / OTel GenAI 采集适配器、
AgenTracer GRPO 微调 tracer 路线。

## 轮次三：阶段四A 确定性层 ✅（2026-08-25，计划见 plan_阶段四.md）

对照 refs/ 原文实现五个确定性模块（其中 CHIEF 2602.23701 与 DoVer
2512.06749 两篇已用 paper-fetch 补入 refs/ 供后续轮次）：

- [x] `represent/idg`（R2 信息依赖图，GraphTracer 思想——已撤稿只吸收
      思想不采数字）：R0 `refs` 即 usage 边，建图 O(V+E) 零 LLM【适配：
      原文需事后引用抽取（模式匹配/辅助 LLM），采集层已提供】。验收：
      四类故障 GT 根因 ∈ 终态祖先闭包；两例 honest miss 如实记录
      （step_repetition 重复结果从未被引用、premature 无据提交 refs 为空
      ——"信息浪费/缺失"型故障不在依赖链上，由 R5/rule_pack 补位）
- [x] `represent/hierarchy_tree`（R4，CodeTracer）：exploration=兄弟/
      state-changing=子节点；域映射【适配：SWE edit/install → 研究问答
      search/read=探索，LLM_CALL/HANDOFF/submit=改状态】；stage 用 R0
      phase；tree.md 每步一行压缩渲染（13 事件→8 行）
- [x] `attribute/rg_ug`（L0 确定性，2608.01913 §4.3 逐式）：沙盒补
      `env.qrels`（E/G 两层按构造已知，经 meta["qrels"] 携带）；episode
      按 search 边界切分；R_k/C_k/Δ_k/G* 集合运算；判定 correct/RG_
      directional/RG_last_hop/UG_true_extraction/UG_boundary + episode
      效用（productive/redundant/unproductive）+k*/wasted_tail/visit
      precision；Hypothesis 映射【适配：轨迹级标签→agent/step】。验收：
      构造标签 7/7；18 故障轨迹 step 15/18（miss=step_repetition 映射
      边界）；UG_boundary 由合成 fixture 覆盖（现任务 gold 均单文档）
- [x] `analyze/drift_detect`（2511.19933 三类漂移定义→工程实现）：分组键
      (model_version×prompt_version×time_window)（schema 已预留）；
      行为特征（kind/action 直方图、长度分箱、失败率、重复调用率）+
      PSI（ε 平滑）；三对照族：version（同 prompt 跨模型桶）/behavior
      （同模型跨窗比行为）/data（同模型跨窗比任务构成）——统计实现全部
      【适配：论文无算法，综述明示 PSI 为工程选型】。新语料
      `generate_drift_corpus`（w1 基线/w2 换模型/w3 换任务构成/w4 同构成
      注入重复行为）。验收：三族全检出（PSI 11.46/1.06/6.14）、稳定
      语料零误报、确定性
- [x] `classify/inducer`（AgentDebugX §3.4）：mast_judge 增 `allow_novel`
      残差通道（novel 标签必附 symptom 短语）+`extra_modes_file` 加载
      人工接受的扩展模式；inducer 词面聚类（3-gram Jaccard≥0.35）+
      支持度门控（簇≥3）→ 每簇一提案（NM-n，命名=症状实词 top-3，不含
      故障类型词防泄漏）+ kinship（最近 MAST）+ seed 去重；**提案永不
      自动生效**，`atap taxonomy accept` 人工闸门。新故障
      `agent_deadlock`（双 agent 互等，MAST 14 无对应=天然残差；
      EXTRA_FAULTS 独立注册不进默认语料，旧数字不变）。伪判官 novel
      行为=复发 agent 间消息提取症状+扩展模式词面匹配。验收：deadlock×3
      → 恰好 1 提案；接受后新码打标闭环；全可标语料 0 提案

配套：沙盒 `retrieval_detour` 故障（RG last-hop 靶：泛化查询命中
evidence 永不中 gold）；configs/pipeline_offline_v4.yaml（4A 全栈）与
pipeline_drift.yaml；CLI `corpus --drift` 与 `taxonomy accept`。

**验收：179 测试全绿（154 旧+25 新：idg/tree 7、rg_ug 6、drift 5、
inducer 7）；demo 六故障 6/6×4 与 v3 全栈 15/18·147 calls、SBFL 12/18·
42 calls 全部回归不变；compare 表新增 v4 行：rg_ug step 15/18·agent
15/18·42 calls（LLM 调用全部来自 judge_eval/mast_judge，rg_ug 本身零调用）。**

### 轮次四~六（阶段四B/C/D，待实施）

见 plan_阶段四.md：4B=claim_ledger+claim_audit（DRIFT）/tree_diagnosis
（CodeTracer 诊断）/hcg+chief（CHIEF）；4C=沙盒检查点重放基建+
counterfactual_replay（TraceElephant）+dover（DoVer）；4D=langfuse/
otel 采集适配器。

## 轮次四：阶段四B LLM 表征与归因升级 ✅（2026-08-25）

- [x] `represent/claim_ledger`（R3，DRIFT §4-A 式 2）：六元组
  (text/introduced/first_effective/reuse/type/status) + task_goal +
  hard_constraints；LLM 全局单遍、3-5 条紧凑、finalized 仅终答；
  【适配】span=R0 事件
- [x] `attribute/claim_audit`（DRIFT §4-B/C/Tracer）：支撑四级
  （DIRECT/WEAK/MISSING/CONFLICTING）→ verdict 四值 → 保守回溯取最早
  失支撑主张引入位；【适配】B+C 合并为一次调用（原文 claim×auditor
  逐路路由），共 2 调用；约束主张核查最终答案引用（Constraint 家族）
- [x] `attribute/tree_diagnosis`（CodeTracer §3.2）：两次调用近似
  "先树后钻"（原文为命令式 agent 交互）——①tree.md 树级定位可疑
  stage（循环/错误信号）②stage 区间+邻居上下文下钻；产物记录
  inspected vs full 渲染行数（省 token 证据）
- [x] `represent/hcg`（CHIEF §4.1）：三层图确定性构建——subtask=R0
  phase 区间、agent 节点=OTAR（observation 沿 refs 归属调用 agent）、
  E_sub/E_agt/E_step 三类边；【适配】原文全 LLM 驱动建图（2.5-3×
  token），Φ 模式留占位由 chief 近似
- [x] `attribute/chief`（CHIEF §4.2-4.3）：oracle 合成（1 调用）→
  subtask F_eval 逆拓扑回溯（1 调用）→ 渐进因果筛选定位（1 调用，
  mechanism∈local/upstream/executor_loop/planning/dataflow_first_
  pollution）；【适配】原文 ≈5+K 次调用压缩为 3 次
- [x] 伪判官新增 8 个 handler（claim_ledger/claim_audit_support/
  claim_audit_trace/tree_diagnosis_stage/tree_diagnosis_drill/
  chief_oracle/chief_eval/chief_localize），全部只依据判官可见文本

**验收：191 测试全绿（179+12）；六故障伪判官 step/agent——chief 6/6
（机制全对）、tree_diagnosis 6/6（stage 定位 6/6、下钻渲染行数 < 全量）、
claim_audit 主场 4/4 + 两例 honest miss（step_repetition/malformed 非
claim 级故障，Tracer 禁选工具步，如实记录）；18 条故障语料 compare——
chief 18/18·54 calls、tree 18/18·36 calls、claim 12/18·57 calls、v3 回归
15/18·147 calls 不变；防泄漏回归覆盖 4B 全部 8 个 prompt。**

### 轮次五~六（阶段四C/D，待实施）

见 plan_阶段四.md：4C=沙盒检查点重放基建+counterfactual_replay
（TraceElephant）+dover（DoVer）；4D=langfuse/otel 采集适配器。

## 轮次五：阶段四C L3 反事实重放 ✅（2026-08-25）

- [x] 沙盒 `ToySandbox.replay_intervene(trajectory, step, edit_text,
      horizon, n_repeats)`：检查点重放+候选步消息**原位替换**（DoVer M1
      语义；编辑文本经故障条件消费决定后缀走向——关键词优先/LLM 兜底，
      与反馈消费同机制【适配：两文均未给缓存/去随机细节】）；horizon=k
      窗口模式（TraceElephant"只验证后续 k 步"）；n_repeats ×3（DoVer）；
      不变量：无编辑效应时后缀 (kind,agent,payload) 逐事件不变
- [x] `attribute/counterfactual_replay`（TraceElephant A.6.3）：候选=其它
      归因算法 Hypothesis 去重按置信度取 ≤3；每候选一次 cf_oracle 调用
      （自推期望+编辑文本）；重放 k=3 窗口；verdict validated（编辑改变
      失败走向，置信+0.2）/refuted（伪因果/症状步，置信−0.3）
- [x] `recover/dover`（DoVer §4）：Trial Segmenter（plan 消息切点）→
      Failure Proposer（mistake 三元组）→ Intervention Recommender
      （category 三类+最小 replacement，"不给答案内容"）→ replay_intervene
      ×3 跑到结束 → Milestone（K=3 任务构造生成）→ Outcome Classifier
      （Validated/Partially/Refuted/Inconclusive 四判定）；消息原位替换+
      结果差分 vs targeted_rerun 的追加反馈（docstring 声明）
- [x] 伪判官新增 6 个 handler（cf_oracle/dover_segment/dover_proposer/
      dover_intervene/dover_milestone/dover_classify）

**验收：201 测试全绿（191+10）；六故障——cf_replay GT 候选 validated 6/6
（症状步候选 refuted 置信下调）、dover mistake=GT 6/6·Validated·恢复
6/6（每干预 ×3）；e2e pipeline_dover.yaml 闭环 18/18·18/18·108 calls、
闭环改善 18/18；防泄漏回归覆盖 4C 全部 6 个 prompt。**

### 轮次六（阶段四D 采集适配器，待实施）

langfuse_source（v3 ingestion/v4 OTLP 双格式）、otel_source（gen_ai
semconv）、roundtrip 验证。

## 轮次六：阶段四D 采集适配器 ✅（2026-08-25）

- [x] `io/langfuse.py`：v3 ingestion batch 导入/导出——R0 事件 ↔
      observation（LLM_CALL→GENERATION，kind/agent/action/phase/refs 入
      metadata["atap"] 命名空间——refs 是 Langfuse 模型没有的引用边，
      必须经 metadata 生存）；parentObservationId ← R0 parent；trace 级
      input→task、meta（qrels/分组键）随 trace metadata；GT（injected_
      fault）不导出（防泄漏）
- [x] `io/otel.py`：OTLP/HTTP JSON 导入/导出——gen_ai.operation.name
      标准属性（chat/execute_tool/invoke_agent）+ 未映射信息入 atap.*
      自定义属性；refs 经 atap.refs 生存并按 ev_id→spanId 重映射
- [x] CLI `atap export --format langfuse|otel`；build_source 分发
      langfuse/otel 新类型
- [x] roundtrip 验收：导出→导入→canonical_events 拍平，事件语义字段级
      等价（树/kind/agent/action/payload/引用边/phase/outcome/qrels）；
      导入轨迹跑 4A 栈结果不变

**验收：208 测试全绿（201+7）。阶段四（4A/4B/4C/4D）全部完成。**

## 轮次七：阶段四论文一致性审计（2026-08-25）

11 路只读 subagent 逐模块对照 refs/ 原文（idg/hierarchy_tree+tree_
diagnosis/rg_ug/inducer/drift_detect/claim_ledger+claim_audit/hcg+chief/
counterfactual_replay/dover/langfuse/otel，报告
`audit_阶段四论文一致性_2026-08-25.md`）。修复：dover classify 运行期
回显故障名（🔴，`_redact_fault_names` + 运行期防泄漏回归测试）、otel
traceId/spanId 非 OTLP hex（🔴，sha256 派生 + atap.* 原 id 生存）、
dover trial 区间 off-by-one、langfuse outcome.score/ts 恢复；约 30 处
【适配/推断/论文未说明/声明弱化】标注失实或缺口补正。保留边界（未改
行为）：claim_audit outcome 行、HCG OTAR result 槽、inducer 与契约的
聚类差异——均已在 docstring/审计报告声明。

**验收：210 测试全绿（208+2）。demo 六故障 6/6×4 不变。**

## 轮次八：统一日志与 LLM 调用审计（2026-08-26，工程能力）

补齐"程序日志"缺口（此前只有 stdout print + 结构化 report.json）：

* `atap/log.py`：`atap` logger（stderr + 替换式 run.log 文件 handler，
  setup 幂等；stdout 只出结果、过程走日志；`atap -v` 开 DEBUG）。分层
  不变量保持——core/** 不 import 日志模块，阶段耗时由 runtime 在
  run_config 末尾统一落 log（stage_log 逐条 INFO）。
* `llm/call_log.py`：`CallLogMixin`（Fake/OpenAI 客户端共用）——
  `complete()` 计时包装，审计记录（ts/client/tag/model/schema/
  latency_ms/messages 完整 prompt/response 截断 20k/usage token/ok+
  error）追加写 `runs/<name>/llm_calls.jsonl`；runtime.run_config 自动
  挂载，compare 的 CountingLLM 向内层透传，库形态默认不挂载。OpenAI
  客户端顺带回填 `LLMResult.usage`（token 用量）。
* CLI：错误改 `log.error`/`log.exception`；demo/compare 补起止 INFO。

**验收：215 测试全绿（210+5：Fake 审计成功/失败路径、OpenAI 包装
离线单测（绕过 __new__ + monkeypatch _create）、setup 幂等/run.log
替换式、demo 端到端 run.log+llm_calls 落盘）。demo：run.log 含全部
阶段耗时与验收数字，llm_calls.jsonl 26 条（judge_eval 14 + mast_judge
6 + all_at_once 6），6/6×4 不变。**

## 轮次九：真实 LLM 冒烟第三轮（2026-08-26，OpenRouter 四档）

四档 `configs/realtest_*.yaml`（密钥只走环境变量）：ox-alpha 冒烟 /
ox-alpha chief / nemotron-3-ultra 冒烟 / claim（nemotron 撞墙后改 ox）。

- **验收数字**：ox 冒烟 step 5/6·agent 6/6·MAST 3/6·**恢复 6/6**
  （闭环二轮失败 0）；ox **chief step 6/6·agent 6/6**（code=None 为
  设计，与离线一致）；nemotron-3-ultra 冒烟 **step 6/6·agent 6/6·
  code 3/6·MAST 3/6·恢复 6/6**（追平 FakeLLM 定位基线，中位 18s）；
  ox claim 覆盖 5/6·step 3/5（claim 台账抽取质量决定上限）。
- **工程修复**：① `http_requests` 配额记账入 llm_calls.jsonl（每次
  complete 的真实 HTTP 请求数，含限流/解析修复重试）；② OpenRouter
  偶发 200+choices=null → 可重试错误退避重试，耗尽显式 LLMError；
  审计包装捕获一切异常（此前 TypeError 裸抛且审计缺记录）；③ claim
  栈 `max_completion_tokens: 8192`（4096 截断 JSON 实测：修复重试 2
  次仍截断）。
- **配额事实**：OpenRouter 50 次/日为**账户级 free-models-per-day**
  （全部 `:free` 模型共享，非每模型独立；429 响应头 X-RateLimit-*
  明示）；`stealth/ox-alpha` 独立池（96+ HTTP 无一次 429）。真实
  调用公式修正：冒烟栈 = 26 + F 次 `feedback_match` 语义兜底（自由
  文本修复建议关键词不中时触发）；chief 18；claim 18。

**验收：216 测试全绿（215+1：空 choices 重试/耗尽/意外异常审计三
路径离线单测）。四档产物在 runs/realtest/（run.log + llm_calls.jsonl
含 usage token 与 http_requests，可直接复核计费）。**

工程结构调整（2026-08-26，最终形态）：代码统一在 `src/` 下且无
`src/atap` 嵌套——`src/core`、`src/llm`、…、`src/cli.py`、
`src/__init__.py`，pyproject `package-dir = {"atap" = "src"}` + 显式
10 项子包清单（import 语义不变，tests/configs/docs/runs 不入发行包）；
根目录只留 README/pyproject/src/tests/configs/docs/runs。计划/审计/
文献与真实 LLM 评测脚本统一在 `docs/`（README 与脚本用法注释内的
路径已同步）。216 测试、demo 6/6×4、闭环管线验收数字均不变；
test_invariants 的 SRC 取 `Path(atap.__file__).parent`，位置无关；
两份审计报告中的 src/atap 路径为当日记录，保留原文。

## 轮次十：上线前真实全量测试执行（2026-08-25，deepseek-v4-flash 直连）

按 `docs/plan_上线前真实测试.md` 执行八档（`configs/final_*.yaml`，
产物 `runs/final/`，报告 `docs/audit_上线前真实测试_2026-08-25.md`）：

* **7/8 档 exit 0、594 条调用零业务失败、判官 prompt 泄漏 0 命中**；
  人工抽检（含逐字对轨核验）无幻觉实锤。P1 过线 6/7：smoke
  6/6·6/6·2/6·恢复 6/6；chief **17/18·18/18**（真实模型最佳）；
  claim 覆盖 14/18·step 7/11；tree 14/18；dover mistake 14/18·
  Validated 17/18·恢复 18/18；v3 闭环/反馈注入 18/18。
* **binary_search 3/18 未达 8/18 门槛，归因=判官能力**：同 corpus 同
  config 下 FakeLLM 基线 15/18（机制正确），deepseek 预测步 13/18
  系统性偏早（二分判官 lower-half 乐观偏置，与 Who&When"二分弱于
  单遍"一致）。处置：主定位用 all_at_once（16/18）/chief，
  bsearch 降级辅助。
* **思考型模型两起预算事件**：v4 的 mast_judge(allow_novel) 首答
  顶满 8192 致 JSON 截断（5 次修复重试全自愈，HTTP 1.17/修复率
  11.9% 双软指标越线）；smoke-corpus 首跑同族但重试仍空 → 该档
  16384 后 mast_judge 18/18 全过。建议 allow_novel/corpus 档统一
  16384（API 实测接受）。
* **⑦ smoke-corpus 续跑时 DeepSeek 账户 402 余额耗尽**（外部阻塞，
  差 ~60 次调用；充值后一条命令收尾，命令在报告 §6）。
* **P0-2 判据精确化（【适配】）**：`feedback_match` 为沙盒环境侧
  匹配器（policy.py 论证在案），prompt 持故障规格不算判官泄漏，
  硬门槛限定判官类 tag；`docs/realtest_audit.py` 实现该区分并
  输出全套审计（P0/P1/P2 + tag 漂移 + `--sample` 抽检，退出码挂
  P0-2）。合计 594 记录 / 627 HTTP / 150 万 tokens（completion
  占 68%），墙钟 ~3h10m。

## 设计要点（对齐文献的两条核心原则）

- 检测 ≠ 归因：analyze 只回答"有没有问题"，attribute 才回答"哪个错误决定了失败"
  （误定位 81% 偏晚、平均滞后 14.5 步——TrajAudit/RootSE）。
- 聚合先于单例：StageAlgorithm 提供 run_corpus 作用域，为跨轨迹算法（SBFL/聚类）预留
  （agent 级 53.5% 可用 vs step 级 14.2%——Who&When）。
