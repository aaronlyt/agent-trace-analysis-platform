# atap 阶段四实施计划

> **状态：四个子轮次全部完成（2026-08-25），208 测试全绿。**
> **论文一致性审计完成（2026-08-25）：11 路只读 subagent 逐模块对照
> refs/ 原文，2 处 🔴 已修复、约 30 处标注失实已修正，210 测试全绿
> ——报告见 audit_阶段四论文一致性_2026-08-25.md。**
> 实施记录见 plan.md 轮次三~六；本文件保留原计划与文献依据。

> 覆盖:R2 信息依赖图(含 CHIEF 层次因果图)、R3 claim 台账(DRIFT)、R4 层级树
> (CodeTracer)、RG/UG 确定性归因(2608.01913)、失败聚类+残差词表扩展
> (AgentDebugX inducer)、分布漂移检测(2511.19933)、L3 动态重放
> (TraceElephant / DoVer)、Langfuse v3 / OTel GenAI 采集适配器。
>
> 算法设计全部对照 refs/ 原文;本轮新增两篇:CHIEF(2602.23701)与
> DoVer(2512.06749)已用 paper-fetch 补入 refs/。标注约定沿用项目惯例:
> 【适配】= 工程适配且论文有依据方向;【推断】= 论文未明说的自行设计;
> 【论文未说明】= 原文缺口,如实保留。

## 0. 总览:四个子轮次与依赖

```
轮次 4a 确定性层(零 LLM prompt)          轮次 4b LLM 表征与归因升级
  represent/idg            ←R0 refs 直出    represent/claim_ledger   (DRIFT A)
  represent/hierarchy_tree ←R0 phase/action attribute/claim_audit    (DRIFT B/C/Tracer)
  attribute/rg_ug          ←沙盒 qrels      attribute/tree_diagnosis (CodeTracer 诊断)
  analyze/drift_detect     ←漂移语料        represent/hcg + attribute/chief (CHIEF)
  classify/inducer(词面路径)
                                            轮次 4c L3 反事实重放
轮次 4d 采集适配器                            sandbox 检查点/消息编辑/续跑基建
  io/langfuse_source                        attribute/counterfactual_replay (TraceElephant)
  io/otel_source                            recover/dover               (DoVer)
```

- 4a 全部确定性、离线可精确验收,先做——同时为 4b/4c 扩语料(qrels、漂移场景、残差场景);
- 4b 依赖 4a 的产物契约(树/图/语料),新增 6 组 LLM prompt + 伪判官规则 + 防泄漏回归;
- 4c 是最大基建改动(沙盒重放),风险隔离在独立轮次;
- 4d 独立,可并行或垫底。
- 每轮 DoD(沿用项目惯例):新测试全绿 + 旧 154 测试回归不变 + 离线 e2e 验收数字
  达标 + subagent 对照原文一致性审计 + README/plan.md 更新。

---

## 1. 轮次 4a:确定性层(跨轨迹 + L0 + 表征基建)

### 1.1 `represent/idg` — R2 信息依赖图(GraphTracer 思想)

**文献依据**:GraphTracer 2510.10581(⚠已撤稿,只吸收思想、不采信数字)。
节点=信息产物 v=(t_v, μ_v, o_v),边=(v_i,v_j) ⇔ o_vj 显式引用 o_vi 且 t_vi<t_vj;
经验规模 |V|≈0.5T、|E|≈2.5|V|;归因排序式 Impact(v)=α·deg⁺(v)+(1−α)·Betweenness(v)。

**数据契约**(产物 `idg`,挂 bundle):
```python
IDGArtifact:
  nodes: [{event_id, index, agent, kind}]        # 只收"被后续引用"的信息产物
  edges: [{src_event_id, dst_event_id}]          # usage 边
  conflicts: [{a, b, note}]                      # 共享后代的两源矛盾对(可选)
```

**算法设计**:
1. 遍历 R0 事件,`refs` 非空事件即派生节点(源节点=检索/工具输出,入度 0);
   【适配】GraphTracer 需事后引用抽取(结构化模式匹配/辅助 LLM),atap 的 R0
   采集层已落 `refs` 引用边,建图为纯确定性 O(V+E),无 LLM;
2. 边=refs 展开(span-id 已由 canonical_events 映射为 event-id);
3. 冲突检测【可选,本期只留接口】:对共享共同后代的两个源节点做两两一致性检查,
   原文用 LLM,atap 留 `conflicts=[]` 占位并在 docstring 声明【论文未说明简化】。

**归因消费接口**(供 4b chief 与未来算法经产物使用,不 import):
`Impact` 排序 + 沿入边反向祖先闭包 A(v_f)(失败终事件),传播路径=根因沿出边
前向遍历。实现放在消费侧(attribute 模块读 `idg` 产物),idg 本身只建图。

**验收**:demo 六故障,GT 根因事件 ∈ 失败终事件祖先闭包 6/6;|V|/|E| 与
0.5T/2.5V 经验值对照记录(玩具轨迹更小,如实记录);确定性 golden 测试。

### 1.2 `represent/hierarchy_tree` — R4 层级树(CodeTracer)

**文献依据**:CodeTracer 2604.11641 §3.1。树节点=归一化 step:exploration 步
(只查环境不改状态)挂**前一步的兄弟**;state-changing 步挂**前一步的子节点**;
节点带 intent+outcome 摘要;stage 为正交维度(stage_ranges 区间);检索靠
tree.md(每行 step_id+短摘要,缩进编码层级)。树索引消融 +18.3pt(三 backbone
平均,Table 4),token 反降(105.1k→56.8k)。

**数据契约**(产物 `tree`):
```python
TreeArtifact:
  nodes: [{step_id, parent, class}]      # class ∈ {exploration, state_changing}
  stage_ranges: [{stage, start, end}]
  tree_md: str                            # 渲染好的压缩索引文本
```

**算法设计**:
1. explore/change 判定:配置化 action→class 映射表。沙盒映射【适配:CodeTracer
   域为代码编辑,研究问答沙盒定义——search/read/verify=exploration;plan 制定、
   消息断言、submit=state_changing】;
2. 建树:顺序扫描,exploration→`parent=prev.parent`,state-changing→
   `parent=prev`;
3. 摘要:确定性模板(action+payload 首行截断 ≤60 字符)【适配:论文摘要生成
   方式未说明,伪判官路径必须确定性;真 LLM 路径可配置】;
4. stage 分段:R0 `phase` 字段直出(沙盒 phase∈plan/search/report 已天然有序)
   【适配:论文五阶段为 SWE 域,映射声明于 docstring】。

**验收**:golden 树结构测试(每故障类型一棵);tree_md 行数 ≪ 事件渲染行数,
压缩比记录。

### 1.3 `attribute/rg_ug` — RG/UG 确定性归因(2608.01913)

**文献依据**:§4.3 逐式。E(q)=主题相关文档集、G(q)⊆E 为 gold 充分集;
episode=一次 search call 起至下一次 search 前;R_k=episode 内 search/visit 返回
且 ∈E(q) 的 doc id 精确匹配;C_k=∪_{j≤k}R_j;G*=C_M∩G(q)。判定:
success→correct;失败∧G*=∅→**RG**;失败∧G*≠∅→**UG**。子类:RG 分
directional(C_M∩E=∅)/last-hop;UG 分 true-extraction(G*=G)/boundary。
扩展量:episode 效用(productive/redundant/unproductive)、k*=最后涌现步、
wasted tail=M−k*、visit precision。

**前置:沙盒补 qrels**。mock 语料构造时已知每任务的相关文档与 gold 集,落
`corpus metadata: {task_id: {evidence: [...], gold: [...]}}`;doc id 全局唯一
贯穿检索与读取返回。

**数据契约**(产物 `rgug` + 统一 Hypothesis):
```python
RGUGArtifact:
  label: correct | RG_directional | RG_last_hop | UG_true_extraction | UG_boundary
  episodes: [{k, agent, docs, utility, delta}]
  first_gold_hit: int | None
  wasted_tail: int
```

**Hypothesis 映射**【适配:论文输出是轨迹级二分+episode 效用,不定位
agent/step;atap 映射——RG→(searcher, 首个连续 unproductive episode 的起点,
root_cause_code=retrieval_gap,fix_suggestion=换查询式);UG→(reporter,
决策步即 G* 非空后仍未引用 gold 的首个决策事件,code=utilization_gap,
fix=饱和即停/强制引用),confidence=1.0(确定性)】。success 判定用沙盒
verifier 结果替代论文 GPT-4o 二值判定【适配:确定性,免 LLM】。

**新故障注入**:`retrieval_gap`(gold 文档存在但 mock 排序置于 top-K 之外)、
`utilization_gap`(gold 正常返回但 reporter 忽略)——RG/UG 标签按构造已知。

**验收**:三类构造场景(RG/UG/correct 各 ≥2)标签 100% 命中;24 条语料上
RG/UG 分布报告;与 all_at_once 的 LLM 调用对比(本算法 0 调用,进 compare 表)。

### 1.4 `analyze/drift_detect` — 分布漂移检测(2511.19933)

**文献依据**:该文是立场论文——给出"检测什么":三类漂移定义
(**version drift**=模型更新导致工作流变化;**data drift**=输入分布偏离;
**behavior drift**=同提示跨时窗输出漂移)+ 监测信号清单(输出方差/格式变化/
行为指标纵向采样)。**无算法、无公式、无阈值**——统计实现全部为
【适配:工程选型,综述明示 PSI 为工程选型非论文内容】。

**算法设计**(run_corpus 作用域,确定性):
1. 分组键:(model_version, prompt_version, window);window=语料批次序号;
2. 特征向量(跨轨迹聚合):agent×action 直方图、轨迹长度分布、循环率
   (loop_detect 产物汇总)、失败率、格式异常率(rule_pack malformed 触发率);
3. 检验:离散特征 PSI(分箱)、连续特征 KS;PSI>0.2 或 p<0.01 告警
   【推断:阈值工程默认,参数化暴露于 config】;
4. 三类漂移判定:version=跨版本桶比较(控 window);data=跨任务域桶;
   behavior=同(任务,版本)跨 window。

**数据契约**(产物 `drift`):
```python
DriftReportArtifact:
  groups: [{key, n_traces, features}]
  pairwise: [{a, b, psi, ks_p, drift_type, alert}]
```

**沙盒漂移语料**(`atap corpus --drift`):①中途切换 FakeLLM"版本"(某版本
注入重复倾向→动作分布变化);②前半 A 任务域、后半 B 域;③同任务分窗加
随机扰动。

**验收**:三类构造漂移全部检出且 drift_type 正确;稳定语料(同分布重复批)
零误报;确定性 golden。

### 1.5 `classify/inducer` — 失败聚类+残差词表扩展(AgentDebugX §3.4)

**文献依据**:§3.4 单段文字——judge 遇 seed 外**复发**失败记 novel-mode
candidate;聚类="label, then lexical or embedding similarity, gated by a
support threshold";每簇提名一个候选模式;对 seed 去重;**提案永不覆盖策展
词表**,人工接受后生效(附录 E:inducer implemented but not yet evaluated,
requires human acceptance)。嵌入模型/阈值/命名模板均【论文未说明】。

**算法设计**(run_corpus 作用域):
1. 残差来源:judge_eval 的 finding 允许 `novel` 输出【适配:现 MAST judge 丢弃
   未知码;加 `allow_novel` 参数——prompt 增"若症状不属于任何允许码,输出
   failure_mode_id='novel' 并给症状短语";伪判官=症状词表未命中但确有失败】;
2. 聚类:字符 3-gram Jaccard 相似度层次聚合(词面路径,确定性);embedding
   路径留接口【推断】;support_threshold=3(簇大小下限,参数化);
3. 提名:每簇一模式,模板命名"{top 症征关键词} {kinship 后缀}",definition=
   簇内高频证据片段拼接,kinship=与 seed(MAST 14)最近的类(词面相似度);
4. 与 seed 去重: nominees 与 seed 词面相似度 >0.8 丢弃【推断:阈值自设】;
5. 产物:`TaxonomyProposalsArtifact {proposals: [{mode_id, label, definition,
   kinship, support, evidence_trace_ids, status: proposed}]}`;接受经 CLI
   `atap taxonomy accept <id>`(写回词表文件,版本号+1;测试模式 auto_accept)。

**新故障注入**:`agent_deadlock`(planner 与 searcher 互相等待,跨 ≥3 轨迹
复现)——论文同款示例场景。

**验收**:deadlock 注入语料→恰好 1 提案、kinship 指向 FM-2.x 协同类;
全可标语料→0 提案;接受后 MAST judge 可用新码打标(闭环)。

**轮次 4a DoD**:全部确定性模块离线验收达标;~35-45 新测试;不变量测试
扩展(新模块 import 图);compare 表纳入 rg_ug(0 LLM calls)。

---

## 2. 轮次 4b:LLM 表征与归因升级(R3 / R4 诊断 / CHIEF)

### 2.1 `represent/claim_ledger` — R3 claim 台账(DRIFT §4-A)

**文献依据**:式(2) 六元组 c_k=(a_k, i_k, b_k, U_k, τ_k, σ_k):a=claim 文本、
i=引入 span、b=首次 consequential span、U=复用 span 集、τ=类型
{entity/constraint/evidence/retrieval/compute/process}、σ=承诺状态
{exploratory/tentative/consequential/finalized}。LLM **全局单遍**抽取(非增量),
只留 decision-critical claims,台账保持 3-5 条;finalized 仅用于提交/最终答案。

**算法设计**:
1. prompt(附录 G Prompt 2 结构):system=仅输出 JSON;user=角色/任务/
   schema/输入 question+事件渲染;校验:类型与状态封闭枚举、step 越界钳制;
2. span= R0 事件【适配:DRIFT 的 span 边界靠边界信号切分,沙盒事件边界天然
   给出】;
3. 伪判官(确定性):从 planner/reporter 消息按句式规则抽 claim
   (如 "use X as the answer"/"document D establishes Y"→evidence 类,
   plan 步→process 类),b=首次被后续事件 refs 的位置,U=所有复用处。

**验收**:六故障语料,伪判官台账条数 3-5、类型合法率 100%;info_withholding
故障的 claim 其 U/b 构造已知(声明了约束但未传达)。

### 2.2 `attribute/claim_audit` — 支撑四级+专家审计+依赖回溯(DRIFT §4-B/C)

**文献依据**:B Support Seeker 逐 consequential claim 出
DIRECT/WEAK/MISSING/CONFLICTING(高召回、故意过度路由);C 专家审计器按
6 类 claim 类型路由,输出 verdict∈{supported, harmful_unsupported_commitment,
conflicting_support, insufficient_but_nonharmful}、decisive_defect(8 值)、
failure_mechanism(9 值)、responsible_span 等;Dependency Tracer 以
prior_support_trace.error_span_ids 为工作集保守回溯——保留 first_error_span
除非其只是查询/工具/重试;输出 {first_error_span, follow-up_error_spans,
error_span_ids}。Ê={s|h(s)=1}:commit/reuse/amplify/finalize 有害 claim 的步。

**算法设计**(三段,共 1+B 个+C 个+1 个 LLM 调用):
1. B:逐 claim 判四级。伪判官:claim 引用的 doc id 是否出现在其引入步之前的
   读取集中(在且被 verifier 支持→DIRECT;在但未验证→WEAK;不在→MISSING;
   与读取内容矛盾→CONFLICTING);
2. C:needs_auditors 路由到对应类型审计器,输出封闭枚举(伪判官规则化:
   如 evidence 类 MISSING→harmful_unsupported_commitment);
3. Tracer:保守回溯得 first_error_span→映射 Hypothesis(step=index,
   agent,root_cause=claim 文本+缺陷码)。
   【适配:DRIFT 的 18 故障 taxonomy(Table 4)作为 C 的 decisive_defect 枚举
   引入,不新建 classify 模块(与 MAST 并存,供 compare)】。

**验收**:伪判官六故障 first_error 命中 ≥5/6(info_withholding/
ungrounded_citation/premature 类故障是 DRIFT 主场);真 LLM 可选跑
(文献校准:FEA +4.2~13.4pp 方向)。

### 2.3 `attribute/tree_diagnosis` — 树索引诊断(CodeTracer §3.2)

**文献依据**:判官三步 workflow——tree.md 定位可疑区域→stage_ranges 映射
区间→按需 inspect steps("Do NOT iterate over all steps")。原文为终端命令式
agent(INSPECT/WRITE/FINALIZE 纪律、反作弊约束:标过的 step 必须 inspect 过、
覆盖 ≥3 stage、可疑 stage 至少查 2 步含一邻居)。

**算法设计**【适配:atap 判官是无状态 JSON 调用,以两次结构化调用近似
"先树后钻",命令式 agent 交互不引入】:
1. 调用①(树级):输入 tree.md+任务;输出可疑 stage 排序列表+理由;
2. 调用②(钻取):渲染 top-1 stage 的完整事件+邻居事件;输出 Hypothesis
   (step/agent/root_cause);
3. 伪判官:在可疑 stage 内取首个 rule_pack/loop_detect 触发事件。
   消费 `tree` 产物(经 bundle,不 import)。

**验收**:六故障 step 命中(伪判官)≥5/6;与 all_at_once 同语料对比:token
消耗应下降(文献方向:树索引省 token 且 +18.3pt,进 compare 表)。

### 2.4 `represent/hcg` + `attribute/chief` — CHIEF 层次因果图归因

**文献依据**:CHIEF 2602.23701(Who&When 算法生成子集 step 级 52% 迄今最高;
token 为 all-at-once 的 2.5~3×)。HCG 三层:subtask 节点(RAG 检索 2 范例
few-shot 分解+Trajectory-Aligned Reflection 校正)/agent 节点(OTAR=
⟨Observation, Thought, Action, Result⟩)/step(数据流边);三类边 E_sub∪E_agt∪
E_step,子任务/agent 边绑定反事实模式 Φ:Bias(u)→Φ Anomaly(v)。回溯:
oracle(非人工,LLM 合成 O_k=⟨G_sub, P_pre, E_key, C_acc⟩,式 2 顺序生成)
引导 top-down 三级剪枝(子任务逆拓扑序 F_eval→agent OTAR 对照→step 细节)。
渐进因果筛选四步:Local Attribution(式 6:Pre(x) 中 Bias→Φ Anomaly(x) 者
非空则上游传播)/Planning-Control(loop group 聚合,planner 重复信号仍同令→
归 planner)/Data-Flow(沿 step 边找首次污染,生成者=根因)/Deviation-Aware
(后续自愈的偏差可逆不担责)。输出 (i*,t*)=时间最早 decisive error(式 1),
完全遵循 Who&When 协议。**不做真重放**,反事实在图上审计近似。

**拆分设计**(算法只经产物解耦):
- `represent/hcg`:建图。LLM 调用≈分解 1+对齐 1+OTAR 1+边 2;
  【适配:RAG 检索退化为静态 few-shot(沙盒任务族小);伪判官——subtask 由
  R0 phase 直出,OTAR 由事件 payload 规则化,step 边复用 idg 产物或 refs】;
  产物 `hcg`(三层节点+三类边+Φ 标注);
- `attribute/chief`:oracle 合成(每 subtask 1 次顺序调用;伪判官从任务 spec+
  语料元数据规则化 G_sub/C_acc)→三级回溯(伪判官:输出 vs C_acc 文本比对)
  →渐进筛选四步(确定性图运算+伪判官 F_eval)→Hypothesis(最早 decisive
  error,confidence=oracle 匹配度)。

**验收**:伪判官六故障 step ≥5/6(malformed 等动作级故障可能不适配 HCG 语义,
如实记录 miss);LLM 调用与 token 计数进 compare(预期 2-3× all_at_once,
文献方向);w/o oracle 变体可配置(对照 45.60%)。

**轮次 4b DoD**:6 组新 prompt 全部过防泄漏回归(prompt 全文无故障类型词/
GT 键);伪判官验收数字达标;~40-50 新测试;compare 表新增
claim_audit/tree_diagnosis/chief 三行。

---

## 3. 轮次 4c:L3 反事实重放(TraceElephant / DoVer)

### 3.1 沙盒基建:检查点重放中间件

**文献依据**:TraceElephant §3.2/A——轻量 LLM API 中间件透明录制请求 payload/
响应/工具交互,重放=从对应执行点重新运行,经中间件改写候选步输入,只看后续
k=3 步;DoVer M1——逐步 checkpoint/加载/改消息/续跑,每干预 ×3 次。两文均未给
缓存/去随机细节【论文未说明】。

**设计**(`sandbox/` 扩展,core 不动):
```python
RunRecord:            # env.record() 产物
  events: list[TraceEvent]           # 即原轨迹
  llm_log: [{step, agent, request, response}]
  tool_log: [{step, call, result}]   # mock 检索结果缓存
env.rerun_from(record, t, edits: dict[step, new_text]) -> RerunResult
  # 截断至 t,对 edits 指定消息原位替换,续跑到结束或 k 步
```
- FakeLLM/mock 按 (agent, request 哈希) 命中录制响应保证前缀一致【推断:设计
  补充】;分支后的新请求走正常生成(FakeLLM 确定性→同输入同输出);
- 不变性测试:无编辑 rerun_from(t)== 原轨迹后缀(逐事件相等),∀t。

### 3.2 `attribute/counterfactual_replay` — L3 终审(TraceElephant A.6.3)

**算法设计**:
1. 输入:bundle 内其它归因器的 Hypothesis 候选(去重后取 ≤3);
2. 每候选:判官一步自推 oracle(该步应有输出)【论文:期望 oracle 模型自推,
   非人工】;`rerun_from(t*, 干预改写)`→只验证后续 k=3 步满足 oracle 且失败
   模式不再现;
3. 输出:候选 verdict∈{validated, refuted} 写回 Hypothesis.confidence/
   evidence(滤伪因果:refuted 候选降权);参数 k=3、候选≤3(论文值);
   FakeLLM 温度恒 0(论文 0.3 为真模型稳定性设定)。

**验收**:六故障——编辑 GT 根因步(伪判官给正确 oracle)→validated 6/6;
编辑非根因步→refuted(构造 ≥2 例);不变性测试通过。

### 3.3 `recover/dover` — do-then-verify 恢复(DoVer §4)

**文献依据**:Pipeline=①Trial Segmenter(规划步/re-plan 步为切点)②每 trial
Summarizer+Failure Proposer(改进 all-at-once,hypothesis=(步,agent,理由))③
Intervention Recommender(严格 JSON:category∈{orchestrator_ledger(最小
FACTS/PLAN_REPLACEMENT 片段), orchestrator_instruction(单个原子 next step),
subagent_instruction};"Keep changes minimal";"Do not give any ground truth")④
checkpoint 重放原位替换消息 ×3 ⑤Milestone Extractor(K≤5)+Evaluator(进度
A(τ̃)−A(τ),achieved/partial/missed+new_path 评估)⑥Outcome Classifier 四判定
(Validated≥2/3 成功;Partially<2/3 成功且≥2/3 忠实执行且进度≥20%;Refuted=
忠实执行仍无进度;Inconclusive 其余,区分"未执行"vs"定位错")。

**算法设计**:
1. 沙盒映射【适配:论文域为 Orchestrator 多 agent;沙盒——orchestrator→
   planner、subagent→searcher/reporter;切点=planner 的 plan/re-plan 消息
   (伪判官确定性识别);milestone 由 GT 答案+语料元数据规则化(需读哪些
   gold 文档、需给出含关键事实的报告),K≤5;无标注场景论文留作 future
   work,沙盒按构造生成】;
2. 主循环=§6 伪代码(segment→summarize→recommend→replay×3→classify);
   refuted 后切下一 trial 假设【论文未说明切换细节,按 trial 批量独立实现】;
3. 产物:每假设 {label, progress, verdict 证据} + 若干预即恢复→复用闭环验证
   (recover 产物回 analyze,现有 `run_closed_loop`);
4. 与 targeted_rerun 的本质区别(docstring 声明):消息**原位替换** vs 追加
   反馈;**结果差分**(里程碑进度)vs 外部 verifier 反馈——论文实测
   CRITIC 式追加反馈基线 0% 翻转 vs DoVer 17.6%(WW-GAIA)。

**验收**:离线(伪判官+FakeLLM)六故障:恢复 ≥5/6(对照 targeted_rerun 6/6
基线,允许一例如实 miss);四判定分布报告;LLM 调用计数进 compare(预计
≈1+T+T'+1+3R+1,T≈2-3)。

**轮次 4c DoD**:重放不变性测试(全步扫描);counterfactual_replay 与 dover
验收达标;~30-40 新测试;README 更新"L3 金标准终审"一节。

---

## 4. 轮次 4d:采集适配器(Langfuse v3 / OTel GenAI)

**规范调研结论**(2026-08):Langfuse v3 数据模型 Trace+Observation
(type∈SPAN/GENERATION/EVENT,parentObservationId 成树),经典 ingestion=
`POST /api/public/ingestion` batch(trace-create/observation-create 事件);
**v4 已弃用该端点**(云上 2026-11 停用)迁 OTLP/HTTP `/api/public/otel/v1/traces`,
原生解析 `gen_ai.*` 与 openinference 属性。OTel GenAI semconv 已迁
semantic-conventions-genai:`gen_ai.operation.name`(必填)、
`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.usage.*`;
内容类(input/output.messages、tool.call.*)为 Opt-In PII;消息
={role, parts[{type: text|tool_call|tool_call_response,...}]}。

### 4.1 `io/langfuse_source.py`
- 输入:v3 ingestion batch JSON 导出文件(兼容 v4 OTLP JSON——span 属性含
  `langfuse.*` 映射);
- 映射:observation.type+name→action(GENERATION→llm_call、tool 类 SPAN→
  tool_call、EVENT→log);input/output/model/usage→payload;id/
  parentObservationId→refs/parent;startTime/endTime→phase 起止;trace 级
  userId/sessionId/tags→trace metadata;
- 丢失声明:Scores(需另端点)、成本单价、v4 无 trace 级 input/output(由根
  observation 重建)——未映射属性整体存 `payload["extras"]` 防二次损失。

### 4.2 `io/otel_source.py`
- 输入:OTLP/HTTP JSON traces;
- 映射:`gen_ai.operation.name`→action(chat→llm_call、execute_tool→
  tool_call、invoke_agent→agent_step、retrieval→search);spanId/parentSpanId/
  tool_call id(请求-响应配对键)→refs;attributes 未消费项→payload extras;
- 校验:Opt-In 未开启时 payload 缺失→显式 warning 不静默。

**验收**:`atap export --format langfuse|otel` 把 demo 轨迹导出→重新导入→
R0 事件语义等价(字段级 roundtrip 断言);OTel gen_ai fixture(手工构造含
tool_call 配对)解析正确;不变量:io/ 不依赖 stage 包(现有测试自动覆盖新模块)。

---

## 5. 框架级改动与测试

- **产物类型注册**(core/bundle.py):新增 idg/tree/ledger/rgug/drift/proposals/
  hcg/replay 八类,只加常量不动核心逻辑;
- **配置**:全部新算法走既有 `@register`+YAML 组合,零改核心;新增
  `configs/pipeline_offline_v4.yaml`(4a+4b 全栈)、`pipeline_dover.yaml`;
- **CLI**:`atap corpus --drift`、`atap taxonomy accept/list`、
  `atap export --format`、compare 表扩展;
- **测试矩阵增量**(预计 +140~180,总数 ~300+):
  - 确定性 golden:idg/tree/rg_ug/drift/重放不变性;
  - 伪判官:claim/chief/tree_diagnosis/dover 各自验收场景;
  - 防泄漏回归:4b 全部新 prompt + dover prompt 全文扫描(禁故障类型词/GT 键);
  - 不变量:新模块 import 图、io 独立性、registry 契约;
  - 审计回归:每轮 subagent 对照原文审计后落 test_audit_fixes 风格回归。

## 6. 验收总表(文献校准)

| 模块 | 轮次 | 离线验收(伪判官/FakeLLM) | 文献参照数字 |
|---|---|---|---|
| idg | 4a | GT 根因 ∈ 祖先闭包 6/6 | |V|≈0.5T,\|E\|≈2.5V(经验) |
| hierarchy_tree | 4a | golden 树 + 压缩比记录 | 树索引 +18.3pt、token 105k→57k |
| rg_ug | 4a | 构造场景标签 100% | RG 占错误 51.6–64.1%(分布对照) |
| drift_detect | 4a | 三漂移全检出/稳定零误报 | 论文无算法(仅定义) |
| inducer | 4a | 恰好 1 提案/0 误提案 | 论文未评测 |
| claim_ledger+audit | 4b | first_error ≥5/6 | FEA +4.2~13.4pp |
| tree_diagnosis | 4b | step ≥5/6 + token 降 | +18.3pt 方向 |
| hcg+chief | 4b | step ≥5/6 | step 52%(算法子集)、token 2.5–3× |
| counterfactual_replay | 4c | validated 6/6 + refuted 构造例 | 动态 33.3% vs 静态 30.3% |
| dover | 4c | 恢复 ≥5/6 + 四判定分布 | trial 成功 17.6–49%、Validated 15–35% |
| langfuse/otel source | 4d | roundtrip 字段级等价 | — |

## 7. 风险与未决

1. **CHIEF/DoVer 无开源**:伪判官只证管路与契约;真 LLM 数字以文献方向校准
   (如"干预优于追加反馈"、"oracle 引导剪枝省 token"),不承诺复现具体 pp;
2. **GraphTracer 已撤稿**:idg 只吸收图模型与排序思想,不引用其任何数字;
3. **inducer/drift 论文本身无算法**:实现以【适配/推断】如实标注,验收自设;
4. **重放一致性**:真 LLM 路径前缀一致依赖响应缓存,冷缓存时退化为论文的
   "重新执行"口径(温度 0.3)——config 二选一,产物记录口径;
5. **DoVer milestone 无标注场景**:沙盒按构造生成;真数据留 future work
   (论文亦如此);
6. **Langfuse v4 迁移窗口**:双格式支持,meta 记录来源端点版本。

## 8. 落地节奏建议

4a → 4b → 4c 串行(依赖),4d 可与任意轮并行。每轮完成即更新 plan.md 进度
与 README 算法表,并按项目惯例跑一轮"多路 subagent 论文一致性审计"。
