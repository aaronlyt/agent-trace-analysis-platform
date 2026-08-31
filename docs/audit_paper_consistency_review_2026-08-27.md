# 论文一致性复审报告（22 路独立 subagent，2026-08-27）

> 范围：src/ 下全部 21 组算法模块 + 2 个 io 适配器，逐组一个只读 subagent，
> 三方比对：refs/ 论文原文（paper.md）× 实现代码 × 测试（读断言 + 实跑）。
> 与前两轮（audit_paper_consistency_2026-08-25.md、audit_stage4_paper_consistency_2026-08-25.md）
> 互补：本轮重点（a）验证前两轮"已修复"声称是否真实落地；（b）用新鲜眼光
> 找新问题；（c）评估测试是否真正断言了论文机制。
> 判定口径：✅忠实 / 🟡已声明适配 / 🔴未声明偏离 / ⚫误读 / 🐛bug；
> 严重级 P0(泄漏/违规) / P1(算法语义错误或未声明实质偏离) / P2(标注失实/
> 修复无效/测试缺口) / P3(瑕疵)。

## 0. 测试执行汇总

- 主线全量基线：`pytest tests/` **216 passed / 0 failed**（复审前后各跑一次，一致）。
- 22 路 subagent 各自实跑其模块测试文件，**全部通过，零失败**。
- 但测试质量层面发现 2 处空洞/恒真断言（见 §3 测试质量），绿灯不代表机制被钉死。

## 1. 总览

| 模块 | 论文 | 评级 | 上轮修复验证 | 新发现 |
|---|---|---|---|---|
| canonical_events | AgentDebugX | 🟡 | P3-8 四项 ✅ | 4（P2×1 测试缺口，P3×3） |
| ssf | TrajAudit | 🟡+🐛 | **P3-5 ❌ 修复无效**；P3-6 ✅ | 3（P2×2） |
| action_signature | TraceProbe | ✅ | P3-1/2/3 ✅ | 5（全 P3） |
| loop_detect | TraceProbe | ✅ | P1-1 ✅（口径+边界回归均在） | 3（全 P3） |
| idg | GraphTracer(撤稿) | 🟡 | 4/4 ✅ | 1（P3） |
| hierarchy_tree + tree_diagnosis | CodeTracer | 🟡 | 4 项标注 ✅ | 3（P2×2） |
| claim_ledger + claim_audit | DRIFT | 🟡+🔴 | 标注 ✅（文档措辞与代码注释不完全一致） | 4（**P1×1**，P2×1） |
| hcg + chief | CHIEF | 🟡+🔴 | 5 项标注 ✅，但 §5/§6 描述已过时 | 4（**P1×1**，P2×1，P3×2） |
| judge_eval | MAST | 🟡 | P1-2 / P1-3 ✅ | 3（P2×1，P3×2） |
| mast_judge + taxonomy | MAST | 🟡 | P2-1 基本✅、P2-2/P2-3 ✅ | 4（P2×1，P3×3） |
| rule_pack | AgentDebugX | ✅ | P3-7 ✅ | 3（P2×1，P3×2） |
| inducer | AgentDebugX §3.4 | 🟡 | 2/3 ✅（plan 契约差异未闭合） | 4（P2×2，P3×2） |
| drift_detect | 2511.19933 | ✅+🟡 | 统计口径 4 项 ✅（PSI 独立验算恒等） | 3（全 P3） |
| all_at_once | Who&When | ✅ | few-shot off-by-one ✅；**12.5 口径 ❌ 未补** | 2（全 P3） |
| binary_search | Who&When | ✅ | P3-4 ✅（Algorithm 2 逐行等价复验） | 3（全 P3） |
| sbfl | FAMAS | ✅ | P3-9 ✅（六式手工验算全对） | 2（P3，测试零数值断言另计） |
| rg_ug | 2608.01913 | ✅+🟡 | 3 项 ✅（独立重算 8 轨迹全一致） | 3（全 P3） |
| counterfactual_replay | TraceElephant | 🟡 | tag/标注 ✅；运行期防泄漏 ✅ | 5（P2×1，P3×4） |
| targeted_rerun | AgentDebug | 🟡 | P3-11/12 ✅ | 3（P2×1 测试失效，P3×2） |
| dover | DoVer | 🟡 | 两处 🔴 修复 ✅（实测 `_redact_fault_names` 生效、运行期扫描真扫全量 calls） | 3（P2×1，P3×2） |
| feedback_injection | AgenTracer | ✅ | P3-10 ✅（数字引用核对一致） | 4（全 P3） |
| io/langfuse + io/otel | 规范型 | ✅ | 🔴 修复全部落地（hex 实测合规） | 5（P2×2，P3×3） |

**总体判断**：前两轮的修复绝大多数真实落地且经得起复验（loop_detect 窗口口径、
judge_eval outcome 剥离、taxonomy 收敛、dover 双 🔴、otel hex 等重点项全部 ✅；
sbfl 六式 / rg_ug 集合运算 / binary_search Algorithm 2 经独立手工验算逐式一致）。
本轮新发现集中在：**2 处 P1**（claim_audit 成功路径泄 gold、chief 的 HCG 图
全管线零消费）、**1 处修复无效**（ssf unfold_line，且其回归测试是空洞的）、
以及一批 P2 级未声明行为/测试失效。默认运行路径无 P0 泄漏。

## 2. P1（本轮新发现，建议优先处理）

1. **claim_audit：include_success=True 时 gold 答案泄入判官 prompt**。
   claim_audit.py:141 + render.py:124-128 + env.py:211——成功轨迹的
   outcome note 含 `matches gold 'all-at-once' (d3)`，judge_view 默认带
   outcome 行 → 标准答案直接进 prompt。docstring"恒 FAILURE、不泄漏"声明
   对该路径失实。默认路径（只失败轨迹）实测无泄漏。
   > **修正（评审修复轮 2026-08-27）**：当时的结论"环境反馈属轨迹本身、
   > 判官可见是构造使然"不成立——VERIFIER 行的 gold 文案是 oracle 标注，
   > 不是轨迹挣来的信息，成功轨迹经 judge_eval 同样泄（实测 14 条
   > judge_eval 调用中 8 条命中）。本轮已改 env.verify 成功文案为不含
   > gold 的措辞，并把"matches gold"加入全部泄漏回归 token 表（扫描不再
   > 剥 TRACE 块）。
2. **chief：HCG 图从未进入任何判官 prompt，全管线零消费**。
   chief.py:222-234 localize prompt 只有"失败 subtask 提示+全轨迹视图"；
   E_step / E_agt / agent_nodes / phi_patterns 无任何消费者（grep 证实）。
   论文 Fig.11 显式含 {graph}、Eq.4 评 OTAR、Eq.6/§4.3.3 依赖 Pre(x)/
   E_step/Φ。hcg.py:37 "consumed by chief" 措辞误导；偏离声明只写了
   剪裁语义，未写"图未入 prompt"。HCG 对归因的实际贡献只剩 subtask 区间
   切分——README"CHIEF step 6/6"的机制基础与论文不符（离线数字是伪判官
   handler 给的，本来就已声明不可比，但机制层缺口应显式声明或补实现）。

## 3. P2（修复无效 / 标注失实 / 未声明行为 / 测试失效）

1. **ssf P3-5 修复无效（上轮修复声称失实）**：ssf.py:147 改成非锚定
   `search`，但 `FOLD_PLACEHOLDER_RE`（render.py:65）本身带 `^…$` 锚定，
   渲染行以 `[7]` 开头 → 仍必然不匹配，unfold_line 原样返回。实证：1143
   字符原文未恢复。需改正则去锚定。
2. **ssf 回归测试空洞**：test_audit_fixes.py:283-294 两条断言对"未展开的
   原行"也成立（startswith 平凡真；"TrajAudit"来自摘要而非展开文本），
   掩盖了上条。
3. **targeted_rerun 测试恒真**：test_recover.py:53 `assert … or True`，
   且检查错字段——UpdateFeedback 精炼实际零断言。
4. **tree_diagnosis 防作弊声明不实**：tree_diagnosis.py:18-20 称 attributed
   step 必含于 drill 渲染区间（模块保证），:169 实际只 clamp 到全轨迹。
5. **tree_diagnosis 同名 stage revisit 未处理**：按名字取第一个 span，
   论文附录 A stage 可 revisit 且各连续块独立 span（沙盒各阶段恒出现
   一次，未暴露，未声明）。
6. **judge_eval few-shot 引用 GT 起始步**：judge_eval.py:74-83 的示例
   "step 3/step 9" 恰为 malformed_tool_call、ungrounded_citation/
   disobey_task_spec 的 GT 起始步，违反第三轮"few-shot 不得引用 GT 步号"
   惯例；反泄漏测试只覆盖 MAST/AAO 的 few-shot，不含 judge 的。
7. **taxonomy FM-2.6 收敛不彻底**：删掉了论文后果子句 "potentially
   resulting in unexpected or undesired behaviors"，与 P2-1 恢复标准
   不一致（FM-2.6 恰是沙盒 malformed_tool_call 的 GT 映射码）。
8. **rule_pack 静默抑制**：loop_detect 产物存在但只含 re_read_churn/
   tool_oscillation 命中时返回空——不回退 R5、不留 skip-note，重读循环
   被漏检（R5 回退本含 FILE_READ，恰在产物存在时被禁用），未声明。
9. **inducer 非确定性**：pseudo_judge.py:364 并列最高频时 `max` 遍历
   set，顺序随哈希盐漂移——跨进程提案 name 不稳定（6 进程实测出现两种
   name），违背 inducer"确定性"声明，accepted 词表可漂移。
10. **inducer kinship 机制实效失效**：本域 kinship 恒 None（实测），
    plan_stage4.md:195 验收"kinship 指向 FM-2.x"未达，"惰性"未声明。
11. **counterfactual_replay 步独立性漏洞**：policy.py:801 fault_removed
    只看 edit_text 是否命中故障关键词、与干预步无关——对任意步（含症状步）
    施以含故障名的编辑即判 validated；"滤伪因果"实由伪判官 oracle 兜底
    而非重放机制本身；:777 docstring 与代码不符。
12. **claim_audit 依赖结构无输入**：b_k/U_k 台账采集后从未送入 B/T
    调用（claims_json 仅 introduced_step），论文 "later spans depend on
    it" / Prompt5 保链无输入，未声明。
13. **dover 失败归因 proposer 输入未切片**：dover.py:222-246 为全量 view，
    未按 Fig 6 切 trial 日志、无 previous_trial_summary——docstring 只声明了
    干预推荐器的全轨迹偏离，此项漏声明（沙盒 T=1 掩盖）。
14. **io 导出静默丢轨迹**：对未展平轨迹（events=[]、仅 raw spans）otel
    导出 0 spans 整条丢失、langfuse 只剩 trace-create；cli.py:134-149 不
    先 flatten 也不警告；现有测试只数 3 条测不出。
15. **io roundtrip 断言缺口**：otel 侧未断言 task/meta/qrels/GT 防泄漏、
    双侧均未断言 outcome.score/note（恰是上轮修点）、refs 只比数量不比指向。
16. **sbfl 测试零公式数值断言**：γ/β/α/n_cf^λ/Kul2/S 具体值、λ=0.9 默认
    均未钉死；test 用 `max(…, key=confidence)` 取 top 而所有 confidence
    恒 0.35，靠 max 取首才碰巧正确（本审计已手工验算公式全对，建议补数值
    回归）。

## 4. P3 摘要（择要，全量见各模块 subagent 记录）

- canonical_events：沙盒从不传 parent → 树拍平/DFS/parent 赋值零测试覆盖、
  dropped_refs>0 分支从未触发；ts 声明半失实（langfuse ts 被丢弃）；未知
  操作回退 "SPAN" 越词表无校验；重复 span id 后写覆盖。
- action_signature：PLAN/AGENT_SPAWN 无条件 RECORDED 不校验成功；锚子串
  误匹配（d1 命中 "[d10"，当前语料 d1–d6 为潜伏）；REASON target=None 使
  LCS 任意互配；失败读计入 M1/M3 与锚集；coverage 分母口径。
- loop_detect：tool_oscillation per-target 子序列推断未标【推断】；targets
  未滤 None；"recurs ≥2" 歧义解读未声明；tool_oscillation 无负例。
- idg：节点缺 o_v（观察/结论本体）字段裁剪未标注。
- chief：docstring 机制枚举漏 upstream_propagation、误写 loop_executor；
  52% 为 w/G 口径而实现是 w/o G（对应 45.60）；hcg E_step 全量 refs 超集
  vs 原文 LLM 筛"meaningful"。
- hcg/chief、tree_diagnosis：运行期防泄漏扫描只覆盖 dover+cf
  （test_stage4c.py:203），chief/tree/claim 无；mast_judge few-shot 防泄漏
  测试未纳入 EXTRA_FAULTS；dover forbidden 列表漏 EXTRA_FAULTS 两键
  （正则已覆盖，无实泄漏）。
- mast_judge：FM-1.1 "explicit"≠"specified"、FM-2.1/2.3 措辞、FM-3.1 窄化、
  FM-3.2 丢 "or inconsistencies" 等改写残留；"MAST J.1"指称错误；空
  definition 可入 prompt。
- all_at_once：12.5 属 With-GT 子列的口径仍未注明（上轮备注项 ❌）；判官
  视图默认含 outcome 行（论文 G.1 无此信息，未声明增强）；failure_mode
  非法码静默置 None 不留痕。
- binary_search：_parse_half 无否定处理（"lower half looks clean"会被
  误判，与 lower-half 偏置叠加的结构性促因）；不可解析答案直接 ValueError
  无重试；全 lower 收敛 s*=0 时 env 回退退化。
- rg_ug：UG 侧与 plan 契约两处行为差异未像 RG 侧那样标注；矛盾检测窗口
  episode 级粒度差；wasted_tail 无 productive 时取 M 未标注。
- counterfactual_replay：intervention_applied 恒 True（TOOL_CALL 实际未
  替换也记 True）；论文 temperature=0.3 未实现未标注；窗口含干预步本身
  （一步偏移）；step 被钳时 window_events≠k。
- dover：_redact_fault_names 正则不剥空格变体（"info withholding"）而
  环境侧接受空格变体——残余回显通道；off-by-one 无多 plan 回归。
- drift_detect：policy.py:558 "19-vs-3" 数字陈旧（实际 19 vs 6）；version
  漂移与时间窗在语料中混淆未声明；同分组双族冗余 skipped。
- inducer：max_proposals 按首现序截断非 support 降序；accept 幂等跳过无
  审计痕迹。
- feedback_injection：成功即停无直接断言；泄漏扫描仅单故障；伪判官 fix
  模板嵌故障名使 fake 管线"恢复"近同义反复（输出侧，非输入泄漏）；
  feedback_match 兜底 prompt 含 fault_type 与判官同 ctx.llm（部署时需
  角色隔离）。
- io：悬挂 parent 产出 parentSpanId=""（违反 16hex）；langfuse 导入 meta
  残留旧 "outcome" 键；trace 级信息仅挂 index==0、空事件轨迹导入消失；
  _attrs 只读 stringValue。
- 上轮审计文档过时两处：阶段四报告 §5/§6 中 chief "context_events=4 窗口"
  与现码不符（现为全轨迹+偏离声明）；§3 claim_audit "取早者"措辞与代码
  注释不一致。

## 5. 测试充分性横向结论

- **数值级断言普遍缺位**：sbfl 六式、binary_search 区间收缩序列、rg_ug
  episode 效用均无"具体数值"断言（本轮 subagent 以手工验算补上了这层，
  全部通过——公式正确但无回归保护）。
- **空洞/恒真断言 2 处**：test_recover.py:53（`or True`）、
  test_audit_fixes.py:283-294（unfold 两条断言平凡真）。
- **运行期防泄漏扫描窄**：仅 dover+cf 有实跑全量 calls 扫描；judge few-shot
  GT 步号、EXTRA_FAULTS 词表、chief/tree/claim 运行期均无。
- **树形/边界 fixture 缺**：canonical_events 嵌套 span（parent 恒 None）、
  hierarchy_tree stage revisit、binary_search 否定措辞答案、loop_detect
  tool_oscillation 负例等。

## 6. 建议修复优先级

1. **P1-1** claim_audit include_success 泄漏：成功路径剥离 outcome note
   （或至少剥离 gold 短语）+ 补运行期泄漏测试覆盖 include_success。
2. **P1-2** chief：要么把 HCG 图/OTAR/Φ 真正喂进 localize/oracle prompt，
   要么 docstring 如实声明"图未入 prompt、HCG 仅提供 subtask 区间"，并修
   正 hcg.py "consumed by chief" 措辞与 README 机制表述。
3. **P2-1/2** ssf unfold_line：去锚定正则真正修复 + 重写空洞回归测试
   （断言表格原文替换了占位符）。
4. **P2 顺序处理**：judge_eval few-shot 步号改造（避开 GT 步号）并纳入
   反泄漏测试；taxonomy FM-2.6 补回后果子句；rule_pack 抑制行为补声明或
   回退；inducer 并列最高频改确定性 tie-break；counterfactual_replay
   fault_removed 加步敏感；dover proposer 输入切片或声明；io 导出对
   raw-span 轨迹先 flatten 或告警；targeted_rerun 恒真断言重写。
5. **P3** 按模块清单择机清理；sbfl/binary_search 补数值级回归测试。

## 7. 复审方法备注

22 路 subagent 均只读（唯一写操作为运行指定 pytest）；公式类模块
（sbfl/rg_ug/drift_detect/binary_search）均做了独立手工验算而非仅读码；
防泄漏结论均基于实跑 prompt 文本扫描而非静态常量。

---

## 8. 修复记录（2026-08-27 同日完成：8 路修复 subagent + 8 路独立验证 subagent）

每项修复由独立于修复者的验证 subagent 复核（验证者做撤修复推演/独立复算/
论文回查，确认回归测试"有牙齿"）。**验收：pytest 260 passed（216→260，
新增/重写 44 个测试用例）；`atap demo` 六故障 step/agent/MAST/恢复 6/6、
闭环 round1 failures=0——与修复前基线完全一致。**

| # | 问题 | 修复 | 回归测试 | 独立验证 |
|---|---|---|---|---|
| P1-1 | claim_audit include_success 泄 gold | 成功轨迹判官视图 `include_outcome=False`（伪判官 claim handler 不读 outcome 行，行为零变化）；docstring 如实声明 | test_claim_audit_include_success_no_gold_leak（撤修复必失败）+ test_stage4b_runtime_prompts_no_fault_leakage（6 故障×claim/tree/chief 运行期全量扫描，禁 8 故障名+GT 键） | VA ✅（含撤修复复演） |
| P1-2 | chief HCG 图零消费未声明 | 声明路线：chief.py 偏离清单新增(6)【适配:机制缺口】；hcg.py "consumed by chief" 改"仅 subtasks 区间被消费"；引用改 w/o G 45.60（论文 Table 1 核实：52.00=w/G）；README 算法表 chief 行如实措辞（验收数字未动） | —（声明性修复，grep 证实零消费者与声明一致） | VH ✅（45.60 论文核实无误） |
| P2-1 | ssf unfold 假修复 | render.py `FOLD_PLACEHOLDER_RE` 去 ^$ 锚定；另一调用点改 fullmatch 保原语义 | 重写 test_unfold_line_expands_rendered_line（断言折叠原文恢复/占位符消失/行头保留；旧锚定正则复演必失败）+ fullmatch 语义新测试 | VB ✅ |
| P2-2 | targeted_rerun 恒真断言 | 删 `or True`，经 rerun.meta["feedback_snippet"] 真实钉住"只增不换"+失败 note 消费 | 原 test 内重写 | VB ✅ |
| P2-3 | feedback_injection 测试缺口 | 成功即停四重断言（rounds==1/reruns==1/reflect 0 次）；泄漏扫描参数化 6 故障（中和 root_cause 判定为增强非削弱——否则 premature 首轮即恢复不触发 reflect） | 2 处新增/扩展 | VB ✅ |
| P2-4 | judge_eval few-shot 引 GT 步号 | 步号改 10/11/14（onset 全集 {1,3,5,8,9} 含 EXTRA_FAULTS 实测） | test_fewshot_step_numbers_do_not_collide_with_gt_onsets（动态算全集） | VC ✅ |
| P2-5 | taxonomy FM-2.6 等收敛不彻底 | FM-2.6 补回后果子句；FM-1.1/2.2/2.3/3.1/3.2 按论文附录 A 原词修正 | test_taxonomy_definitions_align_with_paper_appendix_a | VC ✅（逐条对论文核对） |
| P2-6 | rule_pack 静默抑制 | 按谓词面回退：loop 产物存在时搜索面 settle、re-read 面恒走 R5 FILE_READ 兜底；note 留 consumed/fallback 痕迹；全语料实测回退新增命中 0（e2e 数字不变） | test_no_progress_reread_surface_not_dropped_when_artifact_present | VD ✅（独立遍历 21 轨迹证实） |
| P2-7 | inducer 非确定性 | tie-break 改 sorted((-count,text))[0] | test_inducer_proposal_name_pinned_value（name 钉死，5 个 PYTHONHASHSEED 验证） | VE ✅（自跑多 seed 复核） |
| P2-8 | cf 重放步不敏感 | fault_removed = fault 存在 ∧ 干预步(clamp 后)==onset 步 ∧ 关键词命中；步未知 fail-closed；meta 留 intervention_on_onset_step；dover/cf 既有数字不变 | test_replay_fault_removal_is_step_sensitive | VE ✅（含 clamp 边界推演） |
| P2-9 | dover proposer 未切片 | _exec_range/_slice_view：proposer 只见失败 trial 区间+previous_trial_summary（首 trial 空表）；docstring 声明 | test_dover_proposer_input_sliced_per_trial（3-plan 合成） | VE ✅ |
| P2-10 | io 导出静默丢轨迹 | cli export 前 _ensure_flattened（注册表同源展平）+ stderr log + stdout 计数 | 2 个 raw-span-only 导出测试 | VF ✅（独立复演 26 spans/observations） |
| P2-11 | io roundtrip 断言缺口 | score/note 双侧相等、otel 侧 task/qrels/injected_fault 防泄漏、_refs_sig 按 id 比指向 | test_stage4d.py 内补 | VF ✅ |
| P2-12 | canonical 树拍平零覆盖 | 嵌套 fixture 测试（3 层+跨层 ref+悬空 ref）——顺带**发现并修复重复 span id 真 bug**（旧实现后写覆盖致 ref 错指幽灵事件，探针复现；现保留首个+duplicate_span_ids 计数） | test_canonical_flatten_nested_tree 等 | VG ✅ |
| P2-13 | sbfl 零数值断言 | 数值回归钉死 γ=0.5/β=1/3/n_cf^λ=1.9/Kul2^λ=0.8276/α=7.5788/**S=12.5442430**；三处 max(key=confidence) 改 hypotheses()[0]（稳定排序首位） | test_sbfl 数值测试（pytest.approx+算式自守恒） | VG ✅。**勘误**：修复验证确认 sbfl 代码公式本就正确；复审 subagent 报告中的参考值 25.0885 系 (1+β)(1+γ)=2 重复相乘的算术笔误（论文式(6) 各因子只乘一次），以 12.5442430 为准 |
| P2-14 | tree 声明不实 + revisit | attributed step clamp 到渲染区间 [lo, rng.end]（防作弊声明自此为真）；同名 stage 选含失败事件 span（结构化 error observation 派生失败 index）【推断】声明 | 既有六故障 step==GT 断言保持 | VH ✅ |
| P3 批 | 20+ 项 | canonical kind 别名+ts 节点优先；langfuse meta pop/otel 悬挂 parent 省略/_attrs 声明；binary 否定词解析/LLMError/s*=0 直读；dover redact 空格变体+forbidden 补 EXTRA；cf intervention_applied 如实+窗口/温度声明；mast_judge J.1 逐字引用（论文核实属实）+空定义防护+防泄漏 ALL_FAULTS；action_signature 锚词边界（零行为变化）；idg o_v、hcg E_step、PLAN/AGENT_SPAWN RECORDED、all_at_once 12.5 口径/outcome 声明/非法码留痕、claim_ledger b_k/U_k 钳制+送审+Prompt2 声明、loop_detect 推断声明+None 滤 | 各组新增测试 | 各验证 ✅ |
| 追加 | test_represent.py:222 恒真断言（VB 验证期新发现） | 钉真实行为：n_folded==0 且视图无占位符 | 原测试内重写 | 主线修复+全量绿 |

### 遗留已知残留（低风险/已声明，后续按需处理）

- io 库层直调（绕过 CLI）传 raw 轨迹仍会静默丢——CLI 层已防护，导出函数 docstring 提示；
- dover 切片保留 session 级 outcome 头行（多 trial 时 per-trial 成败为 session 级，T=1 无影响，未声明）；
- binary _parse_half 罕见口语（"no doubt"类）可误翻——prompt 已限定裸短语答案；
- canonical remapped_kinds 统计把纯大小写归一也计数（轻微高估）；
- taxonomy 14 条定义首句仍是同义改写非逐字（无语义扭曲，7/14 有关键词钉死）；
- test_stage4b.py:184-186 `if False else` 死代码（无害）；
- hcg.py:184 行内注释与 docstring 措辞微相左。

