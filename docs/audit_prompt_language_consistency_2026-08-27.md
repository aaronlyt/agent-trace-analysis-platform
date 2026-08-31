# Prompt 语言与论文一致性审计（8 路独立 subagent，2026-08-27）

> 范围：10 个持 prompt 模块的全部判官可见英文文本（22+ system prompt、
> few-shot、user 模板、渲染模板），逐组与 refs/ 论文原文比对 + 一路横向
> 语言/术语一致性检查。只读审计，未改任何文件。
> 判定：逐字 / 忠实改写（语义等价）/ 语义漂移 / 自拟（论文无原文）。

## 1. 总体结论

1. **全部 prompt 均为英文，无中文汉字渗入判官可见文本；无拼写错误，
   语法总体通顺**。唯一的"中文残留"是全角破折号"——"混入 4 处判官可见
   文本（见 §4-1），属排版层面非语言层面。
2. **与论文的一致性分两类**：论文提供 prompt 原文的四组（Who&When G.1/G.2、
   DRIFT 附录 G、DoVer 附录 B、CHIEF Fig.10/11/13）——代码均为**忠实改写/
   压缩转写而非逐字复用**，核心问句、枚举名、输出语义保持，主要偏离已在
   docstring 声明；论文无 prompt 原文的四组（MAST 判官、CodeTracer、
   TraceElephant、AgenTracer §5.3）——自拟，协议要素覆盖良好。
3. **仍存在不一致**：6 处中等级语义偏离（多为未声明）+ 一批低危措辞差异
   + 明显的跨模块术语不统一（同一概念五種说法并存）。无任何一处达到
   "语义反转/错误"程度。

## 2. 中等级问题（语义偏离，建议处理）

| # | 位置 | 论文原文 | 代码现状 | 声明 |
|---|---|---|---|---|
| M1 | binary_search.py:103（refine） | —（G.2 无 refine） | refine prompt 称 "You may consult the MAST failure mode codes" 但该调用从未注入 definitions 块（all_at_once.py:60 有 `{definitions}` 对照）——判官拿不到码表 | 未声明（自相矛盾） |
| M2 | claim_ledger.py:70-73 | Prompt 2 "Record it only if the span also says it has identified/found/verified the candidate…" | 该例外句整句删除，"query text and tool calls themselves do not count" 一刀切 | 未声明 |
| M3 | claim_audit.py:135 | "missing when the trajectory commits…but shows no support for a required decisive link" | "MISSING (no support shown at all)"——丢"已承诺"前提，MISSING 判定面更宽 | 未声明 |
| M4 | dover.py:158 | 四判定 "Partially Validated"（§4.2/Table 3） | 枚举名只写 "Partially" | 未声明 |
| M5 | dover.py:195-198 | — | classify system 自称输入含 "milestone progress"，user 消息实际只给里程碑列表无进度值（prompt 内部不一致）；另 Refuted 用 "no progress" vs 论文 "progress does not exceed 20%" | 未声明 |
| M6 | chief.py:143 | planning_error="planner repeats semantically identical thoughts/commands despite repeated error signals"（限于循环+失败反馈） | "the planning step itself deviates from the task requirements"——泛化为任意计划偏离 | 未声明 |
| M7 | chief.py:199,240-245 | Eq.4 用 P_pre/E_key 做代理级评估 | eval 只传 goal/acceptance、localize 不传任何 oracle 字段——合成的 preconditions/key_evidence **零消费者**（docstring 只声明了 HCG 图无消费者，未覆盖此点） | 部分（相邻声明未覆盖） |
| M8 | counterfactual_replay.py:149-153 | oracle="…anticipated output **if the specified mistake reason had not occurred**" | user 消息只给 step/agent/内容片段，候选 Hypothesis 的 root_cause（mistake reason）不传入——反事实条件被丢弃（影响低：expected 已声明不参与 verdict） | 未声明 |

## 3. 低危措辞差异（择要；论文有原文各组均有若干）

- Who&When：all_at_once 输出 schema 较论文 G.1 增 fix_suggestion/confidence/
  failure_mode 三字段；"(agent roster: …)"、"the full log has {n} steps" 为
  新增；"症状晚于根因"澄清句新增——均未声明但方向无害。
- DRIFT：排除列收窄（"pure query or tool-call line" vs 论文六项枚举）；
  "fake verification" vs 论文 "false verification"；支撑级大写 vs 附录小写。
- DoVer：分段输出自导出 plan_step+exec_range（论文仅步索引+理由）；
  [:120] 截断 vs 论文 "Use all content as-is without truncation"（3 处）；
  硬编码 "3 repeats" 而 n_repeats 可配；"Keep changes minimal and avoid
  global resets" vs 论文两句式。
- CHIEF：oracle 丢 Fig.10 第 5 点 global self-check 与 Eq.2 的 RAG τ_s
  输入；executor_loop 取第二次出现为自拟 tie-break；"progressive causal
  filtering" vs 论文 "screening"；system 称 "failing subtask interval" 而
  user 实给全轨迹（prompt 内部措辞矛盾，全轨迹本身已声明）。
- CodeTracer：neighbor 语义——论文 "parent, sibling, or immediate next
  step"（树邻居）vs 代码前一 stage 尾部 3 事件（时间前驱），注释却称
  "aligned with the paper"（注释失实）；cue 措辞无 "stalled progress"
  对应；无 phase 事件可渲染 "== stage: None ==" 行。
- AgenTracer：reflect 缺论文 App B "DO NOT provide the complete solution
  in the suggested_fix" 护栏。
- MAST：mast_judge user 模板含 outcome 行（已声明）；few-shot 只演示单标签
  而附录 N 示例多为多标签；加载 extra_modes 后 "14 failure modes" 失真。

## 4. 语言与横向一致性问题

### 4-1 语言错误（判官可见文本内）
1. **全角破折号"——"4 处**：judge_eval.py:82、mast_judge.py:88-89、
   mast_judge.py:159、taxonomy.py:111——taxonomy 的经 definitions block
   渗入 all_at_once / tree_diagnosis / mast_judge 三个模块的 prompt（已
   实渲染验证）。
2. 破折号三种风格并存："——"（上述）、"--"（dover.py:198、
   claim_ledger.py:71）、"—"（多数）。
3. Chinglish/生硬处：claim_ledger.py:97 "The task and trajectory follow;"、
   dover.py:175 "give the mistaken agent"、字段名 is_succeed 非惯用。
4. tree_diagnosis.py:128 "task:" vs :185 "Task:" 大小写不一致；few-shot
   拼接 judge_eval 单换行 vs 其余 "\n\n"。

### 4-2 跨模块术语不统一（同一概念多种说法）
- 最早致错步：earliest decisive error（aao/tree/chief）/ earliest
  failure-flipping error + most critical error（binary_search）/ earliest
  erroneous step（dover）/ earliest error span（claim_audit）——四模块五說法。
- 步索引：R0 index（aao）/ [index] at start of rendered lines（chief/
  tree）/ R0 event index（judge_eval）/ line-start [index]（dover）。
- 责任方：responsible agent（aao/chief/tree）/ mistaken agent（dover）/
  responsible party（feedback_injection）/ responsible span（claim_audit）。
- JSON 指令："Output JSON." 11 处有、其余模块无（靠 schema）。
- 枚举描述风格：元组 repr（claim/chief/ledger）vs 竖线列表（dover/
  judge_eval）。
- few-shot 格式：key=value 文本（aao）vs JSON 字面量（mast_judge/
  judge_eval）。
- 事件行字段序：render.py 统一 "[idx] KIND agent"，但六个模块自建候选/
  证据行 "[idx] agent KIND ::"——cf oracle prompt 同屏出现两种顺序。

## 5. 逐组判定汇总

| 组 | 论文 prompt 原文 | 判定 |
|---|---|---|
| Who&When（all_at_once/binary_search） | 有（G.1/G.2） | 忠实改写；区间逻辑/问句/步号基制吻合；M1 |
| MAST（judge_eval/mast_judge/taxonomy） | 无（附录 J 为相关性分析） | taxonomy 14 条全部同义改写（词面重合 44–89%），无语义反转；prompt 自拟要素齐全 |
| DRIFT（claim_ledger/claim_audit） | 有（附录 G Prompt 0/2/3/4/5） | 枚举/核心规则忠实；M2、M3 + 数处收窄 |
| CodeTracer（tree_diagnosis/hierarchy_tree） | 有但为终端 agent 式（已声明改造） | 自拟协议骨架与附录 D 一致；neighbor 注释失实等 4 处低危 |
| CHIEF（chief） | 有（Fig.10/11/13） | 自由转写，四元组/F_eval 忠实；M6、M7 |
| DoVer（dover） | 有（附录 B Fig.5-11） | 自拟压缩版协议等效；M4、M5 |
| TraceElephant+AgenTracer（cf/feedback） | 无 | 自拟要素覆盖良好；M8 + 缺护栏 |
| 横向语言/术语 | — | 无拼写错误；"——"残留 + 术语五說法并存 |

## 6. 建议（未实施，本轮只查不改）

1. P2：M1（refine 注入 definitions 或删该句）、M3（MISSING 补"已承诺"
   前提）、M6（planning_error 收窄回论文定义或声明泛化）、M4（枚举名
   对齐 "Partially Validated"）。
2. P2：清"——"残留（taxonomy 一处影响三个模块的 prompt）、统一破折号
   风格；binary_search/chief 的 prompt 内部自述与实际输入对齐（M1、M5、
   CHIEF "failing subtask interval"）。
3. P3：建跨模块 prompt 术语表（最早致错步/步索引/责任方/JSON 指令/
   枚举描述风格统一），M2/M7/M8 补 docstring 声明。
