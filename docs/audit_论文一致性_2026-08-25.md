# 论文一致性审计报告（12 路独立 subagent，2026-08-25）

> 范围：src/atap 下 12 个算法模块，逐个与 refs/ 论文原文独立比对。
> 判定口径：✅忠实 / 📝已声明适配 / ⚠️未声明偏离 / 🐛疑似bug。
> 所有 subagent 均为只读审计，未修改任何文件。

## 总览

| 阶段 | 模块 | 论文 | 总结论 |
|---|---|---|---|
| 表征 | canonical_events (R0) | AgentDebugX 2607.18754 | 基本一致，已声明适配为主；3 处未声明省略，无 bug |
| 表征 | ssf (R1) | TrajAudit 2605.26563 | 忠实，6 类适配声明全部属实；2 处轻微未声明 + 1 个未接线小 bug |
| 表征 | action_signature (R5) | TraceProbe 2607.06184 | 整体忠实；锚集声明措辞不准确 + 3 个小 bug |
| 分析 | judge_eval | MAST 2503.13657 | 存在未声明偏离：判官视图泄露 outcome（与 J.1 相悖）+ few-shot 性质未声明 |
| 分析 | loop_detect | TraceProbe 2607.06184 | search_loop/tool_oscillation 忠实；re_read_churn/redundant_search 的 10-action window 口径实质放宽（未声明） |
| 分类 | mast_judge + taxonomy | MAST 2503.13657 | 骨架忠实、上轮修复生效；6 处定义残留论文没有的扩写 + few-shot 步号错位 |
| 分类 | rule_pack (L0) | AgentDebugX 2607.18754 | 总体忠实；3 处低危瑕疵 |
| 归因 | all_at_once (L1) | Who&When 2505.00212 | 忠实，无未声明偏离、无 bug |
| 归因 | binary_search (L2) | Who&When 2505.00212 | 核心数值逻辑逐行忠实；1 个潜伏疑点 + 2 处低危未声明 |
| 归因 | sbfl (L0) | FAMAS 2509.13782 | 式(2)–(7) 逐式忠实，无实质偏离、无 bug |
| 恢复 | targeted_rerun | AgentDebug 2509.25370 | 总体忠实，4 处实质偏离全部已声明且理由成立；policy.py 2 个边界情形 |
| 恢复 | feedback_injection | AgenTracer 2509.03312 | 忠实，所有偏离已声明；仅效率瑕疵 |

**总体判断**：机制层复现质量高——公式/伪代码/轮数/阈值等硬数值全部与原文一致（SBFL 六式、二分的 floor-mid 与 ⌈log₂n⌉、SSF 词表与豁免、5 轮/3 轮恢复等）；声明纪律好，绝大多数偏离带【适配】【推断】标记且标记准确。需要处理的问题集中在四处：**loop_detect 的窗口口径（唯一实质的算法语义偏离）**、**judge_eval 的 outcome 泄露（唯一协议相悖）**、**taxonomy 定义残留扩写（上轮 FM-2.6 同类问题未清干净）**、以及 action_signature/mast_judge/ssf 的几个小 bug。

---

## 一、表征（represent/）

### 1. canonical_events vs AgentDebugX

机制语义（有序事件序列、step index、parent 树、agent、采集源无关、诊断不写回轨迹）忠实；plan.md 声明的"省略 error/duration/metadata/artifacts 四字段"属实且标记准确，但省略声明不完整：

- ⚠️ **`module` 字段无对应**：论文 §3.1 与附录 A 字段清单均含 module，schema.py:11-15 的偏离声明逐字列举了 12 个原字段却未列入 module。
- ⚠️ **`inputs`/`outputs` 合并为单一 `payload`**：论文为两个独立字段，声明中列举了却未声明合并。
- ⚠️ **`timestamp` 退化为序号副本**：`_flatten` 中 `ts=float(idx)`（canonical_events.py:61），与 schema.py "单调递增时间戳"自述不符；全库无 ts 消费者，实际影响为零。
- 轻微：EVENT_KINDS 词表缺论文提到的 memory operations 事件类（UI actions 依赖已省略的 artifacts，可归入已声明范围）。

🐛 无。dropped_refs 计数、refs 映射、span id 生成均核验正确。

### 2. ssf vs TrajAudit（Algorithm 1）

骨架忠实（补丁保留 + 失败词保留 + 其余折叠 + 可逆恢复）；docstring/plan 声明的六类适配点（strict 收窄为结构化前缀、loose=strict∪词边界、min_fold_len 豁免、占位符带摘要、fold_ratio 分母口径、94.6%/20.2% 引用）逐一与原文核实全部属实。

- ⚠️ 空观测跳过（ssf.py:83-84 `if not content: continue`）：论文 Alg.1 字面上空观测也会被折叠；实现跳过但仍计入分母。无实质影响。
- ⚠️ `extra_keywords` 扩展参数（ssf.py:66,85）：论文 K 为预定义词典；默认空、不改默认行为，但适配声明清单未提及。
- 🐛 **`unfold_line` 契约不符（未接线）**：docstring 称展开"渲染行"，但 `FOLD_PLACEHOLDER_RE` 锚定 `^⟦folded:`，而渲染行以 `[8]` 开头，正则必然不匹配、静默原样返回。当前无调用方，不影响流水线；一旦按 docstring 用法使用会得到错误的"未展开"结果。建议改非锚定 `search` 或修 docstring。

### 3. action_signature vs TraceProbe

九类动作、七个效果标签全集、效果判定条件、里程碑 M1–M5 骨架、单调 LCS 保守相容规则均与论文 §III-B/D/E 及 Table I/II 一一对应；"效果标签取全集 7 个"、"规范动作类不写回 TraceEvent.action"两个声明准确。

- ⚠️ **锚集声明措辞不准确（impl:12-14, 180-185）**：论文的 oracle-free 回退路线（paper.md:64）是"从该轨迹自身的存活写与测试/导入引用推导"；实现取"同任务成功轨迹的读集"——这是另一种（合理的）oracle-free 构造，但"按原文回退路线"的说法不准确。附带效应：锚集宽度=成功轨迹读过的全部文档，宽于 gold-patch 语义，压低 OFF-ANCHOR 召回。
- ⚠️ 无锚集时的默认标签未声明（impl:296-298）：成功 FILE_READ→JUSTIFIED、成功 SEARCH→RECORDED，二者不对称。
- ⚠️ SEARCH 锚命中为格式启发式（`[d`/` d]`/`, d`），仅行内注释声明，模块 docstring 未列。
- 🐛 **M3 步数取值偏晚（impl:309-315）**：取最后一次锚读 index，正确应为每锚首次读 index 的最大值（读序 A,B,A → 实现 M3=第3步，应为第2步）。仅 step 偏晚，reached 判定不受影响。
- 🐛 docstring M2/M3 映射顺序与实现相反（纯文档瑕疵）。
- 🐛 **`alignment.reference_trace` 填错值（impl:383）**：填的是被比轨迹自己的 task_id，而 anchor 字段中同名字段填 ref_bundle.trace_id（impl:206），语义不一致。

## 二、分析（analyze/）

### 4. judge_eval vs MAST 判官协议

声明范围内大部分要素忠实（few-shot 默认开启及 Table 2 依据、gold 隔离、成功+失败全量评测、检测≠归因边界、过程可见性）。两处未声明偏离：

- ⚠️ **判官视图始终泄露 `outcome: SUCCESS/FAILURE`（最实质项）**：论文 J.1（L557）明确"不向 LLM Annotator 提供成败结果"，判官结论才能与真实成败做相关性分析。实现 `judge_view` 恒含 outcome 行（render.py:106-112），judge_eval 无剥离开关、`only_failures` 默认 False → **默认每条轨迹都把成败标签喂给判官**。伪判官恰是仅凭 outcome 给 9.0/2.5 分——真实判官同样可能被锚定，使"过程质量分"退化为结果标签复读。对照组：mast_judge 已声明此相悖且默认只打失败轨迹使默认路径无影响；judge_eval 是默认全量受影响的一方却零声明。
- ⚠️ **few-shot 示例性质未声明**：论文 few-shot 来自人工标注数据（附录 N 13 例），Table 2 κ 0.58→0.77 由实质示例驱动；实现是 1 条自造输出格式演示。mast_judge 对同等做法已声明，judge_eval 缺同等声明。
- 🐛 轻微：few-shot 示例正文说 step 4/9、输出却定位 step 3，指称歧义；`Finding.severity` 为自由 str 无词表校验（score 有 ge/le 硬校验，severity 应有同等级 `Literal` 校验）。

### 5. loop_detect vs TraceProbe Table II

- ✅ search_loop、tool_oscillation 忠实（含"宽松连续"解读有原文依据且已声明）；阈值 10 确为论文 SWE-Bench Verified 冻结口径，"阈值需按目标基准审计"有原文直接支持（paper.md:137,194）。
- ⚠️ **re_read_churn 窗口口径（loop_detect.py:129-130）**：论文"within a 10-action window"指动作序列上连续 10 个动作；实现 `reads[i:i+10]` 是 **FILE_READ 子序列**上的 10 个读——同一文件 3 次读无论相隔多少其他动作都会触发，实质放宽。docstring"在 window 动作窗内"与实现自身口径矛盾。
- ⚠️ **redundant_search 同类问题（loop_detect.py:163-164）**：SEARCH 子序列窗口，同上。
- 📝 范围裁剪已声明：论文 Table II 单轨迹结构检测器共 8 个（另 2 个语义检测器），实现仅 4 个循环类谓词。

## 三、分类（classify/）

### 6. mast_judge + taxonomy vs MAST

层级（3 类 14 模式代码与英文名）、prompt 结构（轨迹+全部定义+few-shot）、逐标签 reason、上轮修复（FC3 改名 Task Verification、FM-2.6 删除扩写）均确认生效。协议层偏离（自造 few-shot、max_labels 截断、默认只打失败轨迹+outcome 行相悖）均已声明。残留问题：

- ⚠️ **定义残留扩写（与上轮 FM-2.6 同类，按 taxonomy.py:4-7 自我承诺"不扩写"应清理）**：
  - **FM-3.3**（taxonomy.py:96）：括号"假阳性通过、把未核实信息当作已验证依据"+"误导性结论"均为论文没有的内容——会向判官暗示特定判定样式，建议按 FM-2.6 方式收敛回原文；
  - **FM-2.4**（:70）：括号"（需求、约束、发现）"添加枚举；后果由论文"可能影响决策"改为"**导致**下游重复失败"（因果变强且换内容）；
  - FM-1.1（:29）括号枚举、FM-1.3（:39）"预算耗尽"（论文无，且恰与沙盒场景呼应，有引导判官之嫌）、FM-1.2（:34）丢失"表现得像另一个 agent"特有后果、FM-1.5（:49）增"或悬置"；
  - 系统性小项：论文 "potentially" 限定语普遍未保留。
- 🐛 **few-shot 步号与沙盒真实轨迹错位（mast_judge.py:45-47）**：示例称三次相同 search 在"step 5/6/7"、决定性错误 step 6；真实沙盒轨迹（policy.py:169-186）三次 TOOL_CALL 的 R0 索引为 3/5/7，ground truth onset=5。示例规则表述正确但编号与真实轨迹不符，判官模仿"连续编号"模式会产生 step 偏移。建议改为 3/5/7、决定步 5。
- 🐛 边缘：max_labels 截断后超出标签既不入 labels 也不入 invalid_codes，无记录。

### 7. rule_pack vs AgentDebugX

"原文只有一句话定义"声明属实（§3.2 L58，附录无触发条件）；四规则名称/数量一一对应；输出结构（症状定位+种子化归因+free 成本）符合 L0 定位；8/8 测试通过。低危瑕疵：

- ⚠️ docstring 说 premature-success 要求"成功的 FILE_READ"，实现不校验读取成败（收窄，声明与代码择一修正）；`read_actions={"read_doc"}` 硬编码沙盒动作词表未单独标注。
- ⚠️ `_malformed_tool_call` 第二分支把"错误观测"等价于"畸形调用"（环境侧错误也触发）——超出论文词面的轻微外延，通用组件使用时的潜在过触发源。
- 备忘：`d.get(...) or d.get(...)` onset=0 时静默回退（实际恒 ≥1 不触发）；loop_detect 四谓词只消费两个（已声明，玩具域无影响）。

## 四、归因（attribute/）

### 8. all_at_once vs Who&When §4.1 / G.1

**忠实，无未声明偏离、无 bug。** 单遍结构、query+全轨迹、以"最早决定性错误/根因非症状"引导（Eq.5 语义）、Without-GT 设定、54.33=With-GT 列口径声明、few-shot off-by-one 修复（第二次调用=首次重复，语义与序号均正确）、越界钳制透明留痕，全部核验通过。备注级：step 单位为 R0 事件 index（论文为发言条目序号，粒度约为其 2–3 倍），docstring 引用 step 级 12.5 时未提示口径不可直接对比（且该数字为 With-GT 子列）。

### 9. binary_search vs Who&When Algorithm 2

**核心数值逻辑逐行忠实**：区间初始化等价换基、floor 中点、只展示 [low,mid]、lower⇔error-in-L' 收缩、⌈log₂n⌉ 轮上界、s\*=low、A\* 从 s\* 事件读；refine 收尾、0 基 index、env 回退、prompt 措辞四项差异均已声明且属实。

- 🐛 **`_responsible_agent` 漏 AGENT_MESSAGE（binary_search.py:208，低危/潜伏）**：acting 集 {LLM_CALL,TOOL_CALL,HANDOFF} 不含 AGENT_MESSAGE；若 s\* 落在某 agent 的 AGENT_MESSAGE 而前最近 acting 事件属于别的 agent，A\* 会错归。当前沙盒不产生 AGENT_MESSAGE 事件故未暴露。
- ⚠️ 未实现 G.2 的 with-GT prompt 变体且差异清单未列（框架层有意设计，有防泄漏测试固化，属声明缺口）；口径混用：docstring 引 With-GT 数字 23.98/12.50，实现实为 Without-GT 口径（16.59/13.53）。
- ⚠️ 轮次 prompt 片段用 SSF 折叠视图，本文件差异清单未列。

### 10. sbfl vs FAMAS 式 (2)–(7)

**逐式忠实，无未声明实质偏离、无 bug**：γ/β/α/λ-decay/Kul2^λ 计数与合成公式、n_uf 不带衰减、λ=0.9 与 (0.5,1) 开区间校验、降序排序全部与原文一致；两处声明适配（LLM 聚类→R5 签名、排除 env/verifier）属实；docstring 引用指标（57.61/29.35）与论文 Table 2 完全一致。注意点：

- 签名第三元语义：论文 ⟨agent,action,**state**⟩（结果状态）vs 实现 (agent,action_class,**target**)（输入参数指纹）；REASON 类 target=None 坍缩为单一频谱单元使 f 偏大。被"聚类→签名"声明覆盖但 state≠target 未点明。
- 责任步取签名第二次出现（sbfl.py:195-201），注释称对齐 Eq.5"最早决定性错误"——但第二次出现晚于第一次，理由字面相悖（已标【工程选择】，不影响 S 计算与排序）。
- top_k=5 输出为 L0 先验接口扩展（首位即论文 top-1），已声明。

## 五、恢复（recover/）

### 11. targeted_rerun vs AgentDebug Algorithm 1 Stage 3

**总体忠实**：成功轨迹不调试、T\*=∅ 判 Failure、t\* 前缀保留+从 t\* 重跑、≤5 轮（论文 N=5 属实）、成功即停；四处实质偏离（t\* 置信度优先并列取最早步、每轮从原始轨迹重放、UpdateFeedback 弱化、反馈只消费 step+fix_suggestion）全部已声明且理由核实成立（如链式传 rerun 轨迹会因 meta 剥离 injected_fault 而虚假成功）。

- ⚠️ 轻度：UpdateFeedback 弱化版=纯字符串拼接且仅消费 outcome.note，第 2..5 轮失败说明完全相同、反馈只增不换——与论文"refined with more specific guidance"意图有距离，外层已声明"弱化版"但该细节粒度未声明。
- 重放环境 policy.py 两个边界情形（TR 本体无 bug）：
  1. step 越界时 `suffix_start` 兜底 0 → 全量事件接前缀后出现重复事件/序号错位（正常归因不触发，无越界防护）；
  2. 无注入故障的失败轨迹重跑恒成功（`fault=None` → 直接取干净重跑结果）——当前沙盒不可达，接入真实环境会变成系统性虚假恢复。

### 12. feedback_injection vs AgenTracer §5.3

**忠实，所有实质偏离与论文留白均已声明**：仅失败轨迹、反思输入无 GT 泄漏（judge_view 不渲染 meta/gold，rerun meta 已剥离 injected_fault——"τ (w/o G)"要求满足）、3 轮（论文 three rounds 属实）、完整重解（与前缀保留正交）、成功即停、第 1 轮反馈取归因 Hypothesis（📝）、生成者由微调 tracer 换通用判官（📝）、注入形式由 env.resolve 决定（📝论文留白）、沙盒"关键词优先 LLM 兜底"消费（📝论文无此细节）。实验数字引用（+4.8/+14.2/−4.9/−5.5%）准确。

- 🐛 仅效率瑕疵：末轮失败后仍调用一次 `_reflect` 生成永不会被注入的反馈（feedback_rounds 长度可达 max_rounds+1）。

---

## 建议修复优先级

1. **P1 loop_detect 窗口口径**：re_read_churn/redundant_search 改为动作序列 10-action window（或至少修正 docstring 并声明口径差异）——唯一实质的算法语义偏离。
2. **P1 judge_eval outcome 泄露**：judge_view 增加剥离 outcome 的选项（对齐 J.1），或至少在 docstring 声明该偏离——唯一协议相悖且默认路径受影响。
3. **P2 taxonomy 定义收敛**：FM-3.3/FM-2.4 等按 FM-2.6 先例回归附录 A 原文（定义直接进判官 prompt，扩写=引导判官）。
4. **P2 mast_judge few-shot 步号**：改为与沙盒真实轨迹一致（3/5/7、决定步 5）。
5. **P3 小 bug 清单**：action_signature M3 step/reference_trace 字段/锚集声明措辞；binary_search AGENT_MESSAGE；ssf unfold_line；judge_eval severity Literal 校验；sbfl 第二次出现的注释理由；canonical_events 省略声明补齐（module/inputs-outputs 合并/ts=index）。

---

## 修复记录（2026-08-25 同日完成，全部修复并验证）

| # | 问题 | 修复 | 回归测试 |
|---|---|---|---|
| P1-1 | loop_detect 窗口口径（读/搜子序列 ≠ 论文动作序列窗） | `_re_read_churn`/`_redundant_search` 重写为动作序列位置窗口（连续 window 个签名动作内的最大同目标 run）；写把同目标读分组（无中间写语义保持） | test_audit_fixes.py：跨度 5≤window 触发、跨度 7>window 不触发（两谓词各 2 例） |
| P1-2 | judge_eval 泄露 outcome（与 MAST J.1 相悖） | `render_trace/judge_view` 增加 `include_outcome`；judge_eval 默认剥离（`show_outcome=true` 恢复）；伪判官 `find_outcome`→`bool|None` + VERIFIER 行推断兜底；few-shot 自造演示声明 + 示例 step 指称修正（3/4/9 一致） | prompt 无 "outcome:" 断言；show_outcome 恢复断言；成功轨迹仍 ≥8 分 |
| P1-3 | judge_eval severity 无词表校验 | `Literal["minor","major","critical"]` + before-validator 同义词归一（high/low/严重等），未知值显式 LLMError | 同义词归一 2 例 + 拒绝 1 例 |
| P2-1 | taxonomy 定义残留扩写（FM-3.3/2.4/1.1/1.3/1.2/1.5/2.1 等） | 14 条定义逐一回归附录 A 语义：删括号扩写与强因果改写、恢复"可能"限定语、恢复 FM-1.2"表现得像另一个 agent"、FM-2.1"意外"、FM-1.3"任务完成中的错误" | 定义不含污染短语 + 含论文特有语义断言 |
| P2-2 | mast_judge few-shot 步号错位（5/6/7 连续编号真实轨迹不可能） | 示例改为 step 3/5/7、决定步 5（与沙盒 R0 索引、GT onset=5 一致）；all_at_once 示例同步 | 沙盒真实索引 3/5/7、GT=5 与示例一致断言 |
| P2-3 | mast_judge max_labels 截断观测缺口 | 先全量校验代码再截断（无效代码不挤占名额），超出有效标签记入 `truncated_codes` | 4 标签（1 无效）+max_labels=2 → labels=[FM-3.1,FM-1.3]、truncated=[FM-3.2] |
| P3-1 | action_signature M3 步数偏晚（取最后重复读） | 改为每锚**首次**读 index 的最大值 | 合成 d1,d2,d1 读序 → M3 step=3（非 5） |
| P3-2 | action_signature alignment.reference_trace 填被比 task_id | `_alignment` 改收 `ref_trace_id`，与 anchor.reference_trace 同口径 | step_repetition 的 alignment.reference_trace=="q-trajaudit--ok0" |
| P3-3 | 锚集声明措辞（"按原文回退路线"不准确）等 5 处 docstring 缺口 | 改为"oracle-free 新构造（非原文回退路线）"如实措辞；补无锚集默认标签、SEARCH 锚命中格式启发式、AGENT_MESSAGE 排除、M2/M3 映射顺序 | —（声明性修复） |
| P3-4 | binary_search A\* 漏 AGENT_MESSAGE | acting 集补 AGENT_MESSAGE（agent 侧行为直读）+ With-GT 口径/SSF 视图声明补齐 | AGENT_MESSAGE 事件 → 直读其 agent |
| P3-5 | ssf unfold_line 正则锚定行首必不匹配 | 改非锚定 `search` + 原位替换（保留行头） | 渲染行展开含原文且行头保留 |
| P3-6 | ssf 空观测跳过 / extra_keywords 未声明 | 补入 docstring 差异清单 | — |
| P3-7 | rule_pack premature-success 不校验读取成败 | 只认成功读取（错误/无观测的读取不算证据）+ read_actions 沙盒绑定声明 + onset=0 边界修正 | 读取失败后 submit → 规则触发 |
| P3-8 | canonical_events/schema 省略声明不完整 | 补 module 省略、inputs/outputs 合并 payload、ts=index 副本、memory/UI 事件类缺口声明 | — |
| P3-9 | sbfl 第二次出现理由与 Eq.5 表述相悖 | 注释改写：重复类故障的最早决定性错误=首次重复（第二次出现），首次执行本身合法 | — |
| P3-10 | feedback_injection 末轮无效反思调用 | 末轮失败后 break，不再生成永不会注入的反馈 | 3 轮全失败 → feedback_rounds 恰 3 条 |
| P3-11 | policy.py 两处边界（step 越界轨迹畸变 / 无故障失败恒恢复） | rerun_from step 越界钳制+`step_clamped` 留痕；`unexplained_failure` 防护（resolve 与 rerun_from） | —（当前沙盒不可达路径，防御性） |
| P3-12 | targeted_rerun UpdateFeedback 弱化细节未声明 | docstring 补"纯拼接、信息量可能不增"声明 | — |

**验证结果**：154 测试全绿（139 旧 + 15 新增 `tests/test_audit_fixes.py`）；
`atap demo` 六故障 step/agent/MAST/恢复 6/6、闭环 round1 failures=0；
v3 全栈 15/18·18/18·18/18·147 calls（二分 per-fault step 5/6，唯一 miss 仍为
已记录的 step_repetition 5→8 已知偏差）；SBFL 12/18·42 calls（4/6，miss 仍为
malformed/info_withholding 两例已解释边界）——**全部验收数字与修复前基线
一致，行为变化仅限审计修复点本身**。
