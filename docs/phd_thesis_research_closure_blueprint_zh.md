# 博士论文研究收口与写作蓝图

> 证据冻结日期：2026-08-14
> 论文核心主线：**Task-conditioned trajectory-level behavior drift evaluation for closed-loop planning policies**
> 研究阶段状态：`FROZEN_FOR_THESIS_WRITING`
> 联合模型决策：`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`
> 实验收口结论：`RESEARCH_EXPERIMENTS_CAN_BE_FROZEN_FOR_THESIS_WRITING`

本文档是论文写作阶段的当前权威蓝图。正文按科学问题组织，不沿用研发过程中的 Stage 编号叙事。Stage 编号只用于附录、复现索引和证据溯源。现有冻结结果不得触发换 seed、换 epoch、修改模型、调整已查看结果的门槛或在同一 blind 数据上返工。

## 一、一句话核心贡献

本文建立了一个面向闭环规划策略的、任务条件化的轨迹级行为漂移评估框架，明确区分同场景配对归因与异场景非配对发布监控，并通过 Waymo 表示学习与 official nuPlan 闭环实验同时给出正证据和能力边界：学习式表示可以显著增强发布级纵向行为漂移检出，但这种增强不等价于更强的同场景纵向归因能力，也没有证明邻车上下文具有独立增量价值。

英文一句话建议：

> We present a task-conditioned trajectory-level framework that separates same-scenario attribution from unpaired release monitoring, showing that learned representations can substantially improve release-level longitudinal drift detection while exposing clear limits in paired sensitivity and incremental interaction information.

## 二、论文要回答的科学问题与冻结结论

### 科学问题 1：什么是规划策略的轨迹级行为漂移？

本文将 behavior/style 限定为：

> **ego response conditioned on traffic and interaction context**，即策略在给定交通与交互条件下产生的轨迹响应分布，而不是单一速度、加速度或安全评分。

评估对象是策略 rollout 产生的轨迹及其 83D ego-neighbor 上下文序列，经表示映射得到 64D behavior embedding，再通过 Behavior Distribution Discrepancy（BDD，本项目实现为冻结统计协议下的 MMD²）进行分布差异检验。BDD 只表示行为表示分布发生变化，不表示安全性、优劣、法规合规性或用户接受度。

BDD/MMD² 的绝对值依赖表示尺度和 kernel bandwidth，不允许跨 representation 直接比较 raw MMD²。跨表示比较只使用各自 null 下的标准化统计量、预冻结检出率、FPR、任务覆盖率和门禁结果。

### 科学问题 2：配对归因与非配对发布监控是否是同一种能力？

不是。二者是不同 estimand。

- **Same-scenario paired attribution**：同一场景分别运行两个策略，控制场景暴露，估计“在相同场景条件下，策略变化是否引起行为分布变化”。它强调因果归因接近性、低剂量敏感性和任务内一致性。
- **Unpaired release monitoring**：两个软件版本分别在不同场景、不同日志集合上运行，通过 A/A 标定和 log-disjoint 重采样估计“两个版本的总体行为分布是否可区分”。它强调发布级总体漂移、场景异质性下的聚合可靠性和误报控制。

冻结证据表明，不应要求单一表示在这两个 estimand 上统一最优：

1. 在 310 个新 log/scenario-disjoint 的同场景确认对上，old64 能显著区分 assertive 与 conservative planner，overall plus-one p=`9.9999e-06`，五个 pre-treatment task 经 Holm 校正后均显著；但该证据受 lane-assignment fallback 与 embedding distance 相关的质量限制约束。
2. 在更窄的纯纵向 183-pair、四剂量任务中，ego13 通过 4/4 overall 和 12/12 task×dose；old64/A 为 4/4 和 7/12，B/C 只有 3/4 和 2/12。由此不能声称 B/C 恢复了完整的同场景纵向敏感性。
3. 在同一 800-pair pool、相同 n=400 的控制分析中，B/C 的 paired median Z 为 28.295/25.368，高于 old64 的 13.502。这证明历史 paired 较弱不是“配对统计天然不适合 B/C”，也不只是 183 与 400 的样本量差异，而是 treatment、任务范围、场景池和 estimand 的共同作用。

### 科学问题 3：新数据与训练目标是否改善了真实发布条件下的漂移监控？

是，这是论文最强的学习式表示正结果。

Dynamic Builder v2 在不扩大 Waymo source 的条件下恢复了逐帧 semantic slots、lead entry/exit、intermittent following、front identity switch 和状态转换；完整数据包含 51 个 TFRecord、24,872 个 scenario、168,700 个窗口，train/val/test 为 135,046/16,870/16,784，跨 split scenario 重叠为 0。train intermittent-following 从旧 builder 的 0 恢复到 63,415，证明原问题来自静态 slot 与整窗有效率过滤，而不是 Waymo 原始数据缺失。

在冻结的 context-balanced n=400 非配对发布监控中：

| Representation | A/A FPR | A/B detection | 双方向最小检出率 | detection−FPR | 结论 |
|---|---:|---:|---:|---:|---|
| old64 | 5.0% | 66.5% | 62.0% | 61.5 pp | 未通过冻结发布门禁 |
| A-3407 | 3.0% | 90.5% | 90.0% | 87.5 pp | 通过 |
| B-3407 | 5.0% | 100.0% | 100.0% | 95.0 pp | 通过 |
| C-3407 | 6.5% | 99.5% | 99.0% | 93.0 pp | 通过 |
| ego13 | 2.0% | 100.0% | 100.0% | 98.0 pp | 通过；诊断参考 |

B/C 的提升跨三个 seed 稳定。机制分解进一步表明，B/C 相对 old64 的标准化 signal 为 2.586×/2.643×，null noise 为 0.856×/0.927×；signal 对 log-Z 增益的贡献约为 85.9%/92.8%。因此接近 100% 的检出主要来自更强且方向更一致的 planner signal，而不是单纯降低 A/A/null variance。B/C 的 release shift 方向一致性为 0.925/0.927，高于 old64 的 0.815。

该结果支持“学习式表示可改善冻结 nuPlan treatment 下的发布级纵向漂移监控”，但不支持真实整车厂发布可靠性、通用阈值或安全有效性。

### 科学问题 4：新 64D 表示是否全面优于 old64？

没有。必须保留联合负结果。

Waymo Dynamic-v2 test 的 primary seed 3407 上，A/B/C longitudinal delta 分别为 `-0.0232/+0.0248/+0.0159`。三者均通过 following、lateral、behavior proxy 和 retrieval 的综合非劣性，但都没有通过冻结的 primary longitudinal 完整门禁。B-3409 虽通过全部 Waymo 门禁，但 primary seed 在盲测前已固定为 3407，不能事后换 seed。

纯纵向 paired benchmark 中，A 在 learned64 中最好，但只与 old64 同为 4/4 overall、7/12 task×dose；B/C 更弱。故不能声称新 64D 全面优于 old64，也不能以强 unpaired 结果覆盖 Waymo 与 paired 负结果。

冻结联合决策保持：

`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`

### 科学问题 5：C 是否证明了 interaction context 的独立增量价值？

没有证明。

prospective interaction confirmation 的 80/80 official pairs 全部成功。short-headway 相对 long-headway 的 median 差异为：mean speed `+0.289 m/s`、RMS accel `+0.150 m/s²`、front gap `-4.202 m`、finite THW `-2.670 s`；front-gap、finite-THW、closing acceleration response 和 following acceleration response 四项机制门禁均通过。这说明 benchmark 确实构造出了“ego 整体动力学变化较小、interaction response 明确不同”的处置。

但 C full-context 的 null-standardized Z=`28.955`，C neighbor-zero 的 Z=`36.807`；预冻结主端点 ΔZ=`-7.852`，log-cluster bootstrap 95% CI=`[-33.393, 29.219]`。CI 下界不大于 0，而且点估计方向也不支持 full-context 增量。因此只能写：

> C 和 C neighbor-zero 均能检出该 treatment，但现有确认实验没有证明 full-context 相对 neighbor-zero 存在增量 interaction information。

不能写成“interaction-aware C 已验证”，也不能把该结果解释成“邻车/context 永远无用”。它只否定了当前 C、当前训练协议和当前确认范围内的增量证据。

## 三、正式学术贡献

建议正文采用以下五条贡献，不将贡献写成“提出一个新 GRU 模型”。

### 贡献 1：问题定义与双估计目标

定义闭环规划策略的 trajectory-level behavior/style drift 问题，将行为风格表述为交通与交互条件下的 ego 轨迹响应，并形式化区分 same-scenario paired attribution 与 log/scenario-disjoint unpaired release monitoring 两类估计目标。

### 贡献 2：任务条件化行为表示与统计评估框架

构建包含 ego 与五类 semantic neighbor slots 的 83D 时序上下文、64D behavior representation、BDD/MMD 统计检验及 task-conditioned 分层协议，使总体行为漂移能够与 following、lane-change、stop-go、高动态和密集交互等 pre-treatment task 结果共同报告。

### 贡献 3：面向真实版本发布约束的非配对监控方法

提出 A/A-calibrated、log-disjoint、context-balanced 的 pseudo-release 监控流程，分别冻结每种 representation 的 null、阈值、样本量和双方向检出规则，从而在无法要求两个软件版本复现相同路试场景时控制误报并量化发布级检出可靠性。

### 贡献 4：跨数据集闭环实证与 estimand 分离机制

通过 Waymo 训练/测试和 official nuPlan 闭环 rollout，系统比较 controlled paired 与 unpaired release 条件；实证表明二者对 representation 的需求可以分离，并进一步证明 B/C 的发布级提升主要由 planner signal 增强与 shift direction coherence 提高驱动，而非 null variance 人为收缩。

### 贡献 5：正负证据共同定义表示能力边界

给出可复现的能力矩阵：Dynamic-v2 与新纵向训练目标显著增强 learned64 的发布级纵向漂移检出，但 ego13 仍是 controlled longitudinal sensitivity 最强参考，A/B/C 未通过联合 Waymo/paired 门禁，C 也未证明 context 的独立增量价值。由此论证 task-conditioned evaluation 比追求单一“万能 embedding”更符合实际版本评估需求。

## 四、主结果、负结果与 claim boundary

| 结论 | 论文角色 | 允许的表述 | 禁止外推 |
|---|---|---|---|
| 310 对同场景确认中 old64 overall 与五个任务层显著 | 主结果：paired attribution 可行性 | 轨迹级表示能在冻结任务族中检出 planner-conditioned behavior distribution difference | 不等于安全、优劣或因果质量校正后的 planner effect |
| n=400 中 old64/A/B/C 为 66.5%/90.5%/100%/99.5%，FPR 受控 | 核心主结果：release monitoring | 新数据与训练显著增强冻结 treatment 下的发布级纵向漂移检出 | 不等于真实 OEM 80% 发布可靠性或通用阈值 |
| B/C 增益的 85.9%/92.8% 来自 signal | 主结果：机制解释 | 提升主要来自 planner signal 增强和方向一致性 | 不应写成 null calibration 对结果毫无影响 |
| ego13 在 183-pair 四剂量 paired 中唯一通过完整门禁 | 诊断主结果 | 显式 ego 运动学对 controlled longitudinal 差异最敏感 | 不表示 context 无用，也不是最终 behavior style 模型 |
| A/B/C primary 均未通过 Waymo longitudinal 完整门禁 | 负结果/能力边界 | 新模型未获得跨指标联合资格 | 不能挑选 B-3409 替代预冻结 primary seed |
| B/C 在窄纵向 paired 中弱于 ego13，且未通过完整门禁 | 负结果/estimand 边界 | 强 unpaired 能力不保证低剂量、任务内 paired 敏感性 | 不能宣称 learned64 全面恢复纵向敏感性 |
| interaction benchmark 机制通过 | 主结果：benchmark validity | 已实现小 mean-speed/accel 差与明确 gap/THW/response 差异 | 不等于任何 representation 已学到 interaction |
| C full-context 相对 neighbor-zero 的 ΔZ CI 跨 0 | 确认性负结果 | 未证明当前 C 具有增量 interaction information | 不能写“C 已验证”，也不能写“context 永远无用” |
| 联合决策无 candidate 入选 | 最终模型负结果 | 无 A/B/C 满足全部预冻结论文主模型规则 | 不得事后改变联合规则或重新定义主模型 |

### 可作为论文核心 claim 的内容

1. paired attribution 与 unpaired release monitoring 是不同 estimand，所需 representation 能力可以分离。
2. task-conditioned trajectory-level BDD 能在 official closed-loop planner rollout 上检出冻结任务族内的行为分布变化。
3. Dynamic-v2 与新训练目标大幅且跨 seed 稳定地改善了 context-balanced release-level longitudinal drift detection。
4. 该改善主要由标准化 planner signal 增强与方向一致性提高驱动。
5. 完整的正负证据支持按任务和用途选择 representation，而不是假设存在一个统一最优 embedding。

### 不能作为论文 claim 的内容

1. C 已被验证为 interaction-aware representation。
2. 新 64D 在所有 paired、unpaired 和 Waymo 指标上全面优于 old64。
3. B 或 C 是 universal/final validated representation。
4. ego13 证明 neighbor/context 无用。
5. BDD 数值可跨 representation 直接比较，或存在一个可跨数据集复用的通用 BDD 阈值。
6. 当前公开数据实验已经证明真实整车厂软件发布的检出率、功能安全性或道路风险。

## 五、各 representation 在论文中的定位

| Representation | 定义与用途 | 论文定位 | 不能赋予的定位 |
|---|---|---|---|
| old64 | 历史 Stage5D learned 64D；83D context 输入 | 冻结历史 baseline；证明 paired planner behavior difference 可检出的既有表示 | 不称为最终最优模型，也不掩盖其 release detection 只有 66.5% |
| ego13 | 13D ego kinematic 手工表示 | controlled longitudinal sensitivity 的诊断参考上界；帮助识别 learned64 丢失的运动学信号 | 不是最终 style representation；不能用于证明 interaction/context 无用 |
| A | Dynamic-v2 数据 + legacy single-GRU/objective | 数据修复贡献的工程消融；release detection 明显提升 | Waymo longitudinal 下降，不能称为全面改善 |
| B | Dynamic-v2 + single-GRU + clean longitudinal supervision/ranking/sampling | 当前最简单、最强且稳定的 release-level learned engineering candidate；用于支持训练目标贡献 | 未通过联合 Waymo/paired 门禁，不是 universal/final validated model |
| C | 与 B 相同训练条件，dual-branch ego/context encoder | interaction architecture 假设的确认性候选；release monitoring 强 | 未证明相对 neighbor-zero 的 interaction 增量，不能称为 validated interaction-aware model |

论文不选择 A/B/C 中任何一个作为“最终全能主模型”。框架和评估方法是论文主角；不同 representation 是用于揭示任务适配性与能力边界的被评对象。

## 六、推荐论文题目

### 首选题目

**Task-Conditioned Trajectory-Level Behavior Drift Evaluation for Closed-Loop Planning Policies**

中文建议：

**面向闭环规划策略的任务条件化轨迹级行为漂移评估方法研究**

### 备选题目

1. **Paired Attribution and Unpaired Release Monitoring of Trajectory-Level Planner Behavior**
2. **Evaluating Behavior Drift in Closed-Loop Planning Policies Across Paired and Unpaired Scenarios**

不建议以“Interaction-aware GRU”或“New 64D Embedding”为标题，因为联合证据不支持把模型结构作为最终主要创新。

## 七、推荐整篇论文目录与证据安排

### 第 1 章 Introduction

回答为什么 ADE/FDE、碰撞率和规则指标不足以描述版本间 behavior/style drift；引出真实整车厂两个版本通常在不同地点、不同场景路试的现实约束；提出 paired attribution 与 unpaired release monitoring 两个研究问题。

使用内容：一句话贡献、应用背景、研究问题、五条贡献。避免在引言中出现 Stage 编号。

### 第 2 章 Related Work

建议分为：自动驾驶闭环评估、驾驶风格与行为建模、轨迹表示学习、两样本检验与分布漂移、软件版本/模型监控。明确本文不是 planner 优劣排序，也不是提出 SOTA GRU，而是研究任务条件化的行为漂移评估。

### 第 3 章 Problem Formulation

正式定义 trajectory window、traffic/context、policy、representation、task、paired estimand、unpaired estimand、BDD/MMD²、A/A FPR、A/B detection 和 claim boundary。明确 behavior style 的条件化定义与“BDD 不等于安全”的边界。

### 第 4 章 Method / Evaluation Framework

介绍从 official closed-loop rollout 到统一轨迹视图、83D ego-neighbor context、64D representation、task-conditioned BDD、paired permutation null、unpaired A/A calibration、log-disjoint pseudo-release 和 context balancing 的完整流程。

模型只作为 framework 中可替换的 representation module。A/B/C 结构差异放在本章末尾或消融章节，不把 dual-branch 写成已验证贡献。

### 第 5 章 Dataset & Experimental Protocol

介绍 Waymo Dynamic-v2 数据、逐帧 semantic slot、split 防泄漏、新纵向 supervision、nuPlan official closed-loop inventory、planner treatments、pre-treatment task、primary seed、blind authorization、multiple testing 和禁止跨 representation 比 raw MMD² 的规则。

正文只保留最终数据版本与冻结协议。旧 builder、blocked gate、smoke、resume 和历史修复过程移到附录。

### 第 6 章 Controlled Paired Evaluation

分两部分：

1. 310-pair 锁定确认：报告 old64 overall primary 与五个 task strata，说明 task-conditioned paired attribution 的可行性，并披露 lane fallback 质量限制。
2. 183-pair 纯纵向四剂量：统一比较 old64/A/B/C/ego13 的 dose-response、overall 与 task×dose 覆盖，突出 ego13 最强和 B/C 未通过完整门禁。

章节结论不是“paired 失败”，而是：paired 结果依赖 treatment/task，显式 ego kinematics 对窄纵向低剂量更敏感。

### 第 7 章 Unpaired Release Monitoring

先说明 A/A calibration、489 logs、800 pairs、2400 frozen splits、n=200/250/300/400 和双方向评估，再报告 n=400 主结果及 seed stability。把 old64 66.5% 到 A/B/C 90.5%/100%/99.5% 作为全篇最核心工程结果，同时显示 FPR，禁止只画 detection 不画误报。

### 第 8 章 Representation Ablation & Mechanism Analysis

解释 Dynamic-v2 数据、纵向目标和 encoder topology 的可归因关系；用同池 n=400 paired/unpaired 对照排除样本量和 pairing 本身；报告 displacement、shift direction coherence、signal/noise 和 log heterogeneity 分解。章节结论固定为：B/C release 增益主要来自 signal，而非 null variance 下降。

### 第 9 章 Interaction Confirmation

先报告 trajectory mechanism gate，证明 short/long treatment 在 mean speed/accel 较小变化下产生 front gap、finite THW 和 response 差异；再报告 old64/A/B/C/ego13/C-neighbor-zero。主端点只使用 C full-context 减 C neighbor-zero 的 null-standardized ΔZ 与 log-cluster bootstrap CI。明确写出未证明 interaction 增量。

失败的旧 confirmation roster 不进入正文结果，只在本章“prospective validity control”用一段话说明：旧 roster 因 official runnability omission 失败，随后只修复可运行性规则并独立冻结新 roster；完整细节移附录。

### 第 10 章 Discussion

围绕 estimand 而不是模型胜负展开：

1. paired 与 unpaired 为什么不是同一能力；
2. ego13 为什么在 controlled longitudinal 中更强；
3. B/C 为什么在 release monitoring 接近 100%；
4. C 为什么没有表现出 context 增量；
5. 为什么 task-conditioned evaluation 优于单一万能 embedding；
6. 对整车厂异地路试版本监控的可用性与部署前提。

### 第 11 章 Limitations

集中陈述 public dataset、finite scenario/log pool、重复 pseudo-release、非独立真实发布、nuPlan planner treatment、lane fallback、interaction 场景范围、C neighbor-zero 结果、缺少 OEM 真值、BDD 非安全指标和外部可推广性。

### 第 12 章 Conclusion

回到双 estimand 与 task-conditioned evaluation。结论应强调：本文给出一个可审计的评估框架和能力边界；获得了强 release-level 正结果，但没有选出满足全部联合门禁的 universal representation。这是可信的研究结论，不是需要隐藏的失败。

## 八、正文核心图表规划

正文建议控制为 **7 幅图 + 3 张表，共 10 个核心图表**。

### 图 1：整体框架总图

内容：Waymo human trajectories → Dynamic-v2 context builder → representation learning → official nuPlan closed-loop rollout → paired/unpaired evaluation → task-conditioned report。突出 representation 是可替换模块，输出不是 planner ranking。

### 图 2：paired 与 unpaired estimand 概念图

左侧为同一 scenario 的 A/B paired label swap；右侧为不同 logs/scenarios 的 release A/B、各自 A/A calibration。图中明确“same-scenario attribution”与“release-level detectability”回答不同问题。

### 图 3：83D context 与 representation 对照图

展示 ego + front/left-front/left-rear/right-front/right-rear 的时序输入，以及 old64、ego13、A、B、C、C-neighbor-zero 的关系。C 的 dual-branch 只标为 candidate hypothesis，不用成功色突出。

### 表 1：数据与冻结评估协议

汇总 Waymo Dynamic-v2 样本规模、split、防泄漏、nuPlan paired/unpaired pool、任务定义、样本量、null、primary endpoint、multiplicity 和 blind/freeze 规则。

### 图 4：Controlled paired evidence

双面板：A 为 310-pair overall + 五个 task 的 null-standardized 结果；B 为 183-pair 25/50/75/100% dose 下 old64/A/B/C/ego13 的 Z_BDD 或通过单元格热图。图注披露 lane fallback 质量限制。

### 图 5：n=400 unpaired detection 与 FPR

每个 representation 同时显示 A/B detection、双方向最小值和 A/A FPR；标注冻结门槛。主视觉突出 old64 66.5% 与 A/B/C 90.5%/100%/99.5%，但不能省略 ego13 和误报。

### 图 6：paired/unpaired 分离的 signal-noise 机制图

建议三面板：同池 paired Z、release direction coherence、标准化 signal 与 null noise 的 log-Z 增益分解。直接回答 B/C 提升由什么驱动。

### 表 2：Waymo test 与跨域非劣性

列出 primary seed 的 longitudinal delta、CI、following/lateral/behavior/retrieval 非劣性和完整门禁；B-3409 仅放脚注作为 seed stability，不替换 primary。

### 图 7：Interaction confirmation

左侧画 Δmean speed/accel 与 Δfront gap/finite THW/response，证明 mechanism gate；右侧画 C full-context−neighbor-zero 的 ΔZ 与 log-cluster bootstrap 95% CI，清楚显示 CI 跨 0。

### 表 3：最终能力矩阵与 claim boundary

行是 old64/ego13/A/B/C；列是 Waymo noninferiority、controlled paired、unpaired release、interaction increment、最终资格。最后一行写明 `NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。

## 九、Discussion 的正式论证逻辑

### 9.1 为什么 paired 与 unpaired 不是同一能力

paired estimand 通过同场景对齐消除 scenario composition，适合归因和低剂量任务内敏感性；unpaired estimand 面对不同场景暴露，依赖总体 shift 是否能跨日志聚合并超过 A/A variability。表示可能对局部运动学差异敏感，却不一定产生跨场景一致的分布方向；也可能具有稳定的总体 release shift，却在窄任务、低剂量 paired 条件下不够敏感。因此二者互补而非互相替代。

### 9.2 为什么 ego13 在 controlled longitudinal 任务更强

ego13 直接保留 speed、accel、jerk 等与纵向 treatment 紧密相关的显式运动学量，不需要 64D 压缩在多任务目标之间分配容量。窄纵向处置的主要差异正落在这些坐标上，因此其信噪比最高。该结果说明当前 learned64 存在 longitudinal information retention trade-off，不说明 interaction context 在行为定义中无用。

### 9.3 为什么 B/C 在 release monitoring 中接近 100%

Dynamic-v2 恢复了旧 builder 丢失的动态交互窗口，新纵向目标和 sampling 强化了与 planner treatment 一致的方向。B/C 的 signal 是 old64 的约 2.6 倍，shift direction coherence 更高；在 400 场景聚合中，局部 scenario/log heterogeneity 被部分平均，而一致 planner shift 被保留。null noise 仅小幅下降，且 raw-marginal 口径下并未下降，因此结果不是由“更容易的 null”制造。

### 9.4 为什么 C 没有证明 context 增量

可能解释包括：ego 分支已经吸收 treatment 的主要可见后果；现有 interaction supervision 未迫使 context branch 学到条件响应的独立信息；fusion 可能冗余或稀释 context；80-pair benchmark 虽有明确机制差异，但其可辨识结果仍可由 ego trajectory 单独解释。论文只能把这些写为解释性假设，不能在当前 blind confirmation 上选择其中一个因果解释。

### 9.5 为什么 task-conditioned evaluation 更合理

不同版本验证问题需要不同证据：低剂量纵向归因、跨场景发布监控、interaction-conditioned response、横向动态或舒适性不必共享同一最优表示。将 representation、task 和 estimand 联合报告，可以避免 raw BDD 大小崇拜，也避免把某一 benchmark 的成功误写成普适能力。

### 9.6 面向整车厂的现实意义

在两个软件版本无法复现相同道路场景时，A/A-calibrated、log-disjoint、context-balanced 的 unpaired 流程是更接近实际发布监控的原型。部署到公司数据前仍需要以历史同版本路试建立 A/A null、冻结 ODD/task composition、按 log/route/region 聚类重采样，并以独立版本发布进行前瞻确认。当前 100% 不能直接视为真实公司检出率保证。

## 十、Limitations 必须完整保留

1. **公开数据与领域差异**：表示使用 Waymo 训练，评估使用 nuPlan/PDM 闭环，不能自动推广到公司传感器、ODD、planner 和驾驶员分布。
2. **有限库存**：release split 来自冻结的有限 800-pair/489-log pool，2400 次 pseudo-release 是重采样试验，不是 2400 次独立真实软件发布。
3. **场景组成依赖**：检出率依赖 treatment 强度、task mix、样本量和 context balancing；不能形成跨任务通用 BDD 阈值。
4. **planner treatment 人工性**：assertive/conservative 和 short/long headway 是受控参数处置，不代表真实版本差异的全部形态。
5. **paired 质量限制**：310-pair 结果中 lane fallback 与 embedding distance 相关，属于 post-treatment 描述性关联，不能作因果质量调整。
6. **interaction 范围**：interaction confirmation 仅覆盖冻结的纵向 following/closing treatment，且集中在 11 个 logs；不能推广到所有交互、横向博弈或城市复杂行为。
7. **context 增量未证实**：C full-context 没有显著优于 neighbor-zero；当前 interaction-aware 假设仍是负结果。
8. **模型资格失败**：A/B/C 都没有同时通过 Waymo、paired、unpaired 和 interaction 联合规则；没有最终 universal learned representation。
9. **统计含义边界**：BDD 检出 distribution shift，不评价安全、舒适性优劣、责任、风险或法规合规。
10. **缺少真实 OEM 前瞻数据**：当前方法适合作为工程候选与公开数据证据，尚未完成真实整车厂版本发布的独立前瞻验证。

## 十一、仅放附录的研发内容

正文不按 Stage 流水账展开。以下内容保留用于复现、审计和答辩追问：

| 附录 | 内容 | 对应历史材料 |
|---|---|---|
| Appendix A | 83D schema、五 semantic slots、mask、track-id、derivative reset、global33 标准化 | Stage 5D、6R、6O-v2 |
| Appendix B | Dynamic Builder v2 pilot、旧 builder 结构性过滤、intermittent=0 根因、人工语义审查 | Stage 6Q/6R |
| Appendix C | A/B/C 统一 trainer、公平随机流、训练预算、checkpoint SHA、resume ledger | Stage 6T/6U |
| Appendix D | BDD/MMD²、kernel bandwidth、paired permutation、A/A calibration、Holm、bootstrap 公式 | Stage 7 M6.1–M6.6、Stage 6P |
| Appendix E | paired 全 task×dose 表、quality tiers、lane fallback 敏感性、310-pair provenance | Stage 6J/K、Stage 7 M6.5/M6.6 |
| Appendix F | n=200/250/300/400 全 operating curves、双方向结果、三个 seed 稳定性 | Stage 6P/6V |
| Appendix G | same-pool paired/unpaired 几何、log/scenario heterogeneity、signal/noise 完整分解 | Stage 6W-A |
| Appendix H | interaction roster、THW sentinel 处理、mechanism gates、cluster bootstrap | Stage 6S-v2/v3 |
| Appendix I | v2 roster runnability omission、61/80 complete 的失败审计与 prospective repair | Stage 6S-v2 execution freeze |
| Appendix J | smoke、blocked/superseded 输出、失败路径和 no-post-hoc 规则 | Stage 6D–6G、早期 6S、其他开发记录 |
| Appendix K | 全部 manifest、SHA256、环境、依赖、命令和 reproducibility index | QUICK_REFERENCE 与各 freeze manifest |

Stage6S-v2 的执行失败不能删除，也不能用 61 个成功子集包装为主结果；它应作为 prospective benchmark runnability 设计教训放在附录。早期 smoke、阈值 probe、superseded builder 和下载/环境迁移过程不进入论文正文。

## 十二、当前 evidence gaps 与是否需要补实验

### 不影响当前核心论文收口的 evidence gaps

1. 尚无真实整车厂 A/A 历史发布和独立 A/B 发布数据；因此真实 OEM 可靠性属于外部验证空缺。
2. 尚未证明 C 或任何 learned64 具有独立 interaction-context 增量；因此 interaction-aware 主模型主张必须删除，而不是补写。
3. 尚未获得同时通过 Waymo、paired、unpaired 和 interaction 联合门禁的 universal representation；论文应把它作为能力边界。
4. interaction confirmation 的 log 数量有限，不能覆盖横向博弈、cut-in、merge 等更广交互类型。

### 是否存在必须补做的实验

**没有。** 当前证据足以支撑经过收窄后的核心主张：提出并验证 task-conditioned trajectory-level behavior drift evaluation，区分 paired attribution 与 unpaired release monitoring，证明 learned representation 的强 release-level 改善，并通过预冻结负结果界定其 paired、Waymo 和 interaction 边界。

只有在论文必须坚持以下更强主张时，才需要另立新研究阶段，而不是修改当前冻结实验：

- “C 是已验证的 interaction-aware 主模型”；
- “新 64D 是跨任务统一最优表示”；
- “该方法已达到真实整车厂发布可靠性要求”。

这些都不是当前论文完成所必需的 claim。若未来开展，应使用新的训练前协议、全新未使用 confirmation 数据和真实 OEM 前瞻发布数据，不能复用当前 blind 结果调参后再次作为 confirmation。

## 十三、论文写作冻结决策

最终研究叙事固定为：

> **强 release-level 正结果 + estimand 分离机制 + paired/Waymo/interaction 增量负结果 + 明确的 task-conditioned claim boundary。**

论文的主要成果是评估问题定义、双 estimand 方法、task-conditioned 统计框架、发布监控协议和可审计的能力边界，而不是某个 A/B/C 模型胜出。

当前无需继续训练 v3、扩展 Stage6S 库存、修改 A/B/C、调整门槛或新增 post-hoc 主指标。

`RESEARCH_EXPERIMENTS_CAN_BE_FROZEN_FOR_THESIS_WRITING`

## 十四、冻结证据索引

以下 SHA256 绑定本蓝图使用的核心机器可审计证据；正文写作只能在这些冻结结果的 claim boundary 内压缩和重述，不得改变结论方向。

| 证据 | SHA256 |
|---|---|
| 一次性 blind evaluation final manifest | `9aa7c10d50f30b1cb6798e9e7ffc8fe52004c7897291a1cc5134c829ec237e5b` |
| paired/unpaired mechanism + interaction final manifest | `462f1fb6aecbd8c2cc3a4ccf345bb16d8c95a1d47a0f1241b3ee96e6db7b1062` |
| 310-pair confirmation evidence summary | `7c21a3b36da8670aed1f29a4af1d4a91cf6b55c2e4fc5713915445a948b2bb2c` |
| Waymo Dynamic-v2 test result manifest | `6c5bd844974f6a6333a293ab761e23528cbaff06310b89c85bd95f09692c06af` |
| pure-longitudinal paired result manifest | `242f88517bdca31359c163244d26f7f556376dec2065ffc5cde859cec4f3d42b` |
| unpaired release result manifest | `70205230c9a3f10db01e0604dbd6ac75765a4fdc004a086b6f14a5bf46e455fa` |
| interaction representation result manifest | `74b601795decb2f6de90928d527429f27c1320a705d9a4f672400ce4d32412b9` |
