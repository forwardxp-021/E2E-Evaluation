# RBR-64 博士研究总体方案 v2.1（模型发现 → 行为解释 → 人类确认）

> 项目：E2E-Evaluation / 博士论文  
> 文档状态：`RBR64_STAGE_R_ROADMAP_V2_1_OWNER_SYNTHESIS_DRAFT`  
> 更新时间：2026-09-06（Asia/Shanghai）  
> Active branch：`20260825_stageR_new`  
> 版本关系：v2.1 在 v2.0 基础上吸收第三轮 Astra 独立审查与 Scientific Owner 最新判断。  
> 本文**不回写历史冻结协议**，不授权新的 simulation、roster selection 或 RBR training。

---

# 0. 本版相对 v2.0 的关键更新

v2.0 仍然受到上一轮 Astra “增量预测价值优先”的影响，将：

```text
H vs H+RBR incremental predictive validity
```

放在未来 RBR qualification 的主 gate 前面。

第三轮 Astra 审查与 Scientific Owner 复核后，正式调整为：

```text
PRIMARY = Behavior Drift Detection utility

SECONDARY =
H vs H+RBR incremental predictive-value diagnostic
+
semantic probe
+
shortcut / geometry diagnosis
```

因此本论文不再把“证明 64D 比 handcrafted features 信息更多”作为核心问题。

最终核心变成：

> **learned trajectory representation 能否成为一个模型化的行为发现层，在当前常规 KPI 尚未定义某类问题时，首先发现软件版本行为发生了值得关注的变化；随后再通过行为特征分析和人类试驾完成解释、确认和工程固化。**

---

# 1. 与两个 v1.0 文档的关系

## 1.1 R0 Protocol v1.0

保持：

```text
HISTORICAL_FROZEN_PROTOCOL
DO_NOT_OVERWRITE
```

其 D0–D5、资产防火墙、hypothesis-level 状态、anti-selection、representation/readout/BDD 分离等原则继续有效。

## 1.2 RBR Representation-V2 总体方案 v1.0

保持：

```text
HISTORICAL_SCIENTIFIC_DIRECTION
```

其中继续保留：

- learned trajectory representation 不应被训练成 handcrafted geometry 的复制品；
- handcrafted semantics 是 anchor / interpretation tool，不是 latent geometry 的唯一真值；
- representation、measurement readout 与 BDD statistic 必须分离；
- final confirmation 必须与 development evidence 隔离。

但其“至少三个 residual family”“RBR-A/B/C 连续推进”“下一步仍是 R0”等内容已被真实 Stage R 结果取代。

---

# 2. 博士论文真正要解决的现实问题

现实自动驾驶软件版本评价通常有大量人工 KPI，例如：

- 平均速度；
- acceleration；
- jerk；
- yaw rate；
- curvature；
- lateral acceleration；
- THW；
- gap；
- lane-change duration；
- safety / comfort / efficiency 指标。

这些 KPI 对**已知问题**非常有效。

但它们有一个天然限制：

> 只有当研发团队已经知道“应该测什么”，才会去定义对应 handcrafted feature。

现实中常见：

```text
所有已知 KPI 都正常
        ↓
离线与仿真看起来可接受
        ↓
实际试驾才发现一种过去没有显式 KPI 的行为变化
        ↓
例如：
新版本每次变道都犹豫、试探、回撤、迟迟不 commit
        ↓
版本被拒绝释放
        ↓
团队事后新增“犹豫变道”相关 handcrafted feature
```

这说明传统 KPI 体系更擅长：

> **closed-set monitoring：监控已经被定义的问题。**

而本论文真正要解决的是：

> **open-ended behavior drift discovery：在问题尚未被命名、尚未有 KPI 时，先发现“行为发生了值得关注的变化”。**

因此本论文不以：

> 用无数 handcrafted features 穷举所有驾驶行为

作为核心内容。

---

# 3. 最终工程与科研工作流

本论文正式采用三层闭环：

```text
Layer 1 — Model-based Discovery
RBR / learned trajectory representation + BDD
        ↓
首先发现：
“这个软件版本的行为和参考版本不一样”

Layer 2 — Behavioral Diagnosis / Explanation
已有 KPI + feature attribution + 新机制指标
        ↓
定位：
“差异主要集中在哪些任务、阶段、行为形态？”

Layer 3 — Human Semantic Confirmation
针对性试驾 / 专家复核
        ↓
确认：
“这是可接受的风格差异，还是影响发布的真实体验问题？”
```

若该问题被确认具有持续业务价值：

```text
confirmed novel behavior issue
        ↓
新增明确 handcrafted KPI / rule
        ↓
进入下一代常规版本监控体系
```

因此三者职责不同：

```text
Learned representation / BDD
= discovery layer

Handcrafted features
= diagnosis + interpretation + operationalization layer

Human driving / expert review
= final semantic and release-decision confirmation layer
```

不是谁替代谁，而是形成闭环。

---

# 4. 论文核心科学问题

最终推荐：

> **如何利用 learned trajectory representation 构建一个 closed-loop planning-policy behavior drift discovery layer，使其能够在预定义常规 handcrafted KPI 尚未显式描述某类行为变化时，发现软件版本之间的行为分布漂移，并支持后续行为特征分析与针对性人类试驾完成定位、解释和确认？**

英文：

> **How can a learned trajectory representation serve as a behavior-drift discovery layer for closed-loop planning policies, detecting release-level behavioral changes that are not explicitly represented by the current routine handcrafted KPI set, and supporting subsequent behavioral diagnosis and targeted human confirmation?**

该问题拆成四层：

1. **Realization**：planner / generator 意图是否真的在 closed loop 中实现？
2. **Discovery**：learned representation + BDD 是否能发现版本行为变化？
3. **Diagnosis**：变化可以被哪些已知或新增行为指标解释？
4. **Confirmation**：该变化是否在人类体验与发布决策中具有实际意义？

---

# 5. “Residual” 的最终定义

本论文不使用：

> residual = 人类永远无法手工描述的信息。

这是不可维护的强主张。

正式定义：

> **Residual-to-F0 behavior：在指定适用域、测量窗口与预定义常规 handcrafted summary set F0 的监控合同之外，仍存在的、可通过独立行为机制或后续专家复核确认的闭环行为结构差异。**

其中：

```text
F0 =
CURRENT_ROUTINE_HANDCRAFTED_MONITORING_SET
```

Residual 是相对于：

> **当前已经部署 / 预定义的实际监控体系**

而不是相对于：

> 所有未来可能发明的 handcrafted feature。

因此：

```text
RBR detects a new behavior pattern
        ↓
researchers later define a new handcrafted feature
```

并不会推翻模型发现价值。

恰恰相反，这构成：

> **模型发现 → 人类理解 → 指标固化**

的成功闭环。

---

# 6. 为什么不能用无限 handcrafted feature 作为核心路线

理论上可以不断增加：

- hesitation count；
- retreat depth；
- recommit delay；
- brake-phase count；
- gap-probing count；
- oscillation count；
- 任何未来想到的规则。

但这会形成：

```text
发现一个新模式
→ 写一个规则
→ 规则能检测
→ 再寻找下一个规则外模式
```

这仍然是 closed-set monitoring。

本论文要解决的是：

> **在问题尚未被定义之前，用模型先发现“这里有异常行为差异”。**

因此：

```text
handcrafted feature expansion
!=
primary scientific solution
```

它应该是模型发现后的：

```text
interpretation / operationalization step
```

---

# 7. RBR 的价值不在“64 > 13”

本论文不声称：

```text
64 dimensions > 13 dimensions
therefore more useful information
```

维数更高不保证：

- 表征没有 collapse；
- 没有 noise；
- 没有 shortcut；
- 保留了目标行为；
- BDD geometry 更有效。

RBR真正需要证明的是：

> **学习得到的 trajectory representation 是否把对版本行为漂移有用的结构组织到了一个更适合检测的空间中。**

因此即使：

```text
H contains the same underlying discriminative information
```

RBR 仍可能有价值，如果：

```text
BDD_RBR
```

在相同：

- FPR；
- sample budget；
- operating condition；
- data firewall；

下明显优于：

```text
BDD_H
```

这支持：

> learned representation 对行为漂移测量具有实际效用。

---

# 8. PRIMARY 与 SECONDARY 的最终层级

## 8.1 PRIMARY — Behavior Drift Detection Utility

未来 Level 2 的核心问题：

> **在相同误报约束与样本预算下，RBR-BDD 是否比预注册 handcrafted challenger BDD 更敏感、更稳定或样本效率更高？**

不比较 raw MMD²。

推荐比较：

- detection rate at fixed FPR；
- sample efficiency；
- detection gain with CI；
- null-calibrated sensitivity；
- bidirectional consistency；
- release-emulation performance。

形式上可定义：

\[
\Delta_{BDD}(m,\pi)
=
P(detect | RBR,m,\pi)
-
P(detect | H,m,\pi)
\]

其中：

- \(m\)：预冻结 batch size；
- \(\pi\)：预冻结 drift proportion；
- FPR contract 相同。

## 8.2 SECONDARY — Incremental Predictive Value

保留：

```text
H
vs
H + RBR
```

作为解释性诊断。

可能出现：

### A
```text
H+RBR better
BDD_RBR better
```

支持：

> 指定 readout 下有增量预测价值，并转化为 BDD 效用。

### B
```text
H+RBR not better
BDD_RBR better
```

支持：

> RBR未证明新增标签信息，但将已有行为信息组织成更适合 BDD 的 geometry。

这仍然是有价值的正结果。

### C
```text
H+RBR better
BDD_RBR not better
```

说明：

> representation 有可读增量，但 frozen BDD 未能利用。

### D

两者都未支持：

> 当前候选没有获得 Level 2 qualification。

因此：

```text
INCREMENTAL_RISK_TEST =
SECONDARY
NOT_A_PRECONDITION_FOR_BDD
```

---

# 9. Paired 与 Unpaired 的最终定位

## 9.1 Paired BDD

同一 scenario：

```text
Reference_i
vs
Target_i
```

主要用于：

- mechanism attribution；
- controlled sensitivity；
- representation diagnosis。

## 9.2 Unpaired / Release Emulation

不同日志 / 场景构成：

```text
Release Reference sample
vs
Release Target sample
```

更接近生产真实问题：

> 软件新版本整体行为分布是否发生漂移？

因此 Level 2 的优先 Primary utility endpoint：

```text
UNPAIRED_RELEASE_EMULATION_BDD
```

paired BDD：

```text
CONTROLLED_SUPPORTING_EVIDENCE
```

如果 fresh data 容量不足以支撑可靠 unpaired inference，则必须在 E 解盲前预先缩窄为：

```text
PAIRED_PROSPECTIVE_BDD_ONLY
```

禁止实验后把 paired 成功写成 release-level validation。

---

# 10. HLC 最终处置

保持：

```text
HLC_V4_CURRENT_CANDIDATE = REJECTED

HLC_CURRENT_GENERATOR_BRANCH =
CLOSED_BY_SCOPE_AFTER_ENGINEERING_NONCONVERGENCE

HLC_V5 = NOT_AUTHORIZED

REMAINING_HLC_RUNS = NOT_AUTHORIZED

HLC_SCIENTIFIC_IMPOSSIBILITY =
NOT_ESTABLISHED
```

保留 Astra 三个 forensic 修正：

1. V4 名义形态在审查的 ideal-tracking 条件下仍未达到 frozen monotonic gate；
2. treatment terminal 是 overshoot + residual lateral motion，而非简单“尚未到中心”；
3. rolling planner future terminal 与 Primary80 realized terminal 不等价。

同时收紧：

- V4 只能说改善了运动学一致性与部分 reference feasibility；
- 不再说 intrinsic morphology feasibility 已解决；
- 多轮 development identities 不同，但科学判断沿前序结果持续演化，不能称相互独立；
- offline PASS 不等于 closed-loop evidence；
- B1 safety FAIL 必须保留。

HLC 的学术价值：

> 展示 planner trajectory 看起来合理，为什么仍不足以成为 behavior representation benchmark。

---

# 11. R0 / Stage7L 历史状态修正

## 11.1 D3

正式状态：

```text
D3 = INCONCLUSIVE
```

只能补充：

> simple full64 dilution 未被当前证据建立为充分解释。

不能把附带描述替换 formal state。

## 11.2 Stage7L

严格写：

```text
B seed3407 = Primary
```

old64 / A / C 为 secondary / supporting comparison。

禁止写成：

> old64/A/B/C 的 Primary 全部失败。

## 11.3 D1

可写：

```text
tested known semantics are decodable
```

但跨域 semantic transfer 保留其原有不确定性边界。

---

# 12. TSB 当前定位

当前：

```text
TSB_FAMILY_DEVELOPMENT_CANDIDATE =
FROZEN

DEV_CAL:
mechanism = 8/8
F_match = 8/8
safety = 8/8
measurement = 8/8
```

但：

```text
LOW_ORDER_NUISANCE_ELIMINATED =
NOT_ESTABLISHED

TSB_CLEAN_RESIDUAL_TASK =
NOT_ESTABLISHED
```

原因包括：

- mean speed / path length / mean abs accel delta 靠近 caliper；
- `end_minus_start_speed` 在 8 个 DEV-CAL pair 中已有明显 label signal。

因此 TSB 更准确叫：

> **通过 development matching contract 的 closed-loop temporal-mechanism candidate**

而不是：

> handcrafted summaries cannot detect this task。

---

# 13. TSB 的三个未来问题

## Q1 — Mechanism Qualification

fresh closed loop 是否稳定实现：

```text
baseline one brake phase
treatment two brake phases
release fraction
second peak ratio
F_match
safety
measurement
technical completeness
```

## Q2 — Existing Monitoring Signal Audit

不再问：

> H classifier 是否接近100%，若是就停止。

改为：

> **F / ego13 / H 已经保留多少信号？在真正的 BDD operating condition 下，是否仍存在值得研究的检测敏感性、样本效率或 geometry 问题？**

## Q3 — RBR-BDD Utility

最终问：

> **冻结 RBR-BDD 是否在相同 FPR 与 sample budget 下，比 handcrafted BDD 更有效地发现已确认的 closed-loop behavior shift？**

Q3 才是 Level 2 核心。

---

# 14. Handcrafted baseline 的最终职责

## F0 / Routine KPI

代表：

> 当前实际已经使用 / 当前项目常规预定义的行为 KPI。

它是 residual 定义的参考合同。

## ego13

代表现有项目的标准人工 behavior summary。

## H — Strong Development-Informed Challenger

一次性冻结，用于防止故意给 RBR 一个过弱 baseline。

允许包含：

- F summaries；
- ego13；
- fixed time-bin acceleration；
- fixed-lag autocorrelation；
- braking-mass temporal moments。

必须承认：

```text
H =
DEVELOPMENT_INFORMED
```

## O — Mechanism Positive Control

例如：

```text
brake_phase_count
release_fraction
second_peak_ratio
```

用途：

> 确认生成行为确实是目标 morphology。

O 不进入要求 RBR 超越的 Primary challenger。

---

# 15. 模型发现与后续人工特征新增的关系

假设真实项目中发生：

```text
新版本变道更加犹豫
```

此时：

```text
F0 / routine KPI =
没有“犹豫变道”指标
```

RBR-BDD首先报警：

```text
release behavior drift detected
```

随后研发人员：

1. 定位 lane-change task；
2. 查看最具代表性的 trajectory；
3. 发现 advance / pause / retreat / delayed commit；
4. 设计：
   - retreat count；
   - commit latency；
   - monotonicity；
   - hesitation score；
5. 进行定向试驾；
6. 人类确认：
   > 这是影响体验 / 发布的真实问题；
7. 新 KPI 加入后续版本常规监控。

此时不能说：

> “既然后来人工 feature 也能测，所以模型当初没有价值。”

模型真正提供的是：

> **在 feature 尚不存在之前触发 discovery。**

这就是：

```text
MODEL-FIRST DISCOVERY
→ FEATURE-BASED EXPLANATION
→ HUMAN CONFIRMATION
```

---

# 16. 模型不必自动命名新行为

论文不要求 RBR 一开始直接输出：

```text
“这是犹豫变道”
```

第一阶段只需要：

> **知道哪里有异常行为差异。**

推荐层级：

```text
1. Release-level alarm
2. Task / scenario localization
3. Representative trajectory retrieval
4. Feature / mechanism analysis
5. Human targeted drive
```

因此：

> Detection 可以由模型完成；semantic naming 可以在下游完成人机协同。

---

# 17. TSB fresh qualification 的 cohort 原则

优先采用：

```text
freeze candidate roster
        ↓
run whole roster
        ↓
apply whole-roster qualification rule
        ↓
PASS / FAIL
```

避免：

```text
run many
→ keep only mechanism-success pairs
→ call them prospective cohort
```

若使用资格后 subset，必须明确标记：

```text
POST_TREATMENT_QUALIFIED_CONDITIONAL_BENCHMARK
```

并报告完整分母及推断边界。

默认推荐：

```text
WHOLE_FROZEN_ROSTER
```

---

# 18. TSB sample size

不再采用：

```text
29/29
```

作为默认 generator-reliability Primary。

本博士不需要把：

> generator population success probability >90%

设为主要研究问题。

样本量应服务于：

```text
BDD Primary estimand
```

由以下因素共同决定：

- independent logs；
- paired / unpaired design；
- batch size；
- drift proportion；
- minimum useful detection gain；
- fixed FPR；
- null calibration precision；
- total simulation budget。

S1 需要冻结：

```text
Q size
sample-size decision rule
budget ceiling
primary operating condition
```

E 最终规模可以在 development variance estimation 后、E 解盲前冻结。

禁止：

```text
E差一点显著
→追加样本
```

---

# 19. 数据角色

## U — Encoder Training / Validation

用于：

- RBR training；
- checkpoint selection；
- normalization。

## Q — Fresh TSB Qualification

用于：

- generator；
- mechanism；
- safety；
- applicability qualification。

Q 完成后可预注册转为：

```text
D = development evidence
```

用于：

- H fit；
- normalization；
- readout；
- power estimation；
- kernel-development budget。

但 Q 不得重新成为 E。

## E — Locked Scientific Test

唯一用于最终：

```text
PRIMARY BDD comparison
```

以及预注册 secondary diagnostics。

E 在以下全部冻结前不得解盲：

- encoder；
- H；
- BDD statistic；
- kernel/bandwidth；
- null；
- sample design；
- operating condition。

---

# 20. RBR 最小训练原则

仍然：

```text
RBR_TRAINING = NOT_AUTHORIZED
```

未来若授权：

- one primary architecture family；
- fixed small number of seeds；
- no E-based rescue；
- no best-seed cherry-picking；
- handcrafted semantic reconstruction 不定义 latent geometry；
- 不使用 F / ego13 distance matrix 对齐 z64；
- checkpoint selection 不看 TSB scientific E result。

RBR 的核心职责是：

> 形成有利于 behavior drift discovery 的 representation space。

---

# 21. RBR Training 值得开展的最低条件

不是：

```text
H classifier must be imperfect
```

也不是：

```text
H+RBR must be expected to win
```

而是：

1. fresh TSB mechanism qualification 有效；
2. applicability / safety / data firewall 合法；
3. F / ego13 / H 已量化现有信号与替代解释；
4. 仍存在一个明确、可证伪、未被廉价 handcrafted-BDD 解决的 BDD 使用问题；
5. 有足够独立数据与可负担的 Primary BDD test；
6. RBR architecture/search budget 已冻结；
7. E 尚未暴露。

即：

> **RBR训练的理由是存在一个值得研究的 BDD measurement problem，而不是为了证明 RBR 一定有“新信息”。**

---

# 22. Level 1 / 2 / 3 Claim

## Level 1 — Dissertation Base Target

本论文主贡献：

> **闭环自动驾驶行为漂移评价必须区分干预意图、实际行为机制、表征可读性与分布测量效用；前一层成立不保证后一层有效。**

Level 1 不依赖新 RBR 获得阳性结果。

## Level 2 — Model-based Behavior Discovery Upgrade

若未来成功：

> **在预定义 TSB 适用域与冻结评价合同下，RBR 在相同误报约束和样本预算下，相对于预注册 handcrafted trajectory challenger，提高了对已确认 closed-loop behavior morphology drift 的 BDD 检测敏感性 / 样本效率。**

如果 secondary risk test 也支持，可附加：

> 在指定 readout protocol 下观察到增量预测价值。

如果只有 paired test：

> 只写 controlled prospective BDD sensitivity。

只有完成 unpaired E 才写：

> release-level / release-emulation monitoring。

## Level 3

多个 independent residual families、interaction/context、broader generalization：

```text
FUTURE WORK
```

---

# 23. 最终四阶段路线

## S1 — Scope & TSB–BDD Protocol Freeze

冻结：

- HLC closure wording；
- Residual-to-F0 定义；
- TSB applicability；
- Q roster policy；
- Q sample size；
- paired/unpaired primary claim；
- PRIMARY BDD operating condition；
- F0 / ego13 / H challenger roles；
- O positive control；
- BDD statistic；
- normalization；
- kernel / bandwidth selection budget；
- null calibration；
- independent FPR evaluation；
- detection gain / sample-efficiency metric；
- Q→D role；
- E firewall；
- technical schema。

禁止：

```text
simulation
roster execution
RBR training
TSB tuning
HLC reopening
```

## S2 — Fresh TSB Qualification & Handcrafted-BDD Audit

先完成 Q1：

```text
mechanism / safety / F_match / applicability
```

再执行：

```text
F0 / ego13 / H signal audit
+
handcrafted BDD development analysis
```

Owner 判断：

> 是否仍存在值得 RBR 解决的 BDD measurement problem？

若否：

```text
S4
```

若是：

```text
S3
```

## S3 — One-shot RBR–BDD Qualification

训练：

```text
one bounded RBR program
```

正式 Primary：

```text
BDD_RBR vs BDD_H
under matched FPR and sample budget
```

Secondary：

- H vs H+RBR risk；
- semantic probes；
- geometry；
- shortcut；
- paired sensitivity；
- mechanism attribution limits。

任何：

```text
PASS / FAIL / INCONCLUSIVE
```

都进入 S4。

不重新设计模型。

## S4 — Thesis Evidence Closure

根据最终证据：

```text
Level 1
or
Level 2
```

完成论文。

禁止：

```text
missing positive result
→ new HLC
→ new family
→ new RBR rescue
```

---

# 24. 最终论文方法闭环

```text
Known KPI Monitoring
        |
        | may miss previously unnamed behavior changes
        v
Learned Representation / BDD
        |
        | discovery
        v
Behavior Drift Alarm
        |
        | localization
        v
Task / Scenario / Episode Clusters
        |
        | interpretation
        v
Existing + Newly Designed Behavioral Features
        |
        | semantic diagnosis
        v
Targeted Human Driving / Expert Review
        |
        | confirmation
        v
Release Decision
        |
        | operationalization
        v
New KPI added to future routine monitoring
```

不是：

> learned model 替代 handcrafted feature。

而是：

> **learned model负责发现未知；handcrafted feature负责把未知变成可解释、可监控的已知；人类负责最终语义与发布价值判断。**

---

# 25. 当前一致性判断

Scientific Owner 当前认为，Astra 最新三轮审查、v2.1 与博士真实应用目标已经在以下核心点达成一致：

```text
1. BDD IS PRIMARY.

2. H+RBR incremental prediction IS SECONDARY.

3. HLC current branch is closed by scope,
   not scientifically proven impossible.

4. TSB is a frozen development candidate,
   not yet a clean residual benchmark.

5. Residual is defined relative to a frozen routine handcrafted set F0,
   not relative to every possible handcrafted feature.

6. The project does NOT aim to enumerate infinite handcrafted features.

7. The learned model's main purpose is behavior-drift discovery,
   especially for behavior dimensions not yet explicitly modeled by routine KPIs.

8. Handcrafted features remain essential for diagnosis and explanation
   after a drift is discovered.

9. Human targeted driving / expert review remains the final semantic and
   release-decision confirmation layer.

10. A later handcrafted KPI that explains a model-discovered drift does NOT
    invalidate the model's discovery value.

11. The model need not directly name “hesitation”.
    Detecting a reliable behavior difference first is sufficient;
    semantic interpretation can occur downstream.

12. A successful Level-2 RBR claim is primarily:
    better calibrated BDD sensitivity / sample efficiency under the frozen
    evaluation contract, not “information impossible for humans to handcraft”.
```

---

# 26. 当前立即状态

```text
HLC_V5 = NOT_AUTHORIZED
REMAINING_HLC_RUNS = NOT_AUTHORIZED

TSB_SIMULATION = NOT_YET_AUTHORIZED

RBR_TRAINING = NOT_AUTHORIZED

PRIMARY_FUTURE_QUESTION =
CAN_A_LEARNED_TRAJECTORY_REPRESENTATION_ENABLE_MORE_EFFECTIVE
BEHAVIOR_DRIFT_DISCOVERY_THAN_THE_CURRENT_HANDCRAFTED_MONITORING_SPACE?

NEXT_STAGE =
S1_SCOPE_AND_TSB_BDD_PROTOCOL_FREEZE
```

---

# 27. 一句话论文灵魂

> **不是试图事先为所有可能的驾驶行为变化设计无限多 KPI，而是利用 learned trajectory representation 先发现“我们尚未定义的问题”，再用行为特征和人类试驾把异常解释、确认并最终固化为新的工程指标。**

---

`RBR64_STAGE_R_ROADMAP_V2_1_OWNER_SYNTHESIS_DRAFT`
