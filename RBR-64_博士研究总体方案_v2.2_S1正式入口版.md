# RBR-64 博士研究总体方案 v2.2（Stage S1 正式入口版）

> 项目：E2E-Evaluation / 博士论文  
> 文档状态：`RBR64_STAGE_R_ROADMAP_V2_2_S1_ENTRY_DRAFT_FOR_FREEZE`  
> 更新时间：2026-09-06（Asia/Shanghai）  
> Active branch：`20260825_stageR_new`  
> 版本关系：v2.2 在 v2.1 基础上吸收 Astra 最终审查与 Scientific Owner 最终意见。  
> 本文用于定义当前研究主线并正式进入 **Stage S1 — Scope & TSB–BDD Protocol Freeze**。  
> 本文不回写任何历史冻结结果，不授权 simulation、roster execution 或 RBR training。

---

# 0. v2.2 相对 v2.1 的最终修订

v2.1 已经完成研究主线从“64D 信息增量”向“模型化行为发现 / BDD 效用”的重定位。

v2.2 进一步完成以下证据边界修正：

1. **BDD utility 正式确认为未来 Level 2 的 PRIMARY。**
2. **H vs H+RBR incremental predictive-risk 正式降为 SECONDARY diagnostic。**
3. “模型发现未知行为 → feature 解释 → 人类确认”保留为**应用愿景 / 工程工作流**，而不是当前 Route A 已经验证的完整科学主张。
4. 当前实验目标收紧为：
   > **在不为目标行为机制专门新增常规监控 KPI 的条件下，检验 learned trajectory representation 是否能改善 closed-loop behavior drift detection。**
5. BDD alarm 表示“行为分布发生变化”，不自动表示异常、退化或 release-blocking issue。
6. `BDD_RBR > BDD_H` 只能直接支持冻结 pipeline 的 BDD 效用改善；“更好的 latent geometry”只能作为兼容解释，不能直接视为已证明根因。
7. Q→D 的角色转换必须**在 Q 执行前预注册**。
8. Encoder architecture / checkpoint selection 必须仅依据 **U-training / U-validation 的冻结训练目标**；不得使用 Q/D 的 TSB scientific outcome 选择 encoder。
9. 完成 unpaired E 只能支持对应 frozen public-data / simulation setting 下的 **release emulation**；不能直接写成真实量产软件发布验证。
10. S1 必须冻结一个唯一 **Primary BDD comparison metric / operating point**，禁止事后在 detection、sample efficiency、stability 中挑最有利者作为主结果。
11. R0 D3 formal state 保持 `INCONCLUSIVE`。
12. Stage7L formal Primary 保持 `B seed3407`，old64/A/C 为 supporting/secondary evidence。

---

# 1. 博士论文的现实问题

传统自动驾驶行为评价依赖大量 handcrafted KPI：

- speed / acceleration / jerk；
- yaw rate / curvature / lateral acceleration；
- THW / gap；
- lane-change duration；
- safety / comfort / efficiency；
- 以及随着历史问题不断增加的新规则。

这些 KPI 对“已经知道的问题”很有效。

真正困难的是：

> **当某种行为变化尚未被团队命名、尚未有专门 KPI 时，如何在正式试驾之前发现“这个版本的行为与参考版本不一样”？**

现实例子：

```text
已有 KPI 全部正常
        ↓
新版本仿真结果看起来正常
        ↓
实际试驾发现：
变道明显犹豫 / 试探 / 回撤 / 延迟 commit
        ↓
版本拒绝释放
        ↓
团队事后新增 hesitation-related KPI
```

因此本论文不试图：

> 为所有未来行为问题预先穷举无限 handcrafted features。

而是研究：

> **能否建立一个 model-based behavior-drift discovery layer，在当前 routine KPI 尚未专门刻画目标行为时，首先发现版本行为分布发生了值得分析的变化。**

---

# 2. 应用愿景与论文当前可验证主张必须分开

## 2.1 应用愿景

推荐工程闭环：

```text
Model-based behavior drift discovery
        ↓
Task / scenario / episode localization
        ↓
Behavioral feature / mechanism analysis
        ↓
Targeted expert review / human driving
        ↓
Semantic & release-value confirmation
        ↓
New KPI operationalization
```

简称：

> **Model-first discovery → Feature-based diagnosis → Human confirmation**

这一工作流符合真实自动驾驶质量研发流程。

## 2.2 当前论文 Route A 实际验证范围

当前 S1–S4 主要验证：

```text
closed-loop mechanism validity
        +
representation / BDD detection utility
        +
supporting behavioral diagnosis
```

并**不自动证明**：

- 能发现研究者此前完全未知的机制 family；
- 能自动命名“犹豫变道”等新行为；
- human confirmation workflow 已被完整实证验证；
- drift alarm 一定对应质量退化或 release-blocking issue。

因此正式实验目标写成：

> **在不为目标行为机制专门新增 routine monitoring KPI 的条件下，检验 learned trajectory representation 是否能在相同误报约束与样本预算下改善对 closed-loop behavior drift 的检测。**

---

# 3. 论文核心科学问题

最终推荐：

> **如何建立一套面向 closed-loop planning-policy software release 的行为漂移评价方法，区分干预意图、实际行为机制、表征敏感性与分布检测效用，并检验 learned trajectory representation 是否能在固定 routine handcrafted monitoring contract 下提供更有效的 behavior drift discovery？**

英文工作表述：

> **How can closed-loop planning-policy behavior drift be evaluated by separating intervention intent, realized behavioral mechanism, representation sensitivity, and distributional detection utility; and can a learned trajectory representation improve behavior-drift discovery under a fixed routine handcrafted monitoring contract?**

---

# 4. Residual 的最终定义

正式定义：

> **Residual-to-F0 behavior：在指定适用域、测量窗口和预定义 routine handcrafted summary set F0 的监控合同下，仍存在的、由独立机制测量确认的 closed-loop behavior structure difference。**

其中：

```text
F0 =
CURRENT_ROUTINE_HANDCRAFTED_MONITORING_SET
```

必须明确：

- residual 是相对于 **F0** 的；
- 不是相对于所有未来可能发明的 handcrafted feature；
- 不是“人类无法描述”；
- 不是“所有 classifier 都无法从 F0 中预测标签”；
- 不是与已知语义天然正交的 latent subspace。

后来新增 KPI 能解释一个由模型先发现的 drift：

> 不会否定模型此前的 discovery value。

---

# 5. Learned representation 与 handcrafted KPI 的最终职责

## Learned representation / BDD

主要职责：

> **发现当前 routine KPI monitoring space 没有充分显式覆盖的行为漂移。**

## Handcrafted KPI / features

主要职责：

- known-issue monitoring；
- downstream diagnosis；
- mechanism explanation；
- engineering operationalization。

## Expert / human driving

主要职责：

- semantic interpretation；
- experience relevance；
- release-value judgement。

注意：

> 这是推荐的工程职责分工，不是互斥能力边界。

routine KPI 与 model-based BDD 在实际系统中应并行存在。

---

# 6. BDD 是 PRIMARY

未来 Level 2 的主要科学问题：

> **在相同误报约束、相同样本预算和同一 frozen operating condition 下，RBR-BDD 是否比预注册 handcrafted challenger BDD 提供更好的 behavior-drift detection utility？**

禁止比较：

```text
raw MMD²_RBR
vs
raw MMD²_H
```

因为不同 representation 的 kernel geometry 与尺度不同。

Primary 必须比较可公平解释的实际检测量。

---

# 7. S1 必须冻结唯一 Primary BDD 指标

v2.2 不提前替 S1 决定最终数值，但要求 S1 在 execution 前只选一个 Primary。

候选包括：

### Candidate A — Detection gain at fixed operating point

\[
\Delta_{\mathrm{BDD}}
=
P_{\mathrm{detect}}(RBR)
-
P_{\mathrm{detect}}(H)
\]

在固定：

```text
FPR
batch size m
drift proportion π
```

下比较。

### Candidate B — Sample efficiency

比较达到预定义 target detection probability 所需样本量。

### Candidate C — 其他预定义效用量

只有在 S1 给出科学理由后可用。

S1 必须确定：

```text
ONE_PRIMARY_BDD_METRIC
ONE_PRIMARY_OPERATING_POINT
ONE_PRIMARY_COMPARISON
```

其余全部 Secondary。

禁止：

```text
detection更有利 → 用detection
sample efficiency更有利 → 改成sample efficiency
stability更有利 → 再改成stability
```

---

# 8. H+RBR incremental predictive value = SECONDARY

保留：

```text
H
vs
H + RBR
```

作为辅助解释。

可能分支：

## A

```text
incremental prediction SUPPORTED
BDD gain SUPPORTED
```

支持：

> 指定 readout 下有增量预测价值，且该增量与 BDD 效用改善同时出现。

## B

```text
incremental prediction NOT SUPPORTED
BDD gain SUPPORTED
```

直接支持：

> frozen RBR-BDD pipeline 具有更高检测效用。

允许讨论：

> 该结果与 representation organization / geometry 更有利的解释相容。

但不能直接写：

> geometry improvement was proven.

## C

```text
incremental prediction SUPPORTED
BDD gain NOT SUPPORTED
```

说明：

> 可读预测增量没有转化为 frozen BDD utility。

## D

两者均未支持：

> 当前 Level 2 RBR qualification 未成功。

因此：

```text
H_PLUS_RBR_RISK_TEST =
SECONDARY
NO_PRECONDITION_ROLE
```

---

# 9. Paired 与 Unpaired 的正式角色

## 9.1 Paired controlled BDD

同一 scenario 的 baseline / treatment：

用于：

- mechanism attribution；
- controlled sensitivity；
- representation diagnosis。

状态：

```text
CONTROLLED_SUPPORTING_EVIDENCE
```

## 9.2 Unpaired release emulation

不同 independent logs / scenarios 构造 reference 与 target release sample：

用于：

> 模拟真实版本级行为分布比较。

优先 Level 2 Primary：

```text
UNPAIRED_RELEASE_EMULATION_BDD
```

但只有在：

- independent logs 足够；
- calibration 合法；
- FPR evaluation 独立；
- sample design / power 足够；

时才能使用。

如果数据容量不足，S1 必须在 E 解盲前预定义 fallback：

```text
PAIRED_PROSPECTIVE_BDD_ONLY
```

此时主张同步收缩。

完成 unpaired E 也只能写：

> release emulation under the frozen study setting

不能直接写：

> validated real production release monitoring.

---

# 10. HLC 最终状态

```text
HLC_V4_CURRENT_CANDIDATE = REJECTED

HLC_CURRENT_GENERATOR_BRANCH =
CLOSED_BY_SCOPE_AFTER_ENGINEERING_NONCONVERGENCE

HLC_V5 = NOT_AUTHORIZED

REMAINING_HLC_RUNS = NOT_AUTHORIZED

HLC_SCIENTIFIC_IMPOSSIBILITY =
NOT_ESTABLISHED
```

必须保留：

- V4 nominal morphology 在审查 ideal-tracking 条件下仍未通过 frozen monotonic gate；
- treatment terminal 是 overshoot + residual lateral motion；
- rolling future reference endpoint 与 Primary80 realized endpoint 不等价；
- safety FAIL；
- offline feasibility 不是独立 closed-loop success evidence；
- development rounds 使用不同 identities，但科学判断彼此连续演化。

HLC 在论文中的价值：

> 一个 planner-intent 看似合理、但 closed-loop realization 和 benchmark qualification 失败的前瞻反例。

---

# 11. R0 / Stage7L 历史边界

## D3

正式：

```text
D3 = INCONCLUSIVE
```

只允许附带说明：

> simple full64 dilution 未被建立为充分解释。

## Stage7L

正式：

```text
B seed3407 = PRIMARY
```

old64 / A / C 为 supporting / secondary。

## D1

允许：

```text
tested semantics are decodable
```

但 cross-domain semantic transfer 的正式不确定性保持不变。

---

# 12. TSB 当前状态

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

因此当前定位：

> **closed-loop temporal-mechanism development candidate that passed the frozen development matching contract**

而不是：

> handcrafted methods cannot detect the behavior.

---

# 13. TSB 的三个未来问题

## Q1 — Mechanism Qualification

fresh logs 中能否稳定满足：

```text
technical completeness
measurement validity
baseline one-phase
treatment two-phase
release fraction
second peak ratio
F_match
official safety
applicability
```

## Q2 — Existing Monitoring Signal Audit

问：

> **F0 / ego13 / H 已经保留多少相关信号？在目标 BDD operating condition 下，现有 handcrafted monitoring 是否已经解决该使用问题，还是仍存在值得研究的检测敏感性 / sample-efficiency / geometry 问题？**

不是：

> H classifier 是否完美？

## Q3 — RBR-BDD Utility

问：

> **冻结 RBR-BDD 是否在同一 FPR、sample budget 和 operating condition 下比 handcrafted BDD 更有效？**

Q3 是 Level 2 Primary。

---

# 14. Handcrafted comparison contract

## F0 — Routine KPI

代表当前实际 / 项目标准监控空间。

## ego13

代表 Generation-1 项目的标准人工行为摘要。

## H — Development-informed strong challenger

角色：

> 防止故意使用过弱 handcrafted baseline。

H 必须在 S1 一次性定义并冻结：

- exact feature list；
- time bins；
- lags；
- smoothing；
- derivatives；
- units；
- validity；
- duplicate-column handling；
- scaler；
- BDD preprocessing；
- tuning budget。

必须明确：

```text
H =
DEVELOPMENT_INFORMED_HANDCRAFTED_CHALLENGER
```

不能声称与 TSB development 完全无关。

## O — Mechanism positive control

例如：

```text
brake_phase_count
release_fraction
second_peak_ratio
```

只用于：

> mechanism confirmation / interpretation。

不加入要求 RBR 超越的 Primary challenger。

---

# 15. 禁止“无限增加 handcrafted features 追杀模型”

S1 必须冻结：

```text
F0
ego13
H
O
```

一旦 E 解盲：

禁止：

```text
RBR detects pattern X
→ invent handcrafted feature X
→ add it to Primary H
→ rerun comparison
```

同样禁止：

```text
H performs strongly
→ remove H features
→ weaken H readout
→ rerun comparison
```

任何未来新增 mechanism-specific feature：

> 只能进入 downstream diagnosis / future operational KPI。

不能回写当前 Primary comparison。

---

# 16. Q→D 必须在 Q 执行前预注册

正确顺序：

```text
before Q execution:
pre-register role transition
        ↓
run Q
        ↓
if qualification stage completes under protocol:
Q becomes outcome-exposed development asset D
for exactly predefined uses
```

允许 D 用于：

- H fit 中允许的数据依赖部分；
- scaler；
- readout；
- kernel / bandwidth development；
- calibration-development；
- variance / power estimation。

Q/D 不得：

- 成为 E；
- 重新被描述为 untouched；
- 用于选择 encoder architecture / checkpoint。

---

# 17. Encoder selection 必须只依据 U

未来若授权 RBR：

```text
U =
encoder training + validation only
```

Encoder 的：

- architecture；
- training objective；
- checkpoint；
- seed-handling rule；

只能依据预冻结 U-training/U-validation objective。

禁止：

```text
train several encoders
→ compare TSB Q/D outcome
→ choose best encoder
→ test on E
```

即使 E 未暴露，这仍然属于 TSB outcome-driven model selection。

允许 D 调整：

- frozen-capacity readout；
- kernel；
- bandwidth；
- calibration；

但其预算必须 S1 预先冻结，并与 encoder selection 分开。

---

# 18. TSB cohort qualification policy

优先：

```text
WHOLE_FROZEN_ROSTER
```

流程：

```text
freeze roster
→ execute whole roster
→ apply frozen whole-roster qualification rule
→ PASS / FAIL
```

这样避免：

```text
run many
→ retain mechanism-success pairs only
→ call survivors prospective cohort
```

若 S1 最终选择 conditional qualified subset design，必须明确：

```text
POST_TREATMENT_QUALIFIED_CONDITIONAL_BENCHMARK
```

并预注册：

- complete denominator；
- qualification selection rule；
- inference limitations；
- null / exchangeability implications。

默认不采用此设计。

---

# 19. Sample size

不再使用：

```text
29/29
```

作为默认目标。

TSB generator population reliability 不是当前博士 Primary。

S1 的样本设计服务于：

```text
PRIMARY BDD ESTIMAND
```

需共同考虑：

- independent log count；
- Q qualification budget；
- paired / unpaired design；
- batch size；
- drift proportion；
- target FPR；
- minimum useful BDD gain；
- A/A calibration precision；
- A/A evaluation precision；
- E power；
- simulation budget。

S1 必须冻结：

```text
Q size
Q stopping policy
overall simulation budget ceiling
primary BDD sample-size decision rule
```

允许：

> 在 Q→D 后估计 variance / power 参数，并在 E 解盲前一次性冻结最终 E size。

禁止：

```text
look at E
→ almost significant
→ add samples
```

---

# 20. 数据角色

## U

Encoder training / validation。

## Q

Fresh TSB mechanism / safety / F_match / applicability qualification。

## D

预先规定的 Q-after-qualification development role，或其它独立 development assets。

## E

Locked final scientific test。

E 只用于：

```text
PRIMARY BDD comparison
+
pre-registered secondary diagnostics
```

E 在以下全部冻结前不得解盲：

- encoder；
- H；
- BDD statistic；
- normalization；
- kernel / bandwidth；
- null；
- calibration；
- primary operating condition；
- primary comparison metric；
- E sample size；
- analysis plan。

---

# 21. RBR training 最低授权条件

当前：

```text
RBR_TRAINING = NOT_AUTHORIZED
```

未来授权不要求：

```text
H classifier imperfect
```

也不要求：

```text
H+RBR expected to win
```

至少要求：

1. TSB fresh qualification有效；
2. applicability / safety / data firewall闭合；
3. F0 / ego13 / H 的 signal 与替代解释已量化；
4. 仍存在一个明确、可证伪、未被廉价 handcrafted-BDD 在目标 operating point 充分解决的 measurement problem；
5. Primary BDD test 设计完整且可负担；
6. encoder training objective / architecture budget 已冻结；
7. U/Q/D/E 角色冻结；
8. E 未暴露。

---

# 22. Level 1 / Level 2 / Level 3

## Level 1 — Dissertation base contribution

核心科学陈述：

> **闭环自动驾驶行为漂移评价必须区分干预意图、实际行为机制、表征可读性与分布测量效用；前一层成立不能保证后一层有效。**

支持来源：

- Stage6/7/7L；
- R0；
- R1；
- R2；
- HLC negative development；
- TSB development candidate；
- paired/unpaired distinction；
- benchmark / measurement failure analysis。

Level 1 不依赖新 RBR 获得阳性结果。

## Level 2 — Model-based BDD utility

如果成功：

> **在预定义 TSB 适用域和冻结 evaluation contract 下，RBR 在相同 FPR、样本预算与主要使用条件下，相对于预注册 handcrafted challenger，提高了对已确认 closed-loop behavior morphology drift 的 BDD 检测效用。**

可根据实际 Primary 进一步写：

- detection sensitivity；
- sample efficiency；
- 或 S1 最终冻结的唯一主要效用量。

如果 secondary incremental-risk 也支持：

> 可附加“在指定 readout protocol 下具有增量预测价值”。

只有完成合格 unpaired E：

> 才允许写 `release-emulation`。

不写：

> real production release validation.

## Level 3

多个 independent residual families / interaction / broader unknown-morphology discovery：

```text
FUTURE WORK
```

---

# 23. 应用愿景：Model-first Discovery

最终工程愿景：

```text
Routine KPI monitoring
        +
Learned representation / BDD
        ↓
Behavior drift signal
        ↓
Task / scenario / episode localization
        ↓
Representative trajectory inspection
        ↓
Behavioral mechanism / new feature design
        ↓
Targeted expert review / human driving
        ↓
Release-value judgement
        ↓
New KPI operationalization
```

论文当前主要实证：

> **discovery layer + mechanism validity + BDD utility。**

下游：

- representative-case retrieval；
- explanation usability；
- human confirmation effectiveness；

只有实际执行相应研究后才能升级为实证主张。

---

# 24. BDD alarm 的语义

必须写：

```text
BDD alarm =
BEHAVIOR DISTRIBUTION CHANGE DETECTED
```

不能自动写：

```text
anomaly
degradation
bad behavior
release-blocking issue
```

因为版本变化可能是：

- 更保守；
- 更积极；
- 更舒适；
- 风格变化但仍可接受。

实际价值判断需要：

- mechanism analysis；
- engineering context；
- expert / human review。

---

# 25. Stage S1 正式授权边界

从 v2.2 开始：

```text
S1_PROTOCOL_WORK = AUTHORIZED
```

授权范围仅限：

- read-only repo / evidence review；
- protocol / schema / SAP drafting；
- deterministic zero-run code for protocol tooling；
- schema-faithful fixture tests；
- sample-size / power design；
- static applicability derivation；
- documentation / manifests；
- tests that do not invoke simulation or runner.run.

明确禁止：

```text
simulation
runner.run
TSB rollout
HLC rollout
roster execution
new scientific identity exposure
TSB parameter tuning
HLC V5
RBR training
encoder checkpoint selection from Q/D
E construction / unblinding
```

---

# 26. Stage S1 必须产出的最终包

建议 S1 只产出一组扁平化、互相引用而非重复复制的冻结文件。

## S1-A — Scope / Claim Freeze

必须明确：

- HLC closure；
- Level 1 / 2 claim；
- application vision vs current empirical claim；
- Residual-to-F0 definition；
- paired / unpaired claim boundary。

## S1-B — TSB Applicability Contract

必须重新论证：

- current R2 TSB candidate；
- inherited 2.0 m/s floor；
- low-speed measurability；
- Primary80；
- safety eligibility；
- reference completeness；
- independent log rule。

不得根据 future outcome 调整。

## S1-C — Qualification Protocol

冻结：

- whole-roster policy；
- Q size；
- run order；
- technical vs scientific failure；
- no replacement；
- no rerun；
- stop policy；
- budget ceiling；
- Q→D pre-registration。

## S1-D — Handcrafted Challenger Contract

冻结：

```text
F0
ego13
H
O
```

及：

- exact implementation；
- normalization；
- missingness；
- feature validity；
- no post-E expansion rule。

## S1-E — BDD Statistical Analysis Plan

必须冻结：

- paired / unpaired Primary；
- one Primary BDD metric；
- one operating point；
- FPR；
- batch size / drift composition decision；
- RBR vs H fair-tuning budget；
- null calibration；
- independent FPR evaluation；
- CI；
- bootstrap / resampling independent unit；
- multiplicity；
- E size decision rule；
- success / fail / inconclusive。

## S1-F — Secondary Diagnostic SAP

冻结：

- H vs H+RBR risk；
- semantic probes；
- geometry diagnostic；
- shortcut audit；
- mechanism-attribution wording。

它们不能取代 BDD Primary。

## S1-G — Canonical Technical Schema

目标：

```text
one canonical schema
one executor
one passive recorder
one analyzer
one primary manifest
one stage authorization
```

并用真实 schema-faithful fixture 覆盖：

- 80 states / 79 controller transitions；
- timestamp irregularity；
- missing/wrong keys；
- unit mismatch；
- one/two-phase boundaries；
- low-speed invalidity；
- F_match boundary；
- safety fail；
- duplicate identity；
- budget violation；
- analyzer/production field parity。

---

# 27. S1 结束的唯一 Owner 决策

S1 完成后不自动进入仿真。

Scientific Owner 只做一个正式判断：

```text
TSB_FRESH_QUALIFICATION_SIMULATION =
AUTHORIZED
or
NOT_AUTHORIZED
```

授权前必须确认：

- protocol 完整；
- candidate SHA 完整；
- applicability 定义合理；
- Q roster selection rule 合法；
- Primary BDD estimand 明确；
- H / O / F0 / ego13 冻结；
- data firewall 完整；
- simulation budget 明确；
- no hidden outcome-dependent tuning path。

---

# 28. 最终四阶段路线

```text
S1
Scope & TSB–BDD Protocol Freeze
        ↓
S2
Fresh TSB Qualification
+
Handcrafted-BDD Development Audit
        ↓
Scientific Owner decision:
Is there still a meaningful BDD problem worth RBR?
        ↓
if YES
S3
One-shot RBR–BDD Qualification
        ↓
S4
Thesis Evidence Closure

if NO at S2
→ directly S4
```

任何 S3：

```text
PASS
FAIL
INCONCLUSIVE
```

都进入 S4。

不再开启：

- HLC rescue；
- new family rescue；
- second RBR rescue。

---

# 29. 当前最高层共识

```text
1. BDD IS PRIMARY.

2. H+RBR INCREMENTAL PREDICTION IS SECONDARY.

3. MODEL-FIRST DISCOVERY → FEATURE DIAGNOSIS → HUMAN CONFIRMATION
   IS THE APPLICATION VISION.

4. CURRENT ROUTE A MAINLY VALIDATES THE DISCOVERY / BDD LAYER,
   NOT A COMPLETE UNKNOWN-MECHANISM + HUMAN-STUDY PIPELINE.

5. RESIDUAL IS RELATIVE TO F0,
   NOT TO ALL POSSIBLE HANDCRAFTED FEATURES.

6. THE DISSERTATION DOES NOT AIM TO ENUMERATE INFINITE KPIs.

7. HANDCRAFTED FEATURES REMAIN ESSENTIAL FOR DIAGNOSIS
   AND FUTURE OPERATIONAL MONITORING.

8. BDD ALARM MEANS BEHAVIOR CHANGE, NOT AUTOMATIC DEGRADATION.

9. HLC IS CLOSED BY SCOPE / ENGINEERING NONCONVERGENCE,
   NOT SCIENTIFIC IMPOSSIBILITY.

10. TSB IS A FROZEN DEVELOPMENT CANDIDATE,
    NOT YET A CLEAN RESIDUAL BENCHMARK.

11. ENCODER SELECTION MUST BE U-ONLY.

12. Q→D ROLE MUST BE PRE-REGISTERED BEFORE Q.

13. LEVEL-2 PRIMARY CLAIM IS BDD UTILITY UNDER A FROZEN FAIR COMPARISON.

14. REAL UNKNOWN-MORPHOLOGY DISCOVERY AND HUMAN-CONFIRMATION VALIDATION
    REMAIN STRONGER FUTURE EXTENSIONS UNLESS DIRECTLY TESTED.
```

---

# 30. 一句话论文灵魂

> **不是试图预先为所有可能的驾驶行为变化设计无限多 KPI，而是研究 learned trajectory representation 能否在固定的常规监控合同下更有效地发现行为漂移，并把这些发现交给后续行为分析和专家判断去解释其含义与工程价值。**

---

`RBR64_STAGE_R_ROADMAP_V2_2_S1_ENTRY_DRAFT_FOR_FREEZE`
