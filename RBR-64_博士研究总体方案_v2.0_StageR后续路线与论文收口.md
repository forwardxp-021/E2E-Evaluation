# RBR-64 博士研究总体方案 v2.0（Stage R 后续路线与论文收口）

> 项目：E2E-Evaluation / 博士论文  
> 文档状态：`RBR64_STAGE_R_ROADMAP_V2_DRAFT_FOR_OWNER_FREEZE`  
> 更新时间：2026-09-06（Asia/Shanghai）  
> Active branch：`20260825_stageR_new`  
> 目的：统一 2026-09 初最新 Scientific Owner、Astra 独立评审与现有 Stage R 证据，作为后续 Work 执行前的唯一“当前思路文档”。  
> 本文**不回写历史冻结协议**，不授权新的 simulation、roster selection 或 RBR training。

---

# 0. 与两个 v1.0 文档的关系

现有两份 v1.0 文档职责不同，不能简单覆盖。

## 0.1 《R0 Representation & Measurement Audit Protocol v1.0》

该文档是**已经冻结的历史协议**，记录 R0 在 2026-08-26 时的审计问题、资产角色、统计合同与训练授权边界。

其历史价值必须保留：

- R0 D0–D5 的问题定义；
- development / audit / future confirmation 的数据防火墙思想；
- hypothesis-level evidence state；
- D1 semantic retention；
- D3 representation / readout / BDD statistic 分离；
- D4 residual benchmark 的三重资格门；
- anti-selection / anti-leakage 原则。

因此：

```text
R0_PROTOCOL_V1_0 =
HISTORICAL_FROZEN_PROTOCOL
DO_NOT_OVERWRITE
```

后续发现不能回写该协议，只能通过新版本的总体路线、addendum 或后续协议记录。

## 0.2 《RBR-64 Representation-V2 博士研究最终总体方案 v1.0》

该文档是当时的**研究方向总纲**，核心思想仍然成立：

\[
Z_{64}
=
Known\ Behavioral\ Semantics
+
Meaningful\ Residual\ Behavior
\]

以及：

- Representation 与 BDD Measurement 分离；
- handcrafted semantics 是 anchor，不是 latent geometry 定义；
- residual 必须具有可验证行为意义；
- final confirmation 不得使用已解盲 development evidence；
- RBR 不应为了“超越 ego13”而被训练成 handcrafted geometry 的复制品。

但 v1.0 中以下内容已经被 Stage R 真实执行结果部分取代：

- “下一步唯一任务是编写 R0 protocol”；
- “Residual benchmark 至少三个 family”作为当前强制门；
- RBR-A/B/C 连续开发是默认路线；
- HLC/TSB/IP 尚处于设计假设；
- R1→R4 的原始阶段定义。

因此：

```text
RBR64_OVERALL_PLAN_V1_0 =
HISTORICAL_SCIENTIFIC_DIRECTION

THIS_V2 =
CURRENT_FORWARD_ROADMAP
```

---

# 1. 当前博士研究的核心问题

本论文当前最稳健的核心问题不再是：

> 能否训练一个一定优于 ego13 的 64D embedding？

而是：

> **如何建立一套面向 closed-loop planning policy 的行为漂移评估方法，严格区分行为干预意图、闭环实现、representation 信息、measurement/readout 和统计检测，并检验 learned trajectory representation 是否在预定义手工摘要之外提供有用的轨迹级增量信息。**

英文工作表述：

> **How can closed-loop planning-policy behavior drift be evaluated in a way that separates intervention intent, realized behavioral mechanism, representation content, measurement/readout, and statistical detection; and under what conditions does a learned trajectory representation provide useful incremental information beyond predefined handcrafted summaries?**

这一表述允许：

- learned representation 得到正结果；
- learned representation 得到负结果；
- residual benchmark 开发失败；
- 某一行为 family 只在有限适用域成立。

论文闭环不再依赖“必须出现一个成功的 RBR”。

---

# 2. 当前证据链已经建立的事实

## 2.1 Generation-1

Stage6 / Stage7 / Stage7L 已经支持：

1. learned representation 可以检测一部分 planner-conditioned behavior drift；
2. B 在某些 release-level unpaired 设置中很强；
3. interaction planner treatment 可以被机制确认，但未证明 context encoder 的独立增量；
4. Stage7L pure-lateral planner treatment 成立；
5. Stage7L 中 ego13 极敏感，而 learned old64/A/B/C 的 Primary BDD 没有可靠检出；
6. “信息可 probe”与“BDD 可检测”不能等同。

## 2.2 R0

R0 正式澄清：

```text
D1_INFORMATION_RETENTION = SUPPORTED
KNOWN_SEMANTICS_DECODABLE = YES
```

但：

```text
KNOWN_SEMANTICS_DECODABLE
!=
BDD_SUITABILITY
```

D3 的正式解释必须保持谨慎：

```text
SIMPLE_FULL64_DILUTION_HYPOTHESIS =
NOT_SUPPORTED_AS_SUFFICIENT_EXPLANATION

MEASUREMENT_READOUT_PROBLEM =
NOT_FULLY_RESOLVED
```

不能写成：

> 已经证明 Stage7L 失败主要由于 temporal information loss。

该因果结论仍未建立。

## 2.3 R1

R1 首次 prospective residual benchmark 证明：

> planner-space intent 不等于 closed-loop realized mechanism。

HLC：

```text
F_match = 12/12
mechanism = 0/12
transfer = ATTENUATED
```

TSB：

```text
F_match = 12/12
mechanism = 0/12
transfer = COLLAPSED
```

因此 R1 的核心方法学发现是：

> **Residual benchmark 必须在 realized closed-loop behavior 层确认，不能把 planner trajectory 的形态直接当作 representation evaluation label。**

## 2.4 R2

R2-A 将 transfer failure 分解到 planner/replanning → LQR → vehicle 链。

R2-B：

- TSB development candidate 达到 8/8 mechanism / F_match / safety / measurement；
- HLC 达到 mechanism 6/8，但 endpoint 0/8。

随后 HLC 经 BH、BI、BJ 多轮 architecture development，依次暴露：

1. constant re-anchor 不形成 target-center attractor；
2. XY 与 heading/curvature 不一致导致 controller-interface 缺陷；
3. kinematically consistent V3 暴露原 morphology time-scale lateral-acceleration 不可行；
4. V4 修复运动学接口与 intrinsic morphology feasibility；
5. offline joint-support audit显示 V4 在部分 moving-regime geometry 上可行；
6. fresh V4 canary 最终仍未通过 frozen mechanism + endpoint + safety 联合门禁。

---

# 3. HLC 最终处置

## 3.1 当前可以支持的结论

```text
HLC_V4_CURRENT_CANDIDATE =
REJECTED

HLC_CURRENT_GENERATOR_BRANCH =
ENGINEERING_NONCONVERGENCE

REMAINING_HLC_RUNS =
NOT_AUTHORIZED

HLC_V5 =
NOT_AUTHORIZED_IN_CURRENT_DISSERTATION_SCOPE
```

## 3.2 当前不能支持的结论

```text
HLC_SCIENTIFIC_IMPOSSIBILITY =
NOT_ESTABLISHED

HLC_CONSTRUCT_PHYSICALLY_IMPOSSIBLE =
NOT_ESTABLISHED
```

当前停止 HLC 的理由是：

> 经多轮相互隔离的 architecture development 后，当前 branch 在 fresh closed-loop canary 上仍未通过冻结联合门；继续进入 predictive controller/vehicle-aware constrained optimization 将成为新的控制器研究项目，超出当前博士论文的合理范围。

这是：

```text
PROJECT_SCOPE_TERMINATION
```

不是：

```text
SCIENTIFIC_IMPOSSIBILITY_PROOF
```

## 3.3 Astra 独立评审新增的三个重要修正

### A. V4 名义机制本身缺少冻结 monotonic gate 裕量

即使理想跟踪 V4 名义 progress，按 frozen 10 Hz / median3 / deadzone mechanism 计算：

```text
treatment monotonic fraction ≈ 0.9502554
delta vs ideal monotonic baseline ≈ -0.0497446
```

而冻结要求：

```text
delta <= -0.10
```

因此：

> “V4 本来满足 Option-B，只是 controller 抹掉 retreat”这一解释不成立。

更准确：

> **V4 nominal morphology 本身没有足够的 frozen mechanism margin。**

### B. endpoint 不是“还差一点到中心”，而是过冲且仍未稳定

fresh treatment terminal：

```text
raw progress ≈ 1.148342
absolute target offset ≈ 0.342582 m
lateral velocity ≈ 0.275049 m/s
```

target offset 从约 2.54 m 降到 0.34 m 只能说明 gross error 下降。

不能推出：

```text
stable target-center attraction established
```

更准确：

> treatment 已越过 target center，terminal 时仍未消除横向运动。

### C. planner rolling future terminal 与 scientific Primary80 terminal 不同

V4 stitching 存在 minimum rolling horizon。

因此：

```text
offline future-trajectory terminal feasibility
!=
realized Primary80 endpoint feasibility
```

过去大量 offline PASS 只能证明：

> reference trajectory 在自身 future horizon 内满足相应 feasibility。

不能替代：

> actual episode 在 7.9 s scientific endpoint 已 settling。

---

# 4. HLC construct 的论文定位

HLC Option-B 不应写成“人类所有变道犹豫”的充分表征。

更准确名称建议：

> **retreat-based hesitant lane-change morphology**

或：

> **HLC Option-B detectable-retreat morphology**

它测试的是：

```text
advance
→ detectable retreat
→ recommit
```

并同时要求：

- departure；
- retreat count；
- commitment delay；
- monotonic penalty。

现实“犹豫”还可以表现为：

- pause；
- late commit；
- slow probing；
- waiting；
- deceleration；
- abort without retreat。

因此论文只保留：

> 一个有明确机制定义的 controlled temporal morphology benchmark development case。

不做自然驾驶心理状态推断。

---

# 5. TSB 当前状态：保留，但不能直接视为“干净 residual benchmark”

当前 TSB development candidate：

```text
mechanism = 8/8
F_match = 8/8
safety = 8/8
measurement = 8/8
```

candidate 继续冻结，不调参、不重跑。

但 Astra review 发现：

> F_match PASS 不等于 low-order nuisance 已被消除。

DEV-CAL 的低阶差异具有稳定方向，而且部分非常接近 frozen caliper：

```text
mean speed delta ≈ 0.65 m/s
caliper ≈ 0.708 m/s

path length delta ≈ 5.14–5.18 m
caliper ≈ 5.384 m

mean abs accel delta ≈ 0.10 m/s²
caliper ≈ 0.118 m/s²
```

更关键的是：

```text
end_minus_start_speed
```

在现有 8 个 DEV-CAL pair 中 baseline 与 treatment 的观察范围完全分离。

因此当前只能说：

```text
TSB_F_MATCH_CONTRACT_PASS = SUPPORTED

LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED

TSB_IS_CLEAN_RESIDUAL_INFORMATION_TASK = NOT_ESTABLISHED
```

---

# 6. TSB 后续的核心问题已经改变

未来 TSB 不再只回答：

> generator 能不能稳定形成 brake-release-brake？

还要连续回答三层问题。

## TSB-Q1 — Mechanism qualification

在 fresh、outcome-unexposed logs 上：

```text
baseline exactly one brake phase
treatment exactly two brake phases
release fraction pass
second-peak ratio pass
F_match pass
safety pass
measurement valid
technical complete
```

是否稳定成立？

## TSB-Q2 — Construct discriminability

在 fresh data 上：

> predefined low-order / generic handcrafted features 是否已经足以可靠解决 baseline vs treatment 标签？

如果一个简单手工基线已经几乎完美解决标签，则：

```text
TSB_MECHANISM_BENCHMARK = MAY_BE_VALID

TSB_RBR_INCREMENTAL_TEST_VALUE = LOW / ABSENT
```

这不是 generator failure。

## TSB-Q3 — Incremental representation value

若任务仍有合理增量空间：

> RBR 是否在 predefined strong handcrafted baseline 之外提供有实际意义的 trajectory-level predictive value？

只有 Q1 + Q2 支持继续，才值得训练 RBR。

---

# 7. TSB applicability 必须重新论证

原 R1 的 `initial speed >= 2.0 m/s` 不能机械继承。

原因：

当前 R2 TSB candidate 的 braking schedule 已不同：

```text
baseline:
-1.45 m/s² × 1.8 s

treatment:
-2.4 × 0.9
+1.4 × 1.3
-2.4 × 0.9
```

原 2.0 m/s applicability 证明不自动覆盖当前 generator。

未来适用性必须在 fresh execution 前重新冻结，且只使用 pre-outcome 信息。

至少考虑：

- official initial speed；
- initial acceleration；
- controller state / steering state；
- road curvature；
- Primary80 全窗口 low-speed measurability；
- current generator 的 expected speed-loss envelope；
- `LOW_SPEED_ENDSTOP` frozen semantics；
- reference completeness；
- scenario safety eligibility；
- log/token independence；
- raw provenance closure。

禁止：

```text
根据未来 run outcome
→ 提高 speed floor
→ 删除失败 pair
```

未来 applicability 若受开发证据启发，必须明确标记：

```text
POST_DEVELOPMENT_SCOPE_AMENDMENT
```

---

# 8. Fresh TSB qualification：当前只冻结设计原则，不冻结最终样本数

Astra 提出 29/29 的理由：

> 若 29 个独立 pair 全部通过，则 pair-level success probability 的单侧 95% 下界超过 90%。

数学上成立。

但当前还不能把 `29` 直接冻结为最终样本量。

必须先区分两个不同 estimand。

## 8.1 Generator reliability estimand

若问题是：

> eligible population 中，TSB generator 的 pair-level qualification success probability 是否至少高于某个值？

则应使用正式 reliability / binomial design。

这种设计可以：

- 允许少量 failure；
- 用 CI 对 success probability 做 population-level 推断；
- 样本量由目标下界、容许 failure 数决定。

## 8.2 Scientific benchmark cohort estimand

若问题是：

> 我们要形成一个所有 pair 都机制有效、安全、F_match 的固定 benchmark cohort，用于后续 representation evaluation。

则：

```text
cohort member must pass all qualification gates
```

合理。

但 cohort 数量主要应由：

- future representation statistical power；
- independent log diversity；
- compute budget；

决定，而不是单独由 90% reliability 公式决定。

## 8.3 当前正式状态

```text
TSB_FRESH_PAIR_COUNT =
PROTOCOL_PARAMETER_NOT_YET_FROZEN

29_PAIRS =
SCIENTIFICALLY_MOTIVATED_PROPOSAL
NOT_FINAL_AUTHORIZATION
```

---

# 9. TSB fresh qualification 的治理原则

未来 Q qualification set 必须满足：

- independent log 为独立单位；
- historical R1/R2 development/outcome-exposed logs 全部排除；
- token/log uniqueness；
- baseline→treatment paired design；
- 两臂独立重置相同 initial state；
- 不共享可变 planner/controller state；
- roster selection 只使用 frozen pre-treatment applicability；
- scientific failure 不得改叫“不适用”后删除；
- infrastructure failure 与 scientific failure 分离；
- no identity replacement after scientific failure；
- no parameter update；
- no threshold change；
- no outcome-based applicability change；
- 全部 started / excluded / incomplete / failed / passed 必须报告。

TSB qualification 的 Primary gates 保持：

```text
measurement
baseline one-phase
treatment two-phase
release fraction
second-peak ratio
F_match
official safety
technical completeness
```

禁止把 HLC endpoint gate 临时移植到 TSB。

---

# 10. 强手工 baseline：必须在 RBR 之前冻结

未来不允许再使用：

```text
RBR significant
+
F_match PASS
=
Residual information proven
```

新的主问题应是：

> **RBR 是否在预定义 strong handcrafted trajectory representation 之外提供有实际意义的增量预测价值？**

建议分四层 baseline。

## 10.1 F-only

当前 TSB nuisance-control features。

作用：

> 判断最基础低阶摘要是否已经解决任务。

## 10.2 ego13

保持 Generation-1 的项目手工基线。

## 10.3 Strong generic handcrafted temporal baseline H

Astra 建议的 30D H 可作为候选，但当前不是自动冻结项。

H 的构造原则应是：

- generic；
- 不按观察到的两个 brake peak 对齐；
- 不读取 mechanism label；
- 固定时间箱；
- 固定 autocorrelation；
- braking-mass temporal moments；
- 低容量、可解释；
- 在 fresh scientific test 前完全冻结。

候选内容：

```text
原 F 四项
+
ego13
+
固定时间箱 longitudinal acceleration summaries
+
固定 lag autocorrelation
+
braking-mass temporal centroid / spread
```

H 的最终维数和去重规则在 protocol freeze 时确定。

H 必须明确称：

```text
DEVELOPMENT_INFORMED_HANDCRAFTED_CHALLENGER
```

不能包装成项目最初即存在的 baseline。

## 10.4 Mechanism-defining positive control O

TSB frozen mechanism variables：

```text
brake phase count
release fraction
second-peak ratio
validity indicator
```

属于：

```text
POSITIVE_CONTROL
```

不能放入主 baseline 后再要求 RBR 超过它。

因为 O 本身定义了 treatment label 的机制。

---

# 11. RBR Incremental Validity：未来新的 Primary representation question

若 TSB fresh qualification 成功，且 strong H 没有耗尽任务可分性，未来 RBR Primary question 应变成：

\[
oxed{
Does\ RBR\ add\ useful\ trajectory\ information
beyond\ predefined\ handcrafted\ summaries?
}
\]

而不是：

\[
RBR\ detects\ treatment?
\]

## 11.1 Primary comparison

候选主比较：

```text
H + fixed initial covariates
vs
H + fixed initial covariates + RBR
```

独立单位：

```text
log / pair
```

两个 arm 的 loss 先在 pair 内平均。

## 11.2 Primary risk metric

Astra建议 Brier risk difference 是合理 candidate：

\[
\Delta =
R(f_H)-R(f_{H+Z})
\]

但当前 `0.01` minimum useful improvement：

```text
DELTA_MIN = 0.01
```

仍只是 proposal。

在 protocol freeze 前必须结合：

- H baseline risk；
- scientific materiality；
- sample size；
- downstream BDD relevance；

再冻结。

## 11.3 Readout

Primary 使用：

- regularized logistic；
- H 可以预定义低自由度 nonlinear basis；
- Z 使用 linear readout；
- H+Z 使用同一 H basis + 64D linear term；
- hyperparameters 只在 readout development data 选择。

不能：

```text
RBR significant
H not significant
→ 宣布两者有显著差异
```

必须直接检验 paired risk improvement。

---

# 12. “Temporal residual information”措辞需要收紧

即使：

```text
H + RBR > H
```

也只能直接支持：

> **trajectory-level incremental predictive information beyond predefined handcrafted summaries**

它不自动证明：

> temporal ordering information

因为 RBR 可能利用：

- nonlinear amplitude；
- local shape；
- endpoint dynamics；
- higher-order trajectory structure。

若论文希望使用更强的：

> temporal ordering information

则建议增加一个预冻结 order-invariant raw-sequence baseline 或等价控制。

例如：

```text
same per-frame raw channels
+
permutation-invariant pooling
```

与 ordered temporal RBR 对比。

没有该控制时，论文主词保持：

```text
trajectory-level incremental information
```

而非：

```text
new temporal semantic discovery
```

---

# 13. Incremental validity 不能替代 BDD

这是当前 v2.0 相对于 Astra路线的关键补充。

本论文原始主线仍然是：

> planning-policy behavior drift detection。

因此未来 Stage RBR evaluation 必须有两层 gate。

## Gate 1 — Representation Incremental Validity

回答：

> RBR 是否在 H 之外提供有实际意义的 trajectory-level增量信息？

如果 Gate 1 fail：

```text
RBR_LEVEL2_UPGRADE = STOP
```

## Gate 2 — BDD Suitability

只有 Gate 1 通过后，才回答：

> frozen RBR + frozen BDD/MMD protocol 是否能够把该 representation 信息转化为可靠 release-level drift detection？

必须包括：

- frozen BDD statistic；
- representation-specific null；
- null calibration；
- FPR；
- paired / unpaired 设计中适用的 detection；
- confidence interval；
- 不得在 E test outcome 上选 kernel/readout。

因为：

```text
classification/readout gain
!=
BDD geometry suitability
```

这正是 Stage7L 后必须闭合的问题。

---

# 14. 数据角色重新简化

未来最少区分：

## U — Encoder Training / Validation

自然轨迹或其它允许的 training asset。

用途：

- train encoder；
- checkpoint selection；
- normalization。

不得包含最终 Q/D/E 相同 logs。

## Q — TSB Fresh Qualification

只用于：

- generator mechanism；
- F_match；
- safety；
- applicability；
- construct discriminability screening。

Q 一旦 outcome 暴露：

```text
OUTCOME_EXPOSED
```

不得成为最终科学 E。

## D — Readout / Handcrafted Challenger Development

用于：

- H finalization；
- scaler；
- readout；
- calibration；
- variance / power estimation。

为了控制 simulation 成本，允许研究：

> Q 在 generator qualification 完成后降级为 outcome-exposed D。

如果采用此方案，必须在 Q 运行前预注册：

```text
Q_AFTER_QUALIFICATION_ROLE =
READOUT_DEVELOPMENT_ONLY
NOT_FINAL_TEST
```

这样可以避免为了 D 再开一整套不必要 rollout。

## E — Locked Scientific Test

唯一用于：

- formal incremental-validity test；
- frozen BDD qualification。

E 在 encoder/readout/statistics 全冻结前不得解盲。

---

# 15. RBR 最小训练方案：现在只冻结原则，不冻结网络细节

目前不授权训练。

如果未来 TSB fresh qualification + construct value gate 通过，才冻结一个最小 architecture。

## 15.1 必须保留的 Gen-1 教训

禁止再次：

```text
Want Z64 to exceed ego13
while training latent geometry to mimic ego13/F distances
```

因此：

- handcrafted semantics 只用于 post-hoc frozen probe；
- 不把 handcrafted feature distance 作为 latent metric target；
- 不把 ego13 similarity matrix 作为 latent geometry supervision；
- 不用 TSB label 选 checkpoint；
- 不用 future E outcome 选 architecture。

## 15.2 最小 architecture 原则

候选可为简单 TCN，但目前：

```text
TCN_6_DILATION =
CANDIDATE
NOT_FROZEN
```

主方向：

```text
ego raw temporal sequence
+
validity mask
+
time-aware temporal encoder
+
mask-aware pooling
→ shared 64D
```

TSB-only primary 不强制 context encoder。

## 15.3 一个 architecture family

未来若授权训练，建议：

```text
one primary architecture family
three fixed seeds
fixed architecture-search budget
no TSB-outcome architecture rescue
```

checkpoint 只按 U-validation 的 self-supervised / raw-sequence objective选择。

---

# 16. RBR Training Authorization Gate

即使 TSB Q 全部通过，也不能自动训练 RBR。

训练至少要求：

```text
TSB_FRESH_QUALIFICATION = PASS

TSB_APPLICABILITY = FROZEN

H_STRONG_HANDCRAFTED_BASELINE = FROZEN

INCREMENTAL_VALIDITY_PROTOCOL = FROZEN

BDD_SECONDARY_GATE = FROZEN

U/Q/D/E_FIREWALL = FROZEN

RBR_ARCHITECTURE_SEARCH_BUDGET = FROZEN

RBR_TEST_E = UNEXPOSED
```

还需要一个 construct-value gate：

> strong H 不能已经把任务解决到几乎不存在预定义的 minimum useful improvement space。

若 H-only 已接近 perfect：

```text
RBR_INCREMENTAL_TEST_SPACE =
INSUFFICIENT

RBR_TRAINING =
NOT_REQUIRED_FOR_THIS_TSB_TASK
```

这不是 RBR failure。

---

# 17. 论文 claim 分层

## Level 1 — 当前保证完成目标

### 最低证据

- Stage6/7/7L frozen evidence；
- R0 diagnostic；
- R1 residual benchmark failure；
- R2 planner→controller transfer analysis；
- HLC architecture-development negative chain；
- TSB development positive candidate；
- clear data/governance boundaries。

### 最强可辩护表述

> 提出并实证审查了一套面向 closed-loop planning policy 的行为漂移评估方法，将 intervention intent、realized behavior、representation content、measurement/readout 与 statistical detection 分离，并系统揭示 planner intervention、closed-loop transfer、nuisance matching 和 representation measurement 的多类失效模式。

### 禁止

- general RBR validated；
- HLC scientifically impossible；
- learned representation universally superior；
- residual behavior cannot be described by handcrafted features。

### 当前状态

```text
LEVEL_1 =
SUPPORTED_AS_DISSERTATION_BASE_TARGET
```

## Level 2 — 有界升级目标

要求：

- fresh TSB qualification pass；
- strong H baseline protocol；
- incremental RBR Gate 1 pass；
- frozen BDD Gate 2有合格结果；
- data firewall完整。

允许写：

> 在预定义 TSB 适用域上，RBR 相对于指定 handcrafted trajectory baseline 提供额外 trajectory-level 信息，并在冻结 BDD 协议下表现出相应 drift-detection value。

禁止：

- universal residual representation；
- interaction representation；
- lateral residual generalization；
- beyond all handcrafted features。

## Level 3 — 当前不承诺

多个 independent residual families + broader RBR qualification。

当前：

```text
LEVEL_3 =
FUTURE_WORK / OPTIONAL_EXTENSION
```

不作为博士毕业闭环前提。

---

# 18. 当前推荐路线：Route A

正式推荐：

```text
CLOSE HLC CURRENT BRANCH

↓

ONE BOUNDED TSB QUALIFICATION OPPORTUNITY

↓

ONLY IF THE TASK STILL HAS INCREMENTAL VALUE SPACE:
ONE BOUNDED RBR TRAINING/EVALUATION PROGRAM

↓

THESIS CLOSURE
```

不推荐 Route B（立即开发第二 residual family）或 Route C（HLC predictive optimization V5）进入当前博士范围。

---

# 19. 最终未来阶段：最多四个

## Stage S1 — Scope & TSB Protocol Freeze

### 科学问题

TSB candidate 是否有合理、前瞻、可验证的适用域？  
该任务是否值得作为 RBR incremental-value benchmark？

### 允许

- zero-run protocol work；
- applicability derivation；
- schema simplification；
- H definition；
- sample-size design；
- data firewall design；
- BDD Gate 2 design。

### 禁止

- simulator；
- roster selection；
- TSB parameter changes；
- HLC V5；
- RBR training。

### 输出

- HLC closure addendum；
- TSB applicability contract；
- TSB qualification SAP；
- sample-size decision；
- H challenger contract；
- O positive-control contract；
- U/Q/D/E firewall；
- incremental-validity SAP；
- BDD secondary SAP；
- simplified canonical schema。

### Gate

只有所有定义闭合，才申请 Stage S2 simulation authorization。

### Failure

若 TSB applicability 或 construct value 无法合理闭合：

```text
STOP RESIDUAL/RBR PATH
PROCEED LEVEL_1 THESIS
```

---

## Stage S2 — Fresh TSB Qualification & Handcrafted Challenge

### 科学问题

冻结 TSB candidate 能否在 fresh logs 上稳定成立？  
strong handcrafted H 是否已经基本解决任务？

### 允许

- 明确授权后的 TSB qualification simulation；
- frozen H/F/ego13 analysis；
- no RBR encoder training。

### 禁止

- generator tuning；
- failed identity replacement；
- threshold change；
- outcome-driven applicability revision；
- RBR results。

### 输出

- qualification ledger；
- mechanism/F/safety/applicability results；
- H/F/ego13 discriminability；
- power / sample-size estimates for possible RBR study。

### Gate

进入 S3 必须同时满足：

```text
TSB qualification = PASS
task has nontrivial incremental-value space
data budget sufficient
```

### Failure

任一失败：

```text
STOP RESIDUAL/RBR EXPERIMENTAL PATH
PROCEED LEVEL_1 THESIS
```

---

## Stage S3 — One-shot Restricted RBR Qualification

### 科学问题

RBR 是否提供 predefined H 之外的有用 trajectory-level增量信息？  
该信息能否被 frozen BDD 有效读出？

### 允许

- one primary architecture family；
- fixed seeds；
- U/D only for development；
- E only after full freeze。

### 禁止

- architecture rescue after E；
- best-seed picking；
- E-based kernel/readout tuning；
- new residual family；
- HLC reopen。

### Primary Gate 1

```text
Incremental validity beyond H
```

### Gate 2

```text
Frozen BDD suitability / null calibration
```

### 输出

- semantic retention；
- primary incremental-risk comparison；
- BDD result；
- all seeds；
- null/calibration；
- negative or positive conclusion。

### Failure

无论：

```text
PASS
FAIL
INCONCLUSIVE
```

都进入 S4。

不允许第二轮 RBR rescue。

---

## Stage S4 — Thesis Evidence Closure

### 科学问题

现有证据能够支持哪个 claim level？

### 允许

- synthesis；
- thesis writing；
- figures；
- reproducibility package；
- limitation / future work。

### 禁止

```text
because one positive result is missing
→ reopen HLC
→ invent new family
→ retrain another RBR
```

### 最终输出

默认：

```text
LEVEL_1 THESIS
```

若 S3 满足所有 Level2 条件：

```text
UPGRADE TO LEVEL_2
```

---

# 20. Governance 简化

B1 schema mismatch 已证明：

> governance complexity 本身会产生 infrastructure risk。

未来不再复制 B0/B0.1/B0.2/B1/B1.1 的层层 wrapper。

建议统一：

```text
one canonical schema
one executor
one passive recorder
one analyzer
one primary manifest
one stage authorization
```

仍保留：

```text
NOT_RUN
TECHNICAL_INCOMPLETE
MEASUREMENT_INVALID
SCIENTIFIC_FAIL
PASS
```

不能压缩成一个 boolean。

必须有真实 schema-faithful end-to-end tests，覆盖 execution → serializer → metrics → analyzer，而不是 mock 掉关键路径。

---

# 21. 永久冻结内容

以下不得回写：

- Stage6/7/7L frozen evidence；
- R0 v1.0 formal protocol；
- R1 official outcomes；
- R2-A/B/BH/BI/BJ historical outcomes；
- HLC frozen mechanism；
- HLC negative canary；
- B1 infrastructure state；
- B1.1 recovery state；
- TSB current development candidate parameters/results；
- historical thresholds/calipers；
- protected CSV；
- outcome-exposed identities；
- original combined-family goal未建立的事实。

---

# 22. 当前明确禁止的科研行为

禁止：

```text
HLC V5
remaining HLC canary runs
TSB parameter retuning
tightening TSB calipers because low-order features separate
dropping discriminative F features to create RBR advantage
choosing only difficult scenarios
changing labels
adding noise to handcrafted baselines
selecting best RBR seed
training on E
tuning BDD from E
reusing outcome-exposed identities
rewriting old PASS/FAIL
calling HLC scientifically impossible
calling TSB F_match "low-order features eliminated"
calling classification gain "BDD success"
calling H+RBR gain "new temporal semantics" without an ordering control
```

---

# 23. 当前允许的论文结论边界

## 已支持

- planner-intent 与 realized closed-loop mechanism 必须分离；
- nuisance-control PASS 不等于 label information 被消除；
- representation information retention 与 BDD readout capability 必须分离；
- HLC current branch did not converge；
- TSB current candidate has positive development evidence；
- current residual benchmark work contains post-development scope amendments；
- prospective confirmation 与 development evidence 必须分离。

## 尚未支持

- RBR-64 superior to ego13；
- RBR captures novel temporal semantics；
- TSB is a clean residual-information benchmark；
- HLC impossible；
- interaction residual capability；
- broad residual behavior generalization；
- two families sufficient for universal representation；
- classification gain guarantees BDD gain。

---

# 24. 当前立即执行事项

现在：

```text
DO NOT RUN SIMULATION
DO NOT SELECT TSB ROSTER
DO NOT TRAIN RBR
```

下一步唯一正式任务：

> **Stage S1 — Scope & TSB Protocol Freeze**

需要把本 v2.0 的候选设计正式化为：

- HLC closure addendum；
- TSB applicability；
- TSB fresh qualification estimand；
- final sample-size rationale；
- H / O controls；
- incremental-validity SAP；
- BDD Gate 2 SAP；
- U/Q/D/E firewall；
- canonical schema；
- execution simplification；
- stop conditions。

完成 S1 后，Scientific Owner 再决定：

```text
TSB_SIMULATION_AUTHORIZED?
```

而不是由 Work 自动继续。

---

# 25. 最终研究主线

```text
Generation-1
        ↓
controlled / release-level BDD evidence
        ↓
Stage7L learned-vs-handcrafted divergence
        ↓
R0 representation / measurement diagnosis
        ↓
R1 prospective residual benchmark
        ↓
planner intent ≠ realized behavior
        ↓
R2 controller-transfer / generator development
        ↓
HLC negative development
+
TSB positive development candidate
        ↓
TSB construct-value and fresh qualification
        ↓
if worthwhile:
one-shot RBR incremental qualification
+
BDD suitability
        ↓
thesis evidence closure
```

最终论文的重点不是证明：

> “64D 一定比 13D 强。”

而是建立：

> **一个可证伪、可审计的 closed-loop behavior evaluation methodology，能够区分机制实现、低阶可分性、representation 增量信息与 BDD detection，并诚实保留正结果、负结果和不可判定结果。**

---

`RBR64_STAGE_R_ROADMAP_V2_DRAFT_FOR_OWNER_FREEZE`
