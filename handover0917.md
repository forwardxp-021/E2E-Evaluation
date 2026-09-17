# E2E-Evaluation 博士研究项目权威交接

> **状态：`CURRENT_STAGE_S1_FROZEN_S2_PREFLIGHT_NEXT`**  
> 更新时间：2026-09-17（Asia/Shanghai）  
> 仓库：`forwardxp-021/E2E-Evaluation`  
> 当前研发分支：`20260825_stageR_new`  
> 当前已核对远端 HEAD：`9764ea46a91b6495ec9063616c0ec1ce6b1045f3`  
> 当前阶段：**Stage S1 已完成并正式冻结；下一步是 S2-Preflight（metadata-only eligibility census + production execution binding）**  
> 当前核心状态：**S2 simulation 尚未授权；RBR 正式训练尚未授权；E 构造/访问尚未授权。**  
> 本文件替代 2026-09-05 的旧 handover 作为当前唯一实时入口；旧文件仅保留为历史快照。

---

# 0. 给下一个 conversation / Work / Astra session 的启动指令

## 0.1 第一原则

当前已经不再处于“是否继续 HLC / 是否采用 TSB-only / 是否把 H+RBR 当 Primary”的路线争论阶段。

经过三轮 Astra 独立科学审查、Scientific Owner 复核以及 v2.2 路线冻结，目前已形成明确共识：

```text
BDD = PRIMARY

H+RBR incremental predictive value = SECONDARY

HLC current branch =
CLOSED_BY_SCOPE_AFTER_ENGINEERING_NONCONVERGENCE

TSB =
FROZEN_DEVELOPMENT_CANDIDATE_PENDING_FRESH_QUALIFICATION

Residual =
RELATIVE_TO_FROZEN_PROJECT_HANDCRAFTED_BASELINE F0_project (= ego13)

RBR_TRAINING = NOT_AUTHORIZED

S2_SIMULATION = NOT_AUTHORIZED

NEXT =
S2-PREFLIGHT
metadata-only eligibility census
+
production execution binding
```

下一个 session 不应重新发散大方向，不应重新讨论 HLC V5，也不应提前设计/训练 RBR。

## 0.2 当前最重要的 governing documents

优先阅读顺序：

1. `AGENTS.md`
2. 本 `handover.md`
3. `RBR-64_博士研究总体方案_v2.2_S1正式入口版.md`
4. `docs/stageR/s1/S1_Scope_and_Claim_Freeze_Draft_v0.1.md`
5. `docs/stageR/s1/S1_TSB_Applicability_Contract_Draft_v0.1.md`
6. `docs/stageR/s1/S1_TSB_Fresh_Qualification_Protocol_Draft_v0.1.md`
7. `docs/stageR/s1/S1_Handcrafted_Challenger_Contract_Draft_v0.1.md`
8. `docs/stageR/s1/S1_BDD_Statistical_Analysis_Plan_Draft_v0.1.md`
9. `docs/stageR/s1/S1_Secondary_Diagnostics_SAP_Draft_v0.1.md`
10. `docs/stageR/s1/S1_Data_Firewall_Draft_v0.1.json`
11. `docs/stageR/s1/S1_Canonical_Schema_Draft_v0.1.json`
12. `docs/stageR/s1/S1_Protocol_Design_Report_v0.1.md`
13. `docs/stageR/s1/S1_Protocol_Design_Manifest_v0.1.json`
14. R0 / R1 / R2 历史冻结协议与报告
15. Stage6 / Stage7 / Stage7L 冻结 BDD 证据

启动时先核对：

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
git rev-parse origin/20260825_stageR_new
```

如果 HEAD 不是 `9764ea46a91b6495ec9063616c0ec1ce6b1045f3` 或其后续明确授权提交，必须先确认最新状态。

## 0.3 当前默认权限

除非 Scientific Owner 明确授权：

```text
SIMULATION = 0
RUNNER_RUN = 0
NEW_SCIENTIFIC_OUTCOME_EXPOSURE = 0

S2_EXECUTION = NOT_AUTHORIZED
Q20_EXECUTION = NOT_AUTHORIZED
RBR_TRAINING = NOT_AUTHORIZED
E_CONSTRUCTION = NOT_AUTHORIZED
E_ACCESS = NOT_AUTHORIZED

HLC_V5 = NOT_AUTHORIZED
REMAINING_HLC_RUNS = NOT_AUTHORIZED

TSB_PARAMETER_CHANGE = FORBIDDEN
THRESHOLD_CHANGE = FORBIDDEN
OUTCOME_EXPOSED_IDENTITY_REUSE = FORBIDDEN
```

允许的下一步仅是：

```text
S2-PREFLIGHT
metadata-only
zero-simulation
```

---

# 1. 一分钟项目摘要

## 1.1 论文当前真正的主线

论文不再以：

> “做一个 64D 表征并证明它比 13D 手工特征信息更多”

作为核心。

当前主线已经收敛为：

> **如何建立一套面向 closed-loop planning-policy software release 的行为漂移评价方法，区分干预意图、实际行为机制、表征敏感性与分布检测效用，并检验 learned trajectory representation 是否能在固定 handcrafted monitoring contract 下提供更有效的 behavior drift discovery。**

核心 Primary：

> **在相同误报约束、相同样本预算、相同 operating condition 下，RBR-BDD 是否比预注册 handcrafted challenger BDD 提供更好的检测效用。**

不是比较 raw MMD²。

## 1.2 应用愿景

工程应用愿景：

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

典型真实动机是“变道犹豫”：现有 KPI 可能都正常，但实际试驾才发现新版本每次变道明显犹豫，随后团队才新增 hesitation-related KPI。模型的理想价值是在 KPI 尚未存在时，先提示“这个版本的行为和参考版本显著不同”。

但必须严格区分：

```text
APPLICATION_VISION
!=
CURRENTLY_PROVEN_CLAIM
```

当前 Route A 主要实证验证：

```text
closed-loop mechanism validity
+
representation/BDD detection utility
```

当前并未证明：

- 能发现任意未知 behavior family；
- 能自动命名“犹豫变道”；
- 人类确认闭环已经完成正式实验；
- BDD alarm 一定意味着退化或 release-blocking。

---

# 2. Astra 三轮审查后的最终共识

## 2.1 第一轮 Astra 的核心贡献

发现 HLC V4 的关键问题：

1. V4 nominal morphology 即使 ideal tracking，也不能满足冻结 monotonic gate；
2. treatment terminal 不是“离中心还差 0.34 m”，而是已经 overshoot target center 且仍有 lateral motion；
3. rolling planner future terminal 不等于 scientific Primary80 realized terminal；
4. 一个 fresh canary 足以否决当前 V4 candidate、取消剩余 14 runs，但不能证明 HLC scientific construct 不可能；
5. TSB 的 F_match PASS 不等于 low-order nuisance 已消除。

## 2.2 第二轮 Astra 的核心贡献

提出 Route A：

```text
close HLC
→ bounded TSB qualification
→ if worthwhile, one bounded RBR qualification
→ thesis closure
```

并提出 strong handcrafted challenger H、incremental validity、Level 1 / Level 2 / Level 3 claim。

Scientific Owner 后续修正：

- 29/29 不应自动成为 sample-size；
- H+RBR classifier gain 不能替代 BDD；
- BDD 必须保留为 thesis 主任务；
- temporal-order claim 不能过强；
- H 是 development-informed challenger，不是标准答案。

## 2.3 第三轮 Astra 的最终修正

Astra接受 Scientific Owner 挑战并最终确认：

```text
BDD = PRIMARY

H+RBR incremental predictive-risk = SECONDARY
```

撤回：

```text
H classifier near-perfect
→ RBR no longer worth studying
```

因为：

```text
classification/readout utility
!=
distributional BDD geometry / finite-sample detection utility
```

RBR 即使没有增加“新的标签信息”，也可能把已有信息组织成更适合 BDD 的 representation geometry。

但“更好的 geometry”本身只是兼容解释，除非单独验证，不能直接写成已证明根因。

---

# 3. 论文最终科学定位

## 3.1 Level 1 — Dissertation base contribution

当前不依赖未来 RBR 阳性结果即可成立的主张：

> **闭环自动驾驶行为漂移评价必须区分干预意图、实际行为机制、表征可读性与分布测量效用；前一层成立不能保证后一层有效。**

已有证据来源：

- Stage6 / Stage7 / Stage7L；
- R0 representation/measurement diagnosis；
- R1 residual benchmark failure；
- R2 controller-transfer analysis；
- HLC negative development chain；
- TSB frozen development candidate；
- paired / unpaired BDD distinction；
- benchmark / measurement / governance failure analysis。

注意：Level 1 的科学贡献不是“做了很多 wrapper / manifest / 失败了很多次”，而是不同评价层之间的不等价关系及其前瞻反例与诊断方法。

## 3.2 Level 2 — Model-based BDD utility

若未来成功：

> **在预定义 TSB 适用域和冻结 evaluation contract 下，RBR 在相同 FPR、样本预算与 operating condition 下，相对于预注册 handcrafted challenger，提高了对已确认 closed-loop behavior morphology drift 的 BDD 检测效用。**

如果 Secondary H vs H+RBR risk 也支持，才可附加：

> 在指定 readout protocol 下观察到增量预测价值。

只有真正完成合格 unpaired E，才能写：

```text
release-emulation
```

不能直接写：

```text
real production release validation
```

## 3.3 Level 3

多个 independent residual families、interaction/context generalization、真实 unknown-family discovery、人类确认实验：

```text
FUTURE WORK
```

当前博士不承诺。

---

# 4. Residual 的最终定义

不要再写：

> residual = handcrafted 永远无法描述的信息。

正式概念：

> **Residual-to-F0_project behavior：在指定适用域、测量窗口和预定义 project handcrafted representation F0_project 下，仍存在的、由独立 closed-loop mechanism measurement 确认的 behavior structure difference。**

当前：

```text
F0_project = ego13
```

它是项目既有的 predefined/routine handcrafted behavior representation。

它不是公司量产 KPI inventory 的声明。

注意角色必须严格分开：

```text
F_match =
4D TSB matching descriptors

F0_project =
ego13

H =
development-informed strong handcrafted challenger

O =
mechanism positive control
```

---

# 5. Handcrafted comparison contract

## 5.1 F_match

仅用于 TSB nuisance/matching contract：

```text
mean_speed
end_minus_start_speed
path_length
mean_abs_accel
```

F_match PASS：

```text
!=
low-order nuisance eliminated
```

DEV-CAL 8 pair 中四项 delta 均方向一致，尤其 `end_minus_start_speed` 可高度区分 arm，因此不能把当前 TSB 称为“clean residual task”。

## 5.2 F0_project = ego13

项目历史标准 handcrafted representation。

用途：

- predefined project baseline；
- residual 定义参考；
- secondary historical baseline。

## 5.3 H — strong development-informed challenger

未来 Primary handcrafted BDD comparator。

当前冻结约 30D，允许：

- F summaries；
- ego13；
- fixed temporal acceleration bins；
- fixed-lag autocorrelation；
- braking-mass temporal centroid/spread。

必须承认：

```text
H = DEVELOPMENT_INFORMED_HANDCRAFTED_CHALLENGER
```

禁止 E 结果出来后：

- 新增针对 RBR 发现模式的 feature；
- 删除 H 中表现太强的 feature；
- 换弱 readout；
- 修改 normalization。

## 5.4 O — mechanism positive control

例如：

```text
brake_phase_count
interstage_release_fraction
second_brake_peak_ratio
validity flags
```

用途：

> 证明 TSB treatment 机制确实实现。

O 不进入 Primary RBR vs H 竞赛。

---

# 6. RBR 模型训练架构边界（非常重要）

这是后来多次 conversation 混乱的地方，必须严格记住。

## 6.1 正确概念架构

```text
trajectory / context X
        ↓
temporal encoder / GRU
        ↓
shared z64
       /   \
      /     \
Human Semantic Head    Generic Representation / Temporal Head(s)
      ↓                         ↓
   L_sem                    L_repr / L_ssl
      \                         /
       └──── backprop ─────────┘
                ↓
             encoder
```

形式化：

```text
z64 = Encoder(X)
semantic prediction = h_sem(z64)

L =
lambda_sem * L_sem
+
lambda_repr * L_repr
+
optional pre-frozen auxiliary terms
```

## 6.2 Human Semantic Head 的职责

必须：

```text
backpropagate through z64 into encoder
```

目的：

> **明确强制 z64 保留已知人工驾驶语义，而不是让模型完全自己“乱学”。**

人工语义 targets 只来自 U-authorized training data。

禁止：

- TSB treatment label；
- TSB mechanism label；
- O labels；
- Q outcome；
- D outcome；
- E outcome；
- ego13 distance-matrix alignment；
- handcrafted geometry imitation loss。

## 6.3 Human Semantic Head 和 post-hoc semantic probe 不是一回事

```text
Human Semantic Head
= training-time supervision
= directly shapes z64

Post-hoc semantic probe
= Secondary diagnostic
= evaluates what z64 retained
```

不要再混。

## 6.4 BDD 使用什么

Primary BDD：

```text
uses frozen z64 directly
```

不是：

```text
semantic-head prediction
```

因此 semantic head 保证 known semantics retained，但 BDD 仍在 shared learned representation space 上运行。

## 6.5 当前尚未冻结的模型细节

S1 只冻结架构原则，不冻结：

- exact encoder layers；
- GRU hidden size；
- TCN / Transformer 是否替代 GRU；
- semantic head exact MLP；
- lambda weights；
- exact SSL / reconstruction / temporal objective；
- number of seeds beyond future bounded protocol。

这些属于未来 S3 pre-training protocol。

当前：

```text
RBR_TRAINING = NOT_AUTHORIZED
```

---

# 7. BDD Primary 的冻结合同

## 7.1 Primary estimand

当前冻结：

```text
ΔBDD =
P(alarm_RBR)
-
P(alarm_H)
```

在：

```text
nominal FPR α = .05
batch size m = 20 independent logs per release arm
drift fraction π = .50
minimum useful gain δ* = .10
```

下比较。

## 7.2 π=.50 的语义

必须写：

```text
SYNTHETIC MODERATE-DRIFT BENCHMARK
```

它不是：

- real-fleet prevalence；
- production drift incidence；
- ODD population estimate；
- 50%生产场景都会变化的假设。

它只是为了公平比较 representation sensitivity 而 prospectively frozen 的 operating point。

## 7.3 Primary 成功规则

设 log-aware 95% CI：

```text
[L, U]
```

Success：

```text
L > 0
AND
point estimate ΔBDD >= .10
```

Fail：

```text
U < .10
```

否则：

```text
INCONCLUSIVE
```

不要求：

```text
L > .10
```

## 7.4 FPR gate

两个 pipelines：

```text
RBR
H
```

均要求 independent FPR evaluation。

Nominal：

```text
α = .05
```

一侧 simultaneous 95% upper bound：

```text
<= .075
```

阈值不能在 E 上重新调。

## 7.5 不允许比较 raw MMD²

不同 representation 的 kernel scale / geometry 不同。

因此：

```text
raw MMD²_RBR > raw MMD²_H
```

不能作为 representation superiority 结论。

比较的是：

- detection probability；
- fixed-FPR utility；
- sample budget；
- CI。

---

# 8. Paired / Unpaired 的最终角色

## 8.1 Paired

同一 scenario 的 baseline/treatment。

用于：

- controlled mechanism attribution；
- controlled sensitivity；
- representation diagnosis。

只能支持：

```text
same-scenario controlled evidence
```

## 8.2 Unpaired

不同 logs / scenarios 组成 reference / target releases。

优先用于：

```text
release emulation
```

未来若容量足够：

```text
UNPAIRED_RELEASE_EMULATION_BDD = PRIMARY
```

如果容量不够，必须在 E 解盲前预定义：

```text
PAIRED_PROSPECTIVE_BDD_ONLY
```

且主张同步收缩。

不能实验后把 paired success 改写成 release-level monitoring validation。

---

# 9. BDD calibration / uncertainty 的关键规则

当前 S1 SAP 已冻结：

- unbiased quadratic-time MMD²；
- RBF kernel；
- representation-specific standardization；
- representation-specific median-bandwidth rule；
- 相同数学 tuning budget；
- zero supervised Primary kernel search；
- independent A/A calibration；
- independent E_AA evaluation；
- independent E_AB evaluation；
- same release draws for H/RBR；
- 10,000 inner Monte Carlo draws只是数值精度，不是独立 release 样本；
- 2,000 outer source-log cluster bootstrap 用于 source-log uncertainty；
- calibration uncertainty 在 outer bootstrap 中传播；
- 如果 log-level interval support 不足：

```text
PRIMARY = INCONCLUSIVE
```

禁止：

```text
Monte Carlo n=10000
→ 当成10000个独立release
```

---

# 10. S1 已完成并冻结

## 10.1 S1 final owner state

```text
S1_PROTOCOL_FROZEN = YES

S1_SCOPE_AND_CLAIMS = FROZEN
S1_TSB_APPLICABILITY = FROZEN
S1_Q_QUALIFICATION_PROTOCOL = FROZEN
S1_HANDCRAFTED_COMPARISON = FROZEN
S1_PRIMARY_BDD_SAP = FROZEN
S1_SECONDARY_DIAGNOSTICS = FROZEN
S1_DATA_FIREWALL = FROZEN
S1_RBR_TRAINING_ARCHITECTURE_BOUNDARY =
FROZEN_AT_PRINCIPLE_LEVEL
```

仍然：

```text
S2_SIMULATION = NOT_AUTHORIZED
RBR_TRAINING = NOT_AUTHORIZED
E_ACCESS = NOT_AUTHORIZED
```

---

# 11. TSB applicability 已冻结

原 R1：

```text
initial_speed >= 2.0 m/s
```

只能保留为历史 provenance。

当前 R2 TSB candidate：

baseline：

```text
-1.45 m/s² × 1.8 s
```

treatment：

```text
-2.4 × 0.9
+1.4 × 1.3
-2.4 × 0.9
```

精确离散 nominal cumulative loss：

```text
baseline = 2.61 m/s
treatment max = 2.50 m/s
```

LOW_SPEED_ENDSTOP measurement floor：

```text
1.0 m/s
```

因此冻结 prospective nominal screen：

```text
initial_speed >= 3.61 m/s
```

正式标签：

```text
POST_DEVELOPMENT_PROSPECTIVE_SCOPE_AMENDMENT
NOMINAL_MEASURABILITY_SCREEN
NOT_A_CLOSED_LOOP_GUARANTEE
```

如果 future selected scenario 实际闭环仍出现 LOW_SPEED_ENDSTOP：

```text
SCIENTIFIC_FAILURE
```

禁止事后提高 floor 把失败样本排除。

---

# 12. Q qualification 已冻结

## 12.1 Q target

```text
Q_TARGET = 20 independent logs/pairs
MAX_RUNS = 40
```

每 pair：

```text
baseline
treatment
```

## 12.2 Q12

```text
Q_MINIMUM_OPTION = 12
```

只允许：

- metadata-only census 后；
- 任何 Q rollout 前；
- 任何 Q scientific outcome 暴露前；
- 由 Scientific Owner 单独决定。

一旦 Q20 execution 开始：

```text
cannot redefine to Q12
```

12/12 early success：

```text
!= qualification success
```

## 12.3 whole-roster rule

```text
ALL_PAIRS_JOINT_PASS
FIRST_SCIENTIFIC_FAILURE_STOP
NO_SURVIVOR_SELECTION
NO_REPLACEMENT
```

第一个 scientific failure：

```text
Q FAIL
stop remaining runs
```

这不是 generator population reliability inference。

它只是 frozen benchmark construction qualification contract。

## 12.4 scientific vs infrastructure failure

Scientific failure 包括：

- LOW_SPEED_ENDSTOP；
- baseline mechanism fail；
- treatment mechanism fail；
- F_match fail；
- official safety fail。

Infrastructure failure：

- schema mismatch；
- corrupt/missing artifacts；
- runner failure；
- callback failure；
- incomplete technical lifecycle。

Infrastructure failure：

```text
Q = INCOMPLETE / INFRASTRUCTURE_STOP
```

不自动算 scientific negative。

但也不允许自动 retry。

---

# 13. U / Q / D / E Data Firewall

## U

```text
encoder training / validation only
```

Encoder 的 architecture、training objective、checkpoint、seed-handling rule 只能依据 U。

禁止用 Q/D/E 选 encoder。

## Q

Fresh TSB qualification。

只用于：

- mechanism；
- F_match；
- safety；
- applicability；
- technical completeness。

## Q→D

必须在 Q execution 前预注册。

当前已预注册：

```text
if Q PASS:
Q becomes outcome-exposed D
```

用于：

- H scaler；
- bounded readout；
- kernel/bandwidth development；
- calibration development；
- variance/power estimation。

Q 不能重新成为 E。

## E

最终 locked scientific test。

唯一用于：

```text
Primary BDD comparison
+
pre-registered Secondary diagnostics
```

在以下全部冻结前不得 materialize / access：

- encoder；
- H；
- scaler；
- kernel；
- bandwidth；
- null；
- calibration；
- operating point；
- Primary metric；
- E sample size；
- analysis plan。

---

# 14. 当前下一步：S2-Preflight

当前不是直接执行 Q20。

下一阶段：

> **S2-PRE — Metadata-Only Eligibility Census + Production Execution Binding**

目标：

1. materialize complete eligibility census；
2. 检查真正 independent unit；
3. 验证 Q20 是否有至少 20 个 fresh eligible independent logs；
4. 只在 capacity 足够时 materialize Q20 NOT_RUN roster；
5. bind exact production executor/runtime/config/analyzer SHA；
6. freeze full arm reset contract；
7. freeze precontext identity contract；
8. prepare inactive Q20 budget authorization；
9. zero-run integration tests；
10. 最后等待 Scientific Owner 单独授权 S2。

仍然：

```text
simulation = 0
runner.run = 0
scientific outcome exposure = 0
```

---

# 15. S2-Preflight eligibility census

未来 census 只能使用 pre-outcome metadata。

必须审计：

- source universe；
- log ID；
- scenario token；
- session/source grouping；
- initial speed；
- route/reference completeness；
- timestamp metadata；
- provenance；
- historical exposure；
- U/Q/D/E reservation conflicts；
- engineering reservation；
- initial_speed >= 3.61；
- forward route support；
- source independence。

禁止使用：

- realized TSB mechanism；
- treatment trace；
- safety outcome；
- F_match outcome；
- LOW_SPEED_ENDSTOP outcome。

---

# 16. Q20 deterministic selection

若 >=20 eligible independent units：

冻结 selection：

```text
salt = S1_TSB_Q_v0.1

sort by SHA256(salt | log_id)

within log:
SHA256(salt | scenario_token)

take first eligible token per independent log
take first 20 independent units
```

仅 materialize roster：

```text
execution status = NOT_RUN
exposure = UNEXPOSED
```

不能执行。

如果 capacity 只有 12–19：

```text
Q20_CAPACITY_NOT_AVAILABLE
```

先停，Owner 决定是否启用 Q12 fallback。

---

# 17. S2-Preflight production execution binding

未来 S2 必须绑定唯一生产链：

```text
frozen stage authorization
→ one executor
→ official nuPlan lifecycle
→ frozen TSB planner/generator
→ actual controller/LQR
→ passive recorder
→ canonical serializer
→ production analyzer/dispatcher
→ official safety artifacts
→ one primary result manifest
```

必须绑定 SHA：

- TSB candidate；
- planner/generator；
- runner entrypoint；
- scenario builder；
- simulation setup；
- controller；
- tracker；
- motion model；
- passive actual-LQR recorder；
- serializer；
- analyzer；
- mechanism evaluator；
- F_match evaluator；
- safety adapter；
- canonical metric parser；
- stage authorization；
- output root；
- budget ledger。

Unknown 必须写：

```text
UNKNOWN
AMBIGUOUS
BLOCKED
```

不能猜。

---

# 18. Full reset contract

`controller.reset()` 本身不够。

baseline / treatment 每一 arm 必须 fresh construct 或 proven reset：

- scenario/simulation instance；
- planner；
- controller；
- LQR tracker；
- motion model；
- history buffer；
- callbacks；
- recorder；
- RNG state；
- mutable cache；
- output filesystem state；
- planner episode clock；
- treatment generator internal state。

只允许共享：

```text
proven immutable read-only map/cache
```

---

# 19. Precontext identity contract

在 treatment divergence 前，baseline/treatment 应一致：

- scenario identity；
- initial EgoState；
- map/route；
- replay observations；
- initial speed；
- initial acceleration if available；
- steering/angular state if available；
- controller config；
- random seed；
- time origin；
- PRE_CONTEXT rows；
- hashes。

每个字段必须标：

```text
DIRECTLY_BINDABLE
HASH_VERIFIABLE
NOT_AVAILABLE
```

禁止虚构 equality field。

---

# 20. HLC 最终科学状态

```text
HLC_V4_FRESH_CANARY =
VALID_NEGATIVE_ENGINEERING_RESULT

HLC_V4_CANDIDATE =
REJECTED

HLC_CURRENT_GENERATOR_BRANCH =
CLOSED_BY_SCOPE_AFTER_ENGINEERING_NONCONVERGENCE

HLC_V5 =
NOT_AUTHORIZED

REMAINING_HLC_RUNS =
NOT_AUTHORIZED

HLC_SCIENTIFIC_IMPOSSIBILITY =
NOT_ESTABLISHED
```

Astra forensic corrections：

### nominal mechanism insufficiency

V4 idealized nominal progress：

```text
monotonic fraction ≈ 0.9502554
delta vs ideal baseline ≈ -0.0497446
```

frozen gate：

```text
<= -0.10
```

所以不能再写：

> V4 mechanism 原本有效，只是 controller washout。

### terminal interpretation

treatment terminal：

```text
raw progress ≈ 1.148342
offset abs ≈ 0.342582 m
lat vel ≈ 0.275049 m/s
```

准确解释：

> 已过 target center 且仍有 lateral motion。

### rolling terminal mismatch

```text
generator future horizon terminal
!=
Primary80 realized terminal
```

offline future PASS 不能替代 closed-loop settling proof。

---

# 21. TSB 当前科学状态

```text
TSB_FAMILY_DEVELOPMENT_CANDIDATE =
FROZEN_PENDING_FRESH_QUALIFICATION
```

DEV-CAL：

```text
8/8 measurement
8/8 baseline one-phase
8/8 treatment two-phase
8/8 mechanism
8/8 F_match
8/8 safety
```

candidate SHA：

```text
7c37fdd2d939e9282adafcd98a76571c0ce9c0812e618c758b004098e5e09538
```

但：

```text
LOW_ORDER_NUISANCE_ELIMINATED =
NOT_ESTABLISHED

TSB_CLEAN_RESIDUAL_TASK =
NOT_ESTABLISHED
```

因此：

> TSB 是 frozen development candidate，不是已经 scientific-qualified residual benchmark。

---

# 22. R0 formal state corrections

## D1

```text
KNOWN_SEMANTICS_DECODABLE = YES
D1_INFORMATION_RETENTION = SUPPORTED
```

但：

```text
semantic decodability
!=
BDD suitability
```

cross-domain semantic transfer 保留历史限制。

## D2

```text
INCONCLUSIVE
```

## D3

formal state：

```text
INCONCLUSIVE
```

允许：

> simple full64 dilution 未被建立为充分解释。

不能再把 `NOT_SUPPORTED` 当成 D3 完整 formal state。

## D4

历史资产无法建立 executable residual benchmark：

```text
NOT_EVALUABLE_WITH_EXISTING_HISTORICAL_ASSETS
```

触发 R1。

---

# 23. Stage7L formal correction

Stage7L：

```text
B seed3407 = Primary
```

old64 / A / C：

```text
supporting / secondary
```

不能写：

> old64/A/B/C 的 Primary 全部失败。

Stage7L 关键事实：

```text
planner mechanism = PASS
B Primary BDD = FAIL
ego13 = highly sensitive
```

---

# 24. 历史证据链简表

## Stage7 M6.5

```text
310 paired scenarios
620 official rollouts
MMD² ≈ 0.004469
paired p ≈ 1e-5
```

## Stage6J/K

```text
183 same-scenario pairs
366 rollouts
old64 dose100 Z ≈ 9.23
dose25 = minimum detected
```

## Stage6P

```text
old64 66.5%
B 100%
C 99.5%
ego13 100%
```

## Stage6W

B/C context-balanced signal：

```text
~2.6× old64
```

## Stage6S-v3

interaction mechanism：

```text
PASS
```

但 context increment：

```text
NOT ESTABLISHED
```

## Stage7L

```text
80 scenarios
79 logs
400/400 rollout success

planner mechanism PASS
B Primary FAIL
ego13 strong
```

---

# 25. R1 / R2 核心证据

## R1 official

48/48 technical complete。

HLC：

```text
F_match 12/12
mechanism 0/12
```

TSB：

```text
measurement/mechanism 0/12
```

核心：

> planner intent 不等于 realized closed-loop mechanism。

## R1-B3

HLC：

```text
REALIZED_TRANSFER = ATTENUATED
```

TSB：

```text
REALIZED_TRANSFER = COLLAPSED
```

## R2-A

HLC transfer variable gain / lag / settling。

TSB attenuation主要发生：

```text
replanning/future trajectory
→ LQR command
```

不是简单 LQR→vehicle loss。

## R2-B

TSB：

```text
8/8 development success
```

HLC：

```text
mechanism 6/8
endpoint 0/8
```

---

# 26. Governance / technical lessons

B1 schema mismatch证明：

> governance system 自身也可能制造 infrastructure risk。

未来原则：

```text
one canonical schema
one executor
one passive recorder
one analyzer
one primary manifest
one stage authorization
```

保留状态区分：

```text
NOT_RUN
TECHNICAL_INCOMPLETE
MEASUREMENT_INVALID
SCIENTIFIC_FAIL
PASS
```

不能压成 boolean。

---

# 27. Protected assets

Protected CSV：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/
behavior_events_v2/behavior_event_metrics_v2.csv
```

SHA256：

```text
e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8
```

不得 overwrite、commit、clean 或 bulk delete。

---

# 28. 永久禁止操作

```text
git reset --hard
git clean
git add .
bulk delete outputs
```

禁止：

- 重训 old64/A/B/C；
- 重跑 R1 official；
- 重跑 R2-A / R2-B / BH / BI / B1 exposed identities；
- HLC V5；
- HLC remaining14；
- 调 TSB candidate；
- 调 frozen HLC/TSB threshold；
- outcome-based applicability revision；
- survivor selection；
- result-based H weakening；
- E 结果后改 Primary operating point；
- best-seed cherry-pick；
- Q/D 选 encoder；
- E 扩样本追显著；
- benchmark未确认前训练RBR。

Unknown 必须写：

```text
UNKNOWN
NOT_FOUND
AMBIGUOUS
BLOCKED
```

不能猜。

---

# 29. 当前 Git / provenance

分支：

```text
20260825_stageR_new
```

当前已核对远端 HEAD：

```text
9764ea46a91b6495ec9063616c0ec1ce6b1045f3
```

commit：

```text
stageS1.1: close Owner-approved semantics and training boundaries
```

近期关键节点：

```text
2f21b437... B1.1 offline recovery
d7b2f711... S1 draft package sync
9764ea46... S1.1 Owner closure repair
```

S1.1：

```text
simulation=0
runner.run=0
new scientific identities exposed=0
RBR training=0
E access=0
```

---

# 30. 当前 Scientific Owner 正式状态

```text
S1_PROTOCOL_FROZEN = YES

S1_SCOPE_AND_CLAIMS = FROZEN
S1_TSB_APPLICABILITY = FROZEN
S1_Q_QUALIFICATION_PROTOCOL = FROZEN
S1_HANDCRAFTED_COMPARISON = FROZEN
S1_PRIMARY_BDD_SAP = FROZEN
S1_SECONDARY_DIAGNOSTICS = FROZEN
S1_DATA_FIREWALL = FROZEN
S1_RBR_TRAINING_ARCHITECTURE_BOUNDARY =
FROZEN_AT_PRINCIPLE_LEVEL

S2_PREFLIGHT =
AUTHORIZED_AS_ZERO_SIMULATION_METADATA_ONLY_WORK

S2_Q20_EXECUTION =
NOT_AUTHORIZED

RBR_TRAINING =
NOT_AUTHORIZED

E_ACCESS =
NOT_AUTHORIZED
```

---

# 31. 下一个 conversation 的第一项任务

不要重新讨论整体路线。

第一项实际任务：

> **执行 S2-PRE — Metadata-Only Eligibility Census + Production Execution Binding**

必须回答：

1. frozen source universe 中有多少候选？
2. 经过 exposure/reservation/provenance/3.61m/s/route/reference 筛选后，有多少 eligible independent units？
3. independence unit 到底是 log、session 还是其它？
4. 是否存在 Q20 capacity？
5. 如果存在，deterministic Q20 NOT_RUN roster hash 是什么？
6. execution binding 是否 PASS？
7. full-arm reset contract 是否 PASS？
8. precontext identity contract 是否 PASS？
9. budget authorization draft 是否保持 inactive？
10. protected CSV hash 是否仍不变？

---

# 32. S2-Preflight 最终允许状态

只能是：

```text
S2_PREFLIGHT_READY_FOR_OWNER_AUTHORIZATION

or

S2_PREFLIGHT_BLOCKED_Q20_CAPACITY

or

S2_PREFLIGHT_BLOCKED_EXECUTION_BINDING

or

S2_PREFLIGHT_BLOCKED_RESET_CONTRACT

or

S2_PREFLIGHT_BLOCKED_OTHER
```

不能自行写：

```text
S2_AUTHORIZED
```

---

# 33. 如果 S2-Preflight PASS，下一步才是什么

只有 Scientific Owner 审完 S2-Preflight 后，才可能签：

```text
S2_TSB_FRESH_QUALIFICATION = AUTHORIZED
```

届时才允许：

```text
Q20
20 pairs
40 max simulator entries
baseline→treatment
whole roster all pass
first scientific failure stop
```

在此之前：

```text
RUNNER_RUN = 0
```

---

# 34. 当前最重要的“不要再搞错”的五件事

1. **Human Semantic Head 是训练时 head，会反向约束 encoder，不是 post-hoc probe。**
2. **Primary BDD 用 z64，不用 semantic-head prediction。**
3. **F0_project = ego13；F_match 是 4D matching descriptor；H 是 Primary challenger；O 是 mechanism control。**
4. **3.61 m/s 是 nominal measurability screen，不是 closed-loop safety guarantee。**
5. **当前下一步是 S2-Preflight，不是 S2 rollout，更不是 RBR training。**

---

# 35. 一句话论文灵魂

> **不是试图预先为所有可能的驾驶行为变化设计无限多 KPI，而是研究 learned trajectory representation 能否在固定的常规 handcrafted monitoring contract 下更有效地发现行为漂移，并把这些发现交给后续行为分析和专家判断去解释其含义与工程价值。**

---

# 36. 一句话当前执行状态

> **S1 已正式冻结；现在只做 S2-Preflight 的 metadata-only census 与 production binding，任何 fresh simulation、Q20 rollout、RBR training 或 E access 都尚未授权。**

---

`CURRENT_STAGE_S1_FROZEN_S2_PREFLIGHT_NEXT`
