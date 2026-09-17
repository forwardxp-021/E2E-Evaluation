# RBR-64 博士研究总体方案 v2.3

> 项目：E2E-Evaluation / 博士论文
> 文档状态：`RBR64_STAGE_R_ROADMAP_V2_3_CONTROLLED_VALIDATION_FREEZE_DRAFT`
> 更新时间：2026-09-17（Asia/Shanghai）
> Active branch：`20260825_stageR_new`
> 版本关系：v2.3 在 v2.2 基础上，保留论文科学主线与核心防泄漏约束，重新定义 nuPlan 的研究角色、历史 exposure 判定和 S2–S4 路线。
> 本文用于定义当前研究主线并进入 **Stage S1 — Scope, Exposure & TSB–BDD Protocol Freeze**。
> 本文不回写任何历史冻结结果，不授权 simulation、roster execution、RBR training 或结果解盲。

---

# 0. v2.3 相对 v2.2 的核心修订

v2.2 已确立以下原则，v2.3 全部保留：

1. BDD utility 是未来 Level 2 的 PRIMARY；
2. H vs H+RBR incremental predictive value 是 SECONDARY diagnostic；
3. HLC 关闭；
4. TSB 是 frozen development candidate；
5. encoder selection 必须 U-only；
6. H 必须是 development-informed strong challenger；
7. Primary 开始后禁止 post-hoc tuning、feature chasing 与 survivor selection。

v2.3 的修订不是改变论文问题，而是纠正 nuPlan 数据角色中过度严格的“fresh session”逻辑：

1. **nuPlan 是 closed-loop controlled validation platform，不是 RBR training corpus。**
2. **历史上运行、查看或用于其他实验，不自动使一个 nuPlan session 失去当前研究价值。**
3. 是否可用于某项证据，改由 **claim-relevant outcome-driven contamination** 判断，而不是由“是否曾经使用过”一刀切判断。
4. 旧的 Q→D→E 绝对不可复用链条，改为更符合受控仿真实验的三层角色：

   ```text
   Development
   Evaluation
   Confirmatory Robustness
   ```

5. exposure 按“session × claim × 被调整对象 × outcome 使用方式”登记；同一 session 可以对一个 claim 被污染，而对另一个 claim 仍然有效。
6. benchmark development exposure 与 detector-comparison exposure 明确区分：用于调试 TSB 机制，不等于用于调 RBR-vs-H 结论。
7. 大规模冻结 Evaluation 是主要受控证据；低暴露或未暴露子集用于 confirmatory robustness，而不是让少量所谓 fresh session 独自承担整篇论文。
8. 保留 session-level cluster statistics、完整分母、固定 roster、no replacement 和 no survivor selection。
9. S2 完成 benchmark stabilization 与 exposure audit；S3 执行冻结 RBR-BDD evaluation 与 confirmatory robustness；S4 完成证据闭合。

v2.3 的基本原则：

> **历史 exposure 需要透明记录和与主张匹配的限制；只有当历史 outcome 被用于选择、优化或修改当前被检验的方法、协议或样本时，才构成该主张的实质污染。**

---

# 1. 博士论文的现实问题

传统自动驾驶行为评价依赖大量 handcrafted KPI：

- speed / acceleration / jerk；
- yaw rate / curvature / lateral acceleration；
- THW / gap；
- lane-change duration；
- safety / comfort / efficiency；
- 以及随着历史问题不断增加的新规则。

这些 KPI 对“已经知道的问题”很有效。真正困难的是：

> **当某种行为变化尚未被团队命名、尚未有专门 KPI 时，如何在正式试驾之前发现“这个版本的行为与参考版本不一样”？**

现实例子：

```text
已有 KPI 全部正常
        ↓
新版本仿真结果看起来正常
        ↓
实际试驾发现：变道犹豫 / 试探 / 回撤 / 延迟 commit
        ↓
版本拒绝释放
        ↓
团队事后新增 hesitation-related KPI
```

本论文不试图为所有未来行为问题预先穷举无限 handcrafted features，而是研究：

> **能否建立一个 model-based behavior-drift discovery layer，在当前 routine KPI 尚未专门刻画目标行为时，首先发现版本行为分布发生了值得分析的变化。**

---

# 2. 应用愿景与当前可验证主张

## 2.1 应用愿景

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

## 2.2 当前 Route A 的验证范围

当前 S1–S4 主要验证：

```text
closed-loop mechanism validity
        +
representation / BDD detection utility
        +
supporting behavioral diagnosis
```

并不自动证明：

- 能发现研究者此前完全未知的 mechanism family；
- 能自动命名“犹豫变道”等新行为；
- human confirmation workflow 已被完整实证验证；
- drift alarm 一定对应质量退化或 release-blocking issue；
- nuPlan 结果已等价于真实量产软件发布验证。

正式实验目标仍然是：

> **在不为目标行为机制专门新增 routine monitoring KPI 的条件下，检验 learned trajectory representation 是否能在相同误报约束与样本预算下改善对 closed-loop behavior drift 的检测。**

---

# 3. 论文核心科学问题

> **如何建立一套面向 closed-loop planning-policy software release 的行为漂移评价方法，区分干预意图、实际行为机制、表征敏感性与分布检测效用，并检验 learned trajectory representation 是否能在固定 routine handcrafted monitoring contract 下提供更有效的 behavior drift discovery？**

英文工作表述：

> **How can closed-loop planning-policy behavior drift be evaluated by separating intervention intent, realized behavioral mechanism, representation sensitivity, and distributional detection utility; and can a learned trajectory representation improve behavior-drift discovery under a fixed routine handcrafted monitoring contract?**

---

# 4. Residual 的最终定义

> **Residual-to-F0 behavior：在指定适用域、测量窗口和预定义 routine handcrafted summary set F0 的监控合同下，仍存在的、由独立机制测量确认的 closed-loop behavior structure difference。**

其中：

```text
F0 = CURRENT_ROUTINE_HANDCRAFTED_MONITORING_SET
```

必须明确：

- residual 是相对于 F0 的；
- 不是相对于所有未来可能发明的 handcrafted feature；
- 不是“人类无法描述”；
- 不是“所有 classifier 都无法从 F0 中预测标签”；
- 不是与已知语义天然正交的 latent subspace。

后来新增 KPI 能解释一个由模型先发现的 drift，不会否定模型此前的 discovery value。

---

# 5. Learned representation 与 handcrafted KPI 的职责

## 5.1 Learned representation / BDD

> **发现当前 routine KPI monitoring space 没有充分显式覆盖的行为漂移。**

## 5.2 Handcrafted KPI / features

- known-issue monitoring；
- downstream diagnosis；
- mechanism explanation；
- engineering operationalization。

## 5.3 Expert / human driving

- semantic interpretation；
- experience relevance；
- release-value judgement。

这是推荐的工程职责分工，不是互斥能力边界。routine KPI 与 model-based BDD 在实际系统中应并行存在。

---

# 6. BDD 是 PRIMARY

Level 2 的主要科学问题：

> **在相同误报约束、相同样本预算和同一 frozen operating condition 下，RBR-BDD 是否比预注册 handcrafted challenger BDD 提供更好的 behavior-drift detection utility？**

禁止直接比较：

```text
raw MMD²_RBR vs raw MMD²_H
```

因为不同 representation 的 kernel geometry 与尺度不同。Primary 必须比较可公平解释的实际检测量。

---

# 7. 唯一 Primary BDD 指标

S1 必须在任何 claim-bearing Evaluation 前只选一个 Primary。

候选 A：固定 FPR、batch size `m` 和 drift proportion `π` 下的 detection gain：

\[
\Delta_{\mathrm{BDD}} = P_{\mathrm{detect}}(RBR)-P_{\mathrm{detect}}(H)
\]

候选 B：达到预定义 target detection probability 所需的样本量。

候选 C：仅在 S1 给出科学理由后采用的其他预定义效用量。

必须冻结：

```text
ONE_PRIMARY_BDD_METRIC
ONE_PRIMARY_OPERATING_POINT
ONE_PRIMARY_COMPARISON
```

其余全部为 Secondary。禁止在 detection、sample efficiency、stability 中事后挑选最有利者。

---

# 8. H+RBR incremental predictive value = SECONDARY

保留 `H vs H+RBR` 作为辅助解释：

| Incremental prediction | BDD gain | 解释 |
|---|---|---|
| Supported | Supported | 指定 readout 下有增量预测价值，且与 BDD 效用改善同时出现 |
| Not supported | Supported | 直接支持 frozen RBR-BDD pipeline 检测效用更高；可说与更有利 geometry 相容，但不能说 geometry 已被证明 |
| Supported | Not supported | 可读预测增量没有转化为 frozen BDD utility |
| Not supported | Not supported | 当前 Level 2 RBR qualification 未成功 |

因此：

```text
H_PLUS_RBR_RISK_TEST = SECONDARY
NO_PRECONDITION_ROLE
```

---

# 9. Paired 与 Unpaired 的角色

## 9.1 Paired controlled BDD

同一 scenario 的 baseline / treatment 用于：

- mechanism attribution；
- controlled sensitivity；
- representation diagnosis；
- 受控 intervention 下的 detector comparison。

默认状态为 `CONTROLLED_SUPPORTING_EVIDENCE`。若 independent-log 数量不足，且解盲前已预注册，也可成为收缩后的 Primary。

## 9.2 Unpaired release emulation

用不同 independent logs / scenarios 构造 reference 与 target release sample，用于模拟版本级行为分布比较。

优先 Level 2 Primary 为：

```text
UNPAIRED_RELEASE_EMULATION_BDD
```

前提是 independent logs 足够、calibration 合法、FPR evaluation 独立、power 足够、sampling frame 与 exposure strata 已冻结，且不存在 outcome-driven scenario selection。

若容量不足，S1 必须在主评估解盲前预定义 fallback：

```text
PAIRED_CONTROLLED_BDD_ONLY
```

完成 unpaired Evaluation 也只能写 `release emulation under the frozen nuPlan study setting`，不能写成真实量产发布验证。

---

# 10. nuPlan 的正式研究角色

## 10.1 平台定义

nuPlan 在本论文中的正式角色是：

> **closed-loop controlled validation platform**

它用于执行 frozen baseline 与 frozen treatment、验证 closed-loop mechanism、生成受控 rollout、比较冻结的 RBR-BDD 与 H-BDD，以及模拟版本级行为分布比较。

nuPlan 不是 RBR encoder training corpus。RBR encoder 的训练和模型选择只能依赖 U。

## 10.2 “历史上用过”不等于“科学上作废”

以下历史行为本身不自动导致 session 排除：

- 曾用于 Stage6/7/7L；
- 曾用于 old64 / ego13 或其他表示实验；
- 曾运行普通 planner 或旧版本 planner；
- 曾用于基础设施、schema、recorder 或 simulation smoke test；
- 曾查看轨迹、日志或安全输出；
- 曾用于与当前 Primary claim 无关的问题；
- 曾生成旧 rollout，但当前将用 frozen pipeline 重新运行。

历史 session identity exposure 与当前 RBR-vs-H 结论被 outcome-driven 优化不是同一件事。

## 10.3 新 rollout 与旧 exposure

在冻结 baseline、TSB、RBR、H、BDD protocol 和 roster 后重新运行，可以产生新的 closed-loop rollout。但：

- 新 rollout 不会自动消除过去的 claim-relevant contamination；
- 与当前 claim 无关的旧 exposure 也不会自动污染新的冻结评估；
- 是否可用取决于旧 outcome 是否改变了当前被检验对象或样本选择。

---

# 11. Claim-relevant outcome-driven contamination

## 11.1 正式定义

若某个 session 的历史 outcome 被用于以下任一目的，则它对相应 claim 构成 contamination：

1. 选择或修改 RBR architecture、objective、checkpoint、seed rule；
2. 选择、增加、删除或变换 H features；
3. 选择 BDD statistic、kernel、bandwidth、normalization、threshold 或 operating point；
4. 修改 TSB 参数，而当前 claim 又依赖 TSB 在未知场景上的机制泛化；
5. 决定哪些 session 保留、删除、替换或重复运行；
6. 根据 RBR-vs-H 输赢改变 Primary metric、sample size 或 stopping rule；
7. 根据期望结论修改 applicability、failure 或 exclusion rule。

核心判断问题：

> **如果没有看到该 session 的 outcome，当前被检验的方法、协议、主指标或纳入决定是否可能不同？**

若答案为“是”，必须标记 claim-relevant contamination。

## 11.2 污染是 claim-specific

同一 session 可能：

- 对“TSB 是否能在未见场景泛化”构成 development exposure；
- 对“冻结 RBR-BDD 是否优于冻结 H-BDD”仍然有效；
- 对“完全 prospective external validation”不够强；
- 对机制示例、failure analysis 或工程复现仍然有价值。

因此不得只维护 `fresh / not fresh`，而要维护：

```text
session × claim × outcome observed × decision influenced × evidence role
```

## 11.3 重新运行不能洗白实质污染

若 session 已被用于根据 RBR-vs-H outcome 调整当前 detector 或 Primary protocol，即使重新运行，也不能称为 untouched confirmatory evidence。它仍可用于 development、debugging、sensitivity analysis 和透明的 supporting analysis。

---

# 12. Exposure taxonomy

每个 session 至少按以下类别登记。类别可多选，最终角色按具体 claim 判定。

## E0 — Administrative / infrastructure exposure

下载、索引、路径检查、schema/recorder/loader 测试、非科学 smoke test。默认 `NO SCIENTIFIC EXCLUSION`。

## E1 — Historical non-claim scientific exposure

其他 Stage、old64/ego13、旧 planner 或与当前 TSB、RBR-vs-H Primary 无关的历史研究。若旧 outcome 未修改当前 pipeline 或样本纳入，默认 `ELIGIBLE FOR FROZEN EVALUATION`。

## E2 — Benchmark / generator development exposure

用于 TSB 参数、mechanism、F_match、controller transfer、safety 或 applicability 开发。

- 若 detector 和比较协议未依据这些 session 的 RBR-vs-H outcome 调整，可进入冻结 RBR-vs-H Evaluation；
- 不宜单独支持“TSB 对完全未见场景的广泛泛化”；
- 必须在机制泛化分析中标为 development-exposed stratum。

## E3 — Measurement / detector development exposure

用于 H、RBR readout、kernel、bandwidth、calibration-development、power/variance estimation 或 detector debugging。对被调组件默认 `DEVELOPMENT ONLY`，除非跨拟合结构已预定义且独立性可证明。

## E4 — Primary comparison outcome exposure

查看 RBR-vs-H 输赢后改方法、协议、样本、sample size、主指标或 operating point。对同一 Primary claim：

```text
NOT ELIGIBLE FOR CONFIRMATORY EVIDENCE
```

## E5 — Low/No claim-relevant exposure

在当前 claim 上没有发现 outcome-driven influence path。优先用于 `CONFIRMATORY ROBUSTNESS`。低暴露必须由 ledger 证明，不能仅凭记忆声明。

---

# 13. 三层证据角色

v2.3 不再把 Q、D、E 视为 session 永久且互斥的身份，而改成与用途绑定的证据角色。

## 13.1 Development

用于 TSB benchmark、mechanism measurement、H、BDD implementation、calibration-development、power planning 和 debugging。

Development 允许 outcome exposure，但必须记录：

```text
what was observed
what decision changed
which component was tuned
which later claim is affected
```

Development 结果不能被重新描述为 untouched confirmatory evidence。

## 13.2 Evaluation

在全部方法、protocol、roster、exclusion rule 与 SAP 冻结后，对较大规模 nuPlan pool 统一重新执行 baseline/treatment，完成主要受控比较。

Evaluation 可包含 E0、E1、对 RBR-vs-H claim 不构成 detector contamination 的 E2，以及其他经 audit 判定 claim-eligible 的历史 session。

Evaluation 必须：

- 使用完整、预注册的 sampling frame；
- 保留全部执行分母；
- 不按机制成功或 detector 表现挑选 survivor；
- 报告 exposure strata；
- 采用 session-level cluster-aware inference；
- 预定义 development-exposed 与 low-exposure strata 的分析。

## 13.3 Confirmatory Robustness

用于检验主要结果是否在低/无 claim-relevant exposure、替代合法抽样或预定义独立子集上保持方向与实际意义。

其作用是强化可信度、约束未知场景泛化表述，并检查结果是否由 development-exposed sessions 驱动；不要求机械复制一个“临床式绝对新鲜测试集”。

若低暴露 session 有限：

- 不把小样本非显著直接解释为主要结果失败；
- 报告 effect direction、uncertainty 与 compatibility；
- 主证据仍来自冻结的大规模 controlled Evaluation；
- 泛化措辞按证据收缩。

## 13.4 三层角色不等于随意复用

禁止：

```text
看 Evaluation outcome
→ 回到 Development 修改方法
→ 再把同一 Evaluation 称为第一次最终检验
```

解盲后修改必须创建新版本，把原 Evaluation 明确降为 development-exposed evidence，并使用预留且未受影响的 confirmatory asset，或结束当前 claim。

---

# 14. HLC 最终状态

```text
HLC_V4_CURRENT_CANDIDATE = REJECTED
HLC_CURRENT_GENERATOR_BRANCH = CLOSED_BY_SCOPE_AFTER_ENGINEERING_NONCONVERGENCE
HLC_V5 = NOT_AUTHORIZED
REMAINING_HLC_RUNS = NOT_AUTHORIZED
HLC_SCIENTIFIC_IMPOSSIBILITY = NOT_ESTABLISHED
```

必须保留：V4 nominal morphology 在 ideal-tracking 条件下仍未通过 frozen monotonic gate；treatment terminal 是 overshoot + residual lateral motion；rolling future reference endpoint 与 Primary80 realized endpoint 不等价；safety FAIL；offline feasibility 不是独立 closed-loop success evidence；development rounds 的科学判断连续演化。

HLC 的论文价值是一个 planner-intent 看似合理、但 closed-loop realization 和 benchmark qualification 失败的前瞻反例。v2.3 对 exposure 规则的更新不构成重开 HLC 的理由。

---

# 15. R0 / Stage7L 历史边界

## 15.1 D3

```text
D3 = INCONCLUSIVE
```

只允许附带说明：simple full64 dilution 未被建立为充分解释。

## 15.2 Stage7L

```text
B seed3407 = PRIMARY
```

old64 / A / C 为 supporting / secondary。

## 15.3 D1

允许写 `tested semantics are decodable`，但 cross-domain semantic transfer 的不确定性保持不变。

---

# 16. TSB 当前状态

```text
TSB_FAMILY_DEVELOPMENT_CANDIDATE = FROZEN

DEV_CAL:
mechanism = 8/8
F_match = 8/8
safety = 8/8
measurement = 8/8

LOW_ORDER_NUISANCE_ELIMINATED = NOT_ESTABLISHED
TSB_CLEAN_RESIDUAL_TASK = NOT_ESTABLISHED
```

当前定位：

> **closed-loop temporal-mechanism development candidate that passed the frozen development matching contract**

不能写成 `handcrafted methods cannot detect the behavior`。

历史上用于 TSB 开发的 session 必须标为 benchmark-development-exposed；若未使用其 detector comparison outcome 调整 RBR/H/BDD，可进入冻结 RBR-vs-H Evaluation；但不应单独承担 TSB unseen-session generalization claim。

---

# 17. TSB 的三个未来问题

## 17.1 Q1 — Mechanism Stability

在冻结、较大规模、分层报告的 pool 中，能否稳定满足：

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

Q1 不要求全部证据来自从未运行过的 session，但必须分别报告 benchmark-development-exposed、historical-non-claim 和 low/no claim-relevant exposure strata。

## 17.2 Q2 — Existing Monitoring Signal Audit

> **F0 / ego13 / H 已保留多少相关信号？在目标 BDD operating condition 下，现有 handcrafted monitoring 是否已解决该问题，还是仍存在检测敏感性、sample-efficiency 或 geometry 问题？**

问题不是“H classifier 是否完美”。

## 17.3 Q3 — RBR-BDD Utility

> **冻结 RBR-BDD 是否在同一 FPR、sample budget 和 operating condition 下比冻结 handcrafted BDD 更有效？**

Q3 是 Level 2 Primary。

---

# 18. Handcrafted comparison contract

## 18.1 F0 — Routine KPI

代表当前实际 / 项目标准监控空间。

## 18.2 ego13

代表 Generation-1 项目的标准人工行为摘要。

## 18.3 H — Development-informed strong challenger

H 防止故意使用过弱 handcrafted baseline。S1 必须一次性冻结 exact feature list、time bins、lags、smoothing、derivatives、units、validity、duplicate handling、scaler、BDD preprocessing 和 tuning budget。

```text
H = DEVELOPMENT_INFORMED_HANDCRAFTED_CHALLENGER
```

不能声称 H 与 TSB development 完全无关。

## 18.4 O — Mechanism positive control

例如 `brake_phase_count`、`release_fraction`、`second_peak_ratio`，只用于 mechanism confirmation / interpretation，不加入要求 RBR 超越的 Primary challenger。

---

# 19. 禁止无限增加 handcrafted features 追杀模型

S1 必须冻结 `F0 / ego13 / H / O`。一旦 claim-bearing Evaluation 解盲，禁止：

```text
RBR detects X → invent feature X → add to Primary H → rerun
H performs strongly → remove H features → weaken H → rerun
```

未来新增 mechanism-specific feature 只能进入 downstream diagnosis / future operational KPI，不能回写当前 Primary comparison。

---

# 20. Encoder selection 必须只依据 U

未来若授权 RBR：

```text
U = encoder training + validation only
```

Encoder architecture、training objective、checkpoint 和 seed-handling rule 只能依据预冻结 U objective。

禁止：

```text
train several encoders
→ compare any nuPlan TSB outcome
→ choose best encoder
→ report frozen Evaluation
```

即使最后测试的是另一批 session，这仍属于 TSB outcome-driven model selection。

Development assets 可以按冻结预算调整 frozen-capacity readout、kernel、bandwidth 和 calibration，但必须与 encoder selection 分离。

---

# 21. Roster、cohort 与 no-survivor-selection

优先：

```text
WHOLE_FROZEN_EVALUATION_ROSTER
```

流程：

```text
define sampling frame
→ freeze roster and exposure ledger
→ execute whole roster
→ apply frozen technical/applicability rules
→ retain complete denominator
→ report all pre-defined strata
```

禁止 `run many → retain mechanism-success pairs only → call survivors Evaluation cohort`。

若必须条件化于“机制实际形成”，只能使用预注册的：

```text
POST_TREATMENT_QUALIFIED_CONDITIONAL_BENCHMARK
```

并报告完整分母、qualification rule/rate、对 null/exchangeability 的影响、conditional 与 unconditional estimand 区别和 inference limitations。

---

# 22. Sample size

不再以 `29/29` 为默认目标。TSB generator population reliability 不是当前博士 Primary，样本设计服务于 `PRIMARY BDD ESTIMAND`。

需共同考虑 independent session/log count、exposure strata、paired/unpaired design、batch size、drift proportion、target FPR、minimum useful gain、A/A precision、cluster correlation、simulation budget 和 confirmatory capacity。

主 Evaluation 解盲前必须冻结：

```text
overall simulation budget ceiling
development budget
evaluation size
confirmatory robustness allocation
primary sample-size decision rule
stopping rule
```

允许使用与 Primary outcome 隔离的 Development assets 估计 variance/power；禁止 `look at Evaluation → almost significant → add samples`。

---

# 23. 数据与证据角色

## 23.1 U — Representation learning corpus

只用于 encoder training、validation、architecture/checkpoint/seed-rule selection。U 不使用 TSB treatment/mechanism label、nuPlan RBR-vs-H outcome 或 Evaluation/Confirmatory result。

## 23.2 Development assets

用于 benchmark、H、readout、BDD implementation、calibration-development 和 power planning；每项使用都写入 exposure ledger。

## 23.3 Evaluation assets

用于冻结的主要 controlled comparison。资格不是“从未被碰过”，而是：

> **没有对当前 claim 形成未控制的 outcome-driven influence path。**

## 23.4 Confirmatory Robustness assets

优先选择 E5 或最低 claim-relevant exposure strata，检查主要结果的可迁移性和稳健性。

## 23.5 Diagnosis assets

用于 mechanism explanation、representative-case retrieval、error analysis 和 future KPI design；不得反向修改 Primary。

---

# 24. Exposure ledger 与可审计防火墙

S1 必须创建 canonical exposure ledger，每条记录至少包括：

```text
session_id / log_id / scenario_ids
historical stages and runs
outcomes viewed
viewer / decision owner
decision influenced
component affected
claim affected
exposure class E0–E5
permitted evidence role
rationale
timestamp / artifact reference
```

对每个 session 和 claim 依次回答：看过什么 outcome；是否改变 TSB/RBR/H/BDD/metric/roster/exclusion；影响 benchmark claim、detector claim 还是两者；允许什么 evidence role；是否需分层或敏感性分析。

若历史记录不完整，标记 `EXPOSURE_UNCERTAIN`：不自动永久排除，不优先进入 Confirmatory Robustness，可在 Evaluation 中单列敏感性分析，并保守表述。

禁止结果出炉后把表现好的历史 session 改标 eligible、把表现差的改标 contaminated，或用“technical failure”删除科学失败。

---

# 25. 统计独立性与 cluster 管理

重复 scenario、同一 log 内多个 scenario、同一 session 内多个 rollout 不能被当作完全独立样本。

Primary 和关键 secondary inference 必须：

- 预定义 statistical unit；
- 以 session 或更高独立单元做 cluster-aware bootstrap/permutation/resampling；
- 防止同一 session 泄漏到 calibration 与 FPR evaluation；
- paired design 保留配对关系；
- unpaired design 保留 log/session 独立性；
- 报告 session、log、scenario 与 rollout count；
- 不通过切碎一个 session 人为放大 n。

默认：

```text
INDEPENDENCE MANAGEMENT UNIT = SESSION
```

历史使用资格与统计独立性是两个问题：exposure audit 管理 outcome-driven bias，cluster statistics 管理相关性与不确定性；二者都必须满足。

---

# 26. RBR training 最低授权条件

当前：

```text
RBR_TRAINING = NOT_AUTHORIZED
```

未来授权不要求 H classifier imperfect，也不要求 H+RBR expected to win。至少要求：

1. TSB definition、mechanism measurement 与 applicability contract 已冻结；
2. exposure ledger 完成，claim-relevant influence path 可审计；
3. F0 / ego13 / H signal 与替代解释已量化；
4. 仍存在明确、可证伪、未被廉价 handcrafted-BDD 充分解决的问题；
5. Primary BDD test 完整且可负担；
6. encoder objective / architecture budget 已冻结；
7. U 与 nuPlan validation 角色明确隔离；
8. 三层证据角色和 sampling frame 已冻结；
9. claim-bearing Evaluation 尚未解盲；
10. no survivor selection、cluster inference 和 post-hoc prohibition 已进入协议。

---

# 27. Level 1 / Level 2 / Level 3

## 27.1 Level 1 — Dissertation base contribution

> **闭环自动驾驶行为漂移评价必须区分干预意图、实际行为机制、表征可读性与分布测量效用；前一层成立不能保证后一层有效。**

支持来源包括 Stage6/7/7L、R0/R1/R2、HLC negative development、TSB candidate、paired/unpaired distinction、benchmark/measurement failure analysis 和 exposure-aware controlled validation framework。Level 1 不依赖新 RBR 阳性结果。

## 27.2 Level 2 — Model-based BDD utility

若成功：

> **在预定义 TSB 适用域和冻结 nuPlan controlled-evaluation contract 下，RBR 在相同 FPR、样本预算与主要使用条件下，相对于预注册 handcrafted challenger，提高了对已确认 closed-loop behavior morphology drift 的 BDD 检测效用。**

若 secondary incremental-risk 也支持，可附加“在指定 readout protocol 下具有增量预测价值”。只有完成合格 unpaired Evaluation 才允许写 `release-emulation under the frozen study setting`。

若低暴露 Confirmatory Robustness 方向一致，可说明主要结论在较低 claim-relevant exposure 子集中得到稳健性支持。

不得写 `real production release validation` 或 `universally generalizes to unseen nuPlan sessions`，除非对应证据确实存在。

## 27.3 Level 3

多个 independent residual families、interaction、broader unknown-morphology discovery 和完整 human-confirmation validation 均为 `FUTURE WORK`。

---

# 28. 应用愿景：Model-first Discovery

```text
Routine KPI monitoring + Learned representation / BDD
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

论文当前主要实证是 `discovery layer + mechanism validity + BDD utility`。representative-case retrieval、explanation usability 和 human confirmation effectiveness 只有实际执行研究后才能升级为实证主张。

---

# 29. BDD alarm 的语义

```text
BDD alarm = BEHAVIOR DISTRIBUTION CHANGE DETECTED
```

不能自动写成 anomaly、degradation、bad behavior 或 release-blocking issue。版本变化可能更保守、更积极、更舒适，或只是可接受的风格变化。价值判断需要 mechanism analysis、engineering context 和 expert/human review。

---

# 30. Stage S1 正式授权边界

```text
S1_PROTOCOL_AND_EXPOSURE_WORK = AUTHORIZED
```

授权范围：read-only evidence review、historical exposure reconstruction、protocol/schema/SAP drafting、zero-run tooling、schema-faithful fixture tests、sample-size/power design、static applicability derivation、sampling-frame/roster drafting、documentation/manifests，以及不调用 simulation 或 `runner.run` 的测试。

明确禁止：

```text
simulation / runner.run / TSB rollout / HLC rollout
evaluation roster execution / new scientific outcome exposure
TSB parameter tuning / HLC V5 / RBR training
encoder checkpoint selection from nuPlan outcome
claim-bearing Evaluation construction or unblinding
```

---

# 31. Stage S1 必须产出的最终包

## S1-A — Scope / Claim Freeze

明确 HLC closure、Level 1/2 claim、application vision vs empirical claim、Residual-to-F0、paired/unpaired boundary、nuPlan platform role，以及 historical use ≠ automatic exclusion。

## S1-B — TSB Applicability & Benchmark Contract

冻结 current R2 candidate、2.0 m/s floor、low-speed measurability、Primary80、safety eligibility、reference completeness、mechanism measurement 和 exposure strata 角色。不得按未来 Evaluation outcome 调整。

## S1-C — Exposure & Evidence-role Protocol

冻结 E0–E5 taxonomy、canonical ledger、claim-specific contamination rule、三层证据角色、uncertain exposure policy、whole-roster、no replacement/no survivor selection、technical vs scientific failure、budget ceiling 和解盲后版本管理。

## S1-D — Handcrafted Challenger Contract

冻结 `F0 / ego13 / H / O` 及 exact implementation、normalization、missingness、feature validity、tuning data/budget 和 no post-Evaluation expansion rule。

## S1-E — BDD Statistical Analysis Plan

冻结 paired/unpaired Primary、唯一 metric 和 operating point、FPR、batch/drift composition、fair-tuning budget、null calibration、independent FPR evaluation、CI、session-level cluster resampling、multiplicity、Evaluation size、confirmatory robustness、exposure-stratified sensitivity 和 success/fail/inconclusive rule。

## S1-F — Secondary Diagnostic SAP

冻结 H vs H+RBR risk、semantic probes、geometry diagnostic、shortcut audit、mechanism-attribution wording 和 diagnosis 非反馈边界。它们不能取代 BDD Primary。

## S1-G — Canonical Technical Schema

目标：

```text
one canonical schema / executor / passive recorder / analyzer
one primary manifest / exposure ledger / stage authorization
```

fixture 覆盖 80 states/79 transitions、timestamp irregularity、missing/wrong keys、unit mismatch、phase boundary、low-speed invalidity、F_match boundary、safety fail、duplicate/cluster identity、exposure-role conflict、budget violation 和 analyzer/production parity。

---

# 32. S1 结束的 Owner 决策

S1 完成后不自动进入仿真。Scientific Owner 判断：

```text
S2_BENCHMARK_STABILIZATION_AND_AUDIT = AUTHORIZED / NOT_AUTHORIZED
```

授权前确认 protocol、candidate SHA、applicability、sampling frame/roster rule、Primary estimand、H/O/F0/ego13、U-only rule、exposure ledger、三层证据边界、budget、无隐藏 outcome-dependent path、no survivor selection 和 session-level cluster statistics 均已闭合。

---

# 33. 重新定义的 S2 / S3 / S4 路线

## 33.1 S2 — Benchmark Stabilization, Exposure Audit & Evaluation Readiness

S2 不证明所有 session 都“fresh”，而是形成可审计、可冻结、可用于受控比较的 benchmark 与证据框架：

1. 完成历史 exposure reconstruction；
2. 为 session 分配 claim-specific role；
3. 在 Development budget 内完成必要的 TSB mechanism/safety/measurement stabilization；
4. 完成 F0/ego13/H signal audit；
5. 完成 BDD implementation、A/A calibration 与 power planning；
6. 冻结 Evaluation frame、roster、strata、sample size 与 SAP；
7. 识别并保留 Confirmatory Robustness 子集；
8. 确认 encoder selection 严格 U-only。

S2 结束时 Scientific Owner 决定：

```text
IS THERE A MEANINGFUL, IDENTIFIABLE,
AND AFFORDABLE BDD PROBLEM WORTH RBR EVALUATION?
```

若否，S2 → S4，以 Level 1、benchmark analysis 和 negative result 闭合；若是，S2 → S3。

## 33.2 S3 — Frozen RBR–BDD Controlled Evaluation

```text
U-only RBR training / selection
        ↓
freeze encoder + H + BDD + metric + roster + SAP
        ↓
run full frozen nuPlan Evaluation roster
        ↓
locked Primary analysis
        ↓
pre-registered exposure-stratified analyses
        ↓
Confirmatory Robustness analysis
```

证据结构：

1. Primary：大规模 frozen controlled Evaluation 中 RBR-BDD vs H-BDD；
2. Supporting：paired mechanism attribution、F0/ego13、H+RBR；
3. Robustness：低暴露子集、替代合法 sampling 或预定义 sensitivity；
4. Diagnosis：机制解释、代表 case、future KPI，不反馈修改 Primary。

S3 结果可为 `PASS / FAIL / INCONCLUSIVE`，均进入 S4。

S3 禁止看结果后扩展 H、选择 encoder、更换 Primary、删除不利 session、把 development-exposed subset 伪装成 untouched confirmation，以及开启 HLC/new family/second RBR rescue。

## 33.3 S4 — Thesis Evidence Closure

S4 固化 Level 1/2 主张，汇总 positive/negative/inconclusive evidence，报告完整 denominator、exposure strata 和 clustered uncertainty，明确 controlled evaluation、release emulation 与真实量产验证边界，完成 reproducibility package、artifact manifest 与 limitations；新增 KPI 只进入 diagnosis/future operationalization。

最终路线：

```text
S1 Scope, Exposure & Protocol Freeze
        ↓
S2 Benchmark Stabilization + Exposure Audit + Evaluation Readiness
        ↓
Scientific Owner decision
        ↓ YES
S3 Frozen RBR–BDD Controlled Evaluation + Confirmatory Robustness
        ↓
S4 Thesis Evidence Closure

NO at S2 → directly S4
```

---

# 34. 结果解释矩阵

| Primary Evaluation | Confirmatory Robustness | 允许结论 |
|---|---|---|
| Supported | Compatible | 冻结 nuPlan contract 下 RBR-BDD 改善检测效用，且未显示由 development-exposed sessions 单独驱动 |
| Supported | Uncertain due to low power | 大规模 controlled Evaluation 支持 utility；跨 exposure-stratum 稳健性仍不确定 |
| Supported mainly in development-exposed stratum | Weak/incompatible | 结果可能依赖 benchmark exposure 或场景构成；低暴露稳健性未建立，主张收缩 |
| Not supported | Any | 冻结 contract 下未获得 RBR-BDD 优于 strong H-BDD 的预定义证据；不得 post-hoc rescue |
| Mechanism unstable | Any | benchmark validity 不足，不能解释为 learned representation 的一般成败；仍支持 Level 1 |

---

# 35. 当前最高层共识

```text
1. BDD IS PRIMARY.
2. H+RBR INCREMENTAL PREDICTION IS SECONDARY.
3. NUPLAN IS A CLOSED-LOOP CONTROLLED VALIDATION PLATFORM,
   NOT THE RBR TRAINING CORPUS.
4. HISTORICAL SESSION USE DOES NOT AUTOMATICALLY MEAN EXCLUSION.
5. EVIDENCE ROLE IS DETERMINED BY CLAIM-RELEVANT
   OUTCOME-DRIVEN CONTAMINATION.
6. CONTAMINATION IS CLAIM-SPECIFIC, NOT A PERMANENT SESSION LABEL.
7. DEVELOPMENT / EVALUATION / CONFIRMATORY ROBUSTNESS
   REPLACE THE ABSOLUTE Q→D→E NON-REUSE LOGIC.
8. FROZEN LARGE-SCALE CONTROLLED EVALUATION IS THE MAIN EVIDENCE;
   LOW-EXPOSURE DATA PROVIDE CONFIRMATORY ROBUSTNESS.
9. SESSION-LEVEL CLUSTER STATISTICS REMAIN MANDATORY.
10. SURVIVOR SELECTION, OUTCOME-DRIVEN ROSTER EDITING,
    AND POST-HOC SAMPLE INCREASE ARE FORBIDDEN.
11. MODEL-FIRST DISCOVERY → FEATURE DIAGNOSIS → HUMAN CONFIRMATION
    IS THE APPLICATION VISION.
12. RESIDUAL IS RELATIVE TO F0, NOT ALL POSSIBLE FEATURES.
13. HANDCRAFTED FEATURES REMAIN ESSENTIAL FOR DIAGNOSIS.
14. BDD ALARM MEANS BEHAVIOR CHANGE, NOT AUTOMATIC DEGRADATION.
15. HLC IS CLOSED BY SCOPE / ENGINEERING NONCONVERGENCE.
16. TSB IS A FROZEN DEVELOPMENT CANDIDATE,
    NOT YET A CLEAN RESIDUAL BENCHMARK.
17. ENCODER ARCHITECTURE / CHECKPOINT SELECTION MUST BE U-ONLY.
18. H MUST REMAIN A DEVELOPMENT-INFORMED STRONG CHALLENGER.
19. NO POST-HOC TUNING OR FEATURE CHASING AFTER UNBLINDING.
20. LEVEL-2 PRIMARY CLAIM IS BDD UTILITY
    UNDER A FROZEN FAIR CONTROLLED COMPARISON.
21. UNKNOWN-MORPHOLOGY DISCOVERY, BROAD UNSEEN-SCENARIO
    GENERALIZATION, AND HUMAN CONFIRMATION REQUIRE DIRECT EVIDENCE.
```

---

# 36. 一句话论文灵魂

> **不是试图预先为所有可能的驾驶行为变化设计无限多 KPI，而是研究 learned trajectory representation 能否在固定的常规监控合同与可审计的受控仿真实验中，更有效地发现 closed-loop behavior drift，并把这些发现交给后续行为分析和专家判断去解释其含义与工程价值。**

---

`RBR64_STAGE_R_ROADMAP_V2_3_CONTROLLED_VALIDATION_FREEZE_DRAFT`
