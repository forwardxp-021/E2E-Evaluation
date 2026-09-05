# E2E-Evaluation 博士研究项目权威交接

> **状态：`CURRENT_STAGE_R_R2_HANDOVER_UPDATED_FOR_INDEPENDENT_REVIEW`**  
> 更新时间：2026-09-05（Asia/Shanghai）  
> 仓库：`forwardxp-021/E2E-Evaluation`  
> 当前研发分支：`20260825_stageR_new`  
> 当前已知远端 HEAD：`2f21b437a105067cfb19932ba7799fc4f4a40eca`  
> 当前内容 tree：`0f2173f63b96a670663387bbf9f2d49547c0e545`  
> 当前阶段：**Stage R / R2 — Residual Benchmark Enablement 与 RBR 前置验证**  
> 当前核心状态：**RBR 正式训练仍未授权；TSB family development candidate 已冻结；HLC V4 fresh canary 为有效负结果；HLC 后续路线等待独立 Scientific Owner review。**  
> 本文件替代 2026-08-20 的旧 thesis-closure handover 作为当前实时入口；旧文件只保留为历史快照，不再代表当前执行状态。

---

## 0. 给下一个 conversation / Work / Astra session 的启动指令

### 0.1 第一原则

当前不是“继续 Stage7L”或“只写论文”的阶段。2026-08-20 之后项目重新打开了 **Stage R**，目标是解释 learned64 在 Stage7L pure-lateral prospective benchmark 中失败的根因，并建立一个能够真正验证 **Residual Behavior Representation (RBR-64)** 的 prospective residual benchmark。

下一个 session 在任何写操作前，必须先理解：

1. Stage6/Stage7/Stage7L 证据仍然冻结；
2. Stage R 是在这些冻结证据之上的**新研究分支**；
3. RBR 尚未正式训练；
4. 当前真正的 blocker 是：**residual benchmark 必须先在 closed-loop realized behavior 层产生可验证的 residual mechanism**；
5. HLC 与 TSB 两个 residual family 的当前状态不同，不能混写；
6. HLC 当前只有一个 fresh V4 canary pair 的正式 closed-loop evidence，不能把“当前 candidate 失败”升级成“整个 HLC scientific construct 已证明不可能”。

### 0.2 推荐阅读顺序

1. `AGENTS.md`
2. `handover.md`
3. `docs/stageR/r1/` 下 R0/R1 closure、B3 root-cause 与 outcome-exposure 文件
4. `docs/stageR/r2/R2_A_Controller_Transfer_Identification_Report_v1.md`
5. `docs/stageR/r2/R2_A_TSB_Replanning_Transfer_Audit_v1.md`
6. `docs/stageR/r2/R2_A_R2B_Generator_Architecture_Decision_v1.md`
7. `docs/stageR/r2/R2_B_Controller_Aware_Generator_Development_Report_v1.md`
8. `docs/stageR/r2/R2_BH_HLC_Target_Capture_Development_Report_v1.md`
9. `docs/stageR/r2/` 中 R2-BI / BJ-A / A2 / A3 / B0 / B0.1 / B0.2 / B1 / B1.1 的当前报告与 manifest
10. `docs/stageR/r2/r2_bh_tsb_family_development_candidate_v1.0.json`
11. `tools/r2_b_controller_aware_generator_v1.py`
12. HLC V4 generator/planner 与 frozen HLC evaluator
13. nuPlan `TwoStageController` / `LQRTracker` 实现
14. `docs/stage7l_e_prospective_representation_bdd_report_zh.md`
15. `outputs/final_standardized_bdd_style_report_card_v2_stage7l/final_standardized_bdd_style_report_card_zh.md`

启动时先核对：

```bash
git status --short --branch
git rev-parse HEAD
git log -1 --oneline
git rev-parse origin/20260825_stageR_new
```

### 0.3 当前默认权限

除非 Scientific Owner 明确授权，默认：

```text
RUNNER_RUN = 0
NEW_SIMULATION = NOT_AUTHORIZED
R2_C = NOT_AUTHORIZED
CONFIRMATORY_SMOKE = NOT_AUTHORIZED
RBR_TRAINING = NOT_AUTHORIZED
RBR_A/B/C = NOT_AUTHORIZED
THRESHOLD_CHANGE = FORBIDDEN
OUTCOME_EXPOSED_IDENTITY_REUSE = FORBIDDEN
```

下一个 Astra/Reviewer session 推荐先做 **read-only independent scientific review**，不要直接改代码或设计 HLC V5。

---

# 1. 一分钟项目摘要

## 1.1 论文基础主线

原论文主线已经建立：

> **Task-conditioned trajectory-level behavior drift evaluation framework for closed-loop planning policies**

核心目标是：

> 如何判断两个 E2E/planning policy release 的驾驶行为是否发生漂移、漂移发生在哪些行为维度，并区分 controlled same-scenario attribution 与 production-style unpaired release monitoring。

Stage6/Stage7/Stage7L 已经形成冻结证据链：

- official nuPlan controlled same-scenario confirmation；
- pure longitudinal treatment；
- interaction treatment；
- prospective pure-lateral treatment；
- paired 与 unpaired BDD；
- old64 / A / B / C / ego13 representation qualification；
- standardized BDD Style Report Card。

这些结果仍然有效，不因 Stage R 重新解释或覆盖。

## 1.2 Stage R 为什么重新打开研究

Stage7L prospective pure-lateral benchmark 中：

- planner-level pure-lateral mechanism确认成功；
- ego13 对该 treatment 高度敏感；
- old64 / A / B / C learned representation 均未可靠检出；
- Primary B-3407 为：

```text
BDD/null-q95 = 0.435802×
Z_BDD = -0.065037
p = 0.411906
FAIL
```

这暴露出一个关键问题：

> learned64 能解码已知 handcrafted semantics，但其 latent geometry / measurement sensitivity 可能不足以表达更细粒度的 temporal / interaction residual behavior。

因此 Stage R 正式提出：

# Residual Behavior Representation（RBR-64）

核心假设：

> Real driving behavior contains latent degrees of freedom beyond finite handcrafted features. A representation trained directly from raw temporal/contextual trajectories should preserve known semantics while also capturing additional temporal/interaction residual information.

形式化：

```text
z_i = f(e_i | c_i)
```

其中 `z_i` 表示在给定 context 下 ego 实际选择的 response。

未来 release-level drift 比较：

```text
P_r(z | task, context stratum)
```

RBR 目标不是复刻 ego13，而是：

```text
known semantic retention
+
residual temporal / interaction sensitivity
+
context robustness
```

## 1.3 当前最重要的科学前置条件

在训练 RBR 之前，必须先有一个可靠 residual benchmark：

```text
F_handcrafted(baseline) ≈ F_handcrafted(treatment)
```

同时：

```text
M_residual(baseline) != M_residual(treatment)
```

而且这个 residual mechanism 必须存在于：

> **closed-loop realized ego behavior**

不能只存在于 planner trajectory intent。

因此：

```text
RBR_FORMAL_TRAINING = NOT_AUTHORIZED
```

直到 residual benchmark prospective qualification 成立。

---

# 2. R0：D0–D5 诊断框架与结论

最初 Stage R 使用 D0–D5 诊断 Gen-1 representation。

## 2.1 D0 Temporal Information Loss

结论：

```text
D0 = MIXED_NOT_GENERALIZED
```

历史 Gen-1 learned representation：

- Waymo T=80；
- Stage7L T=150；
- final GRU hidden；
- 无 mask / valid-length aware pooling；
- Stage7L full150；
- learned 83D 未标准化并包含历史 sentinel 999 风险。

允许的结论是：

> temporal contract / masking / pooling / readout 存在风险。

禁止声称：

> Stage7L 失败是由 80→150 长度变化单独造成。

## 2.2 D1 Information & Geometry

9 个 CORE known semantic targets 在 in-domain probe 中可从 learned representation 解码。

正式结论：

```text
KNOWN_SEMANTICS_DECODABLE = YES
D1_INFORMATION_RETENTION = SUPPORTED
```

但：

> “可以 probe 出来”不等于 latent geometry 适合 BDD。

## 2.3 D2 Context / Response Leakage

neighbor-zero / interaction diagnostic 不能建立干净 causal conclusion。

正式状态：

```text
D2_CONTEXT_RESPONSE_SEPARATION = INCONCLUSIVE
```

不能写 `CONTEXT_LEAKAGE = YES`，也不能写 `CONTEXT_LEAKAGE = NO`。

## 2.4 D3 Measurement Readout

task projection / restricted readout 没有救活 Stage7L learned BDD。

正式结论：

```text
D3_SIMPLE_FULL64_DILUTION_HYPOTHESIS = NOT_SUPPORTED
TASK_PROJECTION_RESCUE = NO
```

## 2.5 D4 Residual Benchmark

R0 时历史资产无法构造满足：

```text
handcrafted semantics matched
+
residual temporal/interaction mechanism different
```

的 executable prospective benchmark。

正式状态：

```text
R0_D4 = NOT_EVALUABLE_WITH_EXISTING_HISTORICAL_ASSETS
```

这直接触发 R1。

## 2.6 D5 External Assets

Person2Drive / StyleDrive / 外部个性化驾驶资产仍可作为未来 external validation 参考，但不是当前 blocker。

---

# 3. R1：第一次 Prospective Residual Benchmark Enablement

R1 设计两个 residual family。

## 3.1 HLC — Hesitant Lane Change

目标行为：

```text
advance
→ hold
→ retreat
→ recommit
```

冻结 realized mechanism：

```text
baseline retreat = 0
treatment retreat >= 1
commit latency delta >= 0.5 s
treatment monotonic fraction <= baseline - 0.10
```

F_match Primary：

```text
mean_speed
end_minus_start_speed
path_length
```

endpoint：

```text
terminal offset <= 0.25 m
heading error <= 0.05 rad
lateral velocity <= 0.25 m/s
paired route progress delta <= 1.5 m
```

engineering：

```text
lateral acceleration <= 6.0 m/s²
yaw rate <= 1.0 rad/s
curvature <= 0.5 1/m
```

## 3.2 TSB — Two-Stage Braking

baseline：

```text
one braking phase
```

treatment：

```text
brake
→ release
→ brake
```

冻结 mechanism：

```text
baseline exactly 1 brake phase
treatment exactly 2 brake phases
release fraction >= 0.15
second peak ratio >= 0.50
```

brake phase threshold：

```text
acceleration <= -0.80 m/s²
```

## 3.3 R1 official B2.9-E

最终完整执行：

```text
48/48 technical complete
24 pairs
Primary80 exact traces
run_runners lifecycle complete
metric callbacks complete
safety adapter complete
retry = 0
replacement = 0
```

HLC：

```text
context        12/12 PASS
F_match        12/12 PASS
engineering    12/12 PASS
mechanism       0/12 PASS
endpoint        6/12 PASS
safety         11/12 PASS
```

12/12 mechanism failure 唯一 reason：

```text
MONOTONIC_PENALTY_LT_0P1
```

TSB：

```text
context          12/12 PASS
F_match          12/12 PASS
measurement       0/12 OK
mechanism         0/12 PASS
safety           11/12 PASS
```

R1 结论：

```text
R1_EXECUTION = COMPLETE
R1_TECHNICAL_RUNTIME_VALIDITY = PASS
R1_CONTEXT_CONTROL = PASS
R1_F_MATCH_CONTROL = PASS
R1_HLC_REALIZED_MECHANISM = FAIL
R1_TSB_REALIZED_MECHANISM = FAIL
R1_RESIDUAL_BENCHMARK_ENABLEMENT =
FAILED_UNDER_FROZEN_R1_CONTRACT
```

注意：

> 这不是 infrastructure failure。F_match 成功，说明 residual-pair nuisance matching 思路有效。失败发生在 intended residual mechanism 没有可靠穿过 closed-loop transfer。

---

# 4. R1-B3：Realized Mechanism Transfer Forensic

B3 是 0 simulation、只读 forensic。

## 4.1 HLC

realized baseline monotonic：

```text
1.0 / 1.0 / 1.0 / 1.0 / 1.0
```

treatment：

```text
0.905146 / 0.913578 / 0.927766 / 0.932607 / 0.950015
```

realized delta：

```text
-0.094854 / -0.086422 / -0.072234 / -0.067393 / -0.049985
```

ideal generator delta：

```text
-0.127303
```

frozen gate：

```text
<= -0.10
```

HLC transfer ratio：

```text
0.392646 / 0.529392 / 0.567418 / 0.678869 / 0.745104
```

retreat：

```text
baseline = 0 for 12/12
treatment >=1 for 12/12
```

latency：

```text
12/12 PASS
median delta ≈ 3.10 s
```

正式解释：

```text
GENERATOR_INTENT = VALID
REALIZED_TRANSFER = ATTENUATED
MEASUREMENT_ERROR = NOT_SUPPORTED
```

## 4.2 TSB

baseline / treatment：

```text
NO_BRAKE_PHASE = 12/12
```

intended-window realized decel medians：

```text
baseline ≈ 0.768 m/s²
treatment first ≈ 0.249 m/s²
treatment second ≈ 0.334 m/s²
```

ideal generator：

```text
baseline 1 phase
treatment 2 phases
release fraction ≈ 0.333
second peak ratio ≈ 1.0
```

主导根因：

```text
BRAKE_AMPLITUDE_ATTENUATION = 12/12
REALIZED_TRANSFER = COLLAPSED
```

## 4.3 R1 数据防火墙

R1 official 24 identities：

```text
OUTCOME_EXPOSED = TRUE
R1_SCIENTIFIC_HISTORY_ONLY = TRUE
R2_DEVELOPMENT_USE_FORBIDDEN = TRUE
R2_CONFIRMATORY_USE_FORBIDDEN = TRUE
```

---

# 5. R2-A：Controller Transfer Identification

使用 fresh engineering-only：

```text
HLC 8 identities
TSB 8 identities
overlap with R1/historical blacklist = 0
```

总有效工程运行：

```text
HLC 40
TSB 40
80 effective runs
4 technical reruns
84 actual runner.run engineering calls
scientific simulation = 0
```

## 5.1 HLC transfer

retreat gain：

```text
0.526 / 0.765 / 0.869 / 0.996 / 1.238
```

tracking lag：

```text
0.2 / 0.3 / 0.3 / 0.4 / 0.4 s
```

settling delay：

```text
0.10 / 0.25 / 0.40 / 0.45 / 4.40 s
```

说明：

> 不能用一个静态 gain 同时补偿 retreat、recommit 与 settling。

## 5.2 TSB transfer

generator→LQR gain 中位数：

```text
0.454706
```

LQR→realized gain 中位数：

```text
0.830694
```

主要 attenuation 发生在：

> repeated replanning / future trajectory processing → LQR command

32/32 two-pulse runs：

```text
two distinct phases = 0
phase loss = 32
phase merge = 0
```

R2-A surrogate：

```text
small deterministic linear model
no ML
engineering-only
```

LOIO：

```text
HLC retreat MAE median = 0.006514
TSB peak decel MAE median = 0.012151 m/s²
TSB timing MAE median = 0.028099 s
```

推荐：

```text
CONTROLLER_AWARE_PRECOMPENSATION
+
DEV_ONLY_OFFLINE_FEEDBACK_CALIBRATION
```

---

# 6. R2-B：Controller-Aware Generator Development

fresh DEV-CAL：

```text
HLC 8
TSB 8
overlap with R1/R2-A/historical blacklist = 0
```

## 6.1 TSB

仅 1 轮达到：

```text
measurement OK = 8/8
baseline one-phase = 8/8
treatment two-phase = 8/8
mechanism = 8/8
F_match = 8/8
safety = 8/8
```

冻结 candidate：

```text
TSB_FAMILY_DEVELOPMENT_CANDIDATE_FROZEN
validation_status = PENDING_FRESH_R2C_VALIDATION
```

candidate SHA256：

```text
7c37fdd2d939e9282adafcd98a76571c0ce9c0812e618c758b004098e5e09538
```

参数：

```text
baseline:
-1.45 m/s² × 1.8 s

treatment:
first  -2.4 m/s² × 0.9 s
release +1.4 m/s² × 1.3 s
second -2.4 m/s² × 0.9 s
```

注意：这是 DEV-CAL candidate，不是 fresh scientifically validated benchmark。

## 6.2 HLC R2-B

4 轮达到冻结上限，最终：

```text
mechanism = 6/8
F_match = 8/8
endpoint = 0/8
engineering = 8/8
safety = 8/8
```

endpoint 主导 failure：

```text
treatment terminal offset
```

所以：

```text
R2_B_DEVELOPMENT_NOT_CONVERGED
```

---

# 7. HLC Architecture Development Chain

## 7.1 R2-BH — Target Capture V2

V1 旧 re-anchor：

```text
xy = source*(1-progress)+target*progress
xy += current_ego_xy - xy[0]
```

synthetic forensic 证明 current residual target-lane offset 会被整个未来 trajectory 原样平移，不能形成真正 target-center attractor。

BH V2 尝试：

```text
BEHAVIOR_MORPHOLOGY
+
TARGET_CAPTURE
```

但产生新的 controller-interface 缺陷：

- capture 修改 final `xy`；
- heading 没有从 final `xy` 重算；
- `(x,y,heading)` 不是同一条运动学曲线；
- LQR lateral demand 主要通过 heading/curvature 进入 controller；
- state0 exact current ego 时 initial lateral/heading error为0；
- capture 的未来横向位置变化没有形成一致的 controller-visible curvature；
- fixed deadline 后 residual 未收敛时出现 state0→state1 几何 hard jump。

BH fresh DEV-ARCH：

```text
mechanism = 0/8
endpoint = 0/8
F_match = 8/8
engineering = 8/8
safety = 4/8
```

正式结论：

```text
V1_CONSTANT_REANCHOR_DIAGNOSIS = SUPPORTED
V2_BEHAVIOR_CAPTURE_SEPARATION_PRINCIPLE = RETAIN
V2_IMPLEMENTATION = REJECTED
```

## 7.2 R2-BI — Kinematically Consistent V3

V3 修复：

```text
final XY
→ tangent heading
→ curvature
```

并保证：

- state0 exact；
- controller-visible curvature；
- exact frozen LQR shadow 与 actual return一致；
- fail-closed feasibility gate。

第一次 treatment 在 `t=1.1 s` fail-closed：

```text
max lateral acceleration = 7.391761 m/s²
frozen limit = 6.0 m/s²
```

解析发现原 frozen morphology 的 advance / recommit 时间尺度对普通 lane width 本身过激。

正式状态：

```text
V3_KINEMATIC_CONTROLLER_INTERFACE = PASS
PRIMARY_FAILURE =
TREATMENT_MORPHOLOGY_TIME_SCALE_INFEASIBLE
```

## 7.3 R2-BJ-A — V4 Offline Morphology Redesign

V4：

- 保留 XY→heading→curvature；
- 保留 controller observability；
- 去除破坏 divergence C2 continuity 的正 lag shift；
- 放缓 morphology；
- common→treatment at `1.1 s` 保证 P/V/A 连续。

intrinsic lateral acceleration：

```text
advance ≈ 2.057 m/s²
retreat ≈ 1.668 m/s²
recommit ≈ 2.815 m/s²
```

最初 Cartesian envelope 把全数据库独立的速度极值与曲率极值做笛卡尔组合，导致大量失败。后来判定该 Cartesian edge 是 adversarial stress test，不能直接作为真实 HLC joint-support applicability model。

## 7.4 A2 — Joint Support Audit

历史 outcome-blind HLC opportunity：

```text
57 audited
47 complete joint records
10 extraction failures
```

47 条完整记录：

```text
45,120 offline V4 cases
45,120 PASS
native-only infeasible = 0
generated increment infeasible = 0
composite infeasible = 0
terminal settling infeasible = 0
```

历史 `0.082281 1/m` 曲率极值被识别为 terminal short-segment gradient artifact。

## 7.5 A3 — Prospective Applicability Audit

完整扫描冻结 source universe：

```text
5,386,575 deduplicated tokens
```

预冻结 hash-ranked frame：

```text
256
EARLY_STOP = FALSE
```

256/256：

```text
topology ambiguity          147
duplicate / short segment    74
reference too short           4
route occurrence ambiguity    3
full V4 component stage      28
V4 generated/composite fail  11
complete PASS                17
```

11 个 V4 failure 全部集中在：

```text
v_audit ≈ 0.162 – 1.931 m/s
```

17 个 pass 最低：

```text
4.318 m/s
```

提示 V4 更像 moving-regime HLC architecture，而不是 near-stop/crawl lane-change architecture。

注意：`3.0 m/s` moving-regime floor 是 **development-stage estimand revision**，不能伪装成 outcome-independent 原始 scientific threshold。

---

# 8. R2-BJ-B0/B1：Fresh HLC V4 Engineering Canary

进入真实 canary 前已经完成：

- frozen exact baseline→treatment `[1,2]` slice；
- unique `runner.run()` call point；
- exact run budget；
- architecture failure 原子持久化；
- actual LQR passive recorder；
- exact frozen shadow；
- 80 planner calls / 79 controller transitions contract；
- post-run analyzer outcome 前冻结；
- no rerun / no replacement / no parameter update。

## 8.1 B1 真实 canary

唯一一次执行：

```text
runner.run actual calls = 2
budget 2 → 0

baseline = TECHNICAL_COMPLETE
treatment = TECHNICAL_COMPLETE

realized trace = 80 each
planner gate = 80/80 PASS each
actual LQR rows = 79 each
actual-shadow = 79/79 exact agreement
max command difference = 0
architecture failure audit = absent
```

但 frozen analyzer 发生：

```text
KeyError: capture_end_abs_s
```

所以历史 B1 状态永久保留：

```text
R2_BJ_B1_CANARY_INFRASTRUCTURE_FAILURE_STOPPED
```

不重跑。

## 8.2 B1.1 Offline Analyzer Schema Recovery

唯一允许修复：

```text
capture["capture_end_abs_s"]
→
capture["nominal_capture_end_abs_s"]
```

其它 analyzer 源码、阈值、gate 不变。

offline invocation：

```text
1
budget 1 → 0
runner.run = 0
```

恢复结果：

```text
CANARY_TECHNICAL_COMPLETE_MECHANISM_OR_ENDPOINT_FAIL
```

### Mechanism

```text
FAIL
TREATMENT_RETREAT_LT_ONE
MONOTONIC_PENALTY_LT_0P1

commit latency delta = +1.899923 s
monotonic delta = 0.0
```

当前 treatment 实现了更晚 commit，但没有形成 frozen Option-B 所要求的 detectable retreat。

### Endpoint

baseline：

```text
PASS
offset = 0.021593 m
heading error = 0.033338 rad
lateral velocity = 0.172622 m/s
```

treatment：

```text
FAIL
offset = 0.342582 m
heading error = 0.052064 rad
lateral velocity = 0.275049 m/s
route progress delta = 0.031520 m PASS
```

### F_match

```text
PASS
```

absolute deltas：

```text
mean speed = 0.002750
end-minus-start speed = 0.106555
path length = 0.011338
```

### Engineering

```text
PASS
```

treatment maxima：

```text
lateral acceleration = 0.805108 m/s²
yaw rate = 0.156025 rad/s
curvature = 0.030235 1/m
```

### Target capture

```text
capture-start offset = 2.536799 m
terminal offset = 0.342582 m
decline = PASS
post-deadline hard jump absent = PASS
```

因此：

> V4 target-center attraction 在方向上有效，但 frozen Primary80 horizon 内最终 settling 不足。

### Official safety

```text
FAIL

baseline at-fault collisions = 2
treatment at-fault collisions = 1
both drivable-area compliance = true
```

禁止解释成 treatment 更安全或更危险。

当前还应做的一个只读解释性检查：

> 将 collision timestamp 与 departure / retreat / recommit / capture window 对齐，判断 collision 是否会干扰对 mechanism failure 的 causal interpretation。

这不会改变 frozen safety FAIL，也不会重新开放 HLC canary。

---

# 9. 当前 HLC 科学状态：必须严格区分四个层次

## 9.1 已经成立

```text
HLC_V4_FRESH_CANARY = VALID_NEGATIVE_ENGINEERING_RESULT
HLC_V4_CURRENT_CANDIDATE = REJECTED
REMAINING_14_BJ_B_RUNS = NOT_AUTHORIZED
```

## 9.2 很有支持，但不能写成一般定理

开发链显示明显 tension：

```text
detectable retreat
+
controller/kinematic feasibility
+
Primary80 endpoint settling
```

但正式状态只能是：

```text
THREE_WAY_TENSION =
SUPPORTED_AS_WORKING_EXPLANATION
```

不能写：

```text
HLC_STRUCTURAL_IMPOSSIBILITY = PROVEN
```

原因：不同 architecture 版本使用不同 outcome-exposed development cohorts，未做 same-identity controlled architecture ablation。

## 9.3 不能声称

```text
HLC_SCIENTIFIC_CONSTRUCT = FAILED
HLC_IS_PHYSICALLY_IMPOSSIBLE
```

当前最多可以说：

> current HLC generator branch 经多轮治理约束开发后，在 fresh V4 canary 上仍未通过 frozen mechanism + endpoint 联合门禁。

## 9.4 当前项目决策建议

为了博士 scope 和避免继续基于 outcome-exposed development evidence 迭代：

```text
HLC_CURRENT_GENERATOR_BRANCH =
CLOSED_BY_SCOPE_AFTER_FRESH_CANARY_FAILURE
```

比写 `HLC_IS_IMPOSSIBLE` 更科学。

是否真正完全关闭 HLC，还是只允许“一次 fundamentally different final attempt”，应交给独立 Astra review。

---

# 10. TSB 当前状态

TSB 必须与 HLC 分开。

当前：

```text
TSB_FAMILY_DEVELOPMENT_CANDIDATE_FROZEN
PENDING_FRESH_R2C_VALIDATION
```

已有 DEV-CAL：

```text
8/8 measurement OK
8/8 baseline one-phase
8/8 treatment two-phase
8/8 mechanism
8/8 F_match
8/8 safety
```

禁止：

- 重新调 TSB 参数；
- 重跑 TSB DEV-CAL identities；
- 把 DEV-CAL 8/8 写成 fresh scientific confirmation。

未来合理路线：

```text
TSB candidate
→ fresh TSB-only R2-C
→ if PASS:
   TSB-specific residual benchmark confirmed
```

但 TSB-only 只能支持 longitudinal temporal residual claim。

不能自动恢复：

```text
FULL_RBR_GENERAL_RESIDUAL_BEHAVIOR_QUALIFICATION
```

未来应区分：

```text
TSB_SPECIFIC_RBR_QUALIFICATION
```

与：

```text
FULL_RBR_QUALIFICATION
```

如果论文希望声称广义 residual behavior，仍建议至少有第二个 independent residual family，或者明确缩小 claim。

---

# 11. RBR 正式方法：当前冻结原则

## 11.1 命名

正式名称：

```text
Residual Behavior Representation (RBR-64)
```

禁止 formalize 为 DriveDNA。

## 11.2 Representation 原则

RBR 不是：

```text
4 × 16 hard semantic blocks
```

而是：

```text
shared z64
+
frozen low-capacity task projections/readouts
```

handcrafted features是 semantic anchors，不是 latent geometry definition。

## 11.3 训练方向

candidate architecture 尚未冻结，但合理方向包括：

```text
raw temporal ego stream
+
interaction/context stream
+
multiscale temporal modeling
+
conditional fusion
+
attentive / mask-aware pooling
→ 64D
```

可考虑：

- TCN / temporal conv；
- Transformer / attention；
- ego/context dual stream；
- mask-aware pooling；
- valid-length aware representation。

但在 residual benchmark 未被 fresh confirmation 前：

```text
RBR_FORMAL_TRAINING = NOT_AUTHORIZED
```

---

# 12. Stage6 / Stage7 / Stage7L 冻结证据摘要

这些历史结果仍是论文基础，不要因为 Stage R 覆盖。

## 12.1 Stage7 M6.5

```text
310 complete scenario pairs
620 official rollouts
overall MMD² = 0.0044693963
paired p ≈ 1e-5
5/5 pre-treatment tasks pass Holm
```

## 12.2 Stage6J/K longitudinal

```text
183 same-scenario pairs
366 rollouts
Δ mean speed ≈ +0.915 m/s
Δ RMS accel ≈ +0.182 m/s²
old64 dose100 Z ≈ 9.23
25/50/75/100% realized gates pass
```

## 12.3 Stage6P unpaired

| Representation | A/B detection | A/A FPR | min bidirectional detection |
|---|---:|---:|---:|
| old64 | 66.5% | 5.0% | 62% |
| A | 90.5% | 3.0% | 90% |
| B | 100.0% | 5.0% | 100% |
| C | 99.5% | 6.5% | 99% |
| ego13 | 100.0% | 2.0% | 100% |

## 12.4 Stage6W

paired median Z：

```text
old64 ≈ 13.50
B ≈ 28.29
C ≈ 25.37
```

context-balanced unpaired signal：

```text
B ≈ 2.59× old64
C ≈ 2.64× old64
```

## 12.5 Stage6S-v3 interaction

```text
80/80 rollout
interaction mechanism PASS
C full Z ≈ 28.95
C neighbor-zero Z ≈ 36.81
ΔZ ≈ -7.85
95% CI ≈ [-33.39, 29.22]
```

正式解释：

> interaction mechanism positive confirmation + no demonstrated incremental context benefit for C under this frozen experiment。

## 12.6 Stage7L prospective pure-lateral

```text
80 scenarios
79 logs
400/400 rollouts

planner mechanism PASS
B Primary BDD FAIL
ego13 highly sensitive
```

Primary B：

```text
0.435802×
Z=-0.065037
p=0.411906
```

ego13：

```text
13.087068×
Z=40.201025
```

---

# 13. Git / provenance / 关键提交

当前 Stage R 分支：

```text
20260825_stageR_new
```

当前已知 remote HEAD：

```text
2f21b437a105067cfb19932ba7799fc4f4a40eca
stageR: record B1.1 offline recovery result
```

当前 tree：

```text
0f2173f63b96a670663387bbf9f2d49547c0e545
```

近期关键节点：

```text
1a626e98...  B2.9-E official R1 scientific smoke result
78f3a94c...  R1-B3 forensic
0007d51d...  R2-A controller transfer identification
16a93163...  R2-B controller-aware generator development
72941e78...  R2-BH target-capture V2 negative development
accbbb1a...  R2-BI kinematic V3 fail-closed
d745d770...  BJ-A offline V4 feasibility envelope
fb8a29ac...  A2 joint-support audit
1e7f9f78...  A3 prospective applicability audit
c1df6902...  B0.1 production execution path
39a0a536...  B0.2 actual-LQR observability freeze
5e8c5b31...  B1 canary infrastructure-stop result
2f21b437...  B1.1 offline recovery result
```

---

# 14. 受保护资产与永久限制

## 14.1 Protected CSV

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/
  behavior_events_v2/behavior_event_metrics_v2.csv
```

SHA256：

```text
e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8
```

不得覆盖、提交或清理。

## 14.2 永久禁止操作

禁止：

```text
git reset --hard
git clean
bulk delete outputs
blind git add .
```

禁止：

- 重训 old64/A/B/C；
- 事后换 primary seed；
- 用 outcome 调 MMD threshold；
- 重跑 R1 official；
- 重跑 R2-A / R2-B / BH / BI / B1 outcome-exposed identities；
- 重新选择 HLC canary identity；
- 调整 frozen HLC / TSB scientific thresholds；
- 因 failure 删除 unsafe / mechanism-fail identity；
- 将 engineering canary 结果写成 confirmatory science；
- 在 benchmark 未确认前启动 RBR formal training。

Unknown 必须写：

```text
UNKNOWN
NOT_FOUND
AMBIGUOUS
BLOCKED
```

不能猜。

---

# 15. Outcome-exposed / data firewall 原则

所有已经用于：

- R1 official；
- R1 B3 forensic；
- R2-A identification；
- R2-B calibration；
- R2-BH；
- R2-BI；
- BJ engineering canary；

的 identity，一旦 outcome 被观察：

```text
PERMANENT_ENGINEERING_ONLY / HISTORY_ONLY
```

禁止进入：

```text
future generator tuning
R2-C validation
confirmatory smoke
RBR scientific evidence
```

未运行但已被冻结为 engineering roster 的 identity，也不能自动回收进 future scientific roster；当前默认从严处理。

---

# 16. 当前最重要的未决科学问题

下一个独立 reviewer 不应默认接受已有 Sol/Work 结论，应重新审视：

1. Residual benchmark construct 本身是否 scientifically well-posed？
2. HLC `advance → retreat → recommit` 是否是合理 residual construct，还是过度人工？
3. 当前 fresh V4 canary 真正证明了什么、没有证明什么？
4. 一个 fresh canary pair 是否足以终止 HLC development？
5. 如果 HLC 停止：TSB-only 是否足以进入 RBR？是否应找第二 residual family？是否应缩小 thesis claim？
6. 如果 HLC 继续：最多只应考虑一个 fundamentally different final architecture attempt，不能继续 retreat/capture 参数微调。
7. 是否存在 post-selection / moving-goalpost risk，特别是 moving-regime speed floor、applicability narrowing、多 cohort development 与 TSB-only scope amendment。

---

# 17. 当前推荐的独立 Astra Review 任务

下一个 Astra Work conversation 应先做：

```text
READ_ONLY_INDEPENDENT_SCIENTIFIC_REVIEW
```

不要：

```text
edit code
run simulator
run runner.run
implement V5
select roster
change threshold
train RBR
```

Reviewer 重点回答：

1. R0→R1→R2 scientific strategy 是否成立；
2. residual benchmark construct 是否合理；
3. HLC Option-B 是否合理；
4. B1.1 canary 的证据强度与解释边界；
5. 一个 fresh canary pair 是否足以终止 HLC；
6. 当前 HLC failure 更像 implementation、controller incompatibility、horizon tension 还是 construct problem；
7. 如果继续，只提出一个 fundamentally different architecture；
8. 如果停止，TSB-only / second family / claim reduction 三条路线如何选；
9. TSB-only 能授权多大的 RBR claim；
10. data governance / post-selection / overfitting 风险；
11. 到 RBR training 与 thesis closure 的最短可靠路线。

每个主要结论应分类：

```text
SUPPORTED
PLAUSIBLE
NOT_ESTABLISHED
CONTRADICTED
```

---

# 18. 当前 Scientific Owner 暂定状态

在 Astra independent review 前，推荐暂时冻结为：

```text
R1_RESIDUAL_BENCHMARK_ENABLEMENT =
FAILED_UNDER_FROZEN_R1_CONTRACT

R2_A_CONTROLLER_TRANSFER_IDENTIFICATION =
COMPLETE

TSB_FAMILY_DEVELOPMENT_CANDIDATE =
FROZEN_PENDING_FRESH_R2C

HLC_V4_FRESH_CANARY =
VALID_NEGATIVE_ENGINEERING_RESULT

HLC_V4_CANDIDATE =
REJECTED

HLC_CURRENT_GENERATOR_BRANCH =
PAUSED_PENDING_INDEPENDENT_REVIEW

HLC_GENERAL_STRUCTURAL_IMPOSSIBILITY =
NOT_ESTABLISHED

REMAINING_14_HLC_RUNS =
NOT_AUTHORIZED

HLC_V5 =
NOT_AUTHORIZED

COMBINED_G_R2 =
NOT_AVAILABLE

TSB_ONLY_R2C =
DESIGN_ELIGIBLE_BUT_NOT_EXECUTION_AUTHORIZED

FULL_RBR_QUALIFICATION =
NOT_AUTHORIZED

TSB_SPECIFIC_RBR_QUALIFICATION =
PENDING_FRESH_TSB_R2C

RBR_FORMAL_TRAINING =
NOT_AUTHORIZED
```

---

# 19. 当前下一步

推荐顺序：

```text
更新 handover
    ↓
新开 Astra Work conversation
    ↓
read-only independent review
    ↓
回到 Scientific Owner 决策
    ↓
三选一：
A. HLC current branch正式归档 + TSB-only
B. HLC归档 + 新第二 residual family
C. HLC仅再允许一次 fundamentally different final attempt
    ↓
再决定是否授权 TSB R2-C / 第二 family / RBR
```

---

# 20. 最后检查清单

新的 session 在任何写操作前，应能回答：

1. Stage7L 为什么是 Stage R 的直接动机？
2. 为什么“known semantics decodable”不等于 representation 适合 BDD？
3. 为什么 residual benchmark 必须在 realized closed-loop behavior 层成立？
4. R1 为什么不是 infrastructure failure？
5. HLC R1 是 ATTENUATED，TSB R1 是 COLLAPSED，有什么区别？
6. R2-A 为什么证明 TSB attenuation主要在 generator/replanning→LQR，而非 LQR→vehicle？
7. TSB 当前为什么只能称 development candidate？
8. HLC R2-B 为什么是 mechanism 6/8 但 endpoint 0/8？
9. BH V2 为什么是 XY-heading/curvature interface bug？
10. BI V3 为什么 fail-closed？
11. V4 offline joint-support 为什么不能替代 fresh closed-loop canary？
12. B1 为什么技术 rollout有效但历史 scientific adjudication无效？
13. B1.1 为什么可以离线恢复而不能重跑？
14. B1.1 当前 frozen gate 的原始结果是什么？
15. 为什么当前 V4 canary可以否决 V4 candidate，但不能证明 HLC impossible？
16. 为什么剩余14 HLC runs仍然不应执行？
17. TSB-only若 fresh validation成功，能支持什么 RBR claim、不能支持什么？
18. 哪些 identity 已 outcome-exposed，绝对不能进入 confirmatory science？
19. protected CSV SHA 是否仍保持不变？
20. 当前是否仍然禁止 RBR formal training？

如果这些问题任一不清楚，先读本文件和 StageR 权威报告，不要启动 simulation、roster selection 或 RBR。

---

`CURRENT_STAGE_R_R2_HANDOVER_UPDATED_FOR_INDEPENDENT_REVIEW`
