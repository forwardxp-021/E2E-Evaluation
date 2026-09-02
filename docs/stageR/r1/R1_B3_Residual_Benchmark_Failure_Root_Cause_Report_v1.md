# R1 B3 Residual Benchmark Failure Root Cause Report v1

## 结论

`R1_RESIDUAL_BENCHMARK_ENABLEMENT = FAILED_UNDER_FROZEN_R1_CONTRACT` 保持不变。本阶段只读加载 B2.9-E 的 24 个冻结 evaluator 结果，并用冻结函数进行离线机制诊断；没有 simulation、rerun、selector、阈值或 generator 修改。

HLC 的 generator intent 在理想轨迹上满足冻结机制，但 closed-loop realized monotonic effect 被系统性削弱：12/12 pair 的 status、retreat 与 latency 条件均通过，12/12 唯一失败均为 `MONOTONIC_PENALTY_LT_0P1`。TSB 的理想 schedule 可产生 1-vs-2 brake phases，但正式 realized 轨迹中 baseline 与 treatment 均为 12/12 `NO_BRAKE_PHASE`，主要是 intended brake windows 的减速度幅值未达到冻结 `-0.80 m/s²` phase threshold，而不是 low-speed end-stop。

## 输入闭包

- 24/24 evaluator 与 committed pair gate table 一致；科学结果没有被重新计算或替代。
- B2.9-E raw output SHA：1080/1080 文件通过，mismatch=0。
- `simulation=0`、`runner.run=0`、`run_runners=0`、RBR=0。

## HLC realized mechanism

| 指标 | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| baseline monotonic | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| treatment monotonic | 0.905146 | 0.913578 | 0.927766 | 0.932607 | 0.950015 |
| delta monotonic | -0.094854 | -0.086422 | -0.072234 | -0.067393 | -0.049985 |
| latency delta (s) | 3.099219 | 3.099752 | 3.10004 | 3.125692 | 3.200307 |
| monotonic transfer ratio | 0.392646 | 0.529392 | 0.567418 | 0.678869 | 0.745104 |

理想 generator：baseline monotonic=1.0，treatment monotonic=0.872697，delta=-0.127303，相对冻结 -0.10 gate 的 margin=-0.027303。该项仅为 `ANALYTICAL_GENERATOR_INTENT_DIAGNOSTIC_ONLY`。

HLC endpoint 为 6/12 PASS；6 个失败中 5 个仅由 treatment terminal lateral velocity 触发，1 个仅由 treatment terminal offset 触发。heading 与 paired native route progress 失败均为 0。

state1→realized 对照的 tracking projection gain 中位数：baseline=0.758696，treatment=0.74303；最佳相关 lag 中位数分别为 3.5 与 4.0 个 nominal samples。

## TSB realized mechanism

- baseline status：`NO_BRAKE_PHASE=12`、`LOW_SPEED_ENDSTOP=0`、`OK=0`。
- treatment status：`NO_BRAKE_PHASE=12`、`LOW_SPEED_ENDSTOP=0`、`OK=0`。
- baseline/treatment brake phase count 均为 12 个 0-phase。
- intended-window peak decel（min/p25/median/p75/max）：baseline `0.755252/0.766246/0.768126/0.768714/0.77638`；treatment first `0.222297/0.229697/0.249224/0.264614/0.266804`；treatment second `0.244677/0.331599/0.3335/0.333564/0.335647` m/s²。
- treatment release window descriptive realization：12/12；release 存在，但前后两个 brake phase 的幅值没有形成冻结 measurement phase。
- ideal generator：baseline 1 phase、peak 1.0 m/s²；treatment 2 phases、first/second peak 0.9 m/s²、release fraction 0.333333、second peak ratio 1.0。仅为 `GENERATOR_INTENT_DIAGNOSTIC_ONLY`。
- state1→realized tracking projection gain 中位数：baseline=0.582744，treatment=0.164061；treatment command transfer 相对冻结 phase semantics 已 collapsed。

## F_match 与 safety

HLC 12/12、TSB 12/12 F_match PASS，因此 `HANDCRAFTED_NUISANCE_MATCHING = SUCCESSFUL`，caliper 未重定义。

Safety 仅失败 2 pair：HLC pair01 为 baseline drivable-area noncompliance（collision=0），treatment 通过；TSB pair21 baseline 为 collision=1 且 drivable=false，treatment 为 collision=2 且 drivable=false。该模式不是 12/12 family-wide systematic blocker。

## Root-cause classification

| 判定项 | HLC | TSB |
|---|---|---|
| GENERATOR_INTENT_VALID | YES | YES |
| PLANNER_INTENT_TO_REALIZED_TRANSFER | ATTENUATED | COLLAPSED |
| MEASUREMENT_IMPLEMENTATION_ERROR | NOT_SUPPORTED | NOT_SUPPORTED |
| F_MATCH_CONTROL | PASS | PASS |
| ENDPOINT_SETTLING | PARTIAL | NOT_APPLICABLE_BY_FROZEN_TSB_CONTRACT |
| SAFETY_SYSTEMATIC_BLOCKER | NO | NO |

冻结 measurement 实现与 evaluator 输出逐 pair exact parity，且 observed numeric values 能直接解释 gate failure，因此当前证据不支持 measurement implementation error。

## Governance

建议的 R2 architecture repair family 是 `CONTROLLER_AWARE_TRAJECTORY_SHAPING + FEEDBACK_CALIBRATED_GENERATOR`，但任何 numerical amplitude/duration calibration 必须使用 fresh、永久 engineering-only identities。24 个 R1 official identities 已 outcome-exposed，只能用于只读历史失败诊断，禁止参与 R2 development、model selection 或 confirmatory smoke。

`MEASUREMENT_THRESHOLD_RELAXATION = NOT_RECOMMENDED_FROM_R1_OUTCOME`。`RBR_FORMAL_TRAINING = NOT_AUTHORIZED`。
