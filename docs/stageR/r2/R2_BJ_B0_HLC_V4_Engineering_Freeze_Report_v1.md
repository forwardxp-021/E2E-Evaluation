# R2-BJ-B0 HLC V4 Engineering Roster 与零运行执行冻结报告

## 结论

本阶段完成了 outcome-blind 的 BJ-B engineering-only roster、8 个 pair 和 16 个 intended run 的冻结，并完成 16/16 官方 nuPlan 执行对象的零运行构造。状态为：

`R2_BJ_B0_ZERO_RUN_EXECUTION_PACKAGE_FROZEN_READY_FOR_CANARY_OWNER_REVIEW`

该状态只表示控制面与执行对象可构造，不构成任何运行授权。`BJ_B_ENGINEERING_SIMULATION_AUTHORIZED=false`、`CANARY_AUTHORIZED=false`、`NEW_RUN_BUDGET=0`、`RUNNER_RUN=0`。

## 选择结果

唯一候选池为 A5 provenance 中 34 条 `MOVING_REGIME_V4_APPLICABLE` 记录。选择只使用预注册 salt、scenario token、log ID、map、lane-change direction 和 `v_audit` speed band。未使用 V4 component margin、曲率/横摆率/横向加速度 margin、route coverage margin、闭环结果或人工轨迹判断。

精确重放得到字典序最小 selection-rank tuple：

`(1, 2, 3, 4, 5, 6, 7, 9)`

| roster | selection rank | token | direction | map | v_audit (m/s) | speed band |
|---:|---:|---|---|---|---:|---|
| 1 | 1 | `cc1abd3989065d8d` | right | Pittsburgh | 5.375735 | [3,6) |
| 2 | 2 | `1ec5332f171f5872` | right | Pittsburgh | 8.947916 | [6,9) |
| 3 | 3 | `5735883e91ef57e4` | right | Pittsburgh | 6.002119 | [6,9) |
| 4 | 4 | `7188743b78cb5631` | left | Las Vegas | 8.598139 | [6,9) |
| 5 | 5 | `57e7cbc7cc465554` | right | Pittsburgh | 9.331809 | [9,12) |
| 6 | 6 | `6408758176c25fb4` | right | Pittsburgh | 9.671629 | [9,12) |
| 7 | 7 | `c5a22460be1c59c2` | left | Pittsburgh | 11.828636 | [9,12) |
| 8 | 9 | `74801411f6d15a0d` | right | Pittsburgh | 14.134407 | [12,+∞) |

配额精确为 left/right = 2/6、Las Vegas/Pittsburgh = 1/7、四个速度带 = 1/3/3/1。token 与 log 均为 8/8 唯一，和已冻结历史及永久 exclusion 的交集为 0。

选择的 8 条立即进入 `PERMANENT_ENGINEERING_ONLY`，目前仍是 `ROSTER_FROZEN_NOT_YET_OUTCOME_EXPOSED`；永久禁止 R2-C、confirmatory smoke 和 RBR 使用。剩余 26 条保持 `UNSELECTED_OUTCOME_UNEXPOSED_POOL`，没有 reserve 或 replacement 顺序。

## Pair 与执行冻结

8 个 pair 各含 baseline、treatment，形成 16 个 intended runs；顺序严格为 selection rank，再按 baseline→treatment。每对双臂共享 scenario token/log、初始状态、V2.3 source/target reference SHA、route progression、pre-treatment physical context、TwoStageController/LQR 配置、seed 和 Primary80 控制器。唯一允许的双臂差异为冻结 V4 treatment morphology/capture。

零运行审计结果：

- A5 输入及 V4 关键 SHA：闭合；
- 16/16 full Hydra config：resolved；
- 16/16 exact official scenario resolution：1；
- 16/16 pair binding lookup：通过；
- 16/16 output path：新鲜且互不碰撞；
- 16/16 planner：`R2BJB0HLCV4EngineeringPlanner`；
- 16/16 time controller：`R1Primary80ScientificTimeControllerV1`，`number_of_iterations()=81`；
- 16/16 SimulationRunner：构造成功；
- 8/8 pairs、每对 11 个 `0.0...1.0 s` planner-call construction：baseline/treatment exact equal；
- simulation started：0；`runner.run` calls：0。

## 在线 fail-closed 门

新 runtime wrapper 不修改 V4 `_states` 或参数，只对同一输出进行记录与冻结门检查。每个 planner call 检查 curvature、yaw rate、lateral acceleration、state0 exact current pose、state0→state1 distance excess、state0 tangent mismatch、XY-heading consistency、terminal target-frame residual、rolling stitching horizon、controller-visible steering 的有限性，以及 pre-divergence 双臂全轨迹一致性。

任何架构门失败分类为 `ARCHITECTURE_FAILURE`，动作固定为 `STOP_CURRENT_RUN` 和 `STOP_REMAINING_SCHEDULE`；identity replacement 与 parameter update 均禁止。Hydra、scenario resolution、runner construction 和 output path 问题独立归为 `INFRASTRUCTURE_FAILURE`。本阶段没有预授权技术重跑。

## 下一步边界

本阶段只冻结未来 canary：selection rank 1、1 identity、1 pair、2 runs，顺序 baseline→treatment。必须由 Scientific Owner 在下一阶段绑定完整 component manifest、schedule、pair binding SHA 并给予正预算后，才能越过 simulator-start 前的最终授权门。

本阶段未启动 R2-C、confirmatory smoke、TSB 或 RBR，也未修改 V4、任何 scientific/kinematic threshold 或 protected CSV。
