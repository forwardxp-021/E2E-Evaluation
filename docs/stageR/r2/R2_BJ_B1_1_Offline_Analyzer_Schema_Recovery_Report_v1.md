# R2-BJ-B1.1 离线 Analyzer Schema Recovery 报告

## 结论

本阶段唯一一次离线恢复分析已完成，未构造 runner、未启动 simulator，也未调用 `runner.run()`。恢复结论为：

`R2_BJ_B1_1_OFFLINE_RECOVERY = CANARY_TECHNICAL_COMPLETE_MECHANISM_OR_ENDPOINT_FAIL`

历史结论继续永久保留：

`HISTORICAL_B1_STATE = R2_BJ_B1_CANARY_INFRASTRUCTURE_FAILURE_STOPPED`

恢复结果是独立 disposition，不覆盖、不 supersede 历史 B1 analyzer 结果或 control artifacts。

## Pre-recovery 冻结点

- 本地 commit：`27bbe910865e79000bfba16ea512b896109cbfcf`
- 远端 commit：`c88554ea2adb87ee10342ae40ba8de984859564b`
- 共同 tree：`4feee85345cf6eb4450dba81f512f3e34ef70426`
- pre-recovery manifest SHA256：`204a439b790e9fe06678fc438b4d7ffafee056019c7917ed537a0351ec29a444`
- recovery analyzer SHA256：`e7becddeeb6baea998aee9129e838c36808c2bc73e66a3dd7a9be733964c6788`

Recovery analyzer 与原 analyzer 只有一行差异：deadline 字段从 `capture_end_abs_s` 接到冻结 V4 schema 的 `nominal_capture_end_abs_s`。没有 fallback、字段猜测、阈值改动或 scientific gate 改动。

## Invocation 与结果完整性

- B1 manifest 所列 component、raw、trace、telemetry、metric、control artifacts：SHA closure PASS
- offline analyzer invocation：1
- remaining offline budget：0
- analyzer 内置 deterministic double evaluation：PASS
- 完整结果 SHA256：`761532070a61dc744e742c212c72c24e3d36bc8fee12b44805f63e41824e16ed`
- `runner.run()`：0
- engineering/scientific/TSB simulation：0/0/0

## 冻结 gate 原样结果

| Gate | 结果 | 冻结输出摘要 |
|---|---|---|
| Mechanism | FAIL | `TREATMENT_RETREAT_LT_ONE`、`MONOTONIC_PENALTY_LT_0P1`；commit latency delta `1.899923 s`；monotonic delta `0.0` |
| Endpoint | FAIL | baseline PASS；treatment 的 offset、heading、lateral velocity FAIL，route progress delta PASS |
| F_match | PASS | mean speed、end-minus-start speed、path length 均 PASS |
| Engineering | PASS | 两臂 lateral acceleration、yaw rate、curvature 均在冻结 limits 内 |
| Official safety | FAIL | baseline at-fault collisions `2`；treatment `1`；两臂 drivable-area compliance 均为 true |
| Actual-shadow observability | PASS | 两臂 79/79，一致性最大绝对差为 0 |
| Treatment target-offset decline | PASS | capture-start `2.536799 m` → terminal `0.342582 m` |
| Post-deadline hard jump absent | PASS | 两臂均未触发冻结 `0.25 m` 判据 |

Endpoint 数值：baseline terminal offset `0.021593 m`、heading error `0.033338 rad`、lateral velocity `0.172622 m/s`；treatment 分别为 `0.342582 m`、`0.052064 rad`、`0.275049 m/s`。paired route-progress delta 为 `0.031520 m`，通过 `1.5 m` 冻结上限。

F_match absolute deltas 为 mean speed `0.002750`、end-minus-start speed `0.106555`、path length `0.011338`，均低于原冻结 caliper。

Engineering maxima：baseline lateral acceleration/yaw rate/curvature 为 `0.520186 / 0.100484 / 0.019410`；treatment 为 `0.805108 / 0.156025 / 0.030235`，对应冻结上限保持 `6.0 / 1.0 / 0.5`。

## 处置边界

本次结果未达到 remaining cohort readiness。剩余 14 runs、R2-C、confirmatory smoke 和 RBR 均继续未授权。不得再次调用 offline recovery analyzer，不得技术重跑、替换 identity、修改参数或阈值。
