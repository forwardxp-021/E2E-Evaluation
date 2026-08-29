# R1 TSB Mechanism Applicability Evidence v0.1

状态：`PROPOSED_REQUIRES_OWNER_APPROVAL`。推导只使用 frozen dt=0.1 s、baseline/Option-A acceleration schedules 与 `LOW_SPEED_ENDSTOP` 定义，未读取 12 个 B2.1 initial speeds。

## 解析推导

baseline 在离散网格上包含 10 个 `-1.0 m/s² × 0.1 s` 积分区间，总速度损失为 1.0 m/s。由于制动后低速状态持续远超过 0.5 s，为避免 `speed < 1.0 m/s` 连续至少 0.5 s，必须满足 `v0 - 1.0 ≥ 1.0`，即 baseline floor 为 2.0 m/s。

Option-A 两段制动总损失 0.9 m/s，中间 release 恢复 0.28 m/s，净损失 0.62 m/s；对应 floor 为 1.62 m/s。两臂共同可评价取最大值，因此：

`proposed_initial_speed_floor_mps = 2.0`

## 合成穷举验证

在 `[0,4] m/s`、步长 0.001 m/s 的 outcome-blind synthetic grid 上，用原 frozen mechanism calculator 同时评价两臂；第一个 joint `OK` 网格点为 2.000 m/s，与解析结果一致。离散速度构造规范到 12 位，以保持十进制协议边界的确定性，不改变 threshold。

HLC 同样受 `<1 m/s for ≥0.5 s` applicability 约束，但 HLC speed 取决于 native geometry realization；当前证据不支持自动增加 initial-speed threshold，若需要数值必须由 owner 单独批准。
