# R1 HLC Generator 端点与有效性合同 v0.1

状态：`STRUCTURAL_RULES_READY_NUMERICAL_ENDPOINT_OPTIONS_PENDING_OWNER_APPROVAL`。

该合同独立于 Primary F_match，角色仅为 `GENERATOR_VALIDITY / SAFETY DEVELOPMENT DIAGNOSTICS`，不是 future R4 survivor-selection rule。

必须保持相同 source lane、intended target lane 和 lane-change direction，并要求 baseline/treatment 都完成目标 lane transition。每对轨迹还必须审计 terminal lateral offset、terminal heading error、terminal lateral velocity、route progress、phase continuity、curvature、lateral acceleration 与 yaw-rate。

已存在的 treatment-independent engineering limits 保持为：横向加速度 `≤6.0 m/s²`、yaw-rate `≤1.0 rad/s`、curvature `≤0.5 m⁻¹`。本阶段不新冻结 endpoint 数值 gate。

供 owner 下一轮选择的数值方案：

- `OPTION_ENDPOINT_STRICT`：offset `0.15 m`、heading `0.03 rad`、lateral velocity `0.15 m/s`、pair route-progress delta `1.0 m`。
- `OPTION_ENDPOINT_RESOLUTION_BASED`：offset `0.25 m`、heading `0.05 rad`、lateral velocity `0.25 m/s`、pair route-progress delta `1.5 m`。

两组值只来自 map geometry、0.1 s sampling resolution、vehicle kinematics 与既有 engineering limits，均为 `OWNER_REVIEW`，尚非 scientific gate。
