# R1 官方合规技术 Smoke 一次性授权 v1.0

状态：`AUTHORIZED_ONCE_AFTER_ZERO_BUDGET_PREFLIGHT`。

本授权仅对应 R1 Phase B2 的 fresh official compliant technical smoke；它在 0/48 的 preflight 通过后生成，绑定本地提交 `bbba281…`、同树远端提交 `9cf47f89…`、24-scenario roster、V3 replay contract、官方 metric canonicalizer、冻结 context/mechanism、HLC/TSB generator、planner、executor 与 simulation config。

## 唯一允许日程

- R-HLC：12 个 fresh scenarios，每个仅 `HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE` 与 `HLC_TREATMENT_HLC_GEN_V2_OPTION_B`。
- R-TSB：12 个 fresh scenarios，每个仅 `TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING` 与 `TSB_TREATMENT_TSB_GEN_V2_OPTION_A`。
- 总预算为 48 个 `OFFICIAL_CLOSED_LOOP_RUN`；每次 simulation 前 claim，第 49 次必须在 simulation 前拒绝。

## 停止与保留规则

任何 official command、trace、binding、DB/map/route、metric/context canonicalization 或 run identity 的技术失败，立即停止整个 smoke；不得替换 scene、重跑第三次或改 config。

mechanism、F_match、endpoint、engineering 或官方 safety 未通过不是技术失败：必须保留该 pair 并继续执行完整的冻结日程，不能删 pair、换 candidate、调 threshold 或改 generator。

本授权不授权 formal R1 development rollout、RBR-A/B/C training、representation/BDD/probe 读取或任何新 planner rollout。
