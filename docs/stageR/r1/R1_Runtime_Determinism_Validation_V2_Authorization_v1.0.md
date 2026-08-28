# R1 Runtime Determinism Validation V2 授权 v1.0

scientific owner 已一次性授权 B1.2 的 replacement official runtime determinism validation：最多新增
8 个 `OFFICIAL_CLOSED_LOOP_RUN`，固定为原 outcome-blind roster 的 4 个 scenario 各执行 `V2_RUN_A` 与
`V2_RUN_B` 一次。

唯一允许的 arm 为：R-HLC 的 `DECISIVE_MONOTONIC_LANE_CHANGE` baseline 与 R-TSB 的
`SINGLE_CONTINUOUS_BRAKING` baseline。treatment、48-call smoke、roster re-selection、RBR training 均未获授权。

V1 的 `R-HLC__25944935eadb52f1__RUN_A` 永久保留为 `TECHNICAL_FAILURE / HISTORICAL_FAILED_EXECUTION`；
它不构成 V2 evidence，也不计入 V2 的 8-run cap。V2 必须先通过零预算的 interface preflight；若任一
V2 official run 失败，整批立即停止并需再次 owner review。

完整 SHA binding（包括修复后的 planner 与 V2 专用执行器）、roster 不可变性和历史失败排除规则见同名 JSON。
