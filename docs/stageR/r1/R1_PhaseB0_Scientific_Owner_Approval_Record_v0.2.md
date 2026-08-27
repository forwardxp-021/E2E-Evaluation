# R1 Phase B0 科学负责人批准记录 v0.2

状态：`APPROVED_AS_RECORDED`；作用域：`PROSPECTIVE_R1_ONLY`。

## 正式决定

- HLC mechanism × F_match：`MARGINALLY_FEASIBLE`；heading 与 retreat/recommit 机制存在结构重叠，但交集非空。
- 批准 HLC Amendment `OPTION_B`：未来 R1 HLC Primary F_match 仅包含 `ego13.mean_speed`、`ego13.end_minus_start_speed`、`ego13.path_length`；`ego13.heading_change_abs_total` 改为 `SECONDARY_MECHANISM_PROXIMAL_AUDIT`。
- HLC：`NO_IMPLEMENTATION_DEFINITION_BUG_CONFIRMED`，不因旧 0/6 diagnostic 启动 bug-fix。
- TSB：`JOINTLY_FEASIBLE`；旧 generator 根因为 `GENERATOR_PROFILE_REDESIGN_REQUIRED`，不是 implementation bug。
- 批准 `TSB_GEN_V2_OPTION_A` 进入参数冻结准备：`-0.9 m/s² × 0.5 s`、`+0.4 m/s² × 0.7 s`、`-0.9 m/s² × 0.5 s`。
- official nuPlan runtime：`NOT_READY`。
- 新 compliant 48-call smoke：`NOT_AUTHORIZED`。

## 不变量

本批准记录不修改 R0 v1.0、`R0_D4_Family_Specific_Matching_Contract_v0.1`、Wave3 D4 历史结论或冻结的 HLC/TSB mechanism thresholds；也不授权 roster selection、smoke 或 RBR training。
