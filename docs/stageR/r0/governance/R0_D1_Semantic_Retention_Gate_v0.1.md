# R0 D1 Semantic Retention Gate v0.1

## Frozen decision contract

`FROZEN_FOR_R0_V1_PROTOCOL`。CORE targets 在任何新 representation evaluation 前按语义覆盖与既有 development target support 选定；未使用 Stage7L outcome、embedding、BDD 或 probe outcome。

| Family | CORE targets | Family pass |
|---|---|---|
| longitudinal | `ego13.mean_speed`, `ego13.end_minus_start_speed`, `ego13.rms_accel` | 至少 2/3 |
| lateral | `ego13.rms_yaw_rate`, `ego13.heading_change_abs_total`, `raw33.lane_change_count_proxy -> any_count_gt_0` | 至少 2/3 |
| interaction | `raw33.mean_front_distance`, `raw33.mean_rel_speed`, `raw33.front_pressure_score` | 至少 2/3 |

连续 target 使用 log/source-grouped held-out linear ridge：Primary `R² >= 0.10` 且 log-cluster 95% CI lower bound `>0`；同时必须报告 MAE/NRMSE、Spearman 与 calibration slope。分类 target 使用 grouped linear logistic：balanced accuracy `>=0.60` 且 95% CI lower bound `>0.50`；同时报告 AUROC 与 macro-F1。

模块级 `D1_KNOWN_SEMANTIC_INFORMATION_PRESENT=SUPPORTED` 要求至少 2/3 semantic families 在至少两个 learned representation families 中通过；A/B/C 各自要求至少 2/3 seeds 方向一致。old64 单 seed 只能作 descriptive corroboration。

少于 30 个独立 log/source groups，或分类 target 任一类少于 50 个独立 groups，结果必须为 `INCONCLUSIVE`。单一 probe failure、样本不足或 CI 过宽不得解释为 information absent。没有 R0_AUDIT_HOLDOUT 时，所有结果仍限定为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。

机器合同：`docs/stageR/r0/manifests/r0_d1_core_semantic_targets_v0.1.json`。
