# Standardized Fixed-Dimension BDD Evaluation Report

> 协议：`standardized_fixed_dimension_bdd_protocol_v1`
> 状态：`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`
> 边界：不训练、不改 checkpoint、不改 planner、不重选场景；Stage7 的新表示导出均为 `POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`，不会改写 Stage6V 联合结论。

## 1. 三类 Reference（必须分开读）

- **Behavior Reference**：每一行均明确 Reference planner/version 与 Target planner/version；semantic delta 统一为 **Target − Reference**。
- **Null Reference**：paired 行使用该 representation 自己的 pair-label-swap/randomization null；unpaired 行使用该 representation 自己独立的 A/A calibration。BDD 行均保留 null q95。
- **Representation Baseline**：old64 是历史 baseline。A/B/C/ego13 对 old64 的比较仅限检测能力与各自null标准化结果；**禁止用 raw MMD² 跨表示排序**。

## 2. 固定行为维度 × representation 主矩阵

单元格为 `BDD/null-q95 ratio / Z_BDD`。Best capability 按各自null内的 Z 与检出状态描述，绝不按 raw MMD² 排名。`N/A` 是证据缺口，不是没有差异。

| behavior_dimension   | old64           | A               | B               | C               | ego13            | best_capability                                      |
|:---------------------|:----------------|:----------------|:----------------|:----------------|:-----------------|:-----------------------------------------------------|
| 总体行为漂移               | 2.69× / Z=11.08 | 5.41× / Z=25.53 | 5.23× / Z=21.16 | 4.75× / Z=18.08 | 23.26× / Z=86.10 | ego13 (max within-null Z=86.10; no raw-MMD² ranking) |
| 自由流速度                | N/A             | N/A             | N/A             | N/A             | N/A              | N/A                                                  |
| 纵向加速/减速              | 2.39× / Z=9.23  | 2.65× / Z=10.79 | 2.74× / Z=10.33 | 2.40× / Z=8.48  | 8.87× / Z=35.09  | ego13 (max within-null Z=35.09; no raw-MMD² ranking) |
| 跟车行为                 | 1.85× / Z=5.63  | 1.61× / Z=5.43  | 1.72× / Z=5.25  | 1.59× / Z=4.61  | 4.81× / Z=18.74  | ego13 (max within-null Z=18.74; no raw-MMD² ranking) |
| 逼近前车响应               | 6.84× / Z=27.98 | 7.41× / Z=26.45 | 7.39× / Z=30.60 | 6.83× / Z=28.95 | 11.14× / Z=35.91 | ego13 (max within-null Z=35.91; no raw-MMD² ranking) |
| 纵向平顺性                | 2.39× / Z=9.23  | 2.65× / Z=10.79 | 2.74× / Z=10.33 | 2.40× / Z=8.48  | 8.87× / Z=35.09  | ego13 (max within-null Z=35.09; no raw-MMD² ranking) |
| 车道保持                 | N/A             | N/A             | N/A             | N/A             | N/A              | N/A                                                  |
| 变道行为                 | 2.04× / Z=7.44  | 2.81× / Z=10.86 | 2.50× / Z=9.12  | 2.66× / Z=9.15  | 6.66× / Z=22.80  | ego13 (max within-null Z=22.80; no raw-MMD² ranking) |
| 横向动态                 | 1.63× / Z=5.73  | 2.93× / Z=11.61 | 2.99× / Z=10.43 | 3.01× / Z=10.44 | 6.57× / Z=22.01  | ego13 (max within-null Z=22.01; no raw-MMD² ranking) |
| 前车间距/车头时距交互          | 6.84× / Z=27.98 | 7.41× / Z=26.45 | 7.39× / Z=30.60 | 6.83× / Z=28.95 | 11.14× / Z=35.91 | ego13 (max within-null Z=35.91; no raw-MMD² ranking) |
| 纵向跟车交互响应             | 6.84× / Z=27.98 | 7.41× / Z=26.45 | 7.39× / Z=30.60 | 6.83× / Z=28.95 | 11.14× / Z=35.91 | ego13 (max within-null Z=35.91; no raw-MMD² ranking) |
| 横向间隙接受/横向交互          | N/A             | N/A             | N/A             | N/A             | N/A              | N/A                                                  |
| 汇入/让行/切入响应           | 1.43× / Z=4.16  | 1.22× / Z=2.98  | 1.71× / Z=4.95  | 1.83× / Z=5.30  | 6.03× / Z=19.79  | ego13 (max within-null Z=19.79; no raw-MMD² ranking) |

## 3. 同一跟车工况：Stage6J/K dose100 following_interaction

Behavior Reference：`pdm_closed_longitudinal_conservative_v2 → pdm_closed_longitudinal_assertive_v2`；Null Reference：各 representation 的冻结 paired label-swap null；60个相同场景、52个相同log。跟车方向只由 speed/accel 语义解释，因此为 `TARGET_MORE_ACTIVE_FOLLOWING`，不写成 `CLOSER`。

| representation_id   |   raw_mmd2 |   null_q95 |   bdd_to_null_q95_ratio |    z_bdd |   raw_p_value |   corrected_p_value | detection_or_pass   |   n_pairs |   n_logs | semantic_delta_target_minus_reference                                                                      | semantic_direction           |
|:--------------------|-----------:|-----------:|------------------------:|---------:|--------------:|--------------------:|:--------------------|----------:|---------:|:-----------------------------------------------------------------------------------------------------------|:-----------------------------|
| old64               |  0.0170672 | 0.00922696 |                 1.84971 |  5.63432 |   0.000689993 |         0.00689993  | True                |        60 |       52 | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280] | TARGET_MORE_ACTIVE_FOLLOWING |
| A                   |  0.0106978 | 0.00665659 |                 1.6071  |  5.42684 |   0.000269997 |         0.00269997  | True                |        60 |       52 | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280] | TARGET_MORE_ACTIVE_FOLLOWING |
| B                   |  0.0127901 | 0.00742114 |                 1.72347 |  5.25451 |   0.00137999  |         0.0165598   | True                |        60 |       52 | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280] | TARGET_MORE_ACTIVE_FOLLOWING |
| C                   |  0.0126281 | 0.00794476 |                 1.58949 |  4.61431 |   0.00336997  |         0.0370696   | True                |        60 |       52 | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280] | TARGET_MORE_ACTIVE_FOLLOWING |
| ego13               |  0.0402803 | 0.00837334 |                 4.81055 | 18.7377  |   9.9999e-06  |         0.000119999 | True                |        60 |       52 | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280] | TARGET_MORE_ACTIVE_FOLLOWING |

## 4. 相同 interaction confirmation：Stage6S-v3（80对、11 log）

Behavior Reference：`pdm_closed_interaction_long_headway_v2 → pdm_closed_interaction_short_headway_v2`。front-gap、finite THW、closing acceleration、following-pressure acceleration 使用完全相同的冻结轨迹机制；THW仅为有限物理值，排除 sentinel/cap。每个 representation 的三条语义子行共享同一个 parent BDD，不能当作三次独立检验。

| representation_id   |   raw_mmd2 |   null_q95 |   bdd_to_null_q95_ratio |   z_bdd |   raw_p_value | detection_or_pass   | semantic_delta_target_minus_reference                                                | semantic_direction                         | evidence_status                                     |
|:--------------------|-----------:|-----------:|------------------------:|--------:|--------------:|:--------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------|:----------------------------------------------------|
| old64               |  0.0632019 | 0.00924585 |                 6.83571 | 27.9764 |    9.9999e-06 | True                | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT            |
| A                   |  0.0491725 | 0.00663854 |                 7.40712 | 26.4538 |    9.9999e-06 | True                | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT            |
| B                   |  0.0470387 | 0.00636297 |                 7.39257 | 30.603  |    9.9999e-06 | True                | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT            |
| C                   |  0.0393332 | 0.00576273 |                 6.82544 | 28.9549 |    9.9999e-06 | True                | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT            |
| ego13               |  0.237887  | 0.0213538  |                11.1402  | 35.905  |    9.9999e-06 | True                | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT            |
| C_neighbor_zero     |  0.137493  | 0.0114619  |                11.9957  | 36.8066 |    9.9999e-06 | True                | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | DIAGNOSTIC_C_NEIGHBOR_ZERO_NOT_A_MAIN_MATRIX_COLUMN |

C-neighbor-zero 的既有诊断：`C full − C neighbor-zero ΔZ = -7.852`，log-cluster 95% CI `[-33.393, 29.219]`，增量 interaction gate = `False`。这不改变主矩阵中的 C 列。

## 5. 同一变道场景 slice：Stage7（事后描述性）

Behavior Reference：`pdm_closed_conservative_v1 → pdm_closed_assertive_v1`；60个预处理scenario_type为 changing_lane 的固定场景。它是 lane-change **场景切片**，不自动证明ego完成了变道；semantic direction 因此保持限制性表述。

| representation_id   |   raw_mmd2 |   null_q95 |   bdd_to_null_q95_ratio |    z_bdd |   raw_p_value |   corrected_p_value | detection_or_pass   | semantic_delta_target_minus_reference                   | semantic_direction                                   |
|:--------------------|-----------:|-----------:|------------------------:|---------:|--------------:|--------------------:|:--------------------|:--------------------------------------------------------|:-----------------------------------------------------|
| old64               |  0.0287843 | 0.0140866  |                 2.04338 |  7.43819 |   6.99993e-05 |         0.000349997 | True                | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023] | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE |
| A                   |  0.0256197 | 0.00911842 |                 2.80967 | 10.863   |   9.9999e-06  |         4.99995e-05 | True                | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023] | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE |
| B                   |  0.0256236 | 0.0102481  |                 2.50033 |  9.12016 |   9.9999e-06  |         4.99995e-05 | True                | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023] | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE |
| C                   |  0.0276068 | 0.0103965  |                 2.6554  |  9.14775 |   9.9999e-06  |         4.99995e-05 | True                | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023] | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE |
| ego13               |  0.0693422 | 0.0104165  |                 6.65699 | 22.8026  |   9.9999e-06  |         4.99995e-05 | True                | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023] | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE |

## 6. 业务 Style Report Card（Primary contrast：Conservative → Assertive）

下表使用 B 作为当前最简单的 learned release-level candidate，用于让业务读者看到固定contrast中各切片的差异。它不是‘B优于所有representation’的证明，详见主矩阵。

| behavior_dimension   | contrast_label                                       | representation_id   |   bdd_to_null_q95_ratio |    z_bdd | corrected_p_value       | semantic_delta_target_minus_reference                                                                              | semantic_direction                                          | evidence_status                              |
|:---------------------|:-----------------------------------------------------|:--------------------|------------------------:|---------:|:------------------------|:-------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:---------------------------------------------|
| 总体行为漂移               | pdm_closed_assertive_v1 | pdm_closed_conservative_v1 | B                   |                 5.23341 | 21.1623  | N/A_DESCRIPTIVE_OVERALL | mean_speed +1.281 m/s; 95% CI [+1.120, +1.448]; rms_accel +0.235 m/s²; 95% CI [+0.206, +0.264]                     | MIXED_NO_SINGLE_STYLE_DIRECTION                             | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 跟车行为                 | pdm_closed_assertive_v1 | pdm_closed_conservative_v1 | B                   |                 1.42034 |  3.8229  | 0.011979880201197989    | following_mean_speed +0.952 m/s; 95% CI [+0.667, +1.280]; following_rms_accel +0.246 m/s²; 95% CI [+0.193, +0.298] | TARGET_MORE_ACTIVE_FOLLOWING                                | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 变道行为                 | pdm_closed_assertive_v1 | pdm_closed_conservative_v1 | B                   |                 2.50033 |  9.12016 | 4.9999500004999955e-05  | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023]                                                            | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE        | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 横向动态                 | pdm_closed_assertive_v1 | pdm_closed_conservative_v1 | B                   |                 2.98926 | 10.4288  | 7.999920000799993e-05   | mean_abs_yaw_rate +0.055 rad/s; 95% CI [+0.019, +0.120]                                                            | TARGET_HIGHER_LATERAL_EXCITATION_PROXY                      | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 汇入/让行/切入响应           | pdm_closed_assertive_v1 | pdm_closed_conservative_v1 | B                   |                 1.70731 |  4.94658 | 0.0064199358006419936   | mean_front_distance +0.062 m; 95% CI [-3.355, +3.753]                                                              | N/A_DENSE_OR_VULNERABLE_PROXY_NOT_A_MERGE_YIELD_CUTIN_EVENT | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |

## 7. Representation gate 分拆（不再使用模糊 frozen_gate_result）

| representation_id   | representation_baseline                      | stage6jk_paired_gate_pass   | stage6p_unpaired_gate_pass   | waymo_gate_pass                     | interaction_increment_gate_pass   | stage6v_joint_candidate_gate_pass   |
|:--------------------|:---------------------------------------------|:----------------------------|:-----------------------------|:------------------------------------|:----------------------------------|:------------------------------------|
| old64               | old64                                        | False                       | False                        | N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE | N/A_C_ONLY_DIAGNOSTIC             | N/A_NOT_ABC_CANDIDATE               |
| A                   | compared_to_old64_by_capability_not_raw_mmd2 | False                       | True                         | False                               | N/A_C_ONLY_DIAGNOSTIC             | False                               |
| B                   | compared_to_old64_by_capability_not_raw_mmd2 | False                       | True                         | False                               | N/A_C_ONLY_DIAGNOSTIC             | False                               |
| C                   | compared_to_old64_by_capability_not_raw_mmd2 | False                       | True                         | False                               | False                             | False                               |
| ego13               | compared_to_old64_by_capability_not_raw_mmd2 | True                        | True                         | N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE | N/A_C_ONLY_DIAGNOSTIC             | N/A_NOT_ABC_CANDIDATE               |

## 8. 确认性与事后描述性边界

- **原预冻结确认性证据**：Stage6J/K dose-response paired 及 Stage6S-v3 interaction confirmation；它们的任务、样本、null与统计均沿用冻结输出。
- **事后标准化描述性证据**：Stage7 old64/A/B/C/ego13 共用既有310对assertive/conservative rollout、固定pre-treatment task membership、primary seed3407及固定100,000次pair swap。其目的只是补齐同工况横向矩阵，不能取代Stage6V endpoint，也不能触发训练返工。
- **unpaired release**：Stage6P属于representation scorecard，不被伪装成某个方向的行为画像；A/A FPR和detection仍单独解释。

## 9. Evidence gaps / N/A

| dimension_id        | behavior_dimension   | evidence_status                   | missing                                      |
|:--------------------|:---------------------|:----------------------------------|:---------------------------------------------|
| LON.FREE_FLOW_SPEED | 自由流速度                | N/A_NO_FROZEN_FREE_FLOW_SLICE     | 没有冻结的同维度场景、BDD及绑定semantic delta；N/A不等同于没有差异。 |
| LAT.LANE_KEEPING    | 车道保持                 | N/A_NO_FROZEN_LANE_KEEPING_SLICE  | 没有冻结的同维度场景、BDD及绑定semantic delta；N/A不等同于没有差异。 |
| INT.LATERAL_GAP     | 横向间隙接受/横向交互          | N/A_NO_FROZEN_LATERAL_GAP_OUTCOME | 没有冻结的同维度场景、BDD及绑定semantic delta；N/A不等同于没有差异。 |

## 10. 直接回答

1. **同一跟车工况**：第3节列出old64/A/B/C/ego13逐一的 raw MMD²、null q95、ratio、Z、p与Holm；这是同一60对条件，允许比较各自相对null的检测强度，不允许比较raw MMD²大小。
2. **同一纵向工况**：主矩阵的`纵向加速/减速`来自Stage6J/K dose100 overall；完整25/50/75/100与四个scope的逐representation行在`standardized_bdd_long.csv`中保留。
3. **同一变道工况**：第5节给出Stage7固定60对场景slice的所有representation BDD。该证据是post-hoc descriptive，且不足以声称已验证ego executed lane-change差异。
4. **interaction工况**：第4节给出Stage6S-v3相同80对的逐representation BDD；其轨迹机制已先行通过。C不具有相对于C-neighbor-zero的已证实增量interaction信息。
5. **Reference定义**：每条长表行都分别携带behavior_reference、target、null_reference和representation baseline语境。
6. **每维最可靠表示**：主矩阵给出按within-representation Z/检出描述的best capability；不构成universal representation排名。
7. **结论边界**：Stage6J/K、Stage6S-v3是继承的确认性结果；Stage7全表示矩阵是事后描述性。
8. **完整矩阵**：13个固定维度均已出现；无法支持的维度保持N/A并列出证据缺口。

`STANDARDIZED_FIXED_DIMENSION_BDD_MATRIX_COMPLETE`
