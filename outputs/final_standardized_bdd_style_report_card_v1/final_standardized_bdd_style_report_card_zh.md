# Final Standardized BDD Style Report Card

> 最终协议：`standardized_fixed_dimension_bdd_protocol_v2_final_render_only`
> 最终状态：`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`
> 本报告只重新组织已冻结数值；未训练、未仿真、未导出新embedding、未重算BDD、未改变Stage6V结论。

## 第一页：Behavior Drift / Style Report Card

### 报告身份

- **Behavior Reference**：逐行列出；回答谁相对谁发生变化。
- **Target**：逐行列出；所有semantic delta固定为`Target − Behavior Reference`。
- **Evaluation mode**：逐行列出paired/unpaired；本页现有固定行为对比均为paired。
- **Primary Representation**：`B`。
- **Null Reference**：逐行绑定B自己的paired randomization q95；`BDD/null-q95 = 1.0×`是统计背景参考线。
- **身份边界**：B是用于测量行为漂移的representation，**不是**被评价的planner/version本身。

本页只回答Target相对Behavior Reference发生了哪些行为变化。不同来源的固定treatment不被混写为一个Behavior Reference；每行均保留自己的Reference、Target与Null Reference。

| behavior_dimension   | behavior_reference                      | target                                  | evaluation_mode   | null_reference                                                                                    | n_scenarios   | n_logs   | BDD/null-q95   | Z_BDD   | significance   | semantic_delta_target_minus_reference                                                                                   | semantic_direction                                          | evidence_status                              |
|:---------------------|:----------------------------------------|:----------------------------------------|:------------------|:--------------------------------------------------------------------------------------------------|:--------------|:---------|:---------------|:--------|:---------------|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:---------------------------------------------|
| 总体行为漂移               | pdm_closed_conservative_v1              | pdm_closed_assertive_v1                 | paired            | within_scenario_pair_label_swap; common seeded sign stream across representations                 | 310           | 257      | 5.23×          | 21.16   | 显著             | mean_speed +1.281 m/s; 95% CI [+1.120, +1.448]; rms_accel +0.235 m/s²; 95% CI [+0.206, +0.264]                          | MIXED_NO_SINGLE_STYLE_DIRECTION                             | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 自由流速度                | N/A                                     | N/A                                     | N/A               | N/A                                                                                               | N/A           | N/A      | N/A            | N/A     | N/A            | N/A                                                                                                                     | N/A                                                         | N/A_NO_FROZEN_FREE_FLOW_SLICE                |
| 纵向加速/减速              | pdm_closed_longitudinal_conservative_v2 | pdm_closed_longitudinal_assertive_v2    | paired            | representation-specific paired within-scenario label-swap randomization inherited from Stage6J/K  | 183           | 156      | 2.74×          | 10.33   | 显著             | delta_mean_speed +0.915 m/s; 95% CI [+0.758, +1.078]; delta_rms_accel +0.182 m/s²; 95% CI [+0.146, +0.217]              | TARGET_HIGHER_LONGITUDINAL_EXCITATION                       | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT     |
| 跟车行为                 | pdm_closed_longitudinal_conservative_v2 | pdm_closed_longitudinal_assertive_v2    | paired            | representation-specific paired within-scenario label-swap randomization inherited from Stage6J/K  | 60            | 52       | 1.72×          | 5.25    | 显著             | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280]              | TARGET_MORE_ACTIVE_FOLLOWING                                | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT     |
| 逼近前车响应               | pdm_closed_interaction_long_headway_v2  | pdm_closed_interaction_short_headway_v2 | paired            | representation-specific paired within-scenario label-swap randomization inherited from Stage6S-v3 | 80            | 11       | 7.39× †        | 30.60   | 显著             | delta_mean_accel_during_closing_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450]                                               | TARGET_MAINTAINS_MORE_ACCEL_DURING_CLOSING                  | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT     |
| 纵向平顺性                | pdm_closed_longitudinal_conservative_v2 | pdm_closed_longitudinal_assertive_v2    | paired            | representation-specific paired within-scenario label-swap randomization inherited from Stage6J/K  | 183           | 156      | 2.74×          | 10.33   | 显著             | delta_rms_jerk +0.228 m/s³; 95% CI [+0.142, +0.319]                                                                     | TARGET_HIGHER_LONGITUDINAL_JERK                             | SHARED_PARENT_BDD_SEMANTIC_PROXY             |
| 车道保持                 | N/A                                     | N/A                                     | N/A               | N/A                                                                                               | N/A           | N/A      | N/A            | N/A     | N/A            | N/A                                                                                                                     | N/A                                                         | N/A_NO_FROZEN_LANE_KEEPING_SLICE             |
| 变道行为                 | pdm_closed_conservative_v1              | pdm_closed_assertive_v1                 | paired            | within_scenario_pair_label_swap; common seeded sign stream across representations                 | 60            | 60       | 2.50×          | 9.12    | 显著             | mean_abs_yaw_rate +0.018 rad/s; 95% CI [+0.012, +0.023]                                                                 | N/A_TASK_SLICE_DOES_NOT_CONFIRM_EXECUTED_LANE_CHANGE        | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 横向动态                 | pdm_closed_conservative_v1              | pdm_closed_assertive_v1                 | paired            | within_scenario_pair_label_swap; common seeded sign stream across representations                 | 60            | 59       | 2.99×          | 10.43   | 显著             | mean_abs_yaw_rate +0.055 rad/s; 95% CI [+0.019, +0.120]                                                                 | TARGET_HIGHER_LATERAL_EXCITATION_PROXY                      | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 前车间距/车头时距交互          | pdm_closed_interaction_long_headway_v2  | pdm_closed_interaction_short_headway_v2 | paired            | representation-specific paired within-scenario label-swap randomization inherited from Stage6S-v3 | 80            | 11       | 7.39× †        | 30.60   | 显著             | delta_median_front_gap_m -4.202 m; 95% CI [-5.791, -1.181]; delta_median_finite_thw_s -2.670 s; 95% CI [-3.687, -2.275] | TARGET_SHORTER_GAP_OR_THW                                   | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT     |
| 纵向跟车交互响应             | pdm_closed_interaction_long_headway_v2  | pdm_closed_interaction_short_headway_v2 | paired            | representation-specific paired within-scenario label-swap randomization inherited from Stage6S-v3 | 80            | 11       | 7.39× †        | 30.60   | 显著             | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450]                                    | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE                  | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT     |
| 横向间隙接受/横向交互          | N/A                                     | N/A                                     | N/A               | N/A                                                                                               | N/A           | N/A      | N/A            | N/A     | N/A            | N/A                                                                                                                     | N/A                                                         | N/A_NO_FROZEN_LATERAL_GAP_OUTCOME            |
| 汇入/让行/切入响应           | pdm_closed_conservative_v1              | pdm_closed_assertive_v1                 | paired            | within_scenario_pair_label_swap; common seeded sign stream across representations                 | 63            | 57       | 1.71×          | 4.95    | 显著             | mean_front_distance +0.062 m; 95% CI [-3.355, +3.753]                                                                   | N/A_DENSE_OR_VULNERABLE_PROXY_NOT_A_MERGE_YIELD_CUTIN_EVENT | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |

† These semantic dimensions share the same parent task-level BDD and are not independent BDD tests.

### 三类Reference的永久定义

- **Behavior Reference**：定义谁相对谁变化，并与Target共同决定semantic delta方向。
- **Null Reference**：paired randomization q95或unpaired A/A calibration q95；ratio中的`1.0×`仅表示统计背景参考线。
- **Representation Baseline**：`old64`历史baseline，只比较检测能力，不定义行为方向。
- 禁止使用模糊术语“reference BDD”。

## 第二页：Representation Qualification Matrix

### 固定13维Treatment标准化敏感度

单元格为`BDD/null-q95 ratio / Z_BDD`。该Treatment下最高标准化检测敏感度，只表示在这一已知treatment上相对各representation自身null的标准化敏感度；不表示最完整、最通用或全局最优的behavior representation，也不得据此宣称ego13是‘全局最佳representation’。禁止跨representation排序raw MMD²。

| behavior_dimension   | old64             | A                 | B                 | C                 | ego13              | 该Treatment下最高标准化检测敏感度       |
|:---------------------|:------------------|:------------------|:------------------|:------------------|:-------------------|:----------------------------|
| 总体行为漂移               | 2.69× / Z=11.08   | 5.41× / Z=25.53   | 5.23× / Z=21.16   | 4.75× / Z=18.08   | 23.26× / Z=86.10   | ego13 (within-null Z=86.10) |
| 自由流速度                | N/A               | N/A               | N/A               | N/A               | N/A                | N/A                         |
| 纵向加速/减速              | 2.39× / Z=9.23    | 2.65× / Z=10.79   | 2.74× / Z=10.33   | 2.40× / Z=8.48    | 8.87× / Z=35.09    | ego13 (within-null Z=35.09) |
| 跟车行为                 | 1.85× / Z=5.63    | 1.61× / Z=5.43    | 1.72× / Z=5.25    | 1.59× / Z=4.61    | 4.81× / Z=18.74    | ego13 (within-null Z=18.74) |
| 逼近前车响应               | 6.84× / Z=27.98 † | 7.41× / Z=26.45 † | 7.39× / Z=30.60 † | 6.83× / Z=28.95 † | 11.14× / Z=35.91 † | ego13 (within-null Z=35.91) |
| 纵向平顺性                | 2.39× / Z=9.23    | 2.65× / Z=10.79   | 2.74× / Z=10.33   | 2.40× / Z=8.48    | 8.87× / Z=35.09    | ego13 (within-null Z=35.09) |
| 车道保持                 | N/A               | N/A               | N/A               | N/A               | N/A                | N/A                         |
| 变道行为                 | 2.04× / Z=7.44    | 2.81× / Z=10.86   | 2.50× / Z=9.12    | 2.66× / Z=9.15    | 6.66× / Z=22.80    | ego13 (within-null Z=22.80) |
| 横向动态                 | 1.63× / Z=5.73    | 2.93× / Z=11.61   | 2.99× / Z=10.43   | 3.01× / Z=10.44   | 6.57× / Z=22.01    | ego13 (within-null Z=22.01) |
| 前车间距/车头时距交互          | 6.84× / Z=27.98 † | 7.41× / Z=26.45 † | 7.39× / Z=30.60 † | 6.83× / Z=28.95 † | 11.14× / Z=35.91 † | ego13 (within-null Z=35.91) |
| 纵向跟车交互响应             | 6.84× / Z=27.98 † | 7.41× / Z=26.45 † | 7.39× / Z=30.60 † | 6.83× / Z=28.95 † | 11.14× / Z=35.91 † | ego13 (within-null Z=35.91) |
| 横向间隙接受/横向交互          | N/A               | N/A               | N/A               | N/A               | N/A                | N/A                         |
| 汇入/让行/切入响应           | 1.43× / Z=4.16    | 1.22× / Z=2.98    | 1.71× / Z=4.95    | 1.83× / Z=5.30    | 6.03× / Z=19.79    | ego13 (within-null Z=19.79) |

† These semantic dimensions share the same parent task-level BDD and are not independent BDD tests.

### Release与联合门禁资格

| representation_id   |   stage6p_n400_detection |   stage6p_n400_aa_fpr |   stage6p_n400_direction_min_detection | stage6jk_paired_gate_pass   | stage6p_unpaired_gate_pass   | waymo_gate_pass                     | interaction_increment_gate_pass   | stage6v_joint_candidate_gate_pass   | applicability_boundary                                                     |
|:--------------------|-------------------------:|----------------------:|---------------------------------------:|:----------------------------|:-----------------------------|:------------------------------------|:----------------------------------|:------------------------------------|:---------------------------------------------------------------------------|
| old64               |                    0.665 |                 0.05  |                                   0.62 | False                       | False                        | N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE | N/A_C_ONLY_DIAGNOSTIC             | N/A_NOT_ABC_CANDIDATE               | 历史Representation Baseline；用于能力比较，不定义行为方向。                                  |
| A                   |                    0.905 |                 0.03  |                                   0.9  | False                       | True                         | False                               | N/A_C_ONLY_DIAGNOSTIC             | False                               | Dynamic-data-only候选；release检出提升，但未通过Waymo与paired联合门禁。                      |
| B                   |                    1     |                 0.05  |                                   1    | False                       | True                         | False                               | N/A_C_ONLY_DIAGNOSTIC             | False                               | 当前最简单的learned release-level工程候选；不是通用或最终验证representation。                   |
| C                   |                    0.995 |                 0.065 |                                   0.99 | False                       | True                         | False                               | False                             | False                               | dual-branch候选；release检出强，但未证明full-context相对neighbor-zero的增量interaction信息。  |
| ego13               |                    1     |                 0.02  |                                   1    | True                        | True                         | N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE | N/A_C_ONLY_DIAGNOSTIC             | N/A_NOT_ABC_CANDIDATE               | controlled treatment高敏感参考；不能解释为通用style representation或neighbor/context无价值。 |

Primary Representation选择说明：B在Stage6P context-balanced n=400达到100.0% detection、5.0% A/A FPR，并且是结构更简单的learned release-level工程候选；这不推翻其Stage6J/K、Waymo与Stage6V联合门禁未通过的事实。

### ego13固定解释边界

ego13在当前多个controlled treatments中具有最高within-null standardized sensitivity，但这些treatment大量直接作用于ego运动学。因此不能解释为ego13是通用style representation，不能解释为neighbor/context无价值。learned64的主要强正结果仍包括production-style unpaired release monitoring；representation能力必须按deployment/evaluation task解释。

### Shared-parent统计身份

Closing response、Front-gap / THW interaction、Longitudinal following interaction对每个representation均绑定同一个Stage6S-v3 parent task-level BDD；三条语义解释只计为一个独立BDD检验。机器审计中的parent IDs为：

`stage6s_v3:old64:following_interaction, stage6s_v3:A:following_interaction, stage6s_v3:B:following_interaction, stage6s_v3:C:following_interaction, stage6s_v3:ego13:following_interaction`

## 一眼可答的最终结论

1. **跟车BDD（Primary Representation B）**：`1.72× / Z=5.25`；Behavior Reference为longitudinal conservative v2，Target为longitudinal assertive v2；60 scenario / 52 log，确认性结果。
2. **变道BDD（B）**：`2.50× / Z=9.12`；Behavior Reference为conservative v1，Target为assertive v1；固定60场景，`POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION`。它是变道场景slice，不证明ego一定执行了变道。
3. **纵向加速/减速BDD（B）**：`2.74× / Z=10.33`；Stage6J/K dose100确认性结果。
4. **interaction BDD（B）**：`7.39× / Z=30.60 †`；long-headway v2 → short-headway v2，80 pair / 11 log，确认性结果。
5. **Null Reference**：以上每项均使用B自己的冻结paired randomization q95；Behavior Reference与Target逐行显示。
6. **该Treatment下最高标准化检测敏感度**：当前有证据的矩阵行均为ego13；这只是controlled treatment下的within-null敏感度，不是全局排名。
7. **共享parent test**：逼近前车响应、前车间距/THW交互、纵向跟车交互响应共享Stage6S-v3 parent BDD，均以`†`标识，不是三次独立检验。
8. **证据身份**：Stage6J/K与Stage6S-v3为继承的预冻结confirmatory；Stage7 overall/lane-change/lateral/dense-interaction为post-hoc standardized descriptive；N/A维度不补实验。

`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`
