# Final Standardized BDD Style Report Card — Stage7L Prospective Evidence Addendum

> 基础schema状态：`FINAL_STANDARDIZED_BDD_REPORTING_SYSTEM_FROZEN`
> Stage7L证据整合状态：`STAGE7L_E_PROSPECTIVE_EVIDENCE_INTEGRATED_FOR_THESIS`
> 没有重算既有统计；Stage7L-E数值逐值继承E2冻结结果。

## 第一页：Behavior Drift / Style Report Card

Primary Representation仍为B；B只是测量representation，不是被评价的planner。每行独立声明Behavior Reference、Target和Null Reference。semantic方向只来自Target−Reference物理指标。

| behavior_dimension | behavior_reference | target | evaluation_mode | primary_representation | null_reference | n_scenarios | n_logs | bdd_to_null_q95_ratio | z_bdd | significance | semantic_delta_target_minus_reference | semantic_direction | evidence_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 总体行为漂移 | pdm_closed_conservative_v1 | pdm_closed_assertive_v1 | paired | B | within_scenario_pair_label_swap; common seeded sign stream across representations | 310 | 257 | 5.233408136820748 | 21.162317701590048 | 显著 | mean_speed +1.281 m/s; 95% CI [+1.120, +1.448]; rms_accel +0.235 m/s²; 95% CI [+0.206, +0.264] | MIXED_NO_SINGLE_STYLE_DIRECTION | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 自由流速度 | N/A | N/A | N/A | B | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A_NO_FROZEN_FREE_FLOW_SLICE |
| 纵向加速/减速 | pdm_closed_longitudinal_conservative_v2 | pdm_closed_longitudinal_assertive_v2 | paired | B | representation-specific paired within-scenario label-swap randomization inherited from Stage6J/K | 183 | 156 | 2.7377746118140065 | 10.328348889183347 | 显著 | delta_mean_speed +0.915 m/s; 95% CI [+0.758, +1.078]; delta_rms_accel +0.182 m/s²; 95% CI [+0.146, +0.217] | TARGET_HIGHER_LONGITUDINAL_EXCITATION | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 跟车行为 | pdm_closed_longitudinal_conservative_v2 | pdm_closed_longitudinal_assertive_v2 | paired | B | representation-specific paired within-scenario label-swap randomization inherited from Stage6J/K | 60 | 52 | 1.7234724560466528 | 5.254514960298006 | 显著 | delta_mean_speed +0.917 m/s; 95% CI [+0.625, +1.246]; delta_rms_accel +0.234 m/s²; 95% CI [+0.187, +0.280] | TARGET_MORE_ACTIVE_FOLLOWING | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 逼近前车响应 | pdm_closed_interaction_long_headway_v2 | pdm_closed_interaction_short_headway_v2 | paired | B | representation-specific paired within-scenario label-swap randomization inherited from Stage6S-v3 | 80 | 11 | 7.392571575590812 | 30.60300889203967 | 显著 | delta_mean_accel_during_closing_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MAINTAINS_MORE_ACCEL_DURING_CLOSING | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 纵向平顺性 | pdm_closed_longitudinal_conservative_v2 | pdm_closed_longitudinal_assertive_v2 | paired | B | representation-specific paired within-scenario label-swap randomization inherited from Stage6J/K | 183 | 156 | 2.7377746118140065 | 10.328348889183347 | 显著 | delta_rms_jerk +0.228 m/s³; 95% CI [+0.142, +0.319] | TARGET_HIGHER_LONGITUDINAL_JERK | SHARED_PARENT_BDD_SEMANTIC_PROXY |
| 车道保持 | N/A | N/A | N/A | B | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A_NO_FROZEN_LANE_KEEPING_SLICE |
| 变道行为 | pure_lateral_execution_gentle_dose0_60.0m | pure_lateral_execution_sharp_dose100_54.0m | paired | B | representation-specific same-scenario within-pair label-swap; 100000 randomizations; plus-one p; own null q95 | 80 | 79 | 0.4358024100249059 | -0.0650366602360071 | 不显著（预注册Primary FAIL；raw p=0.411906） | duration -0.200160 s; RMS lateral accel +0.055832 m/s²; peak yaw rate +0.014404 rad/s | TARGET_SHORTER_DURATION_HIGHER_LATERAL_EXCITATION | PROSPECTIVE_PLANNER_MECHANISM_POSITIVE_B_PRIMARY_BDD_FAILED |
| 横向动态 | pure_lateral_execution_gentle_dose0_60.0m | pure_lateral_execution_sharp_dose100_54.0m | paired | B | representation-specific same-scenario within-pair label-swap; 100000 randomizations; plus-one p; own null q95 | 38 | 38 | 0.8563292326513613 | 1.5142694880034469 | 不显著（secondary Holm p=1.0；LOW_N mixed proxy） | N/A | MIXED_PROXY | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_BDD_NOT_SIGNIFICANT |
| 前车间距/车头时距交互 | pdm_closed_interaction_long_headway_v2 | pdm_closed_interaction_short_headway_v2 | paired | B | representation-specific paired within-scenario label-swap randomization inherited from Stage6S-v3 | 80 | 11 | 7.392571575590812 | 30.60300889203967 | 显著 | delta_median_front_gap_m -4.202 m; 95% CI [-5.791, -1.181]; delta_median_finite_thw_s -2.670 s; 95% CI [-3.687, -2.275] | TARGET_SHORTER_GAP_OR_THW | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 纵向跟车交互响应 | pdm_closed_interaction_long_headway_v2 | pdm_closed_interaction_short_headway_v2 | paired | B | representation-specific paired within-scenario label-swap randomization inherited from Stage6S-v3 | 80 | 11 | 7.392571575590812 | 30.60300889203967 | 显著 | delta_mean_accel_during_following_pressure_mps2 +0.085 m/s²; 95% CI [+0.022, +0.450] | TARGET_MORE_ACCEL_UNDER_FOLLOWING_PRESSURE | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 横向间隙接受/横向交互 | N/A | N/A | N/A | B | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A_NO_FROZEN_LATERAL_GAP_OUTCOME |
| 汇入/让行/切入响应 | pdm_closed_conservative_v1 | pdm_closed_assertive_v1 | paired | B | within_scenario_pair_label_swap; common seeded sign stream across representations | 63 | 57 | 1.7073093301169444 | 4.946576033681239 | 显著 | mean_front_distance +0.062 m; 95% CI [-3.355, +3.753] | N/A_DENSE_OR_VULNERABLE_PROXY_NOT_A_MERGE_YIELD_CUTIN_EVENT | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |

Stage7L prospective更新要点：Sharp相对Gentle的换道时长缩短、RMS横向加速度和峰值yaw-rate升高，但B的BDD Primary不显著。这不是矛盾：前者回答行为是否物理变化，后者回答B是否能检出该分布变化。

† Stage6S-v3的closing/front-gap/following三行共享同一parent task-level BDD，不是独立检验。

## 第二页：Representation Qualification Matrix

单元格只比较各representation自身null下的BDD/q95和Z_BDD；禁止跨representation比较raw MMD²。`该Treatment下最高标准化检测敏感度`不表示全局最佳。

| behavior_dimension | old64 | A | B | C | ego13 | highest_standardized_sensitivity_on_this_treatment | evidence_status |
|---|---|---|---|---|---|---|---|
| 总体行为漂移 | 2.69× / Z=11.08 | 5.41× / Z=25.53 | 5.23× / Z=21.16 | 4.75× / Z=18.08 | 23.26× / Z=86.10 | ego13 (within-null Z=86.10) | POST_HOC_STANDARDIZED_DESCRIPTIVE_EVALUATION |
| 自由流速度 | N/A | N/A | N/A | N/A | N/A | N/A | N/A_NO_FROZEN_FREE_FLOW_SLICE |
| 纵向加速/减速 | 2.39× / Z=9.23 | 2.65× / Z=10.79 | 2.74× / Z=10.33 | 2.40× / Z=8.48 | 8.87× / Z=35.09 | ego13 (within-null Z=35.09) | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 跟车行为 | 1.85× / Z=5.63 | 1.61× / Z=5.43 | 1.72× / Z=5.25 | 1.59× / Z=4.61 | 4.81× / Z=18.74 | ego13 (within-null Z=18.74) | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 逼近前车响应 | 6.84× / Z=27.98 † | 7.41× / Z=26.45 † | 7.39× / Z=30.60 † | 6.83× / Z=28.95 † | 11.14× / Z=35.91 † | ego13 (within-null Z=35.91) | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 纵向平顺性 | 2.39× / Z=9.23 | 2.65× / Z=10.79 | 2.74× / Z=10.33 | 2.40× / Z=8.48 | 8.87× / Z=35.09 | ego13 (within-null Z=35.09) | SHARED_PARENT_BDD_SEMANTIC_PROXY |
| 车道保持 | N/A | N/A | N/A | N/A | N/A | N/A | N/A_NO_FROZEN_LANE_KEEPING_SLICE |
| 变道行为 | 0.58× / Z=-0.65 | 0.46× / Z=-0.02 | 0.44× / Z=-0.07 | 0.55× / Z=0.37 | 13.09× / Z=40.20 | ego13 (within-null Z=40.20) | PROSPECTIVE_PRIMARY_B_FAILED_SECONDARY_EGO13_HOLM_SIGNIFICANT |
| 横向动态 | 0.69× / Z=0.05 | 1.03× / Z=2.00 | 0.86× / Z=1.51 | 1.04× / Z=2.22 | 6.62× / Z=20.27 | ego13 (within-null Z=20.27) | PROSPECTIVE_SECONDARY_LOW_N_MIXED_PROXY_ONLY_EGO13_HOLM_SIGNIFICANT |
| 前车间距/车头时距交互 | 6.84× / Z=27.98 † | 7.41× / Z=26.45 † | 7.39× / Z=30.60 † | 6.83× / Z=28.95 † | 11.14× / Z=35.91 † | ego13 (within-null Z=35.91) | SHARED_PARENT_BDD_SEMANTIC_PROXY |
| 纵向跟车交互响应 | 6.84× / Z=27.98 † | 7.41× / Z=26.45 † | 7.39× / Z=30.60 † | 6.83× / Z=28.95 † | 11.14× / Z=35.91 † | ego13 (within-null Z=35.91) | INHERITED_PRE_FROZEN_CONFIRMATORY_RESULT |
| 横向间隙接受/横向交互 | N/A | N/A | N/A | N/A | N/A | N/A | N/A_NO_FROZEN_LATERAL_GAP_OUTCOME |
| 汇入/让行/切入响应 | 1.43× / Z=4.16 | 1.22× / Z=2.98 | 1.71× / Z=4.95 | 1.83× / Z=5.30 | 6.03× / Z=19.79 | ego13 (within-null Z=19.79) | POST_HOC_STANDARDIZED_DESCRIPTIVE_PROXY_ONLY |

### Release、联合门禁与Stage7L资格

| representation_id | stage6p_n400_detection | stage6p_n400_aa_fpr | stage6jk_paired_gate_pass | stage6p_unpaired_gate_pass | waymo_gate_pass | interaction_increment_gate_pass | stage7l_e_dose100_lane_change_status | stage7l_e_dose100_lane_change_bdd_over_null_q95 | stage7l_e_dose100_lane_change_z_bdd | stage6v_joint_candidate_gate_pass | applicability_boundary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| old64 | 0.665 | 0.05 | False | False | N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE | N/A_C_ONLY_DIAGNOSTIC | SECONDARY_HOLM_NOT_SIGNIFICANT | 0.5804694203926174 | -0.6457639955813627 | N/A_NOT_ABC_CANDIDATE | 历史Representation Baseline；用于能力比较，不定义行为方向。 |
| A | 0.905 | 0.03 | False | True | False | N/A_C_ONLY_DIAGNOSTIC | SECONDARY_HOLM_NOT_SIGNIFICANT | 0.4596285043285111 | -0.02142146563116 | False | Dynamic-data-only候选；release检出提升，但未通过Waymo与paired联合门禁。 |
| B | 1.0 | 0.05 | False | True | False | N/A_C_ONLY_DIAGNOSTIC | PRE_REGISTERED_PRIMARY_FAILED | 0.4358024100249059 | -0.0650366602360071 | False | 当前最简单的learned release-level工程候选；不是通用或最终验证representation。 |
| C | 0.995 | 0.065 | False | True | False | False | SECONDARY_HOLM_NOT_SIGNIFICANT | 0.5453479320798916 | 0.3730130704153008 | False | dual-branch候选；release检出强，但未证明full-context相对neighbor-zero的增量interaction信息。 |
| ego13 | 1.0 | 0.02 | True | True | N/A_NOT_A_STAGE6T_LEARNED_CANDIDATE | N/A_C_ONLY_DIAGNOSTIC | SECONDARY_HOLM_SIGNIFICANT | 13.08706829583308 | 40.20102515841172 | N/A_NOT_ABC_CANDIDATE | controlled treatment高敏感参考；不能解释为通用style representation或neighbor/context无价值。 |

## 证据优先级与历史保留

- `LAT.LANE_CHANGE`和`LAT.DYNAMICS`主显示使用Stage7L prospective dose100 evidence。
- 原Stage7 60场景post-hoc lane-change/lateral结果完整保留在`historical_stage7_posthoc_lateral_evidence.csv`和combined long table中，但不再作为横向主显示。
- Stage7L B Primary失败不会被ego13 secondary成功替代；Primary身份保持B。
- Free-flow speed、lane keeping、lateral gap interaction继续N/A，不补实验。

## 一眼可答的最终结论

1. 跟车BDD（B）：`1.72× / Z=5.25`，Stage6J/K confirmatory。
2. 变道BDD（B）：Stage7L prospective `0.436× / Z=-0.065 / p=0.411906`，Primary FAIL；物理mechanism同时PASS。
3. 纵向BDD（B）：`2.74× / Z=10.33`，Stage6J/K confirmatory。
4. interaction BDD（B）：`7.39× / Z=30.60 †`，Stage6S-v3 confirmatory。
5. 横向最高within-null标准化敏感度为ego13（Stage7L dose100 Z=40.201），不等于全局最佳。
6. Stage7历史post-hoc横向结果保留，但证据等级低于Stage7L prospective。
7. Stage6V联合决策仍为`NO_ABC_CANDIDATE_QUALIFIES_UNDER_PRE_FROZEN_RULE`。

`STAGE7L_E_PROSPECTIVE_EVIDENCE_INTEGRATED_FOR_THESIS`
