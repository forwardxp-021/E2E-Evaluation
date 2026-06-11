# Stage 6C v2 task-conditioned behavior-event BDD report

BDD detects distribution shift in learned embedding space. Task-specific metrics explain the drift direction.

本报告的主评价单元是 driving task / behavior-event slice 内的 BDD；hard_brake、late_brake 等 outcome-style 表现只应作为可选 post-hoc 诊断，而不是主结果。

## Task BDD summary

| task_key | task_value | strength_filter | detector_strength | detector_strength_counts | n_A(before) | n_B(before) | n_A(after) | n_B(after) | BDD_MMD | bootstrap_mean | bootstrap_std | in_CI | CI95 | p_value | interpretation |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| task_cutin_response | cutin_response | all | proxy | {"proxy": 10948} | 564 | 117 | 564 | 117 | 0.219525 | 0.224554 | 0.0196429 | True | [0.186595, 0.257522] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_lane_change | lane_change | all | strong | {"strong": 61236} | 448 | 4068 | 448 | 4068 | 0.178129 | 0.178653 | 0.00749525 | True | [0.164511, 0.19289] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_queue_approach | queue_approach | all | proxy | {"proxy": 26636, "strong": 13211} | 1840 | 606 | 1840 | 606 | 0.177014 | 0.179206 | 0.00704613 | True | [0.164493, 0.192847] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_lead_brake_response | lead_brake_response | all | proxy | {"proxy": 21350, "strong": 13481} | 1548 | 559 | 1548 | 559 | 0.165798 | 0.166393 | 0.00849615 | True | [0.149551, 0.179677] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_following | following | all | strong | {"strong": 43939} | 2150 | 621 | 2150 | 621 | 0.165289 | 0.167508 | 0.00579602 | True | [0.157983, 0.179374] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_yield_conflict | yield_conflict | all | strong | {"strong": 52543} | 1514 | 1144 | 1514 | 1144 | 0.145184 | 0.146369 | 0.00488229 | True | [0.13777, 0.155155] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_hesitation | hesitation | all | strong | {"strong": 52549} | 965 | 1461 | 965 | 1461 | 0.100805 | 0.102128 | 0.00539625 | True | [0.0936501, 0.114465] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |

## Style metric explanation layer

| task_key | metric | n_A | n_B | mean_A | mean_B | B_minus_A | effect_size |
|---|---|---:|---:|---:|---:|---:|---:|
| task_lane_change | lc_max_lateral_speed | 448 | 4068 | 0.863194 | 4.1898 | 3.32661 | 2.72587 |
| task_lane_change | lc_sharpness_score | 448 | 4068 | 0.0887802 | 0.500008 | 0.411228 | 2.19183 |
| task_lane_change | lc_rms_lateral_accel | 448 | 4068 | 0.243677 | 1.24761 | 1.00394 | 1.98081 |
| task_yield_conflict | conflict_accel_score | 1514 | 1144 | 1.14434 | 4.23075 | 3.0864 | 1.81722 |
| task_lane_change | lc_rms_yaw_rate | 448 | 4068 | 0.0192614 | 0.204693 | 0.185431 | 1.80241 |
| task_cutin_response | cutin_min_ttc | 564 | 117 | 1.87787 | 0.825329 | -1.05254 | -1.56283 |
| task_yield_conflict | assertiveness_score | 1514 | 1144 | 0.925437 | 2.91935 | 1.99391 | 1.50606 |
| task_cutin_response | cutin_min_thw | 564 | 117 | 3.13901 | 1.24113 | -1.89788 | -1.41022 |
| task_lane_change | lc_heading_change_total | 448 | 4068 | 0.172099 | 1.31617 | 1.14407 | 1.21739 |
| task_lane_change | lc_rms_curvature | 448 | 4068 | 0.00340246 | 0.047717 | 0.0443145 | 1.15394 |
| task_cutin_response | cutin_gap_min | 564 | 117 | 12.8864 | 7.42498 | -5.46145 | -1.12144 |
| task_lane_change | lc_target_front_gap_min | 178 | 520 | 21.426 | 10.2387 | -11.1872 | -1.02044 |
| task_lane_change | lc_duration | 431 | 3966 | 2.24153 | 4.17766 | 1.93613 | 0.976099 |
| task_yield_conflict | yield_conflict_score | 1514 | 1144 | 3.5319 | 6.06072 | 2.52882 | 0.975334 |
| task_lead_brake_response | lead_brake_max_jerk_after_lead_brake | 1548 | 559 | 19.0998 | 36.8814 | 17.7815 | 0.96819 |
| task_following | following_max_abs_jerk | 2150 | 621 | 19.7889 | 36.9911 | 17.2022 | 0.948469 |
| task_queue_approach | queue_rms_jerk | 1840 | 606 | 6.13238 | 11.2447 | 5.11232 | 0.926583 |
| task_queue_approach | queue_stop_smoothness_score | 1840 | 606 | -4.05201 | -7.2535 | -3.20149 | -0.893548 |
| task_yield_conflict | rear_pressure_response_score | 1070 | 709 | 7.93788 | 13.9787 | 6.0408 | 0.885422 |
| task_cutin_response | cutin_gap_initial | 564 | 117 | 13.2942 | 7.83677 | -5.45744 | -0.82521 |
| task_cutin_response | cutin_jerk_after_cutin | 564 | 117 | 3.66032 | 7.74515 | 4.08483 | 0.823715 |
| task_following | following_rms_jerk | 2150 | 621 | 6.4689 | 11.2992 | 4.83034 | 0.820588 |
| task_following | following_aggressiveness_score | 2150 | 621 | 2.2068 | 3.77597 | 1.56917 | 0.818783 |
| task_cutin_response | cutin_yielding_response_score | 564 | 117 | 3.88229 | 2.27829 | -1.604 | -0.815599 |
| task_hesitation | hesitation_lc_duration | 203 | 831 | 2.06552 | 3.84633 | 1.78081 | 0.775387 |
| task_lane_change | lc_target_rear_gap_min | 169 | 751 | 22.8797 | 13.0983 | -9.78141 | -0.752201 |
| task_queue_approach | queue_time_to_stop | 730 | 173 | 3.20301 | 1.22197 | -1.98105 | -0.751368 |
| task_lane_change | lc_gap_acceptance_score | 262 | 1089 | 0.0806394 | 0.146299 | 0.0656597 | 0.724787 |
| task_cutin_response | cutin_peak_decel_after_cutin | 564 | 117 | 0.79079 | 1.55205 | 0.761256 | 0.689814 |
| task_following | following_late_brake_score | 2150 | 621 | 1.04573 | 3.83046 | 2.78473 | 0.686059 |
| task_lead_brake_response | lead_brake_late_response_score | 1548 | 559 | 0.690225 | 1.28567 | 0.595443 | 0.64134 |
| task_lead_brake_response | lead_brake_peak_decel_after_lead_brake | 1548 | 559 | 1.74146 | 3.16744 | 1.42598 | 0.633937 |
| task_queue_approach | queue_peak_decel | 1840 | 606 | 1.97164 | 3.2623 | 1.29066 | 0.621841 |
| task_following | following_min_thw | 2150 | 621 | 3.21446 | 2.1152 | -1.09926 | -0.587752 |
| task_following | following_peak_decel | 2150 | 621 | 2.02824 | 3.29892 | 1.27068 | 0.58734 |
| task_following | following_min_front_distance | 2150 | 621 | 18.2389 | 12.2477 | -5.99117 | -0.567024 |
| task_lead_brake_response | lead_brake_min_thw_after_lead_brake | 1548 | 559 | 3.19181 | 2.12126 | -1.07055 | -0.556907 |
| task_queue_approach | queue_distance_when_start_decel | 1602 | 539 | 24.0914 | 18.0329 | -6.05851 | -0.518806 |
| task_yield_conflict | small_gap_speed_maintain_score | 1514 | 1144 | 5.93681 | 8.38938 | 2.45256 | 0.469395 |
| task_following | following_mean_front_distance | 2150 | 621 | 24.4163 | 18.7003 | -5.71604 | -0.453942 |

## Metric quality warnings

- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_peak_decel', 'p99': 88.18155059814399, 'max': 973.9443969726562, 'min': 0.0, 'expected_range': [0.0, 12.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_jerk', 'p99': 346.9703537791367, 'max': 5425.427204522052, 'min': 0.31034740834822716, 'expected_range': [-80.0, 80.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_jerk', 'p99': 1736.2067031860295, 'max': 19413.407592773438, 'min': 0.8535861968994141, 'expected_range': [-80.0, 80.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_yaw_rate', 'p99': 3.1423631071896705, 'max': 16.370475958360878, 'min': 0.0006558123303606169, 'expected_range': [-2.0, 2.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_yaw_rate', 'p99': 12.593758583068844, 'max': 31.415817260742188, 'min': 0.0017428398132324219, 'expected_range': [-2.0, 2.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_lateral_accel', 'p99': 6.118913210639888, 'max': 194.0924073832214, 'min': 0.015322627441702343, 'expected_range': [-8.0, 8.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_lateral_accel', 'p99': 31.955756664275825, 'max': 771.9446516036987, 'min': 0.03143956098938361, 'expected_range': [-8.0, 8.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_curvature', 'p99': 1.8439568756130393, 'max': 1312.4450825882468, 'min': 8.30845516729268e-05, 'expected_range': [-1.0, 1.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_curvature', 'p99': 12.766613260247453, 'max': 11738.86489868164, 'min': 0.0001924390472764212, 'expected_range': [-1.0, 1.0]}

## Skipped tasks

- `task_overtake_opportunity`: below_min_bin_size (n_A=497, n_B=76, validity=valid)
- `task_overtake_executed`: below_min_bin_size (n_A=65, n_B=63, validity=valid)

## Interpretation guide

- `negative_control_random`: sanity check; task BDD should be low and not systematic.
- `pseudo_agg_vs_cons`: positive control; style drift should localize to relevant behavior tasks.
- `scene_confounding_control`: confounding diagnosis; drift may concentrate where task exposure or dynamic interaction pressure differs.

## Warnings

- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_peak_decel', 'p99': 88.18155059814399, 'max': 973.9443969726562, 'min': 0.0, 'expected_range': [0.0, 12.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_jerk', 'p99': 346.9703537791367, 'max': 5425.427204522052, 'min': 0.31034740834822716, 'expected_range': [-80.0, 80.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_jerk', 'p99': 1736.2067031860295, 'max': 19413.407592773438, 'min': 0.8535861968994141, 'expected_range': [-80.0, 80.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_yaw_rate', 'p99': 3.1423631071896705, 'max': 16.370475958360878, 'min': 0.0006558123303606169, 'expected_range': [-2.0, 2.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_yaw_rate', 'p99': 12.593758583068844, 'max': 31.415817260742188, 'min': 0.0017428398132324219, 'expected_range': [-2.0, 2.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_lateral_accel', 'p99': 6.118913210639888, 'max': 194.0924073832214, 'min': 0.015322627441702343, 'expected_range': [-8.0, 8.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_lateral_accel', 'p99': 31.955756664275825, 'max': 771.9446516036987, 'min': 0.03143956098938361, 'expected_range': [-8.0, 8.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_curvature', 'p99': 1.8439568756130393, 'max': 1312.4450825882468, 'min': 8.30845516729268e-05, 'expected_range': [-1.0, 1.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_curvature', 'p99': 12.766613260247453, 'max': 11738.86489868164, 'min': 0.0001924390472764212, 'expected_range': [-1.0, 1.0]}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_lead_brake_response', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 21350, "strong": 13481}', 'proxy_fraction': 0.6129597197898424}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_queue_approach', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 26636, "strong": 13211}', 'proxy_fraction': 0.6684568474414636}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_cutin_response', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 10948}', 'proxy_fraction': 1.0}
- completed: {'warning': 'completed', 'valid_task_count': 7, 'skipped_task_count': 2, 'embedding_rows': 164871, 'event_rows': 164871}
