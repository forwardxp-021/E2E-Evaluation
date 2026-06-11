# Stage 6C v2 task-conditioned behavior-event BDD report

BDD detects distribution shift in learned embedding space. Task-specific metrics explain the drift direction.

本报告的主评价单元是 driving task / behavior-event slice 内的 BDD；hard_brake、late_brake 等 outcome-style 表现只应作为可选 post-hoc 诊断，而不是主结果。

## Task BDD summary

| task_key | task_value | strength_filter | detector_strength | detector_strength_counts | n_A(before) | n_B(before) | n_A(after) | n_B(after) | BDD_MMD | bootstrap_mean | bootstrap_std | in_CI | CI95 | p_value | interpretation |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| task_lane_change | lane_change | all | strong | {"strong": 61236} | 719 | 2847 | 719 | 2847 | 0.224173 | 0.225638 | 0.00875455 | True | [0.212772, 0.244427] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_cutin_response | cutin_response | all | proxy | {"proxy": 10948} | 824 | 114 | 824 | 114 | 0.21089 | 0.21466 | 0.0116959 | True | [0.194974, 0.236418] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_hesitation | hesitation | all | strong | {"strong": 52549} | 1112 | 1966 | 1112 | 1966 | 0.176213 | 0.176848 | 0.00669485 | True | [0.164012, 0.192297] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_queue_approach | queue_approach | all | proxy | {"proxy": 26636, "strong": 13211} | 2863 | 431 | 2863 | 431 | 0.146251 | 0.146622 | 0.00550037 | True | [0.138908, 0.159969] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_yield_conflict | yield_conflict | all | strong | {"strong": 52543} | 2365 | 1084 | 2365 | 1084 | 0.138207 | 0.139538 | 0.00423153 | True | [0.131357, 0.148725] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_following | following | all | strong | {"strong": 43939} | 3184 | 472 | 3184 | 472 | 0.136325 | 0.137444 | 0.00387823 | True | [0.130218, 0.144421] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_lead_brake_response | lead_brake_response | all | proxy | {"proxy": 21350, "strong": 13481} | 2439 | 432 | 2439 | 432 | 0.135319 | 0.135918 | 0.00502076 | True | [0.126859, 0.145364] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_overtake_opportunity | overtake_opportunity | all | proxy | {"proxy": 8202} | 547 | 107 | 547 | 107 | 0.134527 | 0.141291 | 0.0126963 | True | [0.120671, 0.172169] | 0.00990099 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |

## Style metric explanation layer

| task_key | metric | n_A | n_B | mean_A | mean_B | B_minus_A | effect_size |
|---|---|---:|---:|---:|---:|---:|---:|
| task_overtake_opportunity | overtake_max_abs_jerk | 547 | 107 | 18.6126 | 68.6922 | 50.0797 | 3.94932 |
| task_following | following_max_abs_jerk | 3184 | 472 | 18.9671 | 66.14 | 47.1729 | 3.44715 |
| task_following | following_aggressiveness_score | 3184 | 472 | 2.07304 | 6.79483 | 4.72179 | 3.2223 |
| task_lead_brake_response | lead_brake_max_jerk_after_lead_brake | 2439 | 432 | 18.4853 | 63.4626 | 44.9772 | 3.19613 |
| task_queue_approach | queue_stop_smoothness_score | 2863 | 431 | -3.96006 | -12.8699 | -8.9098 | -3.17167 |
| task_queue_approach | queue_rms_jerk | 2863 | 431 | 6.07447 | 19.5449 | 13.4704 | 3.04114 |
| task_following | following_rms_jerk | 3184 | 472 | 6.14491 | 20.1074 | 13.9625 | 2.99352 |
| task_overtake_opportunity | overtake_peak_accel | 547 | 107 | 1.73622 | 6.12514 | 4.38891 | 2.84961 |
| task_overtake_opportunity | overtake_peak_decel | 547 | 107 | 1.803 | 6.97932 | 5.17632 | 2.71622 |
| task_following | following_peak_decel | 3184 | 472 | 1.8371 | 6.44343 | 4.60633 | 2.54882 |
| task_queue_approach | queue_peak_decel | 2863 | 431 | 1.84565 | 6.19481 | 4.34916 | 2.45716 |
| task_lead_brake_response | lead_brake_peak_decel_after_lead_brake | 2439 | 432 | 1.53297 | 6.0969 | 4.56393 | 2.41017 |
| task_yield_conflict | conflict_accel_score | 2365 | 1084 | 1.63393 | 4.65817 | 3.02424 | 1.78056 |
| task_cutin_response | cutin_min_ttc | 824 | 114 | 1.76025 | 0.836953 | -0.923294 | -1.48605 |
| task_cutin_response | cutin_jerk_after_cutin | 824 | 114 | 3.50582 | 10.3627 | 6.85687 | 1.45103 |
| task_following | following_late_brake_score | 3184 | 472 | 0.989496 | 6.0998 | 5.11031 | 1.45064 |
| task_lane_change | lc_sharpness_score | 719 | 2847 | 0.189751 | 0.497733 | 0.307982 | 1.38851 |
| task_cutin_response | cutin_min_thw | 824 | 114 | 2.9851 | 1.28063 | -1.70447 | -1.38073 |
| task_lane_change | lc_rms_lateral_accel | 719 | 2847 | 0.480292 | 1.2449 | 0.76461 | 1.31667 |
| task_lane_change | lc_max_lateral_speed | 719 | 2847 | 1.92854 | 3.83249 | 1.90396 | 1.18778 |
| task_lane_change | lc_rms_yaw_rate | 719 | 2847 | 0.0702344 | 0.200716 | 0.130482 | 1.06819 |
| task_cutin_response | cutin_peak_decel_after_cutin | 824 | 114 | 0.771909 | 1.84392 | 1.07201 | 1.02903 |
| task_overtake_opportunity | overtake_execution_score | 547 | 107 | 1.5314 | 3.02945 | 1.49805 | 0.990565 |
| task_queue_approach | queue_front_speed_mean | 2863 | 431 | 5.38113 | 9.59946 | 4.21833 | 0.914697 |
| task_overtake_opportunity | overtake_opportunity_score | 547 | 107 | 2.82999 | 4.42673 | 1.59674 | 0.908332 |
| task_following | following_min_thw | 3184 | 472 | 3.11477 | 1.53173 | -1.58305 | -0.898239 |
| task_queue_approach | queue_front_speed_min | 2863 | 431 | 2.88064 | 6.78163 | 3.90099 | 0.896301 |
| task_yield_conflict | assertiveness_score | 2365 | 1084 | 1.50734 | 2.89915 | 1.39181 | 0.896252 |
| task_following | following_front_closing_rate_mean | 3184 | 472 | 6.08024 | 10.9462 | 4.86592 | 0.8843 |
| task_following | following_mean_thw | 3184 | 472 | 5.73396 | 2.81075 | -2.92321 | -0.873623 |
| task_lead_brake_response | lead_brake_min_thw_after_lead_brake | 2439 | 432 | 3.08912 | 1.55569 | -1.53342 | -0.849725 |
| task_lane_change | lc_heading_change_total | 719 | 2847 | 0.452679 | 1.34964 | 0.896964 | 0.840665 |
| task_following | following_front_closing_rate_p95 | 3184 | 472 | 10.2149 | 15.1842 | 4.96928 | 0.827968 |
| task_yield_conflict | yield_conflict_score | 2365 | 1084 | 4.11491 | 6.06554 | 1.95062 | 0.778824 |
| task_overtake_opportunity | overtake_execution_rate_proxy | 547 | 107 | 0.146252 | 0.439252 | 0.293 | 0.768988 |
| task_lane_change | lc_duration | 674 | 2754 | 2.79748 | 4.11057 | 1.31309 | 0.677877 |
| task_lane_change | lc_rms_curvature | 719 | 2847 | 0.0187271 | 0.0475819 | 0.0288547 | 0.669812 |
| task_lead_brake_response | lead_brake_min_ttc_after_lead_brake | 2404 | 432 | 2.91083 | 1.32776 | -1.58307 | -0.640415 |
| task_queue_approach | queue_creep_after_stop_score | 1459 | 51 | 0.435154 | 0.57947 | 0.144316 | 0.602168 |
| task_queue_approach | queue_front_stopped_ratio | 2863 | 431 | 0.259426 | 0.075261 | -0.184165 | -0.579328 |

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

- `task_overtake_executed`: below_min_bin_size (n_A=80, n_B=47, validity=valid)

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
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_overtake_opportunity', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 8202}', 'proxy_fraction': 1.0}
- completed: {'warning': 'completed', 'valid_task_count': 8, 'skipped_task_count': 1, 'embedding_rows': 164871, 'event_rows': 164871}
