# Stage 6C v2 task-conditioned behavior-event BDD report

BDD detects distribution shift in learned embedding space. Task-specific metrics explain the drift direction.

本报告的主评价单元是 driving task / behavior-event slice 内的 BDD；hard_brake、late_brake 等 outcome-style 表现只应作为可选 post-hoc 诊断，而不是主结果。

## Task BDD summary

| task_key | task_value | strength_filter | detector_strength | detector_strength_counts | n_A(before) | n_B(before) | n_A(after) | n_B(after) | BDD_MMD | bootstrap_mean | bootstrap_std | in_CI | CI95 | p_value | interpretation |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| task_overtake_opportunity | overtake_opportunity | all | proxy | {"proxy": 8202} | 383 | 405 | 383 | 405 | 0.00268214 | 0.00470089 | 0.00126229 | False | [0.00274222, 0.00693981] | 0.168317 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_cutin_response | cutin_response | all | proxy | {"proxy": 10948} | 509 | 555 | 509 | 555 | 0.000965651 | 0.00236504 | 0.000677045 | False | [0.00154325, 0.00382321] | 0.930693 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_hesitation | hesitation | all | strong | {"strong": 52549} | 2629 | 2566 | 2629 | 2566 | 0.000462313 | 0.000886678 | 0.000253972 | False | [0.000587502, 0.00147443] | 0.39604 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_yield_conflict | yield_conflict | all | strong | {"strong": 52543} | 2623 | 2572 | 2623 | 2572 | 0.000412219 | 0.0008359 | 0.000218431 | False | [0.000476163, 0.00139173] | 0.435644 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_lead_brake_response | lead_brake_response | all | proxy | {"proxy": 21350, "strong": 13481} | 1677 | 1710 | 1677 | 1710 | 0.000394265 | 0.000859221 | 0.00023219 | False | [0.00057618, 0.00146917] | 0.574257 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_following | following | all | strong | {"strong": 43939} | 2125 | 2149 | 2125 | 2149 | 0.000306625 | 0.000717984 | 0.000240941 | False | [0.000424448, 0.00135045] | 0.772277 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_queue_approach | queue_approach | all | proxy | {"proxy": 26636, "strong": 13211} | 1920 | 1968 | 1920 | 1968 | 0.000304638 | 0.000696864 | 0.000168738 | False | [0.000440058, 0.000992333] | 0.792079 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |
| task_lane_change | lane_change | all | strong | {"strong": 61236} | 3050 | 3080 | 3050 | 3080 | 0.000286281 | 0.000734166 | 0.000224783 | False | [0.000410972, 0.00123035] | 0.871287 | Task-conditioned BDD is computed within the same behavior-event slice; inspect task-specific metrics below for semantic drift direction. |

## Style metric explanation layer

| task_key | metric | n_A | n_B | mean_A | mean_B | B_minus_A | effect_size |
|---|---|---:|---:|---:|---:|---:|---:|
| task_cutin_response | cutin_reaction_delay_to_brake | 26 | 25 | 1.58077 | 0.86 | -0.720769 | -0.557508 |
| task_cutin_response | cutin_late_response_score | 26 | 25 | 1.58077 | 0.86 | -0.720769 | -0.557508 |
| task_overtake_opportunity | overtake_target_lane_rear_gap | 138 | 138 | 18.4899 | 16.8025 | -1.68734 | -0.156959 |
| task_cutin_response | cutin_peak_decel_after_cutin | 509 | 555 | 1.04342 | 0.882786 | -0.160634 | -0.135803 |
| task_cutin_response | cutin_jerk_after_cutin | 509 | 555 | 4.98115 | 4.41224 | -0.568903 | -0.105636 |
| task_overtake_opportunity | overtake_execution_rate_proxy | 383 | 405 | 0.26893 | 0.224691 | -0.0442381 | -0.102695 |
| task_cutin_response | cutin_speed_drop_after_cutin | 509 | 555 | 0.289736 | 0.194943 | -0.0947926 | -0.0906833 |
| task_overtake_opportunity | overtake_time_to_initiate | 359 | 384 | 0.872981 | 1.02135 | 0.148374 | 0.0897062 |
| task_cutin_response | cutin_yielding_response_score | 509 | 555 | 3.81803 | 3.64361 | -0.174413 | -0.0806816 |
| task_cutin_response | cutin_gap_initial | 509 | 555 | 11.5768 | 11.1164 | -0.460404 | -0.0745545 |
| task_lead_brake_response | lead_brake_min_thw_after_lead_brake | 1677 | 1710 | 2.60208 | 2.72734 | 0.125257 | 0.0688038 |
| task_lead_brake_response | lead_brake_reaction_delay | 661 | 657 | 2.04054 | 2.16225 | 0.121708 | 0.0603081 |
| task_yield_conflict | conflict_accel_score | 2623 | 2572 | 2.46627 | 2.3597 | -0.106576 | -0.0542964 |
| task_overtake_opportunity | overtake_target_lane_front_gap | 291 | 316 | 20.4069 | 21.0288 | 0.621882 | 0.0514071 |
| task_queue_approach | queue_time_to_stop | 777 | 783 | 2.47748 | 2.6166 | 0.139125 | 0.0508853 |
| task_overtake_opportunity | overtake_min_front_gap_before | 383 | 405 | 15.5723 | 15.234 | -0.338279 | -0.0447691 |
| task_following | following_min_thw | 2125 | 2149 | 2.67625 | 2.75453 | 0.0782727 | 0.0436558 |
| task_cutin_response | cutin_gap_min | 509 | 555 | 11.1514 | 10.9434 | -0.208024 | -0.0418315 |
| task_lead_brake_response | lead_brake_min_ttc_after_lead_brake | 1659 | 1693 | 2.42485 | 2.51618 | 0.0913332 | 0.0380053 |
| task_hesitation | hesitation_lateral_velocity_sign_change_count | 2629 | 2566 | 7.11069 | 6.85776 | -0.252933 | -0.0376467 |
| task_lane_change | lc_gap_acceptance_score | 1006 | 1032 | 0.135909 | 0.1393 | 0.00339095 | 0.0368894 |
| task_queue_approach | queue_final_front_gap | 1920 | 1968 | 20.4412 | 19.9897 | -0.451419 | -0.0364925 |
| task_lead_brake_response | lead_brake_front_decel_start_time | 1677 | 1710 | 1.39428 | 1.33912 | -0.0551527 | -0.0334684 |
| task_following | following_min_front_distance | 2125 | 2149 | 15.6051 | 15.2919 | -0.313202 | -0.031337 |
| task_queue_approach | queue_creep_after_stop_score | 777 | 783 | 0.442802 | 0.450374 | 0.00757256 | 0.0311981 |
| task_lane_change | lc_oscillation_score | 3050 | 3080 | 3.4259 | 3.32549 | -0.100415 | -0.0301821 |
| task_yield_conflict | assertiveness_score | 2623 | 2572 | 1.91151 | 1.86322 | -0.0482861 | -0.0293822 |
| task_hesitation | hesitation_lc_duration | 858 | 852 | 3.2704 | 3.33521 | 0.064815 | 0.02879 |
| task_following | following_mean_front_distance | 2125 | 2149 | 21.5945 | 21.2563 | -0.338187 | -0.0285909 |
| task_lane_change | lc_duration | 2910 | 2963 | 3.68368 | 3.73753 | 0.0538526 | 0.0264469 |
| task_lane_change | lc_target_front_gap_min | 567 | 567 | 12.1762 | 11.8837 | -0.292511 | -0.0262148 |
| task_lead_brake_response | lead_brake_speed_drop_after_lead_brake | 1677 | 1710 | 1.92894 | 2.02109 | 0.0921573 | 0.0262018 |
| task_cutin_response | cutin_min_ttc | 509 | 555 | 1.5727 | 1.55368 | -0.0190161 | -0.0260884 |
| task_following | following_mean_thw | 2125 | 2149 | 4.90494 | 4.99148 | 0.0865339 | 0.02524 |
| task_cutin_response | cutin_min_thw | 509 | 555 | 2.62493 | 2.59028 | -0.0346494 | -0.0247401 |
| task_yield_conflict | gap_pressure_score | 2623 | 2572 | 0.22425 | 0.226219 | 0.00196924 | 0.0215593 |
| task_hesitation | hesitation_score | 2629 | 2566 | 10.7113 | 10.5728 | -0.13846 | -0.0208964 |
| task_hesitation | hesitation_lc_oscillation_score | 2629 | 2566 | 10.8745 | 10.7352 | -0.139286 | -0.0208262 |
| task_lane_change | lc_target_rear_gap_min | 671 | 694 | 14.6481 | 14.3985 | -0.249596 | -0.0192083 |
| task_hesitation | hesitation_abort_like_score | 2629 | 2566 | 0.267402 | 0.259158 | -0.00824383 | -0.0187145 |

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

- `task_overtake_executed`: below_min_bin_size (n_A=103, n_B=91, validity=valid)

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
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_following', 'bdd_mmd': 0.0003066249999998938, 'ci95_low': 0.0004244484374999136, 'ci95_high': 0.0013504492187500212, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 2000, 'effective_n_B': 2000, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_lead_brake_response', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 21350, "strong": 13481}', 'proxy_fraction': 0.6129597197898424}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_lead_brake_response', 'bdd_mmd': 0.000394265304664021, 'ci95_low': 0.0005761797642849287, 'ci95_high': 0.0014691726678752568, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 1677, 'effective_n_B': 1710, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_queue_approach', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 26636, "strong": 13211}', 'proxy_fraction': 0.6684568474414636}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_queue_approach', 'bdd_mmd': 0.0003046382289071392, 'ci95_low': 0.0004400577559033225, 'ci95_high': 0.00099233336747776, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 1920, 'effective_n_B': 1968, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_lane_change', 'bdd_mmd': 0.0002862812499999645, 'ci95_low': 0.0004109718749998548, 'ci95_high': 0.00123034765625008, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 2000, 'effective_n_B': 2000, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_cutin_response', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 10948}', 'proxy_fraction': 1.0}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_cutin_response', 'bdd_mmd': 0.0009656507880644316, 'ci95_low': 0.0015432479885840844, 'ci95_high': 0.003823209338022082, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 509, 'effective_n_B': 555, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- task_bdd_uses_proxy_detector: {'warning': 'task_bdd_uses_proxy_detector', 'task_key': 'task_overtake_opportunity', 'dominant_detector_strength': 'proxy', 'detector_strength_counts': '{"proxy": 8202}', 'proxy_fraction': 1.0}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_overtake_opportunity', 'bdd_mmd': 0.0026821377798387225, 'ci95_low': 0.002742216334002934, 'ci95_high': 0.006939809352798826, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 383, 'effective_n_B': 405, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_hesitation', 'bdd_mmd': 0.00046231250000006163, 'ci95_low': 0.0005875015625000313, 'ci95_high': 0.001474430468749993, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 2000, 'effective_n_B': 2000, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- observed_bdd_outside_bootstrap_ci: {'warning': 'observed_bdd_outside_bootstrap_ci', 'task_key': 'task_yield_conflict', 'bdd_mmd': 0.00041221874999997077, 'ci95_low': 0.0004761632812499139, 'ci95_high': 0.0013917273437501208, 'mmd_estimator_config': {'max_mmd_samples_requested': 2000, 'effective_n_A': 2000, 'effective_n_B': 2000, 'kernel_block_size': 1024, 'observed_and_bootstrap_share_initial_subsample': True}}
- completed: {'warning': 'completed', 'valid_task_count': 8, 'skipped_task_count': 1, 'embedding_rows': 164871, 'event_rows': 164871}
