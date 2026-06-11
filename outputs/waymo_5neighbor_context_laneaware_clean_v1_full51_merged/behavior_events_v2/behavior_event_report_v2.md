# Stage 6C v2 behavior-event build report

Stage 6C v2 is **Task-conditioned behavior-event BDD**. This builder creates task slices and task-specific style metrics; BDD is computed by `stage6c_task_conditioned_bdd_report.py`.

## Reliability notes

- following and yield_conflict are currently the most reliable strong detectors.
- cutin, overtake, and much of lead/queue remain proxy-based.
- lane_change and hesitation are usable only if positive_ratio is not broad after tightening; this report emits `lane_change_detector_broad` or `hesitation_detector_broad` when positive_ratio > 0.40.
- TTC/THW sentinels and invalid time gaps are reported as NaN, not as 999-style diagnostic scores.

- total_rows: 164871
- shard_count: 35
- missing metrics are stored as NaN, never as silent zero fills.

## Task diagnostics

| task_key | positive_label | negative_label | positive_count | negative_count | unknown_count | positive_ratio | unknown_ratio | event_validity |
|---|---|---|---:|---:|---:|---:|---:|---|
| task_following | following | not_following | 43939 | 120932 | 0 | 0.266505 | 0 | valid |
| task_lead_brake_response | lead_brake_response | no_lead_brake_response | 34831 | 130040 | 0 | 0.211262 | 0 | valid |
| task_queue_approach | queue_approach | no_queue_approach | 39847 | 125024 | 0 | 0.241686 | 0 | valid |
| task_lane_change | lane_change | no_lane_change | 61236 | 103635 | 0 | 0.371418 | 0 | valid |
| task_cutin_response | cutin_response | no_cutin_response | 10948 | 153923 | 0 | 0.0664034 | 0 | valid |
| task_overtake_opportunity | overtake_opportunity | no_overtake_opportunity | 8202 | 156669 | 0 | 0.049748 | 0 | valid |
| task_overtake_executed | overtake_executed | no_overtake_executed | 2011 | 162860 | 0 | 0.0121974 | 0 | valid |
| task_hesitation | hesitation | no_hesitation | 52549 | 112322 | 0 | 0.318728 | 0 | valid |
| task_yield_conflict | yield_conflict | no_yield_conflict | 52543 | 112328 | 0 | 0.318692 | 0 | valid |

## Metric diagnostics

| metric | valid_count | valid_rate | min | p01 | p50 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| following_mean_thw | 44036 | 0.267094 | 0.191995 | 0.800281 | 3.95221 | 16.3832 | 29.4682 |
| following_min_thw | 44036 | 0.267094 | 0.00661912 | 0.256639 | 2.28329 | 10.043 | 29.4682 |
| following_mean_front_distance | 44036 | 0.267094 | 3.42606 | 7.20604 | 18.1607 | 62.6379 | 115.718 |
| following_min_front_distance | 44036 | 0.267094 | 0.153171 | 3.13575 | 12.1674 | 51.8453 | 110.469 |
| following_front_closing_rate_mean | 44036 | 0.267094 | -19.975 | -1.88891 | 5.89072 | 24.869 | 40.8924 |
| following_front_closing_rate_p95 | 44036 | 0.267094 | -4.46387 | 0.956636 | 10.3222 | 29.9856 | 49.084 |
| following_peak_decel | 164871 | 1 | 0 | 0 | 1.93339 | 12 | 12 |
| following_rms_jerk | 164871 | 1 | 0.116716 | 0.404261 | 6.94697 | 34.861 | 69.4716 |
| following_max_abs_jerk | 164871 | 1 | 0.252419 | 1.41143 | 20.9521 | 80 | 80 |
| following_late_brake_score | 44036 | 0.267094 | 0 | 0 | 0.983827 | 13.5163 | 906.665 |
| following_aggressiveness_score | 164871 | 1 | 0.0608181 | 0.347137 | 3.86422 | 20.8882 | 40.7358 |
| lead_brake_front_decel_start_time | 41325 | 0.250651 | 0.1 | 0.1 | 0.6 | 7.2 | 7.9 |
| lead_brake_ego_brake_start_time | 143292 | 0.869116 | 0 | 0 | 0.4 | 7.7 | 7.9 |
| lead_brake_reaction_delay | 15616 | 0.0947165 | 0 | 0 | 1.3 | 7.5 | 7.8 |
| lead_brake_min_ttc_after_lead_brake | 40682 | 0.24675 | 0.0147164 | 0.177721 | 1.87551 | 16.3651 | 29.9361 |
| lead_brake_min_thw_after_lead_brake | 41035 | 0.248892 | 0.01613 | 0.269337 | 2.2785 | 13.5146 | 29.9834 |
| lead_brake_peak_decel_after_lead_brake | 41325 | 0.250651 | 0 | 0 | 1.58351 | 12 | 12 |
| lead_brake_max_jerk_after_lead_brake | 41325 | 0.250651 | 0 | 0.94212 | 18.5583 | 80 | 80 |
| lead_brake_speed_drop_after_lead_brake | 41325 | 0.250651 | 0 | 0 | 0 | 13.3245 | 25.1504 |
| lead_brake_late_response_score | 41035 | 0.248892 | 0.0299127 | 0.0779585 | 0.505099 | 4.06091 | 31.6482 |
| queue_distance_when_start_decel | 38419 | 0.233025 | 2.58112 | 6.22504 | 20.4846 | 71.5226 | 119.197 |
| queue_time_to_stop | 59323 | 0.359815 | 0 | 0 | 0 | 7.7 | 7.9 |
| queue_final_front_gap | 44036 | 0.267094 | 2.8131 | 5.05013 | 18.7696 | 79.5361 | 138.751 |
| queue_peak_decel | 164871 | 1 | 0 | 0 | 1.93339 | 12 | 12 |
| queue_rms_jerk | 164871 | 1 | 0.116716 | 0.404261 | 6.94697 | 34.861 | 69.4716 |
| queue_stop_smoothness_score | 164871 | 1 | -40.7358 | -22.6588 | -4.42359 | -0.386287 | -0.0608181 |
| queue_creep_after_stop_score | 59323 | 0.359815 | 0.0370126 | 0.108909 | 0.353691 | 0.974807 | 0.999997 |
| queue_front_speed_min | 44036 | 0.267094 | 0 | 0 | 2.4156 | 21.0675 | 37.2059 |
| queue_front_speed_mean | 44036 | 0.267094 | 0 | 0 | 5.99757 | 22.1483 | 37.6712 |
| queue_front_stopped_ratio | 164871 | 1 | 0 | 0 | 0 | 1 | 1 |
| lc_rms_yaw_rate | 164871 | 1 | 0.000229364 | 0.00209346 | 0.014899 | 0.547749 | 1.56085 |
| lc_rms_curvature | 164871 | 1 | 5.60787e-05 | 0.000213045 | 0.00734372 | 0.198292 | 0.741298 |
| lc_heading_change_total | 164871 | 1 | 0.00344825 | 0.0167882 | 0.188169 | 6.41808 | 8 |
| lc_max_lateral_speed | 164871 | 1 | 0.00288499 | 0.0281225 | 0.358232 | 5 | 5 |
| lc_rms_lateral_accel | 164871 | 1 | 0.00409529 | 0.0192817 | 0.256413 | 2.20236 | 5.71735 |
| lc_duration | 58611 | 0.355496 | 0.1 | 0.1 | 3.7 | 6.9 | 7.6 |
| lc_oscillation_score | 164871 | 1 | 0 | 0 | 6.5 | 24.5 | 40 |
| lc_target_front_gap_min | 45114 | 0.273632 | 0.585603 | 2.93132 | 7.03763 | 50.3418 | 118.496 |
| lc_target_rear_gap_min | 45661 | 0.27695 | 0.285004 | 2.9812 | 7.80396 | 58.2911 | 121.777 |
| lc_gap_acceptance_score | 69320 | 0.42045 | 0.00821171 | 0.017774 | 0.147518 | 0.332756 | 3.50873 |
| lc_sharpness_score | 164871 | 1 | 0.00159438 | 0.0088338 | 0.0972719 | 0.841575 | 2.16919 |
| cutin_gap_initial | 12207 | 0.0740397 | 0.171075 | 2.9206 | 10.6262 | 49.8387 | 105.295 |
| cutin_gap_min | 10948 | 0.0664034 | 0.171075 | 2.89752 | 9.87462 | 24.2262 | 24.9844 |
| cutin_min_ttc | 10947 | 0.0663974 | 0.00430349 | 0.122351 | 1.52147 | 3.58436 | 22.0375 |
| cutin_min_thw | 10948 | 0.0664034 | 0.00661912 | 0.168838 | 2.46272 | 6.7339 | 14.5283 |
| cutin_reaction_delay_to_brake | 461 | 0.00279613 | 0 | 0 | 0.9 | 5.56 | 7.3 |
| cutin_peak_decel_after_cutin | 12207 | 0.0740397 | 0 | 0 | 0.70072 | 6.54178 | 12 |
| cutin_jerk_after_cutin | 12207 | 0.0740397 | 0 | 0.0454486 | 3.72683 | 30.4386 | 80 |
| cutin_speed_drop_after_cutin | 12207 | 0.0740397 | 0 | 0 | 0 | 6.16911 | 25.1504 |
| cutin_yielding_response_score | 164871 | 1 | 0 | 0 | 0 | 13.8046 | 29.4981 |
| cutin_late_response_score | 461 | 0.00279613 | 0 | 0 | 0.9 | 5.56 | 7.3 |
| overtake_opportunity_score | 85517 | 0.51869 | -9.9446 | -0.247671 | 1 | 9.25656 | 19.2676 |
| overtake_execution_score | 164871 | 1 | 0 | 0 | 1.7216 | 6.26338 | 35.7411 |
| overtake_execution_rate_proxy | 164871 | 1 | 0 | 0 | 0 | 1 | 1 |
| overtake_time_to_initiate | 7722 | 0.0468366 | 0 | 0 | 0.2 | 7.2 | 7.9 |
| overtake_peak_accel | 8202 | 0.049748 | -1.15764 | -0.357837 | 1.99526 | 8 | 8 |
| overtake_peak_decel | 8202 | 0.049748 | 0 | 0 | 1.98545 | 12 | 12 |
| overtake_max_abs_jerk | 8202 | 0.049748 | 0.383568 | 1.29518 | 19.9716 | 80 | 80 |
| overtake_min_front_gap_before | 8202 | 0.049748 | 0.209686 | 3.14837 | 13.4762 | 34.04 | 34.9922 |
| overtake_target_lane_front_gap | 45114 | 0.273632 | 0.585603 | 2.93132 | 7.03763 | 50.3418 | 118.496 |
| overtake_target_lane_rear_gap | 45661 | 0.27695 | 0.285004 | 2.9812 | 7.80396 | 58.2911 | 121.777 |
| hesitation_score | 164871 | 1 | 0 | 0.5 | 6.5 | 24.5 | 40 |
| hesitation_lc_duration | 58611 | 0.355496 | 0.1 | 0.1 | 3.7 | 6.9 | 7.6 |
| hesitation_yaw_sign_change_count | 164871 | 1 | 0 | 0 | 10 | 33 | 50 |
| hesitation_lateral_velocity_sign_change_count | 164871 | 1 | 0 | 0 | 3 | 23 | 42 |
| hesitation_lc_oscillation_score | 164871 | 1 | 0 | 0 | 6.5 | 24.5 | 40 |
| hesitation_abort_like_score | 164871 | 1 | 0 | 0 | 0 | 1 | 1 |
| hesitation_speed_drop | 164871 | 1 | 0 | 0 | 0 | 14.2396 | 29.4981 |
| hesitation_evidence_count | 164871 | 1 | 0 | 0 | 1 | 3 | 5 |
| yield_conflict_score | 164871 | 1 | -4.01953 | -0.117055 | 3.14043 | 11.2285 | 50.2566 |
| yielding_score | 164871 | 1 | 0 | 0 | 1.29215 | 9.9321 | 20.7491 |
| assertiveness_score | 164871 | 1 | 0 | 0 | 1.91818 | 8.59665 | 53.6116 |
| gap_pressure_score | 85517 | 0.51869 | 0.00821171 | 0.0194178 | 0.133543 | 0.343988 | 6.52865 |
| conflict_accel_score | 52543 | 0.318692 | -1.54437 | -0.413294 | 2.12009 | 8 | 8 |
| small_gap_speed_maintain_score | 52543 | 0.318692 | 0.999085 | 1.07209 | 6.35494 | 22.5799 | 35.9308 |
| rear_pressure_response_score | 45661 | 0.27695 | -14.3197 | -1.44603 | 12.3054 | 32.8228 | 142.721 |
| courtesy_score | 164871 | 1 | -1.00962 | -0.0506459 | 1.03992 | 8.99917 | 18.1032 |

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
- raw_metric_physically_implausible: {'warning': 'raw_metric_physically_implausible'}
- physical_metric_clipping_applied: {'warning': 'physical_metric_clipping_applied', 'smoothing_window': 5}

## Degenerate/all_unknown tasks

- None

## Warnings

- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 0, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000000/neighbor_slot_ids.npy'}
- lead_brake_selective_detector_enabled: {'warning': 'lead_brake_selective_detector_enabled', 'shard_id': 0, 'shard_path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000000', 'front_speed_preferred_with_closing_derivative_fallback': True}
- cutin_true_slot_transition_not_implemented_using_gap_drop_proxy: {'warning': 'cutin_true_slot_transition_not_implemented_using_gap_drop_proxy', 'shard_id': 0, 'shard_path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000000', 'slot_ids_available': True}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 1, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000001/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 2, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000002/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 3, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000003/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 4, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000004/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 5, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000005/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 6, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000006/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 7, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000007/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 8, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13/shards/shard_000008/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 9, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000000/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 10, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000001/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 11, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000002/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 12, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000003/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 13, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000004/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 14, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000005/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 15, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000006/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 16, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000007/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 17, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26/shards/shard_000008/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 18, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000000/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 19, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000001/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 20, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000002/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 21, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000003/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 22, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000004/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 23, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000005/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 24, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000006/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 25, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000007/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 26, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39/shards/shard_000008/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 27, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000000/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 28, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000001/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 29, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000002/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 30, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000003/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 31, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000004/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 32, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000005/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 33, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000006/neighbor_slot_ids.npy'}
- neighbor_slot_ids_loaded_with_pickle: {'warning': 'neighbor_slot_ids_loaded_with_pickle', 'neighbor_slot_ids_loaded_with_pickle': True, 'shard_id': 34, 'path': '/Users/liuqing/Projects/01_E2E_QA_Code/E2E-Evaluation/outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51/shards/shard_000007/neighbor_slot_ids.npy'}
- detector_strength_not_strong: {'warning': 'detector_strength_not_strong', 'task_key': 'task_lead_brake_response', 'detector_strength': 'proxy', 'counts': {'proxy': 151390, 'strong': 13481}, 'rows': 164871}
- detector_strength_not_strong: {'warning': 'detector_strength_not_strong', 'task_key': 'task_cutin_response', 'detector_strength': 'proxy', 'counts': {'proxy': 164871}, 'rows': 164871}
- detector_strength_not_strong: {'warning': 'detector_strength_not_strong', 'task_key': 'task_overtake_opportunity', 'detector_strength': 'proxy', 'counts': {'proxy': 164871}, 'rows': 164871}
- detector_strength_not_strong: {'warning': 'detector_strength_not_strong', 'task_key': 'task_overtake_executed', 'detector_strength': 'proxy', 'counts': {'proxy': 164871}, 'rows': 164871}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_peak_decel', 'p99': 88.18155059814399, 'max': 973.9443969726562, 'min': 0.0, 'expected_range': [0.0, 12.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_jerk', 'p99': 346.9703537791367, 'max': 5425.427204522052, 'min': 0.31034740834822716, 'expected_range': [-80.0, 80.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_jerk', 'p99': 1736.2067031860295, 'max': 19413.407592773438, 'min': 0.8535861968994141, 'expected_range': [-80.0, 80.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_yaw_rate', 'p99': 3.1423631071896705, 'max': 16.370475958360878, 'min': 0.0006558123303606169, 'expected_range': [-2.0, 2.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_yaw_rate', 'p99': 12.593758583068844, 'max': 31.415817260742188, 'min': 0.0017428398132324219, 'expected_range': [-2.0, 2.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_lateral_accel', 'p99': 6.118913210639888, 'max': 194.0924073832214, 'min': 0.015322627441702343, 'expected_range': [-8.0, 8.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_lateral_accel', 'p99': 31.955756664275825, 'max': 771.9446516036987, 'min': 0.03143956098938361, 'expected_range': [-8.0, 8.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_rms_curvature', 'p99': 1.8439568756130393, 'max': 1312.4450825882468, 'min': 8.30845516729268e-05, 'expected_range': [-1.0, 1.0]}
- metric_physical_range_warning: {'warning': 'metric_physical_range_warning', 'source': 'raw', 'metric_name': 'raw_max_abs_curvature', 'p99': 12.766613260247453, 'max': 11738.86489868164, 'min': 0.0001924390472764212, 'expected_range': [-1.0, 1.0]}
- raw_metric_physically_implausible: {'warning': 'raw_metric_physically_implausible'}
- physical_metric_clipping_applied: {'warning': 'physical_metric_clipping_applied', 'smoothing_window': 5}
- completed: {'warning': 'completed', 'total_rows': 164871, 'elapsed_sec': 99.69483280181885}
