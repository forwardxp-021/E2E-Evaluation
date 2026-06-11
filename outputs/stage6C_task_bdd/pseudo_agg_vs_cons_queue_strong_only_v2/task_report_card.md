# Stage 6C v2 task-conditioned behavior-event BDD report

BDD detects distribution shift in learned embedding space. Task-specific metrics explain the drift direction.

本报告的主评价单元是 driving task / behavior-event slice 内的 BDD；hard_brake、late_brake 等 outcome-style 表现只应作为可选 post-hoc 诊断，而不是主结果。

## Task BDD summary

- No task passed the min_bin_size / validity filters.

## Style metric explanation layer

- No valid task-specific metric deltas were available.

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

- `task_queue_approach`: below_min_bin_size (n_A=1233, n_B=50, validity=valid)

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
- completed: {'warning': 'completed', 'valid_task_count': 0, 'skipped_task_count': 1, 'embedding_rows': 164871, 'event_rows': 164871}
