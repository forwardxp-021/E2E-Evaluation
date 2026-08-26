# R0 D4 Family-Specific Matching Contract v0.1

## Status

`READY_FOR_R0_V1_PROTOCOL_FREEZE`。本合同取代 24 项 global F_match 设计；它不执行 representation、BDD、rollout 或训练。

## Role rules

- `F_match` 只控制该 residual family 必须消除的核心固定人工摘要；
- `Context_match` 只允许 treatment/response 发生前已测量的 context；
- `M_behavior` 只确认 rollout/episode 后的 morphology/mechanism；
- 同一 feature 在同一 family 内不得同时是 Primary F_match 与 Primary M_behavior；
- 三个 family 不要求共享 F_match；
- whole-window THW/front-gap/closing 等会受 response 影响，全部移出 Primary F_match/Context_match；只有 frozen pre-treatment anchor 版本可作 Context_match。

## R-HLC

Primary F_match (4)：`ego13.mean_speed`, `ego13.end_minus_start_speed`, `ego13.heading_change_abs_total`, `ego13.path_length`。

Context_match：`context.map_location`, `context.road_class`, `context.log_id`, `context.intended_lane_change_direction`, `context.initial_speed_mps`, `context.initial_lane_offset_m`, `context.traffic_density`, `context.neighbor_availability_pattern`, `context.target_lane_initial_front_gap_m`, `context.target_lane_initial_rear_gap_m`。

M_behavior：`raw33.lane_change_count_proxy`, `raw33.lane_change_duration_mean_proxy`, `raw33.lane_change_oscillation_score_proxy`, `mechanism.hesitation_retreat_count`, `mechanism.commit_latency_s`, `mechanism.monotonic_transition_fraction`。

其余 target 均为 `Semantic_probe_only`，不参与该 family 的 Primary matching/mechanism gate。

## R-TSB

Primary F_match (4)：`ego13.mean_speed`, `ego13.end_minus_start_speed`, `ego13.mean_abs_accel`, `ego13.path_length`。

Context_match：`context.map_location`, `context.road_class`, `context.log_id`, `context.initial_speed_mps`, `context.initial_front_gap_m`, `context.initial_lead_relative_speed_mps`, `context.initial_thw_s`, `context.traffic_density`, `context.neighbor_availability_pattern`, `context.planned_stop_or_hazard_class`。

M_behavior：`raw33.rms_accel`, `raw33.rms_jerk`, `raw33.max_abs_jerk`, `mechanism.brake_phase_count`, `mechanism.interstage_release_fraction`, `mechanism.second_brake_peak_ratio`。

其余 target 均为 `Semantic_probe_only`，不参与该 family 的 Primary matching/mechanism gate。

## R-IP

Primary F_match (3)：`ego13.mean_speed`, `ego13.end_minus_start_speed`, `ego13.path_length`。

Context_match：`context.map_location`, `context.road_class`, `context.log_id`, `context.intended_lane_change_direction`, `context.initial_speed_mps`, `context.traffic_density`, `context.neighbor_availability_pattern`, `context.target_lane_initial_front_gap_m`, `context.target_lane_initial_rear_gap_m`, `context.target_lane_initial_rear_closing_speed_mps`, `context.gap_opportunity_class`。

M_behavior：`raw33.left_gap_acceptance_proxy`, `raw33.right_gap_acceptance_proxy`, `raw33.yielding_score_proxy`, `raw33.assertiveness_score_proxy`, `mechanism.gap_acceptance_latency_s`, `mechanism.minimum_accepted_rear_gap_m`, `mechanism.yield_response_onset_s`。

其余 target 均为 `Semantic_probe_only`，不参与该 family 的 Primary matching/mechanism gate。

## D4 development fallback

`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 冻结为 development-only bounded fallback。每个 Primary F_match 使用既有 Waymo TRAIN robust-IQR 证据中的 `0.10 × IQR` caliper，并要求该 family 全部核心 feature 通过。

它可用于 R0/R1 benchmark development、hard-negative construction 与 feasibility diagnosis；其科学状态是 `NOT_FORMAL_PHYSICAL_EQUIVALENCE`、`NOT_R4_CONFIRMATORY_EQUIVALENCE`。因此 D4 可成为 `CONDITIONALLY_NONBLOCKING_FOR_RBR_DEVELOPMENT`，但 fallback 本身不授权 RBR training。

R4 outcome 解盲前必须把每个 family 的 physical/material margin、TOST/IUT、cluster rule 与 final roster 一并冻结。若做不到，R4 equivalence 为 `NOT_EVALUABLE`，不得用 development caliper 替代。

机器角色表：`docs/stageR/r0/manifests/r0_d4_family_specific_feature_roles_v0.1.csv`；fallback：`docs/stageR/r0/manifests/r0_d4_development_balance_fallback_v1.json`。
