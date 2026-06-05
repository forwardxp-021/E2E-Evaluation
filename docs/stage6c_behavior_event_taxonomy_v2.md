# Stage 6C v2 — Task-conditioned behavior-event BDD taxonomy

Stage 6C v2 的名称是 **Task-conditioned behavior-event BDD**。它用于诊断：在相同 driving task / behavior-event context 下，A/B driving policies 或 model versions 的 learned behavior embedding distribution 是否发生 style drift，并用 task-specific handcrafted metrics 解释 drift 方向。

核心原则：

```text
Event bin = task slice / comparable driving context
BDD = embedding distribution difference within this task
Style metrics = semantic explanation of the detected drift
```

因此 v2 不再把 hard_brake、late_brake 等 outcome bins 作为主要评价对象。这些 outcome-style bins 只保留为可选 post-hoc diagnostics；主报告必须以 task-conditioned BDD 为主。

## 1. 研究定位

1. **Stage 6C v2 是 task-conditioned behavior style drift diagnosis**：先定义可比较的驾驶任务切片，再在每个任务内比较 A/B embedding distribution。
2. **主指标是 embedding-based BDD**：BDD 衡量同一 task slice 内 learned behavior embedding 的 distribution difference。
3. **Task-specific handcrafted metrics 是解释层**：它们帮助说明 BDD 的方向，例如更小 THW、更大 jerk、更高 yaw rate；它们不是 BDD 的替代品。
4. **旧 outcome bins 降级为 post-hoc diagnostics**：hard_brake / late_brake 等只能用于解释行为表现，不应主导 Stage 6C v2 结论。
5. **推荐完整实验集**：
   - `negative_control_random`
   - `pseudo_agg_vs_cons`
   - `scene_confounding_control`
6. **报告解释方式**：
   - `negative_control_random`：sanity check，BDD 应低且不呈系统性漂移。
   - `pseudo_agg_vs_cons`：positive control，style drift 应定位到 following、lane-change、overtake、yield conflict 等相关 tasks。
   - `scene_confounding_control`：confounding diagnosis，drift 可能集中在 task exposure imbalance 或 dynamic interaction pressure 中。

## 2. 输出与诊断约定

`tools/stage6c_build_behavior_events_v2.py` 输出：

- `behavior_event_bins_v2.csv`：每行一个 window，包含 `global_row` 与 task label。
- `behavior_event_metrics_v2.csv`：每行一个 window，包含 `global_row` 与 task-specific style metrics。
- `behavior_event_schema_v2.json`：task labels、detector strength、diagnostics 与阈值。
- `behavior_event_report_v2.md`：构建报告。
- `behavior_event_warnings_v2.json`：缺失 raw arrays、proxy detector、degenerate task 等 warning。

所有 task detector 输出三值标签：positive label / negative label / `unknown`。缺失 metric 必须写 `NaN`，不得用 0 静默填充。

## 3. First-priority task-conditioned behavior events

Stage 6C v2 首批只定义以下 8 类优先事件；其中 overtake 拆为 opportunity 与 execution 两个 task columns，因为 task exposure 与 behavior response 需要分开解释。

### E1. Following / Car-following

- **Task column**：`task_following`
- **Positive / negative label**：`following` / `not_following`
- **Goal**：在 following 场景下比较 A/B 的 following distance、THW、braking intensity 与 comfort。
- **Detector**：front vehicle 有足够有效帧；front distance 与 THW 有效；可选要求 ego speed 高于 low-speed threshold。
- **Metrics**：
  - `following_mean_thw`
  - `following_min_thw`
  - `following_mean_front_distance`
  - `following_min_front_distance`
  - `following_front_closing_rate_mean`
  - `following_front_closing_rate_p95`
  - `following_peak_decel`
  - `following_rms_jerk`
  - `following_max_abs_jerk`
  - `following_late_brake_score`
  - `following_aggressiveness_score`
- **Interpretation**：如果 BDD 高且 B 的 THW/front gap 更低、decel/jerk 更高，则 B 呈现更贴近跟车与更急制动风格。

### E2. Lead Vehicle Braking Response

- **Task column**：`task_lead_brake_response`
- **Positive / negative label**：`lead_brake_response` / `no_lead_brake_response`
- **Goal**：当前车刹车时比较 reaction delay、safety margin、braking intensity 与 comfort。
- **Detector**：front vehicle 有效；front vehicle 出现显著减速；front gap / THW 有效。若 raw lead acceleration 不可用，可用 front closing-rate derivative 作 proxy，并记录 warning / detector strength。
- **Metrics**：
  - `lead_brake_front_decel_start_time`
  - `lead_brake_ego_brake_start_time`
  - `lead_brake_reaction_delay`
  - `lead_brake_min_ttc_after_lead_brake`
  - `lead_brake_min_thw_after_lead_brake`
  - `lead_brake_peak_decel_after_lead_brake`
  - `lead_brake_max_jerk_after_lead_brake`
  - `lead_brake_speed_drop_after_lead_brake`
  - `lead_brake_late_response_score`
- **Interpretation**：若 B 反应更晚、TTC/THW 更低且 decel/jerk 更强，则 B 在 lead-braking response 中更不提前预判且舒适性更差。

### E3. Queue Approach / Stopped-traffic Approach

- **Task column**：`task_queue_approach`
- **Positive / negative label**：`queue_approach` / `no_queue_approach`
- **Goal**：接近慢行/停车队列时比较 early-vs-late braking、final gap 与 smoothness。
- **Detector**：front vehicle 存在；front vehicle speed 低或接近 0；ego 从较高速度接近。若 front speed 不可用，可用 front gap、THW 与 closing-rate 作 conservative proxy。
- **Metrics**：
  - `queue_distance_when_start_decel`
  - `queue_time_to_stop`
  - `queue_final_front_gap`
  - `queue_peak_decel`
  - `queue_rms_jerk`
  - `queue_stop_smoothness_score`
  - `queue_creep_after_stop_score`
- **Interpretation**：若 B 开始减速更晚、final gap 更短、peak decel 与 jerk 更高，则 B 的 queue approach 更晚且更急。

### E4. Lane Change

- **Task column**：`task_lane_change`
- **Positive / negative label**：`lane_change` / `no_lane_change`
- **Goal**：在 lane-change events 下比较 steering sharpness、lateral smoothness 与 gap acceptance。
- **Detector**：lateral displacement / lane-change count proxy 为正，或 yaw / curvature / heading change / lateral speed 较高。
- **Metrics**：
  - `lc_rms_yaw_rate`
  - `lc_rms_curvature`
  - `lc_heading_change_total`
  - `lc_max_lateral_speed`
  - `lc_rms_lateral_accel`
  - `lc_duration`
  - `lc_oscillation_score`
  - `lc_target_front_gap_min`
  - `lc_target_rear_gap_min`
  - `lc_gap_acceptance_score`
  - `lc_sharpness_score`
- **Interpretation**：如果 BDD 高且 B 的 yaw/curvature/lateral accel 更高、target-lane gap 更小，则 B 的变道更 sharp 且更 assertive。

### E5. Cut-in Response

- **Task column**：`task_cutin_response`
- **Positive / negative label**：`cutin_response` / `no_cutin_response`
- **Goal**：其他车辆 cut in 时比较 response delay、braking timing、safety margin 与 comfort。
- **Detector**：neighbor 从 side/front-side 转入 front；或 front gap 突然下降且出现新 front vehicle。优先使用 `neighbor_seq.npy` 与 `neighbor_slot_ids.npy`；如果 slot IDs 不可用，使用 conservative proxy 并写 warning。
- **Metrics**：
  - `cutin_gap_initial`
  - `cutin_gap_min`
  - `cutin_min_ttc`
  - `cutin_reaction_delay_to_brake`
  - `cutin_peak_decel_after_cutin`
  - `cutin_jerk_after_cutin`
  - `cutin_speed_drop_after_cutin`
  - `cutin_yielding_response_score`
  - `cutin_late_response_score`
- **Interpretation**：若 B reaction delay 更长、min TTC 更低、peak decel/jerk 更高，则 B 对 cut-in 响应更晚且更突兀。

### E6. Overtake / Passing Opportunity and Execution

- **Task columns**：`task_overtake_opportunity` 与 `task_overtake_executed`
- **Positive / negative labels**：
  - `overtake_opportunity` / `no_overtake_opportunity`
  - `overtake_executed` / `no_overtake_executed`
- **Goal**：在 overtake opportunity 下比较 passing willingness 与 acceleration/braking aggressiveness。
- **Detector**：opportunity positive 要求 front vehicle present 且更慢、front gap 不太远、adjacent lane/context 可用；execution positive 要求 opportunity 存在且 ego 有 lane-change-like / passing-like maneuver，或加速并绕过慢车。
- **Metrics**：
  - `overtake_opportunity_score`
  - `overtake_execution_score`
  - `overtake_execution_rate_proxy`
  - `overtake_time_to_initiate`
  - `overtake_peak_accel`
  - `overtake_peak_decel`
  - `overtake_max_abs_jerk`
  - `overtake_min_front_gap_before`
  - `overtake_target_lane_front_gap`
  - `overtake_target_lane_rear_gap`
- **Interpretation**：若 B execution score 更高、time to initiate 更短、accel 更高且 accepted gap 更小，则 B 更愿意超车且更激进。

### E7. Hesitation / Aborted Maneuver

- **Task column**：`task_hesitation`
- **Positive / negative label**：`hesitation` / `no_hesitation`
- **Goal**：比较 driver/model 在 maneuvering 中是否犹豫或出现 abort-like 行为。
- **Detector**：高 yaw / lateral velocity sign changes；lane-change duration 长；lane-change attempt without completion；speed/yaw/lateral motion oscillatory。
- **Metrics**：
  - `hesitation_score`
  - `hesitation_lc_duration`
  - `hesitation_yaw_sign_change_count`
  - `hesitation_lateral_velocity_sign_change_count`
  - `hesitation_lc_oscillation_score`
  - `hesitation_abort_like_score`
  - `hesitation_speed_drop`
- **Interpretation**：若 BDD 高且 B oscillation 更高、duration 更长、abort-like score 更高，则 B 更犹豫/更不果断。

### E8. Yield Conflict / Interaction Assertiveness

- **Task column**：`task_yield_conflict`
- **Positive / negative label**：`yield_conflict` / `no_yield_conflict`
- **Goal**：比较 driver/model 在 interaction pressure 下更倾向 yield 还是 compete。
- **Detector**：small front/side/rear gaps；存在 interaction pressure；side/front vehicle closing；ego 在 conflict 下保持速度或加速。
- **Metrics**：
  - `yield_conflict_score`
  - `yielding_score`
  - `assertiveness_score`
  - `gap_pressure_score`
  - `conflict_accel_score`
  - `small_gap_speed_maintain_score`
  - `rear_pressure_response_score`
  - `courtesy_score`
- **Interpretation**：若 B assertiveness 与 conflict accel 更高、yielding/courtesy 更低，并在小 gap 下保持速度，则 B 更 competitive / less yielding。

## 4. 使用建议

1. 先运行 `tools/stage6c_build_behavior_events_v2.py` 构建 task bins 与 metrics。
2. 再运行 `tools/stage6c_task_conditioned_bdd_report.py` 分别分析 `negative_control_random`、`pseudo_agg_vs_cons` 与 `scene_confounding_control`。
3. 解释结论时先读 task-level BDD，再读该 task 对应 metrics 的 B-A delta；不要把 outcome-style hard_brake / late_brake 当成主结论。
