# Stage 6C v2 — Task-conditioned behavior-event BDD taxonomy

Stage 6C v2 的名称是 **Task-conditioned behavior-event BDD**。它用于诊断：在相同 driving task / behavior-event context 下，A/B driving policies 或 model versions 的 learned behavior embedding distribution 是否发生 style drift，并用 task-specific handcrafted metrics 解释 drift 方向。

核心原则：

```text
Event bin = task slice / comparable driving context
BDD = embedding distribution difference within this task
Style metrics = semantic explanation of the detected drift
```

因此 v2 不再把 hard_brake、late_brake 等 outcome bins 作为主要评价对象。这些 outcome-style bins 只保留为可选 post-hoc diagnostics；主报告必须以 task-conditioned BDD 为主。

**当前可靠性结论（Stage 6C v2 quality pass）**：`following` 与 `yield_conflict` 目前是最可靠的 strong detectors；`cutin`、`overtake` 以及相当一部分 `lead_brake` / `queue` 仍然是 proxy-based；`lane_change` 与 `hesitation` 已收紧，但只有在 `positive_ratio <= 0.40`、且没有 `lane_change_detector_broad` / `hesitation_detector_broad` warning 时，才适合作为稳定 task-conditioned BDD 结论。

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
- **Detector**：front vehicle 有效；当前实现用 front closing-rate derivative 作为 lead-brake proxy signal，并记录 `lead_brake_uses_front_closing_derivative_proxy` / `task_lead_brake_response_strength=proxy`。只有未来接入可靠 raw lead acceleration 后，才可称为 true lead acceleration detector。
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
- **Detector**：front vehicle 存在；若 front speed column 可用，则检查 front stopped ratio，并输出 `queue_front_speed_min`、`queue_front_speed_mean`、`queue_front_stopped_ratio`。若 front speed 不可用，则只可用 front gap、THW 与 closing-rate 作 conservative proxy，并记录 `queue_approach_uses_gap_thw_closing_proxy`。
- **Metrics**：
  - `queue_distance_when_start_decel`
  - `queue_time_to_stop`
  - `queue_final_front_gap`
  - `queue_peak_decel`
  - `queue_rms_jerk`
  - `queue_stop_smoothness_score`
  - `queue_creep_after_stop_score`
  - `queue_front_speed_min`
  - `queue_front_speed_mean`
  - `queue_front_stopped_ratio`
- **Interpretation**：若 B 开始减速更晚、final gap 更短、peak decel 与 jerk 更高，则 B 的 queue approach 更晚且更急。

### E4. Lane Change

- **Task column**：`task_lane_change`
- **Positive / negative label**：`lane_change` / `no_lane_change`
- **Goal**：在 lane-change events 下比较 steering sharpness、lateral smoothness 与 gap acceptance。
- **Detector**：保守 lane-change detector 必须有足够横向位移；yaw-rate 或 heading-change 不能单独触发。Positive 条件为：`lateral_range >= --lane_change_lateral_range_m`（默认 2.5m），或存在 `lc_duration` 且 `lateral_range >= --lane_change_min_lateral_range_m`（默认 1.5m），或 heading/yaw evidence 高且横向位移至少达到该最小阈值。若 `positive_ratio > 0.40`，构建报告与 warnings 会写入 `lane_change_detector_broad`，此时只可作为 broad lateral-maneuver proxy 解释。
- **Metrics**：
  - `lc_rms_yaw_rate`
  - `lc_rms_curvature`
  - `lc_heading_change_total`
  - `lc_max_lateral_speed`（使用 `--lateral_speed_abs_cap` 默认 5.0m/s 裁剪后的值）
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
- **Detector**：true cut-in 需要 stable neighbor ID 的 side/front-side → front slot transition；当前实现尚未实现该 transition detector，因此只使用 front gap late appearance / sudden drop 的 conservative proxy，并记录 `cutin_true_slot_transition_not_implemented_using_gap_drop_proxy`。
- **Metrics**：
  - `cutin_gap_initial`
  - `cutin_gap_min`
  - `cutin_min_ttc`（仅来自真实 TTC column；不可用则为 `NaN`）
  - `cutin_min_thw`
  - `cutin_reaction_delay_to_brake`
  - `cutin_peak_decel_after_cutin`
  - `cutin_jerk_after_cutin`
  - `cutin_speed_drop_after_cutin`
  - `cutin_yielding_response_score`
  - `cutin_late_response_score`
- **Interpretation**：若 B reaction delay 更长、真实 min TTC 或 THW 更低、peak decel/jerk 更高，则 B 对 cut-in proxy/true cut-in slice 的响应更晚且更突兀；解释前必须检查 `task_cutin_response_strength`。

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
- **Detector**：必须先满足 maneuver context，并且至少满足 `--hesitation_min_evidence_count`（默认 2）个 evidence components：yaw sign changes 达阈值、lateral velocity sign changes 达阈值、`lc_duration >= --long_lane_change_s`、abort-like partial maneuver、maneuver 中明显 speed drop。`--hesitation_sign_changes` 默认 8。若 `positive_ratio > 0.40`，构建报告与 warnings 会写入 `hesitation_detector_broad`，此时不应作为稳定 hesitation 结论。
- **Metrics**：
  - `hesitation_score`
  - `hesitation_lc_duration`
  - `hesitation_yaw_sign_change_count`
  - `hesitation_lateral_velocity_sign_change_count`
  - `hesitation_lc_oscillation_score`
  - `hesitation_abort_like_score`
  - `hesitation_speed_drop`
  - `hesitation_evidence_count`
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

## 5. Detector strength and proxy limitations

Stage 6C v2 的核心概念保持不变：

```text
Event bin = task slice / comparable driving context
BDD = embedding distribution difference within this task
Style metrics = semantic explanation of the detected drift
```

但每个 task slice 的 detector 可信度必须显式解释。`behavior_event_bins_v2.csv` 为每个 task 输出 `*_strength` 列，取值为 `strong` / `proxy` / `weak_proxy` / `unknown`；`behavior_event_schema_v2.json` 汇总 detector strength counts；`task_bdd_summary.csv` 与 `task_report_card.md` 也会展示 positive task rows 的 detector strength distribution。解释 BDD 之前必须先检查这些 strength 与 warnings。

### 5.1 TTC 与 THW 命名边界

- `TTC` 只表示 `neighbor_seq.npy` 中真实 TTC column（当前 Waymo 5-neighbor builder layout 为 column 9）。
- `THW` 只表示 time headway（当前 layout 为 column 10）。
- 当前 v2 不允许把 THW 当作 TTC 输出；如果某个 shard 缺少 TTC column，则 `lead_brake_min_ttc_after_lead_brake`、`cutin_min_ttc` 等 TTC metrics 写为 `NaN`，并记录 `ttc_column_unavailable_metric_set_nan`。
- TTC/THW 加载后会清理哨兵和不合理值：`>=999`、`<=0`、TTC 大于 `--ttc_valid_max_s`（默认 30s）、THW 大于 `--thw_valid_max_s`（默认 30s）均写为 `NaN`；正式 metrics 与 diagnostic scores 不得出现 999 哨兵值。
- THW proxy 必须命名为 THW，例如 `lead_brake_min_thw_after_lead_brake`、`following_min_thw`、`cutin_min_thw`。

### 5.2 Cut-in response：true transition detector vs proxy fallback

- **True cut-in detector**：只有在实现了 stable neighbor ID 的 side/front-side → front slot transition 检测后，才可以把 `task_cutin_response_strength` 标为 `strong`。
- **当前实现**：仍使用 front vehicle late appearance 或 front-gap sudden drop 的 conservative proxy；即使 `neighbor_slot_ids.npy` 可加载，当前版本也不会声称真实 slot-ID transition 已实现。
- 当前实现会记录 `cutin_true_slot_transition_not_implemented_using_gap_drop_proxy`，并把 detector strength 标为 `proxy` 或 `weak_proxy`。

### 5.3 Lead-brake response：front-closing derivative proxy

- 当前 v2 没有直接读取 raw front vehicle acceleration。
- `task_lead_brake_response` 使用 `front_closing_rate` 的时间导数作为 lead-brake proxy signal，因此 detector strength 为 `proxy`，并记录 `lead_brake_uses_front_closing_derivative_proxy`。
- 在加入可靠 raw lead acceleration 之前，文档与报告不得声称该 detector 使用了真实 lead vehicle acceleration。

### 5.4 Queue approach：front speed available vs proxy fallback

- 如果 `neighbor_seq.npy` 提供 front neighbor speed column（当前 Waymo 5-neighbor builder layout 为 column 11），queue detector 会输出 `queue_front_speed_min`、`queue_front_speed_mean`、`queue_front_stopped_ratio` 诊断，并优先使用 stopped-front condition。
- 如果 front speed 不可用，则只使用 front gap、THW、closing-rate 与 ego speed 的 conservative proxy，并记录 `queue_approach_uses_gap_thw_closing_proxy`。
- 只有确认 front speed diagnostics 有效时，才把 queue approach 解释为 stopped-traffic approach；否则应解释为 queue-approach proxy。

### 5.5 Lateral / heading quality control

- `lc_max_lateral_speed` 使用 smoothed lateral velocity 后再按 `--lateral_speed_abs_cap`（默认 5.0m/s）裁剪，避免把瞬时噪声当作可解释 lane-change style。
- `lc_heading_change_total` 使用 wrap_angle 归一化后的 heading increment 总和，并按 `--heading_change_total_cap`（默认 8.0rad）封顶；lane-change 与 hesitation detector 也使用封顶后的 heading total。
- `behavior_event_schema_v2.json` 的 raw/clipped diagnostics 会记录 `raw_max_lateral_speed`、`clipped_max_lateral_speed`、`raw_heading_change_total`、`clipped_heading_change_total`，用于区分原始噪声与正式下游指标。
- `queue_distance_when_start_decel` 是距离指标，不属于 deceleration metric；physical warning 只应匹配 `peak_decel`、`*_decel_after_*`、`*_peak_decel` 等真实减速度指标。
