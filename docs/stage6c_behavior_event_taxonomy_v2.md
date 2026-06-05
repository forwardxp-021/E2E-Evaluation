# Stage 6C Behavior-Event Taxonomy v2：Task-conditioned Style Drift

## 0. 设计动机

当前 Stage 6C 的 exposure/outcome bins 过粗，容易把“任务切片”和“行为结果”混在一起。Stage 6 的研究目标是：

> 在相同 driving task / event 下，学习到的 behavior embedding 是否能区分 A/B driving policies 或 model versions 的行为风格差异？

因此 v2 不再把 outcome bins 作为主要评价对象，而是将 **behavior-event bin 定义为 task slice / comparable driving context**，在每个 task 内计算 embedding distribution difference（BDD），再用 handcrafted style metrics 解释 drift 方向。

核心原则：

- **Event bin = task slice / comparable driving context**；
- **BDD = embedding distribution difference within this task**；
- **Style metrics = semantic explanation of the detected drift**。

推荐主报告表述：

> The main Stage 6C report emphasizes exposure/task-conditioned BDD. Outcome-like handcrafted metrics are used as semantic explanations, not as the primary evaluation object.

## 1. 输出与行对齐约束

新增脚本：`tools/stage6c_build_behavior_events_v2.py`。

输入：

- `--shard_manifest` 指向 sharded dataset manifest；
- 每个 shard 优先读取 raw arrays：
  - `ego_seq.npy`；
  - `neighbor_seq.npy`；
  - `neighbor_slot_ids.npy`（如果存在，用于记录/诊断 neighbor slot 语义）；
  - `meta.npy`；
  - `interaction_feat_style.npy`（只作为可选辅助/一致性检查，不只依赖 33 aggregate features）。

输出目录内生成：

1. `behavior_event_bins_v2.csv`：每个 event detector 的 `positive` / `negative` / `unknown` 标签；
2. `behavior_event_metrics_v2.csv`：每个 task 的 style explanation metrics；
3. `behavior_event_schema_v2.json`：taxonomy、阈值、array layout 假设、event/metric diagnostics；
4. `behavior_event_report_v2.md`：中文/英文混合的可读诊断报告；
5. `behavior_event_warnings_v2.json`：缺失 array、metadata mismatch、退化 event 等 warning。

行对齐要求：

- 必须保留 `global_row`，按 `shard_manifest` 的 shard 顺序从 0 递增；
- 必须保留 `shard_id` 与 `local_row`；
- 必须尽量透传 metadata：`scenario_id`、`target_agent_id`、`start`、`window_len`、`split`；
- 不得默认合并 `ego_seq.npy` / `neighbor_seq.npy` 等大数组；
- 缺失 metric 必须写 `NaN`，不得用 0 填充。

## 2. Detector 与 diagnostics 统一规则

每个 event detector 必须输出三值状态：

- `positive`：该 window 属于该 task slice；
- `negative`：raw signals 可用，但该 window 不属于该 task slice；
- `unknown`：缺少必要 raw signal、shape 不满足要求、或 detector 无法判断。

每个 event 必须报告 validity diagnostics：

- `positive_ratio`；
- `unknown_ratio`；
- `n_positive`；
- `n_negative`；
- `degenerate`：当 `positive_ratio < 0.01` 或 `positive_ratio > 0.95` 时标记为 true。

每个 metric 必须报告 metric diagnostics：

- `valid_count`；
- `valid_rate`；
- `p01` / `p50` / `p99` / `min` / `max`。

BDD 报告中应跳过或单独标注 unknown-heavy / degenerate event，避免将无效切片写成研究结论。

## 3. Primary behavior-event taxonomy

### 3.1 Following / car-following

**Goal**：在 following task 下，比较 A/B 是否在 following distance、THW、braking intensity、comfort 方面存在风格差异。

**Detector**：

- front vehicle valid for sufficient frames；
- valid front distance and THW；
- 可选：ego speed above low-speed threshold，避免把停车/低速蠕行误判为 following。

**Metrics**：

- `mean_thw`, `min_thw`；
- `mean_front_distance`, `min_front_distance`；
- `front_closing_rate_mean`, `front_closing_rate_p95`；
- `peak_decel`, `rms_jerk`, `max_abs_jerk`；
- `late_brake_score`；
- `following_aggressiveness_score`。

**BDD interpretation**：

- BDD 升高说明同为 following 的 embedding 分布发生漂移；
- 若 `mean_thw` / `mean_front_distance` 下降且 `peak_decel` / `rms_jerk` 上升，可解释为更贴近、更晚刹或更激进；
- 若 THW 上升且 jerk 降低，可解释为更保守/更舒适。

### 3.2 Lane change

**Goal**：在 lane-change events 下，比较 steering sharpness 与 gap acceptance。

**Detector**：

- `lane_change_count_proxy > 0`；或
- high yaw / curvature / heading change / lateral displacement。

**Metrics**：

- `rms_yaw_rate`, `rms_curvature`, `heading_change_total`；
- `max_lateral_speed`, `rms_lateral_accel`；
- `lane_change_duration`；
- `lane_change_oscillation_score`；
- `target_front_min_gap_during_lane_change`, `target_rear_min_gap_during_lane_change`；
- `lane_change_sharpness_score`；
- `gap_acceptance_score`。

**BDD interpretation**：

- BDD 升高说明同为 lane-change 的行为 embedding 分布不同；
- sharpness 指标升高表示更急转向/更激烈 lateral control；
- target lane gaps 降低或 `gap_acceptance_score` 升高表示更小 gap acceptance。

### 3.3 Overtake / passing

**Goal**：在 overtake opportunity 下，比较 willingness to pass 以及 acceleration/braking aggressiveness。

**Detector**：

- front vehicle present and slower；
- front gap not too far；
- adjacent lane/context available。

**Metrics**：

- `overtake_opportunity_score`；
- `overtake_execution_score`；
- `time_to_initiate_overtake`；
- `peak_accel_during_overtake`；
- `peak_decel_during_overtake`；
- `jerk_during_overtake`；
- `min_front_gap_before_overtake`；
- `target_lane_front_gap`；
- `target_lane_rear_gap`。

**BDD interpretation**：

- BDD 衡量同等 overtake opportunity 下 embedding 是否漂移；
- execution score / acceleration 上升表示更愿意通过或更积极超车；
- delay 变长、execution score 降低表示更保守或更迟疑。

### 3.4 Cut-in response

**Goal**：当其他车辆 cut in 时，比较 response delay、braking timing 与 comfort。

**Detector**：

- neighbor transitions from side/front-side context into front position；或
- front gap suddenly decreases with a newly appearing front vehicle。

**Metrics**：

- `cutin_gap_initial`；
- `cutin_gap_min`；
- `cutin_min_ttc`；
- `reaction_delay_to_brake`；
- `peak_decel_after_cutin`；
- `jerk_after_cutin`；
- `speed_drop_after_cutin`；
- `yielding_response_score`；
- `late_response_score`。

**BDD interpretation**：

- BDD 升高表示同等 cut-in exposure 下 response style 发生变化；
- reaction delay / late response 上升表示响应更晚；
- peak decel / jerk 上升表示制动更急、舒适性下降；
- yielding response 上升表示更让行。

### 3.5 Hesitation / aborted maneuver

**Goal**：比较 driver/model 在 maneuvering 过程中是否更容易犹豫或 abort。

**Detector**：

- high yaw / lateral velocity sign changes；
- long lane-change duration；
- lane-change attempt without completion；
- oscillatory speed / yaw / lateral motion。

**Metrics**：

- `hesitation_score`；
- `lane_change_duration`；
- `yaw_sign_change_count`；
- `lateral_velocity_sign_change_count`；
- `lane_change_oscillation_score`；
- `abort_like_score`；
- `speed_drop_during_hesitation`。

**BDD interpretation**：

- BDD 升高说明 maneuvering task 内的 embedding 分布变化；
- hesitation/oscillation/abort-like 指标上升表示更犹豫或横向控制更反复。

### 3.6 Yield conflict / interaction assertiveness

**Goal**：比较 driver/model 在 conflict 下更倾向 yield 还是 compete。

**Detector**：

- small front / side / rear gaps；
- interaction pressure；
- side/front vehicle closing；
- ego maintains speed or accelerates under conflict。

**Metrics**：

- `yielding_score`；
- `assertiveness_score`；
- `gap_pressure_score`；
- `conflict_accel_score`；
- `small_gap_speed_maintain_score`；
- `rear_pressure_response`；
- `courtesy_score`。

**BDD interpretation**：

- BDD 升高说明相似 conflict exposure 下 interaction style 变化；
- assertiveness / conflict accel 上升表示更竞争；
- yielding / courtesy 上升表示更让行；
- gap pressure 必须作为 context 强度解释，不应单独当作安全结论。

## 4. Second-priority events

### 4.1 Free cruising stability

**Goal**：在无明显 front pressure 与 lane-change 的自由巡航中，比较 speed / yaw / jerk 稳定性。

**Detector**：front vehicle absent or far、非 lane-change、ego speed above low-speed threshold。

**Metrics**：`cruise_speed_std`、`cruise_yaw_rate_rms`、`cruise_rms_jerk`。

### 4.2 Stop-and-go / low-speed creep

**Goal**：在低速/拥堵蠕行中比较 creep smoothness 与停车-起步风格。

**Detector**：low-speed frames 占比高，或 stop frames 占比高。

**Metrics**：`stop_go_low_speed_ratio`、`stop_go_stopped_ratio`、`rms_jerk`、`peak_decel`。

### 4.3 Risk proximity

**Goal**：定位小 gap / 小 TTC 的 proximity slice，但不直接作为 outcome 主评价。

**Detector**：front / side / rear minimum gap 小，或 TTC 小。

**Metrics**：`risk_min_any_gap`、`risk_min_ttc`、`peak_decel`、`max_abs_jerk`。

### 4.4 Interaction comfort

**Goal**：在存在 interaction 的窗口中解释 comfort drift。

**Detector**：任意 neighbor valid 或 interaction pressure 可用。

**Metrics**：`interaction_comfort_rms_jerk`、`interaction_comfort_rms_yaw_rate`、`rms_lateral_accel`。

## 5. 后续 task-conditioned BDD 使用方式

v2 artifacts 生成后，Stage 6C 的 BDD 应优先在 primary task slice 内运行：

- `following == positive`；
- `lane_change == positive`；
- `overtake == positive`；
- `cutin_response == positive`；
- `hesitation == positive`；
- `yield_conflict == positive`。

计划用于三类 split：

- `negative_control_random`：同分布随机拆分；期望 task-conditioned BDD 低且无系统性方向；
- `pseudo_agg_vs_cons`：伪 aggressive vs conservative；期望 following / lane-change / conflict 等 task 内 BDD 明显升高，并由 style metrics 解释方向；
- `scene_confounding_control`：场景混杂控制；检查 task-conditioned BDD 是否仍被 scene confounding 推高。

主报告必须优先强调 exposure/task-conditioned BDD，而不是 outcome bins。Outcome-like metrics 只能写成 drift explanation，例如“在 following task 内，A/B 的 BDD 升高，同时 B 的 THW 更低、peak decel 更高，因此解释为更激进 following style”。
