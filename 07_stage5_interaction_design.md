# Stage 5：Lane-aware 5-neighbor Interaction-conditioned Behavior Embedding 设计方案

> 文档定位：Stage 5 设计对齐文档（design-first），用于明确输入、监督信号、建模路线与评估方案。  
> 非目标：本阶段不启动训练，不改写 Stage 4G/4H/4I 既有流程。

## 1. 研究目标

Stage 5 的核心目标是把 Stage 4G 的 **ego-only trajectory embedding** 扩展为 **interaction-conditioned behavior embedding**。  
即：不仅编码目标车自身轨迹形态，还编码其与周边车辆互动后的行为风格。

重点覆盖三类风格信号：

1. 纵向跟驰风格（longitudinal following style）
   - 是否跟车过近
   - 是否持续施压前车
   - 跟驰时的舒适性与动态平顺性
2. 横向稳定性 / 变道风格（lateral stability / lane-change style）
   - 变道频率与方向偏好
   - 横向稳定性与摆动
   - 变道过程持续时间与激进程度
3. 交互 / 缺口 / 让行风格（interaction / gap / yielding style）
   - 对左右侧 gap 的接受偏好
   - 后车压力下的行为变化
   - 是否更倾向让行或更 assertive

明确边界：

- Stage 5 **不是**对 Stage 4G 的替代。  
- Stage 4G 继续作为当前最佳 trajectory-only baseline。  
- Stage 5 是在此基础上的 interaction-aware 扩展路线。

## 2. 输入对象与窗口定义

- 样本单位仍为固定长度轨迹窗口。
- 默认窗口长度为 80 帧。
- Waymo 采样频率为 10Hz。
- 80 帧对应 8 秒。
- 该设置继承 Stage 4 工程管线，属于工程一致性选择。
- `window_len` 保持可配置，不锁死为 80。

输入对象说明：

- 评估对象是 target vehicle（ego-like vehicle），不要求是 Waymo 原始自车。
- 任意满足质量过滤条件的 vehicle agent 都可构成一个 target sample。
- Stage 5 的“ego”语义指“当前样本中的目标车”，不是数据采集车身份。

## 3. Lane-aware 5-neighbor assignment

### 3.1 固定 slot 定义

采用固定 5 个邻车 slot：

- `front`
- `left_front`
- `left_rear`
- `right_front`
- `right_rear`

### 3.2 lane-aware 优先分配规则

1. `front`
   - 与目标车同 lane
   - 候选满足 `s_neighbor > s_ego`
   - 取纵向距离最近者
2. `left_front`
   - 位于左侧相邻 lane
   - 候选满足 `s_neighbor > s_ego`
   - 取纵向距离最近者
3. `left_rear`
   - 位于左侧相邻 lane
   - 候选满足 `s_neighbor < s_ego`
   - 取纵向距离最近者
4. `right_front`
   - 位于右侧相邻 lane
   - 候选满足 `s_neighbor > s_ego`
   - 取纵向距离最近者
5. `right_rear`
   - 位于右侧相邻 lane
   - 候选满足 `s_neighbor < s_ego`
   - 取纵向距离最近者

### 3.3 fallback 机制

当 lane/map 关联失败时：

- 退化为 ego-centric 几何分配（基于相对位置与朝向分区）。
- fallback 触发必须落日志，便于后续统计与排查。

### 3.4 lane-map 依赖概念

实现 lane-aware 分配需要以下地图语义：

- `lane_id`
- lane centerline
- 轨迹点投影到 lane 的 `s/l` 坐标
- 相邻左/右 lane 拓扑关系
- lane heading

### 3.5 已知风险

- 路口区域 lane 投影可能失败。
- lane topology 在并线/岔路区域可能存在歧义。
- fallback 比例升高会降低 slot 语义纯度，因此必须记录并纳入诊断指标。

## 4. Model input schema

### 4.1 ego 每帧输入

`ego_x_local`  
`ego_y_local`  
`ego_vx_local`  
`ego_vy_local`  
`ego_heading_local`  
`ego_speed`  
`ego_accel`  
`ego_yaw_rate`

关键约束：

- heading 优先使用 Waymo 原始 object heading。
- yaw_rate 优先使用原始 heading 差分计算。
- velocity-heading 推导的 `yaw_rate_proxy` 仅作为 fallback。

### 4.2 neighbor（每个 slot）每帧输入

`valid_mask`  
`dx_ego`  
`dy_ego`  
`rel_vx_ego`  
`rel_vy_ego`  
`distance`  
`longitudinal_gap`  
`lateral_gap`  
`closing_rate`  
`ttc_proxy`  
`thw_proxy`  
`neighbor_speed`  
`neighbor_accel`  
`neighbor_heading_rel`  
`neighbor_yaw_rate`

输入约束：

- 不直接输入 `neighbor_global_x` / `neighbor_global_y` / 原始全局 heading。
- `neighbor_speed` 与 `neighbor_accel` 可保留（平移/旋转不变标量状态）。
- `neighbor_heading_rel` 可保留（相对 ego heading 的角度）。

### 4.3 张量格式

Flatten 版本：

- `X_context: [B, T, D_ego + 5 * D_neighbor]`

Slot encoder 版本：

- `ego_seq: [B, T, D_ego]`
- `neighbor_seq: [B, 5, T, D_neighbor]`
- `neighbor_mask: [B, 5]`

### 4.4 配套元数据文件

- `context_feature_names.json`
- `neighbor_slot_names.json`
- `neighbor_slot_valid_ratio.csv`

## 5. Weak supervision feature groups

弱监督特征为 window-level summary，不直接作为 encoder 输入，而用于：

- soft contrastive targets
- auxiliary head targets
- metric alignment targets
- evaluation / pseudo labels

### A. Longitudinal comfort / following features

`rms_accel`  
`rms_jerk`  
`max_abs_accel`  
`max_abs_jerk`  
`mean_thw`  
`min_thw`  
`front_mean_distance`  
`front_min_distance`  
`front_closing_rate_mean`  
`front_closing_rate_p95`

> 命名统一为 “longitudinal comfort / following features”，避免仅称 comfort features。

### B. Lateral stability / lane-change features

`rms_yaw_rate`  
`rms_curvature`  
`heading_change_total`  
`lane_change_count`  
`lane_change_rate`  
`lane_change_left_count`  
`lane_change_right_count`  
`lane_change_duration_mean`  
`max_lateral_speed`  
`rms_lateral_accel`  
`lane_change_oscillation_score`

说明：

- `lane_change_rate = lane_change_count / window_duration`
- 变道行为是驾驶风格中的关键可辨识信号。

### C. Interaction / gap / yielding features

`front_pressure_score`  
`left_front_min_gap`  
`left_rear_min_gap`  
`right_front_min_gap`  
`right_rear_min_gap`  
`left_gap_min`  
`right_gap_min`  
`left_gap_acceptance_proxy`  
`right_gap_acceptance_proxy`  
`rear_vehicle_pressure_proxy`  
`yielding_score_proxy`  
`assertiveness_score_proxy`

## 6. Explicit head design

三类显性 head 主要用于辅助训练与可解释诊断；最终检索/距离评估仍以 embedding `z` 为核心。

### A. Longitudinal Head（预测）

`rms_accel`  
`rms_jerk`  
`max_abs_accel`  
`max_abs_jerk`  
`mean_thw`  
`min_thw`  
`front_min_distance`  
`front_closing_rate_p95`

### B. Lateral Head（预测）

`rms_yaw_rate`  
`rms_curvature`  
`heading_change_total`  
`lane_change_count`  
`lane_change_rate`  
`max_lateral_speed`  
`rms_lateral_accel`

### C. Interaction Head（预测）

`front_pressure_score`  
`left_gap_min`  
`right_gap_min`  
`left_rear_min_gap`  
`right_rear_min_gap`  
`gap_acceptance_score`  
`yielding_score_proxy`  
`assertiveness_score_proxy`

目标：证明 `z` 同时保留 longitudinal / lateral / interaction 可解释信息。

## 7. Model architecture

## Version A：Flatten Context GRU

输入：

- `X_context [B, T, D_ego + 5 * D_neighbor]`

结构：

`Context sequence -> GRU -> projection head -> z -> longitudinal/lateral/interaction heads`

定位：

- 快速工程验证版本
- 与当前 Stage 4G 代码形态最接近

## Version B：Slot Encoder + Attention Pooling

输入：

- `ego_seq [B, T, D_ego]`
- `neighbor_seq [B, 5, T, D_neighbor]`
- `neighbor_mask [B, 5]`

结构：

1. `ego_seq -> ego encoder -> h_ego`
2. 对每个 slot `k`：
   - `neighbor_seq_k -> shared neighbor encoder -> h_k`
   - `h_k + slot_embedding[k]`
3. `h_ego` 对 slot 表示执行 masked attention 或 gated pooling，得到 `h_context`
4. 融合：`concat(h_ego, h_context) -> projection head -> z -> 三个显性 heads`

优势：

- 保留 slot 语义
- 更自然处理缺失邻车
- 可输出 attention 权重
- 支持可解释性诊断：
  - 跟驰场景中 `front` 权重升高
  - 左变道场景中左侧 slot 权重升高
  - 右变道场景中右侧 slot 权重升高

## 8. Loss design

完整目标（未来态）定义：

```text
L_total =
  L_soft_style
  + lambda_long_aux * L_longitudinal_head
  + lambda_lat_aux * L_lateral_head
  + lambda_int_aux * L_interaction_head
  + lambda_long_metric * L_longitudinal_metric_alignment
  + lambda_lat_metric * L_lateral_metric_alignment
  + lambda_int_metric * L_interaction_metric_alignment
```

实现原则：首次实现不要一次性全部开启。

推荐分阶段落地：

- Stage 5A：数据 schema + lane-aware neighbor 抽取
- Stage 5B：Flatten context GRU + 保留 Stage 4G comfort metric alignment，先验证 5-neighbor 输入收益
- Stage 5C：加入 lateral head 与 interaction head
- Stage 5D：加入 lateral / interaction metric alignment
- Stage 5E：升级为 slot encoder + attention pooling

## 9. Evaluation plan

### 9.1 复用 Stage 4 指标

`centroid accuracy`  
`hit@1`  
`topK same-label fraction`  
`rms_jerk_delta`  
`rms_yaw_rate_delta`  
`rms_curvature_delta`  
`mean_speed_delta`

### 9.2 新增 Stage 5 交互指标

`front_min_thw_delta`  
`front_min_distance_delta`  
`front_closing_pressure_delta`  
`lane_change_rate_delta`  
`left_gap_min_delta`  
`right_gap_min_delta`  
`gap_acceptance_delta`  
`yielding_score_delta`

### 9.3 诊断指标

`neighbor slot coverage`  
`front_valid_ratio`  
`left_front_valid_ratio`  
`left_rear_valid_ratio`  
`right_front_valid_ratio`  
`right_rear_valid_ratio`  
`lane-aware assignment success rate`  
`fallback assignment rate`  
`attention weight distribution`（slot encoder 版本）

## 10. Expected outputs

Stage 5 预计新增输出文件：

- `context_traj.npy`
- `context_mask.npy`
- `neighbor_slot_ids.npy`
- `neighbor_slot_names.json`
- `context_feature_names.json`
- `interaction_feat_style.npy`
- `interaction_feature_names.json`
- `neighbor_context_summary.json`
- `neighbor_slot_valid_ratio.csv`
- `lane_assignment_debug.csv`

约束：不得覆盖既有 Stage 4 产物：

- `traj.npy`
- `feat_style.npy`
- `embeddings_row_level_comfort_metric.npy`

## 11. Risks and reviewer-facing notes

- 5-neighbor 上下文提高行为真实性，但会提升对 lane-map 质量的依赖。
- lane-aware assignment 在路口/拓扑复杂区域可能歧义。
- weak supervision features 属于人工构造统计，定位应是 metric-aligned behavior embedding，而非纯无监督发现。
- 不应直接使用邻车全局坐标作为输入。
- heading 在可用时应优先使用 raw heading，而非仅依赖 velocity proxy。
- pseudo labels 仍是评估标签，不作为训练真值标签。


## Stage 5A implementation status

- Status: planned / implementation in progress.
- This stage is currently focused on data construction and diagnostics only.
- No Stage 5 training results are claimed in this document section.
