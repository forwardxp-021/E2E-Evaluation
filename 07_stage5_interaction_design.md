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

- Stage 5A-v1 geometric fallback passed.
- Stage 5A-v2 true lane-aware assignment is now required before training.
- Do not proceed to Stage 5B if fallback_assignment_rate remains 1.0.


## Stage 5A-v3 车道感知槽位规则收紧
- front_max_distance = 120m
- side_front_max_distance = 80m
- side_rear_max_distance = 120m
- lane_lateral_tolerance = 2.0m
- heading_diff_threshold = 45°
- static_speed_threshold = 0.5m/s

静止前车不会被自动丢弃：只要同车道且在前方可作为有效 front，并在诊断中标记为 `neighbor_is_static`。

路口场景说明：Waymo 在路口常有地图，但相邻车道拓扑可能存在歧义。Stage 5 清洁训练建议优先 `lane_context_quality=good`；无地图或 ego lane 缺失可在 clean 模式下丢弃。

## Stage 5A-V4 / Full51 Lane-aware 5-neighbor Context Dataset

Stage 5A 已从早期 prototype/sample 版本扩展到 full51 merged 数据集。

数据集目录：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged`

关键统计：

- `n_windows_kept = 164871`
- `n_shards = 35`
- `split_counts`：
  - `train = 131998`
  - `val = 16481`
  - `test = 16392`
- `context_dim = 83`
- `feature_dim = 33`
- `fallback_assignment_rate = 0`
- `nonfinite_output_detected = 0`
- `good_lane_context_rate ≈ 0.99`

解读：

这意味着 Stage 5 数据已不再是 toy subset，而是可用于正式训练/评估的全量分片数据集；lane-aware 5-neighbor context 稳定，特征输出干净。

重要约束：

- 该数据集是 **sharded** 形式。
- **不要**把 shard 直接拼接成一个巨大的 monolithic `.npy`。
- shard 内及 shard 间行顺序必须保持不变，后续 embedding 导出与评估对齐依赖该顺序。

## Stage 5A Feature Schema

Stage 5 现已采用严格 33 维特征 schema。

schema 路径：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json`

有序 33 维特征如下（按索引）：

0. `rms_accel`
1. `rms_jerk`
2. `max_abs_accel`
3. `max_abs_jerk`
4. `mean_thw`
5. `min_thw`
6. `mean_front_distance`
7. `min_front_distance`
8. `mean_rel_speed`
9. `p95_rel_speed`
10. `rms_yaw_rate`
11. `rms_curvature`
12. `heading_change_total`
13. `lane_change_count_proxy`
14. `lane_change_rate_proxy`
15. `lane_change_left_count_proxy`
16. `lane_change_right_count_proxy`
17. `lane_change_duration_mean_proxy`
18. `max_lateral_speed`
19. `rms_lateral_accel`
20. `lane_change_oscillation_score_proxy`
21. `front_pressure_score`
22. `left_front_min_gap`
23. `left_rear_min_gap`
24. `right_front_min_gap`
25. `right_rear_min_gap`
26. `left_gap_min`
27. `right_gap_min`
28. `left_gap_acceptance_proxy`
29. `right_gap_acceptance_proxy`
30. `rear_vehicle_pressure_proxy`
31. `yielding_score_proxy`
32. `assertiveness_score_proxy`

注意事项：

- `mean_speed` **不属于** Stage 5 schema。
- `std_rel_speed` **不属于** Stage 5 schema。
- 使用 `p95_rel_speed` 替代 `std_rel_speed`。
- 后续评估必须按 schema 名称解析索引，禁止依赖硬编码 fallback 索引。

## Stage 5B: Context GRU Baseline Embedding

训练脚本：

`tools/train_context_behavior_embedding.py`

Stage 5B baseline 命令（历史基线）：

```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 256 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_temperature 1.0 \
  --metric_alignment \
  --metric_loss_weight 0.1 \
  --metric_loss_type huber \
  --metric_targets all \
  --device cuda \
  --seed 42 \
  --overwrite
```

Stage 5B 输出目录：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1`

Stage 5B embedding 导出目录：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings`

Embedding manifest：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings/embedding_manifest.json`

Embedding 统计：

- `embedding_dim = 64`
- `total_rows = 164871`
- `embedding_shards = 35`
- `split = all`
- `nonfinite_embedding_detected = 0`
- 行对齐遵循 source shard 行顺序

解读：

Stage 5B 是 context-aware GRU 的基线 embedding，作为 Stage 5D group-weighted loss 改进前的对照。

## Stage 5C-V1: Preliminary Evaluation and Its Problem

评估脚本：

`tools/evaluate_context_embedding.py`

Stage 5C 初始评估比较了：

- `learned_context_embedding`
- `raw_feature`
- `pca_feature`
- `context_l2`
- `random`

初始现象：

- `learned_context_embedding` 明显优于 `random` 和 `context_l2`；
- 但落后于 `raw_feature` 与 `pca_feature`。

问题根因：

`evaluation_summary.json` 显示：

- `feature_names_used = []`
- 使用了 fallback feature indices；
- 错误假设了 `mean_speed` 与 `std_rel_speed`；
- evaluator 并不知道真实 33D schema。

结论：

Stage 5C-V1 仅是 smoke test，不是 paper-grade 证据。

## Stage 5C-1: Strict Feature Schema Evaluation

Stage 5C-1 修复了 evaluator 有效性，关键改动包括：

- 显式加载 `feature_schema.json`。
- 默认启用 strict feature schema。
- 禁止 fallback feature index resolution。
- 缺失 required features 直接报错。
- optional features 缺失时 warning 并跳过。
- 不再评估 `mean_speed` 与 `std_rel_speed`。
- 用 `p95_rel_speed` 替代 `std_rel_speed`。

paper-grade 校验结果：

- `strict_feature_schema = true`
- `feature_schema_loaded = true`
- no fallback feature index was used
- `missing_required_features = []`
- `warnings = []`
- `paper_grade_valid = true`
- `actual_eval_samples = 16392`
- `row_alignment_checks.aligned = true`

评估命令：

```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_schema_fixed \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

期望输出文件：

- `evaluation_summary.json`
- `retrieval_metrics.csv`
- `style_distance_correlation.csv`
- `context_sensitivity_metrics.csv`
- `retrieval_bar.png`
- `feature_delta_correlation_bar.png`
- `pca_embedding.png`
- `pca_feature.png`
- `evaluation_report.md`

## Stage 5C-2: Category-wise Evaluation

新增 Stage 5C-2 的原因：

仅看 global retrieval 会掩盖 embedding 在不同行为维度上的差异，因此新增 category-wise correlation summary 与 learned-win feature 分析。

新增输出文件：

- `category_correlation_summary.csv`
- `category_retrieval_summary.csv`
- `learned_win_features.csv`

类别分组：

- `longitudinal_comfort`
  - `rms_accel`
  - `rms_jerk`
  - `max_abs_accel`
  - `max_abs_jerk`
- `following_interaction`
  - `mean_thw`
  - `min_thw`
  - `mean_front_distance`
  - `min_front_distance`
  - `mean_rel_speed`
  - `p95_rel_speed`
  - `front_pressure_score`
  - `rear_vehicle_pressure_proxy`
- `lateral_lane_dynamics`
  - `rms_yaw_rate`
  - `rms_curvature`
  - `heading_change_total`
  - `lane_change_count_proxy`
  - `lane_change_rate_proxy`
  - `max_lateral_speed`
  - `rms_lateral_accel`
  - `lane_change_oscillation_score_proxy`
  - `left_front_min_gap`
  - `left_rear_min_gap`
  - `right_front_min_gap`
  - `right_rear_min_gap`
  - `left_gap_min`
  - `right_gap_min`
  - `left_gap_acceptance_proxy`
  - `right_gap_acceptance_proxy`
- `behavior_proxy`
  - `yielding_score_proxy`
  - `assertiveness_score_proxy`

Stage 5C-2 retrieval 结果：

| representation | hit@1 | hit@5 | mean_same_label_fraction_at_5 |
|---|---:|---:|---:|
| learned_context_embedding | 0.191862 | 0.490300 | 0.174024 |
| raw_feature | 0.266227 | 0.585774 | 0.233821 |
| pca_feature | 0.266959 | 0.595839 | 0.236323 |
| context_l2 | 0.085713 | 0.267692 | 0.080942 |
| random | 0.010920 | 0.060090 | 0.012384 |

结论：

- `learned_context_embedding` 显著优于 `context_l2` 与 `random`；
- 但在 global retrieval 上仍弱于 `raw_feature` 与 `pca_feature`。

Stage 5C-2 category-wise correlation summary：

| category | learned_context_embedding | raw_feature | pca_feature |
|---|---:|---:|---:|
| longitudinal_comfort | 0.150833 | 0.172702 | 0.174362 |
| following_interaction | 0.302917 | 0.469712 | 0.467968 |
| lateral_lane_dynamics | 0.266777 | 0.251786 | 0.251469 |
| behavior_proxy | 0.190567 | 0.296595 | 0.298821 |

核心结论：

- learned embedding 全局上仍低于 raw/pca retrieval baseline；
- learned embedding 在 lateral/lane-change dynamics 最强；
- 在 `lateral_lane_dynamics` 类别均值上优于 raw_feature 与 pca_feature；
- following/front-distance 相关目标仍明显偏弱；
- 这直接驱动 Stage 5D。

重要 feature-level learned wins（learned embedding 更强）：

- `lane_change_count_proxy`
- `lane_change_rate_proxy`
- `lane_change_oscillation_score_proxy`
- `max_lateral_speed`
- `heading_change_total`
- `rms_yaw_rate`
- `rms_lateral_accel`

解释：

序列编码器并非简单复制手工特征距离；它在横向时序行为上优于静态 feature-space 距离。但在 following interaction、THW、front distance 与前后车压力上仍存在欠表达。

## Stage 5D Training Adjustment Principle

Stage 5B 使用了 general context GRU embedding + 全局 style/metric 目标。Stage 5C-2 的诊断是：该 embedding 有效，但结构不均衡：

- 对 lateral/lane-change dynamics 捕捉较强；
- 对 following/front-distance interaction 捕捉偏弱。

因此 Stage 5D **不改数据集**，而是改训练目标：使用 multi-objective group-weighted representation learning，让 embedding `z` 同时保留多类行为结构：

1. `longitudinal_comfort`
2. `following_interaction`
3. `lateral_lane_dynamics`
4. `lateral_gap_interaction`
5. `behavior_proxy`

每组都包含两类约束：

- auxiliary regression loss
- group metric alignment loss（embedding pairwise distance 对齐该组 feature pairwise distance）

简化公式：

`Total loss = style loss + weighted auxiliary losses + weighted group metric alignment losses`

权重决定 embedding 几何重点：

- following 权重过低：THW/front distance/rel speed 欠表达；
- following 权重过高：following 主导、lateral 结构被冲淡；
- lateral 权重过低：yaw/heading/lane-change 结构减弱；
- lateral 权重过高：可能恢复 lateral，但牺牲 following interaction。

目标不是单项极值，而是多交互风格的**平衡表示**。

研究逻辑：

- Stage 5D-v1：ablation，证明上调 following 权重可以修复 following 弱项；
- Stage 5D-balanced-v2：回调 following、上调 lateral，保持 following 强化同时恢复 lateral。

## Stage 5B Baseline Result

关键结果：

- hit@5 = `0.490300`
- longitudinal_comfort = `0.150833`
- following_interaction = `0.302917`
- lateral_lane_dynamics = `0.266777`
- behavior_proxy = `0.190567`

解释：

- Stage 5B learned embedding 有效（优于 random/context_l2）；
- strongest signal 在 lateral/lane-change dynamics；
- following/front-distance interaction 偏弱；
- 这直接驱动 Stage 5D。

## Stage 5D-v1: Group-weighted Following Enhancement

训练思想：

- 上调 following_interaction 相关损失权重；
- lateral 权重保持中等；
- 目标：修复 Stage 5B following 弱项。

关键结果：

- hit@5 = `0.507992`
- longitudinal_comfort = `0.151584`
- following_interaction = `0.582954`
- lateral_lane_dynamics = `0.204637`
- behavior_proxy = `0.355707`

解释：

- Stage 5D-v1 成功强化 following_interaction，且超过 raw/pca；
- behavior_proxy 明显提升；
- 但存在 following 过校正；
- lateral_lane_dynamics 从 Stage 5B 的 `0.266777` 下降到 `0.204637`；
- 因此 v1 是关键 ablation，但不是最终推荐模型。

结论（固定表述）：

Stage 5D-v1 proves that group-weighted following losses are effective, but also reveals a trade-off: over-emphasizing following interaction can partially erase lateral dynamic structure.

## Stage 5D-balanced-v2: Current Recommended Model

训练思想：

- 相比 v1，下调 following 权重；
- 相比 v1，上调 lateral dynamics 权重；
- 保持 following 显著高于 Stage 5B；
- 目标：following 与 lateral 两类结构平衡。

关键结果：

- hit@1 = `0.213092`
- hit@5 = `0.526232`
- mean_same_label_fraction_at_5 = `0.189776`
- longitudinal_comfort = `0.171751`
- following_interaction = `0.501998`
- lateral_lane_dynamics = `0.245608`
- behavior_proxy = `0.322344`

相对 raw/pca：

- Global retrieval 仍低于 raw_feature / pca_feature；
- following_interaction 超过 raw_feature 与 pca_feature；
- lateral_lane_dynamics 与 raw/pca 接近；
- behavior_proxy 超过 raw_feature 与 pca_feature；
- longitudinal_comfort 与 raw/pca 接近。

重要 feature-level learned wins：

- `mean_thw_delta`、`min_thw_delta`
- `mean_front_distance_delta`、`min_front_distance_delta`
- `mean_rel_speed_delta`、`p95_rel_speed_delta`
- `front_pressure_score_delta`
- `lane_change_count_proxy_delta`、`lane_change_rate_proxy_delta`、`lane_change_oscillation_score_proxy_delta`
- `max_lateral_speed_delta`、`rms_yaw_rate_delta`、`heading_change_total_delta`
- `yielding_score_proxy_delta`

平衡结论：

不应声称 learned embedding 在 global retrieval 上全面超过 handcrafted baselines；应表述为：Stage 5D-balanced-v2 是当前最优 learned trade-off 表示，在多个行为类别上实现超过或接近 feature baselines。

## Final Comparison Table

```bash
| Model | hit@5 | longitudinal | following | lateral | behavior_proxy | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Stage 5B baseline | 0.490300 | 0.150833 | 0.302917 | 0.266777 | 0.190567 | strong lateral, weak following |
| Stage 5D-v1 | 0.507992 | 0.151584 | 0.582954 | 0.204637 | 0.355707 | following over-correction |
| Stage 5D-balanced-v2 | 0.526232 | 0.171751 | 0.501998 | 0.245608 | 0.322344 | best current trade-off |
```

进展逻辑不是随机调参，而是受控的 multi-objective trade-off 研究：

- Stage 5B 先诊断弱项；
- Stage 5D-v1 证明 following 可被显著强化；
- Stage 5D-balanced-v2 恢复平衡并成为当前推荐模型。

## Stage 5D-balanced-v2 Commands

训练命令（Stage 5D 不使用 `--metric_alignment`）：

```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_temperature 1.0 \
  --metric_loss_type huber \
  --style_loss_weight 1.0 \
  --aux_longitudinal_weight 0.5 \
  --aux_following_weight 1.2 \
  --aux_lateral_dynamics_weight 1.5 \
  --aux_lateral_gap_weight 1.0 \
  --aux_behavior_proxy_weight 0.5 \
  --metric_longitudinal_weight 0.5 \
  --metric_following_weight 1.5 \
  --metric_lateral_dynamics_weight 1.5 \
  --metric_lateral_gap_weight 1.0 \
  --metric_behavior_proxy_weight 0.5 \
  --device cuda \
  --seed 42 \
  --overwrite
```

导出命令：

```bash
python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings \
  --split all
```

评估命令：

```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

## Next Immediate Actions

1. Treat Stage 5D-balanced-v2 as the current recommended Stage 5 model.
2. Fix evaluator report to generate dynamic conclusions.
3. Update result tables in this design document whenever a new model is trained.
4. Do not run many more weight sweeps immediately unless needed.
5. Next research step: final comparison + paper framing
   - Stage 5B vs Stage 5D-v1 vs Stage 5D-balanced-v2
   - learned embedding vs raw_feature / pca_feature / context_l2 / random
   - public-human trajectory validation narrative
   - relation to earlier lateral_stable findings

## Stage 5E Final Comparison

### 为什么新增 Stage 5E

Stage 5B / Stage 5D-v1 / Stage 5D-balanced-v2 已经完成训练与评估。新增 Stage 5E 的目的，是在**不改训练逻辑、不重建数据集**前提下，统一读取三组既有评估产物并给出最终对比结论，形成可复现实验报告与当前推荐模型。

### 比较脚本路径

- `tools/compare_stage5_embedding_runs.py`

### 运行命令

```bash
python tools/compare_stage5_embedding_runs.py \
  --stage5b_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_schema_fixed \
  --stage5d_v1_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval \
  --stage5d_v2_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison \
  --overwrite
```

### 期望输出目录

- `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison`

### 期望输出文件

- `final_stage5_model_comparison.csv`
- `final_stage5_category_comparison.csv`
- `final_stage5_retrieval_comparison.csv`
- `final_stage5_learned_win_summary.csv`
- `final_stage5_recommendation.md`
- `final_stage5_comparison_plot.png`

### Stage 5E 兼容性说明（learned-win）

- `tools/compare_stage5_embedding_runs.py` 会优先读取 `learned_win_features.csv`。
- 若 `learned_minus_raw_feature` / `learned_minus_pca_feature` 缺失，脚本会在比较阶段内部自动计算：
  - `learned_minus_raw_feature = learned_corr - raw_corr`
  - `learned_minus_pca_feature = learned_corr - pca_corr`
- 若 `learned_win_features.csv` 不可用或字段不足，脚本会回退到 `style_distance_correlation.csv`，按 `target_feature` + `representation` 透视后再计算两类 delta。
- 因此 Stage 5E 比较与旧版 evaluator 产物兼容，不要求 evaluator 预先写出 delta 列。
- 注意：paper-grade 校验不放宽，仍要求 `paper_grade_valid=true`、`strict_feature_schema=true`、`feature_schema_loaded=true`、`row_alignment_checks.aligned=true`。

### Final comparison（当前记录）

| model | hit@5 | following_interaction | lateral_lane_dynamics | behavior_proxy | interpretation |
|---|---:|---:|---:|---:|---|
| Stage 5B baseline | 0.490300 | 0.302917 | 0.266777 | - | meaningful baseline; strong lateral; weak following |
| Stage 5D-v1 group_weighted | 0.507992 | 0.582954 | 0.204637 | - | following strongly improved but lateral over-corrected downward |
| Stage 5D-balanced-v2 | 0.526232 | 0.501998 | 0.245608 | 0.322344 | best current trade-off and current recommended Stage 5 model |

### 当前推荐

- **Stage 5D-balanced-v2**。

### 研究解释

Stage 5D 系列属于受控的多目标权衡（multi-objective trade-off）实验：
- v1 强化 following_interaction 后出现 lateral_lane_dynamics 下探；
- balanced-v2 在保持较强 following 的同时恢复 lateral，并在 behavior_proxy 上超过特征基线；
- 该路线是有约束的目标权重调节，不是随机调参。

## Stage 5F：论文级实验固化与章节材料整理

### 5F.1 Final Experiment Summary

Stage 5F 不是新的训练阶段，也不是新的数据构建阶段。其定位是将 Stage 5A~5E 的结果固化为可复核、可复现、可直接写入论文实验章节的材料。

Stage 5A~5E 主链路总结：

- Stage 5A：lane-aware 5-neighbor context dataset
- Stage 5B：Flatten Context GRU baseline
- Stage 5C：strict-schema evaluation（paper-grade validity）
- Stage 5D：group-weighted multi-objective training
- Stage 5E：final comparison

当前 Stage 5 推荐模型：**Stage 5D-balanced-v2**。

### 5F.2 Paper-ready Tables

**Table 1：Dataset statistics（Stage 5A）**

- `n_windows_kept = 164871`
- `n_shards = 35`
- `train = 131998`
- `val = 16481`
- `test = 16392`
- `context_dim = 83`
- `feature_dim = 33`
- `fallback_assignment_rate = 0`
- `nonfinite_output_detected = 0`
- `good_lane_context_rate ≈ 0.99`

**Table 2：Behavior feature groups（33-D strict schema）**

- `longitudinal_comfort`
- `following_interaction`
- `lateral_lane_dynamics`
- `behavior_proxy`

**Table 3：Final model comparison（Stage 5E）**

| Model | hit@5 | longitudinal | following | lateral | behavior_proxy | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Stage 5B baseline | 0.490300 | 0.150833 | 0.302917 | 0.266777 | 0.190567 | strong lateral, weak following |
| Stage 5D-v1 | 0.507992 | 0.151584 | 0.582954 | 0.204637 | 0.355707 | following over-correction |
| Stage 5D-balanced-v2 | 0.526232 | 0.171751 | 0.501998 | 0.245608 | 0.322344 | best current trade-off |

**Table 4：Learned-win summary**

- Stage 5B beats both raw/pca on 8 features
- Stage 5D-v1 beats both raw/pca on 10 features
- Stage 5D-balanced-v2 beats both raw/pca on 17 features

### 5F.3 Paper-ready Figures List

- **Figure 1：Stage 5 interaction-aware embedding pipeline**  
  Purpose：展示 Waymo trajectory → lane-aware 5-neighbor context → encoder → embedding → evaluation。  
  Source：后续手工绘制或生成。
- **Figure 2：Lane-aware 5-neighbor slot definition**  
  Purpose：展示 `front / left_front / left_rear / right_front / right_rear` 五个槽位定义。  
  Source：示意图。
- **Figure 3：Group-weighted multi-objective training design**  
  Purpose：展示 style loss + auxiliary heads + group metric alignment 的训练结构。  
  Source：方法图。
- **Figure 4：Stage 5 final comparison bar chart**  
  Purpose：比较 Stage 5B / 5D-v1 / 5D-balanced-v2 的 `hit@5`、following、lateral、behavior_proxy。  
  Source file：`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison/final_stage5_comparison_plot.png`。
- **Figure 5：Category-wise learned vs raw/pca comparison**  
  Purpose：展示 learned embedding 在各类别上相对 raw/pca 的胜负与近似持平区域。  
  Source：`final_stage5_category_comparison.csv`（可后续绘图）。
- **Figure 6：Learned-win feature summary**  
  Purpose：展示 learned 同时超过 raw/pca 的目标特征数量统计。  
  Source：`final_stage5_learned_win_summary.csv`。
- **Figure 7：Future BDD / E2E style report card**  
  Purpose：连接 Stage 5 embedding 与 Stage 6 E2E 模型风格对比。  
  Source：未来示意图。

### 5F.4 Method Description Draft

我们将每个驾驶片段表示为长度为 80 帧的 ego-centric interaction context sequence。对于每个目标车辆窗口，首先基于 lane-aware assignment 选择 `front`、`left_front`、`left_rear`、`right_front`、`right_rear` 五个邻车槽位，并将其相对运动编码为 83 维时序输入。编码器采用 Flatten Context GRU，将窗口映射到 64 维 context-aware behavior embedding。

训练监督采用严格 33 维 interaction feature schema 作为弱监督与评估目标，并通过 group-weighted multi-objective loss 共同塑造 embedding 空间。具体而言，总损失由 style loss、辅助回归头（auxiliary regression heads）以及 group-wise metric alignment 组成，用于同时保持全局行为结构与类别级可解释性。为保证实验可复现性，Stage 5C 之后的评估全程启用 strict schema，通过名称解析特征索引并禁止 silent feature-index mismatch，从而确保不同模型和不同运行批次之间的指标具有可比性。

### 5F.5 Evaluation Protocol

评估设置（固定）：

- `eval split = test`
- `actual_eval_samples = 16392`

表示对比集合：

- `learned_context_embedding`
- `raw_feature`
- `pca_feature`
- `context_l2`
- `random`

核心指标：

- `hit@1`
- `hit@5`
- `mean_same_label_fraction_at_5`
- style-distance Spearman correlation
- category-wise correlation
- learned-win feature count
- context sensitivity

strict schema 校验要求：

- `feature_schema_loaded = true`
- `strict_feature_schema = true`
- no fallback index
- `paper_grade_valid = true`
- `row_alignment_checks.aligned = true`

### 5F.6 Results and Ablation Interpretation Draft

Stage 5B baseline 是有意义的 context-aware 基线：它已经学到可用行为结构，但呈现“lateral 强、following 弱”的不均衡格局。

Stage 5D-v1 在 following_interaction 上从 `0.302917` 显著提升到 `0.582954`，同时 lateral 从 `0.266777` 下降到 `0.204637`。这说明 following 权重强化是有效的，但会引入 lateral 过校正。

Stage 5D-balanced-v2 将 hit@5 提升到 `0.526232`，following 保持在 `0.501998`，并把 lateral 恢复到 `0.245608`，同时 `behavior_proxy = 0.322344`。整体上它给出了当前最优的多目标折中。

因此，Stage 5D 不是随机调参，而是由 Stage 5C 诊断驱动的受控 multi-objective trade-off 研究：先验证 following 可强化，再通过权重重平衡恢复 lateral，并最终得到论文可报告的稳定折中点。

### 5F.7 Current Recommended Model Statement

Current recommended Stage 5 model：**Stage 5D-balanced-v2**。

推荐理由：

- 在 learned 表示内部取得当前最佳全局 retrieval（hit@5 最优）
- `following_interaction` 超过 raw/pca 特征基线
- `behavior_proxy` 超过 raw/pca 特征基线
- `longitudinal_comfort` 与特征基线接近（near-tie）
- `lateral_lane_dynamics` 与特征基线接近（near-tie）
- 在 17 个目标特征上同时超过 raw/pca

平衡结论：当前不宣称 learned embedding 已经全局超越 raw/pca 特征检索；更准确的结论是 Stage 5D-balanced-v2 是目前最好的 learned representation，并且是进入 BDD / E2E style comparison 的可靠基础。

### 5F.8 Limitations and Next Steps

当前限制：

- learned embedding 仍未在全局检索上全面超过 raw/pca baselines
- Stage 5 使用的是 Waymo public human trajectories，而非私有 E2E 实车日志
- pseudo labels / features 属于弱监督，不是 ground truth
- 当前是 trajectory-level behavior evaluation，不是 closed-loop simulation
- Slot Encoder + Attention Pooling 尚未完成正式评估
- Stage 6 真实 E2E style comparison 仍是后续工作

下一步：

- Stage 5G：可选 Slot Encoder + Attention Pooling 消融
- Stage 6：E2E model style comparison / BDD report card

### 5F.9 Reproducibility Commands

Stage 5D-balanced-v2 训练命令（示意，保持与前文一致配置）：

```bash
python tools/train_context_gru.py   --dataset_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged   --run_name context_gru_stage5d_balanced_v2   --embedding_dim 64   --group_metric_weight_following 1.20   --group_metric_weight_lateral 1.00   --group_metric_weight_longitudinal 1.00   --group_metric_weight_behavior_proxy 1.10
```

导出命令：

```bash
python tools/export_context_embeddings.py   --dataset_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged   --run_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2
```

评估命令：

```bash
python tools/evaluate_context_embeddings.py   --dataset_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged   --eval_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval_final   --strict_feature_schema
```

Stage 5E 最终对比命令：

```bash
python tools/compare_stage5_embedding_runs.py   --stage5b_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_final   --stage5d_v1_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval_final   --stage5d_v2_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval_final   --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison   --overwrite
```

说明：Stage 5D 路线不使用 `--metric_alignment`，而是通过 group-specific metric weights 实现多目标权衡。

### Synthetic Policy / BDD 框架回顾

Synthetic policy 属于 Stage 1-3 的受控验证框架，构建了多种已知风格变体：

- conservative
- aggressive
- lateral_stable
- comfort
- following_safe
- assertive
- yielding

其目的在于：通过已知行为差异，验证 embedding 与 style drift 指标是否能正确感知风格变化。

BDD（Behavioral Distribution Distance）定义为：比较 policy/model A 与 B 在 embedding 分布上的距离，即 `BDD(A, B)`。

可选实现包括：

- MMD
- Wasserstein
- Fréchet distance
- energy distance

关系总结：Stage 5 提供了更强的 interaction-aware encoder；Stage 6 将在此基础上对真实 E2E 模型版本计算 BDD，并形成风格漂移与行为差异报告。
