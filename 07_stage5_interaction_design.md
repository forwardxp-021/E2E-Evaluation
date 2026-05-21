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

### Stage 5F 的目标

Stage 5F 不是新的训练阶段，也不是新的数据构建阶段。Stage 5F 的定位是**论文级实验固化**：将 Stage 5A~5E 的已完成工作，整理为可追溯、可复现、可直接进入论文实验章节的完整叙事。

Stage 5F 重点固化的内容包括：

- 数据集构建摘要（Stage 5A）
- 特征 schema 摘要（33-D 严格 schema）
- 模型设计摘要（Stage 5B/5D）
- 训练目标演化（从 baseline 到 group-weighted）
- 评估协议（Stage 5C 严格评估）
- 最终对比（Stage 5E）
- 当前推荐模型
- 限制与边界
- 后续阶段方向

关键说明：Stage 5F 的输出**不是新模型**，而是把 Stage 5 的实验链路在 `07_stage5_interaction_design.md` 中固化为干净、可核查的论文级材料。

### Stage 5A-E 总链路

Stage 5 全链路如下：

- Stage 5A：lane-aware 5-neighbor context dataset
- Stage 5B：Flatten Context GRU baseline
- Stage 5C：strict-schema evaluation
- Stage 5D：group-weighted multi-objective training
- Stage 5E：final comparison
- Stage 5F：paper-level consolidation

这意味着 Stage 5 已经不再只是“数据 + 训练”实验，而是形成了完整的 interaction-aware behavior embedding 验证管线：

Waymo public human trajectory  
↓  
lane-aware 5-neighbor context dataset  
↓  
Flatten Context GRU / group-weighted encoder  
↓  
64-D behavior embedding  
↓  
strict-schema evaluation  
↓  
final model comparison  
↓  
paper-ready behavior representation conclusion

### Stage 5A 数据集摘要

数据集目录：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged`

关键统计：

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

论文级解释：

- 数据采用 lane-aware assignment，而非纯几何邻车选择。
- 固定五个邻车槽位：`front`、`left_front`、`left_rear`、`right_front`、`right_rear`。
- 数据以 shard 形式组织，保证大规模样本可处理。
- 行对齐（row alignment）是后续 embedding 导出与评估的前提。
- 不应把数据粗暴合并为单个超大 `.npy`。
- lane-aware slot 的稀疏性是预期行为，不是 bug。

### 33-D interaction feature schema and behavior groups

Stage 5 评估采用严格 33-D schema，并按行为组进行分析：

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

重要约束：

- `mean_speed` 不在 Stage 5 schema 中。
- `std_rel_speed` 不在 Stage 5 schema 中。
- Stage 5 使用 `p95_rel_speed` 替代 `std_rel_speed`。
- 严格 schema 防止 silent index mismatch。
- 特征索引必须由 schema 名称解析，不能依赖硬编码下标。

### 模型版本与训练目标演化

Stage 5B baseline：

- 模型：Flatten Context GRU
- 输入：`context_traj [N, 80, 83]`
- `embedding_dim = 64`
- 训练目标：
  - soft contrastive loss
  - global metric alignment loss
- 结果：
  - meaningful baseline
  - lateral 强
  - following 弱

Stage 5D-v1：

- 模型：group-weighted multi-objective loss
- 目的：上调 following_interaction
- 结果：following 大幅提升，但 lateral 下探（over-correction）
- 解释：证明 following 可被显式强化，但不是最终模型

Stage 5D-balanced-v2：

- 相比 v1，下调 following 权重
- 相比 v1，上调 lateral dynamics 权重
- 目标：在保持 following 的同时恢复 lateral 结构
- 当前推荐模型

训练调整原则：

Stage 5D 不是随机调参，而是由 Stage 5C 诊断驱动的受控多目标权衡研究：

- Stage 5C 发现 Stage 5B 的 following_interaction 偏弱、lateral_lane_dynamics 偏强；
- Stage 5D-v1 回答“following 能否通过 group weights 强化”；
- Stage 5D-balanced-v2 回答“following 强化后 lateral 能否恢复”；
- 最终形成可解释的 trade-off 路线，而非盲目试参。

### Group-weighted multi-objective loss

总损失形式：

`Total loss = style loss + weighted auxiliary regression losses + weighted group metric alignment losses`

直观解释：

- style loss：保持整体行为几何结构。
- auxiliary heads：约束 embedding 保留各行为组可回归信息。
- group metric alignment：让 embedding 距离与各行为组 feature-space 距离一致。

权重机制解释：

- following 权重过低：THW / front distance / rel speed 会被弱表达。
- following 权重过高：embedding 会被 following 主导并削弱 lateral。
- lateral 权重过低：yaw / heading / lane-change 结构变弱。
- lateral 权重过高：可能恢复 lateral 但损害 following。
- balanced-v2 被选中，是因为其在 following 与 lateral 之间取得更优平衡，同时提升全局 learned retrieval。

### 评估协议

对比表示：

- `learned_context_embedding`
- `raw_feature`
- `pca_feature`
- `context_l2`
- `random`

核心指标：

- retrieval `hit@1`
- retrieval `hit@5`
- `mean_same_label_fraction_at_5`
- style-distance Spearman correlation
- category-wise correlation
- learned-win feature count
- context sensitivity

严格 schema 有效性要求：

- `feature_schema_loaded = true`
- `strict_feature_schema = true`
- no fallback feature index
- `paper_grade_valid = true`
- `row_alignment_checks.aligned = true`

阶段解释：

- Stage 5C-v1 属于 preliminary（使用 fallback 索引）。
- Stage 5C-1 修复 strict schema 并达到 paper-grade validity。
- Stage 5C-2 增加 category-wise 分析。
- Stage 5E 完成跨模型最终比较。

### Stage 5E 最终对比结果

| Model | hit@5 | longitudinal | following | lateral | behavior_proxy | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Stage 5B baseline | 0.490300 | 0.150833 | 0.302917 | 0.266777 | 0.190567 | strong lateral, weak following |
| Stage 5D-v1 | 0.507992 | 0.151584 | 0.582954 | 0.204637 | 0.355707 | following over-correction |
| Stage 5D-balanced-v2 | 0.526232 | 0.171751 | 0.501998 | 0.245608 | 0.322344 | best current trade-off |

解释：

- Stage 5B 建立了有意义的 context-aware baseline，但 following 明显偏弱。
- Stage 5D-v1 证明了 following 可以被显著强化，但对 lateral 产生过校正。
- Stage 5D-balanced-v2 在全局 learned retrieval、following 保持、lateral 恢复与 behavior_proxy 上取得最好折中，是当前推荐模型。

### learned-win feature analysis

- Stage 5B：同时超过 raw/pca 的特征数为 8，主要集中在 lateral dynamics。
- Stage 5D-v1：同时超过 raw/pca 的特征数为 10，主要集中在 following/front-distance/yielding。
- Stage 5D-balanced-v2：同时超过 raw/pca 的特征数为 17，覆盖 following、lateral dynamics、comfort、yielding 多维度。

这说明 balanced-v2 并非只提升单一类别，而是把 learned representation 的优势扩展到更多行为维度。

balanced-v2 关键 learned wins（示例）：

- `mean_thw`
- `min_thw`
- `mean_front_distance`
- `min_front_distance`
- `mean_rel_speed`
- `p95_rel_speed`
- `front_pressure_score`
- `rear_vehicle_pressure_proxy`
- `rms_yaw_rate`
- `heading_change_total`
- `lane_change_count_proxy`
- `lane_change_rate_proxy`
- `max_lateral_speed`
- `rms_lateral_accel`
- `lane_change_oscillation_score_proxy`
- `yielding_score_proxy`

### Stage 5 论文级结论

我们已经构建了 lane-aware 的 5-neighbor interaction-aware 轨迹数据集，并在此基础上训练 context-aware behavior embedding。严格 schema 评估表明 learned embedding 具备有效行为结构信息。

从阶段性诊断看，Stage 5B 揭示了结构不均衡：lateral 强、following 弱；Stage 5D-v1 证明 following 可被增强，但会带来 lateral 退化；Stage 5D-balanced-v2 在两者之间取得当前最佳 trade-off，因此被选为当前推荐模型。

同时必须保持审慎：当前 learned embedding 仍未在 global retrieval 上全面超过 raw_feature / pca_feature，因此不能宣称“完全战胜手工特征”。但 balanced-v2 已提供了坚实的 learned representation 基础，可支撑后续 BDD / E2E 风格对比研究。

### 关键命令与可复现路径

关键评估目录：

- Stage 5B eval final：
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_final`
- Stage 5D-v1 eval final：
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval_final`
- Stage 5D-balanced-v2 eval final：
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval_final`
- Stage 5E final comparison：
  - `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison`

Stage 5E 对比命令：

```bash
python tools/compare_stage5_embedding_runs.py \
  --stage5b_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_final \
  --stage5d_v1_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval_final \
  --stage5d_v2_eval outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_eval_final \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/stage5_final_comparison \
  --overwrite
```

Stage 5D-balanced-v2 训练/导出/评估命令见前文“Stage 5D-balanced-v2 Commands”章节，可直接复现。注意：Stage 5D 路线不使用 `--metric_alignment`，而是使用 group-specific metric 权重。

### 当前限制

- learned embedding 仍未在 global retrieval 上全面超过 raw/pca baselines。
- Stage 5 使用的是 Waymo public human trajectory，不是私有 E2E 实车日志。
- pseudo style labels 与 interaction features 属于弱监督，不是 ground truth。
- 当前属于 trajectory-level behavior evaluation，不是闭环仿真验证。
- Slot Encoder + Attention Pooling 尚未完成正式对比评估。
- Stage 6 的真实 E2E style comparison 仍是后续工作。

### 后续阶段规划

Stage 5G（可选架构消融）：Slot Encoder + Attention Pooling

- 当前主线是 Flatten Context GRU。
- slot encoder 可能提升交互可解释性。
- 可分别编码 `ego/front/left_front/left_rear/right_front/right_rear`。
- 通过 attention pooling 聚合上下文表示。
- 该方向属于后续研究，不阻塞当前 Stage 5F 固化。

Stage 6（E2E model style comparison / BDD report card）：

- 使用 Stage 5D-balanced-v2 embedding 对比两版 E2E 模型轨迹数据。
- 计算 BDD（Behavioral Distribution Distance）。
- 输出驾驶风格报告卡（report card）。
- 输出类别级差异（category deltas）。
- 输出场景切片差异（scenario-sliced deltas）。
- 输出 top drift cases。
- 服务管理层、学术审稿与工程同事的共同决策需求。

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
