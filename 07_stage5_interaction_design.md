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

## Stage 5D: Group-weighted Training Improvement

动机：

Stage 5C-2 表明 Stage 5B embedding 在 lateral dynamics 上较强，但在 following_interaction 上偏弱。Stage 5D 的目标是保留 lateral 优势，同时增强 following/front-distance interaction。

训练脚本：

`tools/train_context_behavior_embedding.py`

与 Stage 5B 的关键差异：

- Stage 5D **不再使用** `--metric_alignment`。
- Stage 5D 使用 group-specific metric loss weights：
  - `--metric_longitudinal_weight`
  - `--metric_following_weight`
  - `--metric_lateral_dynamics_weight`
  - `--metric_lateral_gap_weight`
  - `--metric_behavior_proxy_weight`

Stage 5D 正确训练命令：

```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1 \
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
  --aux_following_weight 1.5 \
  --aux_lateral_dynamics_weight 1.0 \
  --aux_lateral_gap_weight 1.0 \
  --aux_behavior_proxy_weight 0.5 \
  --metric_longitudinal_weight 0.5 \
  --metric_following_weight 2.0 \
  --metric_lateral_dynamics_weight 1.0 \
  --metric_lateral_gap_weight 1.0 \
  --metric_behavior_proxy_weight 0.5 \
  --device cuda \
  --seed 42 \
  --overwrite
```

补充说明：

- `batch_size=64` 用于更稳健的 GPU 显存控制。
- 若显存充足，可后续试验 `batch_size=128` 或 `256`。
- 训练支持 tqdm 进度条。
- 日志优先模式可用 `--no_progress` 关闭 tqdm。

Stage 5D 预期输出目录：

`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1`

预期训练产物：

- `best_model.pt`
- `training_config.json`
- `feature_group_config.json`
- `train_log.csv`
- `training_summary.json`

Stage 5D 目标判据（相对 Stage 5B）：

- `following_interaction` mean correlation 从 `0.302917` 朝 `0.38+` 提升；
- `lateral_lane_dynamics` 保持在 `0.25` 附近或以上；
- `hit@5` 从 `0.4903` 朝 `0.52+` 提升；
- learned embedding 仍需稳定优于 `context_l2` 与 `random`。

## Current Stage 5 Status

截至本次更新：

1. Stage 5A full51 lane-aware 5-neighbor context 数据集已完成。
2. Stage 5B context GRU baseline 已训练并评估。
3. Stage 5C evaluator 已切换为 strict-schema 且具备 paper-grade 有效性。
4. Stage 5C-2 得到关键科学结论：
   - learned embedding 在 lateral/lane-change dynamics 上强于 feature baselines；
   - following/front-distance interaction 仍偏弱。
5. Stage 5D 是当前进行中的 active stage：
   - 加入 group-weighted auxiliary + metric losses；
   - 提升 following_interaction 权重；
   - 保持 lateral dynamics 优势。

## Paper-level Interpretation

Stage 5 提供了首条超越 synthetic policy rollout 的、基于真实公共人类轨迹数据的验证路径。当前结果并非“全面胜利”：`learned_context_embedding` 在全局上尚未超过 handcrafted raw/pca feature baselines。但已有一个关键正结果：序列式 embedding 在 lateral 与 lane-change 的时序动态表达上优于静态特征空间距离，这与 earlier lateral_stable 结论（yaw-rate / lateral stability 对可分性关键）一致。

平衡结论应明确：

- Stage 5B embedding 有意义，但并不完整；
- 它在 lateral dynamics 上较强；
- 它在 following/front-distance interaction 上较弱；
- 需要 Stage 5D 继续强化 interaction-awareness。

## Known Pitfalls / Lessons Learned

1. 不要依赖 fallback feature indices；feature schema 必须显式给定。
2. `mean_speed` 与 `std_rel_speed` 不属于 Stage 5 schema，必须用 `p95_rel_speed`。
3. `raw_feature` 与 `pca_feature` 是强 baseline；仅优于 random/context_l2 还不够。
4. global retrieval 会掩盖类别差异，必须做 category-wise 评估。
5. Stage 5D 训练命令必须 **不包含** `--metric_alignment`，改用 group-specific metric weights。
6. 文档必须与代码变更同步更新，且给出可运行命令与期望输出。

## Next Immediate Actions

1. 在 `tools/train_context_behavior_embedding.py` 中确认（或修复）训练进度条支持。
2. 用修正后的 Stage 5D 命令启动训练。
3. 导出 Stage 5D embeddings（已有命令可复用）：

```bash
python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1/best_model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_embeddings \
  --batch_size 256 \
  --device cuda \
  --split all \
  --overwrite
```

4. 对 Stage 5D embeddings 重新运行 Stage 5C evaluator。
5. 对比 Stage 5D 与 Stage 5B：
   - global retrieval
   - category-wise correlation
   - learned-win features
   - following_interaction
   - lateral_lane_dynamics
