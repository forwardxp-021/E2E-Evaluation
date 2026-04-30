# E2E-Evaluation (20260413_features_new_design)

基于 Waymo 轨迹数据学习驾驶风格 embedding 的实验工程。

## 项目目标

从自车与前车的对齐轨迹中构建 style feature，用 feature-guided soft contrastive 训练轨迹编码器，最后通过 UMAP、线性探针和邻域一致性验证 embedding 是否编码了行为风格信息。

## 近期代码更改总结 (2026-04)

### 1) 数据构建重构为滑窗 + 新 style 特征
- `build_dataset.py` 新增滑窗参数：`--window_len`、`--stride`。
- 每个 scenario 按窗口输出样本，不再只输出整段场景级样本。
- 新增 20 维 style 特征计算函数 `compute_style_features(...)`，并写出：
  - `output/feat_style_raw.npy`：原始特征（可能含 NaN）
  - `output/feat_style.npy`：NaN 填 0 后全局标准化特征（训练默认建议使用）
  - `output/feature_names_style.json`：特征名映射
- 保留兼容开关 `--save_legacy_features`，需要时可额外输出旧特征到 `feat_legacy.npy` 和兼容文件 `feat.npy`。

### 2) TFRecord 解析健壮性增强
- `build_dataset.py` 增加 `parse_scenario_from_record(...)`：
  - 支持“直接 Scenario proto”
  - 支持“tf.train.Example 中 bytes 特征包裹 Scenario”
- 增加 `_scenario_looks_valid(...)` 做解析后有效性检查，降低脏样本导致的崩溃风险。

### 3) 训练数据读取兼容修复（已解决真实报错）
- `dataset.py` 中 `TrajFeatureDataset` 对 `traj.npy` 进行统一 `float32` 转换。
- 修复问题：`numpy.object_` 轨迹样本在 `collate_variable_traj` 中无法转 tensor，报错
  - `TypeError: can't convert np.ndarray of type numpy.object_`
- 修复后已通过训练冒烟验证。

### 4) 导出脚本修复与增强（已解决真实报错）
- `export_embeddings.py` 删除了误粘贴的重复代码块，修复
  - `SyntaxError: 'return' outside function`
- `TrajOnlyDataset` 同步增加轨迹 `float32` 转换，避免导出阶段再次触发 object dtype 问题。
- 保留 `--checkpoint_path` 作为 `--checkpoint` 的兼容别名，并支持 `--split_path` 行数一致性校验。

### 5) 评估流程对齐新特征
- 推荐在评估中显式指定：
  - `--feat_path output/feat_style.npy`
  - `--feature_names_path output/feature_names_style.json`
- 已完成一轮 `train -> export -> evaluate` 全链路跑通。

## 新 style 特征 (20D)

顺序与 `feature_names_style.json` 一致：

1. `acc_abs_p95`
2. `acc_abs_p99`
3. `acc_rms`
4. `jerk_abs_p95`
5. `jerk_abs_p99`
6. `jerk_rms`
7. `yaw_rate_rms`
8. `yaw_rate_abs_p95`
9. `heading_change_total`
10. `speed_control_oscillation`
11. `cf_valid_frac`
12. `thw_p50`
13. `thw_p20`
14. `thw_iqr`
15. `v_rel_p50`
16. `closing_gain_kv`
17. `gap_gain_kd`
18. `desired_gap_d0`
19. `acc_sync_lag`
20. `acc_sync_corr`

## 快速使用

### 1) 构建数据

```bash
python build_dataset.py \
  --tfrecord_glob "/mnt/d/WMdata/*.tfrecord-*" \
  --output_dir output \
  --min_ego_speed 5.5 \
  --window_len 80 \
  --stride 20 \
  --min_points_cf 20 \
  --kd_min 1e-3 \
  --d0_min_gap 1.0 \
  --d0_max_gap 200.0 \
  --d0_log1p \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### desired_gap_d0 清洗说明（病态长尾防护）

- `desired_gap_d0` 来自拟合关系 `a_e ≈ kv*v_rel + kd*d + b` 的零加速度间距 `d0=-b/kd`。当 `kd` 过小或拟合病态时，`d0` 会数值爆炸并污染 feature 距离，导致 soft target 过平。
- 数据构建中已增加两层防护：
  - 拟合阶段：`abs(kd) < kd_min` 时不计算 `d0`（置为 NaN）。
  - 特征阶段：对窗口级 `d0` 做 sanitize（`<=d0_min_gap` 置 NaN、clip 到 `[d0_min_gap,d0_max_gap]`、可选 `log1p` 压缩长尾）。
- 相关参数：
  - `--kd_min`（默认 `1e-3`）
  - `--d0_min_gap`（默认 `1.0`）
  - `--d0_max_gap`（默认 `200.0`）
  - `--d0_log1p / --no-d0_log1p`（默认开启）

## 为什么需要工况门控（条件感知训练）

### 问题
直接在全体样本间做特征距离对比存在一个根本缺陷：**不同工况（速度/跟车距离/相对速度/跟车覆盖率）下的驾驶行为是不可比的**。例如：
- 高速行驶时的急刹与低速跟车时的轻刹，加速度指标天然不同，但不能据此判断它们风格相似或不同。
- 非跟车段（cf_valid_frac ≈ 0）没有 thw、kv、kd 等跟车特征，与跟车段的特征距离毫无意义（cf 维度都填 0，错误地认为"完全一样"）。

这导致评估中 `neighbor_consistency ratio_mean > 1`（embedding 邻居的特征差异反而比随机邻居更大），以及线性探针 Spearman 偏弱。

### 解决方案：工况门控（kNN 模式）
- **工况向量**：从 `traj.npy` 和 `front.npy` 计算每个样本的工况特征 `[speed_mean, dist_mean, vrel_mean, cf_valid_frac]`
- **kNN 门控**（`--cond_mode knn`，推荐）：对每个 anchor 选取工况距离最近的 `cond_k` 个样本（距离用鲁棒尺度 MAD/IQR/STD 归一化，无需手动调容差）。仅在候选数为 0 时触发最后兜底（回退全局）。
- **硬盒门控**（`--cond_mode hard_box`，保留向后兼容）：基于绝对容差盒子过滤，候选数不足时退回全局。

### 2) 训练（推荐：kNN 工况门控 + 混合 SupCon）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_cond_knn \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 0.2 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

> **说明**：`--feat_clip_value 3.0` 在特征归一化之后、距离计算之前，将标准化特征值裁剪到 [-3, 3]，可有效抑制 jerk/yaw 等长尾维度对距离计算的影响（推荐值 3.0；默认 0.0 表示不裁剪，与旧行为完全兼容）。

训练日志中新增诊断指标：
- `cond_cands`：每个 anchor 平均可用的工况兼容候选数（knn 模式下应接近 cond_k）
- `cond_fallback`：触发最后兜底（候选数=0）的 anchor 比例（knn 模式下应接近 0）
- `supcon`/`softkl`：hybrid 模式下两个分项的损失值

### 2b) 旧版硬盒门控训练（向后兼容）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_cond_hybrid \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode hard_box \
  --cond_speed_tol 2 --cond_dist_tol 5 --cond_vrel_tol 1 \
  --cond_cf_bucket_edges "0.2,0.6" --min_cond_candidates 8 \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 0.2 \
  --eval_every 10 --skip_val_clustering
```

### 2c) 不使用工况门控的基础训练（向后兼容）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_style_masked_ls_k1_a3 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --eval_every 10
```

### 3) 导出全量 embedding

```bash
python export_embeddings.py \
  --traj_path output/traj.npy \
  --split_path output/split.npy \
  --checkpoint output/run_cond_knn/best_model.pth \
  --output_path output/run_cond_knn/embeddings_all.npy
```

### 4) 评估（含工况感知邻域一致性）

```bash
python evaluate_embedding.py \
  --embeddings_path output/run_cond_knn/embeddings_all.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --eval_split test \
  --feature_names_path output/feature_names_style.json \
  --analysis_dir output/run_cond_knn/analysis_best \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --plot_first_k 20 --k_neighbors 10 \
  --umap_neighbors 30 --umap_min_dist 0.1 \
  --seed 42 --kmeans_clusters 8
```

工况感知评估额外生成 `neighbor_results_cond.csv`：
- `ratio_mean`：在工况兼容候选集内的随机基线比较（更公平）
- `mean_cond_candidates`：每个 anchor 在工况内的平均候选数
- `frac_fallback`：因候选数不足而退回全局随机基线的比例

## rel_kinematics 输入模式

### 动机

原始 `raw_xyv` 模式直接将 ego 轨迹的 `[x, y, vx, vy]` 送入 GRU，缺少显式的相对运动信息。对于 jerk、yaw_rate、thw 等风格特征，它们本质上是**差分**和**相对量**，如果在输入层就提供这些归纳偏置，模型更容易学习出可分离的风格维度。

`rel_kinematics` 模式从对齐的 ego/front 窗口计算 12 维逐帧特征后再送入 GRU，可改善 jerk/yaw/thw 等维度的邻域一致性。

### 12 维特征说明（dt = 0.1 s，Waymo 10 Hz）

| 索引 | 名称 | 公式 |
|------|------|------|
| 0 | `ego_v` | √(vx²+vy²) |
| 1 | `front_v` | √(front_vx²+front_vy²) |
| 2 | `v_rel` | ego_v − front_v |
| 3 | `dx` | front_x − ego_x |
| 4 | `dy` | front_y − ego_y |
| 5 | `dist` | √(dx²+dy²) |
| 6 | `closing_rate` | diff(dist) / dt（t=0 置 0） |
| 7 | `ego_a` | diff(ego_v) / dt（t=0 置 0） |
| 8 | `front_a` | diff(front_v) / dt（t=0 置 0） |
| 9 | `ego_heading` | atan2(vy, vx) |
| 10 | `ego_yaw_rate` | wrap(diff(ego_heading)) / dt（t=0 置 0） |
| 11 | `thw` | dist / max(ego_v, ε) |

填充帧被置零（用 `lengths` mask）。角度差通过 `wrap_angle` 归约到 `[-π, π]`。

### 示例训练命令（rel_kinematics）

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_relkin_knn \
  --input_mode rel_kinematics --dt 0.1 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 1.0 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

启动时会打印：
```
Input mode: rel_kinematics (12-dim) | dt=0.1s | front loaded: <N> windows
```

### 示例导出命令（rel_kinematics）

```bash
python export_embeddings.py \
  --traj_path output/traj.npy \
  --front_path output/front.npy \
  --split_path output/split.npy \
  --checkpoint output/run_relkin_knn/best_model.pth \
  --output_path output/run_relkin_knn/embeddings_all.npy \
  --input_mode rel_kinematics --dt 0.1
```

> **注意**：`--input_mode` 和 `--dt` 必须与训练时使用的值一致，否则模型结构不匹配会导致权重加载失败。

## 关键输出文件

- `output/traj.npy`: 自车滑窗轨迹
- `output/front.npy`: 前车滑窗轨迹
- `output/feat_style_raw.npy`: 原始 style 特征
- `output/feat_style.npy`: 训练用标准化 style 特征
- `output/feature_names_style.json`: 特征名
- `output/split.npy`: train/val/test
- `output/<run>/best_model.pth`: 最优模型
- `output/<run>/embeddings_all.npy`: 全量 embedding
- `output/<run>/analysis_*/neighbor_results.csv`: 全局邻域一致性
- `output/<run>/analysis_*/neighbor_results_cond.csv`: 工况内邻域一致性（条件感知评估）
- `output/<run>/analysis_*`: 评估结果图表与 CSV

---

## 合成策略 Rollout：在无真实 E2E 推理代码时验证 Embedding 区分能力

### 背景与目标

在没有多个 E2E 模型 rollout 数据时，可以利用已有的 Waymo log-replay 窗口生成若干条规则控制策略（conservative / aggressive / lateral_stable）的合成轨迹，用于验证 style embedding 是否能够区分不同策略的驾驶行为。

### 关键文件

| 脚本 | 功能 |
|------|------|
| `generate_policy_rollouts.py` | 从已有 traj/front 窗口为每个策略生成模拟自车轨迹 |
| `compute_style_features.py` | 对已有 traj/front npy 计算 20D style 特征（不依赖 TFRecord） |
| `evaluate_policy_separation.py` | 用 embedding 做策略分类 + Recall@K 检索评估 |
| `evaluate_policy_separation_aligned.py` | Source-aligned 策略区分评估（按 source_index 分组，控制场景分布） |

### 三种合成策略说明

| 策略 | THW 目标 | 最大加速度 | Jerk 限制 | 横向稳定性 |
|------|----------|------------|-----------|------------|
| `conservative` | 2.5 s | ±1.5/3.0 m/s² | 0.5 m/s²/step | 强（yaw_rate_clip=0.05 rad/step） |
| `aggressive` | 1.0 s | ±3.5/5.0 m/s² | 2.0 m/s²/step | 弱（yaw_rate_clip=0.20 rad/step） |
| `lateral_stable` | 1.8 s | ±2.0/3.5 m/s² | 0.8 m/s²/step | 极强（yaw_rate_clip=0.01 rad/step, heading_smooth_alpha=0.7） |

### 完整工作流（复制可用命令）

#### a) 生成合成策略 Rollout

```bash
python generate_policy_rollouts.py \
  --src_traj_path  output/traj.npy \
  --src_front_path output/front.npy \
  --src_split_path output/split.npy \
  --src_meta_path  output/meta.npy \
  --output_dir     output_policy_rollouts \
  --dt 0.1 \
  --policies "conservative,aggressive,lateral_stable" \
  --seed 42
```

输出文件：
- `output_policy_rollouts/traj.npy` — 模拟自车轨迹（N_policies × N_src 个窗口）
- `output_policy_rollouts/front.npy` — 原始前车轨迹（直接复制）
- `output_policy_rollouts/policy_id.npy` — 策略标签（int）
- `output_policy_rollouts/scenario_id.npy` — 场景 ID（来自 meta 或自动生成）
- `output_policy_rollouts/source_index.npy` — 回溯原始窗口索引
- `output_policy_rollouts/split.npy` — train/val/test 分割
- `output_policy_rollouts/policy_names.json` — 策略 id→名称映射

#### b) 计算 Style 特征

```bash
python compute_style_features.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --output_dir output_policy_rollouts
```

#### c) 训练 Embedding

```bash
python train_embedding.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --feat_path  output_policy_rollouts/feat_style.npy \
  --feat_raw_path output_policy_rollouts/feat_style_raw.npy \
  --split_path output_policy_rollouts/split.npy \
  --output_dir output_policy_rollouts/run_relkin_knn \
  --input_mode rel_kinematics --dt 0.1 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 1.0 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

#### d) 导出全量 Embedding

```bash
python export_embeddings.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --split_path output_policy_rollouts/split.npy \
  --checkpoint output_policy_rollouts/run_relkin_knn/best_model.pth \
  --output_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --input_mode rel_kinematics --dt 0.1
```

#### e) 评估策略区分能力

```bash
python evaluate_policy_separation.py \
  --embeddings_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path  output_policy_rollouts/policy_id.npy \
  --split_path      output_policy_rollouts/split.npy \
  --policy_names_path output_policy_rollouts/policy_names.json \
  --eval_split test \
  --k_neighbors 10 \
  --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_policy \
  --seed 42
```

输出：
- `policy_separation_summary.json` — 分类准确率、Macro-F1、Recall@K（汇总）
- `policy_retrieval.csv` — 每个测试样本的 Recall@K

### 注意事项

- 合成轨迹基于简单纵向控制器 + yaw-rate 限幅，**不追踪车道**，仅保持与原始轨迹方向大致对齐。
- `front.npy` 中的前车轨迹保持不变（外生 log-replay，不响应 ego）。
- 下游 `train_embedding.py` / `export_embeddings.py` / `evaluate_embedding.py` 均无需修改，直接指向新 output_dir 即可。
- 若 `--src_split_path` 未提供，脚本会按 scenario_id MD5 哈希自动分配 train/val/test（与 `build_dataset.py` 一致）。

---

## Synthetic policy rollouts: source-aligned evaluation

### Background

The standard `evaluate_policy_separation.py` measures global classification and
retrieval performance but does not control for *scenario distribution*: if different
policies happen to be evaluated on systematically different scenarios, the metric may
reflect difficulty differences rather than policy style differences.

`evaluate_policy_separation_aligned.py` addresses this by grouping samples by their
`source_index` — each source window was rolled out under every policy exactly once, so
within a source group the only variable is the policy.  All four computations below are
therefore scenario-controlled.

### Computations

| Step | Description | Output |
|------|-------------|--------|
| (a) Coverage validation | Checks each `(source_index, policy_id)` pair appears exactly once; reports missing / duplicate counts | summary JSON |
| (b) Within-source pairwise distances | Euclidean + cosine distance between every policy pair within each source group | `policy_pairwise_dist.csv` |
| (c) Centroid classification accuracy | Nearest-centroid prediction using train-split centroids; evaluated per source group on eval split | summary JSON |
| (d) Within-source retrieval applicability + margin | Check whether within-source same-policy NN retrieval is well-defined; report mean/median within-source distance margin | summary JSON |

### Copy-pastable commands (using default paths)

#### Step 1 — Generate rollouts

```bash
python generate_policy_rollouts.py \
  --src_traj_path  output/traj.npy \
  --src_front_path output/front.npy \
  --src_split_path output/split.npy \
  --src_meta_path  output/meta.npy \
  --output_dir     output_policy_rollouts \
  --dt 0.1 \
  --policies "conservative,aggressive,lateral_stable" \
  --seed 42
```

#### Step 2 — Compute style features

```bash
python compute_style_features.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --output_dir output_policy_rollouts
```

#### Step 3 — Train embedding

```bash
python train_embedding.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --feat_path  output_policy_rollouts/feat_style.npy \
  --feat_raw_path output_policy_rollouts/feat_style_raw.npy \
  --split_path output_policy_rollouts/split.npy \
  --output_dir output_policy_rollouts/run_relkin_knn \
  --input_mode rel_kinematics --dt 0.1 \
  --epochs 50 --batch_size 64 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --cond_mode knn --cond_k 24 --cond_scale_mode mad \
  --cond_cf_bucket_edges "0.2,0.6" \
  --loss_mode hybrid --pos_topk 8 --w_supcon 1.0 --w_soft 1.0 \
  --feat_clip_value 3.0 \
  --eval_every 10 --skip_val_clustering
```

#### Step 4 — Export embeddings

```bash
python export_embeddings.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --split_path output_policy_rollouts/split.npy \
  --checkpoint output_policy_rollouts/run_relkin_knn/best_model.pth \
  --output_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --input_mode rel_kinematics --dt 0.1
```

#### Step 5 — Run standard policy separation evaluation

```bash
python evaluate_policy_separation.py \
  --embeddings_path output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path  output_policy_rollouts/policy_id.npy \
  --split_path      output_policy_rollouts/split.npy \
  --policy_names_path output_policy_rollouts/policy_names.json \
  --eval_split test \
  --k_neighbors 10 \
  --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_policy \
  --seed 42
```

#### Step 6 — Run source-aligned evaluation

```bash
python evaluate_policy_separation_aligned.py \
  --embeddings_path   output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path    output_policy_rollouts/policy_id.npy \
  --source_index_path output_policy_rollouts/source_index.npy \
  --split_path        output_policy_rollouts/split.npy \
  --eval_split test \
  --analysis_dir output_policy_rollouts/run_relkin_knn/analysis_aligned \
  --seed 42
```

### Outputs

| File | Description |
|------|-------------|
| `policy_separation_aligned_summary.json` | Coverage stats, centroid accuracy, pairwise distance stats (mean/median), within-source retrieval applicability and margin |
| `policy_pairwise_dist.csv` | Per-source-group pairwise (Euclidean + cosine) distances for each policy pair |

### Notes

- `source_index.npy` is generated automatically by `generate_policy_rollouts.py`
  alongside the other output files.
- Samples where a source group does not have all policies present (e.g. at split
  boundaries) are excluded from computations (b) and (d); the coverage report in step
  (a) will list any such gaps.
- Centroids for classification (step c) are always estimated from the **train** split,
  regardless of `--eval_split`.
- If each source has only one sample per policy (common aligned setup), within-source
  same-policy nearest-neighbour retrieval is undefined; summary JSON will report
  `retrieval_applicable=false` and set NN hit-rate/chance to `null` instead of a
  misleading numeric 0.0.

---

## Embedding interpretability demo: retrieval + trajectory replay

`tools/embedding_retrieval_demo.py` provides the most intuitive way to visually
verify that embeddings cluster/separate driving styles into different regions.  Given a
query window it:

1. Retrieves the Top-K most-similar windows in embedding space.
2. Overlays ego + front trajectories (aligned to the query's initial position and heading).
3. Plots time-series style signals: speed, acceleration, jerk, and a curvature proxy.

### Prerequisites

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

No additional dependencies beyond the standard project requirements.

### Minimal command examples

#### Global retrieval (query against all items in the selected split)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 \
    --topk 5 \
    --mode global \
    --split_filter test
```

#### Within-source retrieval (only other rows sharing the same source meta-key)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 \
    --mode within-source
```

#### Select query by scenario ID (instead of array index)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_scenario_id "my_scenario_id" \
    --query_start 10 \
    --topk 5 \
    --mode global
```

#### Exclude same-scenario neighbours (prevent trivial retrieval)

```bash
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 \
    --topk 5 \
    --mode global \
    --exclude_same_scenario
```

#### Self-contained smoke test (no data files needed)

```bash
python tools/embedding_retrieval_demo.py --smoke_test
```

### Outputs

All files are written to `outputs/<run_id>/` (configurable with `--output_dir` and
`--run_id`).

| File | Description |
|------|-------------|
| `retrieval_table.csv` | Top-K results with index, meta fields, distance, and excluded flag |
| `traj_overlay.png` | Ego + front trajectories overlaid in aligned coordinates |
| `timeseries.png` | Speed / accel / jerk / curvature-proxy time series for query and Top-K |
| `summary.json` | Run parameters: mode, distance, topk, exclusions, data paths |

### Explanation of plots and what to look for

**`traj_overlay.png`** — Both ego and front trajectories are translated so the query
starts at the origin and rotated so the query's initial velocity vector points along +x.
This makes cross-scenario overlays comparable.  If the embedding is meaningful you
should see that Top-K retrieved trajectories follow a similar *shape* to the query (e.g.
similar following distance, similar lateral deviation).

**`timeseries.png`** — Shows four derived style signals sampled at `--dt` seconds per
step:
- **speed** — `sqrt(vx² + vy²)`
- **accel** — finite-difference of speed
- **jerk** — finite-difference of accel
- **curvature proxy** — `yaw_rate / max(speed, ε)`, where yaw_rate is estimated from
  heading differences.  This is an *approximation*; label it accordingly in any paper.

For a well-trained embedding the retrieved trajectories should show **similar profiles**
to the query across all four signals, especially in the features the embedding was
trained on (THW, jerk, lateral yaw-rate, etc.).

### Within-source limitation (no explicit policy_id)

The base dataset (`build_dataset.py`) stores meta as
`(scenario_id, start, window_len, front_id)` with **no** explicit `policy_id` field.
In `within-source` mode the script groups all rows sharing the same meta-key tuple and
plots all of them against the query.  When data was produced by `generate_policy_rollouts.py`
there will be exactly `n_policies` rows per meta-key (one per policy), and you can
inspect their relative ordering by distance to verify separability.  If you need
precise per-policy labels use the `policy_id.npy` output of `generate_policy_rollouts.py`
and the aligned evaluator (`evaluate_policy_separation_aligned.py`).

### Running the smoke / unit tests

```bash
python scripts/smoke_test_retrieval_demo.py
```

### PR2 interpretability demo (`tools/embedding_interpretability_demo.py`)

For PR2-style interpretability (same-source triplet + global retrieval cards), use:

```bash
python tools/embedding_interpretability_demo.py \
  --data_dir output_policy_rollouts \
  --out_dir outputs/embedding_demo/case_000 \
  --embedding feat_style \
  --split test \
  --mode both \
  --projection both \
  --case_selection best_human_readable \
  --topk 5 \
  --source_key_fields scenario_id,start,window_len,front_id \
  --auto_select_valid_source
```

If `front_id` changes across policy rollouts, relax grouping with:

```bash
--source_key_fields scenario_id,start,window_len
```

The demo requires multi-policy rollout rows (typically 3 rows per source key).  
Check `summary.json -> diagnostics` to verify:
- row counts before/after split,
- source-group size histograms,
- policy-id availability/source/counts,
- core array shapes (`embedding/meta/traj/front/split`).

When `policy_id` is unavailable, hit@k for same-policy retrieval is intentionally set to `null` and a warning explains that nearest-neighbour visualization is still possible but same-policy verification is not.

#### Embedding interpretability demo outputs (for paper/presentation)

- `summary.json`: includes `policy_mapping`, `case_selection`, within-source distances, retrieval hit@k, and diagnostics.
- `embedding_2d_projection.png` / `embedding_2d_projection.csv`: PCA projection (visualization only; lossy).
- `embedding_2d_projection_umap.png` / `.csv`: produced when `--projection umap|both` and `umap-learn` is available.
- `embedding_distance_matrix.png` / `.csv`: within-source embedding distances with per-cell numeric annotation and policy labels.
- `within_source_triplet.png`, `within_source_style_signals.png`, `within_source_style_fingerprint_kinematic.png`, `within_source_style_fingerprint_dynamics.png`, `within_source_style_fingerprint_normalized.png`, `within_source_style_fingerprint.csv`: same-source policy contrast and style statistics.
- `global_retrieval_cards.png`, `global_retrieval_style_signals.png`, `retrieval_table.csv`, `style_fingerprint.csv`: global nearest-neighbor interpretability and style fingerprints.
- `interpretability_report.md`: auto-generated textual summary from summary/CSV outputs.

Interpretation guidance:
- PCA/UMAP are visualization-only; benchmark conclusions should rely on aligned metrics and high-dimensional embedding distances.
- Lack of perfectly separated 2D clusters does not invalidate high-dimensional separation.
- Metadata (`policy_id`, `policy_name`, `source_index`) is required for policy-level same-source contrast and same-policy hit@k verification.

## Experiment 2: Lateral_stable Ablation and Parameter Sweep

新增脚本：`tools/run_lateral_stable_ablation.py`，用于批量运行 `lateral_stable` 参数消融（生成 + population 评估 + 汇总 + 推荐 + 报告 + 图表）。

### Debug 命令
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable
```

### Full 命令
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_full \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### 仅预览配置（不执行）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_plan \
  --dry_run
```

输出包括：
- `ablation_summary.csv` / `ablation_summary.json`
- `ablation_recommendation.json`
- `ablation_report.md`
- 汇总图：`ablation_*.png`
- 每个 config 独立目录下的 `rollouts/` 与 `population_eval/`

## Experiment 2: Lateral_stable Ablation and Parameter Sweep

**Motivation**: Experiment 1 showed p2/lateral_stable is recognizable but remains too close to conservative, so p2 is not consistently an independent third style.

**Script**: `tools/run_lateral_stable_ablation.py`

**Required inputs**:
- `--source_data_dir` containing generator-compatible `traj.npy` and `front.npy` (optional but recommended `split.npy`, `meta.npy`).
- Existing tools: `generate_policy_rollouts.py`, `tools/evaluate_policy_population.py`.

**Debug command (max_sources=100)**:
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

**Dry-run command**:
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --dry_run
```

**Full command**:
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_full \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

**Output directory structure**:
- `base_output_dir/<config_name>/rollouts/`
- `base_output_dir/<config_name>/population_eval/`
- `base_output_dir/ablation_summary.csv`
- `base_output_dir/ablation_summary.json`
- `base_output_dir/ablation_recommendation.json`
- `base_output_dir/ablation_report.md`
- `base_output_dir/ablation_p2_separation_margin.png`
- `base_output_dir/ablation_p2_farthest_rate.png`
- `base_output_dir/ablation_pairwise_distances.png`
- `base_output_dir/ablation_retrieval_classification.png`
- `base_output_dir/ablation_p2_style_metrics.png`
- `base_output_dir/ablation_tradeoff_plot.png`

**How to interpret core metrics**:
- `p2_farthest_rate`: higher is better.
- `mean_p2_separation_margin > 0`: p2 is farther from both p0/p1 than p0-p1 are from each other.
- Lower `p2_rms_yaw_rate_proxy_mean` indicates stronger lateral stability.
- Lower `p2_rms_jerk_mean` indicates smoother longitudinal comfort.
- Retrieval + centroid metrics quantify policy discriminability.

**Known limitations**:
- Synthetic policies (no human labels).
- Replayed front vehicle (not full multi-agent closed loop).
- No sensor rendering/perception stack.


## Experiment 2 Ablation（必须产出 base_output_dir 聚合文件）

### 推荐命令（可直接复制）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### 期望输出结构
```text
outputs/ablation_debug/
  ablation_summary.csv
  ablation_summary.json
  ablation_recommendation.json
  ablation_report.md
  ablation_p2_separation_margin.png
  ablation_p2_farthest_rate.png
  ablation_pairwise_distances.png
  ablation_retrieval_classification.png
  ablation_p2_style_metrics.png
  ablation_tradeoff_plot.png

  baseline_current/
    rollouts/
    population_eval/
      population_summary.json

  no_lateral_smoothing/
    rollouts/
    population_eval/
      population_summary.json

  lateral_only/
    rollouts/
    population_eval/
      population_summary.json

  comfort_only/
    rollouts/
    population_eval/
      population_summary.json

  full_strong_lateral_stable/
    rollouts/
    population_eval/
      population_summary.json
```

> 完成标准：`ablation_summary.csv` 与 `ablation_report.md` 必须存在于 `base_output_dir` 根目录。
