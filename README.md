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

### 2) 训练

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --feat_path output/feat_style.npy \
  --split_path output/split.npy \
  --output_dir output/run_style_win80_stride20 \
  --epochs 50 \
  --batch_size 64 \
  --temperature 0.1 \
  --feat_norm none \
  --tau_mode anchor_median \
  --gate_topm 0
```

### Soft target ablation（tau vs local scaling）

- 原始（tau）：
  - `--feat_sim tau --tau_mode anchor_median`
- local scaling（推荐）：
  - `--feat_sim local_scale --ls_k 10 --ls_mode row --ls_sigma_min 1e-3 --ls_alpha 1.0`

local scaling 的目标是让 soft target 更局部化，避免 `effective_k` 接近 batch size。

可用以下命令验收（期望日志里 `effective_k_mean` 显著下降）：

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --feat_path output/feat_style.npy \
  --split_path output/split.npy \
  --output_dir output/run_style_ls_k10 \
  --epochs 50 --batch_size 64 \
  --temperature 0.1 \
  --feat_norm none \
  --feat_sim local_scale \
  --ls_k 10 --ls_mode row --ls_sigma_min 1e-3 \
  --ls_alpha 1.5
```

### 缺失感知特征距离（处理 cf 缺失导致的伪相似）

- `feat_style.npy` 在构建时会把 NaN/inf 填 0；对于 cf 相关维度（11–19）缺失较多时，“共同缺失样本对”会在特征距离上被错误拉近。
- 训练时可额外提供 `feat_style_raw.npy` 生成有效维 mask，仅在双方都有效的维度上计算 pairwise feature distance，并按共同有效维度数归一化，减少伪近邻。

推荐命令（local scaling + masked distance）：

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --feat_path output/feat_style.npy \
  --feat_raw_path output/feat_style_raw.npy \
  --split_path output/split.npy \
  --output_dir output/run_style_masked_ls_k1_a3 \
  --epochs 50 --batch_size 64 \
  --temperature 0.1 \
  --feat_norm none \
  --feat_sim local_scale --ls_k 1 --ls_mode row --ls_alpha 3 \
  --feat_dist_mode masked --min_common_dims 5 \
  --eval_every 10
```

快速扫参（降低训练时评估开销）可把评估频率调低，并按需跳过 val 聚类评估：

```bash
python train_embedding.py \
  --traj_path output/traj.npy \
  --feat_path output/feat_style.npy \
  --split_path output/split.npy \
  --output_dir output/run_style_sweep_fast \
  --epochs 50 --batch_size 64 \
  --feat_sim local_scale \
  --ls_k 10 --ls_mode row --ls_sigma_min 1e-3 --ls_alpha 1.5 \
  --eval_every 10 \
  --skip_val_clustering
```

### 3) 导出全量 embedding

```bash
python export_embeddings.py \
  --traj_path output/traj.npy \
  --split_path output/split.npy \
  --checkpoint output/run_style_win80_stride20/best_model.pth \
  --output_path output/run_style_win80_stride20/embeddings_all.npy
```

### 4) 评估

```bash
python evaluate_embedding.py \
  --embeddings_path output/run_style_win80_stride20/embeddings_all.npy \
  --feat_path output/feat_style.npy \
  --split_path output/split.npy \
  --eval_split test \
  --feature_names_path output/feature_names_style.json \
  --analysis_dir output/run_style_win80_stride20/analysis_best \
  --plot_first_k 20 \
  --k_neighbors 10 \
  --umap_neighbors 30 \
  --umap_min_dist 0.1 \
  --seed 42 \
  --kmeans_clusters 8
```

## 关键输出文件

- `output/traj.npy`: 自车滑窗轨迹
- `output/front.npy`: 前车滑窗轨迹
- `output/feat_style_raw.npy`: 原始 style 特征
- `output/feat_style.npy`: 训练用标准化 style 特征
- `output/feature_names_style.json`: 特征名
- `output/split.npy`: train/val/test
- `output/<run>/best_model.pth`: 最优模型
- `output/<run>/embeddings_all.npy`: 全量 embedding
- `output/<run>/analysis_*`: 评估结果图表与 CSV
