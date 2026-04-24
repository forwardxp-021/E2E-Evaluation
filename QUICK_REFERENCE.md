# E2E-Evaluation 项目快速参考

## 项目结构
```
E2E-Evaluation/
├── build_dataset.py       # 从 Waymo TFRecord 构建 traj/feat/meta/split
├── dataset.py             # TrajFeatureDataset + KNN pair 预计算
├── model.py               # TrajectoryEncoder (GRU + MLP head)
├── loss.py                # SoftContrastiveLoss + multi_positive_infonce
├── train_embedding.py     # 训练主脚本
├── export_embeddings.py   # 导出全量 embedding (.npy)
├── evaluate_embedding.py  # UMAP 可视化 + 线性探针 + 邻域一致性
├── data/                  # 数据目录（数据不提交）
└── README.md              # 项目说明
```

## 工作流

### 第1步: 构建数据集
```bash
python build_dataset.py \
    --tfrecord_glob "/mnt/d/WMdata/*.tfrecord-*" \
    --output_dir output \
    --min_ego_speed 5.5 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```
**输出**: `output/traj.npy`, `feat.npy`, `meta.npy`, `split.npy`, `summary.txt`, `summary.csv`  
**特征维度**: 20D（标准化后）

### 第2步: 模型训练
```bash
python train_embedding.py \
    --traj_path output/traj.npy \
    --feat_path output/feat.npy \
    --split_path output/split.npy \
    --output_dir output \
    --epochs 50
```
**损失**: `SoftContrastiveLoss`（特征引导软对比）  
**输出**: `output/best_model.pth`, `output/model_final.pth`

### 第3步: 导出全量 embedding
```bash
python export_embeddings.py \
    --traj_path output/traj.npy \
    --checkpoint output/best_model.pth \
    --output_path output/embeddings_all.npy
```
**输出**: `output/embeddings_all.npy`，shape `(N, 64)`

### 第4步: 评估分析
```bash
python evaluate_embedding.py \
    --embeddings_path output/embeddings_all.npy \
    --feat_path output/feat.npy \
    --split_path output/split.npy \
    --analysis_dir output/analysis
```
**输出**: UMAP 散点图、线性探针 R²/Spearman、邻域一致性分析

## 关键参数

### build_dataset.py
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tfrecord_glob` | `/mnt/d/WMdata/*.tfrecord-*` | TFRecord 文件匹配模式 |
| `--output_dir` | `output` | 输出目录 |
| `--min_ego_speed` | `5.5` | 最低自车速度过滤阈值 (m/s) |
| `--train_ratio` | `0.8` | 训练集比例 |
| `--val_ratio` | `0.1` | 验证集比例 |
| `--test_ratio` | `0.1` | 测试集比例 |
| `--limit_files` | `None` | 限制处理文件数（调试用） |

**特征维度**: 20D
- 相对速度 (3): 均值, 标差, 正向比例
- THW (3): 均值, 标差, 最小值
- Jerk (3): 均值, 标差, 95分位
- 相对加速度 (2): 均值, 标差
- 反应时间 (1)
- 横向 (3): 偏航率标差, 变道次数, 变道时长
- 速度归一化 (2): 均值, 标差
- 稳定性 (2): 速度标差, 加速度标差
- 自车速度均值 (1)

### train_embedding.py
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `64` | 批大小 |
| `--lr` | `1e-3` | 学习率 (AdamW) |
| `--hidden_dim` | `128` | GRU 隐层维度 |
| `--emb_dim` | `64` | embedding 输出维度 |
| `--temperature` | `0.1` | 对比损失温度 |
| `--eval_every` | `2` | 每 N 轮评估一次 |
| `--n_clusters` | `3` | KMeans 聚类数 |

### evaluate_embedding.py
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--k_neighbors` | `10` | 邻域一致性分析的邻居数 |
| `--umap_neighbors` | `30` | UMAP n_neighbors |
| `--umap_min_dist` | `0.1` | UMAP min_dist |
| `--umap_max_points` | `50000` | UMAP 最大样本数 |
| `--ridge_alpha` | `1.0` | 线性探针正则强度 |

## 数据维度总结

| 模块 | 输入 | 输出 |
|---|---|---|
| `build_dataset.py` | Waymo TFRecord | traj `(N,T,4)`, feat `(N,20)`, meta `(N,3)`, split `(N,)` |
| `TrajectoryEncoder` | `(B,T,4)` 轨迹 | `(B,64)` L2 归一化 embedding |
| `export_embeddings.py` | traj.npy + checkpoint | `embeddings_all.npy (N,64)` |
| `evaluate_embedding.py` | embeddings + feat + split | UMAP 图, 探针结果, 邻域分析 |

## 常见问题

### Q: 如何修改特征维度?
A: 修改 `build_dataset.py` 中 `compute_features()` 函数的返回列表（当前 20D）

### Q: 如何修改 embedding 维度?
A: 通过 `train_embedding.py --emb_dim <N>` 指定，`export_embeddings.py --emb_dim <N>` 保持一致

### Q: 如何只处理少量文件调试?
A: 使用 `--limit_files 5` 参数限制读取文件数

### Q: 数据集划分如何保证可重复?
A: `assign_split()` 使用 `scenario_id` 的 MD5 哈希值确定性划分，无随机性

---

## Synthetic Policy Rollouts: End-to-End Pipeline & Verification

### Minimum Reproduction Commands

```bash
# Step 1 — Generate synthetic policy rollouts (conservative / aggressive / lateral_stable)
python generate_policy_rollouts.py \
  --src_traj_path  output/traj.npy \
  --src_front_path output/front.npy \
  --src_split_path output/split.npy \
  --src_meta_path  output/meta.npy \
  --output_dir     output_policy_rollouts \
  --dt 0.1 \
  --policies "conservative,aggressive,lateral_stable" \
  --seed 42

# Step 2 — Compute style features
python compute_style_features.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --front_path output_policy_rollouts/front.npy \
  --output_dir output_policy_rollouts

# Step 3 — Train embedding model on policy rollouts
python train_embedding.py \
  --traj_path  output_policy_rollouts/traj.npy \
  --feat_path  output_policy_rollouts/feat.npy \
  --split_path output_policy_rollouts/split.npy \
  --output_dir output_policy_rollouts/run_relkin_knn \
  --epochs 50

# Step 4 — Export embeddings
python export_embeddings.py \
  --traj_path      output_policy_rollouts/traj.npy \
  --checkpoint     output_policy_rollouts/run_relkin_knn/best_model.pth \
  --output_path    output_policy_rollouts/run_relkin_knn/embeddings_all.npy

# Step 5 — Aligned evaluation (source-controlled policy separation)
python evaluate_policy_separation_aligned.py \
  --embeddings_path   output_policy_rollouts/run_relkin_knn/embeddings_all.npy \
  --policy_id_path    output_policy_rollouts/policy_id.npy \
  --source_index_path output_policy_rollouts/source_index.npy \
  --split_path        output_policy_rollouts/split.npy \
  --eval_split test \
  --analysis_dir      output_policy_rollouts/run_relkin_knn/analysis_aligned
```

### Verifying Policy Separation with Aligned Metrics

After running Step 5, inspect `policy_separation_aligned_summary.json` and the printed output.
The two most informative indicators are:

1. **`p0_vs_p2` pairwise Euclidean distance** (`pairwise_distances.p0_vs_p2.euclidean_mean`)  
   Measures how far apart `conservative` (p0) and `lateral_stable` (p2) embeddings are
   within the same source window.  A healthy separation shows a mean ≥ 0.30 (the larger
   the better).  If this value is low (< 0.15) the `lateral_stable` policy has not yet
   differentiated itself from `conservative`.

2. **`policy_2` centroid classification accuracy** (`centroid_classification.per_policy_accuracy."2"`)  
   Measures how well the embedding of `lateral_stable` examples can be classified back to
   their correct policy using train-split centroids.  Target ≥ 0.60 (chance = 0.333 for 3
   policies).  If this number stays near chance the embedding is not capturing the
   lateral-stability style signal.

**Typical healthy result** (395 test sources, 1185 eval samples):
```
p0_vs_p1  euclidean mean ≈ 0.71
p0_vs_p2  euclidean mean ≈ 0.35   ← key separation signal
p1_vs_p2  euclidean mean ≈ 0.50
Centroid accuracy overall ≈ 0.66  (chance 0.333)
  policy_0 ≈ 0.64
  policy_1 ≈ 0.69
  policy_2 ≈ 0.66                 ← key lateral_stable classification
```

### `lateral_stable` Policy Parameters

The `lateral_stable` policy introduces a distinct driving style through two lateral
mechanisms and tighter longitudinal comfort bounds:

| Parameter | Default | Meaning |
|---|---|---|
| `yaw_rate_clip` | `0.01 rad/step` | Maximum per-step heading change. Smaller → smoother lateral motion. |
| `heading_smooth_alpha` | `0.7` | EMA smoothing weight applied to the desired heading target. |
| `thw_target` | `1.8 s` | Desired time-headway (between conservative 2.5 s and aggressive 1.0 s). |
| `jerk_limit` | `0.8 m/s²/step` | Maximum longitudinal jerk. |
| `a_max` | `2.0 m/s²` | Maximum acceleration. |
| `a_min` | `-3.5 m/s²` | Maximum deceleration. |

**`heading_smooth_alpha` semantics:**

```
heading_smooth_alpha = 0.0  → no smoothing: desired heading tracks source exactly
heading_smooth_alpha = 0.5  → moderate smoothing: heading updates at half the raw delta
heading_smooth_alpha = 0.7  → strong smoothing (default): heading lags source by ~3 steps
heading_smooth_alpha → 1.0  → heading frozen: no response to source path changes (extreme)
```

Formally the EMA update is:
```
smoothed_target ← smoothed_target + (1 − alpha) × wrap(raw_target − smoothed_target)
```
A value close to `1.0` means `(1 − alpha) → 0`, so the target barely moves per step
(strong smoothing / very slow update). A value of `0.0` disables smoothing entirely.

**CLI overrides for `lateral_stable`** (all optional; print in "Final effective parameters"):
```bash
python generate_policy_rollouts.py ... \
  --lateral_stable_yaw_rate_clip 0.005 \
  --heading_smooth_alpha         0.8 \
  --lateral_stable_thw_target    2.0 \
  --lateral_stable_jerk_limit    0.6 \
  --lateral_stable_a_max         1.8 \
  --lateral_stable_a_min        -3.0
```

### Retrieval Metric Note

In the standard within-source setting (1 sample per policy per source), the
**Nearest-neighbour hit rate** is structurally impossible to be non-zero: every
within-source neighbour has a *different* policy label.  The evaluator detects this
automatically and reports `N/A` with an explanation rather than the misleading `0.0`.
The `pairwise_distances` and `centroid_classification` metrics are unaffected and remain
the primary indicators of policy separation quality.

---

## 依赖库

详见 `requirements-cpu.txt`。主要依赖：
- `torch` (CPU 版)
- `tensorflow-cpu`
- `waymo-open-dataset`
- `scikit-learn`
- `umap-learn`
- `numpy`, `pandas`, `matplotlib`

## 最近重构 (2026-04-07)

✅ 重构 `build_dataset.py`：添加 CLI 参数、输出 meta/split、MD5 确定性划分、速度过滤  
✅ 新增 `dataset.py`：`TrajFeatureDataset` + 变长轨迹 collate + KNN pair 预计算  
✅ 新增 `model.py`：`TrajectoryEncoder`（GRU + MLP head，64D 输出）  
✅ 新增 `loss.py`：`SoftContrastiveLoss`（特征引导软对比）  
✅ 重构 `train_embedding.py`：适配新架构，支持 split 文件  
✅ 新增 `export_embeddings.py`：全量 embedding 导出  
✅ 新增 `evaluate_embedding.py`：UMAP + 线性探针 + 邻域一致性  
✅ 移除旧脚本：`generate_embeddings.py`, `visualize_umap.py`, `analyze_style_embedding.py`, `analysis/`, `scripts/`, `docs/`
