# E2E-Evaluation

E2E Evaluation research in automated driving — learning driving-style representations
from trajectory data (Waymo Open Dataset).

---

## 项目概述

本项目从 Waymo Open Dataset 轨迹数据中学习**驾驶风格表示（embedding）**，判断是否编码了激进/保守/平滑/跟车等驾驶风格信息。

核心流程：
1. **数据构建** — 从原始 TFRecord 提取自车+前车对齐轨迹，输出 legacy(20D) 与 style(20D) 两套特征，按 MD5 哈希确定性地划分 train/val/test
2. **模型训练** — GRU 轨迹编码器 + 特征引导软对比损失，输出 L2 归一化 embedding
3. **导出 embedding** — 对全量数据运行推理，保存为 `embeddings_all.npy`
4. **评估分析** — UMAP 可视化、线性探针、邻域一致性分析

---

## 快速开始

```bash
pip install -r requirements-cpu.txt

# 1. 构建数据集
python build_dataset.py \
    --tfrecord_glob "/mnt/d/WMdata/*.tfrecord-*" \
    --output_dir output \
    --min_ego_speed 5.5 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 2. 训练 embedding 模型
python train_embedding.py \
    --traj_path output/traj.npy \
    --feat_path output/feat_style.npy \
    --split_path output/split.npy \
    --output_dir output \
    --epochs 50

# 3. 导出全量 embedding
python export_embeddings.py \
    --traj_path output/traj.npy \
    --checkpoint output/best_model.pth \
    --output_path output/embeddings_all.npy

# 4. 评估分析
python evaluate_embedding.py \
    --embeddings_path output/embeddings_all.npy \
    --feat_path output/feat_style.npy \
    --split_path output/split.npy \
    --feature_names_path output/feature_names_style.json \
    --analysis_dir output/analysis
```

---

## 仓库结构

```
build_dataset.py       # 从 Waymo TFRecord 构建 traj/feat/meta/split
dataset.py             # TrajFeatureDataset + KNN pair 预计算
model.py               # TrajectoryEncoder (GRU + MLP head)
loss.py                # SoftContrastiveLoss + multi_positive_infonce
train_embedding.py     # 训练主脚本
export_embeddings.py   # 导出全量 embedding (.npy)
evaluate_embedding.py  # UMAP 可视化 + 线性探针 + 邻域一致性分析
data/
  .gitkeep             # 数据目录占位（数据不提交）
output/                # 生成的模型、embedding 及分析结果（gitignored）
```

---

## 数据与特征

### `build_dataset.py` 输出

| 文件 | 说明 |
|---|---|
| `output/traj.npy` | 对象数组，每行为 `(T, 4)` 轨迹片段 `[x, y, vx, vy]` |
| `output/front.npy` | 对象数组，每行为 `(T, 4)` 前车对齐轨迹 `[x, y, vx, vy]` |
| `output/feat_legacy.npy` | `(N, 20)` float32，标准化后的 legacy 对照特征 |
| `output/feat_style_raw.npy` | `(N, 20)` float32，新 style 特征（未标准化，含 NaN） |
| `output/feat_style.npy` | `(N, 20)` float32，新 style 特征（NaN→0 后全局标准化，训练默认） |
| `output/feat.npy` | 与 `feat_legacy.npy` 相同（兼容旧脚本） |
| `output/feature_names_style.json` | style 特征名列表（与 `feat_style.npy` 维度严格对齐） |
| `output/meta.npy` | `(N, 3)` 对象数组 `(scenario_id, ego_idx, front_id)` |
| `output/split.npy` | `(N,)` 字符串数组 `"train"/"val"/"test"`（MD5 确定性划分） |
| `output/summary.txt` | 数据集统计摘要 |
| `output/summary.csv` | 同上（CSV 格式） |

### 新 style 特征（20D）

| 维度 | 特征名 | 含义 |
|---|---|---|
| 0 | `acc_abs_p95` | 自车加速度绝对值 95 分位 |
| 1 | `acc_abs_p99` | 自车加速度绝对值 99 分位 |
| 2 | `acc_rms` | 自车加速度 RMS |
| 3 | `jerk_abs_p95` | 自车 jerk 绝对值 95 分位 |
| 4 | `jerk_abs_p99` | 自车 jerk 绝对值 99 分位 |
| 5 | `jerk_rms` | 自车 jerk RMS |
| 6 | `yaw_rate_rms` | 横摆角速度 RMS |
| 7 | `yaw_rate_abs_p95` | 横摆角速度绝对值 95 分位 |
| 8 | `heading_change_total` | 累计航向变化绝对值 |
| 9 | `speed_control_oscillation` | 速度增量符号翻转率 |
| 10 | `cf_valid_frac` | 严格跟车态有效占比 |
| 11 | `thw_p50` | 跟车态 THW 中位数 |
| 12 | `thw_p20` | 跟车态 THW 20 分位 |
| 13 | `thw_iqr` | 跟车态 THW 四分位距 |
| 14 | `v_rel_p50` | 跟车态相对速度中位数 |
| 15 | `closing_gain_kv` | 跟驰拟合闭合速度增益 |
| 16 | `gap_gain_kd` | 跟驰拟合间距增益 |
| 17 | `desired_gap_d0` | 跟驰拟合期望间距 |
| 18 | `acc_sync_lag` | 跟车态前后车加速度同步滞后（帧） |
| 19 | `acc_sync_corr` | 跟车态前后车加速度最大互相关 |

---

## 模型架构

```
TrajectoryEncoder
  GRU(input=4, hidden=128, batch_first=True)
  → 最后时刻隐状态 h[-1]
  → Linear(128, 128) → ReLU → Linear(128, 64)
  → L2 normalize → embedding (64D)
```

训练使用 `SoftContrastiveLoss`（特征引导软对比损失）：批内所有样本对根据特征空间距离计算软标签权重，不依赖硬正负对挖掘。

---

## 评估指标

`evaluate_embedding.py` 产出以下分析结果（保存在 `output/analysis/`）：

| 分析 | 说明 |
|---|---|
| UMAP 可视化 | 按各维度特征着色的 2D 散点图 |
| 线性探针（Ridge Regression）| 各特征的 R² 和 Spearman 相关系数 |
| 邻域一致性 | embedding 邻域 vs. 特征邻域 vs. 随机基线的 overlap 比较 |

# Feature-Guided Behavior Embedding (Soft Contrastive)

This project trains a trajectory-only encoder with feature-guided soft contrastive supervision.

## 1) Build Dataset

```bash
conda run -n waymo_dev python build_dataset.py \
	--output_dir output
```

Main outputs:

- `output/traj.npy`
- `output/feat_style.npy`
- `output/split.npy`

## 2) Train

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/train_embedding.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--output_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output" \
	--epochs 50 \
	--batch_size 64 \
	--temperature 0.1 \
	--feat_norm none \
	--tau_mode anchor_median \
	--gate_topm 0
```

Important training outputs:

- `output/best_model.pth`
- `output/model_final.pth`
- `output/embeddings_test.npy`

New loss-related CLI options:

- `--feat_norm {none,batch_std,l2}`
	- `none` (default): no extra normalization inside loss.
	- `batch_std`: batch-wise per-dimension standardization.
	- `l2`: per-sample L2 normalization.
- `--tau_mode {batch_median,anchor_median}`
	- `anchor_median` is the default.
- `--gate_topm INT`
	- `0` (default) disables gating.
	- `>0` keeps only top-M nearest feature neighbors per anchor in the softmax.

Per-epoch logs include:

- loss
- valid anchors per batch
- tau statistics (`tau_feat` or `tau_feat_mean`/`tau_feat_median`)
- `sim_feat_entropy_mean`
- `effective_k_mean`

## 3) Export Full Embeddings (optional but recommended)

Use this to evaluate with train/test split-aware probe:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/export_embeddings.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--checkpoint "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/best_model.pth" \
	--output_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/embeddings_all.npy"
```

## 4) Evaluate (UMAP + Probe + Neighbor Consistency)

### 4.1 Split-aware evaluation (recommended)

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/evaluate_embedding.py" \
	--embeddings_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/embeddings_all.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--eval_split test \
	--analysis_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/analysis" \
	--kmeans_clusters 8
```

### 4.2 Evaluate with test-only embeddings

```bash
conda run -n waymo_dev python evaluate_embedding.py \
	--embeddings_path output/embeddings_test.npy \
	--feat_path output/feat_style.npy \
	--split_path output/split.npy \
	--eval_split test \
	--analysis_dir output/analysis
```

### 4.3 Evaluate without split file (random holdout for probe)

```bash
conda run -n waymo_dev python evaluate_embedding.py \
	--embeddings_path output/embeddings_all.npy \
	--feat_path output/feat_style.npy \
	--analysis_dir output/analysis \
	--probe_test_ratio 0.2
```

Evaluation outputs in `output/analysis`:

- `probe_results.csv` (Linear/Ridge: R2, RMSE, Spearman)
- `neighbor_results.csv` (embedding-neighbor vs random-neighbor statistics)
- `umap_coordinates.csv`
- `umap_base.png`
- `umap_feat_*.png`
- `umap_kmeans_labels.png` (if `--kmeans_clusters > 0`)

## 5) Ablation Runs (train + export + evaluate)

For fair comparison, use the same evaluation settings for all runs:

- `--umap_neighbors 30`
- `--umap_min_dist 0.1`
- `--seed 42`

### 5.1 Baseline (old behavior replication)

Train:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/train_embedding.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--output_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_baseline" \
	--epochs 50 --batch_size 64 --temperature 0.1 \
	--feat_norm batch_std --tau_mode batch_median --gate_topm 0
```

Export full embeddings:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/export_embeddings.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--checkpoint "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_baseline/best_model.pth" \
	--output_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_baseline/embeddings_all.npy"
```

Evaluate:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/evaluate_embedding.py" \
	--embeddings_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_baseline/embeddings_all.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--eval_split test \
	--analysis_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_baseline/analysis" \
	--umap_neighbors 30 --umap_min_dist 0.1 --seed 42 --kmeans_clusters 8
```

### 5.2 Remove Batch Standardization

Train:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/train_embedding.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--output_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_no_batch_std" \
	--epochs 50 --batch_size 64 --temperature 0.1 \
	--feat_norm none --tau_mode batch_median --gate_topm 0
```

Export full embeddings:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/export_embeddings.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--checkpoint "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_no_batch_std/best_model.pth" \
	--output_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_no_batch_std/embeddings_all.npy"
```

Evaluate:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/evaluate_embedding.py" \
	--embeddings_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_no_batch_std/embeddings_all.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--eval_split test \
	--analysis_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_no_batch_std/analysis" \
	--umap_neighbors 30 --umap_min_dist 0.1 --seed 42 --kmeans_clusters 8
```

### 5.3 Per-Anchor Tau

Train:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/train_embedding.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--output_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_anchor_tau" \
	--epochs 50 --batch_size 64 --temperature 0.1 \
	--feat_norm none --tau_mode anchor_median --gate_topm 0
```

Export full embeddings:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/export_embeddings.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--checkpoint "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_anchor_tau/best_model.pth" \
	--output_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_anchor_tau/embeddings_all.npy"
```

Evaluate:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/evaluate_embedding.py" \
	--embeddings_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_anchor_tau/embeddings_all.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--eval_split test \
	--analysis_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_anchor_tau/analysis" \
	--umap_neighbors 30 --umap_min_dist 0.1 --seed 42 --kmeans_clusters 8
```

### 5.4 Gated Neighborhood

Train:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/train_embedding.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--output_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_gated" \
	--epochs 50 --batch_size 64 --temperature 0.1 \
	--feat_norm none --tau_mode anchor_median --gate_topm 32
```

Export full embeddings:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/export_embeddings.py" \
	--traj_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/traj.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--checkpoint "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_gated/best_model.pth" \
	--output_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_gated/embeddings_all.npy"
```

Evaluate:

```bash
conda run -n waymo_dev python "/home/king/liuqingphd/20260402_add_feature-based positive pairs/evaluate_embedding.py" \
	--embeddings_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_gated/embeddings_all.npy" \
	--feat_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/feat_style.npy" \
	--split_path "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/split.npy" \
	--eval_split test \
	--analysis_dir "/home/king/liuqingphd/20260402_add_feature-based positive pairs/output/ablation_gated/analysis" \
	--umap_neighbors 30 --umap_min_dist 0.1 --seed 42 --kmeans_clusters 8
```

## 6) One-Click Run for All 4 Ablations

```bash
#!/usr/bin/env bash
set -euo pipefail

PY="conda run -n waymo_dev python"
ROOT="/home/king/liuqingphd/20260402_add_feature-based positive pairs"
TRAIN="$ROOT/train_embedding.py"
EXPORT="$ROOT/export_embeddings.py"
EVAL="$ROOT/evaluate_embedding.py"

TRAJ="$ROOT/output/traj.npy"
FEAT="$ROOT/output/feat_style.npy"
SPLIT="$ROOT/output/split.npy"

# name feat_norm tau_mode gate_topm
CONFIGS=(
	"ablation_baseline batch_std batch_median 0"
	"ablation_no_batch_std none batch_median 0"
	"ablation_anchor_tau none anchor_median 0"
	"ablation_gated none anchor_median 32"
)

for cfg in "${CONFIGS[@]}"; do
	read -r NAME FEAT_NORM TAU_MODE GATE_TOPM <<< "$cfg"
	OUT_DIR="$ROOT/output/$NAME"
	ANALYSIS_DIR="$OUT_DIR/analysis"

	echo "[RUN] $NAME"

	$PY "$TRAIN" \
		--traj_path "$TRAJ" \
		--feat_path "$FEAT" \
		--split_path "$SPLIT" \
		--output_dir "$OUT_DIR" \
		--epochs 50 --batch_size 64 --temperature 0.1 \
		--feat_norm "$FEAT_NORM" --tau_mode "$TAU_MODE" --gate_topm "$GATE_TOPM"

	$PY "$EXPORT" \
		--traj_path "$TRAJ" \
		--split_path "$SPLIT" \
		--checkpoint "$OUT_DIR/best_model.pth" \
		--output_path "$OUT_DIR/embeddings_all.npy"

	$PY "$EVAL" \
		--embeddings_path "$OUT_DIR/embeddings_all.npy" \
		--feat_path "$FEAT" \
		--split_path "$SPLIT" \
		--eval_split test \
		--analysis_dir "$ANALYSIS_DIR" \
		--umap_neighbors 30 --umap_min_dist 0.1 --seed 42 --kmeans_clusters 8

	echo "[DONE] $NAME"
done

echo "All ablations finished."
```
