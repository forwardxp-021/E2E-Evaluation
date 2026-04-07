# E2E-Evaluation

E2E Evaluation research in automated driving — learning driving-style representations
from trajectory data (Waymo Open Dataset).

---

## 项目概述

本项目从 Waymo Open Dataset 轨迹数据中学习**驾驶风格表示（embedding）**，判断是否编码了激进/保守/平滑/跟车等驾驶风格信息。

核心流程：
1. **数据构建** — 从原始 TFRecord 提取自车+前车对齐轨迹，计算 20 维驾驶风格特征，按 MD5 哈希确定性地划分 train/val/test
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
    --feat_path output/feat.npy \
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
    --feat_path output/feat.npy \
    --split_path output/split.npy \
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
| `output/feat.npy` | `(N, 20)` float32，标准化后的驾驶风格特征 |
| `output/meta.npy` | `(N, 3)` 对象数组 `(scenario_id, ego_idx, front_id)` |
| `output/split.npy` | `(N,)` 字符串数组 `"train"/"val"/"test"`（MD5 确定性划分） |
| `output/summary.txt` | 数据集统计摘要 |
| `output/summary.csv` | 同上（CSV 格式） |

### 特征维度（20D）

| 维度 | 特征名 | 含义 |
|---|---|---|
| 0 | `rel_speed_mean` | 相对速度均值（激进程度） |
| 1 | `rel_speed_std` | 相对速度标准差 |
| 2 | `rel_speed_pos_frac` | 相对速度为正的比例 |
| 3 | `thw_mean` | 时距均值（跟车习惯） |
| 4 | `thw_std` | 时距标准差 |
| 5 | `thw_min` | 时距最小值 |
| 6 | `jerk_mean` | Jerk 均值 |
| 7 | `jerk_std` | Jerk 标准差 |
| 8 | `jerk_p95` | Jerk 95 分位（平滑性） |
| 9 | `rel_acc_mean` | 相对加速度均值 |
| 10 | `rel_acc_std` | 相对加速度标准差 |
| 11 | `reaction_time` | 制动反应时间（帧数） |
| 12 | `yaw_rate_std` | 横摆角速率标准差 |
| 13 | `lane_change_count` | 变道次数 |
| 14 | `lane_change_duration` | 变道时长比例 |
| 15 | `speed_norm_mean` | 归一化速度均值（速度偏好） |
| 16 | `speed_norm_std` | 归一化速度标准差 |
| 17 | `ego_speed_std` | 自车速度标准差（稳定性） |
| 18 | `ego_acc_std` | 自车加速度标准差 |
| 19 | `ego_speed_mean` | 自车速度均值 |

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
