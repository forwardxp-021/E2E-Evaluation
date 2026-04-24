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

## 合成策略 Rollout（generate_policy_rollouts.py）

### lateral_stable 可分性调优
`lateral_stable` 策略被设计为**第三种独立风格**（"横向稳定 + 舒适但不保守"），与 `conservative`（大间距/低动态）和 `aggressive`（小间距/高动态）在 embedding 空间中形成明显区分。关键设计：
- **thw_target = 1.4 s**：处于 conservative (2.5 s) 和 aggressive (1.0 s) 之间，但纵向动态更软
- **jerk_limit = 0.35 m/s²/step**：比 conservative (0.5) 更软，避免纵向特征与之重合
- **yaw_rate_clip = 0.02 rad/step**：适度的横向约束，在 embedding 中保留横向信号
- **heading_smooth_alpha = 0.45**：中等 EMA 平滑，与 conservative (0.0) 有明显差异

### generate_policy_rollouts.py 参数
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--src_traj_path` | `output/traj.npy` | 源轨迹文件 |
| `--src_front_path` | `output/front.npy` | 源前车轨迹文件 |
| `--src_split_path` | `None` | 源 split 文件（可选） |
| `--src_meta_path` | `None` | 源 meta 文件（可选） |
| `--output_dir` | `output_policy_rollouts` | 输出目录 |
| `--policies` | `conservative,aggressive,lateral_stable` | 要生成的策略列表 |
| `--dt` | `0.1` | 时间步长 (s) |
| `--seed` | `42` | 随机种子 |
| `--conservative_yaw_rate_clip` | `None` | 覆盖 conservative 的 yaw_rate_clip |
| `--aggressive_yaw_rate_clip` | `None` | 覆盖 aggressive 的 yaw_rate_clip |
| `--lateral_stable_yaw_rate_clip` | `None` | 覆盖 lateral_stable 的 yaw_rate_clip（默认 0.02） |
| `--heading_smooth_alpha` | `None` | 覆盖 lateral_stable 的 heading EMA 平滑系数（默认 0.45） |
| `--lateral_stable_thw_target` | `None` | 覆盖 lateral_stable 的 thw_target（默认 1.4 s） |
| `--lateral_stable_jerk_limit` | `None` | 覆盖 lateral_stable 的 jerk_limit（默认 0.35） |
| `--lateral_stable_a_max` | `None` | 覆盖 lateral_stable 的 a_max（默认 1.5 m/s²） |
| `--lateral_stable_a_min` | `None` | 覆盖 lateral_stable 的 a_min（默认 -2.8 m/s²） |

### 基本生成命令
```bash
python generate_policy_rollouts.py \
    --src_traj_path  output/traj.npy \
    --src_front_path output/front.npy \
    --src_split_path output/split.npy \
    --src_meta_path  output/meta.npy \
    --output_dir     output_policy_rollouts
```

### 参数扫描示例（无需修改代码）
```bash
# 调整 lateral_stable 纵向参数以提升可分性
python generate_policy_rollouts.py \
    --src_traj_path  output/traj.npy \
    --src_front_path output/front.npy \
    --output_dir     output_policy_rollouts_sweep1 \
    --lateral_stable_thw_target 1.2 \
    --lateral_stable_jerk_limit 0.25 \
    --lateral_stable_a_max 1.3 \
    --lateral_stable_a_min -2.5 \
    --lateral_stable_yaw_rate_clip 0.02 \
    --heading_smooth_alpha 0.45
```
生成后对比 `Per-policy active parameters` 摘要输出，确认参数已生效，并观察 `yaw_rate|abs|p95` 变化。

### 冒烟测试
```bash
python scripts/smoke_test_policy_rollouts.py
```
验证输出形状正确且 `lateral_stable` 的 `yaw_rate_p95` 与 `aggressive` 有显著差异。



### Q: 如何修改特征维度?
A: 修改 `build_dataset.py` 中 `compute_features()` 函数的返回列表（当前 20D）

### Q: 如何修改 embedding 维度?
A: 通过 `train_embedding.py --emb_dim <N>` 指定，`export_embeddings.py --emb_dim <N>` 保持一致

### Q: 如何只处理少量文件调试?
A: 使用 `--limit_files 5` 参数限制读取文件数

### Q: 数据集划分如何保证可重复?
A: `assign_split()` 使用 `scenario_id` 的 MD5 哈希值确定性划分，无随机性

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
