# 代码审查与重构记录

## 2026-04-07 重大重构（sync from laptop A1000）

本次重构对项目进行了全面升级，涉及模型架构、数据管道和评估流程。

---

### 重构内容

#### ✅ `build_dataset.py` — 重构为 CLI 脚本

| 变更 | 说明 |
|---|---|
| 添加 `argparse` CLI | 所有路径和参数均可命令行配置，不再有硬编码路径 |
| 新增 `align_tracks()` | 自车与前车按时间步对齐，确保序列长度一致 |
| 新增 `get_ego_speeds()` | 提取自车速度序列，用于场景过滤 |
| 新增 `assign_split()` | MD5 哈希确定性划分 train/val/test，无随机性、可重复 |
| 新增 `write_summary()` | 自动写出 `summary.txt` 和 `summary.csv` 统计摘要 |
| 输出从 2 个增至 4 个 | 新增 `meta.npy`（scenario/track 元数据）和 `split.npy`（划分标签） |
| 特征维度 19D → 20D | 新增 `ego_speed_mean` 特征 |
| 速度过滤 | `--min_ego_speed`（默认 5.5 m/s）过滤低速/静止场景 |

#### ✅ 新增 `dataset.py`

- `TrajFeatureDataset`：支持变长轨迹，读取 traj/feat/split，按 split key 索引
- `precompute_knn_pairs()`：基于特征距离预计算 KNN 正负样本对（cosine 或 L2）
- `collate_variable_traj()`：将变长轨迹 pad 并返回 `lengths`，供 GRU pack 使用

#### ✅ 新增 `model.py` — `TrajectoryEncoder`

旧架构（`TrajEncoder` LSTM + `FeatureEncoder` + `projector`）替换为：
```
GRU(input=4, hidden=128) → h[-1] → Linear(128,128) → ReLU → Linear(128,64) → L2 normalize
```
- 仅编码轨迹，特征仅作为监督信号，不进入网络前向路径
- 输出 64D L2 归一化 embedding

#### ✅ 新增 `loss.py` — `SoftContrastiveLoss`

- 特征引导软对比损失：批内所有对按特征相似度计算软标签权重
- `multi_positive_infonce()`：支持多正样本的 SupCon 风格 InfoNCE
- 不依赖硬正负对挖掘，避免 pair 质量问题

#### ✅ `train_embedding.py` — 适配新架构

- 使用 `TrajFeatureDataset` + `collate_variable_traj` 加载变长轨迹
- 使用 `split.npy` 划分 train/val/test，不再自行划分
- 使用 `SoftContrastiveLoss` 替代旧对比损失
- 保存 `best_model.pth`（含完整 checkpoint dict）和 `model_final.pth`

#### ✅ 新增 `export_embeddings.py`

- 对全量 `traj.npy` 运行推理，输出 `embeddings_all.npy (N, 64)`
- 保持样本顺序与 traj/feat/split 完全对齐

#### ✅ 新增 `evaluate_embedding.py`

- UMAP 可视化（按各维度特征着色）
- Ridge 线性探针（R² + Spearman 相关系数）
- 邻域一致性分析（embedding 邻域 vs. 特征邻域 vs. 随机基线）

#### ✅ 移除旧脚本

| 移除文件 | 替代方案 |
|---|---|
| `generate_embeddings.py` | `export_embeddings.py` |
| `visualize_umap.py` | `evaluate_embedding.py` |
| `analyze_style_embedding.py` | `evaluate_embedding.py` |
| `analysis/umap_feature_coloring.py` | `evaluate_embedding.py` |
| `scripts/umap_analysis.py` | `evaluate_embedding.py` |
| `docs/umap_validation_checklist.md` | `README.md` 和 `QUICK_REFERENCE.md` |

---

## 2026-03-23 早期修复记录

| 级别 | 文件 | 问题 | 状态 |
|---|---|---|---|
| 严重 | `generate_embeddings.py` | `feat_dim` 硬编码为 19，与训练不一致 | ✅ 已在本次重构中随文件一起移除 |
| 中度 | `build_dataset.py` | 特征维度注释 20D 与实际 19D 不符 | ✅ 已修复（当前实际为 20D） |
| 中度 | `build_dataset.py` | 数据路径硬编码 | ✅ 已改为 `--tfrecord_glob` CLI 参数 |
| 低度 | `build_dataset.py` | `reaction_time` 默认值处理 | ✅ 保持 `0.0` 作为默认值 |
| 低度 | `visualize_umap.py` | 非交互后端下调用 `plt.show()` | ✅ 已随文件一起移除 |
