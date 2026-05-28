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

### Stage 5B 训练性能/内存排查（GPU 利用率低）
- GPU 利用率低通常不是模型太慢，而是 **DataLoader / CPU / 内存瓶颈**。
- 优先使用 `mmap` 方式加载 shard（避免一次性将大 `.npy` 完整读入 RAM）。
- 建议先从 `batch_size=64` 开始稳定跑通，再逐步调大。
- 可先尝试：`--num_workers 2 --pin_memory` 提升主机到 GPU 的喂数效率。
- 如果系统 RAM 已经很高，优先保持 `--num_workers 0 --cache_shards 1`，降低并发加载压力。
- 训练日志出现 `Killed` 通常是 **系统 RAM OOM**，而不是 CUDA OOM。

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
- **yaw_rate_clip = 0.02 rad/step**：per-step heading delta clip，适度的横向约束，在 embedding 中保留横向信号
- **heading_smooth_alpha = 0.45**：EMA 平滑系数，对 desired heading 做指数移动平均
  - `0.0` = 不平滑（默认用于 conservative / aggressive），heading 直接跟随 source
  - 接近 `1.0` = 更强平滑 / 更慢更新（heading 变化极平缓，几乎不跟随 source 转向）
  - `0.45` = 中等平滑，与 conservative (0.0) 有明显差异，在 embedding 中形成独立横向风格

> **注意**：`heading_smooth_alpha` 越大，lateral_stable 的横向 heading 变化越缓慢，风格越"稳"；
> 但过大（>0.8）会导致轨迹偏移源路径，不推荐。

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

## Aligned 评估工作流（evaluate_policy_separation_aligned.py）

### 最小复现命令（生成 → 训练 → aligned eval）

```bash
# Step 1: 生成 policy rollouts
python generate_policy_rollouts.py \
    --src_traj_path  output/traj.npy \
    --src_front_path output/front.npy \
    --src_split_path output/split.npy \
    --src_meta_path  output/meta.npy \
    --output_dir     output_policy_rollouts

# Step 2: 训练 embedding（以 policy_rollouts 数据为训练集）
python train_embedding.py \
    --traj_path  output_policy_rollouts/traj.npy \
    --feat_path  output_policy_rollouts/feat.npy \
    --split_path output_policy_rollouts/split.npy \
    --output_dir output_policy_rollouts/run_demo

# Step 3: 导出全量 embedding
python export_embeddings.py \
    --traj_path      output_policy_rollouts/traj.npy \
    --checkpoint     output_policy_rollouts/run_demo/best_model.pth \
    --output_path    output_policy_rollouts/run_demo/embeddings_all.npy

# Step 4: Aligned 评估
python evaluate_policy_separation_aligned.py \
    --embeddings_path   output_policy_rollouts/run_demo/embeddings_all.npy \
    --policy_id_path    output_policy_rollouts/policy_id.npy \
    --source_index_path output_policy_rollouts/source_index.npy \
    --split_path        output_policy_rollouts/split.npy \
    --eval_split        test \
    --analysis_dir      output_policy_rollouts/run_demo/analysis_aligned
```

### 如何用 aligned 指标验证 policy separation

`evaluate_policy_separation_aligned.py` 输出 `policy_separation_aligned_summary.json`，
关键指标解读如下：

#### (b) Within-source pairwise distances — 检查 p0_vs_p2 距离是否被拉开

```
"p0_vs_p2": {"euclidean_mean": ...}   ← lateral_stable (p2) vs conservative (p0)
"p0_vs_p1": {"euclidean_mean": ...}   ← conservative vs aggressive (应最大)
"p1_vs_p2": {"euclidean_mean": ...}   ← aggressive vs lateral_stable
```

**验证指标（generator 方向正确的信号）**：
- `p0_vs_p1` 应最大（保守 vs 激进，风格差异最大）
- `p0_vs_p2` < `p0_vs_p1` 但 > 0（p2 与 p0 有差异，说明 lateral_stable 已与 conservative 分离）
- `p0_vs_p2` 变化趋势：随着 `yaw_rate_clip` 降低或 `heading_smooth_alpha` 增大，该距离应增大

#### (c) Within-source centroid classification accuracy — 检查 policy_2 准确率

```
"centroid_classification": {
    "accuracy": ...            ← overall, 应远高于 chance (0.3333)
    "per_policy_accuracy": {
        "0": ...,              ← conservative
        "1": ...,              ← aggressive
        "2": ...               ← lateral_stable ← 重点观察
    }
}
```

**验证指标**：
- overall accuracy > 0.60（明显高于 chance=0.3333 = good）
- `policy_2` accuracy 提升是 lateral_stable 可分性改善的直接信号
- 若 `policy_2` 准确率提升但 `policy_0` 下降，说明 lateral_stable 在往 conservative 方向漂移

#### (d) Within-source retrieval applicability + margin

```
"within_source_retrieval": {
    "retrieval_mode": "within_source",
    "retrieval_applicable": false,
    "retrieval_reason": "... one sample per policy ...",
    "nearest_neighbor_hit_rate": null,
    "nearest_neighbor_chance": null,
    "mean_within_source_margin": ...      ← 应 > 0
}
```

**说明**：
- within-source 每个 source 通常只有 1 个样本/policy，因此“same-policy 最近邻命中率”在定义上可能无效。
- 当无有效 same-policy 正样本可检索时，summary.json 会明确记录 `retrieval_applicable=false`，避免误导性的 `0.0000`。
- 此时应重点看：`pairwise_distances`、`centroid_classification`、`mean_within_source_margin`。

### 冒烟测试（aligned retrieval 回归测试）
```bash
python scripts/smoke_test_aligned_retrieval.py
```
验证 coverage 的 missing/duplicate 统计、以及 within-source 检索在无正样本时会被标记为 N/A。



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

## 最近更新 (2026-04-27)

✅ 修复 `evaluate_policy_separation_aligned.py`：within-source NN 检索在无正样本时改为显式 N/A  
　　→ 原因：within-source 每 policy 只有 1 个样本时，same-policy 最近邻定义无效  
　　→ 修复：summary.json 记录 `retrieval_mode/retrieval_applicable/retrieval_reason`，并将 hit_rate/chance 置 `null`  
✅ 新增 `scripts/smoke_test_aligned_retrieval.py`：防止 false-0.0 回归测试  
✅ 更新 `QUICK_REFERENCE.md`：  
　　→ 补充 `heading_smooth_alpha` 含义（0.0=不平滑，接近 1.0=更强平滑/更慢更新）  
　　→ 新增 aligned 评估工作流与 policy separation 验证指南（p0_vs_p2 距离、policy_2 centroid accuracy）
✅ 新增 `tools/embedding_retrieval_demo.py`：embedding 可解释性 demo（检索 + 轨迹回放）  
✅ 新增 `scripts/smoke_test_retrieval_demo.py`：检索 demo 单元/冒烟测试

## Embedding 可解释性 Demo（tools/embedding_retrieval_demo.py）

### 功能

给定一个 query 窗口（by `--query_index` 或 `--query_scenario_id`），脚本：
1. 在 embedding 空间检索 Top-K 最相似窗口（euclidean 或 cosine 距离）
2. 将 ego + front 轨迹在 query 初始坐标系下对齐叠加，输出 `traj_overlay.png`
3. 输出风格信号时序图（speed / accel / jerk / curvature proxy），输出 `timeseries.png`

### 检索模式

| 模式 | 说明 |
|------|------|
| `--mode global` | 在选定 split 的所有样本中检索 |
| `--mode within-source` | 仅检索与 query 共享相同 meta-key `(scenario_id, start, window_len, front_id)` 的其他行 |

> **within-source 限制**：基础数据集中没有显式 `policy_id` 字段。within-source 模式按
> meta-key 分组，把同组所有行（可能是不同 policy 的 rollout）全部绘出。若需要严格
> per-policy 标签，请用 `generate_policy_rollouts.py` 生成的 `policy_id.npy` 并搭配
> `evaluate_policy_separation_aligned.py`。

### 常用命令

```bash
# Global 检索（test split，默认 Top-5）
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 --topk 5 --mode global

# Within-source 检索
python tools/embedding_retrieval_demo.py \
    --emb_path   output_policy_rollouts/feat_style.npy \
    --meta_path  output_policy_rollouts/meta.npy \
    --traj_path  output_policy_rollouts/traj.npy \
    --front_path output_policy_rollouts/front.npy \
    --split_path output_policy_rollouts/split.npy \
    --query_index 0 --mode within-source

# Smoke test（无需数据文件）
python tools/embedding_retrieval_demo.py --smoke_test

# 单元测试
python scripts/smoke_test_retrieval_demo.py
```

### 输出文件（outputs/<run_id>/）

| 文件 | 说明 |
|------|------|
| `retrieval_table.csv` | Top-K 结果（index、meta 字段、distance、excluded 标记） |
| `traj_overlay.png` | 对齐后的 ego + front 轨迹叠加图 |
| `timeseries.png` | speed / accel / jerk / curvature proxy 时序对比图 |
| `summary.json` | 运行参数（mode、distance、topk、数据路径等） |

## Embedding interpretability demo

新增脚本：`tools/embedding_interpretability_demo.py`，用于**可解释可视化**，不修改 benchmark 指标定义。

### 1) Same-source triplet demo
在同一 `source_key = scenario_id|start|window_len|front_id` 下，比较不同 policy 的轨迹与信号，展示受控条件下风格分离。

### 2) Global retrieval demo
给定 query，执行跨 source 的 Top-K 检索，输出卡片图与信号对比，观察 embedding 邻居是否呈现相近驾驶风格。

### 3) within-source 与 global 区别
- `within-source`：受控对比（同源窗口）
- `global`：跨源检索（风格相似性）

### 4) 风格信号定义（轨迹级）
- `speed = sqrt(vx^2 + vy^2)`
- `accel = d(speed)/dt`
- `jerk = d(accel)/dt`
- `yaw_rate_proxy = d(unwrap(atan2(vy,vx)))/dt`
- `curvature_proxy = yaw_rate_proxy / max(speed, eps)`
- 若有 `front.npy`：`gap` 与 `thw = gap/max(speed,eps)`

### 5) 限制说明
- `yaw_rate_proxy/curvature_proxy` 来自速度方向估计，是近似 proxy
- demo 需要多 policy rollout 数据（同 source 至少 3 条记录）才能稳定展示 p0/p1/p2 对比
- 若 `policy_id` 缺失，`summary.json` 会写明 `policy_id_source=unavailable`，并将 same-policy hit@k 置为 `null`
- 此时会给出强提醒：global retrieval 仅能展示最近邻，不能验证 same-policy style retrieval
- 跨 source 轨迹叠加仅用于风格参考，不代表同一场景几何对齐

### 运行示例
```bash
python tools/embedding_interpretability_demo.py \
  --data_dir output_policy_rollouts \
  --out_dir outputs/embedding_demo/case_000 \
  --embedding feat_style \
  --split test \
  --query_index 0 \
  --mode both \
  --projection both \
  --case_selection best_hit_at_k \
  --distance euclidean \
  --topk 5 \
  --source_key_fields scenario_id,start,window_len,front_id \
  --auto_select_valid_source \
  --exclude_same_source \
  --exclude_same_scenario
```

> 若 rollout 的 `front_id` 在不同 policy 下不一致，可改用：  
> `--source_key_fields scenario_id,start,window_len`

### summary.json 诊断字段（重点看）
- `diagnostics.n_total_rows` / `n_rows_after_split`
- `diagnostics.n_unique_source_keys_total` / `n_unique_source_keys_after_split`
- `diagnostics.source_group_size_histogram_total` / `..._after_split`
- `diagnostics.has_policy_id` / `policy_id_source` / `policy_id_counts`
- `diagnostics.split_array_shape` / `embedding_shape` / `meta_shape` / `traj_shape` / `front_shape`

这组字段可直接判断：
1) split 是否把每个 source 只保留成单条（导致 within-source 失效）  
2) 当前数据是否包含可用 `policy_id`（或至少可恢复）  
3) 是否满足“每 source ≥3 条”的可解释 triplet 前提

### Smoke test
```bash
python tools/embedding_interpretability_demo.py \
  --out_dir outputs/embedding_demo/smoke \
  --smoke_test
```

### 关键输出文件
- `summary.json`：含 `diagnostics`（group histogram / policy_id 可用性 / shape）
- `embedding_2d_projection.png` + `embedding_2d_projection.csv`：PCA 2D 投影（仅可视化；query 星标 + Top-K 红圈 + rank）
- `embedding_2d_projection_umap.png` + `.csv`：`--projection umap|both` 且安装 `umap-learn` 时输出（仅可视化）
- `embedding_distance_matrix.png` + `embedding_distance_matrix.csv`：同源 embedding 距离矩阵（图中含数值标注）
- `within_source_triplet.png` / `within_source_style_signals.png` / `within_source_style_fingerprint_kinematic.png` / `within_source_style_fingerprint_dynamics.png` / `within_source_style_fingerprint_normalized.png` / `within_source_style_fingerprint.csv`：同源 policy 对比与风格统计
- `global_retrieval_cards.png` / `global_retrieval_style_signals.png` / `retrieval_table.csv` / `style_fingerprint.csv`：跨源 Top-K 检索解释
- `interpretability_report.md`：自动文本报告（query、同源距离、Top-K、hit@1/hit@k、局限性）

解释建议：
- PCA/UMAP 是降维可视化，不能替代高维 embedding 距离与 aligned evaluator 指标。
- 2D 上不出现完美三团，并不意味着高维空间没有有效分离。
- policy-level 解释依赖 `policy_id/policy_name/source_index` 元数据完整性。

## Experiment 2: lateral_stable Ablation Sweep

### 一键运行（debug）
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir output \
  --base_output_dir outputs/ablation_debug \
  --max_sources 100 \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable
```

### 常用参数
- `--dry_run`：仅打印命令与生效参数，不执行。
- `--skip_generation`：只跑评估（复用已生成 rollouts）。
- `--skip_evaluation`：只生成 rollouts。
- `--embedding {feat_style,feat_style_raw,feat,feat_legacy}`
- `--split {train,val,test}`
- `--distance {euclidean,cosine}`
- `--topk INT`
- `--configs a,b,c`（按名称选择消融子集）

## Experiment 2: Lateral_stable Ablation and Parameter Sweep

- **Purpose**: Test which lateral_stable controls improve p2 independence while keeping comfort/stability metrics acceptable.
- **Script**: `tools/run_lateral_stable_ablation.py`
- **Required inputs**: `--source_data_dir` with `traj.npy`, `front.npy` (plus `split.npy` / `meta.npy` if available).

### Debug command
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

### Dry-run command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_debug \
  --configs baseline_current,no_lateral_smoothing,lateral_only,comfort_only,full_strong_lateral_stable \
  --dry_run
```

### Full command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/ablation_full \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5
```

### Main outputs
- `ablation_summary.csv`, `ablation_summary.json`
- `ablation_recommendation.json`
- `ablation_report.md`
- `ablation_p2_separation_margin.png`
- `ablation_p2_farthest_rate.png`
- `ablation_pairwise_distances.png`
- `ablation_retrieval_classification.png`
- `ablation_p2_style_metrics.png`
- `ablation_tradeoff_plot.png`
- per-config `population_eval/`

### Interpretation
- Higher `p2_farthest_rate` is better.
- `mean_p2_separation_margin > 0` means p2 is a stronger independent mode.
- Lower `p2_rms_yaw_rate_proxy_mean` means stronger lateral stability.
- Lower `p2_rms_jerk_mean` means smoother comfort.
- Retrieval/centroid metrics measure style discriminability.

### Limitations
- Synthetic policies (not human labels).
- Replayed front-vehicle setup (not full closed-loop multi-agent simulation).
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

## Experiment 2B: Local Fine-Grained Sweep Around full_strong_lateral_stable

### Motivation
Run a focused local sweep around `full_strong_lateral_stable` to improve p2 separation while preserving comfort and lateral stability.

### Script usage
`python tools/run_lateral_stable_ablation.py --config_set local_fine ...`

### Dry run
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_debug \
  --config_set local_fine \
  --dry_run
```

### Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_debug \
  --config_set local_fine \
  --max_sources 100 \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Full command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/local_sweep_full \
  --config_set local_fine \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Output files
- `local_sweep_summary.csv`, `local_sweep_summary.json`
- `local_sweep_recommendation.json`, `local_sweep_report.md`
- `local_sweep_integrity_report.json`, `local_sweep_rollout_sanity.csv`
- `local_sweep_p2_separation_margin.png`, `local_sweep_p2_farthest_rate.png`
- `local_sweep_pairwise_distances.png`, `local_sweep_retrieval_classification.png`
- `local_sweep_p2_style_metrics.png`, `local_sweep_tradeoff_yaw_vs_margin.png`
- `local_sweep_tradeoff_jerk_vs_margin.png`, `local_sweep_delta_vs_center.png`

### Interpretation
Broad ablation compares families; local sweep tests nearby parameter perturbations around the best broad config. If separation margin remains negative, conclude: **p2 independence improved but remains incomplete**.

### Limitations
No public data validation yet.

## Experiment 2C: recommended_lateral_stable_v2 Final Comparison

### Experiment 2B result
Local fine sweep selected `recommended_lateral_stable_v2` (`yaw_008_jerk_020`).

### Recommended lateral_stable v2 parameters
- `heading_smooth_alpha = 0.75`
- `yaw_rate_clip = 0.008`
- `thw_target = 1.70`
- `jerk_limit = 0.200`
- `a_max = 1.275`
- `a_min = -2.52`

### Final comparison command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/final_lateral_stable_v2 \
  --config_set final_compare \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Debug command
```bash
python tools/run_lateral_stable_ablation.py \
  --source_data_dir <SOURCE_DATA_DIR> \
  --base_output_dir outputs/final_lateral_stable_v2_debug \
  --config_set final_compare \
  --max_sources 100 \
  --embedding feat_style \
  --split test \
  --distance euclidean \
  --topk 5 \
  --overwrite
```

### Expected outputs
- `final_config_comparison_summary.csv`
- `final_config_comparison_summary.json`
- `final_config_comparison_report.md`
- `final_config_p2_separation.png`
- `final_config_margin.png`
- `final_config_classification_retrieval.png`
- `final_config_style_metrics.png`
- `final_config_tradeoff.png`
- `ablation_integrity_report.json`

### How to interpret
- `p2_farthest_rate` higher is better.
- `mean_p2_separation_margin` closer to or above 0 is better.
- `centroid_accuracy_p2` measures p2 recognizability.
- `p2_rms_jerk` lower means smoother longitudinal behavior.
- `p2_rms_yaw_rate_proxy` lower means stronger lateral stability.
- negative `mean_p2_separation_margin` means p2 is not yet fully independent.

### Limitations
- Synthetic policy rollout only.
- Replayed front vehicle.
- No real human driver labels yet.
- No sensor rendering / perception stack.
- PCA / UMAP are visualization only.


## Phase 4A: Public Human Trajectory External Validation Scaffold

Purpose: validate whether embedding structure transfers beyond synthetic generator artifacts using trajectory-level weak-label evaluation.

### Unified input format
`traj.npy`, optional `front.npy`, `meta.npy`, `split.npy`, `feat_style.npy`, optional `feat_style_raw.npy`, optional `feature_names_style.json`, optional `embeddings.npy`.

### Pseudo-label assignment
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir <HUMAN_DATA_DIR> \
  --out_dir outputs/vehicledata_validation/pseudo_labels \
  --label_mode percentile \
  --target_quantile 0.25 \
  --dt 0.1
```

### Evaluation
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir <HUMAN_DATA_DIR> \
  --label_dir outputs/vehicledata_validation/pseudo_labels \
  --out_dir outputs/vehicledata_validation/eval \
  --embedding_path <OPTIONAL_EMBEDDING_PATH> \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --projection pca
```

Baselines-only mode:
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir <HUMAN_DATA_DIR> \
  --label_dir outputs/vehicledata_validation/pseudo_labels \
  --out_dir outputs/vehicledata_validation/eval_baselines_only \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --projection pca
```

### Outputs
Pseudo-label outputs include summary/report/distribution files. Evaluation outputs include `human_validation_summary.json`, `human_validation_report.md`, `baseline_comparison_summary.csv`, retrieval/classification/correlation/cluster artifacts and figures.

### Interpretation and limitations
Pseudo labels are rule-based weak labels (not ground truth) for external validation only. Label-defining features can leak into classification metrics, so retrieval, cluster fingerprints, and baseline comparisons must be interpreted jointly.

### Smoke tests
Both scripts support `--smoke_test` and generate synthetic arrays locally without external dataset downloads.

- `evaluate_vehicledata_validation.py` new flags: `--allow_skip_learned`, `--retrieval_mode strict`, exclusion flags.
- Learned embedding mismatches now fail by default; optional skip records warnings and marks `learned_embedding_evaluated=false`.
- Retrieval outputs now include chance/lift metrics and strict anti-leakage behavior.
- Expected outputs: baseline_* plots, cluster_size_distribution.png, cluster_style_fingerprint.png/csv, cluster_label_distribution.csv.

## Embedding alignment requirement

- 评估阶段的 `traj/meta/feat_style/pseudo_label` 是 row-level 数组，learned embedding 必须同样 row-level。
- `embedding.shape[0]` 必须等于样本行数 `N`。
- source-level embedding 默认禁止自动扩展；仅可在 `--allow_source_level_embedding_expansion` 下用于调试，并会标记 `learned_embedding_valid_for_policy_eval=false`。

`data1` 提示：
- `traj` = 33471 rows
- `embeddings` = 11157 rows
- 11157x3=33471，表示 source-level + 3 rollout/policy，不是 row-level learned embedding。

TODO（脚本占位）：
```bash
python tools/export_row_level_embeddings.py \
  --data_dir data1 \
  --model_ckpt <CHECKPOINT> \
  --out_path data1/embeddings_row_level.npy
```


## 阶段 4B：Waymo 真实人类轨迹数据提取

### 命令
```bash
python tools/build_waymo_human_trajectory_dataset.py \
  --waymo_dir <WAYMO_TFRECORD_DIR> \
  --out_dir outputs/waymo_human_v1 \
  --window_len 80 \
  --stride 20 \
  --min_speed 1.0 \
  --max_files 5 \
  --max_scenarios 200 \
  --max_agents_per_scenario 64 \
  --split_by_scenario \
  --overwrite

python tools/build_waymo_human_trajectory_dataset.py \
  --out_dir outputs/waymo_human_smoke \
  --smoke_test \
  --overwrite
```

后续 Stage 4C：
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir outputs/waymo_human_v1 \
  --out_dir outputs/waymo_human_v1/pseudo_labels \
  --label_mode percentile \
  --target_quantile 0.25 \
  --dt 0.1 \
  --dataset_type human_public

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1 \
  --label_dir outputs/waymo_human_v1/pseudo_labels \
  --out_dir outputs/waymo_human_v1/eval_baselines_only \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

### 期望行为
- 从原始 Waymo 场景中提取真实 human vehicle agent 的 observed trajectory window。
- 不调用 synthetic policy generator。
- 不生成 p0/p1/p2。
- 不生成 policy_id / policy_name。
- 输出统一格式 npy 文件。
- 每一行对应一个真实 human agent trajectory window。
- split 按 scenario_id hash 分配，避免同一 scenario 泄漏到不同 split。
- 自动计算 style features 和标准化特征。
- 自动生成 build_summary.json 和 build_report.md。

### 通过标准
- out_dir 下生成 traj.npy / front.npy / meta.npy / split.npy / feat_style.npy / feat_style_raw.npy / feature_names_style.json。
- meta.npy 中 dataset_type = human_public。
- meta.npy 中不包含 policy_id / policy_name。
- len(traj) == len(front) == len(meta) == len(split) == feat_style.shape[0]。
- build_summary.json 中 n_windows_kept > 0。
- split_counts 中 train/val/test 至少有一个非空，正式运行应三者都有数据。
- front_found_rate 被记录。
- feature_names_style.json 与 feat_style 的列数一致。
- smoke_test 可以不依赖真实 Waymo 数据运行成功。


## 阶段 4D：训练并导出 Waymo human row-level learned embedding

### 1. 命令
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --batch_size 1024 \
  --device cuda \
  --traj_nan_mode interpolate \
  --max_traj_nan_ratio 0.2 \
  --overwrite
```

### 2. 期望行为
- Waymo human `traj.npy` 可能包含 NaN，因为观测轨迹可能部分无效。
- `export_human_row_embeddings.py` 必须复用训练脚本相同的轨迹清洗与局部归一化逻辑。
- 导出的 embedding 必须与 `traj.npy` 行对齐（row-aligned）。
- 若 `normalize_local` 产生非有限值（NaN/Inf），必须立即失败，禁止保存坏 embedding。
- 若 checkpoint 训练过程中出现 NaN loss，禁止导出，必须先修复并重训。

### 3. 通过标准
- 控制台输出 raw/sanitized 的 NaN/Inf 统计。
- `embedding_export_summary.json` 与 `embedding_export_debug.json` 成功生成。
- `embeddings_row_level.npy` 全量 finite，且 `shape[0] == len(traj.npy)`。
- `row_aligned = true`（官方 Stage 4D 默认不允许 drop）。


## 阶段 4E：jerk/comfort-aware learned embedding 训练

### 命令
同 README 的三条命令（训练/导出/评估），并可追加：
```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned \
  --out_dir outputs/waymo_human_v1_full51/paper_tables
```

### 期望行为
- 训练保持 Stage 4D v1 可复现（uniform 默认）。
- jerk_comfort 模式重点提升舒适性相关差异建模。
- 输出可直接用于论文表格。

### 通过标准
- 评估摘要包含 learned_embedding_evaluated=true。
- style_distance_correlation.csv 含各指标 valid_pairs_* 列。
- human_validation_report.md 的 next steps 与 Stage 4D 已完成状态一致。

# Stage 4F：comfort-aware auxiliary regression

## 1. 命令

```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode uniform \
  --aux_regression \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --aux_loss_weight 0.2 \
  --aux_loss_type huber \
  --device cuda \
  --seed 42 \
  --overwrite

python tools/evaluate_aux_predictions.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/model.pt \
  --eval_split test \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --batch_size 1024 \
  --device cuda \
  --out_path outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/aux_prediction_metrics_test.json

python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_aux.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_aux.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca

python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
    stage4f_comfort_aux=outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e_stage4f
```

## 2. 期望行为
- 使用 train split 训练。
- 不使用 pseudo labels 训练。
- 在 soft contrastive loss 基础上增加 comfort auxiliary regression。
- evaluate_aux_predictions.py 是训练后、导出前的必做诊断，用于验证 auxiliary regression head 是否真的学到舒适性目标。
- evaluate_aux_predictions.py 与训练/导出共享同一套轨迹 NaN 清洗逻辑（sanitize + normalize_local），可处理 Waymo human traj.npy 的非有限值。
- 报告 rms_accel / rms_jerk / max_abs_accel / max_abs_jerk / mean_thw / min_thw 的 MAE / RMSE / Spearman。
- 该诊断独立于 embedding retrieval/classification 评估。
- 导出 row-aligned embedding。
- 在 test split 上评估 learned vs baselines。
- 与 Stage 4D / Stage 4E 对比。

## 3. 通过标准
- train_total_loss / val_total_loss finite。
- aux_loss finite。
- outputs/waymo_human_v1_full51/human_embedding_model_comfort_aux/aux_prediction_metrics_test.json 存在。
- aux_prediction_metrics_test.json 中 row_aligned=true。
- traj_nan_count_after_sanitize=0。
- aux_head_loaded=true。
- rms_jerk / max_abs_jerk 的 Spearman 为有限值（finite），或显式报告为 N/A（含原因与 warning）。
- 若 rms_jerk Spearman 近似 0，视为 Stage 4F 未学到 jerk（即使训练 loss 有限）。
- 若 aux prediction 指标良好但 embedding jerk correlation 未提升，记录为“aux head 学到但未转移到 embedding geometry”。
- embeddings_row_level_comfort_aux.npy shape = [168191, 64]。
- evaluation learned_embedding_evaluated=true。
- learned 的 classification/retrieval 明显高于 random。
- rms_jerk_delta correlation 相比 Stage 4D v1 有明显提升。
- 如果 jerk 未提升，报告中明确记录 Stage 4F 未达到目标。

# Stage 4G：comfort metric alignment

## 1. 命令

Training:

```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode uniform \
  --aux_regression \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --aux_loss_weight 0.2 \
  --aux_loss_type huber \
  --comfort_metric_alignment \
  --metric_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --metric_loss_weight 0.1 \
  --metric_loss_type mse \
  --metric_distance euclidean \
  --device cuda \
  --seed 42 \
  --overwrite
```

Aux prediction diagnostic:

```bash
python tools/evaluate_aux_predictions.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/model.pt \
  --eval_split test \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --batch_size 1024 \
  --device cuda \
  --out_path outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/aux_prediction_metrics_test.json
```

Export:

```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite
```

Evaluate:

```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

Compare Stage 4D / 4E / 4F / 4G:

```bash
python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
    stage4f_comfort_aux=outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
    stage4g_comfort_metric=outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e_stage4f_stage4g
```

```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4g_comfort_metric
```

## 2. 期望行为

- Stage 4G 在 Stage 4F 的 auxiliary regression 基础上，进一步增加 comfort metric alignment loss。
- auxiliary regression 让 embedding 中可解码 jerk/comfort 信息。
- comfort metric alignment 直接约束 embedding pairwise distance 与 comfort feature pairwise distance 对齐。
- 训练仍然只使用 train split。
- 不使用 pseudo labels 训练。
- pseudo labels 仅用于 test split evaluation。
- 导出的 embeddings_row_level_comfort_metric.npy 必须与 traj.npy 行对齐。
- 评估重点不是只看分类和检索，还要重点看 rms_jerk_delta correlation 是否高于 Stage 4D / 4F。
- Stage 4G 是为了让 embedding geometry 更 jerk/comfort-sensitive。

## 3. 通过标准

- train_total_loss / val_total_loss finite。
- train_metric_loss / val_metric_loss finite。
- aux_loss finite。
- embeddings_row_level_comfort_metric.npy.shape[0] == len(traj.npy) == 168191。
- embedding 全部 finite，无 NaN/Inf。
- human_validation_summary.json 中 learned_embedding_evaluated=true。
- learned 的 classification / retrieval 明显高于 random。
- rms_jerk_delta correlation 相比 Stage 4D v1 的约 0.0697 有明显提升。
- 目标：
  - rms_jerk_delta >= 0.15：初步有效
  - rms_jerk_delta >= 0.20：明显改善
  - rms_jerk_delta >= 0.30：非常理想
- 如果 rms_jerk_delta 未提升，但 aux prediction 仍然很好，说明 metric alignment 权重或距离形式还需要调整。
- 如果 classification/retrieval 接近 random，说明 metric alignment 过强导致 embedding 崩塌。
- compare_stage4d_stage4e_stage4f_stage4g/comparison_summary.csv 生成。

# Stage 4H：4G sanity check — shuffled comfort metric target

## 1. 命令

Training:
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode uniform \
  --aux_regression \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --aux_loss_weight 0.2 \
  --aux_loss_type huber \
  --comfort_metric_alignment \
  --metric_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --metric_loss_weight 0.1 \
  --metric_loss_type mse \
  --metric_distance euclidean \
  --metric_target_shuffle \
  --metric_target_shuffle_seed 42 \
  --device cuda \
  --seed 42 \
  --overwrite
```

Aux prediction diagnostic:
```bash
python tools/evaluate_aux_predictions.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled/model.pt \
  --eval_split test \
  --aux_targets rms_accel,rms_jerk,max_abs_accel,max_abs_jerk,mean_thw,min_thw \
  --batch_size 1024 \
  --device cuda \
  --out_path outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled/aux_prediction_metrics_test.json
```

Export:
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric_shuffled/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_shuffled.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite
```

Evaluation:
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric_shuffled \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_shuffled.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```

Comparison:
```bash
python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
    stage4f_comfort_aux=outputs/waymo_human_v1_full51/eval_with_learned_comfort_aux \
    stage4g_comfort_metric=outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
    stage4h_metric_shuffled=outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric_shuffled \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_to_stage4h
```

## 2. 期望行为

- Stage 4H 是 sanity check，不是主方法。
- 它使用与 Stage 4G 相同的模型和训练流程。
- 唯一区别是 comfort metric alignment 的 target 被打乱。
- 如果 Stage 4G 的提升是真实的，Stage 4H 的 jerk correlation 应该明显低于 Stage 4G。
- auxiliary regression target 不打乱。
- pseudo labels 仍然不用于训练。
- 这个实验用于排除代码 bug、随机偶然、以及无意义 metric alignment 也能提升的可能性。

## 3. 通过标准

- 训练 loss finite。
- 导出 embedding row-aligned。
- evaluation learned_embedding_evaluated=true。
- compare_stage4d_to_stage4h/comparison_summary.csv 生成。
- Stage 4H 的 rms_jerk_delta 应明显低于 Stage 4G。
- 如果 Stage 4H 仍然接近 Stage 4G 的 rms_jerk_delta，需要警惕 metric alignment 实现或评估存在泄漏/bug。
- Stage 4G 仍应作为主结果，Stage 4H 仅作为 sanity check。

# Stage 4I：最终结果固化与论文图表包

## 1. 命令

```bash
python tools/generate_stage4_final_report.py \
  --out_dir outputs/waymo_human_v1_full51/stage4_final_report
```

```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned_comfort_metric \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model_comfort_metric/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_comfort_metric_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4g_comfort_metric
```

## 2. 期望行为

- 汇总 Stage 4D/4E/4F/4G/4H 的结果。
- 生成最终 ablation 表格。
- 生成 Stage 4G learned vs baselines 表格。
- 生成 Stage 4G auxiliary prediction 表格。
- 生成 Stage 4H shuffled-target sanity check 表格。
- 生成论文可用的图和 Markdown 报告。
- 不启动新训练，不修改模型。

## 3. 通过标准

- stage4_final_report.md 存在且非空。
- table_stage4_ablation.md / .csv 存在。
- table_stage4g_learned_vs_baselines.md / .csv 存在。
- table_stage4g_aux_prediction.md / .csv 存在。
- Stage 4G auxiliary prediction 表格不能全是 NaN。
- table_stage4h_sanity_check.md / .csv 存在。
- table_stage4h_sanity_check.md 必须使用按指标定制的解释文本。
- 若 Stage 4H 的 yaw/curvature 高于 Stage 4G，报告不得错误宣称其“退化”。
- 报告必须明确：sanity check 的关键证据是 jerk collapse + retrieval/classification 回落。
- figure_stage4_style_correlation.png 存在。
- figure_stage4_jerk_delta.png 存在。
- stage4_final_numbers.json 存在。
- 报告明确写出 Stage 4G 是 current best。
- 报告明确写出 Stage 4H shuffled target 使 jerk improvement 消失。
- 报告明确写出限制：pseudo labels 是 weak labels，4G 是 metric-aligned embedding，不是纯无监督发现。


# Stage 5：interaction-aware input design

## 1. 命令

> 当前仅做设计评审，不新增训练命令。

```bash
python -m py_compile tools/train_human_behavior_embedding.py
grep -R "Stage 5" -n README.md QUICK_REFERENCE.md 07_stage5_interaction_design.md
```

## 2. 期望行为

- 本阶段是设计阶段，不启动训练。
- 明确 5-neighbor lane-aware 输入设计。
- 明确 weak supervision feature 分组。
- 明确 longitudinal / lateral / interaction 三个显性 head。
- 明确 flatten GRU 与 slot encoder 两种架构路线。
- 不覆盖 Stage 4G 结果。

## 3. 通过标准

- 07_stage5_interaction_design.md 存在。
- README.md 中能看到 Stage 5 的高层说明。
- QUICK_REFERENCE.md 中能看到 Stage 5 设计阶段说明。
- 文档明确说明 5 个 neighbor slot。
- 文档明确说明 heading 使用 raw heading 优先。
- 文档明确说明 longitudinal / lateral / interaction 三组 features。
- 文档明确说明三个 explicit heads。
- 文档明确说明 Version A flatten GRU 与 Version B slot encoder + attention。
- 文档明确说明本阶段不启动训练。


# Stage 5A：lane-aware 5-neighbor context 数据构建

## 1. 命令

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --out_dir outputs/waymo_5neighbor_context_smoke \
  --smoke_test \
  --overwrite
```

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite
```

## 2. 期望行为

- 从 Waymo 场景中提取 target vehicle 轨迹窗口。
- 为每个 target vehicle 分配 front / left_front / left_rear / right_front / right_rear 五个邻车 slot。
- 优先使用 lane-aware assignment。
- lane-aware 失败时 fallback 到 ego-centric geometric assignment。
- 输出 ego_seq / neighbor_seq / context_traj / context_mask / interaction features。
- 输出 slot coverage、fallback rate、heading fallback rate 等诊断信息。
- 不启动模型训练。
- 不覆盖 Stage 4G 输出。

## 3. 通过标准

- smoke test 能跑通。
- context_traj.npy / ego_seq.npy / neighbor_seq.npy / context_mask.npy 存在。
- interaction_feat_style.npy 和 interaction_feat_style_raw.npy 存在。
- build_summary.json 中包含 slot_valid_ratio、fallback_assignment_rate、heading_proxy_fallback_rate。
- neighbor_slot_valid_ratio.csv 存在。
- lane_assignment_debug.csv 存在。
- build_report.md 用中文说明数据规模、slot coverage、fallback 情况、限制。
- context_traj.npy 和 interaction_feat_style.npy 无 NaN/Inf。
- Stage 4 结果文件未被覆盖。

### Stage 5A 非有限值排查（补充）

若 Stage 5A 在非有限值断言/报错处失败：
- 检查 `nonfinite_debug_*.json`。
- 常见原因是：窗口整体 valid_ratio 达标，但内部仍包含 Waymo invalid 帧（`x/y/vx/vy/heading` 为 NaN）。
- 构建脚本应对该类帧执行插值/清洗（sanitize），并确保最终输出不含 NaN/Inf。

通过标准：
- `build_summary.json` 中 `trajectory_nan_count_after_sanitize = 0`。
- `context_traj.npy` 的 finite 检查为 `true`。
- `interaction_feat_style.npy` 的 finite 检查为 `true`。
- 成功构建时不应产生 `nonfinite_debug_*.json`。

# Stage 5A-v2：真正 lane-aware 5-neighbor assignment

## 1. 命令

Smoke test:

python tools/build_waymo_5neighbor_context_dataset.py \
  --out_dir outputs/waymo_5neighbor_context_laneaware_smoke \
  --smoke_test \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite

Small real data:

python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --overwrite

Geometric-only debug baseline:

python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_geometric_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode geometric_only \
  --overwrite

## 2. 期望行为

- 优先使用 Waymo map/lane 信息给 target vehicle 找 current lane、left lane、right lane。
- 根据 lane s 坐标分配 front / left_front / left_rear / right_front / right_rear。
- lane-aware 失败时才 fallback 到几何分配。
- 输出 lane projection / fallback / slot method 诊断。
- 不启动训练。

## 3. 通过标准

- smoke test 中 lane_assignment_success_rate > 0。
- real small data 中 lane_assignment_success_rate 不能为 0。
- fallback_assignment_rate 不能等于 1.0。
- build_summary.json 包含 lane projection 诊断字段。
- lane_assignment_debug.csv 包含 ego_lane_id / neighbor_lane_id / delta_s 等字段。
- 如果 fallback_assignment_rate > 0.5，需要暂停 Stage 5B 训练，先分析原因。
- context_traj.npy / interaction_feat_style.npy 无 NaN/Inf。
- Stage 4 结果不被覆盖。

## Stage 5A-v2 卡住排查（lane-aware，中文）

如果脚本看起来卡住：
- 先看进度条（TFRecord / scenario / target agents / windows）。
- 检查 `build_summary.json` 里的 `timing_seconds`，确认是否 `lane_projection` 占比过高。
- 检查 `lane_projection_avg_candidate_lanes` 是否过大。
- 降低 `--lane_topk_candidates`。
- 降低 `--lane_search_radius`。
- 临时切到 `--assignment_mode geometric_only` 做 baseline/debug。

### 小规模 lane-aware（保守投影限制）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --lane_search_radius 20 \
  --lane_topk_candidates 32 \
  --overwrite
```

### geometric-only 调试
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_geometric_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode geometric_only \
  --overwrite
```


## Stage 5A-v3（lane-aware + fallback）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_with_geometric_fallback \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --overwrite
```

## Stage 5A-v4（normal clean，保留 good + ambiguous_intersection）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --overwrite
```

## Stage 5A-v4（strict quality，仅保留 good）
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_strict_v1_small \
  --max_files 2 \
  --max_scenarios 50 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --drop_if_lane_context_ambiguous \
  --overwrite
```

期望行为（中文）：
- normal clean run 保留 good + ambiguous_intersection，丢弃 bad/fallback。
- strict run 仅保留 good。
- empty neighbor slots 不应自动触发 ambiguous_intersection。
- slot coverage 与 empty slot ratio 单独报告。

通过标准（中文）：clean run 完成；lane_context_quality_counts 合理；good_lane_context_rate 不应仅因 slot 为空而接近 0；empty_slot_ratio_by_slot 存在；assignment_method_counts_by_slot 每个 slot 总和等于 n_windows_kept；context_traj.npy 与 interaction_feat_style.npy 无 NaN/Inf。

### Stage 5A-v3 常见错误与排查（clean 模式）

常见错误：
1. `timing referenced before assignment`
   - 原因：`timing` 初始化太晚。
   - 修复：在 `main` 开始处（参数解析与 out_dir 创建后）立即初始化 `t_global` 和 `timing`。

2. `boolean index did not match`
   - 原因：clean filtering 后输出列表行数不一致（先 append 后过滤）。
   - 修复：先完成过滤判断，再原子化统一 append。

3. `CSV dict contains fields not in fieldnames`
   - 原因：debug row schema 不统一。
   - 修复：统一 `LANE_DEBUG_FIELDS`，写 CSV 时使用 `normalize_debug_row`。

4. `lane_assignment_success_rate > 1.0`（或 `current_lane_found_rate > 1.0`）
   - 原因：分子在 clean filtering 之前累计，分母用 `n_windows_kept`（clean filtering 之后），导致分母不一致。
   - 修复：summary 中拆分 pre-filter 与 kept 计数：
     - pre-filter：`lane_assignment_success_count_pre_filter`、`current_lane_found_count_pre_filter`、`left_lane_found_count_pre_filter`、`right_lane_found_count_pre_filter`。
     - kept：`lane_assignment_success_count_kept`、`current_lane_found_count_kept`、`left_lane_found_count_kept`、`right_lane_found_count_kept`、`fallback_assignment_count_kept`。
   - 主指标 rate 统一使用 kept 分母：`n_windows_kept`。
   - 额外输出 pre-filter rate：`lane_assignment_success_rate_pre_filter`、`current_lane_found_rate_pre_filter`。
   - 可选启用 `--strict_summary_validation`，当任一主指标 rate > 1.0 时直接报错。

通过标准：
- clean run 不报错。
- `split.npy / meta.npy / interaction_feat_style.npy / context_traj.npy` 行数一致。
- `assignment_method_counts_by_slot` 每个 slot 的总数等于 `n_windows_kept`。
- `lane_assignment_debug.csv` 可以正常写出。
- `build_summary.json` 包含 clean filtering drop counts。
- `lane_assignment_success_count_kept <= n_windows_kept` 且 `current_lane_found_count_kept <= n_windows_kept`。
- 主指标 rate（`lane_assignment_success_rate/current_lane_found_rate/left_lane_found_rate/right_lane_found_rate/fallback_assignment_rate`）均 `<=1.0`。

## Stage 5A full51 安全构建（流式分片，避免 OOM）

```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --waymo_dir /mnt/d/WMdata \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51 \
  --max_files 51 \
  --max_agents_per_scenario 64 \
  --window_len 80 \
  --stride 20 \
  --dt 0.1 \
  --min_valid_ratio 0.8 \
  --min_speed 1.0 \
  --agent_types vehicle \
  --assignment_mode lane_aware_only \
  --front_max_distance 120 \
  --side_front_max_distance 80 \
  --side_rear_max_distance 120 \
  --lane_lateral_tolerance 2.0 \
  --slot_heading_diff_deg 45 \
  --static_speed_threshold 0.5 \
  --drop_if_no_lane_map \
  --drop_if_ego_lane_missing \
  --drop_if_lane_context_bad \
  --drop_if_lane_context_ambiguous \
  --streaming \
  --output_shard_size 5000 \
  --overwrite
```

说明：
- 不要用非 streaming 模式跑 full51。
- 如果在 “Processing TFRecord files” 阶段被 killed，通常表示 scenario 被整体堆在内存里。
- 请使用 streaming 或 `--file_start/--file_end` 分段运行。

## Stage 5A-v5 故障排查（临时文件依赖）

如果出现 `FileNotFoundError: /tmp/old.py`：
- 说明代码错误依赖了 Codex 临时文件。
- 仓库代码不能依赖 `/tmp/old.py`。
- 需要运行 `python tools/check_no_tmp_dependencies.py`。
- 所有 helper function 必须在仓库源码内定义或从 `tools` 模块导入。

## Stage 5A-v5 故障排查（slot_method 接口不一致）

如果出现：
`AttributeError: 'SlotAssignResult' object has no attribute 'slot_method'`

原因：
build script 与 `lane_aware_assignment.py` 的接口不一致。

修复原则：
- 优先从 `assign.per_slot_debug` 推导每个 slot 的 `assignment_method`。
- 不要假设 `SlotAssignResult` 存在未定义字段。
- smoke test 必须覆盖 streaming mode。

通过标准：
- `python -m py_compile tools/build_waymo_5neighbor_context_dataset.py` 通过。
- `python tools/check_no_tmp_dependencies.py` 通过。
- smoke test 通过。
- full51 streaming 命令可启动，且不会出现 `/tmp/old.py` 的 FileNotFoundError。

## Stage 5A-v5 Streaming 排障（新增）

### 常见现象
- 只看到 `Processing TFRecord files: 0/51` 不动。

### 原因
- 只有外层文件进度条，没有内部 scenario 进度；
- 或第一个 TFRecord 内部处理耗时较长。

### 修复
- streaming 模式必须显示 scenario 级别进度或每 N 个 scenario 的 heartbeat；
- full51 前必须先跑 `max_scenarios=10` 和 `max_scenarios=50` 的 streaming debug 命令。

### 通过标准
- `max_scenarios=10` 能快速完成；
- `max_scenarios=50` 能生成 shard；
- `build_summary.json` 保留完整诊断字段；
- `row_index` 不重复；
- full51 不再一次性缓存所有 scenario；
- 不出现 `/tmp/old.py`；
- 不出现 `SlotAssignResult` 接口错误。

## Stage 5A：并行分片结果合并

命令（示例）：

```bash
python tools/merge_waymo_5neighbor_context_shards.py \
  --input_roots \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_13_26 \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_26_39 \
    outputs/waymo_5neighbor_context_laneaware_clean_v1_part_39_51 \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --recompute_global_standardization \
  --overwrite
```

预期行为：
- 以 manifest/统计信息方式合并 Stage 5A 并行分片输出；
- 默认不在 merged root 生成 `ego_seq.npy`、`neighbor_seq.npy`、`context_traj.npy` 等超大单体文件；
- 重新基于 train split 计算全局交互特征标准化，并回写每个 shard 的 `interaction_feat_style.npy`；
- 输出 `shard_manifest.json`、`build_summary.json`、`merged_build_summary.json`、`build_report.md`。

通过标准：
- `shard_manifest.json` 存在；
- `build_summary.json` 存在；
- `interaction_feature_standardization.json` 存在；
- merged `n_windows_kept` 等于四个输入分片之和；
- `fallback_assignment_rate` 仍为 0；
- `good_lane_context_rate` 仍为 1；
- global standardization 的 `train_count > 0`；
- 默认不创建 monolithic 大 `.npy` 文件。

# Stage 5A：重建 sharded summary

## 1. 命令

```bash
python tools/rebuild_waymo_5neighbor_context_summary.py \
  --data_root outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
  --overwrite

python tools/rebuild_waymo_5neighbor_context_summary.py \
  --data_root outputs/waymo_5neighbor_context_laneaware_clean_v1_part_00_13 \
  --validate_only
```

## 2. 期望行为

- 从 shards 反扫 split/meta/context_mask/debug csv。
- 重建 build_summary.json。
- 修复 split_counts 为空的问题。
- 不拼接大型 npy。
- 不重新生成数据。

## 3. 通过标准

- split_counts 不为空。
- sum(split_counts) == n_windows_kept。
- assignment_method_counts_by_slot 每个 slot 合计等于 n_windows_kept。
- nonfinite_output_detected = 0。
- build_report.md 显示 summary_rebuilt_from_shards=true。
- 不修改 Stage 4。

# Stage 5B：Flatten Context GRU 训练

## 1. 命令

### 1.1 Preflight（真实数据 smoke）
```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_smoke \
  --batch_size 64 \
  --epochs 1 \
  --max_train_samples 2048 \
  --max_val_samples 512 \
  --device cuda \
  --smoke_test_real_data \
  --overwrite
```

### 1.2 全量训练
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

如遇 CUDA OOM：将 `--batch_size` 降到 `128`。

### 1.3 导出 embedding（按 shard）
```bash
python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1/model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings \
  --batch_size 512 \
  --device cuda \
  --split all \
  --overwrite
```

## 2. 期望行为

- 从 `shard_manifest.json` 读取 Stage 5A 数据。
- 不拼接大型 npy。
- `context_traj.npy` 作为输入。
- `interaction_feat_style.npy` 作为弱监督。
- 训练 Flatten Context GRU。
- 导出 row-aligned sharded embedding。
- 不修改 Stage 4 / Stage 5A 数据构建逻辑。

## 3. 通过标准

- preflight 1 epoch 能跑通。
- Stage 5B smoke 通过标准：`train_loss` / `val_loss` 为有限值且 `model.pt` 存在。
- 单 epoch 时 `loss_curve.png` / `val_loss_curve.png` 看起来“空白”通常只是可视化尺度问题，不代表训练失败。
- `training_summary.json` 存在。
- `context_dim` 和 `feature_dim` 正确记录。
- full training 不 OOM。
- full training 建议开启 `--metric_alignment`。
- `embedding_manifest.json` 存在。
- exported embedding 总行数 = 164871。
- `nonfinite_embedding_detected = 0`。

# Exact Stage 5C evaluator command
```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

## Expected outputs
Training output dir:
- `model.pt`
- `best_model.pt`
- `training_config.json`
- `feature_group_config.json`
- `train_log.csv`
- `training_summary.json`

Embedding output dir:
- `embedding_manifest.json`
- `embeddings/` (shard-aligned outputs)
- optional merged `embeddings.npy`

Evaluation output dir:
- `evaluation_summary.json`
- `evaluation_report.md`
- `category_correlation_summary.csv`
- retrieval/correlation plots and CSVs

## Success criteria
- learned still beats `random/context_l2`.
- following_interaction mean correlation improves vs Stage 5B baseline.
- lateral_lane_dynamics advantage is preserved.
- global retrieval improves, or at least does not degrade significantly.

# Stage 5D 组加权训练（context GRU）

### 1. 命令
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

python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1/best_model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings \
  --split all \
  --merge_embeddings

python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

### 2. 期望行为
- 训练脚本会读取既有 Stage 5 shard 数据与 `feature_schema.json`，按特征名解析分组索引；不会重建 Stage 5A 数据集。
- 训练输出目录会保存模型、最优模型、训练配置、分组配置、日志和 summary。
- 导出脚本会生成与 shard 行顺序对齐的 embedding 文件与 manifest。
- 评估脚本使用现有 Stage 5C evaluator，对新 embedding 进行 strict-schema paper-grade 评估。

### 3. 通过标准
- 训练命令可正常启动并产生 `best_model.pt` 与 `training_summary.json`。
- `feature_group_config.json` 中可看到按 feature name 解析出的 group indices。
- 导出命令产出 `embedding_manifest.json`，且 `nonfinite_embedding_detected=0`。
- 评估命令产出 `evaluation_summary.json` 与 `category_correlation_summary.csv`，可用于验证 following 是否提升且 lateral 优势是否保持。

## Stage 6A：非配对风格漂移评估

## 1. 命令

```bash
python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --feature_path <interaction_feat_style.npy> \
  --feature_schema_path <feature_schema.json> \
  --split_path <split.npy> \
  --experiment_name neg_ctrl

python tools/stage6_compare_unpaired_style.py \
  --context_traj_path <context_traj.npy> \
  --feature_path <interaction_feat_style.npy> \
  --feature_schema_path <feature_schema.json> \
  --encoder_ckpt <stage5d_balanced_v2.pt> \
  --a_indices_path outputs/stage6A_splits/neg_ctrl/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/neg_ctrl/b_indices.npy \
  --output_dir outputs/stage6A_compare/neg_ctrl
```

## 2. 期望行为

读取 Stage 5 处理产物与 Stage 5D 编码器，输出 BDD、类别/特征/slice 漂移、top drift case、markdown report 与最小图表；不会触发新训练或下载新数据。

## 3. 通过标准

输出目录包含 `bdd_summary.json`、`category_delta.csv`、`feature_delta.csv`、`scenario_slice_delta.csv`、`top_drift_cases.csv`、`style_report_card.md` 与 `plots/*.png`。

## Stage 6A 非配对风格漂移（Issue #114）


### Stage 6A（Issue #116）推荐：full51 分片清单模式


#### Stage 6A-1：Negative control

## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

随机同分布拆分 A/B，作为“无真实风格漂移”的负对照，BDD 与 permutation p-value 不应提示显著漂移。

## 3. 通过标准

- 结果目录包含 split 与 compare 全套产物。
- BDD 接近 0 且 p-value 不显著（与历史负对照量级一致）。

#### Stage 6A-2：Pseudo style positive control

## 1. 命令

```bash
python tools/stage6_build_ab_splits.py \
  --mode pseudo_style_aggressive_vs_conservative \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name pseudo_agg_vs_cons \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/pseudo_agg_vs_cons \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

A（conservative_like）与 B（aggressive_like）应产生更明显风格差异；BDD 应高于 negative control，类别/特征变化应可解释 B 更激进、舒适性更弱。

## 3. 通过标准

- BDD 明显高于负对照。
- `category_delta.csv` 与 `feature_delta.csv` 支撑“更激进/更不舒适”解释。

#### Stage 6A-3：Scene confounding control

## 1. 命令

```bash
python tools/stage6_build_ab_splits.py \
  --mode scene_confounding_control \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name scene_confounding \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/scene_confounding \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

A（easy_scene_like）与 B（complex_scene_like）场景代理分布不同，BDD 可能抬升，报告应提示潜在 scenario/ODD confounding。

## 3. 通过标准

- `scenario_slice_delta.csv` 存在且可解释哪些场景代理切片驱动差异。
- warning 与 report card 提示“可能由场景分布差异驱动”。


## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
EMB_ROOT=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings

python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $DATA_ROOT/shard_manifest.json \
  --feature_schema_path $DATA_ROOT/feature_schema.json \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random

python tools/stage6_compare_unpaired_style.py \
  --source_shard_manifest $DATA_ROOT/shard_manifest.json \
  --embedding_manifest $EMB_ROOT/embedding_manifest.json \
  --feature_schema_path $DATA_ROOT/feature_schema.json \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --indices_are_test_relative \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random
```

## 2. 期望行为

- 默认按 Stage 5 full51 的 `shard_manifest.json` 读取分片特征与 split，不依赖根目录扁平 `.npy`。
- compare 默认按 Stage 5D-balanced-v2 的 `embedding_manifest.json` 读取分片 embedding。
- `--feature_path/--split_path` 与 `--embedding_path` 仅保留为 legacy fallback。

## 3. 通过标准

- split 目录产出 `a_indices.npy`、`b_indices.npy`、`split_summary.json`。
- compare 目录产出 `bdd_summary.json`、`category_delta.csv`、`feature_delta.csv`、`stage6_warnings.json` 与图表。
- 日志包含“使用 shard_manifest + embedding_manifest 模式（Stage5 full51 推荐路径）”提示。

### 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
FEATURE_PATH=$DATA_ROOT/interaction_feat_style.npy
SCHEMA_PATH=$DATA_ROOT/feature_schema.json
SPLIT_PATH=$DATA_ROOT/split.npy
CONTEXT_PATH=$DATA_ROOT/context_traj.npy
EMBEDDING_PATH=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embeddings.npy
CKPT=$DATA_ROOT/context_gru_stage5d_group_weighted_v1/best_model.pt

ls -lh \
  $FEATURE_PATH \
  $SCHEMA_PATH \
  $SPLIT_PATH \
  $CONTEXT_PATH \
  $EMBEDDING_PATH \
  $CKPT
```

> 重要警告：如果 `context_traj.npy`、`interaction_feat_style.npy` 或 `split.npy` 在 full51 merged 目录下缺失，单体数组命令不可用。若已有与 feature 行对齐的 embedding，请优先使用 `--embedding_path` 模式。

```bash
# 1) negative_control_random
python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --split_path $SPLIT_PATH \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random

# 2) pseudo_style_aggressive_vs_conservative
python tools/stage6_build_ab_splits.py \
  --mode pseudo_style_aggressive_vs_conservative \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --split_path $SPLIT_PATH \
  --output_dir outputs/stage6A_splits \
  --experiment_name pseudo_style_aggressive_vs_conservative

# 3) scene_confounding_control
python tools/stage6_build_ab_splits.py \
  --mode scene_confounding_control \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --split_path $SPLIT_PATH \
  --output_dir outputs/stage6A_splits \
  --experiment_name scene_confounding_control
```

```bash
# A. embedding_path 模式（推荐）
python tools/stage6_compare_unpaired_style.py \
  --embedding_path $EMBEDDING_PATH \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --top_k 20

# B. context/encoder 模式（回退）
python tools/stage6_compare_unpaired_style.py \
  --context_traj_path $CONTEXT_PATH \
  --feature_path $FEATURE_PATH \
  --feature_schema_path $SCHEMA_PATH \
  --encoder_ckpt $CKPT \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare_ctx/negative_control_random \
  --device cuda \
  --batch_size 256 \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --top_k 20
```

### 2. 期望行为
- split 脚本读取 `feature/split/schema`，生成 A/B 索引与 split summary。
- compare 脚本读取 A/B、embedding（或 context+ckpt）与 feature，输出 BDD、category/feature/slice/case 与报告卡。
- compare 脚本会写 `stage6_warnings.json`，提醒缺失特征、未标定 BDD、元数据缺失等风险。

### 3. 通过标准
- 三个 split 实验都能生成 `a_indices.npy`、`b_indices.npy`。
- compare 产物包含：
  - `bdd_summary.json`
  - `bdd_bootstrap_samples.csv`
  - `bdd_permutation_samples.csv`
  - `category_delta.csv`
  - `feature_delta.csv`
  - `scenario_slice_delta.csv`
  - `top_drift_cases.csv`
  - `stage6_warnings.json`
  - `style_report_card.md`
- `feature_delta.csv` 的 `permutation_p_value` 不应全是 1.0（除非数据本身极端巧合）。

## Stage 6A（推荐流程，Manifest 模式，仅此为当前建议）

## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python -m py_compile \
  tools/stage6_build_ab_splits.py \
  tools/stage6_compare_unpaired_style.py \
  tools/stage6_generate_report_card.py

python tools/stage6_build_ab_splits.py \
  --mode negative_control_random \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --eval_split test \
  --output_dir outputs/stage6A_splits \
  --experiment_name negative_control_random \
  --seed 42 \
  --overwrite

python tools/stage6_compare_unpaired_style.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6A_compare/negative_control_random \
  --num_bootstrap 50 \
  --num_permutation 100 \
  --max_mmd_samples 2000 \
  --max_top_case_candidates 5000 \
  --top_k 20 \
  --seed 42 \
  --overwrite
```

## 2. 期望行为

- 使用 `shard_manifest.json` + `embedding_manifest.json`，按分片顺序对齐读取特征和 embedding。
- A/B 索引为全局行号；top drift 不再构建完整 A×B 距离矩阵，避免 OOM。
- 即使无可用切片，也会输出仅含表头的 `scenario_slice_delta.csv`，并继续生成报告卡。

## 3. 通过标准

- 无需 `--embedding_path __unused_manifest_mode_guard_workaround__` 之类临时参数。
- 输出目录包含 `bdd_summary.json`、`category_delta.csv`、`feature_delta.csv`、`scenario_slice_delta.csv`、`top_drift_cases.csv`、`stage6_warnings.json`、`style_report_card.md` 和核心图表。
- 上述命令可直接复制执行且不依赖根目录扁平 `interaction_feat_style.npy/split.npy/context_traj.npy`。

> LEGACY/DEPRECATED：任何依赖 `$DATA_ROOT/interaction_feat_style.npy`、`$DATA_ROOT/split.npy`、`$DATA_ROOT/context_traj.npy`、`context_gru_stage5d_group_weighted_v1*` 的 Stage 6A 命令都不是当前推荐流程。


## Stage 6A — Full Negative Control（Manifest 模式，当前标准流程）

## 1. 命令

```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python -m py_compile   tools/stage6_build_ab_splits.py   tools/stage6_compare_unpaired_style.py   tools/stage6_generate_report_card.py

python tools/stage6_build_ab_splits.py   --mode negative_control_random   --shard_manifest $SHARD_MANIFEST   --feature_schema_path $FEATURE_SCHEMA   --eval_split test   --output_dir outputs/stage6A_splits   --experiment_name negative_control_random   --seed 42   --overwrite

python tools/stage6_compare_unpaired_style.py   --embedding_manifest $EMBEDDING_MANIFEST   --shard_manifest $SHARD_MANIFEST   --feature_schema_path $FEATURE_SCHEMA   --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy   --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy   --feature_groups_config configs/stage6_feature_groups.yaml   --output_dir outputs/stage6A_compare/negative_control_random   --num_bootstrap 50   --num_permutation 100   --max_mmd_samples 2000   --max_top_case_candidates 5000   --top_k 20   --seed 42   --overwrite
```

## 2. 期望行为

- `negative_control_random` 是在 `test` 集内做同分布随机切分（A/B）。
- 期望结果：BDD 很小、`p-value` 不显著、category/feature effect size 整体较小。
- 该实验用于验证 Stage 6A 不会把同分布样本误报为 style drift。
- 输出目录：`outputs/stage6A_compare/negative_control_random/`，包含：
  - `bdd_summary.json`
  - `category_delta.csv`
  - `feature_delta.csv`
  - `scenario_slice_delta.csv`
  - `top_drift_cases.csv`
  - `stage6_warnings.json`
  - `style_report_card.md`
  - `plots/`

## 3. 通过标准

- `BDD_MMD` 约 `0.0004`、`p-value` 约 `0.396`（数量级接近即可），表示整体分布漂移不显著。
- category/feature 的 effect size 应整体较小；若个别特征 p-value 显著但 effect size 很小，不应过度解读。
- `scenario_slice_delta.csv` 存在（即使仅表头也要有明确 warning）。

> 重要说明：
> - Stage 6A 当前推荐路径是 manifest 模式，不推荐根目录扁平 npy 作为主流程。
> - 不推荐命令中使用：`$DATA_ROOT/interaction_feat_style.npy`、`$DATA_ROOT/split.npy`、`$DATA_ROOT/context_traj.npy`、`context_gru_stage5d_group_weighted_v1`。
> - 以上仅作为 legacy/fallback。
> - 当前推荐 embedding 模型：`context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json`。
>
> 已知限制：scenario slicing 依赖 `feature_schema.json` 可用代理特征；若 speed/density proxy 缺失会产生 warnings。Stage 6B 需要更丰富 scene metadata/scene descriptor matching。

## Stage 6B — Baseline Comparison and Scenario-Controlled BDD

### 1. 命令
```bash
DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python tools/stage6b_compare_baselines.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/negative_control_random/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/negative_control_random/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6B_baselines/negative_control_random \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --pca_dim 16 --seed 42 --overwrite

python tools/stage6b_compare_baselines.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6B_baselines/pseudo_agg_vs_cons \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --pca_dim 16 --seed 42 --overwrite

python tools/stage6b_compare_baselines.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --feature_groups_config configs/stage6_feature_groups.yaml \
  --output_dir outputs/stage6B_baselines/scene_confounding \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --pca_dim 16 --seed 42 --overwrite

python tools/stage6b_scenario_balanced_bdd.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/scene_confounding/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/scene_confounding/b_indices.npy \
  --output_dir outputs/stage6B_balanced/scene_confounding \
  --balance_keys lateral_activity_bin \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --min_bin_size 100 --seed 42 --overwrite

python tools/stage6b_scenario_balanced_bdd.py \
  --embedding_manifest $EMBEDDING_MANIFEST \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --a_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/a_indices.npy \
  --b_indices_path outputs/stage6A_splits/pseudo_agg_vs_cons/b_indices.npy \
  --output_dir outputs/stage6B_balanced/pseudo_agg_vs_cons \
  --balance_keys lateral_activity_bin \
  --num_bootstrap 50 --num_permutation 100 --max_mmd_samples 2000 --min_bin_size 100 --seed 42 --overwrite

python tools/stage6b_summarize_experiments.py \
  --experiment_roots \
  outputs/stage6A_compare/negative_control_random \
  outputs/stage6A_compare/pseudo_agg_vs_cons \
  outputs/stage6A_compare/scene_confounding \
  outputs/stage6B_baselines/negative_control_random \
  outputs/stage6B_baselines/pseudo_agg_vs_cons \
  outputs/stage6B_baselines/scene_confounding \
  outputs/stage6B_balanced/scene_confounding \
  outputs/stage6B_balanced/pseudo_agg_vs_cons \
  --output_dir outputs/stage6B_summary \
  --overwrite
```

### 2. 期望行为
- baseline 脚本读取 manifest 模式特征与 embedding，输出 embedding/feature/PCA-feature 三套 MMD 统计与特征效应量。
- balanced 脚本按 `lateral_activity_bin` 做 A/B bin 内配平，再对比 raw 与 balanced BDD。
- summarize 脚本汇总 Stage6A/6B 输出，生成统一校准表与图。

### 3. 通过标准
- 三个 baseline 实验均生成 `baseline_summary.json`、`baseline_mmd.csv`、`top_feature_effects.csv` 和图。
- 两个 balanced 实验均生成 `balanced_bdd_summary.json`，且 `bins_used` 非空。
- summary 生成 `stage6b_calibration_table.csv` 与三张汇总图。

## Stage 6B — ODD bins and Behavior-event BDD

### 1. 命令
```bash
# 推荐 Stage6A/6B 工作流（manifest 模式）
# 1) shard_manifest.json
# 2) feature_schema.json
# 3) context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --inspect_metadata

DATA_ROOT=outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
SHARD_MANIFEST=$DATA_ROOT/shard_manifest.json
FEATURE_SCHEMA=$DATA_ROOT/feature_schema.json
EMBEDDING_MANIFEST=$DATA_ROOT/context_gru_stage5d_balanced_v2_embeddings/embedding_manifest.json
RAW_SCENARIO_DIR=<path_to_original_waymo_scenario_tfrecords>

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --output_dir $DATA_ROOT/map_odd_features_v1 \
  --overwrite

python tools/stage6b_build_odd_bins.py \
  --map_odd_manifest $DATA_ROOT/map_odd_features_v1/map_odd_manifest.json \
  --shard_manifest $SHARD_MANIFEST \
  --output_dir $DATA_ROOT/map_odd_bins_v1 \
  --overwrite

python tools/stage6b_build_behavior_event_bins.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --output_dir $DATA_ROOT/behavior_event_bins_v1 \
  --overwrite
```

### 2. 期望行为
- ODD bins 用于外部场景公平性控制（map/static context）。
- 若 `map_match_valid` 低于阈值，构建脚本会失败；不会再输出默认全零伪特征。
- Behavior-event bins 用于定位漂移发生在哪些驾驶任务。
- `event_lateral_activity_bin` 仅用于行为报告，不作为主 ODD 控制变量。

### 3. 通过标准
- `map_odd_feat.npy` 与分片 `interaction_feat_style.npy` 行对齐。
- `stage6b_build_map_odd_features.py` 必须报告 metadata/raw scenario overlap，且 `map_match_valid` 非零。
- `odd_bins.csv` / `behavior_event_bins.csv` 必须包含 `global_row, shard_id, local_row`。
- ODD 平衡后可输出 `BDD_odd_balanced`，并与 `BDD_overall` 对比解释。


## Legacy 说明

旧的 root-level flat npy 命令仅保留兼容，不再作为 Stage6A/6B 主流程推荐。

## Stage 6B（map-derived ODD）安全流程（2026-05 更新）

### 1. 命令

```bash
python -m py_compile \
  tools/stage6b_build_map_odd_features.py \
  tools/stage6b_build_odd_bins.py \
  tools/stage6b_build_behavior_event_bins.py \
  tools/stage6b_bin_bdd_report.py \
  tools/stage6b_scenario_balanced_bdd.py

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --output_dir $DATA_ROOT/map_odd_features_v1_debug \
  --max_scenarios 1000 \
  --inspect_metadata

python tools/stage6b_build_map_odd_features.py \
  --shard_manifest $SHARD_MANIFEST \
  --feature_schema_path $FEATURE_SCHEMA \
  --raw_scenario_dir $RAW_SCENARIO_DIR \
  --output_dir $DATA_ROOT/map_odd_features_v1_debug \
  --max_scenarios 1000 \
  --min_match_rate 0.01 \
  --overwrite

python tools/stage6b_build_odd_bins.py \
  --map_odd_manifest $DATA_ROOT/map_odd_features_v1_debug/map_odd_manifest.json \
  --shard_manifest $SHARD_MANIFEST \
  --output_dir $DATA_ROOT/map_odd_bins_v1_debug \
  --min_map_match_rate 0.01 \
  --overwrite
```

### 2. 期望行为

- 先做语法检查，再做 metadata inspect。
- inspect 阶段会检查 processed/raw scenario overlap、context_traj 是否存在、窗口字段是否存在。
- 仅当 inspect 建议可运行时再执行 debug build。
- map ODD 分箱仅使用 `map_match_valid=1` 行计算分位点，`map_match_valid=0` 行统一标记 `unknown`。
- behavior-event bins 保持独立报告层，不与 map ODD bins 混用。
- 旧版 flat-npy Stage6 命令视为 **LEGACY/DEPRECATED**，不建议继续作为主流程。

### 3. 通过标准

- `py_compile` 全部通过。
- inspect 输出包含 overlap 与 context 检查，recommendation 为可执行状态。
- map_odd 输出包含 `map_odd_warnings.json` 且 `map_match_valid_rate` 不低于阈值（或显式允许低匹配率）。
- odd_bin_report 同时给出 valid/invalid 统计与 valid/overall 分布。
- behavior_event_bin_report 明确声明 `event_lateral_activity_bin` 是 behavior-contaminated。
