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
