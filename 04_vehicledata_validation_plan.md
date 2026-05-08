# 04_vehicledata_validation_plan — Public Human Trajectory External Validation

> 更新时间：2026-05-07  
> 所属阶段：阶段 4  
> 当前状态：尚未开始，下一大阶段  
> 目的：验证 behavior embedding 是否不仅适用于 synthetic policy，也能解释真实人类驾驶轨迹。

---

## 1. 为什么需要阶段 4

当前阶段 1-3 已经证明：

- controlled synthetic rollout 下，embedding 有 policy-level 区分能力；
- p0/p1/p2 可以通过 within-source aligned metrics、classification、retrieval 进行评价；
- lateral_stable 机制可以通过 ablation 和 local sweep 被解释；
- `recommended_lateral_stable_v2` 明显优于 baseline。

但当前最大学术风险仍然是：

> synthetic policy 过于规则化，embedding 可能只是在识别 generator artifact。

因此，阶段 4 必须使用公开真实人类轨迹数据做 external validation。

目标不是证明真实人类一定严格分成 p0/p1/p2，而是证明：

> embedding 在真实 human driving trajectories 上也能形成可解释的 behavior structure。

---

## 2. 阶段 4 核心问题

阶段 4 需要回答：

1. 在真实人类轨迹中，embedding 近邻是否具有相似 style signals？
2. 用规则构造的 pseudo style labels 是否能被 embedding 区分？
3. embedding distance 是否与 jerk/yaw/curvature/THW 等物理风格差异相关？
4. embedding 聚类后的 style fingerprint 是否可解释？
5. learned embedding 是否优于 raw feature distance、trajectory distance、random embedding 等 baseline？

---

## 3. 候选公开数据集

候选数据集：

```text
Waymo Open Motion Dataset
Argoverse Motion Forecasting
nuScenes prediction / tracking
INTERACTION
highD / inD
```

### 当前建议

优先选择与当前工程兼容度最高的数据：

1. **Waymo Open Motion Dataset**  
   优点：当前项目已经围绕 Waymo 轨迹格式做了较多处理；场景丰富。  
   缺点：下载/解析环境成本较高。

2. **INTERACTION / inD / highD**  
   优点：轨迹数据结构更轻，适合快速 external validation。  
   缺点：道路类型和 Waymo 城市场景差异较大。

建议策略：

- 若当前 Waymo 解析链路已经稳定，优先 Waymo；
- 若想快速做 proof-of-concept，可先用 INTERACTION / inD / highD。

---

## 4. 数据转换目标格式

阶段 4 最重要的工程目标是把公开 human trajectory 转成当前统一格式：

```text
traj.npy
front.npy
meta.npy
split.npy
feat_style.npy
feat_style_raw.npy
feature_names_style.json
source_index.npy optional
pseudo_label.npy optional
pseudo_label_name.npy optional
```

其中：

- `traj.npy`：ego trajectory；
- `front.npy`：lead/front vehicle trajectory，如果没有可靠 front，可为空或使用 nearest-front 规则构造；
- `meta.npy`：scenario_id、agent_id、start、window_len、front_id 等；
- `split.npy`：train/val/test；
- `feat_style.npy`：标准化 style features；
- `feat_style_raw.npy`：原始 style features；
- `pseudo_label.npy`：规则构造的 pseudo style label。

---

## 5. Pseudo-label validation 方案

真实数据没有 policy label，因此使用规则构造 pseudo style labels。

### 5.1 aggressive-like

候选条件：

```text
high mean_speed
low THW
high accel_rms / jerk_rms
high closing speed
```

直观含义：

> 速度更高、跟车更近、加减速更强。

---

### 5.2 conservative-like

候选条件：

```text
low mean_speed
high THW
low jerk_rms
low accel_rms
```

直观含义：

> 更慢、更大时距、更平顺。

---

### 5.3 lateral-stable-like

候选条件：

```text
low yaw_rate_rms
low curvature_rms
low heading_change_total
smooth heading
```

直观含义：

> 横向变化小，方向变化稳定。

---

## 6. Pseudo-label 构造注意事项

必须避免过强的规则循环论证。

例如，如果 pseudo label 完全由 `jerk_rms` 决定，然后又用 `jerk_rms` 证明 embedding 好，就会被审稿人质疑。

建议：

1. label 构造用少量核心规则；
2. validation 用不同维度的 style fingerprint；
3. 同时报告 retrieval 和 cluster fingerprint；
4. 与 baseline 对比；
5. 对 pseudo label 的阈值做 sensitivity analysis。

---

## 7. 阶段 4 评价指标

### 7.1 pseudo-label classification

类似 synthetic policy 的 centroid classification：

```text
pseudo_label centroid -> eval classification
```

指标：

```text
overall accuracy
per-label accuracy
confusion matrix
chance level
```

---

### 7.2 same pseudo-label retrieval

对 human trajectory embedding 做 global retrieval：

```text
query -> Top-K nearest neighbors
```

指标：

```text
hit@1_same_pseudo_label
hit@K_same_pseudo_label
mean_same_label_fraction_topK
```

---

### 7.3 embedding distance vs style delta correlation

计算 embedding distance 与 style delta 的 Spearman correlation：

```text
mean_speed_delta
rms_jerk_delta
rms_yaw_rate_delta
rms_curvature_delta
mean_thw_delta
min_thw_delta
```

---

### 7.4 cluster style fingerprint

对 human trajectory embedding 聚类，输出每个 cluster 的 style summary：

```text
mean_speed
rms_jerk
rms_yaw_rate_proxy
rms_curvature_proxy
mean_thw
min_thw
```

目标：

> 聚类结果是否对应可解释的 driving behavior modes。

---

### 7.5 retrieval visualization

复用 PR2 demo 思路：

```text
query trajectory
Top-K retrieved trajectories
style signals
retrieval table
```

但解释时应强调：

> human trajectory retrieval 没有真实 policy label，pseudo label 只是弱监督解释。

---

## 8. 必须加入的 baseline

阶段 4 必须与 baseline 对比，否则顶会说服力不足。

最小 baseline suite：

| baseline | 说明 |
|---|---|
| raw feature distance | 直接用 handcrafted style feature 做距离 |
| feature-only retrieval | 不用 learned embedding，直接检索 feature nearest neighbors |
| trajectory distance | 使用轨迹几何距离，例如 L2 / Frechet / DTW |
| random embedding | 随机向量对照 |
| untrained encoder | 随机初始化模型 encoder |
| PCA feature embedding | 对 handcrafted feature 做 PCA 后检索 |

核心比较：

```text
learned embedding vs handcrafted features vs trajectory distance vs random baseline
```

---

## 9. 建议脚本规划

建议新增：

```text
tools/build_human_trajectory_dataset.py
tools/assign_pseudo_style_labels.py
tools/evaluate_human_style_embedding.py
tools/human_retrieval_demo.py
```

### 9.1 build_human_trajectory_dataset.py

功能：

- 读取公开数据集；
- 提取 ego/front trajectory；
- 滑窗；
- 生成统一 `.npy` 文件；
- 计算 style features；
- 生成 split。

### 9.2 assign_pseudo_style_labels.py

功能：

- 根据 raw style features 构造 pseudo labels；
- 支持 percentile-based thresholds；
- 输出 label distribution；
- 避免类别极端不均衡。

### 9.3 evaluate_human_style_embedding.py

功能：

- pseudo-label classification；
- same pseudo-label retrieval；
- style-distance correlation；
- cluster fingerprint；
- baseline comparison。

### 9.4 human_retrieval_demo.py

功能：

- 选 query；
- Top-K 检索；
- 轨迹与 style signal 可视化；
- 自动生成 report。

---

## 10. 预期输出

```text
human_validation_summary.json
human_validation_report.md
pseudo_label_distribution.csv
pseudo_label_classification.csv
pseudo_label_confusion_matrix.png
human_retrieval_topk.csv
human_retrieval_summary.csv
style_distance_correlation.csv
cluster_style_fingerprint.csv
cluster_style_fingerprint.png
human_embedding_pca.png
human_embedding_umap.png
baseline_comparison_summary.csv
baseline_comparison_report.md
```

---

## 11. 阶段 4 成功标准

最低成功标准：

1. 可以把一个公开人类轨迹数据集转成当前统一格式；
2. pseudo labels 分布不过度失衡；
3. learned embedding 的 pseudo-label retrieval 明显高于 random；
4. embedding distance 与至少部分 style delta 有正相关；
5. cluster fingerprint 有可解释结构；
6. learned embedding 不弱于至少一部分 simple baselines。

更强成功标准：

1. learned embedding 明显优于 raw feature 和 trajectory distance baseline；
2. pseudo-label classification 明显高于 chance；
3. retrieval cases 人类可解释；
4. 多数据集验证一致。

---

## 12. 当前任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| 数据集选择 | 未开始 | Waymo / INTERACTION / inD / highD 待定 |
| 数据转换脚本 | 未开始 | 需要统一成 traj/front/meta/split |
| pseudo-label 规则 | 未开始 | 需要先设计阈值与类别平衡 |
| human embedding evaluation | 未开始 | 阶段 4 核心 |
| baseline suite | 未开始 | 顶会必要 |
| retrieval demo | 未开始 | 复用 PR2 思路 |
| report 文档 | 未开始 | 最终形成 validation evidence |

---

## 13. 阶段 4 的论文作用

阶段 4 是把论文从 synthetic benchmark 推向更强贡献的关键。

它解决的问题是：

> 当前方法是否只是在识别 synthetic generator artifact？

如果阶段 4 成功，论文可以更有力地声称：

> The proposed trajectory-level behavior embedding captures interpretable driving behavior structure not only in controlled synthetic policy rollouts, but also in public human trajectory data.

---

## 14. 下一步建议

正式进入阶段 4 前，先完成一个设计 PR：

```text
04_vehicledata_validation_plan.md
```

然后让 Codex 实现最小闭环：

```text
1. 选一个数据集
2. 转成统一格式
3. 构造 pseudo labels
4. 跑 pseudo-label retrieval
5. 输出 baseline comparison
```

建议第一版不要追求复杂模型，先把数据链路和评价指标跑通。


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
