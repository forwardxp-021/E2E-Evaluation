# 01_experiment_1_population_eval — Population-level Policy Separation

> 更新时间：2026-05-07  
> 所属阶段：阶段 2  
> 目的：从单个 hero case 扩展到全量 test source 的统计验证。

---

## 1. 实验目标

阶段 1 的 interpretability demo 已经可以解释单个 source 下的 p0/p1/p2 行为差异，但论文不能只依赖单个案例。因此阶段 2 的目标是：

> 在全量 test split 上验证 behavior embedding 是否具备 policy-level 可分性、可分类性、可检索性，并检查 lateral_stable 是否真正形成独立第三类风格。

核心问题：

1. embedding 是否能区分 p0/p1/p2？
2. 同一个 source 下 p0/p1/p2 的 embedding 距离分布如何？
3. p2/lateral_stable 是否普遍远离 p0 和 p1？
4. centroid classification 是否明显高于 chance？
5. global retrieval 是否能找回同 policy 样本？
6. embedding distance 是否与 jerk/yaw/curvature/THW 等 style signal 差异一致？

---

## 2. 数据设定

当前 test split 结构：

```text
395 sources × 3 policies = 1185 samples
```

policy 映射：

```text
p0 = conservative
p1 = aggressive
p2 = lateral_stable
```

source 表示同一个原始场景窗口，通常由如下字段构成：

```text
scenario_id + start + window_len + front_id
```

评估中优先使用 `source_index.npy` 作为 authoritative grouping key。

---

## 3. 输入文件

population evaluator 依赖：

```text
feat_style.npy
feat_style_raw.npy optional
traj.npy
front.npy
meta.npy
split.npy
source_index.npy
policy_id.npy
policy_name.npy
source_key.npy optional
```

要求：

- 所有数组按 row index 对齐；
- 每个完整 source group 需要包含 p0/p1/p2；
- `policy_id` 与 `policy_name` 映射一致；
- split 由 scenario_id hash 分配。

---

## 4. 核心输出

```text
population_summary.json
population_report.md
per_source_pairwise_distances.csv
per_source_style_summary.csv
centroid_classification.csv
centroid_confusion_matrix.csv
global_retrieval_summary.csv
global_retrieval_topk.csv
pairwise_distance_boxplot.png
p2_separation_margin_hist.png
p2_farthest_rate_bar.png
centroid_confusion_matrix.png
retrieval_hit_at_k_bar.png
policy_style_fingerprint_boxplot.png
p2_distance_vs_style_delta_scatter.png
embedding_2d_population_pca.png
```

---

## 5. 核心指标

### 5.1 coverage diagnostics

检查全量数据完整性：

```text
n_total_rows
n_rows_after_split
n_unique_sources_after_split
source_group_size_histogram_after_split
n_complete_sources
n_incomplete_sources
policy_counts_after_split
warnings
```

当前结论：数据结构正确，test split 中每个 source 均有 p0/p1/p2。

---

### 5.2 within-source pairwise embedding distance

对每个完整 source，计算：

```text
d_p0_p1
d_p0_p2
d_p1_p2
```

重点指标：

```text
p2_farthest_rate
mean_p2_separation_margin
median_p2_separation_margin
pct_p2_separation_margin_gt_0
```

定义：

```text
p2_farthest = d(p0,p2) > d(p0,p1) and d(p1,p2) > d(p0,p1)
```

```text
p2_separation_margin = min(d(p0,p2), d(p1,p2)) - d(p0,p1)
```

解释：

- margin > 0：p2 比 p0-p1 还远，说明 p2 有较强独立性；
- margin < 0：p2 仍接近 p0 或 p1，独立性不足。

---

### 5.3 centroid classification

用 train split 计算每个 policy 的 centroid，在 eval split 上按最近 centroid 分类。

指标：

```text
centroid_accuracy_overall
centroid_accuracy_p0
centroid_accuracy_p1
centroid_accuracy_p2
```

chance level：

```text
1 / 3 = 0.3333
```

阶段 2 结果显示 centroid accuracy 明显高于 chance，说明 embedding 含有 policy-level behavior information。

---

### 5.4 global retrieval

对 eval split 中每个 query，在全局样本中检索 Top-K 最近邻。

默认排除：

- query 自身；
- same source_index；
- same scenario_id if available。

指标：

```text
retrieval_hit_at_1
retrieval_hit_at_k
retrieval_mean_same_policy_count_topk
retrieval_mean_same_policy_fraction_topk
```

阶段 2 结果显示 global retrieval 效果较强，说明 embedding 具备跨 source 检索同 policy / 相似 behavior 的能力。

---

### 5.5 style-distance correlation

计算 embedding distance 与 style signal delta 的相关性，例如：

```text
mean_speed_delta_spearman
rms_jerk_delta_spearman
rms_yaw_rate_delta_spearman
rms_curvature_delta_spearman
mean_thw_delta_spearman
```

阶段 2 中，jerk delta 与 embedding distance 的相关性较明显，说明 embedding 对纵向舒适性差异较敏感。

---

## 6. 阶段 2 主要结论

### 正向结论

1. test split 数据组织正确；
2. embedding 能明显区分 policy，centroid classification 高于 chance；
3. global retrieval 能较好找回同 policy 样本；
4. embedding distance 与部分物理 style 差异，尤其 jerk，有较强一致性。

### 关键限制

初始配置下：

```text
p2_farthest_rate ≈ 0.05
mean_p2_separation_margin < 0
```

说明 p2/lateral_stable 虽然可识别、可检索，但整体上更接近 conservative，而不是稳定成为独立第三类风格。

---

## 7. 阶段 2 对后续工作的意义

阶段 2 的最大价值是把问题从：

```text
embedding 能不能区分 policy？
```

推进到：

```text
怎样设计 lateral_stable generator，才能让 p2 更像真正独立的第三种风格？
```

因此自然进入阶段 3：generator ablation 与 local fine sweep。

---

## 8. 当前任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| population evaluator | 完成 | 支持全量 source 统计 |
| coverage diagnostics | 完成 | test split 为 395×3 |
| pairwise distance distribution | 完成 | p0-p1 / p0-p2 / p1-p2 |
| p2_farthest_rate | 完成 | 发现初始 p2 独立性不足 |
| centroid classification | 完成 | 明显高于 chance |
| global retrieval hit@k | 完成 | 检索能力较强 |
| style-distance correlation | 完成 | jerk delta 相关性明显 |
| 论文表格整理 | 待做 | 后续与 v2 final compare 一起整理 |
