# 05_baseline_plan — Baseline Suite 设计

> 更新时间：2026-05-07  
> 所属阶段：阶段 4 / 论文补强  
> 目的：补齐 baseline，对抗“embedding 只是 handcrafted feature 包装”或“synthetic artifact”的审稿质疑。

---

## 1. 为什么必须做 baseline

当前阶段 1-3 已经完成：

- PR2 interpretability demo；
- population-level policy separation；
- broad ablation；
- local fine sweep；
- recommended_lateral_stable_v2 final compare。

这些证明了在 controlled synthetic rollout setting 下，behavior embedding 具备 policy-level discriminability 和 retrieval capability。

但如果要冲高水平会议，审稿人一定会问：

> 你的 learned embedding 是否真的比简单 handcrafted features、trajectory distance、random embedding 更好？

因此 baseline suite 是论文可信度的关键。

---

## 2. Baseline 的核心作用

Baseline 要回答：

1. learned embedding 是否优于 raw style feature distance？
2. learned embedding 是否优于简单 trajectory geometry distance？
3. learned embedding 是否优于随机向量或未训练 encoder？
4. learned embedding 的 retrieval / classification / style correlation 是否有实际增益？
5. public human trajectory validation 中，learned embedding 是否仍有优势？

---

## 3. 最小 baseline suite

建议至少实现以下 baseline：

| baseline | 说明 | 优先级 |
|---|---|---|
| raw_feature_distance | 直接用 handcrafted style features 计算距离 | P0 |
| feature_only_retrieval | 用 `feat_style.npy` 直接做 retrieval | P0 |
| trajectory_l2_distance | 用轨迹点 L2 距离做 baseline | P0 |
| random_embedding | 随机向量对照 | P0 |
| untrained_encoder | 使用随机初始化 encoder 导出 embedding | P1 |
| pca_feature_embedding | 对 handcrafted features 做 PCA 后检索 | P1 |
| dtw_distance | 动态时间规整轨迹距离 | P1 |
| frechet_distance | 曲线 Frechet 距离 | P2 |

---

## 4. Baseline 1：raw_feature_distance

### 定义

直接使用 `feat_style.npy` 或 `feat_style_raw.npy` 作为表示向量，计算距离：

```text
d(i,j) = ||feat_style_i - feat_style_j||
```

### 作用

回答：

> learned embedding 是否只是 handcrafted feature 的简单复制？

### 评价方式

在同一套 evaluator 中比较：

```text
centroid classification
retrieval hit@1 / hit@K
same-policy fraction@TopK
style-distance correlation
```

---

## 5. Baseline 2：feature_only_retrieval

### 定义

不训练 encoder，直接用 handcrafted style feature 做 global retrieval。

### 作用

回答：

> 如果直接用 style feature 做检索，是否已经足够？

如果 learned embedding 没有明显优于该 baseline，则论文贡献需要重新表述为 benchmark/evaluator，而不是 embedding model。

---

## 6. Baseline 3：trajectory_l2_distance

### 定义

将 ego trajectory 对齐到 local frame 后，计算逐时刻 L2 距离：

```text
d(i,j) = mean_t ||p_i(t) - p_j(t)||_2
```

建议 local normalization：

- 起点平移到原点；
- 初始 heading 旋转到 +x；
- 统一 window_len。

### 作用

回答：

> 简单轨迹几何距离是否已经能区分 policy？

### 注意

trajectory distance 可能更关注空间形状，而不关注 jerk / yaw / THW 等 style。

---

## 7. Baseline 4：random_embedding

### 定义

为每条样本生成同维度随机向量：

```text
z_i ~ N(0, I)
```

### 作用

提供 sanity check。

预期结果：

- centroid classification 接近 chance；
- retrieval hit@K 接近随机水平；
- style-distance correlation 接近 0。

---

## 8. Baseline 5：untrained_encoder

### 定义

使用当前 encoder architecture，但随机初始化，不训练，直接导出 embedding。

### 作用

回答：

> 提升是否来自训练，而不是模型结构或输入统计本身？

---

## 9. Baseline 6：pca_feature_embedding

### 定义

对 handcrafted style features 做 PCA，降到与 learned embedding 相同维度或较低维度：

```text
feat_style -> PCA -> z_pca
```

### 作用

回答：

> 简单线性降维是否已经能达到类似效果？

---

## 10. Baseline 7：DTW / Frechet distance

### 定义

使用轨迹序列距离：

- DTW：允许时间轴非线性对齐；
- Frechet：衡量曲线形状距离。

### 作用

提供经典轨迹相似性对照。

### 优先级

P1/P2。可以后做，不影响第一版 baseline suite。

---

## 11. 统一评价指标

所有 baseline 应使用同一套评价协议：

### synthetic policy setting

```text
centroid_accuracy_overall
centroid_accuracy_p0/p1/p2
retrieval_hit@1
retrieval_hit@K
same_policy_fraction@TopK
d_p0_p1_mean
d_p0_p2_mean
d_p1_p2_mean
p2_farthest_rate
mean_p2_separation_margin
style-distance Spearman correlation
```

### public human trajectory setting

```text
pseudo_label_classification_accuracy
same_pseudo_label_retrieval_hit@1 / hit@K
same_pseudo_label_fraction@TopK
style-distance Spearman correlation
cluster style fingerprint separability
```

---

## 12. 建议新增脚本

```text
tools/evaluate_baselines.py
```

功能：

1. 加载统一数据格式；
2. 构造 baseline representation / distance；
3. 复用 population evaluator 的分类与检索逻辑；
4. 输出 baseline comparison 表格和图。

可选拆分：

```text
tools/build_baseline_embeddings.py
tools/evaluate_representation.py
```

---

## 13. 建议输出文件

```text
baseline_summary.csv
baseline_summary.json
baseline_report.md
baseline_retrieval_summary.csv
baseline_centroid_classification.csv
baseline_style_distance_correlation.csv
baseline_comparison_bar.png
baseline_retrieval_bar.png
baseline_classification_bar.png
baseline_style_correlation_bar.png
```

---

## 14. 论文中 baseline 表格建议

最终论文表格可包含：

| Method | Centroid Acc ↑ | Hit@1 ↑ | Hit@5 ↑ | Same@5 ↑ | Jerk Corr ↑ | Yaw Corr ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Random embedding | | | | | | |
| Trajectory L2 | | | | | | |
| Raw feature | | | | | | |
| PCA feature | | | | | | |
| Learned embedding | | | | | | |

对 public human trajectory，可替换为 pseudo-label 指标。

---

## 15. 成功标准

最小成功标准：

1. learned embedding 明显优于 random embedding；
2. learned embedding 在 retrieval 或 classification 上不弱于 raw feature；
3. learned embedding 在 style-distance correlation 上至少对部分关键 style signal 有优势；
4. trajectory distance 无法同时覆盖 jerk/yaw/THW 等行为信号；
5. baseline 结果能支持 benchmark 贡献。

强成功标准：

1. learned embedding 在 synthetic 和 human validation 中均优于主要 baseline；
2. learned embedding 的 retrieval case 更符合人类可解释 style；
3. style fingerprint 与 embedding cluster 一致。

---

## 16. 风险与应对

### 风险 1：raw feature baseline 很强

可能说明 learned embedding 主要学习了 handcrafted feature。

应对：

- 论文贡献强调 benchmark/evaluator；
- 分析 embedding 是否在 retrieval 可视化或跨场景泛化上更好；
- 增加 condition-aware / scene-controlled validation。

### 风险 2：trajectory distance 很强

可能说明几何轨迹已经足够区分 synthetic policy。

应对：

- 强调 trajectory distance 难以解释 jerk/yaw/THW；
- 加入 style signal correlation；
- 在 public human data 上验证更复杂行为结构。

### 风险 3：human pseudo label 不稳定

应对：

- 做 threshold sensitivity；
- 报告 label distribution；
- 不把 pseudo label 当真实 ground truth，只作为 weak validation。

---

## 17. 当前任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| baseline suite 设计 | 完成初稿 | 本文档 |
| raw feature baseline | 未开始 | P0 |
| feature-only retrieval | 未开始 | P0 |
| trajectory L2 baseline | 未开始 | P0 |
| random embedding baseline | 未开始 | P0 |
| untrained encoder baseline | 未开始 | P1 |
| PCA feature baseline | 未开始 | P1 |
| DTW / Frechet baseline | 未开始 | P1/P2 |
| baseline comparison report | 未开始 | 阶段 4 必要 |

---

## 18. 下一步建议

优先实现最小 baseline suite：

```text
raw_feature_distance
feature_only_retrieval
trajectory_l2_distance
random_embedding
```

先在 synthetic final_compare 数据上跑通，再迁移到 public human trajectory validation。
