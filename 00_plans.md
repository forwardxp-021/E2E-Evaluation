# 00_plans — E2E 行为风格评估论文路线与进展

> 更新时间：2026-05-07  
> 当前分支：`20260507_vehicledata_validation`  
> 项目定位：博士论文 / 论文实验工程  
> 研究对象：自动驾驶端到端（E2E）决策/规划模型的 trajectory-level closed-loop 行为评估

---

## 1. 一句话总结当前路线

我们要做的不是传统感知/检测/渲染闭环，也不是只看 ADE/FDE、collision rate 这类传统指标，而是建立一套 **trajectory-level behavior evaluation benchmark**：

> 给定任意规划/决策 policy 的轨迹级 rollout，将其编码为 behavior embedding，并通过 aligned policy separation、global retrieval、style fingerprint、style drift / BDD 等指标，评价不同 policy 或不同模型版本之间的驾驶风格差异。

当前研究可以概括为：

```text
source window
  ↓
controlled synthetic policy rollout generator
  ↓
p0 / p1 / p2 policy rollouts
  ↓
behavior embedding / style feature representation
  ↓
within-source aligned evaluation
  ↓
population-level statistics
  ↓
ablation / local sweep
  ↓
recommended_lateral_stable_v2
  ↓
public human trajectory external validation（下一阶段）
```

---

## 2. 研究背景与最初动机

最初的问题来自自动驾驶端到端（E2E）模型开发中的一个真实观察：

- E2E 决策/规划模型越来越像“驾驶员”；
- 模型版本之间会出现明显驾驶风格变化；
- 有的版本更激进，有的版本更保守，有的版本更关注舒适性；
- 传统评价指标很难描述“风格变化”；
- 现实中评价人类司机时，并不会要求司机持续解释内部决策逻辑，而是看其实际驾驶行为。

因此，本研究的核心想法是：

> 对 E2E 自动驾驶系统，应该增加一套类似评价人类驾驶员的行为风格评价体系。

这个体系重点不是解释神经网络内部，而是评价它输出的 trajectory-level behavior。

---

## 3. 当前论文定位

建议论文定位为：

> **A trajectory-level behavior evaluation benchmark for closed-loop planning policies.**

也就是：

- 输入：trajectory rollout；
- 输出：behavior embedding / style feature / report card；
- 指标：policy separation、style drift、global retrieval、BDD、comfort/risk/style fingerprint；
- 适用对象：synthetic policy、rule-based policy、learning-based planner、E2E planner，只要能输出轨迹即可。

重要边界：

- 不做 sensor rendering；
- 不做 perception stack；
- 不要求实车私有数据作为第一阶段前提；
- 不评估完整 perception-to-control stack；
- 关注 planning policy behavior。

论文中可以使用这样的表述：

> The benchmark is model-agnostic and accepts any trajectory-level rollout, including learned E2E planners, rule-based policies, and synthetic policies.

---

## 4. 核心研究问题

当前围绕以下问题展开：

### Q1. Behavior embedding 是否能区分不同 policy 的驾驶行为？

需要证明：

- 不同 policy 在 embedding 空间中有可分性；
- embedding 不是随机的；
- centroid classification 明显高于 chance；
- global retrieval 能找回同 policy / 相似行为样本。

### Q2. 同一个 source 下，不同 policy 的差异是否可解释？

需要证明：

- 同一 source window 下，p0/p1/p2 轨迹不同；
- 差异可以用 speed、accel、jerk、yaw_rate、curvature、THW 等 style signal 解释；
- embedding distance matrix 能反映这些差异。

### Q3. lateral_stable 是否真的形成独立第三类风格？

这是当前阶段最重要的具体问题。

目前结论：

- lateral_stable 可识别、可检索；
- recommended_lateral_stable_v2 显著提升了 p2 的可识别性和稳定性；
- 但 mean p2 separation margin 仍为负，因此 p2 independence 仍然 incomplete。

### Q4. 这个方法是否只是在识别 synthetic generator artifact？

这是最大审稿风险。

下一阶段需要用公开真实人类轨迹数据做 external validation。

---

## 5. 当前数据与实验设定

### 5.1 数据来源

当前主要使用 Waymo 轨迹数据构造 source window。

### 5.2 当前 synthetic policy rollout 设定

对同一个 source window 生成三个 policy rollout：

| policy_id | policy_name | 当前含义 |
|---|---|---|
| p0 | conservative | 保守型 policy |
| p1 | aggressive | 激进型 policy |
| p2 | lateral_stable | 横向稳定 / 舒适型 policy |

### 5.3 当前关键假设

当前为 trajectory-level closed-loop / ego-only rollout：

- ego 自车根据不同 policy 生成 rollout；
- front vehicle 使用 replay 轨迹；
- front vehicle 不受 ego policy 影响；
- 不做完整多智能体闭环仿真。

这个设定的优点：

- 成本低；
- 可控；
- 可复现；
- 适合对齐同一 source 下的 policy 差异。

限制：

- 不是完整 multi-agent closed-loop；
- 不能声称等价于实车闭环；
- 需要在论文中明确边界。

---

## 6. 关键概念定义

### 6.1 source / source window

`source` 指一个原始场景窗口，例如：

```text
scenario_id + start + window_len + front_id
```

它代表同一段场景上下文：

- 同一个场景；
- 同一个起始时间；
- 同一个窗口长度；
- 同一个前车关系。

### 6.2 within-source

`within-source` 指在同一个 source window 下比较不同 policy。

例如：

```text
source_i:
  p0 rollout
  p1 rollout
  p2 rollout
```

它的价值在于控制场景变量，让差异主要来自 policy。

### 6.3 embedding

`embedding` 是将一段驾驶轨迹/行为压缩成固定长度向量，例如：

```text
trajectory window -> z ∈ R^D
```

embedding 空间中的距离表示行为差异，而不是物理米制距离。

### 6.4 embedding distance matrix

同一 source 下 p0/p1/p2 的 embedding 两两距离矩阵。

它回答：

> 模型认为这三种 policy 的行为有多不同？

### 6.5 global retrieval

给定一个 query embedding，在全局测试集里检索 Top-K 最近邻。

它回答：

> embedding 是否能跨 source 找回相似行为 / 同 policy 样本？

### 6.6 p2_farthest_rate

定义：

```text
p2_farthest = true if d(p0,p2) > d(p0,p1) and d(p1,p2) > d(p0,p1)
```

`p2_farthest_rate` 表示有多少比例的 source 中，p2 比 p0-p1 更远。

这是判断 lateral_stable 是否形成独立第三类的重要指标。

### 6.7 p2 separation margin

定义：

```text
p2_separation_margin = min(d(p0,p2), d(p1,p2)) - d(p0,p1)
```

解释：

- margin > 0：p2 比 p0-p1 之间还远，说明 p2 有较强独立性；
- margin < 0：p2 仍然更接近 p0 或 p1，独立性不完全。

---

## 7. 阶段规划与当前进展

当前整体规划分为四个阶段：

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| 阶段 1 | PR2 interpretability demo | 已完成 |
| 阶段 2 | population-level 统计 | 已完成 |
| 阶段 3 | generator ablation + local sweep | 基本完成，正在收尾固化 |
| 阶段 4 | public human trajectory external validation | 尚未开始，下一大阶段 |

---

## 8. 阶段 1：PR2 interpretability demo

### 8.1 目标

做一个人类能直观看懂的 demo，展示 embedding 是否能区分不同 policy / driving style。

### 8.2 已完成内容

已经支持：

- 加载 `source_index.npy`；
- 加载 `policy_id.npy`；
- 加载 `policy_name.npy`；
- source group 按 source_index 对齐；
- policy-aware PCA / UMAP；
- within-source triplet 展示；
- within-source style signals；
- embedding distance matrix；
- global retrieval cards；
- global retrieval style signals；
- 自动生成 interpretability report。

### 8.3 当前正式保留输出

```text
embedding_2d_projection.png
embedding_2d_projection_umap.png
embedding_distance_matrix.png
within_source_triplet.png
within_source_style_signals.png
within_source_style_fingerprint_dynamics.png
within_source_style_fingerprint_kinematic.png
within_source_style_fingerprint_normalized.png
global_retrieval_cards.png
global_retrieval_style_signals.png
interpretability_report.md
summary.json
retrieval_table.csv
style_fingerprint.csv
```

### 8.4 当前结论

PR2 已经可以作为人类可解释展示工具，不建议继续无限美化图。

### 8.5 任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| policy_id/source_index metadata 修复 | 完成 | 从 generator 到 demo 已打通 |
| within-source triplet 图 | 完成 | 可展示同 source 下三 policy 轨迹 |
| embedding distance matrix | 完成 | 可解释高维 embedding 距离 |
| policy-colored PCA/UMAP | 完成 | 仅作为 visualization，不作为强证据 |
| global retrieval demo | 完成 | 可展示 Top-K 与 query 的行为相似性 |
| interpretability_report.md | 完成 | 可自动记录 query/source/retrieval/limitations |

---

## 9. 阶段 2：population-level 统计

### 9.1 目标

从单个 hero case 扩展到全部 test sources，证明结果不是偶然。

### 9.2 已完成内容

实现了 population evaluator，输出：

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

### 9.3 数据完整性结论

当前 test split 数据结构正确：

```text
395 sources × 3 policies = 1185 samples
```

每个 source 都有 p0/p1/p2 三条 rollout。

### 9.4 核心结果

阶段 2 的关键发现：

- centroid classification accuracy 明显高于 chance；
- global retrieval hit@1 / hit@5 很好；
- embedding 具备 policy-level discriminability；
- 但 p2/lateral_stable 在初始配置下并不是独立第三极，更接近 conservative。

典型结论：

```text
centroid classification accuracy ≈ 0.64
chance = 0.333
retrieval hit@1 ≈ 0.82
retrieval hit@5 ≈ 0.98
p2_farthest_rate ≈ 0.05
mean_p2_separation_margin < 0
```

### 9.5 阶段 2 的学术意义

阶段 2 证明了：

> embedding 能区分 policy，也能检索同 policy / 相似 behavior。

同时也发现了：

> 原始 lateral_stable 不足以成为独立第三类，需要 generator ablation 和参数优化。

### 9.6 任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| population evaluator | 完成 | 支持全量 source 统计 |
| pairwise distance distribution | 完成 | p0-p1 / p0-p2 / p1-p2 |
| p2_farthest_rate | 完成 | 发现初始 p2 独立性不足 |
| centroid classification | 完成 | 明显高于 chance |
| global retrieval hit@k | 完成 | 检索能力较强 |
| style-distance correlation | 完成 | jerk delta 与 embedding distance 相关较明显 |

---

## 10. 阶段 3：generator ablation 与 local fine sweep

### 10.1 目标

证明 lateral_stable 的差异来自明确 generator 机制，而不是偶然。

重点机制包括：

- heading delta clip / yaw_rate_clip；
- heading_smooth_alpha；
- thw_target；
- jerk_limit；
- a_max / a_min。

---

## 10.2 阶段 3A：broad ablation

### 已比较配置

```text
baseline_current
no_lateral_smoothing
weak_lateral_stable
strong_yaw_clip
strong_heading_smoothing
comfort_only
lateral_only
full_strong_lateral_stable
```

### 工程修复

初版 ablation 出现过严重问题：所有 config 指标完全一样。

后来增加了完整性检查：

```text
effective_config.json
file_fingerprints.json
ablation_integrity_report.json
ablation_rollout_sanity.csv
--overwrite
hash check
```

确保不同 config 真的产生不同 rollout / embedding。

### broad ablation 结论

- `full_strong_lateral_stable` 是 broad ablation 最优；
- `strong_yaw_clip` 是最关键的单项机制；
- `comfort_only` 不足以形成独立 lateral_stable；
- `lateral_only` 会破坏舒适性和可分性；
- lateral_stable 需要横向稳定 + 纵向舒适联合塑形。

---

## 10.3 阶段 3B：local fine sweep

### 目标

围绕 broad ablation 最优配置 `full_strong_lateral_stable` 做细粒度搜索。

### 搜索重点

- yaw_rate_clip；
- jerk_limit；
- heading_smooth_alpha；
- thw_target。

### local sweep 最优配置

```text
yaw_008_jerk_020
```

参数：

```text
heading_smooth_alpha = 0.75
yaw_rate_clip = 0.008
thw_target = 1.70
jerk_limit = 0.200
a_max = 1.275
a_min = -2.52
```

这个配置后来被固化为：

```text
recommended_lateral_stable_v2
```

### local sweep 结论

相比 `full_strong_lateral_stable`，`yaw_008_jerk_020`：

- 提升 p2_farthest_rate；
- 改善 mean_p2_separation_margin；
- 显著提升 centroid_accuracy_p2；
- 提升 retrieval same-policy fraction；
- 降低 p2_rms_jerk；
- 降低 p2_rms_yaw_rate_proxy；
- 基本保持 THW。

---

## 10.4 阶段 3C：final compare / recommended_lateral_stable_v2 固化

### 目标

形成论文级最终三配置对比表：

```text
baseline_current
full_strong_lateral_stable
recommended_lateral_stable_v2
```

### 当前已完成结果

`recommended_lateral_stable_v2` 参数：

```text
heading_smooth_alpha = 0.75
yaw_rate_clip = 0.008
thw_target = 1.70
jerk_limit = 0.200
a_max = 1.275
a_min = -2.52
```

三配置对比显示：

| 指标 | baseline_current | full_strong_lateral_stable | recommended_lateral_stable_v2 |
|---|---:|---:|---:|
| p2_farthest_rate | 0.0489 | 0.0810 | 0.0954 |
| mean_p2_separation_margin | -2.3983 | -2.1522 | -1.9354 |
| centroid_accuracy_p2 | 0.6354 | 0.7283 | 0.8439 |
| retrieval_hit@1 | 0.8127 | 0.8197 | 0.8298 |
| retrieval_hit@k | 0.9218 | 0.9153 | 0.9294 |
| same-policy fraction@TopK | 0.7896 | 0.7983 | 0.8102 |
| p2_rms_jerk | 1.4173 | 1.2421 | 1.1441 |
| p2_rms_yaw_rate_proxy | 0.0211 | 0.0151 | 0.0139 |
| p2_rms_curvature_proxy | 0.00334 | 0.00239 | 0.00214 |
| p2_mean_thw | 1.4308 | 1.5067 | 1.5093 |

### 阶段 3 当前结论

`recommended_lateral_stable_v2` 已经可以作为后续实验的推荐配置。

它通过更紧的 yaw-rate clipping 和更严格的 jerk limitation：

- 显著提升 p2/lateral_stable 的可识别性；
- 提升检索一致性；
- 降低 jerk；
- 降低 yaw_rate_proxy；
- 降低 curvature proxy；
- 保持/提升 THW。

但需要谨慎表达：

> p2 independence is improved but incomplete.

因为：

```text
mean_p2_separation_margin 仍然为负。
```

### 阶段 3 任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| broad ablation pipeline | 完成 | 已支持多配置对比 |
| integrity check | 完成 | 防止不同 config 复用同一输出 |
| broad ablation 分析 | 完成 | 找到 full_strong_lateral_stable |
| local fine sweep | 完成 | 找到 yaw_008_jerk_020 |
| recommended_lateral_stable_v2 固化 | 完成/待代码最终确认 | 已生成 final compare 输出 |
| final comparison summary | 完成 | 三配置论文级对比表已生成 |
| README 更新 | 待最终确认 | 每次代码变更需同步更新 |

---

## 11. 阶段 4：public human trajectory external validation

### 11.1 阶段 4 尚未开始

这是下一大阶段。

### 11.2 为什么必须做

当前最大的学术风险是：

> synthetic policy 过于规则化，embedding 可能只是在识别 generator artifact。

因此需要用公开真实人类轨迹做 external validation。

### 11.3 阶段 4 目标

证明：

> embedding 不仅能区分 synthetic policy，在真实 human driving trajectory 上也能形成可解释的 behavior structure。

### 11.4 可选数据集

候选：

```text
Waymo Open Motion Dataset
Argoverse Motion Forecasting
nuScenes prediction / tracking
INTERACTION
highD / inD
```

建议优先使用与当前工程兼容度最高的数据。

### 11.5 推荐方案：pseudo-label validation

真实数据没有 policy label，因此构造 pseudo style labels：

#### aggressive-like

```text
high mean speed
low THW
high accel / jerk
```

#### conservative-like

```text
low speed
high THW
low jerk
```

#### lateral-stable-like

```text
low yaw_rate RMS
low curvature RMS
smooth heading
```

### 11.6 需要验证的指标

- pseudo-label classification；
- same pseudo-label global retrieval；
- embedding distance vs style delta correlation；
- cluster style fingerprint；
- retrieval case visualization；
- baselines comparison。

### 11.7 阶段 4 任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| 数据集选择 | 未开始 | 需要确定 Waymo / Argoverse / nuScenes 等 |
| human trajectory 数据转换 | 未开始 | 需要转成当前 traj/front/meta/split 格式 |
| pseudo-label 规则定义 | 未开始 | 需要避免过拟合和规则循环论证 |
| human embedding 生成 | 未开始 | 复用现有 embedding 或单独导出 |
| pseudo-label classification | 未开始 | 验证 embedding 是否编码真实风格 |
| retrieval validation | 未开始 | 检索同 pseudo style 样本 |
| cluster fingerprint | 未开始 | 检查聚类是否可解释 |
| baseline comparison | 未开始 | 顶会必要 |

---

## 12. 必须补的 baseline

如果要冲高水平会议，必须补 baseline。

当前建议 baseline：

| baseline | 用途 |
|---|---|
| raw handcrafted feature distance | 判断 embedding 是否只是 feature 包装 |
| feature-only retrieval | 对比 learned embedding retrieval |
| trajectory distance | 轨迹几何距离 baseline |
| DTW / Frechet distance | 序列轨迹距离 baseline |
| random embedding | 随机对照 |
| untrained encoder | 模型结构随机初始化对照 |
| PCA feature embedding | 简单降维 baseline |

### baseline 当前状态

| 任务 | 状态 |
|---|---|
| baseline 设计 | 初步提出 |
| baseline 代码 | 未开始 |
| baseline 实验 | 未开始 |
| baseline 论文表格 | 未开始 |

---

## 13. 当前可写入论文的核心结论

### 13.1 synthetic controlled validation

> The learned behavior embedding is policy-discriminative and retrieval-capable under controlled synthetic rollout settings.

### 13.2 within-source aligned evaluation

> Within-source comparison controls scene variation and allows policy-induced behavior differences to be measured directly.

### 13.3 lateral_stable 机制结论

> Lateral-stable behavior requires joint lateral and longitudinal shaping. Stronger yaw-rate clipping and stricter jerk limitation improve p2 recognizability, retrieval consistency, yaw-rate stability, and longitudinal comfort.

### 13.4 推荐配置结论

> `recommended_lateral_stable_v2` outperforms both the original baseline and full_strong_lateral_stable in p2 classification accuracy, retrieval consistency, separation margin, jerk, and yaw-rate proxy.

### 13.5 谨慎限制

> However, p2 independence remains incomplete because the average p2 separation margin is still negative.

---

## 14. 当前不应该夸大的结论

不能写：

```text
lateral_stable 已经完全成为独立第三类驾驶风格。
```

不能写：

```text
embedding 已经证明适用于真实人类驾驶风格。
```

不能写：

```text
该 benchmark 等价于完整自动驾驶闭环仿真。
```

应该写：

```text
lateral_stable 的独立性显著增强，但仍未完全成立。
```

```text
当前结论主要来自 controlled synthetic rollout，下一步需要 public human trajectory external validation。
```

```text
当前 benchmark 是 trajectory-level closed-loop policy behavior evaluation，不包含 sensor rendering / perception stack。
```

---

## 15. 当前论文成熟度判断

当前状态：

```text
已有论文雏形和投稿潜力，但还不是顶会-ready。
```

强项：

- 问题真实；
- trajectory-level benchmark 定位清楚；
- controlled rollout generator 有价值；
- aligned within-source evaluation 很关键；
- PR2 demo / population eval / ablation / local sweep 已成体系；
- recommended_lateral_stable_v2 结果清晰。

短板：

- 主要还是 synthetic policy validation；
- public human trajectory external validation 未做；
- baselines 未补齐；
- p2 independence 仍不完全；
- benchmark package 需要整理成论文协议。

顶会判断：

| 当前状态 | 可能性 |
|---|---|
| 现在直接投顶会主会 | 偏低 |
| 补完 public validation + baselines | 有机会冲强会/顶会 workshop，主会机会提升 |
| 再接入真实 planner rollout / public planner benchmark | 顶会主会机会进一步提升 |

---

## 16. 接下来最优先任务

### P0：完成阶段 3 收尾

- 确认 `recommended_lateral_stable_v2` 已固化；
- 确认 `final_compare` 三配置对比输出稳定；
- 确认 README 已更新；
- 整理 final comparison 表格和图。

### P1：启动阶段 4 public human trajectory validation

需要先设计文档，不要直接写代码。

待确定：

- 选哪个公开数据集；
- 如何转成当前统一格式；
- pseudo-label 如何定义；
- 是否复用当前 embedding；
- 需要哪些 baseline；
- 评价指标和图表输出。

### P2：设计 baseline suite

先做最小 baseline：

```text
raw feature distance
trajectory distance
random embedding
feature-only retrieval
```

### P3：论文结构草稿

建议开始维护：

```text
paper_outline.md
experiment_table.md
```

---

## 17. 建议后续新增文档

建议在仓库中逐步增加：

```text
00_plans.md                          # 当前总计划与进展
01_experiment_1_population_eval.md    # 阶段 2 详细总结
02_experiment_2_ablation.md           # 阶段 3 broad ablation 总结
03_experiment_2b_local_sweep.md       # local sweep 总结
04_vehicledata_validation_plan.md     # 阶段 4 public/human 数据验证计划
05_baseline_plan.md                   # baseline suite 计划
paper_outline.md                      # 论文结构草稿
```

---

## 18. 当前项目状态总览

| 模块 | 当前状态 | 下一步 |
|---|---|---|
| synthetic policy generator | 可用 | 使用 recommended_lateral_stable_v2 |
| PR2 interpretability demo | 完成 | 保持，不再过度美化 |
| population evaluator | 完成 | 后续可复用于 v2 / human validation |
| broad ablation | 完成 | 已得机制结论 |
| local fine sweep | 完成 | 已得 v2 配置 |
| final compare | 完成/待代码最终确认 | 作为论文表格 |
| README | 待确认 | 必须同步最新命令 |
| public human trajectory validation | 未开始 | 下一大阶段 |
| baselines | 未开始 | 顶会必要 |
| paper outline | 未开始 | 建议尽快启动 |

---

## 19. 最后结论

截至目前，项目已经完成了从单案例 demo 到全量统计、再到 generator 机制 ablation 和推荐配置优化的完整链路。

当前最重要成果是：

> `recommended_lateral_stable_v2` 已经显著优于 baseline_current 和 full_strong_lateral_stable，可以作为后续实验默认推荐配置。

当前最重要限制是：

> 结论仍主要来自 synthetic controlled rollout，必须通过 public human trajectory validation 和 baseline suite 来提升论文可信度。

下一阶段建议正式进入：

```text
阶段 4：public human trajectory external validation
```

目标是把当前工作从 synthetic benchmark 推进到更有顶会潜力的 behavior evaluation benchmark。
