# paper_outline — Trajectory-level Behavior Evaluation Benchmark 论文结构草稿

> 更新时间：2026-05-07  
> 当前阶段：阶段 3 收尾 / 阶段 4 规划中  
> 当前定位：自动驾驶端到端（E2E）决策/规划 policy 的 trajectory-level behavior evaluation benchmark

---

# 论文暂定标题（候选）

## 候选 1

```text
A Trajectory-level Behavior Evaluation Benchmark for Closed-loop Planning Policies
```

当前最推荐。

特点：

- 不把贡献局限为 embedding model；
- 强调 benchmark / evaluation；
- 不限定必须是 E2E planner；
- trajectory-level 边界清晰。

---

## 候选 2

```text
Behavior Embedding and Style Evaluation for Closed-loop Planning Policies
```

更偏 representation learning。

风险：

- 容易被要求更强模型创新；
- benchmark/evaluator 贡献会被弱化。

---

## 候选 3

```text
Evaluating Driving Style in Trajectory-level Closed-loop Planning Policies
```

更偏 behavior/style evaluation。

适合：

- 如果最终 embedding 模型创新不强；
- 更强调 interpretability / retrieval / report card。

---

# 1. Introduction

## 1.1 背景

端到端（E2E）自动驾驶决策/规划模型越来越表现出类似人类驾驶员的行为特征：

- 不同模型版本会呈现不同驾驶风格；
- 有的更激进；
- 有的更保守；
- 有的更关注舒适性；
- 行为变化可能明显影响用户体验和风险。

但当前主流指标：

```text
ADE
FDE
collision rate
off-road rate
rule violation
```

很难描述：

```text
style drift
comfort
aggressiveness
yielding tendency
behavior similarity
```

---

## 1.2 问题定义

提出问题：

> Can we evaluate planning policies using trajectory-level behavior representations rather than only rule-based metrics?

进一步提出：

- policy behavior embedding；
- policy retrieval；
- style fingerprint；
- behavior distance / BDD；
- aligned within-source evaluation。

---

## 1.3 关键挑战

### 挑战 1

不同场景之间的 variation 会掩盖 policy 差异。

### 挑战 2

真实 E2E 系统难以大规模获得闭环 rollout。

### 挑战 3

缺少 trajectory-level behavior benchmark。

### 挑战 4

缺少 behavior-level interpretability。

---

## 1.4 本文核心思路

本文提出：

```text
controlled synthetic policy rollout generator
+ behavior embedding
+ aligned within-source evaluation
+ retrieval / style fingerprint / report card
```

在 trajectory-level closed-loop setting 下评价 policy behavior。

---

## 1.5 本文贡献（当前版本）

建议贡献写法：

### Contribution 1

提出 trajectory-level behavior evaluation benchmark：

- 不依赖 sensor rendering；
- 不依赖 perception stack；
- 接受任意 trajectory rollout。

### Contribution 2

提出 controlled synthetic policy rollout generator：

- 同一 source 下生成多种 policy rollout；
- aligned comparison 控制场景变量；
- 支持 policy separation evaluation。

### Contribution 3

提出 behavior embedding evaluation pipeline：

- within-source pairwise separation；
- centroid classification；
- global retrieval；
- style fingerprint；
- behavior report card。

### Contribution 4

通过 broad ablation 和 local fine sweep，分析 lateral_stable 的机制：

- yaw-rate clipping；
- heading smoothing；
- jerk limitation；
- longitudinal comfort shaping。

### Contribution 5（阶段 4 后补）

在公开真实人类轨迹数据上验证 embedding behavior structure。

---

# 2. Related Work

建议分为：

## 2.1 Autonomous Driving Evaluation

- ADE/FDE
- collision metrics
- rule-based evaluation
- simulation-based evaluation

指出问题：

> 传统指标缺少 behavior/style representation。

---

## 2.2 Behavior Cloning / Driving Style Modeling

- driver style modeling
- aggressive vs conservative driving
- driver embeddings
- imitation learning style control

区别：

> 本文重点不是 imitation，而是 evaluation benchmark。

---

## 2.3 Representation Learning

- contrastive learning
- trajectory embedding
- retrieval-based representation

说明：

> 本文 embedding 更强调 evaluation / interpretability，而不是 SOTA representation learning。

---

## 2.4 Trajectory Similarity / Retrieval

- DTW
- Frechet distance
- retrieval systems

作为 baseline 对照。

---

# 3. Problem Formulation

## 3.1 Trajectory-level closed-loop setting

定义：

```text
source window
→ policy rollout
→ trajectory sequence
→ embedding
```

---

## 3.2 Source / within-source

定义：

```text
source = scenario_id + start + window_len + front_id
```

within-source：

同一 source 下比较不同 policy。

---

## 3.3 Policy set

当前：

```text
p0 = conservative
p1 = aggressive
p2 = lateral_stable
```

---

## 3.4 Behavior embedding

定义：

```text
trajectory window -> z ∈ R^D
```

embedding distance 代表 behavior similarity，而不是物理米制距离。

---

# 4. Controlled Synthetic Policy Rollout Generator

## 4.1 Generator overview

输入：

```text
source window
```

输出：

```text
p0/p1/p2 rollout
```

---

## 4.2 Conservative policy

描述：

- larger THW；
- lower accel；
- smoother braking。

---

## 4.3 Aggressive policy

描述：

- smaller THW；
- stronger accel；
- stronger braking。

---

## 4.4 Lateral_stable policy

核心机制：

```text
heading_smooth_alpha
yaw_rate_clip
thw_target
jerk_limit
a_max / a_min
```

强调：

- 横向更稳；
- 纵向更平顺；
- 不只是 conservative。

---

## 4.5 Recommended lateral_stable v2

当前推荐：

```text
heading_smooth_alpha = 0.75
yaw_rate_clip = 0.008
thw_target = 1.70
jerk_limit = 0.200
a_max = 1.275
a_min = -2.52
```

说明其来自 broad ablation + local fine sweep。

---

# 5. Behavior Embedding and Evaluation Pipeline

## 5.1 Embedding generation

介绍：

- embedding 输入；
- feature normalization；
- optional contrastive learning；
- current representation choice。

注意：

不要把论文重点放在模型创新。

---

## 5.2 Within-source aligned evaluation

定义：

```text
d(p0,p1)
d(p0,p2)
d(p1,p2)
```

以及：

```text
p2_farthest_rate
p2_separation_margin
```

---

## 5.3 Centroid classification

train centroid → eval classification。

---

## 5.4 Global retrieval

定义：

```text
query embedding -> Top-K nearest neighbors
```

强调：

- retrieval 不只是同 source；
- 更强调跨场景相似 behavior。

---

## 5.5 Style fingerprint

包括：

```text
mean_speed
rms_jerk
rms_yaw_rate
rms_curvature
mean_thw
```

用于解释 embedding cluster / retrieval case。

---

# 6. Interpretability Demo

对应阶段 1 / PR2。

展示：

```text
embedding_2d_projection
within_source_triplet
embedding_distance_matrix
retrieval_cards
style_fingerprint
```

重点：

> embedding 是否能以人类可理解方式展示 style difference。

---

# 7. Population-level Evaluation

对应阶段 2。

## 7.1 Dataset statistics

```text
395 sources × 3 policies = 1185 samples
```

---

## 7.2 Policy separation

展示：

```text
pairwise distance distributions
p2 separation margin
p2 farthest rate
```

---

## 7.3 Classification and retrieval

展示：

```text
centroid accuracy
retrieval hit@1 / hit@5
same-policy fraction
```

---

## 7.4 Style-distance correlation

展示 embedding distance 与 jerk/yaw/THW delta 的相关性。

---

# 8. Generator Ablation and Local Fine Sweep

对应阶段 3。

## 8.1 Broad ablation

展示：

```text
baseline_current
strong_yaw_clip
comfort_only
lateral_only
full_strong_lateral_stable
```

结论：

- yaw-rate clip 是关键机制；
- comfort_only 不够；
- lateral_only 不够；
- lateral + longitudinal shaping 必须联合。

---

## 8.2 Local fine sweep

展示：

```text
yaw_008_jerk_020
```

成为：

```text
recommended_lateral_stable_v2
```

---

## 8.3 Final compare

最终表格：

```text
baseline_current
full_strong_lateral_stable
recommended_lateral_stable_v2
```

---

## 8.4 Key limitation

必须诚实写：

```text
mean_p2_separation_margin remains negative.
```

说明：

> p2 independence is improved but incomplete.

---

# 9. Public Human Trajectory External Validation

对应阶段 4。

当前还未完成。

---

## 9.1 Motivation

解决：

> synthetic generator artifact 风险。

---

## 9.2 Pseudo-label validation

定义：

```text
aggressive-like
conservative-like
lateral-stable-like
```

---

## 9.3 Human retrieval and clustering

验证：

- same pseudo-label retrieval；
- cluster style fingerprint；
- style-distance correlation。

---

## 9.4 Baseline comparison

与：

```text
raw feature
trajectory distance
random embedding
PCA feature
```

对比。

---

# 10. Discussion

建议讨论：

## 10.1 Why within-source matters

控制场景 variation。

---

## 10.2 Why trajectory-level evaluation is useful

适合：

- planner iteration；
- E2E version drift；
- comfort/style analysis。

---

## 10.3 Current limitations

必须诚实：

```text
1. synthetic rollout
2. no full multi-agent simulation
3. no sensor rendering
4. p2 independence incomplete
5. pseudo labels are weak supervision
```

---

## 10.4 Future directions

包括：

- real planner rollout；
- real E2E policy comparison；
- multi-agent closed-loop；
- learned style report card；
- behavior drift tracking。

---

# 11. Conclusion

建议核心结论：

> We present a trajectory-level behavior evaluation benchmark for closed-loop planning policies. Using controlled synthetic policy rollouts and aligned within-source evaluation, we show that behavior embeddings can capture policy-level driving behavior differences and support classification, retrieval, and interpretable style analysis. Through broad ablation and local fine sweep, we identify joint lateral and longitudinal shaping as the key mechanism for lateral_stable behavior. The proposed recommended_lateral_stable_v2 improves p2 recognizability, retrieval consistency, yaw-rate stability, and jerk comfort, while public human trajectory validation is identified as the next critical step toward stronger generalization.

---

# 附录建议

## Appendix A

Implementation details。

---

## Appendix B

More retrieval cases。

---

## Appendix C

More ablation tables。

---

## Appendix D

Pseudo-label threshold sensitivity。

---

## Appendix E

Dataset conversion details。


## Stage 4A/4B/4C/4D 更新（2026-05）
- Stage 4A：确认 data1 为 synthetic_rollout scaffold，不可作为公开人类验证证据。
- Stage 4B：Waymo human builder 已完成，full51 抽取 168191 条窗口，来自 24872 scenarios。
- Stage 4C：full51 baseline-only 已完成，learned 尚未在 human_public full51 上评估。
- Stage 4D：新增 row-level learned embedding 训练/导出与 learned-vs-baselines 对比能力。
- 反泄漏声明：pseudo labels 为弱标签且来自规则/特征，必须联合 strict retrieval、baseline、style correlation、cluster fingerprint 解读。


## 新增结果表
- Dataset statistics
- Pseudo label distribution
- Learned vs baselines
- Style-distance correlation（含 valid pair count）
- Training/export summary
