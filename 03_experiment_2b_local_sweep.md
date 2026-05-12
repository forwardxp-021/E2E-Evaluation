# 03_experiment_2b_local_sweep — Local Fine Sweep 与 recommended_lateral_stable_v2

> 更新时间：2026-05-07  
> 所属阶段：阶段 3B / 3C  
> 目的：围绕 broad ablation 最优配置做局部精细搜索，并固化推荐 lateral_stable v2 配置。

---

## 1. 实验背景

阶段 3A broad ablation 发现：

- `full_strong_lateral_stable` 是 broad ablation 阶段最优；
- `strong_yaw_clip` 是最关键的单项机制；
- `comfort_only` 和 `lateral_only` 都不足以形成理想 lateral_stable；
- p2/lateral_stable 的可识别性和稳定性提升了，但 mean p2 separation margin 仍为负。

因此阶段 3B 的目标是：

> 围绕 `full_strong_lateral_stable` 做局部精细 sweep，寻找更好的 p2/lateral_stable 参数组合。

---

## 2. Local sweep 中心配置

中心配置为：

```text
local_center_full_strong
```

参数：

```text
heading_smooth_alpha = 0.75
yaw_rate_clip = 0.010
thw_target = 1.70
jerk_limit = 0.245
a_max = 1.275
a_min = -2.52
```

该配置来自 broad ablation 的 `full_strong_lateral_stable`。

---

## 3. Local sweep 搜索范围

局部搜索重点：

```text
yaw_rate_clip
heading_smooth_alpha
thw_target
jerk_limit
```

主要候选包括：

```text
yaw_006
yaw_008
yaw_012
yaw_015
alpha_065
alpha_085
thw_150
thw_190
jerk_020
jerk_030
yaw_008_alpha_085
yaw_008_thw_190
yaw_006_thw_190
yaw_008_jerk_020
balanced_strong
```

设计原则：

- 不做大规模全因子搜索；
- 围绕当前最优点做小范围探索；
- 重点观察 p2 separation 与 comfort/stability 的 tradeoff。

---

## 4. Local sweep 最优配置

local sweep 推荐的最佳配置为：

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

该配置后来固化为：

```text
recommended_lateral_stable_v2
```

---

## 5. 与 local center 的对比

相对于 `local_center_full_strong`，`yaw_008_jerk_020` 的核心变化是：

```text
yaw_rate_clip: 0.010 -> 0.008
jerk_limit:    0.245 -> 0.200
```

也就是：

- 横向 yaw-rate clip 更紧；
- 纵向 jerk limit 更严格。

典型改善：

| 指标 | local_center_full_strong | yaw_008_jerk_020 / recommended_v2 | 结论 |
|---|---:|---:|---|
| p2_farthest_rate | 0.0810 | 0.0954 | 提升 |
| mean_p2_separation_margin | -2.1522 | -1.9354 | 更接近 0 |
| centroid_accuracy_p2 | 0.7283 | 0.8439 | 大幅提升 |
| retrieval same-policy fraction | 0.7983 | 0.8102 | 提升 |
| p2_rms_jerk_mean | 1.2421 | 1.1441 | 降低，纵向更平顺 |
| p2_rms_yaw_rate_proxy_mean | 0.0151 | 0.0139 | 降低，横向更稳 |
| p2_mean_thw | 1.5067 | 1.5093 | 基本保持 |

---

## 6. Final compare 三配置对比

阶段 3C 对以下三种配置做最终对比：

```text
baseline_current
full_strong_lateral_stable
recommended_lateral_stable_v2
```

`recommended_lateral_stable_v2` 参数：

```text
heading_smooth_alpha = 0.75
yaw_rate_clip = 0.008
thw_target = 1.70
jerk_limit = 0.200
a_max = 1.275
a_min = -2.52
```

最终对比结果：

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

---

## 7. 核心机制结论

local sweep 说明：

> 更紧的 yaw-rate clipping + 更严格的 jerk limitation 是提升 lateral_stable 的关键组合。

具体而言：

1. 降低 `yaw_rate_clip` 能增强横向稳定性；
2. 降低 `jerk_limit` 能增强纵向平顺性；
3. 二者联合比单独调某一个更有效；
4. 过大的 `thw_target` 不一定更好，可能让 p2 更像 conservative；
5. p2 的可识别性、检索一致性和 style stability 可以同步改善。

---

## 8. 当前推荐配置

后续 synthetic policy rollout 默认推荐使用：

```text
recommended_lateral_stable_v2
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

用途：

- 后续 population-level evaluation；
- interpretability demo；
- external validation 前的 synthetic benchmark 固定配置；
- 论文 ablation table。

---

## 9. 重要限制

虽然 recommended_lateral_stable_v2 明显优于 baseline 和 full_strong，但仍需谨慎表述。

不能说：

```text
p2/lateral_stable 已经完全成为独立第三类。
```

因为：

```text
mean_p2_separation_margin 仍然为负。
```

应该说：

```text
recommended_lateral_stable_v2 significantly improves p2 recognizability, retrieval consistency, yaw-rate stability, and jerk comfort, but p2 independence remains incomplete.
```

中文：

> recommended_lateral_stable_v2 显著提升了 p2 的可识别性、检索一致性、横向稳定性和纵向平顺性，但 p2 的独立性仍未完全成立。

---

## 10. 论文可用结论

可以写入论文：

> Local fine-grained sweep shows that reducing yaw_rate_clip from 0.010 to 0.008 and tightening jerk_limit from 0.245 to 0.200 yields the best lateral_stable configuration. The resulting recommended_lateral_stable_v2 improves p2 centroid accuracy, global retrieval consistency, p2 separation margin, jerk comfort, and yaw-rate stability compared with both the original baseline and the broad-ablation full-strong configuration. However, the average p2 separation margin remains negative, indicating that p2 independence is improved but not complete.

---

## 11. 当前任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| local fine sweep | 完成 | 16 个局部配置 |
| yaw_008_jerk_020 选择 | 完成 | local sweep 最优 |
| recommended_lateral_stable_v2 固化 | 完成/待代码最终确认 | 已生成 final compare 输出 |
| final compare | 完成 | 三配置对比表可用于论文 |
| final compare plots | 完成 | margin / retrieval / tradeoff 等 |
| README 更新 | 待最终确认 | 必须同步最新命令 |
| 阶段 3 总结 | 完成 | 可以进入阶段 4 |
