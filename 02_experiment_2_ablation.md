# 02_experiment_2_ablation — Lateral_stable Broad Ablation

> 更新时间：2026-05-07  
> 所属阶段：阶段 3A  
> 目的：验证 lateral_stable 的差异是否来自明确 generator 机制，而不是偶然现象。

---

## 1. 实验背景

阶段 2 的 population-level evaluation 表明：

- behavior embedding 已具备 policy-level 可分类性；
- global retrieval 能较好找回同 policy 样本；
- 但原始 p2/lateral_stable 并没有稳定形成独立第三类风格；
- p2 更接近 conservative，而不是同时远离 conservative 和 aggressive。

因此阶段 3A 的 broad ablation 目标是：

> 分解 lateral_stable 的控制机制，验证哪些参数真正改善 p2 的可识别性、横向稳定性、纵向平顺性和 embedding separation。

---

## 2. 被验证的机制

当前 lateral_stable 相关机制包括：

```text
heading_smooth_alpha
per-step heading delta clip / yaw_rate_clip
thw_target
jerk_limit
a_max
a_min
```

其中：

- `yaw_rate_clip` / heading delta clip 主要控制横向角速度变化；
- `heading_smooth_alpha` 控制 desired heading 的 EMA smoothing；
- `thw_target` 控制期望时距；
- `jerk_limit` 控制纵向舒适性；
- `a_max / a_min` 控制加速度上下界。

---

## 3. Broad ablation configs

阶段 3A 比较了以下配置：

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

各配置含义：

| config | 目的 |
|---|---|
| baseline_current | 当前默认 lateral_stable 设置 |
| no_lateral_smoothing | 关闭 heading smoothing |
| weak_lateral_stable | 弱化 lateral stable 约束 |
| strong_yaw_clip | 强化 yaw_rate / heading delta 限制 |
| strong_heading_smoothing | 强化 heading smoothing |
| comfort_only | 只保留纵向舒适塑形 |
| lateral_only | 只保留横向稳定控制，弱化纵向舒适 |
| full_strong_lateral_stable | 横向稳定 + 纵向舒适联合增强 |

---

## 4. 工程完整性修复

初版 broad ablation 出现过一个严重问题：

> 8 个 config 的所有指标完全一样。

这说明当时 ablation 可能复用了同一份 rollout / embedding，或参数没有真正传入 generator。

后来增加了完整性检查：

```text
effective_config.json
file_fingerprints.json
rollout_sanity_summary.json
ablation_integrity_report.json
ablation_rollout_sanity.csv
--overwrite
```

这些检查用于确认：

- 每个 config 的输出目录独立；
- 每个 config 的 effective parameters 被记录；
- 不同 config 的 `traj.npy` / `feat_style.npy` / `population_summary.json` 不应全部相同；
- 若所有 config 输出完全相同，则 ablation 判为 invalid。

这是该阶段非常重要的工程经验：

> ablation 不是只看配置名不同，必须证明不同配置真的改变了 rollout 或 embedding。

---

## 5. Broad ablation 核心结果

修复完整性问题后，不同 config 的指标开始出现真实差异。

### 5.1 full_strong_lateral_stable 成为 broad ablation 最优

相对于 baseline_current，`full_strong_lateral_stable` 改善了：

- `p2_farthest_rate`；
- `mean_p2_separation_margin`；
- `centroid_accuracy_p2`；
- `retrieval_mean_same_policy_fraction_topk`；
- `p2_rms_jerk_mean`；
- `p2_rms_yaw_rate_proxy_mean`；
- `p2_mean_thw`。

典型结果：

```text
baseline_current:
  p2_farthest_rate ≈ 0.0489
  mean_p2_separation_margin ≈ -2.3983
  p2_rms_yaw_rate_proxy_mean ≈ 0.0211
  p2_rms_jerk_mean ≈ 1.4173
  centroid_accuracy_p2 ≈ 0.6354

full_strong_lateral_stable:
  p2_farthest_rate ≈ 0.0810
  mean_p2_separation_margin ≈ -2.1522
  p2_rms_yaw_rate_proxy_mean ≈ 0.0151
  p2_rms_jerk_mean ≈ 1.2421
  centroid_accuracy_p2 ≈ 0.7283
```

结论：

> full_strong_lateral_stable 显著提升 p2 的可识别性、横向稳定性和纵向舒适性。

---

### 5.2 strong_yaw_clip 是最关键单项机制

`strong_yaw_clip` 明显改善：

- p2_farthest_rate；
- p2 separation margin；
- p2 yaw_rate proxy；
- p2 classification。

结论：

> 限制 heading delta / yaw rate 是制造 lateral_stable 风格最有效的单项机制。

---

### 5.3 heading_smoothing 单独增强不够

`strong_heading_smoothing` 对 yaw_rate 有轻微改善，但对 p2 独立性帮助有限。

结论：

> heading smoothing 单独不是关键，必须与 yaw clip / comfort shaping 组合使用。

---

### 5.4 comfort_only 不是好方向

`comfort_only` 只做纵向舒适，不做 lateral 约束，结果显示：

- p2_farthest_rate 下降；
- separation margin 变差；
- p2 yaw_rate 变差；
- p2 classification 下降。

结论：

> Longitudinal comfort shaping alone is insufficient to create a distinct lateral-stable behavior mode.

---

### 5.5 lateral_only 明显失败

`lateral_only` 只保留横向控制，弱化纵向舒适，导致：

- p2 jerk 明显升高；
- p2 centroid accuracy 接近随机；
- retrieval 下降；
- p2 独立性没有改善。

结论：

> lateral_stable 不是单纯横向控制，必须横向稳定 + 纵向舒适一起做。

---

## 6. Broad ablation 论文结论

阶段 3A 可以写成：

> Experiment 2 shows that lateral_stable behavior requires joint lateral and longitudinal shaping. Strong yaw-rate clipping improves p2 separation and reduces lateral variation, while comfort-only and lateral-only variants are insufficient. The best broad-ablation configuration is full_strong_lateral_stable, which improves p2 recognizability, jerk comfort, and yaw-rate stability. However, the p2 separation margin remains negative, indicating that p2 is more distinguishable but not yet a fully independent third style.

中文：

> lateral_stable 需要横向稳定和纵向舒适联合塑形。强 yaw-rate clip 是关键机制；comfort_only 与 lateral_only 都不足以形成独立风格。full_strong_lateral_stable 是 broad ablation 阶段最优，但 p2 独立性仍不完全。

---

## 7. 当前任务状态

| 任务 | 状态 | 说明 |
|---|---|---|
| broad ablation pipeline | 完成 | 支持多配置运行与汇总 |
| integrity check | 完成 | 修复了 config 指标完全相同的问题 |
| full_strong_lateral_stable | 完成 | broad ablation 最优配置 |
| strong_yaw_clip 机制验证 | 完成 | 关键单项机制 |
| comfort_only / lateral_only 对比 | 完成 | 证明单机制不足 |
| broad ablation report | 完成 | 可作为论文机制实验基础 |
| 后续 local fine sweep | 完成 | 见 `03_experiment_2b_local_sweep.md` |
