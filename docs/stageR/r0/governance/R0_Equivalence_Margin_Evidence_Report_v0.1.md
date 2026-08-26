# R0 Equivalence Margin Evidence Report v0.1

## Decision

`24/24 F_match margins = REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。本次没有批准物理等效边界，也没有用 raw population SD 或 power 机械填 margin。每项均标记 `NO_DEFENSIBLE_PHYSICAL_MARGIN_YET`。

证据来自既有 Waymo dynamic-v2 TRAIN development tensors，共 135046 rows / 36 shards；未读取 representation、embedding、BDD 或 future outcome。24 项均报告 finite/slot validity、结构零、sentinel/extreme、Tukey outlier、median/IQR 与 p05/p25/p50/p75/p95。若同一 scenario 有多个自然窗口，另报 pooled within-scenario SD；这不是重复传感测量误差。

raw33 在固定顺序的前 256 个 train rows 上由 `ego_seq.npy + neighbor_seq.npy` 重算，最大绝对差为 4.76837158e-06（小于 1e-5，float32-consistent）；这是计算再现性，不是 measurement noise floor。

## THW special audit

- mean_thw：valid rows=88787，median=3.89712644 s，IQR=6.85634196 s，p95=288.888196 s，aggregated value >=999 rows=476。
- min_thw：valid rows=88787，median=2.5012548 s，IQR=1.96116096 s，p95=10.9363485 s，aggregated value >=999 rows=476。

大 SD 不是可直接使用的 margin：THW 定义直接聚合 `front[:,10]`，front slot 稀疏时当前实现返回结构零；有效 front 中的极长/999-like headway 又形成 heavy tail。robust quantiles 与 slot-valid filtering 明显比 population SD 更适合描述分布，但仍不能给出物理/人类可感知等效阈值。

## Numerical option boundary

机器证据表最多只给出一个 `OPTION_C = 0.10 × development IQR`，且仅定义为 descriptor-balance sensitivity caliper。它不是 physical/material tolerance，也不是 repeatability/noise floor，不得作为 TOST margin，除非 Scientific Owner 另行批准并解释科学含义。没有合法 repeated-measurement 资产，因此不虚构 OPTION_B；没有物理阈值依据，因此不虚构 OPTION_A。
