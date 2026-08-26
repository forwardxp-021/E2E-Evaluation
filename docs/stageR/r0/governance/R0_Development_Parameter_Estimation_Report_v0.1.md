# R0 Development Parameter Estimation Report v0.1

## 1. 范围与结论

本报告只使用 `R0_DEVELOPMENT` 合同、历史方法配置、target scaler 与只读描述统计形成 pre-freeze proposal。没有运行 RBR-A/B/C 训练，没有仿真，没有读取 future audit/R4 outcome，没有修改历史产物。

机器提案：`docs/stageR/r0/manifests/r0_parameterization_proposal_v0.1.csv`。

结论：18 项 parameter proposal 中 16 项达到 `READY_FOR_FREEZE` 提案状态；2 项仍需 scientific owner 批准。`READY_FOR_FREEZE` 表示依据和数值已经可审阅，不表示 R0 v1 已经冻结。

## 2. 使用的 development evidence

- R0 local contract verification：A/B/C checkpoint、T80/T150、pooling/mask、ego13、Stage7L/Stage6P MMD/null；
- Waymo Dynamic-v2 manifest：168700 rows，scenario split overlap=0；
- raw33 train-only scaler：135046 train rows、33D population mean/std；
- ego13 Stage6L reference scaler：dose100 conservative 183 rows的 median/IQR；
- Stage7L mask/context 的只读统计：每 dose 80 rows，53×149 + 27×150 valid length；
- 历史 Stage7L 100000 paired permutations与 Stage6P 20000 bandwidth pair draws只作为计算/方法依据，不自动继承为 R0 参数。

未使用任何 candidate representation performance 来选择 holdout、reserved pool、kernel、rank、margin 或 scenario。

## 3. 全局统计参数

| 参数 | Proposal | 来源 | 状态 |
|---|---|---|---|
| alpha | 0.05 | PHYSICAL_RATIONALE | READY_FOR_FREEZE |
| confidence level | 0.95 | PHYSICAL_RATIONALE | READY_FOR_FREEZE |
| multiplicity | 每个预声明 module/family 内 Holm；D4 multi-feature 用 intersection-union | COMPUTATIONAL_PRECISION | READY_FOR_FREEZE |
| bootstrap | 5000 cluster replicates | COMPUTATIONAL_PRECISION | READY_FOR_FREEZE |
| permutation | 49999，plus-one p | COMPUTATIONAL_PRECISION | READY_FOR_FREEZE |

49999 次 permutation 的最小 plus-one p 为 `1/50000=0.00002`；当真实 p 约 0.05 时，Monte Carlo SE 约 0.001。若冻结后 runtime preflight 不能满足预算，只能按 SAP 预声明的精度规则上调/下调，不能看 outcome 后改次数。

本阶段未运行 model-level runtime benchmark，因此不声称 encoder/probe 的实测运行时间。5000/49999 是计算精度 proposal，不是已验证 SLA。

## 4. D0/D1 proposal

- event bins：150 帧按 `0–49 / 50–99 / 100–149` 分成相等 5 s support；
- temporal minimum effect：绝对 paired standardized retention difference `>=0.10`，95% CI 排除 0，且至少 2/3 seed 同向；
- 该 0.10 是 bounded small-effect development proposal，不具有直接物理单位，仍需 scientific owner 批准；
- linear probe family：linear ridge/linear logistic；
- ridge grid：`1e-4` 到 `1e4` 的九点 log grid；
- continuous target：R² primary，MAE/NRMSE、Spearman、calibration slope secondary；
- categorical target：balanced accuracy primary，AUROC、macro-F1 secondary；
- 全部 target-level 报告 log-cluster 95% CI，不只报告 pooled aggregate。

## 5. D2 proposal

Context-shuffle strata：

```text
scenario_family
× lane_change_direction
× initial_speed_tertile
× traffic_density_tertile
× neighbor_availability_pattern
× event_phase_bin
```

每 cell 至少 4 个 independent units；稀疏时按固定顺序合并 event phase、density、speed，永不跨 scenario family。匹配和 coarsening 只使用 pre-treatment/context 信息。

OOD boundary：每个 reference metric 使用 treatment-label-blind development q99；四项指标中至少两项越界时标记 `ABLATION_OOD_DOMINATED`。该规则是 diagnostic boundary，不是 causal gate。

## 6. D3 proposal

- Primary kernel：single RBF；
- bandwidth：每个 representation/readout 在 treatment-label-blind R0_DEVELOPMENT reference bank 上，以正 off-diagonal pair distance median 固定一次；
- paired/unpaired 使用各自合法 null，但不再静默混用 cell-adaptive 与 fixed-pool bandwidth；
- projection ranks：`{1,2,4,8,16}`，max 16；
- selection：development semantic-retention metric 先过 null-calibration gate；1 SE 内选最小 rank；
- FPR gate proposal：nominal 0.05 下 upper 95% CI `<=0.075`，仍需 scientific owner 批准。

## 7. D4 variability 与 margin 边界

24 个 F_match candidate 的历史自然波动已写入 `r0_equivalence_margin_proposal_v0.1.csv`：

- ego13 使用 Stage6L conservative 183-row reference IQR；
- raw33 使用 Waymo train 135046-row population SD。

这些统计量只能描述 natural variability。当前没有 repeated-measurement noise/reproducibility 与业务容忍度证据，因此 24 个 equivalence margin 全部保持空值并标记：

```text
REQUIRES_SCIENTIFIC_OWNER_APPROVAL
```

没有用 power 反推 margin。

## 8. Holdout outcome-blind inventory

清单：`docs/stageR/r0/manifests/r0_audit_holdout_candidate_inventory_v0.1.csv`。

Waymo train/val/test、Stage6P、Stage7/M6、Stage7L 均已用于训练、选择、representation evaluation 或已解盲；现有 nuPlan remainder 缺少能够证明“从未使用且对所有历史 roster 无 overlap”的 authoritative identity ledger。

```text
R0_AUDIT_HOLDOUT_UNAVAILABLE_FROM_EXISTING_ASSETS
```

最小新增数据获取方案：

1. 选择一个未进入当前仓库历史分析的新 source release 或 prospective acquisition；
2. 在任何 embedding/probe/BDD 运行前生成 source/log/scenario/token identity ledger 与 SHA；
3. 用 hash-sorted、source/log-disjoint 规则一次性锁定 audit roster；
4. 只用 pre-treatment/context/runnability 信息做 exclusions；
5. independent-unit 数量由已冻结 effect/margin 和 cluster-aware power 决定；在 margin 未批准前不伪造最小样本数；
6. audit roster 不参与 threshold、rank、kernel、probe capacity 或 margin 选择。

## 9. 限制与当前授权

- 这些参数仍是 v0.1 proposal；
- 没有可用 R0_AUDIT_HOLDOUT；
- future R4 source/token roster 尚未锁定；
- 24 个 equivalence margin 和两项 materiality/calibration 数值仍待批准；
- SAP 为 draft；
- `RBR_TRAINING_NOT_AUTHORIZED` 持续有效。
