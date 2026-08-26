# R0 Statistical Analysis Plan v0.1

## 1. 状态

```text
SAP_DRAFT_NOT_FROZEN
ALLOWED_EVIDENCE = DEVELOPMENT_DIAGNOSTIC_EVIDENCE
RBR_A/B/C_TRAINING_AUTHORIZATION = NOT_AUTHORIZED
```

机器版本：`docs/stageR/r0/manifests/r0_statistical_analysis_plan_v0.1.json`。机器文件包含 24 个 hypothesis records；本文件给出解释与执行边界。

## 2. 数据角色

- parameterization：仅 `R0_DEVELOPMENT`；
- `R0_AUDIT_HOLDOUT`：`NOT_AVAILABLE`；
- `FUTURE_R4_RESERVED_POOL`：`NOT_AVAILABLE`；
- historical Waymo test、Stage6/7/7L 只能作为已解盲 development evidence；
- 未建立 audit holdout 前不得使用 confirmatory/prospective validation 措辞。

## 3. 共通统计合同

| 字段 | Proposal |
|---|---|
| alpha | 0.05 |
| confidence | 0.95 |
| multiplicity | Holm within predeclared module/family |
| bootstrap | 5000，默认 log cluster |
| permutation | 49999，plus-one p |
| independence unit | scenario 或 same-scenario pair |
| split unit | scenario/source grouping，跨角色 identity overlap=0 |
| bootstrap cluster | log；没有 log identity 时用 scenario 并降级说明 |
| permutation unit | paired 为 scenario pair；unpaired 仅允许 log-disjoint group label |
| missing values | 保留 mask/sentinel 语义，不静默插补 |
| outliers | 只允许预声明 physical/quality exclusions |
| status model | execution COMPLETE/BLOCKED；hypothesis 五级结果 |

连续 probe：R² primary；MAE/NRMSE、Spearman、calibration slope secondary。分类 probe：balanced accuracy primary；AUROC、macro-F1 secondary。所有 target-level effect 报告 cluster-aware 95% CI。

## 4. Hypothesis families

### D0

- `D0_LENGTH_EFFECT`；
- `D0_POSITION_RETENTION_ASSOCIATION`；
- `D0_POOLING_EFFECT`；
- `D0_MASK_PADDING_SENSITIVITY`。

D0-C 只允许同一 hidden sequence 上的 last/mean/max。D0-B 是 matched quasi-experimental。first80/last80/event80 仅 descriptive。必须并列 frozen-probe-across-view 与 same-capacity-refit-probe-per-view。

### D1

- `D1_KNOWN_SEMANTIC_INFORMATION_PRESENT`；
- `D1_CROSS_DOMAIN_SEMANTIC_TRANSFER`；
- `D1_GEOMETRY_DEGENERACY`。

Probe family 固定为 linear ridge/logistic；ridge alpha grid 为 `1e-4…1e4` 九点。target 以 `r0_target_definition_v0.1.json` 为准。

### D2

- `D2_RESPONSE_SENSITIVITY`；
- `D2_CONTEXT_SENSITIVITY`；
- `D2_PAIRING_SENSITIVITY`；
- `D2_SHORTCUT_RISK`；
- `D2_ABLATION_OOD_RISK`。

Context shuffle 只在 frozen matching strata 内执行。Ablation 是 sensitivity 诊断；四项 OOD metric 中至少两项超过 development q99 时降级为 `ABLATION_OOD_DOMINATED`。

### D3

- `D3_FULL64_SIGNAL_DILUTION`；
- `D3_PROJECTED_READOUT_GAIN`；
- `D3_NULL_CALIBRATION_PRESERVED`。

Primary kernel 为 single RBF。bandwidth 按 representation/readout 在 treatment-label-blind development reference bank 上固定 positive off-diagonal median。projection ranks 为 `{1,2,4,8,16}`，1 SE 内选最小 rank。rank、bandwidth、probe training 与 null calibration 资产必须隔离。

### D4

每个 `R-HLC / R-TSB / R-IP` 分别记录：

- `D4_DESCRIPTOR_EQUIVALENCE_<FAMILY>`；
- `D4_MECHANISM_DIFFERENCE_<FAMILY>`；
- `D4_OUTCOME_BLIND_FEASIBILITY_<FAMILY>`。

F_match 与 M_behavior 严格分离。Equivalence primary 为 TOST 或双侧 90% CI 完全落入 frozen margin；多个 F_match 使用 intersection-union。24 个 margin 尚未获 scientific owner 批准，因此 D4 equivalence 不能执行为 frozen audit。

## 5. Kernel、null 与 calibration

- paired：same-scenario pair label swap；
- unpaired：只允许与 estimand 相符的 log-disjoint A/A calibration；
- paired null 与 unpaired A/A null 不混用；
- raw MMD² 不跨 representation 直接比较；
- projected vs full64 用同一 scenario/release replicate 做 paired effect 与 cluster CI；
- multikernel 若未来启用，必须另作预声明 aggregate 或 multiplicity family，不能挑最显著 kernel。

## 6. Equivalence 与 whole-roster

- `p>0.05` 不代表 equivalent；
- power 只检验 owner-approved margin 的可测性，不定义 margin；
- controlled treatment Primary 使用 whole-frozen-roster/intention-to-evaluate；
- rollout 后只可 whole-roster mechanism gate；
- mechanism-success subset 只能预声明为 Secondary；
- representation outcome 不参与 roster 保留/删除。

## 7. 当前 unresolved fields

1. D0 standardized minimum temporal effect 0.10 的 scientific materiality；
2. D3 nominal 0.05 下 FPR upper-CI 0.075 gate；
3. 24 个 F_match equivalence margin；
4. `R0_AUDIT_HOLDOUT` identity/source；
5. `FUTURE_R4_RESERVED_POOL` exact source/token roster；
6. v1 protocol/SAP 的 owner approval 与 SHA binding。

在这些 blocking fields 解决前，本 SAP 不能改名为 frozen v1.0，也不授权 RBR training。
