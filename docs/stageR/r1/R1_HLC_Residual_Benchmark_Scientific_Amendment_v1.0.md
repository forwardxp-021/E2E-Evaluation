# R1 HLC Residual Benchmark 科学修订 v1.0

状态：`FROZEN_PROSPECTIVE_R1_SCIENTIFIC_AMENDMENT`。该修订只作用于未来 R1 HLC benchmark，在 RBR training、representation evaluation、正式 R1 development roster 和 future R4 confirmation 之前完成。

## 修订内容

新的 Primary F_match 为：

1. `ego13.mean_speed`
2. `ego13.end_minus_start_speed`
3. `ego13.path_length`

`ego13.heading_change_abs_total` 重分类为 `SECONDARY_MECHANISM_PROXIMAL_AUDIT`。原 `0.0492160141 rad` caliper 仅保留作描述审计；它不参与 Primary pair qualification、matching distance 或 roster eligibility，也不得用于删除不利 pair。

## 科学依据与边界

依据仅为预先完成的 analytical/synthetic physical compatibility audit：heading absolute total 会直接响应 retreat/recommit morphology，和 Primary mechanism 存在结构重叠；同时非空可行交集确实存在。旧 smoke 成功率未参与修订理由，也没有 outcome-driven parameter tuning。

未来论文只允许声称 `PREDEFINED_LOW_ORDER_F_MATCH_CONTROLLED` 和 `MECHANISTICALLY_DISTINCT_TEMPORAL_MORPHOLOGY`；禁止声称 `ALL_HANDCRAFTED_FEATURES_MATCHED` 或 `HANDCRAFTED_FEATURES_CANNOT_DETECT`。

## 不回溯

本修订不修改 R0 v1.0、R0 D4 family-specific matching contract、Wave3 D4 历史结果或冻结 HLC mechanism thresholds。
