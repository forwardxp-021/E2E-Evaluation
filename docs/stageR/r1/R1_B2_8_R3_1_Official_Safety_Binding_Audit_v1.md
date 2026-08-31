# R1 B2.8-R3.1 Official Safety 绑定审计 v1

## 结论

已定位并复用既有冻结的 official safety 语义；本轮未创建新 metric、threshold、weight、majority rule 或 aggregate rule。

## 已冻结来源

- 原始 official metric engine 输出：`no_ego_at_fault_collisions.parquet` 与 `drivable_area_compliance.parquet`；
- 历史 canonicalizer：`tools/r1_official_metric_canonicalizer.py`；
- 历史合同：`docs/stageR/r1/r1_official_metric_canonicalization_contract_v1.0.json`；
- 历史双臂 pass 语义：每臂 `number_of_all_at_fault_collisions_stat_value == 0` 且 `drivable_area_compliance_stat_value == true`，pair 为 baseline 与 treatment 均通过。

## R3.1 接线

`tools/r1_b2_8_r3_1_official_safety_adapter.py` 仅调用历史 canonicalizer，并将 raw Parquet provenance SHA、canonical payload SHA 与冻结的 arm/pair pass 语义交给 post-run evaluator dispatcher。缺失文件、重复文件、多行、缺列、NaN、非二元 drivable 值与非整数/负 collision count 都 fail-closed。

该 adapter 不读取 representation、BDD 或 RBR，不做 post-hoc eligibility，也不修改阈值。
