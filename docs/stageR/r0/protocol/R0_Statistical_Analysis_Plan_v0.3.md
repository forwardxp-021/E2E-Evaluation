# R0 Statistical Analysis Plan v0.3

## Status

`READY_FOR_R0_V1_PROTOCOL_FREEZE`。这不等于分析已执行或 RBR training 已授权。

## Readiness semantics

- 无 R0_AUDIT_HOLDOUT：不阻塞 protocol freeze，不阻塞 R0 execution；全部科学结果限定为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。
- 19 runnable clean logs vs 150-log reference：`EXECUTION_CAPACITY_LIMITATION`；131-log gap 只保留为未来 confirmatory planning reference。
- `FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR` 已通过 `R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1` 冻结；final confirmation roster 在 R1 outcome-blind 形成。

## D1

CORE targets 共 9 项，longitudinal/lateral/interaction 各 3。连续 gate 为 held-out grouped `R²>=0.10` 且 cluster 95% CI lower>0；分类 gate 为 balanced accuracy>=0.60 且 lower>0.50；每族至少 2/3 targets。独立 groups 不足时为 `INCONCLUSIVE`。

## D4

Primary F_match 改为 family-specific：R-HLC=4、R-TSB=4、R-IP=3。Context_match 只使用 pre-treatment anchor；M_behavior 与同 family F_match 零交集。`D4_DEVELOPMENT_BALANCE_FALLBACK_V1` 采用 0.10×development IQR，只用于 development balance/feasibility，不是 formal equivalence。R4 physical/material margins 必须在 R4 outcome 解盲前冻结。

## Authorization

`R0_EXECUTION_READY_WITH_DEVELOPMENT_DIAGNOSTIC_EVIDENCE`；`RBR_TRAINING_NOT_AUTHORIZED`；`R4_FINAL_CONFIRMATION_ROSTER_NOT_FROZEN`。
