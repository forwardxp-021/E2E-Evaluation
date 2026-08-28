# R1 Runtime Determinism Validation V3 报告 v1.1

## 结论

`R1_RUNTIME_DETERMINISM_VALIDATION_V3 = PASS`。在本 V3 authorization 绑定的 runtime 环境内，
`BACKGROUND_REPLAY_DETERMINISM = VERIFIED_ON_BOUND_RUNTIME`，
`OFFICIAL_REPLAY = READY_FOR_TECHNICAL_SMOKE_REVIEW`。

此结论只回答 bound official nuPlan runtime 是否严格可重复；没有 HLC/TSB treatment、F_match、BDD、
representation、probe 或 RBR 的科学行为分析。

## V2 诊断与 V3 修正范围

V2 首条官方运行已确认返回码为 0、trace 为 149 steps、planner binding 存在，且恰好存在 collision 与
drivable-area Parquet。V2 唯一失败原因是旧 executor 仅发现 JSON metric。V3 不修改任何 generator 或
scientific metric，只将既有 Stage7L 的两个官方字段冻结为 canonical Parquet payload。

metric parser 的零预算 preflight 为 `PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED`：两组有效 fixture 可解析，
missing/duplicate/missing-column/empty-table 均 fail-closed；V2 Parquet 仅作路径/schema/column compatibility
验证，未解释其值。

## V3 执行与比较

新 V3 ledger 依冻结顺序完成 8/8 次 pre-run claim 和 official run。全部 run 均无 technical failure；第九次
pre-run claim 在 simulator 启动前被拒绝。四个 A/B pair 的 15 个冻结类别均 exact canonical equality：

| family | scenario token | A/B 结果 | 精确类别 |
|---|---|---|---:|
| R-HLC | 25944935eadb52f1 | `EXACTLY_EQUAL` | 15/15 |
| R-HLC | ef3172a208cc5dd7 | `EXACTLY_EQUAL` | 15/15 |
| R-TSB | b486f9cf33a85455 | `EXACTLY_EQUAL` | 15/15 |
| R-TSB | 3edcce9e7e19573f | `EXACTLY_EQUAL` | 15/15 |

其中 collision 与 drivable comparison 使用 canonical semantic payload 的 JSON SHA/精确相等；Parquet
container SHA 仅作为 provenance。V3 未出现 float difference、first differing step 或 affected field。

## 约束与判定

- frozen roster SHA 仍为 `fc5c52a15eef9f71adb6f279e99bb4a0a6312fdc6013671c75550703c2759ac6`；没有 selector、replacement 或 ordering 改动。
- V1/V2 的失败永久保留为历史技术执行，未计入 V3 cap，也未用作 V3 evidence。
- `SCIENTIFIC_PROTOCOL_DEVIATION = NO`。V2 的 Parquet discovery 遗漏是 execution-output integration defect，不是 primary metric/threshold/mechanism 的 protocol 变更；V3 的 canonical binding 是本轮已授权的 execution-completeness correction。
- 已满足提交给 scientific owner 审核 48-call smoke 的 runtime 条件，但 `NEW_COMPLIANT_48_CALL_SMOKE = PENDING_SEPARATE_SCIENTIFIC_OWNER_AUTHORIZATION`，本报告不自动授权或启动它。
