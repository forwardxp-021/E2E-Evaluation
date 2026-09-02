# R1 B2.9-D Scientific Owner 48-Run Authorization Request v0.1

## 请求事项

B2.9-D 已完成 outcome-blind scientific roster rebuild、48-run schedule、24 pair pre-outcome binding、final executor 与完整 SHA 闭包。现仅请求 Scientific Owner 判断：是否授权下述 final manifest 对应的冻结 48-run official smoke 执行一次。

- final execution manifest SHA256：`88d1d36ef721c43dda4ce5907d2a85968142279238a4c2f11309b7b2eebe2877`
- 当前 `OFFICIAL_SMOKE_AUTHORIZED = false`
- 当前 `NEW_RUN_BUDGET = 0`
- 当前 `ACTUAL_OFFICIAL_RUNS = 0`
- 当前 `RBR_A/B/C = NOT_AUTHORIZED`

## 新 roster 摘要

- roster v3.0：24 个 unique identities；R-HLC 12，R-TSB 12。
- HLC retained/replaced：11/1。
- TSB retained/replaced：11/1。
- effective permanent exclusion：45 个 token/log identity；Attempt 1 的 `b1be12bca092597a` 保留 `OFFICIAL_ATTEMPT_CONSUMED = true`。
- source universe 复用 `r1_fresh_smoke_source_universe_v0.1.json`；未因 Attempt 1 或 canary scientific outcome 重选。

## 确定性 replacements

- R-HLC：旧 identity `b1be12bca092597a` → 新 identity `720f0657ad1c5980`；new rank `005964847f4dccee1fb6c24b8affd9b5568206f64c75a4fc17ef3e696c9f3e66`；原因：OFFICIAL_ATTEMPT_CONSUMED_AND_PERMANENT_ENGINEERING_CANARY_EXCLUDED。
- R-TSB：旧 identity `49e7a60a807f58e8` → 新 identity `2b8c769f31e5553d`；new rank `00023447afba74332cc2b08fd7138f326771fb33b32256c26708f4b45575cced`；原因：ADDITIVE_ELIGIBILITY_EXCLUDED。

## 执行前闭包

- exact single scenario resolution：48/48 PASS。
- full runner construction：48/48 PASS。
- planner class：48/48 exact `R1OfficialTechnicalSmokePlannerV3_1`。
- time controller：48/48 exact `R1Primary80ScientificTimeControllerV1`，`number_of_iterations() = 81`。
- frozen pair binding lookup：48/48 PASS；24/24 binding 在 simulation 前完成。
- dispatcher structural invocation：24/24 PASS（仅 contract-valid synthetic 80-row REALIZED trace 与 real-format temporary parquet）。
- complete transitive SHA closure：PASS。
- `runner.run()` 调用：0；official simulation：0；consumed budget：0。

## 语义冻结

HLC 的 planner reference 使用 `ROUTE_CONTINUOUS_V2_3`；measurement reference 继续使用 `FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT`。本轮没有修改 measurement numerics、threshold、mechanism、F_match 或 safety contract。

## Scientific Owner 唯一待决问题

是否对 final manifest SHA256 `88d1d36ef721c43dda4ce5907d2a85968142279238a4c2f11309b7b2eebe2877` 授权一次冻结的 48-run official smoke？在收到匹配该 SHA 的显式授权前，executor 保持 fail-closed。
