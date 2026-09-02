# R1 B2.9-D Official Smoke 技术停止报告 v1.0

## 结论

一次性授权已被使用，official smoke 在完成第 1 个 pair 的 baseline/treatment 两条 simulation 后，因 post-run metric callback 生命周期未执行而技术停止。剩余 46 个 run 未启动；没有重试、identity replacement、threshold change 或 RBR。

本次状态是：

- `OFFICIAL_SMOKE = STOPPED_TECHNICAL_FAILURE_INCOMPLETE`
- `SCIENTIFIC_RESULT = NOT_EVALUABLE`
- `OFFICIAL_RUNS_COMPLETED = 2`
- `CONSUMED_REAL_BUDGET = 2`
- `FURTHER_OFFICIAL_RUNS_AUTHORIZED = false`

## 授权与执行边界

- 授权远端 commit：`8841b770a6c9fe88d93385ba0e75932728c3e874`
- 授权 final manifest SHA256：`88d1d36ef721c43dda4ce5907d2a85968142279238a4c2f11309b7b2eebe2877`
- 授权范围：`ONCE`
- 初始预算：48；已 claim/完成：2；算术未消费：46，但技术停止后不可继续使用。
- 第 3 个 run 的输出根没有创建。

## 已完成的两条 run

`R1B29D-01-R-HLC-BASELINE` 与 `R1B29D-01-R-HLC-TREATMENT` 均完成 80 行 Primary trace：iteration 0...79 恰好一次、来源均为 `REALIZED_CURRENT_EGO`、时间戳严格递增。

这只能证明两条 simulation 与 Primary80 trace 技术完成。由于 official safety parquet 缺失，pair evaluator 没有形成科学结果，不得把两条 trace 解读为 scientific PASS/FAIL。

## 技术失败

自动 pair evaluation 报错：

```text
MetricCanonicalizationError:
collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet
```

两个 run 均留下一个 `metrics/*.pickle.temp`，但 parquet 数量均为 0。代码路径显示：冻结 executor 直接调用 `runners[0].run()`；nuPlan 的 `run_runners(...)` 才会在 runner 完成后调用 `multi_main_callback.on_run_simulation_end()`，该回调负责把临时 metric 文件汇总为 parquet。因此 safety canonicalizer 在冻结位置找不到 `no_ego_at_fault_collisions.parquet` 和 `drivable_area_compliance.parquet`。

这是 execution control-plane / post-run callback wiring 的技术基础设施错误，不是 planner、Primary80、HLC mechanism 或 scientific gate 失败。

## 冻结处置

- 本轮进程已退出，剩余 schedule 停止。
- 不重试已完成的两条 run。
- 不从 run order 3 继续。
- 不人工转换临时 metric 后伪造本次 pair evaluation。
- 不修改 roster、schedule、pair binding、identity、threshold、mechanism、F_match 或 safety definition。
- `RBR_A/B/C = NOT_AUTHORIZED`。

如需修复 callback lifecycle 并执行新的 official package，必须先形成新的 versioned executor/manifest，并由 Scientific Owner 重新授权；当前一次性授权不能复用。
