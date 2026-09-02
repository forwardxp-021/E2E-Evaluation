# R1 B2.9-E Post-Run Callback Lifecycle Forensic v1

## 结论

B2.9-D 的精确技术根因是 final executor 绕过了已经在 B2.9-C 验证成功的 nuPlan 完整执行生命周期：它在构造唯一 `SimulationRunner` 后直接调用 `runners[0].run()`，因此没有执行 `run_runners(...)` 中的 runner report 持久化和 `multi_main_callback.on_run_simulation_end()`。仿真与 Primary80 trace 本身已经完成；失败发生在仿真后的 callback/metric aggregation 生命周期，不是 planner V3.1、Primary80 time controller、V2.3 route-continuous builder 或 scientific semantics 失败。

## 逐项路径对照

| 环节 | B2.9-C 成功路径 | B2.9-D 失败路径 | 证据与影响 |
|---|---|---|---|
| `build_simulations` | 构造官方 scenario、callbacks 与 runner | 同样完成构造 | 两者都得到实际 `SimulationRunner`，不是 construction failure |
| `SimulationRunner.run` | 由 nuPlan `run_runners` 经 `execute_runners` 间接调用 | executor 直接调用 `runners[0].run()` | D 的直接调用只完成单 runner 生命周期 |
| `run_runners` | `tools/r1_b2_9_c_cross_family_canary.py` 调用 `nuplan.planning.script.utils.run_runners(...)` | 未调用 | C 的 metric parquet、runner report、安全适配和 dispatcher 均完成；D 在 post-run 阶段失败 |
| main callback | `run_runners` 持有并调度 `common_builder.multi_main_callback` | 被绕过 | D 未进入全局仿真结束回调 |
| `on_run_simulation_end` | master node 调用一次 | 未调用 | D 的 metric 临时序列化未被聚合 |
| metric temp serialization | 每个 scenario 先产生临时 metric 序列化 | 两个已完成 run 均产生一个 `*.pickle.temp` | 说明 simulation 与 metric callback 的 per-run 部分已运行 |
| metric parquet aggregation | `MetricFileCallback.on_run_simulation_end()` 读取临时文件并按 metric 写 parquet | 未发生 | D 每个 run 的 parquet 数为 0，缺少 canonical safety 输入 |
| runner report | `run_runners` 保存 `runner_report.parquet` | 未保存 | D 绕过 `save_runner_reports` |
| safety canonicalizer | 从 `no_ego_at_fault_collisions.parquet` 与 `drivable_area_compliance.parquet` 读取冻结官方安全指标 | 因两个 parquet 均不存在而 fail-closed | 错误为 `collision:MISSING_EXPECTED_METRIC_FILE:no_ego_at_fault_collisions.parquet` |

## 科学运行时已完成的证据

- B2.9-D 两条已完成 run 均生成精确 80 行 `REALIZED_CURRENT_EGO` trace，iteration 为 `0...79` 且时间戳严格递增。
- 两条 run 均使用冻结的 `R1OfficialTechnicalSmokePlannerV3_1` 与 Primary80 控制器，并完成实际 simulation runner 执行。
- 两条 run 均留下 per-scenario metric 临时序列化与 simulation log，证明失败点晚于 planner/runtime execution。
- 冻结 safety canonicalizer 在缺少 parquet 时按合同停止；frozen evaluator 从未被调用，因此科学状态保持 `NOT_EVALUABLE`。

## B2.9-E 修复边界

B2.9-E 只引入共享 primitive `run_one_with_full_nuplan_lifecycle(...)`。该 primitive 要求 exactly one runner，并调用 nuPlan `run_runners(...)`；返回前验证 runner report、最终 metric parquet，以及两个冻结 safety parquet 均已存在。它不选择 identity、不改 scientific parameter、不做 outcome decision。

新的 official executor 与 exact-executor engineering canary 使用同一个 primitive。executor 源码不直接调用 `SimulationRunner.run()`。pair evaluator 只能在 primitive 成功返回、baseline/treatment 两个 arm 均 technical complete 后调用。

## Exact-executor canary 取证

- 实际授权的 simulation canary 为 4 次：永久 `SCIENTIFIC_USE_FORBIDDEN` 的 1 个 HLC pair 与 1 个 TSB pair，各 baseline/treatment 一次。
- 4/4 经共享 `run_runners` 生命周期完成；4/4 trace 精确 80 行；4/4 生成最终 metric parquet 与 runner report；4/4 safety adapter structural complete；2/2 pair dispatcher complete。
- 第一次 A01 在 runner construction 和 simulation 之前因 B2.9-E Hydra 环境绑定缺失而 fail-closed：`simulation_started=false`、`run_runners_called=false`、trace rows 为 0，不计入 4 次 canary，也不构成 simulation rerun。修复同一 config binding 后，A02 使用全新 output root 执行 4 次授权 canary；`actual_simulation_reruns=0`。
- canary scientific PASS/FAIL 仅为 `DESCRIPTIVE_ONLY`，不得用于 selector、roster、threshold 或 scientific conclusion。

## 旧 Attempt 永久处置

- B2.9-D once authorization 已消费；`OFFICIAL_ATTEMPTS_CONSUMED = 2`，剩余 46 run 未获授权。
- `outputs/r1_b2_9_d_official_smoke_once_v1/` 与 authorization、technical stop record、technical stop report 保持只读。
- 未删除、覆盖、append、补跑 callback、补生成 parquet 或补做旧 pair scientific evaluation。
- 旧两条 trace 永久为 `ATTEMPT_HISTORY_ONLY`，不作为 B2.9-E 新 scientific pair input。

## 最终判定

`FINAL_EXECUTOR_LIFECYCLE_DIVERGED_FROM_VALIDATED_B2_9_C_CANARY_PATH` 得到代码和产物双重支持。B2.9-E 修复的是 post-run callback lifecycle wiring；科学 planner/runtime 与冻结科学合同未修改。
