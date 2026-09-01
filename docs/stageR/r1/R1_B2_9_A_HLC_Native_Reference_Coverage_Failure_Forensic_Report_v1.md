# R1 B2.9-A HLC Native Reference Coverage Failure Forensic Report v1

## 1. 结论

Attempt 1 已正式终止，scientific evidence 为 `NOT_EVALUABLE`。实际异常由 `source_reference_xy` 首先抛出；iteration 33 时 source 和 target 同时越界。失败不是单一 zero-weight source 问题：baseline 的 source 权重确实全为 0，但 active target 权重全为 1 且也已耗尽。

- run：`R1B27-01-R-HLC-BASELINE`
- scenario：`b1be12bca092597a`
- pair：`R1B27-01-R-HLC`
- failure iteration：33；ego time：`1632403000699918 us`
- realized ego：`x=588572.3414243052`，`y=4475807.52703238`，`speed=12.280398578314419 m/s`
- source lane / target lane：`18524 / 18525`
- required 7.9 s future distance：`97.01514876868391 m`

| reference | current arc (m) | native total (m) | requested max (m) | remaining margin (m) |
|---|---:|---:|---:|---:|
| source | 48.90729693937228 | 145.38099317720324 | 145.9224457080562 | -0.5414525308529505 |
| target | 49.06373189707505 | 145.55582470498388 | 146.07888066575896 | -0.523055960775082 |

iteration 32 仍有效：source/target margin 分别为 `0.6835893120259016 / 0.700773459100418 m`；iteration 33 是第一个 zero-crossing，与真实失败严格对齐。

## 2. 精确异常链

冻结 builder 在 `tools/r1_closed_loop_benchmark_v2_1.py:324` 先采样 source，再在 `:325` 采样 target。`tools/r1_prospective_generator_contract_v2.py:104-105` 对超出 native arclength 的 query 抛出固定异常。因此 traceback 的实际 first failing reference 是 source；离线分别计算又证明 target 在同一调用也无效。

```text
tools/r1_b2_8_r3_3_execute_frozen_48run_smoke.py:87 main
tools/r1_b2_8_r3_3_execute_frozen_48run_smoke.py:83 run
tools/r1_b2_8_r3_3_execute_frozen_48run_smoke.py:72 r3_2.run
tools/r1_b2_8_r3_2_execute_frozen_48run_smoke.py:106 runners[0].run()
nuplan/planning/simulation/runner/simulations_runner.py:113 planner.compute_trajectory
tools/r1_official_technical_smoke_planner_v2_2.py:82 super().compute_trajectory
tools/r1_official_technical_smoke_planner_v2_1.py:115 compute_planner_trajectory
tools/r1_official_technical_smoke_planner_v2_1.py:103 builder(*args)
tools/r1_closed_loop_benchmark_v2_1.py:324 sample source_reference_xy
tools/r1_prospective_generator_contract_v2.py:105 ValueError: NATIVE_REFERENCE_COVERAGE_FAIL_NO_EXTRAPOLATION
```

外层冻结执行器将其转为：`STOPPED_ON_TECHNICAL_FAILURE_REQUIRES_OWNER_REVIEW:R1B27-01-R-HLC-BASELINE:ValueError`。

## 3. zero-weight 判定

iteration 33 的输出 absolute time 为 `[3.3, 11.2] s`。冻结 baseline progress 在该区间全部为 1，因此 source weight 全为 0、target weight 全为 1。

- 实际首先抛出的 source：`UNNECESSARY_ZERO_WEIGHT_SOURCE_REFERENCE_EVALUATION_RAISED_FIRST`。
- 同时越界的 active target：`ACTIVE_TARGET_NATIVE_REFERENCE_REPLAN_COVERAGE_EXHAUSTION_SIMULTANEOUS`。

所以仅采用 active-reference-aware sampling 跳过 source 后，target 会在同一 iteration 继续 fail-closed。

## 4. 原 pre-rollout gate 的真实实现

B2.7 的实现只在 candidate 的初始 ego 上投影一次 source/target arc，并分别调用一次 baseline/treatment 80-frame builder（`tools/r1_b2_7_freeze_official_smoke_roster_v2.py:306-314`）。它证明的是：

`A. initial-state one-shot 80-frame construction`

它没有把 future realized ego 作为下一次 replan 的 current state，也没有检查 80 个 rolling calls 各自再要求 7.9 秒 native coverage。

B2.5 只构造 planner/context/route/evaluator，不调用 HLC rolling trajectory。B2.6 虽测试多个 iteration 的公开 planner dispatch，但 fixture 使用旧 `roster_v1.0`，且 `_official_input()` 在每个 iteration 都把 x/y/heading/speed 重置为同一个 initial state，只改变时间戳（`tools/r1_b2_6_official_dispatch_preflight.py:22-26,41-43`）。因此 phase clock 被覆盖，空间 rolling depletion 没有被覆盖。B2.8-R2/R3 的 zero-run 仅构造 SimulationRunner，并在 `runner.run()` 前停止。

## 5. 12 个冻结 HLC identity 的 nominal rolling audit

方法：只使用冻结 official native source/target references、initial state、HLC generator 和 constant initial speed。先生成初始 one-shot 0–79 nominal geometry，再把每一帧重新投影为对应 rolling call 的 nominal current ego，并在该点重新施加 7.9 秒输出 envelope。没有 simulator、controller 或 selector 调用。

| scenario token | speed | initial source margin | initial target margin | baseline first invalid | treatment first invalid |
|---|---:|---:|---:|---:|---:|
| b1be12bca092597a | 11.982115 | 41.806 | 41.851 | 35 | 35 |
| 5292d0ee192e5b51 | 11.975973 | 11.197 | 11.213 | 10 | 10 |
| 4fa52d891cb057f3 | 12.587409 | 36.935 | 36.979 | 30 | 30 |
| 86eb22d3fb9d5878 | 12.718387 | 29.319 | 29.382 | 24 | 24 |
| 8c223edd967d5902 | 12.089784 | 19.541 | 19.557 | 17 | 17 |
| 6464ad553b205b77 | 11.926056 | 40.482 | 40.533 | 34 | 34 |
| 87dc430b0397561c | 13.024392 | 23.933 | 23.977 | 19 | 19 |
| 97c8d9bd277c5497 | 12.137317 | 6.661 | 6.677 | 6 | 6 |
| 3606b3c7132f5f96 | 11.663395 | 51.259 | 51.276 | 44 | 44 |
| a090bc2ea0fd523d | 11.530309 | 45.806 | 45.848 | 40 | 40 |
| f52b69619b04507e | 12.531452 | 11.642 | 11.658 | 10 | 10 |
| ef95bc7d18095d38 | 9.098596 | 11.850 | 11.864 | 14 | 14 |

12/12 均通过 initial one-shot；12/12 的两个 arm 均预测在 iteration 6–44 内耗尽，全部早于 80。这是 `TECHNICAL_DIAGNOSTIC_ONLY`，不得当作 scientific outcome。

## 6. nuPlan controller 的实际 trajectory 消费

绑定的 `closed_loop_nonreactive_agents` 使用 TwoStageController + LQR，time controller 每次推进 0.1 秒。TwoStageController 每步只传播一次，但 LQR 的 `get_interpolated_reference_trajectory_poses()` 会从 trajectory start 一直重采样到 trajectory end，并调用 `get_state_at_times()`；当前 planner 返回 80 states `[0.0,7.9] s`，所以每个 0.1 秒 step 实际最远查询 `+7.9 s`。

LQR config 是 `discretization_time=0.1`、`tracking_horizon=10`。其显式控制 lookahead 为：reference velocity `+1.0 s`，curvature profile `0.0...0.9 s`。但速度/曲率由整条重采样 trajectory 的全局正则化最小二乘拟合得到，当前实现确实读取了全部 7.9 秒。若缩短 planner horizon，满足显式 lookahead 的无 clamp 最小网格为 11 states / 1.0 s；但拟合输入随之改变，不能声称 closed-loop controller behavior exact parity。

## 7. 失败尝试 SHA inventory

| artifact | SHA256 |
|---|---|
| authorization record | `d544effed8ffab783710483cb8223f43d1d5c5aacdce49a7f161f6a41833a31b` |
| committed stop report | `ccee289fcacbfcb4532f5bd139bf4c046c55460ca0f2005a2cf597752685e589` |
| local stop JSON | `16f15ebe4f8cb9dff9a3e3381cf105712628f7e8b2869fa840770059cf085b0d` |
| 34-row realized trace | `85d43b5d5e67a4d2cb1730a7cde26e3aebf8337c3b50018fd2761546b5efce2c` |
| raw log | `bb00306a10909760e6acd58b75d72371adaeaff81175f0db0165781f2cc29ae8` |
| raw nuboard | `e834bc48eed922e4d8b5e2053b9d9465458a1e48a6830c604442694975b068b8` |

上述文件未覆盖、删除或移动。

## 8. Consumed identity policy

`b1be12bca092597a / R1B27-01-R-HLC` 已发生一次 official runner attempt；当前 authorization 下永久禁止 retry。建议保留 Attempt 1 的 permanent execution-attempt exclusion 记录，同时仅允许 Scientific Owner 在新的 protocol/planner/map binding、全新 run ID、全新 manifest 与全新一次性授权下决定是否做 `VERSIONED_TECHNICAL_REPAIR_RERUN`。这不是对 Attempt 1 的 retry，也不得擦除 consumed 记录。

本阶段 `simulation_executed=false`，`RBR_A/B/C=NOT_AUTHORIZED`。
