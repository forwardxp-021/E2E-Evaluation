# R1 B2.6 最终官方调度与 HLC 测量符合性修复报告 v1

## 结论

B2.6 已在零 candidate enumeration、零 roster selection、零 planner rollout、零 simulation、零 RBR 的边界内完成四项 B2.5 pre-rollout implementation nonconformance 修复。`OFFICIAL_SMOKE_PLANNER_V2_1`、`OFFICIAL_SMOKE_EVALUATOR_V2_1`、`ABSOLUTE_EPISODE_CLOCK_BINDING`、`HLC_REALIZED_PROGRESS_V1`、`HLC_TERMINAL_ROUTE_PROGRESS_V1` 与 `OFFICIAL_DISPATCH_PREFLIGHT_V1` 均在测试通过后版本化冻结。

这是一项 execution integration implementation repair，不是 scientific protocol deviation。HLC Option-B、TSB Option-A、两 family 的 mechanism 与 Primary F_match、HLC endpoint/engineering、TSB floor=2.0、HLC clearance 数值均未修改。

## 正式调度与绝对时钟

Planner V2.1 完整实现 bound nuPlan 1.2.2 `AbstractPlanner` 当前接口：`name()`、`observation_type()`、`initialize()`、`compute_trajectory()`、`compute_planner_trajectory()` 与 `generate_planner_report()`。公开入口 `compute_trajectory(current_input)` 显式委托 `compute_planner_trajectory(current_input)`，并通过真实 `PlannerInput` 调用验证返回 `InterpolatedTrajectory`。

动态 phase clock 不再读取 future roster row 的 `absolute_episode_time_s`。首次调用必须为 iteration 0，并把该物理 `time_us` 绑定为 episode start；后续每次 replan 由 `current_input.iteration.index/time_us` 确定绝对 episode 时间，记录 iteration、物理时间、nominal/physical elapsed、phase source 和当次 generator phase。非单调调用 fail closed，replan 不允许 phase reset。`t_anchor=iteration 10`、`t_diverge=iteration 11` 不变。

实际零 rollout preflight 共调用公开 dispatch 16 次：R-HLC 使用 iterations 0/1/5/11/16/25/35/45/60，R-TSB 使用 0/1/5/11/16/25/35。所有调用均返回 `InterpolatedTrajectory`，state0 与 current ego 相同，construction parity 通过，route builder 为 `build_native_route_reference_v1_1`。HLC 的 advance/hold/retreat/recommit 与 TSB 的 first-brake/release/second-brake 均随绝对时钟推进，没有随 replan 重启。

## HLC 原生 realized progress

`hlc_realized_lane_transition_progress_v1_0` 对每个 realized ego frame 分别在 source 与 target native reference 上投影，构造局部 source→target 跨车道向量，再计算 normalized progress。source center 为约 0，target center 为约 1；raw progress 完整保留，只有进入已冻结 mechanism 时才 clip `[0,1]`。拓扑投影歧义 fail closed。

21 个 B2.6 专项测试覆盖直线 source-only、target-only、线性过渡、advance→retreat→recommit、曲线平行车道、不同采样密度与歧义投影。纵向-only source-lane 运动的 progress 全程为 0，HLC mechanism 返回 `NO_DEPARTURE`，因此不能误触发 commitment。

## HLC paired route progress 与 evaluator

`terminal_native_route_progress_v1_0` 把 baseline/treatment 的 terminal realized ego 分别投影到同一个 frozen native route reference，得到 `s_baseline_terminal`、`s_treatment_terminal`，并以二者绝对差作为 paired route progress。1.5 m gate 保持不变，route source 与 canonical route SHA 写入 audit；path-length difference 不再充当 surrogate。

Evaluator V2.1 的 HLC Primary 签名显式要求 source reference、target reference 与 native route reference/source。Primary 顺序为 realized current ego → native HLC progress → timestamp-aware mechanism → frozen Primary F_match → timestamp-aware endpoint/native route progress → frozen engineering → official safety。planned trajectory 继续仅为 `SECONDARY_GENERATOR_INTENT_ONLY`，planned-first 会 fail closed。

完全 synthetic 的 baseline monotonic lane change 与 treatment Option-B hesitation pair 已经由 V2.1 全链评估：context identity、mechanism、F_match、endpoint/native route progress、engineering 和 official safety payload 均实际进入 pipeline；该测试不读取历史 outcome，也不形成科学结论。

## 冻结合同回归

以下既有文件未修改并保持 SHA：

| 绑定 | SHA256 |
|---|---|
| closed-loop residual measurement v1.1 | `96fbdad467b6e1b321c963f83b1b98ca49f8032cf85b572bb1a5a3a774d7f6a1` |
| TSB mechanism applicability v1.0 | `8d249dad707b58337029cd24fa825549a9b05dfc5c9b24f5dc9993956fde8cf2` |
| HLC clearance v1.1 | `ec41c1c651186d78e7bc0e4a5401133d12bc311d98e134597a69937eae12b46e` |
| HLC geometry realization v1.1 | `820bf18f26fd13e0c052786aecd322c87442e02c20550f3c5a0bdaef831d183e` |
| TSB route realization v1.1 | `69da56611a3e782e3939ebafa1ce064d1ef4702eab96c0c34e28040b69fe1132` |
| HLC mechanism v1.0 | `f2e20d6e8c443f92d5fd8458069f431d6ba949069d54e703f6ca4bf2df5ce0e8` |
| TSB mechanism v1.0 | `375d3c07fbdf8d5ed6f0ebe5b8056221dec20ae732529076bce0f2e4695ccece` |

源实现回归确认 HLC mechanism thresholds、TSB mechanism thresholds、HLC/TSB Primary F_match、HLC endpoint limits、HLC engineering limits、TSB floor=2.0、HLC clearance numerics 全部保持。新增 binding 只固定正确执行语义，不改变 scientific numerics。

## 测试与冻结状态

- B2.6 专项对抗测试：`21/21 PASS`。
- B2.5、B2.4、closed-loop benchmark/context 回归：见最终 SHA manifest 的 combined regression 结果。
- 语法检查、JSON 校验、临时依赖检查：见最终 SHA manifest。
- `simulation_launched=false`，`actual_candidates_enumerated=0`，`actual_roster_selected=false`，`new_rollout_count=0`。

Selector v0.7 已继承 `MASTER_SEED=2026082701`、既有 salt SHA 与至少 40 个 blacklist identities，状态为 `READY_FOR_FINAL_SCIENTIFIC_OWNER_ENUMERATION_AUTHORIZATION`。该状态只说明最终实现前置条件已满足，不等于授权。

## 剩余阻断与授权边界

当前只剩科学负责人的两个显式决策：是否授权 fresh outcome-blind enumeration 并冻结 24 identities；以及是否在 roster 冻结后授权 48-run official smoke。当前状态继续为：

- `ENUMERATION = NOT_AUTHORIZED`
- `NEW_ROLLOUT = NOT_AUTHORIZED`
- `R1_FORMAL_DEVELOPMENT_ROSTER = NOT_READY`
- `RBR_A/B/C = NOT_AUTHORIZED`

因此本阶段在冻结 B2.6 实现与证据后停止，不枚举、不仿真、不训练。
