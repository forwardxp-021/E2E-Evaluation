# R1 B2.8-R1 执行集成修复与零运行重新预检

## 结论

`PASS_COMPLETE_EXECUTION_PATH_ZERO_RUN`。本轮仅修复 execution wiring，未启动 official simulation，实际 official runs 为 `0`，消耗预算为 `0`。

V2.1 planner 未修改。新增 V2.2 仅在 `compute_trajectory` 入口被动记录已实现的 `PlannerInput.history.current_state`，然后将同一输入交给 V2.1 原有逻辑。HLC 与 TSB 各一个 frozen input 的 80 次 trajectory state（位置、heading、speed、timestamp）均 exact identical。

48/48 Hydra composition 与 frozen schedule/roster 逐行绑定均通过；任何缺失、歧义或 arm 不匹配均在 simulator start 前 fail-closed。

## 外部 observer 评估

对本地绑定 nuPlan runtime 的 callback 接口进行了只读审计。其 `on_step_start` 仅接收 `SimulationSetup` 与 planner，而 `on_step_end` 接收的是规划后写入的 `SimulationHistorySample`；两者都不提供 planner-call-entry 的 `PlannerInput.history.current_state`。因此不能以外部 callback 无歧义地记录本轮冻结定义的 observation point。

据此采用 V2.2 的最小版本化 instrumentation：只在 V2.1 `compute_trajectory` 调用前写出当前状态，未改 trajectory generation。HLC 与 TSB 的 synthetic parity 均以 80 次相同 `PlannerInput` 验证 position、heading、speed 与 timestamp exact identical。

## 授权状态

新的 execution SHA 尚未获得运行授权：`OFFICIAL_SMOKE_AUTHORIZED=false`，`NEW_EXECUTION_RUN_BUDGET=0`，RBR 未授权。
