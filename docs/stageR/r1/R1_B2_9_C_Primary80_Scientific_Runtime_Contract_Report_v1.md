# R1 B2.9-C Primary80 科学运行时合同报告 v1

## 冻结结论

Primary 固定为 `REALIZED_CURRENT_EGO` iterations `0...79`，planner calls 恰好 80。科学 time-controller 继承 nuPlan 1.2.2 `StepSimulationTimeController`，唯一 override 是 `number_of_iterations=min(official_scenario_iterations,81)`；场景少于 81 iterations 时显式 `NOT_EVALUABLE/FAIL_CLOSED`。

0.1 s 时间网格、TwoStageController/LQR、observation、ego controller、metric engine、两 family generator、机制、F_match、endpoint、工程限制和安全阈值均未改变。80 个 realized 状态覆盖 0.0...7.9 s，runner 在 8.0 s 边界终止；iteration >=80 不进入 planner、Primary trace、安全或 evaluator。

该 horizon 来自既有冻结的 80-frame/8.0-second Primary、HLC 8 s clearance 与 evaluator 的 0...79 输入合同。A02 的 post-Primary failure 只用于确认运行时边界应与既有测量边界对齐；没有读取科学 pair outcome、representation、BDD 或 RBR。

## 路线不变量

审计 `2876` 条 selected transitions，target route-consistency violation 为 `0`，状态 `PASS`。V2.3 在 V2.2 全部 fail-closed 规则之上，新增 source/target 必须落在同一冻结 roadblock occurrence 的强制检查。
