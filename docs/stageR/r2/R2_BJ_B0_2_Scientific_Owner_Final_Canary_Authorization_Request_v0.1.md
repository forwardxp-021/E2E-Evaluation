# R2-BJ-B0.2 Scientific Owner 最终 Canary 授权申请

## 请求结论

B0.2 已在不运行 simulator 的前提下，完成实际 TwoStageController/LQR 被动遥测与 canary 结果分析链的前置冻结。现仅请求 Scientific Owner 审阅；本文件不构成执行授权。

正式状态保持：

- `BJ_B_ENGINEERING_SIMULATION_AUTHORIZED = FALSE`
- `CANARY_AUTHORIZED = FALSE`
- `NEW_RUN_BUDGET = 0`
- `RUNNER_RUN = 0`

## 不变的冻结资产

- B0 roster 未改动。
- B0 schedule 未改动，唯一未来 slice 仍为 run order 1 baseline、run order 2 treatment。
- B0 pair bindings 未改动。
- B0.1 唯一生产 `runner.run()` 调用点、授权门、预算账本、baseline→treatment 顺序与停止语义未改动。
- V4 generator、planner、morphology/capture 参数及全部阈值未改动。

## B0.2 新增闭合项

1. 在真实 runner 构造后，对 `R1Primary80ScientificTimeControllerV1(81)`、`TwoStageController` 与 `LQRTracker` 作精确类型/迭代数验证。
2. 在预算认领前完成 recorder 可安装性验证，并在既有唯一 `runner.run()` 调用点之前安装到 `TwoStageController._tracker.track_trajectory`。
3. recorder 原样返回实际 LQR 的同一结果对象；独立重算冻结 LQR acceleration 与 tire steering rate shadow。写入失败、非有限值、少于或多于 79 行均按 infrastructure failure 停止，禁止重试。
4. 明确区分 80 行 `controller_visible_telemetry.jsonl` 为 `PLANNER_REFERENCE_STEERING / NOT_ACTUAL_CONTROLLER_COMMAND`，以及每 arm 79 行 actual-controller transitions。
5. 预冻结 pair analyzer：只读调用冻结 timestamp-aware realized extraction、native progress、HLC Option-B V2、Evaluator V2.1、F-match、四项 endpoint、engineering gates 与 official safety adapter，并增加 actual/shadow、capture-start→terminal target-frame offset 及 post-deadline hard-jump 审计。
6. 分析结果只有四个冻结状态；任何 architecture failure 优先于普通或 infrastructure 分类，任何 READY 状态都不自动授权剩余 14 runs。

## 零运行证据

- 2/2 official runner construction：PASS。
- 2/2 Primary80 time controller：精确类名且 `number_of_iterations() = 81`。
- 2/2 ego controller：`TwoStageController`。
- 2/2 tracker：`LQRTracker`。
- 2/2 passive actual-LQR recorder installation：PASS。
- 正式 output root：不存在。
- 正式 control root：不存在。
- simulator / `runner.run()`：0。

## Owner 若授权时必须绑定

未来独立授权记录必须同时、单向绑定：

- B0 component manifest SHA256：`35a1282328b461f0b1edbbd39a4284870382ad52a83bd2975d9a91bc0ece1cf9`
- B0 schedule SHA256：`5493c5b402a3bc954d83d0914451c1f3dd38cddcfad8244291cf0a846d88918d`
- B0 pair-binding SHA256：`4e4eee55b816c8fa79cdc41ed0f8f99d9bd778e14c747ee13704998f71366950`
- B0.1 execution-component manifest SHA256：`f768033497a43f23cb1abdb674bf742737655e9d59b62d421eb5fa8dbf573568`
- B0.2 execution-observability manifest SHA256：`dac808cb1f75c26c15223226d9b3c296de0256ff007427a86a2a7d14f6b5b62c`
- exact run IDs：`R2BJB0-HLC-01-BASELINE`、`R2BJB0-HLC-01-TREATMENT`
- exact run orders：`[1, 2]`
- exact budget：`2`
- 固定 production output/control roots。

Owner 需要回答的唯一问题是：是否针对上述精确 SHA 闭合包，授权一次 baseline→treatment、总预算 2 的 engineering canary？

即使未来 canary 达到 READY，剩余 14 runs 仍需新的 Scientific Owner 授权；R2-C、confirmatory smoke 与 RBR 均未授权。
