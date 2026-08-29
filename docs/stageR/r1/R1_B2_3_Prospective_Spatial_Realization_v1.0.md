# R1 B2.3 Prospective Spatial Realization v1.0

状态：`IMPLEMENTED_FOR_SYNTHETIC_AND_READONLY_VALIDATION_NOT_ROLLOUT_AUTHORIZED`。

## TSB

baseline 与 treatment 共用一条 official native route reference，由 current ego 所在 lane/connector 沿 frozen route roadblocks 的 native outgoing-edge topology 构建。未来轨迹从 current ego 积分，phase clock 使用 absolute episode time；`TSB_GEN_V2_OPTION_A` 的 `-0.9×0.5 s / +0.4×0.7 s / -0.9×0.5 s` 及 baseline profile 均未改变。禁止冻结 initial x/y/heading 的 straight-line realization，禁止重规划重置初速/phase。

## HLC

Option-B progress schedule 保持不变。每次 replan 第一状态为 current ego，后续 source/target query 来自 native map geometry；任何 80-frame query 超出 native coverage 都 fail closed，禁止 silent extrapolation。

实现位于 `tools/r1_closed_loop_benchmark_v2.py`。本实现只通过 synthetic/unit 与旧资产只读诊断，未接入或启动 official rollout。
