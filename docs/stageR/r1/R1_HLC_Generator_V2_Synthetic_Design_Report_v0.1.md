# R1 HLC Generator V2 合成设计报告 v0.1

状态：三个方案均为 `PROPOSED_NOT_FROZEN`；没有运行真实 scenario。

## 共用约束

baseline 始终是 `DECISIVE_MONOTONIC_LANE_CHANGE`，treatment 始终是 `HESITANT_RETREAT_RECOMMIT`，各 phase 用 quintic C2 joins。冻结 mechanism threshold 未改变；新三项 Primary F_match 使用原 R0 raw-scale caliper。合成包络覆盖速度 `4.992095–13.292885 m/s`、lane width `2.7/3.2/4.2 m`，每个选项 12 个 cell。

## 方案结果

|方案|advance / hold / retreat / recommit|总时长|mechanism pair|三项 F_match|最差横向加速度|最差 yaw-rate|最差 curvature|安全裕量判断|
|---|---|---:|---|---|---:|---:|---:|---|
|A|0→0.35/1.2s；0.5s；退 0.15/0.8s；2.2s|4.7s|latency Δ 2.6s；monotonic Δ -0.126296|12/12|5.292296|0.999022|0.197363|通过，但 yaw 裕量仅 0.000978|
|B|0→0.38/1.4s；0.6s；退 0.16/1.0s；2.4s|5.4s|latency Δ 3.2s；monotonic Δ -0.127303|12/12|4.403973|0.850528|0.167253|12/12 通过，裕量中等|
|C|0→0.42/1.6s；0.7s；退 0.18/1.2s；2.6s|6.1s|latency Δ 3.8s；monotonic Δ -0.135493|12/12|3.728392|0.728908|0.144300|12/12 通过，裕量最大但时长最长|

所有 synthetic fixtures 的 baseline/treatment 均精确到达目标 parallel-lane center，终端 heading error、lateral velocity 和 pair route-progress delta 均为 0；这不替代真实地图下的 endpoint 审计。`heading_change_abs_total` 仅作 secondary descriptive audit，部分 cell 超过旧 caliper，不影响 Primary qualification，也未据此删除 pair。

owner 下一轮须在 A/B/C 中选择并单独冻结；本报告不自行选择 generator。
