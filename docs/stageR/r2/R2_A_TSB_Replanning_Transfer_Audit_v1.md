# R2-A TSB Replanning Transfer Audit v1

## 审计范围

本审计只读取 8 个永久 engineering-only DEV identity 的 40 个有效 TSB 运行。冻结 excitation 在仿真前一次写定；没有在线改参、identity replacement、scientific threshold tuning 或 confirmatory 使用。

## 重复 replanning 语义

- Planner 每 0.1 s 以 absolute episode time 重建未来 80-state trajectory；LQR 使用 0.1 s discretization、10-step（1.0 s）lookahead。
- 随 episode time 前移，first-brake、release、second-brake 的边界每次相对 lookahead 向左移动 1 个 sample；阶段尾部逐步缩短，越过边界后从 lookahead 消失。这是 `phase shortening / boundary migration / phase disappearance` 的确定性来源。
- 两脉冲运行的 1 s lookahead 内 first-brake 可见 sample 数分布为 `{"n": 2528, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 9.0}`；release 为 `{"n": 2528, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 11.0}`；second-brake 为 `{"n": 2528, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 9.0}`。
- LQR 内部会从完整轨迹 pose 拟合 velocity/curvature profile；其 1 s reference speed 与显式 planner state10 speed 的 RMSE 分布为 `{"n": 40, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}` m/s。这是 trajectory fitting 的直接 telemetry 量化，不是 inverse-controller tuning。

## 对 transfer 的解释

中心 two-pulse 相对 single-brake reference 的 LQR peak-command 比为 `0.442374`，realized peak-decel 比为 `0.412484`。第一段制动尚在发生时，1 s lookahead 已逐步包含 release；LQR 因而看到被缩短、随后消失的第一段制动目标。release 后的正向速度意图又通过轨迹拟合与 motion-model 一阶滞后延续到 second-brake 边界，导致第二相位易合并或丢失。

结论：`absolute-time replanning + trajectory fitting` 决定 controller 可见目标，`LQR tracking attenuation + release-window carryover` 进一步削弱 realized phase formation。此结论只用于 R2-B architecture，不产生新的科学阈值。
