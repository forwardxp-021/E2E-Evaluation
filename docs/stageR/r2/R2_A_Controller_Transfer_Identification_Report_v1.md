# R2-A Controller Transfer Identification Report v1

## 状态

`CONTROLLER_TRANSFER_MODEL_DIAGNOSTIC = COMPLETE`。选择 8 个 HLC 与 8 个 TSB fresh DEV identities；与 R1 official 和 historical blacklist 的重叠均为 0。全部身份永久标记为 R2 engineering-only，禁止 R2 confirmatory 与 RBR scientific use。

冻结设计包含 HLC 5 条 excitation × 8 identities = 40 个有效运行，TSB 5 条 excitation × 8 identities = 40 个有效运行。由于 4 次只由技术故障触发的 fresh-root 重跑，实际 engineering simulations 为 84；scientific simulations 为 0。

## HLC transfer

- commanded→realized retreat gain：`{"n": 32, "min": 0.525976, "p25": 0.764702, "median": 0.868925, "p75": 0.995953, "max": 1.237549}`。
- commanded monotonic effect：`{"n": 32, "min": 0.774906, "p25": 0.848249, "median": 0.873463, "p75": 0.882841, "max": 0.908678}`；realized monotonic effect：`{"n": 32, "min": 0.825025, "p25": 0.873601, "median": 0.905245, "p75": 0.93108, "max": 0.967943}`。
- derivative cross-correlation lag：`{"n": 40, "min": 0.2, "p25": 0.3, "median": 0.3, "p75": 0.4, "max": 0.4}` s。
- commit lag：`{"n": 40, "min": 0.099966, "p25": 0.300155, "median": 0.400108, "p75": 0.500027, "max": 0.600201}` s。
- 以 p>=0.95 且保持到终点作为纯 engineering 描述的 settling delay：`{"n": 31, "min": 0.099955, "p25": 0.24996, "median": 0.399822, "p75": 0.450045, "max": 4.39917}` s。

HLC retreat morphology 在 closed-loop 中可传递，但 gain 与 lag 随 identity/条件变化；因此单一静态缩放不能同时处理深度、recommit 与 settling。

## TSB transfer

- first-brake peak-decel gain：`{"n": 40, "min": 0.0, "p25": 0.349745, "median": 0.39265, "p75": 0.687517, "max": 0.7855}`。
- realized first-brake peak decel：`{"n": 40, "min": 0.0, "p25": 0.314845, "median": 0.585797, "p75": 0.618765, "max": 0.7855}` m/s²。
- first/release/second peak lag 分别为 `{"n": 40, "min": 0.000107, "p25": 0.09991, "median": 0.199871, "p75": 1.299038, "max": 1.299973}`、`{"n": 32, "min": 0.000112, "p25": 0.099668, "median": 0.199282, "p75": 1.224879, "max": 1.29998}`、`{"n": 32, "min": 0.098987, "p25": 0.099911, "median": 0.149792, "p75": 0.199837, "max": 0.200486}` s。
- release response（相对 first-brake peak 的 realized acceleration 回升）为 `{"n": 32, "min": 0.225758, "p25": 0.288939, "median": 0.472373, "p75": 0.586966, "max": 0.770686}` m/s²，32/32 为正；这是零边界的 descriptive telemetry，不是新的 scientific threshold。
- two-pulse phase formation：`{"two_pulse_measurement_phase_count_distribution": {"0": 32}, "phase_loss_count": 32, "phase_merge_count": 0, "two_distinct_phases_count": 0, "phase_merge_probability": 0.0, "phase_loss_probability": 1.0, "release_positive_response_count": 32, "release_positive_response_denominator": 32, "release_positive_response_definition": "REALIZED_ACCEL_IN_RELEASE_WINDOW_RISES_ABOVE_FIRST_BRAKE_PEAK;DESCRIPTIVE_ZERO_BOUNDARY_ONLY"}`。

Telemetry 将 attenuation 分为两段：generator→LQR 与 LQR→realized。中心 two-pulse 比 single-brake reference 更弱，原因不是 scientific threshold，而是 repeated replanning 使 1 s lookahead 提前混入 release，再叠加 trajectory fitting、LQR/motion-model attenuation 和 release carryover。

## Surrogate 与验证

采用小型 deterministic linear surrogate，没有 ML 黑盒。HLC leave-one-identity-out retreat-depth MAE 分布为 `{"n": 32, "min": 0.000402, "p25": 0.002549, "median": 0.006514, "p75": 0.015329, "max": 0.023851}`；TSB peak-decel MAE 为 `{"n": 40, "min": 0.000344, "p25": 0.002044, "median": 0.012151, "p75": 0.021964, "max": 0.067706}` m/s²；TSB timing MAE 为 `{"n": 40, "min": 2e-05, "p25": 0.00033, "median": 0.028099, "p75": 0.028729, "max": 0.085417}` s。

## 边界

没有改变 scientific threshold，没有冻结最终 R2 generator 参数，没有选择 confirmatory identities，没有启动 RBR。R1 frozen assets、B2.9-E raw output 与 B3 forensic assets均未修改。
