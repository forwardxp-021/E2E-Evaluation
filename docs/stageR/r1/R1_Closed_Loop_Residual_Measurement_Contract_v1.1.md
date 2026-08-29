# R1 Closed-loop Residual Measurement Contract v1.1

状态：`FROZEN_PROSPECTIVE_TIMESTAMP_AWARE`。v1.0 历史文件保持不动。

Primary 是 80 个连续 official simulator iterations：`iteration_index=0...79`，nominal cadence 10 Hz、nominal window `[0.0,8.0)`。实际物理时间戳必须有限且严格递增，并逐帧原样保留；禁止插值、外推和物理时间重标。`delta time_us` **不要求**精确等于 100000。

longitudinal acceleration、yaw rate、curvature 和 HLC `dp/dt` 全部使用 actual physical timestamps。0.3 s 与 0.5 s 的 frozen duration gates 仍分别解释为 nominal 3 samples 与 5 samples，不因 official timestamp jitter 改变 mechanism threshold。

HLC Primary：timestamp-aware mechanism、三项 Primary F_match、endpoint、timestamp-aware engineering。TSB Primary：timestamp-aware mechanism 与 Primary F_match。Safety 仍由 official closed-loop metric 决定。`INITIAL_PLANNED_TRAJECTORY` 只作 secondary generator-intent diagnostic，不决定 Primary readiness。
