# R1 TSB Baseline Discrete Execution Binding v1.0

状态：`TSB_BASELINE_DISCRETE_EXECUTION_BOUND`。

既有 `SINGLE_CONTINUOUS_BRAKING` baseline 保持不变：acceleration=`-1.0 m/s²`，nominal 10 Hz grid 上 active sample indices 为 11...20，共 10 个 active integration intervals，离散总速度损失为 1.0 m/s。

追溯来源：`tools/r1_residual_generators.py::TSB_BASELINE` SHA-256 `3c39322c0fbc82d9b5494c3ea9966606081a238ada93ea00eae4c286eb1e03f0`；当前批准实现 `tools/r1_official_technical_smoke_planner.py::tsb_profile` SHA-256 `7afb15ab4196fcc6952d003672baa925c2b44b43e7f8e4daf2941e4690c9f7cc`。本 binding 未读取或使用 B2.1 outcome。
