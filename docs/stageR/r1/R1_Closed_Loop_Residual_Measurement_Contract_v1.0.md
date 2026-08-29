# R1 Closed-loop Residual Measurement Contract v1.0

状态：`PROSPECTIVE_IMPLEMENTATION_CONTRACT_PENDING_FINAL_FREEZE`。继承 `R1_HLC_Residual_Benchmark_Scientific_Amendment_v1.0`，不修改 frozen mechanism thresholds 或 F_match calipers。

## Primary measurement

Primary source 唯一为 `REALIZED_CLOSED_LOOP_EGO_TRAJECTORY`：official simulation iteration 0–79、80 frames、dt=0.1 s、窗口 `[0.0,8.0)`。必须读取每次 simulator iteration 的 actual `current_ego`；禁止插值、外推或用 planned trajectory 替代。iteration index、current-ego physical timestamp 任一不满足 exact grid 时，状态为 `NOT_EVALUABLE_TEMPORAL_GRID`。

- HLC Primary：mechanism、三项 Primary F_match、endpoint、engineering。
- HLC Primary F_match：mean speed、end-minus-start speed、path length。
- HLC `heading_change_abs_total`：仅 `SECONDARY_MECHANISM_PROXIMAL_AUDIT`。
- TSB Primary：mechanism、Primary F_match。
- Safety：official closed-loop metric。

## Secondary measurement

`INITIAL_PLANNED_TRAJECTORY` 仅为 `GENERATOR_INTENT_DIAGNOSTIC`，不得决定 Primary pair readiness，不得替换 realized failure/not-evaluable 状态。
