# R1 B2.3 Scientific Owner 审批记录 v1.0

状态：`OWNER_REVIEW_RECORDED_FOR_B2_4_FINAL_FREEZE`。

1. `REALIZED_CLOSED_LOOP_EGO` 批准为 Primary；`INITIAL_PLANNED_TRAJECTORY` 仅为 `SECONDARY_GENERATOR_INTENT_ONLY`。
2. `CURRENT_EGO_CONTINUOUS_REPLAN` 获批，但 `trajectory[0]` identity 不足以单独通过；未来段必须接受 structural first-segment continuity audit。
3. HLC map applicability 获批，禁止新增 geometry numerical threshold。
4. TSB initial-speed floor `2.0 m/s`：`OWNER_APPROVED_CONDITIONAL_ON_EXPLICIT_BASELINE_EXECUTION_BINDING`。
5. HLC clearance numerics 获批：8.0 s、0.1 s nominal query、0.25 s actor interpolation gap、3.0 m longitudinal buffer、0.5 m lateral buffer。0.25/3.0/0.5 继承既有 Stage7L treatment-independent engineering clearance，不是 R1 outcome-derived threshold。
6. ego footprint 仅允许 official runtime vehicle parameters；fallback 禁止。actor footprint 仅允许 official track dimensions。map/actor extrapolation 均禁止。
7. exact `100000 us` physical cadence rule 被拒绝用于 final freeze；official timestamp jitter 本身不得自动触发 `NOT_EVALUABLE`。

保持不变：HLC Option-B、TSB Option-A、所有 frozen mechanism numerical thresholds 与 F_match calipers。本审批不授权 rollout、enumeration、roster 或 RBR。
