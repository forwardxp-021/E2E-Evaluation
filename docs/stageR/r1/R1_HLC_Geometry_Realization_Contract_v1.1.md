# R1 HLC Geometry Realization Contract v1.1

状态：`FROZEN_PROSPECTIVE_FINAL_XY_TANGENT_HEADING`。

HLC Option-B progress schedule 完全不变。source/target 均来自 native map geometry 且禁止 extrapolation。最终 XY 先由 frozen progress 混合 native source/target，再保留 current-ego residual anchor；Primary heading、yaw 与 curvature 仅从最终 XY tangent/actual timestamps 推导。

禁止分别 `np.unwrap(source_heading)`、`np.unwrap(target_heading)` 后直接线性混合。`+179°/-179°` synthetic 通过 short angular geometry；state0 exact，并对 state0→state1 执行 structural audit。没有新增 geometry threshold，也没有 rollout。
