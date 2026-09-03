# R2-BH HLC Target-Capture Architecture Forensic v1

## 结论

`V1_REANCHOR_DIAGNOSIS = PASS`。该结论来自代码与确定性合成几何审计，未运行 simulation。

## 数学结果

V1 先构造 `xy_before = source·(1-p)+target·p`，再对整条轨迹施加常量平移 `current_xy-xy_before[0]`。当 `p=1` 时，`xy_before=target`，所以终点相对 target center 的偏移等于当前相对 target 的偏移。常量 re-anchor 保证 state0 identity，却同时把当前 target-frame residual 搬到了全部 future states，不能形成有限时间 target-center attractor。

合成测试覆盖当前偏移 `0, +0.25, +0.50, -0.25, -0.50 m`，五种情况下 planned terminal offset 分别保持为相同数值，5/5 精确支持该不变量。

## R2-BH V2 原则

V2 将 behavior morphology 与 target capture 分离。state0 仍严格等于 current ego；state1+ 的 target-frame lateral/heading residual 使用固定 absolute-episode-time quintic 权重衰减，并在固定 capture end 归零。该内部 capture signal 不替代 frozen realized `p(t)` measurement，不使用自由空间路径或 geometry extrapolation。
