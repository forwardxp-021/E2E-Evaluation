# R1 HLC Map Geometry Applicability Contract v0.1

状态：`PROPOSED_REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。本合同只使用 physics/map justified、pre-treatment deterministic 规则，不使用 B2.1 outcomes，不拟合任何 curvature、heading、speed 或 coverage percentage threshold。

## 必须全部满足

1. native source lane 可唯一解析；native target lane 可唯一解析且为 source 的原生相邻 lane；
2. source/target 与 frozen route 一致、同向行驶；
3. 两条 reference 均无 reversal、自交或退化 segment；
4. frozen 80-frame/8 s generator 的每一次 source/target query 都能由 native map topology 返回；
5. Primary 路径严格 `NO_EXTRAPOLATION`；缺失即 applicability fail，不允许 silent polyline extension；
6. pre-rollout deterministic generator audit 同时满足既有冻结 engineering limits：lateral acceleration ≤6 m/s²、yaw rate ≤1 rad/s、curvature ≤0.5 m⁻¹。

以上均为 exact topology/既有冻结工程门禁。若 future native map API 仍需要新的数值 geometry rule，必须返回 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`，不得从历史 12 cases 选择数值。
