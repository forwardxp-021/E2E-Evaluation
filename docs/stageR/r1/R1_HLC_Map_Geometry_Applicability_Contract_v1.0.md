# R1 HLC Map Geometry Applicability Contract v1.0

状态：`FROZEN_PROSPECTIVE_OUTCOME_BLIND`。

必须同时满足：native source、native adjacent target、route consistency、same travel direction、no reversal、no self-intersection、全部 80-frame native query coverage 和 `NO_EXTRAPOLATION`。

pre-rollout deterministic engineering audit 仅使用既有冻结 limits：lateral acceleration ≤6 m/s²、yaw rate ≤1 rad/s、curvature ≤0.5 m⁻¹。不得加入任何其它 geometry numeric cutoff；不得从 B2.1 outcome 拟合 heading、coverage、speed 或 curvature threshold。
