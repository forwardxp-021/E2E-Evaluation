# R1 TSB Route Realization Contract v1.1

状态：`FROZEN_PROSPECTIVE_ROUTE_CONTINUITY`。

Option-A 与 baseline acceleration profiles 完全不变。baseline/treatment 使用同一 native route reference；当前 native edge 与其 outgoing topology 用于消解 repeated roadblock occurrence。current ego signed lateral offset 和 heading offset 进入 offset-preserving route construction，state0 exact，state1+ 不得 centerline snap。

所有 query 由 native route coverage 提供，禁止 extrapolation。curved road、nonzero lateral/heading offset 和 repeated roadblock 已通过 synthetic adversarial test；未执行 rollout。
