# R1 HLC Pretreatment Dynamic Clearance Proposal v0.1

状态：`PROPOSED_NOT_FROZEN_ALL_NEW_NUMERICS_REQUIRE_OWNER_APPROVAL`。本设计借鉴 `tools/stage7l_dynamic_clearance.py` 的 outcome-blind common-envelope 思想，但不继承其 15 s、60 m、5 m/s 或其他 Stage7L 数值。

## R1 专用设计

- 时间域固定为 R1 实际 8 s、dt=0.1 s；使用原始 replay tracks、official map 与 traffic-light/route 数据，不读 planner outcome、representation、BDD 或 RBR。
- 用 unchanged HLC baseline 与 Option-B progress schedule 在 native source/target geometry 上构造共同空间包络；同一个 envelope 同时覆盖 baseline+treatment，不能按 arm/dose 改变。
- 每个时间点对 source-to-target convex corridor 与 original replay actor footprint 做占用冲突检查；不得外推 reference 或 actor track。
- ego footprint 应优先来自 official runtime vehicle parameters，而非新硬编码。若建议复用既有 Stage7L footprint/buffer，其冻结来源必须先由 owner 确认；参考实现 SHA-256 为 `50253b75eed8473b1141b3b76d51ce755f1a8df1f7a2f0690a4075b35a3df129`。

## 尚未批准的数值

actor interpolation gap、longitudinal/lateral buffer、footprint fallback、track-horizon completeness 等所有新增数值均为 `TBD_REQUIRES_OWNER_APPROVAL`。本阶段不冻结数值、不选择 scenario、不运行 smoke。
