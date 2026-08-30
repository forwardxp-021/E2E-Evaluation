# R1 HLC Pretreatment Dynamic Clearance Contract v1.1

v1.1 只增加 replay observation horizon completeness 语义与 oriented-footprint conformance，不修改 v1.0 的任何数值。

全局 official replay frame/lidar timestamp stream 必须覆盖 iterations 0...79 / `[0,8s)`：存在起点覆盖、终点覆盖，且任意内部 observation gap 不超过已冻结的 0.25 s。单个 actor 可以进入、退出、出现或消失，不要求每个 track 覆盖 8 秒。

只有 `GLOBAL_OBSERVATION_HORIZON_COMPLETE` 且 original replay tracks 为空，才能记为 `DYNAMIC_CLEAR_NO_ACTORS`；缺少全局时域绑定一律 `NOT_ELIGIBLE`。

公共包络以每个 arm 的 prospective XY tangent 为 heading，包含 baseline 与 treatment 的完整 official runtime oriented ego footprint，再使用原冻结 3.0 m longitudinal / 0.5 m lateral buffer。禁止 5×2 generic fallback、arm-specific eligibility 和 rollout 后重算 eligibility。
