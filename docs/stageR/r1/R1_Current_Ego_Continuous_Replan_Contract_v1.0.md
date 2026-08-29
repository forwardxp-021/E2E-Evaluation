# R1 Current-Ego Continuous Replan Contract v1.0

状态：`PROSPECTIVE_IMPLEMENTATION_CONTRACT_PENDING_FINAL_FREEZE`，适用于 HLC 与 TSB。

每次 planner call 的 `trajectory[0]` 必须由同一个 `current_ego` 对象构造，position、heading、speed、timestamp 全部 exact identity；这不是事后 tolerance gate。审计必须分别报告四项 `first-state minus current-ego`，任一非零即 implementation failure。

所有 treatment phase 使用 `ABSOLUTE_EPISODE_TIME`。重规划只从 current ego 向未来积分，不得把 phase clock、初始位姿或初始速度重置到 episode 0。TSB baseline/treatment 使用同一 native route reference；HLC 使用当前 native source/target corridor，Option-B progress schedule 不变。
