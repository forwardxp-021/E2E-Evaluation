# R1 Current-Ego Continuous Replan Contract v1.1

状态：`FROZEN_PROSPECTIVE_STRUCTURAL_CONTINUITY`。

每次 HLC/TSB planner call 的 `trajectory[0]` 必须与 current ego position、heading、speed、timestamp exact construction identity。treatment phase clock 使用 `ABSOLUTE_EPISODE_TIME`。

仅满足 state0 identity 不足以通过：state0→state1 必须来自同一 native/offset-preserving reference construction，并报告 first-segment distance、actual dt、tangent heading error 与构造来源。该 audit 不增加新的 numerical geometry threshold；它验证不存在重新锚定、centerline snap、重复 phase-0 或非有限/零长度 future segment。
