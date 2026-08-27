# R1 官方 runtime determinism 核验报告 v0.2（失败收束）

## 执行范围

本次执行严格限于 B1.1 的 baseline-only runtime reproducibility；没有 HLC/TSB treatment、48-call smoke、
development roster、R4、representation、BDD、probe 或 RBR 操作。

冻结 runtime roster 共四个永久隔离场景：R-HLC 两个、R-TSB 两个，均来自只读且 outcome-blind 的选择流程。
计划预算为 8 个 `OFFICIAL_CLOSED_LOOP_RUN`。

## 实际结果

第一条 pre-run claim 为 `R-HLC__25944935eadb52f1__RUN_A`。官方 nuPlan 成功建造了一个 scenario，
但运行器报出 `AttributeError: R1RuntimeDeterminismPlanner has no compute_trajectory`；因此 scenario 本身失败。
该外层命令返回码为 0 并不表示 simulation 成功。

没有产生历史/context/traffic/background/ego/planner trace，也没有 collision 或 off-road/drivable metric，故下列
15 类冻结比较均不能完成：身份、map、route、history、raw context、canonical context、step、timestamp、traffic、
background、ego、planner、collision、drivable 和 technical status。

## 决策

- 已 claim official run：1/8；RUN_B 与余下三场景均未启动。
- background replay：`NOT_VERIFIED`；official replay：`NOT_READY`。
- 不允许把缺失 trace 当作相等；不存在可报告的 float max-absolute-difference。
- 已立即停止，没有第三次运行、重选场景或为通过而调整 threshold。
- 修复后的 planner 入口仅做静态代码修正，未用于本次受约束运行；任何新 runtime validation 需要新 scientific-owner authorization。

该失败是执行器接口缺陷，不是 scientific protocol deviation，也没有形成任何科学 outcome 解释。
