# R1 正式 Development Roster Freeze 就绪性 v0.1

结论：`NOT_READY`。

本文件不创建 HLC 48 或 TSB 58 的正式 roster。技术烟雾 roster 的 12 个历史/R0-development scenario 永久标为 `TECHNICAL_SMOKE_ONLY`、`EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER` 和 `EXCLUDED_FROM_FUTURE_R4_CONFIRMATION`。

## 已满足的前置条件

- frozen 10-frame canonical context 工具可执行，36 个 paired candidate checks 的 `pre_context_raw_hash` 与 `canonical_context_json_hash` 均相同。
- HLC Option-B 与 TSB Option-A synthetic unit tests 14/14 通过。
- 48/48 trajectory-only technical smoke rollout 已执行；无非有限数或运行时中断。
- TSB 三个 candidate 均通过 frozen trajectory F_match（6/6 each）；HLC NOMINAL 与 STRONG 均通过 HLC mechanism pair gate（6/6 each）。

## 阻断项

|项|观察|影响|
|---|---|---|
|HLC frozen F_match|MILD/NOMINAL/STRONG 的 heading-change caliper 均 0/6|无 HLC candidate 同时达成 F_match 与 mechanism gate|
|HLC 运动学完整性|三档 treatment 皆超过预声明 lateral/yaw/curvature 至少一项上限，0/6|不得推荐 HLC generator|
|TSB mechanism gate|MILD=1 phase；NOMINAL/STRONG 虽 2 phase 但 release fraction <0.15，均 0/6|无 TSB candidate 达成冻结 gate|
|official external runtime|本机无 `nuplan`；未运行 official background replay、off-road/collision 或 route/traffic-light API|technical core 不能替代正式 external-planner integration|

因此不满足“至少一个 candidate 同时具有可用 mechanism、frozen F_match trajectory/route 和无实现缺陷”的 readiness 条件。`RECOMMENDED_AFTER_TECHNICAL_SMOKE` 候选为零；不得为了完成规划目标而把失败 candidate 标为推荐。

## 下一步（需新的科学负责人/实现授权）

1. 先排查并以 versioned implementation-definition amendment 处理任何可证实的 state-machine 或 external planner integration bug；不得改变 frozen context/mechanism threshold。
2. 在不重复或扩展本次 48 条技术烟雾额度的前提下，恢复 official runtime/history/background replay 与 route-control API 的实现核验。
3. 只有上述阻断项消除、重新取得明确 smoke/roster 授权后，才能 outcome-blind freeze 正式 48/58 roster。
