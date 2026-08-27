# R1 技术烟雾报告 v1

状态：`TECHNICAL_SMOKE_COMPLETE_CORE_ONLY`。本次恰好执行 48 条 trajectory-only technical rollouts：R-HLC 6 个历史/R0-development scenario 的 baseline+3 candidates（24 条）和 R-TSB 对应 24 条。没有创建 48/58 正式 development roster，没有读取 embedding、BDD、probe、checkpoint 或 RBR。

## roster 与隔离

- roster 由固定 salt `R1_PHASEB_TECHNICAL_SMOKE_ROSTER_V1` 对历史/R0-development source 的 scenario token/log 做 deterministic hash 排序；各 family 取 6 个、均至少 3 个 logs。
- 全部条目为 `TECHNICAL_SMOKE_ONLY`、`EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER` 与 `EXCLUDED_FROM_FUTURE_R4_CONFIRMATION`。
- pre-context 使用被批准的 `CONDITION_IDENTICAL_1S_WARMUP`（完整 10 帧、0.1s），不是缩短窗口。每个 pair 均在生成前核验 raw history 与 canonical context hash 相同。

## 结果

|family|candidate|技术执行|F_match|机制 pair gate|运动学完整性|建议|
|---|---|---:|---:|---:|---:|---|
|R-HLC|HLC_MILD|0/6|0/6|0/6|0/6|NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE|
|R-HLC|HLC_NOMINAL|0/6|0/6|6/6|0/6|NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE|
|R-HLC|HLC_STRONG|0/6|0/6|6/6|0/6|NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE|
|R-TSB|TSB_MILD|6/6|6/6|0/6|6/6|NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE|
|R-TSB|TSB_NOMINAL|6/6|6/6|0/6|6/6|NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE|
|R-TSB|TSB_STRONG|6/6|6/6|0/6|6/6|NOT_RECOMMENDED_AFTER_TECHNICAL_SMOKE|

## 安全与 runtime 边界

所有显示为运动学完整性的结果仅代表有限值、时间单调、非负速度及 HLC 横向加速度/yaw/curvature 的预声明上限。`nuplan` runtime 可用性为 `False`；本机缺少完整 external runtime 时，未声称 official closed-loop background replay、碰撞/off-road 或 traffic-light API safety 已通过。因此任何 `RECOMMENDED_AFTER_TECHNICAL_SMOKE` 仅是 core generator 的技术建议，不是正式 generator freeze 或 scientific efficacy 结论。

## 不可变性

候选参数 JSON 和 smoke roster 均在任何 candidate rollout 前写入并由 execution manifest SHA 绑定。未因本报告中的通过或失败修改 context/mechanism 定义、F_match caliper 或 threshold。
