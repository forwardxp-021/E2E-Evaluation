# R2-BJ-A4 Scientific Owner 准备度请求 v0.1

## 结论

`APPLICABLE_POOL_INSUFFICIENT`。

冻结 source universe 在全部 1621 个 canonical logs 穷尽后，仅能形成 557 条满足基础 HLC 条件且 token/log 全局唯一的记录，低于预注册的 768 条 frame，缺口为 211。因此 A4 frame 未冻结完成，未打开任何 A4 speed、topology、curvature 或 V4 predicate 结果，并保持 fail-closed；不得进入 BJ-B。

## 历史处置

A3 原 11 条 generated/composite failure 的 `v_audit` 均低于 `3.0 m/s`，新增适用域处置为 `LOW_SPEED_OUTSIDE_V4_APPLICABILITY`，不回写其 A3 failure、不加入 outcome blacklist。A2 历史 `3feb5f93f24e5b77` 保持 `HISTORICAL_OPPORTUNITY_NOT_APPLICABLE_UNDER_CURRENT_V2_3`。原 10 条 legacy failure 中重新可构造且低速的记录按 low-speed applicability 处置。

## 治理

V4、morphology/capture 参数及全部阈值未改变。未选择 BJ-B roster；`runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动。
