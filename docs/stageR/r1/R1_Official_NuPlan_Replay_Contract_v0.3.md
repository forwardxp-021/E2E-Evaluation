# R1 官方 nuPlan Replay 合同 v0.3（V2 fail-closed）

## 状态

这不是 v1.0 通过合同。`BACKGROUND_REPLAY_DETERMINISM = NOT_VERIFIED`，
`OFFICIAL_REPLAY = NOT_READY`，并且本文件不授权 48-call technical smoke 或任何 planner rollout。

## 已绑定条件

- `MASTER_SEED=2026082701`，DB fingerprint 为
  `5b53ad42497fe6926c73936970658a3717d1c2cc51077812d5284a57fd242489`，map fingerprint 为
  `a85e17eba18e5fdd65148705844b8f189bb4d4373a1d82805e1f8ffd4ae8afb3`。
- 使用原四行 frozen runtime roster；HLC 只允许 decisive monotonic baseline，TSB 只允许 single continuous
  braking baseline；无 treatment。
- V2 interface preflight 已以零预算通过，确认当前 repaired planner 能经 nuPlan `AbstractPlanner` 接口调用。

## V2 失败收束

V2 的首条已 claim 运行完成了 149 个 simulation steps，并写入 planner trace/binding；但执行器只扫描 JSON
metric payload，未将 nuPlan 写出的 Parquet collision/drivable metrics 计入冻结比较。因此该 run 按执行器规则
被标为 technical failure，整批在 1/8 立即停止。

没有 RUN_B、没有可接受的 A/B comparison、没有四对 15-category exact equality，也没有第九次 pre-run
拒绝检查。因此任何声称 bound-runtime background replay 已验证的表述均不成立。

## 未来条件

本合同不提供恢复执行的权限。只有新的 scientific-owner authorization，且新的 executor/metric-discovery
binding、fresh fail-closed ledger 与冻结 roster 处理规则均被明确记录后，才可考虑新的 runtime validation。
