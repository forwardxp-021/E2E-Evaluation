# R1 Development Roster Freeze Readiness v0.4

## 总状态：NOT_READY

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| HLC amendment | READY | owner 已绑定 Option B；没有基于新 outcome 改参。 |
| HLC generator | READY | V2 Option B 已冻结。 |
| HLC endpoint | READY | resolution-based Primary 与 strict secondary 已冻结。 |
| TSB generator | READY | V2 baseline/treatment 合同未改。 |
| official nuPlan DB | READY | 本地 DB 与 map binding 已核验。 |
| replay seed | READY | `MASTER_SEED=2026082701` 已绑定。 |
| fresh source universe | READY | 只读、outcome-blind source universe 已存在。 |
| 48-call executor | NOT_READY | 本次 runtime executor 首条 official run 出现接口缺陷；虽已修正代码但未获重新授权验证。 |
| traffic-light / route trace | NOT_READY | 首次运行未产生 trace。 |
| collision / offroad metric | NOT_READY | 首次运行未产生官方 metric。 |
| background replay | NOT_READY | 4 个 A/B pair 均未完成。 |

## 48-call smoke

`NEW_COMPLIANT_48_CALL_SMOKE = PENDING_SEPARATE_SCIENTIFIC_OWNER_AUTHORIZATION`，并且当前尚不具备 owner-review 放行条件：
runtime determinism 为 `NOT_VERIFIED`，official replay 为 `NOT_READY`。本文件不选择真实 48-call roster，
不执行 smoke，也不授予任何执行权限。

## 隔离与下一步

四个 runtime-validation scenario/log 已永久加入 future smoke blacklist，future selector salt 已在任何 future
candidate enumeration 前冻结。后续若要恢复 runtime validation，必须先由 scientific owner 明确授权新的绑定执行；
不能复用本次已 claim 的 RUN_A 作为通过证据。
