# R1 Development Roster Freeze 就绪性 v0.5

## 总状态：NOT_READY

v0.4 保持为历史记录，未被改写。本版本仅加入 V2 的 fail-closed 结果；不创建 development roster，也不授权
48-call smoke。

| 组件 | 状态 | 说明 |
|---|---|---|
| HLC Option B / endpoint | `READY` | 既有冻结不变；本次未修改 protocol。 |
| TSB baseline/treatment contract | `READY` | 既有冻结不变；本次只运行 baseline HLC 的首条 V2 run。 |
| official DB/map binding | `READY` | V2 authorization 中的 SHA binding 未变。 |
| V2 planner interface | `READY_BY_ZERO_BUDGET_PREFLIGHT` | `AbstractPlanner` mock 预检通过；不等于 replay verification。 |
| 48-call core budget executor | `READY_BY_FAIL_CLOSED_PREFLIGHT` | 可 pre-claim 并在上限前拒绝；本文件不授予其使用权。 |
| official closed-loop execution path | `NOT_READY_AFTER_V2_TECHNICAL_FAILURE` | V2 executor 对 Parquet metric 发现遗漏，1/8 后停止。 |
| background replay determinism | `NOT_VERIFIED` | 没有形成任何 V2 A/B pair。 |
| collision/offroad metric comparison | `NOT_EVALUABLE` | 指标文件实际存在但不在已冻结 V2 executor 的 JSON discovery 集合内。 |
| new compliant 48-call smoke | `PENDING_SEPARATE_SCIENTIFIC_OWNER_AUTHORIZATION` | runtime determinism 仍未验证；明确禁止启动。 |
| RBR A/B/C | `NOT_AUTHORIZED` | training authorization 未改变。 |

四个 runtime-validation scenario/log 继续永久隔离，不得重选或迁入 development/technical-smoke/R4 roster。
如需修复 V2 executor 的 output discovery，必须作为新的实现授权处理；不得把本次已 claim 的 V2_RUN_A
或未完成的 V2 cap 当作可恢复余额自动使用。
