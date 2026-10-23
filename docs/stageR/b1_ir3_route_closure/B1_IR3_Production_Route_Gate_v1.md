# B1-IR3 Production Route Gate v1

生产 gate：`tools/b1_native_route_infrastructure.py::production_route_precheck_v1_2`；调用点位于 B1 executor 的预算认领和 runner entry 之前。

## 五个不同概念

| 概念 | 本轮执行规则 |
|---|---|
| ROUTE_IDENTITY_VALID | 由 map name + 原完整 ordered IDs 重新计算 hash，必须等于冻结 spec |
| INITIAL_ROUTE_QUERY_PASS | 同一 production successor、原初始 ego、原冻结距离函数的有限请求必须成功 |
| LATENT_TOPOLOGY_RISK | 记录 v1.1 greedy 全 suffix 诊断；若有 dead end，必须有修复后原生路径证据，不能只重命名状态 |
| RUNTIME_ROUTE_COMPATIBILITY | 初始有限查询和修复后完整剩余 native suffix 均通过；逐段存在真实 outgoing edge，roadblock 顺序精确相同 |
| FULL_EPISODE_COMPATIBILITY_EVIDENCE | 静态 native topology + 冻结真实 call-entry 回归；不等同于新的完整闭环执行成功 |

同时核验 precontext hash。Identity 不符为 TECHNICAL_ERROR；初始有限请求失败为 INCOMPATIBLE；完整 native suffix 仍无法解析则 AMBIGUOUS。只有 COMPATIBLE 才可通过 route gate，且仍必须通过单独 authorization、fresh factory、callback、预算及输出 namespace 等门。

B1-18/20 原 AMBIGUOUS 的具体原因是已定位的 greedy dead end；现有同源 kernel 使用地图中已有分支，在相同 frozen route 上通过 native suffix 检查，因此满足这里的工程准入条件。没有降低科学阈值或豁免 unresolved path。

不要求所有理论未来 ego state 都可兼容。若实际运行离开原 route、无法解析当前 edge、出现终点覆盖不足或其他技术问题，继续失败并停止。不能用静态 COMPATIBLE 掩盖真实执行异常，也不能自动多次试跑。

## 授权和 attempt

本轮 `B1_RESUME_EXECUTION=NOT_AUTHORIZED`。后续 Owner 授权必须绑定新 source commit/hash，包括 v1.2 模块、旧 v1.1 依赖、planner import、executor、冻结配置和原 arm specs。

18 treatment 的原 attempt 1、35 次已认领预算和 5 个 NOT_RUN arm 原样保留。attempt 2 如获授权必须是新的 attempt namespace 与精确 slice；现有无重试账本不会被本轮重置或复用成一份新“未执行”账本。MAX_TOTAL_TECHNICAL_ATTEMPTS_PER_ARM=2 不变。

旧完整结果 01～17 NO_RERUN；19 与20 首次执行、18 baseline 首次执行和18 treatment attempt 2 均要 Owner 明确授权。分母为原20对，不替换、不删除。
