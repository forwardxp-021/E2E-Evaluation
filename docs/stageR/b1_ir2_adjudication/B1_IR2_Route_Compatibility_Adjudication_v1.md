# B1-IR2 Route Compatibility Adjudication v1

结论：`CURRENT_PRECHECK = OVER_STRICT`（指 B1-IR v1）。无限前向请求暴露了真实的 route topology 缺陷，但不能据此判定缺陷必然在当前有限 episode 被消费。该缺陷也不是虚构：B1-18 treatment 已在其有限生产请求中真实触发。

## 逐案比较

全部 route index 为零起始。两 arm 的 frozen route identity 相同；完整 route 序列、native edge traversal、初始输入、各次成功调用和首个失败输入见 `B1_IR2_Route_Audit_v1.json`。前向容量由同一 production builder 的请求边界二分探测得到，仅作诊断。

| Pair | 缺失 19339 的 route index | Runtime 当前 cursor | Runtime reference 最大消费 index | 原始结果 | 初始容量 / 首次请求（m） |
|---|---:|---|---:|---|---|
| 06 | 4 | 1..1 | 1 | 两 arm 各 80/80 成功 | 292.055 / 98.249 |
| 10 | 11 | 2..2 | 4 | 两 arm 各 80/80 成功 | 769.722 / 93.571 |
| 14 | 4 | 1..1 | 1 | 两 arm 各 80/80 成功 | 199.596 / 68.004 |
| 18 | 3 | treatment 0..0 | 2；iteration 53 请求 index 3 时失败 | baseline NOT_RUN；treatment 53 成功 + 1 失败 | 142.157 / 82.683 |
| 20 | 10 | NOT_RUN | UNKNOWN | 两 arm NOT_RUN | 715.175 / 85.113 |

06/10/14 的缺陷既没有被当前 ego cursor 到达，也没有被成功生成 reference 所需的前向 horizon 消费。完整 episode 为 iterations 0–79，nominal 0–7.9 s；每次 reference 仍向前规划 7.9 s，不能只凭自车行驶距离判断 reference 范围。本轮回放了全部 480 个有限生产查询，确认它们没有请求有问题的 transition。它们的历史有效性为 YES；一般未来 route 风险为 YES，两者并不冲突。

## B1-18

分类：`RUNTIME_PATH_DEPENDENT_INFRASTRUCTURE_DEFECT`。

原 treatment trace 有 54 行 call-entry state，planner telemetry 有 53 行成功产出。最后一行是 iteration 53，nominal 5.3 s，rear axle 为 (588079.912849, 4475752.550421)，speed=10.409060 m/s。冻结距离函数请求 82.231575 m，而该 state 的 native route 可解析边界约为 81.612 m，随后需要进入 19339 但无 outgoing successor。同一函数在已记录输入上复现同一异常；这不是一次新 planner call。

失败发生在 reference lookahead，需要的 successor 未必已被车身实际驶入。初始查询通过，因此不能归类为起点的确定性不兼容。当前 route/stack 未修复；attempt 1 永久保留，不具备现配置下可执行的 attempt 2 条件。技术 retry 的类别资格不等于执行资格或 Owner 授权。

## B1-20

最终：`ROUTE_AMBIGUOUS`；`DO_NOT_ENTER_RUNNER`。

初始有限请求通过，缺陷位于更远的 route index 10。无限检查不能证明它会被正式 80-step episode 消费，故撤回原“已证明 INCOMPATIBLE”的含义。但是当前没有 future realized position、edge selection 和 required distance 的完备静态范围证明，不能宣布整个闭环 compatible。715 m 的初始容量仅为起点一条分支的结果，不能替代全 episode 证明。

## 历史有效性与未来 gate

06/10/14 的既有完整结果全部保留；不做 retroactive invalidation，不重跑。五个存在 tail defect 的 pair 的一般未来兼容性标为 AMBIGUOUS。当前 production gate 仅接受 COMPATIBLE，ambiguity 因此继续阻止新的 arm entry。这是尚未闭合的保守处理规则，不声称已经获得全 episode production equivalence；本轮主状态据此为 `B1_IR2_BLOCKED_ROUTE_ADJUDICATION`。
