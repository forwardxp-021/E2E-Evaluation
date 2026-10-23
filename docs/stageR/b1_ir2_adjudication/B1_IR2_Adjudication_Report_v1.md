# B1-IR2 Adjudication Report v1

本轮状态：`B1_IR2_BLOCKED_ROUTE_ADJUDICATION`。

正式 B1 状态保持 `B1_TSB_BENCHMARK_QUALIFICATION_INCOMPLETE_INFRASTRUCTURE`。本轮完成原始 safety 判定、历史 route 矛盾分析和恢复计划冻结；没有修复 production route builder，也没有证明 B1-18/20 的未来完整闭环兼容性，因此不能宣布可以恢复执行。

## Recovered official safety

唯一科学规则来自执行前冻结的 `tools/r1_b2_8_r3_1_official_safety_adapter.py::FROZEN_PAIR_PASS_SEMANTICS`：每 arm 的 `number_of_all_at_fault_collisions_stat_value == 0` 且 `drivable_area_compliance_stat_value is true`；pair 必须 baseline AND treatment。解析复用 `r1_official_metric_canonicalizer.py`，只读取恢复后的官方 parquet，不从 trajectory 重算 safety，不引入阈值。

| 项目 | 结果 |
|---|---:|
| Official-safety evaluable arms | 34/34 |
| Collision PASS / FAIL | 34 / 0 |
| Drivable-area PASS / FAIL | 34 / 0 |
| Joint arm safety PASS / FAIL | 34 / 0 |
| Joint pair safety PASS / FAIL / ambiguous | 17 / 0 / 0 |

Arm CSV 记录每项 metric 的绝对路径、SHA256、原始 execution manifest SHA、metric config hash、offline finalize tool SHA、adapter/canonicalizer SHA 和判定规则。全部原始输入及 544 个恢复指标文件的 hash 均与 B1-IR recovery audit 核对。

## Existing 17 full contracts

原 runner 完成状态和 80 行 trace/planner telemetry、79 行 recorder transition、官方 16 类 metric、非空 simulation-log serializer 输出、原 execution manifest、artifact hashes、precontext 与 route identity 均已核验。机制和 F_match 直接继承冻结 Pair Results v1，没有重算。

| 01–17 pair gate | 结果 |
|---|---:|
| Mechanism PASS | 17/17 |
| F_match PASS | 17/17 |
| Official safety PASS | 17/17 |
| Full-contract complete | 17/17 |
| Full-contract scientific PASS / FAIL / ambiguous | 17 / 0 / 0 |

这里的 complete 是原执行加离线恢复的审计结论，不伪造新 `ARM_COMPLETE` manifest。历史缺失的 runner_report 没有被合成；runner 完成依据原已冻结的执行 manifest 和预算账本，并与完整 serializer/telemetry 交叉核验。前 17 对保持 NO_RERUN。上述 17/17 不能代替完整 20 SESSION 分母，也不证明 low-order nuisance 已消除。

## Route adjudication

旧 B1-IR precheck 使用无限前向请求，检查所有 frozen successor。Production 每次只请求当前 state 下冻结 `_tsb_required_distance` 给出的有限 7.9 秒规划距离。`CURRENT_PRECHECK = OVER_STRICT` 是对 B1-IR v1 的判定；它将全 route 缺陷直接等同于当前 arm 不兼容，证据不足。

06/10/14 的两 arm 共 480 次既有 production 输入回放全部通过，没有消费缺失 successor。历史执行有效。B1-18 treatment 的既有 trace 在 iteration 53（nominal 5.3 s）可由同一生产函数确定性复现异常：请求 82.231575 m，当前位置仅能解析约 81.612 m。故属于 `RUNTIME_PATH_DEPENDENT_INFRASTRUCTURE_DEFECT`，同时是给定该已记录输入的确定性结构失败，不是 safety 或机制失败。

新 checker 分别记录初始真实请求和全 topology 风险。初始真实请求均通过；全 route 缺陷改标 latent risk / AMBIGUOUS，不再声称初始不兼容。未来 realized state 未知，不能用历史成功或 nominal progress 猜测未来必然通过。B1-20 为 `ROUTE_AMBIGUOUS`，当前 gate 继续阻止 runner entry；它的初始可解析距离约 715.175 m、初始请求约 85.113 m，尚无覆盖整个闭环的静态证明。

本轮没有 route 修复或 retry 授权。B1-18/20 需要对应基础设施闭合或 Owner 对原分母技术未完成状态的明确规则；不能偷偷替换 SESSION，也不能删除分母成员。

## Authorization and zero-run

`B1_RESUME_EXECUTION=NOT_AUTHORIZED; RBR_TRAINING=NOT_AUTHORIZED; PRIMARY_EVALUATION=NOT_AUTHORIZED`。

`SIMULATION=0; RUNNER_RUN=0; TSB_ROLLOUT=0; HLC_ROLLOUT=0; RBR_TRAINING=0; PRIMARY_EVALUATION=0; V_EXECUTION=0; C_EXECUTION=0; NEW_SCIENTIFIC_OUTCOME_EXPOSURE=0`。

这里新增的是对已运行 B1 artifacts 的判定，不是新的 rollout outcome。没有执行 RBR inference、z64、H、BDD、MMD 或 Primary comparison。测试对 runner、planner、controller 入口设置禁止调用保护。

## 下一步

仅建议 B1-18/20 route infrastructure closure：保留原 identities、冻结科学代码和 20-pair denominator，证明或修复生产实际需要的路由查询，并明确剩余 ambiguity 的处理。之后才可进入 Scientific Owner review → B1 Resume Authorization。本轮完成后停止。
