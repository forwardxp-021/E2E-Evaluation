# R2-BJ-B0.1 Scientific Owner Canary Execution Readiness Request v0.1

## 请求结论

R2-BJ-B0.1 已补齐唯一两次 HLC V4 engineering canary 的生产执行控制面，现请求 Scientific Owner 审阅其执行就绪性。本文件不是执行授权；正式状态继续保持 `CANARY_AUTHORIZED = FALSE`、`NEW_RUN_BUDGET = 0`、`RUNNER_RUN = 0`。

## 冻结继承

B0 roster、16-run intended schedule、8 个 pair bindings、V4 generator/planner、morphology/capture 参数和全部阈值均未修改。生产路径继续绑定：

- B0 component manifest SHA256：`35a1282328b461f0b1edbbd39a4284870382ad52a83bd2975d9a91bc0ece1cf9`
- B0 schedule SHA256：`5493c5b402a3bc954d83d0914451c1f3dd38cddcfad8244291cf0a846d88918d`
- B0 pair-binding SHA256：`4e4eee55b816c8fa79cdc41ed0f8f99d9bd778e14c747ee13704998f71366950`
- B0.1 execution-component manifest SHA256：`f768033497a43f23cb1abdb674bf742737655e9d59b62d421eb5fa8dbf573568`
- protected CSV SHA256：`e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8`

B0.1 manifest 不包含自身，也不包含未来 Owner authorization record；授权记录只能单向引用 manifest。

## 唯一可申请 canary

未来唯一可申请的 schedule slice 固定为同一 pair 的两次执行：

1. run order 1：`R2BJB0-HLC-01-BASELINE`
2. run order 2：`R2BJB0-HLC-01-TREATMENT`

共同绑定 scenario token `cc1abd3989065d8d` 和 log `2021.10.01.16.53.37_veh-44_01126_01602`。run order 3–16 不可访问；不得换序、replacement、technical rerun、参数更新或重新选择 canary。

## 生产控制面闭合

真实 `runner.run()` 只有一个调用点。protected CSV、B0/B0.1 SHA closure、Owner 授权、精确预算 2、精确 `[1,2]` slice、Owner 绑定的 output/control roots 和路径无碰撞检查在控制流上均先于 runner construction；预算认领与 attempt ledger 原子持久化又先于每次 `runner.run()`。

baseline 只有在 runner report、80-row realized trace、80-row planner telemetry、80-row controller-visible telemetry、官方 safety metric 和 runner report 全部完整后，才允许构造 treatment runner。任何 architecture 或 infrastructure failure 均停止当前和剩余 schedule，不重试、不替换、不改参数。两次 attempt 后预算必须为 0，第三次调用机械拒绝；Owner 绑定 control root 防止通过更换 ledger 位置重复消费同一授权。

## 失败审计保证

新增 wrapper 不修改冻结 B0 `_states`、轨迹、参数或 controller 输入。正常路径返回与 B0 planner 完全相同的 trajectory object；若捕获 `B0ArchitectureViolation`，wrapper 会在异常重新抛出前，以临时文件、`fsync` 和原子 rename 写入独立 failure audit。该记录包含 run/pair/arm、iteration、absolute time、failure codes、完整 `error.audit`、realized current ego 和各冻结 SHA，并明确 `STOP_CURRENT_RUN` 与 `STOP_REMAINING_SCHEDULE`。

## 零运行验证

focused mutation tests 已覆盖：关闭授权、零预算、各冻结 SHA 错误、run 3 越界、treatment→baseline 倒序、schedule row 篡改、output collision 和 control-root 篡改均在 runner construction 前失败；测试内临时正授权仅使用 mock runner，验证成功路径恰好两次、baseline/treatment 先后顺序与预算 `2→1→0`。baseline/treatment architecture failure、baseline infrastructure failure、telemetry 不完整、runner construction failure、ledger serialization failure、第三次 attempt 和授权重复消费均按合同 fail-closed；failure audit 在异常传播后仍可完整解析，且 nuPlan 若将 planner 异常包装成失败 report，也不会把已持久化的 architecture failure 降级为 infrastructure failure。

未构造或启动真实 simulator，未调用真实 `runner.run()`。

## 请求 Owner 的下一项决定

若 Owner 决定授权，必须另建一次性 authorization record，并同时绑定上述四个 manifest/schedule/pair SHA、精确两个 run ID、精确预算 2，以及冻结的 output/control roots。当前关闭态授权文件不得直接改称已授权。

在 Owner 作出该决定前：

```text
BJ_B_ENGINEERING_SIMULATION_AUTHORIZED = FALSE
CANARY_AUTHORIZED = FALSE
NEW_RUN_BUDGET = 0
RUNNER_RUN = 0
R2_C_STARTED = FALSE
CONFIRMATORY_SMOKE_STARTED = FALSE
RBR_STARTED = FALSE
```
