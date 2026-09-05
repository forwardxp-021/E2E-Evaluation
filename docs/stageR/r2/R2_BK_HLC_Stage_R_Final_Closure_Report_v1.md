# R2-BK HLC Stage R 最终关闭报告

## 最终处置

HLC V4 canary 被冻结为 `VALID_NEGATIVE_ENGINEERING_RESULT`。HLC Stage R generator development 已关闭；不授权 HLC V5，也不授权 B0 roster 剩余 14 runs。由于 HLC 未收敛，完整的跨 family `G_R2` candidate 从未建立。

## B1.1 冻结结果

本报告只转录 SHA 绑定的 B1.1 recovery 结果，不重算、不重分类、不放宽 gate：

| 冻结 gate | 结果 | 证据 |
|---|---|---|
| Mechanism | FAIL | treatment 无可检测 retreat；`TREATMENT_RETREAT_LT_ONE`、`MONOTONIC_PENALTY_LT_0P1`；commit-latency delta `1.899923 s`，monotonic delta `0.0` |
| Endpoint | FAIL | treatment offset `0.342582 m`、heading error `0.052064 rad`、lateral velocity `0.275049 m/s` 均越界；route-progress delta `0.03152 m` 通过 |
| F_match | PASS | mean-speed delta `0.002750`、end-minus-start-speed delta `0.106555`、path-length delta `0.011338` 均通过冻结 caliper |
| Engineering | PASS | baseline/treatment max lateral acceleration 分别为 `0.520186/0.805108 m/s²`，冻结工程门通过 |
| target-offset decline | PASS | 原 analyzer 结果原样保留 |
| post-deadline hard jump absent | PASS | 未检测到超过冻结 `0.25 m` 门的 hard jump |
| actual-shadow observability | PASS | actual LQR telemetry 与 shadow audit 完整 |
| Official safety | FAIL | baseline 2 次、treatment 1 次 at-fault collision；两臂 drivable-area compliance 均为 true |

baseline 的 collision 计数为 2、treatment 为 1，只能支持“两臂均未通过冻结的零碰撞安全门”。单个 engineering canary 不构成 arm 间安全效应估计，因此不得解释为 treatment 安全改善，也不得解释为安全恶化。

## 身份治理

B1 canary identity `cc1abd3989065d8d` 标记为 `OUTCOME_EXPOSED_PERMANENT_ENGINEERING_ONLY`。B0 roster 其余 7 个 identity 保持未运行、冻结、永久 engineering-only；不补跑、不替换、不重排。A5 未选择的 26 条仍为 outcome-unexposed，只保留其既有 HLC audit-pool 角色，不自动转作 confirmatory roster，也不加入结果型 source blacklist。

## Family scope

- `COMBINED_G_R2_CLAIM = NOT_AVAILABLE`
- `HLC_CLAIM = DEVELOPMENT_NONCONVERGENCE_NEGATIVE_RESULT`
- `TSB_CLAIM = INDEPENDENT_FAMILY_CANDIDATE_PENDING_FRESH_VALIDATION`
- `CROSS_FAMILY_POOLING = PROHIBITED`

后续 TSB-only 验证是公开记录的 post-development scope amendment；它不能回写为原完整 `G_R2` 预注册成功。

## 执行边界

本阶段 `runner.run=0`、B1.1 recovery analyzer invocation `=0`、simulation `=0`。HLC/TSB 参数和科学阈值均未修改；R2-C、confirmatory smoke 与 RBR 均未启动。
