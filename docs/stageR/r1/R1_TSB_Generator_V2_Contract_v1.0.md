# R1 TSB Generator V2 合同 v1.0

状态：`FROZEN_GENERATOR_PARAMETERS_EXECUTION_NOT_AUTHORIZED`。

## 冻结定义

baseline 保持已批准的 `SINGLE_CONTINUOUS_BRAKING` 定义。treatment 冻结为 `TSB_GEN_V2_OPTION_A`：

- first brake：`-0.9 m/s² × 0.5 s`
- release：`+0.4 m/s² × 0.7 s`
- second brake：`-0.9 m/s² × 0.5 s`

TSB mechanism contract 完全不变：brake `≤-0.80 m/s²`、release `≥-0.20 m/s²`、minimum brake/release 各 `0.3 s`、merge gap `<0.3 s`；pair 仍要求 1 phase vs 2 phases、release fraction `≥0.15`、second peak ratio `≥0.50`。

## 证据与授权边界

参数冻结只依据 Phase B0 的 synthetic-only physical compatibility evidence；没有依据旧 smoke outcome 改参。该合同冻结 generator 参数，但不授权 smoke、真实 scenario rollout、formal roster selection 或 RBR training。
