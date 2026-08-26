# R0 Statistical Analysis Plan v0.2

## Status

`DRAFT_OWNER_NUMERICS_BOUND_BUT_NOT_FROZEN`；`RBR_A/B/C_TRAINING_AUTHORIZATION=NOT_AUTHORIZED`。

本版保留 v0.1 的 24 个 hypothesis records、Holm family、bootstrap/permutation、probe/kernel/bandwidth/rank、whole-roster 与 evidence-level 合同，并正式绑定：

- D0 SESOI：`|paired standardized retention difference| >=0.10` + 95% CI 排除 0 + 至少 2/3 seeds 方向一致；仅解释为 temporal-retention diagnostic。
- D3：nominal FPR=.05 且 two-sided Wilson/预声明等价 95% upper CI `<=.075`；independent null units 不足为 `INCONCLUSIVE`，不降 gate。
- D4：24/24 equivalence margins 仍未批准，TOST/IUT 不得作为 frozen audit 执行。

## Prospective capacity binding

- D0：80% power 的独立 paired units=785；10 scenarios/log、ICC=.10 时 raw units=1492、logs=150。
- D1：缺 target-specific SESOI/prevalence/variance，样本量不可识别。
- D2：每个最终 nonempty stratum 至少 4 independent units；未形成 pre-treatment stratum occupancy 前不可宣称满足。
- D3：当观察计数取 floor(.05n) 时，至少 406 independent null trials 才使 two-sided Wilson upper 95% CI <=.075；依赖性降低 effective n。

当前 clean nuPlan 虽有 111 个 identity-clean logs，但仅 19 个含 runnable token；R0 audit holdout 未冻结，R4 也未冻结。因此本 SAP 不能升级为 frozen v1.0，不授权训练、仿真或 outcome evaluation。
