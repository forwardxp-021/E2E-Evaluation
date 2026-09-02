# R1 B3 R2 Repair Family Decision Matrix v1

## 决策边界

本矩阵只比较 architecture-level repair family，不实施 generator、controller、measurement 或阈值变更，也不从 24 个 outcome-exposed R1 identities 反推任何数值。

| Repair family | 机制 | 优点 | 主要风险 | 建议 |
|---|---|---|---|---|
| A. STRONGER_OPEN_LOOP_TRAJECTORY_INTERVENTION | 保持 measurement threshold，增加 planner-intent margin | 简单，可能恢复 realized separation | 忽略 closed-loop tracking 与 timing lag，可能放大 endpoint/safety 问题 | SECONDARY；仅 fresh engineering-only canary 可开发 |
| B. CONTROLLER_AWARE_TRAJECTORY_SHAPING | 根据 closed-loop tracking dynamics 设计带裕量的 trajectory | 直接针对 HLC attenuation、TSB command collapse 与 settling | 需要独立开发身份和系统辨识，不能用 R1 official outcome 调数值 | PRIMARY_RECOMMENDED |
| C. FEEDBACK_CALIBRATED_GENERATOR | 在永久 engineering-only development identities 上校准 realized mechanism | 能以 realized target 而非 open-loop intent 完成工程闭环 | 必须严守 data firewall，并在 fresh confirmatory identities 上重新冻结 | PRIMARY_RECOMMENDED，与 B 联合 |
| D. MEASUREMENT_THRESHOLD_RELAXATION | 移动 mechanism threshold 使现有结果通过 | 表面上提升 pass rate | 明确 outcome-driven、破坏 R1 合同与确认性解释 | NOT_RECOMMENDED_FROM_R1_OUTCOME |

## 推荐路径

优先采用 B/C：先在全新、永久 engineering-only development identities 上建立 controller-aware shaping 与反馈校准，再冻结 generator 数值；之后使用另一批 fresh、outcome-blind identities 进行 R2 confirmatory scientific smoke。不得把本轮 24 个 R1 identities 用于 calibration、model selection 或 confirmatory selection。

## R2 data firewall

- 本轮 24 个 identities：`OUTCOME_EXPOSED=true`、`R1_SCIENTIFIC_HISTORY_ONLY=true`。
- 禁止：R2 generator tuning、threshold tuning、calibration、model selection、confirmatory smoke。
- R2 development calibration：只允许 fresh permanently engineering-only identities。
- R2 scientific benchmark：必须使用另一批 fresh outcome-blind identities。
- Threshold change：`NOT_RECOMMENDED_FROM_R1_OUTCOME`。
- RBR：`NOT_AUTHORIZED`。
