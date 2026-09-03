# R2-A → R2-B Generator Architecture Decision v1

## 比较

| 方案 | DEV 证据下的优点 | 主要风险 | R2-B disposition |
|---|---|---|---|
| A. STATIC_MARGIN_SCALING | 实现简单，可利用平均 gain | HLC/TSB 的 gain、lag、phase carryover 均依 identity 与 duration 改变；静态倍数不能处理边界迁移 | 不作为主方案 |
| B. CONTROLLER_AWARE_PRECOMPENSATION | 可显式对 1 s LQR lookahead、trajectory fitting、motion-model lag 和 settling 做前馈补偿 | 需用 DEV-only surrogate 给出保守 architecture，不能从 R1 official outcome 调数值 | **推荐主架构** |
| C. FEEDBACK_CALIBRATED_OFFLINE_GENERATOR | 可在永久 engineering-only canary 上验证 realized morphology，覆盖 surrogate 未建模误差 | 必须严格维持 data firewall，且校准 identity 不得进入 confirmatory | **推荐作为 B 的离线验证闭环** |

## 决策

R2-B 推荐 `B + C`：以 controller-aware precompensation 为 generator architecture，用另一批永久 engineering-only development canary 做 outcome-separated offline feedback calibration。A 仅可作为 B 中的初始值，不应单独冻结。

本阶段不冻结任何最终 amplitude、duration、lag compensation 或 scientific threshold。R2 confirmatory roster 尚未建立；RBR A/B/C 仍未授权。
