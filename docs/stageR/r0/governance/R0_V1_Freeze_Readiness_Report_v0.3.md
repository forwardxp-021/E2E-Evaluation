# R0 v1 Freeze Readiness Report v0.3

## Final decision

`NOT_READY_FOR_R0_V1_FREEZE`  
`RBR_A/B/C_TRAINING_NOT_AUTHORIZED`

## Gate summary

| Gate | Result | Blocking |
|---|---|---|
| Parameter proposals | 18/18 Scientific Owner approved | no |
| F_match equivalence margins | 0/24 approved; evidence pack only | yes |
| Authoritative nuPlan global ledger | complete as compact per-log ledger with complete token-set SHA binding | no |
| Clean unused pool | exists: 111 identity-clean logs, 19 runnable logs / 12805 runnable tokens | no |
| R0_AUDIT_HOLDOUT | not frozen; current nuPlan provides 19 runnable logs vs conservative minimum 150 log clusters | yes |
| FUTURE_R4_RESERVED_POOL | not frozen; audit allocation did not complete; Route B retained | yes |
| Sample size gates | D0/D1/D2/D3 not jointly satisfiable/frozen | yes |

至少需要新增 131 个 identity-clean、可运行、具有所需 pre-treatment family metadata 的 nuPlan-equivalent independent logs，才能达到当前 D0 保守设计的 150-log floor。之后必须先 outcome-blind 冻结 audit roster，再从剩余且 log/token-disjoint source 冻结 R4 roster。D1 仍需 owner-approved target-level SESOI/prevalence/variance planning；D2 需 frozen stratum occupancy；D3 需至少 406 个有效独立 null trials。任何一项不足都保持 `INCONCLUSIVE/INSUFFICIENT_FOR_FROZEN_GATE`，不得放宽 gate。

本轮没有运行 representation、BDD、仿真、treatment rollout 或训练，没有修改 Generation-1 历史产物。
