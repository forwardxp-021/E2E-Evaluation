# R0 Scientific Owner Approval Record v0.1

## Binding decision

`18/18 PARAMETER PROPOSALS = SCIENTIFIC_OWNER_APPROVED`。其中先前 16 项 `READY_FOR_FREEZE` proposal 全部批准；D0 与 D3 的两项待审批数值现正式绑定。

- D0：`|paired standardized retention difference| >= 0.10`，且 95% CI 排除 0，且至少 2/3 seeds 方向一致。0.10 仅是 representation temporal-retention diagnostic SESOI，不是驾驶物理或人类可感知阈值。
- D3：nominal FPR=0.05，two-sided 95% CI upper bound 必须 `<=0.075`。独立 null units 不足时结果为 `INCONCLUSIVE`，不得放宽门槛。
- 24 个 F_match equivalence margins：`0/24 APPROVED`，继续为 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`。禁止由 population SD 或 power 机械产生 margin。
- 本记录不授权 RBR-A/B/C 正式训练。

机器绑定：`docs/stageR/r0/manifests/r0_scientific_owner_approval_v0.1.json`。
