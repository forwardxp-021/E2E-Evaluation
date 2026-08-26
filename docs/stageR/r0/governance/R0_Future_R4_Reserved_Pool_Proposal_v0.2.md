# R0 Future R4 Reserved Source or Generator Proposal v0.2

## Two-stage decision

```text
FUTURE_R4_RESERVED_SOURCE_OR_GENERATOR = FROZEN
freeze_form = FROZEN_PROSPECTIVE_ACQUISITION_RULE
FUTURE_R4_CONFIRMATION_ROSTER = NOT_FROZEN_BY_DESIGN
RBR_TRAINING_NOT_AUTHORIZED_BY_THIS_DOCUMENT_ALONE
```

R0 阶段冻结的是 future source/generator 的选择边界，不要求提前形成最终 token roster。最终 roster 可在 R1 的 mechanism、family-specific matching/equivalence 与 runnability 规则稳定后，从 reserved rule 产生的 source universe 中 outcome-blind 形成。

## Frozen prospective source rule

采用 `R4_PROSPECTIVE_CONTROLLED_SOURCE_RULE_V1`：选择本次冻结之后首个新获取、research-licensed、nuPlan-compatible 且具完整 source/log/token/SHA ledger 的 source batch。若同时存在多个合格 batch，按 `(dataset_release_id, source_manifest_sha256)` 字典序唯一确定。必须与 Waymo train/val/historical-test、Stage6/7/7L、R0 development/audit 以及任何已接触 representation outcome 的 identity 全部 log/token-disjoint。

source 内 token 按 `SHA256(2026082601|source_release|log_name|scenario_token)` 排序；最终分配必须 log-disjoint。只允许 pre-treatment eligibility、context、技术 runnability、family coverage 与预注册 power；禁止用 realized mechanism 或 representation/BDD/probe outcome 排序、删除或补样本。

## Controlled generation boundary

生成设计固定为 paired baseline/treatment、whole-roster/intention-to-evaluate，families 为 `R-HLC/R-TSB/R-IP`。在任何 rollout 前仍必须绑定 exact source/token roster、planner/config/code SHA、dose grid、failure/missingness policy 与 power allocation。弱 mechanism 不得改写为技术失败。

## Final roster boundary

`RESERVED_SOURCE_OR_GENERATOR_FREEZE` 与 `FINAL_CONFIRMATION_ROSTER_FREEZE` 是两个独立事件。R1 形成 final roster 时不得改变本规则；R4 outcome 解盲前还必须冻结 family-specific physical/material margins、TOST/IUT、model/readout/kernel/threshold 和 roster SHA。

机器 freeze：`docs/stageR/r0/manifests/r0_future_r4_reserved_source_or_generator_freeze_v0.1.json`。
