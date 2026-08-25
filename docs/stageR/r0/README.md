# StageR / R0 local workspace

Active development branch: `20260825_stageR_new`

```text
stageR_base_commit = 460832bde6266f1367a10bfe00e9b3bc176740ce
r0_governance_initial_commit = 0240032511deab32247c233c469b66a45a4888c8
current_stageR_head = 执行时动态读取 git rev-parse HEAD
```

不要把`460832...`或任何报告生成前的commit继续写成current HEAD。

## Current state

```text
RBR_DIRECTION_FROZEN
R0_SCIENTIFIC_SCOPE_FROZEN
LOCAL_CONTRACT_VERIFICATION_EXECUTED
LOCAL_RESULTS_INTEGRATED
R0_V1_PARAMETERIZATION_IN_PROGRESS
R0_OPERATIONAL_PROTOCOL_NOT_YET_FROZEN
RBR_TRAINING_NOT_AUTHORIZED
```

## Folder roles

- `protocol/`: human-readable R0 protocol versions.
- `governance/`: asset/contract inventory and Work/Codex operating rules.
- `manifests/`: small machine-readable CSV/JSON freeze inputs and templates.
- `handoff/`: bounded instructions passed to local Codex.
- `../../../outputs/stageR/r0_local_audit/`: local verification source outputs; keep local and do not commit.

Current master integration artifacts:

- `governance/R0_Local_Verification_Integration_Report_v0.1.md`
- `governance/R0_V1_Freeze_Readiness_Report_v0.1.md`
- `manifests/r0_asset_inventory_v0.3.csv`
- `manifests/r0_contract_inventory_v0.3.csv`
- `manifests/r0_v1_numerical_freeze_readiness_v0.1.csv`
- `governance/R0_Work_Codex_Operating_Guide_v0.3.md`

v0.2 inventories and guide remain historical versions and are not overwritten.
