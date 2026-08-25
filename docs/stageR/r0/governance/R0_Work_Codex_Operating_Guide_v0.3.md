# R0 Work + Local Codex Operating Guide v0.3

## 1. 当前阶段

```text
ChatGPT Work / Codex = StageR research governance and integration
Local evidence files = read-only contract source
GitHub branch = 20260825_stageR_new
R0 manifests = machine-readable source of truth
```

当前只允许：本地事实整合、R0 v1.0参数化准备、split/SAP/target schema设计和小型治理文档提交。

当前禁止：RBR-A/B/C训练、新模型训练、新nuPlan仿真、历史checkpoint重建、历史outputs覆盖，以及根据未冻结结果调整future confirmation门槛。

## 2. Commit术语

永久区分：

```text
stageR_base_commit = 460832bde6266f1367a10bfe00e9b3bc176740ce
r0_governance_initial_commit = 0240032511deab32247c233c469b66a45a4888c8
current_stageR_head = 执行时动态读取 git rev-parse HEAD
```

`460832...`不再称为current StageR HEAD。任何freeze manifest必须在最终提交后重新读取并绑定实际HEAD。

## 3. 已完成的信息流

```text
R0 protocol/governance initial freeze
        ↓
Local Codex只读核验
        ↓
outputs/stageR/r0_local_audit/
        ↓
R0_Local_Verification_Integration_Report_v0.1.md
r0_asset_inventory_v0.3.csv
r0_contract_inventory_v0.3.csv
        ↓
R0 v1.0 numerical readiness
```

本地outputs不提交Git；只提交从其提炼的小型治理文档、manifest和SHA事实。

## 4. 当前核心文件

1. `docs/stageR/r0/protocol/R0_Representation_Measurement_Audit_Protocol_v0.5_zh.md`
2. `docs/stageR/r0/governance/R0_Asset_Contract_Inventory_v0.2.md`（历史说明，保留不覆盖）
3. `docs/stageR/r0/governance/R0_Local_Verification_Integration_Report_v0.1.md`
4. `docs/stageR/r0/governance/R0_V1_Freeze_Readiness_Report_v0.1.md`
5. `docs/stageR/r0/manifests/r0_asset_inventory_v0.3.csv`
6. `docs/stageR/r0/manifests/r0_contract_inventory_v0.3.csv`
7. `docs/stageR/r0/manifests/r0_v1_numerical_freeze_checklist_v0.2.csv`（输入清单，保留）
8. `docs/stageR/r0/manifests/r0_v1_numerical_freeze_readiness_v0.1.csv`
9. `docs/stageR/r0/handoff/R0_Local_Codex_Verification_Handoff_v0.2.md`
10. `docs/stageR/r0/manifests/r0_training_authorization_manifest_template_v0.2.json`
11. `docs/stageR/r0/manifests/r0_contract_verification_result_template_v0.2.json`
12. `docs/stageR/r0/manifests/r0_protocol_frozen_template_v0.2.json`

旧v0.1/v0.2文件是历史版本，不覆盖、不删除，也不再作为当前master inventory。

## 5. Source status规则

```text
LOCAL_VERIFIED = 有真实本地文件/代码/配置/命令证据
NOT_FOUND = 已按核验范围搜索但没有找到
AMBIGUOUS = 资产存在但证据冲突或authoritative provenance不足
BLOCKED = 当前无法建立可执行合同
```

历史handover、聊天结论或默认builder参数不能单独升级为`LOCAL_VERIFIED`。

## 6. v1.0参数化顺序

1. 读取`r0_asset_inventory_v0.3.csv`和`r0_contract_inventory_v0.3.csv`；
2. 解决或显式保留contract blockers；
3. 建立`R0_DEVELOPMENT`与`R0_AUDIT_HOLDOUT`候选清单；
4. 锁定`FUTURE_R4_RESERVED_POOL`的数据源/token池/generator rule；
5. 仅使用允许的development资产做variance/power/timing estimation；
6. 填写probe、projection、kernel、bandwidth、bootstrap/permutation和equivalence合同；
7. 生成split manifest、target definition和SAP；
8. 最终读取`git rev-parse HEAD`并生成`r0_protocol_frozen.json`；
9. 只有全部授权门禁满足后才能执行R0 audit；RBR训练仍需独立授权。

## 7. Git规则

- 每次先读`git status`并保护已有dirty outputs；
- 禁止`git reset --hard`和`git clean`；
- 不提交checkpoint、tensor、大型outputs或本机临时文件；
- 只精确`git add`本阶段StageR小型文件；
- push前检查 staged diff、文件大小和branch；
- 若本地/远端仅因等价commit分叉，使用非破坏性merge，不覆盖远端历史。

## 8. 当前未解除门禁

```text
R0_OPERATIONAL_PROTOCOL_NOT_YET_FROZEN
RBR_TRAINING_NOT_AUTHORIZED
```

本地合同核验完成不等于v1.0 freeze完成，也不等于候选训练授权。
