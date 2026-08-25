# R0 Work + Local Codex Operating Guide v0.2

## 1. Recommended operating model

R0 v1.0 参数化建议采用：

```text
ChatGPT Work = persistent research control plane
Local Codex/terminal = read-only local evidence verifier
GitHub = committed-code remote reference (`20260825_stageR_new` active StageR branch)
R0 manifests = machine-readable source of truth
```

不建议让任何单一工具独立完成整个 R0 freeze。

## 2. ChatGPT Work responsibilities

Work 负责长期维护和合并：

- R0 Protocol 当前版本；
- `r0_asset_inventory.csv`；
- `r0_contract_inventory.csv`；
- `r0_split_manifest.csv`；
- `r0_target_definition.json`；
- `r0_statistical_analysis_plan.json`；
- numerical freeze checklist；
- decision table；
- protocol deviation log；
- candidate training authorization manifest；
- 最终 `r0_protocol_frozen.json`。

Work 中的任务重点是：**协议一致性、证据层级、冻结顺序、SHA绑定和决策审计**。

## 3. Local Codex responsibilities

本地 Codex 只负责当前聊天无法直接读取的本地事实：

- local git HEAD/status；
- checkpoints 及 SHA256；
- A/B/C seeds 3407/3408/3409；
- train/val/test tensor shape/dtype；
- Stage7L context 的实际 T×83；
- training config/CLI/log；
- encoder forward/pooling/mask；
- scaler/normalization；
- ego13 exact schema；
- MMD/null exact implementation；
- Stage6P/Stage7L manifests；
- 可用于 R0 audit holdout / R4 reserved pool 的 token/log/scenario inventory。

Local Codex 的第一阶段是 **read-only verifier**，不是模型开发 agent。

## 4. Information flow

```text
Work freezes verification questions
        ↓
Local Codex executes read-only contract audit
        ↓
r0_local_contract_verification.json
r0_local_asset_inventory.csv
        ↓
Return to Work
        ↓
Work merges facts and resolves UNKNOWN/AMBIGUOUS
        ↓
Build split + SAP + numerical parameters
        ↓
Generate R0 v1.0 + hashes
        ↓
R0 audits authorized
```

## 5. Important rule

不要在 Local Codex 输出回来之前，由 Work 根据历史对话“补全”未知本地事实。

允许的状态包括：

```text
VERIFIED
NOT_FOUND
AMBIGUOUS
UNKNOWN
```

`UNKNOWN` 比猜测更符合 R0 protocol。

## 6. Recommended Work artifact set

把以下文件作为同一 Work workspace 的核心资产：

1. `R0_Representation_Measurement_Audit_Protocol_v0.4_zh.md`
2. `R0_Asset_Contract_Inventory_v0.1.md`
3. `r0_asset_inventory_v0.1.csv`
4. `r0_contract_inventory_v0.1.csv`
5. `r0_v1_numerical_freeze_checklist_v0.1.csv`
6. `R0_Local_Codex_Verification_Handoff_v0.1.md`
7. `r0_training_authorization_manifest_template.json`
8. `r0_contract_verification_result_template.json`
9. `r0_protocol_frozen_template.json`

之后 Local Codex 回传本地核验文件，再更新这些 master artifacts，而不是重新从聊天记录重建状态。


## 7. StageR local-folder layout and Work visibility

推荐在 `E2E-Evaluation` repo 内使用：

```text
docs/stageR/r0/protocol/
docs/stageR/r0/governance/
docs/stageR/r0/manifests/
docs/stageR/r0/handoff/
outputs/stageR/r0_local_audit/
```

由于 macOS Desktop Work 已打开/获准访问 `E2E-Evaluation` 本地文件夹，放在该 folder 树下的文件可由 local Work 读取。不要依赖“自动猜文件”：在 Work 中明确指定相对路径，例如 `docs/stageR/r0/protocol/R0_Representation_Measurement_Audit_Protocol_v0.5_zh.md`。

网页/移动端 Work 不能直接访问 Mac 本地文件；若希望跨设备使用，上传 frozen snapshot 或已提交 Git 的版本，但本地 repo 中的 master 文件仍作为 StageR source of truth。

当前 branch：`20260825_stageR_new`；2026-08-25 远端 HEAD：`460832bde6266f1367a10bfe00e9b3bc176740ce`。
