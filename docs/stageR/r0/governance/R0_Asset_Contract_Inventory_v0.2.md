# R0 Asset & Contract Inventory v0.2

## 1. Purpose

本文件是 R0 Operational Freeze v1.0 参数化前的资产/合同主台账说明。当前阶段只做**事实核验与冻结准备**，不训练 RBR，不运行新的 confirmation，不重写 Generation-1 历史结论。

## 2. Source-of-truth levels

- `REMOTE_VERIFIED`：当前可从 GitHub branch/commit/file blob 核验的 committed code；
- `REMOTE_DISCOVERED_NOT_HASH_VERIFIED`：已发现路径，但尚未锁定当前 branch blob；
- `HISTORICAL_HANDOVER_ONLY`：旧 handover/历史结论记录，尚未重新核验本地 artifact；
- `LOCAL_VERIFICATION_REQUIRED`：必须在用户本地 repo/outputs/checkpoints 中只读核验；
- `TO_BE_GENERATED`：v1.0 freeze 前必须新生成的 manifest/SAP/holdout/reserved-pool 资产。

任何 `HISTORICAL_HANDOVER_ONLY` 或 `LOCAL_VERIFICATION_REQUIRED` 项不得在 v1.0 中伪装成已验证合同。

## 3. Current remote facts

当前远端 reference 记录为：

```text
repo = forwardxp-021/E2E-Evaluation
active_stageR_branch = 20260825_stageR_new
active_stageR_remote_head = 460832bde6266f1367a10bfe00e9b3bc176740ce
historical_gen1_branch = 20260611_stage7_conclusion
historical_gen1_remote_head = 0f6fefd4363bdfcdeec37f3f7d38782516ba72dd
```

上述 commit SHA 只代表 2026-08-25 核验时的远端 branch HEAD。active StageR branch 的本地 HEAD、dirty status，以及真正产生历史 checkpoint/output 的 commit 仍必须本地只读核验。旧 v0.1 中记录的 `54a89d9...` 实际是 tree SHA，不应继续作为 branch HEAD 使用。

已锁定的远端代码 blob 包括：

- `tools/build_waymo_5neighbor_context_dataset.py` → `cefb105b...`；
- `tools/build_waymo_dynamic_interaction_dataset_v2.py` → `fa81ed46...`；
- `tools/build_nuplan_5neighbor_context_dataset.py` → `4e0b10f1...`；
- `tools/build_standardized_fixed_dimension_bdd_matrix.py` → `538ebfdb...`；
- `tools/build_unified_bdd_posttraining_report.py` → `f45265b3...`。

其中 Waymo builder 的远端默认合同为 `window_len=80, dt=0.1`；Stage7L 实际输入 T、A/B/C 实际训练 T、final-hidden/pooling、normalization 等仍以本地 artifact/config 为最终 source of truth。


## 3A. Recommended repository placement

StageR governance files统一放入：

```text
docs/stageR/r0/
  protocol/
  governance/
  manifests/
  handoff/
```

本地审计运行产物放入：

```text
outputs/stageR/r0_local_audit/
```

治理文档/小型 manifest 可在 review 后提交到 `20260825_stageR_new`；大型 tensor/checkpoint/临时 audit 输出不要为了让 Work 可见而复制进 Git。

## 4. v1.0 blocking local facts

在 v1.0 冻结前至少必须本地核验：

1. local git branch/HEAD/status；
2. old64/A/B/C checkpoints，A/B/C 3407/3408/3409 是否真实存在及 SHA256；
3. Waymo train/val/historical-test tensor 的实际 shape/dtype/mask；
4. Stage7L nuPlan context 的实际 shape 与 inference consumption；
5. encoder forward/pooling/mask/normalization/scaler；
6. ego13 精确13维 schema/formula；
7. MMD kernel/bandwidth/biased-vs-unbiased/null/permutation contract；
8. Stage6P / Stage7L manifest、independence unit 与 historical use；
9. 是否能建立 scenario/log-disjoint `R0_AUDIT_HOLDOUT`；
10. 是否能 outcome-blind 锁定 `FUTURE_R4_RESERVED_POOL`。

## 5. Machine-readable companions

- `r0_asset_inventory_v0.2.csv`
- `r0_contract_inventory_v0.2.csv`
- `r0_v1_numerical_freeze_checklist_v0.2.csv`
- `r0_contract_verification_result_template_v0.2.json`
- `r0_training_authorization_manifest_template_v0.2.json`
- `r0_protocol_frozen_template_v0.2.json`

## 6. Safety / preservation rules

Inventory 阶段应尽量只读。禁止：

```text
git reset --hard
git clean
批量删除 outputs
重新训练缺失 seed 来“补资产”
重新生成/覆盖 frozen historical outputs
```

尤其必须保护历史本地修改资产：

```text
outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv
```

若该文件存在，第一步先记录 SHA256 和 git status，绝不改写。
