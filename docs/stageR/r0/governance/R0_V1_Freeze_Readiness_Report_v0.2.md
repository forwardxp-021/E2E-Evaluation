# R0 v1.0 Freeze Readiness Report v0.2

## 1. 最终决策

```text
NOT_READY_FOR_R0_V1_FREEZE
R0_OPERATIONAL_PROTOCOL_NOT_YET_FROZEN
RBR_TRAINING_NOT_AUTHORIZED
```

本阶段已经完成 provenance、policy、target、parameter、holdout/reserved-pool proposal 与 SAP draft，但没有为了推进项目强行把不存在的资产或缺乏科学依据的 margin 标为 READY。

机器 readiness 表：`docs/stageR/r0/manifests/r0_v1_numerical_freeze_readiness_v0.2.csv`。

## 2. Raw33 provenance

```text
CURRENT_FILE_PROVENANCE_VERIFIED
HISTORICAL_LEDGER_ENTRY_NOT_AVAILABLE
```

36 个当前 `interaction_feat_style_raw.npy` 已逐文件绑定 path、SHA、part/shard、shape、row count、manifest、代码 SHA 与发现时间，总计 168700 rows。当前文件 provenance 问题已通过非破坏性 addendum 解决。

历史 ledger 未包含这些条目的事实继续保留；未修改历史 ledger，也未把当前 SHA 冒充为历史冻结 SHA。因此该历史缺项转为显式 limitation，不再伪装成未知，但不能声称 Generation-1 ledger 当时完整。

## 3. D0 temporal policy

状态：`READY_FOR_FREEZE_REVIEW`，尚未 operationally frozen。

已完成：

- D0-A length/temporal-contract；
- D0-B matched natural position retention（准实验）；
- D0-C same hidden sequence last/mean/max（严格 pooling）；
- D0-D mask/padding sensitivity；
- historical T150 + final hidden + historical padding 单独保留；
- first80/last80/event80 限定为 descriptive；
- frozen probe 与 same-capacity refit probe 分开报告。

仍需批准：standardized minimum temporal effect `0.10` 的科学 materiality。D0 没有选择 future RBR 的 80/150 winner。

## 4. Mask/padding policy

状态：`READY_FOR_FREEZE_REVIEW`，future RBR mask policy 未冻结。

已核实每 dose：53/80 行有效长度 149，27/80 行 150；padding ratio 0.4417%，全为右侧。无效末帧 ego 8D 全零，但 83D context 可保留 neighbor values，因此不能把完整末帧称为全零。learned A/B/C 不消费 mask，ego13 消费 `ego_seq_mask`。

Policy 已区分 historical reproduction、diagnostic views 和 future candidates，并要求 hidden norm、embedding distance、final-valid/final-invalid、sentinel 999 与 valid-mask 一致性检查。实际 diagnostic execution 要在 v1 freeze 后按 frozen SAP 运行。

## 5. Target definition

状态：`READY_FOR_FREEZE_REVIEW`。

`r0_target_definition_v0.1.json` 定义 49 个 target：

- ego13 exact 13D；
- raw33 中 longitudinal、lateral、interaction/context targets；
- longitudinal-v2 3D；
- 每项均含 definition、source path、unit、valid-frame rule、independence unit 和四类使用标志；
- F_match 24 项；
- M_behavior 11 项；
- 两组交集为空。

## 6. Numerical parameter readiness

18 个主 parameter proposal：

- 16 项 `READY_FOR_FREEZE` proposal：alpha、confidence、Holm family、bootstrap 5000、permutation 49999、D0 position bins、linear probe/grid、projection ranks/selection、single RBF、label-blind fixed bandwidth、D2 matching/coarsening、D2 OOD boundary、target-level reporting；
- 2 项 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`：D0 minimum temporal effect 0.10；D3 FPR upper-CI 0.075 gate；
- 24 个 F_match equivalence margin 全部 `REQUIRES_SCIENTIFIC_OWNER_APPROVAL`，数值未伪造。

这里的 `READY_FOR_FREEZE` 是 proposal readiness，不是已经写入 v1 frozen manifest。

## 7. D4 equivalence margin

```text
EQUIVALENCE_METHOD = READY_FOR_FREEZE_REVIEW
EQUIVALENCE_MARGIN_24_OF_24 = REQUIRES_SCIENTIFIC_OWNER_APPROVAL
```

每项已记录 physical scale、unit、历史 natural variability 与行为/业务相关性边界。由于缺少 repeated-measurement reproducibility 与 owner-defined material tolerance，margin 保持空值。Power 没有被用来定义 equivalence。

## 8. R0_AUDIT_HOLDOUT

```text
R0_AUDIT_HOLDOUT = NOT_AVAILABLE
R0_AUDIT_HOLDOUT_UNAVAILABLE_FROM_EXISTING_ASSETS
```

Waymo train/val/historical-test、Stage6P、Stage7/M6、Stage7L 均已使用、解盲或参与 representation evaluation。现有 nuPlan remainder 没有 authoritative unused-token ledger，无法排除对全部历史 roster 的 overlap。

最小解决路径是获取新的 source release/prospective data，在任何 representation 计算前建立 identity/SHA ledger，按 hash-sorted、log-disjoint、outcome-blind 规则锁定 audit roster。样本数必须在 margin/effect 冻结后由 cluster-aware power 决定。

## 9. FUTURE_R4_RESERVED_POOL

```text
FUTURE_R4_RESERVED_POOL = NOT_AVAILABLE
```

Route A 的 existing unused source/token pool 缺 authoritative unused identity ledger。Route B 的 prospective controlled-planner rules 已形成，但 exact source/token roster/config SHA 尚未锁定。规则草案本身不能冒充可用 reserved asset。

正式 RBR 训练前必须先绑定 source/token roster；rollout 后只能 whole-roster mechanism gate 和 whole-roster Primary confirmation，不得筛掉 mechanism 弱的 scenario 后再确认。

## 10. SAP readiness

人类 SAP 和 machine SAP v0.1 均已生成，包含 24 个 hypothesis records、analysis family、alpha/multiplicity、independence/split/bootstrap/permutation unit、probe/kernel/bandwidth/rank/equivalence/status/evidence level。

状态仍为 `DRAFT_NOT_FROZEN`，因为 holdout、reserved pool、equivalence margin及两项 owner gate 未解决。

## 11. Readiness matrix

| Gate | 状态 | 是否阻塞 v1 freeze |
|---|---|---|
| raw33 current provenance | READY | 否 |
| raw33 historical ledger entry | NOT_AVAILABLE，显式 limitation | 否，不得改写历史 |
| D0 policy | READY_FOR_FREEZE_REVIEW | 是，待 owner freeze |
| mask/padding policy | READY_FOR_FREEZE_REVIEW | 是，待 owner freeze |
| target definition | READY_FOR_FREEZE_REVIEW | 是，待 owner freeze |
| core numerical proposals | 16 READY_FOR_FREEZE / 2 OWNER APPROVAL | 是 |
| D4 margins | 24/24 OWNER APPROVAL | 是 |
| R0_AUDIT_HOLDOUT | NOT_AVAILABLE | 是 |
| FUTURE_R4_RESERVED_POOL | NOT_AVAILABLE | 是；且为训练前硬门禁 |
| SAP | DRAFT_READY_FOR_REVIEW | 是 |
| RBR-A/B/C training | NOT_AUTHORIZED | 持续有效 |

## 12. 下一步

1. scientific owner 审批或替换 D0 effect、D3 FPR gate 与 24 个 equivalence margin；
2. 获取并 outcome-blind 锁定新的 R0 audit source/identity roster；
3. 绑定 future R4 exact source/token roster 或 prospective generator/config SHA；
4. 将批准值写入 v1 protocol/SAP/split manifests；
5. 重新运行 SHA binding 与 freeze readiness；
6. 只有得到 `READY_FOR_R0_V1_FREEZE` 后才执行 frozen R0 audit；
7. 当前仍不得启动 RBR-A/B/C 正式训练。
