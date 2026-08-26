# R0 Mask / Padding Audit Policy v0.1

## 1. 状态

```text
POLICY_DRAFT_READY_FOR_FREEZE_REVIEW
FUTURE_RBR_MASK_POLICY_NOT_FROZEN
```

本 policy 严格区分 historical reproduction、diagnostic views 与 future RBR policy candidate。任何 diagnostic 改动都不能回写或重新命名 Generation-1 历史结果。

## 2. 本地已核实基线

Stage7L 五个 dose 的 `ego_seq_mask` 相同：

| 项目 | 每个 dose |
|---|---:|
| rows | 80 |
| T | 150 |
| valid length=149 | 53 |
| valid length=150 | 27 |
| padded/invalid ego frames | 53 / 12000 |
| padding ratio | 0.4416667% |
| false→true transition | 0 |
| final valid rows | 27 |
| final invalid rows | 53 |

所有无效 ego frame 都在右侧末帧。53 个无效末帧的 `ego_seq` 8D 全零，但对应 83D `context_traj` 不保证全零：部分 neighbor slot 可以仍有 presence/relative-state 值。历史 learned encoder 消费完整 150 步，所以“ego mask 为 false”不等于“GRU 的最后一步输入为全零”。

完整 context 中存在 `999` sentinel（各 dose 约 71–76 个元素），用于缺失/不可用邻居距离类字段；本次只读检查没有发现 padded-final context 中的 `999`，但 sentinel 在其他有效 context 步仍直接进入无 normalization 的 encoder。

## 3. A — Historical behavior

Historical reproduction 必须：

1. 使用原始 `[B,150,83]`；
2. A/B/C 不传 mask/length；
3. final hidden 在全部 150 步之后取得；
4. 保留 999 sentinel、slot presence、neighbor values 和原始末帧；
5. ego13 继续单独按 `ego_seq_mask` 聚合；
6. 输出标记 `HISTORICAL_STAGE7L_REPRODUCTION`。

不允许在 historical reproduction 中截到 final valid timestep、清空 invalid context、packed-sequence 或 masked pooling。

## 4. B — Diagnostic views

所有 view 都以同一 scenario/dose/checkpoint 做 paired comparison，并标记 `DIAGNOSTIC_NOT_HISTORICAL`。

| View | 唯一受控变化 | 目的 |
|---|---|---|
| H0 | 无，历史完整输入 | reference |
| D1-final-valid | 对 149 帧行取 timestep 148 的 hidden；150 帧行取 149 | final valid vs historical final timestep |
| D2-context-zero-invalid | 只把 ego-mask=false 的完整 83D context 设 physical zero | 分离无效末帧 neighbor 残留敏感性 |
| D3-truncate-valid | 按 valid length 截断 forward | 分离额外 recurrence step 敏感性 |
| D4-same-hidden-masked-mean | 同一 hidden sequence 只纳入 valid timesteps | mask-aware pooling sensitivity |
| D5-sentinel-control | 按预注册规则保留/替换 999，并同步 slot-valid semantics | sentinel sensitivity/OOD diagnostic |

D2/D5 必须同时记录 physical zero、presence channel、missingness channel、slot mask 与 derived-feature policy；不能只改数值而静默保留矛盾的 valid semantics。

## 5. 必须输出的检查

### 输入与位置

- per-row valid length、padding ratio、padding start/end；
- internal gap 与 false→true transition；
- final valid timestep 与 historical final timestep；
- final-invalid context 的非零 channel 清单；
- sentinel 999 总数、frame位置、slot/channel位置；
- `ego_seq_mask` 与 slot presence/valid mask 的一致性。

### Hidden 与 embedding

- `||h_final_historical||2` 与 `||h_final_valid||2`；
- hidden norm 的 paired difference/ratio；
- historical embedding 与每个 diagnostic embedding 的 L2/cosine distance；
- 按 final-valid/final-invalid、dose、representation、seed 分层；
- log-cluster bootstrap 95% CI；
- training embedding reference 的 centroid/PCA/nearest-neighbor OOD 指标。

### Probe

- frozen-probe-across-view；
- same-capacity-refit-probe-per-view；
- target-level effect 与 seed consistency；
- 当前证据级别只能为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`。

## 6. C — Future RBR policy candidates

以下仅为候选，不在本阶段冻结：

1. explicit valid-length + packed recurrent sequence；
2. mask-aware fixed pooling；
3. fixed T80 with predeclared content alignment；
4. fixed T150 with explicit ego/slot masks and trained missingness semantics；
5. native branch-specific masks for RBR-C。

future policy 必须在训练前定义其 training/inference 同构合同，并在 R0/R1 development 上比较；不得根据 future R4 outcome 选择。

## 7. 命名与结论限制

- 修改 mask/length/pooling 后的结果：`DIAGNOSTIC_*`；
- 原始 A/B/C：仅 `HISTORICAL_STAGE7L_*`；
- 不得把 diagnostic B/C 写成 historical B/C；
- 不得把 ablation sensitivity 写成 causal dependence；
- 不得修改 Stage7L 历史 embeddings、BDD、null 或报告；
- 本 policy 不授权 RBR-A/B/C 训练。
