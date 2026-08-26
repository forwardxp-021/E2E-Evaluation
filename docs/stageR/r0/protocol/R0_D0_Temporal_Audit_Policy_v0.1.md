# R0 D0 Temporal Audit Policy v0.1

## 1. 状态与目的

```text
POLICY_DRAFT_READY_FOR_FREEZE_REVIEW
NOT_OPERATIONALLY_FROZEN
RBR_TEMPORAL_LENGTH_NOT_SELECTED
```

D0 首先诊断 Generation-1 的时间合同，不在本阶段决定未来 RBR 必须使用 80 还是 150 帧。

已核实历史事实：Waymo A/B/C 训练输入为 `T80×D83`；Stage7L inference 实际输入为 `T150×D83`；A/B/C 都用 final hidden；learned encoder 不接收 mask/valid length；Stage7L 存在 149/150 有效帧；ego13 单独使用 `ego_seq_mask`。

## 2. 永久保留的 historical reference

Generation-1 Stage7L 的历史行为固定为：

```text
HISTORICAL_STAGE7L = T150 + final hidden + historical mask/padding/sentinel behavior
```

所有复现必须送入完整 `[B,150,83]`，不新增 mask/length，不裁切，不改变 context 中的 sentinel 或无效末帧邻居值。它只用于复现历史结果，不能被“更合理”的实现替换。

## 3. D0-A — Length / temporal-contract study

问题：在 event 内容、event anchor、pooling、preprocessing 和 mask policy 尽量不变时，`T80` 与 `T150` 是否产生信息保留差异。

Primary 构造要求：

1. 同一 episode、同一 event family、同一 event phase 与同一主要运动段；
2. 80/150 中 event 内容均完整保留；
3. pooling 固定为同一规则；
4. 新增的前/后 context、pad value、mask 与有效长度显式记录；
5. 每一对以同一 scenario 为 paired unit，CI 至少做 log-cluster sensitivity；
6. 只有不改变主要 event 内容的比较可标记 `CONTROLLED_LENGTH_STUDY`。

若 150 帧必须通过新增自然内容或不等价 padding 构造，则降级为：

```text
CONTENT_CONFOUNDED_LENGTH_DIAGNOSTIC
```

不得据此单独声称 pure length effect。

## 4. D0-B — MATCHED_NATURAL_POSITION_RETENTION_STUDY

D0-B 是 matched quasi-experimental study，不是严格因果实验。固定总长度后，按预处理/context 变量匹配：

- scenario family；
- lane-change direction；
- event duration/magnitude；
- initial speed；
- traffic density；
- neighbor availability；
- route/road geometry proxy；
- log cluster。

位置 proposal 在 150 帧、`dt=0.1 s` 下固定为：

| Bin | Frame index | 时间支持 |
|---|---|---|
| early | 0–49 | 前 5 s |
| middle | 50–99 | 中 5 s |
| late | 100–149 | 后 5 s |

主解释只允许写成 `matched natural position-retention association`。不能使用“位置导致遗忘”等因果措辞。若 future controlled time-shift 另行建立，必须单独报告输入 OOD。

## 5. D0-C — SAME_HIDDEN_SEQUENCE_POOLING_STUDY

D0-C 是唯一允许称为严格 pooling effect 的主实验。对同一输入只执行一次 encoder forward，保存同一 hidden sequence `H[B,T,H]`；随后只改变 pooling：

```text
last
mean
max
```

合同：

- A/B：同一个 single-GRU hidden sequence；
- C：ego/context 两个 branch 各自使用同一 hidden sequence、同一 pooling rule，再按历史 16+48 顺序拼接；
- encoder weights、输入、hidden sequence、target、probe capacity 与 split 不变；
- `masked_mean` 与 `recent_k_mean` 仅作 secondary，并归入 D0-D/secondary pooling，因为它们改变了 timestep inclusion；
- learned attention、outcome-tuned weights 和重新训练 encoder 均不属于 D0-C。

只有 `last/mean/max` 在同一 hidden sequence 上的差异可解释为 strict pooling effect。

## 6. D0-D — Mask/Padding sensitivity diagnostic

D0-D 只诊断 historical final-hidden 对有效长度、无效末帧和 context sentinel 的敏感性。允许的 view 必须由 `R0_Mask_Padding_Audit_Policy_v0.1.md` 定义，并标记：

```text
DIAGNOSTIC_NOT_HISTORICAL
```

Primary paired outputs：embedding norm change、cosine/L2 distance、final-valid vs final-invalid strata差异、frozen/refit probe变化。任何修改 mask 后的 B/C 结果不得称为 historical B/C。

## 7. Content-window diagnostics

以下 view 只作为 descriptive content-window diagnostic：

```text
first80
last80
event80
overlap80
full_native
```

它们同时改变内容、event presence、event position 或 support，不能单独证明 final-state forgetting，也不能用来选择 future RBR 的 80/150 winner。

## 8. 两类 probe 必须并列报告

### Frozen-probe-across-view

在 reference view 拟合一次 probe，跨 view 冻结使用。它测量坐标/geometry compatibility。

### Same-capacity-refit-probe-per-view

每个 view 独立拟合，但 target、split、linear family、ridge grid、预算、seed、选择规则完全相同。它测量信息是否仍然可读。

解释矩阵：

| Frozen probe | Refit probe | 允许解释 |
|---|---|---|
| 降 | 稳定 | coordinate/geometry shift 更可能 |
| 降 | 降 | information-retention loss 更可能 |
| 稳定 | 稳定 | 不支持实质 temporal loss |
| 稳定 | 降 | 检查 refit/split/calibration 异常，不作机制结论 |

## 9. 统计与证据边界

- A/B/C 核心指标全部报告 seeds 3407/3408/3409；
- direction consistency 至少 2/3 seeds；
- paired independent unit 为 scenario，bootstrap 默认 log cluster；
- temporal effect proposal 为绝对 paired standardized retention difference `>=0.10`、95% CI 排除 0；该 materiality 数值仍需 scientific owner 批准；
- 当前没有 R0_AUDIT_HOLDOUT，所有结果最多为 `DEVELOPMENT_DIAGNOSTIC_EVIDENCE`；
- first80/last80/event80 不参与 strict D0-C pooling hypothesis；
- D0 不授权训练或选择 RBR architecture。
