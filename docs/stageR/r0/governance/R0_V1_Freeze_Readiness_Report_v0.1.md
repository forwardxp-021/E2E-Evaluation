# R0 v1.0 Freeze Readiness Report v0.1

## 1. 决策

```text
NOT_READY_FOR_V1_FREEZE
RBR_TRAINING_NOT_AUTHORIZED
```

原因不是本地核验未执行，而是核验已经明确暴露出尚未冻结的执行policy、provenance缺口和future data blockers。不得为了推进项目强行标记READY。

## 2. 已解决的blocking items

以下“未知本地事实”已解决：

1. A/B/C 3407/3408/3409：9/9 formal best均存在并与locked SHA匹配；
2. Waymo输入：训练/验证/历史test每行均为`[80,83]`；
3. Stage7L输入：每个dose为`[80,150,83]`，learned encoder实际消费完整150步；
4. pooling：A/B为单GRU final hidden，C为双branch各自final hidden后concat；
5. mask：Stage7L learned encoder不消费mask/length；
6. input normalization：learned input无scaler/normalization；
7. raw33、clean-longitudinal、ego13 scaler合同已定位；
8. ego13有序13维schema和实现SHA已定位；
9. Stage7L paired MMD/kernel/bandwidth/null/permutation合同已定位；
10. Stage6P unpaired A/A calibration、log-disjoint release和scenario-overlap限制已定位。

这些事实可以直接进入R0 v1.0 SAP准备，不需要再次从历史handover推断。

## 3. 剩余blocking items

### B1 — Temporal policy未冻结

事实已清楚：训练`T=80`，Stage7L推理`T=150`。但R0尚未冻结：

- D0 length/position/pooling views；
- 80/150对齐policy；
- 何种比较可称controlled，何种只能descriptive；
- temporal minimum effect。

状态：`NEEDS_DEVELOPMENT_ESTIMATION`，阻塞v1 freeze。

### B2 — Mask/padding policy未冻结

历史Stage7L learned encoder忽略mask，并让right-padding的最终零步参与final-hidden pooling。事实明确，但R0 audit如何处理mask/padding尚未冻结。

状态：`BLOCKED`，阻塞v1 freeze。

### B3 — raw33 authoritative SHA provenance缺口

36个实际`interaction_feat_style_raw.npy`文件存在并已计算当前SHA，但历史shard SHA ledger没有这些条目。

允许方案：建立非破坏性的provenance addendum，记录当前文件SHA、manifest、生成代码和缺口；禁止重写历史ledger并假装当时已登记。

状态：`AMBIGUOUS`，阻塞authoritative data freeze。

### B4 — R0_AUDIT_HOLDOUT未建立

现有Waymo historical test、Stage6P、Stage7和Stage7L均已使用或解盲。Local verification没有找到可以直接标成outcome-blind audit holdout的资产。

状态：`NOT_FOUND / BLOCKED`。

### B5 — FUTURE_R4_RESERVED_POOL未建立

没有发现已冻结的数据源、scenario/token pool或generator rule。

状态：`NOT_FOUND / BLOCKED`。正式RBR训练前必须解决。

### B6 — 关键numerical/SAP参数未形成

仍需development estimation：

- alpha与multiple-testing families；
- bootstrap/permutation repetitions；
- temporal effect、position bins；
- probe capacity与target thresholds；
- D2 matching strata和OOD boundary；
- primary kernel/bandwidth；
- projection rank/selection metric；
- F_match及equivalence margins；
- residual family minimum independent units。

## 4. 已验证contracts

| Contract | 状态 |
|---|---|
| branch/base/governance commit术语 | VERIFIED |
| A/B/C seed与checkpoint SHA | VERIFIED |
| Waymo train/val/test shape与scenario split overlap | VERIFIED |
| Stage7L tensor及actual 150-step consumption | VERIFIED |
| A/B/C architecture与pooling | VERIFIED |
| Stage7L mask/padding历史行为 | VERIFIED |
| learned input无normalization | VERIFIED |
| target scaler合同 | VERIFIED |
| ego13 schema/scaler | VERIFIED |
| Stage7L paired BDD/null | VERIFIED |
| Stage6P unpaired release/null | VERIFIED |

## 5. 未验证或未冻结contracts

| Contract | 状态 | 影响 |
|---|---|---|
| nuPlan slot identity-switch实际调用链细节 | AMBIGUOUS | D2解释边界；当前非最大阻塞 |
| D2 synchronized ablation value/mask contract | BLOCKED | C正式审计 |
| R0 kernel/bandwidth Primary选择 | NEEDS_DEVELOPMENT_ESTIMATION | D3 |
| projection rank/selection | NEEDS_DEVELOPMENT_ESTIMATION | D3 |
| R0 calibration/audit data isolation | BLOCKED | D3 |
| D4 equivalence margins | BLOCKED | residual benchmark |
| raw33 historicalauthoritative SHA completeness | AMBIGUOUS | dataset freeze |

## 6. Numerical parameter readiness

详细机器表：`docs/stageR/r0/manifests/r0_v1_numerical_freeze_readiness_v0.1.csv`。

总体：

- `READY`：branch动态绑定方法、三seed规则、历史independence单位、80/150事实、TOST/CI方法、whole-roster estimand；
- `NEEDS_DEVELOPMENT_ESTIMATION`：多数效应阈值、probe、kernel、bandwidth、rank、matching参数；
- `BLOCKED`：D2 ablation contract、R0 null/audit split、equivalence margin、minimum independent units、holdout、reserved pool；
- `NOT_YET_APPLICABLE`：geometry standalone gate、R4 `δ_NI`最终数值。

`δ_NI`在R0只冻结数学定义、比较量、估计/CI方法和选择原则，当前不冻结数值。

## 7. R0_AUDIT_HOLDOUT status

```text
NOT_FOUND
BLOCKED_FOR_V1_FREEZE
```

不能从现有已解盲资产直接重命名得到。下一步应先形成仅基于identity/history的candidate inventory，再做scenario/log/driver overlap和功效检查，不读取新representation outcome。

## 8. FUTURE_R4_RESERVED_POOL status

```text
NOT_FOUND
BLOCKED_BEFORE_FORMAL_RBR_TRAINING
```

必须锁定数据源/token pool或受控生成规则，并禁止R0/R1/RBR outcome影响pool保留。

## 9. 建议的下一步

1. 建立raw33 provenance addendum；
2. 冻结D0与mask/padding audit policy；
3. 生成R0 target definition草案；
4. 使用R0_DEVELOPMENT做有限的variance/power/runtime estimation；
5. 建立R0 holdout candidate inventory；
6. 建立R4 reserved source/token/generator proposal；
7. 填写SAP和数值门槛；
8. 再次执行v1 freeze readiness检查。

本报告不授权执行上述统计audit，更不授权任何RBR训练。
