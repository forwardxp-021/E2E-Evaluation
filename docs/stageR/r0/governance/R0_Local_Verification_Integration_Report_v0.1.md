# R0 Local Verification Integration Report v0.1

## 1. 范围与结论

本报告只整合本地只读核验事实，不解释模型科学表现，不修改历史输出，不授权训练或仿真。

本地 handoff 状态：`COMPLETE_WITH_EXPLICIT_WARNINGS`。三个要求的机器文件均存在且可解析：

- `outputs/stageR/r0_local_audit/r0_local_contract_verification.json`
- `outputs/stageR/r0_local_audit/r0_local_asset_inventory.csv`
- `outputs/stageR/r0_local_audit/r0_local_command_ledger.jsonl`

当前整合结论：本地合同核验已经执行并完成事实整合，但仍有明确 freeze blockers，因此不能据此声明 R0 v1.0 已冻结或授权 RBR 训练。

## 2. Repository state

| 事实 | 状态 | 值 | 证据/说明 |
|---|---|---|---|
| active branch | VERIFIED | `20260825_stageR_new` | `git branch --show-current` |
| `stageR_base_commit` | VERIFIED | `460832bde6266f1367a10bfe00e9b3bc176740ce` | StageR 基线；local audit 在此commit执行 |
| `r0_governance_initial_commit` | VERIFIED | `0240032511deab32247c233c469b66a45a4888c8` | 刷新远端后 `origin/20260825_stageR_new` 的治理初始commit |
| equivalent local governance commit | VERIFIED | `46440b12eb5c323f02e4dec18cb2521bb00b72a0` | 与`0240032...`父提交相同且tree SHA均为`c4189834...` |
| integration merge base HEAD | VERIFIED | `efa422c20011920d3a8f857ce42e2b0ca6775071` | 非破坏性连接两条等价治理commit后的pre-integration HEAD |
| `current_stageR_head` | VERIFIED | 动态执行 `git rev-parse HEAD` | 最终提交后不得沿用本表中的pre-integration SHA |
| worktree | VERIFIED | dirty，包含大量既有untracked outputs及一个既有modified CSV | 未reset、clean、覆盖或移动 |
| protected modified output | VERIFIED | `behavior_event_metrics_v2.csv` SHA `e8deb933...` | 审计前已修改；本阶段保持只读 |

本地audit是在治理文件提交前的`460832b...`代码树上执行；从该commit到本报告整合前，除`docs/stageR/r0/`治理文件外没有代码差异。

## 3. Handoff执行完整性

| 检查 | 状态 | 证据 |
|---|---|---|
| contract JSON存在且可解析 | VERIFIED | SHA256 `3f699215505a467f75781dcf32a29da741b2114e9a2f3c41eefa61b1281dfb7b` |
| local asset inventory存在且可解析 | VERIFIED | 313 data rows；SHA256 `580ce4521304738f0b29efd6415bf8a783b95c34d3066874a2289659f927d4d0` |
| command ledger存在且逐行JSON可解析 | VERIFIED | 当前文件SHA256 `8303cddbcc1941f0c48180d59de4cf77744f1f4cd50579aab250c24d11e0eebe` |
| A/B/C formal best检查 | VERIFIED | 9/9，missing pair为空 |
| Waymo/Stage7L/ego13/BDD合同检查 | VERIFIED | contract JSON断言检查通过 |
| R0 holdout锁定 | NOT_FOUND | 本地audit未发现outcome-blind候选 |
| R4 reserved pool锁定 | NOT_FOUND | 本地audit未发现已冻结source/token/generator rule |

因此 handoff 执行为：核心核验任务 `COMPLETE`；future split/reserved-pool任务为显式`BLOCKED/NOT_FOUND`，不是遗漏执行。

## 4. A/B/C checkpoint事实

全部formal best checkpoint均安全读取metadata并与历史locked SHA一致。

| Candidate | Seed | SHA256 | 状态 | 路径 |
|---|---:|---|---|---|
| A | 3407 | `353982753f208d27d677c6863a681997b8e28b728573a52fa407807f6fd0298d` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_A_dynamic_data_legacy/seed_3407/best_model.pt` |
| A | 3408 | `8d9886490b9308623abe938b48fc926106dcf1c109800b78952175970a31077c` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_A_dynamic_data_legacy/seed_3408/best_model.pt` |
| A | 3409 | `5e22156f0f0197aca9a3fef1fe0e0db1573efd589f93527edfa25eec9f1c92bd` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_A_dynamic_data_legacy/seed_3409/best_model.pt` |
| B | 3407 | `d8e0de6e74ee29076082aabef27a425b47678e1372c630e4f4a04106ff34265f` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_B_single_gru_recovery/seed_3407/best_model.pt` |
| B | 3408 | `3b8ca8949da185bc25715997d49a64aa4131641409d10f44565a94a9c86f4f35` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_B_single_gru_recovery/seed_3408/best_model.pt` |
| B | 3409 | `c2d54a51bfc13d597b0265c59bfe5377035168d133ab97498cbd2d4a1fa53ac5` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_B_single_gru_recovery/seed_3409/best_model.pt` |
| C | 3407 | `cc6bf3c427534f66f74904c8948bf427cfe9f1152bba4bca0e8342f3fa47433d` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_C_dual_branch/seed_3407/best_model.pt` |
| C | 3408 | `603d56f34b62fb22e6c59d6558fb42b8dbc67ca0897d09660e87dc8e1f09f521` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_C_dual_branch/seed_3408/best_model.pt` |
| C | 3409 | `1b0a10779d5c90559e71af42cbd2d8c7b6611f6ecb2efca6059b426ce51a974e` | VERIFIED | `outputs/stage6t_candidates_v1/candidate_C_dual_branch/seed_3409/best_model.pt` |

补充：old64 best也已核验，SHA256为`909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc`，seed 42。formal ledger状态为`LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK`。

## 5. Tensor、temporal、pooling与normalization事实

| 事实 | 状态 | verified value | evidence path / code path |
|---|---|---|---|
| Waymo Dynamic-v2 manifest | VERIFIED | 36 shards，168700 rows | `outputs/stage6r_dynamic_full51_semantic_strict_v1/stage6r_dynamic_full51_manifest.json`; SHA `c67391c...` |
| Waymo train shape | VERIFIED | logical `[135046,80,83]`, float32 | 36 shard `context_traj.npy` + `split.npy`; per-row `[80,83]` |
| Waymo val shape | VERIFIED | logical `[16870,80,83]`, float32 | 同上 |
| Waymo historical-test shape | VERIFIED | logical `[16784,80,83]`, float32 | 同上；已在Stage6V解盲，只能作historical development |
| cross-split scenario overlap | VERIFIED | 0 | recomputed from shard metadata |
| Stage7L context shape | VERIFIED | 每个dose `[80,150,83]`, float32 | `outputs/stage7l_e_prospective_bdd_v1/contexts/<dose>/context_traj.npy` |
| Stage7L actual learned input | VERIFIED | 完整`[B,150,83]`，无crop/slice | `tools/stage7l_e_run_prospective_bdd.py`; SHA `9eb9afe...` |
| training temporal contract | VERIFIED | `T=80,D=83` | `tools/stage6u_unified_abc_trainer.py`; SHA `b6b08f1...` |
| Stage7L inference temporal contract | VERIFIED | `T=150,D=83` | Stage7L context与inference chain |
| temporal contract match | VERIFIED | `false`（80 vs 150） | local contract JSON |
| A pooling | VERIFIED | single GRU `hidden[-1]` | `LegacySingleGRUEncoder` |
| B pooling | VERIFIED | single GRU `hidden[-1]`; `z[:16]` longitudinal + `z[16:64]` context | `PartitionedSingleGRUEncoder` |
| C pooling | VERIFIED | ego/context两分支各取`hidden[-1]`后concat 16+48 | `DualBranchEncoder` |
| mask consumption | VERIFIED | learned A/B/C不接收mask/length | Stage7L `ego_seq_mask`只供ego13；149帧row补零到150后最终零步参与pooling |
| learned input normalization | VERIFIED | NONE | raw float32 context（含999 sentinel）直接进入encoder |
| raw33 target scaler | VERIFIED | train-only population mean/std，epsilon `1e-6` | `stage6t_global_interaction_target_standardization.json`; SHA `fd92c818...` |
| clean longitudinal target | VERIFIED | train q01/q99 winsorize + train median/IQR normalization | Stage6T/Dynamic-v2合同 |
| raw33 historical SHA ledger completeness | AMBIGUOUS | 36个实际`interaction_feat_style_raw.npy`存在且已算SHA，但不在历史shard SHA ledger | 阻塞authoritative provenance，不代表文件缺失 |

## 6. ego13精确合同

状态：VERIFIED。

实现：`tools/stage6l_prepare_context_representation_ablation.py`，SHA256 `293e420b82b8c983d91d35d5f10ba4e57b7b745ad71308ebb9ef5e8170ee319d`。

有序13维：`mean_speed, std_speed, p95_speed, end_minus_start_speed, rms_accel, mean_abs_accel, p95_abs_accel, rms_jerk, p95_abs_jerk, rms_yaw_rate, mean_abs_yaw_rate, heading_change_abs_total, path_length`。

聚合窗口为`ego_seq_mask`选中的全部valid frames，`dt=0.1s`；accel/jerk/yaw-rate由speed/heading重算。正式median/IQR scaler来自dose100 conservative的183行，NPZ SHA256 `0b5b5a7049b97045f4e006ed97eb7a1500731a591aa33fd517aac99850296a19`。

## 7. BDD/MMD与independence合同

### 7.1 Stage7L paired

状态：VERIFIED。

- estimator：含对角项的biased MMD²；
- kernel：single RBF；
- bandwidth：每个representation×dose×task，在pooled baseline+treated rows的正off-diagonal距离上取exact median，fallback 1.0；
- null：每个scenario pair独立`+/-1` label swap；
- permutation unit：scenario pair；
- repetitions：100000；
- p：`(exceedance+1)/(B+1)`；
- q95：NumPy linear quantile；
- independent unit：same-scenario pair；log cluster不是该历史primary null的置换单位。

Manifest：`docs/stage7l_e_prospective_bdd_manifest_v1.json`，SHA256 `4fae0ede5bb77e86eec7f9aa1222b6605248b746dec8767ba3bd75fed6947a8b`。

### 7.2 Stage6P unpaired release

状态：VERIFIED。

- source：800 pairs / 1600 rows / 489 unique logs / 2400 frozen trials；
- sample sizes：200/250/300/400；
- constraint：release groups log-disjoint，scenario-token overlap forbidden；
- estimator：含对角项biased RBF MMD²；
- bandwidth：每个representation在冻结1600-row pool上用seed 20260811、20000 random index-pair draws拟合一次；
- null/calibration：无permutation；每个representation×sample size只用AA_CALIBRATION q95（higher）设阈值；AA_EVALUATION独立用于FPR；AB不参与阈值拟合。

Config：`configs/stage6p_representation_unpaired_release.json`，SHA256 `6febfe1608e4b312b8ea4107d939e1571ac9548f16cd670befa17a28b58eb33f`。

## 8. Holdout与future pool

| 项目 | 状态 | 结论 |
|---|---|---|
| `R0_AUDIT_HOLDOUT` | NOT_FOUND / BLOCKED | 当前已盘点Waymo test、Stage6P、Stage7、Stage7L都已使用或解盲；没有验证到可直接重标的outcome-blind、scenario/log-disjoint holdout。需要新建明确的候选池及功效/独立性检查，禁止从历史资产冒充。 |
| `FUTURE_R4_RESERVED_POOL` | NOT_FOUND / BLOCKED | 未发现已冻结的数据源/token pool/generator rules。必须在正式RBR训练前用outcome-blind规则另行锁定。 |

## 9. VERIFIED / AMBIGUOUS / BLOCKED汇总

### VERIFIED

- branch、base/governance commit关系及dirty worktree；
- 9/9 A/B/C best checkpoint与locked SHA；
- old64 checkpoint；
- Waymo 168700-row、80×83合同及split counts；
- Stage7L 80×150×83/dose与实际150步消费；
- A/B/C final-hidden pooling与mask不消费；
- learned input无normalization；
- raw33与clean-longitudinal target normalization；
- ego13有序schema、公式、mask窗口和scaler；
- Stage7L paired与Stage6P unpaired的MMD/kernel/null/independence合同。

### AMBIGUOUS

- 36个`interaction_feat_style_raw.npy`实际文件已核验，但历史authoritative shard SHA ledger缺少这些条目；只能确认当前文件SHA，不能声称历史ledger完整。
- Stage7L历史inference loader使用`torch.load(weights_only=False)`；本次audit自身使用`weights_only=True`安全加载。该项是历史实现事实，不等于checkpoint内容不可信。

### BLOCKED / NOT_FOUND

- 尚未冻结80训练/150推理的R0 policy；
- 尚未冻结learned encoder的padding/mask处理policy；
- raw33 authoritative SHA provenance仍有ledger缺口；
- `R0_AUDIT_HOLDOUT`未找到；
- `FUTURE_R4_RESERVED_POOL`未找到。

## 10. 当前治理状态

```text
RBR_DIRECTION_FROZEN
R0_SCIENTIFIC_SCOPE_FROZEN
LOCAL_CONTRACT_VERIFICATION_EXECUTED
LOCAL_RESULTS_INTEGRATED
R0_V1_PARAMETERIZATION_IN_PROGRESS
R0_OPERATIONAL_PROTOCOL_NOT_YET_FROZEN
RBR_TRAINING_NOT_AUTHORIZED
```
