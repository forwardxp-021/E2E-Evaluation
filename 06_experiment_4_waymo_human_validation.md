# 06_experiment_4_waymo_human_validation — Waymo Human Public Validation

## 1. 实验目标
验证轨迹级行为表示在 `human_public` 数据上的可区分、可检索、可解释能力。

## 2. 为什么需要 Stage 4
Stage 3 主要是合成策略；Stage 4 需要外部真实人类轨迹验证，避免只在 synthetic 场景内闭环。

## 3. Stage 4A scaffold 结论
`data1` 不是公开人类验证集，而是基于 Waymo source window 的 synthetic rollout scaffold（含 p0/p1/p2）。可用于评估流程联调，不可作为 public human evidence。

## 4. Stage 4B Waymo human builder 结论
已完成 `tools/build_waymo_human_trajectory_dataset.py`。仅提取真实人类窗口，不生成 p0/p1/p2，不含 policy_id/policy_name。

## 5. Stage 4C full51 baseline-only 结论
`outputs/waymo_human_v1_full51` 上已完成 baseline-only：`learned_embedding_evaluated=false`，仅比较 raw_feature / trajectory_l2 / random / pca_feature。

## 6. 数据规模统计
- full51: `n_files_processed=51`
- `n_scenarios_processed=24872`
- `n_windows_kept=168191`（即 168191 条 human trajectory windows）

## 7. pseudo-label 分布
- n_total: 168191
- n_labeled: 75421
- n_unlabeled: 92770
- pseudo labels 为弱规则标签，不是 ground truth。

## 8. baseline-only 结果总结
随机基线接近 chance；raw_feature/pca_feature 对同风格检索较强；trajectory_l2 对几何/速度相关项相关性较强。指标以 CSV 为准，不硬编码图上估值。

## 9. 目前不能过度声称什么
- 不能声称已完成 human_public learned embedding 验证。
- 不能把 pseudo label 指标当真实行为真值泛化结论。

## 10. 当前限制
存在特征泄漏风险：pseudo labels 来自 style features，需结合 strict retrieval、baseline 对照、style-distance correlation、cluster fingerprint 联合解释。

## 11. 下一步 Stage 4D：row-level learned embedding
必须在 full51 上训练并导出 row-aligned learned embedding（`(N,D)`，N=168191，建议 D=64），并与 raw_feature/trajectory_l2/random/pca_feature 同台评估。

## 12. 当前任务状态表

| Task | Status | Notes |
|---|---|---|
| Stage 4A scaffold | 完成 | data1 synthetic scaffold |
| Waymo smoke builder | 完成 | 36 samples |
| Waymo small extraction | 完成 | 260 samples |
| Waymo medium validation | 完成 | 1953 samples |
| Waymo full51 extraction | 完成 | 168191 samples |
| full51 pseudo labels | 完成 | 75421 labeled |
| full51 baseline-only evaluation | 完成 | learned not evaluated |
| human row-level learned embedding | 未完成 | Stage 4D |
| learned vs baselines | 未完成 | Stage 4D |
| report auto-fill | 待完善 | human_validation_report.md still too empty |

## 13. Stage 4D 当前问题与修复
- 初始 Stage 4D 训练出现 `train_loss=nan` 与 `val_loss=nan`。
- 根因：`traj.npy` 中存在 trajectory/velocity NaN，且 feature 标准化后出现极端值。
- 修复：加入 trajectory NaN 插值与对齐过滤、feature 标准化后 clipping、稳定 soft contrastive loss（log_softmax/softmax + 对角 mask + 非有限检查）。


### Stage 4D export 问题与修复
- export 初始报错：`normalize_local produced non-finite values`。
- 根因：导出脚本未对含 NaN/Inf 的 Waymo human 轨迹执行与训练一致的清洗流程。
- 修复：将 `sanitize_trajectory_array` 与 `normalize_local` 抽取到共享预处理模块，并在训练/导出两侧统一调用。

# Stage 4D：Waymo human row-level learned embedding 结果

Stage 4D 已完成，learned embedding 已在 human_public full51 上完成评估，且导出为 row-aligned。

## 数据规模
- n_files_processed: 51
- n_scenarios_processed: 24872
- n_windows_kept: 168191
- split: train=134637, val=16823, test=16731

## 训练结果
- best_val_loss: 6.101393699645996
- final_train_loss: 6.105895535574213
- final_val_loss: 6.101393699645996

## 导出结果
- embeddings_row_level.npy shape=[168191, 64]
- row_aligned=true

## evaluation summary
- learned_embedding_evaluated=true
- learned_embedding_alignment=row_aligned
- learned_embedding_valid_for_policy_eval=true

## learned vs baselines table
| method | acc | hit@1 | mean_same_topk |
|---|---:|---:|---:|
| learned | 0.665990 | 0.695752 | 0.684713 |
| raw_feature | 0.587933 | 0.910308 | 0.895076 |
| trajectory_l2 | 0.695887 | 0.684389 | 0.680195 |
| random | 0.311418 | 0.424107 | 0.415260 |
| pca_feature | 0.579410 | 0.915990 | 0.896104 |

## style-distance correlation table
| method | mean_speed | rms_jerk | rms_yaw_rate | rms_curvature |
|---|---:|---:|---:|---:|
| learned | 0.571203 | 0.069744 | 0.339976 | 0.502536 |
| raw_feature | - | 0.325408 | 0.318599 | 0.433656 |
| trajectory_l2 | 0.915366 | - | 0.089636 | 0.010284 |
| pca_feature | - | 0.327921 | 0.324885 | 0.453873 |

## 当前结论
learned embedding 提供非随机且可解释的行为结构，分类较强，且对横向/曲率风格敏感。

## 不能过度声称的内容
- 不可声称 learned 全面优于所有 baseline。
- pseudo labels 不是 ground truth。

## 当前短板
jerk/comfort 敏感性偏弱；raw_feature/pca_feature 在检索上仍显著更强。

## Stage 4E 下一步
面向 jerk/comfort 的特征加权训练与消融比较。

| Task | Status | Notes |
|---|---|---|
| Stage 4A scaffold | 完成 | data1 synthetic scaffold |
| Stage 4B Waymo human extraction | 完成 | full51 = 168191 windows |
| Stage 4C baseline-only validation | 完成 | raw_feature / trajectory_l2 / pca_feature |
| Stage 4D row-level learned embedding | 完成 | learned evaluated on full51 |
| Report auto-fill | 部分完成 | needs next-step correction and NaN handling |
| Stage 4E jerk/comfort-aware embedding | 未完成 | next target |

# Stage 4E：jerk/comfort-aware 训练完成后的下一步

Stage 4E 训练已完成（jerk_comfort 权重训练）。当前必须按顺序执行后处理：先导出 row-level embedding，再进行 evaluation，最后对比 Stage 4D v1 vs Stage 4E。

- Stage 4E training has completed.
- Next required step is export.
- Then evaluate.
- Then compare Stage 4D v1 vs Stage 4E.
- Do not overwrite Stage 4D v1 results.
- Stage 4E result should be judged by:
  - learned classification
  - learned hit@1
  - learned mean_same_label_fraction_topk
  - learned rms_jerk_delta correlation
  - learned rms_yaw_rate_delta correlation
  - learned rms_curvature_delta correlation
  - no collapse to random

建议执行命令：
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --batch_size 1024 --device cuda --overwrite

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --eval_split test --distance euclidean --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict --dataset_type human_public --projection pca

python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e
```

论文表格输入必须区分：
- Stage 4D v1: `eval_with_learned` + `human_embedding_model` + `embeddings_row_level.npy`
- Stage 4E: `eval_with_learned_jerk_comfort` + `human_embedding_model_jerk_comfort` + `embeddings_row_level_jerk_comfort.npy`
- 除 `compare_embedding_runs.py` 外，不应混用 Stage 4D 与 Stage 4E 输入。

| Task | Status | Notes |
|---|---|---|
| Stage 4E training | 完成 | jerk_comfort model trained |
| Stage 4E export | 待执行 | export row-level embeddings |
| Stage 4E evaluation | 待执行 | eval_with_learned_jerk_comfort |
| Stage 4D vs 4E comparison | 待执行 | compare_embedding_runs |
| README restore | 完成 | overview document |
| QUICK_REFERENCE restore | 完成 | command document |


## 14. 文档职责恢复（README / QUICK_REFERENCE）
- README.md 与 QUICK_REFERENCE.md 均为持续维护文档。
- README.md 负责项目目标、研究方向、阶段进展与结论。
- QUICK_REFERENCE.md 负责可执行命令、期望行为、通过标准。
- Stage 4E 训练已完成；当前最近一步是按顺序执行 export → evaluation → comparison。
- Stage 4D 与 Stage 4E 输出目录必须分离，禁止覆盖。

| Task | Status | Notes |
|---|---|---|
| README restore | 完成 | overview document |
| QUICK_REFERENCE restore | 完成 | command document |
| Stage 4E training | 完成 | jerk_comfort model trained |
| Stage 4E export | 待执行 | export row-level embeddings |
| Stage 4E evaluation | 待执行 | eval_with_learned_jerk_comfort |
| Stage 4D vs 4E comparison | 待执行 | compare_embedding_runs |

## Stage 4F：为什么要做 comfort auxiliary regression

- Stage 4D learned embedding 的 jerk correlation 明显偏弱（例如 `spearman_rms_jerk_delta` 远低于 yaw/curvature/speed）。
- Stage 4E 的 jerk_comfort 特征加权未提升 jerk correlation，且整体指标轻微回退，说明简单重加权不足。
- Stage 4F 在 soft contrastive 主目标之外，增加从 embedding 到 comfort 特征的辅助回归监督，直接约束 embedding 保留 jerk/comfort 信息。
- Stage 4F 训练仍不使用 pseudo labels；pseudo labels 仅用于下游评估对比。
- 【流程修正】`evaluate_aux_predictions.py` 在早期 Stage 4F workflow 中缺失；现已列为必做诊断步骤（训练后、导出前）。
- 该脚本用于确认 auxiliary head 是否真正预测 jerk/comfort targets，而不只看训练 loss。
- 脚本在 `normalize_local` 前必须先做 Waymo human 轨迹 NaN 清洗（与 train/export 共用 sanitization 逻辑）。
