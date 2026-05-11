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
