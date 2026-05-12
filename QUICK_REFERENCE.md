# QUICK_REFERENCE

## 1. 文档用途
本文件提供可直接执行的工作流命令与检查标准（命令优先）。
项目背景、研究目标与阶段结论请先阅读 [`README.md`](./README.md)。

## 2. Stage 4A：scaffold 验证

### 2.1 分配 pseudo labels（data1）
1. **命令**
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir data1 \
  --out_dir data1/pseudo_labels \
  --dataset_type synthetic
```
2. **期望行为**
- 生成 scaffold 数据的弱标签文件，供 baseline 验证使用。
3. **通过标准**
- `data1/pseudo_labels/pseudo_label_summary.json` 生成且可读取。

### 2.2 baseline-only 评估（data1）
1. **命令**
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir data1 \
  --label_dir data1/pseudo_labels \
  --out_dir outputs/data1_eval_baseline_only \
  --eval_split test \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict
```
2. **期望行为**
- 仅比较 baseline，不加载 learned embedding。
3. **通过标准**
- `human_validation_summary.json` 中 `learned_embedding_evaluated=false`。

### 2.3 bad learned embedding 拒绝测试
1. **命令**
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir data1 \
  --label_dir data1/pseudo_labels \
  --out_dir outputs/data1_eval_bad_embedding \
  --embedding_path outputs/bad_embeddings.npy \
  --eval_split test \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict
```
2. **期望行为**
- 对 shape/finite/row 对齐不满足要求的 embedding 报错或拒绝评估。
3. **通过标准**
- 日志出现明确 rejection 信息（shape mismatch、non-finite、row mismatch 之一）。

### 2.4 allow_skip_learned（保留 baseline 评估）
1. **命令**
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir data1 \
  --label_dir data1/pseudo_labels \
  --out_dir outputs/data1_eval_skip_learned \
  --embedding_path outputs/bad_embeddings.npy \
  --allow_skip_learned \
  --eval_split test \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict
```
2. **期望行为**
- learned 被跳过，baseline 仍输出完整结果。
3. **通过标准**
- `learned_embedding_evaluated=false` 且 baseline 指标文件齐全。

## 3. Stage 4B：Waymo human trajectory extraction

### 3.1 smoke test
1. **命令**
```bash
python tools/build_waymo_human_trajectory_dataset.py --smoke_test
```
2. **期望行为**
- 快速验证构建链路。
3. **通过标准**
- 输出目录成功生成 `traj/front/split` 等基础文件。

### 3.2 full51 构建
1. **命令**
```bash
python tools/build_waymo_human_trajectory_dataset.py \
  --tfrecord_dir /path/to/waymo_tfrecords \
  --out_dir outputs/waymo_human_v1_full51 \
  --limit_files 51
```
2. **期望行为**
- 仅提取 human windows，不做 policy rollout 扩展。
3. **通过标准**
- `build_summary.json` 存在，且 `n_windows_kept=168191`。

## 4. Stage 4C：Waymo human baseline validation

### 4.1 pseudo label 生成
1. **命令**
```bash
python tools/assign_pseudo_style_labels.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --dataset_type human_public
```
2. **期望行为**
- 生成人类轨迹弱标签。
3. **通过标准**
- `pseudo_label_summary.json` 生成且 `n_labeled > 0`。

### 4.2 baseline-only 评估
1. **命令**
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_baseline_only \
  --eval_split test \
  --baselines raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public
```
2. **期望行为**
- 只输出 baseline 对照结果。
3. **通过标准**
- `learned_embedding_evaluated=false`。

## 5. Stage 4D：row-level learned embedding

### 5.1 训练
1. **命令**
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model \
  --embedding_dim 64 --batch_size 512 --epochs 20 --lr 1e-3 \
  --temperature 0.1 --device cuda --seed 42 --overwrite
```
2. **期望行为**
- 仅使用 train split 训练 learned embedding。
3. **通过标准**
- `train_summary.json` 生成，`best_val_loss` 有效。

### 5.2 导出
1. **命令**
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --batch_size 1024 --device cuda --overwrite
```
2. **期望行为**
- 导出与 `traj.npy` 行对齐的 `(N,D)` embedding。
3. **通过标准**
- `embeddings_row_level.npy` 存在且 row-aligned。

### 5.3 评估
1. **命令**
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --eval_split test --distance euclidean --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict --dataset_type human_public --projection pca
```
2. **期望行为**
- learned 与 baseline 同台评估。
3. **通过标准**
- `learned_embedding_evaluated=true` 且 summary/csv 生成。

### 5.4 论文表格
1. **命令**
```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4d
```
2. **期望行为**
- 聚合 Stage 4D 关键指标，生成 paper-ready 表。
3. **通过标准**
- `paper_tables_summary.md` 生成。

## 6. Stage 4E：jerk/comfort-aware embedding

> 说明：Stage 4E 训练已完成；如导出/评估/对比尚未执行，请按下列顺序继续。

### 6.1 训练（已完成）
1. **命令**
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort \
  --embedding_dim 64 --batch_size 512 --epochs 20 --lr 1e-3 \
  --temperature 0.1 --feature_weight_mode jerk_comfort \
  --device cuda --seed 42 --overwrite
```
2. **期望行为**
- 使用 jerk/comfort-aware 权重训练 Stage 4E 模型。
3. **通过标准**
- `human_embedding_model_jerk_comfort/train_summary.json` 存在。

### 6.2 导出（必保留）
1. **命令**
```bash
python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite
```
2. **期望行为**
- 导出 Stage 4E 的 row-level embedding，不覆盖 Stage 4D 文件。
3. **通过标准**
- `embeddings_row_level_jerk_comfort.npy` 存在且行数等于 `traj.npy`。

### 6.3 评估（必保留）
1. **命令**
```bash
python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --eval_split test \
  --distance euclidean \
  --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict \
  --dataset_type human_public \
  --projection pca
```
2. **期望行为**
- 在 human_public test split 上评估 Stage 4E learned。
3. **通过标准**
- `eval_with_learned_jerk_comfort/human_validation_summary.json` 生成且 `learned_embedding_evaluated=true`。

### 6.4 Stage 4D vs 4E 对比（必保留）
1. **命令**
```bash
python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e
```
2. **期望行为**
- 对比分类、检索与 style-distance 相关指标，判断 jerk/comfort 是否改善。
3. **通过标准**
- `compare_stage4d_stage4e/comparison_summary.csv` 生成。

### 6.5 Stage 4E 论文表格（必保留）
1. **命令**
```bash
python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4e_jerk_comfort
```
2. **期望行为**
- 生成 Stage 4E 独立论文表格输入，不与 Stage 4D 混用。
3. **通过标准**
- `paper_tables_stage4e_jerk_comfort/paper_tables_summary.md` 生成。

## 7. 常见检查命令

### 7.1 npy shape
1. **命令**
```bash
python - <<'PY'
import numpy as np
x=np.load('outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy')
print(x.shape)
PY
```
2. **期望行为**
- 输出二维 shape（例如 `(168191, 64)`）。
3. **通过标准**
- 第一维与 `traj.npy` 行数一致。

### 7.2 embedding finite
1. **命令**
```bash
python - <<'PY'
import numpy as np
x=np.load('outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy')
print(np.isfinite(x).all())
PY
```
2. **期望行为**
- 输出 `True`。
3. **通过标准**
- 无 NaN/Inf。

### 7.3 row alignment
1. **命令**
```bash
python - <<'PY'
import numpy as np
traj=np.load('outputs/waymo_human_v1_full51/traj.npy')
emb=np.load('outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy')
print(len(traj), len(emb), len(traj)==len(emb))
PY
```
2. **期望行为**
- 打印相同行数与 `True`。
3. **通过标准**
- `len(traj)==len(emb)`。

### 7.4 train_summary
1. **命令**
```bash
python -m json.tool outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/train_summary.json | head
```
2. **期望行为**
- 看到 `best_val_loss` 等关键字段。
3. **通过标准**
- JSON 可解析，关键字段存在。

### 7.5 embedding_export_summary
1. **命令**
```bash
python -m json.tool outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort_export_summary.json | head
```
2. **期望行为**
- 输出导出统计与对齐信息。
3. **通过标准**
- JSON 可解析，包含 shape/row-aligned 相关字段。

### 7.6 baseline_comparison_summary
1. **命令**
```bash
head outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort/baseline_comparison_summary.csv
```
2. **期望行为**
- 包含 learned 与各 baseline 行。
3. **通过标准**
- CSV 可读取且 method 列包含 learned。

## 8. 常见问题
- `data1` 不是 human_public：它是 synthetic scaffold，仅用于流程验证。
- `embeddings.npy` 行数必须等于 `traj.npy`：评估按行索引对齐样本。
- 不允许自动 source-level expansion：会破坏 row-level 一致性并引入伪重复。
- pseudo labels 不是 ground truth：它们是 weak labels。
- `train_loss = NaN`：先检查轨迹清洗、特征裁剪、学习率与 loss 数值稳定设置。
- export 报 `normalize_local` non-finite：先做 trajectory sanitize，再核查输入是否仍含 NaN/Inf。
- hit@K 可能饱和：类别分布不均且 topK 对高频类更易命中。
