# E2E-Evaluation

## 1. 项目目标
本项目聚焦 **trajectory-level closed-loop evaluation**：
- 不做 sensor rendering；
- 不引入 perception stack；
- 基于轨迹与相对运动构建 behavior embedding；
- 在 policy separation、style retrieval、Waymo human validation 场景验证表示质量。

## 2. 当前阶段总览

| 阶段 | 状态 | 说明 |
|---|---|---|
| Stage 1/2 synthetic rollout | 完成 | p0/p1/p2 policy separation |
| Stage 3 ablation/local sweep | 完成 | lateral_stable mechanism analysis |
| Stage 4A scaffold | 完成 | data1 synthetic scaffold |
| Stage 4B Waymo human extraction | 完成 | full51 = 168191 windows |
| Stage 4C baseline validation | 完成 | baseline-only full51 |
| Stage 4D row-level learned embedding | 完成 | learned evaluated on full51 |
| Stage 4E jerk/comfort-aware embedding | 进行中 | training done; export/eval/compare next |

## 3. 数据目录说明
- `data1/` 是 synthetic rollout scaffold，不是 human validation。
- `outputs/waymo_human_v1_full51/` 是 `human_public` 验证数据。
- 任何 `embeddings_*.npy` 都必须与 `traj.npy` **逐行对齐**（row-aligned）。

## 4. 阶段 4A：scaffold 验证
### 命令
```bash
python tools/build_waymo_human_trajectory_dataset.py --smoke_test
```
### 期望行为
- 仅用于流程联调，验证脚本链路可运行。
- 输出样本来自 scaffold，不代表真实 human_public 结论。
### 通过标准
- 构建脚本无报错。
- 输出目录生成 `traj/front/split` 等基础文件。

## 5. 阶段 4B：Waymo human trajectory extraction
### 命令
```bash
python tools/build_waymo_human_trajectory_dataset.py \
  --tfrecord_dir /path/to/waymo_tfrecords \
  --out_dir outputs/waymo_human_v1_full51 \
  --limit_files 51
```
### 期望行为
- 仅提取真实人类窗口，不生成 policy rollout。
- 输出 full51 数据集供后续验证。
### 通过标准
- `build_summary.json` 生成。
- `n_windows_kept=168191`。

## 6. 阶段 4C：Waymo human baseline validation
### 命令
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
### 期望行为
- Stage 4C 不评估 learned embedding。
- 以 baseline 建立 human_public 对照。
### 通过标准
- `learned_embedding_evaluated=false`。
- baseline 对比文件完整输出。

## 7. 阶段 4D：row-level learned embedding
### 命令
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model \
  --embedding_dim 64 --batch_size 512 --epochs 20 --lr 1e-3 \
  --temperature 0.1 --device cuda --seed 42 --overwrite

python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --batch_size 1024 --device cuda --overwrite

python tools/evaluate_vehicledata_validation.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --label_dir outputs/waymo_human_v1_full51/pseudo_labels \
  --out_dir outputs/waymo_human_v1_full51/eval_with_learned \
  --embedding_path outputs/waymo_human_v1_full51/embeddings_row_level.npy \
  --eval_split test --distance euclidean --topk 5 \
  --baselines learned,raw_feature,trajectory_l2,random,pca_feature \
  --retrieval_mode strict --dataset_type human_public --projection pca

python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4d
```
### 期望行为
- 训练只使用 train split。
- pseudo labels 仅用于 evaluation，不用于训练。
- Stage 4D 输出单独保留，不与 Stage 4E 互相覆盖。
### 通过标准
- `embeddings_row_level.npy` 存在且 row-aligned。
- `eval_with_learned/human_validation_summary.json` 中 `learned_embedding_evaluated=true`。

## 8. 阶段 4E：jerk/comfort-aware embedding 训练后的导出、评估与对比
### 命令
```bash
python tools/train_human_behavior_embedding.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --out_dir outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort \
  --embedding_dim 64 \
  --batch_size 512 \
  --epochs 20 \
  --lr 1e-3 \
  --temperature 0.1 \
  --feature_weight_mode jerk_comfort \
  --device cuda \
  --seed 42 \
  --overwrite

python tools/export_human_row_embeddings.py \
  --data_dir outputs/waymo_human_v1_full51 \
  --checkpoint outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/model.pt \
  --out_path outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort.npy \
  --batch_size 1024 \
  --device cuda \
  --overwrite

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

python tools/compare_embedding_runs.py \
  --runs \
    stage4d_v1=outputs/waymo_human_v1_full51/eval_with_learned \
    stage4e_jerk_comfort=outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --out_dir outputs/waymo_human_v1_full51/compare_stage4d_stage4e

python tools/generate_paper_tables.py \
  --eval_dir outputs/waymo_human_v1_full51/eval_with_learned_jerk_comfort \
  --train_summary outputs/waymo_human_v1_full51/human_embedding_model_jerk_comfort/train_summary.json \
  --export_summary outputs/waymo_human_v1_full51/embeddings_row_level_jerk_comfort_export_summary.json \
  --pseudo_label_summary outputs/waymo_human_v1_full51/pseudo_labels/pseudo_label_summary.json \
  --build_summary outputs/waymo_human_v1_full51/build_summary.json \
  --out_dir outputs/waymo_human_v1_full51/paper_tables_stage4e_jerk_comfort
```
### 期望行为
- Stage 4E 使用 jerk_comfort 特征权重训练 human row-level learned embedding。
- 训练仍然只使用 train split。
- 不使用 pseudo labels 训练。
- pseudo labels 只用于 evaluation。
- 导出的 embeddings_row_level_jerk_comfort.npy 必须与 traj.npy 行对齐。
- eval_with_learned_jerk_comfort 用 test split 评估 learned embedding。
- compare_stage4d_stage4e 对比 Stage 4D v1 和 Stage 4E jerk_comfort。
- 重点观察 Stage 4E 是否提升 rms_jerk_delta / comfort 相关指标，同时不让 classification / retrieval 崩掉。
- Stage 4E 是改进实验，不覆盖 Stage 4D v1 结果。
### 通过标准
- embeddings_row_level_jerk_comfort.npy 存在。
- embeddings_row_level_jerk_comfort.npy.shape[0] == len(traj.npy) == 168191。
- embedding 全部 finite，无 NaN/Inf。
- eval_with_learned_jerk_comfort/human_validation_summary.json 中 learned_embedding_evaluated=true。
- baseline_comparison_summary.csv 中包含 learned。
- learned 的 classification / retrieval 明显高于 random。
- learned 的 rms_jerk_delta correlation 相比 Stage 4D v1 有提升，或者 report 明确说明没有提升。
- learned 的 hit@1 / mean_same_label_fraction_topk 不能明显塌缩到 random 水平。
- compare_stage4d_stage4e/comparison_summary.csv 生成。
- Stage 4D v1 和 Stage 4E 的结果都保留，不互相覆盖。

## 9. 常见问题
- 为什么 data1 不是 human_public？因为 data1 是 synthetic scaffold，仅用于联调。
- 为什么 embeddings.npy 的行数必须等于 traj.npy？因为 evaluation 按行索引对齐样本。
- 为什么不能自动 expand source-level embedding？会破坏 row-level 一致性并引入伪重复。
- 为什么 pseudo labels 不是 ground truth？它们来自规则/特征映射，属于弱标签。
- 为什么 raw_feature / pca_feature 检索很强？它们与 pseudo-label 生成特征空间近似同源。
- 为什么 hit@K 可能饱和？标签分布不均+topK 容易命中高频类。
- 为什么 Waymo traj.npy 可能包含 NaN？原始轨迹切窗与速度差分存在缺测。
- 如果 train_loss 是 NaN 怎么办？检查轨迹清洗、feature clipping、loss 数值稳定设置。
- 如果 export normalize_local 报非有限值怎么办？先做 sanitize，再检查输入轨迹是否仍含 NaN/Inf。

## 10. 论文实验产物
- `build_summary.json`
- `pseudo_label_summary.json`
- `baseline_comparison_summary.csv`
- `style_distance_correlation.csv`
- `paper_tables_summary.md`
- `comparison_summary.csv`

> paper tables 路径约定：
> - Stage 4D v1 使用 `eval_with_learned` + `human_embedding_model` + `embeddings_row_level.npy`。
> - Stage 4E 使用 `eval_with_learned_jerk_comfort` + `human_embedding_model_jerk_comfort` + `embeddings_row_level_jerk_comfort.npy`。
> - 除非通过 `compare_embedding_runs.py`，不要混用 Stage 4D/4E 表格输入。

## 11. 限制与下一步
- pseudo labels are weak labels。
- learned embedding does not yet outperform all baselines。
- Stage 4E focuses on jerk/comfort sensitivity。
- 后续工作：qualitative retrieval、ablation、paper tables 深化、benchmark packaging。
