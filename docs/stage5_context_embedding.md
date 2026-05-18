# Stage 5 Context Embedding (Stage 5D Group-weighted Training)

## Stage 5C-2 findings (baseline)
- evaluator paper-grade flags all pass: strict schema + row alignment + no fallback index.
- learned embedding beats `random` / `context_l2` globally, but still below `raw_feature` / `pca_feature` globally.
- following_interaction is the largest weakness (`learned=0.302917` vs `raw=0.469712`, `pca=0.467968`).
- lateral_lane_dynamics is the key strength (`learned=0.266777` vs `raw=0.251786`, `pca=0.251469`).

## Why Stage 5D is needed
Stage 5D targets the imbalance observed in Stage 5C-2: improve following/front-distance interaction modeling while preserving lateral/lane-change advantage.

## Stage 5D design: group-weighted losses
1. Load `feature_schema.json` and resolve all group indices by feature name.
2. Add 5 auxiliary regression heads:
   - `aux_longitudinal`
   - `aux_following`
   - `aux_lateral_dynamics`
   - `aux_lateral_gap`
   - `aux_behavior_proxy`
3. Auxiliary regression uses **SmoothL1Loss** on normalized feature targets (dataset already standardized).
4. Add per-group metric alignment loss (embedding pairwise distance aligned to per-group feature pairwise distance).
5. Use following-upweighted defaults:
   - style: `1.0`
   - aux: `0.5/1.5/1.0/1.0/0.5`
   - metric: `0.5/2.0/1.0/1.0/0.5`

## Exact training command
```bash
python tools/train_context_behavior_embedding.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1 \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 256 \
  --epochs 20 \
  --lr 1e-3 \
  --metric_loss_type huber \
  --overwrite
```

## Exact embedding export command
```bash
python tools/export_context_row_embeddings.py \
  --shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --checkpoint outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1/best_model.pt \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_embeddings \
  --split all \
  --merge_embeddings
```

## Exact Stage 5C evaluator command
```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_group_weighted_v1_eval \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

## Expected outputs
Training output dir:
- `model.pt`
- `best_model.pt`
- `training_config.json`
- `feature_group_config.json`
- `train_log.csv`
- `training_summary.json`

Embedding output dir:
- `embedding_manifest.json`
- `embeddings/` (shard-aligned outputs)
- optional merged `embeddings.npy`

Evaluation output dir:
- `evaluation_summary.json`
- `evaluation_report.md`
- `category_correlation_summary.csv`
- retrieval/correlation plots and CSVs

## Success criteria
- learned still beats `random/context_l2`.
- following_interaction mean correlation improves vs Stage 5B baseline.
- lateral_lane_dynamics advantage is preserved.
- global retrieval improves, or at least does not degrade significantly.
