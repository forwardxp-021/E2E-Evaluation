# Stage 5 Context Embedding (Stage 5C-1 Schema-Fixed)

## Paths
- Stage 5A dataset root: `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged`
- Stage 5B embeddings root: `outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings`

## Stage 5C purpose
Evaluate whether learned context embeddings preserve behavior/style similarity better than baselines.

## Why Stage 5C v1 was preliminary
- `feature_names_used` was empty.
- Evaluator used fallback hardcoded feature indices.
- Therefore results were smoke-test only, not paper-grade validity.

## Stage 5C-1 fixes
- Canonical `feature_schema.json` (33 ordered dimensions) is defined from the same source as feature construction.
- Builder now writes `feature_schema.json` with manifest outputs.
- Strict schema loading is default in evaluator; no silent fallback in strict mode.
- Context sensitivity now uses named schema features (`mean_thw`, `min_thw`, `mean_front_distance`, `min_front_distance`, `mean_rel_speed`, `std_rel_speed`) with neighbor absolute-delta metric.

## Generate schema (no full rebuild required)
```bash
python tools/write_feature_schema.py \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged
```

Alternative (builder utility mode):
```bash
python tools/build_waymo_5neighbor_context_dataset.py \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged \
  --write_schema_only
```

## Re-run Stage 5C strict evaluation
```bash
python tools/evaluate_context_embedding.py \
  --embedding_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_embeddings/embedding_manifest.json \
  --source_shard_manifest outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/shard_manifest.json \
  --feature_schema outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/feature_schema.json \
  --out_dir outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_schema_fixed \
  --max_eval_samples 20000 \
  --eval_split test \
  --seed 42 \
  --overwrite
```

## Expected output directory
`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_schema_fixed`

## Expected files
- `evaluation_summary.json`
- `retrieval_metrics.csv`
- `style_distance_correlation.csv`
- `context_sensitivity_metrics.csv`
- `retrieval_bar.png`
- `feature_delta_correlation_bar.png`
- `pca_embedding.png`
- `pca_feature.png`
- `evaluation_report.md`

## Interpretation
- `learned_context_embedding > random/context_l2`: embedding is meaningful.
- `learned_context_embedding > raw_feature/pca_feature`: strong evidence.
- `learned_context_embedding < raw_feature/pca_feature`: proceed to Stage 5D model improvements.
