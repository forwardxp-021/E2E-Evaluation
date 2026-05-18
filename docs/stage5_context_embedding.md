# Stage 5 Context Embedding (Stage 5C-1 Strict Schema)

## Current Stage 5 feature schema (33-D)
Canonical feature order in `feature_schema.json`:

`rms_accel, rms_jerk, max_abs_accel, max_abs_jerk, mean_thw, min_thw, mean_front_distance, min_front_distance, mean_rel_speed, p95_rel_speed, rms_yaw_rate, rms_curvature, heading_change_total, lane_change_count_proxy, lane_change_rate_proxy, lane_change_left_count_proxy, lane_change_right_count_proxy, lane_change_duration_mean_proxy, max_lateral_speed, rms_lateral_accel, lane_change_oscillation_score_proxy, front_pressure_score, left_front_min_gap, left_rear_min_gap, right_front_min_gap, right_rear_min_gap, left_gap_min, right_gap_min, left_gap_acceptance_proxy, right_gap_acceptance_proxy, rear_vehicle_pressure_proxy, yielding_score_proxy, assertiveness_score_proxy`.

## Why `mean_speed` and `std_rel_speed` are not required
- They are **not present** in the canonical Stage 5 schema.
- Strict evaluator logic resolves indices by **feature name from schema only**.
- Therefore `mean_speed`/`std_rel_speed` are not evaluated; `p95_rel_speed` is used for relative-speed spread behavior.

## Exact rerun command
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

## Output directory
`outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5b_v1_eval_schema_fixed`

## Expected output files
- `evaluation_summary.json`
- `evaluation_report.md`
- `retrieval_metrics.csv`
- `style_distance_correlation.csv`
- `context_sensitivity_metrics.csv`
- `retrieval_bar.png`
- `feature_delta_correlation_bar.png`

## Interpretation guide
- **`learned_context_embedding`**: target representation from Stage 5B.
- **`raw_feature`**: oracle-like direct handcrafted feature baseline.
- **`pca_feature`**: lower-dimensional handcrafted baseline.
- **`context_l2`**: raw flattened trajectory/context baseline.
- **`random`**: sanity-floor baseline.

Interpretation:
- `learned_context_embedding` > `random` and > `context_l2`: embedding captures meaningful structure.
- Close to `raw_feature`/`pca_feature`: representation is competitive with handcrafted semantics.
- Worse than `raw_feature`/`pca_feature`: embedding training/objective likely needs tuning.
