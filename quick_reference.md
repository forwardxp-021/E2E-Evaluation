# Quick Reference

## Stage 5C-1 strict schema evaluation rerun
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
