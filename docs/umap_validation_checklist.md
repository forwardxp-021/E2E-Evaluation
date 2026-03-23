# UMAP Semantic-Coloring Validation Checklist

> **Related issue**: [Add UMAP feature-coloring analysis to validate driving-style embeddings](../../../issues) (label: `test`)  
> This checklist is reproduced here for long-term maintainability.  
> See `scripts/umap_analysis.py` for the implementation.

---

## Goal

Generate a single UMAP 2D projection for a large sample (default 100 k rows),
then produce **4 scatter plots**—one per style feature—all sharing the **same 2D layout**,
colored by:

| Feature | Description |
|---------|-------------|
| `rel_speed_mean` | Mean relative speed (ego vs. surrounding vehicles) |
| `thw_mean` | Mean time-headway (following-distance proxy) |
| `jerk_p95` | 95th-percentile longitudinal jerk (aggressiveness proxy) |
| `speed_norm` | Normalized speed (baseline / confounder reference) |

---

## 判定标准 / 怎么看图 — Per-Feature Checklist

### Feature 1: `rel_speed_mean`

- [ ] **Monotonic / continuous gradient (核心)**: Color changes smoothly along a dominant axis—not just isolated local patches. Reds should cluster on one side, blues on the other.
- [ ] **Stability across UMAP seeds / params**: Repeat with ≥ 3 random seeds (e.g. 0, 1, 2) and at least 2 different `n_neighbors` values; the gradient pattern must be consistent.
- [ ] **Fragmentation vs smooth gradients**: If the plot looks fragmented into many small "worms/lines", check whether color varies *smoothly along* them (motion-mode artifact) or is *globally structured* (genuine style axis).
- [ ] **Confounder check**: Stratify by `speed_norm` quantile buckets; a residual `rel_speed_mean` gradient should persist within each stratum.
- [ ] **Recordkeeping**: UMAP params (`n_neighbors`, `min_dist`, `metric`, `random_state`, `sample_size`), embedding source (checkpoint / hash), output directory, and saved 2D coords file (`umap_coords_*.npy`).

---

### Feature 2: `thw_mean`

- [ ] **Monotonic / continuous gradient**: Smooth color transition reflecting following-distance style (aggressive vs. cautious drivers separating spatially in the UMAP).
- [ ] **Stability across UMAP seeds / params**: Gradient pattern holds across ≥ 3 seeds and varied `n_neighbors` / `min_dist`.
- [ ] **Fragmentation vs smooth gradients**: Confirm gradient is *global* structure, not random speckle inside clusters.
- [ ] **Confounder check**: Stratify by scene / road-type if available; gradient should persist within each stratum (highway vs. urban).
- [ ] **Recordkeeping**: Save per-feature plot with informative filename, e.g. `umap_thw_mean_n15_d0.100_s42.png`.

---

### Feature 3: `jerk_p95`

- [ ] **Monotonic / continuous gradient**: Smooth color transition reflecting aggressiveness / comfort style.
- [ ] **Stability across UMAP seeds / params**: Gradient stable across ≥ 3 seeds; does not flip or disappear with parameter changes.
- [ ] **Fragmentation vs smooth gradients**: High fragmentation + random coloring → jerk not encoded as a style dimension in the embedding.
- [ ] **Confounder check**: Compare with the `speed_norm` UMAP map — the `jerk_p95` gradient must **not** be identical to the speed gradient everywhere (would indicate speed conflation).
- [ ] **Recordkeeping**: As above; include `jerk_p95` in the run metadata JSON.

---

### Feature 4: `speed_norm` (baseline / confounder reference)

- [ ] **Monotonic / continuous gradient**: Smooth speed-style transition — this is the *baseline* reference feature and usually shows the strongest signal.
- [ ] **Stability across UMAP seeds / params**: Stable across ≥ 3 seeds.
- [ ] **Fragmentation vs smooth gradients**: Speed is often the dominant structure; confirm that other features are *not* simply proxies for speed.
- [ ] **Confounder check**: Use `speed_norm` as the confounder baseline — if all other features' gradients vanish after stratifying by speed, the embedding encodes speed rather than driving style.
- [ ] **Recordkeeping**: 2D coords saved once (`umap_coords_*.npy`) and **reused** for all 4 features to ensure a fair comparison.

---

## Stability Across UMAP Seeds / Params — Protocol

Run the script at least three times varying the seed, then optionally vary `n_neighbors`:

```bash
# Seed sweep
for seed in 0 1 2; do
  python scripts/umap_analysis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.csv \
    --output-dir outputs/umap_seed${seed} \
    --random-state ${seed}
done

# n_neighbors sweep
for nn in 10 15 30; do
  python scripts/umap_analysis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.csv \
    --output-dir outputs/umap_nn${nn} \
    --n-neighbors ${nn}
done
```

Visually compare plots across runs. A robust style axis survives parameter variation.

---

## Outputs Per Run

| File | Description |
|------|-------------|
| `umap_coords_n{N}_d{D}_s{S}.npy` | 2D UMAP coordinates (shape `[n_samples, 2]`) — reuse for alternative colorings |
| `umap_rel_speed_mean_n{N}_d{D}_s{S}.png` | Scatter plot coloured by `rel_speed_mean` |
| `umap_thw_mean_n{N}_d{D}_s{S}.png` | Scatter plot coloured by `thw_mean` |
| `umap_jerk_p95_n{N}_d{D}_s{S}.png` | Scatter plot coloured by `jerk_p95` |
| `umap_speed_norm_n{N}_d{D}_s{S}.png` | Scatter plot coloured by `speed_norm` |
| `run_metadata.json` | UMAP params, data provenance, timestamp |

---

## Pass / Fail Criteria Summary

| Criterion | Pass | Fail |
|-----------|------|------|
| Gradient continuity | Smooth color band across embedding | Random speckle / no spatial structure |
| Seed stability | Pattern consistent across ≥ 3 seeds | Pattern changes dramatically with seed |
| Fragmentation | Color smooth within local blobs | Color random within blobs |
| Confounder (speed) | Other features' gradients persist after speed stratification | All gradients vanish → embedding only encodes speed |
| Recordkeeping | `run_metadata.json` + 2D coords saved | Output files missing or unnamed |
