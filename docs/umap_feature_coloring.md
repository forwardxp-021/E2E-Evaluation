# UMAP / 2-D Feature-Coloring Analysis

> **Related issue:** [Add UMAP feature-coloring analysis to validate driving-style embeddings](https://github.com/forwardxp-021/E2E-Evaluation/issues/1)

## Purpose

This analysis validates whether the learned driving-style embedding truly
encodes **semantic style information** — or whether it mainly captures
low-level motion/scene patterns.

The key diagnostic question is:

> When the embedding is projected to 2-D with UMAP, do hand-crafted style
> features (e.g. `rel_speed_mean`, `thw_mean`, `jerk_p95`, `speed_norm`)
> form **continuous, monotone gradients** across the projection?

A clear gradient means the embedding has implicitly learned to organise
samples along a style axis.  No gradient means the embedding is dominated
by motion pattern or scene context.

---

## Quick Start

### 1 – Install dependencies

```bash
pip install umap-learn numpy pandas matplotlib scipy
```

### 2 – Prepare your data

| File | Shape | Description |
|------|-------|-------------|
| `embeddings.npy` | `(N, D)` | Learned embedding vectors (e.g. 64-dim GRU→MLP output) |
| `features.npy`   | `(N, F)` | Hand-crafted style features, row-aligned with embeddings |

### 3 – Run the script

```bash
python analysis/umap_feature_coloring.py \
    --embeddings path/to/embeddings.npy \
    --features   path/to/features.npy   \
    --feat-names rel_speed_mean thw_mean jerk_p95 speed_norm \
    --out-dir    results/umap_coloring
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--embeddings` | *(required)* | Path to embeddings `.npy` |
| `--features` | *(required)* | Path to feature matrix `.npy` |
| `--feat-names` | *(required)* | Column names for feature matrix |
| `--color-features` | all feat-names | Subset of features to plot |
| `--out-dir` | `results/umap_coloring` | Output directory |
| `--n-neighbors` | `15` | UMAP `n_neighbors` |
| `--min-dist` | `0.1` | UMAP `min_dist` |
| `--metric` | `cosine` | UMAP distance metric |
| `--random-state` | `42` | Random seed |
| `--point-size` | `2.0` | Scatter point size |
| `--alpha` | `0.4` | Scatter point opacity |
| `--dpi` | `150` | Figure resolution |
| `--cmap` | `plasma` | Matplotlib colormap |

---

## Output Files

```
results/umap_coloring/
├── umap_projection.npy      # (N, 2) UMAP coordinates for downstream use
├── rel_speed_mean.png        # Scatter coloured by rel_speed_mean
├── thw_mean.png              # Scatter coloured by thw_mean
├── jerk_p95.png              # Scatter coloured by jerk_p95
├── speed_norm.png            # Scatter coloured by speed_norm
└── gradient_report.csv       # Spearman ρ between UMAP axes and features
```

### `gradient_report.csv` schema

| Column | Description |
|--------|-------------|
| `feature` | Feature name |
| `spearman_rho_umap1` | Spearman ρ with UMAP axis 1 |
| `pval_umap1` | p-value for above |
| `spearman_rho_umap2` | Spearman ρ with UMAP axis 2 |
| `pval_umap2` | p-value for above |
| `max_abs_rho` | `max(|ρ₁|, |ρ₂|)` — primary ranking metric |

---

## Interpreting Results

### Decision thresholds

| `max_abs_rho` | Assessment |
|---------------|------------|
| ≥ 0.30 | **Strong gradient** — embedding encodes this feature as a clear style axis |
| 0.15 – 0.30 | **Moderate gradient** — partial encoding; may improve with longer time horizon |
| < 0.15 | **Weak / no gradient** — feature not encoded; see improvement directions below |

### Overall verdict

| Condition | Verdict |
|-----------|---------|
| ≥ 2 features show **strong** gradient | Embedding successfully encodes driving style |
| Only 1 strong gradient | Partial encoding; targeted improvement needed |
| All weak gradients | Embedding dominated by motion/scene; redesign required |

### Scatter plot reading guide

Look for:
- **Monotone colour sweep** across the projection (low–high in one direction) → strong gradient
- **Locally coherent patches** (similar colour in neighbouring blobs) → weak but non-zero encoding
- **Uniform noise** or **random speckle** → no encoding

---

## Improvement Directions (if gradients are weak)

Based on the project context, if the gradient report shows weak encoding:

1. **Increase temporal scale** – extend trajectory window from T=50 to T=150,
   or use multi-window aggregation (encode 3 consecutive windows and mean-pool).
2. **Add same-vehicle consistency constraint** – pull embeddings of multiple
   segments from the same vehicle together in the loss.
3. **Filter confounded features** – remove features that co-vary with speed
   or road type (scene proxies); keep only cross-scene stable features for
   `feature_structure_loss`.
4. **Quantile supervision** – convert continuous features to rank/quantile
   labels for more robust gradient learning.

---

## Example: Running on synthetic data

```python
import numpy as np

# Generate toy data (replace with real embeddings/features)
N, D, F = 1000, 64, 4
np.save("/tmp/embeddings.npy", np.random.randn(N, D).astype(np.float32))
np.save("/tmp/features.npy",   np.random.randn(N, F).astype(np.float32))
```

```bash
python analysis/umap_feature_coloring.py \
    --embeddings /tmp/embeddings.npy \
    --features   /tmp/features.npy   \
    --feat-names rel_speed_mean thw_mean jerk_p95 speed_norm \
    --out-dir    /tmp/umap_out
```

---

*Last updated: 2026-03-23 — see [issue #1](https://github.com/forwardxp-021/E2E-Evaluation/issues/1) for task history.*
