# UMAP Embedding Visualization – Usage Guide

## Overview

`analysis/umap_embedding_vis.py` projects a learned trajectory embedding to
2-D via [UMAP](https://umap-learn.readthedocs.io) and colors each point with
one of several handcrafted driving-style features (continuous colormap).

The goal is to **diagnose** whether the embedding space contains a semantically
meaningful driving-style structure – i.e. whether continuous gradients
(e.g. left → right: aggressive → conservative) or interpretable axes exist.

---

## Installation

```bash
pip install -r requirements-analysis.txt
```

Required packages: `numpy`, `umap-learn`, `matplotlib`.  
Optional: `pandas` (recommended for CSV feature files).

---

## Input Data Format

### Embeddings file  
A 2-D array of shape `(N, D)` where `N` = number of trajectory samples and `D` = embedding dimension (e.g. 64).

| Format | Notes |
|--------|-------|
| `.npy` | Single array saved with `np.save` |
| `.npz` | Multi-array archive; key `embeddings` or `z` is used (first key if neither exists) |
| `.pkl` | Pickle of `np.ndarray` or `dict` with key `embeddings` / `z` / `embedding` |

Example – saving from a PyTorch training loop:

```python
import numpy as np
np.save("data/embeddings.npy", z_all.cpu().numpy())          # shape (N, 64)
np.savez("data/embeddings.npz", embeddings=z_all.cpu().numpy())
```

### Features file  
A collection of named 1-D arrays, each of length `N`, matching the embedding rows.

**Minimum required columns / keys:**

| Name | Description |
|------|-------------|
| `rel_speed_mean` | Mean relative speed w.r.t. lead vehicle |
| `thw_mean` | Mean time headway |
| `jerk_p95` | 95th-percentile longitudinal jerk |
| `speed_norm` | Normalised ego speed |

You may include any additional features; pass them via `--features-key`.

| Format | Notes |
|--------|-------|
| `.npz` | Recommended. Each feature is a named array: `np.savez("features.npz", rel_speed_mean=..., thw_mean=..., ...)` |
| `.csv` / `.tsv` | Column headers must match feature names |
| `.pkl` | Pickle of `dict[str, np.ndarray]` |
| `.npy` | 2-D array; columns are assigned to features **in the order** given by `--features-key` |

Example – creating a `.npz` from a pandas DataFrame:

```python
import numpy as np
np.savez(
    "data/features.npz",
    rel_speed_mean=df["rel_speed_mean"].values,
    thw_mean=df["thw_mean"].values,
    jerk_p95=df["jerk_p95"].values,
    speed_norm=df["speed_norm"].values,
)
```

---

## Basic Usage

### Minimal command

```bash
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz
```

This generates four colored scatter plots (one per default feature) plus a
combined overview grid, saved under `outputs/umap/`.

### With explicit feature keys

```bash
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz \
    --features-key rel_speed_mean thw_mean jerk_p95 speed_norm
```

### Tuning UMAP parameters

```bash
python analysis/umap_embedding_vis.py \
    --embeddings  data/embeddings.npy \
    --features    data/features.npz \
    --n-neighbors 15 \
    --min-dist    0.05 \
    --metric      euclidean \
    --random-state 42 \
    --n-samples   50000 \
    --output-dir  outputs/umap_nn15
```

### Stability check across random seeds

Providing multiple `--random-state` values runs UMAP once per seed and
produces a side-by-side comparison for each feature – useful for checking
whether gradients are stable across runs:

```bash
python analysis/umap_embedding_vis.py \
    --embeddings   data/embeddings.npy \
    --features     data/features.npz \
    --random-state 42 21 7 \
    --output-dir   outputs/umap_stability
```

### Save raw UMAP coordinates

```bash
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz \
    --save-umap \
    --output-dir outputs/umap
```

The 2-D coordinates are saved as `umap_coords_seed<N>_<timestamp>.npy` for
downstream analyses (e.g. Spearman correlation, probing classifiers).

### CSV feature file

```bash
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.csv \
    --features-key rel_speed_mean thw_mean jerk_p95 speed_norm
```

---

## Output Files

All files are saved to `--output-dir` (default: `outputs/umap/`).

| File pattern | Description |
|---|---|
| `umap_<feature>_seed<N>_<cfg>_<timestamp>.png` | Individual colored scatter per feature |
| `umap_overview_seed<N>_<cfg>_<timestamp>.png` | Combined 2×2 (or larger) grid |
| `umap_stability_<feature>_seeds<…>_<cfg>_<timestamp>.png` | Side-by-side across seeds |
| `umap_coords_seed<N>_<timestamp>.npy` | Raw 2-D coordinates (with `--save-umap`) |

---

## All CLI Options

```
usage: umap_embedding_vis.py [-h]
       --embeddings PATH --features PATH
       [--features-key KEY [KEY ...]]
       [--n-samples N] [--seed INT]
       [--n-neighbors INT] [--min-dist FLOAT] [--metric STR]
       [--random-state INT [INT ...]] [--n-jobs INT]
       [--output-dir DIR] [--format {png,pdf,svg}] [--dpi INT]
       [--point-size FLOAT] [--alpha FLOAT] [--cmap STR]
       [--save-umap] [--stability-feature KEY]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--embeddings` | – | Embedding file (required) |
| `--features` | – | Feature file (required) |
| `--features-key` | 4 defaults | Feature names to color by |
| `--n-samples` | `100000` | Max rows to use (`-1` = all) |
| `--seed` | `0` | Subsampling random seed |
| `--n-neighbors` | `30` | UMAP `n_neighbors` |
| `--min-dist` | `0.1` | UMAP `min_dist` |
| `--metric` | `cosine` | UMAP distance metric |
| `--random-state` | `42` | UMAP random seed(s); multiple values → stability mode |
| `--n-jobs` | `-1` | CPU threads for UMAP |
| `--output-dir` | `outputs/umap` | Output directory |
| `--format` | `png` | File format (`png` / `pdf` / `svg`) |
| `--dpi` | `150` | DPI for raster output |
| `--point-size` | `1.0` | Scatter marker size |
| `--alpha` | `0.4` | Scatter marker alpha |
| `--cmap` | `viridis` | Matplotlib colormap |
| `--save-umap` | off | Save raw 2-D UMAP coordinates |
| `--stability-feature` | first key | Feature used in stability overview title |

---

## Interpreting the Plots

| Observation | Interpretation |
|---|---|
| Continuous color gradient along a clear UMAP axis | Embedding has captured this style dimension ✔ |
| Same gradient stable across seeds | Signal is robust, not an artefact ✔ |
| Fragmented local color patches | Embedding reflects local motion/scene rather than global style ✗ |
| No visible correlation between color and position | Feature not encoded; consider architectural changes ✗ |

**Decision rule (suggested):**  
If at least 2 of the 4 core features show a stable, continuous gradient across seeds → the embedding has captured style semantics.  
If all 4 appear as fragmented noise → focus next on increasing the temporal window or adding vehicle-level consistency constraints.

---

## Troubleshooting

**`KeyError: The following required feature(s) are MISSING …`**  
The feature file does not contain all expected column names.  
- Check spelling and case (`jerk_p95` ≠ `Jerk_P95`).
- Pass the correct names via `--features-key`.
- If your file uses a column prefix (e.g. `feat_speed_norm`), strip it before saving or rename columns.

**`ImportError: umap-learn is required`**  
Run `pip install umap-learn`.

**`ValueError: Row count mismatch …`**  
The embedding and feature files must have the same number of rows `N`.

**UMAP is very slow**  
- Reduce `--n-samples` (e.g. `--n-samples 10000` for quick checks).
- Increase `--n-neighbors` (fewer tree splits).
- Ensure `numba` JIT cache is warm (first run is always slower).
