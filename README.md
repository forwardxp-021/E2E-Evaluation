# E2E-Evaluation

E2E Evaluation research in automated driving.

---

## UMAP Feature-Coloring Analysis

> **Issue**: [Add UMAP feature-coloring analysis to validate driving-style embeddings](../../issues) (label: `test`)  
> **Validation checklist**: [`docs/umap_validation_checklist.md`](docs/umap_validation_checklist.md)

This analysis verifies whether learned trajectory embeddings encode **driving style**
(stable per-driver preference) rather than only local motion patterns or speed proxies.

### How it works

A single UMAP 2D projection is computed for a sample of embeddings. Four scatter plots
are then produced—all sharing the **same 2D layout**—colored by:

| Feature | Meaning |
|---------|---------|
| `rel_speed_mean` | Mean relative speed (ego vs. surrounding vehicles) |
| `thw_mean` | Mean time-headway (following-distance proxy) |
| `jerk_p95` | 95th-percentile jerk (aggressiveness proxy) |
| `speed_norm` | Normalized speed (baseline / confounder reference) |

### Requirements

```bash
pip install umap-learn numpy pandas matplotlib
# Optional for parquet support:
pip install pyarrow
```

### Quick start

```bash
python scripts/umap_analysis.py \
    --embeddings  data/embeddings.npy \
    --features    data/features.csv \
    --output-dir  outputs/umap_run1
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--embeddings PATH` | _(required)_ | Embedding matrix (`.npy` or `.csv`) |
| `--features PATH` | _(required)_ | Feature table (`.csv` or `.parquet`) |
| `--output-dir DIR` | `outputs/umap` | Directory for plots and coords |
| `--n-neighbors N` | `15` | UMAP n_neighbors |
| `--min-dist D` | `0.1` | UMAP min_dist |
| `--metric METRIC` | `cosine` | Distance metric |
| `--random-state SEED` | `42` | Random seed |
| `--sample-size N` | `0` (all) | Rows to sample before UMAP |
| `--dpi DPI` | `150` | Plot resolution |
| `--point-size S` | `0.8` | Scatter point size |
| `--cmap CMAP` | `viridis` | Matplotlib colormap |

### Outputs

Each run saves:
- `umap_coords_n{N}_d{D}_s{S}.npy` — 2D coordinates (reuse for alternative colorings)
- `umap_<feature>_n{N}_d{D}_s{S}.png` — one plot per feature
- `run_metadata.json` — UMAP params, data provenance, timestamp

### Validating results

See [`docs/umap_validation_checklist.md`](docs/umap_validation_checklist.md) for the
full 判定标准 / 怎么看图 checklist covering gradient continuity, seed stability,
fragmentation diagnosis, confounder checks, and recordkeeping requirements.

---

## Repository layout

```
E2E-Evaluation/
├── data/               # Place embedding and feature files here
├── docs/
│   └── umap_validation_checklist.md
├── scripts/
│   └── umap_analysis.py
└── README.md
```
