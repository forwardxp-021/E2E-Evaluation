# E2E-Evaluation

End-to-end evaluation research for automated driving with a focus on
**driving-style representation learning** from trajectory data (Waymo Open Dataset).

---

## Project Overview

The goal of this project is to learn a compact *style embedding* from vehicle
trajectory data such that the embedding space captures semantic driving-style
differences — aggressive vs. conservative, smooth vs. jerky, close-following
vs. cautious, etc.

### Current approach (V3)

| Component | Details |
|-----------|---------|
| Input | Trajectory `[N, T, D]` (position, velocity, acceleration) |
| Encoder | GRU → MLP → embedding (64-dim) |
| Supervision | Hand-crafted style features `[N, F]` (≈20 dims) |
| Loss | `0.2 × contrastive + 1.0 × feature_structure + 0.1 × variance` |

---

## Analysis Scripts

| Script | Description | Docs |
|--------|-------------|------|
| [`analysis/umap_feature_coloring.py`](analysis/umap_feature_coloring.py) | UMAP/2-D projection coloured by style features to validate whether the embedding encodes semantic style gradients | [docs/umap_feature_coloring.md](docs/umap_feature_coloring.md) |
| [`analysis/umap_embedding_vis.py`](analysis/umap_embedding_vis.py) | UMAP 2-D embedding visualization with combined overview grid | [docs/umap_visualization.md](docs/umap_visualization.md) |
| [`scripts/umap_analysis.py`](scripts/umap_analysis.py) | CLI wrapper: runs UMAP + produces 4 per-feature scatter plots + metadata | [docs/umap_validation_checklist.md](docs/umap_validation_checklist.md) |

### Quick-start

```bash
pip install -r requirements-analysis.txt

# Feature-coloring analysis (validates style gradients with Spearman report)
python analysis/umap_feature_coloring.py \
    --embeddings path/to/embeddings.npy \
    --features   path/to/features.npy \
    --feat-names rel_speed_mean thw_mean jerk_p95 speed_norm \
    --out-dir    outputs/umap_coloring

# Standalone UMAP analysis CLI (CSV features, saves coords + metadata)
python scripts/umap_analysis.py \
    --embeddings path/to/embeddings.npy \
    --features   path/to/features.csv \
    --output-dir outputs/umap_run1

# Embedding visualisation tool (multi-feature overview grid)
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz \
    --output-dir outputs/umap
```

---

## 判定标准 / 怎么看图 (Interpretation checklist)

Use the checklist below when reviewing each of the 4 feature-colored UMAP plots.
A full version is available in [docs/umap_validation_checklist.md](docs/umap_validation_checklist.md).

#### `rel_speed_mean` — Relative Speed
- [ ] Look for a **monotonic, continuous colour gradient along an axis** (e.g. left → right transitions from low to high relative speed), not just isolated clusters of a single colour.
- [ ] Re-run with at least 2 different random seeds and 2 different `n_neighbors` values; a meaningful gradient should remain **stable** across settings.
- [ ] Flag if the gradient is **fragmented** into many disconnected patches.
- [ ] **Confounder check:** split data into speed buckets and re-plot within each bucket.

#### `thw_mean` — Time Headway
- [ ] Look for a **monotonic, continuous colour gradient along an axis**.
- [ ] Re-run with at least 2 different random seeds and 2 different `n_neighbors` values.
- [ ] Flag **fragmented gradients** as evidence that headway behaviour is not globally structured.
- [ ] **Confounder check:** if road-type labels are available, compare with those.

#### `jerk_p95` — 95th-Percentile Jerk
- [ ] Look for a **monotonic, continuous colour gradient**.
- [ ] Verify **stability** across seeds and `n_neighbors` settings.
- [ ] Flag if the gradient is **fragmented**.
- [ ] **Confounder check:** check whether the gradient correlates with `speed_norm` alone.

#### `speed_norm` — Normalised Speed
- [ ] Look for a **monotonic, continuous colour gradient**.
- [ ] Verify **stability** across seeds and `n_neighbors` settings.
- [ ] Flag **fragmentation**.
- [ ] **Confounder check:** re-plot within each scene type if scene labels are available.

#### Cross-feature stability
- [ ] Do at least 2 features show **stable gradients** under the same UMAP layout? If yes, the embedding is likely encoding style information.
- [ ] If all 4 features show only fragmented patches, model changes are needed (longer time window, multi-segment aggregation).

#### Recordkeeping
- [ ] Save the `run_record.json` / `run_metadata.json` sidecar alongside every set of plots.
- [ ] Store plots in a versioned directory (e.g. `outputs/umap_coloring/v3_epoch8_seed42/`).

---

## Issue Tracking

Active tasks are tracked as GitHub Issues:

| # | Title | Label |
|---|-------|-------|
| [#1](https://github.com/forwardxp-021/E2E-Evaluation/issues/1) | Add UMAP feature-coloring analysis to validate driving-style embeddings | `test` |

---

## Repository Structure

```
.
├── analysis/                        # Analysis and evaluation scripts
│   ├── umap_feature_coloring.py     # UMAP feature-coloring (Spearman gradient report)
│   └── umap_embedding_vis.py        # UMAP embedding visualisation (overview grid)
├── scripts/
│   └── umap_analysis.py             # CLI: UMAP + 4 scatter plots + metadata
├── data/                            # Data files (not committed – see .gitignore)
├── docs/
│   ├── umap_feature_coloring.md     # Detailed docs for umap_feature_coloring.py
│   ├── umap_visualization.md        # Detailed docs for umap_embedding_vis.py
│   └── umap_validation_checklist.md # Full 判定标准 checklist
├── outputs/                         # Generated plots and records (gitignored)
├── requirements-analysis.txt        # Python dependencies for analysis scripts
└── README.md
```
