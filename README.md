# E2E-Evaluation

End-to-end evaluation research for automated driving with a focus on
**driving-style representation learning**.

## Project Overview

The goal of this project is to learn a compact *style embedding* from vehicle
trajectory data (Waymo Open Dataset) such that the embedding space captures
semantic driving-style differences — aggressive vs. conservative, smooth vs.
jerky, close-following vs. cautious, etc.

### Current approach (V3)

| Component | Details |
|-----------|---------|
| Input | Trajectory `[N, T, D]` (position, velocity, acceleration) |
| Encoder | GRU → MLP → embedding (64-dim) |
| Supervision | Hand-crafted style features `[N, F]` (≈20 dims) |
| Loss | `0.2 × contrastive + 1.0 × feature_structure + 0.1 × variance` |

## Analysis Scripts

| Script | Description | Docs |
|--------|-------------|------|
| [`analysis/umap_feature_coloring.py`](analysis/umap_feature_coloring.py) | UMAP/2-D projection coloured by style features to validate whether the embedding encodes semantic style gradients | [docs/umap_feature_coloring.md](docs/umap_feature_coloring.md) |

## Issue Tracking

Active tasks are tracked as GitHub Issues:

| # | Title | Label |
|---|-------|-------|
| [#1](https://github.com/forwardxp-021/E2E-Evaluation/issues/1) | Add UMAP feature-coloring analysis to validate driving-style embeddings | `test` |

## Repository Structure

```
.
├── analysis/                  # Analysis and evaluation scripts
│   └── umap_feature_coloring.py
├── data/                      # Data (not committed – use .gitignore)
├── docs/                      # Documentation
│   └── umap_feature_coloring.md
└── README.md
```
