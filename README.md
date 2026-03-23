# E2E-Evaluation
E2E Evaluation research in automated driving

## Analysis Tools

### UMAP Embedding Visualisation

Diagnose whether a learned trajectory embedding has captured driving-style
semantics by projecting it to 2-D (UMAP) and colouring each point with
handcrafted style features.

**Quick start:**

```bash
pip install -r requirements-analysis.txt

python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz \
    --output-dir outputs/umap
```

Generates four coloured scatter plots (one each for `rel_speed_mean`,
`thw_mean`, `jerk_p95`, `speed_norm`) plus a combined overview grid.

See **[docs/umap_visualization.md](docs/umap_visualization.md)** for the full
usage guide, input data format, all CLI options, and how to interpret the
output plots.
