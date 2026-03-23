# E2E-Evaluation

E2E Evaluation research in automated driving — learning driving-style representations
from trajectory data (Waymo Open Dataset).

---

## UMAP Feature-Coloring Analysis

> **Tracked in:** GitHub Issue — [Add UMAP feature-coloring analysis to validate driving-style embeddings](../../issues)  
> **Full checklist**: [`docs/umap_validation_checklist.md`](docs/umap_validation_checklist.md)

The goal of this analysis is to verify whether the learned embedding encodes
**driving style** (激进/保守/平滑/跟车习惯) or primarily captures motion/scene patterns.

### Quick-start

```bash
pip install umap-learn matplotlib numpy pandas

# Option A — simple CLI
python analysis/umap_feature_coloring.py \
    --embeddings path/to/embeddings.npy \
    --features   path/to/features.csv \
    --output_dir outputs/umap_coloring \
    --n_neighbors 30 \
    --min_dist 0.1 \
    --seed 42 \
    --sample_size 10000

# Option B — extended CLI (more options, saves 2D coords for reuse across colorings)
python scripts/umap_analysis.py \
    --embeddings  path/to/embeddings.npy \
    --features    path/to/features.csv \
    --output-dir  outputs/umap_run1 \
    --n-neighbors 15 \
    --min-dist    0.1 \
    --metric      cosine \
    --random-state 42 \
    --sample-size 50000
```

Both scripts generate **4 UMAP scatter plots** — one per style feature — all
projected from the **same 2-D embedding** so layouts are directly comparable.

| Plot file | Feature | Driving-style axis |
|---|---|---|
| `umap_colored_rel_speed_mean.png` | `rel_speed_mean` | Aggressiveness (激进程度) |
| `umap_colored_thw_mean.png` | `thw_mean` | Following distance (跟车习惯) |
| `umap_colored_jerk_p95.png` | `jerk_p95` | Smoothness (平滑性) |
| `umap_colored_speed_norm.png` | `speed_norm` | Speed preference (速度偏好) |

A `run_record.json` sidecar is also written containing UMAP params, seed, and
sample size for reproducibility.

---

### 判定标准 / 怎么看图 (Interpretation checklist)

Use the checklist below when reviewing each of the 4 feature-colored UMAP plots.

#### `rel_speed_mean` — Relative Speed
- [ ] Look for a **monotonic, continuous colour gradient along an axis** (e.g. left → right transitions from low to high relative speed), not just isolated clusters of a single colour.
- [ ] Re-run with at least 2 different random seeds (e.g. `--seed 0`, `--seed 123`) and 2 different `n_neighbors` values (e.g. 15 and 50); a meaningful gradient should remain **stable** across settings.
- [ ] Flag if the gradient is **fragmented** into many disconnected patches — this indicates the embedding has not formed a continuous style axis for this feature.
- [ ] **Confounder check:** split data into speed buckets (e.g. `speed_norm` quartiles) and re-plot within each bucket; if the gradient disappears after splitting, `rel_speed_mean` may be a proxy for overall speed rather than a style signal.

#### `thw_mean` — Time Headway
- [ ] Look for a **monotonic, continuous colour gradient along an axis** (short headway → long headway should map to a smooth colour transition in 2-D space).
- [ ] Re-run with at least 2 different random seeds and 2 different `n_neighbors` values; the gradient should be **stable** across settings.
- [ ] Flag **fragmented gradients** (many small patches instead of a smooth band) as evidence that headway behaviour is not globally structured in the embedding.
- [ ] **Confounder check:** if road-type or scene labels are available, colour by those and compare — ensure the headway gradient is not explained purely by road type (e.g. motorway vs. urban).

#### `jerk_p95` — 95th-Percentile Jerk
- [ ] Look for a **monotonic, continuous colour gradient** — one region should contain consistently high jerk (aggressive/jerky driving) and another should contain low jerk (smooth driving).
- [ ] Verify **stability** across seeds and `n_neighbors` settings (gradient should not appear or disappear with minor parameter changes).
- [ ] Flag if the gradient is **fragmented** — high-jerk and low-jerk points interleaved without structure.
- [ ] **Confounder check:** jerk can be inflated by sensor noise at high speeds; check whether the gradient correlates with `speed_norm` alone by colouring those points with speed and comparing.

#### `speed_norm` — Normalised Speed
- [ ] Look for a **monotonic, continuous colour gradient** (slow → fast driving mapped to a smooth axis).
- [ ] Verify **stability** across seeds and `n_neighbors` settings.
- [ ] Flag **fragmentation** — scattered high-speed/low-speed patches without a clear axis.
- [ ] **Confounder check:** `speed_norm` is the most likely scene confound (motorways are fast by definition). If scene/road-type labels are available, re-plot within each scene type; a true style signal should persist within scene type.

#### Cross-feature stability
- [ ] After reviewing all 4 plots: do at least 2 features show **stable gradients** under the same UMAP layout? If yes, the embedding is likely encoding style information.
- [ ] If all 4 features show only fragmented patches, the embedding is primarily capturing motion/scene — model changes (longer time window, multi-segment aggregation) are needed.

#### Recordkeeping
- [ ] Save `run_record.json` (auto-generated) alongside every set of plots. It captures: UMAP `n_neighbors`, `min_dist`, `metric`, random seed, and sample size.
- [ ] Store plots in a versioned directory (e.g. `outputs/umap_coloring/v3_epoch8_seed42/`).
- [ ] Note the epoch/checkpoint used in the directory name or in a `README` inside the output directory.

---

## Repository structure

```
analysis/
  umap_feature_coloring.py      # UMAP feature-coloring script (simple CLI)
scripts/
  umap_analysis.py              # Extended UMAP script (more options, saves 2D coords)
docs/
  umap_validation_checklist.md  # Full 判定标准 checklist
data/
  .gitkeep                      # placeholder — add data files here (not committed)
outputs/                        # generated plots and records (gitignored)
```
