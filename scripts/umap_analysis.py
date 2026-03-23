#!/usr/bin/env python3
"""
umap_analysis.py — UMAP 2D semantic-coloring analysis for driving-style embeddings.

Runs UMAP on a pre-computed embedding matrix and produces:
  1. A single 2D coordinate file  (``umap_coords.npy``)
  2. Four scatter plots, one per style feature, all sharing the same 2D layout:
       - rel_speed_mean
       - thw_mean
       - jerk_p95
       - speed_norm

Usage example
-------------
python scripts/umap_analysis.py \\
    --embeddings  data/embeddings.npy \\
    --features    data/features.csv \\
    --output-dir  outputs/umap_run1 \\
    --n-neighbors 15 \\
    --min-dist    0.1 \\
    --metric      cosine \\
    --random-state 42 \\
    --sample-size  50000

Related issue: see forwardxp-021/E2E-Evaluation#1 "Add UMAP feature-coloring
analysis to validate driving-style embeddings" (label: test).

判定标准 / 怎么看图 checklist is reproduced in docs/umap_validation_checklist.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns of interest
# ---------------------------------------------------------------------------
STYLE_FEATURES = [
    "rel_speed_mean",
    "thw_mean",
    "jerk_p95",
    "speed_norm",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="umap_analysis",
        description="Run UMAP on driving-style embeddings and produce semantic-coloring plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embeddings",
        required=True,
        metavar="PATH",
        help="Path to embedding matrix (.npy, shape [N, D]) or a CSV file whose"
             " first D columns are the embedding dimensions.",
    )
    p.add_argument(
        "--features",
        required=True,
        metavar="PATH",
        help="Path to feature table (.csv or .parquet) that contains at least the"
             f" columns: {', '.join(STYLE_FEATURES)}."
             " Row order must match the embedding matrix.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/umap",
        metavar="DIR",
        help="Directory where 2D coordinates and plots will be saved.",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        metavar="N",
        help="UMAP n_neighbors — controls local vs global structure trade-off.",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        metavar="D",
        help="UMAP min_dist — controls how tightly points are packed.",
    )
    p.add_argument(
        "--metric",
        default="cosine",
        metavar="METRIC",
        help="Distance metric used by UMAP (e.g. cosine, euclidean, manhattan).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for UMAP reproducibility.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        metavar="N",
        help="Number of rows to sample before running UMAP (0 = use all rows).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="DPI",
        help="DPI for saved plots.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=0.8,
        metavar="S",
        help="Scatter point size.",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        metavar="CMAP",
        help="Matplotlib colormap for the scatter plots.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> np.ndarray:
    """Return embedding matrix as float32 numpy array, shape [N, D]."""
    fpath = Path(path)
    if fpath.suffix == ".npy":
        emb = np.load(fpath).astype(np.float32)
    elif fpath.suffix in {".csv", ".tsv"}:
        sep = "," if fpath.suffix == ".csv" else "\t"
        import pandas as pd  # type: ignore
        emb = pd.read_csv(fpath, sep=sep, header=None).values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported embedding file format: {fpath.suffix}")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2-D embedding array, got shape {emb.shape}")
    log.info("Loaded embeddings: shape=%s", emb.shape)
    return emb


def load_features(path: str) -> "pandas.DataFrame":  # noqa: F821
    """Return feature DataFrame. Requires at least the STYLE_FEATURES columns."""
    import pandas as pd  # type: ignore

    fpath = Path(path)
    if fpath.suffix == ".parquet":
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath)

    missing = [c for c in STYLE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature table is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    log.info("Loaded feature table: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    """Fit UMAP and return 2D coordinates, shape [N, 2]."""
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'umap-learn' package is required. Install it with:\n"
            "  pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
        low_memory=False,
    )
    log.info(
        "Running UMAP  n_neighbors=%d  min_dist=%.3f  metric=%s  seed=%d …",
        n_neighbors, min_dist, metric, random_state,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    log.info("UMAP done: output shape=%s", coords.shape)
    return coords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _param_suffix(n_neighbors: int, min_dist: float, random_state: int) -> str:
    return f"n{n_neighbors}_d{min_dist:.3f}_s{random_state}"


def plot_feature(
    coords: np.ndarray,
    values: "pandas.Series",  # noqa: F821
    feature_name: str,
    *,
    output_dir: Path,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    dpi: int,
    point_size: float,
    cmap: str,
) -> Path:
    """Save one scatter plot coloured by *values*. Returns the saved file path."""
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    suffix = _param_suffix(n_neighbors, min_dist, random_state)
    filename = f"umap_{feature_name}_{suffix}.png"
    out_path = output_dir / filename

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=point_size,
        cmap=cmap,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label(feature_name, fontsize=10)
    ax.set_title(
        f"UMAP coloured by {feature_name}\n"
        f"n_neighbors={n_neighbors}  min_dist={min_dist:.3f}  seed={random_state}",
        fontsize=10,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved plot: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Record-keeping helpers
# ---------------------------------------------------------------------------

def save_run_metadata(
    output_dir: Path,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
    sample_size: int,
    n_total: int,
    embeddings_path: str,
    features_path: str,
) -> None:
    """Write a JSON file recording all UMAP parameters and provenance."""
    import json
    import datetime

    meta = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "umap": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state,
            "n_components": 2,
        },
        "data": {
            "embeddings_path": str(embeddings_path),
            "features_path": str(features_path),
            "sample_size": sample_size if sample_size > 0 else n_total,
            "n_total": n_total,
        },
        "features_analysed": STYLE_FEATURES,
    }
    meta_path = output_dir / "run_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Saved run metadata: %s", meta_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    embeddings = load_embeddings(args.embeddings)
    features_df = load_features(args.features)

    n_total = len(embeddings)
    if len(features_df) != n_total:
        log.error(
            "Embedding rows (%d) != feature rows (%d). "
            "Check that the files share the same row order.",
            n_total, len(features_df),
        )
        return 1

    # --- Optional downsampling -----------------------------------------------
    sample_size = args.sample_size
    if sample_size > 0 and sample_size < n_total:
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(n_total, size=sample_size, replace=False)
        idx.sort()
        embeddings = embeddings[idx]
        features_df = features_df.iloc[idx].reset_index(drop=True)
        log.info("Sampled %d / %d rows", sample_size, n_total)
    else:
        sample_size = n_total

    # --- Run UMAP ------------------------------------------------------------
    coords = run_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )

    # --- Save 2D coordinates -------------------------------------------------
    suffix = _param_suffix(args.n_neighbors, args.min_dist, args.random_state)
    coords_path = output_dir / f"umap_coords_{suffix}.npy"
    np.save(coords_path, coords)
    log.info("Saved 2D coords: %s  shape=%s", coords_path, coords.shape)

    # --- Save run metadata ---------------------------------------------------
    save_run_metadata(
        output_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        sample_size=sample_size,
        n_total=n_total,
        embeddings_path=args.embeddings,
        features_path=args.features,
    )

    # --- Produce one plot per feature (all share same UMAP layout) -----------
    saved_plots: list[Path] = []
    for feature in STYLE_FEATURES:
        values = features_df[feature]
        plot_path = plot_feature(
            coords,
            values,
            feature,
            output_dir=output_dir,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
            dpi=args.dpi,
            point_size=args.point_size,
            cmap=args.cmap,
        )
        saved_plots.append(plot_path)

    log.info("All done. Output directory: %s", output_dir.resolve())
    log.info("Plots:\n%s", "\n".join(f"  {p}" for p in saved_plots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
