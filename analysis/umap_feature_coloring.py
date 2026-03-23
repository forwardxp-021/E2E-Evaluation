"""
UMAP Feature-Coloring Analysis for Driving-Style Embeddings
============================================================
Generates 4 UMAP plots (one per style feature) from the same 2D embedding to
validate whether the learned representation encodes driving style vs. motion/scene.

Features colored:
  - rel_speed_mean : relative speed (激进程度)
  - thw_mean       : time headway (跟车习惯)
  - jerk_p95       : 95th-percentile jerk (平滑性)
  - speed_norm     : normalised speed (速度偏好)

Usage
-----
python analysis/umap_feature_coloring.py \
    --embeddings path/to/embeddings.npy \
    --features   path/to/features.csv \
    --output_dir outputs/umap_coloring \
    --n_neighbors 30 \
    --min_dist 0.1 \
    --seed 42 \
    --sample_size 10000

Recordkeeping
-------------
Each run writes a JSON sidecar next to the output plots that captures:
  - UMAP parameters (n_neighbors, min_dist, metric)
  - random seed
  - sample size actually used
  - output directory
This makes results reproducible and auditable (see Issue #<ISSUE_NUMBER>).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature definitions – the four style axes we want to validate
# ---------------------------------------------------------------------------
STYLE_FEATURES = [
    "rel_speed_mean",
    "thw_mean",
    "jerk_p95",
    "speed_norm",
]

FEATURE_LABELS = {
    "rel_speed_mean": "Relative Speed (mean)",
    "thw_mean": "Time Headway (mean)",
    "jerk_p95": "Jerk 95th-percentile",
    "speed_norm": "Normalised Speed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from .npy file.  Shape: (N, D)."""
    emb = np.load(path)
    assert emb.ndim == 2, f"Expected 2-D array, got shape {emb.shape}"
    return emb


def _load_features(path: str) -> pd.DataFrame:
    """Load handcrafted features from CSV.  Must contain STYLE_FEATURES columns."""
    df = pd.read_csv(path)
    missing = [f for f in STYLE_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Feature CSV is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def _subsample(
    embeddings: np.ndarray,
    features: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Randomly subsample rows (aligned)."""
    n = len(embeddings)
    if sample_size >= n:
        return embeddings, features
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return embeddings[idx], features.iloc[idx].reset_index(drop=True)


def _run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    """Fit UMAP and return 2-D coordinates.  Shape: (N, 2)."""
    try:
        from umap import UMAP  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required.  Install with: pip install umap-learn"
        ) from exc

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Core plotting – one figure per feature, all from the *same* 2-D embedding
# ---------------------------------------------------------------------------

def plot_umap_colored_by_features(
    coords_2d: np.ndarray,
    features: pd.DataFrame,
    output_dir: Path,
    alpha: float = 0.4,
    point_size: float = 5.0,
    cmap: str = "viridis",
) -> list[Path]:
    """
    Generate one scatter plot per style feature, colouring each point by
    the feature value.

    Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for feature in STYLE_FEATURES:
        values = features[feature].values.astype(float)

        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=values,
            cmap=cmap,
            alpha=alpha,
            s=point_size,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(FEATURE_LABELS[feature], fontsize=10)

        ax.set_title(
            f"UMAP coloured by: {FEATURE_LABELS[feature]}",
            fontsize=12,
            pad=12,
        )
        ax.set_xlabel("UMAP dim 1", fontsize=10)
        ax.set_ylabel("UMAP dim 2", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.tight_layout()

        out_path = output_dir / f"umap_colored_{feature}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        print(f"  Saved: {out_path}")

    return saved_paths


# ---------------------------------------------------------------------------
# Recordkeeping sidecar
# ---------------------------------------------------------------------------

def _save_run_record(
    output_dir: Path,
    args: argparse.Namespace,
    actual_sample_size: int,
    umap_time_s: float,
    saved_plots: list[Path],
) -> Path:
    """Write a JSON sidecar with all run parameters for reproducibility."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embeddings_path": str(args.embeddings),
        "features_path": str(args.features),
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "n_components": 2,
        },
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "actual_sample_size": actual_sample_size,
        "umap_fit_seconds": round(umap_time_s, 1),
        "output_dir": str(output_dir),
        "plots": [str(p) for p in saved_plots],
    }
    record_path = output_dir / "run_record.json"
    with open(record_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    print(f"  Run record: {record_path}")
    return record_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UMAP feature-coloring analysis for driving-style embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to .npy file with embeddings, shape (N, D).",
    )
    parser.add_argument(
        "--features",
        required=True,
        help=(
            "Path to CSV with handcrafted features.  "
            "Must include: rel_speed_mean, thw_mean, jerk_p95, speed_norm."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/umap_coloring",
        help="Directory where plots and run record are written.",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="UMAP distance metric.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP and subsampling (test across multiple seeds).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="Number of samples to use (set <=0 to use all).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Point transparency for scatter plots.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=5.0,
        help="Point size for scatter plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("UMAP Feature-Coloring Analysis")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/4] Loading embeddings from: {args.embeddings}")
    embeddings = _load_embeddings(args.embeddings)
    print(f"      Shape: {embeddings.shape}")

    print(f"[1/4] Loading features from: {args.features}")
    features = _load_features(args.features)
    assert len(features) == len(embeddings), (
        f"Row count mismatch: embeddings={len(embeddings)}, features={len(features)}"
    )

    # 2. Subsample
    sample_size = args.sample_size if args.sample_size > 0 else len(embeddings)
    print(f"\n[2/4] Subsampling to {sample_size} rows (seed={args.seed})")
    embeddings, features = _subsample(embeddings, features, sample_size, args.seed)
    actual_sample_size = len(embeddings)
    print(f"      Actual sample size: {actual_sample_size}")

    # 3. Fit UMAP (once, shared across all 4 plots)
    print(
        f"\n[3/4] Fitting UMAP  "
        f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, "
        f"metric={args.metric}, seed={args.seed})"
    )
    t0 = time.time()
    coords_2d = _run_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )
    umap_time_s = time.time() - t0
    print(f"      Done in {umap_time_s:.1f}s. Output shape: {coords_2d.shape}")

    # Save the 2-D coordinates so the plot step can be re-run cheaply
    coords_path = output_dir / "umap_coords_2d.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(coords_path, coords_2d)
    print(f"      2-D coords saved: {coords_path}")

    # 4. Plot – one figure per feature
    print(f"\n[4/4] Generating {len(STYLE_FEATURES)} feature-coloring plots")
    saved_plots = plot_umap_colored_by_features(
        coords_2d=coords_2d,
        features=features,
        output_dir=output_dir,
        alpha=args.alpha,
        point_size=args.point_size,
    )

    # Recordkeeping sidecar
    _save_run_record(
        output_dir=output_dir,
        args=args,
        actual_sample_size=actual_sample_size,
        umap_time_s=umap_time_s,
        saved_plots=saved_plots,
    )

    print("\nDone! Plots written to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
