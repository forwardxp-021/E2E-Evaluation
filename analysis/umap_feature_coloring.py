"""
UMAP/2D Feature-Coloring Analysis
==================================
Validates whether learned driving-style embeddings encode semantic style
information by projecting them to 2-D with UMAP and coloring each point by
a hand-crafted style feature.

Usage
-----
    python analysis/umap_feature_coloring.py \
        --embeddings path/to/embeddings.npy \
        --features   path/to/features.npy   \
        --feat-names rel_speed_mean thw_mean jerk_p95 speed_norm \
        --out-dir    results/umap_coloring

Expected file formats
---------------------
embeddings : np.ndarray, shape (N, D)
    Learned embedding vectors (e.g. 64-dim output of GRU→MLP encoder).
features   : np.ndarray, shape (N, F)
    Hand-crafted style features aligned row-wise with embeddings.
feat-names : list[str]
    Column names for the feature matrix (length must equal F).

Output
------
For each feature specified via --color-features (default: all feat-names),
the script saves:
  <out-dir>/<feature_name>.png   – scatter plot coloured by feature value
  <out-dir>/gradient_report.csv  – Spearman ρ between each UMAP axis and
                                   each feature, useful for quantitative
                                   gradient assessment

Related issue
-------------
GitHub Issue: Add UMAP feature-coloring analysis to validate driving-style
embeddings  →  forwardxp-021/E2E-Evaluation#1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import umap
from scipy.stats import spearmanr

matplotlib.use("Agg")  # non-interactive backend, safe for headless environments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UMAP feature-coloring analysis for driving-style embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embeddings", required=True, help="Path to embeddings .npy (N, D)")
    p.add_argument("--features", required=True, help="Path to features .npy (N, F)")
    p.add_argument(
        "--feat-names",
        nargs="+",
        required=True,
        metavar="NAME",
        help="Column names for the feature matrix (length must match F)",
    )
    p.add_argument(
        "--color-features",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Subset of feat-names to use for coloring plots (default: all)",
    )
    p.add_argument("--out-dir", default="results/umap_coloring", help="Output directory")
    p.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--metric", default="cosine", help="UMAP metric")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP")
    p.add_argument("--point-size", type=float, default=2.0, help="Scatter plot point size")
    p.add_argument("--alpha", type=float, default=0.4, help="Scatter plot alpha")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    p.add_argument(
        "--cmap",
        default="plasma",
        help="Matplotlib colormap for continuous features",
    )
    return p


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2-D with UMAP and return the projection."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
        low_memory=True,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


def plot_colored(
    projection: np.ndarray,
    color_values: np.ndarray,
    feature_name: str,
    out_path: str | os.PathLike,
    point_size: float = 2.0,
    alpha: float = 0.4,
    cmap: str = "plasma",
    dpi: int = 150,
) -> None:
    """Scatter plot of 2-D UMAP projection coloured by one feature."""
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=color_values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label=feature_name, fraction=0.046, pad=0.04)

    ax.set_title(f"UMAP projection  –  coloured by {feature_name}", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def compute_gradient_report(
    projection: np.ndarray,
    features: np.ndarray,
    feat_names: list[str],
) -> pd.DataFrame:
    """
    Compute Spearman ρ between each UMAP axis (UMAP-1, UMAP-2) and every
    feature column.  A large |ρ| indicates a monotone gradient along that
    axis, i.e. the embedding has encoded that feature as a continuous axis.
    """
    records = []
    for i, name in enumerate(feat_names):
        rho1, p1 = spearmanr(projection[:, 0], features[:, i])
        rho2, p2 = spearmanr(projection[:, 1], features[:, i])
        records.append(
            {
                "feature": name,
                "spearman_rho_umap1": round(rho1, 4),
                "pval_umap1": round(p1, 6),
                "spearman_rho_umap2": round(rho2, 4),
                "pval_umap2": round(p2, 6),
                "max_abs_rho": round(max(abs(rho1), abs(rho2)), 4),
            }
        )
    df = pd.DataFrame(records).sort_values("max_abs_rho", ascending=False)
    return df


def interpret_gradient_report(df: pd.DataFrame) -> str:
    """Return a short human-readable interpretation of the gradient report."""
    strong = df[df["max_abs_rho"] >= 0.3]
    moderate = df[(df["max_abs_rho"] >= 0.15) & (df["max_abs_rho"] < 0.3)]
    weak = df[df["max_abs_rho"] < 0.15]

    lines = ["=" * 60, "Gradient assessment (Spearman |ρ| thresholds)", "=" * 60]

    if not strong.empty:
        lines.append(
            f"✔  STRONG gradient (|ρ| ≥ 0.30): {', '.join(strong['feature'].tolist())}"
        )
    if not moderate.empty:
        lines.append(
            f"~  MODERATE gradient (|ρ| ∈ [0.15, 0.30)): {', '.join(moderate['feature'].tolist())}"
        )
    if not weak.empty:
        lines.append(
            f"✘  WEAK / NO gradient (|ρ| < 0.15): {', '.join(weak['feature'].tolist())}"
        )

    lines += [
        "",
        "Interpretation guide:",
        "  • ≥2 strong gradients  → embedding encodes style semantics successfully.",
        "  • Only weak gradients  → embedding is dominated by motion/scene patterns;",
        "                           consider multi-window aggregation or longer T.",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading embeddings from: {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"  shape: {embeddings.shape}")

    print(f"Loading features from:   {args.features}")
    features = np.load(args.features)
    print(f"  shape: {features.shape}")

    if embeddings.shape[0] != features.shape[0]:
        raise ValueError(
            f"Row count mismatch: embeddings has {embeddings.shape[0]} rows, "
            f"features has {features.shape[0]} rows."
        )
    if features.shape[1] != len(args.feat_names):
        raise ValueError(
            f"Feature matrix has {features.shape[1]} columns but "
            f"{len(args.feat_names)} names were provided."
        )

    color_features = args.color_features if args.color_features else args.feat_names
    unknown = set(color_features) - set(args.feat_names)
    if unknown:
        raise ValueError(f"Unknown color features (not in --feat-names): {unknown}")

    feat_index = {name: i for i, name in enumerate(args.feat_names)}

    # ------------------------------------------------------------------
    # UMAP projection
    # ------------------------------------------------------------------
    print("\nRunning UMAP …")
    projection = run_umap(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )

    np.save(out_dir / "umap_projection.npy", projection)
    print(f"  Projection saved to: {out_dir / 'umap_projection.npy'}")

    # ------------------------------------------------------------------
    # Coloured scatter plots
    # ------------------------------------------------------------------
    print("\nGenerating coloured scatter plots …")
    for feat_name in color_features:
        col_vals = features[:, feat_index[feat_name]]
        plot_colored(
            projection=projection,
            color_values=col_vals,
            feature_name=feat_name,
            out_path=out_dir / f"{feat_name}.png",
            point_size=args.point_size,
            alpha=args.alpha,
            cmap=args.cmap,
            dpi=args.dpi,
        )

    # ------------------------------------------------------------------
    # Gradient report
    # ------------------------------------------------------------------
    print("\nComputing Spearman gradient report …")
    report_df = compute_gradient_report(projection, features, args.feat_names)
    report_path = out_dir / "gradient_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"  Report saved to: {report_path}")

    print()
    print(interpret_gradient_report(report_df))


if __name__ == "__main__":
    main()
