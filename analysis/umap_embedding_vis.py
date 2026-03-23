"""
UMAP 2D Embedding Visualization for Driving Style Analysis
===========================================================
Diagnoses whether a learned trajectory embedding has captured driving-style
semantics by projecting it to 2D via UMAP and coloring each point with
handcrafted style features (continuous colormap).

Usage example
-------------
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz \
    --features-key rel_speed_mean thw_mean jerk_p95 speed_norm \
    --n-samples  100000 \
    --n-neighbors 30 \
    --min-dist   0.1 \
    --metric     cosine \
    --random-state 42 \
    --output-dir outputs/umap

# Multiple random states (stability check):
python analysis/umap_embedding_vis.py \
    --embeddings data/embeddings.npy \
    --features   data/features.npz \
    --random-state 42 21 7 \
    --output-dir outputs/umap

See docs/umap_visualization.md for full documentation.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required style features (at minimum these must be present)
# ---------------------------------------------------------------------------
REQUIRED_FEATURES = ["rel_speed_mean", "thw_mean", "jerk_p95", "speed_norm"]

# ---------------------------------------------------------------------------
# Helpers: data loading
# ---------------------------------------------------------------------------

def _load_array_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr


def _load_npz_as_dict(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def load_embeddings(path: str) -> np.ndarray:
    """Load embedding array from .npy, .npz (first key), or .pkl file.

    Expected shape: (N, D) – one row per trajectory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = _load_array_npy(p)
    elif suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        keys = data.files
        if len(keys) == 0:
            raise ValueError(f"NPZ file '{path}' contains no arrays.")
        if "embeddings" in keys:
            arr = data["embeddings"]
        elif "z" in keys:
            arr = data["z"]
        else:
            arr = data[keys[0]]
            log.warning(
                "No 'embeddings'/'z' key found in %s; using first key '%s'.",
                path,
                keys[0],
            )
    elif suffix == ".pkl":
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            arr = obj
        elif isinstance(obj, dict):
            candidates = ["embeddings", "z", "embedding"]
            for c in candidates:
                if c in obj:
                    arr = np.asarray(obj[c])
                    break
            else:
                raise ValueError(
                    f"PKL dict has no recognised key ({candidates}). "
                    f"Available: {list(obj.keys())}"
                )
        else:
            raise ValueError(
                f"Unsupported pickle content type: {type(obj)}. "
                "Expected np.ndarray or dict."
            )
    else:
        raise ValueError(
            f"Unsupported embedding file extension '{suffix}'. "
            "Use .npy, .npz, or .pkl."
        )

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding array must be 2-D (N, D), got shape {arr.shape}."
        )
    log.info("Loaded embeddings: shape=%s, dtype=%s", arr.shape, arr.dtype)
    return arr


def load_features(
    path: str,
    required_keys: List[str],
    csv_key_prefix: str = "",
) -> Dict[str, np.ndarray]:
    """Load handcrafted features from .npy, .npz, .pkl, or .csv/.tsv.

    Returns a dict mapping feature name → 1-D array of length N.

    Parameters
    ----------
    path:
        Path to the feature file.
    required_keys:
        Feature names that *must* be present. Raises a clear
        ``KeyError`` listing every missing key.
    csv_key_prefix:
        Optional prefix to strip from CSV column names before matching
        (e.g. ``"feat_"`` so that ``feat_speed_norm`` → ``speed_norm``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    suffix = p.suffix.lower()
    feat_dict: Dict[str, np.ndarray] = {}

    if suffix == ".npy":
        arr = _load_array_npy(p)
        if arr.ndim == 1:
            raise ValueError(
                "Feature .npy file is 1-D. Use .npz with named keys, or "
                "a .csv file with column headers."
            )
        log.warning(
            "Feature .npy is shape %s with no column names; "
            "will attempt to match by position against required keys in order.",
            arr.shape,
        )
        if arr.shape[1] < len(required_keys):
            raise ValueError(
                f"Feature array has only {arr.shape[1]} columns "
                f"but {len(required_keys)} required keys were requested."
            )
        for i, k in enumerate(required_keys):
            feat_dict[k] = arr[:, i].astype(np.float32)

    elif suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        feat_dict = {k: data[k].ravel().astype(np.float32) for k in data.files}

    elif suffix == ".pkl":
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            raise ValueError(
                f"Feature .pkl must contain a dict, got {type(obj)}."
            )
        feat_dict = {k: np.asarray(v).ravel().astype(np.float32) for k, v in obj.items()}

    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            import pandas as pd  # optional but very handy for CSV
        except ImportError:
            # fallback: plain numpy
            log.warning("pandas not installed; using numpy genfromtxt for CSV.")
            raw = np.genfromtxt(p, delimiter=sep, names=True, dtype=np.float32)
            for name in raw.dtype.names:
                col = name.lstrip(csv_key_prefix) if csv_key_prefix else name
                feat_dict[col] = raw[name].astype(np.float32)
        else:
            df = pd.read_csv(p, sep=sep)
            if csv_key_prefix:
                df.columns = [
                    c[len(csv_key_prefix):] if c.startswith(csv_key_prefix) else c
                    for c in df.columns
                ]
            feat_dict = {c: df[c].values.astype(np.float32) for c in df.columns}
    else:
        raise ValueError(
            f"Unsupported feature file extension '{suffix}'. "
            "Use .npy, .npz, .pkl, .csv, or .tsv."
        )

    # Validate required keys
    missing = [k for k in required_keys if k not in feat_dict]
    if missing:
        available = sorted(feat_dict.keys())
        raise KeyError(
            f"The following required feature(s) are MISSING from '{path}':\n"
            f"  Missing : {missing}\n"
            f"  Available: {available}\n"
            "Check the feature file or adjust --features-key accordingly."
        )

    log.info(
        "Loaded features from '%s': %d keys, sample length=%d",
        path,
        len(feat_dict),
        next(iter(feat_dict.values())).shape[0],
    )
    return feat_dict


# ---------------------------------------------------------------------------
# Helpers: sampling
# ---------------------------------------------------------------------------

def subsample(
    embeddings: np.ndarray,
    features: Dict[str, np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Randomly subsample to at most *n_samples* rows."""
    N = embeddings.shape[0]
    lengths = {k: v.shape[0] for k, v in features.items()}
    if len(set(lengths.values()) | {N}) > 1:
        raise ValueError(
            f"Row count mismatch between embeddings ({N}) and features: {lengths}."
        )
    if N <= n_samples:
        log.info("Using all %d samples (n_samples=%d).", N, n_samples)
        return embeddings, features
    idx = rng.choice(N, size=n_samples, replace=False)
    idx.sort()
    log.info("Subsampled %d → %d rows.", N, n_samples)
    return embeddings[idx], {k: v[idx] for k, v in features.items()}


# ---------------------------------------------------------------------------
# UMAP projection
# ---------------------------------------------------------------------------

def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    n_jobs: int = -1,
) -> np.ndarray:
    """Fit and transform embeddings to 2D via UMAP.

    Returns array of shape (N, 2).
    """
    try:
        import umap  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "umap-learn is required. Install it with:\n"
            "  pip install umap-learn\n"
            "or see requirements-analysis.txt."
        )

    log.info(
        "Running UMAP: n_neighbors=%d, min_dist=%.3f, metric=%s, random_state=%d",
        n_neighbors,
        min_dist,
        metric,
        random_state,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        low_memory=True,
        verbose=False,
    )
    embedding_2d = reducer.fit_transform(embeddings)
    log.info("UMAP done. Output shape: %s", embedding_2d.shape)
    return embedding_2d.astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_colormap_plot(
    ax,
    xy: np.ndarray,
    color_values: np.ndarray,
    feature_name: str,
    point_size: float = 1.0,
    alpha: float = 0.5,
    cmap: str = "viridis",
):
    """Draw a 2-D scatter on *ax* colored by *color_values*."""
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color_values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(feature_name, fontsize=9)
    ax.set_title(feature_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


def save_individual_plots(
    xy: np.ndarray,
    features: Dict[str, np.ndarray],
    feature_keys: List[str],
    output_dir: Path,
    tag: str,
    point_size: float = 1.0,
    alpha: float = 0.5,
    cmap: str = "viridis",
    fmt: str = "png",
    dpi: int = 150,
):
    """Save one PNG/PDF per feature, plus one combined overview figure."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for feat_name in feature_keys:
        if feat_name not in features:
            log.warning("Feature '%s' not loaded; skipping plot.", feat_name)
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        _make_colormap_plot(
            ax,
            xy,
            features[feat_name],
            feat_name,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
        )
        fig.tight_layout()
        fname = output_dir / f"umap_{feat_name}_{tag}.{fmt}"
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", fname)
        saved_paths.append(fname)

    # Combined overview (2×2 grid)
    n_cols = 2
    n_rows = max(1, int(np.ceil(len(feature_keys) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).ravel()

    for i, feat_name in enumerate(feature_keys):
        if i >= len(axes_flat):
            break
        if feat_name not in features:
            axes_flat[i].set_visible(False)
            continue
        _make_colormap_plot(
            axes_flat[i],
            xy,
            features[feat_name],
            feat_name,
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
        )

    for j in range(len(feature_keys), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"UMAP Driving-Style Embedding  [{tag}]", fontsize=13, y=1.01)
    fig.tight_layout()
    overview_path = output_dir / f"umap_overview_{tag}.{fmt}"
    fig.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved overview: %s", overview_path)
    saved_paths.append(overview_path)

    return saved_paths


def save_stability_comparison(
    results: Dict[int, np.ndarray],
    features: Dict[str, np.ndarray],
    feature_key: str,
    output_dir: Path,
    base_tag: str,
    point_size: float = 1.0,
    alpha: float = 0.5,
    cmap: str = "viridis",
    fmt: str = "png",
    dpi: int = 150,
):
    """Save a side-by-side comparison across random states for one feature."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if feature_key not in features:
        log.warning("Feature '%s' not in dict; skipping stability plot.", feature_key)
        return

    seeds = sorted(results.keys())
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4.5))
    if len(seeds) == 1:
        axes = [axes]

    for ax, seed in zip(axes, seeds):
        _make_colormap_plot(
            ax,
            results[seed],
            features[feature_key],
            f"{feature_key}\n(seed={seed})",
            point_size=point_size,
            alpha=alpha,
            cmap=cmap,
        )

    fig.suptitle(
        f"UMAP Stability Check – {feature_key}  [{base_tag}]",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"umap_stability_{feature_key}_{base_tag}.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved stability plot: %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "UMAP 2D visualization of driving-style embeddings.\n"
            "Generates one scatter plot per feature colored by a continuous "
            "colormap, plus an overview grid and optional stability comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input ---
    inp = p.add_argument_group("Input")
    inp.add_argument(
        "--embeddings",
        required=True,
        metavar="PATH",
        help="Path to embedding file (.npy | .npz | .pkl). Shape: (N, D).",
    )
    inp.add_argument(
        "--features",
        required=True,
        metavar="PATH",
        help=(
            "Path to handcrafted feature file "
            "(.npy | .npz | .pkl | .csv | .tsv). "
            "Must contain at minimum: rel_speed_mean, thw_mean, jerk_p95, speed_norm."
        ),
    )
    inp.add_argument(
        "--features-key",
        nargs="+",
        default=REQUIRED_FEATURES,
        metavar="KEY",
        help=(
            "Feature column/key names to color the UMAP by. "
            f"Default: {REQUIRED_FEATURES}"
        ),
    )

    # --- Sampling ---
    sam = p.add_argument_group("Sampling")
    sam.add_argument(
        "--n-samples",
        type=int,
        default=100_000,
        metavar="N",
        help="Maximum number of points to use (default: 100 000). Use -1 for all.",
    )
    sam.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="INT",
        help="Random seed for subsampling (default: 0).",
    )

    # --- UMAP parameters ---
    ump = p.add_argument_group("UMAP")
    ump.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        metavar="INT",
        help="UMAP n_neighbors (default: 30).",
    )
    ump.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help="UMAP min_dist (default: 0.1).",
    )
    ump.add_argument(
        "--metric",
        default="cosine",
        metavar="STR",
        help="UMAP distance metric (default: cosine).",
    )
    ump.add_argument(
        "--random-state",
        type=int,
        nargs="+",
        default=[42],
        metavar="INT",
        help=(
            "One or more UMAP random seeds. "
            "Providing >1 enables stability comparison. Default: 42."
        ),
    )
    ump.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        metavar="INT",
        help="Number of CPU threads for UMAP (default: -1 = all).",
    )

    # --- Output ---
    out = p.add_argument_group("Output")
    out.add_argument(
        "--output-dir",
        default="outputs/umap",
        metavar="DIR",
        help="Directory for output figures (default: outputs/umap).",
    )
    out.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output file format (default: png).",
    )
    out.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for rasterised output (default: 150).",
    )
    out.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Marker size in scatter plots (default: 1.0).",
    )
    out.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Marker alpha in scatter plots (default: 0.4).",
    )
    out.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis).",
    )
    out.add_argument(
        "--save-umap",
        action="store_true",
        help="Also save the raw 2-D UMAP coordinates as .npy files.",
    )
    out.add_argument(
        "--stability-feature",
        default=None,
        metavar="KEY",
        help=(
            "When multiple --random-state values are given, produce a "
            "side-by-side stability comparison for this feature. "
            "Default: first key in --features-key."
        ),
    )

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    embeddings = load_embeddings(args.embeddings)

    # Determine features to validate (union of required + user-specified)
    required = list(dict.fromkeys(REQUIRED_FEATURES + args.features_key))
    # Only insist on user-specified features; REQUIRED_FEATURES are also expected
    # but if the user explicitly overrode --features-key we accept that subset.
    user_keys = args.features_key

    features = load_features(args.features, required_keys=user_keys)

    # ------------------------------------------------------------------ #
    # 2. Subsample
    # ------------------------------------------------------------------ #
    n_samples = args.n_samples if args.n_samples > 0 else embeddings.shape[0]
    rng = np.random.default_rng(args.seed)
    embeddings_sub, features_sub = subsample(embeddings, features, n_samples, rng)

    # ------------------------------------------------------------------ #
    # 3. UMAP – one projection per random_state
    # ------------------------------------------------------------------ #
    umap_results: Dict[int, np.ndarray] = {}
    for rs in args.random_state:
        xy = run_umap(
            embeddings_sub,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=rs,
            n_jobs=args.n_jobs,
        )
        umap_results[rs] = xy

        if args.save_umap:
            umap_path = output_dir / f"umap_coords_seed{rs}_{timestamp}.npy"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(umap_path, xy)
            log.info("Saved UMAP coordinates: %s", umap_path)

    # ------------------------------------------------------------------ #
    # 4. Per-seed plots
    # ------------------------------------------------------------------ #
    cfg_tag = (
        f"nn{args.n_neighbors}_md{args.min_dist}_{args.metric}_{timestamp}"
    )
    for rs, xy in umap_results.items():
        tag = f"seed{rs}_{cfg_tag}"
        save_individual_plots(
            xy=xy,
            features=features_sub,
            feature_keys=user_keys,
            output_dir=output_dir,
            tag=tag,
            point_size=args.point_size,
            alpha=args.alpha,
            cmap=args.cmap,
            fmt=args.format,
            dpi=args.dpi,
        )

    # ------------------------------------------------------------------ #
    # 5. Stability comparison (multiple seeds)
    # ------------------------------------------------------------------ #
    if len(args.random_state) > 1:
        stab_feat = args.stability_feature or user_keys[0]
        log.info(
            "Generating stability comparison across %d seeds for '%s'.",
            len(args.random_state),
            stab_feat,
        )
        seeds_tag = "_".join(str(s) for s in sorted(args.random_state))
        base_tag = f"seeds{seeds_tag}_{cfg_tag}"
        for feat in user_keys:
            save_stability_comparison(
                results=umap_results,
                features=features_sub,
                feature_key=feat,
                output_dir=output_dir,
                base_tag=base_tag,
                point_size=args.point_size,
                alpha=args.alpha,
                cmap=args.cmap,
                fmt=args.format,
                dpi=args.dpi,
            )

    log.info("All done. Output directory: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
