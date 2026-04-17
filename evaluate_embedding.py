import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

from dataset import _compute_cond
from loss import build_cond_mask, build_cond_knn_mask


if not hasattr(cbook, "_Stack") and hasattr(cbook, "Stack"):
    cbook._Stack = cbook.Stack


FEATURE_NAME_MAP = {
    0: "rel_speed_mean",
    3: "thw_mean",
    8: "jerk_p95",
    15: "speed_norm",
}


def resolve_feature_name(feature_names: list[str] | None, idx: int) -> str:
    """Resolve feature name by priority: loaded names -> legacy map -> feature_{idx}."""
    if feature_names is not None and 0 <= idx < len(feature_names):
        return feature_names[idx]
    return FEATURE_NAME_MAP.get(idx, f"feature_{idx}")


def replace_unsafe_chars(name: str) -> str:
    """Keep [a-zA-Z0-9_.-] and replace all other characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"

    parser = argparse.ArgumentParser(description="Evaluate learned driving behavior embedding")
    parser.add_argument("--embeddings_path", type=str, default=str(out_dir / "embeddings_all.npy"))
    parser.add_argument("--feat_path", type=str, default=str(out_dir / "feat.npy"))
    parser.add_argument("--feat_raw_path", type=str, default=None, help="Optional raw feature path with NaN/inf values")
    parser.add_argument("--feature_names_path", type=str, default=str(out_dir / "feature_names_style.json"))
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="test",
        help="When split_path is provided, evaluate this split. 'all' uses all rows.",
    )
    parser.add_argument("--analysis_dir", type=str, default=str(out_dir / "analysis"))

    # Condition-aware evaluation inputs.
    parser.add_argument("--traj_path", type=str, default=str(out_dir / "traj.npy"), help="Trajectory file for condition vector (requires --front_path)")
    parser.add_argument("--front_path", type=str, default=None, help="Front vehicle trajectory file for condition-aware evaluation")

    # Condition gating options (should match training settings for consistent evaluation).
    parser.add_argument(
        "--cond_mode",
        type=str,
        choices=["off", "hard_box", "knn"],
        default="off",
        help="Condition gating mode for neighbor consistency evaluation",
    )
    parser.add_argument("--cond_speed_tol", type=float, default=2.0, help="Speed tolerance for condition gating (m/s)")
    parser.add_argument("--cond_dist_tol", type=float, default=5.0, help="Distance tolerance for condition gating (m)")
    parser.add_argument("--cond_vrel_tol", type=float, default=1.0, help="Relative speed tolerance for condition gating (m/s)")
    parser.add_argument(
        "--cond_cf_bucket_edges",
        type=str,
        default="0.2,0.6",
        help="Comma-separated edges for cf_valid_frac bucketing (e.g. '0.2,0.6')",
    )
    parser.add_argument(
        "--min_cond_candidates",
        type=int,
        default=8,
        help="Minimum condition candidates for within-condition neighbor evaluation",
    )
    parser.add_argument("--cond_k", type=int, default=24, help="Number of nearest compatible samples per anchor for cond_mode=knn")
    parser.add_argument(
        "--cond_scale_mode",
        type=str,
        choices=["mad", "iqr", "std"],
        default="mad",
        help="Robust scale estimator for cond_mode=knn distance normalization",
    )

    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_max_points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--feature_std_eps", type=float, default=1e-6)
    parser.add_argument("--neighbor_denominator_eps", type=float, default=1e-3)
    parser.add_argument("--neighbor_clip_quantile", type=float, default=0.99)
    parser.add_argument("--probe_test_ratio", type=float, default=0.2, help="Used only when split_path is not provided")
    parser.add_argument("--probe_min_samples", type=int, default=5, help="Minimum masked samples per split for probe fitting/eval")
    parser.add_argument("--kmeans_clusters", type=int, default=0, help="If >0, save UMAP plot colored by KMeans labels")
    parser.add_argument("--plot_first_k", type=int, default=12, help="UMAP feature-colored plots use the first K feature dims")
    parser.add_argument("--nan_policy", type=str, choices=["ignore", "zero"], default="ignore")
    parser.add_argument("--umap_color_source", type=str, choices=["feat", "raw"], default="feat")
    return parser.parse_args()


def to_split_mask(split: np.ndarray, key: str) -> np.ndarray:
    if split.dtype.kind in ("U", "S", "O"):
        return split.astype(str) == key

    key_map = {"train": 0, "val": 1, "test": 2}
    if key not in key_map:
        raise ValueError(f"Unknown split key: {key}")
    return split == key_map[key]


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)

    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata_average_ties(x)
    yr = rankdata_average_ties(y)
    xz = xr - xr.mean()
    yz = yr - yr.mean()
    denom = (np.sqrt((xz * xz).sum()) * np.sqrt((yz * yz).sum())) + 1e-12
    return float((xz * yz).sum() / denom)


def evaluate_probe(
    emb_train: np.ndarray,
    feat_train: np.ndarray,
    emb_test: np.ndarray,
    feat_test: np.ndarray,
    feat_raw_train: np.ndarray | None,
    feat_raw_test: np.ndarray | None,
    nan_policy: str,
    feature_names: list[str] | None,
    ridge_alpha: float,
    feature_std_eps: float,
    probe_min_samples: int,
) -> pd.DataFrame:
    rows = []
    n_feat = feat_train.shape[1]

    for f_idx in range(n_feat):
        if feat_raw_train is not None and feat_raw_test is not None and nan_policy == "ignore":
            mask_train = np.isfinite(feat_raw_train[:, f_idx])
            mask_test = np.isfinite(feat_raw_test[:, f_idx])
            y_train = feat_train[mask_train, f_idx]
            y_test = feat_test[mask_test, f_idx]
            x_train = emb_train[mask_train]
            x_test = emb_test[mask_test]
        else:
            y_train = feat_train[:, f_idx]
            y_test = feat_test[:, f_idx]
            x_train = emb_train
            x_test = emb_test

        n_train_used = int(len(y_train))
        n_test_used = int(len(y_test))

        if n_train_used < probe_min_samples or n_test_used < probe_min_samples:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": resolve_feature_name(feature_names, f_idx),
                    "spearman_linear": np.nan,
                    "r2_linear": np.nan,
                    "rmse_linear": np.nan,
                    "spearman_ridge": np.nan,
                    "r2_ridge": np.nan,
                    "rmse_ridge": np.nan,
                    "n_train_used": n_train_used,
                    "n_test_used": n_test_used,
                    "note": f"skipped_small_n(n_train={n_train_used}, n_test={n_test_used})",
                }
            )
            continue

        train_std = float(np.std(y_train))
        test_std = float(np.std(y_test))
        if train_std < feature_std_eps or test_std < feature_std_eps:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": resolve_feature_name(feature_names, f_idx),
                    "spearman_linear": np.nan,
                    "r2_linear": np.nan,
                    "rmse_linear": np.nan,
                    "spearman_ridge": np.nan,
                    "r2_ridge": np.nan,
                    "rmse_ridge": np.nan,
                    "n_train_used": n_train_used,
                    "n_test_used": n_test_used,
                    "note": f"skipped_low_variance(train_std={train_std:.3e}, test_std={test_std:.3e})",
                }
            )
            continue

        lin = LinearRegression()
        lin.fit(x_train, y_train)
        pred_lin = lin.predict(x_test)

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(x_train, y_train)
        pred_ridge = ridge.predict(x_test)

        row = {
            "feature_idx": f_idx,
            "feature_name": resolve_feature_name(feature_names, f_idx),
            "spearman_linear": spearman_corr(y_test, pred_lin),
            "r2_linear": float(r2_score(y_test, pred_lin)),
            "rmse_linear": float(np.sqrt(mean_squared_error(y_test, pred_lin))),
            "spearman_ridge": spearman_corr(y_test, pred_ridge),
            "r2_ridge": float(r2_score(y_test, pred_ridge)),
            "rmse_ridge": float(np.sqrt(mean_squared_error(y_test, pred_ridge))),
            "n_train_used": n_train_used,
            "n_test_used": n_test_used,
            "note": "ok",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_neighbor_consistency(
    emb_test: np.ndarray,
    feat_test: np.ndarray,
    feat_raw_test: np.ndarray | None,
    k: int,
    seed: int,
    feature_std_eps: float,
    denominator_eps: float,
    clip_quantile: float,
    nan_policy: str,
    feature_names: list[str] | None,
) -> pd.DataFrame:
    n = len(emb_test)
    if n <= k + 1:
        raise ValueError(f"Need more test samples than k+1, got n={n}, k={k}")

    rng = np.random.default_rng(seed)

    rows = []
    for f_idx in range(feat_test.shape[1]):
        if feat_raw_test is not None and nan_policy == "ignore":
            mask = np.isfinite(feat_raw_test[:, f_idx])
            emb_sub = emb_test[mask]
            f = feat_test[mask, f_idx]
        else:
            emb_sub = emb_test
            f = feat_test[:, f_idx]

        n_sub = int(len(f))
        if n_sub <= k + 1:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": resolve_feature_name(feature_names, f_idx),
                    "ratio_mean": np.nan,
                    "ratio_median": np.nan,
                    "n_eval_used": n_sub,
                    "note": f"skipped_small_n(n={n_sub}, k={k})",
                }
            )
            continue

        f_std = float(np.std(f))
        if f_std < feature_std_eps:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": resolve_feature_name(feature_names, f_idx),
                    "ratio_mean": np.nan,
                    "ratio_median": np.nan,
                    "n_eval_used": n_sub,
                    "note": f"skipped_low_variance(std={f_std:.3e})",
                }
            )
            continue

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(emb_sub)
        neigh_idx = nn.kneighbors(emb_sub, return_distance=False)[:, 1:]  # remove self

        embed_dists = np.abs(f[:, None] - f[neigh_idx]).mean(axis=1)

        # random neighbors per sample without replacement and excluding self
        rand_dists = np.zeros(n_sub, dtype=np.float64)
        all_idx = np.arange(n_sub)
        for i in range(n_sub):
            candidates = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
            ridx = rng.choice(candidates, size=k, replace=False)
            rand_dists[i] = np.abs(f[i] - f[ridx]).mean()

        denom = np.maximum(rand_dists, denominator_eps)
        ratio = embed_dists / denom
        if 0.0 < clip_quantile < 1.0:
            hi = float(np.quantile(ratio, clip_quantile))
            ratio = np.clip(ratio, 0.0, hi)

        rows.append(
            {
                "feature_idx": f_idx,
                "feature_name": resolve_feature_name(feature_names, f_idx),
                "embed_dist_mean": float(np.mean(embed_dists)),
                "random_dist_mean": float(np.mean(rand_dists)),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_median": float(np.median(ratio)),
                "n_eval_used": n_sub,
                "note": "ok",
            }
        )

    return pd.DataFrame(rows)


def _load_cond_for_eval(
    traj_path: str,
    front_path: str,
    split: np.ndarray | None,
    eval_split: str,
    feat_raw: np.ndarray | None,
    feat: np.ndarray,
    eval_idx: np.ndarray,
) -> np.ndarray | None:
    """Load traj/front, compute condition vectors, and return the eval-split subset."""
    try:
        traj_loaded = np.load(traj_path, allow_pickle=True)
        traj = [np.asarray(t, dtype=np.float32) for t in traj_loaded]
        front_loaded = np.load(front_path, allow_pickle=True)
        front = [np.asarray(f, dtype=np.float32) for f in front_loaded]
    except Exception as exc:
        print(f"[WARN] Could not load traj/front for condition-aware eval: {exc}")
        return None

    if len(traj) != len(front):
        print(f"[WARN] traj length ({len(traj)}) != front length ({len(front)}); skipping condition eval")
        return None

    # Build cf_col from feat_raw if available.
    if feat_raw is not None and feat_raw.shape[1] > 10:
        cf_col = feat_raw[:, 10].copy()
        cf_col = np.where(np.isfinite(cf_col), cf_col, 0.0).astype(np.float32)
    else:
        cf_col = feat[:, 10].copy() if feat.shape[1] > 10 else None

    cond_all = _compute_cond(traj, front, cf_col)  # (N, 4)
    return cond_all[eval_idx]  # return eval-split subset


def evaluate_neighbor_consistency_cond(
    emb_test: np.ndarray,
    feat_test: np.ndarray,
    feat_raw_test: np.ndarray | None,
    cond_eval: np.ndarray,
    k: int,
    seed: int,
    feature_std_eps: float,
    denominator_eps: float,
    clip_quantile: float,
    nan_policy: str,
    feature_names: list[str] | None,
    cond_mode: str,
    cond_speed_tol: float,
    cond_dist_tol: float,
    cond_vrel_tol: float,
    cf_bucket_edges: list[float],
    min_cond_candidates: int,
    cond_k: int = 24,
    cond_scale_mode: str = "mad",
) -> pd.DataFrame:
    """Neighbor consistency evaluated within condition-compatible sets only.

    For each anchor, nearest neighbors are found globally in embedding space but
    the random baseline is drawn only from condition-compatible candidates.
    This makes the ratio meaningful within comparable operating conditions.

    Supports cond_mode in {"hard_box", "knn"}:
      - hard_box: uses absolute-tolerance box filter; falls back when < min_cond_candidates.
      - knn: uses robust-scale kNN mask; fallback only for zero-compatible anchors.
    """
    import torch

    n = len(emb_test)
    rng = np.random.default_rng(seed)

    # Build full condition mask [N, N].
    cond_t = torch.as_tensor(cond_eval, dtype=torch.float32)

    if cond_mode == "knn":
        cond_gate_tensor, fallback_rows_global = build_cond_knn_mask(
            cond_t,
            cond_k=cond_k,
            cond_scale_mode=cond_scale_mode,
            cf_bucket_edges=cf_bucket_edges,
        )
        cond_gate = cond_gate_tensor.numpy()  # [N, N] bool, self excluded
        fallback_arr_global: np.ndarray | None = fallback_rows_global.numpy()
    else:  # hard_box
        cond_gate_raw = build_cond_mask(
            cond_t,
            speed_tol=cond_speed_tol,
            dist_tol=cond_dist_tol,
            vrel_tol=cond_vrel_tol,
            cf_bucket_edges=cf_bucket_edges,
        )
        eye = torch.eye(n, dtype=torch.bool)
        cond_gate = (cond_gate_raw & ~eye).numpy()  # [N, N] bool, exclude self
        fallback_arr_global = None  # tracked per-feature loop for hard_box

    rows = []
    for f_idx in range(feat_test.shape[1]):
        if feat_raw_test is not None and nan_policy == "ignore":
            mask = np.isfinite(feat_raw_test[:, f_idx])
            emb_sub = emb_test[mask]
            f = feat_test[mask, f_idx]
            cond_sub = cond_gate[np.ix_(mask, mask)]
            fallback_sub = fallback_arr_global[mask] if fallback_arr_global is not None else None
        else:
            emb_sub = emb_test
            f = feat_test[:, f_idx]
            cond_sub = cond_gate
            fallback_sub = fallback_arr_global

        n_sub = int(len(f))
        if n_sub <= k + 1:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": resolve_feature_name(feature_names, f_idx),
                    "ratio_mean": np.nan,
                    "ratio_median": np.nan,
                    "mean_cond_candidates": np.nan,
                    "frac_fallback": np.nan,
                    "n_eval_used": n_sub,
                    "note": f"skipped_small_n(n={n_sub}, k={k})",
                }
            )
            continue

        f_std = float(np.std(f))
        if f_std < feature_std_eps:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": resolve_feature_name(feature_names, f_idx),
                    "ratio_mean": np.nan,
                    "ratio_median": np.nan,
                    "mean_cond_candidates": np.nan,
                    "frac_fallback": np.nan,
                    "n_eval_used": n_sub,
                    "note": f"skipped_low_variance(std={f_std:.3e})",
                }
            )
            continue

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(emb_sub)
        neigh_idx = nn.kneighbors(emb_sub, return_distance=False)[:, 1:]  # remove self

        embed_dists = np.abs(f[:, None] - f[neigh_idx]).mean(axis=1)

        # Random baseline: draw from condition-compatible candidates only.
        rand_dists = np.zeros(n_sub, dtype=np.float64)
        cand_counts = np.zeros(n_sub, dtype=np.int64)
        all_idx = np.arange(n_sub)

        if cond_mode == "knn":
            # For knn mode, cond_sub already encodes the correct candidate set
            # (cond_k nearest, or all compatible if fewer, or all non-self for fallback rows).
            # frac_fallback comes from fallback_sub (zero-compatible anchors).
            fallback_count = int(fallback_sub.sum()) if fallback_sub is not None else 0
            for i in range(n_sub):
                compat = np.where(cond_sub[i])[0]
                cand_counts[i] = len(compat)
                # build_cond_knn_mask guarantees >=1 candidate per anchor (last-resort
                # fallback expands to all non-self for zero-compatible anchors), so
                # len(compat) >= 1 is always true here for knn mode.
                if len(compat) == 0:
                    # Defensive: should not happen; fall back to all non-self.
                    compat = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
                ridx = rng.choice(compat, size=min(k, len(compat)), replace=False)
                rand_dists[i] = np.abs(f[i] - f[ridx]).mean()
        else:  # hard_box
            fallback_count = 0
            for i in range(n_sub):
                compat = np.where(cond_sub[i])[0]
                cand_counts[i] = len(compat)
                if len(compat) >= min_cond_candidates:
                    ridx = rng.choice(compat, size=min(k, len(compat)), replace=False)
                else:
                    # Fallback: use all non-self samples.
                    fallback_count += 1
                    candidates = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
                    ridx = rng.choice(candidates, size=k, replace=False)
                rand_dists[i] = np.abs(f[i] - f[ridx]).mean()

        denom = np.maximum(rand_dists, denominator_eps)
        ratio = embed_dists / denom
        if 0.0 < clip_quantile < 1.0:
            hi = float(np.quantile(ratio, clip_quantile))
            ratio = np.clip(ratio, 0.0, hi)

        rows.append(
            {
                "feature_idx": f_idx,
                "feature_name": resolve_feature_name(feature_names, f_idx),
                "embed_dist_mean": float(np.mean(embed_dists)),
                "random_dist_mean": float(np.mean(rand_dists)),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_median": float(np.median(ratio)),
                "mean_cond_candidates": float(np.mean(cand_counts)),
                "frac_fallback": float(fallback_count) / max(1, n_sub),
                "n_eval_used": n_sub,
                "note": "ok",
            }
        )

    return pd.DataFrame(rows)


def save_umap_plots(
    emb_eval: np.ndarray,
    feat_eval: np.ndarray,
    feat_raw_eval: np.ndarray | None,
    analysis_dir: Path,
    n_neighbors: int,
    min_dist: float,
    max_points: int,
    seed: int,
    kmeans_clusters: int,
    feature_names: list[str] | None,
    plot_first_k: int,
    umap_color_source: str,
) -> None:
    rng = np.random.default_rng(seed)
    n = len(emb_eval)

    if n > max_points:
        idx = np.sort(rng.choice(np.arange(n), size=max_points, replace=False))
        emb_plot = emb_eval[idx]
        feat_plot = feat_eval[idx]
        feat_raw_plot = feat_raw_eval[idx] if feat_raw_eval is not None else None
    else:
        idx = np.arange(n)
        emb_plot = emb_eval
        feat_plot = feat_eval
        feat_raw_plot = feat_raw_eval if feat_raw_eval is not None else None

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="cosine",
    )
    xy = reducer.fit_transform(emb_plot)

    umap_df = pd.DataFrame(
        {
            "row_index": idx,
            "umap_x": xy[:, 0],
            "umap_y": xy[:, 1],
        }
    )
    umap_df.to_csv(analysis_dir / "umap_coordinates.csv", index=False)

    # Base
    plt.figure(figsize=(8, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.7)
    plt.title("UMAP of Test Embeddings")
    plt.tight_layout()
    plt.savefig(analysis_dir / "umap_base.png", dpi=180)
    plt.close()

    # Feature-colored plots
    for f_idx in range(min(plot_first_k, feat_plot.shape[1])):
        name = resolve_feature_name(feature_names, f_idx)
        safe_name = replace_unsafe_chars(name)

        color_vals = feat_plot[:, f_idx]
        title_suffix = ""
        color_label = name
        if umap_color_source == "raw" and feat_raw_plot is not None:
            raw_vals = feat_raw_plot[:, f_idx].astype(float)
            finite_mask = np.isfinite(raw_vals)
            if finite_mask.any():
                fill_val = float(np.min(raw_vals[finite_mask]))
            else:
                fill_val = 0.0
            color_vals = raw_vals.copy()
            if (~finite_mask).any():
                color_vals[~finite_mask] = fill_val
                title_suffix = " (raw, NaN/inf->min)"
            color_label = f"{name} (raw)"

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=color_vals, s=3, alpha=0.8, cmap="viridis")
        plt.colorbar(sc, label=color_label)
        plt.title(f"UMAP colored by {name}{title_suffix}")
        plt.tight_layout()
        plt.savefig(analysis_dir / f"umap_feat_{safe_name}.png", dpi=180)
        plt.close()

    if kmeans_clusters > 0 and len(emb_plot) >= kmeans_clusters:
        km = KMeans(n_clusters=kmeans_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(emb_plot)
        plt.figure(figsize=(8, 6))
        plt.scatter(xy[:, 0], xy[:, 1], c=labels, s=3, alpha=0.8, cmap="tab20")
        plt.title(f"UMAP colored by KMeans labels (k={kmeans_clusters})")
        plt.tight_layout()
        plt.savefig(analysis_dir / "umap_kmeans_labels.png", dpi=180)
        plt.close()


def load_feature_names(feature_names_path: str | None) -> list[str] | None:
    if not feature_names_path:
        return None
    path = Path(feature_names_path)
    if not path.exists():
        print(f"[WARN] feature names file not found: {path}, fallback to legacy naming")
        return None
    with path.open("r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        print(f"[WARN] invalid feature names format in {path}, fallback to legacy naming")
        return None
    return names


def _load_eval_views(
    emb: np.ndarray,
    feat: np.ndarray,
    split: np.ndarray | None,
    eval_split: str,
    probe_test_ratio: float,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, str]:
    if split is not None:
        if len(feat) != len(split):
            raise ValueError(f"feat rows ({len(feat)}) != split rows ({len(split)})")

        if eval_split == "all":
            eval_mask = np.ones(len(split), dtype=bool)
        else:
            eval_mask = to_split_mask(split, eval_split)

        train_mask = to_split_mask(split, "train")

        if len(emb) == len(feat):
            emb_train = emb[train_mask]
            emb_eval = emb[eval_mask]
        elif len(emb) == int(eval_mask.sum()):
            emb_eval = emb
            emb_train = None
        else:
            raise ValueError(
                f"embeddings rows ({len(emb)}) are neither full-N ({len(feat)}) nor eval-split-only ({int(eval_mask.sum())})"
            )

        feat_train = feat[train_mask]
        feat_eval = feat[eval_mask]
        train_idx = np.flatnonzero(train_mask)
        eval_idx = np.flatnonzero(eval_mask)
        note = f"split_mode(eval_split={eval_split})"
        return emb_train, emb_eval, feat_train, feat_eval, train_idx, eval_idx, note

    if len(emb) != len(feat):
        raise ValueError(
            "split_path is not provided, so embeddings_path must align row-wise with feat_path (same number of rows)"
        )

    n = len(emb)
    if n < 5:
        raise ValueError(f"Too few samples for evaluation without split: n={n}")

    test_n = max(1, int(round(n * probe_test_ratio)))
    test_n = min(test_n, n - 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = np.sort(perm[:test_n])
    train_idx = np.sort(perm[test_n:])

    emb_train = emb[train_idx]
    feat_train = feat[train_idx]
    emb_eval = emb[test_idx]
    feat_eval = feat[test_idx]
    note = f"random_holdout(test_ratio={probe_test_ratio:.2f})"
    return emb_train, emb_eval, feat_train, feat_eval, train_idx, test_idx, note


def main() -> None:
    args = parse_args()

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embeddings_path, allow_pickle=False)
    feat = np.load(args.feat_path, allow_pickle=False)
    feat_raw = np.load(args.feat_raw_path, allow_pickle=False) if args.feat_raw_path else None
    split = np.load(args.split_path, allow_pickle=True) if args.split_path else None
    feature_names = load_feature_names(args.feature_names_path)

    if feat_raw is not None and feat_raw.shape != feat.shape:
        raise ValueError(f"feat_raw shape {feat_raw.shape} must equal feat shape {feat.shape}")
    if args.umap_color_source == "raw" and feat_raw is None:
        print("[WARN] --umap_color_source raw requested but --feat_raw_path is not provided; fallback to feat")

    # Parse cf bucket edges.
    cf_bucket_edges: list[float] = []
    if args.cond_cf_bucket_edges:
        try:
            cf_bucket_edges = [float(x.strip()) for x in args.cond_cf_bucket_edges.split(",") if x.strip()]
        except ValueError as e:
            print(f"[WARN] Invalid --cond_cf_bucket_edges '{args.cond_cf_bucket_edges}': {e}; using empty")

    emb_train, emb_eval, feat_train, feat_eval, train_idx, eval_idx, split_note = _load_eval_views(
        emb=emb,
        feat=feat,
        split=split,
        eval_split=args.eval_split,
        probe_test_ratio=args.probe_test_ratio,
        seed=args.seed,
    )
    feat_raw_train = feat_raw[train_idx] if feat_raw is not None else None
    feat_raw_eval = feat_raw[eval_idx] if feat_raw is not None else None

    # Task 1: probe (requires train embeddings)
    if emb_train is not None:
        probe_df = evaluate_probe(
            emb_train=emb_train,
            feat_train=feat_train,
            emb_test=emb_eval,
            feat_test=feat_eval,
            feat_raw_train=feat_raw_train,
            feat_raw_test=feat_raw_eval,
            nan_policy=args.nan_policy,
            feature_names=feature_names,
            ridge_alpha=args.ridge_alpha,
            feature_std_eps=args.feature_std_eps,
            probe_min_samples=args.probe_min_samples,
        )
    else:
        probe_df = pd.DataFrame(
            [
                {
                    "feature_idx": -1,
                    "feature_name": "N/A",
                    "spearman_linear": np.nan,
                    "r2_linear": np.nan,
                    "spearman_ridge": np.nan,
                    "r2_ridge": np.nan,
                    "rmse_linear": np.nan,
                    "rmse_ridge": np.nan,
                    "n_train_used": 0,
                    "n_test_used": int(len(emb_eval)),
                    "note": f"Skipped probe: no train embeddings available ({split_note})",
                }
            ]
        )

    probe_path = analysis_dir / "probe_results.csv"
    probe_df.to_csv(probe_path, index=False)

    # Task 2: neighbor consistency (test only)
    neigh_df = evaluate_neighbor_consistency(
        emb_test=emb_eval,
        feat_test=feat_eval,
        feat_raw_test=feat_raw_eval,
        k=args.k_neighbors,
        seed=args.seed,
        feature_std_eps=args.feature_std_eps,
        denominator_eps=args.neighbor_denominator_eps,
        clip_quantile=args.neighbor_clip_quantile,
        nan_policy=args.nan_policy,
        feature_names=feature_names,
    )
    neigh_path = analysis_dir / "neighbor_results.csv"
    neigh_df.to_csv(neigh_path, index=False)

    # Task 2b: condition-aware neighbor consistency (optional, requires --front_path).
    neigh_cond_path: Path | None = None
    cond_eval: np.ndarray | None = None
    if args.cond_mode != "off" and args.front_path is not None:
        cond_eval = _load_cond_for_eval(
            traj_path=args.traj_path,
            front_path=args.front_path,
            split=split,
            eval_split=args.eval_split,
            feat_raw=feat_raw,
            feat=feat,
            eval_idx=eval_idx,
        )
        if cond_eval is not None:
            neigh_cond_df = evaluate_neighbor_consistency_cond(
                emb_test=emb_eval,
                feat_test=feat_eval,
                feat_raw_test=feat_raw_eval,
                cond_eval=cond_eval,
                k=args.k_neighbors,
                seed=args.seed,
                feature_std_eps=args.feature_std_eps,
                denominator_eps=args.neighbor_denominator_eps,
                clip_quantile=args.neighbor_clip_quantile,
                nan_policy=args.nan_policy,
                feature_names=feature_names,
                cond_mode=args.cond_mode,
                cond_speed_tol=args.cond_speed_tol,
                cond_dist_tol=args.cond_dist_tol,
                cond_vrel_tol=args.cond_vrel_tol,
                cf_bucket_edges=cf_bucket_edges,
                min_cond_candidates=args.min_cond_candidates,
                cond_k=args.cond_k,
                cond_scale_mode=args.cond_scale_mode,
            )
            neigh_cond_path = analysis_dir / "neighbor_results_cond.csv"
            neigh_cond_df.to_csv(neigh_cond_path, index=False)
    elif args.cond_mode != "off" and args.front_path is None:
        print("[WARN] --cond_mode is not 'off' but --front_path was not provided; skipping condition-aware neighbor eval.")

    # Task 3: UMAP (evaluation split)
    save_umap_plots(
        emb_eval=emb_eval,
        feat_eval=feat_eval,
        feat_raw_eval=feat_raw_eval,
        analysis_dir=analysis_dir,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        max_points=args.umap_max_points,
        seed=args.seed,
        kmeans_clusters=args.kmeans_clusters,
        feature_names=feature_names,
        plot_first_k=args.plot_first_k,
        umap_color_source=args.umap_color_source,
    )

    # Summary
    if emb_train is not None and len(probe_df) > 0:
        best_spearman = float(np.nanmax(probe_df["spearman_ridge"].to_numpy()))
        mean_spearman = float(np.nanmean(probe_df["spearman_ridge"].to_numpy()))
    else:
        best_spearman = float("nan")
        mean_spearman = float("nan")

    mean_neighbor_ratio = float(np.nanmean(neigh_df["ratio_mean"].to_numpy()))

    print("Evaluation summary:")
    print(f"best Spearman (ridge): {best_spearman:.4f}")
    print(f"mean Spearman (ridge): {mean_spearman:.4f}")
    print(f"mean neighbor ratio: {mean_neighbor_ratio:.4f}")
    if neigh_cond_path is not None:
        mean_cond_ratio = float(np.nanmean(neigh_cond_df["ratio_mean"].to_numpy()))
        mean_cond_cands = float(np.nanmean(neigh_cond_df["mean_cond_candidates"].to_numpy()))
        print(f"mean neighbor ratio (cond-aware): {mean_cond_ratio:.4f}")
        print(f"mean cond candidates: {mean_cond_cands:.1f}")
    print(f"eval mode: {split_note}")
    print(f"Saved: {probe_path}")
    print(f"Saved: {neigh_path}")
    if neigh_cond_path is not None:
        print(f"Saved: {neigh_cond_path}")
    print(f"Saved: {analysis_dir / 'umap_coordinates.csv'}")
    print(f"Saved UMAP images in: {analysis_dir}")


if __name__ == "__main__":
    main()
