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
) -> pd.DataFrame:
    rows = []
    n_feat = feat_train.shape[1]
    min_samples = 5

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

        if n_train_used < min_samples or n_test_used < min_samples:
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
    print(f"eval mode: {split_note}")
    print(f"Saved: {probe_path}")
    print(f"Saved: {neigh_path}")
    print(f"Saved: {analysis_dir / 'umap_coordinates.csv'}")
    print(f"Saved UMAP images in: {analysis_dir}")


if __name__ == "__main__":
    main()
