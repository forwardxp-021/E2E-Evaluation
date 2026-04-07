import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors


if not hasattr(cbook, "_Stack") and hasattr(cbook, "Stack"):
    cbook._Stack = cbook.Stack


FEATURE_NAME_MAP = {
    0: "rel_speed_mean",
    3: "thw_mean",
    8: "jerk_p95",
    15: "speed_norm",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"

    parser = argparse.ArgumentParser(description="Evaluate learned driving behavior embedding")
    parser.add_argument("--embeddings_path", type=str, default=str(out_dir / "embeddings_test.npy"))
    parser.add_argument("--feat_path", type=str, default=str(out_dir / "feat.npy"))
    parser.add_argument("--split_path", type=str, default=str(out_dir / "split.npy"))
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
    ridge_alpha: float,
    feature_std_eps: float,
) -> pd.DataFrame:
    rows = []
    n_feat = feat_train.shape[1]

    for f_idx in range(n_feat):
        y_train = feat_train[:, f_idx]
        y_test = feat_test[:, f_idx]

        train_std = float(np.std(y_train))
        test_std = float(np.std(y_test))
        if train_std < feature_std_eps or test_std < feature_std_eps:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": FEATURE_NAME_MAP.get(f_idx, f"feature_{f_idx}"),
                    "spearman_linear": np.nan,
                    "r2_linear": np.nan,
                    "spearman_ridge": np.nan,
                    "r2_ridge": np.nan,
                    "note": f"skipped_low_variance(train_std={train_std:.3e}, test_std={test_std:.3e})",
                }
            )
            continue

        lin = LinearRegression()
        lin.fit(emb_train, y_train)
        pred_lin = lin.predict(emb_test)

        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(emb_train, y_train)
        pred_ridge = ridge.predict(emb_test)

        row = {
            "feature_idx": f_idx,
            "feature_name": FEATURE_NAME_MAP.get(f_idx, f"feature_{f_idx}"),
            "spearman_linear": spearman_corr(y_test, pred_lin),
            "r2_linear": float(r2_score(y_test, pred_lin)),
            "spearman_ridge": spearman_corr(y_test, pred_ridge),
            "r2_ridge": float(r2_score(y_test, pred_ridge)),
            "note": "ok",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_neighbor_consistency(
    emb_test: np.ndarray,
    feat_test: np.ndarray,
    k: int,
    seed: int,
    feature_std_eps: float,
    denominator_eps: float,
    clip_quantile: float,
) -> pd.DataFrame:
    n = len(emb_test)
    if n <= k + 1:
        raise ValueError(f"Need more test samples than k+1, got n={n}, k={k}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(emb_test)
    neigh_idx = nn.kneighbors(emb_test, return_distance=False)[:, 1:]  # remove self

    rng = np.random.default_rng(seed)

    rows = []
    for f_idx in range(feat_test.shape[1]):
        f = feat_test[:, f_idx]

        f_std = float(np.std(f))
        if f_std < feature_std_eps:
            rows.append(
                {
                    "feature_idx": f_idx,
                    "feature_name": FEATURE_NAME_MAP.get(f_idx, f"feature_{f_idx}"),
                    "ratio_mean": np.nan,
                    "ratio_median": np.nan,
                    "note": f"skipped_low_variance(std={f_std:.3e})",
                }
            )
            continue

        embed_dists = np.abs(f[:, None] - f[neigh_idx]).mean(axis=1)

        # random neighbors per sample without replacement and excluding self
        rand_dists = np.zeros(n, dtype=np.float64)
        all_idx = np.arange(n)
        for i in range(n):
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
                "feature_name": FEATURE_NAME_MAP.get(f_idx, f"feature_{f_idx}"),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_median": float(np.median(ratio)),
                "note": "ok",
            }
        )

    return pd.DataFrame(rows)


def save_umap_plots(
    emb_test: np.ndarray,
    feat_test: np.ndarray,
    analysis_dir: Path,
    n_neighbors: int,
    min_dist: float,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = len(emb_test)

    if n > max_points:
        idx = np.sort(rng.choice(np.arange(n), size=max_points, replace=False))
        emb_plot = emb_test[idx]
        feat_plot = feat_test[idx]
    else:
        emb_plot = emb_test
        feat_plot = feat_test

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="cosine",
    )
    xy = reducer.fit_transform(emb_plot)

    # Base
    plt.figure(figsize=(8, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.7)
    plt.title("UMAP of Test Embeddings")
    plt.tight_layout()
    plt.savefig(analysis_dir / "umap_base.png", dpi=180)
    plt.close()

    # Feature-colored plots
    for f_idx in [0, 3, 8, 15]:
        if f_idx >= feat_plot.shape[1]:
            continue

        name = FEATURE_NAME_MAP.get(f_idx, f"feature_{f_idx}")
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(xy[:, 0], xy[:, 1], c=feat_plot[:, f_idx], s=3, alpha=0.8, cmap="viridis")
        plt.colorbar(sc, label=name)
        plt.title(f"UMAP colored by {name}")
        plt.tight_layout()
        plt.savefig(analysis_dir / f"umap_feat_{name}.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.embeddings_path, allow_pickle=False)
    feat = np.load(args.feat_path, allow_pickle=False)
    split = np.load(args.split_path, allow_pickle=True)

    if len(feat) != len(split):
        raise ValueError(f"feat rows ({len(feat)}) != split rows ({len(split)})")

    train_mask = to_split_mask(split, "train")
    test_mask = to_split_mask(split, "test")

    # embeddings may be full-N or test-only
    if len(emb) == len(feat):
        emb_train = emb[train_mask]
        emb_test = emb[test_mask]
    elif len(emb) == int(test_mask.sum()):
        emb_test = emb
        emb_train = None
    else:
        raise ValueError(
            f"embeddings rows ({len(emb)}) are neither full-N ({len(feat)}) nor test-only ({int(test_mask.sum())})"
        )

    feat_train = feat[train_mask]
    feat_test = feat[test_mask]

    # Task 1: probe (requires train embeddings)
    if emb_train is not None:
        probe_df = evaluate_probe(
            emb_train=emb_train,
            feat_train=feat_train,
            emb_test=emb_test,
            feat_test=feat_test,
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
                    "note": "Skipped probe: embeddings file contains test-only rows, no train embeddings available",
                }
            ]
        )

    probe_path = analysis_dir / "probe_results.csv"
    probe_df.to_csv(probe_path, index=False)

    # Task 2: neighbor consistency (test only)
    neigh_df = evaluate_neighbor_consistency(
        emb_test=emb_test,
        feat_test=feat_test,
        k=args.k_neighbors,
        seed=args.seed,
        feature_std_eps=args.feature_std_eps,
        denominator_eps=args.neighbor_denominator_eps,
        clip_quantile=args.neighbor_clip_quantile,
    )
    neigh_path = analysis_dir / "neighbor_results.csv"
    neigh_df.to_csv(neigh_path, index=False)

    # Task 3: UMAP (test only)
    save_umap_plots(
        emb_test=emb_test,
        feat_test=feat_test,
        analysis_dir=analysis_dir,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        max_points=args.umap_max_points,
        seed=args.seed,
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
    print(f"Saved: {probe_path}")
    print(f"Saved: {neigh_path}")
    print(f"Saved UMAP images in: {analysis_dir}")


if __name__ == "__main__":
    main()
