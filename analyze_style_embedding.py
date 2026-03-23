import os
import json
import math
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
# 使用非交互式后端避免GUI错误
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_features(feature_path: str) -> pd.DataFrame:
    """
    支持:
    - csv
    - npy (要求是 structured array 或普通数组; 普通数组会自动命名 feature_0 ...)
    """
    if feature_path.endswith(".csv"):
        df = pd.read_csv(feature_path)
    elif feature_path.endswith(".npy"):
        arr = np.load(feature_path, allow_pickle=True)

        # structured array
        if arr.dtype.names is not None:
            df = pd.DataFrame(arr)
        else:
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D npy feature array, got shape {arr.shape}")
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)
    else:
        raise ValueError("feature_path must be .csv or .npy")
    return df


def load_embeddings(embedding_path: str) -> np.ndarray:
    emb = np.load(embedding_path)
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {emb.shape}")
    return emb


def normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    # 通常 probe / KNN 分析前做标准化更稳
    scaler = StandardScaler()
    return scaler.fit_transform(emb)


def try_import_umap():
    try:
        import umap
        return umap
    except ImportError:
        raise ImportError(
            "umap-learn is not installed. Please run:\n"
            "pip install umap-learn"
        )


def compute_umap(emb: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, random_state: int = 42):
    umap = try_import_umap()
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
    )
    emb_2d = reducer.fit_transform(emb)
    return emb_2d


def save_umap_coordinates(emb_2d: np.ndarray, out_path: str):
    df = pd.DataFrame({
        "umap_x": emb_2d[:, 0],
        "umap_y": emb_2d[:, 1],
    })
    df.to_csv(out_path, index=False)


def plot_umap_colored(
    emb_2d: np.ndarray,
    values: np.ndarray,
    title: str,
    out_path: str,
    point_size: float = 1.0,
    alpha: float = 0.6,
    cmap: str = "viridis",
    value_name: str = "value"
):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=values,
        s=point_size,
        alpha=alpha,
        cmap=cmap,
        linewidths=0
    )
    plt.colorbar(sc, label=value_name)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    # 用于一些极端长尾特征上色时可选
    return pd.Series(arr).rank(pct=True).values


def compute_neighbor_feature_consistency(
    emb: np.ndarray,
    feature_df: pd.DataFrame,
    k: int = 20,
    random_seed: int = 42,
):
    """
    比较三种邻域:
    1) embedding 空间邻域
    2) feature 空间邻域
    3) 随机邻域

    对每个 feature 统计:
    - 邻域内平均方差
    - 邻域内平均绝对偏差
    - 邻域中心样本与邻域均值的平均绝对距离
    """
    rng = np.random.default_rng(random_seed)

    X_emb = StandardScaler().fit_transform(emb)
    X_feat = StandardScaler().fit_transform(feature_df.values)

    n = len(feature_df)
    if k >= n:
        raise ValueError(f"k={k} must be smaller than N={n}")

    # embedding neighbors
    nn_emb = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_emb.fit(X_emb)
    idx_emb = nn_emb.kneighbors(X_emb, return_distance=False)[:, 1:]  # 去掉自身

    # feature neighbors
    nn_feat = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_feat.fit(X_feat)
    idx_feat = nn_feat.kneighbors(X_feat, return_distance=False)[:, 1:]

    # random neighbors
    idx_rand = np.zeros((n, k), dtype=np.int64)
    all_indices = np.arange(n)
    for i in range(n):
        candidates = np.delete(all_indices, i)
        idx_rand[i] = rng.choice(candidates, size=k, replace=False)

    records = []
    feat_arr = feature_df.values
    feat_cols = feature_df.columns.tolist()

    for j, col in enumerate(feat_cols):
        vals = feat_arr[:, j]

        def compute_stats(neighbor_idx):
            neighbor_vals = vals[neighbor_idx]             # [N, k]
            center_vals = vals[:, None]                    # [N, 1]
            var_mean = np.mean(np.var(neighbor_vals, axis=1))
            mad_mean = np.mean(np.mean(np.abs(neighbor_vals - np.mean(neighbor_vals, axis=1, keepdims=True)), axis=1))
            center_to_nb_mean = np.mean(np.mean(np.abs(neighbor_vals - center_vals), axis=1))
            return var_mean, mad_mean, center_to_nb_mean

        emb_var, emb_mad, emb_ctr = compute_stats(idx_emb)
        feat_var, feat_mad, feat_ctr = compute_stats(idx_feat)
        rnd_var, rnd_mad, rnd_ctr = compute_stats(idx_rand)

        records.append({
            "feature": col,
            "embedding_nb_var": emb_var,
            "feature_nb_var": feat_var,
            "random_nb_var": rnd_var,
            "embedding_nb_mad": emb_mad,
            "feature_nb_mad": feat_mad,
            "random_nb_mad": rnd_mad,
            "embedding_center_to_nb_absdiff": emb_ctr,
            "feature_center_to_nb_absdiff": feat_ctr,
            "random_center_to_nb_absdiff": rnd_ctr,
            "embedding_vs_random_var_ratio": emb_var / (rnd_var + 1e-8),
            "embedding_vs_random_absdiff_ratio": emb_ctr / (rnd_ctr + 1e-8),
        })

    return pd.DataFrame(records).sort_values("embedding_vs_random_absdiff_ratio")


def run_probe(
    emb: np.ndarray,
    feature_df: pd.DataFrame,
    model_type: str = "linear",
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    对每个 feature 单独做 probe:
    z -> feature_k
    输出:
    - train/test R²
    - train/test RMSE
    - train/test Spearman
    """
    X = StandardScaler().fit_transform(emb)
    results = []

    for col in feature_df.columns:
        y = feature_df[col].values.astype(np.float32)

        # 去掉 NaN
        valid = np.isfinite(y)
        X_valid = X[valid]
        y_valid = y[valid]

        if len(y_valid) < 50:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=test_size, random_state=random_state
        )

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, pred_train)
        test_r2 = r2_score(y_test, pred_test)
        train_rmse = math.sqrt(mean_squared_error(y_train, pred_train))
        test_rmse = math.sqrt(mean_squared_error(y_test, pred_test))

        train_spr = spearmanr(y_train, pred_train).correlation
        test_spr = spearmanr(y_test, pred_test).correlation

        # 回归系数范数，帮助看信息是否分散编码
        coef_norm = np.linalg.norm(model.coef_) if hasattr(model, "coef_") else np.nan

        results.append({
            "feature": col,
            "model": model_type,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_spearman": train_spr,
            "test_spearman": test_spr,
            "coef_norm": coef_norm,
            "n_samples": len(y_valid),
        })

    df = pd.DataFrame(results).sort_values("test_spearman", ascending=False)
    return df


def compute_embedding_feature_spearman_matrix(
    emb: np.ndarray,
    feature_df: pd.DataFrame,
):
    """
    计算 [Dz, F] 的 Spearman correlation matrix
    """
    dz = emb.shape[1]
    feat_cols = feature_df.columns.tolist()
    mat = np.zeros((dz, len(feat_cols)), dtype=np.float32)

    for i in range(dz):
        for j, col in enumerate(feat_cols):
            y = feature_df[col].values
            valid = np.isfinite(y)
            corr = spearmanr(emb[valid, i], y[valid]).correlation
            mat[i, j] = 0.0 if np.isnan(corr) else corr

    df = pd.DataFrame(mat, columns=feat_cols)
    df.insert(0, "embedding_dim", np.arange(dz))
    return df


def plot_spearman_heatmap(df_corr: pd.DataFrame, out_path: str, figsize=(12, 8)):
    """
    简单 matplotlib heatmap，避免引入 seaborn
    """
    corr_vals = df_corr.drop(columns=["embedding_dim"]).values

    plt.figure(figsize=figsize)
    plt.imshow(corr_vals, aspect="auto")
    plt.colorbar(label="Spearman correlation")
    plt.yticks(
        ticks=np.arange(df_corr.shape[0]),
        labels=df_corr["embedding_dim"].astype(str).tolist(),
        fontsize=6
    )
    plt.xticks(
        ticks=np.arange(corr_vals.shape[1]),
        labels=df_corr.columns[1:].tolist(),
        rotation=90,
        fontsize=8
    )
    plt.title("Embedding Dimension vs Feature Spearman Correlation")
    plt.xlabel("Feature")
    plt.ylabel("Embedding Dimension")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--feature_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--style_features", nargs="+", default=[
        "rel_speed_mean", "thw_mean", "jerk_p95", "speed_norm"
    ])
    parser.add_argument("--scene_features", nargs="+", default=[
        "ego_speed_mean", "neighbor_count", "curvature_mean"
    ])

    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--probe_alpha", type=float, default=1.0)
    parser.add_argument("--max_points_for_umap", type=int, default=100000)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    umap_dir = os.path.join(args.output_dir, "umap")
    ensure_dir(umap_dir)

    print("Loading embeddings...")
    emb = load_embeddings(args.embedding_path)

    print("Loading features...")
    feature_df = load_features(args.feature_path)

    if len(emb) != len(feature_df):
        raise ValueError(f"Embedding rows ({len(emb)}) != feature rows ({len(feature_df)})")

    print(f"Embeddings shape: {emb.shape}")
    print(f"Feature shape: {feature_df.shape}")

    # 只保留数值列
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_df = feature_df[numeric_cols].copy()

    # --------------------------------------------
    # 0) 计算 embedding-feature spearman matrix
    # --------------------------------------------
    print("Computing embedding-feature Spearman matrix...")
    emb_norm = normalize_embeddings(emb)
    corr_df = compute_embedding_feature_spearman_matrix(emb_norm, feature_df)
    corr_csv = os.path.join(args.output_dir, "embedding_feature_spearman.csv")
    corr_df.to_csv(corr_csv, index=False)

    heatmap_path = os.path.join(args.output_dir, "embedding_feature_spearman_heatmap.png")
    plot_spearman_heatmap(corr_df, heatmap_path)

    # --------------------------------------------
    # 1) UMAP
    # --------------------------------------------
    print("Running UMAP...")
    n_for_umap = min(len(emb_norm), args.max_points_for_umap)
    umap_indices = np.arange(len(emb_norm))
    if len(emb_norm) > n_for_umap:
        rng = np.random.default_rng(args.random_state)
        umap_indices = rng.choice(len(emb_norm), size=n_for_umap, replace=False)

    emb_umap_input = emb_norm[umap_indices]
    feat_umap_df = feature_df.iloc[umap_indices].reset_index(drop=True)

    emb_2d = compute_umap(
        emb_umap_input,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state
    )
    save_umap_coordinates(emb_2d, os.path.join(umap_dir, "umap_coordinates.csv"))

    print("Plotting style feature UMAPs...")
    for col in args.style_features:
        if col not in feat_umap_df.columns:
            print(f"[Skip] style feature not found: {col}")
            continue

        values = feat_umap_df[col].values
        plot_umap_colored(
            emb_2d=emb_2d,
            values=values,
            title=f"UMAP colored by {col}",
            out_path=os.path.join(umap_dir, f"umap_{col}.png"),
            point_size=1.0,
            alpha=0.6,
            cmap="viridis",
            value_name=col,
        )

        # 再额外保存 rank-normalized 版本，长尾特征时更容易看梯度
        plot_umap_colored(
            emb_2d=emb_2d,
            values=rank_normalize(values),
            title=f"UMAP colored by {col} (rank normalized)",
            out_path=os.path.join(umap_dir, f"umap_{col}_ranknorm.png"),
            point_size=1.0,
            alpha=0.6,
            cmap="viridis",
            value_name=f"{col}_ranknorm",
        )

    print("Plotting scene feature UMAPs...")
    for col in args.scene_features:
        if col not in feat_umap_df.columns:
            print(f"[Skip] scene feature not found: {col}")
            continue

        values = feat_umap_df[col].values
        plot_umap_colored(
            emb_2d=emb_2d,
            values=values,
            title=f"UMAP colored by {col}",
            out_path=os.path.join(umap_dir, f"umap_{col}.png"),
            point_size=1.0,
            alpha=0.6,
            cmap="viridis",
            value_name=col,
        )

        plot_umap_colored(
            emb_2d=emb_2d,
            values=rank_normalize(values),
            title=f"UMAP colored by {col} (rank normalized)",
            out_path=os.path.join(umap_dir, f"umap_{col}_ranknorm.png"),
            point_size=1.0,
            alpha=0.6,
            cmap="viridis",
            value_name=f"{col}_ranknorm",
        )

    # --------------------------------------------
    # 2) 邻域一致性分析
    # --------------------------------------------
    print("Computing neighbor feature consistency...")
    consistency_df = compute_neighbor_feature_consistency(
        emb=emb_norm,
        feature_df=feature_df,
        k=args.knn_k,
        random_seed=args.random_state,
    )
    consistency_csv = os.path.join(args.output_dir, "neighbor_feature_consistency.csv")
    consistency_df.to_csv(consistency_csv, index=False)

    # --------------------------------------------
    # 3) Linear probe
    # --------------------------------------------
    print("Running linear probe...")
    linear_df = run_probe(
        emb=emb_norm,
        feature_df=feature_df,
        model_type="linear",
        random_state=args.random_state,
    )
    linear_csv = os.path.join(args.output_dir, "linear_probe_results.csv")
    linear_df.to_csv(linear_csv, index=False)

    # --------------------------------------------
    # 4) Ridge probe
    # --------------------------------------------
    print("Running ridge probe...")
    ridge_df = run_probe(
        emb=emb_norm,
        feature_df=feature_df,
        model_type="ridge",
        alpha=args.probe_alpha,
        random_state=args.random_state,
    )
    ridge_csv = os.path.join(args.output_dir, "ridge_probe_results.csv")
    ridge_df.to_csv(ridge_csv, index=False)

    # --------------------------------------------
    # Summary json
    # --------------------------------------------
    summary = {
        "embedding_shape": list(emb.shape),
        "feature_shape": list(feature_df.shape),
        "style_features_requested": args.style_features,
        "scene_features_requested": args.scene_features,
        "style_features_found": [c for c in args.style_features if c in feature_df.columns],
        "scene_features_found": [c for c in args.scene_features if c in feature_df.columns],
        "outputs": {
            "spearman_csv": corr_csv,
            "spearman_heatmap": heatmap_path,
            "umap_dir": umap_dir,
            "neighbor_consistency_csv": consistency_csv,
            "linear_probe_csv": linear_csv,
            "ridge_probe_csv": ridge_csv,
        }
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()