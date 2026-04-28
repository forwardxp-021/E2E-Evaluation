#!/usr/bin/env python3
"""Embedding interpretability demo for trajectory-level driving style embeddings."""

import argparse
from collections import Counter
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-8


def _to_traj_array(row: Any) -> np.ndarray:
    arr = np.asarray(row)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Trajectory row must be (T,>=4), got shape={arr.shape}")
    return arr.astype(np.float64)


def _extract_meta_dict(meta_row: Any) -> Dict[str, Any]:
    if isinstance(meta_row, np.void) and meta_row.dtype.names:
        d = {name: meta_row[name].item() if hasattr(meta_row[name], "item") else meta_row[name] for name in meta_row.dtype.names}
    elif isinstance(meta_row, dict):
        d = dict(meta_row)
    else:
        vals = list(meta_row) if isinstance(meta_row, (list, tuple, np.ndarray)) else [meta_row]
        d = {
            "scenario_id": vals[0] if len(vals) > 0 else "",
            "start": vals[1] if len(vals) > 1 else 0,
            "window_len": vals[2] if len(vals) > 2 else 0,
            "front_id": vals[3] if len(vals) > 3 else "",
        }
        if len(vals) > 4:
            d["policy_id"] = vals[4]
    out = {
        "scenario_id": str(d.get("scenario_id", "")),
        "start": int(d.get("start", 0)),
        "window_len": int(d.get("window_len", 0)),
        "front_id": str(d.get("front_id", "")),
    }
    if "policy_id" in d and d["policy_id"] is not None:
        try:
            out["policy_id"] = int(d["policy_id"])
        except Exception:
            out["policy_id"] = None
    return out


def source_key(meta_row: Any) -> str:
    m = _extract_meta_dict(meta_row)
    return f"{m['scenario_id']}|{m['start']}|{m['window_len']}|{m['front_id']}"


def _make_source_key(meta_row: Any, fields: Sequence[str]) -> str:
    m = _extract_meta_dict(meta_row)
    return "|".join(str(m.get(f, "")) for f in fields)


def _load_metadata_table(
    data_dir: Path,
    n: int,
    meta: Optional[np.ndarray],
    split_arr: np.ndarray,
    source_key_fields: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    warnings: List[str] = []
    sources: Dict[str, str] = {
        "meta_source": "none",
        "source_index_source": "unavailable",
        "policy_id_source": "unavailable",
        "policy_name_source": "unavailable",
        "source_key_source": "source_key_fields",
    }
    table = pd.DataFrame({"row_index": np.arange(n, dtype=int)})

    meta_csv_path = data_dir / "meta.csv"
    if meta_csv_path.exists():
        m = pd.read_csv(meta_csv_path)
        if len(m) == n:
            sources["meta_source"] = "meta.csv"
            table = m.copy()
        else:
            warnings.append(f"meta.csv length={len(m)} does not match n={n}; ignoring meta.csv.")

    if sources["meta_source"] == "none" and meta is not None and len(meta) == n:
        rows = []
        for i in range(n):
            m = _extract_meta_dict(meta[i])
            m["row_index"] = i
            rows.append(m)
        table = pd.DataFrame(rows)
        sources["meta_source"] = "meta.npy"

    if "row_index" not in table.columns:
        table["row_index"] = np.arange(n, dtype=int)
    table["row_index"] = pd.to_numeric(table["row_index"], errors="coerce").fillna(np.arange(n)).astype(int)
    table = table.sort_values("row_index").reset_index(drop=True)
    if len(table) != n or not np.array_equal(table["row_index"].values, np.arange(n)):
        warnings.append("Metadata table row_index is not a complete 0..N-1 range; rebuilding canonical row_index.")
        table = table.iloc[:n].copy()
        table["row_index"] = np.arange(len(table))

    for c, default in [("scenario_id", ""), ("start", 0), ("window_len", 0), ("front_id", "")]:
        if c not in table.columns:
            table[c] = default
    table["scenario_id"] = table["scenario_id"].fillna("").astype(str)
    table["start"] = pd.to_numeric(table["start"], errors="coerce").fillna(0).astype(int)
    table["window_len"] = pd.to_numeric(table["window_len"], errors="coerce").fillna(0).astype(int)
    table["front_id"] = table["front_id"].fillna("").astype(str)

    split_npy = data_dir / "split.npy"
    if "split" not in table.columns:
        table["split"] = split_arr.astype(str)
        sources["split_source"] = "split.npy" if split_npy.exists() else "default_all"
    else:
        table["split"] = table["split"].fillna(split_arr.astype(str) if len(split_arr) == len(table) else "all").astype(str)
        sources["split_source"] = sources["meta_source"]

    src_idx_npy = data_dir / "source_index.npy"
    if src_idx_npy.exists():
        arr = np.load(src_idx_npy, allow_pickle=True)
        if len(arr) == n:
            table["source_index"] = pd.to_numeric(pd.Series(arr), errors="coerce").astype("Int64")
            sources["source_index_source"] = "source_index.npy"
    elif "source_index" in table.columns:
        table["source_index"] = pd.to_numeric(table["source_index"], errors="coerce").astype("Int64")
        sources["source_index_source"] = sources["meta_source"]

    pid_npy = data_dir / "policy_id.npy"
    if pid_npy.exists():
        arr = np.load(pid_npy, allow_pickle=True)
        if len(arr) == n:
            table["policy_id"] = pd.to_numeric(pd.Series(arr), errors="coerce").astype("Int64")
            sources["policy_id_source"] = "policy_id.npy"
    elif "policy_id" in table.columns:
        table["policy_id"] = pd.to_numeric(table["policy_id"], errors="coerce").astype("Int64")
        sources["policy_id_source"] = sources["meta_source"]

    pname_npy = data_dir / "policy_name.npy"
    if pname_npy.exists():
        arr = np.load(pname_npy, allow_pickle=True)
        if len(arr) == n:
            table["policy_name"] = pd.Series(arr).astype(str)
            sources["policy_name_source"] = "policy_name.npy"
    elif "policy_name" in table.columns:
        table["policy_name"] = table["policy_name"].fillna("").astype(str)
        sources["policy_name_source"] = sources["meta_source"]

    if "source_key" in table.columns:
        table["source_key"] = table["source_key"].fillna("").astype(str)
        sources["source_key_source"] = sources["meta_source"]
    else:
        table["source_key"] = (
            table["scenario_id"].astype(str) + "|" + table["start"].astype(str) + "|" +
            table["window_len"].astype(str) + "|" + table["front_id"].astype(str)
        )
        sources["source_key_source"] = "source_key_fields"

    if "source_index" not in table.columns:
        if table["source_key"].nunique() < n:
            table["source_index"] = table.groupby("source_key").ngroup().astype("Int64")
            sources["source_index_source"] = "inferred_from_source_key"
            warnings.append("source_index inferred from source_key groups.")
        else:
            table["source_index"] = pd.Series([pd.NA] * n, dtype="Int64")
    if "policy_id" not in table.columns:
        table["policy_id"] = pd.Series([pd.NA] * n, dtype="Int64")

    if table["policy_id"].isna().all():
        # Safe inference only when every group size identical and policy ordering unambiguous.
        g = table.groupby("source_index", dropna=True)["row_index"].apply(list) if not table["source_index"].isna().all() else pd.Series([], dtype=object)
        if len(g) > 0:
            sizes = sorted({len(v) for v in g.tolist()})
            if len(sizes) == 1 and sizes[0] >= 2:
                inferred = [pd.NA] * n
                for idxs in g.tolist():
                    for pid, row_idx in enumerate(sorted(idxs)):
                        inferred[int(row_idx)] = pid
                table["policy_id"] = pd.Series(inferred, dtype="Int64")
                sources["policy_id_source"] = "inferred_from_within_source_order"
                warnings.append("policy_id inferred from within-source row order.")
    if "policy_name" not in table.columns or table["policy_name"].isna().all():
        table["policy_name"] = table["policy_id"].apply(lambda v: f"policy_{int(v)}" if pd.notna(v) else "")
        if sources["policy_name_source"] == "unavailable":
            sources["policy_name_source"] = "derived_from_policy_id"

    return table, sources, warnings


def _infer_policy_ids(meta: np.ndarray, selected_indices: np.ndarray, source_keys: List[str], aux_policy_id: Optional[np.ndarray]) -> Tuple[List[Optional[int]], str, List[str]]:
    warnings = []
    if aux_policy_id is not None and len(aux_policy_id) == len(meta):
        try:
            parsed = [int(x) for x in aux_policy_id.tolist()]
            return parsed, "policy_id.npy", warnings
        except Exception:
            warnings.append("policy_id.npy found but failed integer parsing; falling back to metadata inference.")

    # explicit policy id from structured/object dict rows
    explicit: List[Optional[int]] = []
    has_any_explicit = False
    for row in meta:
        m = _extract_meta_dict(row)
        pid = m.get("policy_id", None)
        explicit.append(pid)
        has_any_explicit = has_any_explicit or (pid is not None)
    if has_any_explicit and all(v is not None for v in explicit):
        return explicit, "explicit_meta", warnings

    groups: Dict[str, List[int]] = {}
    for i in selected_indices.tolist():
        groups.setdefault(source_keys[i], []).append(i)
    sizes = sorted({len(v) for v in groups.values()})
    if sizes == [3]:
        inferred = [None] * len(meta)
        for _, idxs in groups.items():
            for pid, row_idx in enumerate(sorted(idxs)):
                inferred[row_idx] = pid
        return inferred, "inferred_from_within_source_order", warnings

    warnings.append(
        "policy_id unavailable: explicit field missing/incomplete and within-source group sizes are not consistently 3."
    )
    return explicit, "unavailable", warnings


def _compute_signals(traj: np.ndarray, front: Optional[np.ndarray], dt: float) -> Dict[str, np.ndarray]:
    vx, vy = traj[:, 2], traj[:, 3]
    speed = np.hypot(vx, vy)
    accel = np.gradient(speed, dt)
    jerk = np.gradient(accel, dt)
    heading = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(heading, dt)
    curvature = yaw_rate / np.maximum(speed, EPS)
    out = {
        "speed": speed,
        "accel": accel,
        "jerk": jerk,
        "heading": heading,
        "yaw_rate_proxy": yaw_rate,
        "curvature_proxy": curvature,
    }
    if front is not None:
        dx = front[:, 0] - traj[:, 0]
        dy = front[:, 1] - traj[:, 1]
        gap = np.hypot(dx, dy)
        thw = gap / np.maximum(speed, EPS)
        out["gap"] = gap
        out["thw"] = thw
    return out


def _summary_stats(signals: Dict[str, np.ndarray]) -> Dict[str, float]:
    s = {
        "mean_speed": float(np.mean(signals["speed"])),
        "std_speed": float(np.std(signals["speed"])),
        "mean_accel": float(np.mean(signals["accel"])),
        "std_accel": float(np.std(signals["accel"])),
        "rms_accel": float(np.sqrt(np.mean(signals["accel"] ** 2))),
        "mean_jerk": float(np.mean(signals["jerk"])),
        "rms_jerk": float(np.sqrt(np.mean(signals["jerk"] ** 2))),
        "max_abs_jerk": float(np.max(np.abs(signals["jerk"]))),
        "mean_yaw_rate_proxy": float(np.mean(signals["yaw_rate_proxy"])),
        "std_yaw_rate_proxy": float(np.std(signals["yaw_rate_proxy"])),
        "rms_yaw_rate_proxy": float(np.sqrt(np.mean(signals["yaw_rate_proxy"] ** 2))),
        "mean_curvature_proxy": float(np.mean(signals["curvature_proxy"])),
        "std_curvature_proxy": float(np.std(signals["curvature_proxy"])),
        "rms_curvature_proxy": float(np.sqrt(np.mean(signals["curvature_proxy"] ** 2))),
    }
    if "gap" in signals:
        s["mean_gap"] = float(np.mean(signals["gap"]))
        s["min_gap"] = float(np.min(signals["gap"]))
    if "thw" in signals:
        s["mean_thw"] = float(np.mean(signals["thw"]))
        s["min_thw"] = float(np.min(signals["thw"]))
    return s


def _policy_label(pid: Optional[int], pname: str) -> str:
    if pid is None:
        return "pNA: unknown"
    if pname:
        return f"p{pid}: {pname}"
    return f"p{pid}: policy_{pid}"


def _policy_name_for(pid: Optional[int], policy_mapping: Dict[int, str]) -> str:
    if pid is None:
        return ""
    return policy_mapping.get(pid, f"policy_{pid}")


def _dist(a: np.ndarray, b: np.ndarray, distance: str) -> float:
    if distance == "cosine":
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < EPS or nb < EPS:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    return float(np.linalg.norm(a - b))


def _align_xy(traj: np.ndarray, ref: np.ndarray) -> np.ndarray:
    xy = traj[:, :2].astype(float).copy()
    xy -= ref[0, :2]
    a = math.atan2(ref[0, 3], ref[0, 2])
    ca, sa = math.cos(-a), math.sin(-a)
    x = xy[:, 0] * ca - xy[:, 1] * sa
    y = xy[:, 0] * sa + xy[:, 1] * ca
    return np.stack([x, y], axis=1)


def _get_split(split_arr: Optional[np.ndarray], n: int) -> np.ndarray:
    if split_arr is None:
        return np.array(["all"] * n, dtype=object)
    return np.array([str(x) for x in split_arr], dtype=object)


def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_within_triplet(
    path: Path,
    query_idx: int,
    indices: Sequence[int],
    traj: np.ndarray,
    front: Optional[np.ndarray],
    policy_ids: List[Optional[int]],
    policy_mapping: Dict[int, str],
    skey: str,
):
    q = _to_traj_array(traj[query_idx])
    plt.figure(figsize=(8, 6))
    all_xy: List[np.ndarray] = []
    for i in indices:
        t = _to_traj_array(traj[i])
        xy = _align_xy(t, q)
        all_xy.append(xy)
        pid = policy_ids[i]
        label = f"idx={i}, {_policy_label(pid, _policy_name_for(pid, policy_mapping))}"
        lw = 2.5 if pid in (0, 1, 2) and len(indices) == 3 else 1.8
        plt.plot(xy[:, 0], xy[:, 1], label=label, linewidth=lw)
    if front is not None:
        fq = _to_traj_array(front[query_idx])
        fxy = _align_xy(fq, q)
        all_xy.append(fxy)
        plt.plot(fxy[:, 0], fxy[:, 1], "k--", alpha=0.6, label="front(query)")
    plt.xlabel("x local (m)")
    plt.ylabel("y local (m)")
    plt.title(f"Within-source trajectories | source_key={skey} | query={query_idx}")
    if all_xy:
        pts = np.concatenate(all_xy, axis=0)
        x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        x_margin = max(1.0, 0.08 * (x_max - x_min + EPS))
        y_margin = max(0.5, 0.08 * (y_max - y_min + EPS))
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.legend(fontsize=8)
    _savefig(path)


def _plot_within_signals(
    path: Path,
    indices: Sequence[int],
    traj: np.ndarray,
    front: Optional[np.ndarray],
    policy_ids: List[Optional[int]],
    policy_mapping: Dict[int, str],
    dt: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    keys = ["speed", "accel", "jerk", "yaw_rate_proxy"]
    for i in indices:
        sig = _compute_signals(_to_traj_array(traj[i]), _to_traj_array(front[i]) if front is not None else None, dt)
        pid = policy_ids[i]
        label = f"idx={i}, {_policy_label(pid, _policy_name_for(pid, policy_mapping))}"
        for ax, k in zip(axes.flat, keys):
            ax.plot(sig[k], label=label)
            ax.set_title(k)
            ax.grid(alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    _savefig(path)


def _plot_distance_matrix(
    path: Path,
    csv_path: Path,
    indices: Sequence[int],
    emb: np.ndarray,
    policy_ids: List[Optional[int]],
    policy_mapping: Dict[int, str],
    distance: str,
    source_index: Optional[int],
    source_key: str,
):
    n = len(indices)
    mat = np.zeros((n, n), dtype=float)
    for r, i in enumerate(indices):
        for c, j in enumerate(indices):
            mat[r, c] = _dist(emb[i], emb[j], distance)
    labels = []
    rows = []
    for i in indices:
        pid = policy_ids[i]
        pname = _policy_name_for(pid, policy_mapping)
        labels.append(f"p{pid} {pname} | idx={i}" if pid is not None else f"idx={i}")
    for r, i in enumerate(indices):
        for c, j in enumerate(indices):
            rpid, cpid = policy_ids[i], policy_ids[j]
            rows.append(
                {
                    "row_index": int(i),
                    "row_policy_id": rpid,
                    "row_policy_name": _policy_name_for(rpid, policy_mapping),
                    "col_index": int(j),
                    "col_policy_id": cpid,
                    "col_policy_name": _policy_name_for(cpid, policy_mapping),
                    "distance": float(mat[r, c]),
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label=f"{distance} distance")
    plt.xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(n), labels, fontsize=8)
    for r in range(n):
        for c in range(n):
            plt.text(c, r, f"{mat[r, c]:.3f}", ha="center", va="center", fontsize=8, color="white")
    plt.title(f"Within-source embedding distance matrix | source_index={source_index} | source_key={source_key}")
    _savefig(path)
    return mat


def _plot_embedding_projection(
    path: Path,
    csv_path: Path,
    emb: np.ndarray,
    selected: np.ndarray,
    query_idx: int,
    top_df: pd.DataFrame,
    policy_ids: List[Optional[int]],
    policy_names: List[str],
    policy_mapping: Dict[int, str],
    group_sizes: Dict[str, int],
    source_keys: List[str],
    source_indices: List[Optional[int]],
    splits: List[str],
    projection: str,
    warnings: List[str],
    method: str = "pca",
):
    selected_emb = emb[selected]
    coords = None
    proj_used = method
    explained = None
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(selected_emb)
            proj_used = "umap"
        except Exception:
            warnings.append("UMAP requested but unavailable; falling back to PCA.")
            proj_used = "pca"
    if coords is None:
        x = selected_emb - selected_emb.mean(axis=0, keepdims=True)
        _, svals, vh = np.linalg.svd(x, full_matrices=False)
        basis = vh[:2].T
        coords = x @ basis
        proj_used = "pca"
        denom = float(np.sum(svals ** 2))
        if denom > 0:
            explained = (svals[:2] ** 2) / denom

    idx_to_pos = {int(idx): pos for pos, idx in enumerate(selected.tolist())}
    top_indices = set(top_df["index"].astype(int).tolist()) if len(top_df) else set()
    rank_map = {
        int(r["index"]): int(r["rank"])
        for _, r in top_df.iterrows()
        if pd.notna(r.get("rank")) and not pd.isna(r.get("index"))
    }

    plt.figure(figsize=(8, 6))
    has_policy = any(pid is not None for pid in (policy_ids[i] for i in selected.tolist()))
    if has_policy:
        valid_pids = sorted({int(policy_ids[i]) for i in selected.tolist() if policy_ids[i] is not None})
        cmap = plt.cm.get_cmap("tab10", max(1, len(valid_pids)))
        pid_to_color = {pid: cmap(k) for k, pid in enumerate(valid_pids)}
        legend_handles = []
        for i in selected.tolist():
            pos = idx_to_pos[i]
            c = pid_to_color.get(policy_ids[i], (0.6, 0.6, 0.6, 0.8))
            plt.scatter(coords[pos, 0], coords[pos, 1], c=[c], s=28, alpha=0.7)
        for pid in valid_pids:
            pname = _policy_name_for(pid, policy_mapping)
            legend_handles.append(plt.Line2D([0], [0], marker="o", color="w", label=_policy_label(pid, pname),
                                             markerfacecolor=pid_to_color[pid], markersize=7))
        if legend_handles:
            plt.legend(handles=legend_handles, loc="best", fontsize=8, title="Policies")
    else:
        sizes = np.array([group_sizes[source_keys[i]] for i in selected.tolist()], dtype=float)
        plt.scatter(coords[:, 0], coords[:, 1], c=sizes, cmap="viridis", s=28, alpha=0.7)
        warnings.append("policy_id unavailable for projection coloring.")

    for i in top_indices:
        if i in idx_to_pos:
            pos = idx_to_pos[i]
            plt.scatter(coords[pos, 0], coords[pos, 1], facecolors="none", edgecolors="red", s=120, linewidths=1.6)
            if i in rank_map:
                plt.annotate(f"r{rank_map[i]}", (coords[pos, 0], coords[pos, 1]), fontsize=7, color="red")
    qpos = idx_to_pos[query_idx]
    plt.scatter(coords[qpos, 0], coords[qpos, 1], c="gold", edgecolors="black", marker="*", s=280, linewidths=1.0, label="query")
    if proj_used == "pca":
        plt.title("Embedding 2D projection (PCA; visualization only)")
    else:
        plt.title("Embedding 2D projection (UMAP; visualization only)")
    if explained is not None:
        plt.xlabel(f"PC1 ({100.0 * explained[0]:.1f}%)")
        plt.ylabel(f"PC2 ({100.0 * explained[1]:.1f}%)")
    else:
        plt.xlabel("component 1")
        plt.ylabel("component 2")
    caption = (
        "2D PCA is a lossy projection; policy separation should be judged by high-dimensional metrics and aligned evaluation."
        if proj_used == "pca"
        else "2D UMAP is a nonlinear visualization; policy separation should be judged by high-dimensional metrics and aligned evaluation."
    )
    plt.figtext(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8)
    plt.grid(alpha=0.3)
    _savefig(path)
    csv_rows = []
    for i in selected.tolist():
        pos = idx_to_pos[i]
        pid = policy_ids[i]
        pname = policy_names[i] if policy_names[i] else _policy_name_for(pid, policy_mapping)
        csv_rows.append(
            {
                "index": int(i),
                "pc1": float(coords[pos, 0]),
                "pc2": float(coords[pos, 1]),
                "policy_id": pid,
                "policy_name": pname,
                "source_index": source_indices[i],
                "split": splits[i],
                "is_query": bool(i == query_idx),
                "retrieval_rank": rank_map.get(int(i)),
            }
        )
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)


def _plot_cards(
    path: Path,
    query_idx: int,
    top_df: pd.DataFrame,
    traj: np.ndarray,
    front: Optional[np.ndarray],
    policy_ids: List[Optional[int]],
    policy_mapping: Dict[int, str],
    stats: Dict[int, Dict[str, float]],
):
    rows = 1 + len(top_df)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.6 * rows))
    if rows == 1:
        axes = np.array([axes])

    card_indices = [("Query", query_idx, 0.0)] + [(f"Top-{int(r.rank)}", int(r.index), float(r.distance)) for r in top_df.itertuples()]
    ref = _to_traj_array(traj[query_idx])
    for ridx, (name, idx, dval) in enumerate(card_indices):
        ax_traj, ax_speed = axes[ridx, 0], axes[ridx, 1]
        t = _to_traj_array(traj[idx])
        xy = _align_xy(t, ref)
        sig = _compute_signals(t, _to_traj_array(front[idx]) if front is not None else None, 0.1)
        ax_traj.plot(xy[:, 0], xy[:, 1])
        qpid = policy_ids[query_idx]
        pid = policy_ids[idx]
        same_policy = bool(pid == qpid) if (pid is not None and qpid is not None) else False
        ptxt = _policy_label(pid, _policy_name_for(pid, policy_mapping))
        same_marker = " [same policy]" if same_policy and name != "Query" else ""
        ax_traj.set_title(f"{name}: idx={idx}, {ptxt}{same_marker}, d={dval:.4f}")
        ax_traj.axis("equal")
        ax_traj.grid(alpha=0.3)
        ax_speed.plot(sig["speed"])
        st = stats[idx]
        txt = (
            f"same_policy_as_query={str(same_policy).lower()}\n"
            f"mean_speed={st['mean_speed']:.2f}\n"
            f"rms_jerk={st['rms_jerk']:.2f}\n"
            f"rms_yaw={st['rms_yaw_rate_proxy']:.3f}\n"
            f"rms_curv={st['rms_curvature_proxy']:.3f}"
        )
        if "mean_thw" in st:
            txt += f"\nmean_thw={st['mean_thw']:.2f}"
        ax_speed.text(0.02, 0.98, txt, va="top", ha="left", transform=ax_speed.transAxes, fontsize=8)
        ax_speed.set_title("Speed profile")
        ax_speed.grid(alpha=0.3)
    _savefig(path)


def _plot_global_signals(
    path: Path,
    query_idx: int,
    top_df: pd.DataFrame,
    traj: np.ndarray,
    front: Optional[np.ndarray],
    policy_ids: List[Optional[int]],
    policy_mapping: Dict[int, str],
    dt: float,
    max_signal_topk: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    keys = ["speed", "jerk", "yaw_rate_proxy", "curvature_proxy"]
    qsig = _compute_signals(_to_traj_array(traj[query_idx]), _to_traj_array(front[query_idx]) if front is not None else None, dt)
    for ax, k in zip(axes.flat, keys):
        ax.plot(qsig[k], label=f"query idx={query_idx}", linewidth=2.3)
    for _, r in top_df.head(max_signal_topk).iterrows():
        idx = int(r["index"])
        sig = _compute_signals(_to_traj_array(traj[idx]), _to_traj_array(front[idx]) if front is not None else None, dt)
        pid = policy_ids[idx]
        lbl = f"r{int(r['rank'])} idx={idx} {_policy_label(pid, _policy_name_for(pid, policy_mapping))} d={float(r['distance']):.3f}"
        for ax, k in zip(axes.flat, keys):
            ax.plot(sig[k], label=lbl)
            ax.set_title(k)
            ax.grid(alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    _savefig(path)


def _plot_within_style_fingerprint_bar(
    path: Path,
    csv_path: Path,
    indices: Sequence[int],
    stats: Dict[int, Dict[str, float]],
    policy_ids: List[Optional[int]],
    policy_mapping: Dict[int, str],
):
    metrics = ["mean_speed", "rms_accel", "rms_jerk", "rms_yaw_rate_proxy", "rms_curvature_proxy", "mean_thw", "min_thw"]
    available = [m for m in metrics if any(m in stats[i] for i in indices)]
    rows = []
    for i in indices:
        pid = policy_ids[i]
        pname = _policy_name_for(pid, policy_mapping)
        row = {"index": int(i), "policy_id": pid, "policy_name": pname}
        for m in available:
            row[m] = stats[i].get(m, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    if len(df) == 0 or len(available) == 0:
        return
    def _plot_metric_subset(target_path: Path, metric_subset: List[str], title: str, norm: bool = False):
        subset = [m for m in metric_subset if m in df.columns]
        if not subset:
            return
        x = np.arange(len(subset), dtype=float)
        width = 0.8 / max(1, len(df))
        plt.figure(figsize=(max(7, 1.1 * len(subset)), 4.5))
        for k, row in enumerate(df.itertuples()):
            raw_y = np.array([getattr(row, m) if hasattr(row, m) else np.nan for m in subset], dtype=float)
            y = raw_y.copy()
            if norm:
                for midx, m in enumerate(subset):
                    col = pd.to_numeric(df[m], errors="coerce").astype(float).values
                    lo = np.nanmin(col)
                    hi = np.nanmax(col)
                    denom = hi - lo
                    if np.isfinite(denom) and denom > EPS and np.isfinite(raw_y[midx]):
                        y[midx] = (raw_y[midx] - lo) / denom
                    else:
                        y[midx] = 0.0
            label = _policy_label(row.policy_id, row.policy_name)
            plt.bar(x + (k - (len(df) - 1) / 2.0) * width, y, width=width, label=label, alpha=0.85)
        plt.xticks(x, subset, rotation=25, ha="right")
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)
        if norm:
            plt.ylim(-0.05, 1.05)
        plt.legend(fontsize=8)
        _savefig(target_path)

    _plot_metric_subset(
        path,
        ["mean_speed", "mean_thw", "min_thw"],
        "Within-source style fingerprint (kinematic)",
    )
    _plot_metric_subset(
        path.with_name("within_source_style_fingerprint_dynamics.png"),
        ["rms_accel", "rms_jerk", "rms_yaw_rate_proxy", "rms_curvature_proxy"],
        "Within-source style fingerprint (dynamics)",
    )
    _plot_metric_subset(
        path.with_name("within_source_style_fingerprint_normalized.png"),
        ["mean_speed", "rms_accel", "rms_jerk", "rms_yaw_rate_proxy", "rms_curvature_proxy", "mean_thw", "min_thw"],
        "Within-source style fingerprint (normalized across p0/p1/p2)",
        norm=True,
    )


def _generate_report(path: Path, summary: Dict[str, Any], retrieval_csv: Path):
    retrieval_df = pd.read_csv(retrieval_csv) if retrieval_csv.exists() else pd.DataFrame()
    top_df = retrieval_df[(~retrieval_df.get("excluded", False)) & (retrieval_df.get("rank", 0) <= summary.get("topk", 0))].copy()
    top_cols = [
        c for c in ["rank", "index", "policy_id", "policy_name", "distance", "same_policy_as_query", "mean_speed", "rms_jerk", "rms_yaw_rate_proxy", "rms_curvature_proxy", "mean_thw"]
        if c in top_df.columns
    ]
    md = []
    md.append("# Interpretability Report")
    md.append("")
    md.append("## Query")
    md.append(f"- query_index: {summary.get('query_index')}")
    md.append(f"- source_index: {summary.get('query_source_index')}")
    md.append(f"- source_key: {summary.get('query_source_key')}")
    md.append(f"- query policy: p{summary.get('query_policy_id')}: {summary.get('query_policy_name')}")
    md.append("")
    md.append("## Within-source")
    ws = summary.get("within_source", {})
    md.append(f"- policy list: {ws.get('available_policy_ids')}")
    pdists = ws.get("pairwise_embedding_distances")
    if isinstance(pdists, dict) and len(pdists) > 0:
        md.append("- embedding distances:")
        for k in sorted(pdists.keys()):
            md.append(f"  - {k}: {pdists[k]:.4f}")
    md.append("- style interpretation: compare mean_speed / rms_jerk / yaw_rate_proxy / curvature_proxy / THW across p0-p1-p2 in within_source_style_fingerprint.csv.")
    auto_interp = ws.get("automatic_interpretation", {})
    if isinstance(auto_interp, dict) and len(auto_interp) > 0:
        md.append("- automatic interpretation:")
        for k in ["highest_mean_speed", "lowest_rms_jerk", "lowest_rms_yaw_rate_proxy", "p2_farthest_in_embedding"]:
            if k in auto_interp:
                md.append(f"  - {k}: {auto_interp[k]}")
    md.append("")
    md.append("## Global retrieval Top-K")
    if len(top_df) > 0 and len(top_cols) > 0:
        md.append(top_df[top_cols].to_markdown(index=False))
    else:
        md.append("- No valid Top-K rows.")
    gr = summary.get("global_retrieval", {})
    md.append("")
    md.append(f"- same-policy hit@1: {gr.get('hit_at_1_same_policy')}")
    md.append(f"- same-policy hit@k: {gr.get('hit_at_k_same_policy')}")
    md.append("")
    md.append("## Limitations")
    md.append("- PCA/UMAP are visualization-only and lossy.")
    md.append("- Benchmark conclusions should rely on aligned metrics and high-dimensional embedding distances.")
    md.append("- yaw_rate_proxy / curvature_proxy are approximate proxies from velocity direction.")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def _make_synth(n_src: int = 5, n_policy: int = 3, t: int = 40):
    rng = np.random.default_rng(42)
    n = n_src * n_policy
    emb = np.zeros((n, 8), dtype=np.float32)
    meta = np.empty(n, dtype=object)
    traj = np.empty(n, dtype=object)
    front = np.empty(n, dtype=object)
    split = np.array(["test"] * n, dtype=object)
    idx = 0
    for s in range(n_src):
        for p in range(n_policy):
            x = np.linspace(0, 30, t)
            yaw = 0.02 * p * np.sin(np.linspace(0, 2 * np.pi, t))
            y = np.cumsum(yaw)
            v = 8 + p + 0.6 * np.sin(np.linspace(0, np.pi, t))
            vx, vy = v * np.cos(yaw), v * np.sin(yaw)
            ego = np.stack([x, y, vx, vy], axis=1)
            fr = np.stack([x + 20, y + 0.5, vx * 0.95, vy], axis=1)
            traj[idx], front[idx] = ego, fr
            meta[idx] = (f"scenario_{s:03d}", s * 10, t, f"front_{s}")
            emb[idx] = np.array([s, p, np.mean(v), np.std(v), np.std(yaw), 0.1 * p, 0.01 * s, 1.0], dtype=np.float32)
            idx += 1
    return emb, meta, traj, front, split


def _choose_case(
    mode: str,
    explicit_query_idx: Optional[int],
    selected: np.ndarray,
    source_keys: List[str],
    source_indices: List[Optional[int]],
    policy_ids: List[Optional[int]],
    embeddings: np.ndarray,
    distance: str,
    topk: int,
    stats: Optional[Dict[int, Dict[str, float]]] = None,
    traj: Optional[np.ndarray] = None,
) -> Tuple[int, Dict[str, Any]]:
    selected_list = [int(i) for i in selected.tolist()]
    if explicit_query_idx is not None:
        mode = "query_index"
    if mode == "query_index":
        q = explicit_query_idx if explicit_query_idx is not None else selected_list[0]
        return int(q), {"mode": "query_index", "selected_query_index": int(q), "reason": "Explicit query index.", "score": None}

    groups: Dict[str, List[int]] = {}
    for i in selected_list:
        groups.setdefault(source_keys[i], []).append(i)
    valid_groups = [sorted(v) for v in groups.values() if len(v) >= 3]
    if mode == "first_valid":
        if valid_groups:
            q = valid_groups[0][0]
            return int(q), {"mode": mode, "selected_query_index": int(q), "reason": "First deterministic source with >=3 policies.", "score": None}
        q = selected_list[0]
        return int(q), {"mode": mode, "selected_query_index": int(q), "reason": "Fallback to first selected row.", "score": None}

    if mode == "best_hit_at_k":
        best = (-1, selected_list[0])
        for q in selected_list:
            qpid = policy_ids[q]
            if qpid is None:
                continue
            drows = []
            for i in selected_list:
                if i == q:
                    continue
                drows.append((i, _dist(embeddings[q], embeddings[i], distance)))
            drows.sort(key=lambda x: x[1])
            hits = sum(1 for i, _ in drows[:topk] if policy_ids[i] == qpid)
            best = max(best, (hits, q))
        return int(best[1]), {"mode": mode, "selected_query_index": int(best[1]), "reason": "Maximum same-policy count in top-k.", "score": float(best[0])}

    if mode == "best_p2_separation":
        group_by_source_idx: Dict[int, List[int]] = {}
        for i in selected_list:
            sidx = source_indices[i]
            if sidx is not None:
                group_by_source_idx.setdefault(int(sidx), []).append(i)
        best_score = -1.0
        best_q = selected_list[0]
        for _, g in group_by_source_idx.items():
            idx_by_pid = {policy_ids[i]: i for i in g if policy_ids[i] is not None}
            if 0 not in idx_by_pid or 1 not in idx_by_pid or 2 not in idx_by_pid:
                continue
            p0, p1, p2 = idx_by_pid[0], idx_by_pid[1], idx_by_pid[2]
            score = min(_dist(embeddings[p2], embeddings[p0], distance), _dist(embeddings[p2], embeddings[p1], distance))
            if score > best_score:
                best_score = float(score)
                best_q = int(p2)
        return int(best_q), {"mode": mode, "selected_query_index": int(best_q), "reason": "Largest min(d(p2,p0), d(p2,p1)) in selected split.", "score": (best_score if best_score >= 0 else None)}

    if mode == "best_human_readable":
        group_by_source_idx: Dict[int, List[int]] = {}
        for i in selected_list:
            sidx = source_indices[i]
            if sidx is not None:
                group_by_source_idx.setdefault(int(sidx), []).append(i)
        if not group_by_source_idx:
            q = selected_list[0]
            return int(q), {"mode": mode, "selected_query_index": int(q), "reason": "No valid source_index groups; fallback to first selected.", "score": None}

        component_rows = []
        for _, g in group_by_source_idx.items():
            idx_by_pid = {policy_ids[i]: i for i in g if policy_ids[i] is not None}
            if 0 not in idx_by_pid or 1 not in idx_by_pid or 2 not in idx_by_pid:
                continue
            p0, p1, p2 = idx_by_pid[0], idx_by_pid[1], idx_by_pid[2]
            sep = min(_dist(embeddings[p2], embeddings[p0], distance), _dist(embeddings[p2], embeddings[p1], distance))
            yaw_curv = 0.0
            jerk_diff = 0.0
            endpoint_diff = 0.0
            hit_cnt = 0.0
            if stats is not None:
                yaw_curv = (
                    abs(stats[p2].get("rms_yaw_rate_proxy", 0.0) - stats[p0].get("rms_yaw_rate_proxy", 0.0))
                    + abs(stats[p2].get("rms_curvature_proxy", 0.0) - stats[p0].get("rms_curvature_proxy", 0.0))
                    + abs(stats[p2].get("rms_yaw_rate_proxy", 0.0) - stats[p1].get("rms_yaw_rate_proxy", 0.0))
                    + abs(stats[p2].get("rms_curvature_proxy", 0.0) - stats[p1].get("rms_curvature_proxy", 0.0))
                ) / 2.0
                jerk_diff = (
                    abs(stats[p2].get("rms_jerk", 0.0) - stats[p0].get("rms_jerk", 0.0))
                    + abs(stats[p2].get("rms_jerk", 0.0) - stats[p1].get("rms_jerk", 0.0))
                ) / 2.0
            drows = []
            for i in selected_list:
                if i == p2:
                    continue
                drows.append((i, _dist(embeddings[p2], embeddings[i], distance)))
            drows.sort(key=lambda x: x[1])
            hit_cnt = float(sum(1 for i, _ in drows[:topk] if policy_ids[i] == 2))
            component_rows.append(
                {
                    "p2": int(p2),
                    "sep": float(sep),
                    "yaw_curv": float(yaw_curv),
                    "jerk_diff": float(jerk_diff),
                    "hit_cnt": float(hit_cnt),
                    "endpoint_diff": float(endpoint_diff),
                }
            )

        if not component_rows:
            q = selected_list[0]
            return int(q), {"mode": mode, "selected_query_index": int(q), "reason": "No complete p0/p1/p2 groups; fallback to first selected.", "score": None}

        # endpoint diff uses trajectory endpoints (computed once group candidates are known)
        for row in component_rows:
            p2 = row["p2"]
            sidx = source_indices[p2]
            group = group_by_source_idx[int(sidx)] if sidx is not None else []
            idx_by_pid = {policy_ids[i]: i for i in group if policy_ids[i] is not None}
            if traj is not None and 0 in idx_by_pid and 1 in idx_by_pid:
                ep2 = _to_traj_array(traj[p2])[-1, :2]
                ep0 = _to_traj_array(traj[idx_by_pid[0]])[-1, :2]
                ep1 = _to_traj_array(traj[idx_by_pid[1]])[-1, :2]
                row["endpoint_diff"] = float((np.linalg.norm(ep2 - ep0) + np.linalg.norm(ep2 - ep1)) / 2.0)

        comp_keys = ["sep", "yaw_curv", "jerk_diff", "endpoint_diff", "hit_cnt"]
        norms: Dict[str, Tuple[float, float]] = {}
        for k in comp_keys:
            vals = [r[k] for r in component_rows]
            norms[k] = (min(vals), max(vals))
        for r in component_rows:
            score = 0.0
            for k in comp_keys:
                lo, hi = norms[k]
                if hi - lo > EPS:
                    score += (r[k] - lo) / (hi - lo)
            r["score"] = float(score)
        best = max(component_rows, key=lambda x: x["score"])
        return int(best["p2"]), {"mode": mode, "selected_query_index": int(best["p2"]), "reason": "Best combined human-readable separation score.", "score": float(best["score"]), "components": best}

    q = selected_list[0]
    return int(q), {"mode": mode, "selected_query_index": int(q), "reason": "Unknown case_selection mode fallback.", "score": None}


def run_demo(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if args.smoke_test:
        embeddings, meta, traj, front, split = _make_synth()
        embedding_path = "<synthetic>"
        meta_path = traj_path = front_path = split_path = "<synthetic>"
        data_dir = None
    else:
        data_dir = Path(args.data_dir)
        embeddings = np.load(data_dir / f"{args.embedding}.npy", allow_pickle=True)
        meta = np.load(data_dir / "meta.npy", allow_pickle=True)
        traj = np.load(data_dir / "traj.npy", allow_pickle=True)
        front_file = data_dir / "front.npy"
        front = np.load(front_file, allow_pickle=True) if front_file.exists() else None
        split_file = data_dir / "split.npy"
        split = np.load(split_file, allow_pickle=True) if split_file.exists() else None
        embedding_path = str(data_dir / f"{args.embedding}.npy")
        meta_path = str(data_dir / "meta.npy")
        traj_path = str(data_dir / "traj.npy")
        front_path = str(front_file) if front is not None else None
        split_path = str(split_file) if split is not None else None

    n = len(embeddings)
    if len(traj) != n:
        raise ValueError(f"len(embedding)={n} must equal len(traj)={len(traj)}")
    if front is not None and len(front) != n:
        raise ValueError(f"len(front)={len(front)} must equal len(embedding)={n}")
    if len(meta) != n:
        raise ValueError(f"len(meta)={len(meta)} must equal len(embedding)={n}")
    if not args.smoke_test and data_dir is not None:
        for feat_name in ("feat_style.npy", "feat_style_raw.npy"):
            feat_path = data_dir / feat_name
            if feat_path.exists():
                feat_arr = np.load(feat_path, allow_pickle=True)
                if len(feat_arr) != n:
                    raise ValueError(f"len({feat_name})={len(feat_arr)} must equal len(embedding)={n}")
    split_arr = _get_split(split, n)
    source_key_fields = [x.strip() for x in args.source_key_fields.split(",") if x.strip()]
    if len(source_key_fields) == 0:
        raise ValueError("source_key_fields must include at least one field")

    metadata_sources: Dict[str, str] = {
        "meta_source": "synthetic",
        "source_index_source": "synthetic",
        "policy_id_source": "synthetic",
        "policy_name_source": "synthetic",
        "source_key_source": "synthetic",
    }
    if args.smoke_test:
        md_rows = []
        for i in range(n):
            m = _extract_meta_dict(meta[i])
            md_rows.append({
                "row_index": i,
                "source_index": i // 3,
                "source_key": _make_source_key(meta[i], source_key_fields),
                "policy_id": i % 3,
                "policy_name": f"policy_{i%3}",
                "scenario_id": m["scenario_id"],
                "start": m["start"],
                "window_len": m["window_len"],
                "front_id": m["front_id"],
                "split": split_arr[i],
            })
        metadata_table = pd.DataFrame(md_rows)
    else:
        metadata_table, metadata_sources, md_warn = _load_metadata_table(data_dir, n, meta, split_arr, source_key_fields)
        warnings.extend(md_warn)

    source_keys = metadata_table["source_key"].astype(str).tolist()
    source_index_vals = metadata_table["source_index"]
    has_source_index = not source_index_vals.isna().all()
    source_group_series = metadata_table.groupby("source_index").size().to_dict() if has_source_index else {}
    source_group_sizes_total = {int(k): int(v) for k, v in source_group_series.items()} if has_source_index else {}
    group_sizes_total_key: Dict[str, int] = Counter(source_keys)
    group_hist_total_key: Dict[str, int] = Counter(group_sizes_total_key.values())
    group_hist_total_source_idx: Dict[str, int] = Counter(source_group_sizes_total.values()) if has_source_index else {}

    if args.split == "all":
        selected = np.arange(n)
    else:
        selected = np.where(metadata_table["split"].astype(str).values == args.split)[0]
    if len(selected) == 0:
        raise ValueError(f"No samples for split={args.split}")
    selected_mask = metadata_table["row_index"].isin(selected.tolist())
    group_sizes_after_key: Dict[str, int] = Counter([source_keys[i] for i in selected.tolist()])
    group_hist_after_key: Dict[str, int] = Counter(group_sizes_after_key.values())
    group_sizes_after_source_idx = (
        metadata_table.loc[selected_mask].groupby("source_index").size().to_dict()
        if has_source_index else {}
    )
    group_hist_after_source_idx: Dict[str, int] = Counter(group_sizes_after_source_idx.values()) if has_source_index else {}

    policy_ids: List[Optional[int]] = [
        int(x) if pd.notna(x) else None
        for x in metadata_table["policy_id"].tolist()
    ]
    policy_names: List[str] = [
        str(x) if (x is not None and str(x) != "nan") else ""
        for x in metadata_table["policy_name"].tolist()
    ]
    pid_source = metadata_sources["policy_id_source"]
    has_policy_id = any(pid is not None for pid in policy_ids)

    policy_ids_observed = sorted({pid for pid in policy_ids if pid is not None})
    expected_num_policies = len(policy_ids_observed)
    if not args.smoke_test and data_dir is not None:
        pnames_json = data_dir / "policy_names.json"
        if pnames_json.exists():
            try:
                mapping = json.loads(pnames_json.read_text(encoding="utf-8"))
                if isinstance(mapping, dict):
                    expected_num_policies = max(expected_num_policies, len(mapping))
            except Exception:
                warnings.append("Failed to parse policy_names.json for expected_num_policies.")

    policy_mapping: Dict[int, str] = {}
    for i in selected.tolist():
        pid = policy_ids[i]
        if pid is None:
            continue
        pname = policy_names[i] if policy_names[i] else f"policy_{pid}"
        if pid not in policy_mapping:
            policy_mapping[pid] = pname
    for pid in policy_ids_observed:
        policy_mapping.setdefault(pid, f"policy_{pid}")
    for i in range(n):
        pid = policy_ids[i]
        if pid is not None and not policy_names[i]:
            policy_names[i] = policy_mapping.get(pid, f"policy_{pid}")

    # Precompute stats
    stats: Dict[int, Dict[str, float]] = {}
    for i in selected.tolist():
        sig = _compute_signals(_to_traj_array(traj[i]), _to_traj_array(front[i]) if front is not None else None, args.dt)
        stats[i] = _summary_stats(sig)

    case_mode = args.case_selection
    if args.query_index is not None and args.case_selection == "first_valid":
        case_mode = "query_index"
    query_idx, case_selection_info = _choose_case(
        mode=case_mode,
        explicit_query_idx=args.query_index,
        selected=selected,
        source_keys=source_keys,
        source_indices=[int(x) if pd.notna(x) else None for x in metadata_table["source_index"].tolist()],
        policy_ids=policy_ids,
        embeddings=embeddings,
        distance=args.distance,
        topk=args.topk,
        stats=stats,
        traj=traj,
    )
    if query_idx not in selected.tolist():
        raise ValueError("Selected query_index is not in selected split")
    if policy_ids[query_idx] is None:
        warnings.append(
            "policy_id unavailable: global retrieval can show nearest embedding neighbors, but cannot verify same-policy style retrieval."
        )

    query_meta = {
        "scenario_id": str(metadata_table.at[query_idx, "scenario_id"]),
        "start": int(metadata_table.at[query_idx, "start"]),
        "window_len": int(metadata_table.at[query_idx, "window_len"]),
        "front_id": str(metadata_table.at[query_idx, "front_id"]),
    }
    q_source = source_keys[query_idx]
    q_source_index = metadata_table.at[query_idx, "source_index"] if has_source_index else None
    if args.mode in ("within_source", "both") and group_sizes_after_key.get(q_source, 0) < 2:
        if args.auto_select_valid_source:
            valid_candidates: List[int] = []
            if has_source_index and expected_num_policies > 0:
                selected_df = metadata_table.loc[selected_mask]
                grouped = selected_df.groupby("source_index")
                full_groups = []
                for sidx, gdf in grouped:
                    pset = set(int(x) for x in gdf["policy_id"].dropna().tolist())
                    if len(gdf) >= expected_num_policies and len(pset) >= expected_num_policies:
                        full_groups.append((int(sidx), sorted(gdf["row_index"].astype(int).tolist())))
                if full_groups:
                    full_groups = sorted(full_groups, key=lambda x: x[0])
                    valid_candidates = [full_groups[0][1][0]]
            if not valid_candidates:
                valid_candidates = sorted(i for i in selected.tolist() if group_sizes_after_key[source_keys[i]] >= 3)
            if valid_candidates:
                query_idx = valid_candidates[0]
                query_meta = {
                    "scenario_id": str(metadata_table.at[query_idx, "scenario_id"]),
                    "start": int(metadata_table.at[query_idx, "start"]),
                    "window_len": int(metadata_table.at[query_idx, "window_len"]),
                    "front_id": str(metadata_table.at[query_idx, "front_id"]),
                }
                q_source = source_keys[query_idx]
                q_source_index = metadata_table.at[query_idx, "source_index"] if has_source_index else None
                warnings.append("query source group had fewer than 2 samples; auto-selected replacement from a source group with >=3 samples.")
        else:
            warnings.append("query source group has fewer than 2 samples; within-source demo is not applicable.")

    retrieval_rows = []
    top_rows = []
    if args.mode in ("global", "both"):
        for i in selected.tolist():
            if i == query_idx and args.exclude_self:
                excluded, reason = True, "self"
            else:
                excluded, reason = False, ""
            same_source = source_keys[i] == q_source
            same_scenario = str(metadata_table.at[i, "scenario_id"]) == query_meta["scenario_id"]
            if args.exclude_same_source and same_source and i != query_idx:
                excluded, reason = True, (reason + ";same_source").strip(";")
            if args.exclude_same_scenario and same_scenario and i != query_idx:
                excluded, reason = True, (reason + ";same_scenario").strip(";")
            d = _dist(embeddings[query_idx], embeddings[i], args.distance)
            pid = policy_ids[i]
            qpid = policy_ids[query_idx]
            row = {
                "rank": None,
                "query_index": query_idx,
                "query_policy_id": qpid,
                "query_policy_name": policy_names[query_idx] if policy_names[query_idx] else _policy_name_for(qpid, policy_mapping),
                "query_policy_label": _policy_label(qpid, policy_names[query_idx] if policy_names[query_idx] else _policy_name_for(qpid, policy_mapping)),
                "index": i,
                "policy_id": pid,
                "policy_name": policy_names[i] if policy_names[i] else _policy_name_for(pid, policy_mapping),
                "policy_label": _policy_label(pid, policy_names[i] if policy_names[i] else _policy_name_for(pid, policy_mapping)),
                "source_key": source_keys[i],
                "source_index": int(metadata_table.at[i, "source_index"]) if pd.notna(metadata_table.at[i, "source_index"]) else None,
                "scenario_id": str(metadata_table.at[i, "scenario_id"]),
                "start": int(metadata_table.at[i, "start"]),
                "window_len": int(metadata_table.at[i, "window_len"]),
                "front_id": str(metadata_table.at[i, "front_id"]),
                "split": str(metadata_table.at[i, "split"]),
                "distance": d,
                "distance_type": args.distance,
                "same_policy_as_query": (pid == qpid) if (pid is not None and qpid is not None) else None,
                "same_source_as_query": same_source,
                "same_scenario_as_query": same_scenario,
                "excluded": excluded,
                "exclude_reason": reason,
            }
            row.update({k: stats[i].get(k, np.nan) for k in [
                "mean_speed", "std_speed", "rms_accel", "rms_jerk", "rms_yaw_rate_proxy", "rms_curvature_proxy", "mean_gap", "min_gap", "mean_thw", "min_thw"
            ]})
            retrieval_rows.append(row)
        df = pd.DataFrame(retrieval_rows).sort_values("distance").reset_index(drop=True)
        rank = 1
        for ridx, r in df.iterrows():
            if not r["excluded"]:
                df.at[ridx, "rank"] = rank
                if rank <= args.topk:
                    top_rows.append(df.loc[ridx].to_dict())
                rank += 1
        retrieval_df = df
    else:
        retrieval_df = pd.DataFrame(columns=["rank"])

    top_df = pd.DataFrame(top_rows)
    retrieval_df.to_csv(out_dir / "retrieval_table.csv", index=False)

    # within-source
    within = {
        "enabled": args.mode in ("within_source", "both"),
        "source_key": q_source,
        "available_indices": [],
        "available_policy_ids": [],
        "available_policy_labels": [],
        "applicable": False,
        "reason_if_not_applicable": None,
        "pairwise_embedding_distances": None,
        "automatic_interpretation": {},
    }
    selected_for_fingerprint = {query_idx}

    if within["enabled"]:
        if has_source_index and pd.notna(q_source_index):
            ws = [i for i in selected.tolist() if pd.notna(metadata_table.at[i, "source_index"]) and int(metadata_table.at[i, "source_index"]) == int(q_source_index)]
            within["source_index"] = int(q_source_index)
        else:
            ws = [i for i in selected.tolist() if source_keys[i] == q_source]
            within["source_index"] = None
        within["available_indices"] = ws
        within["available_policy_ids"] = [policy_ids[i] for i in ws]
        within["available_policy_labels"] = [
            _policy_label(policy_ids[i], policy_names[i] if policy_names[i] else _policy_name_for(policy_ids[i], policy_mapping))
            for i in ws
        ]
        selected_for_fingerprint.update(ws)
        min_needed = max(3, expected_num_policies if expected_num_policies > 0 else 3)
        if len(ws) >= min_needed:
            within["applicable"] = True
            ws_sorted = sorted(ws)
            ws_plot = ws_sorted[:max(3, min(len(ws_sorted), expected_num_policies if expected_num_policies > 0 else 3))]
            _plot_within_triplet(out_dir / "within_source_triplet.png", query_idx, ws_plot, traj, front, policy_ids, policy_mapping, q_source)
            _plot_within_signals(out_dir / "within_source_style_signals.png", ws_plot, traj, front, policy_ids, policy_mapping, args.dt)
            mat = _plot_distance_matrix(
                out_dir / "embedding_distance_matrix.png",
                out_dir / "embedding_distance_matrix.csv",
                ws_plot,
                embeddings,
                policy_ids,
                policy_mapping,
                args.distance,
                int(q_source_index) if (q_source_index is not None and pd.notna(q_source_index)) else None,
                q_source,
            )
            _plot_within_style_fingerprint_bar(
                out_dir / "within_source_style_fingerprint_kinematic.png",
                out_dir / "within_source_style_fingerprint.csv",
                ws_plot,
                stats,
                policy_ids,
                policy_mapping,
            )
            within["pairwise_embedding_distances"] = {
                f"{ws_plot[r]}->{ws_plot[c]}": float(mat[r, c]) for r in range(len(ws_plot)) for c in range(len(ws_plot))
            }
            idx_by_pid = {policy_ids[i]: i for i in ws_plot if policy_ids[i] is not None}
            auto_interp: Dict[str, Any] = {}
            if idx_by_pid:
                highest_speed_idx = max(idx_by_pid.values(), key=lambda i: stats[i].get("mean_speed", -np.inf))
                lowest_jerk_idx = min(idx_by_pid.values(), key=lambda i: stats[i].get("rms_jerk", np.inf))
                lowest_yaw_idx = min(idx_by_pid.values(), key=lambda i: stats[i].get("rms_yaw_rate_proxy", np.inf))
                auto_interp["highest_mean_speed"] = _policy_label(policy_ids[highest_speed_idx], _policy_name_for(policy_ids[highest_speed_idx], policy_mapping))
                auto_interp["lowest_rms_jerk"] = _policy_label(policy_ids[lowest_jerk_idx], _policy_name_for(policy_ids[lowest_jerk_idx], policy_mapping))
                auto_interp["lowest_rms_yaw_rate_proxy"] = _policy_label(policy_ids[lowest_yaw_idx], _policy_name_for(policy_ids[lowest_yaw_idx], policy_mapping))
            if 0 in idx_by_pid and 1 in idx_by_pid and 2 in idx_by_pid:
                d20 = _dist(embeddings[idx_by_pid[2]], embeddings[idx_by_pid[0]], args.distance)
                d21 = _dist(embeddings[idx_by_pid[2]], embeddings[idx_by_pid[1]], args.distance)
                d01 = _dist(embeddings[idx_by_pid[0]], embeddings[idx_by_pid[1]], args.distance)
                auto_interp["p2_farthest_in_embedding"] = bool(min(d20, d21) > d01)
            within["automatic_interpretation"] = auto_interp
        else:
            msg = "within-source not applicable: fewer than expected policy samples under query source group"
            warnings.append(msg)
            within["reason_if_not_applicable"] = msg

    if args.mode in ("global", "both"):
        if len(top_df) > 0:
            selected_for_fingerprint.update(top_df["index"].astype(int).tolist())
            _plot_cards(out_dir / "global_retrieval_cards.png", query_idx, top_df, traj, front, policy_ids, policy_mapping, stats)
            _plot_global_signals(out_dir / "global_retrieval_style_signals.png", query_idx, top_df, traj, front, policy_ids, policy_mapping, args.dt, args.max_signal_topk)
        else:
            warnings.append("global retrieval plot skipped: no non-excluded candidates")

    source_index_list = [int(x) if pd.notna(x) else None for x in metadata_table["source_index"].tolist()]
    split_list = metadata_table["split"].astype(str).tolist()
    projection_mode = args.projection
    umap_available = False
    if projection_mode in ("umap", "both"):
        try:
            import umap  # type: ignore # noqa: F401
            umap_available = True
        except Exception:
            warnings.append("UMAP requested but umap-learn is not installed; continuing with PCA visualization.")
    if projection_mode == "both":
        methods = ["pca"] + (["umap"] if umap_available else [])
    elif projection_mode == "umap":
        methods = ["umap"] if umap_available else ["pca"]
    else:
        methods = ["pca"]
    for m in methods:
        if m == "pca":
            _plot_embedding_projection(
                out_dir / "embedding_2d_projection.png",
                out_dir / "embedding_2d_projection.csv",
                embeddings,
                selected,
                query_idx,
                top_df,
                policy_ids,
                policy_names,
                policy_mapping,
                group_sizes_after_key,
                source_keys,
                source_index_list,
                split_list,
                projection_mode,
                warnings,
                method="pca",
            )
        else:
            _plot_embedding_projection(
                out_dir / "embedding_2d_projection_umap.png",
                out_dir / "embedding_2d_projection_umap.csv",
                embeddings,
                selected,
                query_idx,
                top_df,
                policy_ids,
                policy_names,
                policy_mapping,
                group_sizes_after_key,
                source_keys,
                source_index_list,
                split_list,
                projection_mode,
                warnings,
                method="umap",
            )

    fp_rows = []
    for i in sorted(selected_for_fingerprint):
        row = {
            "index": i,
            "source_key": source_keys[i],
            "source_index": int(metadata_table.at[i, "source_index"]) if pd.notna(metadata_table.at[i, "source_index"]) else None,
            "scenario_id": str(metadata_table.at[i, "scenario_id"]),
            "start": int(metadata_table.at[i, "start"]),
            "window_len": int(metadata_table.at[i, "window_len"]),
            "front_id": str(metadata_table.at[i, "front_id"]),
            "split": str(metadata_table.at[i, "split"]),
            "policy_id": policy_ids[i],
            "policy_name": policy_names[i] if policy_names[i] else (f"policy_{policy_ids[i]}" if policy_ids[i] is not None else ""),
        }
        row.update(stats[i])
        fp_rows.append(row)
    pd.DataFrame(fp_rows).to_csv(out_dir / "style_fingerprint.csv", index=False)

    topk_same_policy = None
    hit1 = None
    hitk = None
    hit_reason = None
    if len(top_df) > 0 and policy_ids[query_idx] is not None:
        same = top_df["policy_id"] == policy_ids[query_idx]
        topk_same_policy = int(same.sum())
        hit1 = bool(same.iloc[0])
        hitk = bool(topk_same_policy > 0)
    else:
        hit_reason = "policy_id unavailable" if policy_ids[query_idx] is None else "no retrieval candidates"

    if (not has_source_index or not has_policy_id) and not args.smoke_test:
        warnings.append(
            "This data directory cannot support policy-level interpretability because policy_id/source_index metadata is missing. "
            "Please regenerate rollouts with the updated generator metadata output."
        )

    policy_counts = Counter([pid for pid in policy_ids if pid is not None])
    pname_counts = Counter([p for p in policy_names if p])
    n_sources_with_all_policies = 0
    if has_source_index and expected_num_policies > 0:
        grouped = metadata_table.groupby("source_index")
        for _, gdf in grouped:
            pset = set(int(v) for v in gdf["policy_id"].dropna().tolist())
            if len(pset) >= expected_num_policies:
                n_sources_with_all_policies += 1

    summary = {
        "run_id": str(uuid.uuid4())[:8],
        "data_dir": args.data_dir,
        "embedding_path": embedding_path,
        "meta_path": meta_path,
        "traj_path": traj_path,
        "front_path": front_path,
        "split_path": split_path,
        "source_key_fields": source_key_fields,
        "query_index": query_idx,
        "query_meta": query_meta,
        "query_source_key": q_source,
        "query_policy_id": policy_ids[query_idx],
        "query_policy_name": policy_names[query_idx] if policy_names[query_idx] else (_policy_name_for(policy_ids[query_idx], policy_mapping) if policy_ids[query_idx] is not None else None),
        "query_policy_label": _policy_label(policy_ids[query_idx], policy_names[query_idx] if policy_names[query_idx] else _policy_name_for(policy_ids[query_idx], policy_mapping)),
        "query_source_index": int(q_source_index) if (q_source_index is not None and pd.notna(q_source_index)) else None,
        "case_selection": case_selection_info,
        "projection": args.projection,
        "policy_mapping": {str(k): str(v) for k, v in sorted(policy_mapping.items())},
        "policy_id_source": pid_source,
        "split_filter": args.split,
        "distance": args.distance,
        "topk": args.topk,
        "dt": args.dt,
        "within_source": within,
        "global_retrieval": {
            "enabled": args.mode in ("global", "both"),
            "candidate_scope": "selected_split",
            "exclude_self": args.exclude_self,
            "exclude_same_source": args.exclude_same_source,
            "exclude_same_scenario": args.exclude_same_scenario,
            "n_candidates_before_filter": int(len(selected)),
            "n_candidates_after_filter": int((~retrieval_df.get("excluded", pd.Series([], dtype=bool))).sum()) if len(retrieval_df) else 0,
            "definition": "Top-K nearest by embedding distance after exclusion filters.",
            "hit_at_1_same_policy": hit1,
            "hit_at_k_same_policy": hitk,
            "num_same_policy_in_topk": topk_same_policy,
            "reason_if_hit_rate_unavailable": hit_reason,
        },
        "diagnostics": {
            "n_total_rows": int(n),
            "n_rows_after_split": int(len(selected)),
            "n_unique_source_keys_total": int(len(group_sizes_total_key)),
            "n_unique_source_keys_after_split": int(len(group_sizes_after_key)),
            "source_group_size_histogram_total": {str(k): int(v) for k, v in sorted(group_hist_total_key.items())},
            "source_group_size_histogram_after_split": {str(k): int(v) for k, v in sorted(group_hist_after_key.items())},
            "source_group_size_histogram_by_source_index": {str(k): int(v) for k, v in sorted(group_hist_after_source_idx.items())},
            "source_group_size_histogram_by_source_key": {str(k): int(v) for k, v in sorted(group_hist_after_key.items())},
            "has_source_index": has_source_index,
            "source_index_source": metadata_sources["source_index_source"],
            "has_policy_id": has_policy_id,
            "policy_id_source": pid_source,
            "policy_id_counts": {str(k): int(v) for k, v in sorted(policy_counts.items())},
            "policy_name_counts": {str(k): int(v) for k, v in sorted(pname_counts.items())},
            "n_sources_with_all_policies": int(n_sources_with_all_policies),
            "expected_num_policies": int(expected_num_policies),
            "policy_ids_observed": [int(x) for x in policy_ids_observed],
            "split_array_shape": list(split.shape) if split is not None else None,
            "embedding_shape": list(np.asarray(embeddings).shape),
            "meta_shape": list(np.asarray(meta).shape),
            "traj_shape": list(np.asarray(traj).shape),
            "front_shape": list(np.asarray(front).shape) if front is not None else None,
        },
        "style_signal_definitions": {
            "speed": "sqrt(vx^2 + vy^2)",
            "accel": "finite difference of speed / dt",
            "jerk": "finite difference of accel / dt",
            "heading": "unwrap(atan2(vy, vx))",
            "yaw_rate_proxy": "finite difference of heading / dt",
            "curvature_proxy": "yaw_rate_proxy / max(speed, eps)",
            "gap": "distance(front_xy - ego_xy), if front.npy available",
            "thw": "gap / max(speed, eps), if gap available",
            "limitations": "yaw_rate_proxy and curvature_proxy are approximate proxies from velocity-direction signals.",
        },
        "warnings": warnings,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _generate_report(out_dir / "interpretability_report.md", summary, out_dir / "retrieval_table.csv")

    if args.smoke_test:
        must = ["summary.json", "retrieval_table.csv", "style_fingerprint.csv"]
        for p in must:
            if not (out_dir / p).exists():
                raise RuntimeError(f"smoke_test failed: missing {p}")
        plots = list(out_dir.glob("*.png"))
        if len(plots) < 1:
            raise RuntimeError("smoke_test failed: no plot generated")


def parse_args():
    p = argparse.ArgumentParser(description="Embedding interpretability demo (within-source + global retrieval)")
    p.add_argument("--data_dir", type=str, required=False, default="")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--embedding", type=str, default="feat_style", choices=["feat_style", "feat_style_raw", "feat", "feat_legacy"])
    p.add_argument("--query_index", type=int, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "cosine"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--mode", type=str, default="both", choices=["within_source", "global", "both"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--exclude_self", action="store_true", default=True)
    g.add_argument("--include_self", action="store_true")
    p.add_argument("--exclude_same_source", action="store_true", default=True)
    p.add_argument("--exclude_same_scenario", action="store_true", default=True)
    p.add_argument("--max_signal_topk", type=int, default=3)
    p.add_argument("--source_key_fields", type=str, default="scenario_id,start,window_len,front_id")
    p.add_argument("--auto_select_valid_source", action="store_true")
    p.add_argument("--projection", type=str, default="pca", choices=["pca", "umap", "both"])
    p.add_argument("--case_selection", type=str, default="first_valid", choices=["query_index", "first_valid", "best_hit_at_k", "best_p2_separation", "best_human_readable"])
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()
    if args.include_self:
        args.exclude_self = False
    return args


if __name__ == "__main__":
    run_demo(parse_args())
