#!/usr/bin/env python3
"""Embedding interpretability demo for trajectory-level driving style embeddings."""

import argparse
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


def _infer_policy_ids(meta: np.ndarray, selected_indices: np.ndarray) -> Tuple[List[Optional[int]], str, List[str]]:
    warnings = []
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
        groups.setdefault(source_key(meta[i]), []).append(i)
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


def _plot_within_triplet(path: Path, query_idx: int, indices: Sequence[int], traj: np.ndarray, front: Optional[np.ndarray], policy_ids: List[Optional[int]], skey: str):
    q = _to_traj_array(traj[query_idx])
    plt.figure(figsize=(8, 6))
    for i in indices:
        t = _to_traj_array(traj[i])
        xy = _align_xy(t, q)
        pid = policy_ids[i]
        label = f"idx={i}, p={pid if pid is not None else 'NA'}"
        lw = 2.5 if pid in (0, 1, 2) and len(indices) == 3 else 1.8
        plt.plot(xy[:, 0], xy[:, 1], label=label, linewidth=lw)
    if front is not None:
        fq = _to_traj_array(front[query_idx])
        fxy = _align_xy(fq, q)
        plt.plot(fxy[:, 0], fxy[:, 1], "k--", alpha=0.6, label="front(query)")
    plt.xlabel("x local (m)")
    plt.ylabel("y local (m)")
    plt.title(f"Within-source trajectories | source_key={skey} | query={query_idx}")
    plt.legend(fontsize=8)
    _savefig(path)


def _plot_within_signals(path: Path, indices: Sequence[int], traj: np.ndarray, front: Optional[np.ndarray], policy_ids: List[Optional[int]], dt: float):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    keys = ["speed", "accel", "jerk", "yaw_rate_proxy"]
    for i in indices:
        sig = _compute_signals(_to_traj_array(traj[i]), _to_traj_array(front[i]) if front is not None else None, dt)
        label = f"idx={i}, p={policy_ids[i] if policy_ids[i] is not None else 'NA'}"
        for ax, k in zip(axes.flat, keys):
            ax.plot(sig[k], label=label)
            ax.set_title(k)
            ax.grid(alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    _savefig(path)


def _plot_distance_matrix(path: Path, indices: Sequence[int], emb: np.ndarray, policy_ids: List[Optional[int]], distance: str):
    n = len(indices)
    mat = np.zeros((n, n), dtype=float)
    for r, i in enumerate(indices):
        for c, j in enumerate(indices):
            mat[r, c] = _dist(emb[i], emb[j], distance)
    labels = [f"p{policy_ids[i]}|{i}" if policy_ids[i] is not None else f"i{i}" for i in indices]
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label=f"{distance} distance")
    plt.xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(n), labels, fontsize=8)
    plt.title("Within-source embedding distance matrix")
    _savefig(path)
    return mat


def _plot_cards(path: Path, query_idx: int, top_df: pd.DataFrame, traj: np.ndarray, front: Optional[np.ndarray], policy_ids: List[Optional[int]], stats: Dict[int, Dict[str, float]]):
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
        ax_traj.set_title(f"{name}: idx={idx}, p={policy_ids[idx] if policy_ids[idx] is not None else 'NA'}, d={dval:.4f}")
        ax_traj.grid(alpha=0.3)
        ax_speed.plot(sig["speed"])
        st = stats[idx]
        txt = f"mean_v={st['mean_speed']:.2f}\nrms_jerk={st['rms_jerk']:.2f}\nrms_yaw={st['rms_yaw_rate_proxy']:.3f}\nrms_curv={st['rms_curvature_proxy']:.3f}"
        if "mean_thw" in st:
            txt += f"\nmean_thw={st['mean_thw']:.2f}"
        ax_speed.text(0.02, 0.98, txt, va="top", ha="left", transform=ax_speed.transAxes, fontsize=8)
        ax_speed.set_title("Speed profile")
        ax_speed.grid(alpha=0.3)
    _savefig(path)


def _plot_global_signals(path: Path, query_idx: int, top_df: pd.DataFrame, traj: np.ndarray, front: Optional[np.ndarray], policy_ids: List[Optional[int]], dt: float, max_signal_topk: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    keys = ["speed", "jerk", "yaw_rate_proxy", "curvature_proxy"]
    qsig = _compute_signals(_to_traj_array(traj[query_idx]), _to_traj_array(front[query_idx]) if front is not None else None, dt)
    for ax, k in zip(axes.flat, keys):
        ax.plot(qsig[k], label=f"query idx={query_idx}", linewidth=2.3)
    for _, r in top_df.head(max_signal_topk).iterrows():
        idx = int(r["index"])
        sig = _compute_signals(_to_traj_array(traj[idx]), _to_traj_array(front[idx]) if front is not None else None, dt)
        lbl = f"r{int(r['rank'])} idx={idx} p={policy_ids[idx] if policy_ids[idx] is not None else 'NA'} d={float(r['distance']):.3f}"
        for ax, k in zip(axes.flat, keys):
            ax.plot(sig[k], label=lbl)
            ax.set_title(k)
            ax.grid(alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    _savefig(path)


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


def run_demo(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    if args.smoke_test:
        embeddings, meta, traj, front, split = _make_synth()
        embedding_path = "<synthetic>"
        meta_path = traj_path = front_path = split_path = "<synthetic>"
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
    split_arr = _get_split(split, n)
    if args.split == "all":
        selected = np.arange(n)
    else:
        selected = np.where(split_arr == args.split)[0]
    if len(selected) == 0:
        raise ValueError(f"No samples for split={args.split}")

    query_idx = args.query_index if args.query_index is not None else int(selected.min())
    if query_idx not in selected:
        raise ValueError("query_index is not in selected split")

    policy_ids, pid_source, pid_warn = _infer_policy_ids(meta, selected)
    warnings.extend(pid_warn)

    query_meta = _extract_meta_dict(meta[query_idx])
    q_source = source_key(meta[query_idx])

    # Precompute stats
    stats: Dict[int, Dict[str, float]] = {}
    for i in selected.tolist():
        sig = _compute_signals(_to_traj_array(traj[i]), _to_traj_array(front[i]) if front is not None else None, args.dt)
        stats[i] = _summary_stats(sig)

    retrieval_rows = []
    top_rows = []
    if args.mode in ("global", "both"):
        for i in selected.tolist():
            if i == query_idx and args.exclude_self:
                excluded, reason = True, "self"
            else:
                excluded, reason = False, ""
            same_source = source_key(meta[i]) == q_source
            same_scenario = _extract_meta_dict(meta[i])["scenario_id"] == query_meta["scenario_id"]
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
                "index": i,
                "policy_id": pid,
                "source_key": source_key(meta[i]),
                "scenario_id": _extract_meta_dict(meta[i])["scenario_id"],
                "start": _extract_meta_dict(meta[i])["start"],
                "window_len": _extract_meta_dict(meta[i])["window_len"],
                "front_id": _extract_meta_dict(meta[i])["front_id"],
                "split": split_arr[i],
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
        "applicable": False,
        "reason_if_not_applicable": None,
        "pairwise_embedding_distances": None,
    }
    selected_for_fingerprint = {query_idx}

    if within["enabled"]:
        ws = [i for i in selected.tolist() if source_key(meta[i]) == q_source]
        within["available_indices"] = ws
        within["available_policy_ids"] = [policy_ids[i] for i in ws]
        selected_for_fingerprint.update(ws)
        if len(ws) >= 2:
            within["applicable"] = True
            _plot_within_triplet(out_dir / "within_source_triplet.png", query_idx, ws, traj, front, policy_ids, q_source)
            _plot_within_signals(out_dir / "within_source_style_signals.png", ws, traj, front, policy_ids, args.dt)
            mat = _plot_distance_matrix(out_dir / "embedding_distance_matrix.png", ws, embeddings, policy_ids, args.distance)
            within["pairwise_embedding_distances"] = {
                f"{ws[r]}->{ws[c]}": float(mat[r, c]) for r in range(len(ws)) for c in range(len(ws))
            }
        else:
            msg = "within-source not applicable: fewer than 2 samples under query source_key"
            warnings.append(msg)
            within["reason_if_not_applicable"] = msg

    if args.mode in ("global", "both"):
        if len(top_df) > 0:
            selected_for_fingerprint.update(top_df["index"].astype(int).tolist())
            _plot_cards(out_dir / "global_retrieval_cards.png", query_idx, top_df, traj, front, policy_ids, stats)
            _plot_global_signals(out_dir / "global_retrieval_style_signals.png", query_idx, top_df, traj, front, policy_ids, args.dt, args.max_signal_topk)
        else:
            warnings.append("global retrieval plot skipped: no non-excluded candidates")

    fp_rows = []
    for i in sorted(selected_for_fingerprint):
        m = _extract_meta_dict(meta[i])
        row = {
            "index": i,
            "source_key": source_key(meta[i]),
            "scenario_id": m["scenario_id"],
            "start": m["start"],
            "window_len": m["window_len"],
            "front_id": m["front_id"],
            "split": split_arr[i],
            "policy_id": policy_ids[i],
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

    summary = {
        "run_id": str(uuid.uuid4())[:8],
        "data_dir": args.data_dir,
        "embedding_path": embedding_path,
        "meta_path": meta_path,
        "traj_path": traj_path,
        "front_path": front_path,
        "split_path": split_path,
        "query_index": query_idx,
        "query_meta": query_meta,
        "query_source_key": q_source,
        "query_policy_id": policy_ids[query_idx],
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
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()
    if args.include_self:
        args.exclude_self = False
    return args


if __name__ == "__main__":
    run_demo(parse_args())
