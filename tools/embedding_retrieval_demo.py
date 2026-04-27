"""Embedding interpretability demo: interactive retrieval + trajectory replay.

Visually verifies that embeddings cluster/separate driving styles into different
regions by:
  1) Selecting a query window (by index or meta fields)
  2) Retrieving the Top-K most-similar windows in embedding space
  3) Overlaying ego/front trajectories (aligned to query frame)
  4) Plotting time-series style signals: speed, acceleration, jerk, curvature proxy

Data requirements (all arrays must have aligned row indices):
  embeddings : (N, D) float32  –  feat_style.npy by default
  meta       : (N, 4) object   –  (scenario_id, start, window_len, front_id)
  traj       : (N,)  object    –  each element (T, 4): center_x, center_y, vx, vy
  front      : (N,)  object    –  same layout as traj (lead vehicle)
  split      : (N,)  object    –  "train"/"val"/"test"

Retrieval modes
  --mode global         query against all items in selected split
  --mode within-source  restrict candidates to the same meta-key group
                        (same scenario_id, start, window_len, front_id)

CLI usage examples
  # Global retrieval – default settings
  python tools/embedding_retrieval_demo.py \\
      --emb_path   output_policy_rollouts/feat_style.npy \\
      --meta_path  output_policy_rollouts/meta.npy \\
      --traj_path  output_policy_rollouts/traj.npy \\
      --front_path output_policy_rollouts/front.npy \\
      --split_path output_policy_rollouts/split.npy \\
      --query_index 0 --topk 5

  # Within-source retrieval
  python tools/embedding_retrieval_demo.py \\
      --emb_path   output_policy_rollouts/feat_style.npy \\
      --meta_path  output_policy_rollouts/meta.npy \\
      --traj_path  output_policy_rollouts/traj.npy \\
      --front_path output_policy_rollouts/front.npy \\
      --split_path output_policy_rollouts/split.npy \\
      --query_index 0 --mode within-source

  # Smoke test (no real data needed)
  python tools/embedding_retrieval_demo.py --smoke_test
"""

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAJ_COLS = ["center_x", "center_y", "velocity_x", "velocity_y"]
EPS = 1e-8


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Trajectory-derived signals
# ---------------------------------------------------------------------------

def compute_traj_signals(traj: np.ndarray, dt: float = 0.1) -> Dict[str, np.ndarray]:
    """Compute speed, accel, jerk, and curvature proxy from (T, ≥4) array.

    Columns assumed: [center_x, center_y, vx, vy, ...].
    All derivatives are approximate finite differences; curvature is a proxy
    (yaw_rate / max(speed, EPS)) and is explicitly labelled as such.
    """
    vx = traj[:, 2].astype(float)
    vy = traj[:, 3].astype(float)
    speed = np.hypot(vx, vy)

    # accel ≈ diff(speed) / dt
    accel = np.gradient(speed, dt)

    # jerk ≈ diff(accel) / dt
    jerk = np.gradient(accel, dt)

    # curvature proxy: yaw_rate / speed
    heading = np.arctan2(vy, vx)
    # wrap heading differences to [-π, π]
    d_heading = np.diff(heading)
    d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = np.concatenate([[0.0], d_heading / dt])
    curvature_proxy = yaw_rate / np.maximum(speed, EPS)

    return {
        "speed": speed,
        "accel": accel,
        "jerk": jerk,
        "curvature_proxy": curvature_proxy,
    }


# ---------------------------------------------------------------------------
# Trajectory alignment
# ---------------------------------------------------------------------------

def _align_traj(traj: np.ndarray, ref_traj: np.ndarray) -> np.ndarray:
    """Translate and rotate *traj* so that ref_traj's first point is origin
    and ref_traj's initial velocity vector points along +x.

    Both arrays have shape (T, ≥4): [x, y, vx, vy, ...].
    Returns the aligned copy of *traj* (same shape).
    """
    aligned = traj.copy().astype(float)
    # Translation: offset by query first-point position
    tx = ref_traj[0, 0]
    ty = ref_traj[0, 1]
    aligned[:, 0] -= tx
    aligned[:, 1] -= ty

    # Rotation: align query initial velocity to +x
    vx0, vy0 = ref_traj[0, 2], ref_traj[0, 3]
    angle = math.atan2(vy0, vx0)  # angle of initial heading
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    x_rot = aligned[:, 0] * cos_a - aligned[:, 1] * sin_a
    y_rot = aligned[:, 0] * sin_a + aligned[:, 1] * cos_a
    aligned[:, 0] = x_rot
    aligned[:, 1] = y_rot
    return aligned


# ---------------------------------------------------------------------------
# Retrieval core
# ---------------------------------------------------------------------------

def _meta_key(meta_row) -> tuple:
    """Return the grouping key from a meta row: (scenario_id, start, window_len, front_id)."""
    return (str(meta_row[0]), int(meta_row[1]), int(meta_row[2]), str(meta_row[3]))


def retrieve(
    query_idx: int,
    embeddings: np.ndarray,
    meta: np.ndarray,
    split: np.ndarray,
    split_filter: str,
    mode: str,
    distance: str,
    topk: int,
    exclude_same_scenario: bool,
    exclude_same_source: bool,
) -> pd.DataFrame:
    """Return a DataFrame with retrieval results sorted by distance.

    Columns: index, scenario_id, start, window_len, front_id, distance, excluded.
    The query row itself is always excluded from results.

    Args:
        query_idx: row index of the query.
        embeddings: (N, D) array.
        meta: (N, 4) object array, columns: scenario_id, start, window_len, front_id.
        split: (N,) array of split labels.
        split_filter: only consider rows whose split matches this value.
        mode: "global" or "within-source".
        distance: "euclidean" or "cosine".
        topk: number of results to return.
        exclude_same_scenario: if True, mark same-scenario rows as excluded.
        exclude_same_source: if True, mark same meta-key (same source group) as excluded.
    """
    dist_fn = _euclidean if distance == "euclidean" else _cosine_dist
    q_emb = embeddings[query_idx]
    q_meta_key = _meta_key(meta[query_idx])
    q_scenario = str(meta[query_idx][0])

    rows = []
    split_mask = np.array([str(s) == split_filter for s in split])

    if mode == "within-source":
        # Candidates: same meta-key, different row
        candidates = [
            i for i in range(len(embeddings))
            if _meta_key(meta[i]) == q_meta_key and i != query_idx
        ]
    else:
        # Global: all rows in the requested split, excluding query itself
        candidates = [i for i in range(len(embeddings)) if split_mask[i] and i != query_idx]

    for i in candidates:
        m = meta[i]
        meta_key_i = _meta_key(m)
        excluded = False
        if exclude_same_scenario and str(m[0]) == q_scenario:
            excluded = True
        if exclude_same_source and meta_key_i == q_meta_key:
            excluded = True

        dist = dist_fn(q_emb, embeddings[i])
        rows.append({
            "index": i,
            "scenario_id": str(m[0]),
            "start": int(m[1]),
            "window_len": int(m[2]),
            "front_id": str(m[3]),
            "distance": dist,
            "excluded": excluded,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df_sorted = df.sort_values("distance").reset_index(drop=True)
    # Return topk non-excluded rows (if enough exist), but include all excluded for CSV
    non_excl = df_sorted[~df_sorted["excluded"]]
    excl = df_sorted[df_sorted["excluded"]]
    top_non_excl = non_excl.head(topk)
    result = pd.concat([top_non_excl, excl], ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _traj_as_array(t) -> np.ndarray:
    return np.asarray(t, dtype=float)


def plot_traj_overlay(
    query_idx: int,
    result_df: pd.DataFrame,
    traj: np.ndarray,
    front: np.ndarray,
    meta: np.ndarray,
    out_path: str,
    topk: int,
    quiet: bool = False,
) -> None:
    """Overlay ego + front trajectories (query vs retrieved), aligned to query frame."""
    q_traj_raw = _traj_as_array(traj[query_idx])
    q_front_raw = _traj_as_array(front[query_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_ego, ax_front = axes

    def _plot_one(ax, traj_arr_raw, ref_arr, label, color, alpha=0.8, lw=1.5, zorder=2):
        aligned = _align_traj(traj_arr_raw, ref_arr)
        ax.plot(aligned[:, 0], aligned[:, 1], color=color, alpha=alpha, lw=lw,
                label=label, zorder=zorder)
        ax.plot(aligned[0, 0], aligned[0, 1], "o", color=color, ms=4, zorder=zorder + 1)

    # Query in black
    q_aligned = _align_traj(q_traj_raw, q_traj_raw)
    ax_ego.plot(q_aligned[:, 0], q_aligned[:, 1], color="black", lw=2.5,
                label="query (ego)", zorder=10)
    ax_ego.plot(q_aligned[0, 0], q_aligned[0, 1], "ko", ms=6, zorder=11)

    q_front_aligned = _align_traj(q_front_raw, q_traj_raw)
    ax_front.plot(q_front_aligned[:, 0], q_front_aligned[:, 1], color="black", lw=2.5,
                  label="query (front)", zorder=10)
    ax_front.plot(q_front_aligned[0, 0], q_front_aligned[0, 1], "ko", ms=6, zorder=11)

    cmap = plt.cm.plasma  # type: ignore[attr-defined]
    non_excl = result_df[~result_df["excluded"]].head(topk)
    n_show = len(non_excl)

    for rank, (_, row) in enumerate(non_excl.iterrows()):
        idx = int(row["index"])
        color = cmap(0.2 + 0.6 * rank / max(n_show - 1, 1))
        alpha = max(0.3, 0.85 - rank * 0.1)
        sc_id = str(row["scenario_id"])[:12]
        lbl = f"rank{rank + 1} d={row['distance']:.3f} ({sc_id})"
        t_arr = _traj_as_array(traj[idx])
        f_arr = _traj_as_array(front[idx])
        _plot_one(ax_ego, t_arr, q_traj_raw, lbl, color, alpha=alpha)
        _plot_one(ax_front, f_arr, q_traj_raw, lbl, color, alpha=alpha)

    q_sc = str(meta[query_idx][0])[:20]
    q_start = int(meta[query_idx][1])

    for ax, title in [(ax_ego, f"Ego trajectories (aligned)\nquery idx={query_idx} sc={q_sc} start={q_start}"),
                      (ax_front, "Front (lead) trajectories (aligned)")]:
        ax.set_xlabel("x [m] (aligned)")
        ax.set_ylabel("y [m] (aligned)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    if not quiet:
        print(f"  Saved trajectory overlay → {out_path}")


def plot_timeseries(
    query_idx: int,
    result_df: pd.DataFrame,
    traj: np.ndarray,
    meta: np.ndarray,
    out_path: str,
    topk: int,
    dt: float = 0.1,
    quiet: bool = False,
) -> None:
    """Plot speed/accel/jerk/curvature proxy for query and Top-K retrieved windows."""
    signal_names = ["speed [m/s]", "accel [m/s²]", "jerk [m/s³]", "curvature proxy [1/m] (approx)"]
    signal_keys = ["speed", "accel", "jerk", "curvature_proxy"]

    q_sig = compute_traj_signals(_traj_as_array(traj[query_idx]), dt=dt)
    T = len(q_sig["speed"])
    t_axis = np.arange(T) * dt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    cmap = plt.cm.plasma  # type: ignore[attr-defined]
    non_excl = result_df[~result_df["excluded"]].head(topk)
    n_show = len(non_excl)

    for ax, key, ylabel in zip(axes, signal_keys, signal_names):
        ax.plot(t_axis, q_sig[key], color="black", lw=2.5, label="query", zorder=10)

        for rank, (_, row) in enumerate(non_excl.iterrows()):
            idx = int(row["index"])
            sig = compute_traj_signals(_traj_as_array(traj[idx]), dt=dt)
            T_i = len(sig[key])
            t_i = np.arange(T_i) * dt
            color = cmap(0.2 + 0.6 * rank / max(n_show - 1, 1))
            alpha = max(0.3, 0.8 - rank * 0.1)
            lbl = f"rank{rank + 1} (d={row['distance']:.3f})"
            ax.plot(t_i, sig[key], color=color, alpha=alpha, lw=1.2, label=lbl)

        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time [s]")
    axes[0].legend(fontsize=7, loc="upper right")
    q_sc = str(meta[query_idx][0])[:20]
    fig.suptitle(
        f"Style signals — query idx={query_idx} sc={q_sc}\n"
        "(curvature proxy = yaw_rate / speed, approximate)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    if not quiet:
        print(f"  Saved time-series plot  → {out_path}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding interpretability demo: retrieval + trajectory replay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument("--emb_path", default="output_policy_rollouts/feat_style.npy",
                   help="Path to embedding .npy (feat_style.npy / feat_style_raw.npy / feat.npy)")
    p.add_argument("--meta_path", default="output_policy_rollouts/meta.npy",
                   help="Path to meta.npy; rows are (scenario_id, start, window_len, front_id)")
    p.add_argument("--traj_path", default="output_policy_rollouts/traj.npy",
                   help="Path to ego trajectory .npy")
    p.add_argument("--front_path", default="output_policy_rollouts/front.npy",
                   help="Path to front (lead) vehicle trajectory .npy")
    p.add_argument("--split_path", default=None,
                   help="Path to split.npy; if omitted all rows are treated as 'all'")

    # Query selection
    p.add_argument("--query_index", type=int, default=0,
                   help="Row index of the query window (primary selector)")
    p.add_argument("--query_scenario_id", default=None,
                   help="Find first row whose meta[0] matches this scenario_id")
    p.add_argument("--query_start", type=int, default=None,
                   help="When combined with --query_scenario_id, also match meta[1]==start")

    # Retrieval settings
    p.add_argument("--mode", choices=["global", "within-source"], default="global",
                   help="Retrieval mode")
    p.add_argument("--split_filter", default="test",
                   help="Restrict global retrieval to this split (train/val/test/all)")
    p.add_argument("--topk", type=int, default=5,
                   help="Number of top-K results to return (non-excluded)")
    p.add_argument("--distance", choices=["euclidean", "cosine"], default="euclidean",
                   help="Distance metric")
    p.add_argument("--exclude_same_scenario", action="store_true",
                   help="(global mode) Exclude rows with same scenario_id as query")
    p.add_argument("--exclude_same_source", action="store_true",
                   help="(global mode) Exclude rows with same (scenario_id,start,window_len,front_id)")

    # Output
    p.add_argument("--output_dir", default="outputs",
                   help="Parent output directory; results go into <output_dir>/<run_id>/")
    p.add_argument("--run_id", default=None,
                   help="Sub-directory name for this run (default: auto-generated UUID prefix)")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Time step [s] for derivative computations")

    # Misc
    p.add_argument("--quiet", action="store_true", help="Suppress informational output")
    p.add_argument("--verbose", action="store_true", help="Extra diagnostic output")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run a self-contained smoke test with synthetic data (no files needed)")

    return p.parse_args(argv)


def _resolve_query_index(args, meta: np.ndarray) -> int:
    """Return the query row index, resolving --query_scenario_id / --query_start if provided."""
    if args.query_scenario_id is not None:
        for i, m in enumerate(meta):
            if str(m[0]) == args.query_scenario_id:
                if args.query_start is None or int(m[1]) == args.query_start:
                    return i
        raise ValueError(
            f"No row found with scenario_id={args.query_scenario_id!r}"
            + (f" start={args.query_start}" if args.query_start is not None else "")
        )
    return args.query_index


def run_demo(args: argparse.Namespace) -> str:
    """Execute the retrieval demo; returns the output directory path."""
    if not args.quiet:
        print("Loading data …")

    embeddings = np.load(args.emb_path, allow_pickle=True)
    if embeddings.ndim == 1:
        # Could be object array of rows; try stacking
        embeddings = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    embeddings = embeddings.astype(np.float32)

    meta = np.load(args.meta_path, allow_pickle=True)
    traj = np.load(args.traj_path, allow_pickle=True)
    front = np.load(args.front_path, allow_pickle=True)

    N = len(embeddings)
    if not (len(meta) == len(traj) == len(front) == N):
        raise ValueError(
            f"Array length mismatch: embeddings={N}, meta={len(meta)}, "
            f"traj={len(traj)}, front={len(front)}"
        )

    if args.split_path is not None:
        split = np.load(args.split_path, allow_pickle=True)
        if len(split) != N:
            raise ValueError(f"split length {len(split)} != embeddings length {N}")
        split = np.array([str(s) for s in split])
    else:
        split = np.array(["all"] * N)
        if not args.quiet:
            print("  No split_path provided; treating all rows as split='all'.")

    split_filter = args.split_filter
    if args.split_path is None:
        split_filter = "all"

    query_idx = _resolve_query_index(args, meta)
    if query_idx < 0 or query_idx >= N:
        raise IndexError(f"query_index={query_idx} out of range [0, {N})")

    if not args.quiet:
        q_meta = meta[query_idx]
        print(f"  Query: idx={query_idx}  scenario_id={q_meta[0]}  "
              f"start={q_meta[1]}  window_len={q_meta[2]}  front_id={q_meta[3]}")
        print(f"  Mode: {args.mode}  distance: {args.distance}  topk: {args.topk}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    result_df = retrieve(
        query_idx=query_idx,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter=split_filter,
        mode=args.mode,
        distance=args.distance,
        topk=args.topk,
        exclude_same_scenario=args.exclude_same_scenario,
        exclude_same_source=args.exclude_same_source,
    )

    if result_df.empty:
        print("  WARNING: no candidates found for this query / mode combination.")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    run_id = args.run_id if args.run_id else uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"  Writing outputs to: {out_dir}/")

    # ------------------------------------------------------------------
    # retrieval_table.csv
    # ------------------------------------------------------------------
    csv_path = out_dir / "retrieval_table.csv"
    result_df.to_csv(csv_path, index=False)
    if not args.quiet:
        print(f"  Saved retrieval table  → {csv_path}")
        if args.verbose and not result_df.empty:
            non_excl = result_df[~result_df["excluded"]]
            print(non_excl[["index", "scenario_id", "distance", "excluded"]].head(args.topk).to_string(index=False))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not result_df.empty:
        plot_traj_overlay(
            query_idx=query_idx,
            result_df=result_df,
            traj=traj,
            front=front,
            meta=meta,
            out_path=str(out_dir / "traj_overlay.png"),
            topk=args.topk,
            quiet=args.quiet,
        )
        plot_timeseries(
            query_idx=query_idx,
            result_df=result_df,
            traj=traj,
            meta=meta,
            out_path=str(out_dir / "timeseries.png"),
            topk=args.topk,
            dt=args.dt,
            quiet=args.quiet,
        )
    else:
        if not args.quiet:
            print("  Skipping plots (no retrieval results).")

    # ------------------------------------------------------------------
    # summary.json
    # ------------------------------------------------------------------
    q_meta = meta[query_idx]
    summary = {
        "run_id": run_id,
        "query_index": query_idx,
        "query_meta": {
            "scenario_id": str(q_meta[0]),
            "start": int(q_meta[1]),
            "window_len": int(q_meta[2]),
            "front_id": str(q_meta[3]),
        },
        "mode": args.mode,
        "mode_note": (
            "within-source groups rows by (scenario_id, start, window_len, front_id); "
            "policy_id is NOT stored so all rows sharing the same meta-key are treated "
            "as 'same source, possibly different policy'."
            if args.mode == "within-source"
            else "global retrieval against all rows in the selected split."
        ),
        "split_filter": split_filter,
        "distance": args.distance,
        "topk": args.topk,
        "exclude_same_scenario": args.exclude_same_scenario,
        "exclude_same_source": args.exclude_same_source,
        "n_candidates": int(len(result_df)),
        "n_non_excluded": int((~result_df["excluded"]).sum()) if not result_df.empty else 0,
        "emb_path": str(args.emb_path),
        "meta_path": str(args.meta_path),
        "traj_path": str(args.traj_path),
        "front_path": str(args.front_path),
        "split_path": str(args.split_path),
        "dt": args.dt,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(f"  Saved summary           → {summary_path}")

    return str(out_dir)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_items: int = 60,
    emb_dim: int = 8,
    T: int = 20,
    n_groups: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, meta, traj, front, split) as dummy NumPy arrays."""
    rng = np.random.default_rng(seed)

    # 3 policy clusters in embedding space
    n_policies = 3
    centres = np.zeros((n_policies, emb_dim), dtype=np.float32)
    for p in range(n_policies):
        centres[p, p % emb_dim] = 5.0 * (p + 1)

    embeddings = np.zeros((n_items, emb_dim), dtype=np.float32)
    meta = np.empty(n_items, dtype=object)
    traj_list = []
    front_list = []
    split_list = []

    for i in range(n_items):
        group_id = i % n_groups  # source group index
        policy_id = i % n_policies
        embeddings[i] = centres[policy_id] + rng.standard_normal(emb_dim).astype(np.float32) * 0.1
        meta[i] = (f"scenario_{group_id:03d}", group_id * 5, T, f"front_{group_id:03d}")

        # Simple synthetic trajectory: roughly straight with slight lateral variation
        speed = 10.0 + rng.uniform(-2, 2)
        x0, y0 = float(rng.uniform(0, 50)), float(rng.uniform(-2, 2))
        t_arr = np.zeros((T, 4), dtype=np.float32)
        f_arr = np.zeros((T, 4), dtype=np.float32)
        for t in range(T):
            t_arr[t] = [x0 + speed * t * 0.1, y0, speed, 0.0]
            f_arr[t] = [x0 + 30.0 + (speed - 1.0) * t * 0.1, y0, speed - 1.0, 0.0]
        traj_list.append(t_arr)
        front_list.append(f_arr)
        split_list.append("test" if group_id % 5 == 0 else "train")

    traj_arr = np.empty(n_items, dtype=object)
    front_arr = np.empty(n_items, dtype=object)
    for i in range(n_items):
        traj_arr[i] = traj_list[i]
        front_arr[i] = front_list[i]
    split_arr = np.array(split_list, dtype=object)

    return embeddings, meta, traj_arr, front_arr, split_arr


def run_smoke_test() -> None:
    """Validate retrieval shapes and plotting without requiring real data files."""
    print("Running smoke test …")
    import tempfile

    embeddings, meta, traj, front, split = _make_synthetic_data(n_items=60, T=20)
    N, D = embeddings.shape
    assert N == 60 and D == 8, f"Unexpected shape: {embeddings.shape}"

    # ---- retrieval shape test (global) ----
    result_df = retrieve(
        query_idx=0,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="test",
        mode="global",
        distance="euclidean",
        topk=5,
        exclude_same_scenario=False,
        exclude_same_source=False,
    )
    assert isinstance(result_df, pd.DataFrame), "retrieve() must return a DataFrame"
    assert "distance" in result_df.columns, "distance column missing"
    non_excl = result_df[~result_df["excluded"]]
    assert len(non_excl) <= 5, f"Expected ≤5 non-excluded rows, got {len(non_excl)}"
    print("  [PASS] global retrieval returns correct shape")

    # ---- retrieval shape test (within-source) ----
    result_ws = retrieve(
        query_idx=0,
        embeddings=embeddings,
        meta=meta,
        split=split,
        split_filter="test",
        mode="within-source",
        distance="cosine",
        topk=5,
        exclude_same_scenario=False,
        exclude_same_source=False,
    )
    assert isinstance(result_ws, pd.DataFrame), "within-source retrieve() must return DataFrame"
    print(f"  [PASS] within-source retrieval: {len(result_ws)} candidates")

    # ---- signal computation ----
    t_arr = np.asarray(traj[0], dtype=float)
    sigs = compute_traj_signals(t_arr, dt=0.1)
    for key in ("speed", "accel", "jerk", "curvature_proxy"):
        assert key in sigs, f"Missing signal: {key}"
        assert len(sigs[key]) == len(t_arr), f"Signal {key} length mismatch"
    print("  [PASS] compute_traj_signals returns all signals with correct length")

    # ---- alignment ----
    aligned = _align_traj(t_arr, t_arr)
    assert abs(aligned[0, 0]) < 1e-6, "Aligned first x should be 0"
    assert abs(aligned[0, 1]) < 1e-6, "Aligned first y should be 0"
    print("  [PASS] _align_traj: first point at origin")

    # ---- plotting (uses Agg backend, no display) ----
    with tempfile.TemporaryDirectory() as tmp:
        plot_traj_overlay(
            query_idx=0,
            result_df=result_df,
            traj=traj,
            front=front,
            meta=meta,
            out_path=os.path.join(tmp, "traj_overlay.png"),
            topk=5,
            quiet=True,
        )
        assert os.path.exists(os.path.join(tmp, "traj_overlay.png")), "traj_overlay.png not created"
        print("  [PASS] plot_traj_overlay runs without error")

        plot_timeseries(
            query_idx=0,
            result_df=result_df,
            traj=traj,
            meta=meta,
            out_path=os.path.join(tmp, "timeseries.png"),
            topk=5,
            dt=0.1,
            quiet=True,
        )
        assert os.path.exists(os.path.join(tmp, "timeseries.png")), "timeseries.png not created"
        print("  [PASS] plot_timeseries runs without error")

    print("\nAll smoke tests passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    if args.smoke_test:
        run_smoke_test()
        return

    run_demo(args)


if __name__ == "__main__":
    main()
