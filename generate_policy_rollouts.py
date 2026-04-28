"""Generate synthetic policy rollouts from existing Waymo log-replay windows.

For each source window (ego + front) and each policy (conservative / aggressive /
lateral_stable) a new ego trajectory is simulated using a simple rule-based
longitudinal controller with policy-specific parameters.  The front vehicle
trajectory is kept as the original log-replay window (exogenous – does not
respond to ego).

Outputs written to --output_dir:
    traj.npy           – simulated ego windows, object array (N_policies*N_src, T, 4)
    front.npy          – original front windows replicated per policy, same shape
    policy_id.npy      – int array of length N_policies*N_src (0..P-1)
    scenario_id.npy    – string scenario ids (from source metadata if available,
                         else "src_{i}")
    source_index.npy   – int array mapping each output row back to source window idx
    split.npy          – train/val/test strings; replicated from src if provided,
                         else scenario-stratified 0.8/0.1/0.1
    policy_names.json  – {0: "conservative", 1: "aggressive", 2: "lateral_stable"}

Lateral-stability knobs:
    --conservative_yaw_rate_clip   override yaw_rate_clip for conservative
    --aggressive_yaw_rate_clip     override yaw_rate_clip for aggressive
    --lateral_stable_yaw_rate_clip override yaw_rate_clip for lateral_stable
    --heading_smooth_alpha         EMA smoothing on desired heading for lateral_stable
                                   (0.0=no smoothing, closer to 1.0=stronger; default 0.45)

Lateral-stable longitudinal overrides (for quick parameter sweeps):
    --lateral_stable_thw_target    override thw_target for lateral_stable (default 1.4 s)
    --lateral_stable_jerk_limit    override jerk_limit for lateral_stable (default 0.35 m/s²/step)
    --lateral_stable_a_max         override a_max for lateral_stable (default 1.5 m/s²)
    --lateral_stable_a_min         override a_min for lateral_stable (default -2.8 m/s²)

Usage example:
    python generate_policy_rollouts.py \\
        --src_traj_path output/traj.npy \\
        --src_front_path output/front.npy \\
        --src_split_path output/split.npy \\
        --output_dir output_policy_rollouts

    # Parameter sweep to tune lateral_stable separability:
    python generate_policy_rollouts.py \\
        --src_traj_path output/traj.npy \\
        --src_front_path output/front.npy \\
        --output_dir output_policy_rollouts_sweep \\
        --lateral_stable_thw_target 1.2 \\
        --lateral_stable_jerk_limit 0.25 \\
        --lateral_stable_a_max 1.3 \\
        --lateral_stable_a_min -2.5 \\
        --lateral_stable_yaw_rate_clip 0.02 \\
        --heading_smooth_alpha 0.45
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Policy parameter tables
# ---------------------------------------------------------------------------

# Each policy is described by a dict of scalar parameters consumed by
# _simulate_ego_window().  Tuning guide:
#   thw_target   – desired time-headway [s]
#   a_max        – max longitudinal acceleration [m/s²]
#   a_min        – max deceleration (negative) [m/s²]
#   jerk_limit   – max abs change of acceleration per step [m/s²/step]
#   yaw_rate_clip – max absolute yaw-rate change per step [rad/step]
#   kp_thw       – proportional gain on THW error
#   kd_vrel      – derivative (relative-speed) gain

POLICY_PARAMS: dict[str, dict] = {
    "conservative": {
        "thw_target": 2.5,
        "a_max": 1.5,
        "a_min": -3.0,
        "jerk_limit": 0.5,
        "yaw_rate_clip": 0.05,   # tight lateral stability
        "heading_smooth_alpha": 0.0,  # no heading smoothing
        "kp_thw": 0.6,
        "kd_vrel": 0.4,
    },
    "aggressive": {
        "thw_target": 1.0,
        "a_max": 3.5,
        "a_min": -5.0,
        "jerk_limit": 2.0,
        "yaw_rate_clip": 0.20,   # loose lateral – allows quick steering
        "heading_smooth_alpha": 0.0,  # no heading smoothing
        "kp_thw": 1.5,
        "kd_vrel": 0.8,
    },
    "lateral_stable": {
        # "lateral-stable + comfortable but not conservative"
        # thw_target sits between conservative (2.5 s) and aggressive (1.0 s),
        # but longitudinal dynamics are deliberately softer than conservative to
        # create a distinct third cluster in embedding space.
        "thw_target": 1.4,
        "a_max": 1.5,
        "a_min": -2.8,
        "jerk_limit": 0.35,
        "yaw_rate_clip": 0.02,   # slightly looser than before – keeps lateral signal visible
        "heading_smooth_alpha": 0.45,  # moderate EMA smoothing – separable from conservative
        "kp_thw": 1.0,
        "kd_vrel": 0.6,
    },
}

_DT_DEFAULT = 0.1          # [s] implicit Waymo 10 Hz
_EPS_SPEED = 1e-3          # avoid division by zero in THW
_EPS_DIST = 0.5            # minimum dist to avoid collapse [m]


def _parse_src_meta_row(row: object, fallback_source_index: int, fallback_window_len: int) -> dict:
    """Parse a source metadata row into canonical fields."""
    scenario_id = f"src_{fallback_source_index}"
    start = 0
    window_len = fallback_window_len
    front_id = f"front_{fallback_source_index}"
    if isinstance(row, np.void) and row.dtype.names:
        d = {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in row.dtype.names}
    elif isinstance(row, dict):
        d = dict(row)
    elif isinstance(row, (list, tuple, np.ndarray)):
        vals = list(row)
        d = {
            "scenario_id": vals[0] if len(vals) > 0 else scenario_id,
            "start": vals[1] if len(vals) > 1 else start,
            "window_len": vals[2] if len(vals) > 2 else window_len,
            "front_id": vals[3] if len(vals) > 3 else front_id,
        }
    else:
        d = {}
    scenario_id = str(d.get("scenario_id", scenario_id))
    start = int(d.get("start", start))
    window_len = int(d.get("window_len", window_len))
    front_id = str(d.get("front_id", front_id))
    return {
        "scenario_id": scenario_id,
        "start": start,
        "window_len": window_len,
        "front_id": front_id,
    }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_ego_window(
    src_ego: np.ndarray,
    src_front: np.ndarray,
    params: dict,
    dt: float,
) -> np.ndarray:
    """Simulate a new ego trajectory for a single window.

    The simulation uses the *source* ego's first step to initialise state and
    the source ego's overall heading to keep the generated trajectory roughly
    aligned with the original direction of travel (no lane-tracking).

    Args:
        src_ego:   (T, 4)  source ego   [x, y, vx, vy]
        src_front: (T, 4)  source front [x, y, vx, vy] – exogenous log replay
        params:    policy parameter dict
        dt:        time step [s]

    Returns:
        ego_new: (T, 4)  simulated ego  [x, y, vx, vy]
    """
    T = src_ego.shape[0]
    ego_new = np.empty((T, 4), dtype=np.float32)

    # Initialise from the first frame of the source window.
    x, y = float(src_ego[0, 0]), float(src_ego[0, 1])
    vx, vy = float(src_ego[0, 2]), float(src_ego[0, 3])

    thw_target          = params["thw_target"]
    a_max               = params["a_max"]
    a_min               = params["a_min"]
    jerk_limit          = params["jerk_limit"]
    yaw_rate_clip       = params["yaw_rate_clip"]
    heading_smooth_alpha = params.get("heading_smooth_alpha", 0.0)
    kp                  = params["kp_thw"]
    kd                  = params["kd_vrel"]

    a_prev = 0.0  # previous longitudinal acceleration for jerk clipping

    # EMA state for heading smoothing (initialise to source heading at t=0)
    smoothed_heading_target = float(np.arctan2(src_ego[0, 3], src_ego[0, 2]))

    for t in range(T):
        # ----- Record current state -----
        ego_new[t, 0] = x
        ego_new[t, 1] = y
        ego_new[t, 2] = vx
        ego_new[t, 3] = vy

        if t == T - 1:
            break

        # ----- Compute current kinematics -----
        ego_speed = float(np.hypot(vx, vy))

        fx, fy = float(src_front[t, 0]), float(src_front[t, 1])
        fvx, fvy = float(src_front[t, 2]), float(src_front[t, 3])
        front_speed = float(np.hypot(fvx, fvy))

        dist = float(np.hypot(fx - x, fy - y))
        dist = max(dist, _EPS_DIST)
        thw = dist / max(ego_speed, _EPS_SPEED)

        v_rel = ego_speed - front_speed  # positive → ego faster than front

        # ----- Longitudinal controller (THW + relative-speed) -----
        thw_error = thw - thw_target          # positive → too much gap
        a_target = kp * thw_error - kd * v_rel

        # Jerk-limit: restrict change from previous acceleration
        a_delta = np.clip(a_target - a_prev, -jerk_limit, jerk_limit)
        a_cmd = np.clip(a_prev + a_delta, a_min, a_max)

        a_prev = a_cmd

        # ----- Longitudinal speed update -----
        # Accelerate/decelerate along current heading direction.
        if ego_speed > _EPS_SPEED:
            heading = float(np.arctan2(vy, vx))
        else:
            heading = float(np.arctan2(src_ego[t, 3], src_ego[t, 2]))
        speed_new = max(ego_speed + a_cmd * dt, 0.0)

        # ----- Lateral stability: limit per-step heading change -----
        # Compute desired heading from the source trajectory direction at t+1
        # (so simulated ego stays roughly aligned with the original path).
        src_head_t = float(np.arctan2(src_ego[t, 3], src_ego[t, 2]))
        if t + 1 < T:
            src_head_next = float(np.arctan2(src_ego[t + 1, 3], src_ego[t + 1, 2]))
        else:
            src_head_next = src_head_t

        # Optional EMA smoothing on the heading target (reduces responsiveness
        # to source heading changes, making lateral_stable even smoother).
        if heading_smooth_alpha > 0.0:
            # Wrap the raw target relative to the current EMA to avoid jump
            raw_delta = (src_head_next - smoothed_heading_target + np.pi) % (2 * np.pi) - np.pi
            smoothed_heading_target = smoothed_heading_target + (1.0 - heading_smooth_alpha) * raw_delta
        else:
            smoothed_heading_target = src_head_next

        desired_heading = smoothed_heading_target
        # Wrap heading delta to [-pi, pi]
        dh = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
        dh = np.clip(dh, -yaw_rate_clip, yaw_rate_clip)
        heading_new = heading + dh

        # ----- Update velocity components -----
        vx = speed_new * float(np.cos(heading_new))
        vy = speed_new * float(np.sin(heading_new))

        # ----- Position integration -----
        x += vx * dt
        y += vy * dt

    return ego_new


# ---------------------------------------------------------------------------
# Split assignment (fallback when no src_split_path given)
# ---------------------------------------------------------------------------

def _assign_split_by_hash(scenario_id: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    """Deterministic split assignment via MD5 hash (matching build_dataset.py)."""
    digest = hashlib.md5(str(scenario_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic driving-policy rollouts from existing Waymo log-replay windows."
    )
    parser.add_argument(
        "--src_traj_path",
        type=str,
        default="output/traj.npy",
        help="Path to source traj.npy (object or fixed-shape float32 array).",
    )
    parser.add_argument(
        "--src_front_path",
        type=str,
        default="output/front.npy",
        help="Path to source front.npy.",
    )
    parser.add_argument(
        "--src_split_path",
        type=str,
        default=None,
        help="Optional path to source split.npy; if provided, splits are replicated per policy.",
    )
    parser.add_argument(
        "--src_meta_path",
        type=str,
        default=None,
        help="Optional path to source meta.npy with (scenario_id, start, window_len, front_id) rows.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_policy_rollouts",
        help="Directory to write generated dataset files.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=_DT_DEFAULT,
        help=f"Time step between frames in seconds (default {_DT_DEFAULT} for Waymo 10 Hz).",
    )
    parser.add_argument(
        "--window_len",
        type=int,
        default=None,
        help="Expected window length; inferred from data if not provided.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="conservative,aggressive,lateral_stable",
        help="Comma-separated list of policy names to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used when generating fallback scenario-stratified splits).",
    )
    parser.add_argument(
        "--conservative_yaw_rate_clip",
        type=float,
        default=None,
        help="Override yaw_rate_clip for conservative policy (default: use table value 0.05 rad/step).",
    )
    parser.add_argument(
        "--aggressive_yaw_rate_clip",
        type=float,
        default=None,
        help="Override yaw_rate_clip for aggressive policy (default: use table value 0.20 rad/step).",
    )
    parser.add_argument(
        "--lateral_stable_yaw_rate_clip",
        type=float,
        default=None,
        help="Override yaw_rate_clip for lateral_stable policy (default: use table value 0.02 rad/step).",
    )
    parser.add_argument(
        "--heading_smooth_alpha",
        type=float,
        default=None,
        help=(
            "Override heading EMA smoothing factor for lateral_stable policy "
            "(0.0 = no smoothing, closer to 1.0 = stronger; default: use table value 0.45)."
        ),
    )
    parser.add_argument(
        "--lateral_stable_thw_target",
        type=float,
        default=None,
        help="Override thw_target for lateral_stable policy (default: use table value 1.4 s).",
    )
    parser.add_argument(
        "--lateral_stable_jerk_limit",
        type=float,
        default=None,
        help="Override jerk_limit for lateral_stable policy (default: use table value 0.35 m/s²/step).",
    )
    parser.add_argument(
        "--lateral_stable_a_max",
        type=float,
        default=None,
        help="Override a_max for lateral_stable policy (default: use table value 1.5 m/s²).",
    )
    parser.add_argument(
        "--lateral_stable_a_min",
        type=float,
        default=None,
        help="Override a_min for lateral_stable policy (default: use table value -2.8 m/s²).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in policy_names if p not in POLICY_PARAMS]
    if unknown:
        raise ValueError(
            f"Unknown policy name(s): {unknown}. "
            f"Available: {list(POLICY_PARAMS.keys())}"
        )

    # Apply CLI overrides to a local copy of POLICY_PARAMS
    import copy
    active_params: dict[str, dict] = copy.deepcopy(POLICY_PARAMS)
    _clip_overrides: dict[str, float | None] = {}
    for p_name in POLICY_PARAMS:
        attr = f"{p_name}_yaw_rate_clip"
        _clip_overrides[p_name] = getattr(args, attr, None)
    for p_name, clip_val in _clip_overrides.items():
        if clip_val is not None and p_name in active_params:
            active_params[p_name]["yaw_rate_clip"] = clip_val
    if args.heading_smooth_alpha is not None and "lateral_stable" in active_params:
        active_params["lateral_stable"]["heading_smooth_alpha"] = args.heading_smooth_alpha

    # Apply lateral_stable longitudinal overrides
    long_overrides: dict[str, str] = {
        "thw_target": "lateral_stable_thw_target",
        "jerk_limit": "lateral_stable_jerk_limit",
        "a_max": "lateral_stable_a_max",
        "a_min": "lateral_stable_a_min",
    }
    if "lateral_stable" in active_params:
        for param_key, arg_attr in long_overrides.items():
            val = getattr(args, arg_attr, None)
            if val is not None:
                active_params["lateral_stable"][param_key] = val

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load source data
    # ------------------------------------------------------------------
    print(f"Loading source trajectories from: {args.src_traj_path}")
    src_traj_loaded = np.load(args.src_traj_path, allow_pickle=True)
    src_traj = [np.asarray(t, dtype=np.float32) for t in src_traj_loaded]

    print(f"Loading source front trajectories from: {args.src_front_path}")
    src_front_loaded = np.load(args.src_front_path, allow_pickle=True)
    src_front = [np.asarray(f, dtype=np.float32) for f in src_front_loaded]

    N_src = len(src_traj)
    if len(src_front) != N_src:
        raise ValueError(
            f"Source traj length {N_src} != front length {len(src_front)}"
        )

    if args.window_len is not None:
        for i, t in enumerate(src_traj):
            if len(t) < args.window_len:
                raise ValueError(
                    f"Source window {i} has length {len(t)} < --window_len {args.window_len}"
                )

    # Source splits
    src_split: list[str] | None = None
    if args.src_split_path is not None:
        print(f"Loading source splits from: {args.src_split_path}")
        src_split_arr = np.load(args.src_split_path, allow_pickle=True)
        src_split = [str(s) for s in src_split_arr]
        if len(src_split) != N_src:
            raise ValueError(
                f"Source split length {len(src_split)} != traj length {N_src}"
            )

    # Source meta (scenario ids)
    src_meta_rows: list[dict] = []
    for i in range(N_src):
        src_meta_rows.append(
            _parse_src_meta_row(
                row={},
                fallback_source_index=i,
                fallback_window_len=int(len(src_traj[i])),
            )
        )
    if args.src_meta_path is not None:
        print(f"Loading source metadata from: {args.src_meta_path}")
        meta = np.load(args.src_meta_path, allow_pickle=True)
        if len(meta) == N_src:
            src_meta_rows = [
                _parse_src_meta_row(
                    row=meta[i],
                    fallback_source_index=i,
                    fallback_window_len=int(len(src_traj[i])),
                )
                for i in range(N_src)
            ]
        else:
            print(
                f"  Warning: meta length {len(meta)} != traj length {N_src}; "
                "falling back to generated scenario ids."
            )

    # ------------------------------------------------------------------
    # Simulate rollouts
    # ------------------------------------------------------------------
    P = len(policy_names)
    N_out = P * N_src

    print(f"\nGenerating {P} policies × {N_src} source windows = {N_out} output windows ...")

    out_traj: list[np.ndarray] = []
    out_front: list[np.ndarray] = []
    out_policy_id: list[int] = []
    out_policy_name: list[str] = []
    out_scenario_id: list[str] = []
    out_source_index: list[int] = []
    out_start: list[int] = []
    out_window_len: list[int] = []
    out_front_id: list[str] = []
    out_source_key: list[str] = []
    out_split: list[str] = []

    for p_idx, p_name in enumerate(policy_names):
        params = active_params[p_name]
        print(f"  Policy [{p_idx}] {p_name} ...")
        for i in range(N_src):
            ego_win = src_traj[i]
            front_win = src_front[i]

            sim_ego = _simulate_ego_window(
                src_ego=ego_win,
                src_front=front_win,
                params=params,
                dt=args.dt,
            )

            out_traj.append(sim_ego)
            out_front.append(front_win.copy())
            out_policy_id.append(p_idx)
            out_policy_name.append(p_name)
            out_scenario_id.append(src_meta_rows[i]["scenario_id"])
            out_source_index.append(i)
            out_start.append(src_meta_rows[i]["start"])
            out_window_len.append(src_meta_rows[i]["window_len"])
            out_front_id.append(src_meta_rows[i]["front_id"])
            out_source_key.append(
                f"{src_meta_rows[i]['scenario_id']}|{src_meta_rows[i]['start']}|"
                f"{src_meta_rows[i]['window_len']}|{src_meta_rows[i]['front_id']}"
            )

            if src_split is not None:
                out_split.append(src_split[i])
            else:
                out_split.append(
                    _assign_split_by_hash(src_meta_rows[i]["scenario_id"])
                )

    # ------------------------------------------------------------------
    # Build output arrays
    # ------------------------------------------------------------------
    traj_arr = np.empty(N_out, dtype=object)
    front_arr = np.empty(N_out, dtype=object)
    for k in range(N_out):
        traj_arr[k] = out_traj[k]
        front_arr[k] = out_front[k]

    policy_id_arr = np.array(out_policy_id, dtype=np.int32)
    policy_name_arr = np.array(out_policy_name, dtype=object)
    scenario_id_arr = np.array(out_scenario_id, dtype=object)
    source_index_arr = np.array(out_source_index, dtype=np.int32)
    start_arr = np.array(out_start, dtype=np.int32)
    window_len_arr = np.array(out_window_len, dtype=np.int32)
    front_id_arr = np.array(out_front_id, dtype=object)
    source_key_arr = np.array(out_source_key, dtype=object)
    split_arr = np.array(out_split, dtype=object)
    row_index_arr = np.arange(N_out, dtype=np.int32)

    meta_dtype = np.dtype([
        ("row_index", np.int32),
        ("source_index", np.int32),
        ("source_key", object),
        ("policy_id", np.int32),
        ("policy_name", object),
        ("scenario_id", object),
        ("start", np.int32),
        ("window_len", np.int32),
        ("front_id", object),
        ("split", object),
    ])
    meta_arr = np.empty(N_out, dtype=meta_dtype)
    meta_arr["row_index"] = row_index_arr
    meta_arr["source_index"] = source_index_arr
    meta_arr["source_key"] = source_key_arr
    meta_arr["policy_id"] = policy_id_arr
    meta_arr["policy_name"] = policy_name_arr
    meta_arr["scenario_id"] = scenario_id_arr
    meta_arr["start"] = start_arr
    meta_arr["window_len"] = window_len_arr
    meta_arr["front_id"] = front_id_arr
    meta_arr["split"] = split_arr

    # Alignment and integrity checks.
    assert len(meta_arr) == len(traj_arr) == len(front_arr), "Output metadata and trajectory arrays must align."
    assert np.all(source_index_arr >= 0), "Invalid source_index: expected non-negative integers."
    assert np.all(policy_id_arr >= 0), "Invalid policy_id: expected non-negative integers."
    rows_per_source = Counter(source_index_arr.tolist())
    expected_rows_per_source = len(policy_names)
    underfilled_sources = [s for s, c in rows_per_source.items() if c != expected_rows_per_source]
    if underfilled_sources:
        raise AssertionError(
            "Each source_index should have exactly one row per policy. "
            f"Expected {expected_rows_per_source}, mismatches for source_index values: {underfilled_sources[:10]}"
        )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    np.save(os.path.join(args.output_dir, "traj.npy"), traj_arr)
    np.save(os.path.join(args.output_dir, "front.npy"), front_arr)
    np.save(os.path.join(args.output_dir, "meta.npy"), meta_arr)
    np.save(os.path.join(args.output_dir, "policy_id.npy"), policy_id_arr)
    np.save(os.path.join(args.output_dir, "policy_name.npy"), policy_name_arr)
    np.save(os.path.join(args.output_dir, "scenario_id.npy"), scenario_id_arr)
    np.save(os.path.join(args.output_dir, "source_index.npy"), source_index_arr)
    np.save(os.path.join(args.output_dir, "source_key.npy"), source_key_arr)
    np.save(os.path.join(args.output_dir, "split.npy"), split_arr)
    try:
        import csv

        meta_csv_path = os.path.join(args.output_dir, "meta.csv")
        with open(meta_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "row_index",
                "source_index",
                "source_key",
                "policy_id",
                "policy_name",
                "scenario_id",
                "start",
                "window_len",
                "front_id",
                "split",
            ])
            for i in range(N_out):
                writer.writerow([
                    int(row_index_arr[i]),
                    int(source_index_arr[i]),
                    str(source_key_arr[i]),
                    int(policy_id_arr[i]),
                    str(policy_name_arr[i]),
                    str(scenario_id_arr[i]),
                    int(start_arr[i]),
                    int(window_len_arr[i]),
                    str(front_id_arr[i]),
                    str(split_arr[i]),
                ])
    except Exception as exc:
        raise RuntimeError(f"Failed writing meta.csv: {exc}") from exc

    policy_names_map = {str(idx): name for idx, name in enumerate(policy_names)}
    with open(os.path.join(args.output_dir, "policy_names.json"), "w", encoding="utf-8") as f:
        json.dump(policy_names_map, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    split_counts = Counter(out_split)
    policy_counts = Counter(out_policy_id)

    print(f"\n{'='*60}")
    print(f"Output directory : {args.output_dir}")
    print(f"Total windows    : {N_out}  ({P} policies × {N_src} source windows)")
    print(f"traj shape       : ({N_out}, T, 4)  [object array of per-window arrays]")
    print(f"Split breakdown  : { {k: split_counts[k] for k in sorted(split_counts)} }")
    print(f"Policy breakdown : { {policy_names[k]: policy_counts[k] for k in sorted(policy_counts)} }")
    print(f"Policy names     : {policy_names_map}")

    # ------------------------------------------------------------------
    # Per-policy params summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Per-policy active parameters:")
    param_cols = ["thw_target", "a_max", "a_min", "jerk_limit", "yaw_rate_clip", "heading_smooth_alpha"]
    header = f"  {'Policy':<20}" + "".join(f"  {c:>22}" for c in param_cols)
    print(header)
    print(f"  {'-'*20}" + "".join(f"  {'-'*22}" for _ in param_cols))
    for p_name in policy_names:
        p = active_params[p_name]
        row = f"  {p_name:<20}"
        for c in param_cols:
            val = p.get(c)
            row += f"  {val:>22.4f}" if val is not None else f"  {'N/A':>22}"
        print(row)

    if "lateral_stable" in active_params and "lateral_stable" in policy_names:
        lat = active_params["lateral_stable"]
        print(f"\n{'='*60}")
        print("lateral_stable effective parameters (after defaults + CLI overrides):")
        for key in [
            "yaw_rate_clip",
            "heading_smooth_alpha",
            "thw_target",
            "jerk_limit",
            "a_max",
            "a_min",
        ]:
            print(f"  {key:<22}: {float(lat[key]):.6f}")

    # ------------------------------------------------------------------
    # Per-policy yaw_rate / heading_change report
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Per-policy kinematics report (from generated trajectories):")
    print(f"  {'Policy':<20}  {'yaw_rate|abs|p95':>18}  {'heading_change_total':>22}  {'yaw_rate_clip':>14}")
    print(f"  {'-'*20}  {'-'*18}  {'-'*22}  {'-'*14}")
    for p_idx, p_name in enumerate(policy_names):
        mask = [k for k, pid in enumerate(out_policy_id) if pid == p_idx]
        yaw_abs_vals: list[float] = []
        heading_totals: list[float] = []
        for k in mask:
            traj_k = out_traj[k]  # (T, 4)
            vx_k = traj_k[:, 2]
            vy_k = traj_k[:, 3]
            headings = np.arctan2(vy_k, vx_k)
            dh = np.diff(headings)
            # wrap to [-pi, pi]
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = np.abs(dh) / args.dt
            yaw_abs_vals.extend(yaw_rate.tolist())
            heading_totals.append(float(np.sum(np.abs(dh))))
        yaw_p95 = float(np.percentile(yaw_abs_vals, 95)) if yaw_abs_vals else float("nan")
        heading_mean = float(np.mean(heading_totals)) if heading_totals else float("nan")
        clip_val = active_params[p_name]["yaw_rate_clip"]
        print(f"  {p_name:<20}  {yaw_p95:>18.5f}  {heading_mean:>22.5f}  {clip_val:>14.4f}")

    print(f"{'='*60}")
    print("Files written:")
    for fname in [
        "traj.npy",
        "front.npy",
        "meta.npy",
        "meta.csv",
        "policy_id.npy",
        "policy_name.npy",
        "scenario_id.npy",
        "source_index.npy",
        "source_key.npy",
        "split.npy",
        "policy_names.json",
    ]:
        fpath = os.path.join(args.output_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname:<22}  {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
