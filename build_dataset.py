import argparse
import csv
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
try:
    from waymo_open_dataset.protos import scenario_pb2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'waymo_open_dataset'. Install 'waymo-open-dataset-tf-2-11-0' "
        "and use Python 3.10/3.11 (the official wheel is not compatible with Python 3.12)."
    ) from exc


def _scenario_looks_valid(scenario):
    """Heuristic validity check after protobuf parse."""
    try:
        if len(scenario.tracks) == 0:
            return False
        ego_idx = scenario.sdc_track_index
        return ego_idx is not None and 0 <= ego_idx < len(scenario.tracks)
    except Exception:
        return False


def parse_scenario_from_record(record_bytes):
    """Parse a Scenario from one TFRecord entry.

    Supports two common encodings:
    1) raw Scenario proto bytes
    2) tf.train.Example wrapping Scenario bytes in a bytes_list feature
    """
    scenario = scenario_pb2.Scenario()

    # Case 1: record is already raw Scenario bytes.
    try:
        scenario.ParseFromString(record_bytes)
        if _scenario_looks_valid(scenario):
            return scenario
    except Exception:
        pass

    # Case 2: record is tf.train.Example; scan bytes features for Scenario payload.
    try:
        example = tf.train.Example()
        example.ParseFromString(record_bytes)
        for feature in example.features.feature.values():
            values = feature.bytes_list.value
            if not values:
                continue
            for blob in values:
                candidate = scenario_pb2.Scenario()
                try:
                    candidate.ParseFromString(blob)
                except Exception:
                    continue
                if _scenario_looks_valid(candidate):
                    return candidate
    except Exception:
        pass

    return None


def align_tracks(ego_track, front_track):
    seq_ego, seq_front = [], []
    T = min(len(ego_track.states), len(front_track.states))
    for t in range(T):
        ego_state = ego_track.states[t]
        front_state = front_track.states[t]
        if ego_state.valid and front_state.valid:
            seq_ego.append([
                ego_state.center_x,
                ego_state.center_y,
                ego_state.velocity_x,
                ego_state.velocity_y,
            ])
            seq_front.append([
                front_state.center_x,
                front_state.center_y,
                front_state.velocity_x,
                front_state.velocity_y,
            ])
    return np.asarray(seq_ego), np.asarray(seq_front)


def extract_ego_and_front(scenario):
    if scenario.sdc_track_index is None:
        return None, None, -1

    ego_track = scenario.tracks[scenario.sdc_track_index]
    if len(ego_track.states) == 0:
        return np.empty((0, 4)), None, -1

    front = None
    front_id = -1
    min_forward_dist = 999.0

    for i, track in enumerate(scenario.tracks):
        if i == scenario.sdc_track_index:
            continue

        forward_dists = []
        valid_count = 0
        T = min(len(track.states), len(ego_track.states))

        for t in range(T):
            e = ego_track.states[t]
            o = track.states[t]
            if not (e.valid and o.valid):
                continue

            dx = o.center_x - e.center_x
            dy = o.center_y - e.center_y

            if t < len(ego_track.states) - 1 and ego_track.states[t + 1].valid:
                ego_dx = ego_track.states[t + 1].center_x - e.center_x
                ego_dy = ego_track.states[t + 1].center_y - e.center_y
            else:
                ego_dx = e.velocity_x
                ego_dy = e.velocity_y

            dot = dx * ego_dx + dy * ego_dy
            dist = np.sqrt(dx * dx + dy * dy)
            if dot > 0 and dist < 100:
                forward_dists.append(dist)
                valid_count += 1

        if valid_count > 0:
            mean_forward_dist = float(np.mean(forward_dists))
            if mean_forward_dist < min_forward_dist:
                min_forward_dist = mean_forward_dist
                front = track
                front_id = i

    if front is None:
        return np.empty((0, 4)), None, -1

    ego_aligned, front_aligned = align_tracks(ego_track, front)
    return ego_aligned, front_aligned, front_id


def compute_features(ego, front):
    if front is None or len(front) < len(ego):
        return None

    ego_speed = np.linalg.norm(ego[:, 2:4], axis=1)
    front_speed = np.linalg.norm(front[: len(ego), 2:4], axis=1)

    rel_speed = ego_speed - front_speed
    ego_acc = np.diff(ego_speed, prepend=ego_speed[0])
    front_acc = np.diff(front_speed, prepend=front_speed[0])
    rel_acc = ego_acc - front_acc

    dist = np.linalg.norm(ego[:, :2] - front[: len(ego), :2], axis=1)
    thw = dist / (ego_speed + 1e-3)
    jerk = np.diff(ego_acc, prepend=ego_acc[0])

    front_brake = front_acc < -0.5
    ego_brake = ego_acc < -0.5
    reaction_time = 0.0
    for i in range(len(front_brake)):
        if front_brake[i]:
            for j in range(i, min(i + 10, len(ego_brake))):
                if ego_brake[j]:
                    reaction_time = j - i
                    break
            break

    dx = np.diff(ego[:, 0], prepend=ego[0, 0])
    dy = np.diff(ego[:, 1], prepend=ego[0, 1])
    heading = np.arctan2(dy, dx)
    yaw_rate = np.diff(heading, prepend=heading[0])

    lane_change_count = np.sum(np.abs(yaw_rate) > 0.1)
    lane_change_duration = np.mean(np.abs(yaw_rate) > 0.1)

    speed_norm = ego_speed / (front_speed + 1e-3)

    features = [
        rel_speed.mean(),
        rel_speed.std(),
        np.mean(rel_speed > 0),
        thw.mean(),
        thw.std(),
        thw.min(),
        jerk.mean(),
        jerk.std(),
        np.percentile(jerk, 95),
        rel_acc.mean(),
        rel_acc.std(),
        reaction_time,
        yaw_rate.std(),
        lane_change_count,
        lane_change_duration,
        speed_norm.mean(),
        speed_norm.std(),
        ego_speed.std(),
        ego_acc.std(),
        ego_speed.mean(),
    ]
    return np.asarray(features, dtype=np.float32)


STYLE_FEATURE_NAMES = [
    "acc_abs_p95",
    "acc_abs_p99",
    "acc_rms",
    "jerk_abs_p95",
    "jerk_abs_p99",
    "jerk_rms",
    "yaw_rate_rms",
    "yaw_rate_abs_p95",
    "heading_change_total",
    "speed_control_oscillation",
    "cf_valid_frac",
    "thw_p50",
    "thw_p20",
    "thw_iqr",
    "v_rel_p50",
    "closing_gain_kv",
    "gap_gain_kd",
    "desired_gap_d0",
    "acc_sync_lag",
    "acc_sync_corr",
]
CF_VALID_FRAC_IDX = STYLE_FEATURE_NAMES.index("cf_valid_frac")
D0_IDX = STYLE_FEATURE_NAMES.index("desired_gap_d0")
EPS_DIV_SAFETY = 1e-6


def _wrap_angle_to_pi(x):
    """Wrap scalar/array angles in radians into [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _safe_percentile(arr, q):
    """Return percentile for non-empty array, otherwise NaN."""
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def _speed_control_oscillation(v_e, eps=1e-3):
    """Compute sign-change rate of speed increments, ignoring near-zero increments."""
    dv = np.diff(v_e)
    if dv.size == 0:
        return 0.0
    valid = np.abs(dv) > eps
    dv = dv[valid]
    if dv.size < 2:
        return 0.0
    signs = np.sign(dv)
    return float(np.mean(signs[1:] * signs[:-1] < 0))


def _fit_cf_gains(v_rel_cf, d_cf, a_e_cf, ridge_lambda=1e-3, kd_min=1e-3):
    """Fit kv/kd/d0 in a_e ≈ kv*v_rel + kd*d + b using ridge-regularized least squares.

    Returns:
        (kv, kd, d0, kd_small):
            - d0 is NaN when |kd| < kd_min.
            - kd_small indicates whether d0 was invalidated by the kd_min guard.
    """
    if len(a_e_cf) < 3:
        return np.nan, np.nan, np.nan, False
    # Fit a_e ≈ kv * v_rel + kd * d + b via closed-form ridge for stability.
    x = np.column_stack([v_rel_cf, d_cf, np.ones_like(v_rel_cf)])
    y = a_e_cf
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(3, dtype=np.float64)
    cond = np.linalg.cond(xtx + reg)
    if not np.isfinite(cond) or cond > 1e12:
        return np.nan, np.nan, np.nan, False
    try:
        beta = np.linalg.solve(xtx + reg, x.T @ y)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, False
    kv, kd, b = float(beta[0]), float(beta[1]), float(beta[2])
    if np.abs(kd) < kd_min:
        return kv, kd, np.nan, True
    d0 = -b / kd
    return kv, kd, float(d0), False


def _sanitize_desired_gap_d0(d0_raw: np.ndarray, *, min_gap: float, max_gap: float, log1p: bool) -> np.ndarray:
    """Apply physical constraints and robust compression to desired-gap d0."""
    d0 = np.asarray(d0_raw, dtype=np.float64).copy()
    valid = np.isfinite(d0)
    valid &= d0 > min_gap
    d0[~valid] = np.nan
    if np.any(valid):
        d0[valid] = np.clip(d0[valid], min_gap, max_gap)
        if log1p:
            d0[valid] = np.log1p(d0[valid])
    return d0.astype(np.float32)


def _best_lag_corr(a_e_cf, a_f_cf, max_lag=10, var_eps=1e-8):
    """Find lag in [-max_lag,max_lag] that maximizes Pearson corr between a_e and a_f."""
    if len(a_e_cf) < 3:
        return np.nan, np.nan
    if np.var(a_e_cf) < var_eps or np.var(a_f_cf) < var_eps:
        return np.nan, np.nan
    best_lag, best_corr = np.nan, -np.inf
    # Sweep lag window and keep the lag with max Pearson correlation.
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x = a_e_cf[lag:]
            y = a_f_cf[:-lag]
        elif lag < 0:
            x = a_e_cf[:lag]
            y = a_f_cf[-lag:]
        else:
            x = a_e_cf
            y = a_f_cf
        if len(x) < 3:
            continue
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std < var_eps or y_std < var_eps:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    if not np.isfinite(best_corr):
        return np.nan, np.nan
    return float(best_lag), float(best_corr)


def compute_style_features(
    ego: np.ndarray,
    front: np.ndarray,
    min_points_cf: int = 20,
    kd_min: float = 1e-3,
    d0_min_gap: float = 1.0,
    d0_max_gap: float = 200.0,
    d0_log1p: bool = True,
    return_debug: bool = False,
) -> np.ndarray:
    """
    Compute 20D style feature vector from aligned ego/front trajectories.

    Args:
        ego: (T, 4) aligned ego states [x, y, vx, vy].
        front: (T, 4) aligned front states [x, y, vx, vy].
        min_points_cf: minimum strict car-following mask points required to emit
            car-following stats other than cf_valid_frac.
        kd_min: minimum valid |kd| for desired-gap recovery in CF gain fitting.
        d0_min_gap: physical lower bound for valid desired-gap values.
        d0_max_gap: clipping upper bound for desired-gap values.
        d0_log1p: whether to apply log1p compression after clipping.
        return_debug: when True, return a tuple of (feature, debug_dict).

    Returns:
        feat_style: float32 vector [core(10), car-following(10)].
            core(10): acc/jerk/yaw/heading/speed-control oscillation metrics.
            car-following(10): cf coverage, THW, relative speed, fitted gains,
            and acceleration synchronization lag/correlation.

    Notes:
        Car-following terms are computed under a strict mask
        (v_e>5.5, d∈[8,60], |v_rel|<6), including ridge-fit gains and
        acceleration cross-correlation lag search in [-10, 10].
    """
    v_e = np.linalg.norm(ego[:, 2:4], axis=1)
    v_f = np.linalg.norm(front[:, 2:4], axis=1)

    a_e = np.diff(v_e, prepend=v_e[0])
    a_f = np.diff(v_f, prepend=v_f[0])
    jerk_e = np.diff(a_e, prepend=a_e[0])

    dx = np.diff(ego[:, 0], prepend=ego[0, 0])
    dy = np.diff(ego[:, 1], prepend=ego[0, 1])
    heading_pos = np.arctan2(dy, dx)
    heading_vel = np.arctan2(ego[:, 3], ego[:, 2])
    stationary = np.hypot(dx, dy) < EPS_DIV_SAFETY
    # For near-stationary steps, fall back to velocity heading to avoid unstable arctan2(0,0).
    heading = np.where(stationary, heading_vel, heading_pos)
    yaw_rate = _wrap_angle_to_pi(np.diff(heading, prepend=heading[0]))

    d = np.linalg.norm(front[:, :2] - ego[:, :2], axis=1)
    v_rel = v_e - v_f
    thw = d / (v_e + 1e-3)

    # Strict car-following mask: cruising speed + reasonable distance + small relative speed.
    m_cf = (v_e > 5.5) & (d >= 8.0) & (d <= 60.0) & (np.abs(v_rel) < 6.0)
    cf_valid_frac = float(np.mean(m_cf)) if len(m_cf) > 0 else 0.0
    cf_n = int(np.sum(m_cf))

    heading_diff = np.diff(heading)
    core = [
        _safe_percentile(np.abs(a_e), 95),
        _safe_percentile(np.abs(a_e), 99),
        float(np.sqrt(np.mean(a_e * a_e))),
        _safe_percentile(np.abs(jerk_e), 95),
        _safe_percentile(np.abs(jerk_e), 99),
        float(np.sqrt(np.mean(jerk_e * jerk_e))),
        float(np.sqrt(np.mean(yaw_rate * yaw_rate))),
        _safe_percentile(np.abs(yaw_rate), 95),
        float(np.sum(np.abs(_wrap_angle_to_pi(heading_diff)))) if heading_diff.size > 0 else 0.0,
        _speed_control_oscillation(v_e),
    ]

    if cf_n < min_points_cf:
        cf = [cf_valid_frac] + [np.nan] * 9
        debug_dict = {"d0_raw_unsanitized": np.nan, "kd_small": False}
    else:
        thw_cf = thw[m_cf]
        v_rel_cf = v_rel[m_cf]
        d_cf = d[m_cf]
        a_e_cf = a_e[m_cf]
        a_f_cf = a_f[m_cf]
        thw_q25, thw_q75 = np.percentile(thw_cf, [25, 75])

        kv, kd, d0_raw_unsanitized, kd_small = _fit_cf_gains(
            v_rel_cf=v_rel_cf,
            d_cf=d_cf,
            a_e_cf=a_e_cf,
            ridge_lambda=1e-3,
            kd_min=kd_min,
        )
        d0 = _sanitize_desired_gap_d0(
            np.asarray([d0_raw_unsanitized], dtype=np.float64),
            min_gap=d0_min_gap,
            max_gap=d0_max_gap,
            log1p=d0_log1p,
        )[0]
        lag, corr = _best_lag_corr(a_e_cf=a_e_cf, a_f_cf=a_f_cf, max_lag=10, var_eps=1e-8)

        cf = [
            cf_valid_frac,
            _safe_percentile(thw_cf, 50),
            _safe_percentile(thw_cf, 20),
            float(thw_q75 - thw_q25),
            _safe_percentile(v_rel_cf, 50),
            kv,
            kd,
            d0,
            lag,
            corr,
        ]
        debug_dict = {"d0_raw_unsanitized": float(d0_raw_unsanitized), "kd_small": bool(kd_small)}

    feat_style = np.asarray(core + cf, dtype=np.float32)
    if return_debug:
        return feat_style, debug_dict
    return feat_style


def get_ego_speeds(scenario):
    ego_idx = scenario.sdc_track_index
    if ego_idx is None or ego_idx < 0 or ego_idx >= len(scenario.tracks):
        return np.array([])
    ego_track = scenario.tracks[ego_idx]
    speeds = []
    for st in ego_track.states:
        if st.valid:
            speeds.append(np.hypot(st.velocity_x, st.velocity_y))
    return np.asarray(speeds)


def assign_split(scenario_id, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    digest = hashlib.md5(str(scenario_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def write_summary(output_dir, summary_map):
    output_dir = Path(output_dir)
    txt_path = output_dir / "summary.txt"
    csv_path = output_dir / "summary.csv"

    lines = ["Dataset summary"] + [f"{k}: {v}" for k, v in summary_map.items()]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in summary_map.items():
            writer.writerow([k, v])


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build traj/feat/meta/split from Waymo TFRecords")
    parser.add_argument("--tfrecord_glob", type=str, default="/mnt/d/WMdata/*.tfrecord-*")
    parser.add_argument("--output_dir", type=str, default=str(root / "output"))
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--min_ego_speed", type=float, default=5.5)
    parser.add_argument("--window_len", type=int, default=80)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--min_points_cf", type=int, default=20)
    parser.add_argument("--kd_min", type=float, default=1e-3)
    parser.add_argument("--d0_min_gap", type=float, default=1.0)
    parser.add_argument("--d0_max_gap", type=float, default=200.0)
    parser.add_argument("--d0_log1p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_legacy_features", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_len <= 0:
        raise ValueError(f"--window_len must be > 0, got {args.window_len}")
    if args.stride <= 0:
        raise ValueError(f"--stride must be > 0, got {args.stride}")

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(args.tfrecord_glob))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    print(f"Found {len(files)} TFRecord files via glob: {args.tfrecord_glob}")

    traj_data, front_data, feat_legacy_data, feat_style_raw_data, meta_data, split_data = [], [], [], [], [], []
    d0_raw_unsanitized_list = []
    kd_small_count = 0
    all_speeds = []
    scenarios_filtered = 0
    scenarios_kept = 0

    for file in files:
        dataset_tf = tf.data.TFRecordDataset(file)
        for data in dataset_tf:
            scenario = parse_scenario_from_record(data.numpy())
            if scenario is None:
                scenarios_filtered += 1
                continue

            speeds = get_ego_speeds(scenario)
            if len(speeds) == 0 or np.min(speeds) < args.min_ego_speed:
                scenarios_filtered += 1
                continue

            ego, front, front_id = extract_ego_and_front(scenario)
            if ego is None or front is None or len(ego) == 0 or len(front) == 0:
                scenarios_filtered += 1
                continue

            t = min(len(ego), len(front))
            if t < args.window_len:
                scenarios_filtered += 1
                continue

            split = assign_split(
                scenario.scenario_id,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
            )
            scenario_window_count = 0
            all_speeds.extend(speeds.tolist())
            for start in range(0, t - args.window_len + 1, args.stride):
                end = start + args.window_len
                ego_win = ego[start:end]
                front_win = front[start:end]
                feat_style_raw, debug_dict = compute_style_features(
                    ego=ego_win,
                    front=front_win,
                    min_points_cf=args.min_points_cf,
                    kd_min=args.kd_min,
                    d0_min_gap=args.d0_min_gap,
                    d0_max_gap=args.d0_max_gap,
                    d0_log1p=args.d0_log1p,
                    return_debug=True,
                )
                traj_data.append(ego_win)
                front_data.append(front_win)
                feat_style_raw_data.append(feat_style_raw)
                d0_raw_unsanitized_list.append(debug_dict["d0_raw_unsanitized"])
                kd_small_count += int(debug_dict["kd_small"])
                meta_data.append((scenario.scenario_id, start, args.window_len, front_id))
                split_data.append(split)
                if args.save_legacy_features:
                    feat_legacy = compute_features(ego_win, front_win)
                    if feat_legacy is None:
                        feat_legacy = np.full((20,), np.nan, dtype=np.float32)
                    feat_legacy_data.append(feat_legacy)
                scenario_window_count += 1

            if scenario_window_count > 0:
                scenarios_kept += 1
            else:
                scenarios_filtered += 1

            if args.log_every and scenarios_kept % args.log_every == 0 and scenarios_kept > 0:
                print(f"Processed {scenarios_kept} kept scenarios...")

    traj_data = np.asarray(traj_data, dtype=object)
    front_data = np.asarray(front_data, dtype=object)
    feat_style_raw_data = np.asarray(feat_style_raw_data, dtype=np.float32)
    d0_raw_unsanitized_data = np.asarray(d0_raw_unsanitized_list, dtype=np.float32)
    meta_data = np.asarray(meta_data, dtype=object)
    split_data = np.asarray(split_data, dtype=object)

    if len(feat_style_raw_data) == 0:
        raise RuntimeError(
            "No valid sliding windows after filtering. Try lowering --min_ego_speed or adjusting --window_len/--stride"
        )

    # Missing/invalid style values (mostly CF-only stats when mask is insufficient) are zero-filled
    # before global standardization so training receives dense numeric supervision vectors and
    # keeps behavior consistent with downstream loaders expecting finite float arrays.
    feat_style_raw_filled = np.nan_to_num(feat_style_raw_data, nan=0.0, posinf=0.0, neginf=0.0)
    feat_style_data = (feat_style_raw_filled - feat_style_raw_filled.mean(axis=0)) / (
        feat_style_raw_filled.std(axis=0) + EPS_DIV_SAFETY
    )
    feat_style_data = feat_style_data.astype(np.float32)

    split_counter = Counter(split_data.tolist())

    traj_path = os.path.join(args.output_dir, "traj.npy")
    front_path = os.path.join(args.output_dir, "front.npy")
    feat_path = os.path.join(args.output_dir, "feat.npy")
    feat_legacy_path = os.path.join(args.output_dir, "feat_legacy.npy")
    feat_style_path = os.path.join(args.output_dir, "feat_style.npy")
    feat_style_raw_path = os.path.join(args.output_dir, "feat_style_raw.npy")
    feature_names_style_path = os.path.join(args.output_dir, "feature_names_style.json")
    meta_path = os.path.join(args.output_dir, "meta.npy")
    split_path = os.path.join(args.output_dir, "split.npy")

    np.save(traj_path, traj_data)
    np.save(front_path, front_data)
    np.save(feat_style_path, feat_style_data)
    np.save(feat_style_raw_path, feat_style_raw_data)
    np.save(meta_path, meta_data)
    np.save(split_path, split_data)
    if args.save_legacy_features:
        feat_legacy_data = np.asarray(feat_legacy_data, dtype=np.float32)
        feat_legacy_filled = np.nan_to_num(feat_legacy_data, nan=0.0, posinf=0.0, neginf=0.0)
        feat_legacy_data = (feat_legacy_filled - feat_legacy_filled.mean(axis=0)) / (
            feat_legacy_filled.std(axis=0) + EPS_DIV_SAFETY
        )
        feat_legacy_data = feat_legacy_data.astype(np.float32)
        np.save(feat_path, feat_legacy_data)
        np.save(feat_legacy_path, feat_legacy_data)
    with open(feature_names_style_path, "w", encoding="utf-8") as f:
        json.dump(STYLE_FEATURE_NAMES, f, ensure_ascii=False, indent=2)

    speeds_np = np.asarray(all_speeds, dtype=float) if all_speeds else np.asarray([0.0])
    cf_valid_frac_values = (
        feat_style_raw_data[:, CF_VALID_FRAC_IDX] if feat_style_raw_data.size > 0 else np.asarray([0.0])
    )
    style_nan_counts = np.isnan(feat_style_raw_data).sum(axis=0)
    d0_sanitized_values = feat_style_raw_data[:, D0_IDX]
    d0_raw_valid = d0_raw_unsanitized_data[np.isfinite(d0_raw_unsanitized_data)]
    d0_sanitized_valid = d0_sanitized_values[np.isfinite(d0_sanitized_values)]

    def _safe_fmt_stat(values, q):
        if values.size == 0:
            return "nan"
        return f"{np.percentile(values, q):.6f}"

    summary_map = {
        "tfrecord_glob": args.tfrecord_glob,
        "min_ego_speed": args.min_ego_speed,
        "window_len": args.window_len,
        "stride": args.stride,
        "min_points_cf": args.min_points_cf,
        "kd_min": args.kd_min,
        "d0_min_gap": args.d0_min_gap,
        "d0_max_gap": args.d0_max_gap,
        "d0_log1p": args.d0_log1p,
        "total_windows": len(feat_style_raw_data),
        "scenarios_filtered": scenarios_filtered,
        "scenarios_kept": scenarios_kept,
        "speed_mean": f"{speeds_np.mean():.6f}",
        "speed_std": f"{speeds_np.std():.6f}",
        "speed_min": f"{speeds_np.min():.6f}",
        "speed_max": f"{speeds_np.max():.6f}",
        "train_count": split_counter.get("train", 0),
        "val_count": split_counter.get("val", 0),
        "test_count": split_counter.get("test", 0),
        "traj_shape": str(traj_data.shape),
        "front_shape": str(front_data.shape),
        "feat_style_shape": str(feat_style_data.shape),
        "feat_style_raw_shape": str(feat_style_raw_data.shape),
        "feat_style_dim": len(STYLE_FEATURE_NAMES),
        "d0_raw_valid_count": int(d0_raw_valid.size),
        "d0_raw_min": _safe_fmt_stat(d0_raw_valid, 0),
        "d0_raw_p1": _safe_fmt_stat(d0_raw_valid, 1),
        "d0_raw_p50": _safe_fmt_stat(d0_raw_valid, 50),
        "d0_raw_p99": _safe_fmt_stat(d0_raw_valid, 99),
        "d0_raw_max": _safe_fmt_stat(d0_raw_valid, 100),
        "d0_sanitized_valid_count": int(d0_sanitized_valid.size),
        "d0_sanitized_min": _safe_fmt_stat(d0_sanitized_valid, 0),
        "d0_sanitized_p1": _safe_fmt_stat(d0_sanitized_valid, 1),
        "d0_sanitized_p50": _safe_fmt_stat(d0_sanitized_valid, 50),
        "d0_sanitized_p99": _safe_fmt_stat(d0_sanitized_valid, 99),
        "d0_sanitized_max": _safe_fmt_stat(d0_sanitized_valid, 100),
        "kd_small_count": int(kd_small_count),
        "d0_nan_count": int(np.isnan(d0_sanitized_values).sum()),
        "cf_valid_frac_mean": f"{cf_valid_frac_values.mean():.6f}",
        "cf_valid_frac_std": f"{cf_valid_frac_values.std():.6f}",
        "cf_valid_frac_p10": f"{np.percentile(cf_valid_frac_values, 10):.6f}",
        "cf_valid_frac_p50": f"{np.percentile(cf_valid_frac_values, 50):.6f}",
        "cf_valid_frac_p90": f"{np.percentile(cf_valid_frac_values, 90):.6f}",
        "meta_shape": str(meta_data.shape),
        "split_shape": str(split_data.shape),
    }
    if args.save_legacy_features:
        summary_map["feat_shape"] = str(feat_legacy_data.shape)
        summary_map["feat_legacy_shape"] = str(feat_legacy_data.shape)
    for i, name in enumerate(STYLE_FEATURE_NAMES):
        summary_map[f"feat_style_nan_count_{name}"] = int(style_nan_counts[i])
    write_summary(args.output_dir, summary_map)

    print(f"Saved traj to {traj_path}")
    print(f"Saved front to {front_path}")
    if args.save_legacy_features:
        print(f"Saved legacy feat to {feat_legacy_path} (and {feat_path} for backward compatibility)")
    print(f"Saved style feat to {feat_style_path}")
    print(f"Saved raw style feat to {feat_style_raw_path}")
    print(f"Saved style feature names to {feature_names_style_path}")
    print(f"Saved meta to {meta_path}")
    print(f"Saved split to {split_path}")
    print("Shapes:", traj_data.shape, front_data.shape, feat_style_data.shape, meta_data.shape, split_data.shape)


if __name__ == "__main__":
    main()
