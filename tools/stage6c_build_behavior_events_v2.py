#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tools.stage6c_common import iter_progress, read_json, resolve_path, write_json

EGO = {"x": 0, "y": 1, "vx": 2, "vy": 3, "heading": 4, "speed": 5, "accel": 6, "yaw_rate": 7}
NEI = {"valid": 0, "distance": 5, "closing_rate": 8, "ttc": 9, "thw": 10, "speed": 11}
SLOTS = {0: "front", 1: "left_front", 2: "left_rear", 3: "right_front", 4: "right_rear"}
META_COLUMNS = ["scenario_id", "target_agent_id", "start", "window_len", "split"]
TASK_SPECS = {
    "task_following": ("following", "not_following"),
    "task_lead_brake_response": ("lead_brake_response", "no_lead_brake_response"),
    "task_queue_approach": ("queue_approach", "no_queue_approach"),
    "task_lane_change": ("lane_change", "no_lane_change"),
    "task_cutin_response": ("cutin_response", "no_cutin_response"),
    "task_overtake_opportunity": ("overtake_opportunity", "no_overtake_opportunity"),
    "task_overtake_executed": ("overtake_executed", "no_overtake_executed"),
    "task_hesitation": ("hesitation", "no_hesitation"),
    "task_yield_conflict": ("yield_conflict", "no_yield_conflict"),
}


def finite_values(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(x) -> float:
    v = finite_values(x)
    return float(np.mean(v)) if v.size else np.nan


def safe_min(x) -> float:
    v = finite_values(x)
    return float(np.min(v)) if v.size else np.nan


def safe_max(x) -> float:
    v = finite_values(x)
    return float(np.max(v)) if v.size else np.nan


def safe_percentile(x, q: float) -> float:
    v = finite_values(x)
    return float(np.percentile(v, q)) if v.size else np.nan


def rms(x) -> float:
    v = finite_values(x)
    return float(np.sqrt(np.mean(v * v))) if v.size else np.nan


def safe_ratio(mask) -> float:
    arr = np.asarray(mask)
    return float(np.mean(arr)) if arr.size else np.nan


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-9:
        return np.nan
    return float(num / den)


def nanmean_list(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def wrap_angle(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def count_sign_changes(x, eps: float = 1e-3) -> float:
    v = finite_values(x)
    if v.size < 2:
        return np.nan
    s = np.sign(np.where(np.abs(v) < eps, 0.0, v))
    s = s[s != 0]
    if s.size < 2:
        return 0.0
    return float(np.sum(s[1:] != s[:-1]))



def smooth_signal(x, window: int, enabled: bool = True) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if not enabled or window <= 1 or arr.size == 0:
        return arr.copy()
    window = int(max(1, window))
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, 0.0)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full(arr.shape, np.nan, dtype=float)
    ok = counts > 0
    out[ok] = sums[ok] / counts[ok]
    return out


def clip_abs(x, cap: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if not np.isfinite(cap) or cap <= 0:
        return arr.copy()
    return np.clip(arr, -float(cap), float(cap))


def has_sustained_true(mask, min_frames: int) -> bool:
    return bool(contiguous_true_lengths(np.asarray(mask, dtype=bool)) and max(contiguous_true_lengths(mask)) >= int(max(1, min_frames)))


def contiguous_true_lengths(mask) -> List[int]:
    lengths = []
    cur = 0
    for val in np.asarray(mask, dtype=bool):
        if val:
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


def has_cols(arr: Optional[np.ndarray], cols: Sequence[int]) -> bool:
    return arr is not None and arr.ndim >= 2 and arr.shape[-1] > max(cols)


def slot_array(neighbor_seq: Optional[np.ndarray], slot: int) -> Optional[np.ndarray]:
    if neighbor_seq is None or neighbor_seq.ndim != 3 or neighbor_seq.shape[0] <= slot:
        return None
    return np.asarray(neighbor_seq[slot], dtype=float)


def neighbor_values(slot: Optional[np.ndarray], col: int) -> np.ndarray:
    if not has_cols(slot, [NEI["valid"], col]):
        return np.asarray([], dtype=float)
    valid = np.asarray(slot[:, NEI["valid"]], dtype=float) > 0.5
    values = np.asarray(slot[:, col], dtype=float)
    return np.where(valid, values, np.nan)


def clean_time_gap_values(values: np.ndarray, max_seconds: float, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if arr.size == 0:
        return arr
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError(f"{name} valid max must be positive and finite, got {max_seconds}")
    invalid = (~np.isfinite(arr)) | (arr >= 999.0) | (arr <= 0.0) | (arr > float(max_seconds))
    arr[invalid] = np.nan
    return arr


def valid_ratio(slot: Optional[np.ndarray]) -> float:
    if not has_cols(slot, [NEI["valid"]]):
        return np.nan
    return safe_ratio(np.asarray(slot[:, NEI["valid"]], dtype=float) > 0.5)


def first_index(mask) -> Optional[int]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(idx[0]) if idx.size else None


def first_finite_index(x) -> Optional[int]:
    return first_index(np.isfinite(np.asarray(x, dtype=float)))


def label(value: Optional[bool], positive_label: str, negative_label: str) -> str:
    if value is None:
        return "unknown"
    return positive_label if bool(value) else negative_label


def prefixed(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def load_optional_array(shard_dir: Path, name: str, rows: int, shard_id: int, warnings: List[Dict]):
    path = shard_dir / name
    if not path.exists():
        warnings.append({"warning": "optional_array_missing", "shard_id": int(shard_id), "path": str(path)})
        return None
    if name == "neighbor_slot_ids.npy":
        arr = np.load(path, allow_pickle=True)
        warnings.append({
            "warning": "neighbor_slot_ids_loaded_with_pickle",
            "neighbor_slot_ids_loaded_with_pickle": True,
            "shard_id": int(shard_id),
            "path": str(path),
        })
    else:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.shape[0] != rows:
        raise ValueError(f"Row count mismatch in {path}: expected {rows}, got {arr.shape[0]}")
    return arr


def metadata_from_npy(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        return pd.DataFrame(arr)
    if isinstance(arr, np.ndarray) and arr.shape == () and isinstance(arr.item(), dict):
        return pd.DataFrame(arr.item())
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        vals = arr.tolist()
        if isinstance(vals, list) and (not vals or isinstance(vals[0], dict)):
            return pd.DataFrame(vals)
    return pd.DataFrame(arr)


def load_meta_frame(shard_dir: Path, rows: int, shard_id: int, warnings: List[Dict]) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(rows))
    for name in ["metadata.csv", "meta.csv", "meta.npy"]:
        path = shard_dir / name
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path) if path.suffix == ".csv" else metadata_from_npy(path)
        except Exception as exc:
            warnings.append({"warning": "metadata_load_failed", "shard_id": int(shard_id), "path": str(path), "detail": str(exc)})
            continue
        if len(raw) != rows:
            warnings.append({"warning": "metadata_row_count_mismatch", "shard_id": int(shard_id), "path": str(path), "rows": int(rows), "metadata_rows": int(len(raw))})
            continue
        for col in META_COLUMNS:
            if col in raw.columns:
                frame[col] = raw[col].values
        break
    for col in META_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[META_COLUMNS]


def derive_row(ego: np.ndarray, neighbor: Optional[np.ndarray], slot_ids_available: bool, args) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, float]]:
    strengths = {task: "strong" for task in TASK_SPECS}
    if ego.ndim != 2 or ego.shape[0] < 2 or ego.shape[1] <= max(EGO.values()):
        return ({task: "unknown" for task in TASK_SPECS}, {}, {task: "weak_proxy" for task in TASK_SPECS}, {})

    dt = max(float(args.dt), 1e-6)
    x = np.asarray(ego[:, EGO["x"]], dtype=float)
    y = np.asarray(ego[:, EGO["y"]], dtype=float)
    vy = np.asarray(ego[:, EGO["vy"]], dtype=float)
    heading = np.asarray(ego[:, EGO["heading"]], dtype=float)
    speed_raw = np.asarray(ego[:, EGO["speed"]], dtype=float)
    accel_raw = np.asarray(ego[:, EGO["accel"]], dtype=float)
    yaw_rate_raw = np.asarray(ego[:, EGO["yaw_rate"]], dtype=float)

    speed = smooth_signal(speed_raw, args.smoothing_window, args.enable_signal_smoothing)
    accel_smoothed = smooth_signal(accel_raw, args.smoothing_window, args.enable_signal_smoothing)
    yaw_rate_smoothed = smooth_signal(yaw_rate_raw, args.smoothing_window, args.enable_signal_smoothing)
    vy_smoothed = smooth_signal(vy, args.smoothing_window, args.enable_signal_smoothing)

    raw_jerk = np.diff(accel_raw, prepend=accel_raw[0]) / dt
    raw_lateral_accel = np.diff(vy, prepend=vy[0]) / dt
    raw_curvature = yaw_rate_raw / np.maximum(np.abs(speed_raw), 1e-3)

    accel = np.clip(accel_smoothed, args.accel_min_cap, args.accel_max_cap)
    yaw_rate = clip_abs(yaw_rate_smoothed, args.yaw_rate_abs_cap)
    jerk = clip_abs(np.diff(accel, prepend=accel[0]) / dt, args.jerk_abs_cap)
    lateral_accel = clip_abs(np.diff(vy_smoothed, prepend=vy_smoothed[0]) / dt, args.lateral_accel_abs_cap)
    curvature = clip_abs(yaw_rate / np.maximum(np.abs(speed), 1e-3), args.curvature_abs_cap)
    speed_delta = speed[-1] - speed[0] if np.isfinite(speed[[0, -1]]).all() else np.nan
    peak_decel = min(args.accel_max_cap + abs(args.accel_min_cap), max(0.0, -safe_min(accel))) if np.isfinite(safe_min(accel)) else np.nan
    peak_decel = min(float(args.decel_metric_cap), peak_decel) if np.isfinite(peak_decel) else np.nan
    peak_accel = safe_max(accel)
    raw_diagnostics = {
        "raw_peak_decel": max(0.0, -safe_min(accel_raw)) if np.isfinite(safe_min(accel_raw)) else np.nan,
        "raw_rms_jerk": rms(raw_jerk),
        "raw_max_abs_jerk": safe_max(np.abs(raw_jerk)),
        "raw_rms_yaw_rate": rms(yaw_rate_raw),
        "raw_max_abs_yaw_rate": safe_max(np.abs(yaw_rate_raw)),
        "raw_rms_lateral_accel": rms(raw_lateral_accel),
        "raw_max_abs_lateral_accel": safe_max(np.abs(raw_lateral_accel)),
        "raw_rms_curvature": rms(raw_curvature),
        "raw_max_abs_curvature": safe_max(np.abs(raw_curvature)),
        "clipped_peak_decel": peak_decel,
        "clipped_rms_jerk": rms(jerk),
        "clipped_max_abs_jerk": safe_max(np.abs(jerk)),
        "clipped_rms_yaw_rate": rms(yaw_rate),
        "clipped_max_abs_yaw_rate": safe_max(np.abs(yaw_rate)),
        "clipped_rms_lateral_accel": rms(lateral_accel),
        "clipped_max_abs_lateral_accel": safe_max(np.abs(lateral_accel)),
        "clipped_rms_curvature": rms(curvature),
        "clipped_max_abs_curvature": safe_max(np.abs(curvature)),
    }

    front = slot_array(neighbor, 0)
    lf = slot_array(neighbor, 1)
    lr = slot_array(neighbor, 2)
    rf = slot_array(neighbor, 3)
    rr = slot_array(neighbor, 4)
    front_dist = neighbor_values(front, NEI["distance"])
    front_ttc = clean_time_gap_values(neighbor_values(front, NEI["ttc"]), args.ttc_valid_max_s, "ttc")
    front_thw = clean_time_gap_values(neighbor_values(front, NEI["thw"]), args.thw_valid_max_s, "thw")
    front_closing = neighbor_values(front, NEI["closing_rate"])
    front_speed = neighbor_values(front, NEI["speed"])
    front_closing = smooth_signal(front_closing, args.smoothing_window, args.enable_signal_smoothing)
    front_speed = smooth_signal(front_speed, args.smoothing_window, args.enable_signal_smoothing)
    front_ttc_available = bool(front_ttc.size)
    front_speed_available = bool(front_speed.size)
    front_valid = valid_ratio(front)
    front_dist_valid = safe_ratio(np.isfinite(front_dist)) if front_dist.size else np.nan
    front_thw_valid = safe_ratio(np.isfinite(front_thw)) if front_thw.size else np.nan
    front_known = np.isfinite(front_valid)

    side_slots = [lf, lr, rf, rr]
    side_dist_arrays = [neighbor_values(s, NEI["distance"]) for s in side_slots]
    side_closing_arrays = [neighbor_values(s, NEI["closing_rate"]) for s in side_slots]
    side_dist_values = np.concatenate([finite_values(v) for v in side_dist_arrays]) if any(finite_values(v).size for v in side_dist_arrays) else np.asarray([], dtype=float)
    side_closing_values = np.concatenate([finite_values(v) for v in side_closing_arrays]) if any(finite_values(v).size for v in side_closing_arrays) else np.asarray([], dtype=float)
    target_front_gap = safe_min([safe_min(neighbor_values(lf, NEI["distance"])), safe_min(neighbor_values(rf, NEI["distance"]))])
    target_rear_gap = safe_min([safe_min(neighbor_values(lr, NEI["distance"])), safe_min(neighbor_values(rr, NEI["distance"]))])
    side_min_gap = safe_min(side_dist_values)
    rear_min_gap = safe_min(np.concatenate([finite_values(neighbor_values(lr, NEI["distance"])), finite_values(neighbor_values(rr, NEI["distance"]))]))
    any_min_gap = safe_min([safe_min(front_dist), side_min_gap, rear_min_gap])
    gap_pressure = safe_div(1.0, any_min_gap)
    mean_speed = safe_mean(speed)

    # Lateral / maneuver proxies from raw ego sequence.
    lateral_range = safe_max(y) - safe_min(y) if np.isfinite([safe_max(y), safe_min(y)]).all() else np.nan
    lateral_from_start = np.abs(y - y[0]) if np.isfinite(y[0]) else np.full_like(y, np.nan)
    lc_mask = lateral_from_start >= args.lateral_displacement_m
    lc_lengths = contiguous_true_lengths(lc_mask)
    lc_duration = float(max(lc_lengths) * dt) if lc_lengths else np.nan
    raw_heading_change_total = float(np.nansum(np.abs(wrap_angle(np.diff(heading))))) if heading.size > 1 else np.nan
    heading_change_total = min(raw_heading_change_total, float(args.heading_change_total_cap)) if np.isfinite(raw_heading_change_total) else np.nan
    raw_max_lateral_speed = safe_max(np.abs(vy_smoothed))
    vy_for_lc = clip_abs(vy_smoothed, args.lateral_speed_abs_cap)
    clipped_max_lateral_speed = safe_max(np.abs(vy_for_lc))
    yaw_sign_changes = count_sign_changes(yaw_rate, args.sign_change_eps)
    lat_sign_changes = count_sign_changes(vy_for_lc, args.sign_change_eps)
    lc_oscillation = nanmean_list([yaw_sign_changes, lat_sign_changes])
    lc_duration_exists = bool(np.isfinite(lc_duration) and lc_duration > 0.0)
    lane_change_strong_displacement = bool(np.isfinite(lateral_range) and lateral_range >= args.lane_change_lateral_range_m)
    lane_change_min_displacement = bool(np.isfinite(lateral_range) and lateral_range >= args.lane_change_min_lateral_range_m)
    lane_change_heading_yaw_evidence = bool(
        (np.isfinite(heading_change_total) and heading_change_total >= args.heading_change_rad)
        or (np.isfinite(rms(yaw_rate)) and rms(yaw_rate) >= args.yaw_rate_rms_threshold)
    )
    lane_change_positive = bool(
        lane_change_strong_displacement
        or (lc_duration_exists and lane_change_min_displacement)
        or (lane_change_heading_yaw_evidence and lane_change_min_displacement)
    )
    raw_diagnostics.update({
        "raw_max_lateral_speed": raw_max_lateral_speed,
        "clipped_max_lateral_speed": clipped_max_lateral_speed,
        "raw_heading_change_total": raw_heading_change_total,
        "clipped_heading_change_total": heading_change_total,
    })

    following_positive = None
    if front_known:
        following_positive = bool(
            front_valid >= args.min_valid_front_ratio
            and front_dist_valid >= args.min_valid_front_ratio
            and front_thw_valid >= args.min_valid_front_ratio
            and (not np.isfinite(mean_speed) or mean_speed >= args.low_speed_threshold)
        )

    front_speed_accel = np.diff(front_speed, prepend=front_speed[0]) / dt if front_speed.size else np.asarray([], dtype=float)
    front_speed_decel_mask = np.isfinite(front_speed_accel) & (front_speed_accel <= args.lead_decel_threshold)
    front_closing_derivative_proxy = np.diff(front_closing, prepend=front_closing[0]) / dt if front_closing.size else np.asarray([], dtype=float)
    closing_surge_mask = np.isfinite(front_closing_derivative_proxy) & (front_closing_derivative_proxy >= abs(args.lead_decel_threshold))
    gap_ok = np.isfinite(front_dist) & (front_dist <= args.lead_brake_front_gap_max_m)
    ttc_drop = np.diff(front_ttc, prepend=front_ttc[0]) < 0 if front_ttc.size else np.asarray([], dtype=bool)
    thw_drop = np.diff(front_thw, prepend=front_thw[0]) < 0 if front_thw.size else np.asarray([], dtype=bool)
    ttc_or_thw_drop_ok = True
    if args.lead_brake_require_ttc_or_thw_drop:
        ttc_or_thw_drop_ok = bool((ttc_drop.size and has_sustained_true(ttc_drop, args.lead_brake_min_consecutive_frames)) or (thw_drop.size and has_sustained_true(thw_drop, args.lead_brake_min_consecutive_frames)))
    strong_mask = (front_speed_decel_mask & gap_ok) if front_speed_decel_mask.size == gap_ok.size else np.asarray([], dtype=bool)
    proxy_mask = (closing_surge_mask & gap_ok) if closing_surge_mask.size == gap_ok.size else np.asarray([], dtype=bool)
    strong_lead = bool(
        front_speed_available
        and front_valid >= args.min_valid_front_ratio
        and front_dist_valid >= args.min_valid_front_ratio
        and has_sustained_true(strong_mask, args.lead_brake_min_consecutive_frames)
        and ttc_or_thw_drop_ok
    )
    proxy_lead = bool(
        not strong_lead
        and front_valid >= args.min_valid_front_ratio
        and front_dist_valid >= args.min_valid_front_ratio
        and has_sustained_true(proxy_mask, args.lead_brake_min_consecutive_frames)
        and ttc_or_thw_drop_ok
    )
    lead_mask = strong_mask if strong_lead else proxy_mask
    lead_idx = first_index(lead_mask)
    ego_brake_mask = np.isfinite(accel) & (accel <= args.ego_brake_threshold)
    ego_brake_idx = first_index(ego_brake_mask)
    lead_response_positive = None
    if front_known:
        strengths["task_lead_brake_response"] = "strong" if strong_lead else "proxy"
        lead_response_positive = bool(strong_lead or proxy_lead)
    reaction_delay = np.nan
    if lead_idx is not None and ego_brake_idx is not None and ego_brake_idx >= lead_idx:
        reaction_delay = float((ego_brake_idx - lead_idx) * dt)

    low_front = np.isfinite(front_closing) & (front_closing > args.slower_front_closing_rate)
    front_stopped = np.isfinite(front_speed) & (front_speed <= args.front_speed_low_threshold)
    front_stopped_ratio = safe_ratio(front_stopped) if front_speed_available else np.nan
    queue_strong_stopped_condition = bool(front_speed_available and front_stopped_ratio >= 0.2)
    queue_proxy_condition = bool(safe_percentile(front_thw, 25) <= args.queue_thw_threshold or safe_ratio(low_front) >= 0.2)
    queue_positive = None
    if front_known:
        strengths["task_queue_approach"] = "strong" if front_speed_available else "proxy"
        queue_positive = bool(
            front_valid >= args.min_valid_front_ratio
            and safe_min(front_dist) <= args.queue_front_gap_m
            and safe_mean(speed) >= args.low_speed_threshold
            and (queue_strong_stopped_condition or queue_proxy_condition)
        )
        if queue_positive and not queue_strong_stopped_condition:
            strengths["task_queue_approach"] = "proxy"

    front_valid_bool = np.isfinite(front_dist)
    gap_drop = np.nan
    if np.sum(front_valid_bool) >= 2:
        vals = front_dist[front_valid_bool]
        gap_drop = float(vals[0] - np.nanmin(vals))
    front_appears_late = bool(front_valid_bool.any() and not front_valid_bool[0] and np.nanmin(front_dist) <= args.cutin_max_gap_m)
    # Current v2 cut-in response is a conservative front-gap appearance/drop proxy.
    # Do not mark it strong unless a true side-to-front stable slot-ID transition detector is implemented.
    strengths["task_cutin_response"] = "weak_proxy" if not slot_ids_available else "proxy"
    cutin_positive = None
    if front_known:
        cutin_positive = bool(front_appears_late or (np.isfinite(gap_drop) and gap_drop >= args.cutin_gap_drop_m and safe_min(front_dist) <= args.cutin_max_gap_m))
    cutin_idx = first_finite_index(front_dist) if front_appears_late else (int(np.nanargmin(front_dist)) if np.isfinite(front_dist).any() and np.isfinite(gap_drop) and gap_drop >= args.cutin_gap_drop_m else None)
    cutin_delay = np.nan
    if cutin_idx is not None and ego_brake_idx is not None and ego_brake_idx >= cutin_idx:
        cutin_delay = float((ego_brake_idx - cutin_idx) * dt)

    adjacent_available = np.isfinite(side_min_gap) and side_min_gap >= args.adjacent_available_gap_m
    overtake_opp = None
    if front_known:
        strengths["task_overtake_opportunity"] = "proxy"
        overtake_opp = bool(
            front_valid >= args.min_valid_front_ratio
            and safe_min(front_dist) <= args.overtake_front_gap_m
            and safe_mean(front_closing) > args.slower_front_closing_rate
            and adjacent_available
        )
    overtake_exec = None if overtake_opp is None else bool(overtake_opp and lane_change_positive and (np.isfinite(speed_delta) and speed_delta > 0 or np.isfinite(peak_accel) and peak_accel > 0))
    strengths["task_overtake_executed"] = "proxy"
    overtake_start = first_index(lc_mask | (accel > max(0.2, args.ego_brake_threshold * -0.2)))

    maneuver_context = bool(
        lane_change_positive
        or (np.isfinite(lateral_range) and lateral_range >= args.hesitation_min_lateral_range_m)
        or (np.isfinite(heading_change_total) and heading_change_total >= args.hesitation_min_heading_change_rad)
    )
    abort_like_partial = bool(lane_change_min_displacement and np.isfinite(lateral_range) and lateral_range < args.lane_change_completion_m)
    speed_drop_during_maneuver = bool(maneuver_context and np.isfinite(speed_delta) and -speed_delta >= args.hesitation_min_speed_drop)
    hesitation_components = [
        bool(np.isfinite(yaw_sign_changes) and yaw_sign_changes >= args.hesitation_sign_changes),
        bool(np.isfinite(lat_sign_changes) and lat_sign_changes >= args.hesitation_sign_changes),
        bool(np.isfinite(lc_duration) and lc_duration >= args.long_lane_change_s),
        abort_like_partial,
        speed_drop_during_maneuver,
    ]
    hesitation_evidence_count = int(sum(hesitation_components))
    hesitation_positive = bool(maneuver_context and hesitation_evidence_count >= int(args.hesitation_min_evidence_count))

    side_closing_p95 = safe_percentile(side_closing_values, 95)
    conflict_positive = None
    if neighbor is not None:
        conflict_positive = bool(
            (np.isfinite(any_min_gap) and any_min_gap <= args.conflict_gap_m)
            or (np.isfinite(side_closing_p95) and side_closing_p95 >= args.side_closing_threshold and np.isfinite(side_min_gap) and side_min_gap <= args.conflict_side_gap_m)
        )
    else:
        strengths["task_yield_conflict"] = "weak_proxy"

    def after(mask_start: Optional[int], values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[mask_start:] if mask_start is not None else np.asarray([], dtype=float)

    metrics: Dict[str, float] = {}
    metrics.update(prefixed("following", {
        "mean_thw": safe_mean(front_thw),
        "min_thw": safe_min(front_thw),
        "mean_front_distance": safe_mean(front_dist),
        "min_front_distance": safe_min(front_dist),
        "front_closing_rate_mean": safe_mean(front_closing),
        "front_closing_rate_p95": safe_percentile(front_closing, 95),
        "peak_decel": peak_decel,
        "rms_jerk": rms(jerk),
        "max_abs_jerk": safe_max(np.abs(jerk)),
        "late_brake_score": nanmean_list([safe_div(peak_decel, safe_min(front_thw)), reaction_delay]),
        "aggressiveness_score": nanmean_list([safe_div(1.0, safe_mean(front_thw)), safe_div(1.0, safe_mean(front_dist)), peak_decel, rms(jerk)]),
    }))
    metrics.update(prefixed("lead_brake", {
        "front_decel_start_time": float(lead_idx * dt) if lead_idx is not None else np.nan,
        "ego_brake_start_time": float(ego_brake_idx * dt) if ego_brake_idx is not None else np.nan,
        "reaction_delay": reaction_delay,
        "min_ttc_after_lead_brake": safe_min(after(lead_idx, front_ttc)) if front_ttc_available else np.nan,
        "min_thw_after_lead_brake": safe_min(after(lead_idx, front_thw)),
        "peak_decel_after_lead_brake": max(0.0, -safe_min(after(lead_idx, accel))) if lead_idx is not None else np.nan,
        "max_jerk_after_lead_brake": safe_max(np.abs(after(lead_idx, jerk))),
        "speed_drop_after_lead_brake": max(0.0, speed[lead_idx] - speed[-1]) if lead_idx is not None and np.isfinite(speed[[lead_idx, -1]]).all() else np.nan,
        "late_response_score": nanmean_list([reaction_delay, safe_div(1.0, safe_min(after(lead_idx, front_thw)))]),
    }))
    metrics.update(prefixed("queue", {
        "distance_when_start_decel": front_dist[ego_brake_idx] if ego_brake_idx is not None and ego_brake_idx < len(front_dist) else np.nan,
        "time_to_stop": float(first_index(speed <= args.front_speed_low_threshold) * dt) if first_index(speed <= args.front_speed_low_threshold) is not None else np.nan,
        "final_front_gap": front_dist[np.flatnonzero(np.isfinite(front_dist))[-1]] if np.isfinite(front_dist).any() else np.nan,
        "peak_decel": peak_decel,
        "rms_jerk": rms(jerk),
        "stop_smoothness_score": nanmean_list([-peak_decel if np.isfinite(peak_decel) else np.nan, -rms(jerk) if np.isfinite(rms(jerk)) else np.nan]),
        "creep_after_stop_score": safe_mean(speed[speed <= args.front_speed_low_threshold]) if np.isfinite(speed).any() else np.nan,
        "front_speed_min": safe_min(front_speed) if front_speed_available else np.nan,
        "front_speed_mean": safe_mean(front_speed) if front_speed_available else np.nan,
        "front_stopped_ratio": front_stopped_ratio,
    }))
    metrics.update(prefixed("lc", {
        "rms_yaw_rate": rms(yaw_rate),
        "rms_curvature": rms(curvature),
        "heading_change_total": heading_change_total,
        "max_lateral_speed": clipped_max_lateral_speed,
        "rms_lateral_accel": rms(lateral_accel),
        "duration": lc_duration,
        "oscillation_score": lc_oscillation,
        "target_front_gap_min": target_front_gap,
        "target_rear_gap_min": target_rear_gap,
        "gap_acceptance_score": nanmean_list([safe_div(1.0, target_front_gap), safe_div(1.0, target_rear_gap)]),
        "sharpness_score": nanmean_list([rms(yaw_rate), rms(curvature), rms(lateral_accel)]),
    }))
    metrics.update(prefixed("cutin", {
        "gap_initial": front_dist[cutin_idx] if cutin_idx is not None and cutin_idx < len(front_dist) else np.nan,
        "gap_min": safe_min(front_dist) if cutin_positive else np.nan,
        "min_ttc": safe_min(front_ttc) if cutin_positive and front_ttc_available else np.nan,
        "min_thw": safe_min(front_thw) if cutin_positive else np.nan,
        "reaction_delay_to_brake": cutin_delay,
        "peak_decel_after_cutin": max(0.0, -safe_min(after(cutin_idx, accel))) if cutin_idx is not None else np.nan,
        "jerk_after_cutin": rms(after(cutin_idx, jerk)),
        "speed_drop_after_cutin": max(0.0, speed[cutin_idx] - speed[-1]) if cutin_idx is not None and np.isfinite(speed[[cutin_idx, -1]]).all() else np.nan,
        "yielding_response_score": nanmean_list([max(0.0, -safe_min(after(cutin_idx, accel))) if cutin_idx is not None else np.nan, max(0.0, -speed_delta) if np.isfinite(speed_delta) else np.nan]),
        "late_response_score": cutin_delay,
    }))
    metrics.update(prefixed("overtake", {
        "opportunity_score": nanmean_list([safe_mean(front_closing), safe_div(1.0, safe_min(front_dist)), float(adjacent_available) if np.isfinite(side_min_gap) else np.nan]),
        "execution_score": nanmean_list([float(lane_change_positive), max(0.0, speed_delta) if np.isfinite(speed_delta) else np.nan, max(0.0, peak_accel) if np.isfinite(peak_accel) else np.nan]),
        "execution_rate_proxy": float(overtake_exec) if overtake_exec is not None else np.nan,
        "time_to_initiate": float(overtake_start * dt) if overtake_opp and overtake_start is not None else np.nan,
        "peak_accel": peak_accel if overtake_opp else np.nan,
        "peak_decel": peak_decel if overtake_opp else np.nan,
        "max_abs_jerk": safe_max(np.abs(jerk)) if overtake_opp else np.nan,
        "min_front_gap_before": safe_min(front_dist) if overtake_opp else np.nan,
        "target_lane_front_gap": target_front_gap,
        "target_lane_rear_gap": target_rear_gap,
    }))
    metrics.update(prefixed("hesitation", {
        "score": nanmean_list([yaw_sign_changes, lat_sign_changes, lc_oscillation, lc_duration]),
        "lc_duration": lc_duration,
        "yaw_sign_change_count": yaw_sign_changes,
        "lateral_velocity_sign_change_count": lat_sign_changes,
        "lc_oscillation_score": lc_oscillation,
        "abort_like_score": float(abort_like_partial),
        "speed_drop": max(0.0, -speed_delta) if np.isfinite(speed_delta) else np.nan,
        "evidence_count": float(hesitation_evidence_count),
    }))
    metrics.update({
        "yield_conflict_score": nanmean_list([gap_pressure, side_closing_p95, peak_accel if np.isfinite(peak_accel) else np.nan]),
        "yielding_score": nanmean_list([peak_decel, max(0.0, -speed_delta) if np.isfinite(speed_delta) else np.nan]),
        "assertiveness_score": nanmean_list([max(0.0, speed_delta) if np.isfinite(speed_delta) else np.nan, max(0.0, peak_accel) if np.isfinite(peak_accel) else np.nan, safe_div(1.0, any_min_gap)]),
        "gap_pressure_score": gap_pressure,
        "conflict_accel_score": peak_accel if conflict_positive else np.nan,
        "small_gap_speed_maintain_score": mean_speed if conflict_positive else np.nan,
        "rear_pressure_response_score": safe_percentile(np.concatenate([finite_values(neighbor_values(lr, NEI["closing_rate"])), finite_values(neighbor_values(rr, NEI["closing_rate"]))]) if (finite_values(neighbor_values(lr, NEI["closing_rate"])).size or finite_values(neighbor_values(rr, NEI["closing_rate"])).size) else [], 95),
        "courtesy_score": nanmean_list([peak_decel, max(0.0, -speed_delta) if np.isfinite(speed_delta) else np.nan, -gap_pressure if np.isfinite(gap_pressure) else np.nan]),
    })

    task_values = {
        "task_following": following_positive,
        "task_lead_brake_response": lead_response_positive,
        "task_queue_approach": queue_positive,
        "task_lane_change": lane_change_positive,
        "task_cutin_response": cutin_positive,
        "task_overtake_opportunity": overtake_opp,
        "task_overtake_executed": overtake_exec,
        "task_hesitation": hesitation_positive,
        "task_yield_conflict": conflict_positive,
    }
    events = {task: label(task_values[task], *TASK_SPECS[task]) for task in TASK_SPECS}
    return events, metrics, strengths, raw_diagnostics


def event_diagnostics(events_df: pd.DataFrame, min_ratio: float, max_ratio: float) -> List[Dict]:
    rows = []
    n = len(events_df)
    for task, (pos_label, neg_label) in TASK_SPECS.items():
        vals = events_df[task].astype(str) if task in events_df else pd.Series(["unknown"] * n)
        pos = int((vals == pos_label).sum())
        neg = int((vals == neg_label).sum())
        unk = int((vals == "unknown").sum())
        denom = max(pos + neg, 1)
        pos_ratio = float(pos / denom) if (pos + neg) else 0.0
        unknown_ratio = float(unk / max(n, 1))
        validity = "all_unknown" if unk == n else ("degenerate" if pos_ratio < min_ratio or pos_ratio > max_ratio else "valid")
        rows.append({
            "task_key": task,
            "positive_label": pos_label,
            "negative_label": neg_label,
            "positive_count": pos,
            "negative_count": neg,
            "unknown_count": unk,
            "positive_ratio": pos_ratio,
            "unknown_ratio": unknown_ratio,
            "event_validity": validity,
        })
    return rows


def metric_diagnostics(metrics_df: pd.DataFrame, meta_cols: Sequence[str]) -> List[Dict]:
    rows = []
    n = len(metrics_df)
    for col in metrics_df.columns:
        if col in meta_cols:
            continue
        vals = pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(vals)
        if ok.any():
            finite = vals[ok]
            rows.append({
                "metric": col,
                "valid_count": int(ok.sum()),
                "valid_rate": float(ok.mean()),
                "min": float(np.min(finite)),
                "p01": float(np.percentile(finite, 1)),
                "p50": float(np.percentile(finite, 50)),
                "p99": float(np.percentile(finite, 99)),
                "max": float(np.max(finite)),
            })
        else:
            rows.append({"metric": col, "valid_count": 0, "valid_rate": 0.0, "min": np.nan, "p01": np.nan, "p50": np.nan, "p99": np.nan, "max": np.nan})
    return rows



def physical_expected_ranges(args) -> Dict[str, Tuple[float, float]]:
    return {
        "decel": (0.0, float(args.decel_metric_cap)),
        "jerk": (-float(args.jerk_abs_cap), float(args.jerk_abs_cap)),
        "yaw_rate": (-float(args.yaw_rate_abs_cap), float(args.yaw_rate_abs_cap)),
        "lateral_accel": (-float(args.lateral_accel_abs_cap), float(args.lateral_accel_abs_cap)),
        "curvature": (-float(args.curvature_abs_cap), float(args.curvature_abs_cap)),
    }


def metric_physical_kind(metric_name: str) -> Optional[str]:
    name = metric_name.lower()
    if re.search(r"(^|_)peak_decel($|_)", name) or re.search(r"_decel_after_", name):
        return "decel"
    if "jerk" in name:
        return "jerk"
    if "yaw_rate" in name:
        return "yaw_rate"
    if "lateral_accel" in name:
        return "lateral_accel"
    if "curvature" in name:
        return "curvature"
    return None


def metric_quality_warnings(raw_diag: List[Dict], final_diag: List[Dict], args) -> List[Dict]:
    ranges = physical_expected_ranges(args)
    out: List[Dict] = []
    for source, rows in [("raw", raw_diag), ("final", final_diag)]:
        for row in rows:
            kind = metric_physical_kind(str(row.get("metric", "")))
            if not kind:
                continue
            lo, hi = ranges[kind]
            p99 = float(row.get("p99", np.nan))
            max_v = float(row.get("max", np.nan))
            min_v = float(row.get("min", np.nan))
            exceeds = False
            if kind == "decel":
                exceeds = (np.isfinite(p99) and p99 > hi) or (np.isfinite(max_v) and max_v > hi)
            else:
                exceeds = (np.isfinite(max_v) and max_v > hi) or (np.isfinite(min_v) and min_v < lo) or (np.isfinite(p99) and p99 > hi)
            if exceeds:
                out.append({
                    "warning": "metric_physical_range_warning",
                    "source": source,
                    "metric_name": row.get("metric"),
                    "p99": p99,
                    "max": max_v,
                    "min": min_v,
                    "expected_range": [lo, hi],
                })
    return out


def write_report(path: Path, total_rows: int, shard_count: int, event_diag: List[Dict], metric_diag: List[Dict], warnings: List[Dict]):
    lines = [
        "# Stage 6C v2 behavior-event build report",
        "",
        "Stage 6C v2 is **Task-conditioned behavior-event BDD**. This builder creates task slices and task-specific style metrics; BDD is computed by `stage6c_task_conditioned_bdd_report.py`.",
        "",
        "## Reliability notes",
        "",
        "- following and yield_conflict are currently the most reliable strong detectors.",
        "- cutin, overtake, and much of lead/queue remain proxy-based.",
        "- lane_change and hesitation are usable only if positive_ratio is not broad after tightening; this report emits `lane_change_detector_broad` or `hesitation_detector_broad` when positive_ratio > 0.40.",
        "- TTC/THW sentinels and invalid time gaps are reported as NaN, not as 999-style diagnostic scores.",
        "",
        f"- total_rows: {total_rows}",
        f"- shard_count: {shard_count}",
        "- missing metrics are stored as NaN, never as silent zero fills.",
        "",
        "## Task diagnostics",
        "",
        "| task_key | positive_label | negative_label | positive_count | negative_count | unknown_count | positive_ratio | unknown_ratio | event_validity |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in event_diag:
        lines.append(f"| {row['task_key']} | {row['positive_label']} | {row['negative_label']} | {row['positive_count']} | {row['negative_count']} | {row['unknown_count']} | {row['positive_ratio']:.6g} | {row['unknown_ratio']:.6g} | {row['event_validity']} |")
    lines.extend(["", "## Metric diagnostics", "", "| metric | valid_count | valid_rate | min | p01 | p50 | p99 | max |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in metric_diag:
        lines.append(f"| {row['metric']} | {row['valid_count']} | {row['valid_rate']:.6g} | {row['min']:.6g} | {row['p01']:.6g} | {row['p50']:.6g} | {row['p99']:.6g} | {row['max']:.6g} |")
    metric_quality = [w for w in warnings if w.get("warning") in {"metric_physical_range_warning", "raw_metric_physically_implausible", "physical_metric_clipping_applied"}]
    lines.extend(["", "## Metric quality warnings", ""])
    lines.extend(["- None"] if not metric_quality else [f"- {w.get('warning')}: {w}" for w in metric_quality[:120]])
    deg = [r for r in event_diag if r["event_validity"] != "valid"]
    lines.extend(["", "## Degenerate/all_unknown tasks", ""])
    lines.extend(["- None"] if not deg else [f"- `{r['task_key']}`: {r['event_validity']}, positive_ratio={r['positive_ratio']:.6g}, unknown_ratio={r['unknown_ratio']:.6g}" for r in deg])
    lines.extend(["", "## Warnings", ""])
    lines.extend(["- None"] if not warnings else [f"- {w.get('warning')}: {w}" for w in warnings[:200]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_shards(manifest_path: Path) -> List[Dict]:
    manifest = read_json(manifest_path)
    entries = manifest.get("shards", manifest.get("shard_infos", []))
    if entries:
        return entries
    if "shard_paths" in manifest:
        return [{"shard_path": p} for p in manifest["shard_paths"]]
    raise ValueError(f"No shard entries found in shard manifest: {manifest_path}")


def build(args):
    t0 = time.time()
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"output_dir exists: {out}; use --overwrite")
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.shard_manifest)
    if args.feature_schema_path and not Path(args.feature_schema_path).exists():
        raise FileNotFoundError(f"feature_schema_path does not exist: {args.feature_schema_path}")
    shard_entries = parse_shards(manifest_path)
    warnings: List[Dict] = []
    event_rows: List[Dict] = []
    metric_rows: List[Dict] = []
    raw_metric_rows: List[Dict] = []
    strength_counts: Dict[str, Dict[str, int]] = {task: {} for task in TASK_SPECS}
    emitted_warning_names = set()
    global_row = 0

    for shard_id, shard_info in enumerate(iter_progress(shard_entries, enabled=not args.no_progress, desc="building Stage 6C v2 events", unit="shard")):
        shard_path = shard_info.get("shard_path") or shard_info.get("path")
        if not shard_path:
            raise ValueError(f"Shard entry {shard_id} has no shard_path/path field: {shard_info}")
        shard_dir = resolve_path(manifest_path.parent, shard_path)
        ego_path = shard_dir / "ego_seq.npy"
        if not ego_path.exists():
            raise FileNotFoundError(f"Missing required raw sequence file for Stage 6C v2: {ego_path}")
        ego_arr = np.load(ego_path, mmap_mode="r", allow_pickle=False)
        rows = int(ego_arr.shape[0])
        neighbor_arr = load_optional_array(shard_dir, "neighbor_seq.npy", rows, shard_id, warnings)
        slot_ids_arr = load_optional_array(shard_dir, "neighbor_slot_ids.npy", rows, shard_id, warnings)
        _ = load_optional_array(shard_dir, "interaction_feat_style.npy", rows, shard_id, warnings)
        if neighbor_arr is None:
            warnings.append({"warning": "raw_neighbor_missing_detectors_unknown_or_weak_proxy", "shard_id": int(shard_id), "shard_path": str(shard_path)})
        if slot_ids_arr is None:
            warnings.append({"warning": "neighbor_slot_ids_missing_cutin_uses_conservative_proxy", "shard_id": int(shard_id), "shard_path": str(shard_path)})
        if "lead_brake_selective_detector_enabled" not in emitted_warning_names:
            warnings.append({"warning": "lead_brake_selective_detector_enabled", "shard_id": int(shard_id), "shard_path": str(shard_path), "front_speed_preferred_with_closing_derivative_fallback": True})
            emitted_warning_names.add("lead_brake_selective_detector_enabled")
        if "cutin_true_slot_transition_not_implemented_using_gap_drop_proxy" not in emitted_warning_names:
            warnings.append({"warning": "cutin_true_slot_transition_not_implemented_using_gap_drop_proxy", "shard_id": int(shard_id), "shard_path": str(shard_path), "slot_ids_available": bool(slot_ids_arr is not None)})
            emitted_warning_names.add("cutin_true_slot_transition_not_implemented_using_gap_drop_proxy")
        if neighbor_arr is not None and (neighbor_arr.ndim < 4 or neighbor_arr.shape[-1] <= NEI["ttc"]):
            warnings.append({"warning": "ttc_column_unavailable_metric_set_nan", "shard_id": int(shard_id), "shard_path": str(shard_path), "neighbor_seq_shape": list(neighbor_arr.shape)})
        if neighbor_arr is None or neighbor_arr.ndim < 4 or neighbor_arr.shape[-1] <= NEI["speed"]:
            warnings.append({"warning": "queue_approach_uses_gap_thw_closing_proxy", "shard_id": int(shard_id), "shard_path": str(shard_path), "neighbor_seq_shape": list(neighbor_arr.shape) if neighbor_arr is not None else None})
        meta = load_meta_frame(shard_dir, rows, shard_id, warnings)
        for local_row in range(rows):
            neighbor_row = np.asarray(neighbor_arr[local_row]) if neighbor_arr is not None else None
            events, metrics, strengths, raw_diagnostics = derive_row(np.asarray(ego_arr[local_row]), neighbor_row, slot_ids_arr is not None, args)
            base = {"global_row": int(global_row), "shard_id": int(shard_id), "local_row": int(local_row)}
            for col in META_COLUMNS:
                base[col] = meta.iloc[local_row][col]
            strength_cols = {f"{task}_strength": strengths.get(task, "unknown") for task in TASK_SPECS}
            event_rows.append({**base, **events, **strength_cols})
            metric_rows.append({**base, **metrics})
            raw_metric_rows.append({**base, **raw_diagnostics})
            for task, strength in strengths.items():
                strength_counts[task][strength] = strength_counts[task].get(strength, 0) + 1
            global_row += 1

    event_df = pd.DataFrame(event_rows)
    metric_df = pd.DataFrame(metric_rows)
    raw_metric_df = pd.DataFrame(raw_metric_rows)
    meta_cols = ["global_row", "shard_id", "local_row"] + META_COLUMNS
    event_diag = event_diagnostics(event_df, args.min_event_positive_ratio, args.max_event_positive_ratio)
    for row in event_diag:
        if row["event_validity"] != "valid":
            warnings.append({"warning": "degenerate_or_all_unknown_task", **row})
        if row["task_key"] == "task_lane_change" and row["positive_ratio"] > 0.40:
            warnings.append({"warning": "lane_change_detector_broad", **row})
        if row["task_key"] == "task_hesitation" and row["positive_ratio"] > 0.40:
            warnings.append({"warning": "hesitation_detector_broad", **row})
    for task, counts in strength_counts.items():
        total = sum(counts.values())
        dominant = max(counts, key=counts.get) if counts else "unknown"
        if dominant != "strong":
            warnings.append({"warning": "detector_strength_not_strong", "task_key": task, "detector_strength": dominant, "counts": counts, "rows": int(total)})
    metric_diag = metric_diagnostics(metric_df, meta_cols)
    raw_metric_diag = metric_diagnostics(raw_metric_df, meta_cols)
    clipped_metric_diag = metric_diagnostics(raw_metric_df[[c for c in raw_metric_df.columns if c in meta_cols or c.startswith("clipped_")]], meta_cols) if len(raw_metric_df.columns) else []
    quality_warnings = metric_quality_warnings(raw_metric_diag, metric_diag, args)
    warnings.extend(quality_warnings)
    if any(w.get("source") == "raw" for w in quality_warnings):
        warnings.append({"warning": "raw_metric_physically_implausible"})
    if any(w.get("source") == "final" for w in quality_warnings) or args.enable_signal_smoothing:
        warnings.append({"warning": "physical_metric_clipping_applied", "smoothing_window": int(args.smoothing_window)})
    warnings.append({"warning": "completed", "total_rows": int(len(event_df)), "elapsed_sec": float(time.time() - t0)})

    event_df.to_csv(out / "behavior_event_bins_v2.csv", index=False)
    metric_df.to_csv(out / "behavior_event_metrics_v2.csv", index=False)
    write_json(out / "behavior_event_schema_v2.json", {
        "stage": "Stage 6C v2 — Task-conditioned behavior-event BDD",
        "task_specs": {k: {"positive_label": v[0], "negative_label": v[1]} for k, v in TASK_SPECS.items()},
        "event_diagnostics": event_diag,
        "metric_diagnostics": metric_diag,
        "raw_metric_diagnostics": raw_metric_diag,
        "clipped_metric_diagnostics": clipped_metric_diag,
        "metric_quality_warnings": quality_warnings,
        "detector_strength_counts": strength_counts,
        "detector_strength_columns": [f"{task}_strength" for task in TASK_SPECS],
        "detector_strength_values": ["strong", "proxy", "weak_proxy", "unknown"],
        "schema_notes": {
            "neighbor_slot_ids_loaded_with_pickle": True,
            "ttc_thw_sentinel_and_out_of_range_values_are_nan": True,
            "ttc_metrics_use_true_neighbor_seq_ttc_column_only": True,
            "detector_reliability_note": "following and yield_conflict are currently the most reliable strong detectors; cutin, overtake, and much of lead/queue remain proxy-based; lane_change and hesitation are usable only if positive_ratio is not broad after tightening.",
            "lead_brake_current_detector": "front_speed_deceleration_strong_with_sustained_closing_derivative_proxy_fallback",
            "cutin_current_detector": "front_gap_appearance_or_drop_proxy_no_slot_id_transition",
            "queue_approach_uses_front_speed_when_available_else_gap_thw_closing_proxy": True,
            "lane_change_current_detector": "requires lateral displacement; yaw/heading alone cannot trigger lane_change",
            "hesitation_current_detector": "requires maneuver context and minimum evidence count",
        },
        "raw_array_layout_assumptions": {"ego_seq": EGO, "neighbor_seq_slots": SLOTS, "neighbor_seq": NEI},
        "thresholds": vars(args),
    })
    write_json(out / "behavior_event_warnings_v2.json", warnings)
    write_report(out / "behavior_event_report_v2.md", len(event_df), len(shard_entries), event_diag, metric_diag, warnings)


def parse_args():
    p = argparse.ArgumentParser(description="Build Stage 6C v2 task-conditioned behavior-event bins and task-specific style metrics.")
    p.add_argument("--shard_manifest", required=True, help="Path to sharded dataset manifest JSON.")
    p.add_argument("--feature_schema_path", required=True, help="Path to feature_schema.json; used for provenance and validation.")
    p.add_argument("--output_dir", required=True, help="Output directory for behavior_event_*_v2 artifacts.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it already exists.")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bars.")
    p.add_argument("--dt", type=float, default=0.1, help="Frame duration in seconds.")
    p.add_argument("--smoothing_window", type=int, default=5, help="Rolling moving-average window for derivative-sensitive signals.")
    p.add_argument("--enable_signal_smoothing", action=argparse.BooleanOptionalAction, default=True, help="Enable smoothing before derivative-based metrics and detectors.")
    p.add_argument("--accel_min_cap", type=float, default=-12.0, help="Physical lower cap for smoothed ego acceleration in m/s^2.")
    p.add_argument("--accel_max_cap", type=float, default=8.0, help="Physical upper cap for smoothed ego acceleration in m/s^2.")
    p.add_argument("--decel_metric_cap", type=float, default=12.0, help="Physical cap for reported deceleration metrics in m/s^2.")
    p.add_argument("--jerk_abs_cap", type=float, default=80.0, help="Absolute cap for jerk metrics in m/s^3.")
    p.add_argument("--yaw_rate_abs_cap", type=float, default=2.0, help="Absolute cap for yaw-rate metrics in rad/s.")
    p.add_argument("--lateral_accel_abs_cap", type=float, default=8.0, help="Absolute cap for lateral acceleration metrics in m/s^2.")
    p.add_argument("--curvature_abs_cap", type=float, default=1.0, help="Absolute cap for curvature metrics.")
    p.add_argument("--lateral_speed_abs_cap", type=float, default=5.0, help="Absolute cap for lane-change lateral-speed metrics in m/s.")
    p.add_argument("--heading_change_total_cap", type=float, default=8.0, help="Upper cap for total heading-change metrics in radians.")
    p.add_argument("--ttc_valid_max_s", type=float, default=30.0, help="Maximum valid TTC in seconds; <=0, >=999, and larger values are set to NaN.")
    p.add_argument("--thw_valid_max_s", type=float, default=30.0, help="Maximum valid THW in seconds; <=0, >=999, and larger values are set to NaN.")
    p.add_argument("--min_valid_front_ratio", type=float, default=0.3, help="Minimum valid front frames for front-conditioned tasks.")
    p.add_argument("--min_event_positive_ratio", type=float, default=0.01, help="Below this positive ratio a task is degenerate.")
    p.add_argument("--max_event_positive_ratio", type=float, default=0.95, help="Above this positive ratio a task is degenerate.")
    p.add_argument("--low_speed_threshold", type=float, default=1.0, help="Low ego speed threshold in m/s.")
    p.add_argument("--front_speed_low_threshold", type=float, default=1.0, help="Low/stop speed threshold in m/s for queue metrics.")
    p.add_argument("--lead_decel_threshold", type=float, default=-1.0, help="Lead front-speed deceleration threshold in m/s^2; fallback proxy uses matching closing-rate surge.")
    p.add_argument("--lead_brake_min_consecutive_frames", type=int, default=3, help="Minimum sustained frames for lead-brake detection.")
    p.add_argument("--lead_brake_front_gap_max_m", type=float, default=50.0, help="Maximum front gap for lead-brake response detection.")
    p.add_argument("--lead_brake_require_ttc_or_thw_drop", action=argparse.BooleanOptionalAction, default=True, help="Require sustained TTC or THW decrease for lead-brake response detection.")
    p.add_argument("--ego_brake_threshold", type=float, default=-0.5, help="Ego brake onset threshold in m/s^2.")
    p.add_argument("--hard_brake_threshold", type=float, default=-2.0, help="Hard-brake reference threshold in m/s^2.")
    p.add_argument("--slower_front_closing_rate", type=float, default=0.5, help="Closing-rate threshold for slower lead vehicle proxy.")
    p.add_argument("--queue_front_gap_m", type=float, default=30.0, help="Max front gap for queue approach proxy.")
    p.add_argument("--queue_thw_threshold", type=float, default=2.5, help="THW threshold for queue approach proxy.")
    p.add_argument("--overtake_front_gap_m", type=float, default=35.0, help="Max front gap for overtake opportunity proxy.")
    p.add_argument("--adjacent_available_gap_m", type=float, default=8.0, help="Min adjacent gap for overtake opportunity proxy.")
    p.add_argument("--lateral_displacement_m", type=float, default=2.0, help="Legacy lateral displacement threshold used for lane-change duration measurement.")
    p.add_argument("--lane_change_lateral_range_m", type=float, default=2.5, help="Strong lateral range threshold for conservative lane-change positive detection.")
    p.add_argument("--lane_change_min_lateral_range_m", type=float, default=1.5, help="Minimum lateral range required before duration or heading/yaw evidence can trigger lane-change.")
    p.add_argument("--lane_change_completion_m", type=float, default=3.0, help="Lateral completion threshold for abort-like proxy.")
    p.add_argument("--heading_change_rad", type=float, default=0.25, help="Heading-change threshold for lane-change proxy.")
    p.add_argument("--yaw_rate_rms_threshold", type=float, default=0.10, help="Yaw-rate RMS threshold for lane-change proxy.")
    p.add_argument("--lateral_speed_threshold", type=float, default=0.5, help="Lateral-speed threshold for lane-change proxy.")
    p.add_argument("--sign_change_eps", type=float, default=1e-3, help="Epsilon for sign-change metrics.")
    p.add_argument("--hesitation_min_lateral_range_m", type=float, default=1.0, help="Minimum lateral range for hesitation maneuver context.")
    p.add_argument("--hesitation_min_heading_change_rad", type=float, default=0.15, help="Minimum heading-change total for hesitation maneuver context.")
    p.add_argument("--hesitation_sign_changes", type=float, default=8.0, help="Smoothed sign-change count threshold for hesitation.")
    p.add_argument("--hesitation_min_evidence_count", type=int, default=2, help="Minimum number of hesitation evidence components required with maneuver context.")
    p.add_argument("--hesitation_min_speed_drop", type=float, default=1.0, help="Minimum smoothed speed drop during maneuver for hesitation.")
    p.add_argument("--hesitation_require_maneuver_context", action=argparse.BooleanOptionalAction, default=True, help="Require lane-change/lateral/heading maneuver context for hesitation.")
    p.add_argument("--long_lane_change_s", type=float, default=4.0, help="Long lane-change duration threshold for hesitation.")
    p.add_argument("--cutin_gap_drop_m", type=float, default=8.0, help="Front-gap sudden drop threshold for cut-in proxy.")
    p.add_argument("--cutin_max_gap_m", type=float, default=25.0, help="Max front gap after cut-in proxy.")
    p.add_argument("--conflict_gap_m", type=float, default=8.0, help="Small-gap threshold for yield conflict.")
    p.add_argument("--conflict_side_gap_m", type=float, default=12.0, help="Side-gap threshold under closing pressure.")
    p.add_argument("--side_closing_threshold", type=float, default=1.0, help="Side closing-rate threshold for interaction pressure.")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
