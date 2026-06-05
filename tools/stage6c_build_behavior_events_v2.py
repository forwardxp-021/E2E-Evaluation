#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tools.stage6c_common import iter_progress, read_json, resolve_path, robust_score, score_from_parts, write_json

EGO = {"x": 0, "y": 1, "vx": 2, "vy": 3, "heading": 4, "speed": 5, "accel": 6, "yaw_rate": 7}
NEI = {"valid": 0, "distance": 5, "closing_rate": 8, "thw": 10}
PRIMARY_EVENTS = [
    "following",
    "lane_change",
    "overtake",
    "cutin_response",
    "hesitation",
    "yield_conflict",
]
SECONDARY_EVENTS = [
    "free_cruising_stability",
    "stop_and_go_low_speed_creep",
    "risk_proximity",
    "interaction_comfort",
]
ALL_EVENTS = PRIMARY_EVENTS + SECONDARY_EVENTS
META_COLUMNS = ["scenario_id", "target_agent_id", "start", "window_len", "split"]


def finite(arr):
    return np.asarray(arr, dtype=float)[np.isfinite(arr)]


def safe_mean(arr):
    x = finite(arr)
    return float(np.mean(x)) if x.size else np.nan


def safe_min(arr):
    x = finite(arr)
    return float(np.min(x)) if x.size else np.nan


def safe_max(arr):
    x = finite(arr)
    return float(np.max(x)) if x.size else np.nan


def safe_p(arr, q):
    x = finite(arr)
    return float(np.percentile(x, q)) if x.size else np.nan


def rms(arr):
    x = finite(arr)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else np.nan


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def count_sign_changes(arr, eps=1e-3):
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    signs = np.sign(np.where(np.abs(x) < eps, 0.0, x))
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    return float(np.sum(signs[1:] != signs[:-1]))


def contiguous_true_lengths(mask):
    mask = np.asarray(mask, dtype=bool)
    lengths = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


def required_cols_ok(arr: np.ndarray, cols: Sequence[int]) -> bool:
    return arr.ndim >= 2 and arr.shape[-1] > max(cols)


def neighbor_slot(neighbor_seq: Optional[np.ndarray], slot: int) -> Optional[np.ndarray]:
    if neighbor_seq is None or neighbor_seq.ndim != 3 or neighbor_seq.shape[0] <= slot:
        return None
    return np.asarray(neighbor_seq[slot], dtype=float)


def valid_neighbor_values(slot_arr: Optional[np.ndarray], col: int) -> np.ndarray:
    if slot_arr is None or not required_cols_ok(slot_arr, [NEI["valid"], col]):
        return np.asarray([], dtype=float)
    valid = slot_arr[:, NEI["valid"]] > 0.5
    vals = np.asarray(slot_arr[:, col], dtype=float)
    return np.where(valid, vals, np.nan)


def valid_ratio(slot_arr: Optional[np.ndarray]) -> float:
    if slot_arr is None or not required_cols_ok(slot_arr, [NEI["valid"]]):
        return np.nan
    return float(np.mean(slot_arr[:, NEI["valid"]] > 0.5))


def first_finite_index(arr):
    ok = np.isfinite(arr)
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else None


def status(value: Optional[bool]) -> str:
    if value is None:
        return "unknown"
    return "positive" if bool(value) else "negative"


def load_meta_frame(shard_dir: Path, rows: int, shard_id: int, warnings: List[Dict]) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(rows))
    meta_path = shard_dir / "meta.npy"
    if meta_path.exists():
        try:
            arr = np.load(meta_path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype.names:
                raw = pd.DataFrame(arr)
            elif isinstance(arr, np.ndarray) and arr.shape == () and isinstance(arr.item(), dict):
                raw = pd.DataFrame(arr.item())
            elif isinstance(arr, np.ndarray) and arr.dtype == object:
                vals = arr.tolist()
                raw = pd.DataFrame(vals if isinstance(vals, list) else arr)
            else:
                raw = pd.DataFrame(arr)
            if len(raw) == rows:
                for c in META_COLUMNS:
                    if c in raw.columns:
                        frame[c] = raw[c].values
            else:
                warnings.append({"shard_id": shard_id, "path": str(meta_path), "warning": "meta_row_count_mismatch", "rows": rows, "meta_rows": int(len(raw))})
        except Exception as exc:
            warnings.append({"shard_id": shard_id, "path": str(meta_path), "warning": "meta_load_failed", "detail": str(exc)})
    split_path = shard_dir / "split.npy"
    if "split" not in frame.columns and split_path.exists():
        split = np.load(split_path, allow_pickle=True)
        if len(split) == rows:
            frame["split"] = split.astype(str)
        else:
            warnings.append({"shard_id": shard_id, "path": str(split_path), "warning": "split_row_count_mismatch", "rows": rows, "split_rows": int(len(split))})
    for c in META_COLUMNS:
        if c not in frame.columns:
            frame[c] = np.nan
    return frame[META_COLUMNS]


def load_optional_array(shard_dir: Path, name: str, rows: int, shard_id: int, warnings: List[Dict], mmap_mode="r"):
    path = shard_dir / name
    if not path.exists():
        warnings.append({"shard_id": shard_id, "path": str(path), "warning": "optional_array_missing"})
        return None
    arr = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if arr.shape[0] != rows:
        raise ValueError(f"Row count mismatch in {path}: expected {rows}, got {arr.shape[0]}")
    return arr


def derive_row_metrics(ego: np.ndarray, neighbor: Optional[np.ndarray], dt: float, args) -> Tuple[Dict[str, float], Dict[str, str]]:
    if ego.ndim != 2 or ego.shape[0] < 2 or ego.shape[1] <= max(EGO.values()):
        return {}, {event: "unknown" for event in ALL_EVENTS}

    speed = np.asarray(ego[:, EGO["speed"]], dtype=float)
    accel = np.asarray(ego[:, EGO["accel"]], dtype=float)
    yaw_rate = np.asarray(ego[:, EGO["yaw_rate"]], dtype=float)
    heading = np.asarray(ego[:, EGO["heading"]], dtype=float)
    lateral_pos = np.asarray(ego[:, EGO["y"]], dtype=float)
    lateral_speed = np.asarray(ego[:, EGO["vy"]], dtype=float)
    jerk = np.diff(accel, prepend=accel[0]) / max(dt, 1e-6)
    curvature = yaw_rate / np.maximum(np.abs(speed), 1e-3)
    lateral_accel = np.diff(lateral_speed, prepend=lateral_speed[0]) / max(dt, 1e-6)
    speed_delta = speed[-1] - speed[0] if np.all(np.isfinite([speed[0], speed[-1]])) else np.nan

    front = neighbor_slot(neighbor, 0)
    lf = neighbor_slot(neighbor, 1)
    lr = neighbor_slot(neighbor, 2)
    rf = neighbor_slot(neighbor, 3)
    rr = neighbor_slot(neighbor, 4)
    front_dist = valid_neighbor_values(front, NEI["distance"])
    front_thw = valid_neighbor_values(front, NEI["thw"])
    front_closing = valid_neighbor_values(front, NEI["closing_rate"])
    side_dists = [valid_neighbor_values(s, NEI["distance"]) for s in [lf, lr, rf, rr]]
    rear_dists = [valid_neighbor_values(s, NEI["distance"]) for s in [lr, rr]]
    side_closing = [valid_neighbor_values(s, NEI["closing_rate"]) for s in [lf, lr, rf, rr]]

    front_valid_ratio = valid_ratio(front)
    front_dist_valid_ratio = float(np.mean(np.isfinite(front_dist))) if front_dist.size else np.nan
    thw_valid_ratio = float(np.mean(np.isfinite(front_thw))) if front_thw.size else np.nan
    mean_speed = safe_mean(speed)
    low_speed_ratio = float(np.mean(speed < args.low_speed_mps)) if np.all(np.isfinite(speed)) else np.nan
    stopped_ratio = float(np.mean(speed < args.stop_speed_mps)) if np.all(np.isfinite(speed)) else np.nan

    lane_change_mask = np.abs(lateral_pos - np.nanmedian(lateral_pos)) > args.lateral_displacement_m
    lc_lengths = contiguous_true_lengths(lane_change_mask)
    lane_change_duration = float(max(lc_lengths) * dt) if lc_lengths else np.nan
    lane_change_count_proxy = float(np.sum(lane_change_mask[1:] & (~lane_change_mask[:-1]))) if lane_change_mask.size > 1 else 0.0
    heading_change_total = float(np.nansum(np.abs(wrap_angle(np.diff(heading)))))
    yaw_sign_change_count = count_sign_changes(yaw_rate, args.sign_change_eps)
    lat_sign_change_count = count_sign_changes(lateral_speed, args.sign_change_eps)
    lane_change_oscillation = np.nanmean([yaw_sign_change_count, lat_sign_change_count])

    left_front_min_gap = safe_min(valid_neighbor_values(lf, NEI["distance"]))
    left_rear_min_gap = safe_min(valid_neighbor_values(lr, NEI["distance"]))
    right_front_min_gap = safe_min(valid_neighbor_values(rf, NEI["distance"]))
    right_rear_min_gap = safe_min(valid_neighbor_values(rr, NEI["distance"]))
    target_front_gap = safe_min([left_front_min_gap, right_front_min_gap])
    target_rear_gap = safe_min([left_rear_min_gap, right_rear_min_gap])
    side_min_gap = safe_min(np.concatenate([x[np.isfinite(x)] for x in side_dists]) if any(np.isfinite(x).any() for x in side_dists) else [])
    rear_min_gap = safe_min(np.concatenate([x[np.isfinite(x)] for x in rear_dists]) if any(np.isfinite(x).any() for x in rear_dists) else [])
    min_any_gap = safe_min([safe_min(front_dist), side_min_gap, rear_min_gap])
    side_closing_p95 = safe_p(np.concatenate([x[np.isfinite(x)] for x in side_closing]) if any(np.isfinite(x).any() for x in side_closing) else [], 95)
    front_gap_drop = np.nan
    if np.isfinite(front_dist).sum() >= 2:
        valid_fd = front_dist[np.isfinite(front_dist)]
        front_gap_drop = float(valid_fd[0] - np.nanmin(valid_fd))

    brake_idx = first_finite_index(np.where(accel < -args.brake_threshold_mps2, accel, np.nan))
    front_first_idx = first_finite_index(front_dist)
    reaction_delay = np.nan if brake_idx is None or front_first_idx is None or brake_idx < front_first_idx else float((brake_idx - front_first_idx) * dt)

    lane_change_positive = (
        lane_change_count_proxy > 0
        or heading_change_total >= args.heading_change_rad
        or safe_max(np.abs(lateral_pos - lateral_pos[0])) >= args.lateral_displacement_m
        or rms(yaw_rate) >= args.yaw_rate_rms_threshold
    )
    front_present_known = np.isfinite(front_valid_ratio)
    following_positive = None if not front_present_known else (
        front_valid_ratio >= args.min_front_valid_ratio
        and front_dist_valid_ratio >= args.min_front_valid_ratio
        and thw_valid_ratio >= args.min_front_valid_ratio
        and (not np.isfinite(mean_speed) or mean_speed >= args.low_speed_mps)
    )
    overtake_positive = None if not front_present_known else (
        front_valid_ratio >= args.min_front_valid_ratio
        and safe_mean(front_closing) > args.slower_front_closing_rate
        and safe_min(front_dist) <= args.overtake_front_gap_m
        and (np.isfinite(side_min_gap) and side_min_gap >= args.adjacent_available_gap_m)
    )
    cutin_positive = None if not front_present_known else (
        (front_valid_ratio > 0 and front_valid_ratio < args.min_front_valid_ratio and np.isfinite(safe_min(front_dist)))
        or (np.isfinite(front_gap_drop) and front_gap_drop >= args.cutin_gap_drop_m and safe_min(front_dist) <= args.cutin_max_gap_m)
    )
    hesitation_positive = (
        yaw_sign_change_count >= args.hesitation_sign_changes
        or lat_sign_change_count >= args.hesitation_sign_changes
        or (np.isfinite(lane_change_duration) and lane_change_duration >= args.long_lane_change_s)
        or (lane_change_positive and safe_max(np.abs(lateral_pos - lateral_pos[0])) < args.lane_change_completion_m)
    )
    conflict_positive = (
        (np.isfinite(min_any_gap) and min_any_gap <= args.conflict_gap_m)
        or (np.isfinite(side_closing_p95) and side_closing_p95 >= args.side_closing_threshold)
    )
    free_cruise_positive = None if not front_present_known else (front_valid_ratio < 0.2 and not lane_change_positive and mean_speed >= args.low_speed_mps)
    stop_go_positive = np.isfinite(low_speed_ratio) and low_speed_ratio >= args.low_speed_ratio
    risk_positive = np.isfinite(min_any_gap) and min_any_gap <= args.risk_gap_m
    comfort_positive = (rms(jerk) <= args.comfort_rms_jerk and rms(yaw_rate) <= args.comfort_rms_yaw_rate)

    peak_decel = -safe_min(accel) if np.isfinite(safe_min(accel)) else np.nan
    peak_accel = safe_max(accel)
    late_brake_score = peak_decel / max(safe_min(front_thw), 1e-3) if np.isfinite(peak_decel) and np.isfinite(safe_min(front_thw)) else np.nan
    gap_pressure = np.nanmean([
        1.0 / max(safe_min(front_dist), 1e-3) if np.isfinite(safe_min(front_dist)) else np.nan,
        1.0 / max(side_min_gap, 1e-3) if np.isfinite(side_min_gap) else np.nan,
        1.0 / max(rear_min_gap, 1e-3) if np.isfinite(rear_min_gap) else np.nan,
    ])

    metrics = {
        "mean_thw": safe_mean(front_thw),
        "min_thw": safe_min(front_thw),
        "mean_front_distance": safe_mean(front_dist),
        "min_front_distance": safe_min(front_dist),
        "front_closing_rate_mean": safe_mean(front_closing),
        "front_closing_rate_p95": safe_p(front_closing, 95),
        "peak_decel": peak_decel,
        "rms_jerk": rms(jerk),
        "max_abs_jerk": safe_max(np.abs(jerk)),
        "late_brake_score": late_brake_score,
        "following_aggressiveness_score": np.nanmean([1.0 / max(safe_mean(front_thw), 1e-3) if np.isfinite(safe_mean(front_thw)) else np.nan, peak_decel, rms(jerk)]),
        "rms_yaw_rate": rms(yaw_rate),
        "rms_curvature": rms(curvature),
        "heading_change_total": heading_change_total,
        "max_lateral_speed": safe_max(np.abs(lateral_speed)),
        "rms_lateral_accel": rms(lateral_accel),
        "lane_change_duration": lane_change_duration,
        "lane_change_oscillation_score": lane_change_oscillation,
        "target_front_min_gap_during_lane_change": target_front_gap,
        "target_rear_min_gap_during_lane_change": target_rear_gap,
        "lane_change_sharpness_score": np.nanmean([rms(yaw_rate), rms(curvature), rms(lateral_accel)]),
        "gap_acceptance_score": np.nanmean([1.0 / max(target_front_gap, 1e-3) if np.isfinite(target_front_gap) else np.nan, 1.0 / max(target_rear_gap, 1e-3) if np.isfinite(target_rear_gap) else np.nan]),
        "overtake_opportunity_score": np.nanmean([safe_mean(front_closing), 1.0 / max(safe_min(front_dist), 1e-3) if np.isfinite(safe_min(front_dist)) else np.nan, mean_speed]),
        "overtake_execution_score": np.nanmean([float(lane_change_positive), max(speed_delta, 0.0) if np.isfinite(speed_delta) else np.nan, max(peak_accel, 0.0) if np.isfinite(peak_accel) else np.nan]),
        "time_to_initiate_overtake": lane_change_duration if overtake_positive and lane_change_positive else np.nan,
        "peak_accel_during_overtake": peak_accel if overtake_positive else np.nan,
        "peak_decel_during_overtake": peak_decel if overtake_positive else np.nan,
        "jerk_during_overtake": rms(jerk) if overtake_positive else np.nan,
        "min_front_gap_before_overtake": safe_min(front_dist) if overtake_positive else np.nan,
        "target_lane_front_gap": target_front_gap,
        "target_lane_rear_gap": target_rear_gap,
        "cutin_gap_initial": front_dist[front_first_idx] if front_first_idx is not None else np.nan,
        "cutin_gap_min": safe_min(front_dist) if cutin_positive else np.nan,
        "cutin_min_ttc": safe_min(front_thw) if cutin_positive else np.nan,
        "reaction_delay_to_brake": reaction_delay if cutin_positive else np.nan,
        "peak_decel_after_cutin": peak_decel if cutin_positive else np.nan,
        "jerk_after_cutin": rms(jerk) if cutin_positive else np.nan,
        "speed_drop_after_cutin": -min(speed_delta, 0.0) if cutin_positive and np.isfinite(speed_delta) else np.nan,
        "yielding_response_score": np.nanmean([peak_decel, -speed_delta if np.isfinite(speed_delta) else np.nan]) if cutin_positive else np.nan,
        "late_response_score": reaction_delay if cutin_positive else np.nan,
        "hesitation_score": np.nanmean([yaw_sign_change_count, lat_sign_change_count, lane_change_duration]),
        "yaw_sign_change_count": yaw_sign_change_count,
        "lateral_velocity_sign_change_count": lat_sign_change_count,
        "abort_like_score": float(lane_change_positive and safe_max(np.abs(lateral_pos - lateral_pos[0])) < args.lane_change_completion_m),
        "speed_drop_during_hesitation": -min(speed_delta, 0.0) if hesitation_positive and np.isfinite(speed_delta) else np.nan,
        "yielding_score": np.nanmean([peak_decel, -speed_delta if np.isfinite(speed_delta) else np.nan]),
        "assertiveness_score": np.nanmean([max(speed_delta, 0.0) if np.isfinite(speed_delta) else np.nan, max(peak_accel, 0.0) if np.isfinite(peak_accel) else np.nan, -gap_pressure if np.isfinite(gap_pressure) else np.nan]),
        "gap_pressure_score": gap_pressure,
        "conflict_accel_score": peak_accel if conflict_positive else np.nan,
        "small_gap_speed_maintain_score": mean_speed if conflict_positive and np.isfinite(mean_speed) else np.nan,
        "rear_pressure_response": np.nanmean([safe_p(valid_neighbor_values(lr, NEI["closing_rate"]), 95), safe_p(valid_neighbor_values(rr, NEI["closing_rate"]), 95)]),
        "courtesy_score": np.nanmean([peak_decel, -speed_delta if np.isfinite(speed_delta) else np.nan, -gap_pressure if np.isfinite(gap_pressure) else np.nan]),
        "cruise_speed_std": float(np.nanstd(speed)),
        "cruise_yaw_rate_rms": rms(yaw_rate),
        "cruise_rms_jerk": rms(jerk),
        "stop_go_low_speed_ratio": low_speed_ratio,
        "stop_go_stopped_ratio": stopped_ratio,
        "risk_min_any_gap": min_any_gap,
        "risk_min_ttc": safe_min(front_thw),
        "interaction_comfort_rms_jerk": rms(jerk),
        "interaction_comfort_rms_yaw_rate": rms(yaw_rate),
    }
    events = {
        "following": status(following_positive),
        "lane_change": status(lane_change_positive),
        "overtake": status(overtake_positive),
        "cutin_response": status(cutin_positive),
        "hesitation": status(hesitation_positive),
        "yield_conflict": status(conflict_positive),
        "free_cruising_stability": status(free_cruise_positive),
        "stop_and_go_low_speed_creep": status(stop_go_positive),
        "risk_proximity": status(risk_positive),
        "interaction_comfort": status(comfort_positive),
    }
    return metrics, events


def finalize_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    n = len(metrics_df)
    derived = {
        "following_aggressiveness_score": score_from_parts([
            robust_score(metrics_df.get("mean_thw"), False),
            robust_score(metrics_df.get("mean_front_distance"), False),
            robust_score(metrics_df.get("peak_decel"), True),
            robust_score(metrics_df.get("rms_jerk"), True),
        ], n),
        "lane_change_sharpness_score": score_from_parts([
            robust_score(metrics_df.get("rms_yaw_rate"), True),
            robust_score(metrics_df.get("rms_curvature"), True),
            robust_score(metrics_df.get("rms_lateral_accel"), True),
        ], n),
        "gap_acceptance_score": score_from_parts([
            robust_score(metrics_df.get("target_front_min_gap_during_lane_change"), False),
            robust_score(metrics_df.get("target_rear_min_gap_during_lane_change"), False),
        ], n),
        "hesitation_score": score_from_parts([
            robust_score(metrics_df.get("lane_change_duration"), True),
            robust_score(metrics_df.get("yaw_sign_change_count"), True),
            robust_score(metrics_df.get("lateral_velocity_sign_change_count"), True),
        ], n),
        "gap_pressure_score": score_from_parts([
            robust_score(metrics_df.get("min_front_distance"), False),
            robust_score(metrics_df.get("target_front_min_gap_during_lane_change"), False),
            robust_score(metrics_df.get("target_rear_min_gap_during_lane_change"), False),
            robust_score(metrics_df.get("risk_min_any_gap"), False),
        ], n),
    }
    for name, values in derived.items():
        if values is not None:
            metrics_df[name] = values
    return metrics_df


def event_diagnostics(events_df: pd.DataFrame) -> List[Dict]:
    rows = []
    for event in ALL_EVENTS:
        vals = events_df[event].astype(str)
        n = len(vals)
        pos = int((vals == "positive").sum())
        neg = int((vals == "negative").sum())
        unk = int((vals == "unknown").sum())
        known = pos + neg
        positive_ratio = float(pos / known) if known else np.nan
        unknown_ratio = float(unk / n) if n else np.nan
        rows.append({
            "event": event,
            "n_positive": pos,
            "n_negative": neg,
            "n_unknown": unk,
            "positive_ratio": positive_ratio,
            "unknown_ratio": unknown_ratio,
            "degenerate": bool(np.isfinite(positive_ratio) and (positive_ratio < 0.01 or positive_ratio > 0.95)),
        })
    return rows


def metric_diagnostics(metrics_df: pd.DataFrame, meta_cols: Sequence[str]) -> List[Dict]:
    rows = []
    n = len(metrics_df)
    for col in metrics_df.columns:
        if col in meta_cols:
            continue
        arr = pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(arr)
        vals = arr[ok]
        rec = {"metric": col, "valid_count": int(ok.sum()), "valid_rate": float(ok.sum() / n) if n else np.nan}
        if vals.size:
            rec.update({
                "p01": float(np.percentile(vals, 1)),
                "p50": float(np.percentile(vals, 50)),
                "p99": float(np.percentile(vals, 99)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            })
        else:
            rec.update({"p01": np.nan, "p50": np.nan, "p99": np.nan, "min": np.nan, "max": np.nan})
        rows.append(rec)
    return rows


def schema_obj(args, event_diag, metric_diag, warnings):
    return {
        "version": "stage6c_behavior_event_taxonomy_v2",
        "principle": "Event bin is a task slice/comparable driving context; BDD is computed within each task; style metrics explain drift direction.",
        "event_status_values": ["positive", "negative", "unknown"],
        "primary_events": PRIMARY_EVENTS,
        "secondary_events": SECONDARY_EVENTS,
        "event_diagnostics": event_diag,
        "metric_diagnostics": metric_diag,
        "thresholds": vars(args),
        "raw_array_layout_assumptions": {"ego_seq": EGO, "neighbor_seq_slots": {"0": "front", "1": "left_front", "2": "left_rear", "3": "right_front", "4": "right_rear"}, "neighbor_seq": NEI},
        "warnings_count": len(warnings),
    }


def write_report(path: Path, total_rows: int, total_shards: int, event_diag, metric_diag, warnings):
    deg = [d for d in event_diag if d["degenerate"]]
    lines = [
        "# Stage 6C Behavior-Event Taxonomy v2 Report",
        "",
        "本报告强调 task-conditioned BDD：event bin 是可比较驾驶任务切片，BDD 在任务内部计算，手工指标只用于解释 drift 方向，不作为主要评价对象。",
        "",
        f"- total_rows: {total_rows}",
        f"- total_shards: {total_shards}",
        f"- warnings: {len(warnings)}",
        "",
        "## Event validity diagnostics",
        "",
        "| event | n_positive | n_negative | n_unknown | positive_ratio | unknown_ratio | degenerate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for d in event_diag:
        lines.append(f"| {d['event']} | {d['n_positive']} | {d['n_negative']} | {d['n_unknown']} | {d['positive_ratio']:.6g} | {d['unknown_ratio']:.6g} | {d['degenerate']} |")
    lines.extend(["", "## Metric diagnostics", "", "| metric | valid_count | valid_rate | p01 | p50 | p99 | min | max |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for d in metric_diag:
        lines.append(f"| {d['metric']} | {d['valid_count']} | {d['valid_rate']:.6g} | {d['p01']:.6g} | {d['p50']:.6g} | {d['p99']:.6g} | {d['min']:.6g} | {d['max']:.6g} |")
    lines.extend(["", "## Degenerate events", ""])
    lines.append("- None" if not deg else "\n".join(f"- {d['event']}: positive_ratio={d['positive_ratio']:.6g}" for d in deg))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args):
    t0 = time.time()
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"output_dir exists: {out}; use --overwrite")
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.shard_manifest)
    manifest = read_json(manifest_path)
    shard_entries = manifest.get("shards", manifest.get("shard_infos", []))
    if not shard_entries and "shard_paths" in manifest:
        shard_entries = [{"shard_path": sp} for sp in manifest["shard_paths"]]
    if not shard_entries:
        raise ValueError(f"No shard entries in shard_manifest: {manifest_path}")

    event_rows = []
    metric_rows = []
    warnings = []
    global_row = 0
    for shard_id, shard_info in enumerate(iter_progress(shard_entries, enabled=not args.no_progress, desc="building behavior events v2", unit="shard")):
        shard_dir = resolve_path(manifest_path.parent, shard_info["shard_path"])
        ego_path = shard_dir / "ego_seq.npy"
        if not ego_path.exists():
            raise FileNotFoundError(f"Missing required raw ego sequence file: {ego_path}")
        ego_arr = np.load(ego_path, mmap_mode="r", allow_pickle=False)
        rows = ego_arr.shape[0]
        neighbor_arr = load_optional_array(shard_dir, "neighbor_seq.npy", rows, shard_id, warnings)
        _ = load_optional_array(shard_dir, "neighbor_slot_ids.npy", rows, shard_id, warnings)
        _ = load_optional_array(shard_dir, "interaction_feat_style.npy", rows, shard_id, warnings)
        meta = load_meta_frame(shard_dir, rows, shard_id, warnings)
        for local_row in range(rows):
            metrics, events = derive_row_metrics(np.asarray(ego_arr[local_row]), np.asarray(neighbor_arr[local_row]) if neighbor_arr is not None else None, args.dt, args)
            base = {"global_row": global_row, "shard_id": shard_id, "local_row": local_row}
            for c in META_COLUMNS:
                base[c] = meta.iloc[local_row][c]
            event_rows.append({**base, **events})
            metric_rows.append({**base, **metrics})
            global_row += 1

    events_df = pd.DataFrame(event_rows)
    metrics_df = finalize_scores(pd.DataFrame(metric_rows))
    meta_cols = ["global_row", "shard_id", "local_row"] + META_COLUMNS
    event_diag = event_diagnostics(events_df)
    for diag in event_diag:
        if diag["degenerate"]:
            warnings.append({"warning": "degenerate_event", **diag})
    metric_diag = metric_diagnostics(metrics_df, meta_cols)
    warnings.append({"warning": "completed", "total_rows": int(len(events_df)), "elapsed_sec": float(time.time() - t0)})

    events_df.to_csv(out / "behavior_event_bins_v2.csv", index=False)
    metrics_df.to_csv(out / "behavior_event_metrics_v2.csv", index=False)
    write_json(out / "behavior_event_schema_v2.json", schema_obj(args, event_diag, metric_diag, warnings))
    write_json(out / "behavior_event_warnings_v2.json", warnings)
    write_report(out / "behavior_event_report_v2.md", len(events_df), len(shard_entries), event_diag, metric_diag, warnings)


def parse_args():
    p = argparse.ArgumentParser(description="Build Stage 6C v2 behavior-event bins and style metrics from raw sharded arrays.")
    p.add_argument("--shard_manifest", required=True, help="Path to sharded dataset manifest JSON.")
    p.add_argument("--output_dir", required=True, help="Output directory for v2 behavior-event artifacts.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it already exists.")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    p.add_argument("--dt", type=float, default=0.1, help="Frame time step in seconds.")
    p.add_argument("--min_front_valid_ratio", type=float, default=0.5)
    p.add_argument("--low_speed_mps", type=float, default=2.0)
    p.add_argument("--stop_speed_mps", type=float, default=0.5)
    p.add_argument("--brake_threshold_mps2", type=float, default=1.0)
    p.add_argument("--slower_front_closing_rate", type=float, default=0.5)
    p.add_argument("--overtake_front_gap_m", type=float, default=35.0)
    p.add_argument("--adjacent_available_gap_m", type=float, default=8.0)
    p.add_argument("--lateral_displacement_m", type=float, default=2.0)
    p.add_argument("--lane_change_completion_m", type=float, default=3.0)
    p.add_argument("--heading_change_rad", type=float, default=0.25)
    p.add_argument("--yaw_rate_rms_threshold", type=float, default=0.10)
    p.add_argument("--sign_change_eps", type=float, default=1e-3)
    p.add_argument("--hesitation_sign_changes", type=float, default=3.0)
    p.add_argument("--long_lane_change_s", type=float, default=4.0)
    p.add_argument("--cutin_gap_drop_m", type=float, default=8.0)
    p.add_argument("--cutin_max_gap_m", type=float, default=25.0)
    p.add_argument("--conflict_gap_m", type=float, default=8.0)
    p.add_argument("--side_closing_threshold", type=float, default=1.0)
    p.add_argument("--low_speed_ratio", type=float, default=0.5)
    p.add_argument("--risk_gap_m", type=float, default=5.0)
    p.add_argument("--comfort_rms_jerk", type=float, default=2.0)
    p.add_argument("--comfort_rms_yaw_rate", type=float, default=0.10)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
