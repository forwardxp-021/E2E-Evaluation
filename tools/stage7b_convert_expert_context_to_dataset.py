#!/usr/bin/env python3
from __future__ import annotations

"""Convert Stage 7B.1 nuPlan expert context CSVs to Stage 6C layout.

Stage 7B.2 converts expert ego/object dynamics only.  It writes the sharded
Waymo-style five-neighbor context interface expected by Stage 6C, while
reserving map/ODD fields for Stage 7B.3.  Neighbor slot assignment is geometric
only; no planner simulation, BDD logic, fake rollout, or Stage 6C result file is
modified by this script.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EGO_FEATURES = ["x", "y", "vx", "vy", "heading", "speed", "accel", "yaw_rate"]
NEIGHBOR_SLOTS = ["front", "left_front", "left_rear", "right_front", "right_rear"]
NEIGHBOR_FEATURES = [
    "valid", "dx", "dy", "rvx", "rvy", "distance", "local_x", "local_y",
    "closing_rate", "ttc", "thw", "neighbor_speed", "neighbor_accel",
    "relative_heading", "neighbor_yaw_rate",
]
MAP_ODD_FEATURES_RESERVED = [
    "distance_to_crosswalk_min", "has_crosswalk_near_30m", "distance_to_stop_sign_min",
    "has_stop_sign_near_40m", "lane_curvature_mean", "lane_curvature_max",
    "lane_heading_change_total", "lane_count_near_30m", "road_line_count_near_30m",
    "road_edge_count_near_30m", "crosswalk_count_near_30m", "stop_sign_count_near_40m",
    "speed_bump_count_near_30m", "map_complexity_score", "intersection_proxy",
    "map_match_valid", "fallback_full_scenario_path",
]
REQUIRED_EGO_COLUMNS = ["db_name", "scene_token", "frame_index_in_scene", "lidar_pc_timestamp", "ego_x", "ego_y", "ego_speed", "ego_heading"]
REQUIRED_OBJECT_COLUMNS = ["db_name", "scene_token", "frame_index_in_scene"]
CATEGORY_COLUMNS = ["category_name", "category", "object_category", "tracked_object_type", "object_type"]
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Stage 7B.1 expert CSVs to a Stage 6C-compatible sharded context dataset.")
    parser.add_argument("--expert_ego_csv", required=True)
    parser.add_argument("--expert_objects_csv", required=True)
    parser.add_argument("--selected_scenes_csv")
    parser.add_argument("--output_dir", default="outputs/stage7A_nuplan/expert_context_dataset")
    parser.add_argument("--target_hz", type=float, default=10.0)
    parser.add_argument("--window_sec", type=float, default=8.0)
    parser.add_argument("--stride_sec", type=float, default=4.0)
    parser.add_argument("--num_neighbors", type=int, default=10, help="Input object cap retained in metadata; output always uses five Stage 6C slots.")
    parser.add_argument("--min_window_frames", type=int, default=80)
    parser.add_argument("--front_lateral_tolerance", type=float, default=2.5)
    parser.add_argument("--side_lateral_threshold", type=float, default=2.0)
    parser.add_argument("--rear_tolerance", type=float, default=5.0)
    parser.add_argument("--ttc_cap", type=float, default=999.0)
    parser.add_argument("--thw_cap", type=float, default=999.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path, required: Sequence[str], warnings: List[dict], label: str) -> Tuple[List[dict], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = [c for c in required if c not in columns]
        if missing:
            warnings.append({"type": "missing_columns", "input": str(path), "label": label, "missing_columns": missing})
            raise ValueError(f"{label} CSV {path} is missing required columns: {missing}")
        return list(reader), columns


def to_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Column {key} has non-numeric value {value!r} in row with scene_token={row.get('scene_token')}") from exc


def to_int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Column {key} has non-integer value {value!r} in row with scene_token={row.get('scene_token')}") from exc


def timestamp_to_seconds(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return arr
    diffs = np.diff(np.sort(arr)); diffs = diffs[diffs > 0]
    median_diff = float(np.median(diffs)) if len(diffs) else 0.0
    if median_diff > 1e6:
        return arr / 1e9
    if median_diff > 1e3:
        return arr / 1e6
    return arr


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def rotate_to_local(x: np.ndarray | float, y: np.ndarray | float, heading: np.ndarray | float) -> Tuple[np.ndarray | float, np.ndarray | float]:
    c = np.cos(heading); s = np.sin(heading)
    return c * x + s * y, -s * x + c * y


def finite_diff(values: np.ndarray, ts_sec: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float64)
    return np.gradient(values.astype(np.float64), ts_sec.astype(np.float64), edge_order=1)


def estimate_timing(sorted_rows: List[dict]) -> Tuple[Optional[float], Optional[float], float, np.ndarray]:
    raw_ts = [to_float(r, "lidar_pc_timestamp") for r in sorted_rows]
    ts_sec = timestamp_to_seconds(raw_ts)
    diffs = np.diff(ts_sec); diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None, None, 0.0, ts_sec
    median_dt = float(np.median(diffs))
    source_hz = 1.0 / median_dt if median_dt > 0 else None
    irregularity = float(np.max(np.abs(diffs - median_dt)) / median_dt) if median_dt > 0 else 0.0
    return median_dt, source_hz, irregularity, ts_sec


def target_sample_indices(ts_sec: np.ndarray, target_hz: float) -> List[int]:
    if len(ts_sec) == 0:
        return []
    target_dt = 1.0 / target_hz
    selected = [0]; next_time = float(ts_sec[0]) + target_dt
    for i in range(1, len(ts_sec)):
        if float(ts_sec[i]) + target_dt * 0.25 >= next_time:
            selected.append(i); next_time = float(ts_sec[i]) + target_dt
    return selected


def deterministic_split(scenario_id: str) -> str:
    bucket = int(hashlib.sha1(scenario_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def first_present(row: dict, names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            return value
    return default


def object_xy(row: dict, ego_row: dict) -> Tuple[float, float]:
    if row.get("object_x", "") not in (None, "") and row.get("object_y", "") not in (None, ""):
        return to_float(row, "object_x"), to_float(row, "object_y")
    if row.get("x", "") not in (None, "") and row.get("y", "") not in (None, ""):
        return to_float(row, "x"), to_float(row, "y")
    rel_x = to_float(row, "relative_x", to_float(row, "dx", 0.0))
    rel_y = to_float(row, "relative_y", to_float(row, "dy", 0.0))
    h = to_float(ego_row, "ego_heading")
    return to_float(ego_row, "ego_x") + math.cos(h) * rel_x - math.sin(h) * rel_y, to_float(ego_row, "ego_y") + math.sin(h) * rel_x + math.cos(h) * rel_y


def compute_scene_ego(rows: List[dict], ts_sec: np.ndarray, warnings: List[dict], scene_key: Tuple[str, str]) -> dict:
    x = np.asarray([to_float(r, "ego_x") for r in rows], dtype=np.float64)
    y = np.asarray([to_float(r, "ego_y") for r in rows], dtype=np.float64)
    speed = np.asarray([to_float(r, "ego_speed") for r in rows], dtype=np.float64)
    heading = np.asarray([to_float(r, "ego_heading") for r in rows], dtype=np.float64)
    if "ego_vx" in rows[0] and "ego_vy" in rows[0] and any(r.get("ego_vx", "") not in (None, "") for r in rows):
        vx = np.asarray([to_float(r, "ego_vx") for r in rows], dtype=np.float64)
        vy = np.asarray([to_float(r, "ego_vy") for r in rows], dtype=np.float64)
    else:
        vx = finite_diff(x, ts_sec); vy = finite_diff(y, ts_sec)
        warnings.append({"type": "ego_velocity_finite_difference", "db_name": scene_key[0], "scene_token": scene_key[1]})
    if any(r.get("ego_accel", "") not in (None, "") for r in rows):
        accel = np.asarray([to_float(r, "ego_accel") for r in rows], dtype=np.float64)
    else:
        accel = finite_diff(speed, ts_sec) if len(rows) > 1 else np.zeros_like(speed)
        warnings.append({"type": "ego_accel_finite_difference_or_zero", "db_name": scene_key[0], "scene_token": scene_key[1]})
    if any(r.get("ego_yaw_rate", "") not in (None, "") for r in rows):
        yaw_rate = np.asarray([to_float(r, "ego_yaw_rate") for r in rows], dtype=np.float64)
    else:
        yaw_rate = finite_diff(np.unwrap(heading), ts_sec) if len(rows) > 1 else np.zeros_like(heading)
        warnings.append({"type": "ego_yaw_rate_finite_difference_or_zero", "db_name": scene_key[0], "scene_token": scene_key[1]})
    vx_local, vy_local = rotate_to_local(vx, vy, heading)
    return {"x": x, "y": y, "speed": speed, "heading": heading, "vx": vx, "vy": vy, "vx_local": vx_local, "vy_local": vy_local, "accel": accel, "yaw_rate": yaw_rate}


def build_object_kinematics(objects_by_frame: Dict[int, List[dict]], ego_by_frame: Dict[int, dict], ego_ts_sec_by_frame: Dict[int, float], warnings: List[dict], scene_key: Tuple[str, str]) -> Dict[Tuple[str, int], dict]:
    per_track: Dict[str, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for frame, objs in objects_by_frame.items():
        ego_row = ego_by_frame.get(frame)
        if ego_row is None:
            continue
        ts = float(ego_ts_sec_by_frame.get(frame, to_float(ego_row, "lidar_pc_timestamp")))
        for idx, obj in enumerate(objs):
            token = first_present(obj, ["track_token", "tracked_object_token", "object_id", "token", "id"], f"frame{frame}_obj{idx}")
            ox, oy = object_xy(obj, ego_row)
            per_track[token].append((frame, ts, ox, oy, to_float(obj, "object_heading", to_float(obj, "heading", to_float(ego_row, "ego_heading")))))
    kin: Dict[Tuple[str, int], dict] = {}
    for token, vals in per_track.items():
        vals = sorted(vals, key=lambda v: (v[1], v[0]))
        frames = [v[0] for v in vals]; ts = np.asarray([v[1] for v in vals], dtype=np.float64)
        x = np.asarray([v[2] for v in vals], dtype=np.float64); y = np.asarray([v[3] for v in vals], dtype=np.float64)
        hdg = np.asarray([v[4] for v in vals], dtype=np.float64)
        vx = finite_diff(x, ts); vy = finite_diff(y, ts); spd = np.hypot(vx, vy)
        acc = finite_diff(spd, ts); yr = finite_diff(np.unwrap(hdg), ts)
        for i, frame in enumerate(frames):
            kin[(token, frame)] = {"x": x[i], "y": y[i], "vx": vx[i], "vy": vy[i], "speed": spd[i], "accel": acc[i], "heading": hdg[i], "yaw_rate": yr[i]}
    if per_track:
        warnings.append({"type": "neighbor_kinematics_finite_difference", "db_name": scene_key[0], "scene_token": scene_key[1], "track_count": len(per_track)})
    return kin


def choose_slot(candidates: List[Tuple[float, np.ndarray, str]], used: set) -> Tuple[np.ndarray, str]:
    for _, feat, token in sorted(candidates, key=lambda x: x[0]):
        if token not in used:
            used.add(token); return feat, token
    return np.zeros((len(NEIGHBOR_FEATURES),), dtype=np.float32), ""


def build_stage6c_neighbors(frame_objects: List[dict], ego_row: dict, ego_state: dict, obj_kin: Dict[Tuple[str, int], dict], args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    frame = to_int(ego_row, "frame_index_in_scene")
    ego_x = to_float(ego_row, "ego_x"); ego_y = to_float(ego_row, "ego_y"); ego_heading = to_float(ego_row, "ego_heading")
    ego_vx_l = float(ego_state["vx_local"]); ego_vy_l = float(ego_state["vy_local"]); ego_speed = float(ego_state["speed"])
    cands = {name: [] for name in NEIGHBOR_SLOTS}
    for idx, row in enumerate(frame_objects[: max(args.num_neighbors, 1)]):
        token = first_present(row, ["track_token", "tracked_object_token", "object_id", "token", "id"], f"frame{frame}_obj{idx}")
        ox, oy = object_xy(row, ego_row)
        kin = obj_kin.get((token, frame), {"x": ox, "y": oy, "vx": 0.0, "vy": 0.0, "speed": 0.0, "accel": 0.0, "heading": to_float(row, "object_heading", to_float(row, "heading", ego_heading)), "yaw_rate": 0.0})
        dx, dy = rotate_to_local(float(kin["x"]) - ego_x, float(kin["y"]) - ego_y, ego_heading)
        nvx_l, nvy_l = rotate_to_local(float(kin["vx"]), float(kin["vy"]), ego_heading)
        rvx = float(nvx_l - ego_vx_l); rvy = float(nvy_l - ego_vy_l)
        dist = float(math.hypot(dx, dy))
        closing = float(ego_vx_l - nvx_l)
        ttc = min(args.ttc_cap, dist / max(closing, EPS)) if closing > EPS else args.ttc_cap
        thw = min(args.thw_cap, dist / max(ego_speed, EPS))
        feat = np.asarray([1.0, dx, dy, rvx, rvy, dist, dx, dy, closing, ttc, thw, float(kin["speed"]), float(kin["accel"]), wrap_angle(float(kin["heading"]) - ego_heading), float(kin["yaw_rate"])], dtype=np.float32)
        if dx > 0 and abs(dy) <= args.front_lateral_tolerance:
            cands["front"].append((dx, feat, token))
        if dy > args.side_lateral_threshold and dx >= -args.rear_tolerance:
            cands["left_front"].append((dist - 0.01 * max(dx, 0.0), feat, token))
        if dy > args.side_lateral_threshold and dx < 0:
            cands["left_rear"].append((dist, feat, token))
        if dy < -args.side_lateral_threshold and dx >= -args.rear_tolerance:
            cands["right_front"].append((dist - 0.01 * max(dx, 0.0), feat, token))
        if dy < -args.side_lateral_threshold and dx < 0:
            cands["right_rear"].append((dist, feat, token))
    out = np.zeros((len(NEIGHBOR_SLOTS), len(NEIGHBOR_FEATURES)), dtype=np.float32)
    ids = np.full((len(NEIGHBOR_SLOTS),), "", dtype="U128")
    used = set()
    for i, slot in enumerate(NEIGHBOR_SLOTS):
        out[i], ids[i] = choose_slot(cands[slot], used)
    return out, ids


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_metadata_csv(path: Path, rows: List[dict]) -> None:
    cols = ["scenario_id", "target_agent_id", "start", "window_len", "split", "sample_id", "source", "policy_id", "db_name", "scene_token", "scene_name", "start_frame_index", "end_frame_index", "start_lidar_pc_timestamp", "end_lidar_pc_timestamp", "target_hz", "window_sec", "num_frames", "num_neighbors", "map_odd_status", "slot_assignment_mode"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader(); writer.writerows(rows)


def build_interaction_features(ego_seq: np.ndarray, neighbor_seq: np.ndarray, dt: float, warnings: List[dict]) -> Tuple[np.ndarray, List[str]]:
    try:
        from tools.interaction_context_features import aggregate_interaction_features
        feats = []
        names: List[str] = []
        for i in range(ego_seq.shape[0]):
            feat, names = aggregate_interaction_features(ego_seq[i], neighbor_seq[i], dt)
            feats.append(feat)
        return np.stack(feats).astype(np.float32) if feats else np.zeros((0, 33), dtype=np.float32), names
    except Exception as exc:
        warnings.append({"type": "interaction_feature_generation_failed", "message": str(exc)})
        return np.zeros((ego_seq.shape[0], 0), dtype=np.float32), []


def main() -> int:
    args = parse_args()
    if args.target_hz <= 0 or args.window_sec <= 0 or args.stride_sec <= 0 or args.num_neighbors <= 0:
        raise ValueError("target_hz, window_sec, stride_sec, and num_neighbors must be positive.")
    out_dir = Path(args.output_dir); shard_dir = out_dir / "shards" / "shard_000000"
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[dict] = [{"type": "geometric_slot_assignment_only", "message": "Stage 7B.2 uses geometric neighbor slot assignment only; map/lane-aware assignment will be improved in Stage 7B.3/7B.4."}]
    ego_rows, _ = read_csv_rows(Path(args.expert_ego_csv), REQUIRED_EGO_COLUMNS, warnings, "expert ego")
    object_rows, object_cols = read_csv_rows(Path(args.expert_objects_csv), REQUIRED_OBJECT_COLUMNS, warnings, "expert objects")
    if not any(c in object_cols for c in CATEGORY_COLUMNS):
        warnings.append({"type": "object_category_column_missing", "checked_columns": CATEGORY_COLUMNS})

    scene_names: Dict[Tuple[str, str], str] = {}
    if args.selected_scenes_csv:
        scene_path = Path(args.selected_scenes_csv)
        if scene_path.exists():
            with scene_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    scene_names[(row.get("db_name", ""), row.get("scene_token", ""))] = row.get("scene_name") or row.get("log_name") or ""
        else:
            warnings.append({"type": "missing_selected_scenes_csv", "path": str(scene_path)})

    ego_by_scene: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in ego_rows:
        ego_by_scene[(row["db_name"], row["scene_token"])].append(row)
    objects_by_scene_frame: Dict[Tuple[str, str], Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in object_rows:
        objects_by_scene_frame[(row["db_name"], row["scene_token"])][to_int(row, "frame_index_in_scene")].append(row)

    window_frames = int(args.window_sec * args.target_hz); stride_frames = int(args.stride_sec * args.target_hz)
    min_window_frames = min(args.min_window_frames, window_frames)
    ego_windows: List[np.ndarray] = []; neighbor_windows: List[np.ndarray] = []; slot_id_windows: List[np.ndarray] = []; metadata: List[dict] = []
    dt_values: List[float] = []; hz_values: List[float] = []

    for key, rows in sorted(ego_by_scene.items()):
        rows = sorted(rows, key=lambda r: (to_int(r, "frame_index_in_scene"), to_float(r, "lidar_pc_timestamp")))
        median_dt, source_hz, irregularity, ts_sec = estimate_timing(rows)
        if median_dt is not None: dt_values.append(median_dt)
        if source_hz is not None: hz_values.append(source_hz)
        if irregularity > 0.5:
            warnings.append({"type": "timestamp_irregularity_large", "db_name": key[0], "scene_token": key[1], "relative_irregularity": irregularity})
        sample_indices = target_sample_indices(ts_sec, args.target_hz)
        sampled_rows = [rows[i] for i in sample_indices]
        sampled_ts = ts_sec[sample_indices] if sample_indices else np.asarray([], dtype=np.float64)
        if len(sampled_rows) < min_window_frames:
            warnings.append({"type": "scene_has_too_few_frames", "db_name": key[0], "scene_token": key[1], "sampled_frames": len(sampled_rows), "required_frames": min_window_frames})
            continue
        scene_ego = compute_scene_ego(sampled_rows, sampled_ts, warnings, key)
        ego_by_frame = {to_int(r, "frame_index_in_scene"): r for r in sampled_rows}
        ego_ts_sec_by_frame = {to_int(r, "frame_index_in_scene"): float(sampled_ts[i]) for i, r in enumerate(sampled_rows)}
        obj_kin = build_object_kinematics(objects_by_scene_frame.get(key, {}), ego_by_frame, ego_ts_sec_by_frame, warnings, key)
        ego_scene = np.zeros((len(sampled_rows), len(EGO_FEATURES)), dtype=np.float32)
        neigh_scene = np.zeros((len(sampled_rows), len(NEIGHBOR_SLOTS), len(NEIGHBOR_FEATURES)), dtype=np.float32)
        slot_ids_scene = np.full((len(sampled_rows), len(NEIGHBOR_SLOTS)), "", dtype="U128")
        missing_object_frames = 0
        for i, row in enumerate(sampled_rows):
            frame = to_int(row, "frame_index_in_scene")
            ref_x = to_float(sampled_rows[0], "ego_x"); ref_y = to_float(sampled_rows[0], "ego_y"); ref_h = to_float(sampled_rows[0], "ego_heading")
            lx, ly = rotate_to_local(scene_ego["x"][i] - ref_x, scene_ego["y"][i] - ref_y, ref_h)
            lvx, lvy = rotate_to_local(scene_ego["vx"][i], scene_ego["vy"][i], ref_h)
            ego_scene[i] = np.asarray([lx, ly, lvx, lvy, wrap_angle(scene_ego["heading"][i] - ref_h), scene_ego["speed"][i], scene_ego["accel"][i], scene_ego["yaw_rate"][i]], dtype=np.float32)
            frame_objects = objects_by_scene_frame.get(key, {}).get(frame, [])
            if not frame_objects: missing_object_frames += 1
            ego_state = {name: scene_ego[name][i] for name in ["vx_local", "vy_local", "speed"]}
            neigh_scene[i], slot_ids_scene[i] = build_stage6c_neighbors(frame_objects, row, ego_state, obj_kin, args)
        if missing_object_frames:
            warnings.append({"type": "object_rows_missing_for_frames", "db_name": key[0], "scene_token": key[1], "missing_frame_count": missing_object_frames})
        scenario_id = f"{key[0]}|{key[1]}"; split = deterministic_split(scenario_id)
        for start in range(0, len(sampled_rows) - window_frames + 1, stride_frames):
            end = start + window_frames; sample_id = f"sample_{len(metadata):06d}"
            start_row = sampled_rows[start]; end_row = sampled_rows[end - 1]
            ego_windows.append(ego_scene[start:end]); neighbor_windows.append(np.transpose(neigh_scene[start:end], (1, 0, 2))); slot_id_windows.append(np.transpose(slot_ids_scene[start:end], (1, 0)))
            metadata.append({"scenario_id": scenario_id, "target_agent_id": "ego", "start": start, "window_len": window_frames, "split": split, "sample_id": sample_id, "source": "nuplan_expert", "policy_id": "expert", "db_name": key[0], "scene_token": key[1], "scene_name": scene_names.get(key, ""), "start_frame_index": to_int(start_row, "frame_index_in_scene"), "end_frame_index": to_int(end_row, "frame_index_in_scene"), "start_lidar_pc_timestamp": start_row.get("lidar_pc_timestamp", ""), "end_lidar_pc_timestamp": end_row.get("lidar_pc_timestamp", ""), "target_hz": args.target_hz, "window_sec": args.window_sec, "num_frames": window_frames, "num_neighbors": len(NEIGHBOR_SLOTS), "map_odd_status": "not_built", "slot_assignment_mode": "geometric_v1"})

    if ego_windows:
        ego_seq = np.stack(ego_windows).astype(np.float32); neighbor_seq = np.stack(neighbor_windows).astype(np.float32); neighbor_slot_ids = np.stack(slot_id_windows)
    else:
        warnings.append({"type": "no_windows_generated", "message": "No scene produced a full fixed-length window."})
        ego_seq = np.zeros((0, window_frames, len(EGO_FEATURES)), dtype=np.float32); neighbor_seq = np.zeros((0, len(NEIGHBOR_SLOTS), window_frames, len(NEIGHBOR_FEATURES)), dtype=np.float32); neighbor_slot_ids = np.empty((0, len(NEIGHBOR_SLOTS), window_frames), dtype="U128")

    n_windows, n_slots, n_frames, n_neighbor_features = neighbor_seq.shape
    neighbor_flat = neighbor_seq.transpose(0, 2, 1, 3).reshape(n_windows, n_frames, n_slots * n_neighbor_features)
    context_traj = np.concatenate([ego_seq, neighbor_flat], axis=2).astype(np.float32)
    neighbor_valid = neighbor_seq[..., 0] > 0.5
    context_mask = neighbor_valid.transpose(0, 2, 1).astype(np.float32)
    context_mask_window = neighbor_valid.any(axis=2).astype(np.float32)
    split_arr = np.asarray([m["split"] for m in metadata], dtype="U16")
    meta_arr = np.asarray(metadata, dtype=object)
    interaction_raw, interaction_names = build_interaction_features(ego_seq, neighbor_seq, 1.0 / args.target_hz, warnings)

    np.save(shard_dir / "ego_seq.npy", ego_seq); np.save(shard_dir / "neighbor_seq.npy", neighbor_seq); np.save(shard_dir / "neighbor_slot_ids.npy", neighbor_slot_ids)
    np.save(shard_dir / "context_traj.npy", context_traj); np.save(shard_dir / "context_mask.npy", context_mask); np.save(shard_dir / "context_mask_window.npy", context_mask_window)
    np.save(shard_dir / "meta.npy", meta_arr); np.save(shard_dir / "split.npy", split_arr); np.save(shard_dir / "interaction_feat_style_raw.npy", interaction_raw); np.save(shard_dir / "interaction_feat_style.npy", interaction_raw.copy())
    write_metadata_csv(shard_dir / "metadata.csv", metadata)

    manifest = {"dataset_type": "nuplan_expert_context_stage6c_compatible", "shard_paths": ["shards/shard_000000"], "total_shards": 1, "total_windows": int(ego_seq.shape[0]), "required_files_per_shard": ["ego_seq.npy", "neighbor_seq.npy", "metadata.csv", "split.npy"], "map_odd_feat_path": None, "map_feature_status": "not_built", "next_map_stage": "Stage 7B.3 map/ODD feature builder", "target_hz": args.target_hz, "window_sec": args.window_sec, "stride_sec": args.stride_sec}
    write_json(out_dir / "shard_manifest.json", manifest)
    schema = {"ego_features": EGO_FEATURES, "neighbor_features": NEIGHBOR_FEATURES, "neighbor_slots": NEIGHBOR_SLOTS, "map_odd_features_reserved": MAP_ODD_FEATURES_RESERVED, "interaction_features": interaction_names, "interaction_feature_note": "Canonical 33 features from tools.interaction_context_features.aggregate_interaction_features when generation succeeds."}
    write_json(out_dir / "feature_schema.json", schema)
    write_json(out_dir / "warnings.json", {"warnings": warnings})
    write_json(shard_dir / "shard_summary.json", {"num_windows": int(ego_seq.shape[0]), "ego_seq_shape": list(ego_seq.shape), "neighbor_seq_shape": list(neighbor_seq.shape), "context_traj_shape": list(context_traj.shape), "context_mask_shape": list(context_mask.shape), "context_mask_window_shape": list(context_mask_window.shape), "slot_assignment_mode": "geometric_v1"})

    def summary(values: List[float]) -> dict:
        if not values: return {"count": 0}
        arr = np.asarray(values, dtype=np.float64); return {"count": int(len(arr)), "min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}
    report = f"""# Stage 7B.2 Expert Dynamic Context Conversion Report

## Input paths
- expert_ego_csv: `{args.expert_ego_csv}`
- expert_objects_csv: `{args.expert_objects_csv}`
- selected_scenes_csv: `{args.selected_scenes_csv or ''}`

## Output paths
- shard_manifest.json: `{out_dir / 'shard_manifest.json'}`
- feature_schema.json: `{out_dir / 'feature_schema.json'}`
- shard directory: `{shard_dir}`
- warnings.json: `{out_dir / 'warnings.json'}`

## Scene and timing summary
- number of scenes read: {len(ego_by_scene)}
- source dt summary: `{json.dumps(summary(dt_values), ensure_ascii=False)}`
- source_hz summary: `{json.dumps(summary(hz_values), ensure_ascii=False)}`

## Generated dataset
- number of generated windows: {int(ego_seq.shape[0])}
- ego_seq shape: {list(ego_seq.shape)}
- neighbor_seq shape: {list(neighbor_seq.shape)}
- context_traj shape: {list(context_traj.shape)}
- context_mask shape: {list(context_mask.shape)}
- context_mask_window shape: {list(context_mask_window.shape)}
- metadata rows: {len(metadata)}
- interaction feature shape: {list(interaction_raw.shape)}

Stage 7B.2 dynamic outputs are aligned with the Waymo 5-neighbor context dataset layout except map/ODD features, which are reserved for Stage 7B.3.

## Slot assignment note
Stage 7B.2 uses geometric slot assignment only. Map/lane-aware assignment will be improved in Stage 7B.3/7B.4.

## Map/ODD note
Map/ODD context is not built in Stage 7B.2. It is reserved for Stage 7B.3.

## Warning summary
- warning count: {len(warnings)}
- warning types: {sorted(set(w['type'] for w in warnings))}
"""
    (out_dir / "conversion_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote Stage 6C-compatible Stage 7B.2 dataset to {out_dir} with {ego_seq.shape[0]} windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
