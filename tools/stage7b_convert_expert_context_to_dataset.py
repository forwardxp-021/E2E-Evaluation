#!/usr/bin/env python3
"""Convert Stage 7B.1 nuPlan expert context CSVs into a dynamic context dataset.

Stage 7B.2 intentionally converts only ego/object dynamics.  Stage 6-style
map/ODD feature names are reserved in schema/manifest for Stage 7B.3, but this
script does not parse maps and does not fabricate map features.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EGO_FEATURES = [
    "ego_x",
    "ego_y",
    "ego_speed",
    "ego_accel",
    "ego_heading",
    "ego_yaw_rate",
    "valid",
]

NEIGHBOR_FEATURES = [
    "relative_x",
    "relative_y",
    "relative_distance",
    "object_heading",
    "object_length",
    "object_width",
    "object_height",
    "category_id",
    "valid",
]

CATEGORY_ID_MAPPING = {
    "unknown": 0,
    "": 0,
    "vehicle": 1,
    "pedestrian": 2,
    "bicycle": 3,
    "traffic_cone": 4,
    "barrier": 5,
    "czone_sign": 6,
    "generic_object": 7,
}

MAP_ODD_FEATURES_RESERVED = [
    "distance_to_crosswalk_min",
    "has_crosswalk_near_30m",
    "distance_to_stop_sign_min",
    "has_stop_sign_near_40m",
    "lane_curvature_mean",
    "lane_curvature_max",
    "lane_heading_change_total",
    "lane_count_near_30m",
    "road_line_count_near_30m",
    "road_edge_count_near_30m",
    "crosswalk_count_near_30m",
    "stop_sign_count_near_40m",
    "speed_bump_count_near_30m",
    "map_complexity_score",
    "intersection_proxy",
    "map_match_valid",
    "fallback_full_scenario_path",
]

REQUIRED_EGO_COLUMNS = ["db_name", "scene_token", "frame_index_in_scene", "lidar_pc_timestamp", "ego_x", "ego_y", "ego_speed", "ego_heading"]
REQUIRED_OBJECT_COLUMNS = ["db_name", "scene_token", "frame_index_in_scene", "relative_x", "relative_y", "relative_distance", "rank_by_distance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Stage 7B.1 expert CSVs to Stage 6-style dynamic context arrays.")
    parser.add_argument("--expert_ego_csv", required=True)
    parser.add_argument("--expert_objects_csv", required=True)
    parser.add_argument("--selected_scenes_csv")
    parser.add_argument("--output_dir", default="outputs/stage7A_nuplan/expert_context_dataset")
    parser.add_argument("--target_hz", type=float, default=10.0)
    parser.add_argument("--window_sec", type=float, default=8.0)
    parser.add_argument("--stride_sec", type=float, default=4.0)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--min_window_frames", type=int, default=80)
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
    diffs = np.diff(np.sort(arr))
    diffs = diffs[diffs > 0]
    median_diff = float(np.median(diffs)) if len(diffs) else 0.0
    if median_diff > 1e6:
        return arr / 1e9
    if median_diff > 1e3:
        return arr / 1e6
    return arr


def normalize_category(value: str) -> int:
    key = (value or "unknown").strip().lower()
    return CATEGORY_ID_MAPPING.get(key, 0)


def estimate_timing(sorted_rows: List[dict]) -> Tuple[Optional[float], Optional[float], float, np.ndarray]:
    raw_ts = [to_float(r, "lidar_pc_timestamp") for r in sorted_rows]
    ts_sec = timestamp_to_seconds(raw_ts)
    diffs = np.diff(ts_sec)
    diffs = diffs[diffs > 0]
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
    selected = [0]
    next_time = float(ts_sec[0]) + target_dt
    for i in range(1, len(ts_sec)):
        if float(ts_sec[i]) + target_dt * 0.25 >= next_time:
            selected.append(i)
            next_time = float(ts_sec[i]) + target_dt
    return selected


def build_neighbor_tensor(frame_objects: List[dict], k: int) -> np.ndarray:
    tensor = np.zeros((k, len(NEIGHBOR_FEATURES)), dtype=np.float32)
    ranked = sorted(frame_objects, key=lambda r: (to_float(r, "rank_by_distance", math.inf), to_float(r, "relative_distance", math.inf)))[:k]
    for j, row in enumerate(ranked):
        category_value = row.get("category") or row.get("object_category") or row.get("tracked_object_type") or row.get("object_type") or "unknown"
        tensor[j] = np.asarray([
            to_float(row, "relative_x"),
            to_float(row, "relative_y"),
            to_float(row, "relative_distance"),
            to_float(row, "object_heading", 0.0),
            to_float(row, "object_length", 0.0),
            to_float(row, "object_width", 0.0),
            to_float(row, "object_height", 0.0),
            float(normalize_category(category_value)),
            1.0,
        ], dtype=np.float32)
    return tensor


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.target_hz <= 0 or args.window_sec <= 0 or args.stride_sec <= 0 or args.num_neighbors <= 0:
        raise ValueError("target_hz, window_sec, stride_sec, and num_neighbors must be positive.")

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[dict] = []
    ego_rows, _ = read_csv_rows(Path(args.expert_ego_csv), REQUIRED_EGO_COLUMNS, warnings, "expert ego")
    object_rows, _ = read_csv_rows(Path(args.expert_objects_csv), REQUIRED_OBJECT_COLUMNS, warnings, "expert objects")

    scene_names: Dict[Tuple[str, str], str] = {}
    if args.selected_scenes_csv:
        scene_path = Path(args.selected_scenes_csv)
        if scene_path.exists():
            with scene_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    key = (row.get("db_name", ""), row.get("scene_token", ""))
                    scene_names[key] = row.get("scene_name") or row.get("log_name") or ""
        else:
            warnings.append({"type": "missing_selected_scenes_csv", "path": str(scene_path)})

    ego_by_scene: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in ego_rows:
        ego_by_scene[(row["db_name"], row["scene_token"])].append(row)

    objects_by_scene_frame: Dict[Tuple[str, str], Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in object_rows:
        objects_by_scene_frame[(row["db_name"], row["scene_token"])][to_int(row, "frame_index_in_scene")].append(row)

    window_frames = int(args.window_sec * args.target_hz)
    stride_frames = int(args.stride_sec * args.target_hz)
    if window_frames <= 0 or stride_frames <= 0:
        raise ValueError("window_sec * target_hz and stride_sec * target_hz must both be at least 1 frame.")
    min_window_frames = min(args.min_window_frames, window_frames)

    ego_windows: List[np.ndarray] = []
    neighbor_windows: List[np.ndarray] = []
    metadata: List[dict] = []
    dt_values: List[float] = []
    hz_values: List[float] = []

    for key, rows in sorted(ego_by_scene.items()):
        rows = sorted(rows, key=lambda r: (to_int(r, "frame_index_in_scene"), to_float(r, "lidar_pc_timestamp")))
        median_dt, source_hz, irregularity, ts_sec = estimate_timing(rows)
        if median_dt is not None:
            dt_values.append(median_dt)
        if source_hz is not None:
            hz_values.append(source_hz)
        if irregularity > 0.5:
            warnings.append({"type": "timestamp_irregularity_large", "db_name": key[0], "scene_token": key[1], "relative_irregularity": irregularity})

        sample_indices = target_sample_indices(ts_sec, args.target_hz)
        sampled_rows = [rows[i] for i in sample_indices]
        if len(sampled_rows) < min_window_frames:
            warnings.append({"type": "scene_has_too_few_frames", "db_name": key[0], "scene_token": key[1], "sampled_frames": len(sampled_rows), "required_frames": min_window_frames})
            continue

        ego_seq_scene = []
        neighbor_seq_scene = []
        missing_object_frames = 0
        for row in sampled_rows:
            ego_seq_scene.append([
                to_float(row, "ego_x"),
                to_float(row, "ego_y"),
                to_float(row, "ego_speed"),
                to_float(row, "ego_accel", 0.0),
                to_float(row, "ego_heading"),
                to_float(row, "ego_yaw_rate", 0.0),
                1.0,
            ])
            frame_index = to_int(row, "frame_index_in_scene")
            frame_objects = objects_by_scene_frame.get(key, {}).get(frame_index, [])
            if not frame_objects:
                missing_object_frames += 1
            neighbor_seq_scene.append(build_neighbor_tensor(frame_objects, args.num_neighbors))
        if missing_object_frames:
            warnings.append({"type": "object_rows_missing_for_frames", "db_name": key[0], "scene_token": key[1], "missing_frame_count": missing_object_frames})

        ego_arr = np.asarray(ego_seq_scene, dtype=np.float32)
        neigh_arr = np.asarray(neighbor_seq_scene, dtype=np.float32)
        for start in range(0, len(sampled_rows) - window_frames + 1, stride_frames):
            end = start + window_frames
            sample_id = f"sample_{len(metadata):06d}"
            start_row = sampled_rows[start]
            end_row = sampled_rows[end - 1]
            ego_windows.append(ego_arr[start:end])
            neighbor_windows.append(neigh_arr[start:end])
            metadata.append({
                "sample_id": sample_id,
                "source": "nuplan_expert",
                "policy_id": "expert",
                "db_name": key[0],
                "scene_token": key[1],
                "scene_name": scene_names.get(key, ""),
                "start_frame_index": to_int(start_row, "frame_index_in_scene"),
                "end_frame_index": to_int(end_row, "frame_index_in_scene"),
                "start_lidar_pc_timestamp": start_row.get("lidar_pc_timestamp", ""),
                "end_lidar_pc_timestamp": end_row.get("lidar_pc_timestamp", ""),
                "target_hz": args.target_hz,
                "window_sec": args.window_sec,
                "num_frames": window_frames,
                "num_neighbors": args.num_neighbors,
                "map_odd_status": "not_built",
            })

    if not ego_windows:
        warnings.append({"type": "no_windows_generated", "message": "No scene produced a full fixed-length window."})
        ego_seq = np.zeros((0, window_frames, len(EGO_FEATURES)), dtype=np.float32)
        neighbor_seq = np.zeros((0, window_frames, args.num_neighbors, len(NEIGHBOR_FEATURES)), dtype=np.float32)
    else:
        ego_seq = np.stack(ego_windows).astype(np.float32)
        neighbor_seq = np.stack(neighbor_windows).astype(np.float32)

    np.save(out_dir / "ego_seq.npy", ego_seq)
    np.save(out_dir / "neighbor_seq.npy", neighbor_seq)

    meta_cols = ["sample_id", "source", "policy_id", "db_name", "scene_token", "scene_name", "start_frame_index", "end_frame_index", "start_lidar_pc_timestamp", "end_lidar_pc_timestamp", "target_hz", "window_sec", "num_frames", "num_neighbors", "map_odd_status"]
    with (out_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_cols)
        writer.writeheader()
        writer.writerows(metadata)

    manifest = {
        "dataset_type": "nuplan_expert_context",
        "source": "Stage 7B.1 expert context export",
        "ego_seq_path": "ego_seq.npy",
        "neighbor_seq_path": "neighbor_seq.npy",
        "metadata_path": "metadata.csv",
        "num_samples": int(ego_seq.shape[0]),
        "target_hz": args.target_hz,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "num_neighbors": args.num_neighbors,
        "ego_feature_dim": len(EGO_FEATURES),
        "neighbor_feature_dim": len(NEIGHBOR_FEATURES),
        "map_odd_feat_path": None,
        "map_odd_meta_path": None,
        "map_feature_status": "not_built",
        "next_map_stage": "Stage 7B.3 map/ODD feature builder",
    }
    write_json(out_dir / "shard_manifest.json", manifest)
    write_json(out_dir / "feature_schema.json", {"ego_features": EGO_FEATURES, "neighbor_features": NEIGHBOR_FEATURES, "category_id_mapping": CATEGORY_ID_MAPPING, "map_odd_features_reserved": MAP_ODD_FEATURES_RESERVED})
    write_json(out_dir / "warnings.json", {"warnings": warnings})

    def summary(values: List[float]) -> dict:
        if not values:
            return {"count": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {"count": int(len(arr)), "min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}

    report = f"""# Stage 7B.2 Expert Dynamic Context Conversion Report

## Input paths
- expert_ego_csv: `{args.expert_ego_csv}`
- expert_objects_csv: `{args.expert_objects_csv}`
- selected_scenes_csv: `{args.selected_scenes_csv or ''}`

## Output paths
- ego_seq.npy: `{out_dir / 'ego_seq.npy'}`
- neighbor_seq.npy: `{out_dir / 'neighbor_seq.npy'}`
- metadata.csv: `{out_dir / 'metadata.csv'}`
- shard_manifest.json: `{out_dir / 'shard_manifest.json'}`
- feature_schema.json: `{out_dir / 'feature_schema.json'}`
- warnings.json: `{out_dir / 'warnings.json'}`

## Scene and timing summary
- number of scenes read: {len(ego_by_scene)}
- source dt summary: `{json.dumps(summary(dt_values), ensure_ascii=False)}`
- source_hz summary: `{json.dumps(summary(hz_values), ensure_ascii=False)}`

## Generated dataset
- number of generated windows: {int(ego_seq.shape[0])}
- ego_seq shape: {list(ego_seq.shape)}
- neighbor_seq shape: {list(neighbor_seq.shape)}
- metadata rows: {len(metadata)}

## Warning summary
- warning count: {len(warnings)}
- warning types: {sorted(set(w['type'] for w in warnings))}

## Map/ODD note
Map/ODD context is not built in Stage 7B.2. It is reserved for Stage 7B.3.
"""
    (out_dir / "conversion_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote Stage 7B.2 dynamic context dataset to {out_dir} with {ego_seq.shape[0]} windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
