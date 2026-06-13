#!/usr/bin/env python3
from __future__ import annotations

"""Stage 7B.3 nuPlan map-lite / ODD feature builder.

Reads Stage 7B.2 sharded dynamic-context metadata, reconstructs ego poses from
nuPlan SQLite DBs when possible, and writes row-aligned map/ODD-lite features.
The script is intentionally lightweight: no rendering, no vector-map dump, no
training, and no merge with dynamic features (reserved for Stage 7B.4).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import importlib
import importlib.util
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SENTINEL = -1.0
LAYER_ALIASES = {
    "lane": ["LANE", "lane"],
    "lane_connector": ["LANE_CONNECTOR", "lane_connector"],
    "crosswalk": ["CROSSWALK", "crosswalk"],
    "stop_line": ["STOP_LINE", "stop_line"],
    "roadblock": ["ROADBLOCK", "ROADBLOCK_CONNECTOR", "roadblock", "roadblock_connector"],
    "drivable_area": ["DRIVABLE_AREA", "drivable_area"],
    "road_edge": ["ROAD_EDGE", "road_edge"],
}
FEATURE_GROUPS = {
    "validity": ["map_available", "num_valid_ego_map_queries", "valid_ego_map_query_ratio"],
    "lane_context": [
        "nearby_lane_count_mean", "nearby_lane_count_max", "nearby_lane_connector_count_mean",
        "nearby_lane_connector_count_max", "ego_on_lane_ratio", "ego_on_lane_connector_ratio",
    ],
    "intersection_proxy": [
        "intersection_proxy_ratio", "intersection_proxy_count_mean", "intersection_proxy_count_max",
        "distance_to_intersection_proxy_min", "distance_to_intersection_proxy_mean",
    ],
    "traffic_control_proxy": [
        "crosswalk_count_mean", "crosswalk_count_max", "distance_to_crosswalk_min",
        "stop_line_count_mean", "stop_line_count_max", "distance_to_stop_line_min",
        "traffic_light_related_count_mean", "traffic_light_related_count_max",
    ],
    "geometry": [
        "lane_heading_change_mean", "lane_heading_change_std", "lane_curvature_proxy_mean",
        "lane_curvature_proxy_p95", "lane_curvature_proxy_max", "ego_path_curvature_proxy_mean",
        "ego_path_curvature_proxy_p95",
    ],
    "road_edge_proxy": [
        "distance_to_drivable_area_edge_min", "distance_to_drivable_area_edge_mean",
        "road_edge_count_mean", "road_edge_count_max",
    ],
    "complexity": ["map_object_count_mean", "map_object_count_max", "map_complexity_score", "odd_complexity_score"],
}
FEATURE_NAMES = [name for group in FEATURE_GROUPS.values() for name in group]
KEY_COLUMNS = [
    "scenario_id", "sample_id", "db_name", "scene_token", "scene_name", "start", "window_len",
    "start_frame_index", "end_frame_index", "start_lidar_pc_timestamp", "end_lidar_pc_timestamp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 7B.3 nuPlan map-lite / ODD feature builder aligned to Stage 7B.2 windows.")
    p.add_argument("--nuplan_db_root", required=True, help="Root containing nuPlan SQLite DB files, or a split directory such as .../splits/mini.")
    p.add_argument("--nuplan_map_root", required=True, help="nuPlan map root used by nuPlan map API.")
    p.add_argument("--input_dynamic_dir", default="outputs/stage7A_nuplan/expert_context_dataset", help="Stage 7B.2 output directory containing shard_manifest.json.")
    p.add_argument("--output_dir", default="outputs/stage7b3_nuplan_map_odd", help="Output directory for map/ODD features.")
    p.add_argument("--split", default="mini", help="nuPlan split label for report/API hints.")
    p.add_argument("--max_scenarios", type=int, default=0, help="Optional cap on unique scenarios processed; 0 means all.")
    p.add_argument("--radius_m", type=float, default=50.0, help="Radius for local map object queries.")
    p.add_argument("--sample_stride", type=int, default=5, help="Use every k-th ego pose inside each window for map queries.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out}. Pass --overwrite to replace it.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_dynamic_rows(input_dir: Path) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Any]]:
    manifest_path = input_dir / "shard_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Stage 7B.2 shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, str]] = []
    shard_paths = manifest.get("shard_paths") or []
    if not shard_paths:
        raise ValueError(f"Stage 7B.2 manifest has no shard_paths: {manifest_path}")
    for shard_id, rel in enumerate(shard_paths):
        shard_dir = input_dir / rel
        meta_rows = read_csv(shard_dir / "metadata.csv")
        for local_row, row in enumerate(meta_rows):
            row = dict(row)
            row["shard_id"] = str(shard_id)
            row["local_row"] = str(local_row)
            rows.append(row)
    return rows, [str(input_dir / rel) for rel in shard_paths], manifest


def scenario_limited(rows: List[Dict[str, str]], max_scenarios: int) -> List[Dict[str, str]]:
    if max_scenarios <= 0:
        return rows
    seen: set[str] = set(); out: List[Dict[str, str]] = []
    for r in rows:
        sid = r.get("scenario_id") or f"{r.get('db_name','')}|{r.get('scene_token','')}"
        if sid not in seen:
            if len(seen) >= max_scenarios:
                continue
            seen.add(sid)
        out.append(r)
    return out


def find_db(db_root: Path, db_name: str) -> Optional[Path]:
    candidates = [db_root / db_name, db_root / f"{Path(db_name).stem}.db", db_root / f"{db_name}.db"]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    stem = Path(db_name).stem
    matches = sorted(p for p in db_root.rglob("*") if p.is_file() and p.suffix.lower() in {".db", ".sqlite", ".sqlite3"} and p.stem == stem)
    return matches[0] if matches else None


def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [str(r[1]) for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def bytes_or_text(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, bytes): return v.hex()
    return str(v)


def row_get(row: Optional[sqlite3.Row], col: Optional[str]) -> Any:
    if row is None or not col:
        return None
    try:
        return row[col]
    except Exception:
        return None


def to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        out = float(v)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def first_existing(cols: Sequence[str], names: Sequence[str]) -> str:
    return next((n for n in names if n in cols), "")


def select_rows(con: sqlite3.Connection, table: str, cols: Sequence[str], where: str = "", params: Sequence[Any] = (), order_by: str = "", limit: int = 0) -> List[sqlite3.Row]:
    sql = f"SELECT {', '.join(q(c) for c in cols)} FROM {q(table)}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit > 0:
        sql += f" LIMIT {int(limit)}"
    return list(con.execute(sql, params).fetchall())


def fetch_one_by_value(con: sqlite3.Connection, table: str, cols: Sequence[str], key_col: str, value: Any) -> Optional[sqlite3.Row]:
    for candidate in ([value] if not isinstance(value, str) else sqlite_token_candidates(value)):
        rows = select_rows(con, table, cols, where=f"{q(key_col)}=?", params=(candidate,), limit=1)
        if rows:
            return rows[0]
    return None



def warn_counter_key(prefix: str, layer: str, message: str) -> str:
    msg = (message or "").split("\n", 1)[0][:160]
    return f"{prefix}|{layer}|{msg}"


def sqlite_token_candidates(token: str) -> List[Any]:
    vals: List[Any] = []
    if token:
        vals.append(token)
        try:
            if len(token) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in token):
                vals.append(bytes.fromhex(token))
        except Exception:
            pass
    # Preserve order while deduplicating by representation/type.
    out: List[Any] = []
    seen: set[Tuple[str, str]] = set()
    for v in vals:
        key = (type(v).__name__, bytes_or_text(v))
        if key not in seen:
            seen.add(key); out.append(v)
    return out


def first_row_value(con: sqlite3.Connection, table: str, columns: Sequence[str], where: str = "", params: Sequence[Any] = ()) -> str:
    cols = table_columns(con, table)
    for col in columns:
        if col not in cols:
            continue
        sql = f"SELECT {q(col)} FROM {q(table)} {where} LIMIT 2"
        rows = con.execute(sql, params).fetchall()
        values = [bytes_or_text(r[0]) for r in rows if bytes_or_text(r[0])]
        if len(set(values)) == 1:
            return values[0]
    return ""


def map_name_candidates_from_row(row: Dict[str, str]) -> Tuple[List[str], List[str]]:
    strong: List[str] = []
    weak: List[str] = []
    for k in ("map_name", "location"):
        v = str(row.get(k, "") or "").strip()
        if v and v not in strong:
            strong.append(v)
    for k in ("map_version", "log_name"):
        v = str(row.get(k, "") or "").strip()
        if v and v not in strong and v not in weak:
            weak.append(v)
    return strong, weak


def add_candidate(target: List[str], value: str) -> None:
    value = str(value or "").strip()
    if value and value not in target:
        target.append(value)


def resolve_map_name_candidates_from_db(db_path: Optional[Path], scene_token: str, warnings: List[dict], row: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    """Return strong and weak map-name candidates; final acceptance happens after Map API load."""
    strong, weak = map_name_candidates_from_row(row or {})
    if db_path is None:
        warnings.append({"type": "map_name_resolution_failed", "reason": "db_path_missing", "scene_token": scene_token})
        return strong, weak
    try:
        con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        strong_cols = ("map_name", "location")
        weak_cols = ("map_version", "log_name")
        if "scene" in tables:
            scene_cols = table_columns(con, "scene")
            scene_key = first_existing(scene_cols, ["token", "scene_token", "id"])
            scene_row = fetch_one_by_value(con, "scene", scene_cols, scene_key, scene_token) if scene_key and scene_token else None
            if scene_row is None and scene_cols:
                scene_row = select_rows(con, "scene", scene_cols, limit=1)[0]
            if scene_row is not None:
                for col in strong_cols:
                    add_candidate(strong, bytes_or_text(row_get(scene_row, col) if col in scene_cols else None))
                for col in weak_cols:
                    add_candidate(weak, bytes_or_text(row_get(scene_row, col) if col in scene_cols else None))
                log_ref = first_existing(scene_cols, ["log_token", "log_id", "log"])
                if log_ref and "log" in tables:
                    log_cols = table_columns(con, "log")
                    log_key = first_existing(log_cols, ["token", "log_token", "id"])
                    log_row = fetch_one_by_value(con, "log", log_cols, log_key, row_get(scene_row, log_ref)) if log_key else None
                    if log_row is not None:
                        for col in strong_cols:
                            add_candidate(strong, bytes_or_text(row_get(log_row, col) if col in log_cols else None))
                        for col in weak_cols:
                            add_candidate(weak, bytes_or_text(row_get(log_row, col) if col in log_cols else None))
        if "log" in tables:
            log_cols = table_columns(con, "log")
            for col in strong_cols:
                if col in log_cols:
                    vals = [bytes_or_text(r[0]) for r in con.execute(f"SELECT DISTINCT {q(col)} FROM log WHERE {q(col)} IS NOT NULL LIMIT 2").fetchall() if bytes_or_text(r[0])]
                    if len(vals) == 1: add_candidate(strong, vals[0])
            for col in weak_cols:
                if col in log_cols:
                    vals = [bytes_or_text(r[0]) for r in con.execute(f"SELECT DISTINCT {q(col)} FROM log WHERE {q(col)} IS NOT NULL LIMIT 2").fetchall() if bytes_or_text(r[0])]
                    if len(vals) == 1: add_candidate(weak, vals[0])
        con.close()
        if not strong and not weak:
            warnings.append({"type": "map_name_resolution_failed", "reason": "no_sqlite_map_candidates", "db_path": str(db_path), "scene_token": scene_token})
        return strong, [v for v in weak if v not in strong]
    except Exception as exc:
        warnings.append({"type": "map_name_resolution_failed", "reason": "sqlite_error", "db_path": str(db_path), "scene_token": scene_token, "message": str(exc)})
        return strong, weak

def yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def extract_pose_fields_from_row(row: Optional[sqlite3.Row], ego_cols: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    x_col = first_existing(ego_cols, ["x", "translation_x", "ego_x", "center_x"])
    y_col = first_existing(ego_cols, ["y", "translation_y", "ego_y", "center_y"])
    heading_col = first_existing(ego_cols, ["heading", "yaw", "ego_heading"])
    qw_col = first_existing(ego_cols, ["qw", "rotation_w", "orientation_w"])
    qx_col = first_existing(ego_cols, ["qx", "rotation_x", "orientation_x"])
    qy_col = first_existing(ego_cols, ["qy", "rotation_y", "orientation_y"])
    qz_col = first_existing(ego_cols, ["qz", "rotation_z", "orientation_z"])
    x = to_float(row_get(row, x_col)); y = to_float(row_get(row, y_col))
    if x is None or y is None:
        return None
    heading = to_float(row_get(row, heading_col))
    if heading is None:
        qw = to_float(row_get(row, qw_col)); qx = to_float(row_get(row, qx_col)); qy = to_float(row_get(row, qy_col)); qz = to_float(row_get(row, qz_col))
        if None not in (qw, qx, qy, qz):
            heading = yaw_from_quat(float(qw), float(qx), float(qy), float(qz))
    if heading is None:
        heading = 0.0
    return float(x), float(y), float(heading)


def resolve_lidar_rows_for_scene(con: sqlite3.Connection, scene: sqlite3.Row, scene_cols: Sequence[str], lidar_cols: Sequence[str], warnings: List[dict], db_path: Path, scene_token: str) -> List[sqlite3.Row]:
    max_frames = 20000
    scene_token_col = first_existing(scene_cols, ["token", "scene_token", "id"])
    scene_token_value = row_get(scene, scene_token_col)
    lidar_scene_col = first_existing(lidar_cols, ["scene_token", "scene", "scene_id"])
    order_col = first_existing(lidar_cols, ["frame_index", "lidar_pc_index", "timestamp", "time_us", "utime", "id", "token"])
    order_by = q(order_col) if order_col else ""
    if lidar_scene_col and scene_token_value is not None:
        rows = select_rows(con, "lidar_pc", lidar_cols, where=f"{q(lidar_scene_col)}=?", params=(scene_token_value,), order_by=order_by, limit=max_frames)
        if rows:
            return rows
    start_col = first_existing(scene_cols, ["start_lidar_pc_token", "first_lidar_pc_token", "initial_lidar_pc_token", "lidar_pc_token"])
    end_col = first_existing(scene_cols, ["end_lidar_pc_token", "last_lidar_pc_token", "final_lidar_pc_token"])
    lidar_token_col = first_existing(lidar_cols, ["token", "lidar_pc_token", "id"])
    next_col = first_existing(lidar_cols, ["next_token", "next_lidar_pc_token"])
    if start_col and lidar_token_col:
        token_index = {r[lidar_token_col]: r for r in select_rows(con, "lidar_pc", lidar_cols)}
        current = row_get(scene, start_col); end_token = row_get(scene, end_col)
        sequence: List[sqlite3.Row] = []; seen = set()
        while current is not None and current not in seen and len(sequence) < max_frames:
            seen.add(current); lidar = token_index.get(current)
            if lidar is None:
                warnings.append({"type": "lidar_pc_chain_token_missing", "db_path": str(db_path), "scene_token": scene_token})
                break
            sequence.append(lidar)
            if end_token is not None and current == end_token:
                break
            if not next_col:
                warnings.append({"type": "lidar_pc_chain_without_next_token", "db_path": str(db_path), "scene_token": scene_token})
                break
            current = row_get(lidar, next_col)
        if sequence:
            return sequence
    lidar_ts_col = first_existing(lidar_cols, ["timestamp", "time_us", "utime", "time_stamp"])
    start_time_col = first_existing(scene_cols, ["start_time", "start_timestamp", "start_time_us", "first_timestamp"])
    end_time_col = first_existing(scene_cols, ["end_time", "end_timestamp", "end_time_us", "last_timestamp"])
    if lidar_ts_col and start_time_col and end_time_col:
        start_time = row_get(scene, start_time_col); end_time = row_get(scene, end_time_col)
        if start_time is not None and end_time is not None:
            rows = select_rows(con, "lidar_pc", lidar_cols, where=f"{q(lidar_ts_col)} >= ? AND {q(lidar_ts_col)} <= ?", params=(start_time, end_time), order_by=q(lidar_ts_col), limit=max_frames)
            if rows:
                return rows
    warnings.append({"type": "uncertain_scene_lidar_join", "db_path": str(db_path), "scene_token": scene_token, "scene_columns": list(scene_cols)})
    return []


def load_scene_poses(db_path: Path, scene_token: str, warnings: List[dict]) -> Dict[int, Tuple[float, float, float]]:
    try:
        con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        missing = [t for t in ("scene", "lidar_pc", "ego_pose") if t not in tables]
        if missing:
            warnings.append({"type": "ego_pose_sqlite_load_failed", "db_path": str(db_path), "scene_token": scene_token, "message": f"Missing required tables: {missing}"})
            con.close(); return {}
        scene_cols = table_columns(con, "scene"); lidar_cols = table_columns(con, "lidar_pc"); ego_cols = table_columns(con, "ego_pose")
        scene_key = first_existing(scene_cols, ["token", "scene_token", "id"])
        scene = fetch_one_by_value(con, "scene", scene_cols, scene_key, scene_token) if scene_key and scene_token else None
        if scene is None:
            warnings.append({"type": "scene_token_not_found", "db_path": str(db_path), "scene_token": scene_token})
            con.close(); return {}
        lidar_rows = resolve_lidar_rows_for_scene(con, scene, scene_cols, lidar_cols, warnings, db_path, scene_token)
        lidar_ego_col = first_existing(lidar_cols, ["ego_pose_token", "ego_pose_id", "ego_pose"])
        ego_token_col = first_existing(ego_cols, ["token", "ego_pose_token", "id"])
        if not lidar_ego_col or not ego_token_col:
            warnings.append({"type": "ego_pose_sqlite_load_failed", "db_path": str(db_path), "scene_token": scene_token, "message": f"Missing lidar->ego join columns lidar={lidar_ego_col}, ego={ego_token_col}"})
            con.close(); return {}
        poses: Dict[int, Tuple[float, float, float]] = {}
        for frame_index, lidar in enumerate(lidar_rows):
            ego = fetch_one_by_value(con, "ego_pose", ego_cols, ego_token_col, row_get(lidar, lidar_ego_col))
            pose = extract_pose_fields_from_row(ego, ego_cols)
            if pose is not None:
                poses[frame_index] = pose
        if not poses:
            warnings.append({"type": "scene_has_no_ego_poses", "db_path": str(db_path), "scene_token": scene_token})
        con.close(); return poses
    except Exception as exc:
        warnings.append({"type": "ego_pose_sqlite_load_failed", "db_path": str(db_path), "scene_token": scene_token, "message": str(exc)})
        return {}

def ego_curvature(poses: Sequence[Tuple[float, float, float]]) -> Tuple[float, float]:
    if len(poses) < 3:
        return SENTINEL, SENTINEL
    xy = np.asarray([(p[0], p[1]) for p in poses], dtype=np.float64)
    d = np.diff(xy, axis=0); seg = np.linalg.norm(d, axis=1)
    valid = seg > 1e-3
    if valid.sum() < 2:
        return SENTINEL, SENTINEL
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); dh = np.abs(np.diff(h)); ds = np.maximum((seg[1:] + seg[:-1]) * 0.5, 1e-3)
    curv = dh / ds
    curv = curv[np.isfinite(curv)]
    if curv.size == 0:
        return SENTINEL, SENTINEL
    return float(np.mean(curv)), float(np.percentile(curv, 95))


def load_map_api(map_root: Path, map_name: str, warnings: List[dict]) -> Any:
    if importlib.util.find_spec("nuplan.common.maps.nuplan_map.map_factory") is None:
        warnings.append({"type": "nuplan_map_api_unavailable", "map_name": map_name, "nuplan_map_root": str(map_root), "message": "nuplan map_factory module is not installed."})
        return None
    map_factory = importlib.import_module("nuplan.common.maps.nuplan_map.map_factory")
    try:
        return map_factory.get_maps_api(str(map_root), "nuplan-maps-v1.0", map_name)
    except Exception as exc:
        warnings.append({"type": "nuplan_map_api_unavailable", "map_name": map_name, "nuplan_map_root": str(map_root), "message": str(exc)})
        return None


def resolve_loadable_map_api(map_root: Path, strong_candidates: Sequence[str], weak_candidates: Sequence[str], api_cache: Dict[str, Any], warnings: List[dict], context: Dict[str, str]) -> Tuple[str, Any]:
    """Try candidates with get_maps_api; return only a map_name that actually loads."""
    tried: List[Dict[str, str]] = []
    for strength, candidates in (("strong", strong_candidates), ("weak", weak_candidates)):
        for candidate in candidates:
            if not candidate:
                continue
            if candidate not in api_cache:
                api_cache[candidate] = load_map_api(map_root, candidate, warnings) if map_root.exists() else None
            if api_cache.get(candidate) is not None:
                if strength == "weak":
                    warnings.append({"type": "weak_map_name_candidate_accepted_after_api_load", "map_name": candidate, **context})
                return candidate, api_cache[candidate]
            tried.append({"map_name": candidate, "strength": strength})
    if tried:
        warnings.append({"type": "map_name_candidates_not_loadable", "candidates": tried, **context})
    else:
        warnings.append({"type": "map_name_resolution_failed", "reason": "no_map_name_candidates", **context})
    return "", None


def enum_layers() -> Dict[str, Any]:
    if importlib.util.find_spec("nuplan.common.maps.maps_datatypes") is None:
        return {}
    maps_datatypes = importlib.import_module("nuplan.common.maps.maps_datatypes")
    semantic_map_layer = getattr(maps_datatypes, "SemanticMapLayer", None)
    if semantic_map_layer is None:
        return {}
    return {name: getattr(semantic_map_layer, name) for aliases in LAYER_ALIASES.values() for name in aliases if hasattr(semantic_map_layer, name)}


def point_obj(x: float, y: float) -> Any:
    if importlib.util.find_spec("nuplan.common.actor_state.state_representation") is None:
        return (x, y)
    state_representation = importlib.import_module("nuplan.common.actor_state.state_representation")
    point2d = getattr(state_representation, "Point2D", None)
    return point2d(x, y) if point2d is not None else (x, y)


def layer_value(layer_map: Dict[str, Any], key: str) -> Optional[Any]:
    for alias in LAYER_ALIASES[key]:
        if alias in layer_map:
            return layer_map[alias]
    return None


def safe_count_near(api: Any, layer: Any, x: float, y: float, radius: float) -> Tuple[int, List[Any], bool, str]:
    if api is None:
        return 0, [], False, "api_missing"
    if layer is None:
        return 0, [], False, "layer_missing"
    try:
        res = api.get_proximal_map_objects(point_obj(x, y), radius, [layer])
        objs = res.get(layer, []) if isinstance(res, dict) else []
        return len(objs), list(objs), True, ""
    except Exception as exc:
        return 0, [], False, str(exc)


def shapely_point(x: float, y: float) -> Any:
    if importlib.util.find_spec("shapely.geometry") is None:
        return point_obj(x, y)
    geometry = importlib.import_module("shapely.geometry")
    return geometry.Point(float(x), float(y))


def extract_xy(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try: return float(value[0]), float(value[1])
        except Exception: return None
    for a, b in (("x", "y"), ("x_", "y_")):
        if hasattr(value, a) and hasattr(value, b):
            try: return float(getattr(value, a)), float(getattr(value, b))
            except Exception: return None
    if hasattr(value, "array"):
        try:
            arr = value.array
            return float(arr[0]), float(arr[1])
        except Exception:
            return None
    return None


def path_points(obj: Any) -> List[Tuple[float, float]]:
    candidates = [obj]
    for attr in ("baseline_path", "discrete_path", "interior_edges", "exterior_edges"):
        v = getattr(obj, attr, None)
        if v is not None:
            candidates.append(v)
    pts: List[Tuple[float, float]] = []
    for cand in candidates:
        seq = getattr(cand, "discrete_path", cand)
        if isinstance(seq, (list, tuple)):
            for item in seq:
                if isinstance(item, (list, tuple)) and item and not isinstance(item[0], (int, float)):
                    for sub in item:
                        xy = extract_xy(sub)
                        if xy: pts.append(xy)
                else:
                    xy = extract_xy(item)
                    if xy: pts.append(xy)
        elif hasattr(seq, "coords"):
            try: pts.extend([(float(a), float(b)) for a, b in seq.coords])
            except Exception: pass
    return pts


def geom_distance(x: float, y: float, geom: Any) -> Optional[float]:
    if geom is None:
        return None
    for pt in (shapely_point(x, y), point_obj(x, y), (x, y)):
        try:
            d = float(geom.distance(pt))
            if math.isfinite(d): return d
        except Exception:
            continue
    pts = path_points(geom)
    if pts:
        arr = np.asarray(pts, dtype=np.float64)
        d = np.linalg.norm(arr - np.asarray([x, y], dtype=np.float64), axis=1)
        return float(np.min(d)) if d.size else None
    return None


def dist_to_objs(x: float, y: float, objs: Sequence[Any], warn_counts: Counter) -> float:
    best = math.inf
    for obj in objs:
        tried = False
        for attr in ("polygon", "linestring", "baseline_path", "interior_edges", "exterior_edges"):
            geom = getattr(obj, attr, None)
            if geom is None: continue
            tried = True
            d = geom_distance(x, y, geom)
            if d is not None and math.isfinite(d): best = min(best, d)
        if not tried:
            d = geom_distance(x, y, obj)
            if d is not None and math.isfinite(d): best = min(best, d)
    if not math.isfinite(best) and objs:
        warn_counts["geometry_distance_unavailable"] += 1
    return best if math.isfinite(best) else SENTINEL


def lane_curvature_stats(lane_objs: Sequence[Any], warn_counts: Counter) -> Dict[str, float]:
    heading_changes: List[float] = []; curvs: List[float] = []
    for lane in lane_objs:
        pts = path_points(lane)
        if len(pts) < 3:
            continue
        xy = np.asarray(pts, dtype=np.float64)
        segvec = np.diff(xy, axis=0); seg = np.linalg.norm(segvec, axis=1)
        valid = seg > 1e-3
        if valid.sum() < 2:
            continue
        h = np.unwrap(np.arctan2(segvec[:, 1], segvec[:, 0])); dh = np.abs(np.diff(h)); ds = np.maximum((seg[1:] + seg[:-1]) * 0.5, 1e-3)
        c = dh / ds
        heading_changes.extend([float(v) for v in dh[np.isfinite(dh)]])
        curvs.extend([float(v) for v in c[np.isfinite(c)]])
    if not curvs:
        if lane_objs: warn_counts["lane_curvature_geometry_unavailable"] += 1
        return {}
    return {
        "lane_heading_change_mean": float(np.mean(heading_changes)),
        "lane_heading_change_std": float(np.std(heading_changes)),
        "lane_curvature_proxy_mean": float(np.mean(curvs)),
        "lane_curvature_proxy_p95": float(np.percentile(curvs, 95)),
        "lane_curvature_proxy_max": float(np.max(curvs)),
    }


def feature_for_row(row: Dict[str, str], poses: List[Tuple[float, float, float]], api: Any, layer_map: Dict[str, Any], radius: float, sample_stride: int, warnings: List[dict], query_counts: Counter, resolved_map_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    vals = {name: SENTINEL for name in FEATURE_NAMES}
    vals["map_available"] = 1.0 if api is not None else 0.0
    sampled = poses[::max(sample_stride, 1)] if poses else []
    if not sampled:
        warnings.append({"type": "window_has_no_ego_poses", "sample_id": row.get("sample_id", ""), "scenario_id": row.get("scenario_id", "")})
    q_ok = 0; lane_counts=[]; conn_counts=[]; xwalk_counts=[]; stop_counts=[]; edge_counts=[]; inter_counts=[]; obj_counts=[]; d_inter=[]; d_xwalk=[]; d_stop=[]; d_edge=[]
    lane_geom_objs: List[Any] = []
    layers = {
        "lane": layer_value(layer_map, "lane"), "lane_connector": layer_value(layer_map, "lane_connector"),
        "crosswalk": layer_value(layer_map, "crosswalk"), "stop_line": layer_value(layer_map, "stop_line"),
        "road_edge": layer_value(layer_map, "road_edge"), "roadblock": layer_value(layer_map, "roadblock"),
    }
    for x, y, _ in sampled:
        if api is None:
            continue
        per_pose_success = False
        results: Dict[str, Tuple[int, List[Any]]] = {}
        for lname, layer in layers.items():
            n, objs, ok, msg = safe_count_near(api, layer, x, y, radius)
            query_counts[f"layer:{lname}:success" if ok else f"layer:{lname}:failure"] += 1
            if ok and n == 0: query_counts[f"layer:{lname}:empty"] += 1
            if not ok: query_counts[warn_counter_key("map_query_failed", lname, msg)] += 1
            per_pose_success = per_pose_success or ok
            results[lname] = (n, objs)
        if per_pose_success:
            q_ok += 1
        lane_n, lane_objs = results["lane"]; conn_n, conn_objs = results["lane_connector"]
        xw_n, xw_objs = results["crosswalk"]; st_n, st_objs = results["stop_line"]
        edge_n, edge_objs = results["road_edge"]; rb_n, rb_objs = results["roadblock"]
        lane_geom_objs.extend(lane_objs[:5])
        inter_n = conn_n + rb_n
        lane_counts.append(lane_n); conn_counts.append(conn_n); xwalk_counts.append(xw_n); stop_counts.append(st_n); edge_counts.append(edge_n); inter_counts.append(inter_n)
        obj_counts.append(lane_n + conn_n + xw_n + stop_n + edge_n + rb_n)
        d_inter.append(min([d for d in [dist_to_objs(x, y, conn_objs, query_counts), dist_to_objs(x, y, rb_objs, query_counts)] if d >= 0], default=SENTINEL))
        d_xwalk.append(dist_to_objs(x, y, xw_objs, query_counts)); d_stop.append(dist_to_objs(x, y, st_objs, query_counts)); d_edge.append(dist_to_objs(x, y, edge_objs, query_counts))
    def fill(prefix: str, arr: List[float]) -> None:
        clean = np.asarray([v for v in arr if math.isfinite(float(v)) and float(v) >= 0], dtype=np.float64)
        if clean.size:
            vals[f"{prefix}_mean"] = float(clean.mean()); vals[f"{prefix}_max"] = float(clean.max())
    vals["num_valid_ego_map_queries"] = float(q_ok); vals["valid_ego_map_query_ratio"] = float(q_ok / max(len(sampled), 1))
    fill("nearby_lane_count", lane_counts); fill("nearby_lane_connector_count", conn_counts); fill("crosswalk_count", xwalk_counts); fill("stop_line_count", stop_counts); fill("road_edge_count", edge_counts); fill("intersection_proxy_count", inter_counts); fill("map_object_count", obj_counts)
    vals["ego_on_lane_ratio"] = float(np.mean(np.asarray(lane_counts) > 0)) if lane_counts else SENTINEL
    vals["ego_on_lane_connector_ratio"] = float(np.mean(np.asarray(conn_counts) > 0)) if conn_counts else SENTINEL
    vals["intersection_proxy_ratio"] = float(np.mean(np.asarray(inter_counts) > 0)) if inter_counts else SENTINEL
    for name, arr in [("distance_to_intersection_proxy", d_inter), ("distance_to_crosswalk", d_xwalk), ("distance_to_stop_line", d_stop), ("distance_to_drivable_area_edge", d_edge)]:
        clean = np.asarray([v for v in arr if math.isfinite(float(v)) and float(v) >= 0], dtype=np.float64)
        if clean.size:
            vals[f"{name}_min"] = float(clean.min())
            if f"{name}_mean" in vals: vals[f"{name}_mean"] = float(clean.mean())
    vals["traffic_light_related_count_mean"] = vals["stop_line_count_mean"]
    vals["traffic_light_related_count_max"] = vals["stop_line_count_max"]
    vals.update(lane_curvature_stats(lane_geom_objs, query_counts))
    egomean, egop95 = ego_curvature(poses); vals["ego_path_curvature_proxy_mean"] = egomean; vals["ego_path_curvature_proxy_p95"] = egop95
    if lane_counts:
        comp = (np.mean(lane_counts) + 1.5*np.mean(conn_counts) + 2*np.mean(xwalk_counts) + 2*np.mean(stop_counts)) / 20.0
        vals["map_complexity_score"] = float(comp); vals["odd_complexity_score"] = float(comp + max(vals["intersection_proxy_ratio"], 0.0))
    arr = np.asarray([vals[n] for n in FEATURE_NAMES], dtype=np.float32)
    arr[~np.isfinite(arr)] = SENTINEL
    meta = {k: row.get(k, "") for k in KEY_COLUMNS if k in row}
    meta.update({"map_name": resolved_map_name, "num_ego_poses_used": len(sampled), "map_available": int(vals["map_available"]), "warning_count": 0})
    return arr, meta

def write_meta_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    base = KEY_COLUMNS + ["map_name", "num_ego_poses_used", "map_available", "warning_count", "shard_id", "local_row"]
    cols = [c for c in base if any(c in r for r in rows)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def summary_stats(feat: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = feat[:, i] if feat.size else np.asarray([], dtype=np.float32)
        valid = col[np.isfinite(col) & (col != SENTINEL)]
        out[name] = {"mean": SENTINEL, "std": SENTINEL, "min": SENTINEL, "p50": SENTINEL, "p95": SENTINEL, "max": SENTINEL} if valid.size == 0 else {"mean": float(valid.mean()), "std": float(valid.std()), "min": float(valid.min()), "p50": float(np.percentile(valid, 50)), "p95": float(np.percentile(valid, 95)), "max": float(valid.max())}
    return out



def feature_col(feat: np.ndarray, name: str) -> np.ndarray:
    return feat[:, FEATURE_NAMES.index(name)] if feat.size else np.asarray([], dtype=np.float32)


def add_semantic_warnings(feat: np.ndarray, warnings: List[dict]) -> None:
    if feat.size == 0:
        warnings.append({"type": "semantic_warning", "reason": "empty_feature_matrix"})
        return
    avail = feature_col(feat, "map_available")
    if float(np.mean(avail)) == 0.0:
        warnings.append({"type": "semantic_warning", "reason": "map_available_mean_is_zero"})
    lane_cols = [feature_col(feat, n) for n in ("nearby_lane_count_mean", "nearby_lane_count_max", "nearby_lane_connector_count_mean", "nearby_lane_connector_count_max")]
    if all(np.all((c == SENTINEL) | (c == 0)) for c in lane_cols):
        warnings.append({"type": "semantic_warning", "reason": "all_lane_count_features_sentinel_or_zero"})
    obj_cols = [feature_col(feat, n) for n in ("crosswalk_count_mean", "stop_line_count_mean", "intersection_proxy_count_mean")]
    if all(np.all((c == SENTINEL) | (c == 0)) for c in obj_cols):
        warnings.append({"type": "semantic_warning", "reason": "all_crosswalk_stop_line_intersection_features_sentinel_or_zero"})
    curv_cols = [feature_col(feat, n) for n in ("lane_heading_change_mean", "lane_curvature_proxy_mean", "lane_curvature_proxy_p95", "lane_curvature_proxy_max")]
    if all(np.all(c == SENTINEL) for c in curv_cols):
        warnings.append({"type": "semantic_warning", "reason": "lane_curvature_fields_all_sentinel"})
    ratio = feature_col(feat, "valid_ego_map_query_ratio")
    obj_count = feature_col(feat, "map_object_count_mean")
    valid_ratio = ratio[ratio >= 0]
    if valid_ratio.size and float(np.mean(valid_ratio)) > 0.8 and np.all((obj_count == SENTINEL) | (obj_count == 0)):
        warnings.append({"type": "semantic_warning", "reason": "high_valid_query_ratio_but_all_object_counts_zero"})

def write_report(path: Path, args: argparse.Namespace, rows: List[dict], meta_rows: List[dict], feat: np.ndarray, schema: dict, warnings: List[dict], alignment_ok: bool, query_counts: Counter) -> None:
    stats = summary_stats(feat)
    warn_counts = Counter(w.get("type", w.get("code", "unknown")) for w in warnings)
    resolved = [r.get("map_name", "") for r in meta_rows]
    resolved_nonempty = [m for m in resolved if m]
    unique_maps = sorted(set(resolved_nonempty))
    query_success = {k: v for k, v in query_counts.items() if k.startswith("layer:") and k.endswith(":success")}
    query_failure = {k: v for k, v in query_counts.items() if k.startswith("layer:") and k.endswith(":failure")}
    query_empty = {k: v for k, v in query_counts.items() if k.startswith("layer:") and k.endswith(":empty")}
    total_success = sum(query_success.values()); total_failure = sum(query_failure.values())
    query_success_ratio = float(total_success / max(total_success + total_failure, 1))
    resolution_failures = Counter(w.get("reason", "unknown") for w in warnings if w.get("type") == "map_name_resolution_failed")
    table = "| feature | mean | std | min | p50 | p95 | max |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name in FEATURE_NAMES:
        s = stats[name]; table += f"| {name} | {s['mean']:.4g} | {s['std']:.4g} | {s['min']:.4g} | {s['p50']:.4g} | {s['p95']:.4g} | {s['max']:.4g} |\n"
    avail = feat[:, FEATURE_NAMES.index("map_available")] if feat.size else np.asarray([])
    ratio = feat[:, FEATURE_NAMES.index("valid_ego_map_query_ratio")] if feat.size else np.asarray([])
    ego_pose_counts = np.asarray([int(r.get("num_ego_poses_used", 0) or 0) for r in meta_rows], dtype=np.int64) if meta_rows else np.asarray([], dtype=np.int64)
    windows_with_ego_poses_ratio = float(np.mean(ego_pose_counts > 0)) if ego_pose_counts.size else 0.0
    map_name_loaded_success_ratio = float(len(resolved_nonempty) / max(len(meta_rows), 1))
    lane_feature_names = ["nearby_lane_count_mean", "nearby_lane_count_max", "nearby_lane_connector_count_mean", "nearby_lane_connector_count_max", "ego_on_lane_ratio", "ego_on_lane_connector_ratio"]
    object_feature_names = ["crosswalk_count_mean", "stop_line_count_mean", "intersection_proxy_count_mean", "map_object_count_mean", "road_edge_count_mean"]
    def non_sentinel_nonzero_ratio(names: Sequence[str]) -> float:
        if not feat.size:
            return 0.0
        cols = np.vstack([feature_col(feat, n) for n in names if n in FEATURE_NAMES]).T
        if cols.size == 0:
            return 0.0
        valid = np.isfinite(cols) & (cols != SENTINEL) & (cols != 0)
        return float(np.mean(np.any(valid, axis=1)))
    lane_feature_non_sentinel_ratio = non_sentinel_nonzero_ratio(lane_feature_names)
    object_feature_non_sentinel_ratio = non_sentinel_nonzero_ratio(object_feature_names)
    text = f"""# Stage 7B.3 nuPlan Map/ODD-lite Feature Report

## Purpose
Build lightweight, row-aligned map/ODD proxy features for Stage 7B.2 dynamic context windows. This stage does not serialize full vector maps, render maps, train models, or merge dynamic + map features.

## Inputs / outputs
- input_dynamic_dir: `{args.input_dynamic_dir}`
- output_dir: `{args.output_dir}`
- nuplan_db_root: `{args.nuplan_db_root}`
- nuplan_map_root: `{args.nuplan_map_root}`
- split: `{args.split}`
- radius_m: {args.radius_m}
- sample_stride: {args.sample_stride}

## Counts
- processed rows/windows: {len(rows)}
- scenarios: {len(set(r.get('scenario_id','') for r in rows))}
- feature dimension: {feat.shape[1] if feat.ndim == 2 else 0}

## Feature groups
```json
{json.dumps(schema['feature_groups'], indent=2, ensure_ascii=False)}
```

## Alignment check
- number of dynamic rows == number of map/ODD rows: `{alignment_ok}`
- identifier order preserved: `{alignment_ok}`

## Map name resolution statistics
- unique resolved map names: {len(unique_maps)}
- resolved map names: `{unique_maps[:20]}`
- map_name_loaded_success_ratio: {map_name_loaded_success_ratio:.4f}
- map_name resolution success ratio: {map_name_loaded_success_ratio:.4f}
- resolution failure reasons: `{dict(resolution_failures)}`

## Map availability statistics
- map_available_mean: {float(np.mean(avail)) if avail.size else 0.0:.4f}
- map_available_rows: {int(np.sum(avail > 0.5)) if avail.size else 0} / {len(rows)}

## Ego pose coverage statistics
- windows_with_ego_poses_ratio: {windows_with_ego_poses_ratio:.4f}
- windows_with_ego_poses: {int(np.sum(ego_pose_counts > 0)) if ego_pose_counts.size else 0} / {len(rows)}

## Lane/object non-sentinel statistics
- lane_feature_non_sentinel_ratio: {lane_feature_non_sentinel_ratio:.4f}
- object_feature_non_sentinel_ratio: {object_feature_non_sentinel_ratio:.4f}
- lane/object feature non-sentinel ratio: lane={lane_feature_non_sentinel_ratio:.4f}, object={object_feature_non_sentinel_ratio:.4f}

## Map API query statistics
- query_success_ratio: {query_success_ratio:.4f}
- per-layer success counts: `{dict(query_success)}`
- per-layer failure counts: `{dict(query_failure)}`
- per-layer empty result counts: `{dict(query_empty)}`

## Valid query ratio statistics
- mean: {float(np.mean(ratio[ratio >= 0])) if np.any(ratio >= 0) else SENTINEL:.4f}
- min: {float(np.min(ratio[ratio >= 0])) if np.any(ratio >= 0) else SENTINEL:.4f}
- max: {float(np.max(ratio[ratio >= 0])) if np.any(ratio >= 0) else SENTINEL:.4f}

## Key feature summary
{table}

## Warning summary
- warning count: {len(warnings)}
- warning types: `{dict(warn_counts)}`

## Limitations
- map-lite only; no full vector map serialization.
- Static map proxies only; traffic light state is not included and stop-line counts are used as a static traffic-control proxy.
- Intersection is proxy-based using lane connectors / roadblock-like objects when map API supports them.
- Lane curvature is attempted from lane baseline/discrete geometry; fields remain sentinel only when geometry is unavailable through the installed API.
- Stage 7B.3 is separate from Stage 7B.4 and does not merge with dynamic context features.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.radius_m <= 0 or args.sample_stride <= 0:
        raise ValueError("--radius_m and --sample_stride must be positive.")
    out = Path(args.output_dir); prepare_output(out, args.overwrite)
    warnings: List[dict] = []
    dynamic_rows_all, _, manifest = load_dynamic_rows(Path(args.input_dynamic_dir))
    dynamic_rows = scenario_limited(dynamic_rows_all, args.max_scenarios)
    if len(dynamic_rows) != len(dynamic_rows_all):
        warnings.append({"type": "max_scenarios_subset", "processed_rows": len(dynamic_rows), "dynamic_rows_total": len(dynamic_rows_all), "max_scenarios": args.max_scenarios})
    db_root = Path(args.nuplan_db_root).expanduser(); map_root = Path(args.nuplan_map_root).expanduser()
    if not db_root.exists(): warnings.append({"type": "nuplan_db_root_missing", "path": str(db_root)})
    if not map_root.exists(): warnings.append({"type": "nuplan_map_root_missing", "path": str(map_root)})
    layer_map = enum_layers(); api_cache: Dict[str, Any] = {}; pose_cache: Dict[Tuple[str, str], Dict[int, Tuple[float, float, float]]] = {}; map_candidate_cache: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}; query_counts: Counter = Counter()
    feat_rows: List[np.ndarray] = []; meta_rows: List[Dict[str, Any]] = []
    for row in dynamic_rows:
        before = len(warnings)
        db_name = row.get("db_name", ""); scene_token = row.get("scene_token", "")
        key = (db_name, scene_token)
        db_path = find_db(db_root, db_name) if db_name else None
        if key not in pose_cache:
            if db_path is None:
                warnings.append({"type": "nuplan_db_file_missing", "db_name": db_name, "scenario_id": row.get("scenario_id", "")})
                pose_cache[key] = {}
            else:
                pose_cache[key] = load_scene_poses(db_path, scene_token, warnings)
        if key not in map_candidate_cache:
            map_candidate_cache[key] = resolve_map_name_candidates_from_db(db_path, scene_token, warnings, row)
        start = int(float(row.get("start_frame_index", row.get("start", 0)) or 0)); end = int(float(row.get("end_frame_index", start) or start))
        scene_poses = pose_cache[key]
        poses = [scene_poses[i] for i in range(start, end + 1) if i in scene_poses]
        strong_candidates, weak_candidates = map_candidate_cache[key]
        map_name, api = resolve_loadable_map_api(
            map_root,
            strong_candidates,
            weak_candidates,
            api_cache,
            warnings,
            {"db_name": db_name, "scene_token": scene_token, "scenario_id": row.get("scenario_id", "")},
        )
        feat, meta = feature_for_row(row, poses, api, layer_map, args.radius_m, args.sample_stride, warnings, query_counts, map_name)
        meta.update({"shard_id": row.get("shard_id", ""), "local_row": row.get("local_row", ""), "warning_count": len(warnings) - before})
        feat_rows.append(feat); meta_rows.append(meta)
    feat_arr = np.vstack(feat_rows).astype(np.float32) if feat_rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    nonfinite = ~np.isfinite(feat_arr)
    if nonfinite.any():
        warnings.append({"type": "nonfinite_features_replaced", "count": int(nonfinite.sum())})
        feat_arr[nonfinite] = SENTINEL
    alignment_ok = feat_arr.shape[0] == len(dynamic_rows) == len(meta_rows)
    for key_name, count in sorted(query_counts.items()):
        if key_name.startswith("map_query_failed|"):
            _, layer, message = key_name.split("|", 2)
            warnings.append({"type": "map_query_failed", "layer": layer, "message": message, "count": int(count)})
        elif key_name in {"geometry_distance_unavailable", "lane_curvature_geometry_unavailable"}:
            warnings.append({"type": key_name, "count": int(count)})
    assert feat_arr.ndim == 2, f"map_odd_feat must be 2D, got shape={feat_arr.shape}"
    assert len(meta_rows) == feat_arr.shape[0], f"map_odd_meta rows ({len(meta_rows)}) must match feature rows ({feat_arr.shape[0]})"
    assert np.isfinite(feat_arr).all(), "map_odd_feat contains NaN or inf after sentinel replacement"
    add_semantic_warnings(feat_arr, warnings)
    np.save(out / "map_odd_feat.npy", feat_arr)
    write_meta_csv(out / "map_odd_meta.csv", meta_rows)
    schema = {"stage": "7B.3", "feature_type": "nuplan_map_odd_lite", "num_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES, "feature_groups": FEATURE_GROUPS, "sentinel_value": SENTINEL, "notes": ["Feature order exactly matches map_odd_feat.npy columns.", "Traffic-light-related counts are static stop-line proxies when live traffic-light state is unavailable.", "Intersection proxy uses lane connectors and roadblock-like objects when supported by the installed nuPlan map API.", "Lane geometry curvature is attempted from baseline/discrete paths; ego trajectory curvature fallback is computed from SQLite ego poses."]}
    write_json(out / "map_odd_feature_schema.json", schema)
    write_json(out / "warnings.json", {"warnings": warnings})
    write_report(out / "map_odd_report.md", args, dynamic_rows, meta_rows, feat_arr, schema, warnings, alignment_ok, query_counts)
    print(f"Wrote Stage 7B.3 map/ODD features to {out}: shape={list(feat_arr.shape)}, warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
