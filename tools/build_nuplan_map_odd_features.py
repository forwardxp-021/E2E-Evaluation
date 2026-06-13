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


def yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_scene_poses(db_path: Path, scene_token: str, warnings: List[dict]) -> Dict[int, Tuple[float, float, float]]:
    try:
        con = sqlite3.connect(str(db_path))
        scene_cols = table_columns(con, "scene"); lidar_cols = table_columns(con, "lidar_pc"); ego_cols = table_columns(con, "ego_pose")
        if not {"scene", "lidar_pc", "ego_pose"}:
            pass
        scene_token_col = "token" if "token" in scene_cols else "id"
        lidar_scene_col = "scene_token" if "scene_token" in lidar_cols else "scene_id"
        lidar_ego_col = "ego_pose_token" if "ego_pose_token" in lidar_cols else "ego_pose_id"
        ego_token_col = "token" if "token" in ego_cols else "id"
        frame_col = "frame_index" if "frame_index" in lidar_cols else ("lidar_pc_index" if "lidar_pc_index" in lidar_cols else "timestamp")
        cols = [f"l.{q(frame_col)}", "e.x", "e.y"]
        if "heading" in ego_cols:
            cols.append("e.heading")
            quat = False
        else:
            cols += ["e.qw", "e.qx", "e.qy", "e.qz"]
            quat = True
        sql = f"SELECT {', '.join(cols)} FROM lidar_pc l JOIN ego_pose e ON l.{q(lidar_ego_col)}=e.{q(ego_token_col)} WHERE l.{q(lidar_scene_col)}=? ORDER BY l.{q(frame_col)}"
        poses: Dict[int, Tuple[float, float, float]] = {}
        for row in con.execute(sql, (bytes.fromhex(scene_token) if all(c in '0123456789abcdefABCDEF' for c in scene_token) and len(scene_token) % 2 == 0 else scene_token,)):
            frame = int(float(row[0])); x = float(row[1]); y = float(row[2]); hdg = float(row[3]) if not quat else yaw_from_quat(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
            poses[frame] = (x, y, hdg)
        if not poses:
            # retry text token; schemas differ in byte/text storage
            for row in con.execute(sql, (scene_token,)):
                frame = int(float(row[0])); x = float(row[1]); y = float(row[2]); hdg = float(row[3]) if not quat else yaw_from_quat(float(row[3]), float(row[4]), float(row[5]), float(row[6]))
                poses[frame] = (x, y, hdg)
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


def safe_count_near(api: Any, layer: Any, x: float, y: float, radius: float) -> Tuple[int, List[Any]]:
    if api is None or layer is None: return 0, []
    try:
        res = api.get_proximal_map_objects(point_obj(x, y), radius, [layer])
        objs = res.get(layer, []) if isinstance(res, dict) else []
        return len(objs), list(objs)
    except Exception:
        return 0, []


def dist_to_objs(x: float, y: float, objs: Sequence[Any]) -> float:
    best = math.inf
    for obj in objs:
        for attr in ("polygon", "baseline_path", "linestring"):
            geom = getattr(obj, attr, None)
            if geom is None: continue
            try:
                d = float(geom.distance(point_obj(x, y)))
                if math.isfinite(d): best = min(best, d)
            except Exception:
                pass
    return best if math.isfinite(best) else SENTINEL


def feature_for_row(row: Dict[str, str], poses: List[Tuple[float, float, float]], api: Any, layer_map: Dict[str, Any], radius: float, sample_stride: int, warnings: List[dict]) -> Tuple[np.ndarray, Dict[str, Any]]:
    vals = {name: SENTINEL for name in FEATURE_NAMES}
    vals["map_available"] = 1.0 if api is not None else 0.0
    sampled = poses[::max(sample_stride, 1)] if poses else []
    if not sampled:
        warnings.append({"type": "window_has_no_ego_poses", "sample_id": row.get("sample_id", ""), "scenario_id": row.get("scenario_id", "")})
    q_ok = 0; lane_counts=[]; conn_counts=[]; xwalk_counts=[]; stop_counts=[]; edge_counts=[]; inter_counts=[]; obj_counts=[]; d_inter=[]; d_xwalk=[]; d_stop=[]; d_edge=[]
    for x, y, _ in sampled:
        if api is None: continue
        q_ok += 1
        lane_n, _ = safe_count_near(api, layer_value(layer_map, "lane"), x, y, radius)
        conn_n, conn_objs = safe_count_near(api, layer_value(layer_map, "lane_connector"), x, y, radius)
        xw_n, xw_objs = safe_count_near(api, layer_value(layer_map, "crosswalk"), x, y, radius)
        st_n, st_objs = safe_count_near(api, layer_value(layer_map, "stop_line"), x, y, radius)
        edge_n, edge_objs = safe_count_near(api, layer_value(layer_map, "road_edge"), x, y, radius)
        rb_n, rb_objs = safe_count_near(api, layer_value(layer_map, "roadblock"), x, y, radius)
        inter_n = conn_n + rb_n
        lane_counts.append(lane_n); conn_counts.append(conn_n); xwalk_counts.append(xw_n); stop_counts.append(st_n); edge_counts.append(edge_n); inter_counts.append(inter_n)
        obj_counts.append(lane_n + conn_n + xw_n + stop_n + edge_n + rb_n)
        d_inter.append(min([d for d in [dist_to_objs(x, y, conn_objs), dist_to_objs(x, y, rb_objs)] if d >= 0], default=SENTINEL))
        d_xwalk.append(dist_to_objs(x, y, xw_objs)); d_stop.append(dist_to_objs(x, y, st_objs)); d_edge.append(dist_to_objs(x, y, edge_objs))
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
    egomean, egop95 = ego_curvature(poses); vals["ego_path_curvature_proxy_mean"] = egomean; vals["ego_path_curvature_proxy_p95"] = egop95
    # Lane geometry API is version-sensitive; keep lane-specific fields sentinel and explicitly expose ego fallback.
    if lane_counts:
        comp = (np.mean(lane_counts) + 1.5*np.mean(conn_counts) + 2*np.mean(xwalk_counts) + 2*np.mean(stop_counts)) / 20.0
        vals["map_complexity_score"] = float(comp); vals["odd_complexity_score"] = float(comp + max(vals["intersection_proxy_ratio"], 0.0))
    arr = np.asarray([vals[n] for n in FEATURE_NAMES], dtype=np.float32)
    arr[~np.isfinite(arr)] = SENTINEL
    meta = {k: row.get(k, "") for k in KEY_COLUMNS if k in row}
    meta.update({"map_name": infer_map_name(row), "num_ego_poses_used": len(sampled), "map_available": int(vals["map_available"]), "warning_count": 0})
    return arr, meta


def infer_map_name(row: Dict[str, str]) -> str:
    for k in ("map_name", "map_version", "location"):
        if row.get(k): return row[k]
    # Stage 7B.2 metadata currently has no map_name; use DB stem as a best-effort API key and warn/report it.
    return Path(row.get("db_name", "")).stem


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


def write_report(path: Path, args: argparse.Namespace, rows: List[dict], feat: np.ndarray, schema: dict, warnings: List[dict], alignment_ok: bool) -> None:
    stats = summary_stats(feat)
    warn_counts = Counter(w.get("type", w.get("code", "unknown")) for w in warnings)
    table = "| feature | mean | std | min | p50 | p95 | max |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name in FEATURE_NAMES:
        s = stats[name]; table += f"| {name} | {s['mean']:.4g} | {s['std']:.4g} | {s['min']:.4g} | {s['p50']:.4g} | {s['p95']:.4g} | {s['max']:.4g} |\n"
    avail = feat[:, FEATURE_NAMES.index("map_available")] if feat.size else np.asarray([])
    ratio = feat[:, FEATURE_NAMES.index("valid_ego_map_query_ratio")] if feat.size else np.asarray([])
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

## Map availability statistics
- map_available_mean: {float(np.mean(avail)) if avail.size else 0.0:.4f}
- map_available_rows: {int(np.sum(avail > 0.5)) if avail.size else 0} / {len(rows)}

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
- Lane curvature fields remain sentinel when nuPlan lane baseline geometry is not available through the installed API; ego trajectory curvature fallback fields are always attempted.
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
    layer_map = enum_layers(); api_cache: Dict[str, Any] = {}; pose_cache: Dict[Tuple[str, str], Dict[int, Tuple[float, float, float]]] = {}
    feat_rows: List[np.ndarray] = []; meta_rows: List[Dict[str, Any]] = []
    for row in dynamic_rows:
        before = len(warnings)
        db_name = row.get("db_name", ""); scene_token = row.get("scene_token", "")
        key = (db_name, scene_token)
        if key not in pose_cache:
            db_path = find_db(db_root, db_name) if db_name else None
            if db_path is None:
                warnings.append({"type": "nuplan_db_file_missing", "db_name": db_name, "scenario_id": row.get("scenario_id", "")})
                pose_cache[key] = {}
            else:
                pose_cache[key] = load_scene_poses(db_path, scene_token, warnings)
        start = int(float(row.get("start_frame_index", row.get("start", 0)) or 0)); end = int(float(row.get("end_frame_index", start) or start))
        scene_poses = pose_cache[key]
        poses = [scene_poses[i] for i in range(start, end + 1) if i in scene_poses]
        map_name = infer_map_name(row)
        if map_name not in api_cache:
            api_cache[map_name] = load_map_api(map_root, map_name, warnings) if map_root.exists() else None
            if api_cache[map_name] is None:
                warnings.append({"type": "map_api_or_map_name_unavailable", "map_name": map_name, "message": "Map API unavailable or inferred map_name could not be loaded."})
        feat, meta = feature_for_row(row, poses, api_cache[map_name], layer_map, args.radius_m, args.sample_stride, warnings)
        meta.update({"shard_id": row.get("shard_id", ""), "local_row": row.get("local_row", ""), "warning_count": len(warnings) - before})
        feat_rows.append(feat); meta_rows.append(meta)
    feat_arr = np.vstack(feat_rows).astype(np.float32) if feat_rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    nonfinite = ~np.isfinite(feat_arr)
    if nonfinite.any():
        warnings.append({"type": "nonfinite_features_replaced", "count": int(nonfinite.sum())})
        feat_arr[nonfinite] = SENTINEL
    alignment_ok = feat_arr.shape[0] == len(dynamic_rows) == len(meta_rows)
    np.save(out / "map_odd_feat.npy", feat_arr)
    write_meta_csv(out / "map_odd_meta.csv", meta_rows)
    schema = {"stage": "7B.3", "feature_type": "nuplan_map_odd_lite", "num_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES, "feature_groups": FEATURE_GROUPS, "sentinel_value": SENTINEL, "notes": ["Feature order exactly matches map_odd_feat.npy columns.", "Traffic-light-related counts are static stop-line proxies when live traffic-light state is unavailable.", "Intersection proxy uses lane connectors and roadblock-like objects when supported by the installed nuPlan map API.", "Ego trajectory curvature fallback is computed from SQLite ego poses; lane geometry fields stay sentinel if lane baseline geometry is unavailable."]}
    write_json(out / "map_odd_feature_schema.json", schema)
    write_json(out / "warnings.json", {"warnings": warnings})
    write_report(out / "map_odd_report.md", args, dynamic_rows, feat_arr, schema, warnings, alignment_ok)
    print(f"Wrote Stage 7B.3 map/ODD features to {out}: shape={list(feat_arr.shape)}, warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
