#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from tools.waymo_lane_utils import LaneInfo, _build_lane_geom, find_best_lane_for_agent, project_point_to_lane, wrap_to_pi


def _obj_id(obj: Any) -> str:
    for attr in ("id", "lane_id", "token", "fid"):
        val = getattr(obj, attr, None)
        if val is not None:
            return str(val)
    return str(id(obj))


def _extract_xy(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None
    for a, b in (("x", "y"), ("x_", "y_")):
        if hasattr(value, a) and hasattr(value, b):
            try:
                return float(getattr(value, a)), float(getattr(value, b))
            except Exception:
                return None
    if hasattr(value, "array"):
        try:
            arr = value.array
            return float(arr[0]), float(arr[1])
        except Exception:
            return None
    return None


def nuplan_path_points(obj: Any) -> List[Tuple[float, float]]:
    """Best-effort extraction of nuPlan lane/lane-connector baseline centerline points."""
    candidates = []
    for attr in ("baseline_path", "discrete_path", "centerline", "linestring"):
        val = getattr(obj, attr, None)
        if val is not None:
            candidates.append(val)
    candidates.append(obj)
    pts: List[Tuple[float, float]] = []
    for cand in candidates:
        seq = getattr(cand, "discrete_path", cand)
        if callable(seq):
            try:
                seq = seq()
            except Exception:
                continue
        if isinstance(seq, (list, tuple)):
            for item in seq:
                xy = _extract_xy(item)
                if xy is not None:
                    pts.append(xy)
        elif hasattr(seq, "coords"):
            try:
                pts.extend([(float(x), float(y)) for x, y in seq.coords])
            except Exception:
                pass
        if len(pts) >= 2:
            break
    return pts


def _neighbor_ids(obj: Any, names: Iterable[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        val = getattr(obj, name, None)
        if val is None:
            continue
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        if not isinstance(val, (list, tuple, set)):
            val = [val]
        for nb in val:
            if nb is not None:
                out.append(_obj_id(nb))
    return list(dict.fromkeys(out))


def lane_info_from_nuplan_object(obj: Any, lane_type: str) -> Optional[LaneInfo]:
    pts = nuplan_path_points(obj)
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    geom = _build_lane_geom(arr)
    if geom is None:
        return None
    # nuPlan map object APIs vary by devkit version.  These names cover direct
    # lane adjacency properties when exposed, plus predecessor/successor edge
    # topology for lane connectors.  Missing fields are reported later instead
    # of silently treated as assignment evidence.
    left_ids = _neighbor_ids(obj, ("left_neighbors", "left_neighbor", "adjacent_edges_left", "left_adjacent_edges", "left_adjacent_edge", "left_neighbor_lane"))
    right_ids = _neighbor_ids(obj, ("right_neighbors", "right_neighbor", "adjacent_edges_right", "right_adjacent_edges", "right_adjacent_edge", "right_neighbor_lane"))
    entry_ids = _neighbor_ids(obj, ("incoming_edges", "predecessors", "entry_lanes", "incoming_lane_edges"))
    exit_ids = _neighbor_ids(obj, ("outgoing_edges", "successors", "exit_lanes", "outgoing_lane_edges"))
    return LaneInfo(
        lane_id=_obj_id(obj), centerline_xy=arr, seg_heading=geom[0], seg_len=geom[1], s_prefix=geom[2],
        seg_start_xy=geom[3], seg_vec_xy=geom[4], seg_den=geom[5], bbox_min_xy=geom[6], bbox_max_xy=geom[7],
        bbox_center_xy=geom[8], left_neighbor_lane_ids=left_ids, right_neighbor_lane_ids=right_ids,
        entry_lane_ids=entry_ids, exit_lane_ids=exit_ids, lane_type=lane_type,
        topology_source="nuplan_topology" if (left_ids or right_ids or entry_ids or exit_ids) else "missing_nuplan_topology",
    )



def _lane_length(info: LaneInfo) -> float:
    return float(info.s_prefix[-1]) if len(info.s_prefix) else 0.0


def _dist(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "max": None}
    vals = sorted(float(v) for v in vals)
    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p / 100.0
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"count": len(vals), "p50": pct(50), "p90": pct(90), "p95": pct(95), "max": max(vals)}


def enrich_geometric_adjacency(lanes: Dict[str, LaneInfo], min_offset: float = 2.0, max_offset: float = 8.0, max_heading_diff_deg: float = 20.0) -> Dict[str, int]:
    """Populate missing left/right LaneInfo adjacency from geometry only.

    This is adapter enrichment: it changes only LaneInfo topology fields consumed by
    the existing Stage5D lane-aware assignment, and does not implement a separate
    assignment algorithm. Lane connectors keep predecessor/successor topology but
    are not assigned synthetic left/right neighbors.
    """
    counts: Counter[str] = Counter()
    max_hd = math.radians(max_heading_diff_deg)
    lane_items = [(lid, ln) for lid, ln in lanes.items() if "connector" not in str(ln.lane_type).lower()]
    for ego_id, ego in lane_items:
        if ego.left_neighbor_lane_ids and ego.right_neighbor_lane_ids:
            continue
        best_left: Optional[Tuple[float, str]] = None
        best_right: Optional[Tuple[float, str]] = None
        mid_idx = max(0, len(ego.centerline_xy) // 2)
        p = ego.centerline_xy[mid_idx]
        h = float(ego.seg_heading[min(mid_idx, len(ego.seg_heading) - 1)])
        left_vec = np.array([-math.sin(h), math.cos(h)], dtype=np.float64)
        fwd_vec = np.array([math.cos(h), math.sin(h)], dtype=np.float64)
        for cand_id, cand in lane_items:
            if cand_id == ego_id:
                continue
            proj = project_point_to_lane(p, cand)
            if not proj.get("projection_success"):
                continue
            hd = abs(wrap_to_pi(h - float(proj["heading"])))
            if hd > max_hd:
                counts["geometric_adjacency_direction_mismatch"] += 1
                continue
            q = np.asarray(cand.centerline_xy[max(0, len(cand.centerline_xy) // 2)], dtype=np.float64)
            delta = q - p
            lateral = float(np.dot(delta, left_vec))
            longitudinal = abs(float(np.dot(delta, fwd_vec)))
            score = abs(abs(lateral) - 3.5) + 0.1 * longitudinal
            if min_offset <= abs(lateral) <= max_offset:
                if lateral > 0 and (best_left is None or score < best_left[0]):
                    best_left = (score, cand_id)
                if lateral < 0 and (best_right is None or score < best_right[0]):
                    best_right = (score, cand_id)
        if not ego.left_neighbor_lane_ids and best_left:
            ego.left_neighbor_lane_ids = [best_left[1]]; ego.topology_source = "geometric_lane_adjacency"; counts["geometric_left_added"] += 1
        if not ego.right_neighbor_lane_ids and best_right:
            ego.right_neighbor_lane_ids = [best_right[1]]; ego.topology_source = "geometric_lane_adjacency"; counts["geometric_right_added"] += 1
    return dict(counts)


def build_lane_topology_debug_summary(lanes: Dict[str, LaneInfo], counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    n = len(lanes)
    types = Counter(str(v.lane_type).lower() for v in lanes.values())
    def c(pred): return sum(1 for v in lanes.values() if pred(v))
    connector_no_topology = c(lambda v: "connector" in str(v.lane_type).lower() and not (v.left_neighbor_lane_ids or v.right_neighbor_lane_ids or v.entry_lane_ids or v.exit_lane_ids))
    return {
        "lane_info_count": n,
        "lane_count": sum(v for k, v in types.items() if "connector" not in k),
        "lane_connector_count": sum(v for k, v in types.items() if "connector" in k),
        "lane_type_counts": dict(types),
        "left_adjacency_non_empty_count": c(lambda v: bool(v.left_neighbor_lane_ids)),
        "left_adjacency_non_empty_proportion": c(lambda v: bool(v.left_neighbor_lane_ids)) / n if n else None,
        "right_adjacency_non_empty_count": c(lambda v: bool(v.right_neighbor_lane_ids)),
        "right_adjacency_non_empty_proportion": c(lambda v: bool(v.right_neighbor_lane_ids)) / n if n else None,
        "predecessor_non_empty_count": c(lambda v: bool(v.entry_lane_ids)),
        "predecessor_non_empty_proportion": c(lambda v: bool(v.entry_lane_ids)) / n if n else None,
        "successor_non_empty_count": c(lambda v: bool(v.exit_lane_ids)),
        "successor_non_empty_proportion": c(lambda v: bool(v.exit_lane_ids)) / n if n else None,
        "lane_connectors_with_no_adjacency_or_topology_count": connector_no_topology,
        "centerline_point_count_distribution": _dist([float(len(v.centerline_xy)) for v in lanes.values()]),
        "lane_length_distribution_m": _dist([_lane_length(v) for v in lanes.values()]),
        "topology_source_counts": dict(Counter(v.topology_source for v in lanes.values())),
        "adapter_counts": dict(counts or {}),
        "stage5_limitation": "Stage5 LaneInfo exposes predecessor/successor as entry_lane_ids/exit_lane_ids; current Stage5D assignment uses same/left/right lane relations, so predecessor/successor are diagnostic/topology context only.",
    }


def write_lane_topology_debug_artifacts(out_dir: Path, summary: Dict[str, Any]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "nuplan_lane_topology_debug_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = out_dir / "nuplan_lane_topology_debug_report.md"
    lines = ["# nuPlan LaneInfo Topology Debug", "", f"- LaneInfo objects: `{summary.get('lane_info_count')}`", f"- lane_count: `{summary.get('lane_count')}`", f"- lane_connector_count: `{summary.get('lane_connector_count')}`", f"- left adjacency non-empty: `{summary.get('left_adjacency_non_empty_count')}` / `{summary.get('lane_info_count')}`", f"- right adjacency non-empty: `{summary.get('right_adjacency_non_empty_count')}` / `{summary.get('lane_info_count')}`", f"- predecessor non-empty: `{summary.get('predecessor_non_empty_count')}`", f"- successor non-empty: `{summary.get('successor_non_empty_count')}`", f"- lane connectors without adjacency/topology: `{summary.get('lane_connectors_with_no_adjacency_or_topology_count')}`", f"- centerline point count distribution: `{summary.get('centerline_point_count_distribution')}`", f"- lane length distribution m: `{summary.get('lane_length_distribution_m')}`", "", "## Limitation", "", str(summary.get('stage5_limitation'))]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_json": str(json_path), "report_md": str(md_path)}

def _point_obj(x: float, y: float) -> Any:
    try:
        from nuplan.common.actor_state.state_representation import Point2D
        return Point2D(x, y)
    except Exception:
        return (x, y)


def _layer_values() -> List[Any]:
    try:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        return [getattr(SemanticMapLayer, name) for name in ("LANE", "LANE_CONNECTOR") if hasattr(SemanticMapLayer, name)]
    except Exception:
        return ["LANE", "LANE_CONNECTOR", "lane", "lane_connector"]


def _objects_from_result(res: Any, layer: Any) -> List[Any]:
    if isinstance(res, dict):
        keys = [layer, getattr(layer, "name", None), getattr(layer, "value", None), str(layer), str(layer).lower()]
        for key in keys:
            try:
                if key in res:
                    return list(res.get(key) or [])
            except TypeError:
                pass
    return []


def extract_nuplan_lane_infos(api: Any, ego_xy: np.ndarray, radius: float = 120.0) -> Tuple[Dict[str, LaneInfo], Dict[str, int]]:
    """Convert local nuPlan map objects to Stage-5-compatible LaneInfo objects."""
    counts: Counter[str] = Counter()
    if api is None or ego_xy.size == 0:
        counts["none"] += 1
        return {}, dict(counts)
    objects: List[Tuple[Any, str]] = []
    for x, y in ego_xy:
        for layer in _layer_values():
            try:
                res = api.get_proximal_map_objects(_point_obj(float(x), float(y)), float(radius), [layer])
                lname = str(getattr(layer, "name", layer)).lower()
                objects.extend((obj, lname) for obj in _objects_from_result(res, layer))
                counts["map_query_success"] += 1
            except Exception:
                counts["map_query_failed"] += 1
    lanes: Dict[str, LaneInfo] = {}
    for obj, lname in objects:
        info = lane_info_from_nuplan_object(obj, lname)
        if info is None:
            counts["geometry_unavailable"] += 1
            continue
        lanes[info.lane_id] = info
    counts.update(enrich_geometric_adjacency(lanes))
    for info in lanes.values():
        if info.left_neighbor_lane_ids or info.right_neighbor_lane_ids:
            counts["nuplan_topology"] += 1
        else:
            counts["geometric"] += 1
    if not lanes:
        counts["none"] += 1
    return lanes, dict(counts)
