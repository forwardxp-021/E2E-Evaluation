#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from tools.waymo_lane_utils import LaneInfo, _build_lane_geom, find_best_lane_for_agent, project_point_to_lane


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
    left_ids = _neighbor_ids(obj, ("left_neighbors", "left_neighbor", "adjacent_edges_left", "left_adjacent_edges"))
    right_ids = _neighbor_ids(obj, ("right_neighbors", "right_neighbor", "adjacent_edges_right", "right_adjacent_edges"))
    entry_ids = _neighbor_ids(obj, ("incoming_edges", "predecessors", "entry_lanes"))
    exit_ids = _neighbor_ids(obj, ("outgoing_edges", "successors", "exit_lanes"))
    return LaneInfo(
        lane_id=_obj_id(obj), centerline_xy=arr, seg_heading=geom[0], seg_len=geom[1], s_prefix=geom[2],
        seg_start_xy=geom[3], seg_vec_xy=geom[4], seg_den=geom[5], bbox_min_xy=geom[6], bbox_max_xy=geom[7],
        bbox_center_xy=geom[8], left_neighbor_lane_ids=left_ids, right_neighbor_lane_ids=right_ids,
        entry_lane_ids=entry_ids, exit_lane_ids=exit_ids, lane_type=lane_type,
        topology_source="nuplan_topology" if (left_ids or right_ids) else "geometric_lane_adjacency",
    )


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
    for info in lanes.values():
        if info.left_neighbor_lane_ids or info.right_neighbor_lane_ids:
            counts["nuplan_topology"] += 1
        else:
            counts["geometric"] += 1
    if not lanes:
        counts["none"] += 1
    return lanes, dict(counts)
