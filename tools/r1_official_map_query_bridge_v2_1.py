#!/usr/bin/env python3
"""Read-only binding from the R1 context protocol to the official nuPlan map API."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from tools.waymo_lane_utils import LaneInfo


def _lane_object(map_api: AbstractMap, lane_id: str) -> Any:
    found = []
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        try:
            value = map_api.get_map_object(str(lane_id), layer)
        except ValueError:
            value = None
        if value is not None:
            found.append(value)
    if len(found) != 1:
        raise ValueError(f"OFFICIAL_MAP_LANE_ID_NOT_UNIQUE:{lane_id}")
    return found[0]


def _lane_info(edge: Any, left: str | None, right: str | None) -> LaneInfo:
    points = np.asarray([[float(p.x), float(p.y)] for p in edge.baseline_path.discrete_path], dtype=np.float64)
    if len(points) < 2:
        points = np.asarray(edge.baseline_path.linestring.coords, dtype=np.float64)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if len(points) < 2 or np.any(lengths <= 0):
        raise ValueError("OFFICIAL_MAP_BASELINE_PATH_INVALID")
    bbox_min, bbox_max = points.min(axis=0), points.max(axis=0)
    return LaneInfo(str(edge.id), points, np.arctan2(delta[:, 1], delta[:, 0]), lengths, np.r_[0.0, np.cumsum(lengths)], points[:-1], delta, np.sum(delta * delta, axis=1), bbox_min, bbox_max, (bbox_min + bbox_max) / 2.0, left_neighbor_lane_ids=[] if left is None else [left], right_neighbor_lane_ids=[] if right is None else [right], topology_source="OFFICIAL_NUPLAN_NATIVE_ADJACENCY")


class R1OfficialMapQueryBridgeV2_1:
    """Fail-closed adapter; every geometric answer is produced by AbstractMap objects."""

    def __init__(self, map_api: AbstractMap) -> None:
        if not isinstance(map_api, AbstractMap):
            raise TypeError("OFFICIAL_MAP_API_REQUIRED")
        self._map = map_api

    def _containing_lane(self, xy: Tuple[float, float]) -> Any:
        point = Point2D(float(xy[0]), float(xy[1]))
        found = list(self._map.get_all_map_objects(point, SemanticMapLayer.LANE))
        found += list(self._map.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR))
        unique = {str(edge.id): edge for edge in found}
        if len(unique) != 1:
            raise ValueError("OFFICIAL_MAP_LANE_AMBIGUITY_FAIL_CLOSED")
        return next(iter(unique.values()))

    def lane_for_actor(self, actor: Mapping[str, Any]) -> str:
        return str(self._containing_lane((float(actor["x"]), float(actor["y"]))).id)

    def project(self, lane_id: str, xy: Tuple[float, float]) -> Mapping[str, Any]:
        edge = _lane_object(self._map, str(lane_id))
        point = Point2D(float(xy[0]), float(xy[1]))
        pose = edge.baseline_path.get_nearest_pose_from_position(point)
        arc = float(edge.baseline_path.get_nearest_arc_length_from_position(point))
        dx, dy = point.x - float(pose.x), point.y - float(pose.y)
        lateral = -math.sin(float(pose.heading)) * dx + math.cos(float(pose.heading)) * dy
        discrete = edge.baseline_path.discrete_path
        segment_index = max(0, min(len(discrete) - 2, int(np.argmin([(float(item.x) - point.x) ** 2 + (float(item.y) - point.y) ** 2 for item in discrete]))))
        return {"lane_id": str(edge.id), "arc_m": arc, "s": arc, "lateral_offset_m": lateral, "l": lateral, "distance_to_lane_m": abs(lateral), "distance_to_lane": abs(lateral), "heading": float(pose.heading), "tangent": [math.cos(float(pose.heading)), math.sin(float(pose.heading))], "segment_index": segment_index, "source": "OFFICIAL_NUPLAN_BASELINE_PATH"}

    def native_reference_xy(self, lane_id: str) -> np.ndarray:
        edge = _lane_object(self._map, str(lane_id))
        reference = np.asarray(edge.baseline_path.linestring.coords, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2 or not np.isfinite(reference).all():
            raise ValueError("OFFICIAL_MAP_BASELINE_PATH_INVALID")
        return reference

    def lane_context(self, ego_xy: Tuple[float, float], route_roadblock_ids: Sequence[str]) -> Mapping[str, Any]:
        current = self._containing_lane(ego_xy)
        if str(current.get_roadblock_id()) not in {str(value) for value in route_roadblock_ids}:
            raise ValueError("OFFICIAL_MAP_CURRENT_LANE_OUTSIDE_FROZEN_ROUTE")
        adjacent = current.adjacent_edges
        left, right = adjacent[0], adjacent[1]
        current_id, left_id, right_id = str(current.id), None if left is None else str(left.id), None if right is None else str(right.id)
        infos: Dict[str, LaneInfo] = {current_id: _lane_info(current, left_id, right_id)}
        if left is not None:
            left_adj = left.adjacent_edges
            infos[left_id] = _lane_info(left, None if left_adj[0] is None else str(left_adj[0].id), None if left_adj[1] is None else str(left_adj[1].id))
        if right is not None:
            right_adj = right.adjacent_edges
            infos[right_id] = _lane_info(right, None if right_adj[0] is None else str(right_adj[0].id), None if right_adj[1] is None else str(right_adj[1].id))
        projection = self.project(current_id, ego_xy)
        target_adjacent = []
        for target in (left, right):
            if target is not None:
                target_adjacent.extend(str(value.id) for value in target.adjacent_edges if value is not None)
        return {"valid": True, "current_lane_id": current_id, "left_lane_id": left_id, "right_lane_id": right_id, "tangent": projection["tangent"], "road_class": "LANE_CONNECTOR" if current.__class__.__name__.lower().endswith("connector") else "LANE", "source_immediate_adjacent_lane_ids": [value for value in (left_id, right_id) if value], "target_immediate_adjacent_lane_ids": sorted(set(target_adjacent)), "current_immediate_adjacent_lane_ids": [value for value in (left_id, right_id) if value], "lane_infos": infos, "official_map_api_type": f"{type(self._map).__module__}.{type(self._map).__name__}"}

    def static_stop_control_ahead(self, ego_xy: Tuple[float, float], route_roadblock_ids: Sequence[str]) -> bool:
        edge = self._containing_lane(ego_xy)
        route = {str(value) for value in route_roadblock_ids}
        candidates = [edge] + [candidate for candidate in edge.outgoing_edges if str(candidate.get_roadblock_id()) in route]
        return any(bool(candidate.stop_lines) for candidate in candidates)


__all__ = ["R1OfficialMapQueryBridgeV2_1"]
