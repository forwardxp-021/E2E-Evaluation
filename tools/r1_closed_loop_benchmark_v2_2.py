#!/usr/bin/env python3
"""Route-continuous HLC native-reference construction for B2.9-B.

This module versions only the reference-corridor assembly.  The frozen HLC
progress schedule and trajectory realization remain in v2.1.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from shapely.geometry import LineString, Point

from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1


BUILDER_VERSION = "build_hlc_route_continuous_reference_v2_2"
JOIN_GAP_THRESHOLD_M = "UNKNOWN_NOT_PREDEFINED"


def _edge_xy(edge: Any) -> np.ndarray:
    xy = np.asarray(edge.baseline_path.linestring.coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError(f"ROUTE_CONTINUOUS_INVALID_NATIVE_BASELINE:{edge.id}")
    if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) <= 0.0):
        raise ValueError(f"ROUTE_CONTINUOUS_DUPLICATE_NATIVE_POINT:{edge.id}")
    return xy


def _native_edge(map_api: Any, edge_id: str) -> Any:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    found = []
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        try:
            item = map_api.get_map_object(str(edge_id), layer)
        except ValueError:
            item = None
        if item is not None:
            found.append(item)
    if len(found) != 1:
        raise ValueError(f"ROUTE_CONTINUOUS_EDGE_NOT_EXACTLY_ONE:{edge_id}")
    return found[0]


def _is_connector(edge: Any) -> bool:
    return "connector" in type(edge).__name__.lower()


def _adjacent(left: Any, right: Any) -> bool:
    if str(left.id) == str(right.id):
        return True
    left_ids = {str(item.id) for item in left.adjacent_edges if item is not None}
    right_ids = {str(item.id) for item in right.adjacent_edges if item is not None}
    return str(right.id) in left_ids and str(left.id) in right_ids


def _terminal_lanes(edge: Any) -> List[Any]:
    if not _is_connector(edge):
        return [edge]
    return list(edge.outgoing_edges)


def _direction_continuous(current: Any, successor: Any) -> bool:
    current_xy, successor_xy = _edge_xy(current), _edge_xy(successor)
    a = current_xy[-1] - current_xy[-2]
    b = successor_xy[1] - successor_xy[0]
    return bool(float(np.dot(a, b)) > 0.0)


def _corresponding_pair(source: Any, target: Any) -> bool:
    source_terminals, target_terminals = _terminal_lanes(source), _terminal_lanes(target)
    if len(source_terminals) != 1 or len(target_terminals) != 1:
        return False
    return _adjacent(source_terminals[0], target_terminals[0])


def _next_route_index(route: Sequence[str], cursor: int, roadblock_id: str) -> int | None:
    matches = [index for index in range(cursor + 1, len(route)) if route[index] == str(roadblock_id)]
    return matches[0] if len(matches) == 1 else None


def _select_successor_pair(
    source: Any,
    target: Any,
    route: Sequence[str],
    route_cursor: int,
) -> Tuple[Any, Any, int, Dict[str, Any]]:
    source_out, target_out = list(source.outgoing_edges), list(target.outgoing_edges)
    candidates: List[Tuple[Any, Any, int]] = []
    for source_next in source_out:
        if not _direction_continuous(source, source_next):
            continue
        next_cursor = _next_route_index(route, route_cursor, str(source_next.get_roadblock_id()))
        if next_cursor is None:
            continue
        for target_next in target_out:
            if not _direction_continuous(target, target_next):
                continue
            if not _corresponding_pair(source_next, target_next):
                continue
            candidates.append((source_next, target_next, next_cursor))
    audit = {
        "source_outgoing_count": len(source_out),
        "target_outgoing_count": len(target_out),
        "route_constrained_corresponding_pair_count": len(candidates),
        "candidate_pairs": [
            {
                "source_edge_id": str(item[0].id),
                "target_edge_id": str(item[1].id),
                "source_roadblock_id": str(item[0].get_roadblock_id()),
                "target_roadblock_id": str(item[1].get_roadblock_id()),
                "route_occurrence_index": item[2],
            }
            for item in candidates
        ],
    }
    if len(candidates) != 1:
        raise ValueError(
            "ROUTE_CONTINUOUS_TOPOLOGY_AMBIGUITY_FAIL_CLOSED:"
            f"source={source.id}:target={target.id}:candidates={len(candidates)}"
        )
    selected_source, selected_target, selected_cursor = candidates[0]
    return selected_source, selected_target, selected_cursor, audit


def _append_native(points: List[List[float]], edge: Any) -> Dict[str, Any]:
    xy = _edge_xy(edge)
    gap = 0.0 if not points else float(np.linalg.norm(np.asarray(points[-1]) - xy[0]))
    if points and gap != 0.0:
        raise ValueError(
            "ROUTE_CONTINUOUS_JOIN_PRECISION_UNKNOWN_FAIL_CLOSED:"
            f"edge={edge.id}:observed_gap_m={gap}"
        )
    start = 1 if points else 0
    points.extend(xy[start:].tolist())
    return {
        "edge_id": str(edge.id),
        "roadblock_id": str(edge.get_roadblock_id()),
        "edge_type": type(edge).__name__,
        "native_length_m": float(edge.baseline_path.linestring.length),
        "join_gap_m": gap,
        "join_gap_threshold_m": JOIN_GAP_THRESHOLD_M,
        "join_rule": "EXACT_NATIVE_ENDPOINT_IDENTITY_NO_NUMERIC_TOLERANCE",
    }


def _project_arc(points: Sequence[Sequence[float]], current_ego: Mapping[str, Any]) -> float:
    line = LineString(points)
    point = Point(float(current_ego["rear_axle"]["x"]), float(current_ego["rear_axle"]["y"]))
    return float(line.project(point))


def _length(points: Sequence[Sequence[float]]) -> float:
    return float(LineString(points).length)


def build_hlc_route_continuous_reference_v2_2(
    map_api: Any,
    route_roadblock_ids: Sequence[str],
    source_lane_id: str,
    target_lane_id: str,
    current_ego: Mapping[str, Any],
    required_forward_m: float,
) -> Dict[str, Any]:
    """Build paired native corridors until both cover the requested envelope.

    Successors are accepted only when there is exactly one source-route
    occurrence and exactly one source/target topology-corresponding pair.
    Any branch, merge, reversal, missing edge, or non-exact native join fails
    closed; no distance-based or outcome-based tie break exists.
    """
    if not math.isfinite(float(required_forward_m)) or float(required_forward_m) <= 0.0:
        raise ValueError("ROUTE_CONTINUOUS_REQUIRED_FORWARD_MUST_BE_POSITIVE")
    route = [str(value) for value in route_roadblock_ids]
    if not route:
        raise ValueError("ROUTE_CONTINUOUS_EMPTY_FROZEN_ROUTE")
    source = _native_edge(map_api, str(source_lane_id))
    target = _native_edge(map_api, str(target_lane_id))
    if not _adjacent(source, target):
        raise ValueError("ROUTE_CONTINUOUS_INITIAL_SOURCE_TARGET_NOT_MUTUALLY_ADJACENT")
    initial_rb = str(source.get_roadblock_id())
    if str(target.get_roadblock_id()) != initial_rb:
        raise ValueError("ROUTE_CONTINUOUS_INITIAL_ROADBLOCK_MISMATCH")
    occurrences = [index for index, value in enumerate(route) if value == initial_rb]
    if len(occurrences) != 1:
        raise ValueError("ROUTE_CONTINUOUS_INITIAL_ROUTE_OCCURRENCE_AMBIGUOUS")
    route_cursor = occurrences[0]
    source_points: List[List[float]] = []
    target_points: List[List[float]] = []
    source_components, target_components, transitions = [], [], []
    visited_pairs: set[Tuple[str, str]] = set()
    while True:
        pair = (str(source.id), str(target.id))
        if pair in visited_pairs:
            raise ValueError("ROUTE_CONTINUOUS_DUPLICATE_OR_SELF_INTERSECTION_FAIL_CLOSED")
        visited_pairs.add(pair)
        source_components.append(_append_native(source_points, source))
        target_components.append(_append_native(target_points, target))
        source_arc = _project_arc(source_points, current_ego)
        target_arc = _project_arc(target_points, current_ego)
        source_margin = _length(source_points) - source_arc - float(required_forward_m)
        target_margin = _length(target_points) - target_arc - float(required_forward_m)
        if source_margin >= 0.0 and target_margin >= 0.0:
            break
        source_next, target_next, route_cursor, audit = _select_successor_pair(
            source, target, route, route_cursor
        )
        audit.update(
            {
                "selected_source_edge_id": str(source_next.id),
                "selected_target_edge_id": str(target_next.id),
                "selection_rule": "UNIQUE_ROUTE_CONSTRAINED_TOPOLOGY_CORRESPONDING_PAIR",
            }
        )
        transitions.append(audit)
        source, target = source_next, target_next
        if len(visited_pairs) > len(route):
            raise ValueError("ROUTE_CONTINUOUS_ROUTE_LENGTH_BOUND_EXCEEDED")
    source_line, target_line = LineString(source_points), LineString(target_points)
    if not source_line.is_simple or not target_line.is_simple:
        raise ValueError("ROUTE_CONTINUOUS_SELF_INTERSECTION_FAIL_CLOSED")
    return {
        "builder_version": BUILDER_VERSION,
        "source_reference_xy": np.asarray(source_points, dtype=np.float64),
        "target_reference_xy": np.asarray(target_points, dtype=np.float64),
        "source_current_arc_m": source_arc,
        "target_current_arc_m": target_arc,
        "source_total_length_m": float(source_line.length),
        "target_total_length_m": float(target_line.length),
        "source_remaining_margin_m": source_margin,
        "target_remaining_margin_m": target_margin,
        "required_forward_m": float(required_forward_m),
        "source_components": source_components,
        "target_components": target_components,
        "transitions": transitions,
        "route_occurrence_cursor": route_cursor,
        "extrapolation_used": False,
        "manual_points_used": False,
        "distance_or_outcome_tie_break_used": False,
        "topology_ambiguity": False,
    }


def build_hlc_route_continuous_geometry_v2_2(
    map_api: Any,
    route_roadblock_ids: Sequence[str],
    source_lane_id: str,
    target_lane_id: str,
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    arm: str,
) -> Tuple[Sequence[Dict[str, Any]], Dict[str, Any]]:
    required = max(0.2, float(current_ego["speed_mps"])) * 7.9
    corridor = build_hlc_route_continuous_reference_v2_2(
        map_api,
        route_roadblock_ids,
        source_lane_id,
        target_lane_id,
        current_ego,
        required,
    )
    states = build_hlc_native_geometry_v1_1(
        current_ego,
        absolute_episode_time_s,
        corridor["source_reference_xy"],
        corridor["target_reference_xy"],
        corridor["source_current_arc_m"],
        corridor["target_current_arc_m"],
        arm,
    )
    return states, corridor


__all__ = [
    "BUILDER_VERSION",
    "JOIN_GAP_THRESHOLD_M",
    "build_hlc_route_continuous_geometry_v2_2",
    "build_hlc_route_continuous_reference_v2_2",
]
