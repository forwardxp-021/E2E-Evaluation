#!/usr/bin/env python3
"""Prospective R1 closed-loop residual benchmark primitives (v2).

This module is deliberately independent from the historical B2.1 planner.
It performs no rollout, selection, representation, BDD, probe, checkpoint, or
RBR work.  Frozen HLC Option-B / TSB Option-A phase schedules and mechanism
thresholds are imported as immutable scientific definitions; spatial
realization, measurement source, and replan anchoring are versioned here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from tools.r1_context_mechanism_core import (
    calculate_hlc_option_b,
    calculate_tsb_option_a,
    qualify_hlc_pair,
    qualify_tsb_pair,
)
from tools.r1_official_technical_smoke_planner import (
    HLC_BASELINE,
    HLC_TREATMENT,
    TSB_BASELINE,
    TSB_TREATMENT,
    hlc_progress,
)


DT_SECONDS = 0.1
WINDOW_FRAMES = 80
WINDOW_SECONDS = 8.0
HLC_PRIMARY_F_MATCH_CALIPERS = {
    "mean_speed": 0.708203939,
    "end_minus_start_speed": 0.978755681,
    "path_length": 5.38423459,
}
TSB_PRIMARY_F_MATCH_CALIPERS = {
    **HLC_PRIMARY_F_MATCH_CALIPERS,
    "mean_abs_accel": 0.11777666,
}
HLC_SCIENTIFIC_AMENDMENT = "R1_HLC_Residual_Benchmark_Scientific_Amendment_v1.0"


def wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def exact_realized_window(trace_rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Return actual current-ego states for iterations 0..79 or fail closed."""
    if len(trace_rows) < WINDOW_FRAMES:
        raise ValueError("NOT_EVALUABLE_REALIZED_WINDOW: fewer than 80 simulator iterations")
    rows = list(trace_rows[:WINDOW_FRAMES])
    if [int(row["iteration_index"]) for row in rows] != list(range(WINDOW_FRAMES)):
        raise ValueError("NOT_EVALUABLE_TEMPORAL_GRID: iterations are not exactly 0..79")
    times = [int(row["current_ego"]["time_us"]) for row in rows]
    if any(times[i + 1] - times[i] != 100_000 for i in range(WINDOW_FRAMES - 1)):
        raise ValueError("NOT_EVALUABLE_TEMPORAL_GRID: physical current-ego timestamps are not exact 0.1s")
    return [row["current_ego"] for row in rows]


def trajectory_arrays(states: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(states) != WINDOW_FRAMES:
        raise ValueError("trajectory must contain exactly 80 states")
    time = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    xy = np.asarray([[s["rear_axle"]["x"], s["rear_axle"]["y"]] for s in states], dtype=np.float64)
    heading = np.asarray([s["rear_axle"]["heading"] for s in states], dtype=np.float64)
    speed = np.asarray([s["speed_mps"] for s in states], dtype=np.float64)
    if not np.isfinite(xy).all() or not np.isfinite(heading).all() or not np.isfinite(speed).all():
        raise ValueError("trajectory contains non-finite values")
    return time, xy, heading, speed


def trajectory_descriptors(states: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    time, xy, _, speed = trajectory_arrays(states)
    accel = np.diff(speed, prepend=speed[0]) / DT_SECONDS
    return {
        "mean_speed": round(float(np.mean(speed)), 6),
        "end_minus_start_speed": round(float(speed[-1] - speed[0]), 6),
        "mean_abs_accel": round(float(np.mean(np.abs(accel))), 6),
        "path_length": round(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))), 6),
        "heading_change_abs_total": round(float(np.sum(np.abs(np.diff(np.unwrap(np.asarray([s["rear_axle"]["heading"] for s in states])))))), 6),
    }


def prospective_primary_f_match(
    baseline: Mapping[str, float], treatment: Mapping[str, float], family: str
) -> Dict[str, Any]:
    """Apply the prospective Primary F_match; HLC heading is secondary only."""
    calipers = HLC_PRIMARY_F_MATCH_CALIPERS if family == "R-HLC" else TSB_PRIMARY_F_MATCH_CALIPERS
    delta = {key: round(abs(float(treatment[key]) - float(baseline[key])), 6) for key in calipers}
    by_feature = {key: delta[key] <= limit + 1e-12 for key, limit in calipers.items()}
    return {
        "status": "F_MATCH_PASS" if all(by_feature.values()) else "F_MATCH_FAIL",
        "pass": all(by_feature.values()),
        "primary_features": list(calipers),
        "calipers": calipers,
        "absolute_delta": delta,
        "pass_by_feature": by_feature,
        "heading_change_abs_total": "SECONDARY_MECHANISM_PROXIMAL_AUDIT",
    }


def first_state_error(current_ego: Mapping[str, Any], first: Mapping[str, Any]) -> Dict[str, Any]:
    dx = float(first["rear_axle"]["x"]) - float(current_ego["rear_axle"]["x"])
    dy = float(first["rear_axle"]["y"]) - float(current_ego["rear_axle"]["y"])
    result = {
        "position_error_m": math.hypot(dx, dy),
        "heading_error_rad": abs(wrap_angle(float(first["rear_axle"]["heading"]) - float(current_ego["rear_axle"]["heading"]))),
        "speed_error_mps": abs(float(first["speed_mps"]) - float(current_ego["speed_mps"])),
        "timestamp_error_us": abs(int(first["time_us"]) - int(current_ego["time_us"])),
    }
    result["exact_construction_identity"] = all(value == 0 for value in result.values())
    return result


def polyline_arclength(xy: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("native reference must be finite Nx2 with N>=2")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment <= 0):
        raise ValueError("native reference contains reversal/duplicate segment")
    return np.r_[0.0, np.cumsum(segment)]


def sample_native_reference_no_extrapolation(
    xy: Sequence[Sequence[float]], query_arc_m: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=np.float64)
    arc = polyline_arclength(points)
    query = np.asarray(query_arc_m, dtype=np.float64)
    if np.any(query < -1e-12) or np.any(query > arc[-1] + 1e-12):
        raise ValueError("NATIVE_REFERENCE_COVERAGE_FAIL_NO_EXTRAPOLATION")
    x = np.interp(query, arc, points[:, 0])
    y = np.interp(query, arc, points[:, 1])
    sampled = np.column_stack((x, y))
    index = np.clip(np.searchsorted(arc, query, side="right") - 1, 0, len(points) - 2)
    delta = points[index + 1] - points[index]
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    return sampled, heading


def build_native_route_reference(
    map_api: Any,
    route_roadblock_ids: Sequence[str],
    current_ego: Mapping[str, Any],
    required_forward_m: float | None = None,
) -> np.ndarray:
    """Follow native lane/connector topology through the frozen route sequence.

    No geometric bridge or extrapolated segment is inserted.  The first edge
    is chosen deterministically by official-map distance/heading at current
    ego; every successor must be a native outgoing edge in the next route
    roadblock.
    """
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point

    if not route_roadblock_ids:
        raise ValueError("NATIVE_ROUTE_FAIL: empty route")
    point = Point2D(float(current_ego["rear_axle"]["x"]), float(current_ego["rear_axle"]["y"]))
    ego_heading = float(current_ego["rear_axle"]["heading"])

    def roadblock(route_id: str) -> Any:
        obj = map_api.get_map_object(str(route_id), SemanticMapLayer.ROADBLOCK)
        if obj is None:
            obj = map_api.get_map_object(str(route_id), SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if obj is None:
            raise ValueError(f"NATIVE_ROUTE_FAIL: unresolved route object {route_id}")
        return obj

    route = [str(x) for x in route_roadblock_ids]
    local_edges = list(map_api.get_all_map_objects(point, SemanticMapLayer.LANE))
    local_edges += list(map_api.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR))
    local_edges = [edge for edge in local_edges if str(edge.get_roadblock_id()) in route]
    if not local_edges:
        raise ValueError("NATIVE_ROUTE_FAIL: current ego has no native route edge")
    local_edges.sort(key=lambda edge: (
        abs(wrap_angle(float(edge.baseline_path.get_nearest_pose_from_position(point).heading) - ego_heading)),
        float(edge.baseline_path.linestring.distance(Point(point.x, point.y))),
        str(edge.id),
    ))
    selected = [local_edges[0]]
    start_index = route.index(str(selected[0].get_roadblock_id()))
    current_arc = float(selected[0].baseline_path.get_nearest_arc_length_from_position(point))
    available = float(selected[0].baseline_path.linestring.length) - current_arc
    for route_id in route[start_index + 1:]:
        if required_forward_m is not None and available >= float(required_forward_m):
            break
        candidates = list(roadblock(str(route_id)).interior_edges)
        outgoing = {str(edge.id) for edge in selected[-1].outgoing_edges}
        native = [edge for edge in candidates if str(edge.id) in outgoing]
        if not native:
            raise ValueError(f"NATIVE_ROUTE_FAIL: no native successor into {route_id}")
        selected.append(sorted(native, key=lambda edge: str(edge.id))[0])
        available += float(selected[-1].baseline_path.linestring.length)
    if required_forward_m is not None and available < float(required_forward_m):
        raise ValueError("NATIVE_ROUTE_FAIL: insufficient native forward coverage")
    points = []
    for edge in selected:
        coords = [[float(x), float(y)] for x, y in edge.baseline_path.linestring.coords]
        if points and coords and points[-1] == coords[0]:
            coords = coords[1:]
        points.extend(coords)
    reference = np.asarray(points, dtype=np.float64)
    polyline_arclength(reference)
    return reference


def _state(x: float, y: float, heading: float, speed: float, time_us: int) -> Dict[str, Any]:
    return {"rear_axle": {"x": float(x), "y": float(y), "heading": float(heading)}, "speed_mps": float(speed), "time_us": int(time_us)}


def frozen_tsb_acceleration(absolute_episode_time_s: float, arm: str) -> float:
    t = float(absolute_episode_time_s)
    diverge = 1.1
    if arm == TSB_BASELINE:
        return -1.0 if diverge <= t < diverge + 0.95 else 0.0
    if arm != TSB_TREATMENT:
        raise ValueError(f"unsupported TSB arm: {arm}")
    if diverge <= t < diverge + 0.5:
        return -0.9
    if diverge + 0.5 <= t < diverge + 1.2:
        return 0.4
    if diverge + 1.2 <= t < diverge + 1.7:
        return -0.9
    return 0.0


def build_tsb_route_aligned(
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    route_reference_xy: Sequence[Sequence[float]],
    current_route_arc_m: float,
    arm: str,
) -> Sequence[Dict[str, Any]]:
    """Realize the unchanged TSB profile along one shared native route."""
    times = absolute_episode_time_s + np.arange(WINDOW_FRAMES) * DT_SECONDS
    speed = np.empty(WINDOW_FRAMES, dtype=np.float64)
    distance = np.zeros(WINDOW_FRAMES, dtype=np.float64)
    speed[0] = float(current_ego["speed_mps"])
    for i in range(1, WINDOW_FRAMES):
        accel = frozen_tsb_acceleration(times[i - 1], arm)
        speed[i] = max(0.2, round(float(speed[i - 1] + accel * DT_SECONDS), 12))
        distance[i] = distance[i - 1] + 0.5 * (speed[i - 1] + speed[i]) * DT_SECONDS
    xy, heading = sample_native_reference_no_extrapolation(route_reference_xy, current_route_arc_m + distance)
    start_us = int(current_ego["time_us"])
    states = [_state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000) for i in range(WINDOW_FRAMES)]
    states[0] = dict(current_ego)
    if not first_state_error(current_ego, states[0])["exact_construction_identity"]:
        raise AssertionError("current-ego first-state construction identity failed")
    return states


def build_hlc_native_geometry(
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    source_reference_xy: Sequence[Sequence[float]],
    target_reference_xy: Sequence[Sequence[float]],
    source_current_arc_m: float,
    target_current_arc_m: float,
    arm: str,
) -> Sequence[Dict[str, Any]]:
    """Realize unchanged HLC progress on native geometry without extrapolation."""
    relative = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    absolute = absolute_episode_time_s + relative
    speed = np.full(WINDOW_FRAMES, float(current_ego["speed_mps"]), dtype=np.float64)
    distance = speed * relative
    source, source_heading = sample_native_reference_no_extrapolation(source_reference_xy, source_current_arc_m + distance)
    target, target_heading = sample_native_reference_no_extrapolation(target_reference_xy, target_current_arc_m + distance)
    progress = hlc_progress(absolute, arm)
    xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    heading = np.unwrap(source_heading) * (1.0 - progress) + np.unwrap(target_heading) * progress
    start_us = int(current_ego["time_us"])
    states = [_state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000) for i in range(WINDOW_FRAMES)]
    states[0] = dict(current_ego)
    if not first_state_error(current_ego, states[0])["exact_construction_identity"]:
        raise AssertionError("current-ego first-state construction identity failed")
    return states


def _polyline_self_intersects(xy: np.ndarray) -> bool:
    from shapely.geometry import LineString
    return not bool(LineString(xy).is_simple)


def hlc_map_applicability(
    *,
    source_lane_id: str | None,
    target_lane_id: str | None,
    target_is_native_adjacent: bool,
    source_roadblock_id: str | None,
    target_roadblock_id: str | None,
    route_roadblock_ids: Sequence[str],
    source_reference_xy: Sequence[Sequence[float]],
    target_reference_xy: Sequence[Sequence[float]],
    source_current_arc_m: float,
    target_current_arc_m: float,
    required_forward_m: float,
    engineering: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Outcome-blind exact topology/coverage audit with frozen engineering limits."""
    reasons = []
    if not source_lane_id:
        reasons.append("NATIVE_SOURCE_LANE_UNRESOLVED")
    if not target_lane_id or not target_is_native_adjacent:
        reasons.append("NATIVE_TARGET_ADJACENCY_FAIL")
    route = {str(x) for x in route_roadblock_ids}
    if not source_roadblock_id or str(source_roadblock_id) not in route or source_roadblock_id != target_roadblock_id:
        reasons.append("ROUTE_CONSISTENCY_FAIL")
    try:
        source = np.asarray(source_reference_xy, dtype=np.float64)
        target = np.asarray(target_reference_xy, dtype=np.float64)
        source_arc = polyline_arclength(source)
        target_arc = polyline_arclength(target)
        source_segments = np.diff(source, axis=0)
        target_segments = np.diff(target, axis=0)
        if np.any(np.sum(source_segments[1:] * source_segments[:-1], axis=1) <= 0) or np.any(np.sum(target_segments[1:] * target_segments[:-1], axis=1) <= 0):
            reasons.append("REFERENCE_REVERSAL_FAIL")
        if float(np.median(np.sum(source_segments[: min(len(source_segments), len(target_segments))] * target_segments[: min(len(source_segments), len(target_segments))], axis=1))) <= 0:
            reasons.append("SAME_TRAVEL_DIRECTION_FAIL")
        if _polyline_self_intersects(source) or _polyline_self_intersects(target):
            reasons.append("SELF_INTERSECTION_FAIL")
        if source_arc[-1] - float(source_current_arc_m) < float(required_forward_m) or target_arc[-1] - float(target_current_arc_m) < float(required_forward_m):
            reasons.append("NATIVE_REFERENCE_COVERAGE_FAIL_NO_EXTRAPOLATION")
    except ValueError as exc:
        reasons.append(f"REFERENCE_TOPOLOGY_FAIL:{exc}")
    if engineering is None:
        reasons.append("PRE_ROLLOUT_ENGINEERING_AUDIT_REQUIRED")
    else:
        frozen_limits = {"max_abs_lateral_accel_mps2": 6.0, "max_abs_yaw_rate_radps": 1.0, "max_abs_curvature_inv_m": 0.5}
        for name, limit in frozen_limits.items():
            if float(engineering.get(name, math.inf)) > limit:
                reasons.append(f"FROZEN_ENGINEERING_LIMIT_FAIL:{name}")
    return {
        "status": "HLC_MAP_APPLICABILITY_PASS" if not reasons else "HLC_MAP_APPLICABILITY_FAIL",
        "pass": not reasons,
        "reasons": reasons,
        "no_extrapolation": True,
        "new_numeric_geometry_threshold_used": False,
    }


def hlc_realized_primary_measurement(
    states: Sequence[Mapping[str, Any]],
    source_reference_xy: Sequence[Sequence[float]],
    target_reference_xy: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Measure HLC mechanism/endpoint/engineering from actual ego states."""
    from shapely.geometry import LineString, Point

    time, xy, heading, speed = trajectory_arrays(states)
    source = np.asarray(source_reference_xy, dtype=np.float64)
    target = np.asarray(target_reference_xy, dtype=np.float64)
    source_line, target_line = LineString(source), LineString(target)
    source_arc = np.asarray([source_line.project(Point(p)) for p in xy], dtype=np.float64)
    target_arc = np.asarray([target_line.project(Point(p)) for p in xy], dtype=np.float64)
    source_xy, _ = sample_native_reference_no_extrapolation(source, source_arc)
    target_xy, target_heading = sample_native_reference_no_extrapolation(target, target_arc)
    lane_delta = target_xy - source_xy
    denominator = np.sum(lane_delta * lane_delta, axis=1)
    if np.any(denominator <= 1e-12):
        raise ValueError("HLC_NATIVE_LANE_SEPARATION_NOT_EVALUABLE")
    progress = np.sum((xy - source_xy) * lane_delta, axis=1) / denominator
    mechanism = calculate_hlc_option_b(time, progress, speed, map_valid=True)
    final_velocity = (xy[-1] - xy[-2]) / DT_SECONDS
    normal = np.asarray([-math.sin(target_heading[-1]), math.cos(target_heading[-1])])
    endpoint = {
        "terminal_lateral_offset_to_target_center_m": round(float(target_line.distance(Point(xy[-1]))), 6),
        "terminal_heading_error_rad": round(abs(wrap_angle(float(heading[-1] - target_heading[-1]))), 6),
        "terminal_lateral_velocity_mps": round(abs(float(np.dot(final_velocity, normal))), 6),
        "complete_target_lane_transition": bool(progress[-1] >= 0.75),
        "frozen_limits": {"offset_m_max": 0.25, "heading_error_rad_max": 0.05, "lateral_velocity_mps_max": 0.25},
    }
    velocity = np.gradient(xy, time, axis=0, edge_order=2)
    path_heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    yaw_rate = np.gradient(path_heading, time, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    tangent = velocity / np.maximum(np.linalg.norm(velocity, axis=1, keepdims=True), 1e-12)
    lateral_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    lateral_accel = np.sum(acceleration * lateral_normal, axis=1)
    curvature = yaw_rate / np.maximum(np.linalg.norm(velocity, axis=1), 1e-12)
    engineering = {
        "max_abs_lateral_accel_mps2": round(float(np.max(np.abs(lateral_accel))), 6),
        "max_abs_yaw_rate_radps": round(float(np.max(np.abs(yaw_rate))), 6),
        "max_abs_curvature_inv_m": round(float(np.max(np.abs(curvature))), 6),
        "frozen_limits": {"lateral_accel_mps2_max": 6.0, "yaw_rate_radps_max": 1.0, "curvature_inv_m_max": 0.5},
    }
    return {"measurement_source": "REALIZED_CLOSED_LOOP_EGO_TRAJECTORY", "mechanism": mechanism, "endpoint": endpoint, "engineering": engineering, "progress": progress.tolist()}


def tsb_minimum_initial_speed_evidence(step_mps: float = 0.001) -> Dict[str, Any]:
    """Analytical + exhaustive synthetic evidence, independent of B2.1 speeds."""
    time = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    # Baseline has ten discrete -1.0 m/s2 integration intervals: total loss 1.0.
    analytical = 1.0 + 1.0
    grid = np.arange(0.0, 4.0 + step_mps / 2.0, step_mps)

    def profile(v0: float, arm: str) -> np.ndarray:
        speed = np.empty(WINDOW_FRAMES, dtype=np.float64)
        speed[0] = v0
        for i in range(1, WINDOW_FRAMES):
            speed[i] = max(0.2, round(float(speed[i - 1] + frozen_tsb_acceleration(time[i - 1], arm) * DT_SECONDS), 12))
        return speed

    legal = []
    for v0 in grid:
        statuses = [calculate_tsb_option_a(time, profile(float(v0), arm))["status"] for arm in (TSB_BASELINE, TSB_TREATMENT)]
        if statuses == ["OK", "OK"]:
            legal.append(float(v0))
    synthetic = min(legal) if legal else None
    return {
        "analytical_floor_mps": analytical,
        "synthetic_grid_step_mps": step_mps,
        "synthetic_first_jointly_evaluable_mps": synthetic,
        "match": synthetic is not None and abs(synthetic - analytical) <= step_mps / 2,
        "proposed_initial_speed_floor_mps": analytical,
        "status": "PROPOSED_REQUIRES_OWNER_APPROVAL",
        "b2_1_initial_speeds_used": False,
    }


def evaluate_mechanism_pair(family: str, baseline_speed: Sequence[float], treatment_speed: Sequence[float], progress: Tuple[Sequence[float], Sequence[float]] | None = None) -> Dict[str, Any]:
    time = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    if family == "R-TSB":
        return qualify_tsb_pair(calculate_tsb_option_a(time, baseline_speed), calculate_tsb_option_a(time, treatment_speed))
    if progress is None:
        raise ValueError("HLC progress pair required")
    return qualify_hlc_pair(
        calculate_hlc_option_b(time, progress[0], baseline_speed),
        calculate_hlc_option_b(time, progress[1], treatment_speed),
    )
