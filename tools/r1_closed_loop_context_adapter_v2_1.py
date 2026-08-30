#!/usr/bin/env python3
"""Final prospective R1 closed-loop context adapter v2.1.

The adapter uses iteration index as the canonical ordering coordinate and
preserves every official timestamp as audit data.  It is a fail-closed,
official-map implementation of the frozen Stage5D lane-aware slot semantics;
it performs no rollout or candidate enumeration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

import numpy as np

from tools.r1_context_mechanism_core import build_canonical_context_record
from tools.stage5d_context_core import assign_stage5d_slots
from tools.waymo_lane_utils import LaneInfo


SLOT_NAMES = ("front", "left_front", "left_rear", "right_front", "right_rear")
DYNAMIC_TYPES = {"VEHICLE"}
STAGE5D_SLOT_LIMITS_M = {"front": 120.0, "left_front": 80.0, "left_rear": 120.0, "right_front": 80.0, "right_rear": 120.0}
STAGE5D_LATERAL_TOLERANCE_M = 2.0
STAGE5D_HEADING_TOLERANCE_RAD = math.radians(45.0)
FROZEN_DENSITY_ROUTE_DISTANCE_M = 50.0


class OfficialMapQueryV2_1(Protocol):
    def lane_context(self, ego_xy: Tuple[float, float], route_roadblock_ids: Sequence[str]) -> Mapping[str, Any]: ...
    def project(self, lane_id: str, xy: Tuple[float, float]) -> Mapping[str, Any]: ...
    def lane_for_actor(self, actor: Mapping[str, Any]) -> str | None: ...
    def static_stop_control_ahead(self, ego_xy: Tuple[float, float], route_roadblock_ids: Sequence[str]) -> bool: ...


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _actor_id(actor: Mapping[str, Any]) -> str:
    value = actor.get("track_id") or actor.get("track_token") or actor.get("token")
    if value in (None, ""):
        raise ValueError("official tracked actor is missing stable track ID")
    return str(value)


def _actor_xy(actor: Mapping[str, Any]) -> Tuple[float, float]:
    if "x" in actor and "y" in actor:
        return _finite(actor["x"], "actor.x"), _finite(actor["y"], "actor.y")
    center = actor.get("center")
    if not isinstance(center, Mapping):
        raise ValueError("official tracked actor is missing position")
    return _finite(center.get("x"), "actor.center.x"), _finite(center.get("y"), "actor.center.y")


def _actor_velocity(actor: Mapping[str, Any]) -> Tuple[float, float]:
    velocity = actor.get("velocity")
    vx = actor.get("vx") if "vx" in actor else velocity.get("x") if isinstance(velocity, Mapping) else None
    vy = actor.get("vy") if "vy" in actor else velocity.get("y") if isinstance(velocity, Mapping) else None
    if vx is None or vy is None:
        raise ValueError("CONTEXT_VELOCITY_FAIL_CLOSED: actor velocity is missing")
    return _finite(vx, "actor.vx"), _finite(vy, "actor.vy")


def normalize_official_actors_v2_1(raw: Sequence[Mapping[str, Any]]) -> Sequence[Dict[str, Any]]:
    result = []
    for actor in raw:
        actor_type = str(actor.get("type", "UNKNOWN")).upper()
        if actor_type not in DYNAMIC_TYPES:
            continue
        x, y = _actor_xy(actor)
        vx, vy = _actor_velocity(actor)
        heading_value = actor.get("heading")
        if heading_value is None:
            if math.hypot(vx, vy) <= 0.0:
                raise ValueError("CONTEXT_HEADING_FAIL_CLOSED: stationary actor heading is missing")
            heading_value = math.atan2(vy, vx)
        normalized = {
            "track_id": _actor_id(actor), "type": actor_type, "x": x, "y": y,
            "vx": vx, "vy": vy, "heading": _finite(heading_value, "actor.heading"),
        }
        if actor.get("lane_id") not in (None, ""):
            normalized["lane_id"] = str(actor["lane_id"])
        result.append(normalized)
    return result


def _wrap(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _lane_ids(lane: Mapping[str, Any], family: str, direction: str | None) -> Tuple[str, str | None, str | None, str | None, set[str]]:
    required = ("valid", "current_lane_id", "left_lane_id", "right_lane_id", "tangent", "road_class")
    if any(key not in lane for key in required) or not bool(lane["valid"]):
        raise ValueError("official map query did not resolve current/adjacent lane context")
    current = str(lane["current_lane_id"])
    left = None if lane["left_lane_id"] in (None, "") else str(lane["left_lane_id"])
    right = None if lane["right_lane_id"] in (None, "") else str(lane["right_lane_id"])
    target = None
    if family == "R-HLC":
        target = left if direction == "LEFT" else right
        if target is None:
            raise ValueError("HLC target native adjacent lane is unavailable")
        source_adjacent = lane.get("source_immediate_adjacent_lane_ids")
        target_adjacent = lane.get("target_immediate_adjacent_lane_ids")
        if not isinstance(source_adjacent, Sequence) or isinstance(source_adjacent, (str, bytes)):
            raise ValueError("HLC source immediate native adjacent corridor is missing")
        if not isinstance(target_adjacent, Sequence) or isinstance(target_adjacent, (str, bytes)):
            raise ValueError("HLC target immediate native adjacent corridor is missing")
        density = {current, target, *(str(value) for value in source_adjacent), *(str(value) for value in target_adjacent)}
    else:
        adjacent = lane.get("current_immediate_adjacent_lane_ids")
        if not isinstance(adjacent, Sequence) or isinstance(adjacent, (str, bytes)):
            raise ValueError("TSB current immediate native adjacent corridor is missing")
        density = {current, *(str(value) for value in adjacent)}
    return current, left, right, target, {value for value in density if value}


def _strict_stage5d_equivalent_slots(
    frame: Mapping[str, Any], lane: Mapping[str, Any], family: str, direction: str | None, map_query: OfficialMapQueryV2_1
) -> Tuple[Dict[str, Dict[str, Any]], int, Mapping[str, Any]]:
    ego = frame["ego"]
    ego_xy = (_finite(ego.get("x"), "ego.x"), _finite(ego.get("y"), "ego.y"))
    ego_speed = _finite(ego.get("speed_mps"), "ego.speed_mps")
    ego_heading = _finite(ego.get("heading"), "ego.heading")
    current, left, right, _, density_lanes = _lane_ids(lane, family, direction)
    tangent = np.asarray(lane["tangent"], dtype=np.float64)
    if tangent.shape != (2,) or not np.isfinite(tangent).all() or np.linalg.norm(tangent) <= 0:
        raise ValueError("official current-lane tangent is invalid")
    tangent /= np.linalg.norm(tangent)
    ego_current_projection = map_query.project(current, ego_xy)
    density_ids: set[str] = set()
    actors = {str(_actor_id(actor)): actor for actor in normalize_official_actors_v2_1(frame.get("actors", []))}
    candidate_states: Dict[str, Dict[str, Any]] = {}
    candidate_projections: Dict[str, Dict[str, Any]] = {}
    lane_ids = [value for value in (current, left, right) if value]
    official_ego_by_lane = {lane_id: dict(map_query.project(lane_id, ego_xy)) for lane_id in lane_ids}

    def synthetic_official_lane_info(lane_id: str) -> LaneInfo:
        projection = official_ego_by_lane[lane_id]
        tangent_value = np.asarray(projection.get("tangent", tangent), dtype=np.float64)
        tangent_value /= np.linalg.norm(tangent_value)
        normal = np.asarray([-tangent_value[1], tangent_value[0]])
        center = np.asarray(ego_xy) - float(projection.get("lateral_offset_m", 0.0)) * normal
        points = np.asarray([center - 500.0 * tangent_value, center + 500.0 * tangent_value])
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        return LaneInfo(lane_id, points, np.asarray([math.atan2(tangent_value[1], tangent_value[0])]), lengths, np.r_[0.0, np.cumsum(lengths)], points[:-1], delta, np.sum(delta * delta, axis=1), points.min(axis=0), points.max(axis=0), points.mean(axis=0), left_neighbor_lane_ids=[left] if lane_id == current and left else [], right_neighbor_lane_ids=[right] if lane_id == current and right else [], topology_source="OFFICIAL_PRECOMPUTED_PROJECTION_ADAPTER")

    lane_infos = {lane_id: synthetic_official_lane_info(lane_id) for lane_id in lane_ids}
    for actor_id, actor in actors.items():
        actor_lane_raw = map_query.lane_for_actor(actor)
        if actor_lane_raw in (None, ""):
            continue
        actor_lane = str(actor_lane_raw)
        if actor_lane not in lane_ids:
            continue
        actor_projection = map_query.project(actor_lane, _actor_xy(actor))
        ego_on_actor_lane = official_ego_by_lane[actor_lane]
        actor_arc = _finite(actor_projection.get("arc_m"), "actor_projection.arc_m")
        ego_arc = _finite(ego_on_actor_lane.get("arc_m"), "ego_projection.arc_m")
        ds = actor_arc - ego_arc
        if actor_lane in density_lanes and abs(ds) <= FROZEN_DENSITY_ROUTE_DISTANCE_M:
            density_ids.add(actor_id)
        vx, vy = _actor_velocity(actor)
        candidate_states[actor_id] = {"x": float(actor["x"]), "y": float(actor["y"]), "heading": float(actor["heading"]), "velocity_x": vx, "velocity_y": vy, "speed": math.hypot(vx, vy)}
        candidate_projections[actor_id] = {"lane_id": actor_lane, "s": 500.0 + ds, "l": float(actor_projection.get("lateral_offset_m", actor_projection.get("l", math.nan))), "distance_to_lane": float(actor_projection.get("distance_to_lane_m", actor_projection.get("distance_to_lane", math.nan))), "heading": float(actor_projection.get("heading", 0.0)), "segment_index": int(actor_projection.get("segment_index", 0))}

    ego_state = {"x": ego_xy[0], "y": ego_xy[1], "heading": ego_heading, "speed": ego_speed}
    ego_projection = {"lane_id": current, "s": 500.0, "l": float(ego_current_projection.get("lateral_offset_m", 0.0)), "distance_to_lane": float(ego_current_projection.get("distance_to_lane_m", 0.0)), "heading": float(ego_current_projection.get("heading", math.atan2(tangent[1], tangent[0]))), "segment_index": int(ego_current_projection.get("segment_index", 0))}
    assignment = assign_stage5d_slots(ego_state, candidate_states, lane_infos=lane_infos, assignment_mode="lane_aware_only", config={"lane_lateral_tolerance": STAGE5D_LATERAL_TOLERANCE_M, "slot_heading_diff_deg": 45.0, "front_max_distance": 120.0, "side_front_max_distance": 80.0, "side_rear_max_distance": 120.0, "ego_projection_precomputed": True, "candidate_projections_complete": True, "allow_geometric_adjacent_lane_inference": False}, ego_projection=ego_projection, candidate_projections=candidate_projections)
    if not assignment.lane_assignment_available or assignment.fallback_assignment_used:
        raise ValueError("STAGE5D_LANE_AWARE_ONLY_ASSIGNMENT_UNAVAILABLE")
    debug_by_slot = {row["slot_name"]: row for row in assignment.per_slot_debug}
    selected: Dict[str, Dict[str, Any]] = {}
    for slot in SLOT_NAMES:
        actor_id = assignment.slot_to_agent.get(slot)
        if actor_id is None:
            selected[slot] = {"valid": False}
            continue
        actor = actors[actor_id]
        projection = map_query.project(str(map_query.lane_for_actor(actor)), _actor_xy(actor))
        actor_tangent = np.asarray(projection.get("tangent", tangent), dtype=np.float64)
        actor_tangent /= np.linalg.norm(actor_tangent)
        vx, vy = _actor_velocity(actor)
        lead_speed = float(vx * actor_tangent[0] + vy * actor_tangent[1])
        gap = abs(float(debug_by_slot[slot]["delta_s"]))
        selected[slot] = {
            "valid": True,
            "track_id": actor_id,
            "arc_gap_m": float(gap),
            "lead_relative_speed_mps": lead_speed - ego_speed,
            "thw_s": float(gap) / max(ego_speed, 1e-12),
        }
    return selected, len(density_ids), ego_current_projection


def _frame_semantics(
    frame: Mapping[str, Any], family: str, direction: str | None, route_ids: Sequence[str], map_query: OfficialMapQueryV2_1
) -> Dict[str, Any]:
    ego = frame["ego"]
    ego_xy = (_finite(ego.get("x"), "ego.x"), _finite(ego.get("y"), "ego.y"))
    lane = dict(map_query.lane_context(ego_xy, route_ids))
    slots, density, ego_projection = _strict_stage5d_equivalent_slots(frame, lane, family, direction, map_query)
    output: Dict[str, Any] = {
        "time_s": round(int(frame["iteration_index"]) * 0.1, 6),
        "simulation_iteration_index": int(frame["iteration_index"]),
        "actual_physical_time_us": int(frame["time_us"]),
        "ego_valid": True,
        "map_valid": True,
        "current_required_lane_valid": True,
        "speed_mps": _finite(ego.get("speed_mps"), "ego.speed_mps"),
        "lane_offset_m": _finite(ego_projection.get("lateral_offset_m"), "ego_projection.lateral_offset_m"),
        "legal_projected_dynamic_vehicle_count": density,
        "slots": slots,
    }
    if family == "R-HLC":
        prefix = "left" if direction == "LEFT" else "right"
        output["target_front"] = slots[f"{prefix}_front"]
        output["target_rear"] = slots[f"{prefix}_rear"]
    else:
        output["front"] = slots["front"]
    return output


def stage5d_slot_identity_v2_1(frame: Mapping[str, Any], family: str, direction: str | None, route_ids: Sequence[str], map_query: OfficialMapQueryV2_1) -> Mapping[str, str]:
    """Expose adapter slot identity for the B2.5 authoritative parity audit."""
    semantics = _frame_semantics(frame, family, direction, route_ids, map_query)
    return {slot: str(value["track_id"]) if bool(value.get("valid", False)) else "" for slot, value in semantics["slots"].items()}


def context_source_conformance_matrix_v2_1() -> Sequence[Mapping[str, str]]:
    return [
        {"field": "slot assignment/order", "frozen_source": "tools.stage5d_context_core.assign_stage5d_slots", "v2_1": "direct authoritative call with official precomputed projections; lane_aware_only", "status": "EXACT_PARITY"},
        {"field": "velocity", "frozen_source": "official tracked-object velocity", "v2_1": "missing/nonfinite FAIL_CLOSED", "status": "CONFORMANT"},
        {"field": "traffic density", "frozen_source": "median legal projected dynamic vehicle count", "v2_1": "family corridor, |projected route distance|<=50m", "status": "CONFORMANT"},
        {"field": "stable presence", "frozen_source": ">=8/10 one stable track ID", "v2_1": "unchanged frozen canonicalizer", "status": "CONFORMANT"},
        {"field": "hazard priority", "frozen_source": "route signal > stop control > slow lead > none", "v2_1": "anchor signal/stop plus stable pre-context slow lead", "status": "CONFORMANT"},
        {"field": "temporal ordering", "frozen_source": "10 nominal samples", "v2_1": "iteration 0...9 plus preserved actual time_us", "status": "VERSIONED_CONFORMANT"},
    ]


def build_closed_loop_context_v2_1(
    *,
    family: str,
    scenario_token: str,
    map_version: str,
    route_fingerprint: str,
    initial_state_fingerprint: str,
    log_id: str,
    route_roadblock_ids: Sequence[str],
    frames: Sequence[Mapping[str, Any]],
    map_query: OfficialMapQueryV2_1,
    intended_lane_change_direction: str | None = None,
) -> Dict[str, Any]:
    """Build pre-context iterations 0...9 and anchor-frame iteration 10."""
    if family not in {"R-HLC", "R-TSB"}:
        raise ValueError("family must be R-HLC or R-TSB")
    if len(frames) != 11 or [int(frame.get("iteration_index", -1)) for frame in frames] != list(range(11)):
        raise ValueError("NOT_EVALUABLE_ITERATION_SEQUENCE: context requires iterations 0...10")
    physical = np.asarray([float(frame.get("time_us", math.nan)) for frame in frames], dtype=np.float64)
    if not np.isfinite(physical).all() or np.any(np.diff(physical) <= 0):
        raise ValueError("NOT_EVALUABLE_PHYSICAL_TIMESTAMPS: context timestamps must increase")
    if family == "R-HLC" and intended_lane_change_direction not in {"LEFT", "RIGHT"}:
        raise ValueError("HLC direction must be LEFT or RIGHT")
    pre = [_frame_semantics(frame, family, intended_lane_change_direction, route_roadblock_ids, map_query) for frame in frames[:10]]
    anchor = frames[10]
    anchor_ego = anchor["ego"]
    anchor_xy = (_finite(anchor_ego.get("x"), "anchor.ego.x"), _finite(anchor_ego.get("y"), "anchor.ego.y"))
    anchor_lane = dict(map_query.lane_context(anchor_xy, route_roadblock_ids))
    hazards = []
    if family == "R-TSB":
        if any(str(item.get("status", "")).upper() in {"RED", "YELLOW"} and bool(item.get("route_relevant", False)) for item in anchor.get("traffic_lights", [])):
            hazards.append("ROUTE_SIGNAL_RED_OR_YELLOW")
        if map_query.static_stop_control_ahead(anchor_xy, route_roadblock_ids):
            hazards.append("STATIC_STOP_CONTROL_AHEAD")
        valid_front = [frame["front"] for frame in pre if bool(frame["front"].get("valid", False))]
        ids = {str(item["track_id"]) for item in valid_front}
        if len(valid_front) >= 8 and len(ids) == 1 and float(np.median([float(item["lead_relative_speed_mps"]) for item in valid_front])) < 0.0:
            hazards.append("OBSERVED_SLOW_LEAD")
        if not hazards:
            hazards = ["NONE_OBSERVED"]
    payload: Dict[str, Any] = {
        "family": family,
        "scenario_token": scenario_token,
        "map_version": map_version,
        "route_fingerprint": route_fingerprint,
        "initial_state_fingerprint": initial_state_fingerprint,
        "map_location": map_version,
        "road_class": str(anchor_lane["road_class"]),
        "log_id": log_id,
        "query_version": "r1_closed_loop_context_adapter_v2.1",
        "history_source": "CONDITION_IDENTICAL_1S_WARMUP",
        "t_anchor_s": 1.0,
        "frames": pre,
        "map_source_ids": {"route_roadblock_ids": [str(value) for value in route_roadblock_ids]},
    }
    if family == "R-HLC":
        payload["intended_lane_change_direction"] = intended_lane_change_direction
    else:
        payload["hazard_multi_hot"] = hazards
    canonical = build_canonical_context_record(payload)
    canonical.update({
        "prospective_adapter_version": "v2.1",
        "canonical_ordering_coordinate": "SIMULATION_ITERATION_INDEX",
        "pre_context_iteration_indices": list(range(10)),
        "anchor_frame_iteration_index": 10,
        "t_diverge_iteration_index": 11,
        "actual_pre_context_time_us": [int(value) for value in physical[:10]],
        "actual_anchor_time_us": int(physical[10]),
        "physical_timestamp_delta_exact_100000us_required": False,
        "interpolation_used": False,
        "extrapolation_used": False,
        "physical_timestamp_relabeling_used": False,
        "stage5d_slot_semantics": "AUTHORITATIVE_STAGE5D_EXACT_PARITY_LANE_AWARE_ONLY",
        "context_source_conformance_matrix": list(context_source_conformance_matrix_v2_1()),
    })
    return canonical


__all__ = [
    "build_closed_loop_context_v2_1",
    "context_source_conformance_matrix_v2_1",
    "normalize_official_actors_v2_1",
    "stage5d_slot_identity_v2_1",
]
