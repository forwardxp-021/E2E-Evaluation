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
        result.append({
            "track_id": _actor_id(actor), "type": actor_type, "x": x, "y": y,
            "vx": vx, "vy": vy, "heading": _finite(heading_value, "actor.heading"),
        })
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
    candidates: Dict[str, list[Tuple[float, float, float, float, str, Mapping[str, Any], np.ndarray]]] = {name: [] for name in SLOT_NAMES}
    density_ids: set[str] = set()
    lane_by_slot = {"front": current, "left_front": left, "left_rear": left, "right_front": right, "right_rear": right}
    for actor in normalize_official_actors_v2_1(frame.get("actors", [])):
        actor_lane_raw = map_query.lane_for_actor(actor)
        if actor_lane_raw in (None, ""):
            continue
        actor_lane = str(actor_lane_raw)
        actor_projection = map_query.project(actor_lane, _actor_xy(actor))
        ego_on_actor_lane = map_query.project(actor_lane, ego_xy)
        actor_arc = _finite(actor_projection.get("arc_m"), "actor_projection.arc_m")
        ego_arc = _finite(ego_on_actor_lane.get("arc_m"), "ego_projection.arc_m")
        ds = actor_arc - ego_arc
        lateral = abs(_finite(actor_projection.get("lateral_offset_m"), "actor_projection.lateral_offset_m"))
        actor_heading = _finite(actor["heading"], "actor.heading")
        lane_heading = _finite(actor_projection.get("heading", math.atan2(tangent[1], tangent[0])), "actor_projection.heading")
        heading_diff = abs(_wrap(actor_heading - lane_heading))
        projection_distance = abs(_finite(actor_projection.get("distance_to_lane_m", lateral), "actor_projection.distance_to_lane_m"))
        if actor_lane in density_lanes and abs(ds) <= FROZEN_DENSITY_ROUTE_DISTANCE_M:
            density_ids.add(_actor_id(actor))
        slots = []
        if actor_lane == current and ds > 0:
            slots.append("front")
        if left is not None and actor_lane == left:
            slots.append("left_front" if ds > 0 else "left_rear" if ds < 0 else "")
        if right is not None and actor_lane == right:
            slots.append("right_front" if ds > 0 else "right_rear" if ds < 0 else "")
        for slot in (value for value in slots if value):
            if abs(ds) > STAGE5D_SLOT_LIMITS_M[slot] or lateral > STAGE5D_LATERAL_TOLERANCE_M or heading_diff > STAGE5D_HEADING_TOLERANCE_RAD:
                continue
            actor_tangent = np.asarray(actor_projection.get("tangent", tangent), dtype=np.float64)
            if actor_tangent.shape != (2,) or not np.isfinite(actor_tangent).all() or np.linalg.norm(actor_tangent) <= 0:
                raise ValueError("actor lane tangent is invalid")
            actor_tangent /= np.linalg.norm(actor_tangent)
            candidates[slot].append((abs(ds), projection_distance, lateral, heading_diff, _actor_id(actor), actor, actor_tangent))
    selected: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()
    for slot in SLOT_NAMES:
        rows = [row for row in candidates[slot] if row[4] not in used]
        if not rows or lane_by_slot[slot] is None:
            selected[slot] = {"valid": False}
            continue
        row = sorted(rows, key=lambda value: (value[0], value[1], value[2], value[3], value[4]))[0]
        gap, _, _, _, actor_id, actor, actor_tangent = row
        used.add(actor_id)
        vx, vy = _actor_velocity(actor)
        lead_speed = float(vx * actor_tangent[0] + vy * actor_tangent[1])
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


def context_source_conformance_matrix_v2_1() -> Sequence[Mapping[str, str]]:
    return [
        {"field": "slot assignment/order", "frozen_source": "tools.stage5d_context_core.assign_stage5d_slots", "v2_1": "strict lane-ID/arc/tie-break equivalent; geometric fallback forbidden", "status": "CONFORMANT"},
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
        "stage5d_slot_semantics": "STRICT_EQUIVALENT_NO_GEOMETRIC_FALLBACK",
        "context_source_conformance_matrix": list(context_source_conformance_matrix_v2_1()),
    })
    return canonical


__all__ = [
    "build_closed_loop_context_v2_1",
    "context_source_conformance_matrix_v2_1",
    "normalize_official_actors_v2_1",
]
