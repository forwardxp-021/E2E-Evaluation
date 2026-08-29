#!/usr/bin/env python3
"""Prospective official-map R1 closed-loop context adapter v2.

The adapter accepts normalized official observations plus an official map
query implementation.  It never substitutes hard-coded ABSENT or
NONE_OBSERVED states: missingness is the result of lane-aware queries over the
ten condition-identical warmup iterations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

import numpy as np

from tools.r1_context_mechanism_core import build_canonical_context_record


SLOT_NAMES = ("front", "left_front", "left_rear", "right_front", "right_rear")
DYNAMIC_TYPES = {"VEHICLE"}


class OfficialMapQuery(Protocol):
    """Runtime bridge that must be backed by the official nuPlan map API."""

    def lane_context(self, ego_xy: Tuple[float, float], route_roadblock_ids: Sequence[str]) -> Mapping[str, Any]: ...
    def project(self, lane_id: str, xy: Tuple[float, float]) -> Mapping[str, Any]: ...
    def lane_for_actor(self, actor: Mapping[str, Any]) -> str | None: ...
    def static_stop_control_ahead(self, ego_xy: Tuple[float, float], route_roadblock_ids: Sequence[str]) -> bool: ...


def _actor_id(actor: Mapping[str, Any]) -> str:
    value = actor.get("track_id") or actor.get("track_token") or actor.get("token")
    if value in (None, ""):
        raise ValueError("official tracked actor is missing stable track ID")
    return str(value)


def _actor_xy(actor: Mapping[str, Any]) -> Tuple[float, float]:
    if "x" in actor and "y" in actor:
        return float(actor["x"]), float(actor["y"])
    center = actor.get("center", {})
    return float(center["x"]), float(center["y"])


def _actor_velocity(actor: Mapping[str, Any]) -> Tuple[float, float]:
    velocity = actor.get("velocity", {})
    return float(actor.get("vx", velocity.get("x", 0.0))), float(actor.get("vy", velocity.get("y", 0.0)))


def normalize_trace_observation(raw: Sequence[Mapping[str, Any]]) -> Sequence[Dict[str, Any]]:
    """Normalize one read-only B2.1 raw observation without assigning lanes."""
    result = []
    for actor in raw:
        actor_type = str(actor.get("type", "UNKNOWN")).upper()
        if actor_type not in DYNAMIC_TYPES:
            continue
        x, y = _actor_xy(actor)
        vx, vy = _actor_velocity(actor)
        result.append({"track_id": _actor_id(actor), "type": actor_type, "x": x, "y": y, "vx": vx, "vy": vy})
    return result


def _slot_record(actor: Mapping[str, Any], gap: float, tangent: np.ndarray, ego_speed: float) -> Dict[str, Any]:
    vx, vy = _actor_velocity(actor)
    lead_speed = float(vx * tangent[0] + vy * tangent[1])
    relative = lead_speed - ego_speed
    return {
        "valid": True,
        "track_id": _actor_id(actor),
        "arc_gap_m": abs(float(gap)),
        "lead_relative_speed_mps": relative,
        "thw_s": abs(float(gap)) / max(float(ego_speed), 1e-12),
    }


def _frame_semantics(
    frame: Mapping[str, Any], family: str, direction: str | None, route_ids: Sequence[str], map_query: OfficialMapQuery
) -> Dict[str, Any]:
    ego = frame["ego"]
    ego_xy = (float(ego["x"]), float(ego["y"]))
    lane = dict(map_query.lane_context(ego_xy, route_ids))
    required = ("valid", "current_lane_id", "left_lane_id", "right_lane_id", "tangent", "road_class")
    if any(key not in lane for key in required) or not bool(lane["valid"]):
        raise ValueError("official map query did not resolve current/adjacent lane context")
    current = str(lane["current_lane_id"])
    left = None if lane["left_lane_id"] in (None, "") else str(lane["left_lane_id"])
    right = None if lane["right_lane_id"] in (None, "") else str(lane["right_lane_id"])
    tangent = np.asarray(lane["tangent"], dtype=np.float64)
    tangent /= np.linalg.norm(tangent)
    ego_projection = map_query.project(current, ego_xy)
    ego_arc = float(ego_projection["arc_m"])
    candidates: Dict[str, list[Tuple[float, Mapping[str, Any], np.ndarray]]] = {name: [] for name in SLOT_NAMES}
    density_ids = set()
    actors = [a for a in frame.get("actors", []) if str(a.get("type", "VEHICLE")).upper() in DYNAMIC_TYPES]
    lane_by_slot = {"front": current, "left_front": left, "left_rear": left, "right_front": right, "right_rear": right}
    for actor in actors:
        actor_lane = map_query.lane_for_actor(actor)
        if actor_lane is None:
            continue
        actor_lane = str(actor_lane)
        if actor_lane not in {value for value in (current, left, right) if value is not None}:
            continue
        actor_projection = map_query.project(actor_lane, _actor_xy(actor))
        if actor_lane == current:
            comparison_ego = ego_arc
            slot = "front" if float(actor_projection["arc_m"]) > comparison_ego else None
        else:
            ego_on_actor_lane = map_query.project(actor_lane, ego_xy)
            gap_signed = float(actor_projection["arc_m"]) - float(ego_on_actor_lane["arc_m"])
            side = "left" if actor_lane == left else "right"
            slot = f"{side}_{'front' if gap_signed > 0 else 'rear'}"
        if slot is None:
            continue
        ego_on_lane = map_query.project(actor_lane, ego_xy)
        gap = float(actor_projection["arc_m"]) - float(ego_on_lane["arc_m"])
        actor_tangent = np.asarray(actor_projection.get("tangent", tangent), dtype=np.float64)
        actor_tangent /= np.linalg.norm(actor_tangent)
        candidates[slot].append((abs(gap), actor, actor_tangent))
        if abs(gap) <= 50.0:
            density_ids.add(_actor_id(actor))
    slots: Dict[str, Dict[str, Any]] = {}
    for name in SLOT_NAMES:
        if not candidates[name] or lane_by_slot[name] is None:
            slots[name] = {"valid": False}
        else:
            gap_abs, actor, actor_tangent = min(candidates[name], key=lambda x: x[0])
            sign = -1.0 if name.endswith("rear") else 1.0
            slots[name] = _slot_record(actor, sign * gap_abs, actor_tangent, float(ego["speed_mps"]))
    output: Dict[str, Any] = {
        "time_s": float(frame["time_s"]), "ego_valid": True, "map_valid": True, "current_required_lane_valid": True,
        "speed_mps": float(ego["speed_mps"]), "lane_offset_m": float(ego_projection.get("lateral_offset_m", 0.0)),
        "legal_projected_dynamic_vehicle_count": len(density_ids), "slots": slots, "road_class": str(lane["road_class"]),
    }
    if family == "R-HLC":
        target_slot_prefix = "left" if str(direction).upper() == "LEFT" else "right"
        output["target_front"] = slots[f"{target_slot_prefix}_front"]
        output["target_rear"] = slots[f"{target_slot_prefix}_rear"]
    else:
        output["front"] = slots["front"]
    return output


def build_closed_loop_context_v2(
    *,
    family: str,
    scenario_token: str,
    map_version: str,
    route_fingerprint: str,
    initial_state_fingerprint: str,
    log_id: str,
    route_roadblock_ids: Sequence[str],
    frames: Sequence[Mapping[str, Any]],
    map_query: OfficialMapQuery,
    intended_lane_change_direction: str | None = None,
) -> Dict[str, Any]:
    """Build the frozen canonical semantics from exact warmup iterations 0..9."""
    if family not in {"R-HLC", "R-TSB"}:
        raise ValueError("family must be R-HLC or R-TSB")
    if len(frames) != 10 or [int(x.get("iteration_index", -1)) for x in frames] != list(range(10)):
        raise ValueError("NOT_EVALUABLE_TEMPORAL_GRID: warmup iterations must be exactly 0..9")
    physical_times = [int(x.get("time_us", -1)) for x in frames]
    if any(physical_times[i + 1] - physical_times[i] != 100_000 for i in range(9)):
        raise ValueError("NOT_EVALUABLE_TEMPORAL_GRID: warmup physical timestamps must be exact dt=.1")
    if [round(float(x["time_s"]), 6) for x in frames] != [round(i * 0.1, 6) for i in range(10)]:
        raise ValueError("NOT_EVALUABLE_TEMPORAL_GRID: warmup must be exact iterations 0..9 at dt=.1")
    if family == "R-HLC" and intended_lane_change_direction not in {"LEFT", "RIGHT"}:
        raise ValueError("HLC direction must be LEFT or RIGHT")
    built = [_frame_semantics(x, family, intended_lane_change_direction, route_roadblock_ids, map_query) for x in frames]
    hazards = []
    if family == "R-TSB":
        if any(any(str(t.get("status", "")).upper() in {"RED", "YELLOW"} and bool(t.get("route_relevant", False)) for t in x.get("traffic_lights", [])) for x in frames):
            hazards.append("ROUTE_SIGNAL_RED_OR_YELLOW")
        if any(map_query.static_stop_control_ahead((float(x["ego"]["x"]), float(x["ego"]["y"])), route_roadblock_ids) for x in frames):
            hazards.append("STATIC_STOP_CONTROL_AHEAD")
        if any(x["front"].get("valid") and float(x["front"].get("lead_relative_speed_mps", 0.0)) < 0.0 for x in built):
            hazards.append("OBSERVED_SLOW_LEAD")
        if not hazards:
            hazards = ["NONE_OBSERVED"]
    payload: Dict[str, Any] = {
        "family": family, "scenario_token": scenario_token, "map_version": map_version,
        "route_fingerprint": route_fingerprint, "initial_state_fingerprint": initial_state_fingerprint,
        "map_location": map_version, "road_class": built[0]["road_class"], "log_id": log_id,
        "query_version": "r1_closed_loop_context_adapter_v2", "history_source": "CONDITION_IDENTICAL_1S_WARMUP",
        "t_anchor_s": 1.0, "frames": built,
        "map_source_ids": {"route_roadblock_ids": [str(x) for x in route_roadblock_ids]},
    }
    if family == "R-HLC":
        payload["intended_lane_change_direction"] = intended_lane_change_direction
    else:
        payload["hazard_multi_hot"] = hazards
    canonical = build_canonical_context_record(payload)
    canonical["prospective_adapter_version"] = "v2"
    canonical["hardcoded_absence_forbidden"] = True
    return canonical
