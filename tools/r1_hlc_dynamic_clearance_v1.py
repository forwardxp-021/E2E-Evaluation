#!/usr/bin/env python3
"""R1 HLC pre-treatment dynamic-clearance evaluator v1.0.

Only original replay tracks and the arm-independent common envelope are read.
The evaluator has no planner-outcome, mechanism, F_match, safety,
representation, BDD, probe, checkpoint, or RBR dependency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class R1HLCDynamicClearanceConfig:
    horizon_seconds: float = 8.0
    nominal_query_dt_seconds: float = 0.1
    maximum_actor_interpolation_gap_seconds: float = 0.25
    longitudinal_buffer_m: float = 3.0
    lateral_buffer_m: float = 0.5


def _validate_config(config: R1HLCDynamicClearanceConfig) -> None:
    expected = R1HLCDynamicClearanceConfig()
    if config != expected:
        raise ValueError("R1 HLC dynamic-clearance v1.0 numerics are frozen and may not be overridden")


def _footprint(value: Mapping[str, Any] | None, label: str) -> Tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"NOT_ELIGIBLE: {label} official footprint is missing; fallback is forbidden")
    length = float(value.get("length_m", math.nan))
    width = float(value.get("width_m", math.nan))
    if not math.isfinite(length) or not math.isfinite(width) or length <= 0 or width <= 0:
        raise ValueError(f"NOT_ELIGIBLE: {label} official footprint dimensions are invalid")
    return length, width


def _axis_half_extent(length: float, width: float, yaw: float, axis_heading: float) -> float:
    relative = yaw - axis_heading
    return 0.5 * (abs(length * math.cos(relative)) + abs(width * math.sin(relative)))


def _interpolate_actor(
    times_s: np.ndarray, states: np.ndarray, query_s: float, maximum_gap_s: float
) -> np.ndarray | None:
    if times_s.ndim != 1 or states.ndim != 2 or states.shape != (len(times_s), 5) or not len(times_s):
        raise ValueError("NOT_ELIGIBLE: actor track must be aligned [time, x/y/length/width/yaw]")
    if not np.isfinite(times_s).all() or not np.isfinite(states).all() or np.any(np.diff(times_s) <= 0):
        raise ValueError("NOT_ELIGIBLE: actor track time/state must be finite and strictly increasing")
    if np.any(states[:, 2:4] <= 0):
        raise ValueError("NOT_ELIGIBLE: OFFICIAL_TRACK_DIMENSIONS_REQUIRED")
    index = int(np.searchsorted(times_s, query_s, side="left"))
    if index < len(times_s) and abs(float(times_s[index]) - query_s) <= 1e-12:
        return states[index].copy()
    if index == 0 or index == len(times_s):
        return None
    left, right = index - 1, index
    gap = float(times_s[right] - times_s[left])
    if gap > maximum_gap_s:
        raise ValueError("NOT_ELIGIBLE: actor interpolation gap exceeds frozen 0.25s")
    fraction = (query_s - float(times_s[left])) / gap
    output = states[left] * (1.0 - fraction) + states[right] * fraction
    yaw_delta = (states[right, 4] - states[left, 4] + math.pi) % (2.0 * math.pi) - math.pi
    output[4] = states[left, 4] + fraction * yaw_delta
    return output


def _path_heading(path: np.ndarray, index: int) -> float:
    left = max(index - 1, 0)
    right = min(index + 1, len(path) - 1)
    delta = path[right] - path[left]
    if np.linalg.norm(delta) <= 0:
        raise ValueError("NOT_ELIGIBLE: common-envelope tangent is degenerate")
    return math.atan2(float(delta[1]), float(delta[0]))


def _common_envelope_conflict(
    baseline_xy: np.ndarray,
    treatment_xy: np.ndarray,
    index: int,
    actor: np.ndarray,
    ego_length: float,
    ego_width: float,
    config: R1HLCDynamicClearanceConfig,
) -> bool:
    centers = np.asarray([baseline_xy[index], treatment_xy[index]], dtype=np.float64)
    centerline = 0.5 * (baseline_xy + treatment_xy)
    heading = _path_heading(centerline, index)
    tangent = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    origin = centers[0]
    center_long = (centers - origin) @ tangent
    center_lat = (centers - origin) @ normal
    agent_long = float((actor[:2] - origin) @ tangent)
    agent_lat = float((actor[:2] - origin) @ normal)
    actor_long_extent = _axis_half_extent(float(actor[2]), float(actor[3]), float(actor[4]), heading)
    actor_lat_extent = _axis_half_extent(float(actor[2]), float(actor[3]), float(actor[4]), heading + math.pi / 2.0)
    long_limit = ego_length / 2.0 + actor_long_extent + config.longitudinal_buffer_m
    lat_limit = ego_width / 2.0 + actor_lat_extent + config.lateral_buffer_m
    long_gap = max(float(np.min(center_long) - agent_long), float(agent_long - np.max(center_long)), 0.0)
    lat_gap = max(float(np.min(center_lat) - agent_lat), float(agent_lat - np.max(center_lat)), 0.0)
    return long_gap <= long_limit and lat_gap <= lat_limit


def evaluate_r1_hlc_dynamic_clearance_v1(
    *,
    baseline_xy: Sequence[Sequence[float]],
    treatment_xy: Sequence[Sequence[float]],
    official_runtime_vehicle_parameters: Mapping[str, Any] | None,
    original_replay_tracks: Mapping[str, Mapping[str, Any]],
    config: R1HLCDynamicClearanceConfig = R1HLCDynamicClearanceConfig(),
) -> Dict[str, Any]:
    """Evaluate one frozen, arm-independent 80-frame common envelope."""
    _validate_config(config)
    ego_length, ego_width = _footprint(official_runtime_vehicle_parameters, "ego")
    baseline = np.asarray(baseline_xy, dtype=np.float64)
    treatment = np.asarray(treatment_xy, dtype=np.float64)
    if baseline.shape != (80, 2) or treatment.shape != (80, 2) or not np.isfinite(baseline).all() or not np.isfinite(treatment).all():
        raise ValueError("NOT_ELIGIBLE: both HLC arms require finite [80,2] native geometry")
    query = np.arange(80, dtype=np.float64) * config.nominal_query_dt_seconds
    first_conflict = None
    evaluated = 0
    for token, track in sorted(original_replay_tracks.items()):
        times = np.asarray(track.get("time_s"), dtype=np.float64)
        states = np.asarray(track.get("states"), dtype=np.float64)
        for index, now in enumerate(query):
            actor = _interpolate_actor(times, states, float(now), config.maximum_actor_interpolation_gap_seconds)
            if actor is None:
                continue
            evaluated += 1
            if _common_envelope_conflict(baseline, treatment, index, actor, ego_length, ego_width, config):
                first_conflict = {"track_id": str(token), "iteration_index": index, "nominal_time_s": float(now)}
                break
        if first_conflict is not None:
            break
    result = {
        "status": "DYNAMIC_CLEARANCE_PASS" if first_conflict is None else "DYNAMIC_CLEARANCE_FAIL",
        "eligible": True,
        "pass": first_conflict is None,
        "first_conflict": first_conflict,
        "evaluated_actor_states": evaluated,
        "config": asdict(config),
        "ego_footprint_source": "OFFICIAL_RUNTIME_VEHICLE_PARAMETERS_REQUIRED",
        "actor_footprint_source": "OFFICIAL_TRACK_DIMENSIONS_REQUIRED",
        "common_envelope": "HLC_BASELINE_PLUS_HLC_OPTION_B_TREATMENT",
        "arm_specific_envelope_forbidden": True,
        "map_extrapolation": "FORBIDDEN",
        "actor_extrapolation": "FORBIDDEN",
        "evidence_scope": "PRETREATMENT_ORIGINAL_REPLAY_TRACKS_ONLY",
    }
    return result


__all__ = ["R1HLCDynamicClearanceConfig", "evaluate_r1_hlc_dynamic_clearance_v1"]
