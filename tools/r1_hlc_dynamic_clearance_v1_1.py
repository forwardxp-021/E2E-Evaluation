#!/usr/bin/env python3
"""R1 HLC pre-treatment dynamic clearance v1.1 with horizon completeness."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class R1HLCDynamicClearanceConfigV1_1:
    horizon_seconds: float = 8.0
    nominal_query_dt_seconds: float = 0.1
    maximum_actor_interpolation_gap_seconds: float = 0.25
    longitudinal_buffer_m: float = 3.0
    lateral_buffer_m: float = 0.5


def _not_eligible(reason: str, config: R1HLCDynamicClearanceConfigV1_1) -> Dict[str, Any]:
    return {"status": "NOT_ELIGIBLE", "eligible": False, "pass": False, "reason": reason, "config": asdict(config)}


def _global_horizon(times: Sequence[float], config: R1HLCDynamicClearanceConfigV1_1) -> Dict[str, Any]:
    values = np.asarray(times, dtype=np.float64)
    valid = values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all() and np.all(np.diff(values) > 0)
    beginning = bool(valid and values[0] <= 0.0 + 1e-12)
    end = bool(valid and values[-1] >= 7.9 - 1e-12)
    maximum_gap = float(np.max(np.diff(values))) if valid else math.inf
    interior = bool(valid and maximum_gap <= config.maximum_actor_interpolation_gap_seconds + 1e-12)
    complete = bool(valid and beginning and end and interior)
    return {"status": "GLOBAL_OBSERVATION_HORIZON_COMPLETE" if complete else "GLOBAL_OBSERVATION_HORIZON_INCOMPLETE", "complete": complete, "beginning_coverage": beginning, "end_coverage": end, "interior_gap_coverage": interior, "maximum_observation_gap_seconds": maximum_gap if math.isfinite(maximum_gap) else None, "nominal_query_iterations": [0, 79], "nominal_query_window_seconds": "[0,8)"}


def _footprint(value: Mapping[str, Any] | None) -> Tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError("OFFICIAL_EGO_FOOTPRINT_MISSING")
    length, width = float(value.get("length_m", math.nan)), float(value.get("width_m", math.nan))
    if not math.isfinite(length) or not math.isfinite(width) or length <= 0 or width <= 0:
        raise ValueError("OFFICIAL_EGO_FOOTPRINT_MISSING")
    return length, width


def _heading(path: np.ndarray, index: int) -> float:
    left, right = max(0, index - 1), min(len(path) - 1, index + 1)
    delta = path[right] - path[left]
    if np.linalg.norm(delta) <= 0:
        raise ValueError("DEGENERATE_EGO_XY_TANGENT")
    return math.atan2(float(delta[1]), float(delta[0]))


def _corners(center: np.ndarray, length: float, width: float, yaw: float) -> np.ndarray:
    tangent = np.asarray([math.cos(yaw), math.sin(yaw)])
    normal = np.asarray([-tangent[1], tangent[0]])
    return np.asarray([center + a * length / 2.0 * tangent + b * width / 2.0 * normal for a in (-1.0, 1.0) for b in (-1.0, 1.0)])


def _interpolate(track: Mapping[str, Any], query: float, maximum_gap: float) -> np.ndarray | None:
    times, states = np.asarray(track.get("time_s"), dtype=np.float64), np.asarray(track.get("states"), dtype=np.float64)
    if times.ndim != 1 or states.shape != (len(times), 5) or not len(times) or not np.isfinite(times).all() or not np.isfinite(states).all() or np.any(np.diff(times) <= 0) or np.any(states[:, 2:4] <= 0):
        raise ValueError("INVALID_OFFICIAL_ACTOR_TRACK")
    index = int(np.searchsorted(times, query, side="left"))
    if index < len(times) and abs(float(times[index]) - query) <= 1e-12:
        return states[index].copy()
    if index == 0 or index == len(times):
        return None
    gap = float(times[index] - times[index - 1])
    if gap > maximum_gap + 1e-12:
        raise ValueError("ACTOR_INTERPOLATION_GAP_EXCEEDS_FROZEN_0P25S")
    fraction = (query - float(times[index - 1])) / gap
    output = states[index - 1] * (1.0 - fraction) + states[index] * fraction
    yaw_delta = (states[index, 4] - states[index - 1, 4] + math.pi) % (2.0 * math.pi) - math.pi
    output[4] = states[index - 1, 4] + fraction * yaw_delta
    return output


def _conflict(baseline: np.ndarray, treatment: np.ndarray, index: int, actor: np.ndarray, ego_length: float, ego_width: float, config: R1HLCDynamicClearanceConfigV1_1) -> bool:
    baseline_yaw, treatment_yaw = _heading(baseline, index), _heading(treatment, index)
    all_corners = np.vstack((_corners(baseline[index], ego_length, ego_width, baseline_yaw), _corners(treatment[index], ego_length, ego_width, treatment_yaw)))
    common_yaw = _heading(0.5 * (baseline + treatment), index)
    tangent = np.asarray([math.cos(common_yaw), math.sin(common_yaw)])
    normal = np.asarray([-tangent[1], tangent[0]])
    ego_long, ego_lat = all_corners @ tangent, all_corners @ normal
    actor_corners = _corners(actor[:2], float(actor[2]), float(actor[3]), float(actor[4]))
    actor_long, actor_lat = actor_corners @ tangent, actor_corners @ normal
    return bool(np.max(actor_long) >= np.min(ego_long) - config.longitudinal_buffer_m and np.min(actor_long) <= np.max(ego_long) + config.longitudinal_buffer_m and np.max(actor_lat) >= np.min(ego_lat) - config.lateral_buffer_m and np.min(actor_lat) <= np.max(ego_lat) + config.lateral_buffer_m)


def evaluate_r1_hlc_dynamic_clearance_v1_1(*, baseline_xy: Sequence[Sequence[float]], treatment_xy: Sequence[Sequence[float]], official_runtime_vehicle_parameters: Mapping[str, Any] | None, original_replay_tracks: Mapping[str, Mapping[str, Any]], official_replay_observation_timestamps_s: Sequence[float], config: R1HLCDynamicClearanceConfigV1_1 = R1HLCDynamicClearanceConfigV1_1()) -> Dict[str, Any]:
    if config != R1HLCDynamicClearanceConfigV1_1():
        raise ValueError("frozen clearance numerics may not be overridden")
    horizon = _global_horizon(official_replay_observation_timestamps_s, config)
    if not horizon["complete"]:
        return {**_not_eligible("REPLAY_OBSERVATION_HORIZON_BINDING", config), "observation_horizon": horizon}
    try:
        ego_length, ego_width = _footprint(official_runtime_vehicle_parameters)
    except ValueError as exc:
        return {**_not_eligible(str(exc), config), "observation_horizon": horizon}
    baseline, treatment = np.asarray(baseline_xy, dtype=np.float64), np.asarray(treatment_xy, dtype=np.float64)
    if baseline.shape != (80, 2) or treatment.shape != (80, 2) or not np.isfinite(baseline).all() or not np.isfinite(treatment).all():
        return {**_not_eligible("INVALID_HLC_NATIVE_GEOMETRY", config), "observation_horizon": horizon}
    if not original_replay_tracks:
        return {"status": "DYNAMIC_CLEAR_NO_ACTORS", "eligible": True, "pass": True, "first_conflict": None, "evaluated_actor_states": 0, "observation_horizon": horizon, "config": asdict(config), "envelope": "ORIENTED_FOOTPRINT_COMMON_ENVELOPE_BOTH_ARMS", "pretreatment_only": True}
    evaluated, conflict = 0, None
    for token, track in sorted(original_replay_tracks.items()):
        for index, query in enumerate(np.arange(80, dtype=np.float64) * 0.1):
            try:
                actor = _interpolate(track, float(query), 0.25)
            except ValueError as exc:
                return {**_not_eligible(str(exc), config), "observation_horizon": horizon}
            if actor is None:
                continue
            evaluated += 1
            if _conflict(baseline, treatment, index, actor, ego_length, ego_width, config):
                conflict = {"track_id": str(token), "iteration_index": index, "nominal_time_s": float(query)}
                break
        if conflict:
            break
    return {"status": "DYNAMIC_CLEARANCE_PASS" if conflict is None else "DYNAMIC_CLEARANCE_FAIL", "eligible": True, "pass": conflict is None, "first_conflict": conflict, "evaluated_actor_states": evaluated, "observation_horizon": horizon, "config": asdict(config), "envelope": "ORIENTED_FOOTPRINT_COMMON_ENVELOPE_BOTH_ARMS", "ego_heading_source": "EACH_ARM_PROSPECTIVE_XY_TANGENT", "pretreatment_only": True, "posthoc_recalculation_forbidden": True}


__all__ = ["R1HLCDynamicClearanceConfigV1_1", "evaluate_r1_hlc_dynamic_clearance_v1_1"]
