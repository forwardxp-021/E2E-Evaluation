#!/usr/bin/env python3
"""Outcome-blind deterministic R1 technical-smoke trajectory generators."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.stage7l_pure_lateral_execution_planner import FrozenLaneChangeManeuver, derive_trajectory_states, interpolate_polyline, quintic_blend


SMOKE_DT_SECONDS = 0.1
SMOKE_HORIZON_SECONDS = 4.5
HLC_SMOKE_CANDIDATES: Dict[str, Dict[str, float]] = {
    "HLC_MILD": {"p_hold": 0.30, "retreat_delta_p": 0.10, "hold_seconds": 0.4, "retreat_seconds": 0.4, "recommit_seconds": 0.8},
    "HLC_NOMINAL": {"p_hold": 0.35, "retreat_delta_p": 0.15, "hold_seconds": 0.5, "retreat_seconds": 0.5, "recommit_seconds": 1.0},
    "HLC_STRONG": {"p_hold": 0.40, "retreat_delta_p": 0.20, "hold_seconds": 0.6, "retreat_seconds": 0.6, "recommit_seconds": 1.2},
}
TSB_BASELINE: Dict[str, float] = {"brake_intensity_mps2": -1.0, "brake_duration_seconds": 0.95}
TSB_SMOKE_CANDIDATES: Dict[str, Dict[str, float]] = {
    "TSB_MILD": {"first_brake_intensity_mps2": -0.9, "first_brake_duration_seconds": 0.5, "release_intensity_mps2": 0.3, "release_duration_seconds": 0.3, "second_brake_intensity_mps2": -0.9, "second_brake_duration_seconds": 0.5},
    "TSB_NOMINAL": {"first_brake_intensity_mps2": -1.0, "first_brake_duration_seconds": 0.6, "release_intensity_mps2": 0.3, "release_duration_seconds": 0.4, "second_brake_intensity_mps2": -1.0, "second_brake_duration_seconds": 0.6},
    "TSB_STRONG": {"first_brake_intensity_mps2": -1.2, "first_brake_duration_seconds": 0.6, "release_intensity_mps2": 0.4, "release_duration_seconds": 0.4, "second_brake_intensity_mps2": -1.2, "second_brake_duration_seconds": 0.6},
}


def _time_vector() -> np.ndarray:
    return np.arange(0.0, SMOKE_HORIZON_SECONDS + SMOKE_DT_SECONDS / 2.0, SMOKE_DT_SECONDS, dtype=np.float64)


def _smooth_join(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * quintic_blend(elapsed / duration)


def _hlc_progress(time: np.ndarray, treatment: bool, parameters: Mapping[str, float] | None) -> np.ndarray:
    p = np.zeros_like(time)
    diverge = 1.1
    if not treatment:
        active = time >= diverge
        p[active] = _smooth_join(0.0, 1.0, time[active] - diverge, 2.0)
        return p
    if parameters is None:
        raise ValueError("HLC treatment requires a predeclared parameter candidate")
    advance_seconds = 0.7
    p_hold = float(parameters["p_hold"])
    retreat_target = p_hold - float(parameters["retreat_delta_p"])
    hold_seconds = float(parameters["hold_seconds"])
    retreat_seconds = float(parameters["retreat_seconds"])
    recommit_seconds = float(parameters["recommit_seconds"])
    bounds = [diverge, diverge + advance_seconds, diverge + advance_seconds + hold_seconds, diverge + advance_seconds + hold_seconds + retreat_seconds]
    first = (time >= bounds[0]) & (time < bounds[1])
    p[first] = _smooth_join(0.0, p_hold, time[first] - bounds[0], advance_seconds)
    hold = (time >= bounds[1]) & (time < bounds[2])
    p[hold] = p_hold
    retreat = (time >= bounds[2]) & (time < bounds[3])
    p[retreat] = _smooth_join(p_hold, retreat_target, time[retreat] - bounds[2], retreat_seconds)
    recommit = time >= bounds[3]
    p[recommit] = _smooth_join(retreat_target, 1.0, time[recommit] - bounds[3], recommit_seconds)
    return np.clip(p, 0.0, 1.0)


def generate_hlc_trajectory(maneuver_mapping: Mapping[str, Any], candidate_id: str | None = None) -> Dict[str, Any]:
    """Build HLC baseline (candidate_id None) or one predeclared treatment trajectory."""
    maneuver = FrozenLaneChangeManeuver.from_mapping(maneuver_mapping)
    time = _time_vector()
    treatment = candidate_id is not None
    if candidate_id is not None and candidate_id not in HLC_SMOKE_CANDIDATES:
        raise ValueError(f"unknown HLC smoke candidate: {candidate_id}")
    p = _hlc_progress(time, treatment, HLC_SMOKE_CANDIDATES.get(candidate_id) if candidate_id else None)
    progress = float(maneuver.initial_speed_mps) * time
    source = interpolate_polyline(np.asarray(maneuver.source_reference_xy), maneuver.source_start_arc_m + progress)
    target = interpolate_polyline(np.asarray(maneuver.target_reference_xy), maneuver.target_start_arc_m + progress)
    xy = source * (1.0 - p[:, None]) + target * p[:, None]
    states = derive_trajectory_states(xy, time, wheel_base_m=3.0)
    return {"family": "R-HLC", "arm": "TREATMENT" if treatment else "BASELINE", "candidate_id": candidate_id or "HLC_BASELINE", "time_s": time, "xy": xy, "speed_mps": states["speed"], "progress_p": p, "states": states}


def _tsb_acceleration_profile(time: np.ndarray, candidate_id: str | None) -> np.ndarray:
    acceleration = np.zeros_like(time)
    diverge = 1.1
    if candidate_id is None:
        parameters = TSB_BASELINE
        active = (time >= diverge) & (time < diverge + parameters["brake_duration_seconds"])
        acceleration[active] = parameters["brake_intensity_mps2"]
        return acceleration
    if candidate_id not in TSB_SMOKE_CANDIDATES:
        raise ValueError(f"unknown TSB smoke candidate: {candidate_id}")
    parameters = TSB_SMOKE_CANDIDATES[candidate_id]
    cursor = diverge
    intervals = (("first_brake_intensity_mps2", "first_brake_duration_seconds"), ("release_intensity_mps2", "release_duration_seconds"), ("second_brake_intensity_mps2", "second_brake_duration_seconds"))
    for intensity_key, duration_key in intervals:
        next_cursor = cursor + float(parameters[duration_key])
        active = (time >= cursor) & (time < next_cursor)
        acceleration[active] = float(parameters[intensity_key])
        cursor = next_cursor
    return acceleration


def generate_tsb_trajectory(initial_speed_mps: float, candidate_id: str | None = None) -> Dict[str, Any]:
    """Build a piecewise longitudinal baseline or one fixed two-brake treatment."""
    time = _time_vector()
    acceleration = _tsb_acceleration_profile(time, candidate_id)
    speed = np.empty_like(time)
    x = np.empty_like(time)
    speed[0] = max(float(initial_speed_mps), 3.0)
    x[0] = 0.0
    for index in range(1, len(time)):
        dt = time[index] - time[index - 1]
        speed[index] = max(0.2, speed[index - 1] + acceleration[index - 1] * dt)
        x[index] = x[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * dt
    xy = np.column_stack((x, np.zeros_like(x)))
    return {"family": "R-TSB", "arm": "TREATMENT" if candidate_id else "BASELINE", "candidate_id": candidate_id or "TSB_BASELINE", "time_s": time, "xy": xy, "speed_mps": speed, "commanded_acceleration_mps2": acceleration}


def kinematic_integrity(trajectory: Mapping[str, Any]) -> Dict[str, Any]:
    time = np.asarray(trajectory["time_s"], dtype=np.float64)
    xy = np.asarray(trajectory["xy"], dtype=np.float64)
    speed = np.asarray(trajectory["speed_mps"], dtype=np.float64)
    finite = bool(np.isfinite(time).all() and np.isfinite(xy).all() and np.isfinite(speed).all())
    monotonic_time = bool(np.all(np.diff(time) > 0))
    nonnegative_speed = bool(np.all(speed >= 0.0))
    if trajectory["family"] == "R-HLC":
        states = trajectory["states"]
        lateral_accel = float(np.max(np.abs(states["lateral_accel"])))
        yaw_rate = float(np.max(np.abs(states["yaw_rate"])))
        curvature = float(np.max(np.abs(states["curvature"])))
        bounds_pass = lateral_accel <= 6.0 and yaw_rate <= 1.0 and curvature <= 0.5
    else:
        lateral_accel, yaw_rate, curvature, bounds_pass = 0.0, 0.0, 0.0, True
    return {"status": "KINEMATIC_INTEGRITY_PASS" if finite and monotonic_time and nonnegative_speed and bounds_pass else "KINEMATIC_INTEGRITY_FAIL", "pass": finite and monotonic_time and nonnegative_speed and bounds_pass, "finite": finite, "time_strictly_monotonic": monotonic_time, "nonnegative_speed": nonnegative_speed, "max_abs_lateral_accel_mps2": round(lateral_accel, 6), "max_abs_yaw_rate_radps": round(yaw_rate, 6), "max_abs_curvature_inv_m": round(curvature, 6), "scope": "KINEMATIC_ONLY_NOT_OFFICIAL_CLOSED_LOOP_SAFETY"}
