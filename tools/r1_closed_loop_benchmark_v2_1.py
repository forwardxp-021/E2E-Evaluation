#!/usr/bin/env python3
"""Final prospective R1 closed-loop benchmark primitives (v2.1).

This module does not execute a rollout.  It versions the B2.4 timestamp-aware
measurement, route-continuous spatial realization, and frozen applicability
bindings while leaving the historical B2.1 implementation untouched.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tools.r1_closed_loop_benchmark_v2 import (
    DT_SECONDS,
    HLC_PRIMARY_F_MATCH_CALIPERS,
    TSB_PRIMARY_F_MATCH_CALIPERS,
    WINDOW_FRAMES,
    WINDOW_SECONDS,
    first_state_error,
    frozen_tsb_acceleration,
    polyline_arclength,
    prospective_primary_f_match,
    sample_native_reference_no_extrapolation,
    wrap_angle,
)
from tools.r1_context_mechanism_core import median3
from tools.r1_official_technical_smoke_planner import (
    HLC_BASELINE,
    HLC_TREATMENT,
    TSB_BASELINE,
    TSB_TREATMENT,
    hlc_progress,
)


IMPLEMENTATION_VERSION = "r1_closed_loop_benchmark_v2.1"
NOMINAL_SAMPLE_COUNT_0P3S = 3
NOMINAL_SAMPLE_COUNT_0P5S = 5
TSB_BASELINE_ACCELERATION_MPS2 = -1.0
TSB_BASELINE_ACTIVE_SAMPLE_INDICES = tuple(range(11, 21))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _run_ranges(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(mask):
        if bool(item) and start is None:
            start = index
        if start is not None and (not bool(item) or index == len(mask) - 1):
            end = index if bool(item) and index == len(mask) - 1 else index - 1
            result.append((start, end))
            start = None
    return result


def _validated_vectors(*vectors: Sequence[float]) -> Tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in vectors)
    if len(arrays[0]) < 6 or any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("mechanism inputs must be equal-length vectors of at least six samples")
    if any(value.ndim != 1 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("mechanism inputs must be finite one-dimensional vectors")
    if np.any(np.diff(arrays[0]) <= 0):
        raise ValueError("physical timestamps must be strictly increasing")
    return arrays


def exact_realized_window_v1_1(trace_rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Return 80 consecutive actual ego states while preserving physical time."""
    if len(trace_rows) < WINDOW_FRAMES:
        raise ValueError("NOT_EVALUABLE_REALIZED_WINDOW: fewer than 80 simulator iterations")
    rows = list(trace_rows[:WINDOW_FRAMES])
    if [int(row.get("iteration_index", -1)) for row in rows] != list(range(WINDOW_FRAMES)):
        raise ValueError("NOT_EVALUABLE_ITERATION_SEQUENCE: expected consecutive indices 0...79")
    times = np.asarray([float(row["current_ego"]["time_us"]) for row in rows], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("NOT_EVALUABLE_PHYSICAL_TIMESTAMPS: timestamps must be finite and strictly increasing")
    return [row["current_ego"] for row in rows]


def trajectory_arrays_timestamp_aware(
    states: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(states) != WINDOW_FRAMES:
        raise ValueError("trajectory must contain exactly 80 consecutive states")
    time_us = np.asarray([float(state["time_us"]) for state in states], dtype=np.float64)
    if not np.isfinite(time_us).all() or np.any(np.diff(time_us) <= 0):
        raise ValueError("trajectory physical timestamps must be finite and strictly increasing")
    time = (time_us - time_us[0]) * 1e-6
    xy = np.asarray([[state["rear_axle"]["x"], state["rear_axle"]["y"]] for state in states], dtype=np.float64)
    heading = np.asarray([state["rear_axle"]["heading"] for state in states], dtype=np.float64)
    speed = np.asarray([state["speed_mps"] for state in states], dtype=np.float64)
    if not np.isfinite(xy).all() or not np.isfinite(heading).all() or not np.isfinite(speed).all():
        raise ValueError("trajectory state values must be finite")
    return time, xy, heading, speed


def calculate_hlc_option_b_v2_timestamp_aware(
    time_s: Sequence[float], progress_p: Sequence[float], speed_mps: Sequence[float], map_valid: bool = True
) -> Dict[str, Any]:
    """Frozen Option-B gates with actual-time dp/dt and nominal sample durations."""
    time, p_raw, speed = _validated_vectors(time_s, progress_p, speed_mps)
    p = median3(np.clip(p_raw, 0.0, 1.0))
    result: Dict[str, Any] = {
        "option": "OPTION_B",
        "median3_p": [_round(value) for value in p],
        "map_valid": bool(map_valid),
    }
    if not map_valid:
        return {**result, "status": "MAP_INVALID", "hesitation_retreat_count": None, "commit_latency_s": None, "monotonic_transition_fraction": None}
    if any(end - start + 1 >= NOMINAL_SAMPLE_COUNT_0P5S for start, end in _run_ranges(speed < 1.0)):
        return {**result, "status": "LOW_SPEED_TRANSITION", "hesitation_retreat_count": None, "commit_latency_s": None, "monotonic_transition_fraction": None}
    departures = np.flatnonzero(p >= 0.10)
    if len(departures) == 0:
        return {**result, "status": "NO_DEPARTURE", "hesitation_retreat_count": 0, "commit_latency_s": None, "monotonic_transition_fraction": None}
    departure = int(departures[0])
    commitment: int | None = None
    for start in range(departure, len(p) - NOMINAL_SAMPLE_COUNT_0P5S + 1):
        if np.all(p[start : start + NOMINAL_SAMPLE_COUNT_0P5S] >= 0.75):
            commitment = start
            break
    if commitment is None:
        return {**result, "status": "UNFINISHED_TRANSITION", "departure_time_s": _round(time[departure]), "hesitation_retreat_count": 0, "commit_latency_s": None, "monotonic_transition_fraction": None}
    derivative = np.diff(p) / np.diff(time)
    negative_runs = _run_ranges(derivative <= -0.10)
    candidates: List[Tuple[int, int]] = []
    for start, end in negative_runs:
        if start < departure or start >= commitment or end - start + 1 < NOMINAL_SAMPLE_COUNT_0P3S:
            continue
        recovery_start = min(end + 1, commitment)
        for possible in range(end + 1, max(end + 1, commitment - NOMINAL_SAMPLE_COUNT_0P3S + 1)):
            if np.all(derivative[possible : possible + NOMINAL_SAMPLE_COUNT_0P3S] >= 0.04):
                recovery_start = possible
                break
        event_end = min(commitment, max(end + 1, recovery_start))
        fall = float(p[start] - np.min(p[start : event_end + 1]))
        if fall >= 0.08 - 1e-9 and event_end - start >= 4:
            candidates.append((start, event_end))
    merged: List[Tuple[int, int]] = []
    for start, end in candidates:
        if merged and start - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    deltas = np.diff(p[departure : commitment + 1])
    denominator = float(np.sum(np.maximum(np.abs(deltas) - 0.008, 0.0)))
    monotonic = None if denominator < 0.1 else 1.0 - float(np.sum(np.maximum(-deltas - 0.008, 0.0))) / denominator
    return {
        **result,
        "status": "OK" if monotonic is not None else "NOT_EVALUABLE_MONOTONIC_DENOMINATOR",
        "departure_time_s": _round(time[departure]),
        "commit_time_s": _round(time[commitment]),
        "commit_latency_s": _round(time[commitment] - time[0]),
        "hesitation_retreat_count": len(merged),
        "retreat_episodes": [{"start_time_s": _round(time[start]), "end_time_s": _round(time[end])} for start, end in merged],
        "monotonic_transition_fraction": None if monotonic is None else _round(monotonic),
    }


def calculate_tsb_option_a_v2_timestamp_aware(
    time_s: Sequence[float], longitudinal_speed_mps: Sequence[float]
) -> Dict[str, Any]:
    """Frozen Option-A gates with actual-time finite differences."""
    time, speed_raw = _validated_vectors(time_s, longitudinal_speed_mps)
    speed = median3(speed_raw)
    accel = np.gradient(speed, time, edge_order=2)
    result: Dict[str, Any] = {
        "option": "OPTION_A",
        "median3_speed_mps": [_round(value) for value in speed],
        "acceleration_mps2": [_round(value) for value in accel],
    }
    if any(end - start + 1 >= NOMINAL_SAMPLE_COUNT_0P5S for start, end in _run_ranges(speed < 1.0)):
        return {**result, "status": "LOW_SPEED_ENDSTOP", "brake_phase_count": None, "interstage_release_fraction": None, "second_brake_peak_ratio": None}
    raw = [(start, end) for start, end in _run_ranges(accel <= -0.80) if end - start + 1 >= NOMINAL_SAMPLE_COUNT_0P3S]
    phases: List[Tuple[int, int]] = []
    for start, end in raw:
        if not phases:
            phases.append((start, end))
            continue
        prior_start, prior_end = phases[-1]
        gap = start - prior_end - 1
        release_ranges = _run_ranges(accel[prior_end + 1 : start] >= -0.20)
        has_release = gap >= NOMINAL_SAMPLE_COUNT_0P3S and any(b - a + 1 >= NOMINAL_SAMPLE_COUNT_0P3S for a, b in release_ranges)
        if gap < NOMINAL_SAMPLE_COUNT_0P3S or not has_release:
            phases[-1] = (prior_start, end)
        else:
            phases.append((start, end))
    phase_records = [
        {"start_time_s": _round(time[start]), "end_time_s": _round(time[end]), "peak_decel_mps2": _round(float(np.max(-accel[start : end + 1])))}
        for start, end in phases
    ]
    release_fraction = None
    peak_ratio = None
    if len(phases) >= 2:
        first_start, first_end = phases[0]
        second_start, second_end = phases[1]
        gap_speed = speed[first_end + 1 : second_start]
        first_loss = max(float(speed[first_start] - np.min(speed[first_start : first_end + 1])), 0.1)
        release_fraction = float(np.max(gap_speed) - speed[first_end]) / first_loss if len(gap_speed) else 0.0
        first_peak = max(float(np.max(-accel[first_start : first_end + 1])), 0.8)
        second_peak = float(np.max(-accel[second_start : second_end + 1]))
        peak_ratio = second_peak / first_peak
    return {
        **result,
        "status": "OK" if phases else "NO_BRAKE_PHASE",
        "brake_phase_count": len(phases),
        "brake_phases": phase_records,
        "interstage_release_fraction": None if release_fraction is None else _round(release_fraction),
        "second_brake_peak_ratio": None if peak_ratio is None else _round(peak_ratio),
    }


def trajectory_descriptors_timestamp_aware(states: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    time, xy, heading, speed = trajectory_arrays_timestamp_aware(states)
    accel = np.gradient(speed, time, edge_order=2)
    return {
        "mean_speed": _round(float(np.mean(speed))),
        "end_minus_start_speed": _round(float(speed[-1] - speed[0])),
        "mean_abs_accel": _round(float(np.mean(np.abs(accel)))),
        "path_length": _round(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))),
        "heading_change_abs_total": _round(float(np.sum(np.abs(np.diff(np.unwrap(heading)))))),
    }


def timestamp_aware_hlc_engineering(states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    time, xy, _, _ = trajectory_arrays_timestamp_aware(states)
    velocity = np.gradient(xy, time, axis=0, edge_order=2)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    yaw_rate = np.gradient(heading, time, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    speed = np.linalg.norm(velocity, axis=1)
    tangent = velocity / np.maximum(speed[:, None], 1e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    lateral_accel = np.sum(acceleration * normal, axis=1)
    curvature = yaw_rate / np.maximum(speed, 1e-12)
    return {
        "max_abs_lateral_accel_mps2": _round(float(np.max(np.abs(lateral_accel)))),
        "max_abs_yaw_rate_radps": _round(float(np.max(np.abs(yaw_rate)))),
        "max_abs_curvature_inv_m": _round(float(np.max(np.abs(curvature)))),
        "derivative_time_source": "ACTUAL_PHYSICAL_TIMESTAMPS",
        "frozen_limits": {"lateral_accel_mps2_max": 6.0, "yaw_rate_radps_max": 1.0, "curvature_inv_m_max": 0.5},
    }


def _state(x: float, y: float, heading: float, speed: float, time_us: int) -> Dict[str, Any]:
    return {"rear_axle": {"x": float(x), "y": float(y), "heading": float(heading)}, "speed_mps": float(speed), "time_us": int(time_us)}


def _headings_from_xy(xy: np.ndarray) -> np.ndarray:
    delta = np.diff(xy, axis=0)
    if np.any(np.linalg.norm(delta, axis=1) <= 1e-12):
        raise ValueError("STRUCTURAL_CONTINUITY_FAIL: duplicate future XY state")
    segment = np.arctan2(delta[:, 1], delta[:, 0])
    return np.r_[segment[0], segment]


def _offset_preserving_route_xy(
    reference_xy: Sequence[Sequence[float]], query_arc_m: np.ndarray, current_ego: Mapping[str, Any]
) -> np.ndarray:
    center, center_heading = sample_native_reference_no_extrapolation(reference_xy, query_arc_m)
    normal = np.column_stack((-np.sin(center_heading), np.cos(center_heading)))
    current_xy = np.asarray([current_ego["rear_axle"]["x"], current_ego["rear_axle"]["y"]], dtype=np.float64)
    signed_offset = float(np.dot(current_xy - center[0], normal[0]))
    heading_offset = wrap_angle(float(current_ego["rear_axle"]["heading"]) - float(center_heading[0]))
    distance = query_arc_m - query_arc_m[0]
    total = max(float(distance[-1]), 1e-12)
    u = distance / total
    lateral = signed_offset + math.tan(heading_offset) * distance * np.square(1.0 - u)
    xy = center + normal * lateral[:, None]
    xy[0] = current_xy
    return xy


def build_tsb_route_aligned_v1_1(
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    route_reference_xy: Sequence[Sequence[float]],
    current_route_arc_m: float,
    arm: str,
) -> Sequence[Dict[str, Any]]:
    """Use one native route and preserve current signed lateral/heading offsets."""
    absolute = absolute_episode_time_s + np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    speed = np.empty(WINDOW_FRAMES, dtype=np.float64)
    distance = np.zeros(WINDOW_FRAMES, dtype=np.float64)
    speed[0] = float(current_ego["speed_mps"])
    for index in range(1, WINDOW_FRAMES):
        acceleration = frozen_tsb_acceleration(absolute[index - 1], arm)
        speed[index] = max(0.2, round(float(speed[index - 1] + acceleration * DT_SECONDS), 12))
        distance[index] = distance[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * DT_SECONDS
    query = float(current_route_arc_m) + distance
    xy = _offset_preserving_route_xy(route_reference_xy, query, current_ego)
    heading = _headings_from_xy(xy)
    start_us = int(current_ego["time_us"])
    states = [_state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000) for i in range(WINDOW_FRAMES)]
    states[0] = dict(current_ego)
    if not first_state_error(current_ego, states[0])["exact_construction_identity"]:
        raise AssertionError("current-ego first-state construction identity failed")
    return states


def build_hlc_native_geometry_v1_1(
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    source_reference_xy: Sequence[Sequence[float]],
    target_reference_xy: Sequence[Sequence[float]],
    source_current_arc_m: float,
    target_current_arc_m: float,
    arm: str,
) -> Sequence[Dict[str, Any]]:
    """Realize unchanged Option-B progress; derive heading only from final XY."""
    relative = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    absolute = absolute_episode_time_s + relative
    speed = np.full(WINDOW_FRAMES, float(current_ego["speed_mps"]), dtype=np.float64)
    distance = speed * relative
    source, _ = sample_native_reference_no_extrapolation(source_reference_xy, float(source_current_arc_m) + distance)
    target, _ = sample_native_reference_no_extrapolation(target_reference_xy, float(target_current_arc_m) + distance)
    progress = hlc_progress(absolute, arm)
    xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    current_xy = np.asarray([current_ego["rear_axle"]["x"], current_ego["rear_axle"]["y"]], dtype=np.float64)
    xy += current_xy - xy[0]
    heading = _headings_from_xy(xy)
    start_us = int(current_ego["time_us"])
    states = [_state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000) for i in range(WINDOW_FRAMES)]
    states[0] = dict(current_ego)
    if not first_state_error(current_ego, states[0])["exact_construction_identity"]:
        raise AssertionError("current-ego first-state construction identity failed")
    return states


def structural_first_segment_audit(current_ego: Mapping[str, Any], states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(states) < 2:
        raise ValueError("first-segment audit requires at least two states")
    identity = first_state_error(current_ego, states[0])
    p0 = np.asarray([states[0]["rear_axle"]["x"], states[0]["rear_axle"]["y"]], dtype=np.float64)
    p1 = np.asarray([states[1]["rear_axle"]["x"], states[1]["rear_axle"]["y"]], dtype=np.float64)
    dt = (int(states[1]["time_us"]) - int(states[0]["time_us"])) * 1e-6
    segment = p1 - p0
    distance = float(np.linalg.norm(segment))
    tangent_heading = math.atan2(float(segment[1]), float(segment[0])) if distance > 0 else math.nan
    heading_jump = abs(wrap_angle(tangent_heading - float(current_ego["rear_axle"]["heading"]))) if distance > 0 else math.inf
    return {
        "status": "STRUCTURAL_FIRST_SEGMENT_CONTINUITY_PASS" if identity["exact_construction_identity"] and dt > 0 and distance > 0 and np.isfinite(heading_jump) else "STRUCTURAL_FIRST_SEGMENT_CONTINUITY_FAIL",
        "pass": bool(identity["exact_construction_identity"] and dt > 0 and distance > 0 and np.isfinite(heading_jump)),
        "trajectory_state0_exact": identity["exact_construction_identity"],
        "first_segment_distance_m": _round(distance),
        "first_segment_dt_s": _round(dt),
        "first_segment_tangent_heading_error_rad": _round(heading_jump),
        "audit_role": "STRUCTURAL_CONSTRUCTION_AUDIT_NOT_NEW_NUMERIC_GEOMETRY_GATE",
    }


def resolve_route_occurrence_cursor(
    route_roadblock_ids: Sequence[str], current_roadblock_id: str, native_outgoing_roadblock_ids: Sequence[str]
) -> int:
    """Resolve repeated route IDs using the current edge's native successors."""
    route = [str(value) for value in route_roadblock_ids]
    outgoing = {str(value) for value in native_outgoing_roadblock_ids}
    candidates = [index for index, value in enumerate(route) if value == str(current_roadblock_id)]
    if not candidates:
        raise ValueError("NATIVE_ROUTE_CURSOR_FAIL: current roadblock absent from route")
    compatible = [index for index in candidates if index == len(route) - 1 or route[index + 1] in outgoing]
    if len(compatible) != 1:
        raise ValueError("NATIVE_ROUTE_CURSOR_FAIL: repeated occurrence is not uniquely resolved by native topology")
    return compatible[0]


def tsb_baseline_execution_binding() -> Dict[str, Any]:
    time = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    active = tuple(index for index, value in enumerate(time) if frozen_tsb_acceleration(float(value), TSB_BASELINE) == -1.0)
    if active != TSB_BASELINE_ACTIVE_SAMPLE_INDICES:
        raise AssertionError(f"baseline active samples changed: {active}")
    return {
        "status": "TSB_BASELINE_DISCRETE_EXECUTION_BOUND",
        "behavior": "SINGLE_CONTINUOUS_BRAKING",
        "acceleration_mps2": TSB_BASELINE_ACCELERATION_MPS2,
        "active_sample_indices": list(active),
        "active_integration_interval_count": len(active),
        "nominal_dt_seconds": DT_SECONDS,
        "total_speed_loss_mps": abs(TSB_BASELINE_ACCELERATION_MPS2) * len(active) * DT_SECONDS,
        "implementation_source": "tools/r1_official_technical_smoke_planner.py::tsb_profile",
        "proposal_source": "tools/r1_residual_generators.py::TSB_BASELINE",
        "b2_1_outcomes_used": False,
    }


def tsb_applicability_v1(step_mps: float = 0.001) -> Dict[str, Any]:
    binding = tsb_baseline_execution_binding()
    time = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS

    def profile(initial_speed: float, arm: str) -> np.ndarray:
        speed = np.empty(WINDOW_FRAMES, dtype=np.float64)
        speed[0] = float(initial_speed)
        for index in range(1, WINDOW_FRAMES):
            speed[index] = max(0.2, round(float(speed[index - 1] + frozen_tsb_acceleration(time[index - 1], arm) * DT_SECONDS), 12))
        return speed

    first = None
    for initial in np.arange(0.0, 4.0 + step_mps / 2.0, step_mps):
        statuses = [calculate_tsb_option_a_v2_timestamp_aware(time, profile(float(initial), arm))["status"] for arm in (TSB_BASELINE, TSB_TREATMENT)]
        if statuses == ["OK", "OK"]:
            first = float(initial)
            break
    analytical = 2.0
    if first is None or abs(first - analytical) > step_mps / 2.0:
        raise AssertionError("TSB analytical/synthetic applicability parity failed")
    return {
        "status": "FROZEN_OWNER_APPROVED_BASELINE_EXECUTION_BOUND",
        "TSB_MECHANISM_APPLICABILITY_INITIAL_SPEED_FLOOR_MPS": analytical,
        "domain": "MEASUREMENT_APPLICABILITY_NOT_BEHAVIOR_OUTCOME_THRESHOLD",
        "analytical_floor_mps": analytical,
        "synthetic_first_jointly_evaluable_mps": first,
        "synthetic_grid_step_mps": step_mps,
        "parity": True,
        "baseline_binding": binding,
        "b2_1_initial_speeds_used": False,
    }


__all__ = [
    "HLC_PRIMARY_F_MATCH_CALIPERS",
    "TSB_PRIMARY_F_MATCH_CALIPERS",
    "calculate_hlc_option_b_v2_timestamp_aware",
    "calculate_tsb_option_a_v2_timestamp_aware",
    "exact_realized_window_v1_1",
    "trajectory_descriptors_timestamp_aware",
    "timestamp_aware_hlc_engineering",
    "prospective_primary_f_match",
    "build_tsb_route_aligned_v1_1",
    "build_hlc_native_geometry_v1_1",
    "structural_first_segment_audit",
    "resolve_route_occurrence_cursor",
    "tsb_baseline_execution_binding",
    "tsb_applicability_v1",
]
