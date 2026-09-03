#!/usr/bin/env python3
"""Offline-frozen HLC V4 morphology and kinematic target-capture composition."""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Mapping, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_b_controller_aware_generator_v1 import ARM_BASELINE, ARM_TREATMENT, T_DIVERGE_S  # noqa: E402
from tools.r2_bi_hlc_kinematic_target_capture_generator_v3 import (  # noqa: E402
    CaptureInfeasible,
    _evaluate_quintic,
    _quintic_coefficients,
    _target_frame_residual,
    headings_and_curvature_from_xy,
)


DT_SECONDS = 0.1
QUINTIC_MAX_NORMALIZED_VELOCITY = 15.0 / 8.0
QUINTIC_MAX_NORMALIZED_ACCELERATION = 10.0 * math.sqrt(3.0) / 3.0


def smoothstep5(value: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _phase(result: np.ndarray, t: np.ndarray, start_t: float, duration: float, p0: float, p1: float) -> None:
    active = (t >= start_t) & (t < start_t + duration)
    result[active] = p0 + (p1 - p0) * smoothstep5((t[active] - start_t) / duration)


def morphology_progress(time_s: np.ndarray, arm: str, parameters: Mapping[str, Any]) -> np.ndarray:
    """Absolute-time C2 morphology with zero P/V/A arm difference at divergence."""
    t = np.asarray(time_s, dtype=np.float64)
    result = np.zeros_like(t)
    active = t >= T_DIVERGE_S
    if arm == ARM_BASELINE:
        duration = float(parameters["baseline_transition_duration_s"])
        _phase(result, t, T_DIVERGE_S, duration, 0.0, 1.0)
        result[t >= T_DIVERGE_S + duration] = 1.0
        return result
    if arm != ARM_TREATMENT:
        raise ValueError(f"R2_BJ_A_UNKNOWN_ARM:{arm}")
    if float(parameters.get("lag_precompensation_s", 0.0)) != 0.0:
        raise ValueError("R2_BJ_A_DIRECT_LAG_SHIFT_FORBIDDEN")
    advance = float(parameters["advance_duration_s"])
    p_advance = float(parameters["advance_progress"])
    hold = float(parameters["hold_duration_s"])
    depth = float(parameters["retreat_depth"])
    retreat = float(parameters["retreat_duration_s"])
    recommit = float(parameters["recommit_duration_s"])
    p_retreat = p_advance - depth
    advance_end = T_DIVERGE_S + advance
    hold_end = advance_end + hold
    retreat_end = hold_end + retreat
    recommit_end = retreat_end + recommit
    _phase(result, t, T_DIVERGE_S, advance, 0.0, p_advance)
    result[(t >= advance_end) & (t < hold_end)] = p_advance
    _phase(result, t, hold_end, retreat, p_advance, p_retreat)
    _phase(result, t, retreat_end, recommit, p_retreat, 1.0)
    result[t >= recommit_end] = 1.0
    result[~active] = 0.0
    return result


def phase_boundaries(parameters: Mapping[str, Any]) -> Dict[str, float]:
    advance_end = T_DIVERGE_S + float(parameters["advance_duration_s"])
    hold_end = advance_end + float(parameters["hold_duration_s"])
    retreat_end = hold_end + float(parameters["retreat_duration_s"])
    return {
        "diverge": T_DIVERGE_S,
        "advance_end": advance_end,
        "hold_end": hold_end,
        "retreat_end": retreat_end,
        "recommit_end": retreat_end + float(parameters["recommit_duration_s"]),
    }


def analytic_phase_metrics(
    name: str, progress_start: float, progress_end: float, duration_s: float, lane_separation_m: float
) -> Dict[str, Any]:
    delta = float(progress_end - progress_start)
    duration = float(duration_s)
    separation = float(lane_separation_m)
    return {
        "phase": name,
        "progress_start": float(progress_start),
        "progress_end": float(progress_end),
        "delta_progress": delta,
        "duration_s": duration,
        "maximum_normalized_velocity": QUINTIC_MAX_NORMALIZED_VELOCITY,
        "maximum_normalized_acceleration": QUINTIC_MAX_NORMALIZED_ACCELERATION,
        "lane_separation_scaled_max_lateral_velocity_mps": QUINTIC_MAX_NORMALIZED_VELOCITY * separation * abs(delta) / duration,
        "lane_separation_scaled_max_lateral_acceleration_mps2": QUINTIC_MAX_NORMALIZED_ACCELERATION * separation * abs(delta) / duration**2,
        "boundary_position_velocity_acceleration_continuity": "C2_ZERO_ENDPOINT_VELOCITY_AND_ACCELERATION",
    }


def _stitching_correction(
    base_xy: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    current_abs_s: float,
    future_abs_s: np.ndarray,
    capture: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    start = float(capture["capture_start_abs_s"])
    nominal_end = float(capture["nominal_capture_end_abs_s"])
    horizon = float(capture["minimum_stitching_horizon_s"])
    delta0 = np.asarray(current_xy, dtype=np.float64) - np.asarray(base_xy[0], dtype=np.float64)
    base_velocity = (np.asarray(base_xy[1]) - np.asarray(base_xy[0])) / DT_SECONDS
    base_speed = max(float(np.linalg.norm(base_velocity)), 1e-6)
    desired_velocity = base_speed * np.asarray([math.cos(current_heading), math.sin(current_heading)])
    delta_velocity = desired_velocity - base_velocity
    correction = np.zeros_like(base_xy, dtype=np.float64)
    effective_start = max(start, float(current_abs_s))
    duration = max(nominal_end - effective_start, horizon)
    if float(current_abs_s) < start:
        before = future_abs_s <= start + 1e-12
        correction[before] = delta0
        coefficients = _quintic_coefficients(delta0, np.zeros(2), np.zeros(2), duration)
        active = future_abs_s > start
        elapsed = np.minimum(future_abs_s[active] - start, duration)
        correction[active] = _evaluate_quintic(coefficients, elapsed)
        initial_velocity = np.zeros(2)
        mode = "WAIT_THEN_C2_STITCH"
    else:
        coefficients = _quintic_coefficients(delta0, delta_velocity, np.zeros(2), duration)
        elapsed = np.minimum(future_abs_s - current_abs_s, duration)
        correction = _evaluate_quintic(coefficients, elapsed)
        initial_velocity = delta_velocity
        mode = "ONLINE_C2_STITCH_WITH_GLOBAL_HORIZON_FLOOR"
    return correction, {
        "mode": mode,
        "capture_start_abs_s": start,
        "nominal_capture_end_abs_s": nominal_end,
        "minimum_stitching_horizon_s": horizon,
        "effective_stitching_duration_s": duration,
        "current_vector_residual_m": delta0.tolist(),
        "current_vector_residual_norm_m": float(np.linalg.norm(delta0)),
        "initial_correction_velocity_mps": initial_velocity.tolist(),
        "denominator_zero_special_case_used": False,
        "deadline_hard_jump_used": False,
    }


def compose_kinematic_trajectory(
    morphology_xy: np.ndarray,
    target_reference_xy: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    current_speed_mps: float,
    current_abs_s: float,
    future_abs_s: np.ndarray,
    capture: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Preserve V3 XY→heading→curvature semantics with a non-singular stitching rule."""
    base = np.asarray(morphology_xy, dtype=np.float64)
    target = np.asarray(target_reference_xy, dtype=np.float64)
    future = np.asarray(future_abs_s, dtype=np.float64)
    correction, stitch = _stitching_correction(
        base, np.asarray(current_xy, dtype=np.float64), float(current_heading), float(current_abs_s), future, capture
    )
    xy = base + correction
    xy[0] = np.asarray(current_xy, dtype=np.float64)
    headings, curvature, pose = headings_and_curvature_from_xy(xy, float(current_heading))
    speed = max(float(current_speed_mps), 0.2)
    yaw_rate = curvature * speed
    lateral_acceleration = curvature * speed**2
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    nominal_step = speed * DT_SECONDS
    limits = capture["frozen_feasibility_limits"]
    feasibility = {
        "max_abs_curvature_inv_m": float(np.max(np.abs(curvature))),
        "max_abs_yaw_rate_radps": float(np.max(np.abs(yaw_rate))),
        "max_abs_lateral_acceleration_mps2": float(np.max(np.abs(lateral_acceleration))),
        "state0_to_state1_distance_m": float(lengths[0]),
        "nominal_state_step_distance_m": nominal_step,
        "state0_tangent_mismatch_abs_rad": pose["state0_tangent_mismatch_abs_rad"],
        "future_heading_xy_mismatch_abs_rad": pose["max_future_declared_heading_vs_final_xy_tangent_abs_rad"],
    }
    feasibility["pass"] = bool(
        feasibility["max_abs_curvature_inv_m"] <= float(limits["curvature_inv_m_max"])
        and feasibility["max_abs_yaw_rate_radps"] <= float(limits["yaw_rate_radps_max"])
        and feasibility["max_abs_lateral_acceleration_mps2"] <= float(limits["lateral_accel_mps2_max"])
        and lengths[0] <= nominal_step + float(limits["state0_to_state1_distance_excess_m_max"])
        and pose["state0_tangent_mismatch_abs_rad"] <= float(limits["state0_tangent_mismatch_rad_max"])
        and pose["max_future_declared_heading_vs_final_xy_tangent_abs_rad"] <= float(limits["heading_xy_consistency_rad_max"])
    )
    audit = {
        **stitch,
        "state0_exact_current_xy": bool(np.array_equal(xy[0], np.asarray(current_xy, dtype=np.float64))),
        "state0_exact_current_heading": bool(float(headings[0]) == float(current_heading)),
        "actual_planned_target_frame_offsets_m": [_target_frame_residual(point, target) for point in xy],
        "algebraic_stitching_correction_vectors_xy_m": correction.tolist(),
        "pose_consistency": pose,
        "feasibility": feasibility,
        "final_xy_heading_curvature_single_source": True,
        "controller_visible_curvature_profile": curvature.tolist(),
    }
    if not feasibility["pass"]:
        raise CaptureInfeasible("R2_BJ_A_FROZEN_KINEMATIC_FEASIBILITY_GATE_FAIL", audit)
    return xy, headings, curvature, audit


def validate_parameters(parameters: Mapping[str, Any]) -> None:
    if set(parameters) != {"morphology", "capture"}:
        raise ValueError("R2_BJ_A_PARAMETER_PARTITIONS_INVALID")
    forbidden = {"scenario_token", "log_id", "identity", "per_scenario"}
    if forbidden.intersection(parameters["morphology"]) or forbidden.intersection(parameters["capture"]):
        raise ValueError("R2_BJ_A_SCENARIO_SPECIFIC_PARAMETER_FORBIDDEN")
    morphology = parameters["morphology"]
    required = {
        "baseline_transition_duration_s", "advance_duration_s", "advance_progress", "hold_duration_s",
        "retreat_depth", "retreat_duration_s", "recommit_duration_s", "lag_precompensation_s",
    }
    if not required.issubset(morphology):
        raise ValueError("R2_BJ_A_MORPHOLOGY_PARAMETER_MISSING")
    if float(morphology["lag_precompensation_s"]) != 0.0:
        raise ValueError("R2_BJ_A_DIRECT_LAG_SHIFT_FORBIDDEN")
    if not 0.0 < float(morphology["retreat_depth"]) < float(morphology["advance_progress"]) < 1.0:
        raise ValueError("R2_BJ_A_PROGRESS_ORDER_INVALID")
    capture = parameters["capture"]
    if float(capture["minimum_stitching_horizon_s"]) <= 0.0:
        raise ValueError("R2_BJ_A_STITCHING_HORIZON_NOT_POSITIVE")


__all__ = [
    "ARM_BASELINE", "ARM_TREATMENT", "T_DIVERGE_S", "CaptureInfeasible",
    "QUINTIC_MAX_NORMALIZED_VELOCITY", "QUINTIC_MAX_NORMALIZED_ACCELERATION",
    "analytic_phase_metrics", "compose_kinematic_trajectory", "morphology_progress",
    "phase_boundaries", "smoothstep5", "validate_parameters",
]
