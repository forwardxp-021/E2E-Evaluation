#!/usr/bin/env python3
"""R2-BI HLC V3 kinematic, controller-observable target-capture primitive."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_b_controller_aware_generator_v1 import (  # noqa: E402
    ARM_BASELINE,
    ARM_TREATMENT,
    T_DIVERGE_S,
    hlc_controller_aware_progress,
)


DT_SECONDS = 0.1


class CaptureInfeasible(RuntimeError):
    """Fail-closed marker for a capture that cannot satisfy frozen limits."""

    def __init__(self, reason: str, audit: Mapping[str, Any]):
        super().__init__(f"R2_BI_CAPTURE_INFEASIBLE:{reason}")
        self.reason = reason
        self.audit = dict(audit)


def _wrap(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def _quintic_coefficients(
    position0: np.ndarray, velocity0: np.ndarray, acceleration0: np.ndarray, duration_s: float
) -> np.ndarray:
    """Return vector quintic coefficients with zero terminal position/velocity/acceleration."""
    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError("R2_BI_QUINTIC_DURATION_NOT_POSITIVE")
    a0 = np.asarray(position0, dtype=np.float64)
    a1 = np.asarray(velocity0, dtype=np.float64)
    a2 = 0.5 * np.asarray(acceleration0, dtype=np.float64)
    matrix = np.asarray([
        [duration**3, duration**4, duration**5],
        [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
        [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
    ])
    rhs = np.stack((
        -(a0 + a1 * duration + a2 * duration**2),
        -(a1 + 2.0 * a2 * duration),
        -(2.0 * a2),
    ))
    tail = np.linalg.solve(matrix, rhs)
    return np.vstack((a0, a1, a2, tail))


def _evaluate_quintic(coefficients: np.ndarray, elapsed_s: np.ndarray) -> np.ndarray:
    elapsed = np.asarray(elapsed_s, dtype=np.float64)
    powers = np.column_stack([elapsed**index for index in range(6)])
    return powers @ np.asarray(coefficients, dtype=np.float64)


def headings_and_curvature_from_xy(xy: np.ndarray, state0_heading: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Derive all future headings and curvature from final xy, preserving exact state0 pose."""
    points = np.asarray(xy, dtype=np.float64)
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if len(points) < 3 or np.any(lengths <= 1e-6):
        raise CaptureInfeasible("DUPLICATE_OR_TOO_SHORT_XY_SEGMENT", {"minimum_segment_m": float(np.min(lengths))})
    tangents = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    headings = np.r_[tangents, tangents[-1]]
    state0_tangent_mismatch = abs(float(_wrap(float(state0_heading) - float(tangents[0]))))
    headings[0] = float(state0_heading)
    curvature = np.zeros(len(points), dtype=np.float64)
    curvature[1:] = np.asarray(_wrap(np.diff(headings)), dtype=np.float64) / np.maximum(lengths, 1e-12)
    audit = {
        "state0_geometric_tangent_rad": float(tangents[0]),
        "state0_pose_heading_rad": float(state0_heading),
        "state0_tangent_mismatch_abs_rad": state0_tangent_mismatch,
        "max_future_declared_heading_vs_final_xy_tangent_abs_rad": float(
            np.max(np.abs(np.asarray(_wrap(headings[1:-1] - tangents[1:]), dtype=np.float64)))
        ) if len(points) > 2 else 0.0,
    }
    return headings, curvature, audit


def _target_frame_residual(point: np.ndarray, target_xy: np.ndarray) -> float:
    segment = np.diff(target_xy, axis=0)
    length = np.linalg.norm(segment, axis=1)
    starts = target_xy[:-1]
    u = np.clip(np.sum((point - starts) * segment, axis=1) / np.maximum(length**2, 1e-12), 0.0, 1.0)
    projected = starts + u[:, None] * segment
    index = int(np.argmin(np.sum((point - projected) ** 2, axis=1)))
    tangent = segment[index] / max(float(length[index]), 1e-12)
    return float(np.dot(point - projected[index], np.asarray([-tangent[1], tangent[0]])))


def _correction_profile(
    base_xy: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    current_abs_s: float,
    future_abs_s: np.ndarray,
    capture: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    start = float(capture["capture_start_abs_s"])
    end = float(capture["capture_end_abs_s"])
    if not end > start:
        raise ValueError("R2_BI_CAPTURE_END_MUST_FOLLOW_START")
    delta0 = np.asarray(current_xy, dtype=np.float64) - np.asarray(base_xy[0], dtype=np.float64)
    base_velocity = (np.asarray(base_xy[1]) - np.asarray(base_xy[0])) / DT_SECONDS
    base_speed = max(float(np.linalg.norm(base_velocity)), 1e-6)
    desired_velocity = base_speed * np.asarray([math.cos(current_heading), math.sin(current_heading)])
    current_heading_delta_velocity = desired_velocity - base_velocity
    correction = np.zeros_like(base_xy, dtype=np.float64)
    if current_abs_s >= end - 1e-12:
        audit = {
            "mode": "AT_OR_AFTER_FIXED_DEADLINE",
            "current_vector_residual_m": delta0.tolist(),
            "current_vector_residual_norm_m": float(np.linalg.norm(delta0)),
            "remaining_capture_time_s": max(0.0, end - current_abs_s),
        }
        if float(np.linalg.norm(delta0)) > float(capture["deadline_position_tolerance_m"]):
            raise CaptureInfeasible("FIXED_DEADLINE_MISSED_WITH_NONZERO_RESIDUAL", audit)
        correction[0] = delta0
        return correction, audit
    effective_start = max(start, current_abs_s)
    duration = end - effective_start
    minimum = float(capture["minimum_remaining_capture_time_s"])
    if duration < minimum and float(np.linalg.norm(delta0)) > float(capture["deadline_position_tolerance_m"]):
        raise CaptureInfeasible("INSUFFICIENT_REMAINING_TIME", {
            "remaining_capture_time_s": duration, "minimum_remaining_capture_time_s": minimum,
            "current_vector_residual_norm_m": float(np.linalg.norm(delta0)),
        })
    if current_abs_s < start:
        before = future_abs_s <= start + 1e-12
        correction[before] = delta0
        coefficients = _quintic_coefficients(delta0, np.zeros(2), np.zeros(2), end - start)
        active = (future_abs_s > start) & (future_abs_s < end)
        correction[active] = _evaluate_quintic(coefficients, future_abs_s[active] - start)
        mode = "WAIT_THEN_FIXED_DEADLINE_QUINTIC"
        initial_velocity = np.zeros(2)
    else:
        coefficients = _quintic_coefficients(delta0, current_heading_delta_velocity, np.zeros(2), end - current_abs_s)
        active = future_abs_s < end
        correction[active] = _evaluate_quintic(coefficients, future_abs_s[active] - current_abs_s)
        mode = "ONLINE_FEEDBACK_FIXED_DEADLINE_QUINTIC"
        initial_velocity = current_heading_delta_velocity
    correction[future_abs_s >= end - 1e-12] = 0.0
    return correction, {
        "mode": mode,
        "capture_start_abs_s": start,
        "capture_end_abs_s": end,
        "current_vector_residual_m": delta0.tolist(),
        "current_vector_residual_norm_m": float(np.linalg.norm(delta0)),
        "initial_correction_velocity_mps": initial_velocity.tolist(),
        "remaining_capture_time_s": end - current_abs_s,
        "denominator_zero_special_case_used": False,
    }


def kinematic_target_capture_path(
    base_xy: np.ndarray,
    target_reference_xy: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    current_speed_mps: float,
    current_abs_s: float,
    future_abs_s: np.ndarray,
    capture: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Compose morphology and feedback capture, then derive a single consistent pose trajectory."""
    base = np.asarray(base_xy, dtype=np.float64)
    target = np.asarray(target_reference_xy, dtype=np.float64)
    future = np.asarray(future_abs_s, dtype=np.float64)
    if base.shape != (len(future), 2) or target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("R2_BI_BASE_OR_TARGET_GEOMETRY_SHAPE_INVALID")
    correction, capture_audit = _correction_profile(
        base, np.asarray(current_xy, dtype=np.float64), float(current_heading), float(current_abs_s), future, capture
    )
    xy = base + correction
    xy[0] = np.asarray(current_xy, dtype=np.float64)
    heading, curvature, pose_audit = headings_and_curvature_from_xy(xy, float(current_heading))
    speed = max(float(current_speed_mps), 0.2)
    yaw_rate = curvature * speed
    lateral_acceleration = curvature * speed**2
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    nominal_step = speed * DT_SECONDS
    limits = capture["frozen_feasibility_limits"]
    actual_offsets = [_target_frame_residual(point, target) for point in xy]
    feasibility = {
        "max_abs_curvature_inv_m": float(np.max(np.abs(curvature))),
        "max_abs_yaw_rate_radps": float(np.max(np.abs(yaw_rate))),
        "max_abs_lateral_acceleration_mps2": float(np.max(np.abs(lateral_acceleration))),
        "state0_to_state1_distance_m": float(segment_lengths[0]),
        "nominal_state_step_distance_m": nominal_step,
        "state0_tangent_mismatch_abs_rad": pose_audit["state0_tangent_mismatch_abs_rad"],
        "future_heading_xy_mismatch_abs_rad": pose_audit["max_future_declared_heading_vs_final_xy_tangent_abs_rad"],
        "pass": bool(
            np.max(np.abs(curvature)) <= float(limits["curvature_inv_m_max"])
            and np.max(np.abs(yaw_rate)) <= float(limits["yaw_rate_radps_max"])
            and np.max(np.abs(lateral_acceleration)) <= float(limits["lateral_accel_mps2_max"])
            and segment_lengths[0] <= nominal_step + float(limits["state0_to_state1_distance_excess_m_max"])
            and pose_audit["state0_tangent_mismatch_abs_rad"] <= float(limits["state0_tangent_mismatch_rad_max"])
            and pose_audit["max_future_declared_heading_vs_final_xy_tangent_abs_rad"] <= float(limits["heading_xy_consistency_rad_max"])
        ),
    }
    audit = {
        **capture_audit,
        "state0_exact_current_xy": bool(np.array_equal(xy[0], np.asarray(current_xy, dtype=np.float64))),
        "state0_exact_current_heading": bool(float(heading[0]) == float(current_heading)),
        "actual_planned_target_frame_offsets_m": actual_offsets,
        "algebraic_correction_vectors_xy_m": correction.tolist(),
        "pose_consistency": pose_audit,
        "feasibility": feasibility,
        "final_xy_heading_curvature_single_source": True,
        "controller_visible_curvature_profile": curvature.tolist(),
    }
    if not feasibility["pass"]:
        raise CaptureInfeasible("FROZEN_KINEMATIC_FEASIBILITY_GATE_FAIL", audit)
    return xy, heading, curvature, audit


def behavior_progress(time_s: np.ndarray, arm: str, parameters: Mapping[str, Any]) -> np.ndarray:
    return hlc_controller_aware_progress(time_s, arm, parameters["morphology"])


def validate_parameters(parameters: Mapping[str, Any]) -> None:
    if set(parameters) != {"morphology", "capture"}:
        raise ValueError("R2_BI_PARAMETER_PARTITIONS_MUST_BE_MORPHOLOGY_AND_CAPTURE")
    forbidden = {"scenario_token", "log_id", "identity", "per_scenario"}
    if forbidden.intersection(parameters["morphology"]) or forbidden.intersection(parameters["capture"]):
        raise ValueError("R2_BI_SCENARIO_SPECIFIC_PARAMETER_FORBIDDEN")
    capture = parameters["capture"]
    required = {
        "capture_start_abs_s", "capture_end_abs_s", "minimum_remaining_capture_time_s",
        "deadline_position_tolerance_m", "frozen_feasibility_limits",
    }
    if not required.issubset(capture):
        raise ValueError(f"R2_BI_CAPTURE_PARAMETER_MISSING:{sorted(required-set(capture))}")
    if float(capture["capture_end_abs_s"]) <= float(capture["capture_start_abs_s"]):
        raise ValueError("R2_BI_CAPTURE_TIME_ORDER_INVALID")


__all__ = [
    "ARM_BASELINE", "ARM_TREATMENT", "T_DIVERGE_S", "CaptureInfeasible", "behavior_progress",
    "headings_and_curvature_from_xy", "kinematic_target_capture_path", "validate_parameters",
]
