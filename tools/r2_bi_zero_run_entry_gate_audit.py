#!/usr/bin/env python3
"""Mandatory R2-BI zero-run controller-observability and kinematic entry gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_bi_hlc_kinematic_target_capture_generator_v3 import (  # noqa: E402
    CaptureInfeasible,
    kinematic_target_capture_path,
)


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
SPACE = R2 / "r2_bi_hlc_kinematic_capture_parameter_space_v3.0.json"
CONTRACT = R2 / "r2_bi_hlc_kinematic_capture_architecture_contract_v3.0.json"
OUT = R2 / "r2_bi_mandatory_zero_run_entry_gate_audit_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shadow(xy: np.ndarray, heading: np.ndarray, speed_mps: float) -> Dict[str, Any]:
    """Call the exact frozen LQR tracker on a synthetic pose trajectory; no simulation runner."""
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

    vehicle = get_pacifica_parameters()
    states = [EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(point[0]), float(point[1]), float(angle)),
        rear_axle_velocity_2d=StateVector2D(float(speed_mps), 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0,
        time_point=TimePoint(index * 100_000), vehicle_parameters=vehicle,
    ) for index, (point, angle) in enumerate(zip(xy, heading))]
    tracker = LQRTracker(
        q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], r_lateral=[1.0],
        discretization_time=0.1, tracking_horizon=10, jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
        stopping_proportional_gain=0.5, stopping_velocity=0.2,
    )
    trajectory = InterpolatedTrajectory(states)
    current = SimulationIteration(TimePoint(0), 0)
    initial_velocity, initial_lateral = tracker._compute_initial_velocity_and_lateral_state(current, states[0], trajectory)
    reference_velocity, curvature = tracker._compute_reference_velocity_and_curvature_profile(current, trajectory)
    result = tracker.track_trajectory(current, SimulationIteration(TimePoint(100_000), 1), states[0], trajectory)
    return {
        "initial_velocity_mps": float(initial_velocity),
        "initial_lateral_state": [float(value) for value in initial_lateral],
        "reference_velocity_at_1s_mps": float(reference_velocity),
        "curvature_profile_10": [float(value) for value in curvature],
        "acceleration_command_mps2": float(result.rear_axle_acceleration_2d.x),
        "steering_rate_command_radps": float(result.tire_steering_rate),
        "implementation": "EXACT_FROZEN_NUPLAN_LQR_TRACKER_NO_SIMULATION_RUNNER",
    }


def _corridor(kind: str, direction: int, count: int = 80, speed: float = 10.0) -> np.ndarray:
    s = np.arange(count, dtype=np.float64) * speed * 0.1
    if kind == "STRAIGHT":
        return np.column_stack((s, np.zeros_like(s)))
    if kind == "CURVED":
        radius = 80.0
        angle = s / radius
        return np.column_stack((radius * np.sin(angle), direction * radius * (1.0 - np.cos(angle))))
    u = np.clip(s / 45.0, 0.0, 1.0)
    smooth = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return np.column_stack((s, direction * 3.5 * smooth))


def _build(
    base: np.ndarray, residual_m: float, current_abs_s: float, capture: Mapping[str, Any], speed: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    tangent = math.atan2(float(base[1, 1] - base[0, 1]), float(base[1, 0] - base[0, 0]))
    normal = np.asarray([-math.sin(tangent), math.cos(tangent)])
    current = base[0] + residual_m * normal
    future = current_abs_s + np.arange(len(base), dtype=np.float64) * 0.1
    target = np.vstack((base[0] - 10.0 * np.asarray([math.cos(tangent), math.sin(tangent)]), base, base[-1] + 20.0 * np.asarray([math.cos(tangent), math.sin(tangent)])))
    return kinematic_target_capture_path(base, target, current, tangent, speed, current_abs_s, future, capture)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"R2_BI_VERSIONED_OUTPUT_EXISTS:{OUT}")
    space = json.loads(SPACE.read_text(encoding="utf-8"))
    capture = space["round0"]["capture"]
    tolerances = space["entry_gate_tolerances"]
    cases = []
    all_pass = True
    for kind in ("STRAIGHT", "CURVED", "LANE_CHANGE"):
        directions = (1,) if kind == "STRAIGHT" else (-1, 1)
        for direction in directions:
            base = _corridor(kind, direction)
            for residual in (0.0, 0.25, 0.50, -0.25, -0.50):
                xy, heading, curvature, audit = _build(base, residual, 2.0, capture)
                shadow = _shadow(xy, heading, 10.0)
                state0_identity = bool(np.array_equal(xy[0], base[0] + residual * np.asarray([0.0, 1.0]))) if kind == "STRAIGHT" else audit["state0_exact_current_xy"]
                pose = audit["pose_consistency"]
                actual_terminal = abs(float(audit["actual_planned_target_frame_offsets_m"][-1]))
                pass_case = bool(
                    state0_identity and audit["state0_exact_current_heading"] and audit["feasibility"]["pass"]
                    and pose["max_future_declared_heading_vs_final_xy_tangent_abs_rad"] <= tolerances["heading_xy_consistency_rad_max"]
                    and actual_terminal <= tolerances["actual_target_offset_terminal_abs_m_max"]
                    and abs(float(shadow["initial_lateral_state"][0])) <= 1e-12
                    and abs(float(shadow["initial_lateral_state"][1])) <= 1e-12
                )
                cases.append({
                    "corridor": kind, "target_direction": direction, "current_residual_m": residual,
                    "state0_pose_identity": state0_identity,
                    "state0_to_state1_distance_m": audit["feasibility"]["state0_to_state1_distance_m"],
                    "pose_consistency": pose, "feasibility": audit["feasibility"],
                    "actual_planned_terminal_target_offset_abs_m": actual_terminal,
                    "lqr_shadow": shadow, "pass": pass_case,
                })
                all_pass = all_pass and pass_case
    straight_zero = next(row for row in cases if row["corridor"] == "STRAIGHT" and row["current_residual_m"] == 0.0)
    straight_positive = next(row for row in cases if row["corridor"] == "STRAIGHT" and row["current_residual_m"] == 0.5)
    straight_negative = next(row for row in cases if row["corridor"] == "STRAIGHT" and row["current_residual_m"] == -0.5)
    observability = {
        "zero_residual_no_false_steering": abs(straight_zero["lqr_shadow"]["steering_rate_command_radps"]) <= tolerances["zero_residual_straight_shadow_steering_radps_max"],
        "positive_residual_nonzero_correct_direction": straight_positive["lqr_shadow"]["steering_rate_command_radps"] < -tolerances["nonzero_residual_shadow_steering_radps_min"],
        "negative_residual_nonzero_correct_direction": straight_negative["lqr_shadow"]["steering_rate_command_radps"] > tolerances["nonzero_residual_shadow_steering_radps_min"],
        "positive_command_radps": straight_positive["lqr_shadow"]["steering_rate_command_radps"],
        "negative_command_radps": straight_negative["lqr_shadow"]["steering_rate_command_radps"],
    }
    all_pass = all_pass and all(value for key, value in observability.items() if isinstance(value, bool))
    base0 = _corridor("STRAIGHT", 1)
    plan0 = _build(base0, 0.5, 2.0, capture)
    base1 = np.column_stack((np.arange(80, dtype=np.float64) * 1.0 + 1.0, np.zeros(80)))
    next_xy, next_heading = plan0[0][1], plan0[1][1]
    future1 = 2.1 + np.arange(80, dtype=np.float64) * 0.1
    target1 = np.column_stack((np.linspace(-10.0, 100.0, 500), np.zeros(500)))
    plan1 = kinematic_target_capture_path(base1, target1, next_xy, float(next_heading), 10.0, 2.1, future1, capture)
    overlap_error = float(np.max(np.linalg.norm(plan0[0][1:20] - plan1[0][:19], axis=1)))
    boundary_pass = overlap_error <= tolerances["replanning_overlap_xy_m_max"]
    all_pass = all_pass and boundary_pass
    large_residual_fail_closed = False
    large_reason = None
    try:
        _build(_corridor("STRAIGHT", 1), 4.0, 6.8, capture)
    except CaptureInfeasible as error:
        large_residual_fail_closed = True
        large_reason = error.reason
    all_pass = all_pass and large_residual_fail_closed
    audit = {
        "schema_version": "r2_bi_mandatory_zero_run_entry_gate_audit_v1",
        "status": "R2_BI_ZERO_RUN_ENTRY_GATES_PASS" if all_pass else "R2_BI_SIMULATION_NOT_AUTHORIZED",
        "scientific_simulation_calls": 0, "runner_run_calls": 0,
        "contract": {"path": str(CONTRACT.relative_to(ROOT)), "sha256": _sha(CONTRACT)},
        "parameter_space": {"path": str(SPACE.relative_to(ROOT)), "sha256": _sha(SPACE)},
        "frozen_tolerances": tolerances,
        "synthetic_cases": cases,
        "controller_observability": observability,
        "replanning_boundary": {"maximum_overlap_xy_error_m": overlap_error, "pass": boundary_pass},
        "large_unconverged_residual": {"fail_closed": large_residual_fail_closed, "reason": large_reason},
        "actual_planned_state_offsets_audited": True,
        "additive_residual_field_used_as_success_evidence": False,
        "all_mandatory_gates_pass": all_pass,
    }
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "cases": len(cases), "overlap_error_m": overlap_error, "simulation_calls": 0}, ensure_ascii=False))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
