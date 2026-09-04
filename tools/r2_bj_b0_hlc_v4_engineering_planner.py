#!/usr/bin/env python3
"""BJ-B0 HLC V4 runtime wrapper with frozen per-call fail-closed checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import HLC_BASELINE, HLC_TREATMENT  # noqa: E402
from tools.r2_a_controller_transfer_dev_planner_v1 import _planned_state_payload  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import (  # noqa: E402
    ARM_BASELINE,
    ARM_TREATMENT,
    T_DIVERGE_S,
    validate_parameters,
)
from tools.r2_bj_a_hlc_morphology_feasible_planner_v4 import _states  # noqa: E402


class B0ArchitectureViolation(RuntimeError):
    """An immutable V4 architecture gate failed; all remaining work must stop."""

    def __init__(self, codes: Sequence[str], audit: Mapping[str, Any]):
        self.codes = tuple(codes)
        self.audit = dict(audit)
        super().__init__("R2_BJ_B0_ARCHITECTURE_FAILURE:" + ",".join(self.codes))


def audit_v4_planner_call(
    current: Mapping[str, Any], states: Sequence[Mapping[str, Any]], capture: Mapping[str, Any],
    speed_mps: float, wheel_base_m: float, absolute_s: float,
    counterfactual_states: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Apply only limits already carried by the immutable V4 capture payload."""
    feasibility = capture["feasibility"]
    limits = capture.get("frozen_feasibility_limits") or {
        "curvature_inv_m_max": 0.5,
        "yaw_rate_radps_max": 1.0,
        "lateral_accel_mps2_max": 6.0,
        "state0_to_state1_distance_excess_m_max": 0.35,
        "state0_tangent_mismatch_rad_max": 0.2,
        "heading_xy_consistency_rad_max": 1e-10,
    }
    curvature = np.asarray(capture["controller_visible_curvature_profile"], dtype=np.float64)
    steering = np.arctan(float(wheel_base_m) * curvature)
    target_offsets = np.asarray(capture["actual_planned_target_frame_offsets_m"], dtype=np.float64)
    current_pose = current["rear_axle"]
    state0 = states[0]["rear_axle"]
    exact_pose = all(float(state0[key]) == float(current_pose[key]) for key in ("x", "y", "heading"))
    prediv_equal = True
    if float(absolute_s) < T_DIVERGE_S - 1e-12:
        prediv_equal = counterfactual_states is not None and list(states) == list(counterfactual_states)
    checks = {
        "curvature": float(np.max(np.abs(curvature))) <= float(limits["curvature_inv_m_max"]),
        "yaw_rate": float(np.max(np.abs(curvature))) * float(speed_mps) <= float(limits["yaw_rate_radps_max"]),
        "lateral_acceleration": float(np.max(np.abs(curvature))) * float(speed_mps) ** 2 <= float(limits["lateral_accel_mps2_max"]),
        "state0_exact_current_pose": exact_pose and bool(capture["state0_exact_current_xy"]) and bool(capture["state0_exact_current_heading"]),
        "state0_to_state1_distance_excess": float(feasibility["state0_to_state1_distance_m"]) <= float(feasibility["nominal_state_step_distance_m"]) + float(limits["state0_to_state1_distance_excess_m_max"]),
        "state0_tangent_mismatch": float(feasibility["state0_tangent_mismatch_abs_rad"]) <= float(limits["state0_tangent_mismatch_rad_max"]),
        "XY_heading_consistency": float(feasibility["future_heading_xy_mismatch_abs_rad"]) <= float(limits["heading_xy_consistency_rad_max"]),
        "target_frame_residual": abs(float(target_offsets[-1])) <= 1e-6,
        "rolling_stitching_horizon": float(capture["effective_stitching_duration_s"]) >= float(capture["minimum_stitching_horizon_s"]),
        "controller_visible_steering": bool(np.isfinite(steering).all()),
        "baseline_treatment_pre_divergence_equality": prediv_equal,
    }
    code = {
        "curvature": "CURVATURE_LIMIT", "yaw_rate": "YAW_RATE_LIMIT",
        "lateral_acceleration": "LATERAL_ACCELERATION_LIMIT",
        "state0_exact_current_pose": "STATE0_POSE_MISMATCH",
        "state0_to_state1_distance_excess": "STATE0_STEP_EXCESS",
        "state0_tangent_mismatch": "STATE0_TANGENT_MISMATCH",
        "XY_heading_consistency": "XY_HEADING_MISMATCH", "target_frame_residual": "TARGET_FRAME_RESIDUAL",
        "rolling_stitching_horizon": "STITCHING_HORIZON",
        "controller_visible_steering": "CONTROLLER_VISIBLE_STEERING_INVALID",
        "baseline_treatment_pre_divergence_equality": "PREDIVERGENCE_TRAJECTORY_MISMATCH",
    }
    failed = [code[name] for name, passed in checks.items() if not passed]
    audit = {
        "classification": "PASS" if not failed else "ARCHITECTURE_FAILURE",
        "checks": checks, "failure_codes": failed,
        "max_abs_curvature_inv_m": float(np.max(np.abs(curvature))),
        "max_abs_yaw_rate_radps": float(np.max(np.abs(curvature))) * float(speed_mps),
        "max_abs_lateral_acceleration_mps2": float(np.max(np.abs(curvature))) * float(speed_mps) ** 2,
        "controller_visible_steering_rad": steering.tolist(),
        "terminal_target_frame_offset_abs_m": abs(float(target_offsets[-1])),
        "stop_action": [] if not failed else ["STOP_CURRENT_RUN", "STOP_REMAINING_SCHEDULE"],
    }
    if failed:
        raise B0ArchitectureViolation(failed, audit)
    return audit


class R2BJB0HLCV4EngineeringPlanner(R1OfficialTechnicalSmokePlannerV3_1):
    """Immutable V4 engineering planner; telemetry and gates do not alter its trajectory."""

    def __init__(self, roster_row: Mapping[str, Any], arm: str, parameters: Mapping[str, Any], trace_dir: str, telemetry_dir: str) -> None:
        validate_parameters(parameters)
        super().__init__(roster_row, "R-HLC", HLC_BASELINE if arm == ARM_BASELINE else HLC_TREATMENT, trace_dir)
        self._development_arm = arm
        self._parameters = json.loads(json.dumps(parameters))
        self._telemetry_path = Path(telemetry_dir).expanduser().resolve() / "planner_v4_online_gate.jsonl"

    def name(self) -> str:
        return f"R2BJB0HLCV4EngineeringPlanner_{self._development_arm}"

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        if self._initialization is None:
            raise RuntimeError("R2_BJ_B0_PLANNER_NOT_INITIALIZED")
        if int(current_input.iteration.index) >= 80:
            raise RuntimeError("R2_BJ_B0_PRIMARY80_SECONDARY_PLANNER_CALL_FORBIDDEN")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        absolute = float(self._absolute_episode_clock(current_input, int(current["time_us"]))["absolute_episode_time_s"])
        force_common = absolute < T_DIVERGE_S - 1e-12
        corridor = build_hlc_route_continuous_reference_v2_3(
            self._initialization.map_api, self._row["route_roadblock_ids"], str(self._row["source_lane_id"]),
            str(self._row["target_lane_id"]), current, max(0.2, float(current["speed_mps"])) * 7.9,
        )
        states, progress, capture = _states(current, absolute, corridor, self._development_arm, self._parameters, force_common)
        counterfactual = None
        if force_common:
            other = ARM_TREATMENT if self._development_arm == ARM_BASELINE else ARM_BASELINE
            counterfactual, _, _ = _states(current, absolute, corridor, other, self._parameters, True)
        gate = audit_v4_planner_call(
            current, states, capture, float(current["speed_mps"]),
            float(ego.car_footprint.vehicle_parameters.wheel_base), absolute, counterfactual,
        )
        telemetry = {
            "schema_version": "r2_bj_b0_hlc_v4_online_gate_telemetry_v1.0",
            "iteration": int(current_input.iteration.index), "absolute_episode_time_s": absolute,
            "arm": self._development_arm, "realized_current_ego": current,
            "controller_lookahead": {"states_0_to_10": [_planned_state_payload(states, i) for i in range(11)]},
            "morphology_progress_profile": [float(value) for value in progress], "target_capture": capture,
            "online_gate": gate, "V4_parameter_sha256": hashlib.sha256(json.dumps(self._parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(telemetry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        vehicle = ego.car_footprint.vehicle_parameters
        official = [EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(float(row["rear_axle"]["x"]), float(row["rear_axle"]["y"]), float(row["rear_axle"]["heading"])),
            rear_axle_velocity_2d=StateVector2D(float(row["speed_mps"]), 0.0),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=float(ego.tire_steering_angle),
            time_point=TimePoint(int(row["time_us"])), vehicle_parameters=vehicle,
        ) for row in states]
        return InterpolatedTrajectory(official)


__all__ = ["B0ArchitectureViolation", "R2BJB0HLCV4EngineeringPlanner", "audit_v4_planner_call"]
