#!/usr/bin/env python3
"""Primary80 planner that applies one global R2-B G_R2 parameter set."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import (  # noqa: E402
    DT_SECONDS, WINDOW_FRAMES, _headings_from_xy, _offset_preserving_route_xy,
    _state, build_native_route_reference_v1_1,
)
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import (  # noqa: E402
    HLC_BASELINE, HLC_TREATMENT, TSB_BASELINE, TSB_TREATMENT,
    sample_native_reference_no_extrapolation,
)
from tools.r2_a_controller_transfer_dev_planner_v1 import _planned_state_payload  # noqa: E402
from tools.r2_b_controller_aware_generator_v1 import (  # noqa: E402
    ARM_BASELINE, ARM_TREATMENT, T_DIVERGE_S, hlc_controller_aware_progress,
    tsb_controller_aware_acceleration, validate_global_parameters,
)


def _tsb_required_distance(speed_mps: float, absolute_s: float, arm: str, params: Mapping[str, Any]) -> float:
    speed = float(speed_mps)
    distance = 0.0
    for index in range(1, WINDOW_FRAMES):
        time_s = absolute_s + (index - 1) * DT_SECONDS
        next_speed = max(0.2, speed + tsb_controller_aware_acceleration(time_s, arm, params) * DT_SECONDS)
        distance += 0.5 * (speed + next_speed) * DT_SECONDS
        speed = next_speed
    return distance + 1e-9


def _hlc_states(
    current: Mapping[str, Any], absolute_s: float, corridor: Mapping[str, Any], arm: str,
    params: Mapping[str, Any], force_common_precontext: bool,
) -> Tuple[Sequence[Dict[str, Any]], np.ndarray]:
    relative = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    absolute = absolute_s + relative
    effective_arm = ARM_BASELINE if force_common_precontext else arm
    progress = hlc_controller_aware_progress(absolute, effective_arm, params)
    speed = np.full(WINDOW_FRAMES, float(current["speed_mps"]), dtype=np.float64)
    distance = speed * relative
    source, _ = sample_native_reference_no_extrapolation(
        corridor["source_reference_xy"], float(corridor["source_current_arc_m"]) + distance
    )
    target, _ = sample_native_reference_no_extrapolation(
        corridor["target_reference_xy"], float(corridor["target_current_arc_m"]) + distance
    )
    xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    origin = np.asarray([current["rear_axle"]["x"], current["rear_axle"]["y"]], dtype=np.float64)
    xy += origin - xy[0]
    heading = _headings_from_xy(xy)
    start_us = int(current["time_us"])
    states = [_state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000) for i in range(WINDOW_FRAMES)]
    states[0] = dict(current)
    return states, progress


def _tsb_states(
    current: Mapping[str, Any], absolute_s: float, route: Mapping[str, Any], arm: str,
    params: Mapping[str, Any], force_common_precontext: bool,
) -> Tuple[Sequence[Dict[str, Any]], np.ndarray]:
    effective_arm = ARM_BASELINE if force_common_precontext else arm
    absolute = absolute_s + np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    speed = np.empty(WINDOW_FRAMES, dtype=np.float64)
    distance = np.zeros(WINDOW_FRAMES, dtype=np.float64)
    command = np.zeros(WINDOW_FRAMES, dtype=np.float64)
    speed[0] = float(current["speed_mps"])
    for index in range(1, WINDOW_FRAMES):
        command[index - 1] = tsb_controller_aware_acceleration(absolute[index - 1], effective_arm, params)
        speed[index] = max(0.2, speed[index - 1] + command[index - 1] * DT_SECONDS)
        distance[index] = distance[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * DT_SECONDS
    command[-1] = tsb_controller_aware_acceleration(absolute[-1], effective_arm, params)
    xy = _offset_preserving_route_xy(
        route["reference_xy"], float(route["current_route_arc_m"]) + distance, current
    )
    heading = _headings_from_xy(xy)
    start_us = int(current["time_us"])
    states = [_state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000) for i in range(WINDOW_FRAMES)]
    states[0] = dict(current)
    return states, command


class R2BControllerAwarePlannerV1(R1OfficialTechnicalSmokePlannerV3_1):
    """One shared parameter set per family/round; never token-conditioned."""

    def __init__(
        self, roster_row: Mapping[str, Any], family: str, arm: str,
        parameters: Mapping[str, Any], trace_dir: str, telemetry_dir: str,
    ) -> None:
        validate_global_parameters(family, parameters)
        inherited_arm = (
            HLC_BASELINE if family == "R-HLC" and arm == ARM_BASELINE
            else HLC_TREATMENT if family == "R-HLC"
            else TSB_BASELINE if arm == ARM_BASELINE
            else TSB_TREATMENT
        )
        super().__init__(roster_row, family, inherited_arm, trace_dir)
        self._calibration_arm = arm
        self._global_parameters = dict(parameters)
        self._telemetry_path = Path(telemetry_dir).expanduser().resolve() / "planner_transfer.jsonl"

    def name(self) -> str:
        return f"R2BControllerAwarePlannerV1_{self._family}_{self._calibration_arm}"

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        if self._initialization is None or self._map_bridge is None:
            raise RuntimeError("R2_B_PLANNER_NOT_INITIALIZED")
        if int(current_input.iteration.index) >= 80:
            raise RuntimeError("R2_B_PRIMARY80_SECONDARY_PLANNER_CALL_FORBIDDEN")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        phase = self._absolute_episode_clock(current_input, int(current["time_us"]))
        absolute = float(phase["absolute_episode_time_s"])
        # Entire future trajectory is shared before divergence, preventing preview leakage.
        force_common = absolute < T_DIVERGE_S - 1e-9
        if self._family == "R-HLC":
            corridor = build_hlc_route_continuous_reference_v2_3(
                self._initialization.map_api, self._row["route_roadblock_ids"],
                str(self._row["source_lane_id"]), str(self._row["target_lane_id"]), current,
                max(0.2, float(current["speed_mps"])) * 7.9,
            )
            states, command = _hlc_states(current, absolute, corridor, self._calibration_arm, self._global_parameters, force_common)
        else:
            effective_arm = ARM_BASELINE if force_common else self._calibration_arm
            route = build_native_route_reference_v1_1(
                self._initialization.map_api, self._row["route_roadblock_ids"], current,
                _tsb_required_distance(float(current["speed_mps"]), absolute, effective_arm, self._global_parameters),
            )
            states, command = _tsb_states(current, absolute, route, self._calibration_arm, self._global_parameters, force_common)
        telemetry = {
            "schema_version": "r2_b_controller_aware_planner_telemetry_v1.0",
            "iteration": int(current_input.iteration.index),
            "absolute_episode_time_s": absolute,
            "family": self._family,
            "arm": self._calibration_arm,
            "precontext_common_trajectory_forced": force_common,
            "realized_current_ego": current,
            "controller_lookahead": {"states_0_to_10": [_planned_state_payload(states, i) for i in range(11)]},
            "full_planned_speed_mps": [float(row["speed_mps"]) for row in states],
            "full_command_profile": [float(value) for value in command],
            "global_parameter_sha256": __import__("hashlib").sha256(
                json.dumps(self._global_parameters, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(telemetry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        vehicle = ego.car_footprint.vehicle_parameters
        official = [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(row["rear_axle"]["x"]), float(row["rear_axle"]["y"]), float(row["rear_axle"]["heading"])),
                rear_axle_velocity_2d=StateVector2D(float(row["speed_mps"]), 0.0),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0,
                time_point=TimePoint(int(row["time_us"])), vehicle_parameters=vehicle,
            )
            for row in states
        ]
        return InterpolatedTrajectory(official)


__all__ = ["R2BControllerAwarePlannerV1"]
