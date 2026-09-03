#!/usr/bin/env python3
"""Primary80 HLC V3 planner with one kinematically consistent controller reference."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import DT_SECONDS, WINDOW_FRAMES, _state  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import (  # noqa: E402
    HLC_BASELINE,
    HLC_TREATMENT,
    sample_native_reference_no_extrapolation,
)
from tools.r2_a_controller_transfer_dev_planner_v1 import _planned_state_payload  # noqa: E402
from tools.r2_bi_hlc_kinematic_target_capture_generator_v3 import (  # noqa: E402
    ARM_BASELINE,
    ARM_TREATMENT,
    T_DIVERGE_S,
    behavior_progress,
    kinematic_target_capture_path,
    validate_parameters,
)


def _states(
    current: Mapping[str, Any], absolute_s: float, corridor: Mapping[str, Any], arm: str,
    parameters: Mapping[str, Any], force_common: bool,
) -> Tuple[Sequence[Dict[str, Any]], np.ndarray, Mapping[str, Any]]:
    relative = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    future_absolute = absolute_s + relative
    effective_arm = ARM_BASELINE if force_common else arm
    progress = behavior_progress(future_absolute, effective_arm, parameters)
    speed = np.full(WINDOW_FRAMES, max(float(current["speed_mps"]), 0.2), dtype=np.float64)
    distance = speed * relative
    source, _ = sample_native_reference_no_extrapolation(
        corridor["source_reference_xy"], float(corridor["source_current_arc_m"]) + distance
    )
    target, _ = sample_native_reference_no_extrapolation(
        corridor["target_reference_xy"], float(corridor["target_current_arc_m"]) + distance
    )
    morphology_xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    current_xy = np.asarray([current["rear_axle"]["x"], current["rear_axle"]["y"]], dtype=np.float64)
    xy, heading, curvature, capture = kinematic_target_capture_path(
        morphology_xy, np.asarray(corridor["target_reference_xy"], dtype=np.float64), current_xy,
        float(current["rear_axle"]["heading"]), float(current["speed_mps"]), absolute_s,
        future_absolute, parameters["capture"],
    )
    start_us = int(current["time_us"])
    states = [
        _state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000)
        for i in range(WINDOW_FRAMES)
    ]
    states[0] = dict(current)
    capture = dict(capture)
    capture["derived_curvature_inv_m"] = curvature.tolist()
    return states, progress, capture


class R2BIHLCKinematicTargetCapturePlannerV3(R1OfficialTechnicalSmokePlannerV3_1):
    """Global HLC DEV-KIN planner; no identity-specific parameter adaptation."""

    def __init__(
        self, roster_row: Mapping[str, Any], arm: str, parameters: Mapping[str, Any],
        trace_dir: str, telemetry_dir: str,
    ) -> None:
        validate_parameters(parameters)
        inherited_arm = HLC_BASELINE if arm == ARM_BASELINE else HLC_TREATMENT
        super().__init__(roster_row, "R-HLC", inherited_arm, trace_dir)
        self._development_arm = arm
        self._parameters = json.loads(json.dumps(parameters))
        self._telemetry_path = Path(telemetry_dir).expanduser().resolve() / "planner_kinematic_capture.jsonl"

    def name(self) -> str:
        return f"R2BIHLCKinematicTargetCapturePlannerV3_{self._development_arm}"

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        if self._initialization is None or self._map_bridge is None:
            raise RuntimeError("R2_BI_PLANNER_NOT_INITIALIZED")
        if int(current_input.iteration.index) >= 80:
            raise RuntimeError("R2_BI_PRIMARY80_SECONDARY_PLANNER_CALL_FORBIDDEN")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        phase = self._absolute_episode_clock(current_input, int(current["time_us"]))
        absolute = float(phase["absolute_episode_time_s"])
        force_common = absolute < T_DIVERGE_S - 1e-9
        corridor = build_hlc_route_continuous_reference_v2_3(
            self._initialization.map_api, self._row["route_roadblock_ids"],
            str(self._row["source_lane_id"]), str(self._row["target_lane_id"]), current,
            max(0.2, float(current["speed_mps"])) * 7.9,
        )
        states, progress, capture = _states(
            current, absolute, corridor, self._development_arm, self._parameters, force_common
        )
        telemetry = {
            "schema_version": "r2_bi_hlc_kinematic_target_capture_planner_telemetry_v3.0",
            "iteration": int(current_input.iteration.index), "absolute_episode_time_s": absolute,
            "arm": self._development_arm, "precontext_common_trajectory_forced": force_common,
            "realized_current_ego": current,
            "controller_lookahead": {"states_0_to_10": [_planned_state_payload(states, i) for i in range(11)]},
            "behavior_progress_profile": [float(value) for value in progress],
            "target_capture": capture,
            "actual_planned_state1_target_frame_offset_m": float(capture["actual_planned_target_frame_offsets_m"][1]),
            "actual_planned_terminal_target_frame_offset_m": float(capture["actual_planned_target_frame_offsets_m"][-1]),
            "global_parameter_sha256": hashlib.sha256(
                json.dumps(self._parameters, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(telemetry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        vehicle = ego.car_footprint.vehicle_parameters
        official = [EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(
                float(row["rear_axle"]["x"]), float(row["rear_axle"]["y"]), float(row["rear_axle"]["heading"])
            ),
            rear_axle_velocity_2d=StateVector2D(float(row["speed_mps"]), 0.0),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=float(ego.tire_steering_angle),
            time_point=TimePoint(int(row["time_us"])), vehicle_parameters=vehicle,
        ) for row in states]
        return InterpolatedTrajectory(official)


__all__ = ["R2BIHLCKinematicTargetCapturePlannerV3", "_states"]
