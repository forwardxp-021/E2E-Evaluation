#!/usr/bin/env python3
"""Development-only excitation planner with passive transfer telemetry."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import (  # noqa: E402
    DT_SECONDS,
    WINDOW_FRAMES,
    _headings_from_xy,
    _offset_preserving_route_xy,
    _state,
    build_native_route_reference_v1_1,
)
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import (  # noqa: E402
    HLC_BASELINE,
    HLC_TREATMENT,
    TSB_BASELINE,
    TSB_TREATMENT,
    sample_native_reference_no_extrapolation,
)


TELEMETRY_SCHEMA = "r2_a_controller_transfer_planner_telemetry_v1.0"
CONTROLLER_LOOKAHEAD_STEPS = 10


def _quintic(value: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _smooth(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * _quintic(elapsed / duration)


def hlc_excitation_progress(time_s: np.ndarray, excitation: Mapping[str, Any]) -> np.ndarray:
    t = np.asarray(time_s, dtype=np.float64)
    result = np.zeros_like(t)
    diverge = float(excitation["diverge_s"])
    if excitation["kind"] == "MONOTONIC_REFERENCE":
        active = t >= diverge
        result[active] = _smooth(0.0, 1.0, t[active] - diverge, float(excitation["transition_duration_s"]))
        return np.clip(result, 0.0, 1.0)
    advance_duration = float(excitation["advance_duration_s"])
    advance_progress = float(excitation["advance_progress"])
    hold_duration = float(excitation["hold_duration_s"])
    retreat_duration = float(excitation["retreat_duration_s"])
    recommit_duration = float(excitation["recommit_duration_s"])
    retreat_target = advance_progress - float(excitation["retreat_depth"])
    advance_end = diverge + advance_duration
    hold_end = advance_end + hold_duration
    retreat_end = hold_end + retreat_duration
    recommit_end = retreat_end + recommit_duration
    active = (t >= diverge) & (t < advance_end)
    result[active] = _smooth(0.0, advance_progress, t[active] - diverge, advance_duration)
    result[(t >= advance_end) & (t < hold_end)] = advance_progress
    active = (t >= hold_end) & (t < retreat_end)
    result[active] = _smooth(advance_progress, retreat_target, t[active] - hold_end, retreat_duration)
    active = (t >= retreat_end) & (t < recommit_end)
    result[active] = _smooth(retreat_target, 1.0, t[active] - retreat_end, recommit_duration)
    result[t >= recommit_end] = 1.0
    return np.clip(result, 0.0, 1.0)


def tsb_excitation_acceleration(time_s: float, excitation: Mapping[str, Any]) -> float:
    t = float(time_s)
    start = float(excitation["start_s"])
    first_end = start + float(excitation["first_brake_duration_s"])
    release_end = first_end + float(excitation["release_duration_s"])
    second_end = release_end + float(excitation["second_brake_duration_s"])
    if start <= t < first_end:
        return float(excitation["first_brake_mps2"])
    if first_end <= t < release_end:
        return float(excitation["release_mps2"])
    if release_end <= t < second_end:
        return float(excitation["second_brake_mps2"])
    return 0.0


def tsb_required_forward_m(
    current_speed_mps: float, absolute_episode_time_s: float, excitation: Mapping[str, Any]
) -> float:
    """Exact no-extrapolation distance needed by the frozen 80-state plan."""
    absolute = absolute_episode_time_s + np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    speed = float(current_speed_mps)
    distance = 0.0
    for index in range(1, WINDOW_FRAMES):
        next_speed = max(
            0.2,
            round(speed + tsb_excitation_acceleration(absolute[index - 1], excitation) * DT_SECONDS, 12),
        )
        distance += 0.5 * (speed + next_speed) * DT_SECONDS
        speed = next_speed
    return float(distance + 1e-9)


def build_hlc_excitation_geometry(
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    corridor: Mapping[str, Any],
    excitation: Mapping[str, Any],
) -> Tuple[Sequence[Dict[str, Any]], np.ndarray]:
    relative = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    absolute = absolute_episode_time_s + relative
    speed = np.full(WINDOW_FRAMES, float(current_ego["speed_mps"]), dtype=np.float64)
    distance = speed * relative
    source, _ = sample_native_reference_no_extrapolation(
        corridor["source_reference_xy"], float(corridor["source_current_arc_m"]) + distance
    )
    target, _ = sample_native_reference_no_extrapolation(
        corridor["target_reference_xy"], float(corridor["target_current_arc_m"]) + distance
    )
    progress = hlc_excitation_progress(absolute, excitation)
    xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    current_xy = np.asarray(
        [current_ego["rear_axle"]["x"], current_ego["rear_axle"]["y"]], dtype=np.float64
    )
    xy += current_xy - xy[0]
    heading = _headings_from_xy(xy)
    start_us = int(current_ego["time_us"])
    states = [
        _state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000)
        for i in range(WINDOW_FRAMES)
    ]
    states[0] = dict(current_ego)
    return states, progress


def build_tsb_excitation_geometry(
    current_ego: Mapping[str, Any],
    absolute_episode_time_s: float,
    route: Mapping[str, Any],
    excitation: Mapping[str, Any],
) -> Tuple[Sequence[Dict[str, Any]], np.ndarray]:
    absolute = absolute_episode_time_s + np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    speed = np.empty(WINDOW_FRAMES, dtype=np.float64)
    distance = np.zeros(WINDOW_FRAMES, dtype=np.float64)
    command = np.zeros(WINDOW_FRAMES, dtype=np.float64)
    speed[0] = float(current_ego["speed_mps"])
    for index in range(1, WINDOW_FRAMES):
        command[index - 1] = tsb_excitation_acceleration(absolute[index - 1], excitation)
        speed[index] = max(0.2, round(float(speed[index - 1] + command[index - 1] * DT_SECONDS), 12))
        distance[index] = distance[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * DT_SECONDS
    command[-1] = tsb_excitation_acceleration(absolute[-1], excitation)
    query = float(route["current_route_arc_m"]) + distance
    xy = _offset_preserving_route_xy(route["reference_xy"], query, current_ego)
    heading = _headings_from_xy(xy)
    start_us = int(current_ego["time_us"])
    states = [
        _state(xy[i, 0], xy[i, 1], heading[i], speed[i], start_us + i * 100_000)
        for i in range(WINDOW_FRAMES)
    ]
    states[0] = dict(current_ego)
    return states, command


def _planned_state_payload(states: Sequence[Mapping[str, Any]], index: int) -> Dict[str, Any]:
    row = states[index]
    if index == 0:
        acceleration = None
        lateral_displacement = 0.0
        progress = 0.0
        curvature = None
    else:
        prior = states[index - 1]
        acceleration = (float(row["speed_mps"]) - float(prior["speed_mps"])) / DT_SECONDS
        dx = float(row["rear_axle"]["x"]) - float(states[0]["rear_axle"]["x"])
        dy = float(row["rear_axle"]["y"]) - float(states[0]["rear_axle"]["y"])
        h0 = float(states[0]["rear_axle"]["heading"])
        lateral_displacement = -math.sin(h0) * dx + math.cos(h0) * dy
        progress = math.cos(h0) * dx + math.sin(h0) * dy
        prior_heading = float(prior["rear_axle"]["heading"])
        ds = math.hypot(
            float(row["rear_axle"]["x"]) - float(prior["rear_axle"]["x"]),
            float(row["rear_axle"]["y"]) - float(prior["rear_axle"]["y"]),
        )
        curvature = float((float(row["rear_axle"]["heading"]) - prior_heading + math.pi) % (2 * math.pi) - math.pi) / max(ds, 1e-12)
    return {
        "state_index": index,
        "time_us": int(row["time_us"]),
        "rear_axle": dict(row["rear_axle"]),
        "planned_speed_mps": float(row["speed_mps"]),
        "planned_acceleration_proxy_mps2": acceleration,
        "planned_lateral_displacement_m": lateral_displacement,
        "planned_progress_m": progress,
        "planned_heading_rad": float(row["rear_axle"]["heading"]),
        "planned_curvature_inv_m": curvature,
    }


class R2AControllerTransferDevPlannerV1(R1OfficialTechnicalSmokePlannerV3_1):
    """Parameter-frozen DEV planner; never valid for confirmatory use."""

    def __init__(
        self,
        roster_row: Mapping[str, Any],
        excitation: Mapping[str, Any],
        trace_dir: str,
        telemetry_dir: str,
    ) -> None:
        family = str(roster_row["family"])
        arm = (
            HLC_BASELINE
            if family == "R-HLC" and excitation["kind"] == "MONOTONIC_REFERENCE"
            else HLC_TREATMENT
            if family == "R-HLC"
            else TSB_BASELINE
            if excitation["kind"] == "SINGLE_BRAKE_REFERENCE"
            else TSB_TREATMENT
        )
        super().__init__(roster_row, family, arm, trace_dir)
        self._excitation = dict(excitation)
        self._telemetry_dir = Path(telemetry_dir).expanduser().resolve()
        self._telemetry_path = self._telemetry_dir / "planner_transfer.jsonl"

    @property
    def planner_telemetry_path(self) -> Path:
        return self._telemetry_path

    def name(self) -> str:
        return f"R2AControllerTransferDevPlannerV1_{self._family}_{self._excitation['excitation_id']}"

    def _write_telemetry(
        self,
        current_input: Any,
        current: Mapping[str, Any],
        phase: Mapping[str, Any],
        states: Sequence[Mapping[str, Any]],
        command: np.ndarray,
    ) -> None:
        planned = [_planned_state_payload(states, i) for i in range(CONTROLLER_LOOKAHEAD_STEPS + 1)]
        row = {
            "schema_version": TELEMETRY_SCHEMA,
            "development_only": True,
            "iteration": int(current_input.iteration.index),
            "absolute_episode_time_s": float(phase["absolute_episode_time_s"]),
            "realized_current_ego": current,
            "excitation_id": self._excitation["excitation_id"],
            "family": self._family,
            "controller_lookahead": {
                "discretization_time_s": 0.1,
                "tracking_horizon_steps": CONTROLLER_LOOKAHEAD_STEPS,
                "states_0_to_10": planned,
            },
            "full_planned_speed_mps": [float(state["speed_mps"]) for state in states],
            "full_command_profile": [float(value) for value in command],
            "full_command_profile_kind": "HLC_PROGRESS" if self._family == "R-HLC" else "TSB_ACCELERATION_MPS2",
            "controller_output_source": "SEPARATE_TWO_STAGE_LQR_PASSIVE_WRAPPER",
        }
        self._telemetry_dir.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        if self._initialization is None or self._map_bridge is None:
            raise RuntimeError("planner is not initialized")
        if int(current_input.iteration.index) >= 80:
            raise RuntimeError("R2_A_PRIMARY80_SECONDARY_PLANNER_CALL_FORBIDDEN")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        phase = self._absolute_episode_clock(current_input, int(current["time_us"]))
        absolute = float(phase["absolute_episode_time_s"])
        if self._family == "R-HLC":
            corridor = build_hlc_route_continuous_reference_v2_3(
                self._initialization.map_api,
                self._row["route_roadblock_ids"],
                str(self._row["source_lane_id"]),
                str(self._row["target_lane_id"]),
                current,
                max(0.2, float(current["speed_mps"])) * 7.9,
            )
            states, command = build_hlc_excitation_geometry(current, absolute, corridor, self._excitation)
        else:
            route = build_native_route_reference_v1_1(
                self._initialization.map_api,
                self._row["route_roadblock_ids"],
                current,
                tsb_required_forward_m(float(current["speed_mps"]), absolute, self._excitation),
            )
            states, command = build_tsb_excitation_geometry(current, absolute, route, self._excitation)
        self._write_telemetry(current_input, current, phase, states, command)
        parameters = ego.car_footprint.vehicle_parameters
        official = [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(
                    float(item["rear_axle"]["x"]),
                    float(item["rear_axle"]["y"]),
                    float(item["rear_axle"]["heading"]),
                ),
                rear_axle_velocity_2d=StateVector2D(float(item["speed_mps"]), 0.0),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(int(item["time_us"])),
                vehicle_parameters=parameters,
            )
            for item in states
        ]
        return InterpolatedTrajectory(official)


__all__ = [
    "CONTROLLER_LOOKAHEAD_STEPS",
    "R2AControllerTransferDevPlannerV1",
    "TELEMETRY_SCHEMA",
    "build_hlc_excitation_geometry",
    "build_tsb_excitation_geometry",
    "hlc_excitation_progress",
    "tsb_excitation_acceleration",
    "tsb_required_forward_m",
]
