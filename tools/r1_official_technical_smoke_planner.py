#!/usr/bin/env python3
"""Official two-arm planner for the frozen R1 Phase-B2 technical smoke.

Only the frozen HLC Option-B and TSB Option-A treatments are constructible.
The planner writes a trace for technical/context/generator verification and
does not contain selection, representation, BDD, probe, checkpoint, or RBR
logic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Type

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

from tools.r1_runtime_determinism_planner import (
    _extended_polyline,
    canonical_sha256,
    ego_payload,
    observation_payload,
    traffic_light_payload,
)
from tools.stage7l_pure_lateral_execution_planner import derive_trajectory_states, initial_state_fingerprint, quintic_blend


HLC_BASELINE = "HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE"
HLC_TREATMENT = "HLC_TREATMENT_HLC_GEN_V2_OPTION_B"
TSB_BASELINE = "TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING"
TSB_TREATMENT = "TSB_TREATMENT_TSB_GEN_V2_OPTION_A"
ALLOWED_ARMS = {"R-HLC": {HLC_BASELINE, HLC_TREATMENT}, "R-TSB": {TSB_BASELINE, TSB_TREATMENT}}
TRACE_SCHEMA = "r1_official_technical_smoke_trace_v1.0"
EPISODE_DURATION_SECONDS = 8.0
SAMPLING_TIME_SECONDS = 0.1
EPISODE_FRAME_COUNT = 80


def _episode_times() -> np.ndarray:
    """Return exactly 80 samples [0.0, 8.0) at dt=.1; never an 81-frame grid."""
    values = np.arange(0.0, EPISODE_DURATION_SECONDS, SAMPLING_TIME_SECONDS, dtype=np.float64)
    if len(values) != EPISODE_FRAME_COUNT or not np.array_equal(values, np.arange(EPISODE_FRAME_COUNT, dtype=np.float64) * SAMPLING_TIME_SECONDS):
        raise RuntimeError("frozen 80-frame / 8-second episode sampling is not exact")
    return values


def _smooth(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * quintic_blend(elapsed / duration)


def hlc_progress(relative_s: np.ndarray, arm: str) -> np.ndarray:
    """Construct only the frozen decisive baseline or HLC Option-B treatment."""
    time_s = np.asarray(relative_s, dtype=np.float64)
    progress = np.zeros_like(time_s)
    diverge = 1.1
    if arm == HLC_BASELINE:
        active = time_s >= diverge
        progress[active] = _smooth(0.0, 1.0, time_s[active] - diverge, 2.0)
        return np.clip(progress, 0.0, 1.0)
    if arm != HLC_TREATMENT:
        raise ValueError(f"unsupported HLC arm: {arm}")
    advance_end = diverge + 1.4
    hold_end = advance_end + 0.6
    retreat_end = hold_end + 1.0
    recommit_end = retreat_end + 2.4
    advance = (time_s >= diverge) & (time_s < advance_end)
    progress[advance] = _smooth(0.0, 0.38, time_s[advance] - diverge, 1.4)
    progress[(time_s >= advance_end) & (time_s < hold_end)] = 0.38
    retreat = (time_s >= hold_end) & (time_s < retreat_end)
    progress[retreat] = _smooth(0.38, 0.22, time_s[retreat] - hold_end, 1.0)
    recommit = (time_s >= retreat_end) & (time_s < recommit_end)
    progress[recommit] = _smooth(0.22, 1.0, time_s[recommit] - retreat_end, 2.4)
    progress[time_s >= recommit_end] = 1.0
    return np.clip(progress, 0.0, 1.0)


def tsb_profile(relative_s: np.ndarray, arm: str, initial_speed_mps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct only the frozen one-brake baseline or TSB Option-A treatment."""
    time_s = np.asarray(relative_s, dtype=np.float64)
    acceleration = np.zeros_like(time_s)
    diverge = 1.1
    if arm == TSB_BASELINE:
        acceleration[(time_s >= diverge) & (time_s < diverge + 0.95)] = -1.0
    elif arm == TSB_TREATMENT:
        first_end, release_end, second_end = diverge + 0.5, diverge + 1.2, diverge + 1.7
        acceleration[(time_s >= diverge) & (time_s < first_end)] = -0.9
        acceleration[(time_s >= first_end) & (time_s < release_end)] = 0.4
        acceleration[(time_s >= release_end) & (time_s < second_end)] = -0.9
    else:
        raise ValueError(f"unsupported TSB arm: {arm}")
    speed, distance = np.empty_like(time_s), np.empty_like(time_s)
    speed[0], distance[0] = max(float(initial_speed_mps), 0.2), 0.0
    for index in range(1, len(time_s)):
        delta = float(time_s[index] - time_s[index - 1])
        speed[index] = max(0.2, speed[index - 1] + acceleration[index - 1] * delta)
        distance[index] = distance[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * delta
    return distance, speed, acceleration


class R1OfficialTechnicalSmokePlanner(AbstractPlanner):
    """One frozen arm for one fresh, roster-bound official closed-loop scenario."""

    requires_scenario = True

    def __init__(self, scenario: Any, roster_path: str, runtime_family: str, smoke_arm: str, trace_dir: str) -> None:
        if runtime_family not in ALLOWED_ARMS or smoke_arm not in ALLOWED_ARMS[runtime_family]:
            raise ValueError(f"illegal frozen smoke arm: family={runtime_family}, arm={smoke_arm}")
        self._scenario = scenario
        self._family = runtime_family
        self._arm = smoke_arm
        roster = json.loads(Path(roster_path).expanduser().read_text(encoding="utf-8"))
        matches = [row for row in roster.get("entries", []) if str(row.get("family")) == runtime_family and str(row.get("scenario_token")) == str(scenario.token)]
        if len(matches) != 1:
            raise ValueError(f"scenario {scenario.token} is not uniquely frozen for {runtime_family}")
        self._entry: Dict[str, Any] = dict(matches[0])
        if smoke_arm not in set(self._entry.get("arms", [])):
            raise ValueError(f"roster does not permit arm {smoke_arm}")
        self._trace_dir = Path(trace_dir).expanduser().resolve()
        self._trace_path = self._trace_dir / "planner_trace.jsonl"
        self._initialization: Any = None
        self._initial_time_us: int | None = None
        self._compute_trajectory_runtimes: List[float] = []
        _episode_times()

    def name(self) -> str:
        return f"R1OfficialTechnicalSmokePlanner_{self._family}_{self._arm}"

    def observation_type(self) -> Type[Any]:
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        self._initialization = initialization
        actual_route = [str(value) for value in initialization.route_roadblock_ids]
        expected_route = [str(value) for value in self._entry["route_roadblock_ids"]]
        if actual_route != expected_route:
            raise ValueError("official route-roadblock sequence differs from the frozen roster")
        if self._family == "R-HLC":
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer
            source = initialization.map_api.get_map_object(str(self._entry["source_lane_id"]), SemanticMapLayer.LANE)
            target = initialization.map_api.get_map_object(str(self._entry["target_lane_id"]), SemanticMapLayer.LANE)
            if source is None or target is None:
                raise ValueError("frozen HLC source or target lane is unavailable")
            if str(self._entry["target_lane_id"]) not in {str(edge.id) for edge in source.adjacent_edges if edge is not None}:
                raise ValueError("frozen HLC target lane is no longer native-adjacent")
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        binding = {
            "schema_version": TRACE_SCHEMA,
            "scenario_token": str(self._entry["scenario_token"]), "log_id": str(self._entry["log_id"]),
            "family": self._family, "smoke_arm": self._arm, "map_name": str(self._entry["map_name"]),
            "route_roadblock_ids": actual_route, "route_roadblocks_sha256": canonical_sha256(actual_route),
            "roster_entry_sha256": canonical_sha256(self._entry),
            "episode_window": {"duration_seconds": EPISODE_DURATION_SECONDS, "dt_seconds": SAMPLING_TIME_SECONDS, "frame_count": EPISODE_FRAME_COUNT, "sample_times_sha256": canonical_sha256(_episode_times().tolist()), "convention": "np.arange(0.0,8.0,0.1): [0.0,8.0), 80 samples"},
            "float_comparison": "EXACT_CANONICAL_JSON_NO_TOLERANCE",
        }
        (self._trace_dir / "planner_binding.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _build_hlc(self, ego: Any, relative_s: np.ndarray) -> List[Any]:
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        initial = self._entry["initial_state"]
        progress_m = float(initial["initial_speed_mps"]) * relative_s
        source = _extended_polyline(self._entry["source_reference_xy"], float(self._entry["source_start_arc_m"]) + progress_m)
        target = _extended_polyline(self._entry["target_reference_xy"], float(self._entry["target_start_arc_m"]) + progress_m)
        progress = hlc_progress(relative_s, self._arm)
        xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
        states = derive_trajectory_states(xy, relative_s, ego.car_footprint.vehicle_parameters.wheel_base)
        return [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(states["x"][i]), float(states["y"][i]), float(states["heading"][i])),
                rear_axle_velocity_2d=StateVector2D(float(states["speed"][i]), 0.0),
                rear_axle_acceleration_2d=StateVector2D(float(states["longitudinal_accel"][i]), float(states["lateral_accel"][i])),
                tire_steering_angle=float(states["steering"][i]), time_point=TimePoint(int(ego.time_us + round(float(dt) * 1e6))),
                vehicle_parameters=ego.car_footprint.vehicle_parameters, angular_vel=float(states["yaw_rate"][i]), angular_accel=float(states["angular_accel"][i]),
            )
            for i, dt in enumerate(_episode_times())
        ]

    def _build_tsb(self, ego: Any, relative_s: np.ndarray) -> List[Any]:
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        initial = self._entry["initial_state"]
        distance, speed, acceleration = tsb_profile(relative_s, self._arm, float(initial["initial_speed_mps"]))
        heading = float(initial["initial_heading"])
        x = float(initial["initial_x"]) + distance * math.cos(heading)
        y = float(initial["initial_y"]) + distance * math.sin(heading)
        return [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(x[i]), float(y[i]), heading),
                rear_axle_velocity_2d=StateVector2D(float(speed[i]), 0.0), rear_axle_acceleration_2d=StateVector2D(float(acceleration[i]), 0.0),
                tire_steering_angle=0.0, time_point=TimePoint(int(ego.time_us + round(float(dt) * 1e6))),
                vehicle_parameters=ego.car_footprint.vehicle_parameters, angular_vel=0.0, angular_accel=0.0,
            )
            for i, dt in enumerate(_episode_times())
        ]

    def _append_trace(self, current_input: Any, ego: Any, trajectory: Sequence[Any]) -> None:
        record = {
            "schema_version": TRACE_SCHEMA, "iteration_index": int(current_input.iteration.index), "iteration_time_us": int(current_input.iteration.time_us),
            "initial_history_canonical": [ego_payload(state) for state in current_input.history.ego_states],
            "pre_context_raw": [observation_payload(observation, canonical=False) for observation in current_input.history.observations],
            "canonical_context": [observation_payload(observation, canonical=True) for observation in current_input.history.observations],
            "traffic_light_states": traffic_light_payload(current_input.traffic_light_data), "current_ego": ego_payload(ego),
            "planner_output_trajectory": [ego_payload(state) for state in trajectory],
        }
        record["component_hashes"] = {key: canonical_sha256(record[key]) for key in ("initial_history_canonical", "pre_context_raw", "canonical_context", "traffic_light_states", "current_ego", "planner_output_trajectory")}
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
        ego, _ = current_input.history.current_state
        if self._initial_time_us is None:
            self._initial_time_us = int(ego.time_us)
            expected = str(self._entry["initial_state"]["initial_state_fingerprint"])
            actual = initial_state_fingerprint(ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading, ego.dynamic_car_state.speed, ego.time_us)
            if actual != expected:
                raise ValueError(f"official initial state differs from frozen roster: actual={actual}, expected={expected}")
        relative = max(0.0, (int(ego.time_us) - self._initial_time_us) * 1e-6) + _episode_times()
        trajectory = self._build_hlc(ego, relative) if self._family == "R-HLC" else self._build_tsb(ego, relative)
        self._append_trace(current_input, ego, trajectory)
        return InterpolatedTrajectory(trajectory)

    def compute_trajectory(self, current_input: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.compute_planner_trajectory(current_input)
        finally:
            self._compute_trajectory_runtimes.append(time.perf_counter() - started)

    def generate_planner_report(self, clear_stats: bool = True) -> Any:
        from nuplan.planning.simulation.planner.planner_report import PlannerReport
        report = PlannerReport(compute_trajectory_runtimes=list(self._compute_trajectory_runtimes))
        if clear_stats:
            self._compute_trajectory_runtimes.clear()
        return report
