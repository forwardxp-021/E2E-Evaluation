#!/usr/bin/env python3
"""Bound-runtime baseline-only planner used by the R1 determinism validation.

The planner is intentionally narrow: it reads only the frozen four-row roster,
uses one frozen baseline arm for each family, and writes canonical per-step
trace records.  It has no treatment, outcome, representation, BDD, probe, or
RBR capability.
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

from tools.stage7l_pure_lateral_execution_planner import (
    derive_trajectory_states,
    initial_state_fingerprint,
    quintic_blend,
)


HLC_BASELINE_ID = "HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE"
TSB_BASELINE_ID = "TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING"
TRACE_SCHEMA = "r1_runtime_determinism_trace_v1.0"


def canonical_sha256(value: Any) -> str:
    """Hash one stable, exact JSON representation; no numeric tolerance is used."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _attribute_payload(attribute: Any) -> Dict[str, float | None]:
    return {
        "x": _number(getattr(attribute, "x", None)),
        "y": _number(getattr(attribute, "y", None)),
    }


def ego_payload(ego: Any) -> Dict[str, Any]:
    """Serialize only explicit physical state fields in native float precision."""
    dynamic = ego.dynamic_car_state
    return {
        "time_us": int(ego.time_us),
        "rear_axle": {
            "x": float(ego.rear_axle.x),
            "y": float(ego.rear_axle.y),
            "heading": float(ego.rear_axle.heading),
        },
        "speed_mps": float(dynamic.speed),
        "rear_axle_velocity_2d": _attribute_payload(getattr(dynamic, "rear_axle_velocity_2d", None)),
        "rear_axle_acceleration_2d": _attribute_payload(getattr(dynamic, "rear_axle_acceleration_2d", None)),
        "tire_steering_angle": _number(getattr(ego, "tire_steering_angle", None)),
        "angular_velocity": _number(getattr(dynamic, "angular_velocity", None)),
        "angular_acceleration": _number(getattr(dynamic, "angular_acceleration", None)),
    }


def tracked_object_payload(item: Any) -> Dict[str, Any]:
    box = item.box
    center = box.center
    metadata = item.metadata
    velocity = getattr(item, "velocity", None)
    return {
        "token": str(getattr(item, "token", "")),
        "track_token": None if getattr(item, "track_token", None) is None else str(item.track_token),
        "track_id": getattr(metadata, "track_id", None),
        "timestamp_us": int(getattr(metadata, "timestamp_us", 0)),
        "type": str(getattr(getattr(item, "tracked_object_type", None), "fullname", getattr(item, "tracked_object_type", ""))),
        "center": {"x": float(center.x), "y": float(center.y), "heading": float(center.heading)},
        "box": {"length": float(box.length), "width": float(box.width), "height": float(box.height)},
        "velocity": _attribute_payload(velocity),
    }


def observation_payload(observation: Any, canonical: bool) -> List[Dict[str, Any]]:
    tracked = getattr(getattr(observation, "tracked_objects", None), "tracked_objects", [])
    result = [tracked_object_payload(item) for item in tracked]
    if canonical:
        result.sort(key=lambda row: (row["type"], str(row["track_token"]), row["token"], row["timestamp_us"]))
    return result


def traffic_light_payload(lights: Sequence[Any] | None) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for light in lights or []:
        status = getattr(light, "status", None)
        result.append(
            {
                "lane_connector_id": int(getattr(light, "lane_connector_id", -1)),
                "status": str(getattr(status, "name", status)),
                "timestamp_us": int(getattr(light, "timestamp", 0)),
            }
        )
    result.sort(key=lambda row: (row["lane_connector_id"], row["timestamp_us"], row["status"]))
    return result


def _extended_polyline(points: Sequence[Sequence[float]], arc_m: np.ndarray) -> np.ndarray:
    """Interpolate a reference and extend its terminal tangent if needed.

    Extension is deterministic and prevents a finite local lane reference from
    becoming a runtime-dependent planner failure after the target transition is
    complete.  It does not change the frozen HLC transition schedule.
    """
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError("frozen lane reference must be finite [N,2] with N>=2")
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(segment <= 1e-9):
        keep = np.r_[True, segment > 1e-9]
        xy = xy[keep]
        segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if len(xy) < 2 or np.any(segment <= 1e-9):
        raise ValueError("frozen lane reference has no usable segment")
    arc = np.r_[0.0, np.cumsum(segment)]
    q = np.asarray(arc_m, dtype=np.float64)
    result = np.column_stack((np.interp(np.minimum(q, arc[-1]), arc, xy[:, 0]), np.interp(np.minimum(q, arc[-1]), arc, xy[:, 1])))
    beyond = q > arc[-1]
    if np.any(beyond):
        tangent = (xy[-1] - xy[-2]) / segment[-1]
        result[beyond] = xy[-1] + (q[beyond] - arc[-1])[:, None] * tangent
    return result


def _baseline_hlc_progress(relative_s: np.ndarray) -> np.ndarray:
    """Frozen decisive, monotonic baseline: common prefix then one C2 transition."""
    progress = np.zeros_like(relative_s, dtype=np.float64)
    active = relative_s >= 1.1
    progress[active] = quintic_blend((relative_s[active] - 1.1) / 2.0)
    return np.clip(progress, 0.0, 1.0)


def _braking_displacement(initial_speed_mps: float, relative_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen single continuous -1.0 m/s² braking for 0.95 seconds after 1.1s."""
    time = np.asarray(relative_s, dtype=np.float64)
    acceleration = np.where((time >= 1.1) & (time < 2.05), -1.0, 0.0)
    speed = np.empty_like(time)
    distance = np.empty_like(time)
    speed[0] = max(float(initial_speed_mps), 0.2)
    distance[0] = 0.0
    for index in range(1, len(time)):
        dt = float(time[index] - time[index - 1])
        speed[index] = max(0.2, speed[index - 1] + acceleration[index - 1] * dt)
        distance[index] = distance[index - 1] + 0.5 * (speed[index - 1] + speed[index]) * dt
    return distance, speed, acceleration


class R1RuntimeDeterminismPlanner:
    """Official nuPlan external planner restricted to the two approved baselines."""

    requires_scenario = True

    def __init__(self, scenario: Any, roster_path: str, runtime_family: str, trace_dir: str, horizon_seconds: float = 4.5, sampling_time: float = 0.1) -> None:
        if runtime_family not in ("R-HLC", "R-TSB"):
            raise ValueError(f"unsupported runtime determinism family: {runtime_family}")
        self._scenario = scenario
        self._family = runtime_family
        self._horizon_seconds = float(horizon_seconds)
        self._sampling_time = float(sampling_time)
        if self._horizon_seconds <= 0 or self._sampling_time <= 0:
            raise ValueError("horizon_seconds and sampling_time must be positive")
        payload = json.loads(Path(roster_path).expanduser().read_text(encoding="utf-8"))
        entries = {
            (str(row["family"]), str(row["scenario_token"])): row
            for row in payload.get("entries", [])
        }
        key = (runtime_family, str(scenario.token))
        if key not in entries:
            raise ValueError(f"scenario {scenario.token} is not in the frozen {runtime_family} runtime roster")
        self._entry = entries[key]
        expected_arm = HLC_BASELINE_ID if runtime_family == "R-HLC" else TSB_BASELINE_ID
        if str(self._entry.get("runtime_arm")) != expected_arm:
            raise ValueError(f"frozen roster arm is not the permitted baseline: {self._entry.get('runtime_arm')}")
        self._trace_dir = Path(trace_dir).expanduser().resolve()
        self._trace_path = self._trace_dir / "planner_trace.jsonl"
        self._initialization: Any = None
        self._initial_time_us: int | None = None
        self._trace_header_written = False
        self._compute_trajectory_runtimes: List[float] = []

    def name(self) -> str:
        return f"R1RuntimeDeterminismPlanner_{self._family}"

    def observation_type(self) -> Type[Any]:
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        self._initialization = initialization
        actual_route = [str(value) for value in initialization.route_roadblock_ids]
        if self._family == "R-HLC":
            expected_route = [str(value) for value in self._entry["route_roadblock_ids"]]
            if actual_route != expected_route:
                raise ValueError("official HLC route-roadblock sequence differs from frozen roster")
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer
            source = initialization.map_api.get_map_object(str(self._entry["source_lane_id"]), SemanticMapLayer.LANE)
            target = initialization.map_api.get_map_object(str(self._entry["target_lane_id"]), SemanticMapLayer.LANE)
            if source is None or target is None:
                raise ValueError("frozen HLC source or target lane is unavailable in the official map")
            adjacent = {str(edge.id) for edge in source.adjacent_edges if edge is not None}
            if str(self._entry["target_lane_id"]) not in adjacent:
                raise ValueError("frozen HLC target lane is no longer native-adjacent to the source lane")
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": TRACE_SCHEMA,
            "scenario_token": str(self._entry["scenario_token"]),
            "log_id": str(self._entry["log_id"]),
            "family": self._family,
            "runtime_arm": self._entry["runtime_arm"],
            "map_name": self._entry["map_name"],
            "route_roadblock_ids": actual_route,
            "route_roadblocks_sha256": canonical_sha256(actual_route),
            "roster_entry_sha256": canonical_sha256(self._entry),
            "float_comparison": "EXACT_CANONICAL_JSON_NO_TOLERANCE",
        }
        (self._trace_dir / "planner_binding.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _build_hlc_trajectory(self, ego: Any, global_relative_s: np.ndarray) -> List[Any]:
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        progress_m = float(self._entry["initial_state"]["initial_speed_mps"]) * global_relative_s
        source = _extended_polyline(self._entry["source_reference_xy"], float(self._entry["source_start_arc_m"]) + progress_m)
        target = _extended_polyline(self._entry["target_reference_xy"], float(self._entry["target_start_arc_m"]) + progress_m)
        p = _baseline_hlc_progress(global_relative_s)
        xy = source * (1.0 - p[:, None]) + target * p[:, None]
        states = derive_trajectory_states(xy, global_relative_s, ego.car_footprint.vehicle_parameters.wheel_base)
        future = np.arange(0.0, self._horizon_seconds + self._sampling_time * 0.5, self._sampling_time, dtype=np.float64)
        return [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(states["x"][index]), float(states["y"][index]), float(states["heading"][index])),
                rear_axle_velocity_2d=StateVector2D(float(states["speed"][index]), 0.0),
                rear_axle_acceleration_2d=StateVector2D(float(states["longitudinal_accel"][index]), float(states["lateral_accel"][index])),
                tire_steering_angle=float(states["steering"][index]),
                time_point=TimePoint(int(ego.time_us + round(float(dt_s) * 1e6))),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
                angular_vel=float(states["yaw_rate"][index]),
                angular_accel=float(states["angular_accel"][index]),
            )
            for index, dt_s in enumerate(future)
        ]

    def _build_tsb_trajectory(self, ego: Any, global_relative_s: np.ndarray) -> List[Any]:
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        initial = self._entry["initial_state"]
        distance, speed, acceleration = _braking_displacement(float(initial["initial_speed_mps"]), global_relative_s)
        heading = float(initial["initial_heading"])
        x = float(initial["initial_x"]) + distance * math.cos(heading)
        y = float(initial["initial_y"]) + distance * math.sin(heading)
        future = np.arange(0.0, self._horizon_seconds + self._sampling_time * 0.5, self._sampling_time, dtype=np.float64)
        return [
            EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(x[index]), float(y[index]), heading),
                rear_axle_velocity_2d=StateVector2D(float(speed[index]), 0.0),
                rear_axle_acceleration_2d=StateVector2D(float(acceleration[index]), 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(int(ego.time_us + round(float(dt_s) * 1e6))),
                vehicle_parameters=ego.car_footprint.vehicle_parameters,
                angular_vel=0.0,
                angular_accel=0.0,
            )
            for index, dt_s in enumerate(future)
        ]

    def _append_trace(self, current_input: Any, ego: Any, trajectory: Sequence[Any]) -> None:
        history_ego = [ego_payload(state) for state in current_input.history.ego_states]
        raw_context = [observation_payload(observation, canonical=False) for observation in current_input.history.observations]
        canonical_context = [observation_payload(observation, canonical=True) for observation in current_input.history.observations]
        record = {
            "schema_version": TRACE_SCHEMA,
            "iteration_index": int(current_input.iteration.index),
            "iteration_time_us": int(current_input.iteration.time_us),
            "initial_history_canonical": history_ego,
            "pre_context_raw": raw_context,
            "canonical_context": canonical_context,
            "traffic_light_states": traffic_light_payload(current_input.traffic_light_data),
            "current_ego": ego_payload(ego),
            "planner_output_trajectory": [ego_payload(state) for state in trajectory],
        }
        record["component_hashes"] = {
            key: canonical_sha256(record[key])
            for key in (
                "initial_history_canonical", "pre_context_raw", "canonical_context", "traffic_light_states",
                "current_ego", "planner_output_trajectory",
            )
        }
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
        ego, _ = current_input.history.current_state
        if self._initial_time_us is None:
            self._initial_time_us = int(ego.time_us)
            expected = str(self._entry["initial_state"]["initial_state_fingerprint"])
            actual = initial_state_fingerprint(
                ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading, ego.dynamic_car_state.speed, ego.time_us
            )
            if actual != expected:
                raise ValueError(f"official initial ego state does not match frozen roster: actual={actual}, expected={expected}")
        assert self._initial_time_us is not None
        now_s = max(0.0, (int(ego.time_us) - self._initial_time_us) * 1e-6)
        future = np.arange(0.0, self._horizon_seconds + self._sampling_time * 0.5, self._sampling_time, dtype=np.float64)
        global_relative_s = now_s + future
        trajectory = (
            self._build_hlc_trajectory(ego, global_relative_s)
            if self._family == "R-HLC"
            else self._build_tsb_trajectory(ego, global_relative_s)
        )
        self._append_trace(current_input, ego, trajectory)
        return InterpolatedTrajectory(trajectory)

    def compute_trajectory(self, current_input: Any) -> Any:
        """Provide nuPlan's required timed planner entry point."""
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
