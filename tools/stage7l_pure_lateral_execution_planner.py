#!/usr/bin/env python3
"""Stage7L external planner with a dose-isolated lateral execution channel.

The canonical longitudinal coordinate is route progress on the frozen source-lane
reference.  A dose changes only the quintic blend from the source reference to
the frozen adjacent target reference.  This module intentionally has no BDD or
embedding dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Type

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


DOSE_TRANSITION_LENGTH_M: Dict[str, float] = {
    "dose0": 60.0,
    "dose25": 54.0,
    "dose50": 48.0,
    "dose75": 42.0,
    "dose100": 36.0,
}
SMOKE_PARAMETER_STATUS = "A2_SMOKE_ONLY_NOT_FROZEN_FOR_CONFIRMATION"


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def quintic_blend(u: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def quintic_blend_d1(u: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return 30.0 * x**2 - 60.0 * x**3 + 30.0 * x**4


def quintic_blend_d2(u: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return 60.0 * x - 180.0 * x**2 + 120.0 * x**3


@dataclass(frozen=True)
class FrozenLaneChangeManeuver:
    scenario_token: str
    log_name: str
    db_file: str
    initial_state_fingerprint: str
    initial_x: float
    initial_y: float
    initial_heading: float
    initial_speed_mps: float
    source_lane_id: str
    target_lane_id: str
    source_roadblock_id: str
    target_roadblock_id: str
    direction: str
    route_roadblock_ids: Tuple[str, ...]
    route_fingerprint: str
    trigger_s_route_m: float
    source_start_arc_m: float
    target_start_arc_m: float
    nominal_lane_width_m: float
    horizon_s: float
    background_mode: str
    background_agent_model: str
    background_config_sha256: str
    source_reference_xy: Tuple[Tuple[float, float], ...]
    target_reference_xy: Tuple[Tuple[float, float], ...]
    planner_profile_ids: Tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "FrozenLaneChangeManeuver":
        values = dict(row)
        for key in ("route_roadblock_ids", "planner_profile_ids"):
            values[key] = tuple(str(x) for x in values[key])
        for key in ("source_reference_xy", "target_reference_xy"):
            values[key] = tuple((float(p[0]), float(p[1])) for p in values[key])
        return cls(**values)

    def dose_invariant_payload(self) -> Dict[str, Any]:
        return asdict(self)


class CanonicalLongitudinalProgressGenerator:
    """Dose-blind longitudinal progress on the canonical source reference."""

    def __init__(self, initial_speed_mps: float, target_speed_mps: float, accel_limit_mps2: float) -> None:
        if initial_speed_mps < 0 or target_speed_mps <= 0 or accel_limit_mps2 <= 0:
            raise ValueError("invalid canonical longitudinal parameters")
        self.initial_speed_mps = float(initial_speed_mps)
        self.target_speed_mps = float(target_speed_mps)
        self.accel_limit_mps2 = float(accel_limit_mps2)

    def sample(self, relative_time_s: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.asarray(relative_time_s, dtype=np.float64)
        if t.ndim != 1 or np.any(t < 0) or np.any(np.diff(t) < 0):
            raise ValueError("relative_time_s must be a nonnegative monotonic vector")
        delta = self.target_speed_mps - self.initial_speed_mps
        accel = math.copysign(self.accel_limit_mps2, delta) if abs(delta) > 1e-12 else 0.0
        ramp_s = abs(delta) / self.accel_limit_mps2 if accel else 0.0
        ramp_t = np.minimum(t, ramp_s)
        speed = self.initial_speed_mps + accel * ramp_t
        progress = self.initial_speed_mps * ramp_t + 0.5 * accel * ramp_t**2
        after = np.maximum(t - ramp_s, 0.0)
        progress += self.target_speed_mps * after
        acceleration = np.where(t < ramp_s, accel, 0.0)
        return progress, speed, acceleration

    def fingerprint(self) -> str:
        return canonical_json_sha256(
            {
                "type": self.__class__.__name__,
                "initial_speed_mps": self.initial_speed_mps,
                "target_speed_mps": self.target_speed_mps,
                "accel_limit_mps2": self.accel_limit_mps2,
            }
        )


def polyline_arclength(xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"polyline must be finite [N,2] with N>=2, got {points.shape}")
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(seg <= 1e-9):
        keep = np.r_[True, seg > 1e-9]
        points = points[keep]
        if len(points) < 2:
            raise ValueError("polyline has no positive-length segment")
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(seg)]


def interpolate_polyline(xy: np.ndarray, arc_m: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    s = polyline_arclength(points)
    query = np.asarray(arc_m, dtype=np.float64)
    if np.any(query < -1e-9) or np.any(query > s[-1] + 1e-9):
        raise ValueError(f"route progress exceeds frozen reference: [{query.min()}, {query.max()}] vs {s[-1]}")
    q = np.clip(query, 0.0, s[-1])
    return np.column_stack((np.interp(q, s, points[:, 0]), np.interp(q, s, points[:, 1])))


def build_lateral_positions(
    maneuver: FrozenLaneChangeManeuver, progress_m: np.ndarray, transition_length_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    if transition_length_m <= 0:
        raise ValueError("transition_length_m must be positive")
    source_arc = maneuver.source_start_arc_m + progress_m
    target_arc = maneuver.target_start_arc_m + progress_m
    source = interpolate_polyline(np.asarray(maneuver.source_reference_xy), source_arc)
    target = interpolate_polyline(np.asarray(maneuver.target_reference_xy), target_arc)
    u = (progress_m - maneuver.trigger_s_route_m) / float(transition_length_m)
    weight = quintic_blend(u)
    return source * (1.0 - weight[:, None]) + target * weight[:, None], weight


def derive_trajectory_states(position_xy: np.ndarray, time_s: np.ndarray, wheel_base_m: float) -> Dict[str, np.ndarray]:
    xy = np.asarray(position_xy, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    if xy.shape != (len(t), 2) or len(t) < 5 or np.any(np.diff(t) <= 0):
        raise ValueError("trajectory must be [T,2], T>=5, with strictly increasing time")
    edge_order = 2
    vx = np.gradient(xy[:, 0], t, edge_order=edge_order)
    vy = np.gradient(xy[:, 1], t, edge_order=edge_order)
    ax = np.gradient(vx, t, edge_order=edge_order)
    ay = np.gradient(vy, t, edge_order=edge_order)
    speed = np.hypot(vx, vy)
    heading_unwrapped = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(heading_unwrapped, t, edge_order=edge_order)
    angular_accel = np.gradient(yaw_rate, t, edge_order=edge_order)
    curvature = np.divide(yaw_rate, speed, out=np.zeros_like(speed), where=speed > 0.2)
    steering = np.arctan(wheel_base_m * curvature)
    longitudinal_accel = ax * np.cos(heading_unwrapped) + ay * np.sin(heading_unwrapped)
    lateral_accel = -ax * np.sin(heading_unwrapped) + ay * np.cos(heading_unwrapped)
    values = {
        "x": xy[:, 0], "y": xy[:, 1], "heading": wrap_angle(heading_unwrapped),
        "speed": speed, "longitudinal_accel": longitudinal_accel,
        "lateral_accel": lateral_accel, "yaw_rate": yaw_rate,
        "angular_accel": angular_accel, "curvature": curvature, "steering": steering,
        "vx_global": vx, "vy_global": vy, "ax_global": ax, "ay_global": ay,
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("derived trajectory state contains non-finite values")
    return values


def dynamic_consistency_audit(position_xy: np.ndarray, time_s: np.ndarray, states: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    dt = np.diff(time_s)
    dx = np.gradient(position_xy[:, 0], time_s, edge_order=2)
    dy = np.gradient(position_xy[:, 1], time_s, edge_order=2)
    predicted_vx = states["speed"] * np.cos(states["heading"])
    predicted_vy = states["speed"] * np.sin(states["heading"])
    velocity_error = np.hypot(dx - predicted_vx, dy - predicted_vy)
    heading_jump = np.abs(np.diff(np.unwrap(states["heading"])))
    return {
        "finite": bool(np.isfinite(position_xy).all() and all(np.isfinite(v).all() for v in states.values())),
        "time_strictly_monotonic": bool(np.all(dt > 0)),
        "max_velocity_derivative_error_mps": float(np.max(velocity_error)),
        "max_heading_step_rad": float(np.max(heading_jump)) if len(heading_jump) else 0.0,
        "max_abs_lateral_accel_mps2": float(np.max(np.abs(states["lateral_accel"]))),
        "max_abs_yaw_rate_radps": float(np.max(np.abs(states["yaw_rate"]))),
        "max_abs_curvature_inv_m": float(np.max(np.abs(states["curvature"]))),
    }


def initial_state_fingerprint(x: float, y: float, heading: float, speed_mps: float, time_us: int) -> str:
    return canonical_json_sha256(
        {"x": round(float(x), 6), "y": round(float(y), 6), "heading": round(float(heading), 8),
         "speed_mps": round(float(speed_mps), 6), "time_us": int(time_us)}
    )


class PureLateralExecutionPlanner:
    """Official nuPlan external planner for Stage7L A2 smoke only."""

    requires_scenario = True

    def __init__(
        self,
        scenario: Any,
        manifest_path: str,
        dose_id: str,
        transition_length_m: float | None = None,
        horizon_seconds: float = 4.0,
        sampling_time: float = 0.1,
        target_speed_mps: float = 5.0,
        accel_limit_mps2: float = 1.0,
        audit_dir: str = "",
        parameter_status: str = SMOKE_PARAMETER_STATUS,
    ) -> None:
        if dose_id not in DOSE_TRANSITION_LENGTH_M:
            raise ValueError(f"unknown Stage7L dose: {dose_id}")
        self._scenario = scenario
        self._dose_id = dose_id
        self._transition_length_m = float(
            DOSE_TRANSITION_LENGTH_M[dose_id] if transition_length_m is None else transition_length_m
        )
        if self._transition_length_m <= 0:
            raise ValueError("transition_length_m must be positive")
        self._horizon_seconds = float(horizon_seconds)
        self._sampling_time = float(sampling_time)
        self._target_speed_mps = float(target_speed_mps)
        self._accel_limit_mps2 = float(accel_limit_mps2)
        self._audit_dir = Path(audit_dir).expanduser().resolve() if audit_dir else None
        self._parameter_status = parameter_status
        payload = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
        entries = {str(row["scenario_token"]): row for row in payload["maneuvers"]}
        if scenario.token not in entries:
            raise ValueError(f"scenario token {scenario.token} missing from frozen maneuver manifest")
        self._maneuver = FrozenLaneChangeManeuver.from_mapping(entries[scenario.token])
        self._initialization: Any = None
        self._initial_time_us: int | None = None
        self._generator: CanonicalLongitudinalProgressGenerator | None = None
        self._initial_audit_written = False
        self._compute_trajectory_runtimes: List[float] = []

    def name(self) -> str:
        return f"PureLateralExecutionPlanner_{self._dose_id}"

    def observation_type(self) -> Type[Any]:
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        self._initialization = initialization
        route = tuple(str(x) for x in initialization.route_roadblock_ids)
        if canonical_json_sha256(route) != self._maneuver.route_fingerprint:
            raise ValueError("official initialization route differs from frozen maneuver route")
        source = initialization.map_api.get_map_object(self._maneuver.source_lane_id, SemanticMapLayer.LANE)
        target = initialization.map_api.get_map_object(self._maneuver.target_lane_id, SemanticMapLayer.LANE)
        if source is None or target is None:
            raise ValueError("frozen source or target lane is unavailable in official map")
        adjacent_ids = {str(edge.id) for edge in source.adjacent_edges if edge is not None}
        if self._maneuver.target_lane_id not in adjacent_ids:
            raise ValueError("frozen target lane is no longer a native adjacent edge")
        if source.get_roadblock_id() != target.get_roadblock_id():
            raise ValueError("source and target lane no longer share a roadblock")

    def _write_audit(self, payload: Mapping[str, Any]) -> None:
        if self._audit_dir is None:
            return
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        path = self._audit_dir / f"planner_audit_{self._maneuver.scenario_token}_{self._dose_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego_state, _ = current_input.history.current_state
        if self._initial_time_us is None:
            self._initial_time_us = int(ego_state.time_us)
            actual_fp = initial_state_fingerprint(
                ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading,
                ego_state.dynamic_car_state.speed, ego_state.time_us,
            )
            if actual_fp != self._maneuver.initial_state_fingerprint:
                raise ValueError(
                    "official initial ego state differs from frozen maneuver fingerprint: "
                    f"actual={{'x':{ego_state.rear_axle.x},'y':{ego_state.rear_axle.y},"
                    f"'heading':{ego_state.rear_axle.heading},'speed':{ego_state.dynamic_car_state.speed},"
                    f"'time_us':{ego_state.time_us},'fingerprint':'{actual_fp}'}} "
                    f"expected={{'x':{self._maneuver.initial_x},'y':{self._maneuver.initial_y},"
                    f"'heading':{self._maneuver.initial_heading},'speed':{self._maneuver.initial_speed_mps},"
                    f"'fingerprint':'{self._maneuver.initial_state_fingerprint}'}}"
                )
            self._generator = CanonicalLongitudinalProgressGenerator(
                self._maneuver.initial_speed_mps, self._target_speed_mps, self._accel_limit_mps2
            )
        assert self._generator is not None and self._initial_time_us is not None
        now_relative = max(0.0, (ego_state.time_us - self._initial_time_us) * 1e-6)
        future = np.arange(0.0, self._horizon_seconds + self._sampling_time * 0.5, self._sampling_time)
        global_relative = now_relative + future
        s_route, _, _ = self._generator.sample(global_relative)
        xy, weight = build_lateral_positions(self._maneuver, s_route, self._transition_length_m)
        states = derive_trajectory_states(xy, global_relative, ego_state.car_footprint.vehicle_parameters.wheel_base)
        consistency = dynamic_consistency_audit(xy, global_relative, states)
        trajectory: List[EgoState] = []
        for index, dt_s in enumerate(future):
            trajectory.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(float(states["x"][index]), float(states["y"][index]), float(states["heading"][index])),
                    rear_axle_velocity_2d=StateVector2D(float(states["speed"][index]), 0.0),
                    rear_axle_acceleration_2d=StateVector2D(float(states["longitudinal_accel"][index]), float(states["lateral_accel"][index])),
                    tire_steering_angle=float(states["steering"][index]),
                    time_point=TimePoint(int(ego_state.time_us + round(dt_s * 1e6))),
                    vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
                    angular_vel=float(states["yaw_rate"][index]),
                    angular_accel=float(states["angular_accel"][index]),
                )
            )
        if not self._initial_audit_written:
            self._write_audit(
                {
                    "schema_version": "stage7l_a2_planner_audit_v1",
                    "scenario_token": self._maneuver.scenario_token,
                    "dose_id": self._dose_id,
                    "parameter_status": self._parameter_status,
                    "transition_length_m": self._transition_length_m,
                    "dose_invariant_manifest_sha256": canonical_json_sha256(self._maneuver.dose_invariant_payload()),
                    "canonical_longitudinal_generator_sha256": self._generator.fingerprint(),
                    "s_route_initial_plan_m": s_route.tolist(),
                    "time_initial_plan_s": global_relative.tolist(),
                    "lateral_weight_initial_plan": weight.tolist(),
                    "dynamic_consistency": consistency,
                    "background_mode": self._maneuver.background_mode,
                    "background_agent_model": self._maneuver.background_agent_model,
                    "background_config_sha256": self._maneuver.background_config_sha256,
                    "embedding_or_bdd_read": False,
                }
            )
            self._initial_audit_written = True
        return InterpolatedTrajectory(trajectory)

    def compute_trajectory(self, current_input: Any) -> Any:
        """nuPlan AbstractPlanner-compatible timed entry point."""
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
