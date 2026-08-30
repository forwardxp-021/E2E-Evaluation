#!/usr/bin/env python3
"""R1 official smoke planner V2.1 with bound dispatch and absolute phase clock."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence, Type

import numpy as np

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1, build_native_route_reference_v1_1, build_tsb_route_aligned_v1_1, current_ego_construction_parity_audit
from tools.r1_closed_loop_context_adapter_v2_1 import build_closed_loop_context_v2_1
from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1
from tools.r1_prospective_generator_contract_v2 import ALLOWED_ARMS, HLC_BASELINE, HLC_TREATMENT, TSB_BASELINE, TSB_TREATMENT, frozen_tsb_acceleration, hlc_progress


class R1OfficialTechnicalSmokePlannerV2_1(AbstractPlanner):
    """Bound nuPlan 1.2.2 planner interface; B2.6 itself never runs simulation."""

    requires_scenario = False

    def __init__(self, future_roster_row: Mapping[str, Any], runtime_family: str, smoke_arm: str) -> None:
        if runtime_family not in ALLOWED_ARMS or smoke_arm not in ALLOWED_ARMS[runtime_family]:
            raise ValueError("FROZEN_GENERATOR_ARM_VIOLATION")
        self._row = dict(future_roster_row)
        self._family, self._arm = runtime_family, smoke_arm
        self._initialization = None
        self._map_bridge = None
        self._episode_start_time_us: int | None = None
        self._last_iteration_index: int | None = None
        self._last_iteration_time_us: int | None = None
        self.phase_history: list[Dict[str, Any]] = []
        self.last_construction_audit: Dict[str, Any] | None = None

    def name(self) -> str:
        return f"R1OfficialTechnicalSmokePlannerV2_1_{self._family}_{self._arm}"

    def observation_type(self) -> Type[Any]:
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        expected = [str(value) for value in self._row["route_roadblock_ids"]]
        if [str(value) for value in initialization.route_roadblock_ids] != expected:
            raise ValueError("FUTURE_ROSTER_ROUTE_BINDING_MISMATCH")
        self._initialization = initialization
        self._map_bridge = R1OfficialMapQueryBridgeV2_1(initialization.map_api)
        self._episode_start_time_us = None
        self._last_iteration_index = None
        self._last_iteration_time_us = None
        self.phase_history.clear()

    def build_context_v2_1(self, frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if self._map_bridge is None:
            raise RuntimeError("planner is not initialized")
        return build_closed_loop_context_v2_1(family=self._family, scenario_token=str(self._row["scenario_token"]), map_version=str(self._row["map_name"]), route_fingerprint=str(self._row["route_fingerprint"]), initial_state_fingerprint=str(self._row["initial_state_fingerprint"]), log_id=str(self._row["log_id"]), route_roadblock_ids=self._row["route_roadblock_ids"], frames=frames, map_query=self._map_bridge, intended_lane_change_direction=self._row.get("intended_lane_change_direction"))

    @staticmethod
    def _payload(ego: Any) -> Dict[str, Any]:
        return {"rear_axle": {"x": float(ego.rear_axle.x), "y": float(ego.rear_axle.y), "heading": float(ego.rear_axle.heading)}, "speed_mps": float(ego.dynamic_car_state.speed), "time_us": int(ego.time_us)}

    def _absolute_episode_clock(self, current_input: Any, ego_time_us: int) -> Dict[str, Any]:
        index = int(current_input.iteration.index)
        physical_time_us = int(current_input.iteration.time_us)
        if physical_time_us != int(ego_time_us):
            raise ValueError("PLANNER_INPUT_ITERATION_EGO_TIMESTAMP_MISMATCH")
        if self._episode_start_time_us is None:
            if index != 0:
                raise ValueError("ABSOLUTE_EPISODE_CLOCK_FIRST_CALL_MUST_BE_ITERATION_0")
            self._episode_start_time_us = physical_time_us
        if self._last_iteration_index is not None and (index <= self._last_iteration_index or physical_time_us <= int(self._last_iteration_time_us)):
            raise ValueError("ABSOLUTE_EPISODE_CLOCK_NON_MONOTONIC_REPLAN")
        nominal = float(index) * 0.1
        physical_elapsed = (physical_time_us - int(self._episode_start_time_us)) * 1e-6
        phase_value = float(hlc_progress(np.asarray([nominal]), self._arm)[0]) if self._family == "R-HLC" else float(frozen_tsb_acceleration(nominal, self._arm))
        record = {"iteration_index": index, "physical_time_us": physical_time_us, "derived_nominal_episode_time_s": nominal, "physical_elapsed_episode_time_s": physical_elapsed, "absolute_episode_time_s": nominal, "phase_source": "CURRENT_INPUT_SIMULATION_ITERATION_RELATIVE_TO_FROZEN_EPISODE_START", "t_anchor_iteration": 10, "t_diverge_iteration": 11, "generator_phase_value_at_call": phase_value, "generator_phase_kind": "HLC_PROGRESS" if self._family == "R-HLC" else "TSB_ACCELERATION_MPS2", "phase_reset": False}
        self._last_iteration_index, self._last_iteration_time_us = index, physical_time_us
        self.phase_history.append(record)
        return record

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        if self._initialization is None or self._map_bridge is None:
            raise RuntimeError("planner is not initialized")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        phase = self._absolute_episode_clock(current_input, int(current["time_us"]))
        route = build_native_route_reference_v1_1(self._initialization.map_api, self._row["route_roadblock_ids"], current, max(0.2, float(current["speed_mps"])) * 7.9)
        absolute_episode_time_s = float(phase["absolute_episode_time_s"])
        if self._family == "R-TSB":
            args = (current, absolute_episode_time_s, route["reference_xy"], route["current_route_arc_m"], self._arm)
            builder = build_tsb_route_aligned_v1_1
        else:
            source_id, target_id = str(self._row["source_lane_id"]), str(self._row["target_lane_id"])
            source_xy, target_xy = self._map_bridge.native_reference_xy(source_id), self._map_bridge.native_reference_xy(target_id)
            xy = (current["rear_axle"]["x"], current["rear_axle"]["y"])
            args = (current, absolute_episode_time_s, source_xy, target_xy, float(self._map_bridge.project(source_id, xy)["arc_m"]), float(self._map_bridge.project(target_id, xy)["arc_m"]), self._arm)
            builder = build_hlc_native_geometry_v1_1
        states = builder(*args)
        self.last_construction_audit = current_ego_construction_parity_audit(builder, args, {}, states)
        if not self.last_construction_audit["pass"]:
            raise ValueError("CURRENT_EGO_CONSTRUCTION_PARITY_FAIL")
        parameters = ego.car_footprint.vehicle_parameters
        official = [EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(float(item["rear_axle"]["x"]), float(item["rear_axle"]["y"]), float(item["rear_axle"]["heading"])), rear_axle_velocity_2d=StateVector2D(float(item["speed_mps"]), 0.0), rear_axle_acceleration_2d=StateVector2D(0.0, 0.0), tire_steering_angle=0.0, time_point=TimePoint(int(item["time_us"])), vehicle_parameters=parameters) for item in states]
        return InterpolatedTrajectory(official)

    def compute_trajectory(self, current_input: Any) -> Any:
        """nuPlan 1.2.2 public dispatch explicitly delegates to the V2.1 implementation."""
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


__all__ = ["R1OfficialTechnicalSmokePlannerV2_1"]
