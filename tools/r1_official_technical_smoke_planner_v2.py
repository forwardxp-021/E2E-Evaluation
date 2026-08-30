#!/usr/bin/env python3
"""Versioned R1 official smoke planner V2; constructed in B2.5, never launched there."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Type

import numpy as np

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1, build_native_route_reference_v1_1, build_tsb_route_aligned_v1_1, current_ego_construction_parity_audit
from tools.r1_closed_loop_context_adapter_v2_1 import build_closed_loop_context_v2_1
from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1
from tools.r1_prospective_generator_contract_v2 import ALLOWED_ARMS


class R1OfficialTechnicalSmokePlannerV2(AbstractPlanner):
    """Future official planner uniquely bound to B2.5 integrations."""

    requires_scenario = False

    def __init__(self, future_roster_row: Mapping[str, Any], runtime_family: str, smoke_arm: str) -> None:
        if runtime_family not in ALLOWED_ARMS or smoke_arm not in ALLOWED_ARMS[runtime_family]:
            raise ValueError("FROZEN_GENERATOR_ARM_VIOLATION")
        self._row = dict(future_roster_row)
        self._family, self._arm = runtime_family, smoke_arm
        self._initialization = None
        self._map_bridge = None
        self.last_construction_audit: Dict[str, Any] | None = None

    def name(self) -> str:
        return f"R1OfficialTechnicalSmokePlannerV2_{self._family}_{self._arm}"

    def observation_type(self) -> Type[Any]:
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        expected = [str(value) for value in self._row["route_roadblock_ids"]]
        if [str(value) for value in initialization.route_roadblock_ids] != expected:
            raise ValueError("FUTURE_ROSTER_ROUTE_BINDING_MISMATCH")
        self._initialization = initialization
        self._map_bridge = R1OfficialMapQueryBridgeV2_1(initialization.map_api)

    def build_context_v2_1(self, frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if self._map_bridge is None:
            raise RuntimeError("planner is not initialized")
        return build_closed_loop_context_v2_1(family=self._family, scenario_token=str(self._row["scenario_token"]), map_version=str(self._row["map_name"]), route_fingerprint=str(self._row["route_fingerprint"]), initial_state_fingerprint=str(self._row["initial_state_fingerprint"]), log_id=str(self._row["log_id"]), route_roadblock_ids=self._row["route_roadblock_ids"], frames=frames, map_query=self._map_bridge, intended_lane_change_direction=self._row.get("intended_lane_change_direction"))

    @staticmethod
    def _payload(ego: Any) -> Dict[str, Any]:
        return {"rear_axle": {"x": float(ego.rear_axle.x), "y": float(ego.rear_axle.y), "heading": float(ego.rear_axle.heading)}, "speed_mps": float(ego.dynamic_car_state.speed), "time_us": int(ego.time_us)}

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        """Future execution hook. B2.5 tests must never invoke this method."""
        if self._initialization is None or self._map_bridge is None:
            raise RuntimeError("planner is not initialized")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        route = build_native_route_reference_v1_1(self._initialization.map_api, self._row["route_roadblock_ids"], current, max(0.2, float(current["speed_mps"])) * 7.9)
        if self._family == "R-TSB":
            args = (current, float(self._row.get("absolute_episode_time_s", 0.0)), route["reference_xy"], route["current_route_arc_m"], self._arm)
            states = build_tsb_route_aligned_v1_1(*args)
            builder = build_tsb_route_aligned_v1_1
        else:
            source_id = str(self._row["source_lane_id"]); target_id = str(self._row["target_lane_id"])
            source_xy = self._map_bridge.native_reference_xy(source_id); target_xy = self._map_bridge.native_reference_xy(target_id)
            xy = (current["rear_axle"]["x"], current["rear_axle"]["y"])
            args = (current, float(self._row.get("absolute_episode_time_s", 0.0)), source_xy, target_xy, float(self._map_bridge.project(source_id, xy)["arc_m"]), float(self._map_bridge.project(target_id, xy)["arc_m"]), self._arm)
            states = build_hlc_native_geometry_v1_1(*args)
            builder = build_hlc_native_geometry_v1_1
        self.last_construction_audit = current_ego_construction_parity_audit(builder, args, {}, states)
        if not self.last_construction_audit["pass"]:
            raise ValueError("CURRENT_EGO_CONSTRUCTION_PARITY_FAIL")
        params = ego.car_footprint.vehicle_parameters
        official = [EgoState.build_from_rear_axle(StateSE2(float(item["rear_axle"]["x"]), float(item["rear_axle"]["y"]), float(item["rear_axle"]["heading"])), StateVector2D(float(item["speed_mps"]), 0.0), StateVector2D(0.0, 0.0), 0.0, TimePoint(int(item["time_us"])), params) for item in states]
        return InterpolatedTrajectory(official)


__all__ = ["R1OfficialTechnicalSmokePlannerV2"]
