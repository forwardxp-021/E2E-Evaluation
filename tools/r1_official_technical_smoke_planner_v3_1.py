#!/usr/bin/env python3
"""Scientific-runtime candidate planner: V2.3 HLC plus Primary80 tracing."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import (
    build_tsb_route_aligned_v1_1,
    current_ego_construction_parity_audit,
)
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_geometry_v2_3
from tools.r1_official_technical_smoke_planner_v2_2 import R1OfficialTechnicalSmokePlannerV2_2


class R1OfficialTechnicalSmokePlannerV3_1(R1OfficialTechnicalSmokePlannerV2_2):
    """Preserve frozen arms while enforcing V2.3 and Primary-only trace output."""

    def __init__(
        self,
        future_roster_row: Mapping[str, Any],
        runtime_family: str,
        smoke_arm: str,
        trace_dir: str,
    ) -> None:
        super().__init__(future_roster_row, runtime_family, smoke_arm, trace_dir)
        self.route_continuous_audits: list[Dict[str, Any]] = []
        self._trace_dir_v3_1 = Path(trace_dir).expanduser().resolve()

    def name(self) -> str:
        return f"R1OfficialTechnicalSmokePlannerV3_1_{self._family}_{self._arm}"

    def compute_trajectory(self, current_input: Any) -> Any:
        # Reject an invalid secondary call before the inherited passive writer
        # can create any non-Primary row.
        if int(current_input.iteration.index) >= 80:
            raise RuntimeError("R1_PRIMARY80_SECONDARY_PLANNER_CALL_FORBIDDEN")
        return super().compute_trajectory(current_input)

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        if int(current_input.iteration.index) >= 80:
            raise RuntimeError("R1_PRIMARY80_SECONDARY_PLANNER_CALL_FORBIDDEN")
        if self._initialization is None or self._map_bridge is None:
            raise RuntimeError("planner is not initialized")
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego, _ = current_input.history.current_state
        current = self._payload(ego)
        phase = self._absolute_episode_clock(current_input, int(current["time_us"]))
        absolute_episode_time_s = float(phase["absolute_episode_time_s"])
        if self._family == "R-HLC":
            states, corridor = build_hlc_route_continuous_geometry_v2_3(
                self._initialization.map_api,
                self._row["route_roadblock_ids"],
                str(self._row["source_lane_id"]),
                str(self._row["target_lane_id"]),
                current,
                absolute_episode_time_s,
                self._arm,
            )
            self.route_continuous_audits.append(
                {
                    "iteration_index": int(current_input.iteration.index),
                    "source_remaining_margin_m": corridor["source_remaining_margin_m"],
                    "target_remaining_margin_m": corridor["target_remaining_margin_m"],
                    "source_components": corridor["source_components"],
                    "target_components": corridor["target_components"],
                    "route_progression_invariant": corridor["route_progression_invariant"],
                }
            )
            from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1

            audit_builder = build_hlc_native_geometry_v1_1
            audit_args: Sequence[Any] = (
                current,
                absolute_episode_time_s,
                corridor["source_reference_xy"],
                corridor["target_reference_xy"],
                corridor["source_current_arc_m"],
                corridor["target_current_arc_m"],
                self._arm,
            )
        else:
            from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1

            route = build_native_route_reference_v1_1(
                self._initialization.map_api,
                self._row["route_roadblock_ids"],
                current,
                max(0.2, float(current["speed_mps"])) * 7.9,
            )
            audit_builder = build_tsb_route_aligned_v1_1
            audit_args = (
                current,
                absolute_episode_time_s,
                route["reference_xy"],
                route["current_route_arc_m"],
                self._arm,
            )
            states = audit_builder(*audit_args)
        self.last_construction_audit = current_ego_construction_parity_audit(
            audit_builder, audit_args, {}, states
        )
        if not self.last_construction_audit["pass"]:
            raise ValueError("CURRENT_EGO_CONSTRUCTION_PARITY_FAIL")
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


__all__ = ["R1OfficialTechnicalSmokePlannerV3_1"]
