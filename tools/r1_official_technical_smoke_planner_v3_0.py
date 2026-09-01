#!/usr/bin/env python3
"""Engineering-canary planner with route-continuous official-native HLC refs."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import (
    build_tsb_route_aligned_v1_1,
    current_ego_construction_parity_audit,
)
from tools.r1_closed_loop_benchmark_v2_2 import build_hlc_route_continuous_geometry_v2_2
from tools.r1_official_technical_smoke_planner_v2_2 import (
    R1OfficialTechnicalSmokePlannerV2_2,
    realized_current_ego_row,
)


class Primary80AndSecondaryTraceWriterV1_1:
    """Keep 0...79 primary; preserve later calls as secondary diagnostics."""

    def __init__(self, trace_dir: str) -> None:
        self._trace_dir = Path(trace_dir).expanduser().resolve()
        self._trace_path = self._trace_dir / "realized_current_ego.jsonl"
        self._secondary_path = self._trace_dir / "secondary_diagnostic_realized_current_ego.jsonl"

    @property
    def path(self) -> Path:
        return self._trace_path

    @property
    def secondary_path(self) -> Path:
        return self._secondary_path

    def write(self, current_input: Any) -> Dict[str, Any]:
        row = realized_current_ego_row(current_input)
        iteration = int(row["iteration_index"])
        path = self._trace_path if iteration < 80 else self._secondary_path
        if iteration >= 80:
            row = {
                **row,
                "trace_role": "SECONDARY_DIAGNOSTIC_NOT_PRIMARY",
                "primary_measurement_source": "NOT_PRIMARY_ITERATION_GE_80",
            }
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row


class R1OfficialTechnicalSmokePlannerV3_0(R1OfficialTechnicalSmokePlannerV2_2):
    """V2.2 instrumentation plus the sole B2.9-B reference semantic change."""

    def __init__(
        self,
        future_roster_row: Mapping[str, Any],
        runtime_family: str,
        smoke_arm: str,
        trace_dir: str,
    ) -> None:
        super().__init__(future_roster_row, runtime_family, smoke_arm, trace_dir)
        self._realized_trace_writer = Primary80AndSecondaryTraceWriterV1_1(trace_dir)
        self.route_continuous_audits: list[Dict[str, Any]] = []

    def name(self) -> str:
        return f"R1OfficialTechnicalSmokePlannerV3_0_{self._family}_{self._arm}"

    def compute_planner_trajectory(self, current_input: Any) -> Any:
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
            states, corridor = build_hlc_route_continuous_geometry_v2_2(
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
                    "source_total_length_m": corridor["source_total_length_m"],
                    "target_total_length_m": corridor["target_total_length_m"],
                    "source_remaining_margin_m": corridor["source_remaining_margin_m"],
                    "target_remaining_margin_m": corridor["target_remaining_margin_m"],
                    "source_edge_ids": [item["edge_id"] for item in corridor["source_components"]],
                    "target_edge_ids": [item["edge_id"] for item in corridor["target_components"]],
                    "topology_ambiguity": corridor["topology_ambiguity"],
                }
            )
            audit_args: Sequence[Any] = (
                current,
                absolute_episode_time_s,
                corridor["source_reference_xy"],
                corridor["target_reference_xy"],
                corridor["source_current_arc_m"],
                corridor["target_current_arc_m"],
                self._arm,
            )
            from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1

            audit_builder = build_hlc_native_geometry_v1_1
        else:
            from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1

            route = build_native_route_reference_v1_1(
                self._initialization.map_api,
                self._row["route_roadblock_ids"],
                current,
                max(0.2, float(current["speed_mps"])) * 7.9,
            )
            audit_args = (
                current,
                absolute_episode_time_s,
                route["reference_xy"],
                route["current_route_arc_m"],
                self._arm,
            )
            audit_builder = build_tsb_route_aligned_v1_1
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


__all__ = ["Primary80AndSecondaryTraceWriterV1_1", "R1OfficialTechnicalSmokePlannerV3_0"]
