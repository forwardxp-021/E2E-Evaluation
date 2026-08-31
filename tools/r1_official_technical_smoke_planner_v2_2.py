#!/usr/bin/env python3
"""V2.2 frozen-planner instrumentation for realized-current-ego capture only.

The trajectory construction is inherited unchanged from V2.1.  Immediately
before V2.1 receives a PlannerInput, this version writes the *current* ego
state already realized by the simulator.  It never reads a planned trajectory
while creating the primary trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from tools.r1_official_technical_smoke_planner_v2_1 import R1OfficialTechnicalSmokePlannerV2_1


TRACE_SCHEMA = "r1_b2_8_r1_realized_current_ego_trace_v1.0"
PRIMARY_SOURCE = "REALIZED_CURRENT_EGO"


def realized_current_ego_row(current_input: Any) -> Dict[str, Any]:
    """Extract only the state visible at planner-call entry, fail-closed."""
    ego, _ = current_input.history.current_state
    iteration = int(current_input.iteration.index)
    iteration_time_us = int(current_input.iteration.time_us)
    ego_time_us = int(ego.time_us)
    if iteration != int(current_input.iteration.index) or iteration_time_us != ego_time_us:
        raise ValueError("REALIZED_CURRENT_EGO_ITERATION_TIMESTAMP_MISMATCH")
    return {
        "trace_schema": TRACE_SCHEMA,
        "primary_measurement_source": PRIMARY_SOURCE,
        "observation_timing": "PLANNER_CALL_ENTRY_BEFORE_TRAJECTORY_GENERATION",
        "iteration_index": iteration,
        "current_ego": {
            "time_us": ego_time_us,
            "rear_axle": {
                "x": float(ego.rear_axle.x),
                "y": float(ego.rear_axle.y),
                "heading": float(ego.rear_axle.heading),
            },
            "speed_mps": float(ego.dynamic_car_state.speed),
        },
    }


class RealizedCurrentEgoTraceWriterV1:
    """Append-only passive trace writer; validation happens before evaluation."""

    def __init__(self, trace_dir: str) -> None:
        self._trace_dir = Path(trace_dir).expanduser().resolve()
        self._trace_path = self._trace_dir / "realized_current_ego.jsonl"

    @property
    def path(self) -> Path:
        return self._trace_path

    def write(self, current_input: Any) -> Dict[str, Any]:
        row = realized_current_ego_row(current_input)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row


class R1OfficialTechnicalSmokePlannerV2_2(R1OfficialTechnicalSmokePlannerV2_1):
    """V2.1 trajectory semantics plus passive pre-generation state capture."""

    def __init__(self, future_roster_row: Mapping[str, Any], runtime_family: str, smoke_arm: str, trace_dir: str) -> None:
        super().__init__(future_roster_row, runtime_family, smoke_arm)
        self._realized_trace_writer = RealizedCurrentEgoTraceWriterV1(trace_dir)

    @property
    def realized_trace_path(self) -> Path:
        return self._realized_trace_writer.path

    def compute_trajectory(self, current_input: Any) -> Any:
        # This is a passive observation of the simulator's already-realized
        # state.  V2.1 receives the identical PlannerInput immediately after.
        self._realized_trace_writer.write(current_input)
        return super().compute_trajectory(current_input)


__all__ = [
    "PRIMARY_SOURCE",
    "R1OfficialTechnicalSmokePlannerV2_2",
    "RealizedCurrentEgoTraceWriterV1",
    "TRACE_SCHEMA",
    "realized_current_ego_row",
]
