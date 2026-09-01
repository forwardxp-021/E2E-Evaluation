#!/usr/bin/env python3
"""Engineering-only nuPlan step controller capped at 80 planner calls."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)


class R1B29BEngineeringCanary80CallTimeController(StepSimulationTimeController):
    """Keep official step semantics while ending after planner calls 0...79."""

    def number_of_iterations(self) -> int:
        # StepSimulationTimeController calls the planner for indices
        # 0...(number_of_iterations - 2), hence 81 yields exactly 80 calls.
        return min(super().number_of_iterations(), 81)


__all__ = ["R1B29BEngineeringCanary80CallTimeController"]
