#!/usr/bin/env python3
"""Frozen R1 Primary80 scientific simulation-time controller.

The nuPlan step controller invokes the planner for indices
``0...(number_of_iterations - 2)``.  Returning 81 therefore yields exactly
80 planner calls while preserving the official 0.1 s scenario time points.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)


class R1Primary80ScientificTimeControllerV1(StepSimulationTimeController):
    """End after calls 0...79 and fail closed when the scenario is too short."""

    REQUIRED_CONTROLLER_ITERATIONS = 81

    def number_of_iterations(self) -> int:
        official_iterations = int(super().number_of_iterations())
        if official_iterations < self.REQUIRED_CONTROLLER_ITERATIONS:
            raise ValueError(
                "R1_PRIMARY80_NOT_EVALUABLE_SCENARIO_HAS_FEWER_THAN_81_ITERATIONS:"
                f"official_iterations={official_iterations}"
            )
        return min(official_iterations, self.REQUIRED_CONTROLLER_ITERATIONS)


__all__ = ["R1Primary80ScientificTimeControllerV1"]
