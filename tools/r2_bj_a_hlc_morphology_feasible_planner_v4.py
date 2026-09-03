#!/usr/bin/env python3
"""HLC V4 planner path frozen for R2-BJ-A offline envelope verification."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_closed_loop_benchmark_v2_1 import DT_SECONDS, WINDOW_FRAMES, _state  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import sample_native_reference_no_extrapolation  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import (  # noqa: E402
    ARM_BASELINE,
    compose_kinematic_trajectory,
    morphology_progress,
    validate_parameters,
)


def _states(
    current: Mapping[str, Any],
    absolute_s: float,
    corridor: Mapping[str, Any],
    arm: str,
    parameters: Mapping[str, Any],
    force_common: bool,
) -> Tuple[Sequence[Dict[str, Any]], np.ndarray, Mapping[str, Any]]:
    """Full state construction used by all BJ-A envelope cases."""
    validate_parameters(parameters)
    relative = np.arange(WINDOW_FRAMES, dtype=np.float64) * DT_SECONDS
    future_absolute = float(absolute_s) + relative
    effective_arm = ARM_BASELINE if force_common else arm
    progress = morphology_progress(future_absolute, effective_arm, parameters["morphology"])
    speed = np.full(WINDOW_FRAMES, max(float(current["speed_mps"]), 0.2), dtype=np.float64)
    distance = speed * relative
    source, _ = sample_native_reference_no_extrapolation(
        corridor["source_reference_xy"], float(corridor["source_current_arc_m"]) + distance
    )
    target, _ = sample_native_reference_no_extrapolation(
        corridor["target_reference_xy"], float(corridor["target_current_arc_m"]) + distance
    )
    morphology_xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    current_xy = np.asarray([current["rear_axle"]["x"], current["rear_axle"]["y"]], dtype=np.float64)
    xy, heading, curvature, capture = compose_kinematic_trajectory(
        morphology_xy,
        np.asarray(corridor["target_reference_xy"], dtype=np.float64),
        current_xy,
        float(current["rear_axle"]["heading"]),
        float(current["speed_mps"]),
        float(absolute_s),
        future_absolute,
        parameters["capture"],
    )
    start_us = int(current["time_us"])
    states = [
        _state(xy[index, 0], xy[index, 1], heading[index], speed[index], start_us + index * 100_000)
        for index in range(WINDOW_FRAMES)
    ]
    states[0] = dict(current)
    capture = dict(capture)
    capture["derived_curvature_inv_m"] = curvature.tolist()
    capture["morphology_progress_profile"] = progress.tolist()
    capture["effective_arm"] = effective_arm
    return states, progress, capture


__all__ = ["_states"]
