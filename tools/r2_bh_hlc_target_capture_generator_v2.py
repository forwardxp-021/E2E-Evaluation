#!/usr/bin/env python3
"""R2-BH HLC V2: behavior morphology plus fixed absolute-time target capture."""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Mapping, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_b_controller_aware_generator_v1 import (  # noqa: E402
    ARM_BASELINE,
    ARM_TREATMENT,
    T_DIVERGE_S,
    hlc_controller_aware_progress,
)


def _quintic(value: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def absolute_capture_weight(time_s: np.ndarray, capture: Mapping[str, Any]) -> np.ndarray:
    """C2 weight on the fixed episode clock: one before start and zero at/after end."""
    start = float(capture["capture_start_abs_s"])
    duration = float(capture["capture_duration_s"])
    if duration <= 0:
        raise ValueError("R2_BH_CAPTURE_DURATION_NOT_POSITIVE")
    u = (np.asarray(time_s, dtype=np.float64) - start) / duration
    return 1.0 - _quintic(u)


def replanning_capture_weight(
    current_absolute_s: float, future_absolute_s: np.ndarray, capture: Mapping[str, Any]
) -> np.ndarray:
    """Preserve state0 residual while still reaching the fixed absolute capture end."""
    absolute = absolute_capture_weight(future_absolute_s, capture)
    current = float(absolute_capture_weight(np.asarray([current_absolute_s]), capture)[0])
    if current <= 1e-12:
        result = np.zeros_like(absolute)
    else:
        result = np.clip(absolute / current, 0.0, 1.0)
    # state0 itself is overwritten by the exact current ego; this flag documents identity.
    result[0] = 1.0
    return result


def target_capture_path(
    base_xy: np.ndarray,
    base_heading: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    current_absolute_s: float,
    future_absolute_s: np.ndarray,
    capture: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Decay current residual in the native path frame; no free-space reference is created."""
    xy = np.asarray(base_xy, dtype=np.float64).copy()
    heading = np.asarray(base_heading, dtype=np.float64).copy()
    origin = np.asarray(current_xy, dtype=np.float64)
    normal0 = np.asarray([-math.sin(heading[0]), math.cos(heading[0])])
    lateral_residual = float(np.dot(origin - xy[0], normal0))
    heading_residual = float(math.atan2(math.sin(current_heading - heading[0]), math.cos(current_heading - heading[0])))
    weights = replanning_capture_weight(current_absolute_s, future_absolute_s, capture)
    normals = np.column_stack((-np.sin(heading), np.cos(heading)))
    commanded_lateral = lateral_residual * weights
    commanded_heading = heading_residual * weights
    xy += normals * commanded_lateral[:, None]
    heading = heading + commanded_heading
    # Exact identity is enforced independently from target capture for state0 only.
    xy[0] = origin
    heading[0] = float(current_heading)
    end = float(capture["capture_start_abs_s"]) + float(capture["capture_duration_s"])
    audit = {
        "current_target_frame_lateral_residual_m": lateral_residual,
        "current_heading_residual_rad": heading_residual,
        "capture_weights": weights.tolist(),
        "commanded_lateral_residual_m": commanded_lateral.tolist(),
        "commanded_heading_residual_rad": commanded_heading.tolist(),
        "capture_start_abs_s": float(capture["capture_start_abs_s"]),
        "capture_end_abs_s": end,
        "state0_exact_current_ego": True,
        "state1_plus_zero_after_capture_end": bool(
            current_absolute_s >= end - 1e-12 and np.allclose(commanded_lateral[1:], 0.0)
        ),
    }
    return xy, heading, audit


def behavior_progress(time_s: np.ndarray, arm: str, parameters: Mapping[str, Any]) -> np.ndarray:
    return hlc_controller_aware_progress(time_s, arm, parameters["morphology"])


def validate_parameters(parameters: Mapping[str, Any]) -> None:
    if set(parameters) != {"morphology", "capture"}:
        raise ValueError("R2_BH_PARAMETER_PARTITIONS_MUST_BE_MORPHOLOGY_AND_CAPTURE")
    forbidden = {"scenario_token", "log_id", "identity", "per_scenario"}
    if forbidden.intersection(parameters["morphology"]) or forbidden.intersection(parameters["capture"]):
        raise ValueError("R2_BH_SCENARIO_SPECIFIC_PARAMETER_FORBIDDEN")
    numeric = [
        float(value)
        for partition in parameters.values()
        for value in partition.values()
        if isinstance(value, (int, float))
    ]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise ValueError("R2_BH_NONFINITE_PARAMETER")
    if parameters["capture"].get("lateral_offset_decay_shape") != "QUINTIC_C2":
        raise ValueError("R2_BH_LATERAL_CAPTURE_SHAPE_NOT_FROZEN_QUINTIC_C2")
    if parameters["capture"].get("heading_error_decay_shape") != "QUINTIC_C2":
        raise ValueError("R2_BH_HEADING_CAPTURE_SHAPE_NOT_FROZEN_QUINTIC_C2")


__all__ = [
    "ARM_BASELINE", "ARM_TREATMENT", "T_DIVERGE_S", "absolute_capture_weight",
    "behavior_progress", "replanning_capture_weight", "target_capture_path", "validate_parameters",
]
