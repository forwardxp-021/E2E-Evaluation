#!/usr/bin/env python3
"""Deterministic global G_R2 controller-aware residual generator (DEV candidate)."""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


T_DIVERGE_S = 1.1
ARM_BASELINE = "BASELINE"
ARM_TREATMENT = "TREATMENT"


def _quintic(value: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _smooth(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * _quintic(elapsed / max(float(duration), 1e-9))


def hlc_controller_aware_progress(
    time_s: np.ndarray, arm: str, parameters: Mapping[str, Any]
) -> np.ndarray:
    """Return precompensated planner progress; both arms are identical before 1.1 s."""
    t = np.asarray(time_s, dtype=np.float64)
    result = np.zeros_like(t)
    active = t >= T_DIVERGE_S
    if arm == ARM_BASELINE:
        result[active] = _smooth(
            0.0, 1.0, t[active] - T_DIVERGE_S,
            float(parameters["baseline_transition_duration_s"]),
        )
        return np.clip(result, 0.0, 1.0)
    if arm != ARM_TREATMENT:
        raise ValueError(f"R2_B_UNKNOWN_ARM:{arm}")
    # The shared lag compensation advances only the post-divergence morphology.
    lag = float(parameters.get("lag_precompensation_s", 0.0))
    tau = np.maximum(0.0, t - T_DIVERGE_S + lag)
    advance_duration = float(parameters["advance_duration_s"])
    advance_progress = float(parameters["advance_progress"])
    hold_duration = float(parameters["hold_duration_s"])
    retreat_duration = float(parameters["retreat_duration_s"])
    recommit_duration = float(parameters["recommit_duration_s"])
    retreat_target = advance_progress - float(parameters["retreat_depth"])
    advance_end = advance_duration
    hold_end = advance_end + hold_duration
    retreat_end = hold_end + retreat_duration
    recommit_end = retreat_end + recommit_duration
    mask = active & (tau < advance_end)
    result[mask] = _smooth(0.0, advance_progress, tau[mask], advance_duration)
    mask = active & (tau >= advance_end) & (tau < hold_end)
    result[mask] = advance_progress
    mask = active & (tau >= hold_end) & (tau < retreat_end)
    result[mask] = _smooth(advance_progress, retreat_target, tau[mask] - hold_end, retreat_duration)
    mask = active & (tau >= retreat_end) & (tau < recommit_end)
    result[mask] = _smooth(retreat_target, 1.0, tau[mask] - retreat_end, recommit_duration)
    result[active & (tau >= recommit_end)] = 1.0
    return np.clip(result, 0.0, 1.0)


def tsb_controller_aware_acceleration(
    time_s: float, arm: str, parameters: Mapping[str, Any]
) -> float:
    """Global phase schedule sized for repeated replanning and a 1 s LQR lookahead."""
    t = float(time_s)
    if t < T_DIVERGE_S:
        return 0.0
    if arm == ARM_BASELINE:
        end = T_DIVERGE_S + float(parameters["baseline_duration_s"])
        return float(parameters["baseline_brake_mps2"]) if t < end else 0.0
    if arm != ARM_TREATMENT:
        raise ValueError(f"R2_B_UNKNOWN_ARM:{arm}")
    first_end = T_DIVERGE_S + float(parameters["first_brake_duration_s"])
    release_end = first_end + float(parameters["release_duration_s"])
    second_end = release_end + float(parameters["second_brake_duration_s"])
    if t < first_end:
        return float(parameters["first_brake_mps2"])
    if t < release_end:
        return float(parameters["release_mps2"])
    if t < second_end:
        return float(parameters["second_brake_mps2"])
    return 0.0


def validate_global_parameters(family: str, parameters: Mapping[str, Any]) -> None:
    forbidden = {"scenario_token", "log_id", "identity_parameters", "per_scenario"}
    leaked = forbidden.intersection(parameters)
    if leaked:
        raise ValueError(f"R2_B_SCENARIO_SPECIFIC_PARAMETER_FORBIDDEN:{sorted(leaked)}")
    values = [float(v) for v in parameters.values() if isinstance(v, (int, float))]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("R2_B_NONFINITE_GLOBAL_PARAMETER")
    required = (
        {"baseline_transition_duration_s", "advance_duration_s", "advance_progress", "hold_duration_s", "retreat_depth", "retreat_duration_s", "recommit_duration_s", "lag_precompensation_s"}
        if family == "R-HLC"
        else {"baseline_brake_mps2", "baseline_duration_s", "first_brake_mps2", "first_brake_duration_s", "release_mps2", "release_duration_s", "second_brake_mps2", "second_brake_duration_s"}
    )
    missing = required.difference(parameters)
    if missing:
        raise ValueError(f"R2_B_GLOBAL_PARAMETER_MISSING:{sorted(missing)}")


__all__ = [
    "ARM_BASELINE", "ARM_TREATMENT", "T_DIVERGE_S",
    "hlc_controller_aware_progress", "tsb_controller_aware_acceleration",
    "validate_global_parameters",
]
