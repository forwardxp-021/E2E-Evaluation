#!/usr/bin/env python3
"""Frozen prospective R1 generator definitions without historical planner imports."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


DT_SECONDS = 0.1
WINDOW_FRAMES = 80
WINDOW_SECONDS = 8.0
HLC_BASELINE = "HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE"
HLC_TREATMENT = "HLC_TREATMENT_HLC_GEN_V2_OPTION_B"
TSB_BASELINE = "TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING"
TSB_TREATMENT = "TSB_TREATMENT_TSB_GEN_V2_OPTION_A"
ALLOWED_ARMS = {"R-HLC": {HLC_BASELINE, HLC_TREATMENT}, "R-TSB": {TSB_BASELINE, TSB_TREATMENT}}
HLC_PRIMARY_F_MATCH_CALIPERS = {"mean_speed": 0.708203939, "end_minus_start_speed": 0.978755681, "path_length": 5.38423459}
TSB_PRIMARY_F_MATCH_CALIPERS = {**HLC_PRIMARY_F_MATCH_CALIPERS, "mean_abs_accel": 0.11777666}


def _quintic(value: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _smooth(start: float, end: float, elapsed: np.ndarray, duration: float) -> np.ndarray:
    return start + (end - start) * _quintic(elapsed / duration)


def hlc_progress(relative_s: np.ndarray, arm: str) -> np.ndarray:
    """Unchanged frozen HLC baseline / Option-B schedule."""
    time_s = np.asarray(relative_s, dtype=np.float64)
    progress = np.zeros_like(time_s)
    diverge = 1.1
    if arm == HLC_BASELINE:
        active = time_s >= diverge
        progress[active] = _smooth(0.0, 1.0, time_s[active] - diverge, 2.0)
        return np.clip(progress, 0.0, 1.0)
    if arm != HLC_TREATMENT:
        raise ValueError(f"unsupported HLC arm: {arm}")
    advance_end, hold_end, retreat_end = diverge + 1.4, diverge + 2.0, diverge + 3.0
    recommit_end = diverge + 5.4
    advance = (time_s >= diverge) & (time_s < advance_end)
    progress[advance] = _smooth(0.0, 0.38, time_s[advance] - diverge, 1.4)
    progress[(time_s >= advance_end) & (time_s < hold_end)] = 0.38
    retreat = (time_s >= hold_end) & (time_s < retreat_end)
    progress[retreat] = _smooth(0.38, 0.22, time_s[retreat] - hold_end, 1.0)
    recommit = (time_s >= retreat_end) & (time_s < recommit_end)
    progress[recommit] = _smooth(0.22, 1.0, time_s[recommit] - retreat_end, 2.4)
    progress[time_s >= recommit_end] = 1.0
    return np.clip(progress, 0.0, 1.0)


def frozen_tsb_acceleration(absolute_episode_time_s: float, arm: str) -> float:
    """Unchanged frozen TSB baseline / Option-A acceleration schedule."""
    t = float(absolute_episode_time_s)
    if arm == TSB_BASELINE:
        return -1.0 if 1.1 <= t < 2.05 else 0.0
    if arm == TSB_TREATMENT:
        if 1.1 <= t < 1.6 or 2.3 <= t < 2.8:
            return -0.9
        if 1.6 <= t < 2.3:
            return 0.4
        return 0.0
    raise ValueError(f"unsupported TSB arm: {arm}")


def wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def prospective_primary_f_match(baseline: Mapping[str, float], treatment: Mapping[str, float], family: str) -> Dict[str, Any]:
    calipers = HLC_PRIMARY_F_MATCH_CALIPERS if family == "R-HLC" else TSB_PRIMARY_F_MATCH_CALIPERS
    delta = {key: round(abs(float(treatment[key]) - float(baseline[key])), 6) for key in calipers}
    by_feature = {key: delta[key] <= limit + 1e-12 for key, limit in calipers.items()}
    return {"status": "F_MATCH_PASS" if all(by_feature.values()) else "F_MATCH_FAIL", "pass": all(by_feature.values()), "primary_features": list(calipers), "calipers": calipers, "absolute_delta": delta, "pass_by_feature": by_feature, "heading_change_abs_total": "SECONDARY_MECHANISM_PROXIMAL_AUDIT"}


def first_state_error(current_ego: Mapping[str, Any], first: Mapping[str, Any]) -> Dict[str, Any]:
    dx = float(first["rear_axle"]["x"]) - float(current_ego["rear_axle"]["x"])
    dy = float(first["rear_axle"]["y"]) - float(current_ego["rear_axle"]["y"])
    result = {"position_error_m": math.hypot(dx, dy), "heading_error_rad": abs(wrap_angle(float(first["rear_axle"]["heading"]) - float(current_ego["rear_axle"]["heading"]))), "speed_error_mps": abs(float(first["speed_mps"]) - float(current_ego["speed_mps"])), "timestamp_error_us": abs(int(first["time_us"]) - int(current_ego["time_us"]))}
    result["exact_construction_identity"] = all(value == 0 for value in result.values())
    return result


def polyline_arclength(xy: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("native reference must be finite Nx2 with N>=2")
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment <= 0):
        raise ValueError("native reference contains reversal/duplicate segment")
    return np.r_[0.0, np.cumsum(segment)]


def sample_native_reference_no_extrapolation(xy: Sequence[Sequence[float]], query_arc_m: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=np.float64)
    arc = polyline_arclength(points)
    query = np.asarray(query_arc_m, dtype=np.float64)
    if np.any(query < -1e-12) or np.any(query > arc[-1] + 1e-12):
        raise ValueError("NATIVE_REFERENCE_COVERAGE_FAIL_NO_EXTRAPOLATION")
    sampled = np.column_stack((np.interp(query, arc, points[:, 0]), np.interp(query, arc, points[:, 1])))
    index = np.clip(np.searchsorted(arc, query, side="right") - 1, 0, len(points) - 2)
    delta = points[index + 1] - points[index]
    return sampled, np.arctan2(delta[:, 1], delta[:, 0])


__all__ = [name for name in globals() if name.isupper()] + ["hlc_progress", "frozen_tsb_acceleration", "wrap_angle", "prospective_primary_f_match", "first_state_error", "polyline_arclength", "sample_native_reference_no_extrapolation"]
