#!/usr/bin/env python3
"""Native-geometry HLC progress and paired terminal route progress v1.0."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def _reference(value: Sequence[Sequence[float]], label: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"{label}_NATIVE_REFERENCE_INVALID")
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if np.any(length <= 0):
        raise ValueError(f"{label}_NATIVE_REFERENCE_INVALID")
    return points


def native_projection_v1_0(reference_xy: Sequence[Sequence[float]], point_xy: Sequence[float], *, label: str) -> Dict[str, Any]:
    """Project to one unambiguous native polyline; exact topological ties fail closed."""
    points = _reference(reference_xy, label)
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{label}_PROJECTION_POINT_INVALID")
    delta = np.diff(points, axis=0)
    denominator = np.sum(delta * delta, axis=1)
    fraction = np.clip(np.sum((point - points[:-1]) * delta, axis=1) / denominator, 0.0, 1.0)
    projected = points[:-1] + fraction[:, None] * delta
    distance2 = np.sum((projected - point) ** 2, axis=1)
    minimum = float(np.min(distance2))
    tied = np.flatnonzero(distance2 == minimum)
    arc_prefix = np.r_[0.0, np.cumsum(np.sqrt(denominator))]
    if len(tied) > 1:
        contiguous_joint = bool(np.all(np.diff(tied) == 1))
        if not contiguous_joint:
            raise ValueError(f"{label}_NATIVE_PROJECTION_AMBIGUOUS_FAIL_CLOSED")
    index = int(tied[0])
    return {
        "point_xy": [float(value) for value in projected[index]],
        "arc_m": float(arc_prefix[index] + fraction[index] * math.sqrt(float(denominator[index]))),
        "distance_m": math.sqrt(minimum),
        "segment_index": index,
        "segment_fraction": float(fraction[index]),
        "heading_rad": math.atan2(float(delta[index, 1]), float(delta[index, 0])),
        "projection_semantic": "UNAMBIGUOUS_NATIVE_POLYLINE_OR_FAIL_CLOSED",
    }


def hlc_realized_lane_transition_progress_v1_0(
    *,
    source_reference_xy: Sequence[Sequence[float]],
    target_reference_xy: Sequence[Sequence[float]],
    realized_ego_xy: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Read local cross-lane progress; longitudinal motion cannot become transition."""
    source = _reference(source_reference_xy, "SOURCE")
    target = _reference(target_reference_xy, "TARGET")
    ego = np.asarray(realized_ego_xy, dtype=np.float64)
    if ego.ndim != 2 or ego.shape[1] != 2 or not len(ego) or not np.isfinite(ego).all():
        raise ValueError("REALIZED_EGO_XY_INVALID")
    raw, clipped, audit = [], [], []
    for index, point in enumerate(ego):
        source_projection = native_projection_v1_0(source, point, label="SOURCE")
        target_projection = native_projection_v1_0(target, point, label="TARGET")
        source_point = np.asarray(source_projection["point_xy"], dtype=np.float64)
        target_point = np.asarray(target_projection["point_xy"], dtype=np.float64)
        cross_lane = target_point - source_point
        denominator = float(np.dot(cross_lane, cross_lane))
        if denominator <= 0.0:
            raise ValueError("HLC_LOCAL_CROSS_LANE_VECTOR_DEGENERATE")
        progress = float(np.dot(point - source_point, cross_lane) / denominator)
        raw.append(progress)
        clipped.append(float(np.clip(progress, 0.0, 1.0)))
        audit.append({"frame_index": index, "source_projection": source_projection, "target_projection": target_projection, "local_cross_lane_vector_xy": [float(value) for value in cross_lane], "raw_progress": progress})
    return {"status": "HLC_REALIZED_PROGRESS_EVALUABLE", "raw_progress": raw, "clipped_progress_for_frozen_mechanism": clipped, "frame_audit": audit, "semantic": "LOCAL_SOURCE_TO_TARGET_NATIVE_CROSS_LANE_NORMALIZED_PROGRESS", "longitudinal_surrogate_used": False}


def terminal_native_route_progress_v1_0(
    *,
    baseline_terminal_xy: Sequence[float],
    treatment_terminal_xy: Sequence[float],
    native_route_reference_xy: Sequence[Sequence[float]],
    route_reference_source: str,
) -> Dict[str, Any]:
    """Project both terminal ego positions onto one frozen native route."""
    route = _reference(native_route_reference_xy, "ROUTE")
    baseline = native_projection_v1_0(route, baseline_terminal_xy, label="BASELINE_TERMINAL_ROUTE")
    treatment = native_projection_v1_0(route, treatment_terminal_xy, label="TREATMENT_TERMINAL_ROUTE")
    delta = abs(float(baseline["arc_m"]) - float(treatment["arc_m"]))
    canonical = json.dumps(route.tolist(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"status": "HLC_TERMINAL_ROUTE_PROGRESS_EVALUABLE", "baseline_terminal_route_s_m": float(baseline["arc_m"]), "treatment_terminal_route_s_m": float(treatment["arc_m"]), "paired_route_progress_delta_m": delta, "frozen_limit_m": 1.5, "pass": delta <= 1.5 + 1e-12, "route_reference_source": str(route_reference_source), "route_reference_sha256": hashlib.sha256(canonical).hexdigest(), "path_length_surrogate_used": False, "baseline_projection": baseline, "treatment_projection": treatment}


__all__ = ["hlc_realized_lane_transition_progress_v1_0", "native_projection_v1_0", "terminal_native_route_progress_v1_0"]
