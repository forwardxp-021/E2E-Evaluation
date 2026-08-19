#!/usr/bin/env python3
"""Dose-independent, pre-treatment replay-traffic clearance for Stage7L.

The module deliberately has no planner, dose, embedding, BDD, or rollout
dependencies.  It assesses a common source-to-target lane-change envelope
against original nuPlan replay tracks on a fixed canonical time schedule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.stage7l_pure_lateral_execution_planner import (
    CanonicalLongitudinalProgressGenerator,
    canonical_json_sha256,
    interpolate_polyline,
    polyline_arclength,
)


@dataclass(frozen=True)
class DynamicClearanceConfig:
    """Physical, dose-independent common-envelope parameters."""

    horizon_seconds: float = 15.0
    time_step_seconds: float = 0.1
    maximum_track_interpolation_gap_seconds: float = 0.25
    trigger_route_progress_m: float = 12.0
    gentle_transition_length_m: float = 60.0
    settling_margin_m: float = 10.0
    target_speed_mps: float = 5.0
    accel_limit_mps2: float = 1.0
    ego_length_m: float = 5.0
    ego_width_m: float = 2.0
    longitudinal_buffer_m: float = 3.0
    lateral_buffer_m: float = 0.5

    def fingerprint(self) -> str:
        return canonical_json_sha256(asdict(self))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time_grid(config: DynamicClearanceConfig) -> np.ndarray:
    count = int(round(config.horizon_seconds / config.time_step_seconds))
    return np.linspace(0.0, config.horizon_seconds, count + 1, dtype=np.float64)


def interpolate_track_state(
    timestamps_s: np.ndarray, states: np.ndarray, query_time_s: float, maximum_gap_s: float
) -> Optional[np.ndarray]:
    """Linearly interpolate [x, y, length, width, yaw] without extrapolation."""
    times = np.asarray(timestamps_s, dtype=np.float64)
    values = np.asarray(states, dtype=np.float64)
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values) or not len(times):
        raise ValueError("track timestamps/states must be non-empty aligned arrays")
    index = int(np.searchsorted(times, query_time_s, side="left"))
    if index < len(times) and abs(float(times[index]) - query_time_s) <= 1e-9:
        return values[index].copy()
    if index == 0 or index == len(times):
        return None
    left, right = index - 1, index
    interval = float(times[right] - times[left])
    if interval <= 0 or interval > maximum_gap_s:
        return None
    fraction = (query_time_s - float(times[left])) / interval
    output = values[left] * (1.0 - fraction) + values[right] * fraction
    # yaw must use the shortest angular interpolation path.
    yaw_delta = (values[right, 4] - values[left, 4] + math.pi) % (2.0 * math.pi) - math.pi
    output[4] = values[left, 4] + fraction * yaw_delta
    return output


def axis_half_extent(length_m: float, width_m: float, yaw_rad: float, axis_heading_rad: float) -> float:
    """Projection half-extent of an oriented rectangular footprint onto an axis."""
    relative = float(yaw_rad - axis_heading_rad)
    return 0.5 * (abs(float(length_m) * math.cos(relative)) + abs(float(width_m) * math.sin(relative)))


def common_envelope_conflict(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    tangent_heading_rad: float,
    agent_state: np.ndarray,
    phase: str,
    config: DynamicClearanceConfig,
) -> Tuple[bool, float, float]:
    """Test one replay-agent footprint against the common current-time envelope.

    ``source_xy`` and ``target_xy`` are the two map centerline positions at the
    same canonical route progress.  During transition, their convex lateral
    strip covers every legal 54--60 m treatment profile without receiving a
    dose identifier.  Before/after transition the strip collapses to source or
    target respectively.
    """
    if phase == "source":
        centers = np.asarray([source_xy], dtype=np.float64)
    elif phase == "target":
        centers = np.asarray([target_xy], dtype=np.float64)
    elif phase == "transition":
        centers = np.asarray([source_xy, target_xy], dtype=np.float64)
    else:
        raise ValueError(f"unknown corridor phase: {phase}")
    tangent = np.asarray([math.cos(tangent_heading_rad), math.sin(tangent_heading_rad)], dtype=np.float64)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    origin = centers[0]
    center_longitudinal = (centers - origin) @ tangent
    center_lateral = (centers - origin) @ normal
    agent_xy = np.asarray(agent_state[:2], dtype=np.float64)
    agent_longitudinal = float((agent_xy - origin) @ tangent)
    agent_lateral = float((agent_xy - origin) @ normal)
    agent_length, agent_width, agent_yaw = map(float, agent_state[2:5])
    agent_long_extent = axis_half_extent(agent_length, agent_width, agent_yaw, tangent_heading_rad)
    agent_lateral_extent = axis_half_extent(agent_length, agent_width, agent_yaw, tangent_heading_rad + math.pi / 2.0)
    longitudinal_limit = config.ego_length_m / 2.0 + agent_long_extent + config.longitudinal_buffer_m
    lateral_limit = config.ego_width_m / 2.0 + agent_lateral_extent + config.lateral_buffer_m
    longitudinal_gap = max(float(np.min(center_longitudinal) - agent_longitudinal), float(agent_longitudinal - np.max(center_longitudinal)), 0.0)
    lateral_gap = max(float(np.min(center_lateral) - agent_lateral), float(agent_lateral - np.max(center_lateral)), 0.0)
    return longitudinal_gap <= longitudinal_limit and lateral_gap <= lateral_limit, longitudinal_gap, lateral_gap


def route_heading(reference_xy: np.ndarray, arc_m: float) -> float:
    delta = 0.5
    points = np.asarray(reference_xy, dtype=np.float64)
    total = float(polyline_arclength(points)[-1])
    before_arc = max(0.0, arc_m - delta)
    after_arc = min(total, arc_m + delta)
    if after_arc - before_arc <= 1e-6:
        raise ValueError("MAP_PROJECTION_FAIL: route heading is outside reference")
    before = interpolate_polyline(points, np.asarray([before_arc]))[0]
    after = interpolate_polyline(points, np.asarray([after_arc]))[0]
    vector = after - before
    if float(np.linalg.norm(vector)) <= 1e-6:
        raise ValueError("MAP_PROJECTION_FAIL: local route tangent is degenerate")
    return float(math.atan2(vector[1], vector[0]))


def replay_tracks_from_db(
    db_path: Path, initial_timestamp_us: int, horizon_seconds: float
) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Read original replay tracks only; this never accesses planner output."""
    end_timestamp_us = int(initial_timestamp_us + horizon_seconds * 1e6)
    lidar_query = """
        SELECT timestamp FROM lidar_pc
        WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp
    """
    track_query = """
        SELECT lp.timestamp AS timestamp_us, lower(hex(lb.track_token)) AS track_token,
               lb.x AS x, lb.y AS y,
               COALESCE(NULLIF(lb.length, 0), tr.length) AS length,
               COALESCE(NULLIF(lb.width, 0), tr.width) AS width,
               lb.yaw AS yaw
        FROM lidar_box lb
        JOIN lidar_pc lp ON lp.token = lb.lidar_pc_token
        LEFT JOIN track tr ON tr.token = lb.track_token
        WHERE lp.timestamp >= ? AND lp.timestamp <= ?
        ORDER BY track_token, lp.timestamp
    """
    by_track: Dict[str, List[Tuple[float, float, float, float, float, float]]] = {}
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        lidar_times = np.asarray([int(row["timestamp"]) for row in connection.execute(lidar_query, (initial_timestamp_us, end_timestamp_us))], dtype=np.int64)
        for row in connection.execute(track_query, (initial_timestamp_us, end_timestamp_us)):
            length = float(row["length"] or 0.0); width = float(row["width"] or 0.0)
            if length <= 0.1 or width <= 0.1:
                continue
            token = str(row["track_token"])
            by_track.setdefault(token, []).append((
                (int(row["timestamp_us"]) - initial_timestamp_us) * 1e-6,
                float(row["x"]), float(row["y"]), length, width, float(row["yaw"]),
            ))
    tracks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for token, values in by_track.items():
        raw = np.asarray(values, dtype=np.float64)
        tracks[token] = raw[:, 0], raw[:, 1:]
    return (lidar_times - int(initial_timestamp_us)) * 1e-6, tracks


def replay_horizon_pass(lidar_times_s: np.ndarray, config: DynamicClearanceConfig) -> bool:
    if not len(lidar_times_s):
        return False
    return (
        float(lidar_times_s[0]) <= config.maximum_track_interpolation_gap_seconds
        and float(lidar_times_s[-1]) >= config.horizon_seconds - config.maximum_track_interpolation_gap_seconds
        and bool(np.all(np.diff(lidar_times_s) <= config.maximum_track_interpolation_gap_seconds))
    )


def dynamic_clearance_audit(
    candidate: Mapping[str, Any], db_root: Path, config: DynamicClearanceConfig
) -> Dict[str, Any]:
    """Return a fully pre-treatment dynamic-clearance audit for one candidate."""
    required = {
        "db_file", "official_simulation_initial_timestamp_us", "initial_speed_mps", "source_start_arc_m",
        "target_start_arc_m", "source_reference_xy_json", "target_reference_xy_json", "paired_reference_remaining_m",
    }
    missing = sorted(key for key in required if key not in candidate or candidate[key] in (None, ""))
    if missing:
        raise ValueError(f"candidate missing dynamic-clearance fields: {missing}")
    source_reference = np.asarray(json.loads(str(candidate["source_reference_xy_json"])), dtype=np.float64)
    target_reference = np.asarray(json.loads(str(candidate["target_reference_xy_json"])), dtype=np.float64)
    times = time_grid(config)
    generator = CanonicalLongitudinalProgressGenerator(
        float(candidate["initial_speed_mps"]), config.target_speed_mps, config.accel_limit_mps2
    )
    progress_m, _, _ = generator.sample(times)
    required_reference_m = float(np.max(progress_m))
    base = {
        "dynamic_clearance_config_sha256": config.fingerprint(),
        "dynamic_horizon_seconds": config.horizon_seconds,
        "dynamic_time_step_seconds": config.time_step_seconds,
        "dynamic_required_reference_progress_m": required_reference_m,
        "dynamic_candidate_reference_remaining_m": float(candidate["paired_reference_remaining_m"]),
        "dynamic_eligibility_pre_treatment": True,
        "dynamic_dose_independent": True,
    }
    if float(candidate["paired_reference_remaining_m"]) < required_reference_m:
        return dict(base, dynamic_clearance_pass=False, dynamic_reason_code="INSUFFICIENT_REFERENCE_LENGTH")
    try:
        source_arc = float(candidate["source_start_arc_m"]) + progress_m
        target_arc = float(candidate["target_start_arc_m"]) + progress_m
        source_xy_all = interpolate_polyline(source_reference, source_arc)
        target_xy_all = interpolate_polyline(target_reference, target_arc)
        headings = np.asarray([route_heading(source_reference, arc) for arc in source_arc], dtype=np.float64)
    except Exception as exc:
        return dict(base, dynamic_clearance_pass=False, dynamic_reason_code="MAP_PROJECTION_FAIL", dynamic_error=str(exc))
    initial_timestamp_us = int(float(candidate["official_simulation_initial_timestamp_us"]))
    lidar_times_s, tracks = replay_tracks_from_db(db_root / str(candidate["db_file"]), initial_timestamp_us, config.horizon_seconds)
    if not replay_horizon_pass(lidar_times_s, config):
        return dict(
            base, dynamic_clearance_pass=False, dynamic_reason_code="INSUFFICIENT_TRACK_HORIZON",
            dynamic_lidar_time_min_s=float(lidar_times_s[0]) if len(lidar_times_s) else None,
            dynamic_lidar_time_max_s=float(lidar_times_s[-1]) if len(lidar_times_s) else None,
        )
    evaluated_agent_positions = 0
    missing_interpolation_positions = 0
    first_conflict: Optional[Dict[str, Any]] = None
    for index, now_s in enumerate(times):
        progress = float(progress_m[index])
        if progress < config.trigger_route_progress_m:
            phase = "source"
        elif progress <= config.trigger_route_progress_m + config.gentle_transition_length_m + config.settling_margin_m:
            phase = "transition"
        else:
            phase = "target"
        for track_token, (track_times, track_states) in tracks.items():
            state = interpolate_track_state(track_times, track_states, float(now_s), config.maximum_track_interpolation_gap_seconds)
            if state is None:
                missing_interpolation_positions += 1
                continue
            evaluated_agent_positions += 1
            conflict, longitudinal_gap, lateral_gap = common_envelope_conflict(
                source_xy_all[index], target_xy_all[index], float(headings[index]), state, phase, config
            )
            if conflict:
                reason = {
                    "source": "SOURCE_LANE_DYNAMIC_CONFLICT",
                    "target": "TARGET_FRONT_DYNAMIC_CONFLICT",
                    "transition": "TRANSITION_CORRIDOR_DYNAMIC_CONFLICT",
                }[phase]
                first_conflict = {
                    "dynamic_reason_code": reason, "dynamic_first_conflict_time_s": float(now_s),
                    "dynamic_conflict_track_token": track_token, "dynamic_conflict_phase": phase,
                    "dynamic_conflict_agent_x": float(state[0]), "dynamic_conflict_agent_y": float(state[1]),
                    "dynamic_conflict_agent_length_m": float(state[2]), "dynamic_conflict_agent_width_m": float(state[3]),
                    "dynamic_conflict_longitudinal_gap_m": longitudinal_gap,
                    "dynamic_conflict_lateral_gap_m": lateral_gap,
                }
                break
        if first_conflict is not None:
            break
    coverage_denominator = evaluated_agent_positions + missing_interpolation_positions
    output = dict(base)
    output.update({
        "dynamic_replay_track_count": len(tracks), "dynamic_evaluated_agent_positions": evaluated_agent_positions,
        "dynamic_missing_interpolation_positions": missing_interpolation_positions,
        "dynamic_agent_interpolation_coverage": (
            float(evaluated_agent_positions / coverage_denominator) if coverage_denominator else 1.0
        ),
        "dynamic_common_envelope": "source_to_target_strip_time_aligned_54_to_60m_family",
    })
    if first_conflict is not None:
        output.update(first_conflict)
        output["dynamic_clearance_pass"] = False
    else:
        output.update({"dynamic_clearance_pass": True, "dynamic_reason_code": "DYNAMIC_CLEAR"})
    return output
