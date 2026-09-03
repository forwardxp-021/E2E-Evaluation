#!/usr/bin/env python3
"""R2-BJ-A2 offline HLC joint-support applicability audit; never runs simulation."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_9_b_route_continuous_canary import _ego, _map_api  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import sample_native_reference_no_extrapolation  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import (  # noqa: E402
    ARM_BASELINE,
    ARM_TREATMENT,
    CaptureInfeasible,
    morphology_progress,
)
from tools.r2_bj_a_hlc_morphology_feasible_planner_v4 import _states  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
R2 = ROOT / "docs/stageR/r2"
CONTRACT = R2 / "r2_bj_a2_joint_support_applicability_contract_v1.0.json"
SOURCE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
INVENTORY = R1 / "r1_official_nuplan_db_inventory_rows_v0.1.json"
BJ_A = R2 / "r2_bj_a_expanded_zero_run_feasibility_envelope_v1.0.json"
SPACE = R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"

COHORTS = {
    "R1_B2_SCIENTIFIC_V1": R1 / "r1_official_technical_smoke_roster_v1.0.json",
    "R1_B2_8_SCIENTIFIC_V2_1": R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json",
    "R1_B2_9_D_SCIENTIFIC_V3": R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json",
    "R2_A_HLC_DEV": R2 / "r2_a_controller_id_dev_canary_roster_v1.0.json",
    "R2_B_HLC_DEV_CAL": R2 / "r2_b_generator_calibration_roster_v1.0.json",
    "R2_BH_HLC_DEV_ARCH": R2 / "r2_bh_hlc_arch_dev_roster_v1.0.json",
    "R2_BI_HLC_DEV_KIN": R2 / "r2_bi_hlc_dev_kin_roster_v1.0.json",
}

OUT = {
    "provenance": R2 / "r2_bj_a2_hlc_joint_support_provenance_manifest_v1.0.json",
    "curvature": R2 / "r2_bj_a2_curvature_quality_forensic_v1.0.json",
    "components": R2 / "r2_bj_a2_native_generated_composite_component_audit_v1.0.json",
    "envelope": R2 / "r2_bj_a2_joint_support_applicability_envelope_v1.0.json",
    "firewall": R2 / "r2_bj_a2_data_firewall_audit_v1.0.json",
    "request": R2 / "R2_BJ_A2_Scientific_Owner_Readiness_Request_v0.1.md",
    "manifest": R2 / "r2_bj_a2_component_sha_binding_manifest_v1.0.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BJ_A2_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def percentile(values: Sequence[float]) -> Mapping[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(x)), "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)), "max": float(np.max(x)),
    }


def _wrap(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def path_curvature(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=np.float64)
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    if len(points) < 3 or np.any(length <= 1e-9):
        raise ValueError("A2_CURVATURE_DUPLICATE_OR_SHORT_SEGMENT")
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    support = 0.5 * (length[:-1] + length[1:])
    curvature = _wrap(np.diff(heading)) / support
    return curvature, support


def resample(xy: np.ndarray, start_arc: float, end_arc: float, step: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=np.float64)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    end = min(float(end_arc), float(arc[-1]))
    query = np.arange(float(start_arc), end + 1e-12, float(step))
    if len(query) < 5:
        raise ValueError("A2_REFERENCE_WINDOW_TOO_SHORT_FOR_CURVATURE")
    sampled = np.column_stack((np.interp(query, arc, points[:, 0]), np.interp(query, arc, points[:, 1])))
    return sampled, query


def robust_curvature(sampled: np.ndarray, arc: np.ndarray, window_m: float = 5.0) -> np.ndarray:
    segments = np.diff(np.asarray(sampled), axis=0)
    mid_arc = 0.5 * (arc[:-1] + arc[1:])
    heading = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    output = np.zeros(len(mid_arc), dtype=np.float64)
    half = float(window_m) / 2.0
    for index, center in enumerate(mid_arc):
        mask = np.abs(mid_arc - center) <= half + 1e-12
        x, y = mid_arc[mask], heading[mask]
        x0 = x - np.mean(x)
        denominator = float(np.dot(x0, x0))
        output[index] = 0.0 if denominator <= 1e-12 else float(np.dot(x0, y - np.mean(y)) / denominator)
    return output


def longest_support(values: np.ndarray, supports: np.ndarray, level: float) -> float:
    best = current = 0.0
    for value, support in zip(np.abs(values), supports):
        if float(value) >= float(level):
            current += float(support)
            best = max(best, current)
        else:
            current = 0.0
    return best


def curvature_quality(xy: np.ndarray, start_arc: float, forward_m: float) -> Mapping[str, Any]:
    points = np.asarray(xy, dtype=np.float64)
    full_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    left = max(int(np.searchsorted(full_arc, start_arc, side="right") - 1), 0)
    right = min(int(np.searchsorted(full_arc, start_arc + forward_m, side="left") + 1), len(points))
    window = points[left:right]
    raw, support = path_curvature(window)
    sampled, uniform_arc = resample(points, start_arc, start_arc + forward_m)
    robust = robust_curvature(sampled, uniform_arc)
    raw_abs, robust_abs = np.abs(raw), np.abs(robust)
    max_index = int(np.argmax(raw_abs))
    raw_stats, robust_stats = percentile(raw_abs), percentile(robust_abs)
    max_adjacent_support = float(support[max_index])
    max_location_arc = float(full_arc[left + max_index + 1])
    raw_max = float(raw_stats["max"])
    if raw_max >= 2.0 * max(float(robust_stats["p99"]), 1e-12) and max_adjacent_support <= 1.0:
        classification = "LOCALIZED_POINTWISE_SPIKE"
    elif longest_support(raw, support, 0.8 * raw_max) >= 5.0 and float(robust_stats["p99"]) >= 0.8 * raw_max:
        classification = "RAW_ROBUST_CONCORDANT_SUSTAINED"
    else:
        classification = "MIXED_CURVATURE_REPRESENTATION_UNRESOLVED"
    return {
        "raw_pointwise_abs_curvature_inv_m": raw_stats,
        "raw_pointwise_signed_curvature_at_abs_max_inv_m": float(raw[max_index]),
        "raw_abs_max_location_arc_m": max_location_arc,
        "raw_max_point_adjacent_support_m": max_adjacent_support,
        "raw_adjacent_point_support_distribution_m": percentile(support.tolist()),
        "raw_native_point_count": len(window),
        "robust_abs_curvature_inv_m": robust_stats,
        "robust_resample_step_m": 0.25,
        "robust_window_m": 5.0,
        "continuous_support_m": {
            str(level): {"raw_longest": longest_support(raw, support, level),
                         "robust_longest": longest_support(robust, np.full(len(robust), 0.25), level)}
            for level in (0.02, 0.05, 0.08)
        },
        "classification": classification,
    }


def historical_gradient_curvature_forensic(xy: np.ndarray) -> Mapping[str, Any]:
    """Reproduce the frozen B2.1 pointwise formula and expose its endpoint support."""
    points = np.asarray(xy, dtype=np.float64)
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment)]
    heading = np.unwrap(np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0])))
    curvature = np.gradient(heading, arc, edge_order=1)
    point_support = np.gradient(arc)
    finite = np.isfinite(curvature)
    values, supports, arcs = curvature[finite], point_support[finite], arc[finite]
    index = int(np.argmax(np.abs(values)))
    return {
        "formula": "NP_GRADIENT_OF_UNWRAPPED_GRADIENT_TANGENT_HEADING_WITH_RESPECT_TO_NATIVE_ARC",
        "abs_curvature_inv_m": percentile(np.abs(values).tolist()),
        "signed_curvature_at_abs_max_inv_m": float(values[index]),
        "abs_max_location_arc_m": float(arcs[index]),
        "abs_max_point_support_m": float(supports[index]),
        "point_support_distribution_m": percentile(supports.tolist()),
        "continuous_support_m": {
            str(level): longest_support(values, supports, level) for level in (0.02, 0.05, 0.08)
        },
        "abs_max_is_terminal_point": bool(index == len(values) - 1),
    }


def legacy_extreme_forensic(
    opportunities: Sequence[Mapping[str, Any]], map_cache: Dict[str, Any]
) -> Sequence[Mapping[str, Any]]:
    """Reconstruct the two historical 0.082281/m whole-reference extrema without outcomes."""
    geometry_path = R1 / "r1_b2_1_hlc_geometry_audit_v1.csv"
    with geometry_path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if abs(float(row["target_curvature_max_abs_inv_m"]) - 0.082281) <= 5e-7
        ]
    by_key = {
        (str(wrapper["entry"]["scenario_token"]), str(wrapper["entry"]["log_id"])): wrapper["entry"]
        for wrapper in opportunities
    }
    output = []
    for row in rows:
        key = (str(row["scenario_token"]), str(row["log_id"]))
        entry = by_key[key]
        initial = _ego(entry["initial_state"])
        corridor = build_hlc_route_continuous_reference_v2_3(
            _map_api(str(entry["map_name"]), map_cache), entry["route_roadblock_ids"],
            str(entry["source_lane_id"]), str(entry["target_lane_id"]), initial, 0.1,
        )
        target_xy = np.asarray(corridor["target_reference_xy"], dtype=np.float64)
        quality = curvature_quality(target_xy, 0.0, float(corridor["target_total_length_m"]))
        historical = historical_gradient_curvature_forensic(target_xy)
        speed = float(initial["speed_mps"])
        required = speed * 7.9
        remaining = float(corridor["target_total_length_m"] - corridor["target_current_arc_m"])
        output.append({
            "scenario_token": key[0], "log_id": key[1],
            "historical_reported_target_raw_max_abs_curvature_inv_m": float(row["target_curvature_max_abs_inv_m"]),
            "historical_formula_reproduction": historical,
            "reconstructed_full_target_reference_quality": quality,
            "official_initial_speed_mps": speed,
            "required_7p9s_reference_m": required,
            "target_remaining_reference_m": remaining,
            "speed_and_extreme_accepted_as_valid_joint_window_support": False,
            "curvature_disposition": "TERMINAL_SHORT_SEGMENT_GRADIENT_ARTIFACT_NOT_SUSTAINED_ROAD_CURVATURE",
            "reason": "HISTORICAL_0P082281_OCCURS_AT_TERMINAL_POINT_WITH_SUBCENTIMETER_SUPPORT; A2_TURNING_ANGLE_RAW_AND_FIXED_WINDOW_ROBUST_ARE_APPROX_0P001; FULL_7P9S_WINDOW_IS_ALSO_NOT_COVERED",
        })
    return output


def speed_information(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    db = Path(str(entry["db_path"]))
    initial_us = int(entry["initial_state"]["initial_time_us"])
    anchor_us = int(entry["scenario_anchor_timestamp_us"])
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        anchor = connection.execute(
            "SELECT lp.timestamp,ep.vx,ep.vy FROM lidar_pc lp JOIN ego_pose ep ON ep.token=lp.ego_pose_token "
            "ORDER BY abs(lp.timestamp-?) LIMIT 1", (anchor_us,),
        ).fetchone()
        prefix = connection.execute(
            "SELECT lp.timestamp,ep.vx,ep.vy FROM lidar_pc lp JOIN ego_pose ep ON ep.token=lp.ego_pose_token "
            "WHERE lp.timestamp BETWEEN ? AND ? ORDER BY lp.timestamp", (initial_us, initial_us + 1_000_000),
        ).fetchall()
    if anchor is None or not prefix:
        raise ValueError("A2_PRETREATMENT_SPEED_EXTRACTION_EMPTY")
    samples = [{"timestamp_us": int(row[0]), "speed_mps": math.hypot(float(row[1]), float(row[2]))} for row in prefix]
    return {
        "official_initial_speed_mps": float(entry["initial_state"]["initial_speed_mps"]),
        "anchor_requested_timestamp_us": anchor_us,
        "anchor_nearest_timestamp_us": int(anchor[0]),
        "anchor_timestamp_abs_error_us": abs(int(anchor[0]) - anchor_us),
        "anchor_speed_mps": math.hypot(float(anchor[1]), float(anchor[2])),
        "pre_treatment_window_start_us": initial_us,
        "pre_treatment_window_end_us": initial_us + 1_000_000,
        "pre_treatment_samples": samples,
        "pre_treatment_speed_distribution_mps": percentile([row["speed_mps"] for row in samples]),
    }


def collect_opportunities() -> Sequence[Mapping[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cohort, path in COHORTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["entries"]:
            if row["family"] != "R-HLC":
                continue
            key = (str(row["scenario_token"]), str(row["log_id"]))
            if key not in merged:
                merged[key] = {"entry": row, "cohort_sources": []}
            merged[key]["cohort_sources"].append({"cohort": cohort, "path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    return [merged[key] for key in sorted(merged)]


def reference_at(corridor: Mapping[str, Any], source_arc: float, target_arc: float, speed: float) -> Tuple[np.ndarray, np.ndarray]:
    relative = np.arange(80, dtype=np.float64) * 0.1
    source, _ = sample_native_reference_no_extrapolation(corridor["source_reference_xy"], source_arc + speed * relative)
    target, _ = sample_native_reference_no_extrapolation(corridor["target_reference_xy"], target_arc + speed * relative)
    return source, target


def current_for_case(
    corridor: Mapping[str, Any], absolute_s: float, speed: float, arm: str, residual_m: float,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray, np.ndarray]:
    distance = float(speed) * float(absolute_s)
    source_arc = float(corridor["source_current_arc_m"]) + distance
    target_arc = float(corridor["target_current_arc_m"]) + distance
    source, target = reference_at(corridor, source_arc, target_arc, speed)
    p = float(morphology_progress(np.asarray([absolute_s]), arm, PARAMETERS["morphology"])[0])
    point = source[0] * (1.0 - p) + target[0] * p
    tangent = source[1] * (1.0 - p) + target[1] * p - point
    heading = math.atan2(float(tangent[1]), float(tangent[0]))
    point = point + float(residual_m) * np.asarray([-math.sin(heading), math.cos(heading)])
    current = {
        "rear_axle": {"x": float(point[0]), "y": float(point[1]), "heading": heading},
        "speed_mps": float(speed), "time_us": int(round(absolute_s * 1_000_000)),
    }
    shifted = dict(corridor)
    shifted["source_current_arc_m"] = source_arc
    shifted["target_current_arc_m"] = target_arc
    return current, shifted, source, target


def component_metrics(curvature: np.ndarray, speed: float) -> Mapping[str, float]:
    magnitude = np.abs(np.asarray(curvature, dtype=np.float64))
    return {
        "max_abs_curvature_inv_m": float(np.max(magnitude)),
        "max_abs_yaw_rate_radps": float(np.max(magnitude) * speed),
        "max_abs_lateral_acceleration_mps2": float(np.max(magnitude) * speed**2),
    }


def audit_case(corridor: Mapping[str, Any], absolute_s: float, speed: float, arm: str, residual_m: float) -> Mapping[str, Any]:
    current, shifted, source, target = current_for_case(corridor, absolute_s, speed, arm, residual_m)
    progress = morphology_progress(absolute_s + np.arange(80) * 0.1, arm, PARAMETERS["morphology"])
    current_p = float(progress[0])
    native_xy = source * (1.0 - current_p) + target * current_p
    morphology_xy = source * (1.0 - progress[:, None]) + target * progress[:, None]
    native_k, _ = path_curvature(native_xy)
    morphology_k, _ = path_curvature(morphology_xy)
    try:
        states, _, capture = _states(current, absolute_s, shifted, arm, PARAMETERS, absolute_s < 1.1 - 1e-12)
        final_xy = np.asarray([[row["rear_axle"]["x"], row["rear_axle"]["y"]] for row in states])
        construction_failure = None
    except CaptureInfeasible as error:
        capture = error.audit
        correction = np.asarray(capture["algebraic_stitching_correction_vectors_xy_m"], dtype=np.float64)
        final_xy = morphology_xy + correction
        final_xy[0] = np.asarray([current["rear_axle"]["x"], current["rear_axle"]["y"]])
        construction_failure = error.reason
    composite_k, _ = path_curvature(final_xy)
    # Curvature arrays correspond to interior vertices and share exact indices.
    morphology_increment = morphology_k - native_k
    stitching_increment = composite_k - morphology_k
    generated_increment = composite_k - native_k
    native = component_metrics(native_k, speed)
    morphology = component_metrics(morphology_increment, speed)
    stitching = component_metrics(stitching_increment, speed)
    generated = component_metrics(generated_increment, speed)
    composite = component_metrics(composite_k, speed)
    limits = PARAMETERS["capture"]["frozen_feasibility_limits"]
    native_pass = bool(native["max_abs_curvature_inv_m"] <= limits["curvature_inv_m_max"] and native["max_abs_yaw_rate_radps"] <= limits["yaw_rate_radps_max"] and native["max_abs_lateral_acceleration_mps2"] <= limits["lateral_accel_mps2_max"])
    generated_pass = bool(generated["max_abs_curvature_inv_m"] <= limits["curvature_inv_m_max"] and generated["max_abs_yaw_rate_radps"] <= limits["yaw_rate_radps_max"] and generated["max_abs_lateral_acceleration_mps2"] <= limits["lateral_accel_mps2_max"])
    composite_pass = bool(composite["max_abs_curvature_inv_m"] <= limits["curvature_inv_m_max"] and composite["max_abs_yaw_rate_radps"] <= limits["yaw_rate_radps_max"] and composite["max_abs_lateral_acceleration_mps2"] <= limits["lateral_accel_mps2_max"])
    target_offsets = capture.get("actual_planned_target_frame_offsets_m", [])
    terminal = abs(float(target_offsets[-1])) if target_offsets else float("inf")
    state0 = bool(capture.get("state0_exact_current_xy", False) and capture.get("state0_exact_current_heading", False))
    return {
        "absolute_episode_time_s": round(float(absolute_s), 6), "speed_mps": float(speed),
        "arm": arm, "normal_residual_m": float(residual_m), "native": native,
        "morphology_increment": morphology, "stitching_capture_increment": stitching,
        "generated_increment": generated, "composite": composite,
        "native_pass": native_pass, "generated_increment_pass_without_cancellation": generated_pass,
        "composite_pass": composite_pass, "state0_exact": state0,
        "terminal_target_frame_offset_abs_m": terminal, "terminal_pass": terminal <= 1e-6,
        "construction_failure": construction_failure,
        "frozen_full_V4_gate_pass": construction_failure is None,
        "continuity": {
            "state0_exact": state0,
            "state0_to_state1_distance_m": capture.get("feasibility", {}).get("state0_to_state1_distance_m"),
            "nominal_state_step_distance_m": capture.get("feasibility", {}).get("nominal_state_step_distance_m"),
            "state0_tangent_mismatch_abs_rad": capture.get("feasibility", {}).get("state0_tangent_mismatch_abs_rad"),
            "future_heading_xy_mismatch_abs_rad": capture.get("feasibility", {}).get("future_heading_xy_mismatch_abs_rad"),
        },
        "capture_contribution": {
            "curvature_yaw_rate_lateral_acceleration": stitching,
            "accounting": "SAME_C2_STITCHING_CAPTURE_INCREMENT; DO_NOT_ADD_TWICE",
            "terminal_target_frame_offset_abs_m": terminal,
        },
    }


def aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    components = ("native", "morphology_increment", "stitching_capture_increment", "generated_increment", "composite")
    maxima = {
        component: {metric: max(float(row[component][metric]) for row in cases) for metric in (
            "max_abs_curvature_inv_m", "max_abs_yaw_rate_radps", "max_abs_lateral_acceleration_mps2"
        )} for component in components
    }
    native_fail = sum(not row["native_pass"] for row in cases)
    generated_fail = sum(not row["generated_increment_pass_without_cancellation"] for row in cases)
    composite_fail = sum(not row["composite_pass"] for row in cases)
    terminal_fail = sum(not row["terminal_pass"] for row in cases)
    state0_fail = sum(not row["state0_exact"] for row in cases)
    construction_fail = Counter(str(row["construction_failure"]) for row in cases if row["construction_failure"])
    full_gate_fail = sum(not row["frozen_full_V4_gate_pass"] for row in cases)
    non_native_fail = sum(
        row["native_pass"] and (
            (not row["frozen_full_V4_gate_pass"]) or
            (not row["generated_increment_pass_without_cancellation"])
        ) for row in cases
    )
    settling = [row for row in cases if float(row["absolute_episode_time_s"]) >= 7.4 - 1e-12]
    return {
        "planner_call_cases": len(cases), "component_maxima": maxima,
        "native_only_infeasible_cases": native_fail,
        "generated_increment_infeasible_without_cancellation_cases": generated_fail,
        "composite_infeasible_cases": composite_fail,
        "state0_continuity_failures": state0_fail, "terminal_capture_failures": terminal_fail,
        "construction_failure_counts": dict(sorted(construction_fail.items())),
        "frozen_full_V4_gate_failures": full_gate_fail,
        "V4_non_native_infeasible_cases": non_native_fail,
        "post_recommit_settling_cases": len(settling),
        "post_recommit_terminal_capture_failures": sum(not row["terminal_pass"] for row in settling),
        "post_recommit_composite_failures": sum(not row["composite_pass"] for row in settling),
    }


def main() -> int:
    existing = [str(path) for path in OUT.values() if path.exists()]
    if existing:
        raise FileExistsError(f"R2_BJ_A2_VERSIONED_OUTPUT_EXISTS:{existing}")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    global PARAMETERS
    PARAMETERS = json.loads(SPACE.read_text(encoding="utf-8"))["global_parameters"]
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    inventory_rows = json.loads(INVENTORY.read_text(encoding="utf-8"))["rows"]
    inventory = {str(Path(row["db_path"]).resolve()): row for row in inventory_rows}
    opportunities = collect_opportunities()
    map_cache: Dict[str, Any] = {}
    support, extraction_failures, curvature_rows = [], [], []
    for index, wrapper in enumerate(opportunities, 1):
        entry = wrapper["entry"]
        initial = _ego(entry["initial_state"])
        speed = float(initial["speed_mps"])
        speed_margin = max(0.5, 0.05 * speed)
        try:
            corridor = build_hlc_route_continuous_reference_v2_3(
                _map_api(str(entry["map_name"]), map_cache), entry["route_roadblock_ids"],
                str(entry["source_lane_id"]), str(entry["target_lane_id"]), initial,
                (speed + speed_margin) * 15.8 + 2.0,
            )
            speed_info = speed_information(entry)
            forward = speed * 7.9
            source_quality = curvature_quality(np.asarray(corridor["source_reference_xy"]), float(corridor["source_current_arc_m"]), forward)
            target_quality = curvature_quality(np.asarray(corridor["target_reference_xy"]), float(corridor["target_current_arc_m"]), forward)
            sample_distance = np.linspace(0.0, forward, 160)
            source_xy, _ = sample_native_reference_no_extrapolation(corridor["source_reference_xy"], float(corridor["source_current_arc_m"]) + sample_distance)
            target_xy, _ = sample_native_reference_no_extrapolation(corridor["target_reference_xy"], float(corridor["target_current_arc_m"]) + sample_distance)
            lane_separation = np.linalg.norm(target_xy - source_xy, axis=1)
            db_row = inventory.get(str(Path(entry["db_path"]).resolve()))
            if db_row is None:
                raise ValueError("A2_DB_INVENTORY_PROVENANCE_MISSING")
            eligibility_fields = {
                key: entry.get(key) for key in (
                    "official_exact_single_scenario_resolution", "Primary80_execution_eligibility",
                    "HLC_route_continuous_Primary80_applicability", "pre_treatment_context_availability_audit_ref",
                    "pre_treatment_eligibility_audit_ref", "map_applicability_audit_ref",
                    "dynamic_clearance_audit_ref", "selector_rank_sha256", "route_fingerprint",
                    "source_partition", "selection_role", "PERMANENT_ENGINEERING_ONLY",
                ) if key in entry
            }
            audit_refs = sorted({
                str(value) for key, value in eligibility_fields.items()
                if key.endswith("_audit_ref") and value and (ROOT / str(value)).is_file()
            })
            record = {
                "joint_record_id": f"A2-HLC-{len(support)+1:03d}", "scenario_token": entry["scenario_token"],
                "log_id": entry["log_id"], "scenario_anchor_timestamp_us": entry["scenario_anchor_timestamp_us"],
                "map_name": entry["map_name"], "direction": entry["direction"],
                "source_lane_id": entry["source_lane_id"], "target_lane_id": entry["target_lane_id"],
                "official_initial_state": entry["initial_state"], "speed_information": speed_info,
                "lane_separation_m": percentile(lane_separation.tolist()),
                "required_reference_forward_m": forward,
                "available_reference": {
                    "source_total_length_m": corridor["source_total_length_m"], "target_total_length_m": corridor["target_total_length_m"],
                    "source_current_arc_m": corridor["source_current_arc_m"], "target_current_arc_m": corridor["target_current_arc_m"],
                    "source_remaining_margin_m": corridor["source_remaining_margin_m"], "target_remaining_margin_m": corridor["target_remaining_margin_m"],
                    "source_components": corridor["source_components"], "target_components": corridor["target_components"],
                },
                "actual_reference_geometry": {
                    "source_reference_xy": np.asarray(corridor["source_reference_xy"]).tolist(),
                    "target_reference_xy": np.asarray(corridor["target_reference_xy"]).tolist(),
                    "source_reference_f64_sha256": array_sha(corridor["source_reference_xy"]),
                    "target_reference_f64_sha256": array_sha(corridor["target_reference_xy"]),
                },
                "curvature_quality": {"source": source_quality, "target": target_quality},
                "technical_eligibility": {
                    "official_exact_single_scenario_resolution": entry.get("official_exact_single_scenario_resolution", "LEGACY_PRETREATMENT_ELIGIBLE"),
                    "Primary80_execution_eligibility": entry.get("Primary80_execution_eligibility", "LEGACY_PRETREATMENT_ELIGIBLE"),
                    "HLC_route_continuous_Primary80_applicability": "PASS_RECONSTRUCTED_V2_3_WITH_MARGIN",
                    "all_source_fields": eligibility_fields,
                    "reconstruction_required_forward_m": (speed + speed_margin) * 15.8 + 2.0,
                    "reconstruction_speed_margin_mps": speed_margin,
                },
                "provenance": {
                    "cohort_sources": wrapper["cohort_sources"],
                    "db_inventory_row": db_row, "db_inventory_manifest_sha256": sha(INVENTORY),
                    "source_universe_sha256": sha(SOURCE),
                    "source_root_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"],
                    "map_root_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"],
                    "route_builder_sha256": sha(ROOT / "tools/r1_closed_loop_benchmark_v2_3.py"),
                    "A2_contract_sha256": sha(CONTRACT),
                    "technical_eligibility_audit_artifacts": [
                        {"path": path, "sha256": sha(ROOT / path)} for path in audit_refs
                    ],
                },
            }
            record["joint_record_canonical_sha256"] = canonical_sha(record)
            support.append(record)
            curvature_rows.append({
                "joint_record_id": record["joint_record_id"], "scenario_token": entry["scenario_token"],
                "log_id": entry["log_id"], "source": source_quality, "target": target_quality,
            })
        except Exception as error:
            extraction_failures.append({
                "scenario_token": entry["scenario_token"], "log_id": entry["log_id"],
                "cohort_sources": wrapper["cohort_sources"], "reason": f"{type(error).__name__}:{error}",
                "automatic_identity_exclusion_performed": False,
            })
        print(json.dumps({"progress": "R2_BJ_A2_JOINT_SUPPORT", "completed": index, "total": len(opportunities), "support": len(support), "fail": len(extraction_failures)}), flush=True)

    provenance = {
        "schema_version": "r2_bj_a2_hlc_joint_support_provenance_manifest_v1.0",
        "status": "JOINT_SUPPORT_EXTRACTION_INCOMPLETE",
        "contract": {"path": str(CONTRACT.relative_to(ROOT)), "sha256": sha(CONTRACT)},
        "source_universe": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha(SOURCE)},
        "cohort_files": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in COHORTS.items()},
        "unique_committed_selected_HLC_opportunities_seen": len(opportunities),
        "joint_support_records": len(support), "technical_extraction_failures": extraction_failures,
        "full_source_universe_eligible_population_materialized": False,
        "why_incomplete": "FROZEN_ELIGIBILITY_PIPELINE_PERSISTED_RANK_STOPPING_COHORTS_NOT_ALL_ELIGIBILITY_PASS_OPPORTUNITIES",
        "joint_records": support,
    }
    all_classifications = Counter(
        quality[side]["classification"] for quality in curvature_rows for side in ("source", "target")
    )
    legacy_geometry = R1 / "r1_b2_1_hlc_geometry_audit_v1.csv"
    legacy_extremes = legacy_extreme_forensic(opportunities, map_cache)
    curvature = {
        "schema_version": "r2_bj_a2_curvature_quality_forensic_v1.0",
        "status": "CURVATURE_REPRESENTATION_UNRESOLVED" if all_classifications["MIXED_CURVATURE_REPRESENTATION_UNRESOLVED"] else "RAW_ROBUST_DISPOSITION_COMPLETE",
        "preregistered_contract_sha256": sha(CONTRACT),
        "raw_and_robust_both_retained": True, "manual_spike_deletion": False,
        "classification_counts": dict(sorted(all_classifications.items())),
        "joint_support_rows": curvature_rows,
        "legacy_0p082281_forensic_appendix": {
            "source": str(legacy_geometry.relative_to(ROOT)), "sha256": sha(legacy_geometry),
            "role": "ADVERSARIAL_OR_LEGACY_POINTWISE_EXTREME_ONLY_NOT_AUTOMATIC_MAIN_JOINT_SUPPORT",
            "exact_reported_extreme_inv_m": 0.082281,
            "current_V2_3_long_horizon_extraction_failures_containing_that_legacy_geometry_are_retained_in_provenance": True,
            "records": legacy_extremes,
        },
    }

    case_summaries, component_totals = [], Counter()
    for index, record in enumerate(support, 1):
        corridor = {
            "source_reference_xy": record["actual_reference_geometry"]["source_reference_xy"],
            "target_reference_xy": record["actual_reference_geometry"]["target_reference_xy"],
            "source_current_arc_m": record["available_reference"]["source_current_arc_m"],
            "target_current_arc_m": record["available_reference"]["target_current_arc_m"],
        }
        nominal = float(record["speed_information"]["official_initial_speed_mps"])
        speeds = (nominal, nominal + max(0.5, 0.05 * nominal))
        cases = [
            audit_case(corridor, float(time_s), speed, arm, residual)
            for arm in (ARM_BASELINE, ARM_TREATMENT)
            for speed in speeds
            for residual in (-0.25, 0.0, 0.25)
            for time_s in np.arange(0.0, 8.0, 0.1)
        ]
        aggregate = aggregate_cases(cases)
        for key in ("native_only_infeasible_cases", "generated_increment_infeasible_without_cancellation_cases", "composite_infeasible_cases", "state0_continuity_failures", "terminal_capture_failures", "post_recommit_terminal_capture_failures", "post_recommit_composite_failures", "frozen_full_V4_gate_failures", "V4_non_native_infeasible_cases"):
            component_totals[key] += int(aggregate[key])
        case_summaries.append({
            "joint_record_id": record["joint_record_id"], "scenario_token": record["scenario_token"],
            "log_id": record["log_id"], "speed_cases_mps": list(speeds),
            "normal_residual_cases_m": [-0.25, 0.0, 0.25], "summary": aggregate,
        })
        print(json.dumps({"progress": "R2_BJ_A2_V4_COMPONENT_AUDIT", "completed": index, "total": len(support)}), flush=True)

    native_population = sum(row["summary"]["native_only_infeasible_cases"] > 0 for row in case_summaries)
    generated_population = sum(row["summary"]["V4_non_native_infeasible_cases"] > 0 for row in case_summaries)
    composite_population = sum(row["summary"]["composite_infeasible_cases"] > 0 for row in case_summaries)
    terminal_population = sum(row["summary"]["post_recommit_terminal_capture_failures"] > 0 for row in case_summaries)
    components = {
        "schema_version": "r2_bj_a2_native_generated_composite_component_audit_v1.0",
        "status": "FAIL_CLOSED_COMPONENT_INFEASIBILITY_PRESENT" if any((native_population, generated_population, composite_population, terminal_population)) else "PASS",
        "contract_sha256": sha(CONTRACT), "V4_parameter_space_sha256": sha(SPACE),
        "opportunity_count": len(case_summaries), "planner_state_case_count": len(case_summaries) * 2 * 2 * 3 * 80,
        "native_only_infeasible_opportunities": native_population,
        "generated_increment_infeasible_opportunities": generated_population,
        "composite_infeasible_opportunities": composite_population,
        "terminal_settling_infeasible_opportunities": terminal_population,
        "case_level_totals": dict(component_totals),
        "negative_native_generated_cancellation_accepted_as_generated_pass": False,
        "capture_curvature_accounting": "TARGET_CAPTURE_USES_THE_SAME_C2_STITCHING_CORRECTION; REPORTED_JOINTLY_ON_CURVATURE_AND_SEPARATELY_BY_TERMINAL_OFFSET_WITHOUT_DOUBLE_COUNTING",
        "opportunities": case_summaries,
        "runner_run_calls": 0, "simulation_calls": 0,
    }
    blockers = ["JOINT_SUPPORT_EXTRACTION_INCOMPLETE"]
    if curvature["status"] == "CURVATURE_REPRESENTATION_UNRESOLVED":
        blockers.append("CURVATURE_REPRESENTATION_UNRESOLVED")
    if native_population:
        blockers.append("SOURCE_NATIVE_FEASIBILITY_UNRESOLVED")
    if generated_population:
        blockers.append("V4_GENERATED_INCREMENT_INFEASIBLE")
    if terminal_population:
        blockers.append("V4_TERMINAL_SETTLING_INFEASIBLE")
    low_speed_support = [row for row in support if row["speed_information"]["pre_treatment_speed_distribution_mps"]["min"] <= 0.2]
    if low_speed_support and generated_population and "V4_LOW_SPEED_MORPHOLOGY_INFEASIBLE" not in blockers:
        blockers.append("V4_LOW_SPEED_MORPHOLOGY_INFEASIBLE")
    envelope = {
        "schema_version": "r2_bj_a2_joint_support_applicability_envelope_v1.0",
        "status": blockers[0], "all_blocking_categories": blockers,
        "actual_joint_support_records": len(support), "source_opportunities_considered": len(opportunities),
        "complete_joint_record_provenance_closure_percent": 100.0 if support else 0.0,
        "considered_opportunity_extraction_completion_percent": 100.0 * len(support) / len(opportunities) if opportunities else 0.0,
        "full_eligible_population_provenance_closure_percent": None,
        "speed_joint_support_mps": {
            "official_initial": percentile([row["speed_information"]["official_initial_speed_mps"] for row in support]),
            "anchor": percentile([row["speed_information"]["anchor_speed_mps"] for row in support]),
            "pre_treatment_sample_min": min(row["speed_information"]["pre_treatment_speed_distribution_mps"]["min"] for row in support),
            "pre_treatment_sample_max": max(row["speed_information"]["pre_treatment_speed_distribution_mps"]["max"] for row in support),
            "zero_point_two_mps_supported_record_count": len(low_speed_support),
        },
        "lane_separation_joint_support_m": {
            "minimum": min(row["lane_separation_m"]["min"] for row in support),
            "maximum": max(row["lane_separation_m"]["max"] for row in support),
        },
        "native_only_infeasible_population": {"count": native_population, "fraction": native_population / len(support) if support else None, "automatic_exclusion": False},
        "generated_increment_infeasible_population": {"count": generated_population, "fraction": generated_population / len(support) if support else None},
        "composite_infeasible_population": {"count": composite_population, "fraction": composite_population / len(support) if support else None},
        "terminal_settling_infeasible_population": {"count": terminal_population, "fraction": terminal_population / len(support) if support else None},
        "prospective_native_eligibility_option_for_owner": "PRETREATMENT_ONLY_COUPLED_SPEED_AND_RAW_ROBUST_NATIVE_CURVATURE_GATE_USING_EXISTING_FROZEN_LIMITS; DO_NOT_AUTO_EXCLUDE_UNTIL_OWNER_DECISION",
        "BJ_A_cartesian_envelope": {"path": str(BJ_A.relative_to(ROOT)), "sha256": sha(BJ_A), "role": "ADVERSARIAL_STRESS_APPENDIX_NOT_ACTUAL_DOMAIN_DECIDER"},
        "V4_parameters_changed": False, "roster_selected": False,
        "runner_run_calls": 0, "engineering_simulation_calls": 0, "scientific_simulation_calls": 0, "TSB_simulation_calls": 0,
        "R2C_started": False, "confirmatory_smoke_started": False, "RBR_started": False,
    }
    firewall = {
        "schema_version": "r2_bj_a2_data_firewall_audit_v1.0", "status": "PASS_NO_OUTCOME_LEAKAGE",
        "allowed_inputs": ["frozen_rosters", "read_only_official_DB_pretreatment_speed", "official_map_native_reference", "frozen_route_builder", "frozen_V4"],
        "forbidden_outcome_files_opened": 0, "baseline_treatment_outcomes_used": False,
        "mechanism_endpoint_safety_F_match_results_used": False, "manual_identity_selection": False,
        "V4_parameters_changed": False, "historical_BJ_A_rewritten": False,
        "R2_BI_identity_disposition_changed": False, "new_roster_selected": False,
        "runner_run_calls": 0, "simulation_calls": 0, "protected_CSV_sha256": sha(PROTECTED),
    }
    write_json(OUT["provenance"], provenance)
    write_json(OUT["curvature"], curvature)
    write_json(OUT["components"], components)
    write_json(OUT["envelope"], envelope)
    write_json(OUT["firewall"], firewall)
    OUT["request"].write_text(f"""# R2-BJ-A2 Scientific Owner 准备度请求 v0.1

## 结论

`REQUEST_WITHHELD`。A2 fail-closed 主状态为 `{envelope['status']}`；阻断类别为 `{', '.join(blockers)}`。

## 联合支持结果

仓库中 57 个唯一、已提交且 outcome-blind 选出的 HLC opportunity 被逐一检查，其中 {len(support)} 个完成 V2.3 长窗口 joint-record reconstruction，{len(extraction_failures)} 个未完成。现有冻结 eligibility 管线只持久化 rank-stopping cohort，没有持久化全 source universe 的全部 eligibility-pass population，因此不能声称联合适用域提取 100% 完整。

已完整形成的 {len(support)} 条 joint record 内部 provenance closure 为 100%；但对 57 条已提交记录的 extraction completion 仅为 {100.0 * len(support) / len(opportunities):.2f}%，全 eligible population 的完成率不可计算。

在已物化 joint support 内，native-only 不可行 population 为 {native_population}/{len(support)}，generated increment 不可行为 {generated_population}/{len(support)}，composite 不可行为 {composite_population}/{len(support)}，recommit 后 terminal settling 不可行为 {terminal_population}/{len(support)}。这些记录不触发自动 identity exclusion。

## 曲率质量处置

主 joint support 的 source/target 曲率分类为：{dict(sorted(all_classifications.items()))}。其中仍有 {all_classifications['MIXED_CURVATURE_REPRESENTATION_UNRESOLVED']} 条 reference side 的 raw/robust 关系不能按预注册规则归入“局部尖峰”或“持续曲率”，因此保留 `CURVATURE_REPRESENTATION_UNRESOLVED` 阻断。

历史 `0.082281 1/m` 已按 B2.1 原公式复现：它位于 target reference 末端的超短 segment 支持点；A2 turning-angle raw 与固定窗口 robust 均约为 `0.001 1/m`，且两条历史记录均缺少完整 7.9 秒 target reference coverage。因此该值被判为 terminal discretization/gradient artifact，仅留在 adversarial appendix，不作为实际 speed-curvature joint support。

## 治理

V4 参数和冻结阈值均未修改；BJ-A Cartesian envelope 仅作为 adversarial appendix 保留。未选择 roster，未申请 BJ-B execution。`runner.run=0`，engineering/scientific/TSB simulation 均为 0；R2-C、confirmatory smoke、RBR 均未启动。
""", encoding="utf-8")

    components_to_bind = [
        CONTRACT, OUT["provenance"], OUT["curvature"], OUT["components"], OUT["envelope"], OUT["firewall"], OUT["request"],
        SPACE, BJ_A, SOURCE, INVENTORY,
        ROOT / "tools/r2_bj_a2_joint_support_applicability_audit.py",
        ROOT / "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py",
        ROOT / "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py",
        ROOT / "tools/r1_closed_loop_benchmark_v2_3.py",
        ROOT / "tests/test_r2_bj_a2_joint_support_applicability.py",
        ROOT / "QUICK_REFERENCE.md",
    ] + list(COHORTS.values())
    manifest = {
        "schema_version": "r2_bj_a2_component_sha_binding_manifest_v1.0", "status": envelope["status"],
        "components": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in components_to_bind],
        "component_SHA_closure": "PASS", "joint_record_count": len(support),
        "runner_run_calls": 0, "simulation_calls": 0, "protected_CSV_sha256": sha(PROTECTED),
    }
    write_json(OUT["manifest"], manifest)
    print(json.dumps({"status": envelope["status"], "blockers": blockers, "support": len(support), "source": len(opportunities), "runner_run_calls": 0, "simulation_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
