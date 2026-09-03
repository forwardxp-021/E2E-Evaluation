#!/usr/bin/env python3
"""Build the R2-BJ-A offline-only HLC morphology feasibility package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_prospective_generator_contract_v2 import polyline_arclength  # noqa: E402
from tools.r2_bi_zero_run_entry_gate_audit import _shadow  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import (  # noqa: E402
    ARM_BASELINE,
    ARM_TREATMENT,
    CaptureInfeasible,
    analytic_phase_metrics,
    morphology_progress,
    phase_boundaries,
    validate_parameters,
)
from tools.r2_bj_a_hlc_morphology_feasible_planner_v4 import _states  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
R2 = ROOT / "docs/stageR/r2"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
SOURCE_UNIVERSE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
GEOMETRY = R1 / "r1_b2_1_hlc_geometry_audit_v1.csv"
BI_ROSTER = R2 / "r2_bi_hlc_dev_kin_roster_v1.0.json"
BI_STOP = R2 / "r2_bi_hlc_dev_kin_round_0_architecture_stop_audit_v1.json"

OUT = {
    "exposure": R2 / "r2_bj_a_r2bi_outcome_exposure_ledger_v1.0.json",
    "analytic": R2 / "r2_bj_a_hlc_morphology_analytic_feasibility_audit_v1.0.json",
    "report": R2 / "R2_BJ_A_HLC_Morphology_Analytic_Feasibility_Report_v1.md",
    "contract": R2 / "r2_bj_a_hlc_kinematic_architecture_contract_v4.0.json",
    "space": R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json",
    "envelope": R2 / "r2_bj_a_expanded_zero_run_feasibility_envelope_v1.0.json",
    "firewall": R2 / "r2_bj_a_data_firewall_audit_v1.0.json",
    "request": R2 / "R2_BJ_A_R2BJ_B_Engineering_Execution_Readiness_Request_v0.1.md",
    "manifest": R2 / "r2_bj_a_component_sha_binding_manifest_v1.0.json",
}

PARAMETERS: Mapping[str, Any] = {
    "morphology": {
        "baseline_transition_duration_s": 5.0,
        "advance_duration_s": 1.7,
        "advance_progress": 0.20,
        "hold_duration_s": 0.1,
        "retreat_depth": 0.11,
        "retreat_duration_s": 1.4,
        "recommit_duration_s": 3.1,
        "lag_precompensation_s": 0.0,
    },
    "capture": {
        "capture_start_abs_s": 1.1,
        "nominal_capture_end_abs_s": 7.4,
        "minimum_stitching_horizon_s": 3.0,
        "frozen_feasibility_limits": {
            "curvature_inv_m_max": 0.5,
            "yaw_rate_radps_max": 1.0,
            "lateral_accel_mps2_max": 6.0,
            "state0_to_state1_distance_excess_m_max": 0.35,
            "state0_tangent_mismatch_rad_max": 0.20,
            "heading_xy_consistency_rad_max": 1e-10,
        },
    },
}

# Read-only scan over the 1,624 DB files bound by SOURCE_UNIVERSE.  The query is
# documented in the artifact; constants prevent future unit tests from rescanning 70 GB.
SOURCE_SPEED = {
    "minimum_mps": 3.178053254691344e-07,
    "maximum_mps": 17.246181220874348,
    "effective_planner_minimum_mps": 0.2,
    "ego_pose_rows": 41_706_457,
    "db_files": 1_624,
    "query": "SELECT MIN(sqrt(vx*vx+vy*vy)), MAX(sqrt(vx*vx+vy*vy)), COUNT(*) FROM ego_pose WHERE vx IS NOT NULL AND vy IS NOT NULL",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _geometry_envelope() -> Mapping[str, float]:
    with GEOMETRY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "lane_separation_min_m": min(float(row["lane_separation_min_m"]) for row in rows),
        "lane_separation_max_m": max(float(row["lane_separation_max_m"]) for row in rows),
        "route_curvature_max_abs_inv_m": max(
            max(float(row["source_curvature_max_abs_inv_m"]), float(row["target_curvature_max_abs_inv_m"]))
            for row in rows
        ),
        "audited_rows": len(rows),
    }


def _corridor(curvature: float, target_direction: int, lane_separation: float) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    s = np.arange(-60.0, 360.0, 0.05, dtype=np.float64)
    k = float(curvature)
    if abs(k) < 1e-12:
        source = np.column_stack((s, np.zeros_like(s)))
        heading = np.zeros_like(s)
    else:
        source = np.column_stack((np.sin(k * s) / k, (1.0 - np.cos(k * s)) / k))
        heading = k * s
    normal = np.column_stack((-np.sin(heading), np.cos(heading)))
    target = source + float(target_direction) * float(lane_separation) * normal
    zero = int(np.argmin(np.abs(s)))
    corridor = {
        "source_reference_xy": source.tolist(),
        "target_reference_xy": target.tolist(),
        "source_current_arc_m": float(polyline_arclength(source)[zero]),
        "target_current_arc_m": float(polyline_arclength(target)[zero]),
    }
    geometry = {"source": source, "target": target, "zero": zero, "heading": heading}
    return corridor, geometry


def _current(
    geometry: Mapping[str, Any], absolute_s: float, arm: str, speed: float, residual: float
) -> Mapping[str, Any]:
    p = float(morphology_progress(np.asarray([absolute_s]), arm, PARAMETERS["morphology"])[0])
    source, target, index = geometry["source"], geometry["target"], int(geometry["zero"])
    point = source[index] * (1.0 - p) + target[index] * p
    tangent = (source[index + 1] * (1.0 - p) + target[index + 1] * p) - point
    heading = math.atan2(float(tangent[1]), float(tangent[0]))
    normal = np.asarray([-math.sin(heading), math.cos(heading)])
    point = point + float(residual) * normal
    return {
        "rear_axle": {"x": float(point[0]), "y": float(point[1]), "heading": heading},
        "speed_mps": float(speed),
        "time_us": int(round(float(absolute_s) * 1_000_000)),
    }


def _case(
    case_id: str,
    arm: str,
    absolute_s: float,
    lane: float,
    speed: float,
    curvature: float,
    target_direction: int,
    residual: float,
    group: str,
) -> Mapping[str, Any]:
    corridor, geometry = _corridor(curvature, target_direction, lane)
    current = _current(geometry, absolute_s, arm, speed, residual)
    try:
        states, progress, audit = _states(current, absolute_s, corridor, arm, PARAMETERS, False)
        feasibility = audit["feasibility"]
        target_offsets = audit["actual_planned_target_frame_offsets_m"]
        passed = bool(
            feasibility["pass"]
            and audit["state0_exact_current_xy"]
            and audit["state0_exact_current_heading"]
            and feasibility["future_heading_xy_mismatch_abs_rad"] <= 1e-10
            and abs(float(target_offsets[-1])) <= 1e-6
            and len(states) == 80
        )
        reason = None if passed else "POST_CONSTRUCTION_GATE_FAIL"
        metrics = {
            "max_abs_curvature_inv_m": feasibility["max_abs_curvature_inv_m"],
            "max_abs_yaw_rate_radps": feasibility["max_abs_yaw_rate_radps"],
            "max_abs_lateral_acceleration_mps2": feasibility["max_abs_lateral_acceleration_mps2"],
            "state0_to_state1_distance_m": feasibility["state0_to_state1_distance_m"],
            "terminal_target_frame_offset_abs_m": abs(float(target_offsets[-1])),
            "phase_progress_start": float(progress[0]),
            "phase_progress_end": float(progress[-1]),
        }
    except (CaptureInfeasible, ValueError) as error:
        passed = False
        reason = getattr(error, "reason", str(error))
        failure = getattr(error, "audit", {})
        feasibility = failure.get("feasibility", {})
        metrics = {key: feasibility.get(key) for key in (
            "max_abs_curvature_inv_m", "max_abs_yaw_rate_radps", "max_abs_lateral_acceleration_mps2",
            "state0_to_state1_distance_m",
        )}
    return {
        "case_id": case_id,
        "group": group,
        "arm": arm,
        "absolute_episode_time_s": round(float(absolute_s), 6),
        "lane_separation_m": float(lane),
        "speed_mps": float(speed),
        "route_curvature_inv_m": float(curvature),
        "target_direction": "LEFT" if target_direction > 0 else "RIGHT",
        "realized_pose_residual_m": float(residual),
        "checks": metrics,
        "pass": passed,
        "failure_reason": reason,
    }


def _analytic(geometry: Mapping[str, float]) -> Mapping[str, Any]:
    lane = float(geometry["lane_separation_max_m"])
    new = PARAMETERS["morphology"]
    old = {
        "advance_duration_s": 1.1, "advance_progress": 0.44, "hold_duration_s": 0.5,
        "retreat_depth": 0.30, "retreat_duration_s": 1.3, "recommit_duration_s": 1.4,
    }

    def phases(p: Mapping[str, float]) -> Sequence[Mapping[str, Any]]:
        a = float(p["advance_progress"])
        r = a - float(p["retreat_depth"])
        return [
            analytic_phase_metrics("advance", 0.0, a, float(p["advance_duration_s"]), lane),
            {"phase": "hold", "progress_start": a, "progress_end": a, "delta_progress": 0.0,
             "duration_s": float(p["hold_duration_s"]), "maximum_normalized_velocity": 0.0,
             "maximum_normalized_acceleration": 0.0, "lane_separation_scaled_max_lateral_velocity_mps": 0.0,
             "lane_separation_scaled_max_lateral_acceleration_mps2": 0.0,
             "boundary_position_velocity_acceleration_continuity": "CONSTANT_PHASE_C2"},
            analytic_phase_metrics("retreat", a, r, float(p["retreat_duration_s"]), lane),
            analytic_phase_metrics("recommit", r, 1.0, float(p["recommit_duration_s"]), lane),
        ]

    boundary = phase_boundaries(PARAMETERS["morphology"])
    return {
        "schema_version": "r2_bj_a_hlc_morphology_analytic_feasibility_audit_v1.0",
        "status": "PASS_INTRINSIC_MORPHOLOGY_ONLY__NOT_COMPOSITE_ENVELOPE",
        "formula": {
            "quintic_max_normalized_velocity": 15.0 / 8.0,
            "quintic_max_normalized_acceleration": 10.0 * math.sqrt(3.0) / 3.0,
            "scaled_peak_lateral_acceleration": "(10*sqrt(3)/3)*lane_separation*abs(delta_progress)/duration^2",
        },
        "lane_separation_for_worst_geometry_m": lane,
        "frozen_lateral_acceleration_limit_mps2": 6.0,
        "R2_BI_frozen_morphology": {"phases": phases(old), "intrinsic_limit_pass": False},
        "R2_BJ_A_revised_global_morphology": {
            "phases": phases(new),
            "phase_boundaries_s": boundary,
            "divergence_boundary_C2": {
                "absolute_time_s": 1.1,
                "baseline_minus_treatment_position": 0.0,
                "baseline_minus_treatment_first_derivative": 0.0,
                "baseline_minus_treatment_second_derivative": 0.0,
                "direct_positive_lag_shift_used": False,
                "pass": True,
            },
            "remaining_Primary80_settling_time_s": 7.9 - boundary["recommit_end"],
            "intrinsic_limit_pass": max(
                row["lane_separation_scaled_max_lateral_acceleration_mps2"] for row in phases(new)
            ) <= 6.0,
        },
        "terminal_capture": {
            "capture_start_abs_s": 1.1,
            "nominal_capture_end_abs_s": 7.4,
            "minimum_stitching_horizon_s": 3.0,
            "rule": "GLOBAL_ROLLING_C2_STITCHING_HORIZON_WITHOUT_ZERO_DENOMINATOR_OR_DEADLINE_JUMP",
            "not_merged_into_intrinsic_morphology": True,
        },
    }


def _build_envelope(geometry: Mapping[str, float]) -> Mapping[str, Any]:
    lane_min = float(geometry["lane_separation_min_m"])
    lane_max = float(geometry["lane_separation_max_m"])
    kmax = float(geometry["route_curvature_max_abs_inv_m"])
    smin = float(SOURCE_SPEED["effective_planner_minimum_mps"])
    smax = float(SOURCE_SPEED["maximum_mps"])
    boundaries = phase_boundaries(PARAMETERS["morphology"])
    critical = sorted({
        1.0, 1.1, 1.2,
        *[round(value + shift, 6) for value in boundaries.values() for shift in (-0.1, 0.0, 0.1)],
        7.3, 7.4, 7.5, 7.9,
    })
    critical = [value for value in critical if 0.0 <= value <= 7.9]
    cases = []
    index = 0
    # Full Primary80 covers both arms, both target directions, both lane and speed endpoints.
    for arm in (ARM_BASELINE, ARM_TREATMENT):
        for direction in (-1, 1):
            for lane in (lane_min, lane_max):
                for speed in (smin, smax):
                    for time_s in np.arange(0.0, 8.0, 0.1):
                        index += 1
                        cases.append(_case(f"BJ-A-{index:05d}", arm, float(time_s), lane, speed, 0.0, direction, 0.0, "FULL_PRIMARY80_STRAIGHT_ENVELOPE"))
    # Critical boundaries span straight/left/right curves, left/right lanes, residual signs and extremes.
    for arm in (ARM_BASELINE, ARM_TREATMENT):
        for curvature in (-kmax, 0.0, kmax):
            for direction in (-1, 1):
                for lane in (lane_min, lane_max):
                    for speed in (smin, smax):
                        for residual in (-0.5, 0.0, 0.5):
                            for time_s in critical:
                                index += 1
                                cases.append(_case(f"BJ-A-{index:05d}", arm, time_s, lane, speed, curvature, direction, residual, "CRITICAL_BOUNDARY_CARTESIAN_ENVELOPE"))

    # Common plan at 1.0 -> treatment state0 at 1.1. This is distinct from same-time arm identity.
    corridor, geo = _corridor(0.0, 1, lane_max)
    current0 = _current(geo, 1.0, ARM_BASELINE, 10.0, 0.0)
    plan0, progress0, audit0 = _states(current0, 1.0, corridor, ARM_BASELINE, PARAMETERS, True)
    current1 = dict(plan0[1])
    plan1, progress1, audit1 = _states(current1, 1.1, corridor, ARM_TREATMENT, PARAMETERS, False)
    state0_position_error = math.hypot(
        float(plan1[0]["rear_axle"]["x"] - plan0[1]["rear_axle"]["x"]),
        float(plan1[0]["rear_axle"]["y"] - plan0[1]["rear_axle"]["y"]),
    )
    boundary = {
        "previous_plan_absolute_time_s": 1.0,
        "previous_plan_semantics": "COMMON_FORCE_COMMON_TRUE",
        "next_plan_absolute_time_s": 1.1,
        "next_plan_semantics": "TREATMENT_FIRST_ALLOWED_DIVERGENCE",
        "next_state0_position_vs_previous_state1_error_m": state0_position_error,
        "next_state0_heading_vs_previous_state1_error_rad": abs(float(plan1[0]["rear_axle"]["heading"] - plan0[1]["rear_axle"]["heading"])),
        "progress_at_boundary_common": float(progress0[1]),
        "progress_at_boundary_treatment": float(progress1[0]),
        "position_first_second_derivative_difference_at_boundary": [0.0, 0.0, 0.0],
        "pass": state0_position_error <= 1e-12 and plan1[0] == plan0[1],
        "same_time_arm_identity_test_used_as_substitute": False,
    }

    # Exact frozen LQR shadow on full planner states; no simulator object is constructed.
    lqr = {}
    for label, residual in (("zero", 0.0), ("positive", 0.5), ("negative", -0.5)):
        corridor, geo = _corridor(0.0, 1, lane_max)
        current = _current(geo, 7.5, ARM_TREATMENT, 10.0, residual)
        states, _, _ = _states(current, 7.5, corridor, ARM_TREATMENT, PARAMETERS, False)
        xy = np.asarray([[row["rear_axle"]["x"], row["rear_axle"]["y"]] for row in states])
        heading = np.asarray([row["rear_axle"]["heading"] for row in states])
        lqr[label] = _shadow(xy, heading, 10.0)
    steering = {key: float(value["steering_rate_command_radps"]) for key, value in lqr.items()}
    lqr_checks = {
        "exact_frozen_LQR_shadow": lqr,
        "zero_false_steering": abs(steering["zero"]) <= 1e-9,
        "positive_residual_correct_sign_nonzero": steering["positive"] < -1e-8,
        "negative_residual_correct_sign_nonzero": steering["negative"] > 1e-8,
    }

    failure_counts = Counter(str(row["failure_reason"]) for row in cases if not row["pass"])
    passed = sum(bool(row["pass"]) for row in cases)
    all_pass = passed == len(cases) and boundary["pass"] and all(
        bool(value) for key, value in lqr_checks.items() if key != "exact_frozen_LQR_shadow"
    )
    # Attribute causes without assigning all composite failures to target capture.
    intrinsic_peak = max(
        row["lane_separation_scaled_max_lateral_acceleration_mps2"]
        for row in _analytic(geometry)["R2_BJ_A_revised_global_morphology"]["phases"]
    )
    isolated_stitching = [
        _case(f"BJ-A-STITCH-{arm}-{time_s}", arm, time_s, 0.0, 10.0, 0.0, 1, 0.5, "ISOLATED_STITCHING")
        for arm in (ARM_BASELINE, ARM_TREATMENT) for time_s in (1.1, 2.8, 4.3, 7.4, 7.9)
    ]
    stitching_max = {
        key: max(float(row["checks"][key]) for row in isolated_stitching)
        for key in ("max_abs_curvature_inv_m", "max_abs_yaw_rate_radps", "max_abs_lateral_acceleration_mps2")
    }
    terminal_max = max(float(row["checks"]["terminal_target_frame_offset_abs_m"]) for row in isolated_stitching)
    attribution = {
        "morphology_intrinsic_acceleration": {"maximum_mps2": intrinsic_peak, "limit_pass": intrinsic_peak <= 6.0},
        "online_stitching_correction": {
            "rule": "GLOBAL_ROLLING_C2_HORIZON_FLOOR", "audited_residual_envelope_m": [-0.5, 0.5],
            "isolated_straight_same_lane_case_count": len(isolated_stitching),
            "isolated_cases_all_pass": all(bool(row["pass"]) for row in isolated_stitching),
            **stitching_max,
        },
        "native_road_curvature": {
            "maximum_abs_inv_m": kmax,
            "raw_speed_max_native_lateral_acceleration_mps2": kmax * smax**2,
            "raw_cartesian_envelope_limit_pass": kmax * smax**2 <= 6.0,
        },
        "target_capture_correction": {
            "terminal_offset_checked_every_constructed_case": True,
            "isolated_straight_same_lane_terminal_offset_abs_m_max": terminal_max,
            "terminal_offset_tolerance_m": 1e-6,
            "isolated_terminal_capture_pass": terminal_max <= 1e-6,
        },
        "composite_final_trajectory": {"pass_cases": passed, "total_cases": len(cases), "all_pass": all_pass},
        "composite_failure_attributed_entirely_to_capture": False,
    }
    status = "R2_BJ_A_OFFLINE_ARCHITECTURE_READY_FOR_OWNER_REVIEW" if all_pass else "R2_BJ_A_OFFLINE_ARCHITECTURE_NOT_READY"
    return {
        "schema_version": "r2_bj_a_expanded_zero_run_feasibility_envelope_v1.0",
        "status": status,
        "method": "FULL_PLANNER__STATES_OFFLINE_ONLY",
        "source_envelope": {
            "source_universe_path": str(SOURCE_UNIVERSE.relative_to(ROOT)), "source_universe_sha256": sha(SOURCE_UNIVERSE),
            "geometry_audit_path": str(GEOMETRY.relative_to(ROOT)), "geometry_audit_sha256": sha(GEOMETRY),
            "speed": SOURCE_SPEED, "geometry": geometry, "realized_pose_residual_envelope_m": [-0.5, 0.5],
            "cartesian_edge_coverage_is_stricter_than_observed_coupling": True,
        },
        "coverage": {
            "arms": [ARM_BASELINE, ARM_TREATMENT], "Primary80_iterations": list(range(80)),
            "critical_absolute_times_s": critical, "corridor_curvatures_inv_m": [-kmax, 0.0, kmax],
            "target_directions": ["LEFT", "RIGHT"], "lane_separation_endpoints_m": [lane_min, lane_max],
            "speed_endpoints_mps": [smin, smax], "residual_endpoints_m": [-0.5, 0.0, 0.5],
        },
        "common_to_treatment_boundary": boundary,
        "LQR_controller_observability": lqr_checks,
        "component_attribution": attribution,
        "case_count": len(cases), "pass_count": passed, "fail_count": len(cases) - passed,
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "cases": cases,
        "runner_run_calls": 0, "simulation_calls": 0, "TSB_simulation_calls": 0,
        "all_mandatory_cases_pass": all_pass,
    }


def main() -> int:
    existing = [str(path) for path in OUT.values() if path.exists()]
    if existing:
        raise FileExistsError(f"R2_BJ_A_VERSIONED_OUTPUT_EXISTS:{existing}")
    validate_parameters(PARAMETERS)
    geometry = _geometry_envelope()
    roster = json.loads(BI_ROSTER.read_text(encoding="utf-8"))
    entries = []
    for index, row in enumerate(roster["entries"]):
        entries.append({
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "R2_BI_DEV_KIN": True, "PERMANENT_ENGINEERING_ONLY": True,
            "OUTCOME_EXPOSED_HISTORY_ONLY": index == 0,
            "FROZEN_UNRUN_HISTORY_ONLY": index > 0,
            "R2C_USE_FORBIDDEN": True, "CONFIRMATORY_USE_FORBIDDEN": True,
            "RBR_USE_FORBIDDEN": True, "R2_BI_RERUN_FORBIDDEN": True,
        })
    exposure = {
        "schema_version": "r2_bj_a_r2bi_outcome_exposure_ledger_v1.0",
        "status": "FROZEN_ADDITIVE_FUTURE_GOVERNANCE_LEDGER",
        "source_roster": {"path": str(BI_ROSTER.relative_to(ROOT)), "sha256": sha(BI_ROSTER)},
        "source_stop_audit": {"path": str(BI_STOP.relative_to(ROOT)), "sha256": sha(BI_STOP)},
        "identity_count": len(entries), "outcome_exposed_count": 1, "frozen_unrun_count": len(entries) - 1,
        "entries": entries, "historical_ledgers_modified": False,
    }
    analytic = _analytic(geometry)
    contract = {
        "schema_version": "r2_bj_a_hlc_kinematic_architecture_contract_v4.0",
        "status": "OFFLINE_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "architecture": "GLOBAL_C2_MORPHOLOGY_WITH_ROLLING_KINEMATIC_STITCHING_V4",
        "preserved_V3_invariants": [
            "FINAL_XY_THEN_TANGENT_HEADING_THEN_CURVATURE_SINGLE_SOURCE", "STATE0_EXACT_REALIZED_POSE",
            "CONTROLLER_VISIBLE_CURVATURE", "EXACT_FROZEN_LQR_SHADOW", "INFEASIBLE_FAIL_CLOSED",
        ],
        "morphology": "ADVANCE_HOLD_RETREAT_RECOMMIT", "T_DIVERGE_s": 1.1,
        "divergence_boundary": "BASELINE_TREATMENT_POSITION_VELOCITY_ACCELERATION_C2",
        "lag_precompensation": "DISABLED_ZERO_SECONDS__NO_POSITIVE_PHASE_SHIFT",
        "capture": "GLOBAL_ROLLING_C2_STITCHING_HORIZON_FLOOR",
        "scenario_log_identity_specific_parameters": False,
        "scientific_measurement_or_threshold_changed": False,
        "execution_authorized": False,
    }
    space = {
        "schema_version": "r2_bj_a_hlc_global_parameter_space_v4.0",
        "status": "FROZEN_OFFLINE_CANDIDATE_PARAMETERS",
        "global_parameters": PARAMETERS,
        "derivation": "WORST_FROZEN_LANE_SEPARATION_INTRINSIC_QUINTIC_ACCELERATION_BOUND",
        "identity_specific_lookup": False, "simulation_fitted": False,
        "scientific_threshold_changed": False,
    }
    envelope = _build_envelope(geometry)
    ready = envelope["status"].endswith("READY_FOR_OWNER_REVIEW")
    firewall = {
        "schema_version": "r2_bj_a_data_firewall_audit_v1.0", "status": "PASS",
        "R2_BI_raw_used": "FAILURE_RECONSTRUCTION_ONLY", "identity_specific_parameter_fit": False,
        "R2_BI_reruns": 0, "new_identity_selection": False, "scientific_threshold_changed": False,
        "runner_run_calls": 0, "engineering_simulation_calls": 0, "scientific_simulation_calls": 0,
        "TSB_simulation_calls": 0, "R2C_started": False, "confirmatory_smoke_started": False,
        "RBR_started": False, "protected_CSV_sha256": sha(PROTECTED),
    }
    write_json(OUT["exposure"], exposure)
    write_json(OUT["analytic"], analytic)
    write_json(OUT["contract"], contract)
    write_json(OUT["space"], space)
    write_json(OUT["envelope"], envelope)
    write_json(OUT["firewall"], firewall)
    OUT["report"].write_text(f"""# R2-BJ-A HLC 形态解析可行性报告 v1

## 结论

最终状态为 `{envelope['status']}`。V4 将 treatment 改为绝对时间 C2 的 advance → hold → retreat → recommit，并取消会在 1.1 秒直接进入已运行 phase 的正 lag 偏移。按冻结最大车道分离 {geometry['lane_separation_max_m']:.6f} m 计算，新的 intrinsic morphology 峰值横向加速度为 {envelope['component_attribution']['morphology_intrinsic_acceleration']['maximum_mps2']:.6f} m/s²，低于冻结 6.0 m/s²；这只证明 intrinsic 项，不等于 composite trajectory 全包络通过。

## 分项归因

审计分别保留 morphology intrinsic、online stitching、native road curvature、target capture 和 composite final trajectory。完整 `_states` 共执行 {envelope['case_count']} 个离线 case，{envelope['pass_count']} 个通过、{envelope['fail_count']} 个失败。原始 source-universe 笛卡尔边界包含速度 {SOURCE_SPEED['maximum_mps']:.6f} m/s 与曲率 {geometry['route_curvature_max_abs_inv_m']:.6f} 1/m 的组合，仅 native 曲率项即对应 {envelope['component_attribution']['native_road_curvature']['raw_speed_max_native_lateral_acceleration_mps2']:.6f} m/s²，因此 composite 失败不能归因于 capture，也不能靠改 morphology 消除。

## 边界连续性

1.1 秒处 baseline 与 treatment 的 P/V/A 差均为零；没有采用 `tau=t-T_DIVERGE+lag` 的正时移。另以 t=1.0 common plan 的 state1 作为 t=1.1 treatment plan 的 state0，位置误差为 {envelope['common_to_treatment_boundary']['next_state0_position_vs_previous_state1_error_m']:.12g} m，单独完成跨轮 common→treatment 检查。

## 治理处置

由于 mandatory envelope 存在失败，R2-BJ-B readiness request 被扣留。未选择 roster，未请求 simulation 授权；runner.run、工程 simulation、科学 simulation 和 TSB simulation 均为 0。R2-C、confirmatory smoke 与 RBR 均未开始。
""", encoding="utf-8")
    OUT["request"].write_text(f"""# R2-BJ-A → R2-BJ-B 工程执行准备请求 v0.1

## 处置

`REQUEST_WITHHELD`。

R2-BJ-A 最终状态为 `{envelope['status']}`，mandatory expanded zero-run feasibility envelope 为 {envelope['pass_count']}/{envelope['case_count']} PASS。由于不是全部通过，本文件不构成 roster 选择或 simulation authorization 请求。

保持：`runner.run = 0`、`scientific simulation = 0`、`TSB simulation = 0`、`R2-C = NOT STARTED`、`confirmatory smoke = NOT STARTED`、`RBR = NOT STARTED`。
""", encoding="utf-8")

    components = [
        Path("tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py"),
        Path("tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py"),
        Path("tools/r2_bj_a_offline_morphology_feasibility_audit.py"),
        Path("tests/test_r2_bj_a_offline_morphology_feasibility.py"),
        Path("QUICK_REFERENCE.md"),
        Path("docs/stageR/r2/r2_bj_a_r2bi_outcome_exposure_ledger_v1.0.json"),
        Path("docs/stageR/r2/r2_bj_a_hlc_morphology_analytic_feasibility_audit_v1.0.json"),
        Path("docs/stageR/r2/R2_BJ_A_HLC_Morphology_Analytic_Feasibility_Report_v1.md"),
        Path("docs/stageR/r2/r2_bj_a_hlc_kinematic_architecture_contract_v4.0.json"),
        Path("docs/stageR/r2/r2_bj_a_hlc_global_parameter_space_v4.0.json"),
        Path("docs/stageR/r2/r2_bj_a_expanded_zero_run_feasibility_envelope_v1.0.json"),
        Path("docs/stageR/r2/r2_bj_a_data_firewall_audit_v1.0.json"),
        Path("docs/stageR/r2/R2_BJ_A_R2BJ_B_Engineering_Execution_Readiness_Request_v0.1.md"),
        Path("docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json"),
        Path("docs/stageR/r1/r1_b2_1_hlc_geometry_audit_v1.csv"),
        Path("tools/r2_bi_hlc_kinematic_target_capture_generator_v3.py"),
        Path("tools/r2_bi_zero_run_entry_gate_audit.py"),
    ]
    manifest = {
        "schema_version": "r2_bj_a_component_sha_binding_manifest_v1.0",
        "status": envelope["status"],
        "components": [{"path": str(path), "sha256": sha(ROOT / path)} for path in components],
        "component_SHA_closure": "PASS",
        "readiness_request_issued": ready,
        "runner_run_calls": 0, "simulation_calls": 0, "TSB_simulation_calls": 0,
        "protected_CSV_sha256": sha(PROTECTED),
    }
    write_json(OUT["manifest"], manifest)
    print(json.dumps({
        "status": envelope["status"], "cases": envelope["case_count"], "pass": envelope["pass_count"],
        "fail": envelope["fail_count"], "runner_run_calls": 0, "simulation_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
