#!/usr/bin/env python3
"""Freeze R2-BI V3 architecture, entry tolerances, parameters, and failure taxonomy."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
OUT = {
    "contract": R2 / "r2_bi_hlc_kinematic_capture_architecture_contract_v3.0.json",
    "space": R2 / "r2_bi_hlc_kinematic_capture_parameter_space_v3.0.json",
    "taxonomy": R2 / "r2_bi_hlc_architecture_failure_taxonomy_v1.0.json",
    "authorization": R2 / "r2_bi_scientific_owner_engineering_authorization_v1.0.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BI_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def contract() -> Mapping[str, Any]:
    return {
        "schema_version": "r2_bi_hlc_kinematic_capture_architecture_contract_v3.0",
        "status": "FROZEN_BEFORE_ANY_R2_BI_SIMULATION_OR_ROSTER_SELECTION",
        "architecture": "CONTROLLER_OBSERVABLE_KINEMATIC_TARGET_CAPTURE_V3",
        "semantic_partitions": ["BEHAVIOR_MORPHOLOGY", "TARGET_CAPTURE"],
        "controller_input": "ONE_CONTINUOUS_KINEMATICALLY_CONSISTENT_TRAJECTORY",
        "geometry": {
            "source": "OFFICIAL_NATIVE_SOURCE_TARGET_REFERENCES_ROUTE_CONTINUOUS_V2_3",
            "composition_frame": "NATIVE_CORRIDOR_FRENET_PROGRESS",
            "no_extrapolation": True, "manual_centerline": False,
            "same_frozen_route_progression": True,
        },
        "state0": "EXACT_REALIZED_CURRENT_EGO_POSE",
        "future_pose_rule": "FINAL_XY_FIRST_THEN_TANGENT_HEADING_AND_CURVATURE_DERIVED_FROM_FINAL_XY",
        "capture": {
            "feedback": "CURRENT_REALIZED_STATE_GLOBAL_RULE_ONLY",
            "clock": "ABSOLUTE_EPISODE_TIME_FIXED_START_AND_DEADLINE",
            "correction": "VECTOR_QUINTIC_C2_WITH_TERMINAL_POSITION_VELOCITY_ACCELERATION_ZERO",
            "deadline_behavior": "FAIL_CLOSED_IF_REALIZED_RESIDUAL_EXCEEDS_FROZEN_TOLERANCE",
            "denominator_zero_special_case": False,
            "controller_observability": "NONZERO_RESIDUAL_MUST_CREATE_KINEMATIC_CURVATURE_AND_LQR_STEERING_DEMAND",
        },
        "pre_divergence": {"t_lt_s": 1.1, "baseline_treatment_full_trajectory_identical": True},
        "scientific_realized_progress_measurement_changed": False,
        "scenario_token_log_or_identity_specific_parameter": False,
        "maximum_engineering_rounds": 2,
        "R2BH_round4": False,
        "R2C_or_confirmatory_or_RBR_authorized": False,
    }


def parameter_space() -> Mapping[str, Any]:
    limits = {
        "curvature_inv_m_max": 0.5,
        "yaw_rate_radps_max": 1.0,
        "lateral_accel_mps2_max": 6.0,
        "state0_to_state1_distance_excess_m_max": 0.25,
        "state0_tangent_mismatch_rad_max": 0.03,
        "heading_xy_consistency_rad_max": 1e-10,
    }
    return {
        "schema_version": "r2_bi_hlc_kinematic_capture_parameter_space_v3.0",
        "status": "FROZEN_BEFORE_ENTRY_GATE_AND_ANY_R2_BI_SIMULATION",
        "maximum_rounds": 2,
        "round0": {
            "morphology": {
                "baseline_transition_duration_s": 2.6,
                "advance_duration_s": 1.1, "advance_progress": 0.44,
                "hold_duration_s": 0.5, "retreat_depth": 0.30,
                "retreat_duration_s": 1.30, "recommit_duration_s": 1.40,
                "lag_precompensation_s": 0.30,
            },
            "capture": {
                "capture_start_abs_s": 1.1, "capture_end_abs_s": 7.2,
                "minimum_remaining_capture_time_s": 0.8,
                "deadline_position_tolerance_m": 0.25,
                "decay_shape": "VECTOR_QUINTIC_C2_TERMINAL_PVA_ZERO",
                "frozen_feasibility_limits": limits,
            },
        },
        "bounds": {
            "morphology": {"retreat_depth": [0.30, 0.38], "retreat_duration_s": [1.30, 1.50]},
            "capture": {"capture_start_abs_s": [1.1, 1.1], "capture_end_abs_s": [6.8, 7.2]},
        },
        "round1_deterministic_aggregate_update": {
            "mechanism_not_all": {"retreat_depth_add": 0.06, "retreat_duration_add_s": 0.15},
            "endpoint_not_all": {"capture_end_abs_s_subtract": 0.30},
            "F_match_or_engineering_not_all": "NO_ADDITIONAL_PARAMETER_CHANGE_BEYOND_CLIPPED_ABOVE_RULES",
            "clip_to_bounds": True, "identity_specific_update": False,
        },
        "entry_gate_tolerances": {
            **limits,
            "zero_residual_straight_shadow_steering_radps_max": 1e-9,
            "nonzero_residual_shadow_steering_radps_min": 1e-7,
            "replanning_overlap_xy_m_max": 0.03,
            "actual_target_offset_terminal_abs_m_max": 1e-9,
        },
        "numerical_initialization_sources": [
            "FROZEN_ENGINEERING_LIMITS", "ZERO_SIMULATION_CONTROLLER_INTERFACE_FORENSIC",
            "QUALITATIVE_R2BH_ARCHITECTURE_DIAGNOSIS_ONLY",
        ],
        "R2BH_raw_used_for_V3_numerical_tuning": False,
    }


def taxonomy() -> Mapping[str, Any]:
    return {
        "schema_version": "r2_bi_hlc_architecture_failure_taxonomy_v1.0",
        "status": "FROZEN_BEFORE_ROUND0_SIMULATION",
        "architecture_stop_before_round1": [
            "TREATMENT_NO_DEPARTURE_PLUS_UNFINISHED_TRANSITION_GTE_4_OF_8",
            "CONTROLLER_SHADOW_ACTUAL_COMMAND_DIRECTION_DISAGREEMENT",
            "SYSTEMATIC_XY_HEADING_CURVATURE_INCONSISTENCY",
            "REALIZED_ABS_TARGET_OFFSET_NOT_DECREASING_IN_MORE_THAN_HALF_IDENTITIES",
            "POST_DEADLINE_STATE0_TO_STATE1_HARD_JUMP_REAPPEARS",
            "SYSTEMATIC_ENGINEERING_TRAJECTORY_FEASIBILITY_FAILURE",
        ],
        "round1_allowed_only": "PRE_REGISTERED_NUMERICAL_GLOBAL_CALIBRATION_FAILURE_WITH_NO_ARCHITECTURE_STOP",
        "numerical_global_failures": [
            "MECHANISM_MARGIN_INSUFFICIENT_WITH_VALID_MEASUREMENT_AND_DEPARTURE",
            "ENDPOINT_MARGIN_INSUFFICIENT_WITH_REALIZED_OFFSET_DECLINE",
            "F_MATCH_MARGIN_INSUFFICIENT_WITH_KINEMATIC_CONSISTENCY",
        ],
        "technical_rerun": "FRESH_RUN_ID_AND_OUTPUT_ROOT_TECHNICAL_INFRASTRUCTURE_FAILURE_ONLY",
        "behavior_or_scientific_failure_rerun": False,
        "identity_replacement": False,
    }


def main() -> int:
    if _sha(PROTECTED) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    for key, value in (("contract", contract()), ("space", parameter_space()), ("taxonomy", taxonomy())):
        _write_new(OUT[key], value)
    _write_new(OUT["authorization"], {
        "schema_version": "r2_bi_scientific_owner_engineering_authorization_v1.0",
        "R2_BI_ENGINEERING_ONLY_HLC_SIMULATION_AUTHORIZED_CONDITIONALLY": True,
        "entry_condition": "ALL_MANDATORY_ZERO_RUN_GATES_PASS_BEFORE_ROSTER_SELECTION",
        "scope": "FRESH_DEV_KIN_ONLY_MAX_2_ROUNDS_16_RUNS_PER_ROUND",
        "TSB_simulation_authorized": False, "old_identity_rerun_authorized": False,
        "R2C_confirmatory_RBR_authorized": False,
    })
    print(json.dumps({"status": "R2_BI_PRE_SIMULATION_CONTRACTS_FROZEN", "maximum_rounds": 2}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
