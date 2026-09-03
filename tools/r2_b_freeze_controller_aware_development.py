#!/usr/bin/env python3
"""Freeze fresh R2-B DEV-CAL identities and all pre-simulation contracts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402
from tools.r1_b2_9_d_freeze_pair_bindings import _one as freeze_pair_binding  # noqa: E402
from tools.r1_future_compliant_smoke_selector_v1_3 import canonical_sha  # noqa: E402
from tools.r2_a_freeze_controller_id_design import _select_unique_family_suffix  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
R2A_ROSTER = R2 / "r2_a_controller_id_dev_canary_roster_v1.0.json"
R2A_EXCLUSION = R2 / "r2_a_controller_id_permanent_exclusion_ledger_v1.0.json"
R2A_SURROGATE = R2 / "r2_a_controller_transfer_surrogate_v1.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"

OUT = {
    "exposure": R2 / "r2_b_r2a_identification_outcome_exposure_ledger_v1.0.json",
    "roster": R2 / "r2_b_generator_calibration_roster_v1.0.json",
    "exclusion": R2 / "r2_b_generator_calibration_permanent_exclusion_ledger_v1.0.json",
    "pair_bindings": R2 / "r2_b_generator_calibration_pair_bindings_v1.0.json",
    "contract": R2 / "r2_b_controller_aware_generator_contract_v1.0.json",
    "hlc_space": R2 / "r2_b_hlc_calibration_parameter_space_v1.0.json",
    "tsb_space": R2 / "r2_b_tsb_calibration_parameter_space_v1.0.json",
    "objective": R2 / "r2_b_generator_calibration_objective_v1.0.json",
    "ledger": R2 / "r2_b_generator_calibration_run_ledger_v1.0.json",
    "authorization": R2 / "r2_b_scientific_owner_dev_calibration_authorization_v1.0.json",
}


def read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_B_VERSIONED_FREEZE_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _exposure(roster: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "r2_b_r2a_identification_outcome_exposure_ledger_v1.0",
        "status": "R2_A_IDENTIFICATION_SET_FROZEN_OUTCOME_EXPOSED",
        "source_roster": {"path": str(R2A_ROSTER.relative_to(ROOT)), "sha256": sha(R2A_ROSTER)},
        "identity_count": 16,
        "identities": [
            {
                "family": row["family"],
                "scenario_token": row["scenario_token"],
                "log_id": row["log_id"],
                "IDENTIFICATION_OUTCOME_EXPOSED": True,
                "R2A_IDENTIFICATION_ONLY": True,
                "R2B_CALIBRATION_USE_FORBIDDEN": True,
                "R2C_VALIDATION_USE_FORBIDDEN": True,
                "R2D_CONFIRMATORY_USE_FORBIDDEN": True,
                "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
                "allowed_future_use": "READ_ONLY_R2A_IDENTIFICATION_HISTORY_ONLY",
            }
            for row in roster["entries"]
        ],
        "historical_ledger_modified": False,
    }


def _contract() -> Dict[str, Any]:
    return {
        "schema_version": "r2_b_controller_aware_generator_contract_v1.0",
        "status": "FROZEN_BEFORE_R2_B_DEV_CALIBRATION",
        "architecture": "CONTROLLER_AWARE_PRECOMPENSATION_PLUS_DEV_ONLY_OFFLINE_FEEDBACK_CALIBRATION",
        "generator_name": "G_R2_V1",
        "deterministic": True,
        "global_parameterization": True,
        "per_identity_or_token_lookup_forbidden": True,
        "allowed_inputs": [
            "family", "arm", "absolute_episode_time", "current_ego_state",
            "native_route_or_reference_geometry", "frozen_pretreatment_physical_context",
        ],
        "forbidden_inputs": [
            "future_realized_mechanism", "future_safety", "BDD", "embedding",
            "scenario_token_specific_coefficient", "log_id_specific_coefficient",
        ],
        "PRE_CONTEXT": {"iterations": list(range(10)), "identical_between_arms": True},
        "t_diverge_s": 1.1,
        "HLC": {
            "desired_realized_morphology_separate_from_planner_morphology": True,
            "baseline": "DECISIVE_MONOTONIC_LANE_CHANGE_WITH_ZERO_RETREAT",
            "treatment": "ADVANCE_HOLD_RETREAT_RECOMMIT_SETTLE",
            "precompensation_terms": ["transfer_gain", "tracking_lag", "retreat_duration", "recommit_duration", "settling_allowance"],
        },
        "TSB": {
            "baseline": "ONE_REALIZED_BRAKE_PHASE",
            "treatment": "TWO_REALIZED_BRAKE_PHASES_WITH_RELEASE",
            "precompensation_terms": [
                "first_brake_magnitude_and_effective_duration", "release_magnitude_and_effective_duration",
                "second_brake_magnitude_and_duration", "one_second_LQR_lookahead_phase_composition",
                "absolute_time_boundary_migration", "release_carryover",
            ],
            "constant_gain_inverse_only": False,
        },
        "R2_A_surrogate_role": "INITIALIZATION_ONLY_NO_R2_A_RERUN",
        "scientific_thresholds_modified": False,
        "R1_generator_modified": False,
    }


def _hlc_space() -> Dict[str, Any]:
    return {
        "schema_version": "r2_b_hlc_calibration_parameter_space_v1.0",
        "status": "BOUNDED_AND_FROZEN_BEFORE_FIRST_SIMULATION",
        "maximum_rounds": 4,
        "initialization": "R2_A_ENGINEERING_SURROGATE_AGGREGATE_ONLY",
        "desired_realized": {
            "retreat_depth_target": 0.24,
            "expected_transfer_gain_initial": 0.868925,
            "expected_tracking_lag_s_initial": 0.30,
            "minimum_settling_allowance_s": 1.8,
        },
        "bounds": {
            "baseline_transition_duration_s": [2.2, 3.2],
            "advance_duration_s": [0.9, 1.4],
            "advance_progress": [0.38, 0.55],
            "hold_duration_s": [0.3, 0.8],
            "retreat_depth": [0.24, 0.50],
            "retreat_duration_s": [1.0, 1.8],
            "recommit_duration_s": [1.0, 1.8],
            "lag_precompensation_s": [0.0, 0.4],
        },
        "round0": {
            "baseline_transition_duration_s": 2.6,
            "advance_duration_s": 1.1,
            "advance_progress": 0.44,
            "hold_duration_s": 0.5,
            "retreat_depth": 0.30,
            "retreat_duration_s": 1.3,
            "recommit_duration_s": 1.4,
            "lag_precompensation_s": 0.30,
        },
        "deterministic_update": {
            "mechanism_fail": {"retreat_depth_add": 0.06, "retreat_duration_add_s": 0.15},
            "endpoint_or_engineering_fail": {"recommit_duration_subtract_s": 0.15},
            "settling_fail": {"recommit_duration_subtract_s": 0.10},
            "all_updates_clip_to_bounds": True,
            "aggregate_family_counts_only": True,
        },
        "manual_update_allowed": False,
    }


def _tsb_space() -> Dict[str, Any]:
    return {
        "schema_version": "r2_b_tsb_calibration_parameter_space_v1.0",
        "status": "BOUNDED_AND_FROZEN_BEFORE_FIRST_SIMULATION",
        "maximum_rounds": 4,
        "initialization": "R2_A_GENERATOR_TO_LQR_AND_LQR_TO_REALIZED_AGGREGATE_SURROGATE_ONLY",
        "bounds": {
            "baseline_brake_mps2": [-2.2, -1.1],
            "baseline_duration_s": [1.2, 2.4],
            "first_brake_mps2": [-3.5, -1.8],
            "first_brake_duration_s": [0.7, 1.3],
            "release_mps2": [0.8, 2.2],
            "release_duration_s": [1.0, 1.8],
            "second_brake_mps2": [-3.5, -1.8],
            "second_brake_duration_s": [0.7, 1.3],
        },
        "round0": {
            "start_s": 1.1,
            "baseline_brake_mps2": -1.45,
            "baseline_duration_s": 1.8,
            "first_brake_mps2": -2.40,
            "first_brake_duration_s": 0.9,
            "release_mps2": 1.40,
            "release_duration_s": 1.3,
            "second_brake_mps2": -2.40,
            "second_brake_duration_s": 0.9,
        },
        "deterministic_update": {
            "baseline_one_phase_fail": {"baseline_brake_mps2_subtract": 0.20, "baseline_duration_add_s": 0.10},
            "treatment_phase_or_peak_fail": {"first_brake_mps2_subtract": 0.30, "second_brake_mps2_subtract": 0.30, "brake_duration_add_s": 0.10},
            "release_or_phase_separation_fail": {"release_mps2_add": 0.20, "release_duration_add_s": 0.15},
            "F_match_fail": {"baseline_duration_adjust_toward_treatment_abs_impulse_s": 0.15},
            "all_updates_clip_to_bounds": True,
            "aggregate_family_counts_only": True,
        },
        "manual_update_allowed": False,
    }


def _objective() -> Dict[str, Any]:
    return {
        "schema_version": "r2_b_generator_calibration_objective_v1.0",
        "status": "FROZEN_BEFORE_FIRST_R2_B_SIMULATION",
        "comparison": "LEXICOGRAPHIC_GLOBAL_CANDIDATE_SELECTION",
        "priority": [
            "MECHANISM_REALIZED_PASS_COUNT",
            "F_MATCH_PASS_COUNT",
            "HLC_ENDPOINT_AND_ENGINEERING_PASS_COUNT",
            "OFFICIAL_SAFETY_PASS_COUNT",
            "MINIMUM_MECHANISM_MARGIN",
            "MINIMUM_ENDPOINT_MARGIN",
            "F_MATCH_MARGIN",
        ],
        "success": {
            "HLC": {"mechanism": "ALL", "F_match": "ALL", "engineering": "ALL", "endpoint": "ALL", "safety": "NO_SYSTEMATIC_FAILURE"},
            "TSB": {"measurement_applicability": "ALL", "mechanism": "ALL", "F_match": "ALL", "safety": "NO_SYSTEMATIC_FAILURE"},
        },
        "systematic_safety_failure_definition": "MORE_THAN_HALF_OF_FIXED_DEV_CAL_IDENTITIES_FAIL_PAIR_SAFETY",
        "scientific_threshold_as_optimization_variable": False,
        "identity_replacement_allowed": False,
        "maximum_rounds_per_family": 4,
    }


def main() -> int:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("R2_B_VERSIONED_FREEZE_OUTPUT_EXISTS")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("PROTECTED_CSV_SHA_MISMATCH")
    if not R2A_SURROGATE.is_file():
        raise FileNotFoundError(R2A_SURROGATE)
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    r2a_roster, pre_firewall = read(R2A_ROSTER), read(R2A_EXCLUSION)
    if len(pre_firewall["entries"]) != 85:
        raise RuntimeError("R2_B_EXPECTED_85_IDENTITY_PRE_SELECTION_FIREWALL")
    exposure = _exposure(r2a_roster)
    cutoffs = {
        family: max(row["selector_rank_sha256"] for row in r2a_roster["entries"] if row["family"] == family)
        for family in ("R-HLC", "R-TSB")
    }
    cache: Dict[str, Any] = {}
    selected, audits, source_audits = [], [], {}
    used_tokens, used_logs = set(), set()
    for family in ("R-HLC", "R-TSB"):
        rows, row_audits, source_audit = _select_unique_family_suffix(
            family, cutoffs[family], pre_firewall, cache, used_tokens, used_logs
        )
        for row in rows:
            row.update({
                "PERMANENT_ENGINEERING_ONLY": True,
                "R2C_VALIDATION_USE_FORBIDDEN": True,
                "R2D_CONFIRMATORY_USE_FORBIDDEN": True,
                "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
                "selection_role": "OUTCOME_BLIND_R2_B_GENERATOR_CALIBRATION",
            })
        selected.extend(rows)
        audits.extend(row_audits)
        source_audits[family] = source_audit
        used_tokens.update(row["scenario_token"] for row in rows)
        used_logs.update(row["log_id"] for row in rows)
    if len(selected) != 16 or len(used_tokens) != 16 or len(used_logs) != 16:
        raise RuntimeError("R2_B_DEV_CAL_CARDINALITY_OR_UNIQUENESS_FAIL")
    old_tokens = {row["scenario_token"] for row in pre_firewall["entries"]}
    old_logs = {row["log_id"] for row in pre_firewall["entries"]}
    if any(row["scenario_token"] in old_tokens or row["log_id"] in old_logs for row in selected):
        raise RuntimeError("R2_B_DEV_CAL_OVERLAPS_FIREWALL")
    roster = {
        "schema_version": "r2_b_generator_calibration_roster_v1.0",
        "status": "FROZEN_BEFORE_ANY_R2_B_GENERATOR_OUTCOME",
        "selection_semantics": "CONTINUE_FROZEN_V1_3_HASH_RANK_AFTER_R2_A_SELECTED_PREFIX",
        "source_universe": r2a_roster["source_universe"],
        "pre_selection_firewall": {"path": str(R2A_EXCLUSION.relative_to(ROOT)), "sha256": sha(R2A_EXCLUSION), "identity_count": 85},
        "entries": selected,
        "counts": {"R-HLC": 8, "R-TSB": 8, "total": 16},
        "candidate_audits": audits,
        "source_audits": source_audits,
        "selection_inputs": ["context", "map", "route_reference", "Primary80", "technical_runtime_applicability"],
        "forbidden_selection_inputs_used": [],
        "mechanism_F_match_safety_BDD_representation_outcome_used": False,
    }
    final_entries = list(pre_firewall["entries"]) + [
        {
            "scenario_token": row["scenario_token"], "log_id": row["log_id"], "family": row["family"],
            "sources": [str(OUT["roster"].relative_to(ROOT))],
            "reasons": ["R2_B_GENERATOR_CALIBRATION_DEV_IDENTITY"],
            "PERMANENT_ENGINEERING_ONLY": True,
            "R2C_VALIDATION_USE_FORBIDDEN": True,
            "R2D_CONFIRMATORY_USE_FORBIDDEN": True,
            "RBR_SCIENTIFIC_USE_FORBIDDEN": True,
        }
        for row in selected
    ]
    exclusion = {
        "schema_version": "r2_b_generator_calibration_permanent_exclusion_ledger_v1.0",
        "status": "FROZEN_ADDITIVE_R2_B_DATA_FIREWALL",
        "source_ledger": {"path": str(R2A_EXCLUSION.relative_to(ROOT)), "sha256": sha(R2A_EXCLUSION)},
        "entries": final_entries,
        "counts": {"pre_R2_B": 85, "R2_B_DEV_CAL": 16, "effective_unique_identities": 101},
        "entry_removal_or_reduction_allowed": False,
    }
    pair_cache: Dict[str, Any] = {}
    pair_rows = []
    for index, entry in enumerate(selected, 1):
        binding = freeze_pair_binding(entry, pair_cache)
        binding["pair_id"] = f"R2B-CAL-{entry['family'][2:]}-{index if entry['family']=='R-HLC' else index-8:02d}"
        binding["future_realized_trace_used"] = False
        binding["future_safety_result_used"] = False
        pair_rows.append(binding)
    pair_bindings = {
        "schema_version": "r2_b_generator_calibration_pair_bindings_v1.0",
        "status": "FROZEN_16_OF_16_PRE_OUTCOME_BINDINGS_COMPLETE",
        "pairs": pair_rows,
        "counts": {"R-HLC": 8, "R-TSB": 8, "total": 16},
        "scientific_numerics_changed": False,
    }
    ledger = {
        "schema_version": "r2_b_generator_calibration_run_ledger_v1.0",
        "status": "FROZEN_PRE_EXECUTION",
        "roster_canonical_sha256": canonical_sha(roster),
        "contract_canonical_sha256": canonical_sha(_contract()),
        "hlc_parameter_space_canonical_sha256": canonical_sha(_hlc_space()),
        "tsb_parameter_space_canonical_sha256": canonical_sha(_tsb_space()),
        "objective_canonical_sha256": canonical_sha(_objective()),
        "rounds": [],
        "maximum_rounds_per_family": 4,
        "technical_rerun_policy": "FRESH_RUN_ID_AND_ROOT_TECHNICAL_FAILURE_ONLY",
        "identity_replacement": False,
        "R2C_identities_selected": False,
        "scientific_simulation": False,
        "RBR_started": False,
    }
    authorization = {
        "schema_version": "r2_b_scientific_owner_dev_calibration_authorization_v1.0",
        "R2_B_DEV_CALIBRATION_SIMULATION_AUTHORIZED": True,
        "scope": "FRESH_FROZEN_R2_B_DEV_CAL_IDENTITIES_ONLY_MAX_4_ROUNDS_PER_FAMILY",
        "R1_and_R2A_rerun_authorized": False,
        "R2C_or_confirmatory_authorized": False,
        "RBR_authorized": False,
    }
    for key, value in (
        ("exposure", exposure), ("roster", roster), ("exclusion", exclusion),
        ("pair_bindings", pair_bindings), ("contract", _contract()),
        ("hlc_space", _hlc_space()), ("tsb_space", _tsb_space()),
        ("objective", _objective()), ("ledger", ledger), ("authorization", authorization),
    ):
        write_new(OUT[key], value)
    print(json.dumps({"status": roster["status"], "HLC_DEV_CAL": 8, "TSB_DEV_CAL": 8, "final_exclusion_count": 101}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
