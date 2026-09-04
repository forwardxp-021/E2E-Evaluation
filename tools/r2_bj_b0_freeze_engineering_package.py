#!/usr/bin/env python3
"""Freeze the outcome-blind BJ-B0 HLC engineering roster and pair package."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
SALT = "R2_BJ_B0_DEV_ROSTER_V1_20260904"
PROVENANCE = R2 / "r2_bj_a5_applicable_pool_provenance_manifest_v1.0.json"
CENSUS = R2 / "r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json"
COMPONENT = R2 / "r2_bj_a5_native_generated_composite_component_audit_v1.0.json"
FRAME = R2 / "r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0.json"
SPACE = R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json"
V4_CONTRACT = R2 / "r2_bj_a_hlc_kinematic_architecture_contract_v4.0.json"
PREREG = R2 / "r2_bj_b0_preregistered_roster_selection_contract_v1.0.json"

EXPECTED = {
    PROVENANCE: "55fb83d9c41a1f0e4d43caaebee8ed2a47cf697cc3177aff17b2554c0802f97f",
    CENSUS: "bae7ef8333acd2a8a45acfbe3ddcc2281b242b8a569104edc98b3201eb450adb",
    COMPONENT: "7f18266d20242f0fa16f6dfbe31edd0edfa77ff8febc3cffae9042ffbaa2c19f",
    R2 / "r2_bj_a5_finite_frame_census_envelope_v1.0.json": "44ccc1ca910250e25ad6d0a7697ccf36c1a90140bdd18578fb0be1d2d5f81d9d",
    R2 / "r2_bj_a5_component_sha_binding_manifest_v1.0.json": "54e005c67ce420d11a7b0b8c8134d6c2e1f776b3bfd5c8fa3942f45f8be30f96",
    SPACE: "95b6b726a42f9501f6f5401e8b2e5e179cadb489b74087a09667889efd31a158",
    V4_CONTRACT: "526bbd4335c13dc5177d5f3da80dc8ef0e93b4d8e7ec7346af4096e0955940af",
    ROOT / "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py": "907e118014e1f83ed0004d5a194d75fa389a2e7fc21619c3a3a44dc3c69abae9",
    ROOT / "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py": "066a1fd2dd2eb3fdc25ed4308c115d3a186e8a16e1e8944cbcf1d08d46613b8b",
}

OUT = {
    "roster": R2 / "r2_bj_b0_hlc_v4_engineering_roster_v1.0.json",
    "exclusion": R2 / "r2_bj_b0_permanent_engineering_exclusion_ledger_v1.0.json",
    "unselected": R2 / "r2_bj_b0_unselected_pool_disposition_v1.0.json",
    "schedule": R2 / "r2_bj_b0_hlc_v4_pair_schedule_v1.0.json",
    "bindings": R2 / "r2_bj_b0_exact_pair_binding_manifest_v1.0.json",
    "architecture": R2 / "r2_bj_b0_hlc_v4_execution_architecture_contract_v1.0.json",
    "taxonomy": R2 / "r2_bj_b0_online_failure_taxonomy_v1.0.json",
    "authorization": R2 / "r2_bj_b0_execution_authorization_gate_v1.0.json",
}

HISTORY_FILES = [
    R2 / "r2_a_controller_id_permanent_exclusion_ledger_v1.0.json",
    R2 / "r2_b_generator_calibration_permanent_exclusion_ledger_v1.0.json",
    R2 / "r2_b_r2a_identification_outcome_exposure_ledger_v1.0.json",
    R2 / "r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0.json",
    R2 / "r2_bh_r2b_hlc_outcome_exposure_ledger_v1.0.json",
    R2 / "r2_bi_hlc_dev_kin_permanent_exclusion_ledger_v1.0.json",
    R2 / "r2_bi_r2bh_outcome_exposure_ledger_v1.0.json",
    R2 / "r2_bj_a_r2bi_outcome_exposure_ledger_v1.0.json",
    ROOT / "docs/stageR/r1/r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json",
    ROOT / "docs/stageR/r1/r1_b3_r1_official_outcome_exposure_ledger_v1.0.json",
]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def selection_hash(token: str, log_id: str) -> str:
    return hashlib.sha256(SALT.encode() + b"\0" + token.encode() + b"\0" + log_id.encode()).hexdigest()


def speed_band(value: float) -> str:
    if value < 6.0:
        return "[3.0,6.0)"
    if value < 9.0:
        return "[6.0,9.0)"
    if value < 12.0:
        return "[9.0,12.0)"
    return "[12.0,+inf)"


def _identity_pairs(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        if value.get("scenario_token") and value.get("log_id"):
            yield str(value["scenario_token"]), str(value["log_id"])
        for child in value.values():
            yield from _identity_pairs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identity_pairs(child)


def exact_quota(rows: Iterable[Mapping[str, Any]]) -> bool:
    rows = list(rows)
    def count(key: str, value: str) -> int:
        return sum(row[key] == value for row in rows)
    return (
        count("direction", "left") == 2 and count("direction", "right") == 6
        and count("map_name", "us-nv-las-vegas-strip") == 1
        and count("map_name", "us-pa-pittsburgh-hazelwood") == 7
        and all(count("speed_band", band) == amount for band, amount in {
            "[3.0,6.0)": 1, "[6.0,9.0)": 3, "[9.0,12.0)": 3, "[12.0,+inf)": 1,
        }.items())
    )


def select(pool: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    # combinations() emits ascending-rank tuples in lexicographic order.
    for indexes in itertools.combinations(range(len(pool)), 8):
        candidate = [pool[index] for index in indexes]
        if exact_quota(candidate):
            return candidate
    raise RuntimeError("R2_BJ_B0_EXACT_QUOTA_SUBSET_NOT_FOUND")


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2_BJ_B0_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    failures = [str(path.relative_to(ROOT)) for path, expected in EXPECTED.items() if sha(path) != expected]
    if failures:
        raise RuntimeError(f"R2_BJ_B0_FROZEN_INPUT_SHA_MISMATCH:{failures}")
    provenance, census, frame = read(PROVENANCE), read(CENSUS), read(FRAME)
    if provenance["applicable_pool_count"] != 34 or len(provenance["records"]) != 34:
        raise RuntimeError("R2_BJ_B0_A5_POOL_NOT_EXACT_34")
    by_frame = {row["frame_index"]: row for row in frame["entries"]}
    by_census = {row["census_index"]: row for row in census["entries"]}
    pool = []
    for source in provenance["records"]:
        frozen, result = by_frame[source["frame_index"]], by_census[source["census_index"]]
        if result["final_disposition"] != "MOVING_REGIME_V4_APPLICABLE":
            raise RuntimeError("R2_BJ_B0_NON_APPLICABLE_RECORD_IN_POOL")
        row = {
            **source,
            "map_name": frozen["map_name"], "direction": frozen["direction"],
            "db_path": frozen["db_path"], "db_file": frozen["db_file"],
            "initial_state": frozen["initial_state"], "route_roadblock_ids": frozen["route_roadblock_ids"],
            "route_fingerprint": frozen["route_fingerprint"],
            "source_lane_id": frozen["source_lane_id"], "target_lane_id": frozen["target_lane_id"],
            "scenario_anchor_timestamp_us": frozen["scenario_anchor_timestamp_us"],
            "pre_treatment_speed_information": result["predicate_result"]["closure"]["speed_information"],
            "curvature_disposition": result["predicate_result"]["closure"]["curvature_disposition"],
            "selection_hash_sha256": selection_hash(source["scenario_token"], source["log_id"]),
            "speed_band": speed_band(float(source["v_audit_mps"])),
        }
        pool.append(row)
    pool.sort(key=lambda row: (row["selection_hash_sha256"], row["scenario_token"], row["log_id"]))
    for rank, row in enumerate(pool, 1):
        row["selection_rank"] = rank
    selected = select(pool)
    chosen = {(row["scenario_token"], row["log_id"]) for row in selected}
    unselected = [row for row in pool if (row["scenario_token"], row["log_id"]) not in chosen]

    history = set()
    for path in HISTORY_FILES:
        history.update(_identity_pairs(read(path)))
    overlap = sorted(chosen & history)
    if overlap:
        raise RuntimeError(f"R2_BJ_B0_HISTORY_EXCLUSION_OVERLAP:{overlap}")

    compact = [{key: row[key] for key in (
        "selection_rank", "selection_hash_sha256", "scenario_token", "log_id", "map_name", "direction",
        "v_audit_mps", "speed_band", "frame_index", "census_index", "audit_rank_sha256", "closure_canonical_sha256",
    )} for row in selected]
    roster_entries = []
    for index, row in enumerate(selected, 1):
        roster_entries.append({
            **row, "roster_index": index, "pair_id": f"R2BJB0-HLC-{index:02d}",
            "disposition": "PERMANENT_ENGINEERING_ONLY",
            "outcome_state": "ROSTER_FROZEN_NOT_YET_OUTCOME_EXPOSED",
            "R2_C_CONFIRMATORY_RBR_USE_FORBIDDEN": True,
        })
    roster = {
        "schema_version": "r2_bj_b0_hlc_v4_engineering_roster_v1.0", "status": "FROZEN_ZERO_RUN",
        "selection_contract": str(PREREG.relative_to(ROOT)), "selection_salt": SALT,
        "candidate_pool_count": 34, "entry_count": 8, "exact_quota_pass": exact_quota(selected),
        "lexicographically_smallest_selection_rank_tuple": [row["selection_rank"] for row in selected],
        "history_and_permanent_exclusion_overlap_count": 0, "entries": roster_entries,
        "selected_compact_canonical_sha256": canonical_sha(compact),
    }
    exclusion = {
        "schema_version": "r2_bj_b0_permanent_engineering_exclusion_ledger_v1.0", "status": "FROZEN",
        "entry_count": 8, "entries": [{
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "PERMANENT_ENGINEERING_ONLY": True, "ROSTER_FROZEN_NOT_YET_OUTCOME_EXPOSED": True,
            "R2_C_USE_FORBIDDEN": True, "CONFIRMATORY_SMOKE_USE_FORBIDDEN": True, "RBR_USE_FORBIDDEN": True,
        } for row in roster_entries],
    }
    unselected_doc = {
        "schema_version": "r2_bj_b0_unselected_pool_disposition_v1.0", "status": "FROZEN",
        "entry_count": 26, "reserve_or_replacement_order": "NONE",
        "entries": [{
            "selection_rank": row["selection_rank"], "selection_hash_sha256": row["selection_hash_sha256"],
            "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "disposition": "UNSELECTED_OUTCOME_UNEXPOSED_POOL",
        } for row in unselected],
    }
    runs, bindings = [], []
    for entry in roster_entries:
        pair_id = entry["pair_id"]
        pair_runs = []
        for arm in ("BASELINE", "TREATMENT"):
            run_id = f"R2BJB0-HLC-{entry['roster_index']:02d}-{arm}"
            run = {
                "run_order": len(runs) + 1, "run_id": run_id, "pair_id": pair_id, "family": "R-HLC",
                "arm": arm, "scenario_token": entry["scenario_token"], "log_id": entry["log_id"],
                "seed": 2026090401, "planner": "R2BJB0HLCV4EngineeringPlanner",
                "time_controller": "R1Primary80ScientificTimeControllerV1", "intended_only": True,
            }
            runs.append(run); pair_runs.append(run_id)
        shared = {
            "scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "db_path": entry["db_path"],
            "map_name": entry["map_name"], "direction": entry["direction"], "initial_state": entry["initial_state"],
            "route_roadblock_ids": entry["route_roadblock_ids"], "route_fingerprint": entry["route_fingerprint"],
            "source_lane_id": entry["source_lane_id"], "target_lane_id": entry["target_lane_id"],
            "source_reference_sha256": entry["source_reference_sha256"], "target_reference_sha256": entry["target_reference_sha256"],
            "reference_geometry_locator": {"census_index": entry["census_index"], "predicate_path": str(CENSUS.relative_to(ROOT))},
            "pretreatment_physical_context": entry["pre_treatment_speed_information"],
            "controller_and_simulation_config": "FROZEN_NUPLAN_1_2_2_TWOSTAGE_LQR_CLOSED_LOOP_NONREACTIVE",
            "seed": 2026090401, "Primary80": {"controller_iterations": 81, "recorded_iterations": "0...79"},
            "pre_divergence": {"boundary": "t<1.1s", "complete_trajectory_equality": "REQUIRED_EXACT"},
        }
        bindings.append({
            "pair_id": pair_id, "selection_rank": entry["selection_rank"], "family": "R-HLC",
            "baseline_run_id": pair_runs[0], "treatment_run_id": pair_runs[1], "shared_binding": shared,
            "only_allowed_arm_difference": "FROZEN_V4_HLC_MORPHOLOGY_CAPTURE_TREATMENT",
            "shared_binding_canonical_sha256": canonical_sha(shared), "pre_outcome_complete": True,
        })
    schedule = {
        "schema_version": "r2_bj_b0_hlc_v4_pair_schedule_v1.0", "status": "FROZEN_INTENDED_ONLY",
        "pair_count": 8, "run_count": 16, "order": "SELECTION_RANK_THEN_BASELINE_BEFORE_TREATMENT",
        "run_id_namespace": "R2BJB0-HLC", "runs": runs, "RUNNER_RUN": 0,
    }
    binding_doc = {
        "schema_version": "r2_bj_b0_exact_pair_binding_manifest_v1.0", "status": "FROZEN_PRE_OUTCOME",
        "pair_count": 8, "bindings": bindings, "pair_bindings_canonical_sha256": canonical_sha(bindings),
    }
    architecture = {
        "schema_version": "r2_bj_b0_hlc_v4_execution_architecture_contract_v1.0", "status": "FROZEN_ZERO_RUN",
        "planner_class": "tools.r2_bj_b0_hlc_v4_engineering_planner.R2BJB0HLCV4EngineeringPlanner",
        "immutable_V4_generator": {"path": "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py", "sha256": EXPECTED[ROOT / "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py"]},
        "immutable_V4_states": {"path": "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py", "sha256": EXPECTED[ROOT / "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py"]},
        "immutable_parameters": {"path": str(SPACE.relative_to(ROOT)), "sha256": EXPECTED[SPACE]},
        "online_per_planner_call_gates": [
            "curvature", "yaw_rate", "lateral_acceleration", "state0_exact_current_pose",
            "state0_to_state1_distance_excess", "state0_tangent_mismatch", "XY_heading_consistency",
            "target_frame_residual", "rolling_stitching_horizon", "controller_visible_steering",
            "baseline_treatment_pre_divergence_equality",
        ],
        "threshold_source": "IMMUTABLE_V4_AND_EXISTING_FROZEN_SCIENTIFIC_ENGINEERING_CONTRACTS",
        "architecture_violation_action": ["STOP_CURRENT_RUN", "STOP_REMAINING_SCHEDULE"],
        "IDENTITY_REPLACEMENT": "FORBIDDEN", "PARAMETER_UPDATE": "FORBIDDEN", "technical_rerun_authorized": False,
    }
    taxonomy = {
        "schema_version": "r2_bj_b0_online_failure_taxonomy_v1.0", "status": "FROZEN",
        "ARCHITECTURE_FAILURE": {
            "codes": ["CURVATURE_LIMIT", "YAW_RATE_LIMIT", "LATERAL_ACCELERATION_LIMIT", "STATE0_POSE_MISMATCH", "STATE0_STEP_EXCESS", "STATE0_TANGENT_MISMATCH", "XY_HEADING_MISMATCH", "TARGET_FRAME_RESIDUAL", "STITCHING_HORIZON", "CONTROLLER_VISIBLE_STEERING_INVALID", "PREDIVERGENCE_TRAJECTORY_MISMATCH"],
            "action": ["STOP_CURRENT_RUN", "STOP_REMAINING_SCHEDULE"],
        },
        "INFRASTRUCTURE_FAILURE": {"codes": ["HYDRA_COMPOSITION", "SCENARIO_RESOLUTION", "RUNNER_CONSTRUCTION", "OUTPUT_PATH_COLLISION"], "technical_rerun_authorized": False},
        "identity_replacement": "FORBIDDEN", "parameter_update": "FORBIDDEN",
    }
    authorization = {
        "schema_version": "r2_bj_b0_execution_authorization_gate_v1.0", "status": "DENY_BEFORE_SIMULATOR_START",
        "BJ_B_ENGINEERING_SIMULATION_AUTHORIZED": False, "NEW_RUN_BUDGET": 0, "RUNNER_RUN": 0,
        "required_future_owner_binding": ["authorized_component_manifest_sha256", "authorized_schedule_sha256", "authorized_pair_binding_sha256", "positive_run_budget"],
        "future_canary": {"CANARY_IDENTITIES": 1, "CANARY_PAIRS": 1, "INTENDED_RUNS": 2, "ORDER": "BASELINE_THEN_TREATMENT", "identity_selection_rank": selected[0]["selection_rank"], "scenario_token": selected[0]["scenario_token"], "log_id": selected[0]["log_id"], "CANARY_AUTHORIZED": False, "NEW_RUN_BUDGET": 0},
        "schedule_mismatch_action": "HARD_FAIL_BEFORE_SIMULATOR_START", "SHA_mismatch_action": "HARD_FAIL_BEFORE_SIMULATOR_START",
    }
    for name, value in (("roster", roster), ("exclusion", exclusion), ("unselected", unselected_doc), ("schedule", schedule), ("bindings", binding_doc), ("architecture", architecture), ("taxonomy", taxonomy), ("authorization", authorization)):
        write_new(OUT[name], value)
    print(json.dumps({"selection_rank_tuple": roster["lexicographically_smallest_selection_rank_tuple"], "selected": compact, "outputs": {key: str(path) for key, path in OUT.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
