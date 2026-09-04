import copy
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from tools.r2_bj_b0_execute_frozen_hlc_v4_engineering import authorize_before_simulator_start
from tools.r2_bj_b0_freeze_engineering_package import SALT, exact_quota, speed_band
from tools.r2_bj_b0_hlc_v4_engineering_planner import B0ArchitectureViolation, audit_v4_planner_call


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def test_independent_exact_selection_replay_and_quotas():
    provenance = load("r2_bj_a5_applicable_pool_provenance_manifest_v1.0.json")
    frame = load("r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0.json")
    roster = load("r2_bj_b0_hlc_v4_engineering_roster_v1.0.json")
    by_frame = {row["frame_index"]: row for row in frame["entries"]}
    pool = []
    for row in provenance["records"]:
        source = by_frame[row["frame_index"]]
        digest = hashlib.sha256(
            SALT.encode() + b"\0" + row["scenario_token"].encode() + b"\0" + row["log_id"].encode()
        ).hexdigest()
        pool.append({
            "selection_hash_sha256": digest, "scenario_token": row["scenario_token"], "log_id": row["log_id"],
            "map_name": source["map_name"], "direction": source["direction"],
            "speed_band": speed_band(row["v_audit_mps"]),
        })
    pool.sort(key=lambda row: (row["selection_hash_sha256"], row["scenario_token"], row["log_id"]))
    expected = None
    for indexes in itertools.combinations(range(34), 8):
        if exact_quota([pool[index] for index in indexes]):
            expected = tuple(index + 1 for index in indexes)
            break
    assert expected == (1, 2, 3, 4, 5, 6, 7, 9)
    assert tuple(roster["lexicographically_smallest_selection_rank_tuple"]) == expected
    assert exact_quota(roster["entries"])


def test_roster_pool_dispositions_and_schedule_are_exact():
    roster = load("r2_bj_b0_hlc_v4_engineering_roster_v1.0.json")
    excluded = load("r2_bj_b0_permanent_engineering_exclusion_ledger_v1.0.json")
    unselected = load("r2_bj_b0_unselected_pool_disposition_v1.0.json")
    schedule = load("r2_bj_b0_hlc_v4_pair_schedule_v1.0.json")
    assert roster["candidate_pool_count"] == 34 and roster["entry_count"] == 8
    assert len({row["scenario_token"] for row in roster["entries"]}) == 8
    assert len({row["log_id"] for row in roster["entries"]}) == 8
    assert roster["history_and_permanent_exclusion_overlap_count"] == 0
    assert excluded["entry_count"] == 8 and all(row["PERMANENT_ENGINEERING_ONLY"] for row in excluded["entries"])
    assert unselected["entry_count"] == 26 and unselected["reserve_or_replacement_order"] == "NONE"
    assert all(row["disposition"] == "UNSELECTED_OUTCOME_UNEXPOSED_POOL" for row in unselected["entries"])
    assert schedule["pair_count"] == 8 and schedule["run_count"] == 16
    assert [row["run_order"] for row in schedule["runs"]] == list(range(1, 17))
    assert [row["arm"] for row in schedule["runs"]] == [arm for _ in range(8) for arm in ("BASELINE", "TREATMENT")]


def test_pair_bindings_share_everything_except_frozen_arm_morphology():
    document = load("r2_bj_b0_exact_pair_binding_manifest_v1.0.json")
    assert document["status"] == "FROZEN_PRE_OUTCOME" and document["pair_count"] == 8
    assert all(row["pre_outcome_complete"] for row in document["bindings"])
    assert all(row["only_allowed_arm_difference"] == "FROZEN_V4_HLC_MORPHOLOGY_CAPTURE_TREATMENT" for row in document["bindings"])
    assert all(row["shared_binding"]["pre_divergence"]["complete_trajectory_equality"] == "REQUIRED_EXACT" for row in document["bindings"])


def _valid_gate_payload():
    current = {"rear_axle": {"x": 1.0, "y": 2.0, "heading": 0.3}}
    states = [{"rear_axle": dict(current["rear_axle"])}]
    capture = {
        "controller_visible_curvature_profile": [0.0, 0.01],
        "actual_planned_target_frame_offsets_m": [1.0, 0.0],
        "state0_exact_current_xy": True, "state0_exact_current_heading": True,
        "minimum_stitching_horizon_s": 3.0, "effective_stitching_duration_s": 3.0,
        "feasibility": {"state0_to_state1_distance_m": 1.0, "nominal_state_step_distance_m": 1.0,
                        "state0_tangent_mismatch_abs_rad": 0.0, "future_heading_xy_mismatch_abs_rad": 0.0},
    }
    return current, states, capture


@pytest.mark.parametrize("mutation,code", [
    (("controller_visible_curvature_profile", [0.0, 0.6]), "CURVATURE_LIMIT"),
    (("actual_planned_target_frame_offsets_m", [1.0, 0.01]), "TARGET_FRAME_RESIDUAL"),
    (("effective_stitching_duration_s", 2.9), "STITCHING_HORIZON"),
])
def test_online_architecture_gate_mutations_fail_closed(mutation, code):
    current, states, capture = _valid_gate_payload()
    capture[mutation[0]] = mutation[1]
    with pytest.raises(B0ArchitectureViolation) as error:
        audit_v4_planner_call(current, states, capture, 1.0, 3.0, 1.2)
    assert code in error.value.codes
    assert error.value.audit["stop_action"] == ["STOP_CURRENT_RUN", "STOP_REMAINING_SCHEDULE"]


def test_predivergence_mutation_fails_closed():
    current, states, capture = _valid_gate_payload()
    other = copy.deepcopy(states)
    other[0]["rear_axle"]["x"] += 1e-12
    with pytest.raises(B0ArchitectureViolation) as error:
        audit_v4_planner_call(current, states, capture, 1.0, 3.0, 0.5, other)
    assert "PREDIVERGENCE_TRAJECTORY_MISMATCH" in error.value.codes


def test_authorization_sha_budget_and_schedule_mutations_stop_before_simulator():
    gate = {"BJ_B_ENGINEERING_SIMULATION_AUTHORIZED": True, "NEW_RUN_BUDGET": 2,
            "authorized": {"component_manifest_sha256": "c", "schedule_sha256": "s", "pair_binding_sha256": "b"}}
    authorize_before_simulator_start(gate, "c", "s", "b", 2, True)
    mutations = [
        ({**gate, "BJ_B_ENGINEERING_SIMULATION_AUTHORIZED": False}, "c", "s", "b", 2, True),
        (gate, "wrong", "s", "b", 2, True), (gate, "c", "wrong", "b", 2, True),
        (gate, "c", "s", "b", 3, True), (gate, "c", "s", "b", 2, False),
    ]
    for arguments in mutations:
        with pytest.raises(PermissionError):
            authorize_before_simulator_start(*arguments)


def test_zero_run_constructs_all_16_and_canary_remains_unauthorized():
    audit = load("r2_bj_b0_zero_run_integration_preflight_audit_v1.0.json")
    gate = load("r2_bj_b0_execution_authorization_gate_v1.0.json")
    assert audit["hydra_compositions"] == audit["exact_scenario_resolutions"] == audit["runner_constructions"] == 16
    assert audit["predivergence_exact_pair_count"] == 8
    assert all(row["planner_class"] == "R2BJB0HLCV4EngineeringPlanner" for row in audit["runs"])
    assert all(row["time_controller_class"] == "R1Primary80ScientificTimeControllerV1" and row["time_controller_iterations"] == 81 for row in audit["runs"])
    assert all(row["simulation_started"] is False and row["runner_run_calls"] == 0 for row in audit["runs"])
    assert audit["RUNNER_RUN"] == audit["NEW_RUN_BUDGET"] == 0
    assert gate["BJ_B_ENGINEERING_SIMULATION_AUTHORIZED"] is False
    assert gate["future_canary"]["CANARY_AUTHORIZED"] is False and gate["future_canary"]["NEW_RUN_BUDGET"] == 0
    assert gate["future_canary"]["identity_selection_rank"] == 1


def test_no_forbidden_selection_fields_and_no_tsb():
    roster = load("r2_bj_b0_hlc_v4_engineering_roster_v1.0.json")
    forbidden = {"component_margin", "curvature_margin", "yaw_rate_margin", "lateral_acceleration_margin", "outcome"}
    assert all(not forbidden.intersection(row) for row in roster["entries"])
    assert Counter(row["direction"] for row in roster["entries"]) == {"right": 6, "left": 2}
    assert all(row["pair_id"].startswith("R2BJB0-HLC-") for row in roster["entries"])


def test_component_sha_manifest_closes_current_zero_run_package():
    manifest = load("r2_bj_b0_component_sha_binding_manifest_v1.0.json")
    assert manifest["component_SHA_closure"] == "PASS"
    assert manifest["full_Hydra_compositions"] == manifest["exact_scenario_resolutions"] == 16
    assert manifest["SimulationRunner_constructions"] == 16
    assert manifest["RUNNER_RUN"] == manifest["NEW_RUN_BUDGET"] == 0
    assert manifest["CANARY_AUTHORIZED"] is False
    for row in manifest["components"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    for row in manifest["external_bound_nuplan_1_2_2_runtime_components"]:
        assert hashlib.sha256(Path(row["absolute_path"]).read_bytes()).hexdigest() == row["sha256"]
