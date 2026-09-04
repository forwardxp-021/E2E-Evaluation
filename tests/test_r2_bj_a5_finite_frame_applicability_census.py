import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def test_preregistered_contract_and_frozen_input_shas():
    contract = load("r2_bj_a5_preregistered_finite_frame_census_contract_v1.0.json")
    audit = load("r2_bj_a5_a4_frozen_frame_binding_audit_v1.0.json")
    assert contract["status"] == "PREREGISTERED_BEFORE_ANY_A5_PREDICATE_OUTCOME"
    assert contract["FRAME_SOURCE"] == "EXACT_A4_FROZEN_557_ENTRIES"
    assert contract["CENSUS_TARGET"] == 557
    assert contract["SELECTION"] == "NONE"
    assert contract["RERANK"] is contract["RESCAN_SOURCE_UNIVERSE"] is False
    assert contract["REPLACEMENT"] is contract["EARLY_STOP"] is False
    assert contract["SPEED_FLOOR_MPS"] == 3.0
    assert audit["status"] == "PASS_CENSUS_INPUT_INTEGRITY_CLOSED_BEFORE_A5_OUTCOMES"
    assert audit["frame_entry_count"] == audit["unique_scenario_token_count"] == audit["unique_log_id_count"] == 557
    assert audit["A4_predicate_outcomes_opened_before_frame_freeze"] == 0
    assert audit["historical_permanent_A3_overlap_count"] == 0
    for key in ("A4_frame", "A4_predicate", "V4_generator", "V4_planner"):
        item = contract["frozen_inputs"][key]
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_census_evaluates_exact_frozen_order_without_early_stop():
    frame = load("r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0.json")
    ledger = load("r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json")
    assert ledger["status"] == "COMPLETE_557_OF_557"
    assert ledger["census_target"] == ledger["census_evaluated"] == len(ledger["entries"]) == 557
    assert ledger["EARLY_STOP"] is False
    assert ledger["selection"] == "NONE"
    assert ledger["rerank"] is ledger["replacement"] is False
    assert [(row["scenario_token"], row["log_id"], row["audit_rank_sha256"]) for row in ledger["entries"]] == [
        (row["scenario_token"], row["log_id"], row["audit_rank_sha256"]) for row in frame["entries"]
    ]
    assert ledger["stage_completion_counts"] == {"P03": 341, "P04": 182, "P12": 34}
    assert ledger["applicable_pool_count"] == 34


def test_speed_semantics_and_record_sha_provenance_are_complete():
    ledger = load("r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json")
    for row in ledger["entries"]:
        assert row["complete_input_component_output_SHA_provenance"] is True
        assert row["input_record_canonical_sha256"]
        assert row["predicate_result_canonical_sha256"]
        assert row["scientific_outcome_blacklist_addition"] is False
        if row["v_audit_mps"] is not None:
            assert row["v_audit_mps"] == max(row["official_initial_speed_mps"], row["pre_treatment_max_speed_0_to_1p0s_mps"])
            assert (row["v_audit_mps"] >= 3.0) == (row["moving_regime_speed_gate"] == "PASS")


def test_component_census_has_34_full_960_case_passes_without_cancellation():
    component = load("r2_bj_a5_native_generated_composite_component_audit_v1.0.json")
    assert component["status"] == "PASS_NO_MOVING_REGIME_ARCHITECTURE_FAILURE"
    assert component["component_stage_count"] == len(component["opportunities"]) == 34
    assert component["planner_state_case_count"] == 34 * 960 == 32640
    assert component["moving_regime_component_failure_count"] == 0
    assert component["native_only_infeasible_opportunity_count"] == 0
    assert component["generated_increment_failure_opportunity_count"] == 0
    assert component["composite_failure_opportunity_count"] == 0
    assert component["continuity_failure_opportunity_count"] == 0
    assert component["terminal_settling_failure_opportunity_count"] == 0
    assert component["negative_native_generated_cancellation_accepted"] is False
    assert all(row["planner_state_case_count"] == 960 for row in component["opportunities"])


def test_curvature_and_applicable_pool_provenance_close_at_100_percent():
    curvature = load("r2_bj_a5_curvature_disposition_audit_v1.0.json")
    provenance = load("r2_bj_a5_applicable_pool_provenance_manifest_v1.0.json")
    assert curvature["status"] == "DEFINED_FOR_ALL_REACHED_RECORDS"
    assert curvature["records_reaching_curvature_disposition"] == 34
    assert curvature["undefined_category_count"] == 0
    assert curvature["raw_and_robust_retained"] is True
    assert provenance["status"] == "APPLICABLE_POOL_PROVENANCE_CLOSURE_100_PERCENT"
    assert provenance["applicable_pool_count"] == len(provenance["records"]) == 34
    assert provenance["closure_percent"] == 100.0


def test_final_ready_status_does_not_select_or_run():
    envelope = load("r2_bj_a5_finite_frame_census_envelope_v1.0.json")
    firewall = load("r2_bj_a5_data_firewall_audit_v1.0.json")
    request = (R2 / "R2_BJ_A5_Scientific_Owner_Readiness_Request_v0.1.md").read_text(encoding="utf-8")
    assert envelope["status"] == "R2_BJ_A5_CENSUS_COMPLETE_READY_FOR_BJ_B_OWNER_REVIEW"
    assert envelope["A4_FRAME_CAPACITY"] == envelope["A5_CENSUS_EVALUATED"] == 557
    assert envelope["A5_APPLICABLE_POOL"] == envelope["A5_COMPONENT_STAGE_COUNT"] == 34
    assert envelope["A5_MOVING_REGIME_COMPONENT_FAILURES"] == 0
    assert envelope["BJ_B_ROSTER_SELECTED"] is False
    assert envelope["RUNNER_RUN"] == 0
    assert firewall["status"] == "PASS_NO_OUTCOME_LEAKAGE"
    assert firewall["source_universe_rescanned"] is False
    assert firewall["A5_failure_blacklist_entries_created"] == 0
    assert firewall["engineering_simulation"] == firewall["scientific_simulation"] == firewall["TSB_simulation"] == 0
    assert "不自动选择" in request


def test_current_a5_manifest_binds_all_components():
    manifest = load("r2_bj_a5_component_sha_binding_manifest_v1.0.json")
    assert manifest["status"] == "R2_BJ_A5_CENSUS_COMPLETE_READY_FOR_BJ_B_OWNER_REVIEW"
    assert manifest["component_SHA_closure"] == "PASS"
    assert manifest["A5_CENSUS_EVALUATED"] == 557
    assert manifest["A5_APPLICABLE_POOL"] == 34
    assert manifest["A5_MOVING_REGIME_COMPONENT_FAILURES"] == 0
    assert manifest["BJ_B_ROSTER_SELECTED"] is False
    assert manifest["RUNNER_RUN"] == manifest["simulation_calls"] == 0
    for row in manifest["components"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
