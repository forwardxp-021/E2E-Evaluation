import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def test_preregistration_freezes_frame_speed_curvature_and_zero_run():
    contract = load("r2_bj_a3_preregistered_contract_v1.0.json")
    predicate = load("r2_bj_a3_hlc_prospective_applicability_predicate_v1.0.json")
    assert contract["status"] == "PREREGISTERED_BEFORE_A3_FRESH_CANDIDATE_EVALUATION"
    assert contract["audit_frame"]["size"] == 256
    assert contract["audit_frame"]["early_stop_after_enough_applicable_candidates"] is False
    assert contract["corrected_speed_semantics"]["v_audit"] == "max(v_official_initial,max(v_pre_treatment_0_to_1p0s))"
    assert contract["curvature_representation"]["identity_specific_smoothing"] is False
    assert predicate["scenario_specific_parameters_allowed"] is False
    assert contract["execution"]["runner_run_authorized"] == 0


def test_hash_ranked_frame_is_fixed_unique_outcome_blind_and_not_roster():
    frame = load("r2_bj_a3_hash_ranked_audit_frame_manifest_v1.0.json")
    assert frame["status"] == "FROZEN_BEFORE_FINAL_APPLICABILITY_EVALUATION"
    assert frame["frame_size"] == len(frame["entries"]) == 256
    assert frame["source_scan_accounting"]["source_rows_scanned_after_canonical_DB_dedup"] == 5386575
    assert frame["final_applicability_outcomes_opened_before_frame_freeze"] == 0
    assert frame["candidate_role"] == "AUDIT_FRAME_ONLY_NOT_BJ_B_ROSTER"
    assert len({row["scenario_token"] for row in frame["entries"]}) == 256
    assert len({row["log_id"] for row in frame["entries"]}) == 256
    ranks = [row["audit_rank_sha256"] for row in frame["entries"]]
    assert ranks == sorted(ranks)
    assert frame["frame_entries_canonical_sha256"]


def test_corrected_historical_and_legacy_dispositions_fail_closed_without_rewrite():
    review = load("r2_bj_a3_corrected_speed_envelope_review_v1.0.json")
    legacy = load("r2_bj_a3_legacy_technical_disposition_ledger_v1.0.json")
    assert review["historical_complete_record_count"] == 47
    assert review["pass_count"] == 46
    assert review["status"] == "FAIL_CLOSED"
    for row in review["rows"]:
        info = row.get("corrected_speed_information")
        if info:
            assert info["v_audit_mps"] >= info["official_initial_speed_mps"]
            assert info["v_audit_mps"] >= info["pre_treatment_speed_distribution_mps"]["max"]
            assert info["anchor_used_for_selection_or_eligibility"] is False
    assert legacy["entry_count"] == 10
    assert all(not row["outcome_exclusion"] and not row["added_to_blacklist"] for row in legacy["entries"])
    assert all(row["historical_scientific_result_changed"] is False for row in legacy["entries"])


def test_all_256_are_audited_and_only_17_pass_same_predicate():
    ledger = load("r2_bj_a3_fresh_candidate_eligibility_ledger_v1.0.json")
    assert ledger["audited_count"] == len(ledger["entries"]) == 256
    assert ledger["applicable_count"] == 17
    assert ledger["early_stop_used"] is False
    assert ledger["roster_selected"] is False
    assert ledger["full_source_universe_census_claimed"] is False
    passing = [row for row in ledger["entries"] if row["status"] == "PASS"]
    assert len(passing) == 17
    for row in passing:
        closure = row["closure"]
        assert closure["predicate_status"] == "PASS"
        assert closure["route_coverage"]["extrapolation_used"] is False
        assert closure["component_audit"]["planner_call_cases"] == 960
        assert closure["component_audit"]["frozen_full_V4_gate_failures"] == 0
        assert row["anchor_timestamp_used_for_selection_or_eligibility"] is False


def test_curvature_component_and_provenance_closure_are_explicit():
    curvature = load("r2_bj_a3_curvature_disposition_addendum_v1.0.json")
    component = load("r2_bj_a3_native_generated_composite_component_audit_v1.0.json")
    provenance = load("r2_bj_a3_hlc_joint_support_provenance_manifest_v1.0.json")
    assert curvature["undefined_catch_all_count"] == 0
    assert curvature["raw_and_robust_retained"] is True
    assert curvature["legacy_0p082281_formal_disposition"] == "TERMINAL_SHORT_SEGMENT_GRADIENT_ARTIFACT_NOT_ACTUAL_JOINT_SUPPORT"
    assert component["opportunities_reaching_full_component_stage"] == 28
    assert component["planner_state_case_count"] == 28 * 960
    assert component["opportunities_full_gate_pass"] == 17
    assert component["generated_increment_infeasible_opportunities"] == 11
    assert component["negative_native_generated_cancellation_accepted"] is False
    assert provenance["passing_joint_record_count"] == 17
    assert provenance["closure_percent"] == 100.0


def test_a3_is_not_ready_and_governance_is_zero_run():
    envelope = load("r2_bj_a3_joint_support_envelope_v1.0.json")
    firewall = load("r2_bj_a3_data_firewall_audit_v1.0.json")
    request = (R2 / "R2_BJ_A3_Scientific_Owner_Readiness_Request_v0.1.md").read_text(encoding="utf-8")
    assert envelope["status"] == "JOINT_SUPPORT_EXTRACTION_INCOMPLETE"
    assert "V4_GENERATED_INCREMENT_INFEASIBLE" in envelope["blocking_categories"]
    assert envelope["applicable_count"] == 17
    assert envelope["provenance_geometry_speed_component_closure_percent"] == 100.0
    assert firewall["status"] == "PASS_NO_OUTCOME_LEAKAGE"
    assert firewall["runner_run_calls"] == firewall["engineering_simulation_calls"] == 0
    assert firewall["scientific_simulation_calls"] == firewall["TSB_simulation_calls"] == 0
    assert firewall["BJ_B_roster_selected"] is False
    assert "请求暂缓" in request


def test_current_manifest_binds_current_quick_reference_and_all_components():
    manifest = load("r2_bj_a3_component_sha_binding_manifest_v1.0.json")
    assert manifest["component_SHA_closure"] == "PASS"
    assert manifest["current_QUICK_REFERENCE_bound_here"] is True
    assert manifest["runner_run_calls"] == manifest["simulation_calls"] == 0
    for row in manifest["components"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
