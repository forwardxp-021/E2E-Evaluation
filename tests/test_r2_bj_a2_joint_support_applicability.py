import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
A2_BOUND_COMMIT = "a04f8c4ec5e2448559fb4600ad0c1c830b3fd8ed"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def test_contract_is_preregistered_and_keeps_all_frozen_semantics():
    contract = load("r2_bj_a2_joint_support_applicability_contract_v1.0.json")
    assert contract["status"] == "PREREGISTERED_BEFORE_A2_JOINT_SUPPORT_EVALUATION"
    assert contract["frozen_baseline"]["local_commit"] == "5c991b32c1d26dc5887016e3760a58b0fec64aeb"
    assert contract["frozen_baseline"]["tree"] == "ab13e024df1b3c78e4c529e56b4ecfb03855f410"
    assert contract["frozen_V4"]["parameter_change_authorized"] is False
    assert contract["curvature_representation"]["identity_specific_smoothing"] is False
    assert contract["execution"]["runner_run_authorized"] == 0


def test_joint_records_are_coupled_and_full_population_is_not_overclaimed():
    provenance = load("r2_bj_a2_hlc_joint_support_provenance_manifest_v1.0.json")
    assert provenance["status"] == "JOINT_SUPPORT_EXTRACTION_INCOMPLETE"
    assert provenance["unique_committed_selected_HLC_opportunities_seen"] == 57
    assert provenance["joint_support_records"] + len(provenance["technical_extraction_failures"]) == 57
    assert provenance["full_source_universe_eligible_population_materialized"] is False
    assert provenance["joint_support_records"] > 0
    for row in provenance["joint_records"]:
        assert row["scenario_token"] and row["log_id"] and row["scenario_anchor_timestamp_us"]
        assert row["speed_information"]["pre_treatment_samples"]
        assert row["actual_reference_geometry"]["source_reference_xy"]
        assert row["actual_reference_geometry"]["target_reference_xy"]
        assert row["technical_eligibility"]["HLC_route_continuous_Primary80_applicability"] == "PASS_RECONSTRUCTED_V2_3_WITH_MARGIN"


def test_curvature_raw_robust_forensic_and_legacy_extreme_are_retained():
    audit = load("r2_bj_a2_curvature_quality_forensic_v1.0.json")
    assert audit["raw_and_robust_both_retained"] is True
    assert audit["manual_spike_deletion"] is False
    assert audit["joint_support_rows"]
    for row in audit["joint_support_rows"]:
        for side in ("source", "target"):
            quality = row[side]
            assert "median" in quality["raw_pointwise_abs_curvature_inv_m"]
            assert "p99" in quality["robust_abs_curvature_inv_m"]
            assert "raw_adjacent_point_support_distribution_m" in quality
            assert set(quality["continuous_support_m"]) == {"0.02", "0.05", "0.08"}
    legacy = audit["legacy_0p082281_forensic_appendix"]
    assert legacy["exact_reported_extreme_inv_m"] == 0.082281
    assert len(legacy["records"]) == 2
    assert all(not row["speed_and_extreme_accepted_as_valid_joint_window_support"] for row in legacy["records"])
    for row in legacy["records"]:
        reproduced = row["historical_formula_reproduction"]
        assert reproduced["abs_curvature_inv_m"]["max"] > 0.082
        assert reproduced["abs_max_is_terminal_point"] is True
        assert reproduced["abs_max_point_support_m"] < 0.01
        assert row["reconstructed_full_target_reference_quality"]["robust_abs_curvature_inv_m"]["max"] < 0.002
        assert row["curvature_disposition"] == "TERMINAL_SHORT_SEGMENT_GRADIENT_ARTIFACT_NOT_SUSTAINED_ROAD_CURVATURE"


def test_component_audit_uses_full_states_and_never_hides_generated_increment():
    provenance = load("r2_bj_a2_hlc_joint_support_provenance_manifest_v1.0.json")
    audit = load("r2_bj_a2_native_generated_composite_component_audit_v1.0.json")
    assert audit["opportunity_count"] == provenance["joint_support_records"]
    assert audit["planner_state_case_count"] == audit["opportunity_count"] * 2 * 2 * 3 * 80
    assert audit["negative_native_generated_cancellation_accepted_as_generated_pass"] is False
    assert audit["runner_run_calls"] == audit["simulation_calls"] == 0
    for row in audit["opportunities"]:
        summary = row["summary"]
        assert summary["planner_call_cases"] == 2 * 2 * 3 * 80
        assert summary["post_recommit_settling_cases"] > 0
        assert "generated_increment" in summary["component_maxima"]
        assert "stitching_capture_increment" in summary["component_maxima"]


def test_envelope_and_firewall_fail_closed_without_any_run():
    envelope = load("r2_bj_a2_joint_support_applicability_envelope_v1.0.json")
    firewall = load("r2_bj_a2_data_firewall_audit_v1.0.json")
    assert envelope["status"] == "JOINT_SUPPORT_EXTRACTION_INCOMPLETE"
    assert "JOINT_SUPPORT_EXTRACTION_INCOMPLETE" in envelope["all_blocking_categories"]
    assert envelope["full_eligible_population_provenance_closure_percent"] is None
    assert envelope["complete_joint_record_provenance_closure_percent"] == 100.0
    assert envelope["considered_opportunity_extraction_completion_percent"] < 100.0
    assert envelope["BJ_A_cartesian_envelope"]["role"] == "ADVERSARIAL_STRESS_APPENDIX_NOT_ACTUAL_DOMAIN_DECIDER"
    assert envelope["runner_run_calls"] == envelope["engineering_simulation_calls"] == 0
    assert envelope["scientific_simulation_calls"] == envelope["TSB_simulation_calls"] == 0
    assert firewall["status"] == "PASS_NO_OUTCOME_LEAKAGE"
    assert firewall["forbidden_outcome_files_opened"] == 0
    assert firewall["baseline_treatment_outcomes_used"] is False
    assert firewall["new_roster_selected"] is False


def test_request_withheld_and_sha_manifest_closes():
    request = (R2 / "R2_BJ_A2_Scientific_Owner_Readiness_Request_v0.1.md").read_text(encoding="utf-8")
    manifest = load("r2_bj_a2_component_sha_binding_manifest_v1.0.json")
    assert "REQUEST_WITHHELD" in request
    assert "JOINT_SUPPORT_EXTRACTION_INCOMPLETE" in request
    assert manifest["component_SHA_closure"] == "PASS"
    assert manifest["runner_run_calls"] == manifest["simulation_calls"] == 0
    for row in manifest["components"]:
        historical = subprocess.check_output(
            ["git", "show", f"{A2_BOUND_COMMIT}:{row['path']}"], cwd=ROOT,
        )
        assert hashlib.sha256(historical).hexdigest() == row["sha256"]
