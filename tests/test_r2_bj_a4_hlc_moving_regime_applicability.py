import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


def load(name):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


def test_contract_freezes_moving_regime_speed_and_zero_run():
    contract = load("r2_bj_a4_preregistered_contract_v1.0.json")
    predicate = load("r2_bj_a4_hlc_moving_regime_applicability_predicate_v1.0.json")
    assert contract["estimand"]["name"] == "MOVING_VEHICLE_HESITANT_LANE_CHANGE"
    assert contract["estimand"]["R_HLC_APPLICABILITY_SPEED_FLOOR_MPS"] == 3.0
    assert contract["estimand"]["speed_gate_order"] == "BEFORE_TOPOLOGY_CURVATURE_AND_V4_COMPONENT_AUDIT"
    assert contract["audit_frame"]["size"] == 768
    assert contract["audit_frame"]["EARLY_STOP"] is False
    assert contract["audit_frame"]["GLOBAL_TOKEN_DEDUP"] is True
    assert contract["audit_frame"]["GLOBAL_LOG_DEDUP"] is True
    assert predicate["scenario_specific_tuning"] is False
    assert contract["execution"]["runner_run_authorized"] == 0


def test_exhaustive_unique_log_support_proves_frame_cardinality_shortfall():
    frame = load("r2_bj_a4_hash_ranked_audit_frame_manifest_v1.0.json")
    assert frame["status"] == "APPLICABLE_POOL_INSUFFICIENT"
    assert frame["frame_freeze_complete"] is False
    assert frame["frame_target_size"] == 768
    assert frame["frame_size"] == len(frame["entries"]) == 557
    assert frame["frame_cardinality_shortfall"] == 211
    assert frame["source_scan_accounting"]["source_rows_scanned_after_canonical_DB_dedup"] == 5386575
    assert frame["source_scan_accounting"]["canonical_logs_exhaustively_examined"] == 1621
    assert frame["source_scan_accounting"]["logs_with_at_least_one_basic_HLC_record"] == 557
    assert frame["EARLY_STOP"] is False
    assert frame["A4_predicate_outcomes_opened_before_frame_freeze"] == 0
    assert len({row["scenario_token"] for row in frame["entries"]}) == 557
    assert len({row["log_id"] for row in frame["entries"]}) == 557
    assert [row["audit_rank_sha256"] for row in frame["entries"]] == sorted(
        row["audit_rank_sha256"] for row in frame["entries"]
    )


def test_historical_low_speed_and_topology_dispositions_do_not_rewrite_or_blacklist():
    disposition = load("r2_bj_a4_a3_and_historical_applicability_disposition_v1.0.json")
    assert disposition["A3_generated_composite_failure_count"] == 11
    assert disposition["A3_failure_all_11_below_speed_floor"] is True
    assert all(row["A4_disposition"] == "LOW_SPEED_OUTSIDE_V4_APPLICABILITY" for row in disposition["A3_failures"])
    assert all(row["moving_regime_V4_failure_counted"] is False for row in disposition["A3_failures"])
    topology = disposition["A2_corrected_speed_topology_failure"]
    assert topology["scenario_token"] == "3feb5f93f24e5b77"
    assert topology["A4_disposition"] == "HISTORICAL_OPPORTUNITY_NOT_APPLICABLE_UNDER_CURRENT_V2_3"
    assert topology["topology_builder_modified"] is False
    assert disposition["legacy_reconstructable_but_low_speed_count"] == 2
    assert disposition["outcome_blacklist_entries_created"] == 0


def test_fail_closed_before_partial_predicate_or_component_audit():
    eligibility = load("r2_bj_a4_fresh_candidate_eligibility_ledger_v1.0.json")
    component = load("r2_bj_a4_native_generated_composite_component_audit_v1.0.json")
    curvature = load("r2_bj_a4_curvature_disposition_audit_v1.0.json")
    provenance = load("r2_bj_a4_passing_candidate_provenance_manifest_v1.0.json")
    assert eligibility["status"] == "APPLICABLE_POOL_INSUFFICIENT"
    assert eligibility["A4_predicate_evaluated_count"] == 0
    assert component["status"] == "NOT_EXECUTED_FRAME_FREEZE_FAILED_CLOSED"
    assert component["planner_state_case_count"] == 0
    assert curvature["status"] == "NOT_REACHED_FRAME_FREEZE_FAILED_CLOSED"
    assert curvature["undefined_category_count"] == 0
    assert provenance["passing_count"] == 0


def test_final_disposition_preserves_firewall_and_no_downstream_actions():
    envelope = load("r2_bj_a4_moving_regime_candidate_pool_envelope_v1.0.json")
    firewall = load("r2_bj_a4_data_firewall_audit_v1.0.json")
    request = (R2 / "R2_BJ_A4_Scientific_Owner_Readiness_Request_v0.1.md").read_text(encoding="utf-8")
    assert envelope["status"] == "APPLICABLE_POOL_INSUFFICIENT"
    assert envelope["blocking_category"] == "A4_FRAME_CARDINALITY_UNATTAINABLE_UNDER_GLOBAL_LOG_DEDUP"
    assert envelope["audit_frame_complete"] is False
    assert envelope["audit_frame_shortfall"] == 211
    assert firewall["status"] == "PASS_NO_OUTCOME_LEAKAGE"
    assert firewall["V4_parameters_changed"] is False
    assert firewall["thresholds_changed"] is False
    assert firewall["runner_run_calls"] == firewall["engineering_simulation_calls"] == 0
    assert firewall["scientific_simulation_calls"] == firewall["TSB_simulation_calls"] == 0
    assert firewall["BJ_B_roster_selected"] is False
    assert "不得进入 BJ-B" in request


def test_manifest_sha_closure():
    manifest = load("r2_bj_a4_component_sha_binding_manifest_v1.0.json")
    assert manifest["status"] == "APPLICABLE_POOL_INSUFFICIENT"
    assert manifest["component_SHA_closure"] == "PASS"
    assert manifest["runner_run_calls"] == manifest["simulation_calls"] == 0
    for row in manifest["components"]:
        historical = subprocess.run(
            ["git", "show", f"39922a8af72de382de21eb4bf98326f3639f73d4:{row['path']}"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == row["sha256"]
