"""Read-only contract checks for the R1 B3 forensic artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"


def read(name: str):
    return json.loads((R1 / name).read_text(encoding="utf-8"))


def assert_zero_execution(payload):
    assert payload["execution_this_phase"] == {
        "simulation": 0,
        "runner.run": 0,
        "run_runners": 0,
        "official_rerun": 0,
        "canary_rerun": 0,
        "selector": 0,
        "RBR": 0,
    }


def test_outcome_exposure_firewall_and_frozen_evaluation_parity():
    value = read("r1_b3_r1_official_outcome_exposure_ledger_v1.0.json")
    assert value["identity_count"] == 24
    assert value["frozen_evaluation_validation"]["status"] == (
        "24_OF_24_FROZEN_EVALUATION_TO_COMMITTED_GATE_TABLE_EXACT_MATCH"
    )
    assert value["raw_output_validation"]["files_verified"] == 1080
    assert value["raw_output_validation"]["mismatches"] == 0
    assert len({(row["scenario_token"], row["log_id"]) for row in value["identities"]}) == 24
    assert all(
        row["OUTCOME_EXPOSED"]
        and row["R1_SCIENTIFIC_HISTORY_ONLY"]
        and row["R2_DEVELOPMENT_USE_FORBIDDEN"]
        and row["R2_CONFIRMATORY_SCIENTIFIC_USE_FORBIDDEN"]
        and row["allowed_use"] == "READ_ONLY_FAILURE_DIAGNOSTIC_ONLY"
        for row in value["identities"]
    )
    assert_zero_execution(value)


def test_hlc_realized_mechanism_and_ideal_intent_are_separated():
    value = read("r1_b3_hlc_realized_mechanism_forensic_v1.json")
    assert len(value["pairs"]) == 12
    assert value["evaluation_parity"] == "12_OF_12_EXACT_FROZEN_FUNCTION_PARITY"
    assert value["gate_counts"] == {
        "baseline_retreat_eq_0": 12,
        "treatment_retreat_ge_1": 12,
        "latency_delta_ge_0p5": 12,
        "baseline_and_treatment_status_OK": 12,
        "delta_monotonic_le_minus_0p10": 0,
        "unique_failure_reason_is_MONOTONIC_PENALTY_LT_0P1": 12,
    }
    ideal = value["ideal_generator_diagnostic"]
    assert ideal["role"] == "ANALYTICAL_GENERATOR_INTENT_DIAGNOSTIC_ONLY"
    assert ideal["baseline"]["monotonic_transition_fraction"] == 1.0
    assert ideal["treatment"]["monotonic_transition_fraction"] == 0.872697
    assert ideal["ideal_delta_monotonic_fraction"] == -0.127303
    assert value["threshold_changed"] is False
    assert_zero_execution(value)


def test_hlc_endpoint_failure_breakdown():
    value = read("r1_b3_hlc_endpoint_failure_forensic_v1.json")
    assert len(value["pairs"]) == 12
    assert value["summary"]["endpoint_pass"] == 6
    assert value["summary"]["endpoint_fail"] == 6
    assert value["summary"]["failure_gate_counts"] == {
        "treatment.lateral_velocity": 5,
        "treatment.offset": 1,
    }
    assert value["summary"]["heading_failures"] == 0
    assert value["summary"]["route_progress_failures"] == 0
    assert_zero_execution(value)


def test_tsb_realized_status_and_ideal_generator_morphology():
    value = read("r1_b3_tsb_realized_mechanism_forensic_v1.json")
    assert len(value["pairs"]) == 12
    assert value["summary"]["baseline_status_counts"] == {
        "NO_BRAKE_PHASE": 12,
        "LOW_SPEED_ENDSTOP": 0,
        "OK": 0,
    }
    assert value["summary"]["treatment_status_counts"] == {
        "NO_BRAKE_PHASE": 12,
        "LOW_SPEED_ENDSTOP": 0,
        "OK": 0,
    }
    assert value["summary"]["baseline_brake_phase_count_distribution"] == {"0": 12}
    assert value["summary"]["treatment_brake_phase_count_distribution"] == {"0": 12}
    assert value["summary"]["treatment_release_window_realized"] == 12
    assert value["summary"]["transfer_failure_label_counts"] == {
        "BRAKE_AMPLITUDE_ATTENUATION": 12
    }
    ideal = value["ideal_generator_diagnostic"]
    assert ideal["role"] == "GENERATOR_INTENT_DIAGNOSTIC_ONLY"
    assert ideal["baseline"]["brake_phase_count"] == 1
    assert ideal["treatment"]["brake_phase_count"] == 2
    assert ideal["treatment"]["interstage_release_fraction"] == 0.333333
    assert ideal["treatment"]["second_brake_peak_ratio"] == 1.0
    assert_zero_execution(value)


def test_one_step_transfer_reads_serialized_planner_intent_without_execution():
    value = read("r1_b3_intent_to_realized_transfer_audit_v1.json")
    assert len(value["per_run"]) == 48
    assert all(row["state0_exact_current_ego_identity"] == "79_OF_79" for row in value["per_run"])
    tsb = [row for row in value["per_run"] if row["family"] == "R-TSB"]
    assert len(tsb) == 24
    assert all(row["TSB_planned_state1_delta_vs_frozen_acceleration_dt_max_abs_error"] <= 1e-9 for row in tsb)
    assert value["interpretation"] == {
        "R-HLC": "ATTENUATED_WITH_0P3_TO_0P4S_DESCRIPTIVE_LAG",
        "R-TSB": "TREATMENT_COMMAND_TRANSFER_COLLAPSED_RELATIVE_TO_FROZEN_PHASE_SEMANTICS",
    }
    assert_zero_execution(value)


def test_fmatch_and_safety_forensics_preserve_frozen_results():
    fmatch = read("r1_b3_fmatch_distribution_audit_v1.json")
    assert fmatch["families"]["R-HLC"]["pair_pass_count"] == 12
    assert fmatch["families"]["R-TSB"]["pair_pass_count"] == 12
    assert fmatch["scientific_interpretation"] == "HANDCRAFTED_NUISANCE_MATCHING_SUCCESSFUL"
    assert fmatch["caliper_redefined"] is False
    assert_zero_execution(fmatch)

    safety = read("r1_b3_safety_failure_forensic_v1.json")
    assert safety["safety_failing_pair_count"] == 2
    assert {row["pair_id"] for row in safety["failures"]} == {
        "R1B29E-01-R-HLC",
        "R1B29E-21-R-TSB",
    }
    assert safety["safety_repair_performed"] is False
    assert safety["safety_threshold_changed"] is False
    assert_zero_execution(safety)
