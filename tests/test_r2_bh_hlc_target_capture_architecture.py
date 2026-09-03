import json
from pathlib import Path

import numpy as np

from tools.r2_bh_hlc_target_capture_generator_v2 import (
    ARM_BASELINE,
    ARM_TREATMENT,
    absolute_capture_weight,
    behavior_progress,
    replanning_capture_weight,
    target_capture_path,
    validate_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"


def read(name):
    return json.loads((R2 / name).read_text())


def parameters():
    return read("r2_bh_hlc_arch_parameter_space_v2.0.json")["round0"]


def test_v1_reanchor_invariant_diagnosis_is_supported_for_all_cases():
    audit = read("r2_bh_hlc_v1_reanchor_invariant_audit_v1.json")
    assert audit["diagnosis_pass"] is True
    assert [row["planned_terminal_target_offset_m"] for row in audit["cases"]] == [0.0, 0.25, 0.5, -0.25, -0.5]


def test_fixed_absolute_capture_reaches_zero_and_state0_is_exact():
    params = parameters()
    validate_parameters(params)
    capture = params["capture"]
    start = capture["capture_start_abs_s"]
    end = start + capture["capture_duration_s"]
    time = np.linspace(start, end, 15)
    assert absolute_capture_weight(time, capture)[0] == 1.0
    assert absolute_capture_weight(time, capture)[-1] == 0.0
    future = end + np.arange(10) * 0.1
    weights = replanning_capture_weight(end, future, capture)
    assert np.all(weights[1:] == 0.0)
    base = np.column_stack((np.arange(10, dtype=float), np.full(10, 3.5)))
    heading = np.zeros(10)
    xy, planned_heading, audit = target_capture_path(
        base, heading, np.asarray([0.0, 4.0]), 0.2, end, future, capture
    )
    assert np.array_equal(xy[0], [0.0, 4.0])
    assert planned_heading[0] == 0.2
    assert np.allclose(xy[1:, 1], 3.5)
    assert audit["state1_plus_zero_after_capture_end"] is True


def test_pre_divergence_behavior_is_identical_and_no_scenario_lookup_exists():
    params = parameters()
    time = np.asarray([0.0, 0.5, 1.0, 1.099999])
    assert np.array_equal(
        behavior_progress(time, ARM_BASELINE, params),
        behavior_progress(time, ARM_TREATMENT, params),
    )
    assert "scenario_token" not in json.dumps(params)
    assert "log_id" not in json.dumps(params)


def test_fresh_roster_and_frozen_limits():
    roster = read("r2_bh_hlc_arch_dev_roster_v1.0.json")
    exclusion = read("r2_bh_hlc_arch_permanent_exclusion_ledger_v1.0.json")
    contract = read("r2_bh_hlc_architecture_contract_v2.0.json")
    assert roster["count"] == 8
    assert len({row["scenario_token"] for row in roster["entries"]}) == 8
    assert all(row["PERMANENT_ENGINEERING_ONLY"] for row in roster["entries"])
    assert exclusion["counts"]["effective_unique_identities"] == 109
    assert contract["maximum_rounds"] == 3
    assert contract["R2B_round5"] is False


def test_three_round_stop_and_nonconverged_disposition():
    summary = read("r2_bh_hlc_arch_round_summary_v1.json")
    assert summary["status"] == "R2_BH_DEVELOPMENT_NOT_CONVERGED"
    assert summary["rounds_executed"] == 3
    assert summary["round4_executed"] is False
    assert summary["actual_HLC_engineering_runs"] == 48
    assert summary["final_counts"] == {
        "pairs": 8, "mechanism_pass": 0, "endpoint_pass": 0,
        "endpoint_offset_pass": 0, "heading_pass": 8,
        "lateral_velocity_pass": 8, "route_progress_pass": 7,
        "F_match_pass": 8, "engineering_pass": 8, "safety_pass": 4,
    }
    assert summary["selected_HLC_V2_candidate_frozen"] is False
    assert summary["complete_G_R2_candidate_frozen"] is False
    assert not (R2 / "r2_bh_selected_hlc_generator_parameters_v2.0.json").exists()
    assert not (R2 / "r2_bh_complete_g_r2_development_candidate_manifest_v1.0.json").exists()


def test_capture_command_and_data_firewall_closure():
    capture = read("r2_bh_hlc_target_capture_audit_v1.json")
    firewall = read("r2_bh_generator_data_firewall_audit_v1.json")
    manifest = read("r2_bh_hlc_arch_development_binding_manifest_v1.0.json")
    assert capture["final_round_capture_end_zero_state1_command_16_of_16"] is True
    assert capture["scientific_progress_measurement_changed"] is False
    assert firewall["overlap_historical_R1_R2A_R2B"] == 0
    assert firewall["R2B_HLC_old_identities_rerun"] == 0
    assert firewall["TSB_new_simulation_calls"] == 0
    assert firewall["scientific_thresholds_modified"] is False
    assert manifest["actual_HLC_engineering_runner_run_calls"] == 48
    assert manifest["scientific_simulation_calls"] == 0
    assert manifest["R2C_identities_selected"] is False
    assert manifest["RBR_started"] is False
