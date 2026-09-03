import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.r2_b_controller_aware_generator_v1 import (
    ARM_BASELINE,
    ARM_TREATMENT,
    hlc_controller_aware_progress,
    tsb_controller_aware_acceleration,
    validate_global_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"


def read(name: str):
    return json.loads((R2 / name).read_text())


def test_r2_b_fresh_identity_firewall_and_permanent_disposition():
    roster = read("r2_b_generator_calibration_roster_v1.0.json")
    firewall = read("r2_b_generator_data_firewall_audit_v1.json")
    assert roster["counts"] == {"R-HLC": 8, "R-TSB": 8, "total": 16}
    assert len({row["scenario_token"] for row in roster["entries"]}) == 16
    assert all(row["PERMANENT_ENGINEERING_ONLY"] for row in roster["entries"])
    assert all(row["R2C_VALIDATION_USE_FORBIDDEN"] for row in roster["entries"])
    assert firewall["overlap_with_R1_official"] == 0
    assert firewall["overlap_with_R2_A"] == 0
    assert firewall["overlap_with_pre_R2_B_blacklist"] == 0
    assert firewall["R2C_identities_selected"] is False


def test_generator_is_global_deterministic_and_preserves_precontext():
    hlc = read("r2_b_hlc_calibration_parameter_space_v1.0.json")["round0"]
    tsb = read("r2_b_tsb_calibration_parameter_space_v1.0.json")["round0"]
    validate_global_parameters("R-HLC", hlc)
    validate_global_parameters("R-TSB", tsb)
    with pytest.raises(ValueError, match="SCENARIO_SPECIFIC"):
        validate_global_parameters("R-HLC", {**hlc, "scenario_token": "forbidden"})
    times = np.asarray([0.0, 0.9, 1.0, 1.099999])
    assert np.array_equal(
        hlc_controller_aware_progress(times, ARM_BASELINE, hlc),
        hlc_controller_aware_progress(times, ARM_TREATMENT, hlc),
    )
    baseline = [tsb_controller_aware_acceleration(value, ARM_BASELINE, tsb) for value in times]
    treatment = [tsb_controller_aware_acceleration(value, ARM_TREATMENT, tsb) for value in times]
    assert baseline == treatment


def test_frozen_round_limit_and_convergence_disposition():
    summary = read("r2_b_generator_calibration_round_summary_v1.json")
    assert summary["HLC_rounds_executed"] == 4
    assert summary["TSB_rounds_executed"] == 1
    assert summary["actual_DEV_engineering_runs"] == 80
    assert summary["HLC_final"]["counts"] == {
        "pairs": 8, "mechanism_pass": 6, "F_match_pass": 8,
        "safety_pass": 8, "endpoint_pass": 0, "engineering_pass": 8,
    }
    assert summary["TSB_final"]["counts"]["mechanism_pass"] == 8
    assert summary["TSB_final"]["counts"]["F_match_pass"] == 8
    assert summary["complete_G_R2_candidate_frozen"] is False
    assert summary["fifth_round_executed"] is False
    assert not (R2 / "r2_b_selected_generator_parameters_v1.0.json").exists()


def test_final_manifest_and_protected_asset_closure():
    manifest = read("r2_b_generator_binding_manifest_v1.0.json")
    assert manifest["status"] == "R2_B_DEVELOPMENT_NOT_CONVERGED"
    assert manifest["actual_engineering_runner_run_calls"] == 80
    assert manifest["scientific_simulation_calls"] == 0
    assert manifest["R2C_identities_selected"] is False
    assert manifest["RBR_started"] is False
    assert manifest["raw_DEV_output_provenance"]["run_count"] == 80
    assert hashlib.sha256(PROTECTED.read_bytes()).hexdigest() == (
        "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
    )
