import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import (
    ARM_BASELINE,
    ARM_TREATMENT,
    morphology_progress,
    validate_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
BJ_A_BOUND_COMMIT = "5c991b32c1d26dc5887016e3760a58b0fec64aeb"
ANALYTIC = json.loads((R2 / "r2_bj_a_hlc_morphology_analytic_feasibility_audit_v1.0.json").read_text())
ENVELOPE = json.loads((R2 / "r2_bj_a_expanded_zero_run_feasibility_envelope_v1.0.json").read_text())
SPACE = json.loads((R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json").read_text())


def test_intrinsic_redesign_obeys_acceleration_limit_but_composite_fails_closed():
    revised = ANALYTIC["R2_BJ_A_revised_global_morphology"]
    assert revised["intrinsic_limit_pass"] is True
    assert max(row["lane_separation_scaled_max_lateral_acceleration_mps2"] for row in revised["phases"]) <= 6.0
    assert ANALYTIC["R2_BI_frozen_morphology"]["intrinsic_limit_pass"] is False
    assert ENVELOPE["status"] == "R2_BJ_A_OFFLINE_ARCHITECTURE_NOT_READY"
    assert ENVELOPE["all_mandatory_cases_pass"] is False
    assert ENVELOPE["fail_count"] > 0


def test_divergence_boundary_is_c2_without_positive_lag_shift():
    parameters = SPACE["global_parameters"]
    validate_parameters(parameters)
    assert parameters["morphology"]["lag_precompensation_s"] == 0.0
    at = np.asarray([1.1])
    assert morphology_progress(at, ARM_BASELINE, parameters["morphology"])[0] == 0.0
    assert morphology_progress(at, ARM_TREATMENT, parameters["morphology"])[0] == 0.0
    boundary = ANALYTIC["R2_BJ_A_revised_global_morphology"]["divergence_boundary_C2"]
    assert boundary["pass"] is True
    assert boundary["baseline_minus_treatment_first_derivative"] == 0.0
    assert boundary["baseline_minus_treatment_second_derivative"] == 0.0


def test_phase_morphology_and_primary80_settling_are_retained():
    morphology = SPACE["global_parameters"]["morphology"]
    times = np.asarray([1.1, 2.8, 2.9, 4.3, 7.4, 7.9])
    progress = morphology_progress(times, ARM_TREATMENT, morphology)
    assert progress == pytest.approx([0.0, 0.20, 0.20, 0.09, 1.0, 1.0])
    assert ANALYTIC["R2_BJ_A_revised_global_morphology"]["remaining_Primary80_settling_time_s"] == pytest.approx(0.5)


def test_expanded_full_states_envelope_and_lqr_audit_are_zero_run():
    assert ENVELOPE["method"] == "FULL_PLANNER__STATES_OFFLINE_ONLY"
    assert ENVELOPE["case_count"] == 3296
    assert ENVELOPE["pass_count"] + ENVELOPE["fail_count"] == ENVELOPE["case_count"]
    assert ENVELOPE["common_to_treatment_boundary"]["pass"] is True
    lqr = ENVELOPE["LQR_controller_observability"]
    assert lqr["zero_false_steering"] is True
    assert lqr["positive_residual_correct_sign_nonzero"] is True
    assert lqr["negative_residual_correct_sign_nonzero"] is True
    assert ENVELOPE["runner_run_calls"] == ENVELOPE["simulation_calls"] == 0
    assert ENVELOPE["TSB_simulation_calls"] == 0


def test_native_curvature_is_separate_from_capture_and_explains_raw_edge_failure():
    attribution = ENVELOPE["component_attribution"]
    native = attribution["native_road_curvature"]
    assert native["raw_speed_max_native_lateral_acceleration_mps2"] > 6.0
    assert native["raw_cartesian_envelope_limit_pass"] is False
    assert attribution["morphology_intrinsic_acceleration"]["limit_pass"] is True
    assert attribution["composite_failure_attributed_entirely_to_capture"] is False


def test_r2bi_all_eight_identities_are_permanently_firewalled():
    ledger = json.loads((R2 / "r2_bj_a_r2bi_outcome_exposure_ledger_v1.0.json").read_text())
    assert ledger["identity_count"] == 8
    assert ledger["outcome_exposed_count"] == 1
    assert ledger["frozen_unrun_count"] == 7
    assert all(row["PERMANENT_ENGINEERING_ONLY"] for row in ledger["entries"])
    assert all(row["R2C_USE_FORBIDDEN"] and row["CONFIRMATORY_USE_FORBIDDEN"] for row in ledger["entries"])
    assert all(row["RBR_USE_FORBIDDEN"] and row["R2_BI_RERUN_FORBIDDEN"] for row in ledger["entries"])


def test_firewall_request_and_manifest_are_fail_closed():
    firewall = json.loads((R2 / "r2_bj_a_data_firewall_audit_v1.0.json").read_text())
    manifest = json.loads((R2 / "r2_bj_a_component_sha_binding_manifest_v1.0.json").read_text())
    request = (R2 / "R2_BJ_A_R2BJ_B_Engineering_Execution_Readiness_Request_v0.1.md").read_text()
    assert firewall["status"] == "PASS"
    assert firewall["runner_run_calls"] == firewall["engineering_simulation_calls"] == 0
    assert firewall["R2C_started"] is firewall["confirmatory_smoke_started"] is firewall["RBR_started"] is False
    assert "REQUEST_WITHHELD" in request
    assert manifest["readiness_request_issued"] is False
    assert manifest["component_SHA_closure"] == "PASS"
    for row in manifest["components"]:
        # BJ-A is an immutable historical manifest.  Living documents such as
        # QUICK_REFERENCE.md may legitimately change in later phases, so every
        # historical component is verified against the tree that BJ-A bound.
        historical = subprocess.check_output(
            ["git", "show", f"{BJ_A_BOUND_COMMIT}:{row['path']}"], cwd=ROOT,
        )
        assert hashlib.sha256(historical).hexdigest() == row["sha256"]


def test_scenario_specific_parameter_is_rejected():
    bad = json.loads(json.dumps(SPACE["global_parameters"]))
    bad["capture"]["scenario_token"] = "forbidden"
    with pytest.raises(ValueError, match="SCENARIO_SPECIFIC_PARAMETER_FORBIDDEN"):
        validate_parameters(bad)
