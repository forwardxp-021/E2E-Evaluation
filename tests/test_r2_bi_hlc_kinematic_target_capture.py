import json
from pathlib import Path

import numpy as np
import pytest

from tools.r2_bi_hlc_kinematic_target_capture_generator_v3 import (
    ARM_BASELINE,
    ARM_TREATMENT,
    CaptureInfeasible,
    kinematic_target_capture_path,
    validate_parameters,
)
from tools.r2_bi_hlc_kinematic_target_capture_planner_v3 import _states


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
SPACE = json.loads((R2 / "r2_bi_hlc_kinematic_capture_parameter_space_v3.0.json").read_text())
PARAMETERS = SPACE["round0"]
AUDIT = json.loads((R2 / "r2_bi_mandatory_zero_run_entry_gate_audit_v1.json").read_text())


def _straight(residual: float, absolute: float = 2.0):
    speed = 10.0
    future = absolute + np.arange(80) * 0.1
    base = np.column_stack((np.arange(80) * speed * 0.1, np.zeros(80)))
    target = np.column_stack((np.linspace(-10.0, 100.0, 500), np.zeros(500)))
    current = np.asarray([0.0, residual])
    return kinematic_target_capture_path(
        base, target, current, 0.0, speed, absolute, future, PARAMETERS["capture"]
    )


def test_frozen_zero_run_entry_gate_passes_without_simulation():
    assert AUDIT["status"] == "R2_BI_ZERO_RUN_ENTRY_GATES_PASS"
    assert AUDIT["all_mandatory_gates_pass"] is True
    assert AUDIT["scientific_simulation_calls"] == 0
    assert AUDIT["runner_run_calls"] == 0
    assert len(AUDIT["synthetic_cases"]) == 25


@pytest.mark.parametrize("residual", [0.0, 0.25, 0.50, -0.25, -0.50])
def test_state0_pose_continuity_final_xy_heading_and_terminal_capture(residual):
    xy, heading, curvature, audit = _straight(residual)
    assert np.array_equal(xy[0], np.asarray([0.0, residual]))
    assert heading[0] == 0.0
    assert audit["feasibility"]["pass"] is True
    assert audit["pose_consistency"]["max_future_declared_heading_vs_final_xy_tangent_abs_rad"] <= 1e-10
    assert abs(audit["actual_planned_target_frame_offsets_m"][-1]) <= 1e-9
    assert np.max(np.abs(curvature)) <= 0.5


def test_exact_frozen_lqr_shadow_is_controller_observable_and_directional():
    observed = AUDIT["controller_observability"]
    assert observed["zero_residual_no_false_steering"] is True
    assert observed["positive_residual_nonzero_correct_direction"] is True
    assert observed["negative_residual_nonzero_correct_direction"] is True


def test_replanning_boundary_is_continuous():
    boundary = AUDIT["replanning_boundary"]
    assert boundary["pass"] is True
    assert boundary["maximum_overlap_xy_error_m"] <= SPACE["entry_gate_tolerances"]["replanning_overlap_xy_m_max"]


def test_large_unconverged_residual_fails_closed_instead_of_jumping():
    with pytest.raises(CaptureInfeasible, match="INSUFFICIENT_REMAINING_TIME"):
        _straight(4.0, absolute=6.8)
    assert AUDIT["large_unconverged_residual"]["fail_closed"] is True


def test_predivergence_full_trajectory_identity_and_no_scenario_lookup():
    x = np.linspace(-20.0, 200.0, 1000)
    corridor = {
        "source_reference_xy": np.column_stack((x, np.zeros_like(x))).tolist(),
        "target_reference_xy": np.column_stack((x, np.full_like(x, 3.5))).tolist(),
        "source_current_arc_m": 20.0,
        "target_current_arc_m": 20.0,
    }
    current = {"rear_axle": {"x": 0.0, "y": 0.0, "heading": 0.0}, "speed_mps": 10.0, "time_us": 0}
    baseline, _, _ = _states(current, 0.5, corridor, ARM_BASELINE, PARAMETERS, True)
    treatment, _, _ = _states(current, 0.5, corridor, ARM_TREATMENT, PARAMETERS, True)
    assert baseline == treatment
    bad = json.loads(json.dumps(PARAMETERS))
    bad["capture"]["scenario_token"] = "forbidden"
    with pytest.raises(ValueError, match="SCENARIO_SPECIFIC_PARAMETER_FORBIDDEN"):
        validate_parameters(bad)


def test_actual_round0_fail_closed_disposition_forbids_round1_and_selection():
    stop = json.loads((R2 / "r2_bi_hlc_dev_kin_round_0_architecture_stop_audit_v1.json").read_text())
    summary = json.loads((R2 / "r2_bi_hlc_dev_kin_round_summary_v1.json").read_text())
    ledger = json.loads((R2 / "r2_bi_hlc_dev_kin_run_ledger_v1.0.json").read_text())
    assert stop["status"] == "DIRECT_FAIL_CLOSED_KINEMATIC_FEASIBILITY_VIOLATION"
    assert stop["failure_at_first_allowed_arm_divergence"] is True
    lateral = stop["frozen_feasibility_limit_exceeded"]["lateral_acceleration_mps2"]
    assert lateral["observed"] > lateral["frozen_limit"] == 6.0
    assert stop["classification"]["technical_infrastructure_failure"] is False
    assert stop["classification"]["systematic_across_identity_claimed"] is False
    assert summary["status"] == ledger["status"] == "R2_BI_DEVELOPMENT_NOT_CONVERGED"
    assert summary["round0"]["attempted_runs"] == 2
    assert summary["round0"]["remaining_runs_not_started"] == 14
    assert summary["round1_started"] is False
    assert summary["selected_HLC_V3_parameters_created"] is False
    assert summary["complete_G_R2_candidate_created"] is False
    assert ledger["technical_reruns"] == 0
    assert ledger["TSB_simulation_calls"] == 0
    assert ledger["scientific_simulation_calls"] == 0


def test_data_firewall_and_component_closure_are_frozen_without_raw_git_payloads():
    firewall = json.loads((R2 / "r2_bi_hlc_kinematic_data_firewall_audit_v1.json").read_text())
    manifest = json.loads((R2 / "r2_bi_hlc_kinematic_development_binding_manifest_v1.0.json").read_text())
    assert firewall["status"] == "PASS"
    assert firewall["overlap_historical_R1_R2A_R2B_R2BH"] == 0
    assert firewall["R2B_or_R2BH_identity_resimulation_calls"] == 0
    assert firewall["R2BH_raw_used_for_V3_numerical_tuning"] is False
    assert firewall["scientific_thresholds_modified"] is False
    assert manifest["component_SHA_closure"].startswith("PASS_WITH_NO_SELECTED_HLC")
    assert manifest["raw_outputs_committed"] is False
    assert manifest["actual_HLC_engineering_runner_run_calls"] == 2
    assert manifest["TSB_simulation_calls"] == 0
    assert manifest["scientific_simulation_calls"] == 0
    assert not any(row["path"].startswith("outputs/") for row in manifest["components"])
