from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from tools.r1_b2_6_official_dispatch_preflight import run_official_dispatch_preflight_v1
from tools.r1_closed_loop_benchmark_v2_1 import HLC_PRIMARY_F_MATCH_CALIPERS, TSB_PRIMARY_F_MATCH_CALIPERS, calculate_hlc_option_b_v2_timestamp_aware, timestamp_aware_hlc_engineering, tsb_applicability_v1
from tools.r1_hlc_dynamic_clearance_v1_1 import R1HLCDynamicClearanceConfigV1_1
from tools.r1_hlc_measurement_conformance_v1 import hlc_realized_lane_transition_progress_v1_0, native_projection_v1_0, terminal_native_route_progress_v1_0
from tools.r1_official_technical_smoke_evaluator_v2_1 import R1OfficialTechnicalSmokeEvaluatorV2_1
from tools.r1_official_technical_smoke_planner_v2_1 import R1OfficialTechnicalSmokePlannerV2_1
from tools.r1_prospective_generator_contract_v2 import HLC_BASELINE, HLC_TREATMENT, TSB_TREATMENT, hlc_progress


ROOT = Path(__file__).resolve().parents[1]


def _normalized_row(row):
    value = dict(row)
    value["initial_state_fingerprint"] = value["initial_state"]["initial_state_fingerprint"]
    value["intended_lane_change_direction"] = str(value.get("direction", "")).upper() or None
    return value


@pytest.fixture(scope="module")
def dispatch_results():
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    rows = json.loads((ROOT / "docs/stageR/r1/r1_official_technical_smoke_roster_v1.0.json").read_text())["entries"]
    selected = [_normalized_row(rows[0]), _normalized_row(rows[12])]
    outputs = {}
    for row in selected:
        map_api = get_maps_api(str(ROOT.parent / "nuplan/dataset/maps"), "nuplan-maps-v1.0", row["map_name"])
        indices = [0, 1, 5, 11, 16, 25, 35, 45, 60] if row["family"] == "R-HLC" else [0, 1, 5, 11, 16, 25, 35]
        outputs[row["family"]] = run_official_dispatch_preflight_v1(future_roster_row=row, smoke_arm=row["arms"][1], map_api=map_api, iteration_indices=indices)
    return outputs


def parallel_lanes(width=3.2, samples=401):
    x = np.linspace(0.0, 50.0, samples)
    return np.column_stack((x, np.zeros(samples))), np.column_stack((x, np.full(samples, width)))


def test_01_planner_v2_1_declares_complete_bound_interface():
    required = {"name", "observation_type", "initialize", "compute_trajectory", "compute_planner_trajectory", "generate_planner_report"}
    assert required <= set(R1OfficialTechnicalSmokePlannerV2_1.__dict__)


def test_02_compute_trajectory_is_not_missing_or_inherited_base_dispatch():
    assert R1OfficialTechnicalSmokePlannerV2_1.compute_trajectory is not AbstractPlanner.compute_trajectory
    assert "compute_planner_trajectory(current_input)" in inspect.getsource(R1OfficialTechnicalSmokePlannerV2_1.compute_trajectory)


def test_03_official_dispatch_returns_interpolated_trajectory(dispatch_results):
    assert all(call["returned_type"] == "InterpolatedTrajectory" for call in dispatch_results["R-HLC"]["calls"])
    assert dispatch_results["R-HLC"]["compute_planner_trajectory_delegate_verified"]


def test_04_absolute_episode_clock_selected_iterations_advance(dispatch_results):
    calls = dispatch_results["R-HLC"]["calls"][:6]
    assert [call["iteration_index"] for call in calls] == [0, 1, 5, 11, 16, 25]
    assert [call["phase"]["absolute_episode_time_s"] for call in calls] == [0.0, 0.1, 0.5, 1.1, 1.6, 2.5]
    assert all(call["phase"]["phase_source"].startswith("CURRENT_INPUT_SIMULATION_ITERATION") for call in calls)


def test_05_tsb_treatment_phases_do_not_restart(dispatch_results):
    by_index = {call["iteration_index"]: call for call in dispatch_results["R-TSB"]["calls"]}
    assert by_index[11]["phase"]["generator_phase_value_at_call"] == -0.9
    assert by_index[16]["phase"]["generator_phase_value_at_call"] == 0.4
    assert by_index[25]["phase"]["generator_phase_value_at_call"] == -0.9
    assert by_index[35]["phase"]["generator_phase_value_at_call"] == 0.0
    assert by_index[11]["state1_speed_delta_mps"] < 0 < by_index[16]["state1_speed_delta_mps"]
    assert by_index[25]["state1_speed_delta_mps"] < 0


def test_06_hlc_treatment_advance_hold_retreat_recommit_persist(dispatch_results):
    phase = {call["iteration_index"]: call["phase"]["generator_phase_value_at_call"] for call in dispatch_results["R-HLC"]["calls"]}
    assert phase[11] == 0.0 and 0.0 < phase[16] < phase[25]
    assert phase[25] == 0.38 and phase[35] < phase[25]
    assert phase[45] > 0.22 and phase[60] > phase[45]


def test_07_source_lane_forward_only_progress_is_zero():
    source, target = parallel_lanes()
    ego = source[np.linspace(0, len(source) - 1, 80).astype(int)]
    result = hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)
    assert np.max(np.abs(result["raw_progress"])) == 0.0


def test_08_target_lane_forward_only_progress_is_one():
    source, target = parallel_lanes()
    ego = target[np.linspace(0, len(target) - 1, 80).astype(int)]
    result = hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)
    assert np.array_equal(result["raw_progress"], np.ones(80))


def test_09_linear_transition_progress_is_monotonic_zero_to_one():
    source, target = parallel_lanes(); x = np.linspace(0, 39.5, 80); p = np.linspace(0, 1, 80)
    ego = np.column_stack((x, 3.2 * p))
    result = hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)
    assert result["raw_progress"][0] == 0.0 and result["raw_progress"][-1] == 1.0 and np.all(np.diff(result["raw_progress"]) >= 0)


def test_10_advance_retreat_recommit_morphology_is_preserved():
    source, target = parallel_lanes(); time = np.arange(80) * 0.1; p = hlc_progress(time, HLC_TREATMENT)
    ego = np.column_stack((time * 5.0, 3.2 * p))
    raw = np.asarray(hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)["raw_progress"])
    assert np.max(raw[11:31]) > np.min(raw[31:42]) and raw[-1] == 1.0
    assert np.any(np.diff(raw[25:42]) < 0) and np.any(np.diff(raw[41:]) > 0)


def test_11_curved_source_forward_motion_is_not_transition():
    theta = np.linspace(0.0, 1.2, 321)
    source = np.column_stack((30.0 * np.sin(theta), -30.0 * np.cos(theta)))
    target = np.column_stack((33.2 * np.sin(theta), -33.2 * np.cos(theta)))
    ego = source[np.linspace(0, len(source) - 1, 80).astype(int)]
    raw = np.asarray(hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)["raw_progress"])
    assert np.max(np.abs(raw)) < 1e-12


def test_12_different_sampling_density_is_geometry_consistent():
    source, _ = parallel_lanes(samples=503); _, target = parallel_lanes(samples=37)
    x = np.linspace(0.0, 40.0, 80); ego = np.column_stack((x, np.full(80, 0.8)))
    raw = np.asarray(hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)["raw_progress"])
    assert np.max(np.abs(raw - 0.25)) < 1e-12


def test_13_projection_ambiguity_fails_closed():
    crossing = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="AMBIGUOUS_FAIL_CLOSED"):
        native_projection_v1_0(crossing, [0.5, 0.5], label="CROSSING")


def test_14_path_length_difference_is_not_route_progress_surrogate():
    route = np.column_stack((np.linspace(0, 50, 101), np.zeros(101)))
    audit = terminal_native_route_progress_v1_0(baseline_terminal_xy=[40.0, 0.0], treatment_terminal_xy=[40.0, 3.2], native_route_reference_xy=route, route_reference_source="SYNTHETIC")
    assert audit["paired_route_progress_delta_m"] == 0.0 and not audit["path_length_surrogate_used"]


def test_15_native_terminal_route_projection_is_used():
    route = np.column_stack((np.linspace(0, 50, 101), np.zeros(101)))
    audit = terminal_native_route_progress_v1_0(baseline_terminal_xy=[38.75, 0.0], treatment_terminal_xy=[40.0, 3.2], native_route_reference_xy=route, route_reference_source="FROZEN_NATIVE_ROUTE")
    assert audit["baseline_terminal_route_s_m"] == 38.75 and audit["treatment_terminal_route_s_m"] == 40.0
    assert audit["paired_route_progress_delta_m"] == 1.25 and audit["pass"]


def test_16_longitudinal_only_motion_cannot_trigger_hlc_commitment():
    source, target = parallel_lanes(); x = np.linspace(0, 39.5, 80); ego = np.column_stack((x, np.zeros(80)))
    progress = hlc_realized_lane_transition_progress_v1_0(source_reference_xy=source, target_reference_xy=target, realized_ego_xy=ego)
    mechanism = calculate_hlc_option_b_v2_timestamp_aware(np.arange(80) * 0.1, progress["clipped_progress_for_frozen_mechanism"], np.full(80, 5.0))
    assert mechanism["status"] == "NO_DEPARTURE" and mechanism["commit_latency_s"] is None


def _synthetic_rows(arm):
    time = np.arange(80) * 0.1; p = hlc_progress(time, arm); xy = np.column_stack((time * 5.0, 3.2 * p))
    return [{"iteration_index": index, "current_ego": {"rear_axle": {"x": float(point[0]), "y": float(point[1]), "heading": 0.0}, "speed_mps": 5.0, "time_us": 1_000_000 + index * 100_000}} for index, point in enumerate(xy)]


def test_17_synthetic_evaluator_pair_runs_full_native_pipeline():
    source, target = parallel_lanes(); context = {"pre_context_raw_hash": "same", "canonical_context_json_hash": "same"}
    result = R1OfficialTechnicalSmokeEvaluatorV2_1().evaluate_pair(family="R-HLC", baseline_trace_rows=_synthetic_rows(HLC_BASELINE), treatment_trace_rows=_synthetic_rows(HLC_TREATMENT), baseline_context=context, treatment_context=context, official_safety_canonical_payload={"collision": 0, "drivable": True}, source_reference_xy=source, target_reference_xy=target, native_route_reference_xy=source, native_route_reference_source="SYNTHETIC_FROZEN_NATIVE_ROUTE")
    assert result["mechanism"]["pass"] and result["f_match"]["pass"]
    assert result["endpoint"]["baseline"]["pass"] and result["endpoint"]["treatment"]["pass"]
    assert result["native_route_progress"]["paired_route_progress_delta_m"] == 0.0
    assert result["pipeline"][0] == "REALIZED_CURRENT_EGO_ITERATIONS_0_79"


def test_18_planned_first_cannot_become_primary():
    rows = [{"primary_measurement_source": "PLANNED", "planner_output_trajectory": []}]
    with pytest.raises(ValueError, match="PLANNED_TRAJECTORY_PRIMARY_FORBIDDEN"):
        R1OfficialTechnicalSmokeEvaluatorV2_1().evaluate_pair(family="R-TSB", baseline_trace_rows=rows, treatment_trace_rows=rows, baseline_context={}, treatment_context={}, official_safety_canonical_payload={})


def test_19_frozen_contract_numerics_are_unchanged():
    assert HLC_PRIMARY_F_MATCH_CALIPERS == {"mean_speed": 0.708203939, "end_minus_start_speed": 0.978755681, "path_length": 5.38423459}
    assert TSB_PRIMARY_F_MATCH_CALIPERS == {**HLC_PRIMARY_F_MATCH_CALIPERS, "mean_abs_accel": 0.11777666}
    assert R1HLCDynamicClearanceConfigV1_1() == R1HLCDynamicClearanceConfigV1_1(8.0, 0.1, 0.25, 3.0, 0.5)
    assert tsb_applicability_v1()["TSB_MECHANISM_APPLICABILITY_INITIAL_SPEED_FLOOR_MPS"] == 2.0


def test_20_engineering_limits_remain_frozen():
    result = timestamp_aware_hlc_engineering([row["current_ego"] for row in _synthetic_rows(HLC_TREATMENT)])
    assert result["frozen_limits"] == {"lateral_accel_mps2_max": 6.0, "yaw_rate_radps_max": 1.0, "curvature_inv_m_max": 0.5}


def test_21_v2_1_future_path_does_not_import_historical_b2_1_planner():
    paths = [ROOT / "tools/r1_official_technical_smoke_planner_v2_1.py", ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py", ROOT / "tools/r1_hlc_measurement_conformance_v1.py", ROOT / "tools/r1_b2_6_official_dispatch_preflight.py"]
    assert all("r1_official_technical_smoke_planner import" not in path.read_text() for path in paths)
