from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tools.r1_b2_5_zero_rollout_preflight import launch_official_simulation, validate_selector_inputs_outcome_blind
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1, build_native_route_reference_v1_1, build_tsb_route_aligned_v1_1, current_ego_construction_parity_audit, hlc_endpoint_v1_1_timestamp_aware
from tools.r1_closed_loop_context_adapter_v2_1 import stage5d_slot_identity_v2_1
from tools.r1_hlc_dynamic_clearance_v1_1 import R1HLCDynamicClearanceConfigV1_1, evaluate_r1_hlc_dynamic_clearance_v1_1
from tools.r1_official_ego_vehicle_binding_v1 import official_ego_vehicle_binding_v1
from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1
from tools.r1_official_technical_smoke_evaluator_v2 import R1OfficialTechnicalSmokeEvaluatorV2
from tools.r1_prospective_generator_contract_v2 import HLC_TREATMENT, TSB_BASELINE, hlc_progress
from tools.stage5d_context_core import assign_stage5d_slots


def ego(x=0.0, y=0.0, heading=0.0, speed=4.0, time_us=1_000_000):
    return {"rear_axle": {"x": x, "y": y, "heading": heading}, "speed_mps": speed, "time_us": time_us}


def straight_states(dt_last_us=100_000, lateral_step=0.0):
    rows = []
    for index in range(80):
        time_us = 1_000_000 + index * 100_000 + (dt_last_us - 100_000 if index == 79 else 0)
        rows.append(ego(index * 0.4, lateral_step * index, 0.0, 4.0, time_us))
    return rows


def clearance_kwargs(observation, tracks=None, vehicle=None):
    x = np.arange(80, dtype=np.float64) * 0.5
    path = np.column_stack((x, np.zeros(80)))
    return {"baseline_xy": path, "treatment_xy": path.copy(), "official_runtime_vehicle_parameters": {"length_m": 5.176, "width_m": 2.297} if vehicle is None else vehicle, "original_replay_tracks": tracks or {}, "official_replay_observation_timestamps_s": observation}


def test_01_empty_tracks_incomplete_horizon_is_not_eligible():
    result = evaluate_r1_hlc_dynamic_clearance_v1_1(**clearance_kwargs(np.arange(40) * 0.1))
    assert result["status"] == "NOT_ELIGIBLE"


def test_02_empty_tracks_complete_horizon_is_dynamic_clear_no_actors():
    result = evaluate_r1_hlc_dynamic_clearance_v1_1(**clearance_kwargs(np.arange(80) * 0.1))
    assert result["status"] == "DYNAMIC_CLEAR_NO_ACTORS" and result["observation_horizon"]["complete"]


def test_03_interior_observation_gap_fails_closed():
    times = np.r_[np.arange(20) * 0.1, np.arange(23, 80) * 0.1]
    assert evaluate_r1_hlc_dynamic_clearance_v1_1(**clearance_kwargs(times))["status"] == "NOT_ELIGIBLE"


def test_04_clearance_numerics_cannot_be_overridden():
    with pytest.raises(ValueError, match="may not be overridden"):
        evaluate_r1_hlc_dynamic_clearance_v1_1(**clearance_kwargs(np.arange(80) * 0.1), config=R1HLCDynamicClearanceConfigV1_1(longitudinal_buffer_m=3.1))


def test_05_oriented_both_arm_footprint_catches_yaw_adversary():
    x = np.arange(80, dtype=np.float64) * 0.5
    baseline = np.column_stack((x, np.zeros(80)))
    treatment = baseline.copy(); treatment[39, 1] = -1.0; treatment[41, 1] = 1.0
    track = {"yaw_case": {"time_s": [4.0], "states": [[x[40], 3.2, 0.5, 0.5, 0.0]]}}
    result = evaluate_r1_hlc_dynamic_clearance_v1_1(baseline_xy=baseline, treatment_xy=treatment, official_runtime_vehicle_parameters={"length_m": 5.176, "width_m": 2.297}, original_replay_tracks=track, official_replay_observation_timestamps_s=np.arange(80) * 0.1)
    assert result["status"] == "DYNAMIC_CLEARANCE_FAIL" and result["first_conflict"]["iteration_index"] == 40


def test_06_missing_official_ego_footprint_is_not_eligible():
    result = evaluate_r1_hlc_dynamic_clearance_v1_1(**clearance_kwargs(np.arange(80) * 0.1, vehicle={}))
    assert result["status"] == "NOT_ELIGIBLE" and result["reason"] == "OFFICIAL_EGO_FOOTPRINT_MISSING"


def test_07_tsb_state1_exact_construction_parity_and_tamper_failure():
    reference = np.column_stack((np.linspace(0, 50, 501), np.zeros(501)))
    args = (ego(), 0.0, reference, 0.0, TSB_BASELINE)
    actual = build_tsb_route_aligned_v1_1(*args)
    assert current_ego_construction_parity_audit(build_tsb_route_aligned_v1_1, args, {}, actual)["pass"]
    actual[1]["rear_axle"]["x"] += 1e-12
    assert not current_ego_construction_parity_audit(build_tsb_route_aligned_v1_1, args, {}, actual)["pass"]


def test_08_hlc_state1_exact_construction_parity():
    source = np.column_stack((np.linspace(0, 50, 501), np.zeros(501)))
    target = source + [0.0, 4.0]
    args = (ego(), 0.0, source, target, 0.0, 0.0, HLC_TREATMENT)
    actual = build_hlc_native_geometry_v1_1(*args)
    assert current_ego_construction_parity_audit(build_hlc_native_geometry_v1_1, args, {}, actual)["status"] == "CONSTRUCTION_PARITY_PASS"


class Baseline:
    def __init__(self, coords):
        self._start_x = float(coords[0][0])
        self.linestring = SimpleNamespace(coords=coords, length=float(sum(math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) for i in range(len(coords) - 1))))
        self.discrete_path = [SimpleNamespace(x=x, y=y) for x, y in coords]
    def get_nearest_pose_from_position(self, point): return SimpleNamespace(x=point.x, y=0.0, heading=0.0)
    def get_nearest_arc_length_from_position(self, point): return float(point.x) - self._start_x


class Edge:
    def __init__(self, edge_id, roadblock, coords): self.id, self._roadblock, self.baseline_path, self.outgoing_edges = edge_id, roadblock, Baseline(coords), []
    def get_roadblock_id(self): return self._roadblock


def test_09_repeated_roadblock_builder_calls_second_a_and_reaches_c():
    a1, b, a2, c = Edge("a1", "A", [(0, 0), (1, 0)]), Edge("b", "B", [(1, 0), (2, 0)]), Edge("a2", "A", [(2, 0), (3, 0)]), Edge("c", "C", [(3, 0), (5, 0)])
    a1.outgoing_edges = [b]; b.outgoing_edges = [a2]; a2.outgoing_edges = [c]
    fake_api = SimpleNamespace(get_all_map_objects=lambda point, layer: [a2] if layer.name == "LANE" else [])
    result = build_native_route_reference_v1_1(fake_api, ["A", "B", "A", "C"], ego(x=2.1), 1.5)
    assert result["route_occurrence_cursor"] == 2 and result["native_edge_ids"] == ["a2", "c"]


def test_10_hlc_endpoint_plus_one_ms_remains_evaluable():
    target = np.column_stack((np.linspace(0, 40, 401), np.zeros(401)))
    result = hlc_endpoint_v1_1_timestamp_aware(straight_states(dt_last_us=101_000), target, paired_route_progress_delta_m=0.0)
    assert result["status"] == "HLC_ENDPOINT_PASS" and result["terminal_actual_dt_s"] == 0.101


def test_11_terminal_lateral_velocity_uses_actual_dt():
    states = straight_states(dt_last_us=200_000)
    states[-1]["rear_axle"]["y"] = 0.02
    target = np.column_stack((np.linspace(0, 40, 401), np.zeros(401)))
    result = hlc_endpoint_v1_1_timestamp_aware(states, target, paired_route_progress_delta_m=0.0)
    assert result["terminal_actual_dt_s"] == 0.2 and result["terminal_lateral_velocity_mps"] == 0.1


def test_12_exact_grid_progress_definition_matches_historical():
    from tools.r1_official_technical_smoke_planner import hlc_progress as historical
    time = np.arange(80) * 0.1
    assert np.array_equal(hlc_progress(time, HLC_TREATMENT), historical(time, HLC_TREATMENT))


def test_13_future_v2_path_has_no_historical_b2_1_planner_import():
    root = Path(__file__).resolve().parents[1]
    files = [root / "tools/r1_official_technical_smoke_planner_v2.py", root / "tools/r1_official_technical_smoke_evaluator_v2.py", root / "tools/r1_closed_loop_benchmark_v2_1.py", root / "tools/r1_prospective_generator_contract_v2.py"]
    assert all("from tools.r1_official_technical_smoke_planner import" not in path.read_text() for path in files)


def test_14_planned_primary_fails_closed():
    rows = [{"primary_measurement_source": "PLANNED", "planner_output_trajectory": []}]
    with pytest.raises(ValueError, match="PLANNED_TRAJECTORY_PRIMARY_FORBIDDEN"):
        R1OfficialTechnicalSmokeEvaluatorV2().evaluate_pair(family="R-TSB", baseline_trace_rows=rows, treatment_trace_rows=rows, baseline_context={}, treatment_context={}, official_safety_canonical_payload={})


def test_15_selector_outcomes_fail_closed():
    with pytest.raises(ValueError, match="SELECTOR_OUTCOME_FIELD_FORBIDDEN"):
        validate_selector_inputs_outcome_blind({"candidate": {"safety_outcome": "PASS"}})


def test_16_simulation_launch_is_hard_blocked():
    with pytest.raises(RuntimeError, match="B2_5_SIMULATION_LAUNCH_HARD_BLOCK"):
        launch_official_simulation()


def test_17_official_runtime_ego_binding_has_no_generic_fallback():
    binding = official_ego_vehicle_binding_v1()
    assert binding["length_m"] == 5.176 and binding["width_m"] == 2.297 and not binding["generic_fallback_used"]


class SlotMap:
    centers = {"C": 0.0, "L": 4.0, "R": -4.0}
    def lane_context(self, ego_xy, route_ids): return {"valid": True, "current_lane_id": "C", "left_lane_id": "L", "right_lane_id": "R", "tangent": [1.0, 0.0], "road_class": "LANE", "source_immediate_adjacent_lane_ids": ["L", "R"], "target_immediate_adjacent_lane_ids": ["C"], "current_immediate_adjacent_lane_ids": ["L", "R"]}
    def project(self, lane_id, xy): return {"arc_m": float(xy[0]), "lateral_offset_m": float(xy[1] - self.centers[lane_id]), "distance_to_lane_m": abs(float(xy[1] - self.centers[lane_id])), "heading": 0.0, "tangent": [1.0, 0.0]}
    def lane_for_actor(self, actor): return str(actor["lane_id"])
    def static_stop_control_ahead(self, ego_xy, route_ids): return False


def test_18_stage5d_authoritative_tie_and_all_slot_identity_parity():
    actors = []
    for token, lane, x in [("tie_first", "C", 10.0), ("tie_second", "C", 10.0), ("lf", "L", 12.0), ("lr", "L", -12.0), ("rf", "R", 13.0), ("rr", "R", -13.0)]:
        actors.append({"track_id": token, "type": "VEHICLE", "lane_id": lane, "x": x, "y": SlotMap.centers[lane], "heading": 0.0, "vx": 3.0, "vy": 0.0})
    frame = {"iteration_index": 0, "time_us": 1_000_000, "ego": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed_mps": 4.0}, "actors": actors}
    with patch("tools.r1_closed_loop_context_adapter_v2_1.assign_stage5d_slots", wraps=assign_stage5d_slots) as authoritative:
        identity = stage5d_slot_identity_v2_1(frame, "R-HLC", "LEFT", ["rb"], SlotMap())
    assert authoritative.call_count == 1 and authoritative.call_args.kwargs["assignment_mode"] == "lane_aware_only"
    assert identity == {"front": "tie_first", "left_front": "lf", "left_rear": "lr", "right_front": "rf", "right_rear": "rr"}


@pytest.mark.parametrize("case,actors,expected", [
    ("empty_slot", [], {"front": "", "left_front": "", "left_rear": "", "right_front": "", "right_rear": ""}),
    ("distance_boundary", [{"track_id": "d", "type": "VEHICLE", "lane_id": "C", "x": 120.0, "y": 0.0, "heading": 0.0, "vx": 3.0, "vy": 0.0}], {"front": "d"}),
    ("lateral_boundary", [{"track_id": "l", "type": "VEHICLE", "lane_id": "C", "x": 10.0, "y": 2.0, "heading": 0.0, "vx": 3.0, "vy": 0.0}], {"front": "l"}),
    ("heading_boundary", [{"track_id": "h", "type": "VEHICLE", "lane_id": "C", "x": 10.0, "y": 0.0, "heading": math.pi / 4.0, "vx": 3.0, "vy": 0.0}], {"front": "h"}),
    ("multi_candidate_ranking", [{"track_id": "far", "type": "VEHICLE", "lane_id": "C", "x": 20.0, "y": 0.0, "heading": 0.0, "vx": 3.0, "vy": 0.0}, {"track_id": "near", "type": "VEHICLE", "lane_id": "C", "x": 10.0, "y": 0.0, "heading": 0.0, "vx": 3.0, "vy": 0.0}], {"front": "near"}),
    ("duplicate_suppression", [{"track_id": "dup", "type": "VEHICLE", "lane_id": "C", "x": 10.0, "y": 0.0, "heading": 0.0, "vx": 3.0, "vy": 0.0}, {"track_id": "dup", "type": "VEHICLE", "lane_id": "L", "x": 10.0, "y": 4.0, "heading": 0.0, "vx": 3.0, "vy": 0.0}], {"front": "", "left_front": "dup"}),
])
def test_19_stage5d_boundary_ranking_empty_duplicate_parity(case, actors, expected):
    frame = {"iteration_index": 0, "time_us": 1_000_000, "ego": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed_mps": 4.0}, "actors": actors}
    with patch("tools.r1_closed_loop_context_adapter_v2_1.assign_stage5d_slots", wraps=assign_stage5d_slots) as authoritative:
        identity = stage5d_slot_identity_v2_1(frame, "R-HLC", "LEFT", ["rb"], SlotMap())
    assert authoritative.call_args.kwargs["assignment_mode"] == "lane_aware_only"
    for slot, token in expected.items():
        assert identity[slot] == token, case


def test_20_official_map_bridge_reads_real_nuplan_api():
    import json
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    root = Path(__file__).resolve().parents[1]
    row = json.loads((root / "docs/stageR/r1/r1_official_technical_smoke_roster_v1.0.json").read_text())["entries"][0]
    api = get_maps_api(str(root.parent / "nuplan/dataset/maps"), "nuplan-maps-v1.0", row["map_name"])
    bridge = R1OfficialMapQueryBridgeV2_1(api)
    initial = row["initial_state"]; xy = (initial["initial_x"], initial["initial_y"])
    context = bridge.lane_context(xy, row["route_roadblock_ids"])
    assert context["valid"] and bridge.project(context["current_lane_id"], xy)["source"] == "OFFICIAL_NUPLAN_BASELINE_PATH"


def test_21_map_bridge_lane_ambiguity_fails_closed():
    bridge = object.__new__(R1OfficialMapQueryBridgeV2_1)
    bridge._map = SimpleNamespace(get_all_map_objects=lambda point, layer: [SimpleNamespace(id=f"{layer.name}-1"), SimpleNamespace(id=f"{layer.name}-2")] if layer.name == "LANE" else [])
    with pytest.raises(ValueError, match="AMBIGUITY_FAIL_CLOSED"):
        bridge.lane_for_actor({"x": 0.0, "y": 0.0})
