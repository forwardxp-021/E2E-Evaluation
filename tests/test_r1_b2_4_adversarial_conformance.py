from __future__ import annotations

import math

import numpy as np
import pytest

from tools.r1_closed_loop_benchmark_v2 import prospective_primary_f_match
from tools.r1_closed_loop_benchmark_v2_1 import (
    build_hlc_native_geometry_v1_1,
    build_tsb_route_aligned_v1_1,
    calculate_hlc_option_b_v2_timestamp_aware,
    calculate_tsb_option_a_v2_timestamp_aware,
    exact_realized_window_v1_1,
    resolve_route_occurrence_cursor,
    structural_first_segment_audit,
    tsb_applicability_v1,
)
from tools.r1_closed_loop_context_adapter_v2_1 import build_closed_loop_context_v2_1
from tools.r1_context_mechanism_core import calculate_hlc_option_b, calculate_tsb_option_a
from tools.r1_hlc_dynamic_clearance_v1 import evaluate_r1_hlc_dynamic_clearance_v1
from tools.r1_official_technical_smoke_planner import (
    HLC_TREATMENT,
    TSB_BASELINE,
    TSB_TREATMENT,
    hlc_progress,
    tsb_profile,
)


def ego_state(x: float = 0.0, y: float = 0.0, heading: float = 0.0, speed: float = 5.0, time_us: int = 1_000_000):
    return {"rear_axle": {"x": x, "y": y, "heading": heading}, "speed_mps": speed, "time_us": time_us}


def trace_rows(indices=range(80), jitter_us: int = 0):
    rows = []
    for index in indices:
        time_us = 1_000_000 + index * 100_000 + (jitter_us if index % 2 else 0)
        rows.append({"iteration_index": index, "current_ego": ego_state(x=index * 0.5, time_us=time_us)})
    return rows


class FakeMap:
    centers = {"C": 0.0, "L": 4.0, "R": -4.0, "LL": 8.0}

    def __init__(self, stop=False):
        self.stop = stop

    def lane_context(self, ego_xy, route_ids):
        return {
            "valid": True,
            "current_lane_id": "C",
            "left_lane_id": "L",
            "right_lane_id": "R",
            "tangent": [1.0, 0.0],
            "road_class": "URBAN",
            "source_immediate_adjacent_lane_ids": ["L", "R"],
            "target_immediate_adjacent_lane_ids": ["C", "LL"],
            "current_immediate_adjacent_lane_ids": ["L", "R"],
        }

    def project(self, lane_id, xy):
        center = self.centers[lane_id]
        return {
            "arc_m": float(xy[0]),
            "lateral_offset_m": float(xy[1] - center),
            "distance_to_lane_m": abs(float(xy[1] - center)),
            "heading": 0.0,
            "tangent": [1.0, 0.0],
        }

    def lane_for_actor(self, actor):
        return min(self.centers, key=lambda lane_id: abs(float(actor["y"]) - self.centers[lane_id]))

    def static_stop_control_ahead(self, ego_xy, route_ids):
        return self.stop


def context_frames(*, velocity=True, slow_lead=False, pre_signal=False, anchor_signal=False):
    frames = []
    for index in range(11):
        actor = {
            "track_id": "lead",
            "type": "VEHICLE",
            "lane_id": "C",
            "x": 20.0 + index * 0.3,
            "y": 0.0,
            "heading": 0.0,
        }
        if velocity:
            actor.update(vx=3.0 if slow_lead else 6.0, vy=0.0)
        frames.append({
            "iteration_index": index,
            "time_us": 2_000_000 + index * 100_000 + (1 if index % 2 else 0),
            "ego": {"x": index * 0.5, "y": 0.0, "heading": 0.0, "speed_mps": 5.0},
            "actors": [actor],
            "traffic_lights": ([{"status": "RED", "route_relevant": True}] if (pre_signal and index < 10) or (anchor_signal and index == 10) else []),
        })
    return frames


def build_tsb_context(frames, map_query=None):
    return build_closed_loop_context_v2_1(
        family="R-TSB",
        scenario_token="s",
        map_version="m",
        route_fingerprint="r",
        initial_state_fingerprint="i",
        log_id="l",
        route_roadblock_ids=["rb"],
        frames=frames,
        map_query=map_query or FakeMap(),
    )


def curved_reference(radius=30.0, samples=401):
    angle = np.linspace(0.0, math.pi / 2.0, samples)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))


def test_01_timestamp_plus_one_us_is_accepted():
    assert len(exact_realized_window_v1_1(trace_rows(jitter_us=1))) == 80
    time = np.arange(80) * 0.1 + np.where(np.arange(80) % 2, 1e-6, 0.0)
    progress = hlc_progress(np.arange(80) * 0.1, HLC_TREATMENT)
    assert calculate_hlc_option_b_v2_timestamp_aware(time, progress, np.full(80, 5.0))["status"] != "NOT_EVALUABLE_TEMPORAL_GRID"
    speed = tsb_profile(np.arange(80) * 0.1, TSB_TREATMENT, 5.0)[1]
    assert calculate_tsb_option_a_v2_timestamp_aware(time, speed)["status"] != "NOT_EVALUABLE_TEMPORAL_GRID"


def test_02_non_monotonic_timestamp_fails_closed():
    rows = trace_rows()
    rows[10]["current_ego"]["time_us"] = rows[9]["current_ego"]["time_us"]
    with pytest.raises(ValueError, match="strictly increasing"):
        exact_realized_window_v1_1(rows)


def test_03_missing_iteration_index_fails_closed():
    rows = trace_rows(list(range(79)) + [80])
    with pytest.raises(ValueError, match="consecutive indices"):
        exact_realized_window_v1_1(rows)


def test_04_context_missing_velocity_fails_closed():
    with pytest.raises(ValueError, match="VELOCITY_FAIL_CLOSED"):
        build_tsb_context(context_frames(velocity=False))


def test_05_preframe_signal_does_not_become_anchor_route_red():
    context = build_tsb_context(context_frames(pre_signal=True))
    assert context["context_variables"]["planned_stop_or_hazard_class"] == "NONE_OBSERVED"


def test_06_stable_slow_lead_requires_and_satisfies_eight_of_ten():
    context = build_tsb_context(context_frames(slow_lead=True))
    assert context["context_variables"]["planned_stop_or_hazard_class"] == "OBSERVED_SLOW_LEAD"
    assert context["front_track_audit"]["valid_frame_count"] == 10
    assert context["actual_pre_context_time_us"][1] - context["actual_pre_context_time_us"][0] == 100_001


def test_07_hlc_plus179_minus179_uses_short_geometry_branch():
    source_heading = math.radians(179.0)
    target_heading = math.radians(-179.0)
    length = np.linspace(0.0, 30.0, 301)
    source = np.column_stack((np.cos(source_heading) * length, np.sin(source_heading) * length))
    target = np.column_stack((np.cos(target_heading) * length, 4.0 + np.sin(target_heading) * length))
    current = ego_state(heading=source_heading, speed=2.0)
    states = build_hlc_native_geometry_v1_1(current, 0.0, source, target, 0.0, 0.0, HLC_TREATMENT)
    headings = np.asarray([state["rear_axle"]["heading"] for state in states])
    wrapped_step = (np.diff(headings) + math.pi) % (2 * math.pi) - math.pi
    assert float(np.max(np.abs(wrapped_step))) < math.pi


def test_08_tsb_curved_route_preserves_lateral_and_heading_offset():
    reference = curved_reference()
    current = ego_state(x=29.0, y=0.0, heading=math.pi / 2.0 + 0.1, speed=3.0)
    states = build_tsb_route_aligned_v1_1(current, 0.0, reference, 0.0, TSB_BASELINE)
    assert states[0] == current
    assert structural_first_segment_audit(current, states)["pass"]
    assert np.linalg.norm(np.asarray([states[1]["rear_axle"]["x"], states[1]["rear_axle"]["y"]]) - np.asarray([29.0, 0.0])) < 1.0


def test_09_repeated_roadblock_cursor_uses_native_successor():
    assert resolve_route_occurrence_cursor(["A", "B", "A", "C"], "A", ["C"]) == 2
    with pytest.raises(ValueError, match="not uniquely"):
        resolve_route_occurrence_cursor(["A", "B", "A", "C"], "A", ["B", "C"])


def test_10_trajectory_state_zero_is_exact_current_ego():
    reference = np.column_stack((np.linspace(0, 50, 501), np.zeros(501)))
    current = ego_state(x=0.0, y=1.0, heading=0.05, speed=3.0)
    states = build_tsb_route_aligned_v1_1(current, 0.0, reference, 0.0, TSB_TREATMENT)
    assert states[0] == current


def test_11_state_zero_to_one_has_no_structural_teleport():
    reference = np.column_stack((np.linspace(0, 50, 501), np.zeros(501)))
    current = ego_state(x=0.0, y=0.5, heading=0.02, speed=3.0)
    audit = structural_first_segment_audit(current, build_tsb_route_aligned_v1_1(current, 0.0, reference, 0.0, TSB_BASELINE))
    assert audit["status"] == "STRUCTURAL_FIRST_SEGMENT_CONTINUITY_PASS"
    assert audit["first_segment_distance_m"] < 1.0


def test_12_exact_grid_v1_v2_mechanism_outputs_are_identical():
    time = np.arange(80, dtype=np.float64) * 0.1
    progress = hlc_progress(time, HLC_TREATMENT)
    speed = np.full(80, 5.0)
    assert calculate_hlc_option_b_v2_timestamp_aware(time, progress, speed) == calculate_hlc_option_b(time, progress, speed)
    tsb_speed = tsb_profile(time, TSB_TREATMENT, 5.0)[1]
    assert calculate_tsb_option_a_v2_timestamp_aware(time, tsb_speed) == calculate_tsb_option_a(time, tsb_speed)


def test_13_hlc_heading_is_not_primary_f_match():
    baseline = {"mean_speed": 5.0, "end_minus_start_speed": 0.0, "path_length": 40.0, "heading_change_abs_total": 0.0}
    treatment = dict(baseline, heading_change_abs_total=100.0)
    result = prospective_primary_f_match(baseline, treatment, "R-HLC")
    assert result["pass"]
    assert "heading_change_abs_total" not in result["primary_features"]


def test_14_tsb_floor_analytical_synthetic_parity_and_final_status():
    result = tsb_applicability_v1()
    assert result["parity"]
    assert result["TSB_MECHANISM_APPLICABILITY_INITIAL_SPEED_FLOOR_MPS"] == 2.0
    assert result["status"] == "FROZEN_OWNER_APPROVED_BASELINE_EXECUTION_BOUND"


def test_15_dynamic_clearance_common_envelope_covers_both_arms():
    x = np.linspace(0.0, 40.0, 80)
    baseline = np.column_stack((x, np.zeros(80)))
    treatment = np.column_stack((x, np.linspace(0.0, 4.0, 80)))
    track = {"actor": {"time_s": np.arange(80) * 0.1, "states": np.column_stack((x, np.full(80, 2.0), np.full(80, 4.5), np.full(80, 2.0), np.zeros(80)))}}
    kwargs = {"official_runtime_vehicle_parameters": {"length_m": 4.8, "width_m": 2.0}, "original_replay_tracks": track}
    forward = evaluate_r1_hlc_dynamic_clearance_v1(baseline_xy=baseline, treatment_xy=treatment, **kwargs)
    reverse = evaluate_r1_hlc_dynamic_clearance_v1(baseline_xy=treatment, treatment_xy=baseline, **kwargs)
    assert not forward["pass"] and forward["first_conflict"] == reverse["first_conflict"]
    assert forward["common_envelope"] == "HLC_BASELINE_PLUS_HLC_OPTION_B_TREATMENT"
