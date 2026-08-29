import json
from pathlib import Path

import numpy as np
import pytest

from tools.r1_closed_loop_benchmark_v2 import (
    build_hlc_native_geometry,
    build_native_route_reference,
    build_tsb_route_aligned,
    exact_realized_window,
    first_state_error,
    hlc_map_applicability,
    hlc_realized_primary_measurement,
    prospective_primary_f_match,
    sample_native_reference_no_extrapolation,
    tsb_minimum_initial_speed_evidence,
)
from tools.r1_official_technical_smoke_planner import HLC_TREATMENT, TSB_TREATMENT


ROOT = Path(__file__).resolve().parents[1]


def ego(x=0.0, y=0.0, heading=0.0, speed=4.0, time_us=1_000_000):
    return {"rear_axle": {"x": x, "y": y, "heading": heading}, "speed_mps": speed, "time_us": time_us}


def curved_route():
    theta = np.linspace(0.0, 0.9, 500)
    return np.column_stack((80.0 * np.sin(theta), 80.0 * (1.0 - np.cos(theta))))


def test_realized_primary_requires_exact_physical_grid():
    rows = [{"iteration_index": i, "current_ego": ego(x=i * 0.4, time_us=1_000_000 + i * 100_000)} for i in range(80)]
    assert len(exact_realized_window(rows)) == 80
    rows[9]["current_ego"]["time_us"] += 1
    with pytest.raises(ValueError, match="NOT_EVALUABLE_TEMPORAL_GRID"):
        exact_realized_window(rows)


def test_historical_b2_1_trace_is_readonly_temporal_diagnostic():
    path = sorted((ROOT / "outputs/r1_official_compliant_technical_smoke_v1_1/runs").glob("*/trace/planner_trace.jsonl"))[0]
    rows = []
    with path.open() as handle:
        for i, line in enumerate(handle):
            if i == 80:
                break
            rows.append(json.loads(line))
    with pytest.raises(ValueError, match="NOT_EVALUABLE_TEMPORAL_GRID"):
        exact_realized_window(rows)


def test_tsb_analytical_floor_and_absolute_clock():
    evidence = tsb_minimum_initial_speed_evidence(step_mps=0.001)
    assert evidence["match"]
    assert evidence["proposed_initial_speed_floor_mps"] == 2.0
    route = curved_route()
    current = ego(speed=4.0)
    trajectory = build_tsb_route_aligned(current, 1.2, route, 0.0, TSB_TREATMENT)
    assert first_state_error(current, trajectory[0])["exact_construction_identity"]
    assert trajectory[1]["speed_mps"] < trajectory[0]["speed_mps"]
    assert abs(trajectory[10]["rear_axle"]["heading"] - trajectory[1]["rear_axle"]["heading"]) > 0


def test_hlc_native_no_extrapolation_and_current_ego_anchor():
    source = curved_route()
    target = source + np.asarray([0.0, 3.2])
    current = ego(speed=4.0)
    trajectory = build_hlc_native_geometry(current, 0.0, source, target, 0.0, 0.0, HLC_TREATMENT)
    assert first_state_error(current, trajectory[0])["exact_construction_identity"]
    with pytest.raises(ValueError, match="NO_EXTRAPOLATION"):
        sample_native_reference_no_extrapolation(source, [0.0, 10_000.0])


def test_hlc_primary_f_match_excludes_heading():
    baseline = {"mean_speed": 5.0, "end_minus_start_speed": 0.0, "path_length": 40.0, "mean_abs_accel": 0.0, "heading_change_abs_total": 0.0}
    treatment = dict(baseline, heading_change_abs_total=6.28)
    result = prospective_primary_f_match(baseline, treatment, "R-HLC")
    assert result["pass"]
    assert "heading_change_abs_total" not in result["primary_features"]
    assert result["heading_change_abs_total"] == "SECONDARY_MECHANISM_PROXIMAL_AUDIT"


def test_hlc_map_applicability_uses_native_coverage_and_frozen_limits_only():
    source = np.column_stack((np.arange(100.0), np.zeros(100)))
    target = source + [0.0, 3.2]
    result = hlc_map_applicability(
        source_lane_id="s", target_lane_id="t", target_is_native_adjacent=True,
        source_roadblock_id="rb", target_roadblock_id="rb", route_roadblock_ids=["rb"],
        source_reference_xy=source, target_reference_xy=target,
        source_current_arc_m=0.0, target_current_arc_m=0.0, required_forward_m=79.0,
        engineering={"max_abs_lateral_accel_mps2": 6.0, "max_abs_yaw_rate_radps": 1.0, "max_abs_curvature_inv_m": 0.5},
    )
    assert result["pass"]
    assert result["new_numeric_geometry_threshold_used"] is False


def test_old_roster_native_route_builder_readonly_classification():
    map_root = ROOT.parent / "nuplan/dataset/maps"
    if not map_root.is_dir():
        pytest.skip("local official map assets unavailable")
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    roster = json.loads((ROOT / "docs/stageR/r1/r1_official_technical_smoke_roster_v1.0.json").read_text())["entries"]
    cache, counts, failures = {}, {"R-HLC": 0, "R-TSB": 0}, []
    for entry in roster:
        if entry["map_name"] not in cache:
            cache[entry["map_name"]] = get_maps_api(str(map_root), "nuplan-maps-v1.0", entry["map_name"])
        initial = entry["initial_state"]
        current = ego(initial["initial_x"], initial["initial_y"], initial["initial_heading"], initial["initial_speed_mps"], initial["initial_time_us"])
        try:
            build_native_route_reference(cache[entry["map_name"]], entry["route_roadblock_ids"], current, max(initial["initial_speed_mps"], 0.2) * 7.9)
            counts[entry["family"]] += 1
        except ValueError:
            failures.append(entry["scenario_token"])
    assert counts == {"R-HLC": 12, "R-TSB": 11}
    assert failures == ["f464a2a451d85356"]


def test_hlc_realized_primary_measurement_uses_actual_states():
    source = np.column_stack((np.arange(100.0), np.zeros(100)))
    target = source + [0.0, 3.2]
    states = []
    for i in range(80):
        progress = min(1.0, i / 30.0)
        states.append(ego(x=i * 0.5, y=3.2 * progress, heading=0.0, speed=5.0, time_us=1_000_000 + i * 100_000))
    result = hlc_realized_primary_measurement(states, source, target)
    assert result["measurement_source"] == "REALIZED_CLOSED_LOOP_EGO_TRAJECTORY"
    assert result["endpoint"]["complete_target_lane_transition"]
