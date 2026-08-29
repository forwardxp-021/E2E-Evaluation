import json
from pathlib import Path

import pytest

from tools.r1_closed_loop_context_adapter_v2 import build_closed_loop_context_v2, normalize_trace_observation


ROOT = Path(__file__).resolve().parents[1]


class FakeOfficialMap:
    lanes = {"current": 0.0, "left": 3.2, "right": -3.2}

    def lane_context(self, ego_xy, route_roadblock_ids):
        return {"valid": True, "current_lane_id": "current", "left_lane_id": "left", "right_lane_id": "right", "tangent": [1.0, 0.0], "road_class": "LANE"}

    def project(self, lane_id, xy):
        return {"arc_m": float(xy[0]), "lateral_offset_m": float(xy[1]) - self.lanes[lane_id], "tangent": [1.0, 0.0]}

    def lane_for_actor(self, actor):
        return min(self.lanes, key=lambda lane: abs(float(actor["y"]) - self.lanes[lane]))

    def static_stop_control_ahead(self, ego_xy, route_roadblock_ids):
        return False


def frames():
    out = []
    for i in range(10):
        x = i * 0.5
        out.append({
            "time_s": i * 0.1,
            "iteration_index": i,
            "time_us": 1_000_000 + i * 100_000,
            "ego": {"x": x, "y": 0.0, "speed_mps": 5.0},
            "actors": [
                {"track_id": "front", "type": "VEHICLE", "x": x + 12.0, "y": 0.0, "vx": 3.0, "vy": 0.0},
                {"track_id": "left-front", "type": "VEHICLE", "x": x + 15.0, "y": 3.2, "vx": 5.0, "vy": 0.0},
                {"track_id": "left-rear", "type": "VEHICLE", "x": x - 8.0, "y": 3.2, "vx": 5.0, "vy": 0.0},
            ],
            "traffic_lights": [{"status": "RED", "route_relevant": True}],
        })
    return out


def common(family):
    return dict(family=family, scenario_token="synthetic", map_version="synthetic-map", route_fingerprint="route", initial_state_fingerprint="initial", log_id="log", route_roadblock_ids=["rb"], frames=frames(), map_query=FakeOfficialMap())


def test_hlc_real_slots_stable_ids_and_target_gaps():
    result = build_closed_loop_context_v2(**common("R-HLC"), intended_lane_change_direction="LEFT")
    assert result["context_variables"]["neighbor_availability_pattern"] == "11100"
    assert result["missingness_states"] == {"target_front": "TARGET_FRONT_PRESENT", "target_rear": "TARGET_REAR_PRESENT"}
    assert result["target_track_audit"]["target_front"]["track_ids"] == ["left-front"]
    assert result["context_variables"]["target_lane_initial_front_gap_m"] == 15.0


def test_tsb_front_gap_relative_speed_thw_and_hazard_multihot():
    result = build_closed_loop_context_v2(**common("R-TSB"))
    assert result["missingness_states"]["front"] == "FRONT_PRESENT"
    assert result["context_variables"]["initial_front_gap_m"] == 12.0
    assert result["context_variables"]["initial_lead_relative_speed_mps"] == -2.0
    assert result["context_variables"]["initial_thw_s"] == 2.4
    assert "ROUTE_SIGNAL_RED_OR_YELLOW" in result["hazard_multi_hot_audit"]
    assert "OBSERVED_SLOW_LEAD" in result["hazard_multi_hot_audit"]


def test_existing_readonly_trace_extracts_real_actors_without_forced_absence():
    path = sorted((ROOT / "outputs/r1_official_compliant_technical_smoke_v1_1/runs").glob("*/trace/planner_trace.jsonl"))[0]
    with path.open() as handle:
        first = json.loads(next(handle))
    selected = first["pre_context_raw"][-11:-1]
    normalized = [normalize_trace_observation(frame) for frame in selected]
    assert len(normalized) == 10
    assert sum(len(frame) for frame in normalized) > 0
    assert all(actor["track_id"] for frame in normalized for actor in frame)


def test_all_old_runs_have_real_tracks_but_not_exact_new_temporal_grid():
    paths = sorted((ROOT / "outputs/r1_official_compliant_technical_smoke_v1_1/runs").glob("*/trace/planner_trace.jsonl"))
    assert len(paths) == 48
    actor_runs = stable_runs = exact_runs = 0
    for path in paths:
        rows = []
        with path.open() as handle:
            for i, line in enumerate(handle):
                if i == 10:
                    break
                rows.append(json.loads(line))
        observations = [normalize_trace_observation(row["canonical_context"][-1]) for row in rows]
        actor_runs += any(observations)
        counts = {}
        for observation in observations:
            for actor in observation:
                counts[actor["track_id"]] = counts.get(actor["track_id"], 0) + 1
        stable_runs += any(value >= 8 for value in counts.values())
        times = [int(row["current_ego"]["time_us"]) for row in rows]
        exact_runs += all(times[i + 1] - times[i] == 100_000 for i in range(9))
    assert actor_runs == 48
    assert stable_runs == 48
    assert exact_runs == 0


def test_context_adapter_fails_closed_on_physical_timestamp_jitter():
    payload = common("R-TSB")
    payload["frames"][5]["time_us"] += 1
    with pytest.raises(ValueError, match="NOT_EVALUABLE_TEMPORAL_GRID"):
        build_closed_loop_context_v2(**payload)
