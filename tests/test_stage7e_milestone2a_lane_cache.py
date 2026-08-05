from __future__ import annotations

import numpy as np
import sqlite3

from tools.build_nuplan_5neighbor_context_dataset import (
    scenario_lane_cache_key,
    scenario_query_ego_xy,
    summarize_lane_assignment_rows,
    write_strict_filter_diagnostic,
    resolve_map_name,
)
from tools.nuplan_lane_utils import select_spatial_query_anchors
from tools.waymo_lane_utils import LaneInfo, _build_lane_geom, find_best_lane_for_agent


def _lane(lane_id: str, y: float = 0.0) -> LaneInfo:
    xy = np.asarray([[0.0, y], [20.0, y]], dtype=np.float32)
    geom = _build_lane_geom(xy)
    assert geom is not None
    return LaneInfo(lane_id, xy, *geom)


def test_lane_cache_is_scoped_by_map_and_source_scenario() -> None:
    assert scenario_lane_cache_key("map-a", 3) == ("map-a", 3)
    assert scenario_lane_cache_key("map-a", 4) != scenario_lane_cache_key("map-a", 3)
    assert scenario_lane_cache_key("map-b", 3) != scenario_lane_cache_key("map-a", 3)


def test_scenario_query_positions_include_every_planner_valid_frame() -> None:
    seq = np.zeros((2, 3, 4), dtype=np.float32)
    seq[0, :, :2] = [[0, 0], [1, 0], [2, 0]]
    seq[1, :, :2] = [[10, 0], [11, 0], [12, 0]]
    mask = np.asarray([[True, False, True], [False, True, True]])
    xy = scenario_query_ego_xy(seq, mask)
    np.testing.assert_array_equal(xy, [[0, 0], [2, 0], [11, 0], [12, 0]])


def test_spatial_query_anchors_bound_dense_duplicate_queries() -> None:
    xy = np.asarray([[0, 0], [1, 0], [59, 0], [60, 0], [119, 0], [120, 0]], dtype=float)
    anchors = select_spatial_query_anchors(xy, 60.0)
    np.testing.assert_array_equal(anchors, [[0, 0], [60, 0], [120, 0]])
    assert max(min(np.linalg.norm(point - anchor) for anchor in anchors) for point in xy) < 60.0


def test_map_name_uses_log_db_location_before_cli_fallback(tmp_path) -> None:
    db = tmp_path / "sample-log.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE log (location TEXT, map_version TEXT)")
        connection.execute("INSERT INTO log VALUES (?, ?)", ("las_vegas", "us-nv-las-vegas-strip"))
    cache = {}
    map_name, source = resolve_map_name(
        {"log_name": "sample-log"},
        explicit_map_name="us-nv-las-vegas-strip",
        nuplan_db_root=tmp_path,
        db_map_cache=cache,
    )
    assert map_name == "us-nv-las-vegas-strip"
    assert source == "nuplan_db.log.map_version"
    assert cache == {"sample-log": "us-nv-las-vegas-strip"}


def test_projection_failure_reason_distinguishes_lateral_and_heading() -> None:
    lanes = {"lane": _lane("lane")}
    projection, reason, _ = find_best_lane_for_agent(
        np.asarray([5.0, 10.0]), 0.0, lanes, max_lateral_distance=3.0, max_heading_diff=np.pi / 4
    )
    assert projection is None
    assert reason == "lateral_distance_exceeded"
    projection, reason, _ = find_best_lane_for_agent(
        np.asarray([5.0, 0.0]), np.pi, lanes, max_lateral_distance=3.0, max_heading_diff=np.pi / 4
    )
    assert projection is None
    assert reason == "heading_difference_exceeded"


def test_assignment_diagnostics_are_exactly_frame_weighted() -> None:
    metadata = [
        {"global_row": 0, "scenario_index": 10, "planner_id": 0, "planner_name": "a"},
        {"global_row": 1, "scenario_index": 10, "planner_id": 1, "planner_name": "b"},
    ]
    debug = [
        [
            {"lane_assignment_available": True, "fallback_assignment_used": False, "fallback_reason": ""},
            {"lane_assignment_available": False, "fallback_assignment_used": True, "fallback_reason": "lateral_distance_exceeded"},
        ],
        [
            {"lane_assignment_available": True, "fallback_assignment_used": False, "fallback_reason": ""},
        ],
    ]
    summary = summarize_lane_assignment_rows(debug, metadata)
    assert summary["cache_scope"] == "map_name_plus_source_scenario"
    assert summary["valid_frame_count"] == 3
    assert summary["fallback_frame_count"] == 1
    assert summary["fallback_assignment_used_rate"] == 1 / 3
    assert summary["fallback_reason_counts"] == {"lateral_distance_exceeded": 1}


def test_strict_filter_uses_noncontiguous_source_scenario_ids_and_rejects_ambiguous(tmp_path) -> None:
    planners = ["a", "b"]
    rows = [
        {"scenario_index": scenario, "planner_name": planner}
        for scenario in (17, 18)
        for planner in planners
    ]
    good = {"lane_assignment_available": True, "fallback_assignment_used": False, "lane_context_quality": "good", "current_lane_id": "lane"}
    ambiguous = {**good, "lane_context_quality": "ambiguous_intersection"}
    debug = [[good], [good], [ambiguous], [good]]
    neighbor = np.zeros((4, 5, 1, 15), dtype=np.float32)
    summary = write_strict_filter_diagnostic(
        tmp_path,
        rows,
        planners,
        2,
        debug,
        neighbor,
        {},
        {},
        strict_filter_min_laneaware_ratio=0.8,
        strict_filter_ratio_sweep=[1.0, 0.8],
    )
    assert set(summary["scenario_planner_alignment_after_filtering"]) == {17, 18}
    assert summary["rows_kept"] == 3
    assert summary["scenarios_with_all_planners"] == 1
    assert summary["dropped_by_reason"]["drop_if_lane_context_ambiguous"] == 1
    assert len(summary["strict_filter_ratio_sweep"]) == 2
    assert all(item["scenarios_with_all_planners"] == 1 for item in summary["strict_filter_ratio_sweep"])
