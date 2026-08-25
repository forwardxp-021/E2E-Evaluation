from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dynamic_builder_median5_and_assignment_contract() -> None:
    module = load("build_waymo_dynamic_interaction_dataset_v2")
    values = np.asarray([0.0, 0.0, 100.0, 0.0, 0.0])
    assert np.allclose(module.median5(values), 0.0)
    args = SimpleNamespace(
        lane_max_lateral_distance=3.0,
        lane_max_heading_diff_deg=45.0,
        adjacent_lane_min_offset=2.0,
        adjacent_lane_max_offset=5.5,
        adjacent_lane_max_heading_diff_deg=35.0,
        lane_search_radius=20.0,
        lane_topk_candidates=32,
        front_max_distance=120.0,
        side_front_max_distance=80.0,
        side_rear_max_distance=120.0,
        lane_lateral_tolerance=2.0,
        slot_heading_diff_deg=45.0,
        static_speed_threshold=0.5,
        disable_lane_spatial_index=False,
    )
    config = module.assignment_config(args)
    assert config["ego_projection_precomputed"] is True
    assert config["candidate_projections_complete"] is True
    assert config["allow_geometric_adjacent_lane_inference"] is False


def test_stage6s_mechanism_metrics_use_front_and_closing_response() -> None:
    module = load("stage6s_evaluate_interaction_mechanism")
    ego = np.zeros((6, 8), dtype=np.float32)
    ego[:, 5] = 10.0
    ego[:, 6] = np.asarray([0.0, -1.0, -2.0, -1.0, 0.0, 0.0])
    neighbor = np.zeros((5, 6, 15), dtype=np.float32)
    neighbor[0, :, 0] = 1.0
    neighbor[0, :, 5] = 20.0
    neighbor[0, :, 8] = np.asarray([0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
    neighbor[0, :, 10] = 2.0
    result = module.metrics(ego, neighbor, np.ones(6, dtype=bool))
    assert result["mean_speed"] == 10.0
    assert result["mean_thw"] == 2.0
    assert result["mean_front_gap"] == 20.0
    assert result["mean_accel_during_closing"] < 0.0


def test_stage6r_and_stage6s_configs_preserve_frozen_boundaries() -> None:
    import json

    stage6r = json.loads((ROOT / "configs/stage6r_waymo_dynamic_builder_v2.json").read_text())
    stage6s = json.loads((ROOT / "configs/stage6s_interaction_dominant_nuplan_benchmark.json").read_text())
    assert stage6r["full51_readiness_gate"]["intermittent_train_min"] == 5000
    assert stage6r["assignment_mode"] == "lane_aware_only"
    assert stage6r["pilot_gate"]["semantic_strict_assignment"] is True
    assert "modify_stage6o_v1" in stage6r["forbidden_actions"]
    assert "train_checkpoint" in stage6r["forbidden_actions"]
    assert stage6s["pilot_pair_count"] == 24
    assert {"read_embedding", "read_bdd"}.issubset(stage6s["forbidden_actions"])


def test_stage6r_pilot_event_flags_capture_intermittent_and_switch() -> None:
    module = load("stage6q_audit_waymo_raw_interaction_coverage")
    lead_ids = [None] * 8 + ["a"] * 10 + [None] * 8 + ["b"] * 10
    flags = module.event_flags(
        lead_ids,
        np.asarray([np.nan] * 8 + [20.0] * 10 + [np.nan] * 8 + [15.0] * 10),
        np.asarray([np.nan] * 8 + [1.0] * 10 + [np.nan] * 8 + [2.0] * 10),
        5,
    )
    assert flags["lead_entry"] is True
    assert flags["lead_exit"] is True
    assert flags["intermittent_following_primary"] is True


def _straight_lane(module, lane_id: str, y: float, *, left_relations=None):
    points = np.asarray([[float(x), y] for x in range(12)], dtype=np.float32)
    geom = module._build_lane_geom(points)
    return module.LaneInfo(
        lane_id=lane_id, centerline_xy=points, seg_heading=geom[0], seg_len=geom[1],
        s_prefix=geom[2], seg_start_xy=geom[3], seg_vec_xy=geom[4], seg_den=geom[5],
        bbox_min_xy=geom[6], bbox_max_xy=geom[7], bbox_center_xy=geom[8],
        left_neighbor_lane_ids=["left"] if left_relations else [],
        left_neighbor_relations=left_relations or [],
    )


def test_lane_neighbor_relation_is_active_only_on_local_self_range() -> None:
    lane_module = load("waymo_lane_utils")
    assign_module = load("lane_aware_assignment")
    relation = {"lane_id": "left", "self_start_index": 5, "self_end_index": 8,
                "neighbor_start_index": 5, "neighbor_end_index": 8}
    lanes = {
        "ego": _straight_lane(lane_module, "ego", 0.0, left_relations=[relation]),
        "left": _straight_lane(lane_module, "left", 3.5),
    }
    candidate = {"car": {"x": 8.0, "y": 3.5, "heading": 0.0, "speed": 5.0}}
    candidate_projection = {"car": lane_module.project_point_to_lane([8.0, 3.5], lanes["left"])}
    config = {"ego_projection_precomputed": True, "candidate_projections_complete": True}
    before = assign_module.assign_neighbors_lane_aware(
        {"x": 2.0, "y": 0.0, "heading": 0.0}, candidate, lanes, "lane_aware_only", config,
        lane_module.project_point_to_lane([2.0, 0.0], lanes["ego"]), candidate_projection,
    )
    active = assign_module.assign_neighbors_lane_aware(
        {"x": 6.0, "y": 0.0, "heading": 0.0}, candidate, lanes, "lane_aware_only", config,
        lane_module.project_point_to_lane([6.0, 0.0], lanes["ego"]), candidate_projection,
    )
    assert before.left_lane_id == ""
    assert "left_front" not in before.slot_to_agent
    assert active.left_lane_id == "left"
    assert active.slot_to_agent["left_front"] == "car"


def test_lane_neighbor_candidate_must_be_inside_neighbor_local_range() -> None:
    lane_module = load("waymo_lane_utils")
    assign_module = load("lane_aware_assignment")
    relation = {"lane_id": "left", "self_start_index": 5, "self_end_index": 8,
                "neighbor_start_index": 5, "neighbor_end_index": 8}
    lanes = {
        "ego": _straight_lane(lane_module, "ego", 0.0, left_relations=[relation]),
        "left": _straight_lane(lane_module, "left", 3.5),
    }
    candidates = {
        "outside": {"x": 10.0, "y": 3.5, "heading": 0.0, "speed": 5.0},
        "inside": {"x": 8.0, "y": 3.5, "heading": 0.0, "speed": 5.0},
    }
    projections = {key: lane_module.project_point_to_lane([value["x"], value["y"]], lanes["left"])
                   for key, value in candidates.items()}
    result = assign_module.assign_neighbors_lane_aware(
        {"x": 6.0, "y": 0.0, "heading": 0.0}, candidates, lanes, "lane_aware_only",
        {"ego_projection_precomputed": True, "candidate_projections_complete": True},
        lane_module.project_point_to_lane([6.0, 0.0], lanes["ego"]), projections,
    )
    assert result.slot_to_agent["left_front"] == "inside"


def test_multiple_active_neighbor_relations_are_all_considered() -> None:
    lane_module = load("waymo_lane_utils")
    assign_module = load("lane_aware_assignment")
    relations = [
        {"lane_id": "left_empty", "self_start_index": 5, "self_end_index": 8,
         "neighbor_start_index": 5, "neighbor_end_index": 8},
        {"lane_id": "left_used", "self_start_index": 5, "self_end_index": 8,
         "neighbor_start_index": 5, "neighbor_end_index": 8},
    ]
    lanes = {
        "ego": _straight_lane(lane_module, "ego", 0.0, left_relations=relations),
        "left_empty": _straight_lane(lane_module, "left_empty", 3.5),
        "left_used": _straight_lane(lane_module, "left_used", 7.0),
    }
    candidate = {"car": {"x": 8.0, "y": 7.0, "heading": 0.0, "speed": 5.0}}
    result = assign_module.assign_neighbors_lane_aware(
        {"x": 6.0, "y": 0.0, "heading": 0.0}, candidate, lanes, "lane_aware_only",
        {"ego_projection_precomputed": True, "candidate_projections_complete": True},
        lane_module.project_point_to_lane([6.0, 0.0], lanes["ego"]),
        {"car": lane_module.project_point_to_lane([8.0, 7.0], lanes["left_used"])},
    )
    assert result.slot_to_agent["left_front"] == "car"
