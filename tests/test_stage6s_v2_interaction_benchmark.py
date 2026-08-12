from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools/stage6s_v2_audit_pretreatment_interaction_inventory.py"
    spec = importlib.util.spec_from_file_location("stage6s_v2_inventory", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_mechanism_tool():
    path = ROOT / "tools/stage6s_v2_evaluate_development_mechanism.py"
    spec = importlib.util.spec_from_file_location("stage6s_v2_mechanism", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_confirmation_tool():
    path = ROOT / "tools/stage6s_v2_freeze_confirmation.py"
    spec = importlib.util.spec_from_file_location("stage6s_v2_confirmation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage6s_v2_config_freezes_blinding_and_robust_thw() -> None:
    config = json.loads((ROOT / "configs/stage6s_v2_interaction_benchmark.json").read_text())
    assert config["development"]["pair_count"] == 24
    assert config["development"]["minimum_distinct_logs"] == 4
    assert config["development"]["maximum_scenarios_per_log"] == 6
    assert config["confirmation"]["minimum_pair_count"] == 60
    assert config["confirmation"]["development_log_disjoint"] is True
    assert config["thw_definition"]["exclude_sentinel_or_cap"] is True
    assert config["thw_definition"]["maximum_seconds_exclusive"] == 20.0
    assert {"read_embedding", "read_bdd_or_mmd", "train_checkpoint"}.issubset(config["forbidden_actions"])


def test_pretreatment_audit_uses_invariant_ego_speed_not_world_heading_projection() -> None:
    module = load_tool()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE scenario_tag(token BLOB, lidar_pc_token BLOB, type TEXT, agent_track_token BLOB);
        CREATE TABLE lidar_pc(token BLOB, timestamp INTEGER, scene_token BLOB, ego_pose_token BLOB);
        CREATE TABLE ego_pose(token BLOB, x REAL, y REAL, qw REAL, qx REAL, qy REAL, qz REAL, vx REAL, vy REAL);
        CREATE TABLE lidar_box(lidar_pc_token BLOB, track_token BLOB, x REAL, y REAL, vx REAL, vy REAL, yaw REAL, length REAL);
        """
    )
    scenario, scene, track = b"scenario", b"scene", b"track"
    conn.execute("INSERT INTO scenario_tag VALUES(?,?,?,?)", (b"tag", scenario, "following_lane_with_slow_lead", track))
    for index in range(120):
        lidar = scenario if index == 0 else f"lidar{index}".encode()
        pose = f"pose{index}".encode()
        timestamp = index * 50_000
        conn.execute("INSERT INTO lidar_pc VALUES(?,?,?,?)", (lidar, timestamp, scene, pose))
        # Global heading is north, but ego_pose velocity is body-frame +x.
        conn.execute("INSERT INTO ego_pose VALUES(?,?,?,?,?,?,?,?,?)", (pose, 0.0, index * 0.25, 0.70710678, 0.0, 0.0, 0.70710678, 5.0, 0.0))
        conn.execute("INSERT INTO lidar_box VALUES(?,?,?,?,?,?,?,?)", (lidar, track, 0.0, index * 0.15 + 20.0, 0.0, 3.0, 1.57079632679, 4.5))
    config = json.loads((ROOT / "configs/stage6s_v2_interaction_benchmark.json").read_text())["inventory"]
    cache = module.load_db_cache(conn, {"following_lane_with_slow_lead"})
    row = {
        "db_file": "fixture.db", "log_name": "fixture", "scenario_token": scenario.hex(),
        "scene_token": scenario.hex(), "db_scene_token": scene.hex(),
        "scenario_type": "following_lane_with_slow_lead", "scenario_tag_token": b"tag".hex(),
    }
    result = module.audit_candidate(row, config, cache)
    assert result["ego_median_speed_mps"] == 5.0
    assert result["ego_median_speed_mps"] >= config["ego_median_speed_min_mps"]


def test_mechanism_thw_excludes_sentinel_and_json_is_strict() -> None:
    module = load_mechanism_tool()
    ego = np.zeros((4, 8), dtype=float)
    ego[:, 5] = 5.0
    neighbor = np.zeros((5, 4, 15), dtype=float)
    neighbor[0, :, 0] = 1.0
    neighbor[0, :, 5] = [10.0, 12.0, 14.0, 16.0]
    neighbor[0, :, 10] = [2.0, 999.0, 20.0, 4.0]
    result = module.metrics(
        ego,
        neighbor,
        np.ones(4, dtype=bool),
        {"minimum_seconds_exclusive": 0.0, "maximum_seconds_exclusive": 20.0},
    )
    assert result["median_finite_thw"] == 3.0
    assert result["finite_thw_ratio"] == 0.5
    assert module.json_safe({"count": np.int64(3), "missing": float("nan")}) == {
        "count": 3,
        "missing": None,
    }


def test_confirmation_selection_is_development_disjoint_and_outcome_blind() -> None:
    module = load_confirmation_tool()
    config = json.loads((ROOT / "configs/stage6s_v2_interaction_benchmark.json").read_text())
    config["confirmation"]["target_pair_count"] = 2
    config["confirmation"]["minimum_pair_count"] = 2
    inventory = [
        {"eligible": "True", "log_name": "development_log", "scenario_token": "a"},
        {"eligible": "True", "log_name": "confirmation_log", "scenario_token": "b"},
        {"eligible": "True", "log_name": "confirmation_log", "scenario_token": "c"},
        {"eligible": "True", "log_name": "old_log", "scenario_token": "old"},
    ]
    selected, counts = module.select_confirmation(
        inventory,
        [{"log_name": "development_log", "scenario_token": "a"}],
        [{"log_name": "old_log", "scenario_token": "old"}],
        config,
    )
    assert {row["scenario_token"] for row in selected} == {"b", "c"}
    assert counts["eligible_before_exclusions"] == 4
    assert counts["after_stage6s_v1_token_exclusion"] == 2
