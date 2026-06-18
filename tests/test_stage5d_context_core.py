from pathlib import Path
import ast
import pytest
import numpy as np

from tools import stage5d_context_core as core


def test_stage5d_slot_names_single_source():
    assert core.SLOT_NAMES == ["front", "left_front", "left_rear", "right_front", "right_rear"]
    source = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "tools.stage5d_context_core"]
    assert any(any(alias.name == "SLOT_NAMES" for alias in node.names) for node in imports)


def test_stage5d_context_dim():
    assert core.CONTEXT_DIM == 83
    assert core.CONTEXT_DIM == len(core.EGO_CHANNELS) + len(core.SLOT_NAMES) * len(core.NEIGHBOR_CHANNELS)
    assert len(core.EGO_CHANNELS) == 8
    assert len(core.NEIGHBOR_CHANNELS) == 15


def test_waymo_builder_parity():
    track = np.asarray([
        [0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 2.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    ego, heading, speed = core.build_ego_features_8d(track, track[0, :2], 0.0, 0.1)
    np.testing.assert_allclose(ego[:, 0], [0.0, 0.2], atol=1e-6)
    np.testing.assert_allclose(ego[:, 2], [2.0, 2.0], atol=1e-6)
    feat = core.build_neighbor_features_15d(
        rel_x=10.0, rel_y=0.0, rel_vx=-1.0, rel_vy=0.0,
        ego_forward_speed=2.0, neighbor_speed=1.0, neighbor_accel=0.0,
        heading_rel=0.0, neighbor_yaw_rate=0.0,
    )
    expected = np.asarray([1.0, 10.0, 0.0, -1.0, 0.0, 10.0, 10.0, 0.0, 3.0, 10.0 / 3.0, 5.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(feat, expected, rtol=1e-6, atol=1e-6)


def test_nuplan_builder_schema():
    ego = np.zeros((2, len(core.EGO_CHANNELS)), dtype=np.float32)
    nbr = np.zeros((len(core.SLOT_NAMES), 2, len(core.NEIGHBOR_CHANNELS)), dtype=np.float32)
    ctx = core.build_context_traj_from_standard_tracks(ego, nbr)[None]
    validation = core.validate_stage5d_context(ctx, ego[None], nbr[None])
    assert ctx.shape == (1, 2, 83)
    assert validation["stage5d_core_reused"] is True
    schema = core.make_stage5d_context_schema()
    assert schema["neighbor_slots"] == core.SLOT_NAMES
    assert schema["neighbor_channels_per_slot"] == core.NEIGHBOR_CHANNELS


def test_no_duplicate_schema_constants():
    source = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assert target.id not in {"SLOT_NAMES", "NEIGHBOR_CHANNELS", "STAGE5D_NEIGHBOR_SLOT_NAMES", "STAGE5D_NEIGHBOR_CHANNELS"}
    assert "STAGE5D_NEIGHBOR_SLOT_NAMES" not in source
    assert "STAGE5D_NEIGHBOR_CHANNELS" not in source


def test_stage7e_final_script_has_no_legacy_debug_bridge():
    source = Path("tools/stage7e_embed_stage6_dataset.py").read_text(encoding="utf-8")
    forbidden = [
        "neighbor[:, :5]",
        "neighbor_arr[:, :k]",
        "build_ego_neighbor9_context",
        "build_checkpoint_compatible_context",
        "pad_to_checkpoint_dim",
        "ego_neighbor9",
        "STAGE5D_NEIGHBOR_SLOT_NAMES",
    ]
    for token in forbidden:
        assert token not in source
    assert "context_traj.npy" in source
    assert "does_not_rebuild_context_from_stage7d_neighbor_seq" in source


def test_stage7e_parser_requires_context_dataset_dir(monkeypatch):
    import tools.stage7e_embed_stage6_dataset as stage7e

    monkeypatch.setattr(
        "sys.argv",
        [
            "stage7e_embed_stage6_dataset.py",
            "--checkpoint",
            "model.pt",
            "--output_dir",
            "out",
        ],
    )
    with pytest.raises(SystemExit):
        stage7e.parse_args()


def test_nuplan_builder_imports_full_stage5d_core_schema():
    source = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "tools.stage5d_context_core"]
    imported = {alias.name for node in imports for alias in node.names}
    assert {"SLOT_NAMES", "EGO_CHANNELS", "NEIGHBOR_CHANNELS", "CONTEXT_DIM"}.issubset(imported)


def test_nuplan_builder_does_not_import_stage7d_convert_ego():
    source = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tools.stage7d_export_stage6_compatible_dataset":
            imported = {alias.name for alias in node.names}
            assert "convert_ego" not in imported
    assert "convert_ego(" not in source


def test_nuplan_builder_does_not_import_REQUIRED_PLANNERS():
    source = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tools.stage7d_export_stage6_compatible_dataset":
            imported = {alias.name for alias in node.names}
            assert "REQUIRED_PLANNERS" not in imported
    assert "REQUIRED_PLANNERS" not in source
    assert "--required_planners" in source
    assert "default=[]" in source


def test_nuplan_ego_features_use_stage5d_core():
    import tools.build_nuplan_5neighbor_context_dataset as builder

    seq = np.asarray([
        [10.0, 20.0, np.pi / 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 20.2, np.pi / 2.0, 2.0, 0.0, 0.0, 0.0, 0.1],
        [10.0, 20.4, np.pi / 2.0, 2.0, 0.0, 0.0, 0.0, 0.2],
    ], dtype=np.float32)
    mask = np.asarray([True, True, True])
    ego, heading, speed, dt = builder.build_nuplan_ego_features_8d(seq, mask)
    expected_track = np.asarray([
        [10.0, 20.0, 0.0, 2.0, np.pi / 2.0, 1.0],
        [10.0, 20.2, 0.0, 2.0, np.pi / 2.0, 1.0],
        [10.0, 20.4, 0.0, 2.0, np.pi / 2.0, 1.0],
    ], dtype=np.float32)
    expected_ego, expected_heading, expected_speed = core.build_ego_features_8d(expected_track, expected_track[0, :2], float(expected_track[0, 4]), 0.1)
    np.testing.assert_allclose(ego, expected_ego, atol=1e-6)
    np.testing.assert_allclose(heading, expected_heading, atol=1e-6)
    np.testing.assert_allclose(speed, expected_speed, atol=1e-6)
    assert dt == pytest.approx(0.1)


def test_stage5d_context_core_local_frame_contract():
    track = np.asarray([
        [5.0, 7.0, 0.0, 2.0, np.pi / 2.0, 1.0],
        [5.0, 8.0, 0.0, 2.0, np.pi / 2.0, 1.0],
    ], dtype=np.float32)
    ego, heading, speed = core.build_ego_features_8d(track, track[0, :2], float(track[0, 4]), 0.5)
    # With base heading pi/2, the world +y displacement is local +x; lateral remains zero.
    np.testing.assert_allclose(ego[:, :2], [[0.0, 0.0], [1.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(ego[:, 2:4], [[2.0, 0.0], [2.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(ego[:, 4], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(speed, [2.0, 2.0], atol=1e-6)
    assert np.all(np.isfinite(heading))


def test_slot_switch_makes_temporal_formula_parity_conservative():
    import tools.build_nuplan_5neighbor_context_dataset as builder

    # front slot switches from agent_a to agent_b across a valid transition.
    row_slots = [["agent_a", "agent_b", "agent_b"]] + [["-1", "-1", "-1"] for _ in core.SLOT_NAMES[1:]]
    continuity = builder.slot_continuity_stats([row_slots])
    switch_rates = {k: float(v["slot_id_switch_rate"] or 0.0) for k, v in continuity.items()}
    accel_yaw_rate_matched = all(rate == 0.0 for rate in switch_rates.values())
    stage5d_static_derived_formula_matched = True
    stage5d_temporal_derived_formula_matched = bool(accel_yaw_rate_matched)
    stage5d_derived_formula_matched = bool(stage5d_static_derived_formula_matched and stage5d_temporal_derived_formula_matched)

    assert switch_rates["front"] > 0.0
    assert accel_yaw_rate_matched is False
    assert stage5d_derived_formula_matched is False

    schema = core.make_stage5d_context_schema(accel_yaw_rate_matched=accel_yaw_rate_matched)
    assert schema["stage5d_accel_yaw_rate_formula_matched"] is False
    assert schema["stage5d_derived_formula_matched"] is False


def test_zero_slot_switch_allows_full_derived_formula_parity():
    import tools.build_nuplan_5neighbor_context_dataset as builder

    row_slots = [["agent_a", "agent_a", "agent_a"]] + [["-1", "-1", "-1"] for _ in core.SLOT_NAMES[1:]]
    continuity = builder.slot_continuity_stats([row_slots])
    switch_rates = {k: float(v["slot_id_switch_rate"] or 0.0) for k, v in continuity.items()}
    accel_yaw_rate_matched = all(rate == 0.0 for rate in switch_rates.values())
    stage5d_static_derived_formula_matched = True
    stage5d_temporal_derived_formula_matched = bool(accel_yaw_rate_matched)
    stage5d_derived_formula_matched = bool(stage5d_static_derived_formula_matched and stage5d_temporal_derived_formula_matched)

    assert all(rate == 0.0 for rate in switch_rates.values())
    assert accel_yaw_rate_matched is True
    assert stage5d_derived_formula_matched is True

    schema = core.make_stage5d_context_schema(accel_yaw_rate_matched=accel_yaw_rate_matched)
    assert schema["stage5d_accel_yaw_rate_formula_matched"] is True
    assert schema["stage5d_derived_formula_matched"] is True


def test_nuplan_warnings_validation_does_not_let_core_override_conservative_parity():
    source = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    assert '"validation": {**core_validation, **validation}' in source
    assert '"stage5d_derived_formula_matched": True, "stage5d_closing_formula_matched"' not in source

def test_stage7e_context_builder_fails_lane_aware_only_without_map_name():
    import tools.build_nuplan_5neighbor_context_dataset as builder
    map_name, source = builder.resolve_map_name({}, explicit_map_name="", scenario_map_metadata={})
    assert map_name == ""
    assert source == "unresolved"
    source_text = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    assert 'args.assignment_mode == "lane_aware_only" and (map_name_resolved_rate < 1.0' in source_text
    assert '"severity": "error" if args.assignment_mode == "lane_aware_only" else "warning"' in source_text


def test_stage7e_context_builder_warns_fallback_without_map_name():
    source_text = Path("tools/build_nuplan_5neighbor_context_dataset.py").read_text(encoding="utf-8")
    assert "lane_aware_with_geometric_fallback" in source_text
    assert "No map_name could be resolved" in source_text
    assert '"lane_assignment_available": lane_assignment_available' in source_text
    assert '"map_query_success": map_query_success' in source_text


def test_stage7e_context_builder_accepts_explicit_map_name():
    import tools.build_nuplan_5neighbor_context_dataset as builder
    map_name, source = builder.resolve_map_name({}, explicit_map_name="us-nv-las-vegas-strip", scenario_map_metadata={})
    assert map_name == "us-nv-las-vegas-strip"
    assert source == "cli.--map_name"


def test_stage7c_metadata_contains_map_name_when_scenario_metadata_is_available():
    import tools.stage7c1_run_nuplan_simulation as stage7c
    row = stage7c.scenario_index_row(
        {
            "scenario_index": "0",
            "db_name": "log.db",
            "scene_token": "scene",
            "scenario_id": "scenario",
            "sample_id": "sample",
            "map_name": "us-nv-las-vegas-strip",
            "location": "las_vegas",
            "scenario_type": "following",
        },
        {"planner_id": 2, "planner_name": "idm"},
        "succeeded",
        80,
        0,
    )
    assert row["map_name"] == "us-nv-las-vegas-strip"
    assert row["location"] == "las_vegas"
    assert row["log_name"] == "log"
    assert row["scenario_type"] == "following"
    assert "map_name" in stage7c.SCENARIO_INDEX_COLUMNS


def test_compare_lane_aware_diagnostics_has_no_local_slot_names_fallback():
    source = Path("tools/compare_lane_aware_diagnostics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert any(
        node.module in {"tools.lane_aware_assignment", "tools.stage5d_context_core"}
        and any(alias.name == "SLOT_NAMES" for alias in node.names)
        for node in imports
    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assert target.id != "SLOT_NAMES"
    assert '["front", "left_front", "left_rear", "right_front", "right_rear"]' not in source


def test_compare_lane_aware_diagnose_missing_waymo_metrics_is_inconclusive():
    from tools.compare_lane_aware_diagnostics import diagnose

    result = diagnose({}, {"fallback_assignment_used_rate": 0.9, "candidate_projection_success_rate": 0.1}, 0.2)
    assert result["verdict"] == "inconclusive_missing_waymo_metrics"
    assert result["fallback_rate_comparable"] is False
    assert result["candidate_projection_success_comparable"] is False
    assert result["missing_waymo_metrics"] == {
        "fallback_assignment_used_rate": True,
        "candidate_projection_success_rate": True,
    }


def test_compare_lane_aware_diagnose_flags_nuplan_only_when_comparable_and_worse():
    from tools.compare_lane_aware_diagnostics import diagnose

    by_fallback = diagnose(
        {"fallback_assignment_used_rate": 0.1, "candidate_projection_success_rate": 0.9},
        {"fallback_assignment_used_rate": 0.5, "candidate_projection_success_rate": 0.85},
        0.2,
    )
    assert by_fallback["verdict"] == "nuplan_adapter_or_map_projection_issue"
    assert by_fallback["fallback_rate_comparable"] is True

    by_projection = diagnose(
        {"fallback_assignment_used_rate": 0.1, "candidate_projection_success_rate": 0.9},
        {"fallback_assignment_used_rate": 0.15, "candidate_projection_success_rate": 0.6},
        0.2,
    )
    assert by_projection["verdict"] == "nuplan_adapter_or_map_projection_issue"
    assert by_projection["candidate_projection_success_comparable"] is True


def test_compare_lane_aware_diagnose_generic_when_nuplan_not_clearly_worse():
    from tools.compare_lane_aware_diagnostics import diagnose

    result = diagnose(
        {"fallback_assignment_used_rate": 0.2, "candidate_projection_success_rate": 0.8},
        {"fallback_assignment_used_rate": 0.3, "candidate_projection_success_rate": 0.72},
        0.2,
    )
    assert result["verdict"] == "generic_stage5_lane_aware_limitation_or_inconclusive"


def test_nuplan_slot_sanity_skips_zero_coverage_and_keeps_schema_order():
    import tools.build_nuplan_5neighbor_context_dataset as builder

    stats = {slot: {"coverage_ratio": 0.0, "median_rel_x": None, "median_rel_y": None} for slot in builder.SLOT_NAMES}
    sanity, passed, evaluated, skipped, warnings = builder.evaluate_slot_sanity(stats, 0.05)
    assert passed is True
    assert evaluated == []
    assert skipped == list(builder.SLOT_NAMES)
    assert all(item["passed"] is None and item["status"] == "insufficient_coverage" for item in sanity.values())
    assert {w["type"] for w in warnings} == {"slot_sanity_insufficient_coverage"}
    assert list(builder.SLOT_NAMES) == ["front", "left_front", "left_rear", "right_front", "right_rear"]


def test_nuplan_slot_sanity_fails_covered_wrong_direction():
    import tools.build_nuplan_5neighbor_context_dataset as builder

    stats = {slot: {"coverage_ratio": 0.0, "median_rel_x": None, "median_rel_y": None} for slot in builder.SLOT_NAMES}
    stats["right_front"] = {"coverage_ratio": 0.5, "median_rel_x": 10.0, "median_rel_y": 2.0}
    sanity, passed, evaluated, skipped, warnings = builder.evaluate_slot_sanity(stats, 0.05)
    assert passed is False
    assert evaluated == ["right_front"]
    assert "right_front" not in skipped
    assert sanity["right_front_median_rel_y_lt_0"]["passed"] is False


def test_nuplan_slot_sanity_rejects_invalid_coverage_threshold():
    import pytest
    import tools.build_nuplan_5neighbor_context_dataset as builder

    with pytest.raises(ValueError):
        builder.evaluate_slot_sanity({}, 1.5)
