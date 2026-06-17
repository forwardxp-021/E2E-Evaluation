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
