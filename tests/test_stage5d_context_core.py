from pathlib import Path
import ast
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
