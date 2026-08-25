import numpy as np
import pandas as pd
import pytest

from tools.stage6c_build_behavior_events_v2 import apply_rollout_validity_mask, derive_row, parse_args
from tools.stage7f_aggressive_conservative_paired_delta import row_metrics
from tools.interaction_context_features import aggregate_interaction_features


def test_rollout_validity_mask_removes_padded_derivative_boundary():
    ego = np.zeros((5, 8), dtype=np.float32)
    ego[:4, 3] = [0.0, 1.0, 2.0, 3.0]
    ego[4, 3] = -999.0
    neighbor = np.zeros((5, 5, 15), dtype=np.float32)
    mask = np.asarray([True, True, True, True, False])

    trimmed_ego, trimmed_neighbor, valid_count = apply_rollout_validity_mask(ego, neighbor, mask)

    assert valid_count == 4
    assert trimmed_ego.shape == (4, 8)
    assert trimmed_neighbor.shape == (5, 4, 15)
    lateral_accel = np.diff(trimmed_ego[:, 3], prepend=trimmed_ego[0, 3]) / 0.1
    assert np.max(np.abs(lateral_accel)) == pytest.approx(10.0)


def test_rollout_validity_mask_rejects_noncontiguous_valid_frames():
    ego = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        apply_rollout_validity_mask(
            ego,
            None,
            np.asarray([True, False, True, False]),
        )


def test_missing_rollout_validity_mask_preserves_backward_compatibility():
    ego = np.ones((3, 8), dtype=np.float32)
    trimmed_ego, trimmed_neighbor, valid_count = apply_rollout_validity_mask(ego, None, None)
    assert trimmed_ego is ego
    assert trimmed_neighbor is None
    assert valid_count == 3


def test_paired_delta_metrics_ignore_padded_frames():
    ego = np.zeros((1, 4, 8), dtype=np.float32)
    ego[0, :, 5] = [2.0, 4.0, 6.0, 100.0]
    ego[0, :, 6] = [0.0, 1.0, 1.0, 99.0]
    mask = np.asarray([[True, True, True, False]])

    metrics = row_metrics(0, None, ego, None, mask, pd.Series(dtype=object))

    assert metrics["mean_speed"] == pytest.approx(4.0)
    assert metrics["max_speed"] == pytest.approx(6.0)
    assert metrics["max_abs_accel"] == pytest.approx(1.0)


def test_stage6c_raw_diagnostics_ignore_padded_boundary(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stage6c_build_behavior_events_v2.py",
            "--shard_manifest", "unused.json",
            "--feature_schema_path", "unused_schema.json",
            "--output_dir", "unused_output",
        ],
    )
    args = parse_args()
    ego = np.zeros((5, 8), dtype=np.float32)
    ego[:, 3] = [0.0, 0.2, 0.4, 0.6, -99.0]
    mask = np.asarray([True, True, True, True, False])

    _, _, _, raw = derive_row(ego, None, False, args, validity_mask=mask)

    assert raw["raw_max_abs_lateral_accel"] == pytest.approx(2.0)
    assert np.isnan(raw["raw_max_abs_curvature"])


def test_curvature_features_exclude_near_standstill_division():
    ego = np.zeros((4, 8), dtype=np.float32)
    ego[:, 5] = 0.01
    ego[:, 7] = 0.2
    neighbor = np.zeros((5, 4, 15), dtype=np.float32)

    features, names = aggregate_interaction_features(ego, neighbor, 0.1)

    assert features[names.index("rms_curvature")] == pytest.approx(0.0)
