import numpy as np
import pytest

from tools.stage6l_freeze_context_representation_ablation import context_record
from tools.stage6l_prepare_context_representation_ablation import (
    apply_scaler,
    ego_kinematic_features,
    fit_reference_scaler,
)
from tools.stage6l_run_context_representation_ablation import kernel_analysis
from tools.stage7_m6_scenario_conditioned_bdd import biased_mmd2_from_kernel, rbf_kernel


def test_ego_kinematic_features_are_mask_aware() -> None:
    ego = np.zeros((1, 5, 8), dtype=np.float32)
    ego[0, :4, 0] = [0.0, 1.0, 3.0, 6.0]
    ego[0, :4, 5] = [0.0, 1.0, 3.0, 6.0]
    ego[0, :4, 4] = [0.0, 0.1, 0.3, 0.6]
    ego[0, 4] = 1e6
    mask = np.array([[True, True, True, True, False]])
    features = ego_kinematic_features(ego, mask, dt=1.0)
    assert features.shape == (1, 13)
    assert np.isclose(features[0, 0], 2.5)
    assert np.isclose(features[0, 3], 6.0)
    assert np.isclose(features[0, -1], 6.0)
    assert features.max() < 100.0


def test_reference_scaler_imputes_nonfinite_and_uses_floor() -> None:
    reference = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    median, scale = fit_reference_scaler(reference, 1e-6)
    assert np.allclose(median, [1.0, 4.0])
    assert np.isclose(scale[0], 1e-6)
    transformed = apply_scaler(np.array([[np.nan, 6.0]]), median, scale)
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed, [[0.0, 1.0]])


def test_fast_paired_kernel_statistic_and_contributions_match_direct_mmd() -> None:
    a = np.array([[0.0], [1.0], [3.0]])
    b = np.array([[0.2], [1.5], [2.0]])
    result, samples, contribution = kernel_analysis(a, b, repetitions=31, seed=7)
    pooled = np.vstack([a, b])
    kernel = rbf_kernel(pooled, float(result["bandwidth"]))
    direct = biased_mmd2_from_kernel(kernel, np.arange(3), np.arange(3, 6))
    assert np.isclose(result["mmd2"], direct)
    assert np.isclose(contribution.mean(), direct)
    assert samples.shape == (31,)
    assert 1 / 32 <= result["raw_p"] <= 1.0


def test_formal_freeze_rejects_zero_semantic_neighbor_coverage(tmp_path) -> None:
    for name in [
        "context_traj.npy",
        "ego_seq.npy",
        "ego_seq_mask.npy",
        "interaction_feat_style.npy",
        "metadata.csv",
        "feature_schema.json",
        "stage5d_context_schema.json",
        "nuplan_lane_assignment_by_row.csv",
        "warnings.json",
    ]:
        (tmp_path / name).write_bytes(b"{}" if name.endswith(".json") else b"")
    np.save(tmp_path / "neighbor_seq.npy", np.zeros((366, 5, 150, 15), dtype=np.float32))

    with pytest.raises(ValueError, match="zero semantic-neighbor coverage"):
        context_record(tmp_path)
