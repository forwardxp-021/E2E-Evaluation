import numpy as np

from tools.stage7_m5_representation_mechanism_analysis import (
    minimum_detectable_paired_dz,
    paired_sign_flip_test,
    robust_standardize,
)


def test_robust_standardize_imputes_nonfinite_and_constant_columns() -> None:
    values = np.asarray([[1.0, np.nan, 5.0], [2.0, 3.0, 5.0], [4.0, 7.0, 5.0]])
    result = robust_standardize(values)
    assert np.isfinite(result).all()
    assert np.all(result[:, 2] == 0.0)


def test_paired_sign_flip_detects_consistent_shift() -> None:
    conservative = np.zeros((12, 2))
    assertive = np.ones((12, 2))
    values = np.vstack([assertive, conservative])
    pairs = np.asarray([(index, index + 12) for index in range(12)])
    result = paired_sign_flip_test(values, pairs, repetitions=2000, seed=3)
    assert np.isclose(result["paired_direction_concentration"], 1.0)
    assert result["sign_flip_p"] < 0.01


def test_mde_decreases_with_pair_count() -> None:
    assert minimum_detectable_paired_dz(
        90, alpha=0.05, power=0.8
    ) < minimum_detectable_paired_dz(45, alpha=0.05, power=0.8)
