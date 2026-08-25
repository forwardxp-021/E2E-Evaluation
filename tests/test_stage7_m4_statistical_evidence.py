import numpy as np

from tools.stage7_m4_build_statistical_evidence import (
    endpoint_statistics,
    holm_adjust,
)


def test_holm_adjust_is_monotone_in_sorted_p_order() -> None:
    adjusted = holm_adjust([0.04, 0.001, 0.02])
    assert adjusted[1] == 0.003
    assert adjusted[2] == 0.04
    assert adjusted[0] == 0.04


def test_endpoint_statistics_orients_lower_is_hypothesized_direction() -> None:
    result = endpoint_statistics(
        [-3.0, -2.0, -1.0, -4.0],
        direction=-1,
        bootstrap_repetitions=200,
        seed=7,
    )
    assert result["mean_delta"] == -2.5
    assert result["positive_direction_count"] == 4
    assert result["opposite_direction_count"] == 0
    assert result["paired_cohen_dz_oriented"] > 0


def test_endpoint_bootstrap_is_deterministic() -> None:
    values = np.arange(1.0, 8.0)
    first = endpoint_statistics(values, direction=1, bootstrap_repetitions=200, seed=11)
    second = endpoint_statistics(values, direction=1, bootstrap_repetitions=200, seed=11)
    assert first["mean_ci95_low"] == second["mean_ci95_low"]
    assert first["mean_ci95_high"] == second["mean_ci95_high"]
