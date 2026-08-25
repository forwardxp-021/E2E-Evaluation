from __future__ import annotations

import numpy as np

from tools import stage6j_evaluate_kinematic_gate as gate


def test_row_metrics_respects_mask_and_front_validity() -> None:
    ego = np.zeros((4, 8), dtype=np.float32)
    ego[:, 5] = [1.0, 3.0, 100.0, 100.0]
    ego[:, 6] = [1.0, 2.0, 100.0, 100.0]
    ego[:, 7] = [-0.1, 0.3, 100.0, 100.0]
    neighbor = np.zeros((5, 4, 15), dtype=np.float32)
    neighbor[0, :2, 0] = [1.0, 0.0]
    neighbor[0, :2, 5] = [12.0, 99.0]
    neighbor[0, :2, 10] = [1.5, 99.0]
    result = gate.row_metrics(ego, neighbor, np.asarray([1, 1, 0, 0], dtype=bool))
    assert result["mean_speed"] == 2.0
    assert np.isclose(result["rms_accel"], np.sqrt(2.5))
    assert result["rms_jerk"] == 10.0
    assert np.isclose(result["mean_abs_yaw_rate"], 0.2)
    assert result["mean_front_distance"] == 12.0
    assert result["mean_thw"] == 1.5
    assert result["front_valid_ratio"] == 0.5


def test_cluster_bootstrap_uses_log_clusters_deterministically() -> None:
    values = np.asarray([1.0, 3.0, 5.0])
    clusters = np.asarray(["log-a", "log-a", "log-b"])
    first = gate.cluster_bootstrap_mean_ci(
        values,
        clusters,
        repetitions=2000,
        seed=42,
        confidence_level=0.95,
    )
    second = gate.cluster_bootstrap_mean_ci(
        values,
        clusters,
        repetitions=2000,
        seed=42,
        confidence_level=0.95,
    )
    assert first == second
    assert first[2:] == (3, 2)
    assert first[0] <= np.mean(values) <= first[1]


def test_primary_gate_requires_both_frozen_metrics() -> None:
    config = {
        "primary_gate": {
            "metrics": {
                "delta_mean_speed": {
                    "expected_direction": "positive",
                    "minimum_one_sided_ci_bound": 0.5,
                },
                "delta_rms_accel": {
                    "expected_direction": "positive",
                    "minimum_one_sided_ci_bound": 0.1,
                },
            }
        }
    }
    contrasts = [
        {
            "scope": "overall",
            "metric": "delta_mean_speed",
            "mean_delta_A_minus_B": 1.0,
            "cluster_bootstrap_ci95_low": 0.6,
            "cluster_bootstrap_ci95_high": 1.4,
        },
        {
            "scope": "overall",
            "metric": "delta_rms_accel",
            "mean_delta_A_minus_B": 0.2,
            "cluster_bootstrap_ci95_low": 0.09,
            "cluster_bootstrap_ci95_high": 0.3,
        },
    ]
    passed, decisions = gate.evaluate_gate(contrasts, config)
    assert passed is False
    assert [item["pass"] for item in decisions] == [True, False]
