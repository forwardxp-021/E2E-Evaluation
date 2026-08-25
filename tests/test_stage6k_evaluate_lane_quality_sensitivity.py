from __future__ import annotations

import numpy as np

from tools.stage6k_evaluate_lane_quality_sensitivity import (
    finite_spearman,
    log_cluster_bootstrap_interval,
    task_adjusted_rank_correlation,
)


def test_task_adjusted_rank_removes_task_offsets() -> None:
    quality = np.asarray([1.0, 2.0, 10.0, 20.0])
    distance = np.asarray([10.0, 20.0, 1.0, 2.0])
    tasks = ["a", "a", "b", "b"]
    assert finite_spearman(quality, distance) < 0
    assert task_adjusted_rank_correlation(quality, distance, tasks) > 0.99


def test_cluster_interval_is_reproducible() -> None:
    quality = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    distance = quality.copy()
    tasks = ["a", "a", "a", "b", "b", "b"]
    logs = ["x", "x", "y", "y", "z", "z"]
    first = log_cluster_bootstrap_interval(quality, distance, tasks, logs, adjusted=False, repetitions=100, seed=9)
    second = log_cluster_bootstrap_interval(quality, distance, tasks, logs, adjusted=False, repetitions=100, seed=9)
    assert first == second
    assert first[0] > 0.9
