from __future__ import annotations

import numpy as np

from tools.stage6k_run_longitudinal_dose_bdd import (
    apply_frozen_multiplicity,
    cluster_randomization_test,
    null_diagnostics,
)


def test_null_diagnostics_uses_dose_specific_null() -> None:
    result = null_diagnostics(4.0, np.asarray([1.0, 2.0, 3.0]))
    assert result["paired_null_mean"] == 2.0
    assert result["paired_null_sd"] == 1.0
    assert result["bdd_to_null_q95_ratio"] > 1.0
    assert result["null_standardized_z_bdd"] == 2.0


def test_cluster_flip_is_reproducible_and_keeps_cluster_count() -> None:
    a = np.asarray([[0.0], [0.2], [0.4], [0.6]])
    b = np.asarray([[1.0], [1.2], [1.4], [1.6]])
    logs = ["x", "x", "y", "z"]
    first, first_samples = cluster_randomization_test(
        a, b, logs, repetitions=64, seed=7, progress_label="test"
    )
    second, second_samples = cluster_randomization_test(
        a, b, logs, repetitions=64, seed=7, progress_label="test"
    )
    assert first["n_clusters"] == 3
    assert first["raw_p"] == second["raw_p"]
    np.testing.assert_array_equal(first_samples, second_samples)


def test_multiplicity_uses_four_overall_and_twelve_tasks() -> None:
    rows = []
    for index, label in enumerate(["dose25", "dose50", "dose75", "dose100"], start=1):
        dose = index / 4
        rows.append({"dose_label": label, "nominal_dose": dose, "scope": "overall", "raw_p": 0.001 * index})
        for task in ["a", "b", "c"]:
            rows.append({"dose_label": label, "nominal_dose": dose, "scope": task, "raw_p": 0.001 * index})
    minimum, adjusted = apply_frozen_multiplicity(rows, {"dose25": False, "dose50": True, "dose75": True, "dose100": True})
    assert minimum == 0.5
    assert len([row for row in adjusted if row["reject_holm_0_05"]]) == 16
