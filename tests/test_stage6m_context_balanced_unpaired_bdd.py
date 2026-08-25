import numpy as np
import pandas as pd

from tools.stage6m_run_context_balanced_unpaired_bdd import (
    aggregate_trials,
    calibrate_and_evaluate,
    paired_method_comparisons,
)


TASKS = [
    "following_interaction",
    "lane_change",
    "stop_go_control",
    "high_motion_dynamics",
    "dense_or_vulnerable_interaction",
]


def design() -> dict:
    return {
        "alpha": 0.05,
        "calibration_quantile_method": "higher",
        "minimum_valid_calibration_trials": 1,
        "tasks": [
            {"name": task, "weight": weight}
            for task, weight in zip(TASKS, [0.2, 0.1, 0.2, 0.2, 0.3])
        ],
    }


def trial_rows(experiment_set: str, family: str, repetition: int, offset: float) -> list[dict]:
    rows = []
    for position, scope in enumerate(["overall", *TASKS]):
        rows.append(
            {
                "target_scenarios_per_release": 200,
                "experiment_set": experiment_set,
                "family": family,
                "repetition": repetition,
                "split_seed": repetition,
                "planner_A": "a",
                "planner_B": "b",
                "scope": scope,
                "raw_mmd2": offset + position,
                "standardized_mmd2": offset + position + 0.5,
                "status": "PASS_DESCRIPTIVE_STANDARDIZED_VERSION_DRIFT",
                "n_A": 200,
                "n_B": 200,
                "support_fraction_A": 0.9,
                "support_fraction_B": 0.8,
                "ess_ratio_A": 0.7,
                "ess_ratio_B": 0.6,
                "max_weight_ratio_A": 2.0,
                "max_weight_ratio_B": 3.0,
            }
        )
    return rows


def test_task_statistics_use_frozen_full_pool_weights() -> None:
    aggregated = aggregate_trials(
        pd.DataFrame(trial_rows("AA_CALIBRATION_ASSERTIVE", "AA_CALIBRATION", 0, 1.0)),
        design(),
    )
    task = aggregated.set_index("method").loc["task_conditioned"]
    expected = np.dot([0.2, 0.1, 0.2, 0.2, 0.3], [2, 3, 4, 5, 6])
    assert np.isclose(task["statistic"], expected)
    combined = aggregated.set_index("method").loc["task_context_balanced"]
    assert np.isclose(combined["statistic"], expected + 0.5)
    assert combined["support_fraction_B_min"] == 0.8
    assert combined["max_weight_ratio_B_max"] == 3.0


def test_threshold_is_calibration_only_and_mcnemar_counts_are_paired() -> None:
    rows = []
    rows += trial_rows("AA_CALIBRATION_ASSERTIVE", "AA_CALIBRATION", 0, 0.0)
    rows += trial_rows("AA_EVALUATION_ASSERTIVE", "AA_EVALUATION", 1, 1.0)
    rows += trial_rows("AB_ASSERTIVE_TO_CONSERVATIVE", "AB_EVALUATION", 2, 2.0)
    aggregated = aggregate_trials(pd.DataFrame(rows), design())
    thresholds, evaluated = calibrate_and_evaluate(aggregated, design())
    assert len(thresholds) == 4
    assert set(thresholds["valid_calibration_trials"]) == {1}
    comparisons = paired_method_comparisons(evaluated)
    assert len(comparisons) == 3
    assert set(comparisons["paired_valid_trials"]) == {1}
