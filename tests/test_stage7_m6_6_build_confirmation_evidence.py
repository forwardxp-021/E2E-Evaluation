from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from tools import stage7_m6_6_build_confirmation_evidence as evidence


def test_bootstrap_intervals_are_deterministic_and_contain_estimate() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    first = evidence.bootstrap_mean_ci(values, repetitions=2000, seed=17)
    second = evidence.bootstrap_mean_ci(values, repetitions=2000, seed=17)
    assert first == second
    assert first[0] < values.mean() < first[1]

    x = np.arange(20, dtype=np.float64)
    y = x + np.sin(x)
    low, high = evidence.bootstrap_rank_correlation_ci(
        x, y, repetitions=2000, seed=23
    )
    assert low <= spearmanr(x, y).statistic <= high


def test_task_adjustment_removes_between_task_rank_confounding() -> None:
    rng = np.random.default_rng(20260808)
    tasks = np.repeat(evidence.TASK_ORDER, 40)
    task_level = np.repeat(np.arange(len(evidence.TASK_ORDER)) * 100.0, 40)
    x = task_level + np.tile(np.arange(40), len(evidence.TASK_ORDER))
    y = task_level + np.concatenate([rng.permutation(40) for _ in evidence.TASK_ORDER])

    assert spearmanr(x, y).statistic > 0.9
    adjusted = evidence.task_adjusted_rank_association(x, y, tasks)
    assert abs(adjusted["rho"]) < 0.15
    assert len(adjusted["residual_x"]) == len(x)


def test_pair_task_mapping_uses_equal_pretreatment_scenario_type() -> None:
    scenario_type = evidence.PRETREATMENT_TASKS["following_interaction"][0]
    pair_audit = pd.DataFrame(
        [{"scenario_token": "scenario-1", "row_A": 0, "row_B": 1}]
    )
    metadata = pd.DataFrame(
        [
            {
                "global_row": 0,
                "scenario_token": "scenario-1",
                "scenario_type": scenario_type,
                "planner_name": "a",
                "parameters_json": "{}",
            },
            {
                "global_row": 1,
                "scenario_token": "scenario-1",
                "scenario_type": scenario_type,
                "planner_name": "b",
                "parameters_json": "{}",
            },
        ]
    )
    result = evidence.add_pair_tasks(pair_audit, metadata)
    assert result.loc[0, "task"] == "following_interaction"
    assert result.loc[0, "scenario_type"] == scenario_type

    metadata.loc[1, "scenario_type"] = evidence.PRETREATMENT_TASKS["lane_change"][0]
    with pytest.raises(ValueError, match="unequal pre-treatment"):
        evidence.add_pair_tasks(pair_audit, metadata)


def test_hash_audit_fails_closed_and_json_normalizes_numpy(tmp_path: Path) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("frozen", encoding="utf-8")
    record = {"path": str(locked), "sha256": evidence.sha256_file(locked)}
    evidence.validate_hash_record(record, "fixture")

    locked.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence.validate_hash_record(record, "fixture")

    output = tmp_path / "numpy.json"
    evidence.write_json(output, {"passed": np.bool_(True), "count": np.int64(3)})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "passed": True,
        "count": 3,
    }


def test_quality_attribution_emits_all_scopes_and_labels_post_treatment() -> None:
    rng = np.random.default_rng(13)
    rows = []
    for task in evidence.TASK_ORDER:
        for position in range(8):
            fallback = position / 10.0
            rows.append(
                {
                    "task": task,
                    "embedding_l2_distance": 2.0 * fallback + rng.normal(0, 0.05),
                    "max_pair_fallback_rate": fallback,
                    "fallback_rate_abs_delta": fallback / 2,
                    "max_pair_ambiguous_rate": 0.8 - fallback,
                    "ambiguous_rate_abs_delta": fallback / 3,
                }
            )
    result = evidence.quality_attribution_rows(
        pd.DataFrame(rows), repetitions=200, seed=29
    )
    assert len(result) == 4 * (2 + len(evidence.TASK_ORDER))
    assert {row["scope"] for row in result} == {
        "overall_stratified_bootstrap",
        "task_adjusted_rank_residual",
        "within_task",
    }
    assert all(row["role"] == "descriptive_exploratory_post_treatment" for row in result)
