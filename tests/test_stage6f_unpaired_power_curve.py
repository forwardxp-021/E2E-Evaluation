from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import stage6e_calibrate_unpaired_release as stage6e
from tools import stage6f_unpaired_power_curve as stage6f


def make_config(*, sample_sizes=(10, 20), trials: int = 5) -> dict:
    return {
        "row_id_column": "global_row",
        "pair_id_column": "scenario_token",
        "cluster_column": "log_name",
        "planner_column": "planner_name",
        "planners": {"assertive": "planner_a", "conservative": "planner_b"},
        "invariant_pair_columns": ["map_name", "scenario_type"],
        "covariates": [
            {"name": "map_name", "kind": "categorical", "timing": "pre_treatment"},
            {"name": "scenario_type", "kind": "categorical", "timing": "pre_treatment"},
        ],
        "tasks": [
            {
                "name": "task_one",
                "column": "scenario_type",
                "positive_values": ["task_one"],
                "timing": "pre_treatment",
            }
        ],
        "support_thresholds": {
            "min_support_fraction_per_group": 0.5,
            "min_ess_ratio_per_group": 0.3,
            "max_weight_ratio": 12.0,
            "min_clusters_per_group": 2,
        },
        "sample_sizes_per_release": list(sample_sizes),
        "seed": 128,
        "alpha": 0.1,
        "target_detection_rate": 0.8,
        "target_false_positive_rate": 0.1,
        "calibration_trials_per_planner": trials,
        "evaluation_trials_per_planner": trials,
        "ab_trials_per_direction": trials,
        "split_search_candidates": 12,
        "max_mmd_samples": 1000,
        "minimum_valid_trials": max(4, trials // 2),
    }


def make_dataset(logs: int = 50, shift: float = 3.0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(551)
    rows = []
    embeddings = []
    for index in range(logs):
        base = rng.normal(0.0, 0.6, size=6)
        scenario_type = "task_one" if index % 2 == 0 else "task_two"
        for planner, delta in [("planner_a", shift), ("planner_b", 0.0)]:
            rows.append(
                {
                    "global_row": len(rows),
                    "scenario_token": f"scenario_{index:03d}",
                    "log_name": f"log_{index:03d}",
                    "planner_name": planner,
                    "map_name": "map_x",
                    "scenario_type": scenario_type,
                }
            )
            value = base.copy()
            if delta:
                value[:3] += delta
            embeddings.append(value)
    return pd.DataFrame(rows), np.asarray(embeddings, dtype=np.float64)


def write_inputs(tmp_path: Path, config: dict, metadata: pd.DataFrame, embeddings: np.ndarray):
    embedding_path = tmp_path / "embedding.npy"
    metadata_path = tmp_path / "metadata.csv"
    config_path = tmp_path / "config.json"
    np.save(embedding_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return embedding_path, metadata_path, config_path


def test_power_config_requires_increasing_sizes() -> None:
    config = make_config(sample_sizes=(20, 10))
    with pytest.raises(ValueError, match="strictly increasing"):
        stage6f.validate_power_config(config)


def test_disjoint_size_split_is_reproducible_and_on_target() -> None:
    config = stage6f.validate_power_config(make_config())
    metadata, embeddings = make_dataset(logs=50)
    paired = stage6e.validate_paired_metadata(metadata, config, len(embeddings))
    inventory = stage6e.build_pair_inventory(paired, config)
    first = stage6f.choose_disjoint_logs_for_size(
        inventory, "log_name", 20, np.random.default_rng(8), 12
    )
    second = stage6f.choose_disjoint_logs_for_size(
        inventory, "log_name", 20, np.random.default_rng(8), 12
    )
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert abs(first[2]["selected_scenarios_A"] - 20) <= 1
    assert abs(first[2]["selected_scenarios_B"] - 20) <= 1


def test_split_rejects_unavailable_sample_size() -> None:
    config = stage6f.validate_power_config(make_config())
    metadata, embeddings = make_dataset(logs=30)
    paired = stage6e.validate_paired_metadata(metadata, config, len(embeddings))
    inventory = stage6e.build_pair_inventory(paired, config)
    with pytest.raises(ValueError, match="requires at least"):
        stage6f.choose_disjoint_logs_for_size(
            inventory, "log_name", 20, np.random.default_rng(3), 8
        )


def test_sequential_full_pool_split_supports_exact_half_with_multi_scenario_logs() -> None:
    rows = []
    for index, count in enumerate([3, 3, 2, 2, 1, 1, 1, 1, 1, 1]):
        for scenario in range(count):
            rows.append(
                {
                    "scenario_token": f"s_{index}_{scenario}",
                    "log_name": f"log_{index}",
                    "_support_cell": "cell_a" if index % 2 == 0 else "cell_b",
                }
            )
    inventory = pd.DataFrame(rows)
    logs_a, logs_b, audit = stage6f.choose_disjoint_logs_for_size(
        inventory,
        "log_name",
        8,
        np.random.default_rng(29),
        64,
        "sequential_full_log_pool_v1",
    )
    assert set(logs_a).isdisjoint(logs_b)
    assert audit["selected_scenarios_A"] == 8
    assert audit["selected_scenarios_B"] == 8
    assert audit["log_split_strategy"] == "sequential_full_log_pool_v1"


def test_end_to_end_power_curve_outputs_and_detection(tmp_path: Path) -> None:
    config = make_config(sample_sizes=(10, 20), trials=5)
    metadata, embeddings = make_dataset(logs=50, shift=3.0)
    embedding_path, metadata_path, config_path = write_inputs(tmp_path, config, metadata, embeddings)
    output_dir = tmp_path / "output"
    summary = stage6f.run(
        argparse.Namespace(
            embedding_path=embedding_path,
            metadata_csv=metadata_path,
            config_json=config_path,
            output_dir=output_dir,
            paired_oracle_json=None,
        )
    )
    assert summary["status"] == stage6f.COMPLETE_STATUS
    assert summary["input_audit"]["all_trial_log_overlap_zero"]
    assert summary["input_audit"]["all_trial_scenario_overlap_zero"]
    assert summary["input_audit"]["all_actual_sample_sizes_within_one"]
    assert summary["threshold_audit"]["all_overall_thresholds_pass"]
    operating = pd.read_csv(output_dir / "power_curve_operating_characteristics.csv")
    overall = operating.loc[operating["scope"] == "overall"]
    assert set(overall["target_scenarios_per_release"]) == {10, 20}
    assert (overall["ab_detection_rate"] > overall["aa_false_positive_rate"]).all()
    thresholds = pd.read_csv(output_dir / "power_curve_aa_thresholds.csv")
    assert set(thresholds["target_scenarios_per_release"]) == {10, 20}
    expected = {
        "power_curve_trial_bdd.csv",
        "power_curve_split_audit.csv",
        "power_curve_log_assignments.csv",
        "power_curve_aa_thresholds.csv",
        "power_curve_detection_summary.csv",
        "power_curve_operating_characteristics.csv",
        "fixed_scope_bandwidths.csv",
        "overall_unpaired_power_curve.png",
        "overall_unpaired_power_curve.pdf",
        "stage6f_power_curve_summary.json",
        "stage6f_reproducibility_provenance.json",
        "stage6f_power_curve_report.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})


def test_sufficiency_gate_does_not_extrapolate() -> None:
    operating = pd.DataFrame(
        [
            {
                "target_scenarios_per_release": 100,
                "scope": "overall",
                "aa_false_positive_rate": 0.04,
                "aa_false_positive_wilson95_low": 0.02,
                "aa_false_positive_wilson95_high": 0.08,
                "ab_detection_rate": 0.55,
                "ab_detection_wilson95_low": 0.48,
                "ab_detection_wilson95_high": 0.62,
            },
            {
                "target_scenarios_per_release": 150,
                "scope": "overall",
                "aa_false_positive_rate": 0.03,
                "aa_false_positive_wilson95_low": 0.01,
                "aa_false_positive_wilson95_high": 0.06,
                "ab_detection_rate": 0.70,
                "ab_detection_wilson95_low": 0.63,
                "ab_detection_wilson95_high": 0.76,
            },
        ]
    )
    config = {"target_detection_rate": 0.8, "target_false_positive_rate": 0.05}
    result = stage6f.build_sufficiency_summary(operating, config)
    assert result["status"] == stage6f.TARGET_NOT_REACHED
    assert result["minimum_observed_sample_size_meeting_confidence_targets"] is None
    assert result["extrapolation"] == "FORBIDDEN_OUTSIDE_OBSERVED_SAMPLE_SIZE_RANGE"


def test_full_power_curve_is_reproducible_with_fixed_seed(tmp_path: Path) -> None:
    config = make_config(sample_sizes=(10,), trials=4)
    metadata, embeddings = make_dataset(logs=30, shift=2.5)
    embedding_path, metadata_path, config_path = write_inputs(tmp_path, config, metadata, embeddings)
    outputs = []
    for name in ["first", "second"]:
        output_dir = tmp_path / name
        stage6f.run(
            argparse.Namespace(
                embedding_path=embedding_path,
                metadata_csv=metadata_path,
                config_json=config_path,
                output_dir=output_dir,
                paired_oracle_json=None,
            )
        )
        outputs.append(pd.read_csv(output_dir / "power_curve_trial_bdd.csv"))
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
