from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import stage6d_unpaired_version_bdd as stage6d
from tools import stage6e_calibrate_unpaired_release as stage6e


def make_config(*, trials: int = 12) -> dict:
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
        "post_treatment_columns": ["hard_brake"],
        "support_thresholds": {
            "min_support_fraction_per_group": 0.7,
            "min_ess_ratio_per_group": 0.4,
            "max_weight_ratio": 10.0,
            "min_clusters_per_group": 2,
        },
        "seed": 71,
        "alpha": 0.1,
        "calibration_trials_per_planner": trials,
        "evaluation_trials_per_planner": trials,
        "ab_trials_per_direction": trials,
        "split_search_candidates": 8,
        "max_mmd_samples": 1000,
        "minimum_valid_trials": max(4, trials // 2),
    }


def make_dataset(logs: int = 40, shift: float = 2.5) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(918)
    rows = []
    embeddings = []
    for index in range(logs):
        base = rng.normal(0.0, 0.7, size=6)
        scenario_type = "task_one" if index % 2 == 0 else "task_two"
        for planner, delta in [("planner_a", shift), ("planner_b", 0.0)]:
            row_id = len(rows)
            rows.append(
                {
                    "global_row": row_id,
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


def test_validate_paired_metadata_rejects_missing_planner_row() -> None:
    config = stage6e.validate_config(make_config())
    metadata, embeddings = make_dataset(logs=8)
    bad = metadata.iloc[:-1].copy()
    with pytest.raises(ValueError, match="paired metadata audit failed"):
        stage6e.validate_paired_metadata(bad, config, len(embeddings))


def test_log_split_is_disjoint_and_reproducible() -> None:
    config = stage6e.validate_config(make_config())
    metadata, embeddings = make_dataset(logs=20)
    paired = stage6e.validate_paired_metadata(metadata, config, len(embeddings))
    inventory = stage6e.build_pair_inventory(paired, config)
    split_1 = stage6e.choose_log_split(inventory, "log_name", np.random.default_rng(9), 12)
    split_2 = stage6e.choose_log_split(inventory, "log_name", np.random.default_rng(9), 12)
    assert split_1 == split_2
    assert set(split_1[0]).isdisjoint(split_1[1])
    assert set(split_1[0]) | set(split_1[1]) == set(inventory["log_name"])


def test_no_common_support_is_not_comparable() -> None:
    raw_config = make_config()
    config = stage6e.validate_config(raw_config)
    metadata, embeddings = make_dataset(logs=8)
    metadata.loc[metadata["log_name"].isin(["log_000", "log_001", "log_002", "log_003"]), "map_name"] = "map_a"
    metadata.loc[metadata["log_name"].isin(["log_004", "log_005", "log_006", "log_007"]), "map_name"] = "map_b"
    paired = stage6e.validate_paired_metadata(metadata, config, len(embeddings))
    trial, trial_embeddings, audit = stage6e.build_trial(
        paired,
        embeddings,
        config,
        ["log_000", "log_001", "log_002", "log_003"],
        ["log_004", "log_005", "log_006", "log_007"],
        "planner_a",
        "planner_b",
    )
    bandwidths = {"overall": 1.0, "task_one": 1.0}
    scopes = stage6e.evaluate_trial_scopes(trial, trial_embeddings, config, bandwidths, 44)
    assert audit["log_overlap_count"] == 0
    assert audit["scenario_overlap_count"] == 0
    assert scopes[0]["status"] == stage6d.NOT_COMPARABLE_STATUS
    assert np.isnan(scopes[0]["standardized_mmd2"])


def test_end_to_end_aa_threshold_and_ab_detection(tmp_path: Path) -> None:
    config = make_config(trials=12)
    metadata, embeddings = make_dataset(logs=40, shift=2.5)
    embedding_path, metadata_path, config_path = write_inputs(tmp_path, config, metadata, embeddings)
    output_dir = tmp_path / "output"
    args = argparse.Namespace(
        embedding_path=embedding_path,
        metadata_csv=metadata_path,
        config_json=config_path,
        output_dir=output_dir,
        paired_oracle_json=None,
    )
    summary = stage6e.run(args)
    assert summary["status"] == stage6e.PASS_STATUS
    assert summary["input_audit"]["all_trial_log_overlap_zero"]
    assert summary["input_audit"]["all_trial_scenario_overlap_zero"]
    assert summary["descriptive_evidence_conclusion"] == (
        "AB_SEPARATED_FROM_AA_WITH_STRONG_SINGLE_RELEASE_SENSITIVITY"
    )
    detection = pd.read_csv(output_dir / "release_detection_summary.csv")
    overall = detection.loc[detection["scope"] == "overall"]
    aa_rate = overall.loc[overall["family"] == "AA_EVALUATION", "exceedance_rate"].mean()
    ab_rate = overall.loc[overall["family"] == "AB_EVALUATION", "exceedance_rate"].mean()
    assert ab_rate > aa_rate
    assert ab_rate >= 0.8
    operating = pd.read_csv(output_dir / "release_operating_characteristics.csv")
    overall_operating = operating.loc[operating["scope"] == "overall"].iloc[0]
    assert overall_operating["ab_detection_rate"] > overall_operating["aa_false_positive_rate"]
    assert overall_operating["scope_role"] == "PRIMARY_OVERALL"
    expected = {
        "release_trial_bdd.csv",
        "release_trial_split_audit.csv",
        "release_trial_log_assignments.csv",
        "aa_empirical_thresholds.csv",
        "release_detection_summary.csv",
        "release_operating_characteristics.csv",
        "fixed_scope_bandwidths.csv",
        "stage6e_release_emulation_summary.json",
        "stage6e_reproducibility_provenance.json",
        "stage6e_release_emulation_report.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})


def test_fixed_seed_end_to_end_reproducibility(tmp_path: Path) -> None:
    config = make_config(trials=5)
    metadata, embeddings = make_dataset(logs=24, shift=2.0)
    embedding_path, metadata_path, config_path = write_inputs(tmp_path, config, metadata, embeddings)
    frames = []
    for name in ["one", "two"]:
        args = argparse.Namespace(
            embedding_path=embedding_path,
            metadata_csv=metadata_path,
            config_json=config_path,
            output_dir=tmp_path / name,
            paired_oracle_json=None,
        )
        stage6e.run(args)
        frames.append(pd.read_csv(tmp_path / name / "release_trial_bdd.csv"))
    pd.testing.assert_frame_equal(frames[0], frames[1])
