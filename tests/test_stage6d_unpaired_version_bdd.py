from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import stage6d_unpaired_version_bdd as stage6d


def design() -> dict:
    return stage6d.validate_design(
        {
            "group_column": "version",
            "groups": {"A": "old", "B": "new"},
            "row_id_column": "global_row",
            "cluster_column": "trip_id",
            "reference_distribution": "equal_group_pooled_common_support",
            "covariates": [
                {"name": "city", "kind": "categorical", "timing": "pre_treatment"},
                {"name": "speed_limit", "kind": "continuous", "bins": 2, "timing": "pre_treatment"},
            ],
            "tasks": [
                {
                    "name": "following_opportunity",
                    "column": "following_opportunity",
                    "positive_values": [1],
                    "timing": "pre_treatment",
                }
            ],
            "post_treatment_columns": ["actual_lane_change", "hard_brake"],
            "thresholds": {
                "min_support_fraction_per_group": 0.5,
                "min_ess_ratio_per_group": 0.1,
                "max_weight_ratio": 20.0,
                "min_clusters_per_group": 2,
            },
        }
    )


def synthetic_frame(*, true_shift: float = 0.0, no_overlap: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260809)
    rows = []
    embeddings = []
    global_row = 0
    for version in ["old", "new"]:
        if no_overlap:
            counts = {"city_0": 100, "city_1": 0} if version == "old" else {"city_0": 0, "city_1": 100}
        else:
            counts = {"city_0": 90, "city_1": 10} if version == "old" else {"city_0": 10, "city_1": 90}
        version_shift = true_shift if version == "new" else 0.0
        within_version_position = 0
        for city, count in counts.items():
            city_effect = 0.0 if city == "city_0" else 3.0
            for _ in range(count):
                rows.append(
                    {
                        "global_row": global_row,
                        "version": version,
                        "trip_id": f"{version}_trip_{within_version_position // 5}",
                        "city": city,
                        "speed_limit": 10.0 if city == "city_0" else 20.0,
                        "following_opportunity": int(within_version_position % 2 == 0),
                    }
                )
                embeddings.append(
                    [city_effect + version_shift + rng.normal(0, 0.12), rng.normal(0, 0.12)]
                )
                global_row += 1
                within_version_position += 1
    frame = pd.DataFrame(rows)
    frame, _ = stage6d.coarsen_covariates(frame, design())
    return frame, np.asarray(embeddings, dtype=np.float64)


def test_design_rejects_post_treatment_matching_and_tasks() -> None:
    value = design()
    value["covariates"][0]["timing"] = "post_treatment"
    with pytest.raises(ValueError, match="pre_treatment"):
        stage6d.validate_design(value)

    value = design()
    value["tasks"][0]["timing"] = "post_treatment"
    with pytest.raises(ValueError, match="pre_treatment"):
        stage6d.validate_design(value)


def test_common_support_standardization_removes_pure_scene_composition_shift() -> None:
    frame, embeddings = synthetic_frame()
    result = stage6d.analyze_scope(
        frame,
        embeddings,
        design(),
        scope="overall",
        repetitions=0,
        seed=11,
        max_samples=1000,
    )
    assert result["status"] == stage6d.PASS_STATUS
    assert result["raw_mmd2"] > 0.3
    assert result["standardized_mmd2"] < result["raw_mmd2"] * 0.05


def test_standardization_preserves_true_within_odd_version_shift() -> None:
    frame, embeddings = synthetic_frame(true_shift=1.2)
    result = stage6d.analyze_scope(
        frame,
        embeddings,
        design(),
        scope="overall",
        repetitions=0,
        seed=12,
        max_samples=1000,
    )
    assert result["status"] == stage6d.PASS_STATUS
    assert result["standardized_mmd2"] > 0.05


def test_no_common_support_fails_closed() -> None:
    frame, _ = synthetic_frame(no_overlap=True)
    result = stage6d.build_standardization(frame, design())
    assert result["status"] == stage6d.NOT_COMPARABLE_STATUS
    assert result["checks"]["common_cells_nonempty"] is False
    assert np.all(result["weights"] == 0)


def test_cluster_bootstrap_is_fixed_seed_reproducible() -> None:
    frame, embeddings = synthetic_frame(true_shift=0.5)
    kwargs = dict(
        frame=frame,
        embeddings=embeddings,
        design=design(),
        scope="overall",
        repetitions=8,
        seed=99,
        max_samples=1000,
    )
    first = stage6d.analyze_scope(**kwargs)
    second = stage6d.analyze_scope(**kwargs)
    assert first["bootstrap_rows"] == second["bootstrap_rows"]
    assert first["standardized_cluster_bootstrap_ci95_low"] == second["standardized_cluster_bootstrap_ci95_low"]


def test_end_to_end_writes_auditable_outputs(tmp_path: Path) -> None:
    frame, embeddings = synthetic_frame(true_shift=0.4)
    metadata = frame.drop(columns=[c for c in frame.columns if c.startswith("_cell") or c == "_support_cell"])
    embedding_path = tmp_path / "embedding.npy"
    metadata_path = tmp_path / "metadata.csv"
    design_path = tmp_path / "design.json"
    output_dir = tmp_path / "output"
    np.save(embedding_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    design_path.write_text(json.dumps(design()), encoding="utf-8")

    summary = stage6d.run(
        Namespace(
            embedding_path=embedding_path,
            metadata_csv=metadata_path,
            design_json=design_path,
            output_dir=output_dir,
            bootstrap_repetitions=5,
            max_mmd_samples=1000,
            seed=21,
        )
    )
    assert summary["status"] == stage6d.PASS_STATUS
    expected = {
        "common_support_cells.csv",
        "standardization_row_weights.csv",
        "covariate_balance.csv",
        "task_frequency_shift.csv",
        "overall_bdd_summary.csv",
        "task_bdd_summary.csv",
        "cluster_bootstrap_mmd_samples.csv",
        "stage6d_unpaired_version_summary.json",
        "stage6d_reproducibility_provenance.json",
        "stage6d_unpaired_version_report.md",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert len(pd.read_csv(output_dir / "cluster_bootstrap_mmd_samples.csv")) == 20
