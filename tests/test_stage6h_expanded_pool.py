from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tools import stage6h_merge_expanded_embedding_pool as merge
from tools import stage6h_prepare_expanded_rollout_view as view
from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools.stage7_m6_2_locked_task_bdd import PRETREATMENT_TASKS


PLANNERS = batch.EXPECTED_PLANNERS


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_view_validation_requires_all_490_frozen_successes(tmp_path):
    primary = []
    order = 0
    for task, count in view.EXPECTED_TASK_COUNTS.items():
        scenario_type = PRETREATMENT_TASKS[task][0]
        for _ in range(count):
            order += 1
            primary.append(
                {
                    "collection_order": order,
                    "task": task,
                    "task_rank": order,
                    "log_name": f"log-{order}",
                    "scenario_token": f"token-{order}",
                    "scene_token": f"token-{order}",
                    "scenario_type": scenario_type,
                    "db_file": f"log-{order}.db",
                    "db_scene_token": "scene",
                    "scenario_tag_token": "tag",
                    "selection_role": "stage6g_primary",
                    "selection_salt": "test",
                }
            )
    primary_csv = tmp_path / "primary.csv"
    write_csv(primary_csv, primary, batch.PRIMARY_FIELDS)
    freeze = {
        "status": "FROZEN_BEFORE_STAGE6G_ROLLOUTS",
        "planned_primary_additions": 490,
        "forbidden_inputs_read": {"embedding": False},
        "hashes": {"primary_csv_sha256": batch.sha256_file(primary_csv)},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    statuses = [
        {
            **row,
            "status": "SUCCEEDED",
            "stage7c_output_dir": str(tmp_path / f"output-{row['collection_order']}"),
        }
        for row in primary
    ]
    status_csv = tmp_path / "status.csv"
    write_csv(
        status_csv,
        statuses,
        ["collection_order", "task", "log_name", "scenario_token", "db_file", "status", "stage7c_output_dir"],
    )
    args = argparse.Namespace(
        freeze_manifest=freeze_path,
        primary_csv=primary_csv,
        batch_status_csv=status_csv,
    )
    rows = view.validate_inputs(args)
    assert len(rows) == 490
    assert Counter(row["task"] for row in rows) == Counter(view.EXPECTED_TASK_COUNTS)


def metadata_for_counts(counts, source_prefix):
    rows = []
    scenario_index = 0
    for task, count in counts.items():
        scenario_type = PRETREATMENT_TASKS[task][0]
        for _ in range(count):
            token = f"{source_prefix}-{scenario_index}"
            for planner_id, planner in enumerate(PLANNERS):
                rows.append(
                    {
                        "global_row": len(rows),
                        "tensor_scenario_position": scenario_index,
                        "scenario_index": scenario_index,
                        "planner_id": planner_id,
                        "planner_name": planner,
                        "log_name": f"{source_prefix}-log-{scenario_index}",
                        "map_name": "us-pa-pittsburgh-hazelwood",
                        "location": "",
                        "scenario_token": token,
                        "actual_nuplan_scenario_token": token,
                        "stage7b_scene_token": token,
                        "sample_id": "",
                        "scenario_type": scenario_type,
                    }
                )
            scenario_index += 1
    return pd.DataFrame(rows)


def make_embedding_dir(path, counts, checkpoint, seed):
    path.mkdir()
    metadata = metadata_for_counts(counts, path.name)
    rng = np.random.default_rng(seed)
    np.save(path / "embedding.npy", rng.normal(size=(len(metadata), 64)).astype(np.float32))
    metadata.to_csv(path / "metadata.csv", index=False)
    (path / "embedding_manifest.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "total_rows": len(metadata), "embedding_dim": 64}),
        encoding="utf-8",
    )
    (path / "stage7e_context_schema.json").write_text(
        json.dumps(
            {
                "schema_name": "stage5d83",
                "context_dim": 83,
                "dim_formula": "8+5*15",
                "ego_channels": ["x"],
                "neighbor_channels_per_slot": ["valid"],
                "neighbor_slots": ["front"],
                "channels": [{"index": 0, "name": "x"}],
                "slot_assignment_method": "lane-aware",
            }
        ),
        encoding="utf-8",
    )


def test_merge_builds_exact_800_pair_aligned_pool(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"same checkpoint")
    existing_counts = {
        "following_interaction": 60,
        "lane_change": 60,
        "stop_go_control": 67,
        "high_motion_dynamics": 60,
        "dense_or_vulnerable_interaction": 63,
    }
    new_counts = view.EXPECTED_TASK_COUNTS
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    make_embedding_dir(old_dir, existing_counts, checkpoint, 1)
    make_embedding_dir(new_dir, new_counts, checkpoint, 2)
    output = tmp_path / "combined"
    summary = merge.run(
        argparse.Namespace(
            existing_embedding_dir=old_dir,
            new_embedding_dir=new_dir,
            output_dir=output,
        )
    )
    assert summary["status"] == merge.READY_STATUS
    assert summary["pair_count"] == 800
    assert summary["row_count"] == 1600
    assert summary["task_counts"] == merge.EXPECTED_COMBINED_TASK_COUNTS
    metadata = pd.read_csv(output / "metadata.csv")
    assert metadata["global_row"].tolist() == list(range(1600))
    assert metadata["scenario_token"].nunique() == 800
