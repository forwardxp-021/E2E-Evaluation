import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from tools.stage7_m6_2_locked_task_bdd import PRETREATMENT_TASKS
from tools.stage7_m6_4_freeze_locked_collection import (
    STATUS_BLOCKED,
    STATUS_READY,
    current_planner_fingerprints,
    freeze_collection,
    inventory_candidates,
    select_round_robin,
)
from tools.stage7_m6_scenario_conditioned_bdd import sha256_file
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


PLANNERS = ["pdm_closed_assertive_v1", "pdm_closed_conservative_v1"]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_frozen_inputs(tmp_path, gross_per_task=2):
    metadata = tmp_path / "development.csv"
    pd.DataFrame(
        [
            {
                "scenario_token": "dev-token",
                "log_name": "dev-log",
                "planner_name": planner,
                "parameters_json": json.dumps(PLANNER_PROFILES[planner]["parameters"]),
            }
            for planner in PLANNERS
        ]
    ).to_csv(metadata, index=False)
    tool_dir = Path(__file__).resolve().parents[1] / "tools"
    fingerprints = current_planner_fingerprints(PLANNERS)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_NEW_CONFIRMATION_DATA",
                "development_metadata_sha256": sha256_file(metadata),
                "analysis_tool_sha256": sha256_file(
                    tool_dir / "stage7_m6_2_locked_task_bdd.py"
                ),
                "planner_parameter_fingerprints": fingerprints,
                "task_conditioned_secondary": {
                    "task_definitions": {
                        task: list(types) for task, types in PRETREATMENT_TASKS.items()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    power = tmp_path / "power.json"
    power.write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_LOCKED_CONFIRMATION",
                "m6_2_lock_spec_sha256": sha256_file(lock),
                "power_analysis_tool_sha256": sha256_file(
                    tool_dir / "stage7_m6_3_simulation_power_analysis.py"
                ),
                "required_complete_pairs_by_task": {
                    task: 1 for task in PRETREATMENT_TASKS
                },
                "required_complete_pairs_overall": len(PRETREATMENT_TASKS),
                "planned_gross_pairs_per_task_with_attrition": gross_per_task,
            }
        ),
        encoding="utf-8",
    )
    return metadata, lock, power


def inventory_rows(per_task, db_root):
    rows = []
    index = 0
    for task, types in PRETREATMENT_TASKS.items():
        for task_index in range(per_task):
            index += 1
            db_file = f"log-{index}.db"
            (db_root / db_file).touch()
            rows.append(
                {
                    "db_file": db_file,
                    "log_name": f"log-{index}",
                    "scenario_token": f"token-{index}",
                    "scenario_type": types[0],
                    "db_scene_token": f"scene-{index}",
                    "scenario_tag_token": f"tag-{index}",
                }
            )
    return rows


def make_args(tmp_path, inventory, metadata, lock, power, reserve=1):
    return argparse.Namespace(
        inventory_csv=inventory,
        development_metadata_csv=metadata,
        m6_2_lock_spec=lock,
        power_justification_file=power,
        nuplan_db_root=tmp_path / "db",
        output_dir=tmp_path / "out",
        planner_a=PLANNERS[0],
        planner_b=PLANNERS[1],
        max_per_log=1,
        reserve_per_task=reserve,
        selection_salt="test-freeze",
    )


def test_inventory_excludes_development_overlap_ambiguity_and_missing_db(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir()
    (db_root / "ok.db").touch()
    rows = [
        {"db_file": "ok.db", "log_name": "new", "scenario_token": "ok", "scenario_type": "following_lane_with_lead"},
        {"db_file": "ok.db", "log_name": "dev-log", "scenario_token": "log-overlap", "scenario_type": "following_lane_with_lead"},
        {"db_file": "ok.db", "log_name": "new-2", "scenario_token": "dev-token", "scenario_type": "following_lane_with_lead"},
        {"db_file": "ok.db", "log_name": "new-3", "scenario_token": "ambiguous", "scenario_type": "following_lane_with_lead"},
        {"db_file": "ok.db", "log_name": "new-3", "scenario_token": "ambiguous", "scenario_type": "changing_lane_to_left"},
        {"db_file": "missing.db", "log_name": "new-4", "scenario_token": "missing", "scenario_type": "changing_lane_to_left"},
    ]
    inventory = tmp_path / "inventory.csv"
    write_csv(inventory, rows, ["db_file", "log_name", "scenario_token", "scenario_type"])
    pools, audit, exclusions = inventory_candidates(
        inventory,
        development_tokens={"dev-token"},
        development_logs={"dev-log"},
        db_root=db_root,
        selection_salt="test",
    )
    assert [row["scenario_token"] for row in pools["following_interaction"]] == ["ok"]
    assert audit["eligible_unique_candidates"] == 1
    reasons = "|".join(row["reasons"] for row in exclusions)
    assert "development_log_overlap" in reasons
    assert "development_scenario_overlap" in reasons
    assert "ambiguous_multiple_frozen_scenario_types" in reasons
    assert "db_file_missing" in reasons


def test_round_robin_is_deterministic_and_respects_log_cap():
    pools = {
        "a": [
            {"scenario_token": "a1", "log_name": "shared"},
            {"scenario_token": "a2", "log_name": "a-log"},
        ],
        "b": [
            {"scenario_token": "b1", "log_name": "shared"},
            {"scenario_token": "b2", "log_name": "b-log"},
        ],
    }
    selected, deficits = select_round_robin(pools, {"a": 2, "b": 2}, max_per_log=1)
    assert [row["scenario_token"] for row in selected] == ["a1", "b2", "a2"]
    assert deficits == {"a": 0, "b": 1}


def test_freeze_ready_emits_locked_primary_and_reserve_manifests(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir()
    metadata, lock, power = build_frozen_inputs(tmp_path, gross_per_task=2)
    inventory = tmp_path / "inventory.csv"
    write_csv(
        inventory,
        inventory_rows(3, db_root),
        ["db_file", "log_name", "scenario_token", "scenario_type", "db_scene_token", "scenario_tag_token"],
    )
    args = make_args(tmp_path, inventory, metadata, lock, power, reserve=1)
    args.output_dir.mkdir()
    readiness, exit_code = freeze_collection(args)
    assert exit_code == 0
    assert readiness["status"] == STATUS_READY
    assert (args.output_dir / "m6_4_locked_collection_manifest.json").is_file()
    primary = pd.read_csv(args.output_dir / "m6_4_locked_primary_collection.csv")
    reserve = pd.read_csv(args.output_dir / "m6_4_locked_reserve_collection.csv")
    assert len(primary) == 2 * len(PRETREATMENT_TASKS)
    assert len(reserve) == len(PRETREATMENT_TASKS)
    assert set(primary.scenario_token).isdisjoint(set(reserve.scenario_token))


def test_freeze_blocked_writes_audit_but_not_locked_manifest(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir()
    metadata, lock, power = build_frozen_inputs(tmp_path, gross_per_task=2)
    inventory = tmp_path / "inventory.csv"
    write_csv(
        inventory,
        inventory_rows(1, db_root),
        ["db_file", "log_name", "scenario_token", "scenario_type", "db_scene_token", "scenario_tag_token"],
    )
    args = make_args(tmp_path, inventory, metadata, lock, power, reserve=1)
    args.output_dir.mkdir()
    readiness, exit_code = freeze_collection(args)
    assert exit_code == 2
    assert readiness["status"] == STATUS_BLOCKED
    assert (args.output_dir / "m6_4_inventory_readiness.json").is_file()
    assert not (args.output_dir / "m6_4_locked_collection_manifest.json").exists()
