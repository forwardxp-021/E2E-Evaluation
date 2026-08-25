import argparse
import csv
import json
import sqlite3
from pathlib import Path

import pytest

import tools.stage7_m6_4d_freeze_high_motion_supplement as supplement


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_db(path: Path, token_positions):
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE scene (token BLOB PRIMARY KEY, name TEXT)")
        connection.execute("CREATE TABLE lidar_pc (token BLOB PRIMARY KEY, scene_token BLOB)")
        scene_tokens = []
        for index in range(1, 7):
            scene_token = index.to_bytes(8, "big")
            scene_tokens.append(scene_token)
            connection.execute(
                "INSERT INTO scene(token, name) VALUES (?, ?)",
                (scene_token, f"scene-{index:02d}"),
            )
        for token, position in token_positions.items():
            connection.execute(
                "INSERT INTO lidar_pc(token, scene_token) VALUES (?, ?)",
                (bytes.fromhex(token), scene_tokens[position - 1]),
            )
        connection.commit()
    finally:
        connection.close()


def candidate(token, log_name, db_file="fixture.db"):
    return {
        "task": supplement.TASK,
        "candidate_rank": "1",
        "log_name": log_name,
        "scenario_token": token,
        "scene_token": token,
        "scenario_type": "high_magnitude_speed",
        "db_file": db_file,
        "db_scene_token": "scene",
        "scenario_tag_token": "tag",
        "selection_salt": "old-salt",
    }


def test_top_ranked_candidates_is_deterministic_and_excludes_prior_rows(tmp_path):
    inventory = tmp_path / "eligible.csv"
    rows = [
        candidate("a000000000000001", "old-log"),
        candidate("b000000000000002", "new-log-1"),
        candidate("c000000000000003", "new-log-2"),
        {**candidate("d000000000000004", "other-task"), "task": "lane_change"},
    ]
    write_csv(inventory, rows, rows[0].keys())
    first, audit = supplement.top_ranked_candidates(
        inventory,
        salt="fixed",
        probe_limit=10,
        excluded_tokens={"a000000000000001"},
        excluded_logs={"old-log"},
    )
    second, _ = supplement.top_ranked_candidates(
        inventory,
        salt="fixed",
        probe_limit=10,
        excluded_tokens={"a000000000000001"},
        excluded_logs={"old-log"},
    )
    assert [row["scenario_token"] for row in first] == [
        row["scenario_token"] for row in second
    ]
    assert {row["scenario_token"] for row in first} == {
        "b000000000000002",
        "c000000000000003",
    }
    assert audit["excluded_original_or_development_token"] == 1


def test_inspect_and_select_requires_valid_scene_and_distinct_log(tmp_path):
    db_root = tmp_path / "db"
    db_root.mkdir()
    create_db(
        db_root / "fixture.db",
        {
            "a000000000000001": 1,
            "b000000000000002": 3,
            "c000000000000003": 4,
            "d000000000000004": 3,
        },
    )
    rows = [
        candidate("a000000000000001", "invalid"),
        candidate("b000000000000002", "shared"),
        candidate("c000000000000003", "shared"),
        candidate("d000000000000004", "distinct"),
    ]
    for index, row in enumerate(rows):
        row["stable_rank_sha256"] = f"{index:064x}"
    selected, audit = supplement.inspect_and_select(
        rows, db_root=db_root, required_count=2
    )
    assert [row["scenario_token"] for row in selected] == [
        "b000000000000002",
        "d000000000000004",
    ]
    assert [row["decision"] for row in audit] == [
        "EXCLUDED_INVALID_SCENE_POSITION",
        "SELECTED_TECHNICALLY_RUNNABLE",
        "EXCLUDED_DUPLICATE_SUPPLEMENT_LOG",
        "SELECTED_TECHNICALLY_RUNNABLE",
    ]


def test_run_writes_frozen_primary_and_reserve_without_overlap(tmp_path, monkeypatch):
    db_root = tmp_path / "db"
    db_root.mkdir()
    tokens = [f"{index:016x}" for index in range(10, 16)]
    create_db(db_root / "fixture.db", {token: 3 for token in tokens})
    eligible = tmp_path / "eligible.csv"
    eligible_rows = [candidate(token, f"new-log-{index}") for index, token in enumerate(tokens)]
    write_csv(eligible, eligible_rows, eligible_rows[0].keys())

    simple_fields = ["scenario_token", "log_name"]
    primary = tmp_path / "primary.csv"
    reserve = tmp_path / "reserve.csv"
    development = tmp_path / "development.csv"
    write_csv(primary, [{"scenario_token": "a" * 16, "log_name": "prior-primary"}], simple_fields)
    write_csv(reserve, [{"scenario_token": "b" * 16, "log_name": "prior-reserve"}], simple_fields)
    write_csv(development, [{"scenario_token": "c" * 16, "log_name": "development"}], simple_fields)

    placeholders = {}
    for name in [
        "locked_manifest",
        "batch_status_csv",
        "batch_manifest",
        "m6_4c_audit_summary",
        "quoted_recovery_state",
        "reserve_recovery_state",
        "stage7c_tool",
        "batch_tool",
    ]:
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        placeholders[name] = path
    args = argparse.Namespace(
        **placeholders,
        eligible_inventory=eligible,
        development_metadata_csv=development,
        primary_csv=primary,
        reserve_csv=reserve,
        nuplan_db_root=db_root,
        nuplan_devkit_root=tmp_path,
        tuplan_garage_root=tmp_path,
        selection_salt="fixed-supplement",
        primary_count=2,
        reserve_count=2,
        candidate_probe_limit=10,
        output_dir=tmp_path / "out",
    )
    manifest = {
        "planner_parameter_fingerprints": {},
        "planners": list(supplement.batch.EXPECTED_PLANNERS),
        "required_complete_pairs_by_task": {supplement.TASK: 60},
    }
    monkeypatch.setattr(
        supplement,
        "validate_inputs",
        lambda _args: (manifest, {supplement.TASK: 58}),
    )
    assert supplement.run(args) == 0
    frozen = json.loads(
        (args.output_dir / "m6_4d_locked_supplement_manifest.json").read_text()
    )
    assert frozen["planned_primary_scenarios"] == 2
    assert frozen["maximum_reserve_scenarios"] == 2
    assert frozen["supplement_log_overlap_with_prior_or_development"] == 0
    assert frozen["candidate_probe_decisions"] == {
        "SELECTED_TECHNICALLY_RUNNABLE": 4
    }


def test_existing_output_is_not_overwritten(tmp_path):
    args = argparse.Namespace(output_dir=tmp_path / "out")
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        supplement.run(args)
