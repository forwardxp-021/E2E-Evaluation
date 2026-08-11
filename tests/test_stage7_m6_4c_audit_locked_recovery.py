import argparse
import csv
import json
import sqlite3
from pathlib import Path

import pytest

import tools.stage7_m6_4c_audit_locked_recovery as recovery


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def locked_row(order, rank, token, role, task="high_motion_dynamics"):
    return {
        "collection_order": str(order),
        "task": task,
        "task_rank": str(rank),
        "log_name": "fixture-log",
        "scenario_token": token,
        "scene_token": token,
        "scenario_type": "high_magnitude_speed",
        "db_file": "fixture.db",
        "db_scene_token": "unused-scene-token",
        "scenario_tag_token": "unused-tag-token",
        "selection_role": role,
        "selection_salt": "fixture-salt",
    }


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


def make_fixture(tmp_path: Path):
    db_root = tmp_path / "db"
    db_root.mkdir()
    success_token = "a000000000000001"
    invalid_token = "b000000000000002"
    numeric_token = "1234567890123456"
    scientific_token = "434e024292395713"
    reserve_valid = "c000000000000003"
    reserve_invalid = "d000000000000004"
    create_db(
        db_root / "fixture.db",
        {
            success_token: 3,
            invalid_token: 1,
            numeric_token: 4,
            scientific_token: 3,
            reserve_valid: 3,
            reserve_invalid: 6,
        },
    )
    primary = [
        locked_row(1, 1, success_token, "primary_gross"),
        locked_row(2, 2, invalid_token, "primary_gross"),
        locked_row(3, 3, numeric_token, "primary_gross"),
        locked_row(4, 4, scientific_token, "primary_gross"),
    ]
    reserve = [
        locked_row(1, 1, reserve_valid, "technical_quality_reserve"),
        locked_row(2, 2, reserve_invalid, "technical_quality_reserve"),
    ]
    statuses = [
        {
            "collection_order": "1",
            "task": "high_motion_dynamics",
            "scenario_token": success_token,
            "status": "SUCCEEDED",
            "failure_category": "",
        },
        {
            "collection_order": "2",
            "task": "high_motion_dynamics",
            "scenario_token": invalid_token,
            "status": "FAILED_REVIEW_REQUIRED",
            "failure_category": "OFFICIAL_COMMAND_FAILED",
        },
        {
            "collection_order": "3",
            "task": "high_motion_dynamics",
            "scenario_token": numeric_token,
            "status": "FAILED_REVIEW_REQUIRED",
            "failure_category": "OFFICIAL_COMMAND_FAILED",
        },
        {
            "collection_order": "4",
            "task": "high_motion_dynamics",
            "scenario_token": scientific_token,
            "status": "FAILED_REVIEW_REQUIRED",
            "failure_category": "OFFICIAL_COMMAND_FAILED",
        },
    ]
    primary_csv = tmp_path / "primary.csv"
    reserve_csv = tmp_path / "reserve.csv"
    status_csv = tmp_path / "status.csv"
    write_csv(primary_csv, primary, recovery.PRIMARY_FIELDS)
    write_csv(reserve_csv, reserve, recovery.PRIMARY_FIELDS)
    write_csv(status_csv, statuses, sorted(recovery.STATUS_REQUIRED_FIELDS))
    stage7c_tool = tmp_path / "stage7c.py"
    batch_tool = tmp_path / "batch.py"
    stage7c_tool.write_text("# frozen stage7c\n", encoding="utf-8")
    batch_tool.write_text("# frozen batch\n", encoding="utf-8")
    manifest = {
        "status": recovery.READY_STATUS,
        "ready_to_launch_locked_rollouts": True,
        "planned_primary_scenarios": len(primary),
        "maximum_reserve_scenarios": len(reserve),
        "primary_collection_csv_sha256": recovery.sha256_file(primary_csv),
        "reserve_collection_csv_sha256": recovery.sha256_file(reserve_csv),
        "stage7c_tool_sha256": recovery.sha256_file(stage7c_tool),
        "primary_manifest_sha256": recovery.canonical_rows_hash(primary),
        "reserve_manifest_sha256": recovery.canonical_rows_hash(reserve),
        "required_complete_pairs_by_task": {"high_motion_dynamics": 5},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    batch_manifest_path = tmp_path / "batch_manifest.json"
    batch_manifest_path.write_text(
        json.dumps({"batch_tool_sha256": recovery.sha256_file(batch_tool)}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        locked_manifest=manifest_path,
        primary_csv=primary_csv,
        reserve_csv=reserve_csv,
        batch_status_csv=status_csv,
        batch_manifest=batch_manifest_path,
        nuplan_db_root=db_root,
        stage7c_tool=stage7c_tool,
        batch_tool=batch_tool,
        output_dir=tmp_path / "out",
    )
    return args


def test_scene_position_matches_nuplan_official_boundary(tmp_path):
    db_path = tmp_path / "fixture.db"
    valid = "a000000000000001"
    first = "b000000000000002"
    penultimate = "c000000000000003"
    create_db(db_path, {valid: 3, first: 1, penultimate: 5})

    assert recovery.inspect_token_scene_position(db_path, valid)[
        "official_scene_position_valid"
    ] is True
    assert recovery.inspect_token_scene_position(db_path, first)[
        "official_scene_position_valid"
    ] is False
    assert recovery.inspect_token_scene_position(db_path, penultimate)[
        "official_scene_position_valid"
    ] is False


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("8654415816575948", True),
        ("434e024292395713", True),
        ("434a024292395713", False),
        ("deadbeef12345678", False),
    ],
)
def test_hydra_numeric_like_tokens_require_quotes(token, expected):
    assert recovery.hydra_requires_quoted_token(token) is expected


def test_integration_writes_outcome_blind_recovery_plan(tmp_path):
    args = make_fixture(tmp_path)
    assert recovery.run(args) == 0

    summary = json.loads(
        (args.output_dir / "m6_4c_recovery_audit_summary.json").read_text()
    )
    assert summary["primary_classification_counts"] == {
        "INVALID_SCENE_POSITION": 1,
        "RETRY_WITH_QUOTED_HYDRA_TOKEN": 2,
        "RUNNABLE_SUCCEEDED": 1,
    }
    assert summary["reserve_classification_counts"] == {
        "INVALID_SCENE_POSITION": 1,
        "RUNNABLE_RESERVE": 1,
    }
    assert all(value is False for value in summary["forbidden_inputs_read"].values())

    plan = list(
        csv.DictReader((args.output_dir / "recovery_plan.csv").open(encoding="utf-8"))
    )
    assert [row["action"] for row in plan] == [
        "RETRY_PRIMARY_QUOTED_TOKEN",
        "RETRY_PRIMARY_QUOTED_TOKEN",
        "RUN_FROZEN_RESERVE",
    ]
    quota = list(
        csv.DictReader(
            (args.output_dir / "quota_recovery_projection.csv").open(encoding="utf-8")
        )
    )[0]
    assert quota["projected_complete_pairs"] == "4"
    assert quota["remaining_deficit_after_frozen_recovery"] == "1"
    assert quota["quota_status"] == "SUPPLEMENTAL_PROTOCOL_REQUIRED"


def test_locked_hash_mismatch_fails_before_output_creation(tmp_path):
    args = make_fixture(tmp_path)
    args.stage7c_tool.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        recovery.run(args)
    assert not args.output_dir.exists()


def test_existing_output_directory_is_not_overwritten(tmp_path):
    args = make_fixture(tmp_path)
    args.output_dir.mkdir()
    marker = args.output_dir / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        recovery.run(args)
    assert marker.read_text(encoding="utf-8") == "keep"
