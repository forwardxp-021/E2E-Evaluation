import csv
import json
import sqlite3
from pathlib import Path

import pytest

from tools.stage7p_build_scenario_inventory import (
    INVENTORY_FIELDS,
    build_inventory,
)


def blob(hex_value: str) -> bytes:
    return bytes.fromhex(hex_value)


def create_fixture_db(
    path: Path,
    *,
    log_token: str,
    log_name: str,
    scene_token: str,
    tags,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE log (
                token BLOB PRIMARY KEY NOT NULL,
                logfile TEXT
            );
            CREATE TABLE scene (
                token BLOB PRIMARY KEY NOT NULL,
                log_token BLOB NOT NULL
            );
            CREATE TABLE lidar_pc (
                token BLOB PRIMARY KEY NOT NULL,
                scene_token BLOB
            );
            CREATE TABLE scenario_tag (
                token BLOB PRIMARY KEY NOT NULL,
                lidar_pc_token BLOB NOT NULL,
                type TEXT
            );
            """
        )
        conn.execute("INSERT INTO log(token, logfile) VALUES (?, ?)", (blob(log_token), log_name))
        conn.execute(
            "INSERT INTO scene(token, log_token) VALUES (?, ?)",
            (blob(scene_token), blob(log_token)),
        )
        seen_lidar = set()
        for tag_token, lidar_token, scenario_type in tags:
            if lidar_token not in seen_lidar:
                conn.execute(
                    "INSERT INTO lidar_pc(token, scene_token) VALUES (?, ?)",
                    (blob(lidar_token), blob(scene_token)),
                )
                seen_lidar.add(lidar_token)
            conn.execute(
                "INSERT INTO scenario_tag(token, lidar_pc_token, type) VALUES (?, ?, ?)",
                (blob(tag_token), blob(lidar_token), scenario_type),
            )


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_multi_db_inventory_schema_dedup_tokens_and_flat_pool(tmp_path):
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    db_a = root_a / "log_a.db"
    db_b = root_b / "log_b.db"
    create_fixture_db(
        db_a,
        log_token="01" * 8,
        log_name="log_a",
        scene_token="11" * 8,
        tags=[
            ("21" * 8, "31" * 8, "changing_lane"),
            ("22" * 8, "31" * 8, "changing_lane"),
            ("23" * 8, "31" * 8, "high_lateral_acceleration"),
        ],
    )
    create_fixture_db(
        db_b,
        log_token="02" * 8,
        log_name="log_b",
        scene_token="12" * 8,
        tags=[("24" * 8, "32" * 8, "following_lane_with_slow_lead")],
    )

    output_dir = tmp_path / "outputs"
    flat_root = tmp_path / "flat_pool"
    summary = build_inventory(
        db_roots=[root_b, root_a],
        output_dir=output_dir,
        flat_db_root=flat_root,
    )

    rows = read_csv(output_dir / "all_scenario_tags.csv")
    assert list(rows[0]) == INVENTORY_FIELDS
    assert len(rows) == 3
    assert summary["source_scenario_tag_rows"] == 4
    assert summary["inventory_rows"] == 3
    assert summary["unique_scenario_tokens"] == 2
    assert summary["duplicate_rows_removed"] == 1
    assert summary["db_file_count"] == 2
    assert summary["log_count"] == 2
    assert summary["outcome_blind"] is True
    assert summary["reads_planner_outcomes"] is False

    multi_type_rows = [row for row in rows if row["scenario_token"] == "31" * 8]
    assert {row["scenario_type"] for row in multi_type_rows} == {
        "changing_lane",
        "high_lateral_acceleration",
    }
    assert {row["scene_token"] for row in multi_type_rows} == {"31" * 8}
    assert {row["db_scene_token"] for row in multi_type_rows} == {"11" * 8}
    assert {row["scenario_tag_token"] for row in multi_type_rows} == {"21" * 8, "23" * 8}

    for db_path in (db_a, db_b):
        link = flat_root / db_path.name
        assert link.is_symlink()
        assert not link.readlink().is_absolute()
        assert link.resolve() == db_path.resolve()

    assert (output_dir / "scenario_inventory_summary.json").is_file()
    assert (output_dir / "scenario_inventory_inputs.csv").is_file()
    assert (output_dir / "scenario_inventory_report.md").is_file()
    persisted = json.loads((output_dir / "scenario_inventory_summary.json").read_text())
    assert persisted["schema_version"] == "stage7p_scenario_inventory_v1"
    assert set(persisted["output_sha256"]) == {
        "all_scenario_tags.csv",
        "scenario_inventory_inputs.csv",
        "scenario_inventory_report.md",
    }

    repeated = build_inventory(
        db_roots=[root_a, root_b],
        output_dir=output_dir,
        flat_db_root=flat_root,
        overwrite=True,
    )
    assert repeated["flat_db_pool"]["created"] == 0
    assert repeated["flat_db_pool"]["reused"] == 2


def test_duplicate_db_basename_fails_closed(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    for index, root in enumerate(roots, 1):
        create_fixture_db(
            root / "same.db",
            log_token=f"0{index}" * 8,
            log_name=f"log_{index}",
            scene_token=f"1{index}" * 8,
            tags=[(f"2{index}" * 8, f"3{index}" * 8, "changing_lane")],
        )
    with pytest.raises(ValueError, match="DB basename conflict"):
        build_inventory(db_roots=roots, output_dir=tmp_path / "out")


def test_token_location_conflict_fails_closed(tmp_path):
    root = tmp_path / "dbs"
    shared_token = "44" * 8
    create_fixture_db(
        root / "a.db",
        log_token="01" * 8,
        log_name="log_a",
        scene_token="11" * 8,
        tags=[("21" * 8, shared_token, "changing_lane")],
    )
    create_fixture_db(
        root / "b.db",
        log_token="02" * 8,
        log_name="log_b",
        scene_token="12" * 8,
        tags=[("22" * 8, shared_token, "changing_lane")],
    )
    with pytest.raises(ValueError, match="multiple log/DB locations"):
        build_inventory(db_roots=[root], output_dir=tmp_path / "out")


def test_missing_required_table_fails_closed(tmp_path):
    root = tmp_path / "dbs"
    root.mkdir()
    with sqlite3.connect(root / "broken.db") as conn:
        conn.execute("CREATE TABLE scenario_tag(token BLOB, lidar_pc_token BLOB, type TEXT)")
    with pytest.raises(ValueError, match="Missing required SQLite table 'lidar_pc'"):
        build_inventory(db_roots=[root], output_dir=tmp_path / "out")


def test_flat_pool_refuses_wrong_existing_entry(tmp_path):
    root = tmp_path / "dbs"
    create_fixture_db(
        root / "a.db",
        log_token="01" * 8,
        log_name="log_a",
        scene_token="11" * 8,
        tags=[("21" * 8, "31" * 8, "changing_lane")],
    )
    flat_root = tmp_path / "flat"
    flat_root.mkdir()
    (flat_root / "a.db").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite non-symlink"):
        build_inventory(
            db_roots=[root],
            output_dir=tmp_path / "out",
            flat_db_root=flat_root,
        )
