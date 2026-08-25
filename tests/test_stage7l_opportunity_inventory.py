import csv
import sqlite3
from pathlib import Path

from tools.stage7l_build_lane_change_opportunity_inventory import (
    historical_exclusions,
    official_simulation_initial_token,
    select_source_lane,
)


class Pose:
    heading = 0.0


class Baseline:
    def get_nearest_pose_from_position(self, point):
        return Pose()


class Lane:
    id = "source"
    baseline_path = Baseline()


class FakeMap:
    def get_all_map_objects(self, point, layer):
        return [Lane()]


def test_map_source_lane_requires_heading_compatible_native_lane() -> None:
    lane, reason = select_source_lane(FakeMap(), 0.0, 0.0, 0.05)
    assert lane is not None
    assert lane.id == "source"
    assert reason == "PASS"


def test_historical_exclusion_ledger_deduplicates(tmp_path: Path) -> None:
    roster = tmp_path / "roster.csv"
    with roster.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "scenario_token", "log_name"])
        writer.writeheader(); writer.writerow({"task": "lane_change", "scenario_token": "0123456789abcdef", "log_name": "a"})
    stage7p = tmp_path / "stage7p" / "run"
    stage7p.mkdir(parents=True)
    with (stage7p / "scenario_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual_nuplan_token", "log_name"])
        writer.writeheader(); writer.writerow({"actual_nuplan_token": "0123456789abcdef", "log_name": "a"})
        writer.writerow({"actual_nuplan_token": "fedcba9876543210", "log_name": "b"})
    rows = historical_exclusions(roster, tmp_path / "stage7p")
    assert {row["scenario_token"] for row in rows} == {"0123456789abcdef", "fedcba9876543210"}
    combined = next(row for row in rows if row["scenario_token"] == "0123456789abcdef")
    assert "STAGE7_FROZEN_LANE_CHANGE_60" in combined["exclusion_reason"]
    assert "STAGE7P_TUNING_OR_SMOKE" in combined["exclusion_reason"]


def test_official_initial_token_distinguishes_tagged_and_default_scenarios(tmp_path: Path) -> None:
    db = tmp_path / "log.db"
    anchor = bytes.fromhex("0123456789abcdef")
    prior = bytes.fromhex("fedcba9876543210")
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE lidar_pc (token BLOB, timestamp INTEGER)")
        connection.execute("CREATE TABLE scenario_tag (lidar_pc_token BLOB, type TEXT)")
        connection.execute("INSERT INTO lidar_pc VALUES (?, ?)", (prior, 7_000_000))
        connection.execute("INSERT INTO lidar_pc VALUES (?, ?)", (anchor, 10_000_000))
        connection.commit()
    token, timestamp, types = official_simulation_initial_token(db, anchor.hex(), 10_000_000)
    assert (token, timestamp, types) == (anchor.hex(), 10_000_000, ())
    with sqlite3.connect(db) as connection:
        connection.execute("INSERT INTO scenario_tag VALUES (?, ?)", (anchor, "changing_lane"))
        connection.commit()
    token, timestamp, types = official_simulation_initial_token(db, anchor.hex(), 10_000_000)
    assert (token, timestamp, types) == (prior.hex(), 7_000_000, ("changing_lane",))
