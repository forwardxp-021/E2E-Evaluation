import argparse
import csv
import shlex
from pathlib import Path

import pytest

import tools.stage7_m6_4c_run_locked_recovery as runner
import tools.stage7c1_run_nuplan_simulation as stage7c


@pytest.mark.parametrize("token", ["8654415816575948", "434e024292395713"])
def test_escaped_hydra_token_survives_stage7c_shlex_split(token):
    escaped = runner.escaped_hydra_string(token)
    override = stage7c.scenario_hydra_override_info(
        {"scenario_token": token, "actual_nuplan_token": escaped}
    )["scenario_hydra_overrides"]
    assert shlex.split(override) == [f'scenario_filter.scenario_tokens=["{token}"]']


def test_non_numeric_like_token_is_rejected():
    with pytest.raises(ValueError, match="not Hydra numeric-like"):
        runner.escaped_hydra_string("deadbeef12345678")


def test_build_command_keeps_raw_identity_and_adds_escaped_override(tmp_path):
    args = argparse.Namespace(
        python_executable=Path("/tmp/python"),
        stage7c_tool=Path("/tmp/stage7c.py"),
        nuplan_devkit_root=Path("/tmp/nuplan-devkit"),
        nuplan_db_root=Path("/tmp/db"),
        nuplan_map_root=Path("/tmp/maps"),
        command_timeout_s=3600,
    )
    row = {
        field: ""
        for field in runner.batch.PRIMARY_FIELDS
    }
    row.update(
        {
            "collection_order": "244",
            "task": "high_motion_dynamics",
            "task_rank": "49",
            "log_name": "fixture-log",
            "scenario_token": "434e024292395713",
            "scene_token": "434e024292395713",
            "db_file": "fixture.db",
            "selection_role": "primary_gross",
        }
    )
    command = runner.build_stage7c_command(args, row, tmp_path / "attempt")
    assert "stage7_m6_4c_locked_recovery_mac_v1" in command[-1]
    with (tmp_path / "attempt/context/merged_metadata.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        context = next(csv.DictReader(handle))
    assert context["scenario_token"] == row["scenario_token"]
    assert context["actual_nuplan_token"] == '\\"434e024292395713\\"'


def test_reserve_command_keeps_token_unquoted(tmp_path):
    args = argparse.Namespace(
        python_executable=Path("/tmp/python"),
        stage7c_tool=Path("/tmp/stage7c.py"),
        nuplan_devkit_root=Path("/tmp/nuplan-devkit"),
        nuplan_db_root=Path("/tmp/db"),
        nuplan_map_root=Path("/tmp/maps"),
        command_timeout_s=3600,
    )
    row = {field: "" for field in runner.batch.PRIMARY_FIELDS}
    row.update(
        {
            "collection_order": "2",
            "scenario_token": "f2dfbd8e42c151e0",
            "scene_token": "f2dfbd8e42c151e0",
            "db_file": "fixture.db",
            "selection_role": "technical_quality_reserve",
        }
    )
    runner.build_stage7c_command(
        args, row, tmp_path / "attempt", runner.RESERVE_ACTION
    )
    with (tmp_path / "attempt/context/merged_metadata.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        context = next(csv.DictReader(handle))
    assert context["actual_nuplan_token"] == ""
