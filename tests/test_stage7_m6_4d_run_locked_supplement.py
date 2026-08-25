import argparse
import csv
import json
import shlex
from pathlib import Path

import pytest

import tools.stage7_m6_4d_run_locked_supplement as runner
import tools.stage7c1_run_nuplan_simulation as stage7c


def locked_row(token="434e024292395713"):
    row = {field: "" for field in runner.batch.PRIMARY_FIELDS}
    row.update(
        {
            "collection_order": "1",
            "task": "high_motion_dynamics",
            "task_rank": "1",
            "log_name": "fixture-log",
            "scenario_token": token,
            "scene_token": token,
            "scenario_type": "high_magnitude_speed",
            "db_file": "fixture.db",
            "selection_role": "supplemental_primary",
            "selection_salt": "stage7-m6.4d-high-motion-supplement-v1",
        }
    )
    return row


def test_numeric_like_token_is_quoted_but_normal_hex_is_not():
    assert runner.hydra_actual_token("434e024292395713") == '\\"434e024292395713\\"'
    assert runner.hydra_actual_token("deadbeef12345678") == ""


def test_numeric_like_quote_survives_stage7c_shlex():
    token = "434e024292395713"
    override = stage7c.scenario_hydra_override_info(
        {
            "scenario_token": token,
            "actual_nuplan_token": runner.hydra_actual_token(token),
        }
    )["scenario_hydra_overrides"]
    assert shlex.split(override) == [f'scenario_filter.scenario_tokens=["{token}"]']


def test_build_command_preserves_raw_identity(tmp_path):
    args = argparse.Namespace(
        python_executable=Path("/tmp/python"),
        stage7c_tool=Path("/tmp/stage7c.py"),
        nuplan_devkit_root=Path("/tmp/nuplan-devkit"),
        nuplan_db_root=Path("/tmp/db"),
        nuplan_map_root=Path("/tmp/maps"),
        command_timeout_s=3600,
    )
    row = locked_row()
    command = runner.build_stage7c_command(args, row, tmp_path / "attempt")
    assert "stage7_m6_4d_high_motion_supplement_mac_v1" in command[-1]
    with (tmp_path / "attempt/context/merged_metadata.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        context = next(csv.DictReader(handle))
    assert context["scenario_token"] == row["scenario_token"]
    assert context["actual_nuplan_token"] == '\\"434e024292395713\\"'


def test_reserve_is_forbidden_when_primary_has_no_technical_failure(tmp_path):
    state_path = tmp_path / "primary_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": runner.SCHEMA_VERSION,
                "source": "primary",
                "results": [{"status": "SUCCEEDED"}],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        source="reserve",
        primary_run_state=state_path,
        primary_csv=tmp_path / "primary.csv",
        reserve_csv=tmp_path / "reserve.csv",
    )
    with pytest.raises(ValueError, match="reserve is forbidden"):
        runner.source_definition(
            args,
            {
                "reserve_collection_csv_sha256": "hash",
                "reserve_manifest_sha256": "canonical",
                "maximum_reserve_scenarios": 5,
            },
        )


def test_existing_output_is_not_overwritten(tmp_path):
    args = argparse.Namespace(output_dir=tmp_path / "out")
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run(args)
