import argparse
import json
import shlex

import pytest

from tools import stage6j_run_pure_longitudinal_rollouts as batch
from tools import stage7c1_run_nuplan_simulation as stage7c


def rows(count=3):
    return [
        {
            "collection_order": str(index),
            "source_global_scenario_index": str(index + 10),
            "task": "following_interaction",
            "source_task": "following_interaction",
            "scenario_type": "near_long_vehicle",
            "log_name": f"log-{index}",
            "scenario_token": f"token-{index}",
            "scene_token": f"token-{index}",
            "db_file": f"log-{index}.db",
            "selection_role": "FROZEN_PURE_LONGITUDINAL_PAIRED_PRIMARY",
        }
        for index in range(1, count + 1)
    ]


def args(tmp_path, execute=False, confirmation=""):
    paths = {}
    for name in ["freeze_manifest", "locked_scenarios_csv", "stage7c_tool", "python_executable"]:
        paths[name] = tmp_path / name
        paths[name].write_text("{}", encoding="utf-8")
    for name in ["nuplan_db_root", "nuplan_map_root", "nuplan_data_root", "nuplan_exp_root", "nuplan_devkit_root", "tuplan_garage_root"]:
        paths[name] = tmp_path / name
        paths[name].mkdir()
    return argparse.Namespace(
        **paths,
        expected_nuplan_commit="nuplan-commit",
        expected_tuplan_commit="tuplan-commit",
        output_dir=tmp_path / "out",
        start_order=1,
        end_order=0,
        max_scenarios=0,
        command_timeout_s=10,
        execute=execute,
        confirm_locked_scenarios_sha256=confirmation,
        resume=False,
        retry_failed=False,
    )


def test_selected_rows_preserves_locked_order_and_range(tmp_path):
    options = args(tmp_path)
    options.start_order = 2
    options.end_order = 3
    options.max_scenarios = 1
    assert [row["collection_order"] for row in batch.selected_rows(options, rows(4))] == ["2"]


def test_write_state_reports_pending_and_eta(tmp_path):
    statuses = []
    for index, row in enumerate(rows(), start=1):
        statuses.append(
            {
                **row,
                "status": "SUCCEEDED" if index == 1 else "PENDING",
                "duration_seconds": "30" if index == 1 else "",
            }
        )
    state = batch.write_state(tmp_path, statuses, "start")
    assert state["counts"] == {"SUCCEEDED": 1, "FAILED_REVIEW_REQUIRED": 0, "PENDING": 2}
    assert state["mean_success_duration_seconds"] == 30
    assert state["estimated_remaining_seconds"] == 60
    assert (tmp_path / "batch_state.json").is_file()


def test_dry_run_writes_frozen_batch_manifest_without_running(monkeypatch, tmp_path):
    options = args(tmp_path)
    locked_sha = "locked-sha"
    monkeypatch.setattr(batch, "validate_inputs", lambda unused: ({}, rows(), locked_sha))
    monkeypatch.setattr(batch, "planner_fingerprints", lambda: {name: name for name in batch.PLANNERS})
    result = batch.run(options)
    assert result["status"] == "DRY_RUN_PASS"
    assert result["candidate_count"] == 3
    assert result["batch_state"]["counts"]["PENDING"] == 3
    manifest = json.loads((options.output_dir / "batch_manifest.json").read_text())
    assert manifest["planned_rollout_count"] == 6


def test_execute_requires_exact_locked_hash(monkeypatch, tmp_path):
    options = args(tmp_path, execute=True, confirmation="wrong")
    monkeypatch.setattr(batch, "validate_inputs", lambda unused: ({}, rows(), "correct"))
    monkeypatch.setattr(batch, "planner_fingerprints", lambda: {name: name for name in batch.PLANNERS})
    with pytest.raises(ValueError, match="must exactly match"):
        batch.run(options)


@pytest.mark.parametrize("token", ["434e024292395713", "8654415816575948"])
def test_stage6j_quotes_hydra_numeric_like_token_for_retry(token):
    escaped = batch.hydra_actual_token(token)
    override = stage7c.scenario_hydra_override_info(
        {"scenario_token": token, "actual_nuplan_token": escaped}
    )["scenario_hydra_overrides"]
    assert shlex.split(override) == [f'scenario_filter.scenario_tokens=["{token}"]']


def test_stage6j_does_not_quote_normal_hex_token():
    assert batch.hydra_actual_token("deadbeef12345678") == ""
