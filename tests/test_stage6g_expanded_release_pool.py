import argparse
import csv
from pathlib import Path

from tools import stage6g_freeze_expanded_release_pool as freeze
from tools import stage6g_run_expanded_release_pool as runner
from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools.stage7_m6_2_locked_task_bdd import PRETREATMENT_TASKS


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate(task, token, log_name):
    scenario_type = PRETREATMENT_TASKS[task][0]
    return {
        "task": task,
        "log_name": log_name,
        "scenario_token": token,
        "scene_token": token,
        "scenario_type": scenario_type,
        "db_file": f"{log_name}.db",
        "db_scene_token": f"scene-{token}",
        "scenario_tag_token": f"tag-{token}",
        "selection_salt": "test",
    }


def test_validate_config_requires_exact_bridge_to_target():
    primary = {task: 1 for task in PRETREATMENT_TASKS}
    config = {
        "schema_version": "stage6g_expanded_release_pool_config_v1",
        "existing_pool_size": 5,
        "target_combined_pool_size": 10,
        "primary_additions_by_task": primary,
        "reserve_by_task": {task: 0 for task in PRETREATMENT_TASKS},
    }
    actual_primary, actual_reserve = freeze.validate_config(config)
    assert actual_primary == primary
    assert sum(actual_reserve.values()) == 0


def test_collect_top_candidates_is_deterministic_and_excludes_existing(tmp_path):
    path = tmp_path / "eligible.csv"
    fields = [
        "task", "log_name", "scenario_token", "scenario_type", "db_file",
        "db_scene_token", "scenario_tag_token",
    ]
    rows = []
    for index, task in enumerate(PRETREATMENT_TASKS):
        rows.append(candidate(task, f"token-{index}", f"log-{index}"))
        rows.append(candidate(task, f"token-x-{index}", f"log-x-{index}"))
    write_csv(path, rows, fields)
    first, audit = freeze.collect_top_candidates(
        path, excluded_tokens={"token-0"}, salt="stable", probe_limit=1
    )
    second, _ = freeze.collect_top_candidates(
        path, excluded_tokens={"token-0"}, salt="stable", probe_limit=1
    )
    assert first == second
    assert all(len(values) == 1 for values in first.values())
    assert audit["excluded_existing_tokens_by_task"][next(iter(PRETREATMENT_TASKS))] == 1


def test_selection_protects_rare_task_and_respects_combined_log_cap(tmp_path, monkeypatch):
    tasks = list(PRETREATMENT_TASKS)
    rare, common = tasks[0], tasks[1]
    pools = {task: [] for task in tasks}
    pools[rare] = [candidate(rare, "rare", "shared")]
    pools[common] = [
        candidate(common, "common-shared", "shared"),
        candidate(common, "common-other", "other"),
    ]
    for rows in pools.values():
        for rank, row in enumerate(rows):
            row["stable_rank_sha256"] = f"{rank:064x}"
    monkeypatch.setattr(
        freeze.recovery,
        "inspect_token_scene_position",
        lambda *_: {
            "token_found": True,
            "scene_position": 3,
            "scene_count": 8,
            "official_scene_position_valid": True,
            "hydra_requires_quoted_token": False,
        },
    )
    primary, reserve, _, deficits = freeze.select_technically_runnable(
        pools,
        primary_quotas={task: int(task in {rare, common}) for task in tasks},
        reserve_quotas={task: 0 for task in tasks},
        existing_log_counts={"shared": 1},
        max_per_log=2,
        db_root=tmp_path,
    )
    assert [row["scenario_token"] for row in primary] == ["rare", "common-other"]
    assert reserve == []
    assert not any(deficits.values())


def test_runner_treats_partial_attempt_as_pending(tmp_path):
    row = candidate("following_interaction", "token", "log")
    row.update(collection_order="1", task_rank="1", selection_role="stage6g_primary")
    partial = tmp_path / "rollouts" / batch.scenario_slug(row) / "attempt_001"
    partial.mkdir(parents=True)
    status = runner.status_rows([row], tmp_path)
    assert status[0]["status"] == "PENDING"
    assert status[0]["attempt"] == 1


def test_numeric_hydra_token_is_written_as_quoted_actual_token(tmp_path):
    row = candidate("following_interaction", "123456", "log")
    row.update(collection_order="1", task_rank="1", selection_role="stage6g_primary")
    args = argparse.Namespace(
        python_executable=tmp_path / "python",
        stage7c_tool=tmp_path / "stage7c.py",
        nuplan_devkit_root=tmp_path / "nuplan-devkit",
        nuplan_db_root=tmp_path / "db",
        nuplan_map_root=tmp_path / "maps",
        command_timeout_s=10,
    )
    command = runner.build_stage7c_command(args, row, tmp_path / "attempt")
    assert "--nuplan_simulation_command_template" in command
    with (tmp_path / "attempt/context/merged_metadata.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        written = next(csv.DictReader(handle))
    assert written["actual_nuplan_token"] == '\\"123456\\"'
