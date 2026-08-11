import argparse
import csv
import json
import sys
from pathlib import Path

import pytest

import tools.stage7_m6_4b_run_locked_rollouts as batch


def write_csv(path: Path, rows, fields=batch.PRIMARY_FIELDS):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def locked_row(order: int, task_rank: int, token: str, role: str, task: str = "following_interaction"):
    return {
        "collection_order": str(order),
        "task": task,
        "task_rank": str(task_rank),
        "log_name": f"log-{token}",
        "scenario_token": token,
        "scene_token": token,
        "scenario_type": "near_long_vehicle",
        "db_file": f"log-{token}.db",
        "db_scene_token": f"scene-{token}",
        "scenario_tag_token": f"tag-{token}",
        "selection_role": role,
        "selection_salt": "locked-test",
    }


def make_locked_inputs(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    run_simulation = tmp_path / "nuplan/nuplan/planning/script/run_simulation.py"
    pdm_planner = tmp_path / "tuplan/tuplan_garage/planning/simulation/planner/pdm_planner/pdm_closed_planner.py"
    run_simulation.parent.mkdir(parents=True)
    pdm_planner.parent.mkdir(parents=True)
    run_simulation.write_text("# run simulation\n", encoding="utf-8")
    pdm_planner.write_text("# pdm closed\n", encoding="utf-8")
    db_root = tmp_path / "db"
    db_root.mkdir()
    primary = [
        locked_row(1, 1, "token-a", "primary_gross"),
        locked_row(2, 2, "token-b", "primary_gross"),
    ]
    reserve = [locked_row(1, 1, "token-r", "technical_quality_reserve")]
    for row in primary + reserve:
        (db_root / row["db_file"]).touch()
    primary_csv = tmp_path / "primary.csv"
    reserve_csv = tmp_path / "reserve.csv"
    stage7c = tmp_path / "stage7c.py"
    write_csv(primary_csv, primary)
    write_csv(reserve_csv, reserve)
    stage7c.write_text("# frozen stage7c\n", encoding="utf-8")
    manifest = {
        "status": batch.READY_STATUS,
        "ready_to_launch_locked_rollouts": True,
        "planners": batch.EXPECTED_PLANNERS,
        "planned_primary_scenarios": len(primary),
        "planned_primary_rollouts": len(primary) * 2,
        "maximum_reserve_scenarios": len(reserve),
        "primary_collection_csv_sha256": batch.sha256_file(primary_csv),
        "reserve_collection_csv_sha256": batch.sha256_file(reserve_csv),
        "stage7c_tool_sha256": batch.sha256_file(stage7c),
        "primary_manifest_sha256": batch.canonical_rows_hash(primary),
        "reserve_manifest_sha256": batch.canonical_rows_hash(reserve),
        "planner_parameter_fingerprints": batch.current_planner_fingerprints(batch.EXPECTED_PLANNERS),
        "primary_selected_by_task": {"following_interaction": 2},
        "reserve_selected_by_task": {"following_interaction": 1},
        "selection_salt": "locked-test",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, primary_csv, reserve_csv, stage7c, db_root, manifest, primary, reserve


def validate_fixture(tmp_path: Path):
    manifest_path, primary_csv, reserve_csv, stage7c, db_root, manifest, primary, reserve = make_locked_inputs(tmp_path)

    def commits(repo: Path):
        return "nuplan-commit" if repo.name == "nuplan" else "tuplan-commit"

    result = batch.validate_locked_inputs(
        manifest_path=manifest_path,
        primary_csv=primary_csv,
        reserve_csv=reserve_csv,
        stage7c_tool=stage7c,
        db_root=db_root,
        nuplan_devkit_root=tmp_path / "nuplan",
        tuplan_garage_root=tmp_path / "tuplan",
        expected_nuplan_commit="nuplan-commit",
        expected_tuplan_commit="tuplan-commit",
        commit_resolver=commits,
    )
    assert result[0] == manifest
    assert [row["scenario_token"] for row in result[1]] == ["token-a", "token-b"]
    assert result[3]["stage7c_tool_sha256"] == batch.sha256_file(stage7c)
    return result


def make_pass_output(output: Path, row, planners=batch.EXPECTED_PLANNERS):
    output.mkdir(parents=True, exist_ok=True)
    validation = {
        "pass": True,
        "pseudo_rollout": False,
        "official_success_count": len(planners),
        "trajectory_rows": 20,
        "tensor_validation": {
            "expected_pair_count": len(planners),
            "observed_pair_count": len(planners),
            "missing_pair_count": 0,
            "passed": True,
        },
    }
    (output / "warnings.json").write_text(json.dumps({"validation": validation}), encoding="utf-8")
    (output / "simulation_schema.json").write_text(
        json.dumps(
            {
                "planner_names": planners,
                "simulated_ego_seq_shape": [1, len(planners), 10, 8],
                "same_log_alignment_passed": True,
                "strict_nuplan_token_alignment_passed": True,
                "pseudo_rollout": False,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        output / "scenario_planner_index.csv",
        [{"planner_name": planner, "status": "succeeded"} for planner in planners],
        ["planner_name", "status"],
    )
    write_csv(
        output / "scenario_alignment.csv",
        [
            {
                "target_log_name": row["log_name"],
                "actual_log_name": row["log_name"],
                "target_nuplan_scenario_token": row["scenario_token"],
                "actual_nuplan_scenario_token": row["scenario_token"],
            }
            for _ in planners
        ],
        [
            "target_log_name",
            "actual_log_name",
            "target_nuplan_scenario_token",
            "actual_nuplan_scenario_token",
        ],
    )
    (output / "simulated_ego_seq.npy").touch()
    (output / "simulated_ego_seq_mask.npy").touch()


def make_args(tmp_path: Path, fixture, *, execute=False, resume=False, retry_failed=False):
    manifest, primary, reserve, stage7c, db_root, manifest_payload, _, _ = fixture
    for directory in ["maps", "data", "nuplan", "tuplan"]:
        (tmp_path / directory).mkdir(exist_ok=True)
    return argparse.Namespace(
        manifest_path=manifest,
        primary_csv=primary,
        reserve_csv=reserve,
        nuplan_db_root=db_root,
        nuplan_map_root=tmp_path / "maps",
        nuplan_data_root=tmp_path / "data",
        nuplan_exp_root=tmp_path / "exp",
        nuplan_devkit_root=tmp_path / "nuplan",
        tuplan_garage_root=tmp_path / "tuplan",
        stage7c_tool=stage7c,
        python_executable=Path(sys.executable),
        expected_nuplan_commit="nuplan-commit",
        expected_tuplan_commit="tuplan-commit",
        output_dir=tmp_path / "out",
        start_order=1,
        end_order=0,
        max_scenarios=1,
        command_timeout_s=30,
        execute=execute,
        confirm_primary_manifest_sha256=manifest_payload["primary_manifest_sha256"] if execute else "",
        resume=resume,
        retry_failed=retry_failed,
    )


def test_validate_locked_inputs_accepts_exact_frozen_chain(tmp_path):
    validate_fixture(tmp_path)


def test_validate_locked_inputs_rejects_missing_db_commit_and_planner_fingerprint(tmp_path):
    missing_fixture = make_locked_inputs(tmp_path / "missing")
    (missing_fixture[4] / missing_fixture[6][0]["db_file"]).unlink()
    with pytest.raises(FileNotFoundError, match="locked DB file is missing"):
        batch.validate_locked_inputs(
            manifest_path=missing_fixture[0],
            primary_csv=missing_fixture[1],
            reserve_csv=missing_fixture[2],
            stage7c_tool=missing_fixture[3],
            db_root=missing_fixture[4],
            nuplan_devkit_root=tmp_path / "missing/nuplan",
            tuplan_garage_root=tmp_path / "missing/tuplan",
            expected_nuplan_commit="nuplan-commit",
            expected_tuplan_commit="tuplan-commit",
            commit_resolver=lambda _: "nuplan-commit",
        )

    commit_fixture = make_locked_inputs(tmp_path / "commit")
    with pytest.raises(ValueError, match="nuPlan commit mismatch"):
        batch.validate_locked_inputs(
            manifest_path=commit_fixture[0],
            primary_csv=commit_fixture[1],
            reserve_csv=commit_fixture[2],
            stage7c_tool=commit_fixture[3],
            db_root=commit_fixture[4],
            nuplan_devkit_root=tmp_path / "commit/nuplan",
            tuplan_garage_root=tmp_path / "commit/tuplan",
            expected_nuplan_commit="nuplan-commit",
            expected_tuplan_commit="tuplan-commit",
            commit_resolver=lambda _: "wrong-commit",
        )

    planner_fixture = make_locked_inputs(tmp_path / "planner")
    payload = json.loads(planner_fixture[0].read_text())
    payload["planner_parameter_fingerprints"][batch.EXPECTED_PLANNERS[0]] = "0" * 64
    planner_fixture[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="planner fingerprints"):
        batch.validate_locked_inputs(
            manifest_path=planner_fixture[0],
            primary_csv=planner_fixture[1],
            reserve_csv=planner_fixture[2],
            stage7c_tool=planner_fixture[3],
            db_root=planner_fixture[4],
            nuplan_devkit_root=tmp_path / "planner/nuplan",
            tuplan_garage_root=tmp_path / "planner/tuplan",
            expected_nuplan_commit="nuplan-commit",
            expected_tuplan_commit="tuplan-commit",
            commit_resolver=lambda _: "nuplan-commit",
        )


def test_validate_locked_inputs_rejects_hash_order_duplicate_missing_db_and_commit(tmp_path):
    fixture = make_locked_inputs(tmp_path)
    manifest_path, primary_csv, reserve_csv, stage7c, db_root, manifest, primary, reserve = fixture

    stage7c.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        batch.validate_locked_inputs(
            manifest_path=manifest_path,
            primary_csv=primary_csv,
            reserve_csv=reserve_csv,
            stage7c_tool=stage7c,
            db_root=db_root,
            nuplan_devkit_root=tmp_path / "nuplan",
            tuplan_garage_root=tmp_path / "tuplan",
            expected_nuplan_commit="nu",
            expected_tuplan_commit="tu",
            commit_resolver=lambda _: "nu",
        )

    stage7c.write_text("# frozen stage7c\n", encoding="utf-8")
    bad_primary = [dict(primary[1]), dict(primary[0])]
    bad_primary[0]["collection_order"] = "1"
    bad_primary[1]["collection_order"] = "2"
    write_csv(primary_csv, bad_primary)
    manifest["primary_collection_csv_sha256"] = batch.sha256_file(primary_csv)
    manifest["primary_manifest_sha256"] = batch.canonical_rows_hash(bad_primary)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="task_rank is not contiguous"):
        batch.validate_locked_inputs(
            manifest_path=manifest_path,
            primary_csv=primary_csv,
            reserve_csv=reserve_csv,
            stage7c_tool=stage7c,
            db_root=db_root,
            nuplan_devkit_root=tmp_path / "nuplan",
            tuplan_garage_root=tmp_path / "tuplan",
            expected_nuplan_commit="nu",
            expected_tuplan_commit="tu",
            commit_resolver=lambda _: "nu",
        )

    bad_primary[1]["scenario_token"] = bad_primary[0]["scenario_token"]
    write_csv(primary_csv, bad_primary)
    manifest["primary_collection_csv_sha256"] = batch.sha256_file(primary_csv)
    manifest["primary_manifest_sha256"] = batch.canonical_rows_hash(bad_primary)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate or empty"):
        batch.validate_locked_inputs(
            manifest_path=manifest_path,
            primary_csv=primary_csv,
            reserve_csv=reserve_csv,
            stage7c_tool=stage7c,
            db_root=db_root,
            nuplan_devkit_root=tmp_path / "nuplan",
            tuplan_garage_root=tmp_path / "tuplan",
            expected_nuplan_commit="nu",
            expected_tuplan_commit="tu",
            commit_resolver=lambda _: "nu",
        )


def test_audit_stage7c_output_requires_pair_and_strict_identity(tmp_path):
    row = locked_row(1, 1, "token-a", "primary_gross")
    output = tmp_path / "stage7c"
    make_pass_output(output, row)
    assert batch.audit_stage7c_output(output, batch.EXPECTED_PLANNERS, row)["pass"] is True
    schema = json.loads((output / "simulation_schema.json").read_text())
    schema["strict_nuplan_token_alignment_passed"] = False
    (output / "simulation_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    audit = batch.audit_stage7c_output(output, batch.EXPECTED_PLANNERS, row)
    assert audit["pass"] is False
    assert audit["failure_category"] == "ALIGNMENT_FAILED"


def test_batch_dry_run_then_execute_and_resume_skip(tmp_path, monkeypatch):
    fixture = make_locked_inputs(tmp_path)
    validated = validate_fixture(tmp_path / "validated")
    args = make_args(tmp_path, fixture)
    monkeypatch.setattr(batch, "validate_locked_inputs", lambda **_: validated)
    assert batch.run(args) == 0
    assert json.loads((args.output_dir / "batch_state.json").read_text())["execution_mode"] == "dry_run"
    frozen_batch = json.loads((args.output_dir / "batch_manifest.json").read_text())
    assert frozen_batch["batch_tool_sha256"] == batch.sha256_file(Path(batch.__file__).resolve())
    assert frozen_batch["command_timeout_s"] == 30

    calls = []

    def fake_run_stage7c(run_args, row, attempt_dir):
        calls.append(row["scenario_token"])
        make_pass_output(attempt_dir / "stage7c_output", row)
        return 0

    monkeypatch.setattr(batch, "run_stage7c", fake_run_stage7c)
    args.execute = True
    args.resume = True
    args.confirm_primary_manifest_sha256 = fixture[5]["primary_manifest_sha256"]
    assert batch.run(args) == 0
    assert calls == ["token-a"]
    state = json.loads((args.output_dir / "batch_state.json").read_text())
    assert state["status_counts"] == {"PENDING": 1, "SUCCEEDED": 1}

    monkeypatch.setattr(batch, "run_stage7c", lambda *_: pytest.fail("resume reran a valid output"))
    assert batch.run(args) == 0


def test_corrupt_attempt_is_not_overwritten_without_retry_flag(tmp_path, monkeypatch):
    fixture = make_locked_inputs(tmp_path)
    validated = validate_fixture(tmp_path / "validated")
    args = make_args(tmp_path, fixture, execute=False)
    monkeypatch.setattr(batch, "validate_locked_inputs", lambda **_: validated)
    assert batch.run(args) == 0
    row = fixture[6][0]
    corrupt = args.output_dir / "rollouts" / batch.scenario_slug(row) / "attempt_001"
    corrupt.mkdir(parents=True)
    args.execute = True
    args.resume = True
    args.confirm_primary_manifest_sha256 = fixture[5]["primary_manifest_sha256"]
    monkeypatch.setattr(batch, "run_stage7c", lambda *_: pytest.fail("corrupt output was overwritten"))
    assert batch.run(args) == 2
    assert corrupt.is_dir()
    assert not (corrupt.parent / "attempt_002").exists()


def test_batch_lock_and_deterministic_reserve_proposal(tmp_path):
    lock = tmp_path / "batch.lock"
    with batch.BatchLock(lock):
        with pytest.raises(RuntimeError, match="batch lock already exists"):
            with batch.BatchLock(lock):
                pass
    assert not lock.exists()

    statuses = [
        {**locked_row(2, 2, "b", "primary_gross"), "failure_category": "ALIGNMENT_FAILED"},
        {**locked_row(1, 1, "a", "primary_gross"), "failure_category": "ENVIRONMENT_OR_CONFIG_FAILURE"},
        {**locked_row(3, 3, "c", "primary_gross"), "failure_category": "TRAJECTORY_EXPORT_FAILED"},
    ]
    reserves = [
        locked_row(2, 2, "r2", "technical_quality_reserve"),
        locked_row(1, 1, "r1", "technical_quality_reserve"),
    ]
    proposal = batch.reserve_proposal(statuses, reserves)
    assert [(row["failed_primary_token"], row["reserve_token"]) for row in proposal] == [
        ("b", "r1"),
        ("c", "r2"),
    ]
    assert all(row["approval_status"] == "PROPOSED_NOT_APPROVED_NOT_EXECUTED" for row in proposal)


def test_failure_classification_distinguishes_environment_from_technical(tmp_path):
    log = tmp_path / "driver.log"
    log.write_text("ModuleNotFoundError: No module named x", encoding="utf-8")
    assert batch.classify_process_failure(1, log) == "ENVIRONMENT_OR_CONFIG_FAILURE"
    log.write_text("scenario worker failed", encoding="utf-8")
    assert batch.classify_process_failure(1, log) == "OFFICIAL_COMMAND_FAILED"
    assert batch.classify_process_failure(124, log) == "COMMAND_TIMEOUT"
