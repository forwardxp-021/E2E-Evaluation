#!/usr/bin/env python3
"""Prepare and run the frozen Stage7L-D 80 x 5 official confirmation grid.

This runner is deliberately planner-only.  It does not import torch, load a
checkpoint, export an embedding, or calculate BDD/MMD.  Each frozen
scenario-dose cell has an immutable plan row and an append-preserving attempt
history so an interrupted run can resume without replacing scenarios or
overwriting a valid official rollout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_freeze_confirmation_roster import development_sets


EXPECTED_PROTOCOL_SHA256 = "f5a8b2df5ed60c0384e8181feceab33f3c6f048780e95aab851184e49247490a"
EXPECTED_ROSTER_SHA256 = "90ec9b427636cefc59e6d7ace2507ac8364747e2a38964124be08fdc2a10acf9"
DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")
TRANSITION_LENGTH_M = {"dose0": 60.0, "dose25": 58.5, "dose50": 57.0, "dose75": 55.5, "dose100": 54.0}
PLANNED_FIELDS = (
    "cell_id", "collection_order", "scenario_token", "log_name", "direction", "dose",
    "transition_length_m", "planned", "attempt_id", "start_time", "end_time", "exit_code",
    "official_run_status", "trajectory_available", "failure_category", "retry_count",
)
ATTEMPT_FIELDS = (
    "cell_id", "collection_order", "scenario_token", "log_name", "direction", "dose",
    "transition_length_m", "planned", "attempt_id", "start_time", "end_time", "exit_code",
    "official_run_status", "trajectory_available", "failure_category", "failure_detail",
    "retry_count", "attempt_dir", "trajectory_csv", "planner_audit_path",
)
SUMMARY_FIELDS = (
    "cell_id", "collection_order", "scenario_token", "log_name", "direction", "dose",
    "transition_length_m", "attempt_id", "official_run_status", "trajectory_available",
    "failure_category", "failure_detail", "retry_count", "exit_code", "start_time", "end_time",
    "duration_seconds", "attempt_dir", "trajectory_csv", "official_runs_root",
    "planner_audit_path", "strict_alignment_passed", "trajectory_row_count",
)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def cell_id(collection_order: int, token: str, dose: str) -> str:
    return f"S{collection_order:03d}_{token}_{dose}"


def validate_preflight(args: argparse.Namespace) -> tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Any]]:
    protocol_sha = sha256_file(args.protocol_config)
    roster_sha = sha256_file(args.roster_csv)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(f"protocol SHA mismatch: {protocol_sha} != {EXPECTED_PROTOCOL_SHA256}")
    if roster_sha != EXPECTED_ROSTER_SHA256:
        raise ValueError(f"roster SHA mismatch: {roster_sha} != {EXPECTED_ROSTER_SHA256}")
    protocol = read_json(args.protocol_config)
    authorization = read_json(args.authorization_manifest)
    roster = read_csv(args.roster_csv)
    runnability = read_csv(args.runnability_csv)
    development = read_csv(args.development_ledger)
    maneuvers = read_json(args.maneuver_manifest).get("maneuvers", [])
    maneuver_tokens = {str(row["scenario_token"]) for row in maneuvers}
    tokens = [row["scenario_token"] for row in roster]
    logs = [row["log_name"] for row in roster]
    # Reuse the exact frozen Stage7L-C definition: all historical tokens are
    # scenario exclusions, while only the 26 explicitly labelled Stage7L-B
    # development logs form the development log-disjointness population.
    development_tokens, development_logs = development_sets(development)
    runnable_by_token = {row["scenario_token"]: row for row in runnability}
    checks = {
        "protocol_sha_exact": protocol_sha == EXPECTED_PROTOCOL_SHA256,
        "roster_sha_exact": roster_sha == EXPECTED_ROSTER_SHA256,
        "authorization_status": authorization.get("status") == "STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED",
        "stage7l_d_not_started": authorization.get("stage7l_d_started") is False,
        "stage7l_e_not_authorized": authorization.get("stage7l_e_representation_evaluation_authorized_now") is False,
        "scenario_count_80": len(roster) == 80,
        "left_count_15": sum(row.get("direction") == "left" for row in roster) == 15,
        "right_count_65": sum(row.get("direction") == "right" for row in roster) == 65,
        "unique_logs_79": len(set(logs)) == 79,
        "duplicate_token_zero": len(set(tokens)) == len(tokens) == 80,
        "development_scenario_overlap_zero": not bool(set(tokens) & development_tokens),
        "development_log_overlap_zero": not bool(set(logs) & development_logs),
        "official_runnable_80": len(runnability) == 80 and all(
            runnable_by_token.get(token, {}).get("official_query_runnable", "").lower() == "true" for token in tokens
        ),
        "dynamic_clear_80": all(row.get("dynamic_clearance_status") == "DYNAMIC_CLEAR" for row in roster),
        "static_eligible_80": all(
            row.get("source_lane_id") and row.get("target_lane_id") and row.get("source_reference_sha256")
            and row.get("target_reference_sha256") and row.get("official_query_runnable", "").lower() == "true"
            for row in roster
        ),
        "maneuver_manifest_exact_80": len(maneuvers) == 80 and set(tokens) == maneuver_tokens,
        "treatment_axis_exact": protocol["treatment"]["dose_transition_length_m"] == TRANSITION_LENGTH_M,
        "trigger_exact": float(protocol["treatment"]["trigger_s_route_m"]) == 12.0,
        "planner_horizon_exact": float(protocol["treatment"]["planner_horizon_s"]) == 0.4,
        "sampling_exact": float(protocol["treatment"]["sampling_interval_s"]) == 0.1,
        "scenario_horizon_exact": float(protocol["treatment"]["scenario_horizon_s"]) == 15.0,
        "target_speed_exact": float(protocol["treatment"]["target_speed_mps"]) == 5.0,
        "accel_limit_exact": float(protocol["treatment"]["accel_limit_mps2"]) == 1.0,
        "background_exact": protocol["treatment"]["background_mode"] == "closed_loop_nonreactive_agents",
    }
    for dose, expected in TRANSITION_LENGTH_M.items():
        item = protocol["treatment"]["hydra_planner_configs"][dose]
        path = args.repo_root / item["path"]
        checks[f"{dose}_config_sha"] = path.is_file() and sha256_file(path) == item["sha256"]
        checks[f"{dose}_transition_exact"] = float(protocol["treatment"]["dose_transition_length_m"][dose]) == expected
    planner_item = protocol["source_assets"]["planner_code"]
    planner_path = args.repo_root / planner_item["path"]
    checks["planner_code_sha"] = planner_path.is_file() and sha256_file(planner_path) == planner_item["sha256"]
    for path in (args.nuplan_db_root, args.nuplan_map_root, args.nuplan_data_root, args.nuplan_devkit_root, args.tuplan_garage_root):
        checks[f"path_exists:{path.name}"] = path.exists()
    checks["python_executable_exists"] = args.python_executable.is_file()
    checks["all_80_db_files_exist"] = all((args.nuplan_db_root / row["db_file"]).exists() for row in roster)
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"Stage7L-D preflight failed: {failed}")
    provenance = {
        "schema_version": "stage7l_d_preflight_v1",
        "status": "STAGE7L_D_PREFLIGHT_VALIDATED_NOT_YET_ROLLED_OUT",
        "validated_at": iso_now(),
        "execution_start_commit": git_head(args.repo_root),
        "protocol_sha256": protocol_sha,
        "authorization_sha256": sha256_file(args.authorization_manifest),
        "roster_sha256": roster_sha,
        "maneuver_manifest_sha256": sha256_file(args.maneuver_manifest),
        "planner_code_sha256": sha256_file(planner_path),
        "dose_config_sha256": {
            dose: sha256_file(args.repo_root / protocol["treatment"]["hydra_planner_configs"][dose]["path"])
            for dose in DOSES
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "mechanism_evaluator_sha256": sha256_file(args.mechanism_evaluator),
        "gate_evaluator_sha256": sha256_file(args.gate_evaluator),
        "checks": checks,
        "counts": {"scenario": 80, "left": 15, "right": 65, "unique_logs": 79, "planned_cells": 400},
        "representation_boundary": {
            "embedding_read": False, "checkpoint_read": False, "bdd_computed": False, "mmd_computed": False,
        },
    }
    return protocol, roster, provenance


def initial_plan(roster: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in sorted(roster, key=lambda row: int(row["collection_order"])):
        order = int(item["collection_order"])
        for dose in DOSES:
            cid = cell_id(order, item["scenario_token"], dose)
            rows.append({
                "cell_id": cid, "collection_order": order, "scenario_token": item["scenario_token"],
                "log_name": item["log_name"], "direction": item["direction"], "dose": dose,
                "transition_length_m": TRANSITION_LENGTH_M[dose], "planned": True,
                "attempt_id": f"{cid}_A01", "start_time": "", "end_time": "", "exit_code": "",
                "official_run_status": "PLANNED_NOT_STARTED", "trajectory_available": False,
                "failure_category": "", "retry_count": 0,
            })
    if len(rows) != 400 or len({row["cell_id"] for row in rows}) != 400:
        raise ValueError("planned ledger is not the exact 80 x 5 frozen grid")
    return rows


def prepare_output(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    protocol, roster, provenance = validate_preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.output_dir / "preflight_audit.json"
    plan_path = args.output_dir / "planned_rollout_ledger.csv"
    contract_path = args.output_dir / "stage7l_d_execution_contract.json"
    if preflight_path.is_file():
        existing = read_json(preflight_path)
        immutable_keys = (
            "execution_start_commit", "protocol_sha256", "authorization_sha256", "roster_sha256",
            "maneuver_manifest_sha256", "planner_code_sha256", "dose_config_sha256", "runner_sha256",
            "mechanism_evaluator_sha256", "gate_evaluator_sha256",
        )
        mismatch = [key for key in immutable_keys if existing.get(key) != provenance.get(key)]
        if mismatch:
            raise ValueError(f"resume provenance mismatch: {mismatch}")
        plan = read_csv(plan_path)
        if len(plan) != 400:
            raise ValueError("resume planned ledger is not 400 rows")
        return plan, existing
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"new Stage7L-D output_dir is not empty: {args.output_dir}")
    plan = initial_plan(roster)
    write_csv_atomic(plan_path, plan, PLANNED_FIELDS)
    contract = {
        "schema_version": "stage7l_d_execution_contract_v1",
        "status": "FROZEN_BEFORE_FIRST_CONFIRMATION_ROLLOUT",
        "planned_order": "confirmation_roster collection_order x dose0,dose25,dose50,dose75,dose100",
        "replacement_allowed": False,
        "maximum_automatic_infrastructure_retries_per_cell": args.max_infrastructure_retries,
        "valid_rollout_retry_forbidden": True,
        "safety_aggregation": {
            "denominator": "all_80_frozen_scenarios",
            "level": "scenario_level_conservative_across_all_five_frozen_doses",
            "official_success": "all_five_doses_have_successful_official_rollout_and_valid_trajectory",
            "lane_change_completion": "all_five_doses_complete_the_lane_change",
            "offroad": "any_frozen_dose_offroad",
            "responsible_collision": "any_frozen_dose_has_at_fault_collision",
            "rationale": "faithful implementation of frozen population=all_80_frozen_scenarios_no_post_treatment_deletion; fixed before results",
        },
        "representation_boundary": "NO_EMBEDDING_CHECKPOINT_BDD_OR_MMD",
    }
    provenance["execution_contract_sha256_pending_write"] = True
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance["execution_contract_sha256"] = sha256_file(contract_path)
    provenance.pop("execution_contract_sha256_pending_write", None)
    preflight_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan, provenance


def build_context(path: Path, item: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fields = (
        "collection_order", "source_global_scenario_index", "task", "source_task", "scenario_type",
        "log_name", "scenario_token", "scene_token", "db_file", "selection_role", "actual_nuplan_token",
    )
    row = {
        "collection_order": 1, "source_global_scenario_index": 0, "task": "stage7l_d_confirmation",
        "source_task": "pre_treatment_frozen_confirmation", "scenario_type": "unknown",
        "log_name": item["log_name"], "scenario_token": item["scenario_token"],
        "scene_token": item["scenario_token"], "db_file": f"{item['log_name']}.db",
        "selection_role": "STAGE7L_D_FROZEN_CONFIRMATION_NO_REPLACEMENT",
        "actual_nuplan_token": item["scenario_token"],
    }
    write_csv_atomic(path / "merged_metadata.csv", [row], fields)


def stage7c_command(args: argparse.Namespace, item: Mapping[str, Any], attempt_dir: Path) -> List[str]:
    context = attempt_dir / "context"
    build_context(context, item)
    planner = f"stage7l_b2_pure_lateral_{item['dose']}"
    searchpath = (
        f"[file://{(args.repo_root / 'configs/stage7l_hydra').resolve()},"
        "pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.config.experiments]"
    )
    template = " ".join([
        str(args.python_executable.resolve()),
        str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()),
        "+simulation=closed_loop_nonreactive_agents", "{planner_hydra_overrides}",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{args.nuplan_db_root.resolve()}/{{target_log_name}}.db]",
        "+scenario_builder.scenario_mapping.scenario_map.unknown=[15.0,0.0]",
        "scenario_filter=all_scenarios", "{scenario_hydra_overrides}",
        "worker=single_machine_thread_pool", "worker.max_workers=1", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "gpu=false", "experiment_name=stage7l_d_confirmation_v1",
        "job_name=closed_loop_nonreactive_agents_{planner_name_safe}", "output_dir={output_dir}",
    ])
    return [
        str(args.python_executable.resolve()), str(args.stage7c_tool.resolve()),
        "--context_dir", str(context.resolve()), "--nuplan_db_root", str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root", str(args.nuplan_map_root.resolve()), "--output_dir", str((attempt_dir / "stage7c_output").resolve()),
        "--planners", planner, "--max_scenarios", "1", "--min_timesteps", "20",
        "--require_same_scenario_alignment", "--require_strict_nuplan_token_alignment",
        "--allow_unsafe_pickle_artifacts", "--allow_external_planner_name", "--hydra_searchpath", searchpath,
        "--command_timeout_s", str(args.command_timeout_s), "--nuplan_simulation_command_template", template,
    ]


def inspect_attempt(attempt_dir: Path, item: Mapping[str, Any], exit_code: int) -> Dict[str, Any]:
    output = attempt_dir / "stage7c_output"
    progress_path = output / "stage7c_progress.json"
    schema_path = output / "simulation_schema.json"
    trajectory_path = output / "simulated_ego_trajectory.csv"
    progress = read_json(progress_path) if progress_path.is_file() else {}
    records = progress.get("task_records", [])
    record = records[-1] if records else {}
    trajectory_rows = read_csv(trajectory_path)
    exact_rows = [row for row in trajectory_rows if row.get("scene_token") == item["scenario_token"]]
    schema = read_json(schema_path) if schema_path.is_file() else {}
    strict_alignment = bool(schema.get("strict_nuplan_token_alignment_passed"))
    trajectory_available = len(exact_rows) >= 20
    succeeded = record.get("status") == "succeeded" and trajectory_available and strict_alignment and exit_code == 0
    if succeeded:
        status, category, detail = "SUCCEEDED", "", ""
    else:
        status = str(record.get("status") or "PROCESS_FAILED").upper()
        category = "INFRASTRUCTURE_RUNTIME"
        detail = f"stage7c_exit={exit_code};task_status={record.get('status','missing')};trajectory_rows={len(exact_rows)};strict_alignment={strict_alignment}"
    audit_path = attempt_dir / "planner_audits" / f"planner_audit_{item['scenario_token']}_{item['dose']}.json"
    official_root = output / "official_nuplan_runs"
    return {
        "official_run_status": status, "trajectory_available": trajectory_available,
        "failure_category": category, "failure_detail": detail,
        "trajectory_csv": str(trajectory_path.resolve()) if trajectory_path.is_file() else "",
        "official_runs_root": str(official_root.resolve()),
        "planner_audit_path": str(audit_path.resolve()) if audit_path.is_file() else "",
        "strict_alignment_passed": strict_alignment, "trajectory_row_count": len(exact_rows),
    }


def recover_running_attempts(attempts: List[Dict[str, str]]) -> None:
    now = iso_now()
    for row in attempts:
        if row.get("official_run_status") == "RUNNING":
            row["official_run_status"] = "INTERRUPTED_BEFORE_VALID_ROLLOUT"
            row["end_time"] = now
            row["exit_code"] = "INTERRUPTED"
            row["trajectory_available"] = False
            row["failure_category"] = "INFRASTRUCTURE_RUNTIME"
            row["failure_detail"] = "prior runner exited while attempt ledger row was RUNNING"


def latest_successes(attempts: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for row in attempts:
        if row.get("official_run_status") == "SUCCEEDED" and str(row.get("trajectory_available")).lower() == "true":
            result[row["cell_id"]] = row
    return result


def write_summary(path: Path, plan: List[Dict[str, Any]], attempts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_cell: Dict[str, List[Dict[str, str]]] = {}
    for row in attempts:
        by_cell.setdefault(row["cell_id"], []).append(row)
    summary: List[Dict[str, Any]] = []
    for planned in plan:
        group = by_cell.get(planned["cell_id"], [])
        successful = [row for row in group if row.get("official_run_status") == "SUCCEEDED" and str(row.get("trajectory_available")).lower() == "true"]
        selected = successful[-1] if successful else (group[-1] if group else planned)
        row = dict(planned)
        row.update(selected)
        row["retry_count"] = max(0, len(group) - 1)
        summary.append(row)
    write_csv_atomic(path, summary, SUMMARY_FIELDS)
    return summary


def update_progress(args: argparse.Namespace, summary: List[Dict[str, Any]], started: float, current: str = "") -> None:
    completed = sum(row.get("official_run_status") == "SUCCEEDED" and str(row.get("trajectory_available")).lower() == "true" for row in summary)
    attempted = sum(row.get("official_run_status") not in {"", "PLANNED_NOT_STARTED"} for row in summary)
    elapsed = max(time.monotonic() - started, 0.0)
    avg = elapsed / attempted if attempted else None
    remaining = 400 - completed
    eta_seconds = None if avg is None else avg * remaining
    progress = {
        "schema_version": "stage7l_d_progress_v1", "updated_at": iso_now(), "planned_cells": 400,
        "successful_cells": completed, "attempted_cells": attempted, "completion_fraction": completed / 400,
        "current_cell": current, "elapsed_seconds": elapsed, "average_seconds_per_attempted_cell": avg,
        "estimated_remaining_seconds": eta_seconds, "representation_boundary": "NO_EMBEDDING_CHECKPOINT_BDD_OR_MMD",
    }
    (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    plan, provenance = prepare_output(args)
    if args.prepare_only:
        return {"status": "STAGE7L_D_PREFLIGHT_VALIDATED_AND_400_CELLS_PLANNED", "preflight": provenance}
    attempts_path = args.output_dir / "attempt_ledger.csv"
    summary_path = args.output_dir / "official_run_summary.csv"
    attempts = read_csv(attempts_path)
    recover_running_attempts(attempts)
    write_csv_atomic(attempts_path, attempts, ATTEMPT_FIELDS)
    summary = write_summary(summary_path, plan, attempts)
    successes = latest_successes(attempts)
    started = time.monotonic()
    update_progress(args, summary, started)
    for planned in plan:
        cid = planned["cell_id"]
        if cid in successes:
            continue
        existing = [row for row in attempts if row["cell_id"] == cid]
        retry_limit = 1 + args.max_infrastructure_retries
        while len(existing) < retry_limit and cid not in successes:
            attempt_number = len(existing) + 1
            attempt_id = f"{cid}_A{attempt_number:02d}"
            attempt_dir = args.output_dir / "cells" / cid / attempt_id
            attempt_dir.mkdir(parents=True, exist_ok=False)
            command = stage7c_command(args, planned, attempt_dir)
            start_iso = iso_now()
            row: Dict[str, Any] = {
                **planned, "attempt_id": attempt_id, "start_time": start_iso, "end_time": "", "exit_code": "",
                "official_run_status": "RUNNING", "trajectory_available": False, "failure_category": "",
                "failure_detail": "", "retry_count": attempt_number - 1, "attempt_dir": str(attempt_dir.resolve()),
                "trajectory_csv": "", "planner_audit_path": "",
            }
            attempts.append(row)
            write_csv_atomic(attempts_path, attempts, ATTEMPT_FIELDS)
            update_progress(args, summary, started, cid)
            print(f"[Stage7L-D] START {cid} attempt={attempt_number}/{retry_limit}", flush=True)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([
                str(args.nuplan_devkit_root.resolve()), str(args.tuplan_garage_root.resolve()),
                str(args.repo_root.resolve()), env.get("PYTHONPATH", ""),
            ])
            env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            env["NUPLAN_DATA_ROOT"] = str(args.nuplan_data_root.resolve())
            env["NUPLAN_MAPS_ROOT"] = str(args.nuplan_map_root.resolve())
            env["NUPLAN_EXP_ROOT"] = str(args.nuplan_exp_root.resolve())
            env["STAGE7L_MANEUVER_MANIFEST"] = str(args.maneuver_manifest.resolve())
            env["STAGE7L_PLANNER_AUDIT_DIR"] = str((attempt_dir / "planner_audits").resolve())
            log_path = attempt_dir / "stage7l_d_cell.log"
            cell_started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                log.write("argv: " + json.dumps(command, ensure_ascii=False) + "\n\n")
                log.flush()
                process = subprocess.run(command, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            inspected = inspect_attempt(attempt_dir, planned, process.returncode)
            row.update(inspected)
            row["end_time"] = iso_now()
            row["exit_code"] = process.returncode
            row["duration_seconds"] = time.monotonic() - cell_started
            write_csv_atomic(attempts_path, attempts, ATTEMPT_FIELDS)
            existing = [item for item in attempts if item["cell_id"] == cid]
            if row["official_run_status"] == "SUCCEEDED":
                successes[cid] = row
            summary = write_summary(summary_path, plan, attempts)
            update_progress(args, summary, started, cid)
            print(
                f"[Stage7L-D] DONE {cid} status={row['official_run_status']} "
                f"trajectory={row['trajectory_available']} duration={row['duration_seconds']:.1f}s",
                flush=True,
            )
        # A valid treatment outcome is never retried; an exhausted infrastructure failure remains recorded.
    summary = write_summary(summary_path, plan, attempts)
    update_progress(args, summary, started)
    succeeded = sum(row.get("official_run_status") == "SUCCEEDED" and str(row.get("trajectory_available")).lower() == "true" for row in summary)
    result = {
        "schema_version": "stage7l_d_official_execution_v1",
        "status": "STAGE7L_D_OFFICIAL_ROLLOUT_INVENTORY_COMPLETE" if len(summary) == 400 else "STAGE7L_D_EXECUTION_LEDGER_INVALID",
        "planned_cells": 400, "successful_cells": succeeded, "failed_cells": 400 - succeeded,
        "attempt_count": len(attempts), "replacement_count": 0,
        "embedding_read": False, "checkpoint_read": False, "bdd_computed": False, "mmd_computed": False,
    }
    (args.output_dir / "official_execution_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=root)
    parser.add_argument("--protocol_config", type=Path, default=root / "configs/stage7l_c_prospective_confirmation_protocol_v1.json")
    parser.add_argument("--roster_csv", type=Path, default=root / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_roster.csv")
    parser.add_argument("--maneuver_manifest", type=Path, default=root / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_maneuver_manifest.json")
    parser.add_argument("--runnability_csv", type=Path, default=root / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_runnability_audit.csv")
    parser.add_argument("--development_ledger", type=Path, default=root / "outputs/stage7l_b_final_development_freeze_v1/stage7l_b_final_prior_exclusion_ledger.csv")
    parser.add_argument("--authorization_manifest", type=Path, default=root / "docs/stage7l_c_blind_confirmation_authorization_manifest_v1.json")
    parser.add_argument("--mechanism_evaluator", type=Path, default=root / "tools/stage7l_extract_confirmation_metrics.py")
    parser.add_argument("--gate_evaluator", type=Path, default=root / "tools/stage7l_evaluate_confirmation_gates.py")
    parser.add_argument("--nuplan_db_root", type=Path, default=workspace / "nuplan/dataset/data/cache/locked_pool_expanded_v1")
    parser.add_argument("--nuplan_map_root", type=Path, default=workspace / "nuplan/dataset/maps")
    parser.add_argument("--nuplan_data_root", type=Path, default=workspace / "nuplan/dataset")
    parser.add_argument("--nuplan_exp_root", type=Path, default=root / "outputs")
    parser.add_argument("--nuplan_devkit_root", type=Path, default=workspace / "nuplan-devkit")
    parser.add_argument("--tuplan_garage_root", type=Path, default=workspace / "tuplan_garage")
    parser.add_argument("--stage7c_tool", type=Path, default=root / "tools/stage7c1_run_nuplan_simulation.py")
    parser.add_argument("--python_executable", type=Path, default=Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9"))
    parser.add_argument("--output_dir", type=Path, default=root / "outputs/stage7l_d_one_time_confirmation_v1")
    parser.add_argument("--command_timeout_s", type=int, default=1200)
    parser.add_argument("--max_infrastructure_retries", type=int, default=1, choices=(0, 1))
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
