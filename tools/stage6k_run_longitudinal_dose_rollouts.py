#!/usr/bin/env python3
"""Run the frozen Stage 6K graded pure-longitudinal jobs with resume support."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage6j_run_pure_longitudinal_rollouts as stage6j  # noqa: E402
from tools.stage7_m6_4b_run_locked_rollouts import audit_stage7c_output  # noqa: E402
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES  # noqa: E402


SCHEMA_VERSION = "stage6k_longitudinal_dose_batch_v1"
FREEZE_STATUS = "FROZEN_BEFORE_LONGITUDINAL_DOSE_ROLLOUTS"
JOB_FIELDS = [
    "collection_order", "source_collection_order", "dose", "dose_label", "planner_a", "planner_b",
    "source_global_scenario_index", "task", "source_task", "scenario_type", "log_name", "scenario_token",
    "scene_token", "db_file", "selection_role",
]
STATUS_FIELDS = JOB_FIELDS + [
    "status", "failure_category", "attempt", "return_code", "official_success_count", "trajectory_rows",
    "same_log_alignment_passed", "strict_token_alignment_passed", "started_utc", "ended_utc",
    "duration_seconds", "stage7c_output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Stage 6K dose-response rollouts.")
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--locked_jobs_csv", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--nuplan_data_root", type=Path, required=True)
    parser.add_argument("--nuplan_exp_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--tuplan_garage_root", type=Path, required=True)
    parser.add_argument("--stage7c_tool", type=Path, required=True)
    parser.add_argument("--python_executable", type=Path, required=True)
    parser.add_argument("--expected_nuplan_commit", required=True)
    parser.add_argument("--expected_tuplan_commit", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_order", type=int, default=1)
    parser.add_argument("--end_order", type=int, default=0)
    parser.add_argument("--max_jobs", type=int, default=0)
    parser.add_argument("--command_timeout_s", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm_locked_jobs_sha256", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(JOB_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Stage 6K locked jobs are missing fields: {missing}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def planners_for(row: Mapping[str, str]) -> List[str]:
    planners = [str(row["planner_a"]), str(row["planner_b"])]
    if len(set(planners)) != 2 or not all(planner in PLANNER_PROFILES for planner in planners):
        raise ValueError(f"Invalid Stage 6K planner pair for order {row['collection_order']}: {planners}")
    return planners


def scenario_dir(output_dir: Path, row: Mapping[str, str]) -> Path:
    return output_dir / "rollouts" / f"order_{int(row['collection_order']):04d}_{row['dose_label']}_{row['scenario_token']}"


def attempt_dirs(path: Path) -> List[Path]:
    return sorted(path.glob("attempt_*")) if path.is_dir() else []


def successful_attempt(path: Path, row: Mapping[str, str]) -> Optional[Path]:
    planners = planners_for(row)
    for attempt in reversed(attempt_dirs(path)):
        if audit_stage7c_output(attempt / "stage7c_output", planners, row).get("pass"):
            return attempt
    return None


def validate_inputs(args: argparse.Namespace) -> tuple[Dict[str, Any], List[Dict[str, str]], str]:
    manifest = stage6j.read_json(args.freeze_manifest.resolve())
    rows = read_csv(args.locked_jobs_csv.resolve())
    if manifest.get("status") != FREEZE_STATUS:
        raise ValueError(f"Stage 6K freeze status is not ready: {manifest.get('status')!r}")
    locked_sha = stage6j.sha256_file(args.locked_jobs_csv.resolve())
    if manifest.get("outputs", {}).get("locked_jobs", {}).get("sha256") != locked_sha:
        raise ValueError("Stage 6K locked jobs SHA-256 differs from the freeze manifest")
    if len(rows) != int(manifest.get("job_audit", {}).get("job_count", -1)):
        raise ValueError("Stage 6K locked job count differs from the freeze manifest")
    if [int(row["collection_order"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Stage 6K collection_order must be contiguous and start at 1")
    fingerprints = manifest.get("profile_audit", {}).get("planner_parameter_fingerprints", {})
    for row in rows:
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError(f"DB/log identity mismatch for Stage 6K order {row['collection_order']}")
        if not (args.nuplan_db_root.resolve() / row["db_file"]).is_file():
            raise FileNotFoundError(args.nuplan_db_root.resolve() / row["db_file"])
        for planner in planners_for(row):
            current = stage6j.canonical_hash(PLANNER_PROFILES[planner]["parameters"])
            if current != fingerprints.get(planner):
                raise ValueError(f"Stage 6K planner fingerprint changed after freeze: {planner}")
    for path in [args.python_executable, args.stage7c_tool]:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    for path in [args.nuplan_map_root, args.nuplan_data_root, args.nuplan_exp_root, args.nuplan_devkit_root, args.tuplan_garage_root]:
        if not path.resolve().is_dir():
            raise FileNotFoundError(path.resolve())
    if stage6j.git_commit(args.nuplan_devkit_root.resolve()) != args.expected_nuplan_commit:
        raise ValueError("nuPlan devkit commit differs from the expected Stage 6K commit")
    if stage6j.git_commit(args.tuplan_garage_root.resolve()) != args.expected_tuplan_commit:
        raise ValueError("tuPlan Garage commit differs from the expected Stage 6K commit")
    return manifest, rows, locked_sha


def selected_rows(args: argparse.Namespace, rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    end = args.end_order or len(rows)
    selected = [row for row in rows if args.start_order <= int(row["collection_order"]) <= end]
    if args.max_jobs > 0:
        selected = selected[: args.max_jobs]
    if not selected:
        raise ValueError("Stage 6K execution range selected zero jobs")
    return selected


def initial_status(rows: Sequence[Dict[str, str]], output_dir: Path) -> List[Dict[str, Any]]:
    statuses: List[Dict[str, Any]] = []
    for row in rows:
        path = scenario_dir(output_dir, row)
        success = successful_attempt(path, row)
        attempts = attempt_dirs(path)
        audit = audit_stage7c_output(success / "stage7c_output", planners_for(row), row) if success else {}
        summary = stage6j.read_json(success / "attempt_summary.json") if success and (success / "attempt_summary.json").is_file() else {}
        statuses.append({
            **row, "status": "SUCCEEDED" if success else ("FAILED_REVIEW_REQUIRED" if attempts else "PENDING"),
            "failure_category": audit.get("failure_category", ""), "attempt": len(attempts), "return_code": "",
            "official_success_count": audit.get("official_success_count", 0), "trajectory_rows": audit.get("trajectory_rows", 0),
            "same_log_alignment_passed": audit.get("same_log_alignment_passed", False),
            "strict_token_alignment_passed": audit.get("strict_token_alignment_passed", False),
            "started_utc": summary.get("started_utc", ""), "ended_utc": summary.get("ended_utc", ""),
            "duration_seconds": summary.get("duration_seconds", ""),
            "stage7c_output_dir": str(success / "stage7c_output") if success else "",
        })
    return statuses


def write_state(output_dir: Path, statuses: Sequence[Mapping[str, Any]], started_utc: str) -> Dict[str, Any]:
    labels = sorted({str(row["dose_label"]) for row in statuses})
    counts = {status: sum(row.get("status") == status for row in statuses) for status in ["SUCCEEDED", "FAILED_REVIEW_REQUIRED", "PENDING"]}
    by_dose = {label: {status: sum(row["dose_label"] == label and row.get("status") == status for row in statuses) for status in counts} for label in labels}
    durations = [float(row["duration_seconds"]) for row in statuses if row.get("status") == "SUCCEEDED" and str(row.get("duration_seconds", ""))]
    average = sum(durations) / len(durations) if durations else 0.0
    state = {
        "schema_version": SCHEMA_VERSION, "updated_utc": stage6j.utc_now(), "started_utc": started_utc,
        "total_jobs": len(statuses), "planned_rollouts": len(statuses) * 2, "counts": counts, "counts_by_dose": by_dose,
        "completed_fraction": (counts["SUCCEEDED"] + counts["FAILED_REVIEW_REQUIRED"]) / len(statuses),
        "mean_success_duration_seconds": average,
        "estimated_remaining_seconds": average * counts["PENDING"] if average else None,
    }
    write_csv(output_dir / "batch_scenario_status.csv", statuses, STATUS_FIELDS)
    stage6j.atomic_json(output_dir / "batch_state.json", state)
    return state


def run_one(args: argparse.Namespace, row: Mapping[str, str], attempt: Path) -> Dict[str, Any]:
    stage6j.PLANNERS = planners_for(row)
    args.experiment_name = f"stage6k_longitudinal_{row['dose_label']}_v1"
    return stage6j.run_one(args, row, attempt)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest, rows, locked_sha = validate_inputs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(exist_ok=True)
    batch_manifest_path = output_dir / "batch_manifest.json"
    frozen_batch = {
        "schema_version": SCHEMA_VERSION, "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_sha256": stage6j.sha256_file(args.freeze_manifest.resolve()),
        "locked_jobs_csv": str(args.locked_jobs_csv.resolve()), "locked_jobs_sha256": locked_sha,
        "job_count": len(rows), "planned_rollout_count": len(rows) * 2,
        "planner_fingerprints": manifest["profile_audit"]["planner_parameter_fingerprints"],
        "nuplan_commit": args.expected_nuplan_commit, "tuplan_commit": args.expected_tuplan_commit,
        "stage7c_tool": str(args.stage7c_tool.resolve()), "stage7c_tool_sha256": stage6j.sha256_file(args.stage7c_tool.resolve()),
        "full_embedding_or_bdd_read": False,
    }
    if batch_manifest_path.is_file():
        if stage6j.read_json(batch_manifest_path) != frozen_batch:
            raise ValueError("Existing Stage 6K batch manifest differs from current frozen inputs")
    else:
        stage6j.atomic_json(batch_manifest_path, frozen_batch)
    statuses = initial_status(rows, output_dir)
    started_utc = stage6j.utc_now()
    state = write_state(output_dir, statuses, started_utc)
    candidates = selected_rows(args, rows)
    if not args.execute:
        return {"status": "DRY_RUN_PASS", "candidate_count": len(candidates), "batch_state": state}
    if args.confirm_locked_jobs_sha256 != locked_sha:
        raise ValueError("--confirm_locked_jobs_sha256 must exactly match the frozen Stage 6K job CSV SHA-256")
    if (output_dir / "batch_events.jsonl").is_file() and not args.resume:
        raise ValueError("Existing Stage 6K batch events require --resume")
    status_by_order = {int(row["collection_order"]): row for row in statuses}
    for position, row in enumerate(candidates, start=1):
        order = int(row["collection_order"])
        existing = status_by_order[order]
        if existing["status"] == "SUCCEEDED":
            print(f"[Stage6K] skip succeeded order={order}", flush=True)
            continue
        if existing["status"] == "FAILED_REVIEW_REQUIRED" and not args.retry_failed:
            print(f"[Stage6K] skip failed order={order}; review before --retry_failed", flush=True)
            continue
        path = scenario_dir(output_dir, row)
        path.mkdir(parents=True, exist_ok=True)
        attempt = path / f"attempt_{len(attempt_dirs(path)) + 1:03d}"
        attempt.mkdir()
        print(f"[Stage6K] start {position}/{len(candidates)} order={order} dose={row['dose_label']} token={row['scenario_token']}", flush=True)
        stage6j.append_event(output_dir / "batch_events.jsonl", {"event": "START", "utc": stage6j.utc_now(), "order": order, "dose_label": row["dose_label"], "attempt": attempt.name})
        try:
            result = run_one(args, row, attempt)
        except subprocess.TimeoutExpired:
            result = {**row, "status": "FAILED_REVIEW_REQUIRED", "failure_category": "COMMAND_TIMEOUT", "attempt": len(attempt_dirs(path)), "return_code": 124, "official_success_count": 0, "trajectory_rows": 0, "same_log_alignment_passed": False, "strict_token_alignment_passed": False, "started_utc": stage6j.utc_now(), "ended_utc": stage6j.utc_now(), "duration_seconds": "", "stage7c_output_dir": str((attempt / "stage7c_output").resolve())}
        stage6j.atomic_json(attempt / "attempt_summary.json", result)
        status_by_order[order] = result
        statuses = [status_by_order[int(source["collection_order"])] for source in rows]
        state = write_state(output_dir, statuses, started_utc)
        stage6j.append_event(output_dir / "batch_events.jsonl", {"event": "END", "utc": stage6j.utc_now(), "order": order, "dose_label": row["dose_label"], "attempt": attempt.name, "status": result["status"], "failure_category": result["failure_category"], "duration_seconds": result["duration_seconds"]})
        print(f"[Stage6K] end order={order} status={result['status']} duration={result['duration_seconds']}s counts={state['counts']} eta_s={state['estimated_remaining_seconds']}", flush=True)
    final_status = "COMPLETE" if state["counts"]["PENDING"] == 0 and state["counts"]["FAILED_REVIEW_REQUIRED"] == 0 else "STOPPED_WITH_PENDING_OR_FAILURE"
    result = {"status": final_status, "batch_state": state, "output_dir": str(output_dir)}
    stage6j.atomic_json(output_dir / "batch_result.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
