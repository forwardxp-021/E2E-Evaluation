#!/usr/bin/env python3
"""Run the frozen Stage 6J pure-longitudinal rollouts with resumable auditing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import (  # noqa: E402
    audit_stage7c_output,
    stage7c_environment,
)
from tools.stage7_m6_4c_audit_locked_recovery import hydra_requires_quoted_token  # noqa: E402
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES  # noqa: E402


SCHEMA_VERSION = "stage6j_pure_longitudinal_batch_v1"
FREEZE_STATUS = "FROZEN_BEFORE_PURE_LONGITUDINAL_ROLLOUTS"
PLANNERS = [
    "pdm_closed_assertive_longitudinal_v1",
    "pdm_closed_conservative_longitudinal_v1",
]
LOCKED_FIELDS = [
    "collection_order", "source_global_scenario_index", "task", "source_task",
    "scenario_type", "log_name", "scenario_token", "scene_token", "db_file", "selection_role",
]
CONTEXT_FIELDS = LOCKED_FIELDS + ["actual_nuplan_token"]
STATUS_FIELDS = LOCKED_FIELDS + [
    "status", "failure_category", "attempt", "return_code", "official_success_count",
    "trajectory_rows", "same_log_alignment_passed", "strict_token_alignment_passed",
    "started_utc", "ended_utc", "duration_seconds", "stage7c_output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Stage 6J pure-longitudinal rollouts.")
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--locked_scenarios_csv", type=Path, required=True)
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
    parser.add_argument("--end_order", type=int, default=0, help="0 means the final locked order.")
    parser.add_argument("--max_scenarios", type=int, default=0, help="0 means no additional cap.")
    parser.add_argument("--command_timeout_s", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm_locked_scenarios_sha256", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(LOCKED_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing Stage 6J locked columns: {missing}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: Any) -> None:
    next_path = path.with_name(path.name + ".next")
    next_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(next_path, path)


def append_event(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit(path: Path) -> str:
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"Cannot resolve git commit for {path}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def planner_fingerprints() -> Dict[str, str]:
    return {planner: canonical_hash(PLANNER_PROFILES[planner]["parameters"]) for planner in PLANNERS}


def validate_inputs(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, str]], str]:
    manifest_path = args.freeze_manifest.resolve()
    locked_path = args.locked_scenarios_csv.resolve()
    manifest = read_json(manifest_path)
    rows = read_csv(locked_path)
    if manifest.get("status") != FREEZE_STATUS:
        raise ValueError(f"Stage 6J freeze status is not ready: {manifest.get('status')!r}")
    treatment = manifest.get("treatment_audit", {})
    if treatment.get("pure_longitudinal_treatment") is not True:
        raise ValueError("Stage 6J manifest is not a pure-longitudinal treatment")
    if [treatment.get("planner_a"), treatment.get("planner_b")] != PLANNERS:
        raise ValueError("Stage 6J planner order differs from the frozen batch order")
    locked_sha = sha256_file(locked_path)
    if manifest.get("outputs", {}).get("locked_scenarios", {}).get("sha256") != locked_sha:
        raise ValueError("Locked scenario CSV SHA-256 differs from the freeze manifest")
    if len(rows) != int(manifest.get("selection_audit", {}).get("selected_scenario_count", -1)):
        raise ValueError("Locked scenario row count differs from the freeze manifest")
    orders = [int(row["collection_order"]) for row in rows]
    if orders != list(range(1, len(rows) + 1)):
        raise ValueError("Stage 6J collection_order must be contiguous and start at 1")
    tokens = [row["scenario_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Stage 6J locked scenario tokens are not unique")
    for row in rows:
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError(f"DB/log identity mismatch for order {row['collection_order']}")
        if not (args.nuplan_db_root.resolve() / row["db_file"]).is_file():
            raise FileNotFoundError(args.nuplan_db_root.resolve() / row["db_file"])
    current = planner_fingerprints()
    if current[PLANNERS[0]] != treatment.get("planner_a_parameter_sha256"):
        raise ValueError("Planner A fingerprint changed after Stage 6J freeze")
    if current[PLANNERS[1]] != treatment.get("planner_b_parameter_sha256"):
        raise ValueError("Planner B fingerprint changed after Stage 6J freeze")
    for path in [args.python_executable, args.stage7c_tool]:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    for path in [args.nuplan_map_root, args.nuplan_data_root, args.nuplan_exp_root, args.nuplan_devkit_root, args.tuplan_garage_root]:
        if not path.resolve().is_dir():
            raise FileNotFoundError(path.resolve())
    if git_commit(args.nuplan_devkit_root.resolve()) != args.expected_nuplan_commit:
        raise ValueError("nuPlan devkit commit differs from the explicit expected commit")
    if git_commit(args.tuplan_garage_root.resolve()) != args.expected_tuplan_commit:
        raise ValueError("tuPlan Garage commit differs from the explicit expected commit")
    return manifest, rows, locked_sha


def selected_rows(args: argparse.Namespace, rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    end = args.end_order or len(rows)
    selected = [row for row in rows if args.start_order <= int(row["collection_order"]) <= end]
    if args.max_scenarios > 0:
        selected = selected[: args.max_scenarios]
    if not selected:
        raise ValueError("Stage 6J execution range selected zero scenarios")
    return selected


def scenario_dir(output_dir: Path, row: Mapping[str, str]) -> Path:
    return output_dir / "rollouts" / f"order_{int(row['collection_order']):04d}_{row['scenario_token']}"


def attempt_dirs(path: Path) -> List[Path]:
    return sorted(path.glob("attempt_*")) if path.is_dir() else []


def successful_attempt(path: Path, row: Mapping[str, str]) -> Optional[Path]:
    for attempt in reversed(attempt_dirs(path)):
        if audit_stage7c_output(attempt / "stage7c_output", PLANNERS, row).get("pass"):
            return attempt
    return None


def initial_status(rows: Sequence[Dict[str, str]], output_dir: Path) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in rows:
        path = scenario_dir(output_dir, row)
        success = successful_attempt(path, row)
        attempts = attempt_dirs(path)
        audit = audit_stage7c_output(success / "stage7c_output", PLANNERS, row) if success else {}
        result.append({
            **row,
            "status": "SUCCEEDED" if success else ("FAILED_REVIEW_REQUIRED" if attempts else "PENDING"),
            "failure_category": audit.get("failure_category", ""),
            "attempt": len(attempts),
            "return_code": "",
            "official_success_count": audit.get("official_success_count", 0),
            "trajectory_rows": audit.get("trajectory_rows", 0),
            "same_log_alignment_passed": audit.get("same_log_alignment_passed", False),
            "strict_token_alignment_passed": audit.get("strict_token_alignment_passed", False),
            "started_utc": "", "ended_utc": "", "duration_seconds": "",
            "stage7c_output_dir": str(success / "stage7c_output") if success else "",
        })
    return result


def command_template(args: argparse.Namespace) -> str:
    db_root = args.nuplan_db_root.resolve()
    run_sim = (args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()
    return " ".join([
        str(args.python_executable.resolve()), str(run_sim), "+simulation=closed_loop_nonreactive_agents",
        "{planner_hydra_overrides}", "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{db_root}/{{target_log_name}}.db]",
        "scenario_filter=all_scenarios", "{scenario_hydra_overrides}",
        "worker=single_machine_thread_pool", "worker.max_workers=1", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "gpu=false",
        f"experiment_name={getattr(args, 'experiment_name', 'stage6j_pure_longitudinal_batch_v1')}",
        "job_name=closed_loop_nonreactive_agents_stage7c_{planner_name_safe}", "output_dir={output_dir}",
    ])


def hydra_actual_token(token: str) -> str:
    return f'\\"{token}\\"' if hydra_requires_quoted_token(token) else ""


def run_one(args: argparse.Namespace, row: Mapping[str, str], attempt: Path) -> Dict[str, Any]:
    context = attempt / "context"
    output = attempt / "stage7c_output"
    context.mkdir(parents=True)
    context_row = {**row, "actual_nuplan_token": hydra_actual_token(row["scenario_token"])}
    write_csv(context / "merged_metadata.csv", [context_row], CONTEXT_FIELDS)
    command = [
        str(args.python_executable.resolve()), str(args.stage7c_tool.resolve()),
        "--context_dir", str(context.resolve()),
        "--nuplan_db_root", str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root", str(args.nuplan_map_root.resolve()),
        "--output_dir", str(output.resolve()),
        "--planners", *PLANNERS, "--max_scenarios", "1", "--min_timesteps", "2",
        "--require_same_scenario_alignment", "--require_strict_nuplan_token_alignment",
        "--allow_external_planner_name", "--hydra_searchpath",
        "[pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
        "--command_timeout_s", str(args.command_timeout_s),
        "--nuplan_simulation_command_template", command_template(args),
    ]
    started = utc_now()
    start = time.monotonic()
    log_path = attempt / "stage7c_driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv: " + json.dumps(command, ensure_ascii=False) + "\n\n")
        log.flush()
        proc = subprocess.run(
            command, cwd=str(Path(__file__).resolve().parents[1]), env=stage7c_environment(args),
            stdout=log, stderr=subprocess.STDOUT, text=True, timeout=args.command_timeout_s * len(PLANNERS) + 120,
        )
    audit = audit_stage7c_output(output, PLANNERS, row)
    duration = time.monotonic() - start
    failure = "" if audit.get("pass") else (
        audit.get("failure_category") or ("COMMAND_FAILED" if proc.returncode else "INVALID_STAGE7C_OUTPUT")
    )
    return {
        **row,
        "status": "SUCCEEDED" if audit.get("pass") else "FAILED_REVIEW_REQUIRED",
        "failure_category": failure,
        "attempt": int(attempt.name.split("_")[-1]),
        "return_code": proc.returncode,
        "official_success_count": audit.get("official_success_count", 0),
        "trajectory_rows": audit.get("trajectory_rows", 0),
        "same_log_alignment_passed": audit.get("same_log_alignment_passed", False),
        "strict_token_alignment_passed": audit.get("strict_token_alignment_passed", False),
        "started_utc": started, "ended_utc": utc_now(), "duration_seconds": round(duration, 3),
        "stage7c_output_dir": str(output.resolve()),
    }


def write_state(output_dir: Path, statuses: Sequence[Mapping[str, Any]], started_utc: str) -> Dict[str, Any]:
    counts = {status: sum(row.get("status") == status for row in statuses) for status in ["SUCCEEDED", "FAILED_REVIEW_REQUIRED", "PENDING"]}
    durations = [float(row["duration_seconds"]) for row in statuses if row.get("status") == "SUCCEEDED" and str(row.get("duration_seconds", ""))]
    average = sum(durations) / len(durations) if durations else 0.0
    state = {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": utc_now(), "started_utc": started_utc,
        "total_scenarios": len(statuses), "planned_rollouts": len(statuses) * len(PLANNERS),
        "counts": counts, "completed_fraction": (counts["SUCCEEDED"] + counts["FAILED_REVIEW_REQUIRED"]) / len(statuses),
        "mean_success_duration_seconds": average,
        "estimated_remaining_seconds": average * counts["PENDING"] if average else None,
    }
    write_csv(output_dir / "batch_scenario_status.csv", statuses, STATUS_FIELDS)
    atomic_json(output_dir / "batch_state.json", state)
    return state


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest, rows, locked_sha = validate_inputs(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(exist_ok=True)
    batch_manifest_path = output_dir / "batch_manifest.json"
    frozen_batch = {
        "schema_version": SCHEMA_VERSION,
        "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "locked_scenarios_csv": str(args.locked_scenarios_csv.resolve()),
        "locked_scenarios_sha256": locked_sha,
        "planners": PLANNERS,
        "planner_fingerprints": planner_fingerprints(),
        "scenario_count": len(rows), "planned_rollout_count": len(rows) * len(PLANNERS),
        "nuplan_commit": args.expected_nuplan_commit, "tuplan_commit": args.expected_tuplan_commit,
        "stage7c_tool": str(args.stage7c_tool.resolve()), "stage7c_tool_sha256": sha256_file(args.stage7c_tool.resolve()),
        "full_embedding_or_bdd_read": False,
    }
    if batch_manifest_path.is_file():
        if read_json(batch_manifest_path) != frozen_batch:
            raise ValueError("Existing Stage 6J batch manifest differs from current frozen inputs")
    else:
        atomic_json(batch_manifest_path, frozen_batch)

    statuses = initial_status(rows, output_dir)
    started_utc = utc_now()
    state = write_state(output_dir, statuses, started_utc)
    candidates = selected_rows(args, rows)
    if not args.execute:
        return {"status": "DRY_RUN_PASS", "candidate_count": len(candidates), "batch_state": state}
    if args.confirm_locked_scenarios_sha256 != locked_sha:
        raise ValueError("--confirm_locked_scenarios_sha256 must exactly match the frozen locked CSV SHA-256")
    if (output_dir / "batch_events.jsonl").is_file() and not args.resume:
        raise ValueError("Existing batch events require --resume")

    status_by_order = {int(row["collection_order"]): row for row in statuses}
    for position, row in enumerate(candidates, start=1):
        order = int(row["collection_order"])
        existing = status_by_order[order]
        if existing["status"] == "SUCCEEDED":
            print(f"[Stage6J] skip succeeded order={order}", flush=True)
            continue
        if existing["status"] == "FAILED_REVIEW_REQUIRED" and not args.retry_failed:
            print(f"[Stage6J] skip failed order={order}; pass --retry_failed after review", flush=True)
            continue
        path = scenario_dir(output_dir, row)
        path.mkdir(parents=True, exist_ok=True)
        attempt = path / f"attempt_{len(attempt_dirs(path)) + 1:03d}"
        attempt.mkdir()
        print(f"[Stage6J] start {position}/{len(candidates)} order={order} task={row['task']} token={row['scenario_token']}", flush=True)
        append_event(output_dir / "batch_events.jsonl", {"event": "START", "utc": utc_now(), "order": order, "attempt": attempt.name})
        try:
            result = run_one(args, row, attempt)
        except subprocess.TimeoutExpired:
            result = {**row, "status": "FAILED_REVIEW_REQUIRED", "failure_category": "COMMAND_TIMEOUT", "attempt": len(attempt_dirs(path)), "return_code": 124, "official_success_count": 0, "trajectory_rows": 0, "same_log_alignment_passed": False, "strict_token_alignment_passed": False, "started_utc": utc_now(), "ended_utc": utc_now(), "duration_seconds": "", "stage7c_output_dir": str((attempt / "stage7c_output").resolve())}
        atomic_json(attempt / "attempt_summary.json", result)
        status_by_order[order] = result
        statuses = [status_by_order[int(source["collection_order"])] for source in rows]
        state = write_state(output_dir, statuses, started_utc)
        append_event(output_dir / "batch_events.jsonl", {"event": "END", "utc": utc_now(), "order": order, "attempt": attempt.name, "status": result["status"], "failure_category": result["failure_category"], "duration_seconds": result["duration_seconds"]})
        print(f"[Stage6J] end order={order} status={result['status']} duration={result['duration_seconds']}s counts={state['counts']} eta_s={state['estimated_remaining_seconds']}", flush=True)
    final_status = "COMPLETE" if state["counts"]["PENDING"] == 0 and state["counts"]["FAILED_REVIEW_REQUIRED"] == 0 else "STOPPED_WITH_PENDING_OR_FAILURE"
    result = {"status": final_status, "batch_state": state, "output_dir": str(output_dir)}
    atomic_json(output_dir / "batch_result.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
