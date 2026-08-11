#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools import stage7_m6_4c_audit_locked_recovery as recovery
from tools import stage7_m6_4c_run_locked_recovery as recovery_runner


SCHEMA_VERSION = "stage6g_expanded_release_pool_runner_v1"
READY_STATUS = "FROZEN_BEFORE_STAGE6G_ROLLOUTS"
STATUS_FIELDS = batch.STATUS_FIELDS


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = batch.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} actual={actual}")
    return actual


def source_spec(args: argparse.Namespace, manifest: Mapping[str, Any]) -> Tuple[Path, str, str, int, str]:
    hashes = manifest["hashes"]
    if args.source == "primary":
        return (
            args.primary_csv,
            str(hashes["primary_csv_sha256"]),
            str(manifest["primary_manifest_sha256"]),
            int(manifest["planned_primary_additions"]),
            "stage6g_primary",
        )
    return (
        args.reserve_csv,
        str(hashes["reserve_csv_sha256"]),
        str(manifest["reserve_manifest_sha256"]),
        int(manifest["planned_reserve"]),
        "stage6g_technical_reserve",
    )


def validate_inputs(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Any]]:
    manifest = batch.read_json(args.freeze_manifest)
    if manifest.get("schema_version") != "stage6g_expanded_release_pool_freeze_v1":
        raise ValueError("Stage6G freeze manifest schema mismatch")
    if manifest.get("status") != READY_STATUS or manifest.get("ready_to_launch_rollouts") is not True:
        raise ValueError("Stage6G freeze manifest is not ready")
    if any(bool(value) for value in manifest.get("forbidden_inputs_read", {}).values()):
        raise ValueError("Stage6G freeze reports forbidden post-treatment input use")
    manifest_hash = batch.sha256_file(args.freeze_manifest)
    if args.execute and args.confirm_freeze_manifest_sha256 != manifest_hash:
        raise ValueError("--confirm_freeze_manifest_sha256 must equal the current manifest SHA-256")
    hashes = manifest["hashes"]
    validate_sha(args.selection_tool, str(hashes["freeze_tool_sha256"]), "freeze tool")
    validate_sha(Path(__file__), str(hashes["runner_tool_sha256"]), "runner tool")
    validate_sha(args.stage7c_tool, str(hashes["stage7c_tool_sha256"]), "Stage7C tool")
    validate_sha(args.batch_tool, str(hashes["historical_batch_tool_sha256"]), "historical batch tool")
    source_path, source_csv_hash, canonical_hash, expected_count, expected_role = source_spec(args, manifest)
    validate_sha(source_path, source_csv_hash, f"Stage6G {args.source} CSV")
    rows = batch.read_csv(source_path)
    if len(rows) != expected_count:
        raise ValueError(f"Stage6G {args.source} row count mismatch")
    if batch.canonical_rows_hash(rows) != canonical_hash:
        raise ValueError(f"Stage6G {args.source} canonical hash mismatch")
    if args.execute and args.confirm_source_manifest_sha256 != canonical_hash:
        raise ValueError("--confirm_source_manifest_sha256 must equal the frozen source canonical hash")
    if [int(row["collection_order"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Stage6G collection_order is not contiguous")
    if any(row["selection_role"] != expected_role for row in rows):
        raise ValueError("Stage6G selection_role mismatch")
    primary_status_hash = ""
    if args.source == "reserve":
        if args.primary_status_csv is None:
            raise ValueError("--primary_status_csv is required for reserve execution or dry-run")
        if not args.primary_status_csv.is_file():
            raise FileNotFoundError(args.primary_status_csv)
        with args.primary_status_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            primary_status = list(csv.DictReader(handle))
        failures = Counter(
            row["task"]
            for row in primary_status
            if row.get("status") == "FAILED_REVIEW_REQUIRED"
            and row.get("failure_category") in batch.RESERVE_ELIGIBLE_FAILURES
        )
        if not failures:
            raise ValueError("Stage6G primary status has no reserve-eligible technical failure")
        chosen: List[Dict[str, str]] = []
        chosen_counts: Counter = Counter()
        for row in rows:
            task = row["task"]
            if chosen_counts[task] < failures[task]:
                chosen.append(row)
                chosen_counts[task] += 1
        rows = chosen
        if sum(chosen_counts.values()) < sum(failures.values()):
            raise ValueError("frozen Stage6G reserve cannot cover all primary technical failures")
        primary_status_hash = batch.sha256_file(args.primary_status_csv)

    runtime = manifest["runtime"]
    path_args = {
        "nuplan_db_root": args.nuplan_db_root,
        "nuplan_map_root": args.nuplan_map_root,
        "nuplan_data_root": args.nuplan_data_root,
        "nuplan_exp_root": args.nuplan_exp_root,
        "python_executable": args.python_executable,
    }
    for key, actual in path_args.items():
        if actual.resolve() != Path(runtime[key]).resolve():
            raise ValueError(f"runtime path differs from freeze manifest: {key}")
    if args.command_timeout_s != int(runtime["command_timeout_s"]):
        raise ValueError("command_timeout_s differs from freeze manifest")
    if args.nuplan_devkit_root.resolve() != Path(manifest["nuplan_devkit_root"]).resolve():
        raise ValueError("nuplan_devkit_root differs from freeze manifest")
    if args.tuplan_garage_root.resolve() != Path(manifest["tuplan_garage_root"]).resolve():
        raise ValueError("tuplan_garage_root differs from freeze manifest")
    if batch.current_planner_fingerprints(batch.EXPECTED_PLANNERS) != manifest["planner_parameter_fingerprints"]:
        raise ValueError("planner fingerprints differ from freeze manifest")
    if batch.resolve_git_commit(args.nuplan_devkit_root) != manifest["nuplan_devkit_commit"]:
        raise ValueError("nuPlan commit differs from freeze manifest")
    if batch.resolve_git_commit(args.tuplan_garage_root) != manifest["tuplan_garage_commit"]:
        raise ValueError("tuPlan Garage commit differs from freeze manifest")
    for row in rows:
        technical = recovery.inspect_token_scene_position(
            args.nuplan_db_root / row["db_file"], row["scenario_token"]
        )
        if not technical["official_scene_position_valid"]:
            raise ValueError(f"frozen token is no longer technically runnable: {row['scenario_token']}")
    audit = {
        "freeze_manifest_sha256": manifest_hash,
        "source_csv_sha256": source_csv_hash,
        "source_manifest_sha256": canonical_hash,
        "freeze_tool_sha256": hashes["freeze_tool_sha256"],
        "runner_tool_sha256": hashes["runner_tool_sha256"],
        "stage7c_tool_sha256": hashes["stage7c_tool_sha256"],
        "planner_parameter_fingerprints": manifest["planner_parameter_fingerprints"],
        "nuplan_devkit_commit": manifest["nuplan_devkit_commit"],
        "tuplan_garage_commit": manifest["tuplan_garage_commit"],
        "primary_status_csv_sha256": primary_status_hash,
    }
    return manifest, rows, audit


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    next_path = path.with_name(path.name + ".next")
    batch.write_csv(next_path, rows, fields)
    os.replace(next_path, path)


def event(path: Path, kind: str, **payload: Any) -> None:
    batch.append_jsonl(path, {"event": kind, "utc": batch.utc_now(), **payload})


def status_rows(source_rows: Sequence[Dict[str, str]], output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in source_rows:
        scenario_dir = output_dir / "rollouts" / batch.scenario_slug(source)
        attempts = sorted(scenario_dir.glob("attempt_*")) if scenario_dir.is_dir() else []
        successes = batch.successful_attempts(scenario_dir, batch.EXPECTED_PLANNERS, source)
        completed: List[Tuple[Path, Dict[str, Any]]] = []
        for attempt in attempts:
            summary_path = attempt / "attempt_summary.json"
            if summary_path.is_file():
                completed.append((attempt, batch.read_json(summary_path)))
        if successes:
            attempt_path, audit = successes[-1]
            summary = batch.read_json(attempt_path / "attempt_summary.json")
            payload = {**summary, **audit}
            status = "SUCCEEDED"
            output = attempt_path / "stage7c_output"
        elif completed:
            attempt_path, payload = completed[-1]
            status = "FAILED_REVIEW_REQUIRED"
            output = attempt_path / "stage7c_output"
        else:
            payload = {}
            status = "PENDING"
            output = None
        rows.append(
            {
                **source,
                "status": status,
                "failure_category": payload.get("failure_category", ""),
                "attempt": len(attempts),
                "return_code": payload.get("return_code", ""),
                "official_success_count": payload.get("official_success_count", 0),
                "trajectory_rows": payload.get("trajectory_rows", 0),
                "same_log_alignment_passed": payload.get("same_log_alignment_passed", False),
                "strict_token_alignment_passed": payload.get("strict_token_alignment_passed", False),
                "started_utc": payload.get("started_utc", ""),
                "ended_utc": payload.get("ended_utc", ""),
                "duration_seconds": payload.get("duration_seconds", ""),
                "stage7c_output_dir": str(output.resolve()) if output is not None else "",
            }
        )
    return rows


def write_state(output_dir: Path, source: str, rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> None:
    counts = Counter(str(row["status"]) for row in rows)
    durations = [float(row["duration_seconds"]) for row in rows if row.get("duration_seconds") not in {"", None}]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": batch.utc_now(),
        "source": source,
        "total": len(rows),
        "succeeded": counts["SUCCEEDED"],
        "failed": counts["FAILED_REVIEW_REQUIRED"],
        "pending": counts["PENDING"],
        "running": counts["RUNNING"],
        "completion_fraction": (counts["SUCCEEDED"] + counts["FAILED_REVIEW_REQUIRED"]) / len(rows) if rows else 1.0,
        "mean_completed_duration_seconds": sum(durations) / len(durations) if durations else None,
        "input_audit": dict(audit),
    }
    batch.atomic_write_json(output_dir / "batch_state.json", payload)


def build_stage7c_command(args: argparse.Namespace, row: Mapping[str, str], attempt_dir: Path) -> List[str]:
    context_dir = attempt_dir / "context"
    output_dir = attempt_dir / "stage7c_output"
    context_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    actual_token = (
        recovery_runner.escaped_hydra_string(row["scenario_token"])
        if recovery.hydra_requires_quoted_token(row["scenario_token"])
        else ""
    )
    context_fields = batch.PRIMARY_FIELDS + ["actual_nuplan_token"]
    batch.write_csv(
        context_dir / "merged_metadata.csv",
        [{**row, "actual_nuplan_token": actual_token}],
        context_fields,
    )
    command_template = batch.build_command_template(
        args, args.nuplan_db_root / row["db_file"]
    ).replace("stage7_m6_4b_locked_primary_mac_v1", "stage6g_expanded_release_pool_mac_v1")
    return [
        str(args.python_executable.resolve()),
        str(args.stage7c_tool.resolve()),
        "--context_dir", str(context_dir.resolve()),
        "--nuplan_db_root", str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root", str(args.nuplan_map_root.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--planners", *batch.EXPECTED_PLANNERS,
        "--max_scenarios", "1",
        "--min_timesteps", "2",
        "--require_same_scenario_alignment",
        "--require_strict_nuplan_token_alignment",
        "--allow_external_planner_name",
        "--hydra_searchpath",
        "[pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
        "--command_timeout_s", str(args.command_timeout_s),
        "--nuplan_simulation_command_template", command_template,
    ]


def execute_one(args: argparse.Namespace, row: Mapping[str, str], attempt_dir: Path) -> Dict[str, Any]:
    started = batch.utc_now()
    start = time.monotonic()
    command = build_stage7c_command(args, row, attempt_dir)
    log_path = attempt_dir / "stage7c_driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv: " + json.dumps(command, ensure_ascii=False) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=batch.stage7c_environment(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = int(process.wait())
    duration = time.monotonic() - start
    output_dir = attempt_dir / "stage7c_output"
    audit = batch.audit_stage7c_output(output_dir, batch.EXPECTED_PLANNERS, row)
    if return_code != 0 and audit["failure_category"] == "INVALID_STAGE7C_OUTPUT":
        audit["failure_category"] = batch.classify_process_failure(
            return_code, log_path
        )
    result = {
        **row,
        "status": "SUCCEEDED" if audit["pass"] else "FAILED_REVIEW_REQUIRED",
        "failure_category": audit["failure_category"],
        "return_code": return_code,
        "started_utc": started,
        "ended_utc": batch.utc_now(),
        "duration_seconds": duration,
        "stage7c_output_dir": str(output_dir.resolve()),
        **audit,
    }
    batch.atomic_write_json(attempt_dir / "attempt_summary.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    manifest, source_rows, audit = validate_inputs(args)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "source": args.source, "selected": len(source_rows), "input_audit": audit}, indent=2))
        return 0
    args.nuplan_exp_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "stage6g_runner.lock"
    events_path = args.output_dir / "batch_events.jsonl"
    runner_manifest_path = args.output_dir / "runner_manifest.json"
    expected_runner_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": batch.utc_now(),
        "source": args.source,
        "selected_scenarios": len(source_rows),
        "input_audit": audit,
        "planners": batch.EXPECTED_PLANNERS,
        "forbidden_inputs_read": {
            "embedding": False,
            "bdd": False,
            "effect_size": False,
            "trajectory_metrics_for_selection_or_stopping": False,
            "planner_outcome_for_selection_or_stopping": False,
        },
    }
    if runner_manifest_path.is_file():
        current = batch.read_json(runner_manifest_path)
        if current.get("source") != args.source or current.get("input_audit") != audit:
            raise ValueError("existing Stage6G output belongs to a different frozen invocation")
    else:
        batch.atomic_write_json(runner_manifest_path, expected_runner_manifest)
        event(events_path, "BATCH_CREATED", source=args.source, total=len(source_rows))

    actions = 0
    with batch.BatchLock(lock_path):
        rows = status_rows(source_rows, args.output_dir)
        atomic_write_csv(args.output_dir / "batch_scenario_status.csv", rows, STATUS_FIELDS)
        write_state(args.output_dir, args.source, rows, audit)
        for source in source_rows:
            current = next(row for row in rows if row["scenario_token"] == source["scenario_token"])
            if current["status"] in {"SUCCEEDED", "FAILED_REVIEW_REQUIRED"}:
                continue
            if args.max_actions > 0 and actions >= args.max_actions:
                event(events_path, "INVOCATION_LIMIT_REACHED", max_actions=args.max_actions)
                break
            scenario_dir = args.output_dir / "rollouts" / batch.scenario_slug(source)
            prior_attempts = sorted(scenario_dir.glob("attempt_*")) if scenario_dir.is_dir() else []
            attempt_number = len(prior_attempts) + 1
            attempt_dir = scenario_dir / f"attempt_{attempt_number:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            current.update(status="RUNNING", attempt=attempt_number, started_utc=batch.utc_now())
            atomic_write_csv(args.output_dir / "batch_scenario_status.csv", rows, STATUS_FIELDS)
            write_state(args.output_dir, args.source, rows, audit)
            event(events_path, "SCENARIO_STARTED", collection_order=int(source["collection_order"]), task=source["task"], token=source["scenario_token"], attempt=attempt_number)
            print(f"[Stage6G] START {source['collection_order']}/{len(source_rows)} task={source['task']} token={source['scenario_token']}", flush=True)
            result = execute_one(args, source, attempt_dir)
            result["attempt"] = attempt_number
            current.update(result)
            atomic_write_csv(args.output_dir / "batch_scenario_status.csv", rows, STATUS_FIELDS)
            write_state(args.output_dir, args.source, rows, audit)
            event(events_path, "SCENARIO_FINISHED", collection_order=int(source["collection_order"]), task=source["task"], token=source["scenario_token"], status=result["status"], failure_category=result["failure_category"], duration_seconds=result["duration_seconds"])
            print(f"[Stage6G] DONE order={source['collection_order']} status={result['status']} duration={result['duration_seconds']:.1f}s", flush=True)
            actions += 1
        rows = status_rows(source_rows, args.output_dir)
        atomic_write_csv(args.output_dir / "batch_scenario_status.csv", rows, STATUS_FIELDS)
        write_state(args.output_dir, args.source, rows, audit)
        counts = Counter(row["status"] for row in rows)
        event(events_path, "INVOCATION_FINISHED", succeeded=counts["SUCCEEDED"], failed=counts["FAILED_REVIEW_REQUIRED"], pending=counts["PENDING"])
    print(json.dumps({"source": args.source, "succeeded": counts["SUCCEEDED"], "failed": counts["FAILED_REVIEW_REQUIRED"], "pending": counts["PENDING"]}, indent=2))
    return 0 if counts["FAILED_REVIEW_REQUIRED"] == 0 else 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the frozen Stage6G rollout pool.")
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--reserve_csv", type=Path, required=True)
    parser.add_argument("--selection_tool", type=Path, default=Path("tools/stage6g_freeze_expanded_release_pool.py"))
    parser.add_argument("--stage7c_tool", type=Path, default=Path("tools/stage7c1_run_nuplan_simulation.py"))
    parser.add_argument("--batch_tool", type=Path, default=Path("tools/stage7_m6_4b_run_locked_rollouts.py"))
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--nuplan_data_root", type=Path, required=True)
    parser.add_argument("--nuplan_exp_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--tuplan_garage_root", type=Path, required=True)
    parser.add_argument("--python_executable", type=Path, required=True)
    parser.add_argument("--command_timeout_s", type=int, default=3600)
    parser.add_argument("--source", choices=("primary", "reserve"), default="primary")
    parser.add_argument("--primary_status_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max_actions", type=int, default=0)
    parser.add_argument("--confirm_freeze_manifest_sha256", default="")
    parser.add_argument("--confirm_source_manifest_sha256", default="")
    args = parser.parse_args(argv)
    if args.command_timeout_s < 1 or args.max_actions < 0:
        parser.error("command_timeout_s must be >=1 and max_actions must be >=0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
