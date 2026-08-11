#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools import stage7_m6_4c_audit_locked_recovery as recovery


SCHEMA_VERSION = "stage7_m6_4c_locked_recovery_runner_v1"
QUOTED_RETRY_ACTION = "RETRY_PRIMARY_QUOTED_TOKEN"
RESERVE_ACTION = "RUN_FROZEN_RESERVE"
SUPPORTED_ACTIONS = (QUOTED_RETRY_ACTION, RESERVE_ACTION)
CONTEXT_FIELDS = batch.PRIMARY_FIELDS + ["actual_nuplan_token"]
RESULT_FIELDS = [
    "plan_order",
    "task",
    "action",
    "collection_order",
    "scenario_token",
    "status",
    "failure_category",
    "return_code",
    "duration_seconds",
    "stage7c_output_dir",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = recovery.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} actual={actual}")
    return actual


def validate_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    summary = recovery.read_json(args.audit_summary)
    if summary.get("schema_version") != recovery.SCHEMA_VERSION:
        raise ValueError("audit summary schema is not the authoritative M6.4C audit schema")
    if summary.get("execution_enabled") is not False:
        raise ValueError("audit summary must be an audit-only artifact")
    forbidden = summary.get("forbidden_inputs_read", {})
    if any(bool(value) for value in forbidden.values()):
        raise ValueError("audit summary reports a forbidden post-treatment input read")

    expected = summary.get("input_audit", {})
    paths = {
        "locked_manifest_sha256": args.locked_manifest,
        "primary_csv_sha256": args.primary_csv,
        "reserve_csv_sha256": args.reserve_csv,
        "batch_status_csv_sha256": args.batch_status_csv,
        "batch_manifest_sha256": args.batch_manifest,
        "stage7c_tool_sha256": args.stage7c_tool,
        "batch_tool_sha256": args.batch_tool,
    }
    actual_hashes = {
        key: validate_sha(path, str(expected.get(key, "")), key)
        for key, path in paths.items()
    }
    plan_hash = recovery.sha256_file(args.recovery_plan)
    if args.execute and args.confirm_recovery_plan_sha256 != plan_hash:
        raise ValueError(
            "--confirm_recovery_plan_sha256 must equal the current recovery_plan.csv SHA-256"
        )

    batch_manifest = recovery.read_json(args.batch_manifest)
    runtime_fields = {
        "nuplan_db_root": args.nuplan_db_root,
        "nuplan_map_root": args.nuplan_map_root,
        "nuplan_data_root": args.nuplan_data_root,
        "nuplan_exp_root": args.nuplan_exp_root,
        "python_executable": args.python_executable,
    }
    for key, actual_path in runtime_fields.items():
        frozen_path = Path(str(batch_manifest.get(key, ""))).resolve()
        if actual_path.resolve() != frozen_path:
            raise ValueError(f"{key} differs from the frozen M6.4B runtime: {actual_path}")
    if int(batch_manifest.get("command_timeout_s", 0)) != args.command_timeout_s:
        raise ValueError("command_timeout_s differs from the frozen M6.4B runtime")
    if batch_manifest.get("planners") != batch.EXPECTED_PLANNERS:
        raise ValueError("planner list differs from the frozen M6.4B runtime")
    frozen_runtime = batch_manifest.get("frozen_input_audit", {})
    actual_nuplan_commit = batch.resolve_git_commit(args.nuplan_devkit_root)
    actual_tuplan_commit = batch.resolve_git_commit(args.tuplan_garage_root)
    if actual_nuplan_commit != frozen_runtime.get("nuplan_devkit_commit"):
        raise ValueError("nuPlan commit differs from the frozen M6.4B runtime")
    if actual_tuplan_commit != frozen_runtime.get("tuplan_garage_commit"):
        raise ValueError("tuPlan Garage commit differs from the frozen M6.4B runtime")
    planner_fingerprints = batch.current_planner_fingerprints(batch.EXPECTED_PLANNERS)
    if planner_fingerprints != frozen_runtime.get("planner_parameter_fingerprints"):
        raise ValueError("planner fingerprints differ from the frozen M6.4B runtime")

    return {
        "schema_version": SCHEMA_VERSION,
        "audit_summary": str(args.audit_summary.resolve()),
        "recovery_plan": str(args.recovery_plan.resolve()),
        "recovery_plan_sha256": plan_hash,
        "input_hashes": actual_hashes,
        "frozen_batch_manifest": str(args.batch_manifest.resolve()),
        "planners": batch.EXPECTED_PLANNERS,
        "planner_parameter_fingerprints": planner_fingerprints,
        "nuplan_devkit_commit": actual_nuplan_commit,
        "tuplan_garage_commit": actual_tuplan_commit,
        "execution_action": args.action,
    }


def selected_actions(args: argparse.Namespace) -> List[Dict[str, str]]:
    rows = [
        row
        for row in read_csv(args.recovery_plan)
        if row.get("action") == args.action
    ]
    if not rows:
        raise ValueError(f"recovery plan has no {args.action} actions")
    if any(row.get("approval_status") != "PROPOSED_NOT_EXECUTED" for row in rows):
        raise ValueError("quoted-token retry action has an unexpected approval status")
    rows.sort(key=lambda row: int(row["plan_order"]))
    if args.max_actions:
        rows = rows[: args.max_actions]
    return rows


def source_rows(args: argparse.Namespace) -> Dict[tuple[str, str, str], Dict[str, str]]:
    rows: Dict[tuple[str, str, str], Dict[str, str]] = {}
    for role, path in (
        ("primary_gross", args.primary_csv),
        ("technical_quality_reserve", args.reserve_csv),
    ):
        for row in read_csv(path):
            rows[(role, row["collection_order"], row["scenario_token"])] = row
    return rows


def escaped_hydra_string(token: str) -> str:
    if not recovery.hydra_requires_quoted_token(token):
        raise ValueError(f"token is not Hydra numeric-like and must not use this recovery: {token}")
    # Stage7C shlex-splits its command template. Backslashes preserve the quote
    # characters in the final argv item passed to Hydra/OmegaConf.
    return f'\\"{token}\\"'


def build_stage7c_command(
    args: argparse.Namespace,
    row: Mapping[str, str],
    attempt_dir: Path,
    action: str = QUOTED_RETRY_ACTION,
) -> List[str]:
    context_dir = attempt_dir / "context"
    output_dir = attempt_dir / "stage7c_output"
    context_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    actual_token = (
        escaped_hydra_string(row["scenario_token"])
        if action == QUOTED_RETRY_ACTION
        else ""
    )
    context_row = {**row, "actual_nuplan_token": actual_token}
    write_csv(context_dir / "merged_metadata.csv", [context_row], CONTEXT_FIELDS)
    command_template = batch.build_command_template(
        args, args.nuplan_db_root / row["db_file"]
    ).replace(
        "stage7_m6_4b_locked_primary_mac_v1",
        "stage7_m6_4c_locked_recovery_mac_v1",
    )
    return [
        str(args.python_executable.resolve()),
        str(args.stage7c_tool.resolve()),
        "--context_dir",
        str(context_dir.resolve()),
        "--nuplan_db_root",
        str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root",
        str(args.nuplan_map_root.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--planners",
        *batch.EXPECTED_PLANNERS,
        "--max_scenarios",
        "1",
        "--min_timesteps",
        "2",
        "--require_same_scenario_alignment",
        "--require_strict_nuplan_token_alignment",
        "--allow_external_planner_name",
        "--hydra_searchpath",
        "[pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
        "--command_timeout_s",
        str(args.command_timeout_s),
        "--nuplan_simulation_command_template",
        command_template,
    ]


def execute_one(
    args: argparse.Namespace,
    action: Mapping[str, str],
    row: Mapping[str, str],
    scenario_dir: Path,
) -> Dict[str, Any]:
    attempt_dir = scenario_dir / "attempt_001"
    attempt_dir.mkdir(parents=True)
    command = build_stage7c_command(args, row, attempt_dir, action["action"])
    log_path = attempt_dir / "stage7c_driver.log"
    started = batch.utc_now()
    start_time = time.monotonic()
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
    duration = time.monotonic() - start_time
    output_dir = attempt_dir / "stage7c_output"
    output_audit = batch.audit_stage7c_output(output_dir, batch.EXPECTED_PLANNERS, row)
    if return_code != 0 and output_audit["failure_category"] == "INVALID_STAGE7C_OUTPUT":
        output_audit["failure_category"] = batch.classify_process_failure(return_code, log_path)
    result = {
        **action,
        "schema_version": SCHEMA_VERSION,
        "started_utc": started,
        "ended_utc": batch.utc_now(),
        "status": "SUCCEEDED" if output_audit["pass"] else "FAILED_REVIEW_REQUIRED",
        "return_code": return_code,
        "duration_seconds": duration,
        "stage7c_output_dir": str(output_dir.resolve()),
        **output_audit,
    }
    batch.atomic_write_json(attempt_dir / "attempt_summary.json", result)
    return result


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    runner_manifest = validate_inputs(args)
    actions = selected_actions(args)
    sources = source_rows(args)
    selected: List[tuple[Dict[str, str], Dict[str, str]]] = []
    for action in actions:
        key = (
            action["source_role"],
            action["collection_order"],
            action["scenario_token"],
        )
        row = sources.get(key)
        if row is None:
            raise ValueError(f"frozen source row not found for recovery action: {key}")
        if action["action"] == QUOTED_RETRY_ACTION:
            if action.get("source_role") != "primary_gross":
                raise ValueError("quoted-token recovery is restricted to frozen primary rows")
            escaped_hydra_string(row["scenario_token"])
        elif action.get("source_role") != "technical_quality_reserve":
            raise ValueError("reserve recovery is restricted to frozen reserve rows")
        selected.append((action, row))

    args.output_dir.mkdir(parents=True)
    runner_manifest["execute"] = bool(args.execute)
    runner_manifest["selected_action_count"] = len(selected)
    batch.atomic_write_json(args.output_dir / "recovery_runner_manifest.json", runner_manifest)
    write_csv(
        args.output_dir / "selected_recovery_plan.csv",
        [action for action, _ in selected],
        recovery.RECOVERY_PLAN_FIELDS,
    )
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "selected": len(selected)}, indent=2))
        return 0

    args.nuplan_exp_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for position, (action, row) in enumerate(selected, start=1):
        scenario_dir = args.output_dir / "rollouts" / batch.scenario_slug(row)
        print(
            f"[M6.4C recovery] START {position}/{len(selected)} "
            f"order={row['collection_order']} token={row['scenario_token']}",
            flush=True,
        )
        result = execute_one(args, action, row, scenario_dir)
        results.append(result)
        write_csv(args.output_dir / "recovery_status.csv", results, RESULT_FIELDS)
        print(
            f"[M6.4C recovery] DONE order={row['collection_order']} "
            f"status={result['status']} duration={result['duration_seconds']:.1f}s",
            flush=True,
        )
    state = {
        "schema_version": SCHEMA_VERSION,
        "completed_utc": batch.utc_now(),
        "selected_action_count": len(selected),
        "succeeded": sum(row["status"] == "SUCCEEDED" for row in results),
        "failed": sum(row["status"] != "SUCCEEDED" for row in results),
        "results": results,
    }
    batch.atomic_write_json(args.output_dir / "recovery_state.json", state)
    print(json.dumps({key: state[key] for key in ("succeeded", "failed")}, indent=2))
    return 0 if state["failed"] == 0 else 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated quoted-token retries from the audited M6.4C recovery plan."
    )
    parser.add_argument("--audit_summary", type=Path, required=True)
    parser.add_argument("--recovery_plan", type=Path, required=True)
    parser.add_argument("--locked_manifest", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--reserve_csv", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--batch_tool", type=Path, default=Path("tools/stage7_m6_4b_run_locked_rollouts.py"))
    parser.add_argument("--stage7c_tool", type=Path, default=Path("tools/stage7c1_run_nuplan_simulation.py"))
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--nuplan_data_root", type=Path, required=True)
    parser.add_argument("--nuplan_exp_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--tuplan_garage_root", type=Path, required=True)
    parser.add_argument("--python_executable", type=Path, required=True)
    parser.add_argument("--command_timeout_s", type=int, default=3600)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--action", choices=SUPPORTED_ACTIONS, default=QUOTED_RETRY_ACTION)
    parser.add_argument("--max_actions", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm_recovery_plan_sha256", default="")
    args = parser.parse_args(argv)
    if args.max_actions < 0:
        parser.error("--max_actions must be >= 0")
    if args.command_timeout_s < 1:
        parser.error("--command_timeout_s must be >= 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
