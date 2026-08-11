#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools import stage7_m6_4c_audit_locked_recovery as recovery
from tools import stage7_m6_4c_run_locked_recovery as recovery_runner
from tools import stage7_m6_4d_freeze_high_motion_supplement as supplement


SCHEMA_VERSION = "stage7_m6_4d_locked_supplement_runner_v1"
CONTEXT_FIELDS = batch.PRIMARY_FIELDS + ["actual_nuplan_token"]
RESULT_FIELDS = [
    "collection_order",
    "task",
    "task_rank",
    "selection_role",
    "scenario_token",
    "status",
    "failure_category",
    "return_code",
    "duration_seconds",
    "stage7c_output_dir",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path, required: Sequence[str] = ()) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        return [dict(row) for row in reader]


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


def source_definition(args: argparse.Namespace, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    if args.source == "primary":
        return {
            "path": args.primary_csv,
            "expected_csv_hash": manifest.get("primary_collection_csv_sha256"),
            "expected_manifest_hash": manifest.get("primary_manifest_sha256"),
            "expected_count": int(manifest.get("planned_primary_scenarios", -1)),
            "expected_role": "supplemental_primary",
        }
    if args.primary_run_state is None:
        raise ValueError("--primary_run_state is required when --source reserve")
    primary_state = recovery.read_json(args.primary_run_state)
    if primary_state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("primary_run_state schema mismatch")
    if primary_state.get("source") != "primary":
        raise ValueError("primary_run_state is not a supplemental primary run")
    results = list(primary_state.get("results", []))
    technical_failures = sum(row.get("status") != "SUCCEEDED" for row in results)
    if technical_failures < 1:
        raise ValueError("supplemental primary run has no technical failure; reserve is forbidden")
    return {
        "path": args.reserve_csv,
        "expected_csv_hash": manifest.get("reserve_collection_csv_sha256"),
        "expected_manifest_hash": manifest.get("reserve_manifest_sha256"),
        "expected_count": min(
            technical_failures, int(manifest.get("maximum_reserve_scenarios", -1))
        ),
        "available_count": int(manifest.get("maximum_reserve_scenarios", -1)),
        "expected_role": "supplemental_technical_reserve",
        "primary_run_state_sha256": recovery.sha256_file(args.primary_run_state),
    }


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, Any]]:
    manifest = recovery.read_json(args.supplement_manifest)
    if manifest.get("schema_version") != supplement.SCHEMA_VERSION:
        raise ValueError("supplement manifest schema mismatch")
    if manifest.get("status") != supplement.READY_STATUS:
        raise ValueError("supplement manifest is not frozen before M6.4D rollouts")
    if manifest.get("ready_to_launch_supplemental_rollouts") is not True:
        raise ValueError("supplement manifest is not ready to launch")
    if any(bool(value) for value in manifest.get("forbidden_inputs_read", {}).values()):
        raise ValueError("supplement freeze reports forbidden post-treatment input use")
    manifest_file_hash = recovery.sha256_file(args.supplement_manifest)
    if args.execute and args.confirm_supplement_manifest_sha256 != manifest_file_hash:
        raise ValueError(
            "--confirm_supplement_manifest_sha256 must equal the current supplement manifest file SHA-256"
        )

    input_hashes = manifest.get("input_hashes", {})
    validate_sha(
        args.selection_tool,
        str(input_hashes.get("selection_tool_sha256", "")),
        "M6.4D selection tool",
    )
    validate_sha(
        args.stage7c_tool,
        str(input_hashes.get("stage7c_tool_sha256", "")),
        "frozen Stage7C tool",
    )
    validate_sha(
        args.batch_tool,
        str(input_hashes.get("batch_tool_sha256", "")),
        "frozen M6.4B runner",
    )
    validate_sha(
        args.batch_manifest,
        str(input_hashes.get("batch_manifest_sha256", "")),
        "frozen M6.4B batch manifest",
    )
    source = source_definition(args, manifest)
    validate_sha(
        source["path"], str(source["expected_csv_hash"]), f"supplement {args.source} CSV"
    )
    rows = read_csv(source["path"], batch.PRIMARY_FIELDS)
    if args.source == "reserve":
        if len(rows) != int(source["available_count"]):
            raise ValueError("supplement reserve row count mismatch")
        rows = rows[: int(source["expected_count"])]
    elif len(rows) != int(source["expected_count"]):
        raise ValueError("supplement primary row count mismatch")
    if recovery.canonical_rows_hash(rows if args.source == "primary" else read_csv(source["path"], batch.PRIMARY_FIELDS)) != source[
        "expected_manifest_hash"
    ]:
        raise ValueError(f"supplement {args.source} canonical manifest hash mismatch")
    if args.execute and args.confirm_source_manifest_sha256 != source["expected_manifest_hash"]:
        raise ValueError(
            "--confirm_source_manifest_sha256 must equal the frozen source canonical hash"
        )
    expected_orders = list(range(1, len(rows) + 1))
    if [int(row["collection_order"]) for row in rows] != expected_orders:
        raise ValueError("selected source collection_order is not contiguous")
    if any(row["selection_role"] != source["expected_role"] for row in rows):
        raise ValueError("selected source has an unexpected selection_role")
    if any(row["selection_salt"] != manifest.get("selection_salt") for row in rows):
        raise ValueError("selected source has an unexpected selection_salt")
    for row in rows:
        technical = recovery.inspect_token_scene_position(
            args.nuplan_db_root / row["db_file"], row["scenario_token"]
        )
        if not technical["official_scene_position_valid"]:
            raise ValueError(
                f"supplement token is no longer technically runnable: {row['scenario_token']}"
            )

    batch_manifest = recovery.read_json(args.batch_manifest)
    if batch_manifest.get("planners") != batch.EXPECTED_PLANNERS:
        raise ValueError("planner list differs from the frozen M6.4B runtime")
    runtime_paths = {
        "nuplan_db_root": args.nuplan_db_root,
        "nuplan_map_root": args.nuplan_map_root,
        "nuplan_data_root": args.nuplan_data_root,
        "nuplan_exp_root": args.nuplan_exp_root,
        "python_executable": args.python_executable,
    }
    for key, actual in runtime_paths.items():
        if actual.resolve() != Path(str(batch_manifest.get(key, ""))).resolve():
            raise ValueError(f"{key} differs from the frozen M6.4B runtime")
    if args.command_timeout_s != int(batch_manifest.get("command_timeout_s", -1)):
        raise ValueError("command_timeout_s differs from the frozen M6.4B runtime")
    frozen_runtime = batch_manifest.get("frozen_input_audit", {})
    planner_fingerprints = batch.current_planner_fingerprints(batch.EXPECTED_PLANNERS)
    if planner_fingerprints != frozen_runtime.get("planner_parameter_fingerprints"):
        raise ValueError("planner fingerprints differ from the frozen runtime")
    nuplan_commit = batch.resolve_git_commit(args.nuplan_devkit_root)
    tuplan_commit = batch.resolve_git_commit(args.tuplan_garage_root)
    if nuplan_commit != frozen_runtime.get("nuplan_devkit_commit"):
        raise ValueError("nuPlan commit differs from the frozen runtime")
    if tuplan_commit != frozen_runtime.get("tuplan_garage_commit"):
        raise ValueError("tuPlan Garage commit differs from the frozen runtime")
    audit = {
        "supplement_manifest_sha256": manifest_file_hash,
        "source_manifest_sha256": source["expected_manifest_hash"],
        "source_csv_sha256": source["expected_csv_hash"],
        "selection_tool_sha256": input_hashes["selection_tool_sha256"],
        "stage7c_tool_sha256": input_hashes["stage7c_tool_sha256"],
        "batch_tool_sha256": input_hashes["batch_tool_sha256"],
        "planner_parameter_fingerprints": planner_fingerprints,
        "nuplan_devkit_commit": nuplan_commit,
        "tuplan_garage_commit": tuplan_commit,
    }
    if "primary_run_state_sha256" in source:
        audit["primary_run_state_sha256"] = source["primary_run_state_sha256"]
    return manifest, rows, audit


def hydra_actual_token(token: str) -> str:
    return recovery_runner.escaped_hydra_string(token) if recovery.hydra_requires_quoted_token(token) else ""


def build_stage7c_command(
    args: argparse.Namespace, row: Mapping[str, str], attempt_dir: Path
) -> List[str]:
    context_dir = attempt_dir / "context"
    output_dir = attempt_dir / "stage7c_output"
    context_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    context_row = {**row, "actual_nuplan_token": hydra_actual_token(row["scenario_token"])}
    write_csv(context_dir / "merged_metadata.csv", [context_row], CONTEXT_FIELDS)
    command_template = batch.build_command_template(
        args, args.nuplan_db_root / row["db_file"]
    ).replace(
        "stage7_m6_4b_locked_primary_mac_v1",
        "stage7_m6_4d_high_motion_supplement_mac_v1",
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
    args: argparse.Namespace, row: Mapping[str, str], scenario_dir: Path
) -> Dict[str, Any]:
    attempt_dir = scenario_dir / "attempt_001"
    attempt_dir.mkdir(parents=True)
    command = build_stage7c_command(args, row, attempt_dir)
    log_path = attempt_dir / "stage7c_driver.log"
    started = utc_now()
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
        **row,
        "schema_version": SCHEMA_VERSION,
        "started_utc": started,
        "ended_utc": utc_now(),
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
    manifest, rows, input_audit = validate_inputs(args)
    args.output_dir.mkdir(parents=True)
    runner_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "execute": bool(args.execute),
        "source": args.source,
        "selected_scenarios": len(rows),
        "runner_tool_sha256": recovery.sha256_file(Path(__file__)),
        "supplement_manifest": str(args.supplement_manifest.resolve()),
        "input_audit": input_audit,
        "planners": batch.EXPECTED_PLANNERS,
        "forbidden_inputs_read": {
            "embedding": False,
            "bdd": False,
            "effect_size": False,
            "trajectory_metrics_for_selection_or_stopping": False,
            "planner_outcome_for_selection_or_stopping": False,
        },
    }
    batch.atomic_write_json(args.output_dir / "supplement_runner_manifest.json", runner_manifest)
    write_csv(args.output_dir / "selected_supplement_plan.csv", rows, batch.PRIMARY_FIELDS)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "source": args.source, "selected": len(rows)}, indent=2))
        return 0

    args.nuplan_exp_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        scenario_dir = args.output_dir / "rollouts" / batch.scenario_slug(row)
        print(
            f"[M6.4D supplement] START {position}/{len(rows)} source={args.source} "
            f"order={row['collection_order']} token={row['scenario_token']}",
            flush=True,
        )
        result = execute_one(args, row, scenario_dir)
        results.append(result)
        write_csv(args.output_dir / "supplement_status.csv", results, RESULT_FIELDS)
        print(
            f"[M6.4D supplement] DONE order={row['collection_order']} "
            f"status={result['status']} duration={result['duration_seconds']:.1f}s",
            flush=True,
        )
    succeeded = sum(row["status"] == "SUCCEEDED" for row in results)
    failed = len(results) - succeeded
    state = {
        "schema_version": SCHEMA_VERSION,
        "completed_utc": utc_now(),
        "source": args.source,
        "supplement_manifest_sha256": input_audit["supplement_manifest_sha256"],
        "source_manifest_sha256": input_audit["source_manifest_sha256"],
        "selected_action_count": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "high_motion_complete_pairs_before_supplement": int(
            manifest["complete_pairs_before_supplement_by_task"][supplement.TASK]
        ),
        "high_motion_complete_pairs_after_this_run": int(
            manifest["complete_pairs_before_supplement_by_task"][supplement.TASK]
        )
        + succeeded,
        "results": results,
    }
    batch.atomic_write_json(args.output_dir / "supplement_state.json", state)
    print(
        json.dumps(
            {
                "source": args.source,
                "succeeded": succeeded,
                "failed": failed,
                "high_motion_complete_pairs_after_this_run": state[
                    "high_motion_complete_pairs_after_this_run"
                ],
            },
            indent=2,
        )
    )
    return 0 if failed == 0 else 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen outcome-blind M6.4D high-motion supplement."
    )
    parser.add_argument("--supplement_manifest", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--reserve_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--selection_tool", type=Path, default=Path("tools/stage7_m6_4d_freeze_high_motion_supplement.py"))
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
    parser.add_argument("--primary_run_state", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm_supplement_manifest_sha256", default="")
    parser.add_argument("--confirm_source_manifest_sha256", default="")
    args = parser.parse_args(argv)
    if args.command_timeout_s < 1:
        parser.error("--command_timeout_s must be >= 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
