#!/usr/bin/env python3
"""Fail-closed technical recovery for the frozen R2-A DEV schedule."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r2_a_execute_controller_transfer_dev import (  # noqa: E402
    LEDGER,
    ROOT,
    STATIC_RUN_KEYS,
    _authorization_check,
    _construct,
    _load_frozen,
    write_json,
)


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _actual_run_count() -> int:
    return len(list((ROOT / "outputs").glob("r2_a_controller_transfer_dev_v1*/**/runner_report.parquet")))


def _record_unledgered_bookkeeping_attempt(ledger: Dict[str, Any]) -> None:
    run_id = "R2A-HLC-01-HLC_MONOTONIC_REFERENCE-TECHRERUN01"
    root = ROOT / "outputs/r2_a_controller_transfer_dev_v1_attempt2" / run_id
    if any(row.get("run_root") == str(root.relative_to(ROOT)) for row in ledger.get("technical_reruns", [])):
        return
    counts = {
        "realized_trace": _line_count(root / "trace/realized_current_ego.jsonl"),
        "planner_telemetry": _line_count(root / "telemetry/planner_transfer.jsonl"),
        "controller_commands": _line_count(root / "telemetry/controller_commands.jsonl"),
    }
    if counts != {"realized_trace": 80, "planner_telemetry": 80, "controller_commands": 79}:
        raise RuntimeError("R2_A_UNLEDGERED_ATTEMPT_TELEMETRY_VALIDATION_FAIL")
    ledger.setdefault("technical_reruns", []).append(
        {
            "run_id": run_id,
            "technical_rerun_of": "R2A-HLC-01-HLC_MONOTONIC_REFERENCE",
            "attempt": 2,
            "status": "TECHNICAL_BOOKKEEPING_FAILURE_AFTER_SUCCESSFUL_NUPLAN_RUN",
            "technical_failure": "ValueError:RELATIVE_OUTPUT_ROOT_COULD_NOT_BE_RELATIVIZED",
            "run_root": str(root.relative_to(ROOT)),
            "telemetry_counts": counts,
            "excluded_from_transfer_analysis": True,
            "fresh_run_id_requirement": "NOT_SATISFIED_DUE_TO_LATER_DUPLICATE_DISCOVERY",
        }
    )


def recover_current_failure(output_root: Path) -> Dict[str, Any]:
    _authorization_check()
    ledger, entries, excitations, _ = _load_frozen()
    if ledger.get("status") != "TECHNICAL_FAILURE_STOPPED_REMAINING_SCHEDULE":
        raise PermissionError("R2_A_CURRENT_LEDGER_NOT_STOPPED_ON_TECHNICAL_FAILURE")
    _record_unledgered_bookkeeping_attempt(ledger)
    failed = [row for row in ledger["runs"] if row.get("status") == "TECHNICAL_FAILURE"]
    if len(failed) != 1:
        raise RuntimeError(f"R2_A_EXPECTED_ONE_CURRENT_TECHNICAL_FAILURE:{len(failed)}")
    original = failed[0]
    prior_for_base = [
        row for row in ledger.get("technical_reruns", [])
        if row.get("technical_rerun_of") == original["run_id"]
    ]
    recovery_number = len(prior_for_base) + 1
    recovery = {key: original[key] for key in STATIC_RUN_KEYS}
    recovery.update(
        {
            "run_id": f"{original['run_id']}-TECHRERUN{recovery_number:02d}",
            "attempt": recovery_number + 1,
            "technical_rerun_of": original["run_id"],
            "status": "TECHNICAL_RERUN_AUTHORIZED",
        }
    )
    root = output_root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"R2_A_RECOVERY_OUTPUT_ROOT_REUSE:{root}")
    root.mkdir(parents=True)
    ledger["status"] = "AUTHORIZED_TECHNICAL_RECOVERY_RUNNING"
    write_json(LEDGER, ledger)
    try:
        audit = _construct(
            recovery,
            entries[str(recovery["scenario_token"])],
            excitations[str(recovery["excitation_id"])],
            root,
            True,
        )
    except Exception as exc:
        ledger["status"] = "TECHNICAL_RECOVERY_FAILED_STOPPED"
        ledger.setdefault("technical_reruns", []).append(
            {**recovery, "status": "TECHNICAL_FAILURE", "technical_failure": f"{type(exc).__name__}:{exc}"}
        )
        write_json(LEDGER, ledger)
        raise
    audit["technical_rerun_reason"] = original["technical_failure"]
    audit["technical_rerun_of"] = original["run_id"]
    audit["attempt"] = recovery["attempt"]
    ledger.setdefault("technical_reruns", []).append(audit)
    original.update(
        {
            "status": "TECHNICAL_COMPLETE_AFTER_FRESH_RERUN",
            "effective_run_id": recovery["run_id"],
            "effective_run_root": audit["run_root"],
            "effective_trace_path": audit["trace_path"],
            "effective_planner_telemetry_path": audit["planner_telemetry_path"],
            "effective_controller_command_path": audit["controller_command_path"],
        }
    )
    write_json(LEDGER, ledger)
    print(json.dumps({"progress": "R2_A_TECHNICAL_RECOVERY", "run_id": recovery["run_id"]}), flush=True)
    for row in sorted(ledger["runs"], key=lambda item: int(item["run_order"])):
        if int(row["run_order"]) <= int(original["run_order"]):
            continue
        if row.get("status") != "PLANNED_FROZEN_PRE_EXECUTION":
            raise RuntimeError(f"R2_A_RECOVERY_REMAINING_ROW_NOT_PRISTINE:{row['run_id']}:{row.get('status')}")
        try:
            row_audit = _construct(
                row,
                entries[str(row["scenario_token"])],
                excitations[str(row["excitation_id"])],
                root,
                True,
            )
        except Exception as exc:
            row["status"] = "TECHNICAL_FAILURE"
            row["technical_failure"] = f"{type(exc).__name__}:{exc}"
            ledger["status"] = "TECHNICAL_FAILURE_STOPPED_REMAINING_SCHEDULE"
            ledger["counts"]["executed"] = sum(
                item.get("status", "").startswith("TECHNICAL_COMPLETE") for item in ledger["runs"]
            )
            ledger["counts"]["actual_engineering_runs"] = _actual_run_count()
            write_json(LEDGER, ledger)
            raise
        row.update(row_audit)
        ledger["counts"]["executed"] = sum(
            item.get("status", "").startswith("TECHNICAL_COMPLETE") for item in ledger["runs"]
        )
        ledger["counts"]["technical_reruns"] = len(ledger.get("technical_reruns", []))
        ledger["counts"]["actual_engineering_runs"] = _actual_run_count()
        write_json(LEDGER, ledger)
        print(json.dumps({"progress": "R2_A_ENGINEERING_RECOVERY", "completed": ledger["counts"]["executed"], "total": 80, "run_id": row["run_id"]}), flush=True)
    ledger["status"] = "80_OF_80_EFFECTIVE_RUNS_COMPLETE_PENDING_UNIQUE_FIRST_RUN_RECOVERY"
    ledger["counts"] = {
        "planned": 80,
        "executed": 80,
        "technical_reruns": len(ledger.get("technical_reruns", [])),
        "actual_engineering_runs": _actual_run_count(),
    }
    ledger["scientific_simulations"] = 0
    write_json(LEDGER, ledger)
    return ledger


def finalize_unique_first_run(output_root: Path, audit_output: Path) -> Dict[str, Any]:
    _authorization_check()
    ledger, entries, excitations, _ = _load_frozen()
    if ledger.get("status") != "80_OF_80_EFFECTIVE_RUNS_COMPLETE_PENDING_UNIQUE_FIRST_RUN_RECOVERY":
        raise PermissionError("R2_A_FULL_EFFECTIVE_SCHEDULE_NOT_READY_FOR_FINAL_UNIQUE_RECOVERY")
    original = min(ledger["runs"], key=lambda row: int(row["run_order"]))
    duplicate = [
        row for row in ledger.get("technical_reruns", [])
        if row.get("run_id") == f"{original['run_id']}-TECHRERUN01"
    ]
    if len(duplicate) != 2:
        raise RuntimeError(f"R2_A_EXPECTED_TWO_DUPLICATE_TECHRERUN01_RECORDS:{len(duplicate)}")
    for row in duplicate:
        row.setdefault("family", original["family"])
        row.setdefault("scenario_token", original["scenario_token"])
        row.setdefault("log_id", original["log_id"])
        row.setdefault("excitation_id", original["excitation_id"])
        row["technical_rerun_of"] = original["run_id"]
        row["excluded_from_transfer_analysis"] = True
        row["exclusion_reason"] = "DUPLICATE_TECHNICAL_RERUN_ID_ACROSS_DISTINCT_FRESH_OUTPUT_ROOTS"
    recovery = {key: original[key] for key in STATIC_RUN_KEYS}
    recovery.update(
        {
            "run_id": f"{original['run_id']}-TECHRERUN02",
            "attempt": 4,
            "technical_rerun_of": original["run_id"],
            "status": "TECHNICAL_RERUN_AUTHORIZED_FOR_UNIQUE_ID_CLOSURE",
        }
    )
    root = output_root.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"R2_A_FINAL_RECOVERY_OUTPUT_ROOT_REUSE:{root}")
    root.mkdir(parents=True)
    audit = _construct(
        recovery,
        entries[str(recovery["scenario_token"])],
        excitations[str(recovery["excitation_id"])],
        root,
        True,
    )
    audit["technical_rerun_reason"] = "FRESH_RUN_ID_UNIQUENESS_CLOSURE_AFTER_BOOKKEEPING_FAILURE"
    audit["technical_rerun_of"] = original["run_id"]
    audit["attempt"] = recovery["attempt"]
    ledger["technical_reruns"].append(audit)
    frozen_run_id = str(original["run_id"])
    frozen_attempt = int(original["attempt"])
    frozen_technical_rerun_of = original["technical_rerun_of"]
    original.update(audit)
    original["frozen_run_id"] = frozen_run_id
    original["effective_run_id"] = recovery["run_id"]
    original["run_id"] = frozen_run_id
    original["attempt"] = frozen_attempt
    original["technical_rerun_of"] = frozen_technical_rerun_of
    original["status"] = "TECHNICAL_COMPLETE_AFTER_UNIQUE_FRESH_RERUN"
    original["effective_run_root"] = audit["run_root"]
    original["effective_trace_path"] = audit["trace_path"]
    original["effective_planner_telemetry_path"] = audit["planner_telemetry_path"]
    original["effective_controller_command_path"] = audit["controller_command_path"]
    effective_runs = []
    for row in sorted(ledger["runs"], key=lambda item: int(item["run_order"])):
        effective_runs.append(
            {
                "run_order": row["run_order"],
                "frozen_run_id": row.get("frozen_run_id", row["run_id"]),
                "effective_run_id": row.get("effective_run_id", row["run_id"]),
                "family": row["family"],
                "scenario_token": row["scenario_token"],
                "excitation_id": row["excitation_id"],
                "status": row["status"],
                "run_root": row.get("effective_run_root", row.get("run_root")),
                "trace_path": row.get("effective_trace_path", row.get("trace_path")),
                "planner_telemetry_path": row.get("effective_planner_telemetry_path", row.get("planner_telemetry_path")),
                "controller_command_path": row.get("effective_controller_command_path", row.get("controller_command_path")),
            }
        )
    if len(effective_runs) != 80 or len({row["effective_run_id"] for row in effective_runs}) != 80:
        raise RuntimeError("R2_A_EFFECTIVE_RUN_UNIQUENESS_CLOSURE_FAIL")
    ledger["status"] = "80_OF_80_FROZEN_DEV_RUNS_TECHNICAL_COMPLETE"
    ledger["counts"] = {
        "planned": 80,
        "executed": 80,
        "technical_reruns": len(ledger["technical_reruns"]),
        "actual_engineering_runs": _actual_run_count(),
    }
    ledger["effective_run_ids_unique"] = True
    ledger["scientific_simulations"] = 0
    write_json(LEDGER, ledger)
    result = {
        "schema_version": "r2_a_controller_transfer_execution_audit_v1.0",
        "status": ledger["status"],
        "effective_runs": effective_runs,
        "technical_reruns": ledger["technical_reruns"],
        "counts": ledger["counts"],
        "Primary80_planner_calls_per_effective_run": 80,
        "controller_transitions_per_effective_run": 79,
        "actual_runner_run_calls": ledger["counts"]["actual_engineering_runs"],
        "scientific_simulations": 0,
        "confirmatory_roster_selected": False,
        "RBR_started": False,
    }
    if audit_output.exists():
        raise FileExistsError(f"R2_A_EXECUTION_AUDIT_EXISTS:{audit_output}")
    write_json(audit_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("recover-current", "finalize-unique-first"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.mode == "recover-current":
        result = recover_current_failure(args.output_root)
    else:
        if args.audit_output is None:
            raise ValueError("FINALIZE_UNIQUE_FIRST_REQUIRES_AUDIT_OUTPUT")
        result = finalize_unique_first_run(args.output_root, args.audit_output)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
