#!/usr/bin/env python3
"""Authorized one-arm executor for the frozen B1 TSB qualification roster."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.s2r_production_executor import (  # noqa: E402
    ANALYZER,
    PARAMETERS,
    PLANNER,
    RECORDER,
    ROOT,
    SCHEMA_VERSION as S2R_SCHEMA_VERSION,
    SERIALIZER,
    build_fresh_arm,
    canonical_sha256,
    component_identity,
    runtime_fingerprint,
    sha256_file,
    write_manifest,
)

SCHEMA_VERSION = "b1_tsb_arm_execution_manifest_v1"
ALLOWED_STAGE = "B1_FROZEN_TSB_QUALIFICATION"


class B1ExecutionError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob_matches(commit: str, relative: str, expected_sha: str) -> bool:
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest() == expected_sha


def validate_authorization(
    authorization: Mapping[str, Any], specs_path: Path, roster_path: Path, run_id: str
) -> Mapping[str, Any]:
    if authorization.get("stage") != ALLOWED_STAGE or authorization.get("authorization_status") != "AUTHORIZED":
        raise B1ExecutionError("B1_AUTHORIZATION_NOT_ACTIVE")
    if authorization.get("allowed_role") != "B":
        raise B1ExecutionError("B1_ROLE_NOT_AUTHORIZED")
    if int(authorization.get("max_scientific_arms", 0)) != 40:
        raise B1ExecutionError("B1_RUN_BUDGET_NOT_40")
    if sha256_file(Path(__file__)) != authorization.get("executor_sha256"):
        raise B1ExecutionError("B1_EXECUTOR_SHA_MISMATCH")
    if sha256_file(PARAMETERS) != authorization.get("tsb_config_sha256"):
        raise B1ExecutionError("B1_TSB_CONFIG_SHA_MISMATCH")
    if sha256_file(ANALYZER) != authorization.get("analyzer_sha256"):
        raise B1ExecutionError("B1_ANALYZER_SHA_MISMATCH")
    if sha256_file(SERIALIZER) != authorization.get("schema_sha256"):
        raise B1ExecutionError("B1_SCHEMA_SHA_MISMATCH")
    if sha256_file(PLANNER) != authorization.get("planner_sha256") or sha256_file(RECORDER) != authorization.get("recorder_sha256"):
        raise B1ExecutionError("B1_PRODUCTION_COMPONENT_SHA_MISMATCH")
    if sha256_file(specs_path) != authorization.get("arm_specs_sha256"):
        raise B1ExecutionError("B1_ARM_SPECS_SHA_MISMATCH")
    if sha256_file(roster_path) != authorization.get("roster_sha256"):
        raise B1ExecutionError("B1_ROSTER_SHA_MISMATCH")
    allowed = {str(row["run_id"]): row for row in authorization.get("allowed_runs", [])}
    if len(allowed) != 40 or run_id not in allowed:
        raise B1ExecutionError("B1_RUN_ID_NOT_AUTHORIZED")
    frozen_commit = str(authorization.get("authorized_source_git_sha", ""))
    frozen_files = authorization.get("source_file_hashes", {})
    if not frozen_commit or not frozen_files:
        raise B1ExecutionError("B1_FROZEN_SOURCE_BINDING_MISSING")
    for relative, expected in frozen_files.items():
        path = ROOT / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise B1ExecutionError(f"B1_FROZEN_SOURCE_WORKTREE_MISMATCH:{relative}")
        if not _git_blob_matches(frozen_commit, str(relative), str(expected)):
            raise B1ExecutionError(f"B1_FROZEN_SOURCE_COMMIT_MISMATCH:{relative}")
    return allowed[run_id]


def claim_budget(ledger_path: Path, authorization: Mapping[str, Any], run_id: str) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        if raw:
            ledger = json.loads(raw)
        else:
            ledger = {
                "schema_version": "b1_scientific_arm_budget_ledger_v1",
                "max_scientific_arms": int(authorization["max_scientific_arms"]),
                "runs": {str(row["run_id"]): "NOT_RUN" for row in authorization["allowed_runs"]},
            }
        if ledger["runs"].get(run_id) != "NOT_RUN":
            raise B1ExecutionError(f"B1_RUN_ALREADY_CLAIMED:{run_id}")
        claimed = sum(status != "NOT_RUN" for status in ledger["runs"].values())
        if claimed >= int(ledger["max_scientific_arms"]):
            raise B1ExecutionError("B1_SCIENTIFIC_ARM_BUDGET_EXHAUSTED")
        ledger["runs"][run_id] = "RUNNER_ENTRY_CLAIMED"
        handle.seek(0)
        handle.truncate()
        json.dump(ledger, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_budget_status(ledger_path: Path, run_id: str, status: str) -> None:
    with ledger_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ledger = json.load(handle)
        if ledger["runs"].get(run_id) == "NOT_RUN":
            raise B1ExecutionError("B1_LEDGER_STATUS_WITHOUT_CLAIM")
        ledger["runs"][run_id] = status
        handle.seek(0)
        handle.truncate()
        json.dump(ledger, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def arm_manifest(spec: Mapping[str, Any], arm: Any, authorization: Mapping[str, Any], status: str) -> Mapping[str, Any]:
    identities = component_identity(arm)
    return {
        "schema_version": SCHEMA_VERSION,
        "inherited_s2r_schema_version": S2R_SCHEMA_VERSION,
        "stage": ALLOWED_STAGE,
        "run_id": spec["run_id"],
        "pair_id": spec["pair_id"],
        "arm": spec["arm"],
        "session_id": spec["session_id"],
        "log_id": spec["log_id"],
        "scenario_token": spec["scenario_token"],
        "exposure_class": spec["exposure_class"],
        "role": spec["role"],
        "authorized_source_git_sha": authorization["authorized_source_git_sha"],
        "executor_sha": sha256_file(Path(__file__)),
        "planner_sha": sha256_file(PLANNER),
        "config_hash": arm.config_hash,
        "tsb_config_sha": sha256_file(PARAMETERS),
        "analyzer_sha": sha256_file(ANALYZER),
        "schema_sha": sha256_file(SERIALIZER),
        "runtime_fingerprint": runtime_fingerprint(),
        "precontext_id": arm.precontext_id,
        "route_id": arm.route_id,
        "arm_instance_id": canonical_sha256({"run_id": spec["run_id"], "component_ids": identities}),
        "runner_instance_id": str(identities["runner"]),
        "planner_instance_id": str(identities["planner"]),
        "simulation_instance_id": str(identities["simulation"]),
        "controller_instance_id": str(identities["controller"]),
        "tracker_instance_id": str(identities["tracker"]),
        "motion_model_instance_id": str(identities["motion_model"]),
        "callback_instance_id": str(identities["callbacks"]),
        "recorder_instance_id": str(identities["recorder"]),
        "random_state_instance_id": str(identities["random_state"]),
        "random_state_declaration": {"seed": 2026091801, "fresh_process_per_arm": True},
        "start_timestamp_us": int(spec["start_timestamp_us"]),
        "expected_output_paths": {
            "root": str(arm.run_root),
            "trace": str(arm.run_root / "trace/realized_current_ego.jsonl"),
            "planner_telemetry": str(arm.run_root / "telemetry/planner_transfer.jsonl"),
            "controller_telemetry": str(arm.run_root / "telemetry/actual_lqr_controller_telemetry.jsonl"),
            "raw": str(arm.run_root / "raw"),
            "manifest": str(arm.run_root / "execution_manifest.json"),
        },
        "execution_status": status,
    }


def execute(
    authorization_path: Path, specs_path: Path, roster_path: Path, run_id: str, output_root: Path
) -> Mapping[str, Any]:
    authorization = load_json(authorization_path)
    allowed = validate_authorization(authorization, specs_path, roster_path, run_id)
    specs = load_json(specs_path)
    spec_by_id = {str(row["run_id"]): row for row in specs["arms"]}
    if run_id not in spec_by_id or str(allowed["arm"]) != str(spec_by_id[run_id]["arm"]):
        raise B1ExecutionError("B1_AUTHORIZED_ARM_SPEC_MISMATCH")
    spec = spec_by_id[run_id]
    run_root = output_root / run_id
    arm = build_fresh_arm(spec, run_root)
    manifest_path = run_root / "execution_manifest.json"
    ledger_path = output_root / "B1_Scientific_Arm_Budget_Ledger_v1.json"
    write_manifest(manifest_path, arm_manifest(spec, arm, authorization, "READY_BEFORE_RUNNER_RUN"))
    claim_budget(ledger_path, authorization, run_id)
    try:
        report = arm.runner.run()
        if not bool(getattr(report, "succeeded", False)):
            raise B1ExecutionError("RUNNER_REPORT_NOT_SUCCEEDED")
        arm.recorder.validate_complete()
    except Exception:
        write_manifest(manifest_path, arm_manifest(spec, arm, authorization, "TECHNICAL_INCOMPLETE"))
        update_budget_status(ledger_path, run_id, "TECHNICAL_INCOMPLETE")
        raise
    write_manifest(manifest_path, arm_manifest(spec, arm, authorization, "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS"))
    update_budget_status(ledger_path, run_id, "TECHNICAL_COMPLETE")
    return {"run_id": run_id, "status": "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS", "manifest": str(manifest_path)}


def main() -> int:
    if len(sys.argv) != 7 or sys.argv[1] != "--execute-authorized-arm":
        raise SystemExit(
            "usage: b1_tsb_qualification_executor.py --execute-authorized-arm AUTH SPECS ROSTER RUN_ID OUTPUT_ROOT"
        )
    result = execute(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], Path(sys.argv[6]))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
