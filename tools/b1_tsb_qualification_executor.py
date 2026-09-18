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
from tools.b1_native_route_precheck import precheck_with_production_builder  # noqa: E402
from tools.b1_offline_metric_finalize import EXPECTED_OFFICIAL_METRICS  # noqa: E402

SCHEMA_VERSION = "b1_tsb_arm_execution_manifest_v1"
ALLOWED_STAGE = "B1_FROZEN_TSB_QUALIFICATION"


class B1ExecutionError(RuntimeError):
    pass


LIFECYCLE_ORDER = (
    "RUNNER_COMPLETE",
    "RECORDER_COMPLETE",
    "OFFICIAL_METRICS_COMPLETE",
    "SERIALIZER_COMPLETE",
    "MANIFEST_COMPLETE",
    "ARTIFACT_HASHES_VALIDATED",
)
RETRY_CONTRACT = {
    "maximum_total_attempts_per_arm": 2,
    "authority": "TECHNICAL_RETRY_ALLOWED_IF_OWNER_AUTHORIZED",
    "eligible_failure_classes": [
        "INFRASTRUCTURE_FAILURE",
        "CALLBACK_FINALIZATION_FAILURE",
        "SERIALIZER_FAILURE",
        "ARTIFACT_PERSISTENCE_FAILURE",
        "NATIVE_ROUTE_INFRASTRUCTURE_FAILURE",
    ],
    "ineligible_failure_classes": [
        "SCIENTIFIC_FAIL",
        "MEASUREMENT_INVALID_REALIZED_BEHAVIOR",
        "LOW_SPEED_ENDSTOP",
        "MECHANISM_FAIL",
        "F_MATCH_FAIL",
        "OFFICIAL_SAFETY_FAIL",
    ],
    "same_identity_required": True,
    "replacement_forbidden": True,
    "denominator_rule": "ORIGINAL_PAIR_COUNTS_ONCE;ALL_ATTEMPTS_RETAINED",
    "authoritative_result_rule": "LATEST_OWNER_AUTHORIZED_TECHNICALLY_COMPLETE_ATTEMPT;PRIOR_ATTEMPTS_RETAINED",
}


def validate_retry_request(attempt_number: int, failure_class: str, owner_authorized: bool) -> None:
    """Validate a prospective retry without granting authority."""
    if not owner_authorized:
        raise B1ExecutionError("B1_TECHNICAL_RETRY_OWNER_AUTHORIZATION_REQUIRED")
    if attempt_number != 2:
        raise B1ExecutionError("B1_TECHNICAL_RETRY_MAXIMUM_TOTAL_ATTEMPTS_IS_2")
    if failure_class not in RETRY_CONTRACT["eligible_failure_classes"]:
        raise B1ExecutionError(f"B1_TECHNICAL_RETRY_FAILURE_CLASS_INELIGIBLE:{failure_class}")


def advance_lifecycle(completed: tuple[str, ...], next_state: str) -> tuple[str, ...]:
    """Advance the fail-closed post-run lifecycle by exactly one state."""
    expected = LIFECYCLE_ORDER[len(completed)] if len(completed) < len(LIFECYCLE_ORDER) else None
    if next_state != expected:
        raise B1ExecutionError(f"B1_LIFECYCLE_ORDER_VIOLATION:expected={expected}:received={next_state}")
    return (*completed, next_state)


def official_metric_finalizer_bound(arm: Any) -> bool:
    """Require the official nuPlan MetricFileCallback in the main callback chain."""
    callbacks = getattr(getattr(arm, "common_builder", None), "multi_main_callback", None)
    members = getattr(callbacks, "_main_callbacks", [])
    return any(
        callback.__class__.__module__ == "nuplan.planning.simulation.main_callback.metric_file_callback"
        and callback.__class__.__name__ == "MetricFileCallback"
        for callback in members
    )


def validate_pre_run_lifecycle(arm: Any, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run all zero-rollout gates that must dominate budget claim and runner entry."""
    if not official_metric_finalizer_bound(arm):
        raise B1ExecutionError("B1_OFFICIAL_METRIC_FINALIZER_NOT_BOUND")
    scenario = getattr(getattr(arm, "simulation", None), "_scenario", None)
    map_api = getattr(scenario, "map_api", None)
    if map_api is None:
        raise B1ExecutionError("B1_NATIVE_ROUTE_MAP_API_NOT_BOUND")
    route = precheck_with_production_builder(spec, map_api)
    if route["route_precheck_status"] != "COMPATIBLE":
        raise B1ExecutionError(f"B1_ROUTE_PRECHECK_NOT_COMPATIBLE:{route['failure_code']}")
    return route


def validate_and_hash_artifacts(arm: Any) -> Mapping[str, str]:
    """Require official outputs and hash the immutable post-run artifact set."""
    required = [
        arm.run_root / "trace/realized_current_ego.jsonl",
        arm.run_root / "telemetry/planner_transfer.jsonl",
        arm.run_root / "telemetry/actual_lqr_controller_telemetry.jsonl",
        arm.common_builder.output_dir / arm.cfg.runner_report_file,
        *sorted((arm.run_root / "raw/simulation_log").rglob("*.msgpack.xz")),
    ]
    metric_dir = arm.run_root / "raw/metrics"
    required.extend(metric_dir / f"{name}.parquet" for name in sorted(EXPECTED_OFFICIAL_METRICS))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise B1ExecutionError(f"B1_REQUIRED_POST_RUN_ARTIFACT_MISSING:{missing}")
    return {str(path.relative_to(arm.run_root)): sha256_file(path) for path in required}


def validate_serializer_complete(arm: Any) -> Path:
    """Require the configured official simulation-log serializer output."""
    serialized = sorted((arm.run_root / "raw/simulation_log").rglob("*.msgpack.xz"))
    if len(serialized) != 1 or not serialized[0].is_file() or serialized[0].stat().st_size == 0:
        raise B1ExecutionError(f"B1_SERIALIZER_OUTPUT_INVALID:expected=1:observed={len(serialized)}")
    return serialized[0]


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


def arm_manifest(
    spec: Mapping[str, Any], arm: Any, authorization: Mapping[str, Any], status: str,
    lifecycle: tuple[str, ...] = (), route_precheck: Mapping[str, Any] | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
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
        "lifecycle_states_complete": list(lifecycle),
        "route_precheck": route_precheck,
        "artifact_hashes": dict(artifact_hashes or {}),
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
    route_precheck = validate_pre_run_lifecycle(arm, spec)
    manifest_path = run_root / "execution_manifest.json"
    ledger_path = output_root / "B1_Scientific_Arm_Budget_Ledger_v1.json"
    write_manifest(manifest_path, arm_manifest(spec, arm, authorization, "READY_BEFORE_RUNNER_RUN", route_precheck=route_precheck))
    claim_budget(ledger_path, authorization, run_id)
    lifecycle: tuple[str, ...] = ()
    try:
        report = arm.runner.run()
        if not bool(getattr(report, "succeeded", False)):
            raise B1ExecutionError("RUNNER_REPORT_NOT_SUCCEEDED")
        lifecycle = advance_lifecycle(lifecycle, "RUNNER_COMPLETE")
        arm.recorder.validate_complete()
        lifecycle = advance_lifecycle(lifecycle, "RECORDER_COMPLETE")
        from nuplan.planning.script.utils import save_runner_reports
        save_runner_reports([report], arm.common_builder.output_dir, arm.cfg.runner_report_file)
        arm.common_builder.multi_main_callback.on_run_simulation_end()
        lifecycle = advance_lifecycle(lifecycle, "OFFICIAL_METRICS_COMPLETE")
        validate_serializer_complete(arm)
        lifecycle = advance_lifecycle(lifecycle, "SERIALIZER_COMPLETE")
        write_manifest(
            manifest_path,
            arm_manifest(spec, arm, authorization, "MANIFEST_PENDING_HASH_VALIDATION", lifecycle, route_precheck),
        )
        lifecycle = advance_lifecycle(lifecycle, "MANIFEST_COMPLETE")
        artifact_hashes = validate_and_hash_artifacts(arm)
        lifecycle = advance_lifecycle(lifecycle, "ARTIFACT_HASHES_VALIDATED")
    except Exception:
        write_manifest(manifest_path, arm_manifest(spec, arm, authorization, "TECHNICAL_INCOMPLETE", lifecycle, route_precheck))
        update_budget_status(ledger_path, run_id, "TECHNICAL_INCOMPLETE")
        raise
    write_manifest(manifest_path, arm_manifest(spec, arm, authorization, "ARM_COMPLETE", lifecycle, route_precheck, artifact_hashes))
    update_budget_status(ledger_path, run_id, "ARM_COMPLETE")
    return {"run_id": run_id, "status": "ARM_COMPLETE", "manifest": str(manifest_path)}


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
