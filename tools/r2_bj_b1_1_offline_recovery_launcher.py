#!/usr/bin/env python3
"""One-shot offline B1.1 recovery over immutable B1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.r2_bj_b1_1_offline_recovery_analyzer_v1 import RESULT_STATES, analyze_frozen_canary_pair


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
B1_MANIFEST = R2 / "r2_bj_b1_post_outcome_component_artifact_binding_manifest_v1.0.json"
AUTHORIZATION = R2 / "r2_bj_b1_1_offline_analysis_one_shot_authorization_v1.0.json"
SCHEDULE = R2 / "r2_bj_b0_hlc_v4_pair_schedule_v1.0.json"
RAW_ROOT = ROOT / "outputs/r2_bj_b0_1_canary_once_v1"
CONTROL_ROOT = ROOT / "outputs/r2_bj_b0_1_canary_once_control_v1"
OUTPUT_ROOT = ROOT / "outputs/r2_bj_b1_1_offline_analyzer_recovery_once_v1"
EXPECTED_B1_MANIFEST_SHA256 = "34aebc06b24784a4382f196d25c77ebef4263eb9adab467d1af731834889745c"
EXPECTED_ORIGINAL_ANALYZER_SHA256 = "b0a3daf7cc2234c5c77ad3800e0d15feecc377aca87ffff3141bbb51e8423da6"
EXPECTED_V4_PARAMETER_SHA256 = "95b6b726a42f9501f6f5401e8b2e5e179cadb489b74087a09667889efd31a158"
EXPECTED_B02_MANIFEST_SHA256 = "dac808cb1f75c26c15223226d9b3c296de0256ff007427a86a2a7d14f6b5b62c"


class OfflineRecoveryStop(RuntimeError):
    """Fail-closed offline recovery error."""


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _inventory(root: Path) -> Mapping[str, Any]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        records.append((path.relative_to(root).as_posix(), path.stat().st_size, sha(path)))
    payload = "".join(f"{name}\0{size}\0{digest}\n" for name, size, digest in records).encode("utf-8")
    return {
        "file_count": len(records),
        "total_bytes": sum(row[1] for row in records),
        "canonical_inventory_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_b1_artifact_closure() -> None:
    if sha(B1_MANIFEST) != EXPECTED_B1_MANIFEST_SHA256:
        raise OfflineRecoveryStop("B1_POST_OUTCOME_MANIFEST_SHA_MISMATCH")
    manifest = read(B1_MANIFEST)
    expected_fixed = {
        "tools/r2_bj_b0_2_frozen_canary_pair_analyzer.py": EXPECTED_ORIGINAL_ANALYZER_SHA256,
        "docs/stageR/r2/r2_bj_a_hlc_global_parameter_space_v4.0.json": EXPECTED_V4_PARAMETER_SHA256,
        "docs/stageR/r2/r2_bj_b0_2_execution_observability_sha_manifest_v1.0.json": EXPECTED_B02_MANIFEST_SHA256,
    }
    for relative, expected in expected_fixed.items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise OfflineRecoveryStop(f"FROZEN_INPUT_SHA_MISMATCH:{relative}")
    for row in manifest["frozen_components"] + manifest["committed_result_artifacts"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise OfflineRecoveryStop(f"B1_MANIFEST_COMPONENT_SHA_MISMATCH:{row['path']}")
    authorization = manifest["authorization_binding"]
    authorization_path = ROOT / authorization["path"]
    if not authorization_path.is_file() or sha(authorization_path) != authorization["file_sha256"]:
        raise OfflineRecoveryStop("B1_AUTHORIZATION_FILE_SHA_MISMATCH")
    for row in manifest["control_artifacts"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise OfflineRecoveryStop(f"B1_CONTROL_ARTIFACT_SHA_MISMATCH:{row['path']}")
    for run_id, artifacts in manifest["run_artifacts"].items():
        for row in artifacts:
            path = RAW_ROOT / run_id / row["path"]
            if not path.is_file() or sha(path) != row["sha256"]:
                raise OfflineRecoveryStop(f"B1_RUN_ARTIFACT_SHA_MISMATCH:{run_id}:{row['path']}")
    inventory = manifest["local_raw_output_inventory"]
    for root, key in ((RAW_ROOT, "production_output_root"), (CONTROL_ROOT, "production_control_root")):
        observed, expected = _inventory(root), inventory[key]
        for field in ("file_count", "total_bytes", "canonical_inventory_sha256"):
            if observed[field] != expected[field]:
                raise OfflineRecoveryStop(f"B1_ROOT_INVENTORY_MISMATCH:{key}:{field}")
    protected = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
    if not protected.is_file() or sha(protected) != manifest["protected_CSV_sha256"]:
        raise OfflineRecoveryStop("PROTECTED_CSV_SHA_MISMATCH")


def validate_pre_recovery_manifest(path: Path, expected_sha256: str) -> None:
    if not expected_sha256 or sha(path) != expected_sha256:
        raise OfflineRecoveryStop("PRE_RECOVERY_MANIFEST_SHA_MISMATCH")
    manifest = read(path)
    if manifest.get("self_reference") is not False or manifest.get("status") != "FROZEN_PRE_RECOVERY_NO_ANALYSIS_INVOKED":
        raise OfflineRecoveryStop("PRE_RECOVERY_MANIFEST_POLICY_MISMATCH")
    for row in manifest["components"]:
        candidate = ROOT / row["path"]
        if not candidate.is_file() or sha(candidate) != row["sha256"]:
            raise OfflineRecoveryStop(f"PRE_RECOVERY_COMPONENT_SHA_MISMATCH:{row['path']}")


def exact_runs() -> Sequence[Mapping[str, Any]]:
    rows = [row for row in read(SCHEDULE)["runs"] if int(row["run_order"]) in (1, 2)]
    if [row["run_order"] for row in rows] != [1, 2]:
        raise OfflineRecoveryStop("EXACT_RUN_ORDER_MISMATCH")
    return rows


def _disposition(result: Mapping[str, Any]) -> str:
    state = result.get("result_state")
    if state == RESULT_STATES[3]:
        return "CANARY_COMPLETE_READY_FOR_REMAINING_COHORT_OWNER_REVIEW"
    if state == RESULT_STATES[2]:
        return "CANARY_TECHNICAL_COMPLETE_MECHANISM_OR_ENDPOINT_FAIL"
    return "INFRASTRUCTURE_FAILURE_STOPPED"


def run_once(
    pre_manifest: Path,
    expected_pre_manifest_sha256: str,
    output_root: Path = OUTPUT_ROOT,
    analyzer: Callable[..., Mapping[str, Any]] = analyze_frozen_canary_pair,
) -> Mapping[str, Any]:
    validate_b1_artifact_closure()
    validate_pre_recovery_manifest(pre_manifest, expected_pre_manifest_sha256)
    authorization = read(AUTHORIZATION)
    if not authorization.get("OFFLINE_EXISTING_ARTIFACT_ANALYSIS_AUTHORIZED") or authorization.get("OFFLINE_ANALYZER_INVOCATION_BUDGET") != 1:
        raise OfflineRecoveryStop("OFFLINE_AUTHORIZATION_OR_BUDGET_MISMATCH")
    if authorization.get("NEW_RUN_BUDGET") != 0 or authorization.get("RUNNER_RUN_AUTHORIZED") is not False:
        raise OfflineRecoveryStop("NON_OFFLINE_AUTHORIZATION_PRESENT")
    if output_root.resolve() != Path(authorization["AUTHORIZED_OUTPUT_ROOT"]).resolve():
        raise OfflineRecoveryStop("OUTPUT_ROOT_NOT_AUTHORIZED")
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise OfflineRecoveryStop("OFFLINE_RECOVERY_OUTPUT_ALREADY_EXISTS") from error
    ledger_path = output_root / "offline_invocation_ledger.json"
    ledger = {
        "schema_version": "r2_bj_b1_1_offline_invocation_ledger_v1.0",
        "status": "CLAIMED_BEFORE_ANALYSIS",
        "authorization_sha256": sha(AUTHORIZATION),
        "pre_recovery_manifest_sha256": expected_pre_manifest_sha256,
        "offline_analyzer_invocations": 1,
        "initial_budget": 1,
        "remaining_budget": 0,
        "runner_run": 0,
    }
    atomic_json(ledger_path, ledger)
    try:
        recovered = analyzer(RAW_ROOT, exact_runs())
    except Exception as error:
        recovered = {
            "schema_version": "r2_bj_b1_1_offline_recovery_analyzer_result_v1.0",
            "result_state": RESULT_STATES[1],
            "technical_complete": False,
            "reason": f"{type(error).__name__}:{error}",
            "remaining_14_runs_automatically_authorized": False,
        }
    result = {
        "schema_version": "r2_bj_b1_1_offline_recovery_result_envelope_v1.0",
        "HISTORICAL_B1_STATE": "R2_BJ_B1_CANARY_INFRASTRUCTURE_FAILURE_STOPPED",
        "HISTORICAL_B1_RESULT_SUPERSEDED": False,
        "R2_BJ_B1_1_OFFLINE_RECOVERY": _disposition(recovered),
        "recovery_analyzer_result": recovered,
        "remaining_14_runs_authorized": False,
        "runner_run": 0,
    }
    result_path = output_root / "recovery_analyzer_result.json"
    atomic_json(result_path, result)
    ledger.update({"status": "COMPLETED", "result_sha256": sha(result_path)})
    atomic_json(ledger_path, ledger)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-recovery-manifest", type=Path, required=True)
    parser.add_argument("--pre-recovery-manifest-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "OFFLINE_RECOVERY_NOT_INVOKED", "remaining_budget": 1, "runner_run": 0}, indent=2))
        return 0
    result = run_once(args.pre_recovery_manifest, args.pre_recovery_manifest_sha256)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
