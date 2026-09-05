#!/usr/bin/env python3
"""B0.2 production adapter: install passive actual-LQR telemetry on B0.1's sole run path."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.r2_bj_b0_1_production_canary_launcher import (
    B01_COMPONENT,
    B01ControlPlaneStop,
    AttemptBudgetLedger,
    EXACT_RUN_IDS,
    PRODUCTION_CONTROL_ROOT,
    PRODUCTION_OUTPUT_ROOT,
    _run_one,
    build_production_runner,
    exact_slice,
    read,
    sha,
    validate_execution_component_closure,
    validate_production_control_plane,
    validate_real_technical_completion,
)
from tools.r2_bj_b0_2_passive_actual_lqr_recorder import PassiveActualLQRRecorderV1
from tools.r2_bj_b0_1_failure_persisting_telemetry_wrapper import atomic_json


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
B0_COMPONENT = R2 / "r2_bj_b0_component_sha_binding_manifest_v1.0.json"
B0_SCHEDULE = R2 / "r2_bj_b0_hlc_v4_pair_schedule_v1.0.json"
B0_BINDINGS = R2 / "r2_bj_b0_exact_pair_binding_manifest_v1.0.json"
B02_COMPONENT = R2 / "r2_bj_b0_2_execution_observability_sha_manifest_v1.0.json"
B02_CLOSED_AUTH = R2 / "r2_bj_b0_2_closed_authorization_gate_v1.0.json"


def validate_b02_component_closure(path: Path) -> None:
    manifest = read(path)
    if manifest.get("self_reference") is not False or manifest.get("owner_authorization_included") is not False:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_MANIFEST_REFERENCE_POLICY_INVALID")
    mismatches = []
    for row in manifest.get("components", []):
        candidate = ROOT / str(row["path"])
        if not candidate.is_file() or sha(candidate) != row["sha256"]:
            mismatches.append(str(row["path"]))
    for row in manifest.get("external_bound_nuplan_1_2_2_runtime_components", []):
        candidate = Path(str(row["absolute_path"]))
        if not candidate.is_file() or sha(candidate) != row["sha256"]:
            mismatches.append(str(candidate))
    if not manifest.get("components") or mismatches:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", f"B0_2_COMPONENT_CLOSURE_MISMATCH:{mismatches}")


def validate_b02_authorization(authorization: Mapping[str, Any], manifest_path: Path) -> None:
    expected = authorization.get("authorized", {})
    if expected.get("B0_component_manifest_sha256") != sha(B0_COMPONENT):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_OWNER_B0_COMPONENT_BINDING_MISMATCH")
    if expected.get("B0_schedule_sha256") != sha(B0_SCHEDULE):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_OWNER_SCHEDULE_BINDING_MISMATCH")
    if expected.get("B0_pair_binding_sha256") != sha(B0_BINDINGS):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_OWNER_PAIR_BINDING_MISMATCH")
    if expected.get("B0_1_execution_component_manifest_sha256") != sha(B01_COMPONENT):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_OWNER_B0_1_BINDING_MISMATCH")
    if authorization.get("AUTHORIZED_B0_1_EXECUTION_COMPONENT_MANIFEST_SHA256") != sha(B01_COMPONENT):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_1_OWNER_BINDING_MISSING_FROM_B0_2_AUTH")
    if authorization.get("AUTHORIZED_B0_2_EXECUTION_OBSERVABILITY_MANIFEST_SHA256") != sha(manifest_path):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_OWNER_OBSERVABILITY_BINDING_MISMATCH")
    validate_b02_component_closure(manifest_path)


def zero_run_observability_preflight(
    manifest_path: Path = B02_COMPONENT,
    runner_builder: Any = build_production_runner,
) -> Mapping[str, Any]:
    """Construct and instrument both frozen runners in temporary roots; never call run()."""
    gate = read(B02_CLOSED_AUTH)
    if gate.get("CANARY_AUTHORIZED") or gate.get("NEW_RUN_BUDGET") != 0:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_FORMAL_GATE_NOT_CLOSED")
    if PRODUCTION_OUTPUT_ROOT.exists() or PRODUCTION_CONTROL_ROOT.exists():
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_PRODUCTION_ROOT_EXISTS_DURING_ZERO_RUN")
    validate_execution_component_closure(B01_COMPONENT)
    validate_b02_component_closure(manifest_path)
    rows = []
    with tempfile.TemporaryDirectory(prefix="r2_bj_b0_2_zero_run_") as directory:
        root = Path(directory)
        for run in exact_slice():
            bundle = runner_builder(run, root / str(run["run_id"]))
            recorder = _install_recorder(
                bundle,
                run,
                {"AUTHORIZED_B0_2_EXECUTION_OBSERVABILITY_MANIFEST_SHA256": sha(manifest_path)},
            )
            simulation = bundle.runner._simulation
            rows.append({
                "run_id": run["run_id"],
                "run_order": run["run_order"],
                "arm": run["arm"],
                "time_controller_class": simulation._time_controller.__class__.__name__,
                "time_controller_iterations": int(simulation._time_controller.number_of_iterations()),
                "ego_controller_class": simulation._ego_controller.__class__.__name__,
                "tracker_class": simulation._ego_controller._tracker.__class__.__name__,
                "recorder_installed": bool(getattr(simulation._ego_controller._tracker, "_r2_bj_b0_2_passive_recorder_installed", False)),
                "runner_run_calls": 0,
            })
            recorder.uninstall()
    return {
        "schema_version": "r2_bj_b0_2_zero_run_observability_preflight_v1.0",
        "status": "FROZEN_PRE_OUTCOME_OBSERVABILITY_AND_ANALYSIS_READY_FOR_OWNER_REVIEW",
        "runs": rows,
        "runner_constructions": len(rows),
        "recorder_installations": sum(row["recorder_installed"] for row in rows),
        "actual_controller_expected_rows_per_future_run": 79,
        "planner_reference_steering_rows_per_future_run": 80,
        "production_output_root_exists": PRODUCTION_OUTPUT_ROOT.exists(),
        "production_control_root_exists": PRODUCTION_CONTROL_ROOT.exists(),
        "CANARY_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
        "RUNNER_RUN": 0,
    }


def _install_recorder(bundle: Any, run: Mapping[str, Any], authorization: Mapping[str, Any]) -> PassiveActualLQRRecorderV1:
    simulation = getattr(bundle.runner, "_simulation", None)
    if simulation is None:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_SIMULATION_OBJECT_MISSING")
    recorder = PassiveActualLQRRecorderV1(
        bundle.run_root / "telemetry/actual_lqr_controller_telemetry.jsonl",
        run,
        {
            "B0_component_manifest_sha256": sha(B0_COMPONENT),
            "B0_1_execution_component_manifest_sha256": sha(B01_COMPONENT),
            "B0_2_execution_observability_manifest_sha256": str(
                authorization["AUTHORIZED_B0_2_EXECUTION_OBSERVABILITY_MANIFEST_SHA256"]
            ),
        },
    )
    try:
        recorder.install(simulation._ego_controller, simulation._time_controller)
    except Exception as error:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_RECORDER_INSTALLABILITY_FAILURE") from error
    bundle.b02_actual_lqr_recorder = recorder
    return recorder


def validate_b02_technical_completion(bundle: Any, report: Any) -> None:
    # B0.1's architecture-first validator remains authoritative.
    validate_real_technical_completion(bundle, report)
    recorder = getattr(bundle, "b02_actual_lqr_recorder", None)
    if recorder is None:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_RECORDER_NOT_BOUND")
    try:
        recorder.validate_complete()
    except Exception as error:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_ACTUAL_LQR_TELEMETRY_INCOMPLETE") from error


def run_b02_production_canary(
    authorization: Mapping[str, Any],
    output_root: Path = PRODUCTION_OUTPUT_ROOT,
    control_root: Path = PRODUCTION_CONTROL_ROOT,
    requested_runs: Sequence[Mapping[str, Any]] | None = None,
    observability_manifest: Path = B02_COMPONENT,
    runner_builder: Any = build_production_runner,
    pair_analyzer: Any = None,
) -> Mapping[str, Any]:
    """Run exactly the frozen two-arm slice; closed authorization makes the committed state zero-run."""
    runs = list(exact_slice() if requested_runs is None else requested_runs)
    validate_b02_authorization(authorization, observability_manifest)
    validate_production_control_plane(authorization, runs, output_root, control_root, B01_COMPONENT)
    if output_root.exists() or control_root.exists():
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_PRODUCTION_ROOT_MUST_BE_ABSENT")

    authorization_sha = hashlib.sha256(
        json.dumps(authorization, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    ledger = AttemptBudgetLedger(control_root / "canary_attempt_ledger.json", authorization_sha, budget=2)
    completed: list[str] = []

    for index, run in enumerate(runs):
        if index == 1 and completed != [EXACT_RUN_IDS[0]]:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "TREATMENT_REQUIRES_BASELINE_TECHNICAL_COMPLETE")
        try:
            bundle = runner_builder(run, output_root / str(run["run_id"]))
            _install_recorder(bundle, run, authorization)
        except B01ControlPlaneStop:
            raise
        except Exception as error:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_RUNNER_OR_RECORDER_CONSTRUCTION_FAILURE") from error

        # Installability has now passed. Only now may the one-shot authorization be consumed.
        if index == 0:
            try:
                ledger.claim_authorization_once()
            except Exception as error:
                if isinstance(error, B01ControlPlaneStop):
                    raise
                raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_AUTHORIZATION_LEDGER_PERSISTENCE_FAILURE") from error
        _run_one(bundle, run, ledger, validate_b02_technical_completion)
        completed.append(str(run["run_id"]))

    if ledger.remaining != 0 or len(ledger.attempts) != 2:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_FINAL_BUDGET_OR_ATTEMPT_COUNT_NOT_EXACT")
    if pair_analyzer is None:
        from tools.r2_bj_b0_2_frozen_canary_pair_analyzer import analyze_frozen_canary_pair
        pair_analyzer = analyze_frozen_canary_pair
    analysis = pair_analyzer(output_root, runs)
    try:
        atomic_json(control_root / "canary_pair_analysis.json", analysis)
    except Exception as error:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_2_PAIR_ANALYSIS_PERSISTENCE_FAILURE_STOP_ALL") from error
    return {
        "status": analysis["result_state"],
        "completed_run_ids": completed,
        "runner_run_attempt_count": 2,
        "remaining_budget": 0,
        "analysis": analysis,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=B02_CLOSED_AUTH)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--zero-run-preflight", action="store_true")
    args = parser.parse_args()
    gate = read(args.authorization)
    if args.zero_run_preflight:
        print(json.dumps(zero_run_observability_preflight(), indent=2))
        return 0
    if not args.execute:
        print(json.dumps({
            "status": "R2_BJ_B0_2_ZERO_RUN_CLOSED",
            "CANARY_AUTHORIZED": gate["CANARY_AUTHORIZED"],
            "NEW_RUN_BUDGET": gate["NEW_RUN_BUDGET"],
            "RUNNER_RUN": 0,
        }, indent=2))
        return 0
    print(json.dumps(run_b02_production_canary(gate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
