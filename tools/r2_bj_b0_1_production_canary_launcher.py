#!/usr/bin/env python3
"""Production BJ-B canary control flow; formally closed until a future owner record exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r2_bj_b0_1_failure_persisting_telemetry_wrapper import (  # noqa: E402
    R2BJB01FailurePersistingTelemetryWrapper,
    atomic_json,
)
from tools.r2_bj_b0_hlc_v4_engineering_planner import B0ArchitectureViolation  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
B0_COMPONENT = R2 / "r2_bj_b0_component_sha_binding_manifest_v1.0.json"
B0_SCHEDULE = R2 / "r2_bj_b0_hlc_v4_pair_schedule_v1.0.json"
B0_BINDINGS = R2 / "r2_bj_b0_exact_pair_binding_manifest_v1.0.json"
B01_COMPONENT = R2 / "r2_bj_b0_1_execution_component_sha_manifest_v1.0.json"
B01_SLICE = R2 / "r2_bj_b0_1_exact_two_run_canary_slice_v1.0.json"
B01_CLOSED_AUTH = R2 / "r2_bj_b0_1_closed_authorization_gate_v1.0.json"
PRODUCTION_OUTPUT_ROOT = ROOT / "outputs/r2_bj_b0_1_canary_once_v1"
PRODUCTION_CONTROL_ROOT = ROOT / "outputs/r2_bj_b0_1_canary_once_control_v1"
ROSTER = R2 / "r2_bj_b0_hlc_v4_engineering_roster_v1.0.json"
PARAMETERS = R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"

EXPECTED = {
    B0_COMPONENT: "35a1282328b461f0b1edbbd39a4284870382ad52a83bd2975d9a91bc0ece1cf9",
    B0_SCHEDULE: "5493c5b402a3bc954d83d0914451c1f3dd38cddcfad8244291cf0a846d88918d",
    B0_BINDINGS: "4e4eee55b816c8fa79cdc41ed0f8f99d9bd778e14c747ee13704998f71366950",
    ROOT / "tools/r2_bj_b0_execute_frozen_hlc_v4_engineering.py": "7af447896c0be9f32f78faae4cc1e6b601a63331097deb740d51e3fb49b581ad",
    ROOT / "tools/r2_bj_b0_hlc_v4_engineering_planner.py": "3b837c00bfc55453237437cabda75fa58e5be948fa1f6276963387685e24d3b6",
    PROTECTED: "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8",
}
EXACT_RUN_IDS = ("R2BJB0-HLC-01-BASELINE", "R2BJB0-HLC-01-TREATMENT")
EXACT_RUN_ORDERS = (1, 2)
SLICE_IDENTITY_FIELDS = (
    "run_order", "run_id", "pair_id", "family", "arm", "scenario_token", "log_id",
)


class B01ControlPlaneStop(RuntimeError):
    """A fail-closed production stop; never converted into a retry."""

    def __init__(self, classification: str, reason: str):
        self.classification = classification
        self.reason = reason
        super().__init__(f"R2_BJ_B0_1_{classification}:{reason}")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_execution_component_closure(manifest_path: Path) -> None:
    """Verify every production dependency declared by the non-self-referential manifest."""
    manifest = read(manifest_path)
    if manifest.get("self_reference") is not False or manifest.get("owner_authorization_included") is not False:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_1_MANIFEST_REFERENCE_POLICY_INVALID")
    mismatches = []
    for row in manifest.get("components", []):
        path = ROOT / str(row["path"])
        if not path.is_file() or sha(path) != row["sha256"]:
            mismatches.append(str(row["path"]))
    for row in manifest.get("external_bound_nuplan_1_2_2_runtime_components", []):
        path = Path(str(row["absolute_path"]))
        if not path.is_file() or sha(path) != row["sha256"]:
            mismatches.append(str(path))
    if not manifest.get("components") or mismatches:
        raise B01ControlPlaneStop(
            "INFRASTRUCTURE_FAILURE", f"B0_1_EXECUTION_COMPONENT_CLOSURE_MISMATCH:{mismatches}",
        )


def exact_slice() -> list[Mapping[str, Any]]:
    runs = read(B0_SCHEDULE)["runs"]
    result = [row for row in runs if int(row["run_order"]) in EXACT_RUN_ORDERS]
    if tuple(row["run_id"] for row in result) != EXACT_RUN_IDS:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "FROZEN_SCHEDULE_SLICE_MISMATCH")
    return result


def validate_production_control_plane(
    authorization: Mapping[str, Any], requested_runs: Sequence[Mapping[str, Any]],
    output_root: Path, control_root: Path,
    execution_component_manifest: Path = B01_COMPONENT,
) -> None:
    """Mandatory dominance checks performed before runner construction or simulator start."""
    mismatches = [
        str(path.relative_to(ROOT))
        for path, expected in EXPECTED.items()
        if not path.is_file() or sha(path) != expected
    ]
    if mismatches:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", f"FROZEN_INPUT_SHA_MISMATCH:{mismatches}")
    if not execution_component_manifest.is_file():
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_1_EXECUTION_COMPONENT_MANIFEST_MISSING")
    expected_component = authorization.get("AUTHORIZED_B0_1_EXECUTION_COMPONENT_MANIFEST_SHA256")
    if sha(execution_component_manifest) != expected_component:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_1_EXECUTION_COMPONENT_MANIFEST_SHA_MISMATCH")
    validate_execution_component_closure(execution_component_manifest)
    slice_manifest = read(B01_SLICE)
    if (
        tuple(slice_manifest.get("run_orders", [])) != EXACT_RUN_ORDERS
        or tuple(slice_manifest.get("run_ids", [])) != EXACT_RUN_IDS
        or slice_manifest.get("source_schedule_sha256") != sha(B0_SCHEDULE)
    ):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "B0_1_EXACT_SLICE_MANIFEST_MISMATCH")
    if not authorization.get("BJ_B_ENGINEERING_SIMULATION_AUTHORIZED", False) or not authorization.get("CANARY_AUTHORIZED", False):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "CANARY_NOT_AUTHORIZED")
    if authorization.get("AUTHORIZATION_CONSUMED", False):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZATION_ALREADY_CONSUMED")
    if int(authorization.get("NEW_RUN_BUDGET", 0)) != 2:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZED_BUDGET_NOT_EXACTLY_TWO")
    if tuple(authorization.get("AUTHORIZED_RUN_ORDERS", [])) != EXACT_RUN_ORDERS:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZED_RUN_ORDERS_NOT_EXACT_1_2")
    if tuple(authorization.get("AUTHORIZED_RUN_IDS", [])) != EXACT_RUN_IDS:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZED_RUN_IDS_MISMATCH")
    if Path(str(authorization.get("AUTHORIZED_OUTPUT_ROOT", ""))).expanduser().resolve() != output_root.expanduser().resolve():
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZED_OUTPUT_ROOT_MISMATCH")
    if Path(str(authorization.get("AUTHORIZED_CONTROL_ROOT", ""))).expanduser().resolve() != control_root.expanduser().resolve():
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZED_CONTROL_ROOT_MISMATCH")
    if tuple(int(row["run_order"]) for row in requested_runs) != EXACT_RUN_ORDERS:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "REQUESTED_SCHEDULE_NOT_EXACT_1_2")
    if tuple(str(row["run_id"]) for row in requested_runs) != EXACT_RUN_IDS:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "REQUESTED_RUN_IDS_MISMATCH")
    frozen = exact_slice()
    requested_identity = [
        {field: row[field] for field in SLICE_IDENTITY_FIELDS} for row in requested_runs
    ]
    frozen_identity = [
        {field: row[field] for field in SLICE_IDENTITY_FIELDS} for row in frozen
    ]
    if requested_identity != frozen_identity:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "REQUESTED_SCHEDULE_ROW_BINDING_MISMATCH")
    if any((output_root / str(row["run_id"])).exists() for row in requested_runs):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "OUTPUT_ROOT_COLLISION")
    bindings = authorization.get("authorized", {})
    for key, path in (("B0_component_manifest_sha256", B0_COMPONENT), ("B0_schedule_sha256", B0_SCHEDULE), ("B0_pair_binding_sha256", B0_BINDINGS)):
        if bindings.get(key) != sha(path):
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", f"OWNER_BINDING_MISMATCH:{key}")


class AttemptBudgetLedger:
    """Atomically persisted one-shot budget; claim always precedes runner.run()."""

    def __init__(self, path: Path, authorization_sha256: str, budget: int = 2) -> None:
        self.path = path
        self.authorization_sha256 = authorization_sha256
        self.remaining = int(budget)
        self.attempts: list[dict[str, Any]] = []

    def claim_authorization_once(self) -> None:
        if self.path.exists():
            previous = read(self.path)
            if previous.get("AUTHORIZATION_CONSUMED", False):
                raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "AUTHORIZATION_ALREADY_CONSUMED")
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "ATTEMPT_LEDGER_ALREADY_EXISTS")
        self._persist(True)

    def claim_run(self, run: Mapping[str, Any]) -> None:
        if self.remaining <= 0:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "RUN_BUDGET_EXHAUSTED")
        self.remaining -= 1
        if self.remaining < 0:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "NEGATIVE_BUDGET_FORBIDDEN")
        self.attempts.append({
            "attempt_number": len(self.attempts) + 1,
            "run_order": int(run["run_order"]), "run_id": str(run["run_id"]),
            "pair_id": str(run["pair_id"]), "arm": str(run["arm"]),
            "status": "ATTEMPT_CLAIMED_BEFORE_RUNNER_RUN",
            "budget_remaining_after_claim": self.remaining,
        })
        self._persist(True)

    def finish(self, status: str, reason: str | None = None) -> None:
        self.attempts[-1]["status"] = status
        self.attempts[-1]["reason"] = reason
        self._persist(True)

    def _persist(self, consumed: bool) -> None:
        atomic_json(self.path, {
            "schema_version": "r2_bj_b0_1_execution_attempt_ledger_v1.0",
            "authorization_record_sha256": self.authorization_sha256,
            "AUTHORIZATION_CONSUMED": consumed,
            "initial_budget": 2, "remaining_budget": self.remaining,
            "runner_run_attempt_count": len(self.attempts), "attempts": self.attempts,
            "retry_authorized": False, "replacement_authorized": False,
        })


def _finish_attempt_or_fail_closed(
    ledger: AttemptBudgetLedger, status: str, reason: str | None,
    primary_classification: str,
) -> None:
    try:
        ledger.finish(status, reason)
    except Exception as persistence_error:
        raise B01ControlPlaneStop(
            primary_classification,
            f"ATTEMPT_LEDGER_PERSISTENCE_FAILURE_AFTER_{status}",
        ) from persistence_error


@dataclass
class ProductionRunnerBundle:
    runner: Any
    run_root: Path
    common_builder: Any = None
    cfg: Any = None


def _run_one(
    bundle: ProductionRunnerBundle, run: Mapping[str, Any], ledger: AttemptBudgetLedger,
    completion_validator: Callable[[ProductionRunnerBundle, Any], None],
) -> None:
    try:
        ledger.claim_run(run)
    except B01ControlPlaneStop:
        raise
    except Exception as persistence_error:
        raise B01ControlPlaneStop(
            "INFRASTRUCTURE_FAILURE", "RUN_BUDGET_CLAIM_PERSISTENCE_FAILURE_BEFORE_RUNNER_RUN",
        ) from persistence_error
    try:
        # This is the only production runner.run call site. All checks and budget claim dominate it.
        report = bundle.runner.run()
        if bundle.common_builder is not None:
            from nuplan.planning.script.utils import save_runner_reports
            save_runner_reports([report], bundle.common_builder.output_dir, bundle.cfg.runner_report_file)
            bundle.common_builder.multi_main_callback.on_run_simulation_end()
        completion_validator(bundle, report)
    except B0ArchitectureViolation as error:
        _finish_attempt_or_fail_closed(
            ledger, "ARCHITECTURE_FAILURE_STOP_ALL", ",".join(error.codes), "ARCHITECTURE_FAILURE",
        )
        raise B01ControlPlaneStop("ARCHITECTURE_FAILURE", "STOP_CURRENT_AND_REMAINING") from error
    except B01ControlPlaneStop as error:
        status = "ARCHITECTURE_FAILURE_STOP_ALL" if error.classification == "ARCHITECTURE_FAILURE" else "INFRASTRUCTURE_FAILURE_STOP_ALL"
        _finish_attempt_or_fail_closed(ledger, status, error.reason, error.classification)
        raise
    except Exception as error:
        # Preserve the primary architecture classification even if nuPlan or
        # a post-run lifecycle operation surfaces a secondary exception after
        # the wrapper has already persisted the authoritative audit.
        if (bundle.run_root / "telemetry/architecture_failure_audit.json").exists():
            _finish_attempt_or_fail_closed(
                ledger, "ARCHITECTURE_FAILURE_STOP_ALL",
                "PERSISTED_ARCHITECTURE_FAILURE_PRESENT", "ARCHITECTURE_FAILURE",
            )
            raise B01ControlPlaneStop(
                "ARCHITECTURE_FAILURE", "PERSISTED_ARCHITECTURE_FAILURE_PRESENT_STOP_ALL",
            ) from error
        _finish_attempt_or_fail_closed(
            ledger, "INFRASTRUCTURE_FAILURE_STOP_ALL",
            f"{type(error).__name__}:{error}", "INFRASTRUCTURE_FAILURE",
        )
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "RUNNER_OR_LIFECYCLE_EXCEPTION_STOP_ALL") from error
    _finish_attempt_or_fail_closed(
        ledger, "TECHNICAL_COMPLETE", None, "INFRASTRUCTURE_FAILURE",
    )


def run_production_canary(
    authorization: Mapping[str, Any], output_root: Path, control_root: Path,
    runner_factory: Callable[[Mapping[str, Any], Path], ProductionRunnerBundle],
    completion_validator: Callable[[ProductionRunnerBundle, Any], None],
    requested_runs: Sequence[Mapping[str, Any]] | None = None,
    execution_component_manifest: Path = B01_COMPONENT,
    authorization_record_sha256: str | None = None,
) -> Mapping[str, Any]:
    try:
        runs = list(exact_slice() if requested_runs is None else requested_runs)
        validate_production_control_plane(
            authorization, runs, output_root, control_root, execution_component_manifest,
        )
    except B01ControlPlaneStop:
        raise
    except Exception as validation_error:
        raise B01ControlPlaneStop(
            "INFRASTRUCTURE_FAILURE", "CONTROL_PLANE_VALIDATION_EXCEPTION_BEFORE_RUNNER_CONSTRUCTION",
        ) from validation_error
    canonical_authorization_sha256 = hashlib.sha256(
        json.dumps(authorization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ledger = AttemptBudgetLedger(
        control_root / "canary_attempt_ledger.json",
        authorization_record_sha256 or canonical_authorization_sha256,
        budget=2,
    )
    try:
        ledger.claim_authorization_once()
    except B01ControlPlaneStop:
        raise
    except Exception as persistence_error:
        raise B01ControlPlaneStop(
            "INFRASTRUCTURE_FAILURE", "AUTHORIZATION_LEDGER_PERSISTENCE_FAILURE_BEFORE_RUNNER_CONSTRUCTION",
        ) from persistence_error
    completed = []
    for index, run in enumerate(runs):
        if index == 1 and completed != [EXACT_RUN_IDS[0]]:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "TREATMENT_REQUIRES_BASELINE_TECHNICAL_COMPLETE")
        run_root = output_root / str(run["run_id"])
        try:
            bundle = runner_factory(run, run_root)
        except B01ControlPlaneStop:
            raise
        except Exception as construction_error:
            raise B01ControlPlaneStop(
                "INFRASTRUCTURE_FAILURE", "RUNNER_CONSTRUCTION_EXCEPTION_STOP_ALL",
            ) from construction_error
        _run_one(bundle, run, ledger, completion_validator)
        completed.append(str(run["run_id"]))
    if ledger.remaining != 0 or len(ledger.attempts) != 2:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "FINAL_BUDGET_OR_ATTEMPT_COUNT_NOT_EXACT")
    return {"status": "CANARY_TECHNICAL_COMPLETE", "completed_run_ids": completed, "runner_run_attempt_count": 2, "remaining_budget": 0}


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    from tools.r2_bj_b0_execute_frozen_hlc_v4_engineering import _overrides as b0_overrides
    return b0_overrides(run, entry, raw)


def build_production_runner(run: Mapping[str, Any], run_root: Path) -> ProductionRunnerBundle:
    """Construct one real runner only after the production control-plane authorizes it."""
    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder

    roster = read(ROSTER)
    entry = next(row for row in roster["entries"] if row["scenario_token"] == run["scenario_token"])
    if run_root.exists():
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "OUTPUT_ROOT_COLLISION")
    if official_count(entry["db_path"], run["scenario_token"]) != 1:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "EXACT_SCENARIO_RESOLUTION_NOT_ONE")
    trace, telemetry, raw = run_root / "trace", run_root / "telemetry", run_root / "raw"
    trace.mkdir(parents=True)
    planner = R2BJB01FailurePersistingTelemetryWrapper(
        entry, run["arm"], read(PARAMETERS)["global_parameters"], str(trace), str(telemetry),
        run["run_id"], run["pair_id"], sha(B0_COMPONENT), sha(B0_SCHEDULE), sha(B0_BINDINGS),
    )
    os.environ.update({"R2_BJ_B0_RUN_ID": run["run_id"], "R2_BJ_B0_TRACE_DIR": str(trace), "R2_BJ_B0_TELEMETRY_DIR": str(telemetry)})
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with initialize_config_dir(config_dir=str(config_root)):
        cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
    if "${" in json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "UNRESOLVED_HYDRA")
    common = set_up_common_builder(cfg, "r2_bj_b0_1_production_canary_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "RUNNER_COUNT_NOT_ONE")
    return ProductionRunnerBundle(runners[0], run_root, common, cfg)


def validate_real_technical_completion(bundle: ProductionRunnerBundle, report: Any) -> None:
    # nuPlan may return a failed RunnerReport after catching a planner
    # exception.  The wrapper's atomically persisted record is therefore the
    # authoritative discriminator and must be checked before report status so
    # an architecture failure can never be downgraded to infrastructure.
    if (bundle.run_root / "telemetry/architecture_failure_audit.json").exists():
        raise B01ControlPlaneStop("ARCHITECTURE_FAILURE", "PERSISTED_ARCHITECTURE_FAILURE_PRESENT")
    if not bool(getattr(report, "succeeded", False)):
        raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", "RUNNER_REPORT_NOT_SUCCEEDED")
    expected_jsonl = {
        "realized_trace": bundle.run_root / "trace/realized_current_ego.jsonl",
        "planner_telemetry": bundle.run_root / "telemetry/planner_v4_online_gate.jsonl",
        "controller_visible_telemetry": bundle.run_root / "telemetry/controller_visible_telemetry.jsonl",
    }
    for name, path in expected_jsonl.items():
        if not path.is_file() or len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]) != 80:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", f"PRIMARY80_{name.upper()}_INCOMPLETE")
    for filename in ("no_ego_at_fault_collisions.parquet", "drivable_area_compliance.parquet", str(bundle.cfg.runner_report_file)):
        matches = list(bundle.run_root.rglob(filename))
        if len(matches) != 1:
            raise B01ControlPlaneStop("INFRASTRUCTURE_FAILURE", f"EXPECTED_ARTIFACT_NOT_EXACTLY_ONE:{filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=B01_CLOSED_AUTH)
    parser.add_argument("--output-root", type=Path, default=PRODUCTION_OUTPUT_ROOT)
    parser.add_argument("--control-root", type=Path, default=PRODUCTION_CONTROL_ROOT)
    parser.add_argument("--execute", action="store_true", help="Enter production path; still requires a separately valid Owner authorization record.")
    args = parser.parse_args()
    if not args.execute:
        gate = read(args.authorization)
        print(json.dumps({"status": "ZERO_RUN_CLOSED", "CANARY_AUTHORIZED": gate["CANARY_AUTHORIZED"], "NEW_RUN_BUDGET": gate["NEW_RUN_BUDGET"], "RUNNER_RUN": 0}, indent=2))
        return 0
    result = run_production_canary(
        read(args.authorization), args.output_root, args.control_root,
        build_production_runner, validate_real_technical_completion,
        authorization_record_sha256=sha(args.authorization),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
