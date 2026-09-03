#!/usr/bin/env python3
"""Construct or execute the frozen R2-A controller-transfer DEV design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r2_a_controller_transfer_dev_planner_v1 import R2AControllerTransferDevPlannerV1  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_a_controller_id_dev_canary_roster_v1.0.json"
HLC_GRID = R2 / "r2_a_hlc_excitation_grid_v1.0.json"
TSB_GRID = R2 / "r2_a_tsb_excitation_grid_v1.0.json"
LEDGER = R2 / "r2_a_controller_transfer_run_ledger_v1.0.json"
AUTHORIZATION = R2 / "r2_a_scientific_owner_engineering_simulation_authorization_v1.0.json"
ROSTER_SHA = "aca3b1138a189333dc690edaa6399f5c3eeb32029e9c5971210e07651a951bee"
HLC_GRID_SHA = "1e209aac60f1afddef99304c7f49f12107e5695910e37b72ffc55fa1375bd2d8"
TSB_GRID_SHA = "24b41a54b5e077d10407d0ea2f2d4257693317eccab192b0f221e097367a2969"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
STATIC_RUN_KEYS = (
    "run_order",
    "run_id",
    "family",
    "scenario_token",
    "log_id",
    "excitation_id",
    "attempt",
    "technical_rerun_of",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _frozen_plan_projection(runs: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    projected = []
    for run in runs:
        row = {key: run[key] for key in STATIC_RUN_KEYS}
        row["status"] = "PLANNED_FROZEN_PRE_EXECUTION"
        projected.append(row)
    return projected


def _load_frozen() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if sha256(ROSTER) != ROSTER_SHA or sha256(HLC_GRID) != HLC_GRID_SHA or sha256(TSB_GRID) != TSB_GRID_SHA:
        raise PermissionError("R2_A_FROZEN_ROSTER_OR_GRID_SHA_MISMATCH")
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise PermissionError("PROTECTED_CSV_SHA_MISMATCH")
    roster, hlc, tsb, ledger = read_json(ROSTER), read_json(HLC_GRID), read_json(TSB_GRID), read_json(LEDGER)
    entries = {str(row["scenario_token"]): row for row in roster["entries"]}
    excitations = {
        str(row["excitation_id"]): row
        for row in list(hlc["excitations"]) + list(tsb["excitations"])
    }
    runs = sorted(ledger["runs"], key=lambda row: int(row["run_order"]))
    if len(entries) != 16 or len(runs) != 80 or [row["run_order"] for row in runs] != list(range(1, 81)):
        raise ValueError("R2_A_FROZEN_CARDINALITY_OR_RUN_ORDER_MISMATCH")
    if canonical_sha(_frozen_plan_projection(runs)) != ledger["frozen_run_plan_canonical_sha256"]:
        raise PermissionError("R2_A_FROZEN_RUN_PLAN_SHA_MISMATCH")
    if any(str(run["scenario_token"]) not in entries or str(run["excitation_id"]) not in excitations for run in runs):
        raise ValueError("R2_A_FROZEN_RUN_LOOKUP_MISMATCH")
    return ledger, entries, excitations, roster


def build_planner_from_frozen_binding(run_id: str, trace_dir: str, telemetry_dir: str) -> R2AControllerTransferDevPlannerV1:
    ledger, entries, excitations, _ = _load_frozen()
    matches = [row for row in ledger["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"R2_A_RUN_ID_MATCH_COUNT_NOT_ONE:{run_id}:{len(matches)}")
    run = matches[0]
    entry = entries[str(run["scenario_token"])]
    excitation = excitations[str(run["excitation_id"])]
    if entry["family"] != run["family"]:
        raise ValueError("R2_A_RUN_ROSTER_FAMILY_MISMATCH")
    return R2AControllerTransferDevPlannerV1(entry, excitation, trace_dir, telemetry_dir)


class ControllerCommandRecorderV1:
    """Passive wrapper around the exact bound LQR tracker return value."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.rows = 0

    def install(self, controller: Any) -> None:
        tracker = getattr(controller, "_tracker", None)
        if tracker is None or tracker.__class__.__name__ != "LQRTracker":
            raise TypeError(f"R2_A_EXPECTED_LQR_TRACKER:{type(tracker).__name__}")
        original = tracker.track_trajectory
        recorder = self

        def wrapped(_tracker_self: Any, current_iteration: Any, next_iteration: Any, initial_state: Any, trajectory: Any) -> Any:
            result = original(current_iteration, next_iteration, initial_state, trajectory)
            reference_time_us = int(current_iteration.time_point.time_us + 1_000_000)
            reference = trajectory.get_state_at_time(type(current_iteration.time_point)(reference_time_us))
            row = {
                "schema_version": "r2_a_two_stage_lqr_control_output_v1.0",
                "instrumentation": "PASSIVE_RETURN_VALUE_WRAPPER_NO_BEHAVIOR_CHANGE",
                "iteration": int(current_iteration.index),
                "current_time_us": int(current_iteration.time_point.time_us),
                "next_time_us": int(next_iteration.time_point.time_us),
                "realized_current_speed_mps": float(initial_state.dynamic_car_state.rear_axle_velocity_2d.x),
                "lqr_reference_speed_at_1s_mps": float(reference.dynamic_car_state.rear_axle_velocity_2d.x),
                "acceleration_command_mps2": float(result.rear_axle_acceleration_2d.x),
                "steering_rate_command_radps": float(result.tire_steering_rate),
                "controller": controller.__class__.__name__,
                "tracker": tracker.__class__.__name__,
                "motion_model": getattr(controller, "_motion_model").__class__.__name__,
            }
            recorder.output_path.parent.mkdir(parents=True, exist_ok=True)
            with recorder.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            recorder.rows += 1
            return result

        tracker.track_trajectory = types.MethodType(wrapped, tracker)


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r2_a_controller_transfer_dev_v1",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential",
        "disable_callback_parallelization=true",
        "scenario_builder.max_workers=1",
        "max_callback_workers=1",
        "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0",
        "gpu=false",
        "seed=2026082701",
        "run_metric=false",
        "enable_simulation_progress_bar=false",
        "experiment_name=r2_a_controller_transfer_dev",
        f"job_name={run['run_id']}",
        f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _construct(
    run: Mapping[str, Any],
    entry: Mapping[str, Any],
    excitation: Mapping[str, Any],
    root: Path,
    execute: bool,
) -> Dict[str, Any]:
    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import run_runners, set_up_common_builder
    from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1

    run_root = root / str(run["run_id"])
    trace_dir = run_root / "trace"
    telemetry_dir = run_root / "telemetry"
    raw = run_root / "raw"
    if run_root.exists():
        raise FileExistsError(f"R2_A_FRESH_RUN_ROOT_REQUIRED:{run_root}")
    if official_count(str(entry["db_path"]), str(run["scenario_token"])) != 1:
        raise RuntimeError(f"R2_A_EXACT_SCENARIO_RESOLUTION_NOT_ONE:{run['run_id']}")
    os.environ.update(
        {
            "R2_A_RUN_ID": str(run["run_id"]),
            "R2_A_TRACE_DIR": str(trace_dir),
            "R2_A_TELEMETRY_DIR": str(telemetry_dir),
        }
    )
    planner = R2AControllerTransferDevPlannerV1(entry, excitation, str(trace_dir), str(telemetry_dir))
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with initialize_config_dir(config_dir=str(config_root)):
        cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
    resolved = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if "${" in resolved:
        raise RuntimeError(f"R2_A_UNRESOLVED_HYDRA:{run['run_id']}")
    common = set_up_common_builder(cfg, "r2_a_controller_transfer_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise RuntimeError(f"R2_A_RUNNER_COUNT_NOT_ONE:{run['run_id']}:{len(runners)}")
    controller = runners[0]._simulation._ego_controller
    time_controller = runners[0]._simulation._time_controller
    if time_controller.__class__ is not R1Primary80ScientificTimeControllerV1 or int(time_controller.number_of_iterations()) != 81:
        raise RuntimeError("R2_A_PRIMARY80_CONTROLLER_BINDING_FAIL")
    if controller.__class__.__name__ != "TwoStageController":
        raise RuntimeError(f"R2_A_TWO_STAGE_CONTROLLER_BINDING_FAIL:{controller.__class__.__name__}")
    recorder = ControllerCommandRecorderV1(telemetry_dir / "controller_commands.jsonl")
    recorder.install(controller)
    result: Dict[str, Any] = {
        "run_id": run["run_id"],
        "family": run["family"],
        "scenario_token": run["scenario_token"],
        "log_id": run["log_id"],
        "excitation_id": run["excitation_id"],
        "exact_scenario_resolution": 1,
        "full_hydra_resolved": True,
        "planner_class": planner.__class__.__name__,
        "time_controller_class": time_controller.__class__.__name__,
        "controller_iterations": int(time_controller.number_of_iterations()),
        "ego_controller_class": controller.__class__.__name__,
        "tracker_class": controller._tracker.__class__.__name__,
        "motion_model_class": controller._motion_model.__class__.__name__,
        "runner_count": 1,
        "runner_constructed": True,
        "simulation_executed": execute,
    }
    if not execute:
        return result
    run_runners(runners, common, "r2_a_controller_transfer_running", cfg)
    trace_path = trace_dir / "realized_current_ego.jsonl"
    planner_path = telemetry_dir / "planner_transfer.jsonl"
    controller_path = telemetry_dir / "controller_commands.jsonl"
    counts = {}
    for label, path in (("realized_trace", trace_path), ("planner_telemetry", planner_path), ("controller_commands", controller_path)):
        if not path.is_file():
            raise RuntimeError(f"R2_A_TELEMETRY_FILE_MISSING:{run['run_id']}:{label}")
        counts[label] = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    # Primary80 has 80 planner-call entry observations (iterations 0...79),
    # but only 79 transitions are propagated; the final trajectory has no
    # subsequent controller update by construction.
    if counts != {"realized_trace": 80, "planner_telemetry": 80, "controller_commands": 79}:
        raise RuntimeError(f"R2_A_PRIMARY80_TELEMETRY_COUNT_FAIL:{run['run_id']}:{counts}")
    result.update(
        {
            "status": "TECHNICAL_COMPLETE",
            "run_root": str(run_root.relative_to(ROOT)),
            "trace_path": str(trace_path.relative_to(ROOT)),
            "planner_telemetry_path": str(planner_path.relative_to(ROOT)),
            "controller_command_path": str(controller_path.relative_to(ROOT)),
            "telemetry_counts": counts,
            "controller_output_availability": "AVAILABLE_PASSIVE_LQR_RETURN_VALUE_WRAPPER",
            "runner_run_calls": 1,
            "run_runners_calls": 1,
        }
    )
    return result


def _authorization_check() -> None:
    authorization = read_json(AUTHORIZATION)
    if authorization.get("R2_DEV_ENGINEERING_SIMULATION_AUTHORIZED") is not True:
        raise PermissionError("R2_A_ENGINEERING_SIMULATION_NOT_AUTHORIZED")
    if authorization.get("authorization_scope") != "FRESH_R2_A_DEVELOPMENT_IDENTITIES_ONLY":
        raise PermissionError("R2_A_ENGINEERING_AUTHORIZATION_SCOPE_MISMATCH")


def run(*, execute: bool, output_root: Optional[Path], resume_technical_failure: bool = False) -> Dict[str, Any]:
    ledger, entries, excitations, _ = _load_frozen()
    runs = sorted(ledger["runs"], key=lambda row: int(row["run_order"]))
    if execute:
        _authorization_check()
        if output_root is None:
            raise ValueError("R2_A_EXECUTE_REQUIRES_EXPLICIT_FRESH_OUTPUT_ROOT")
        if output_root.exists():
            raise FileExistsError(f"R2_A_OUTPUT_ROOT_REUSE_FORBIDDEN:{output_root}")
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True)
        root = output_root
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="r2_a_zero_run_")
        root = Path(temporary.name)
    audits = []
    try:
        if execute:
            if resume_technical_failure:
                if ledger.get("status") == "AUTHORIZED_ENGINEERING_EXECUTION_RUNNING":
                    # Recovery attempt 1 completed nuPlan and exact telemetry,
                    # then failed only while relativizing a relative root.
                    prior_root = ROOT / "outputs/r2_a_controller_transfer_dev_v1_attempt2/R2A-HLC-01-HLC_MONOTONIC_REFERENCE-TECHRERUN01"
                    prior_counts = {
                        "realized_trace": sum(bool(line.strip()) for line in (prior_root / "trace/realized_current_ego.jsonl").read_text(encoding="utf-8").splitlines()),
                        "planner_telemetry": sum(bool(line.strip()) for line in (prior_root / "telemetry/planner_transfer.jsonl").read_text(encoding="utf-8").splitlines()),
                        "controller_commands": sum(bool(line.strip()) for line in (prior_root / "telemetry/controller_commands.jsonl").read_text(encoding="utf-8").splitlines()),
                    }
                    if prior_counts != {"realized_trace": 80, "planner_telemetry": 80, "controller_commands": 79}:
                        raise PermissionError("R2_A_PRIOR_BOOKKEEPING_FAILURE_TELEMETRY_NOT_COMPLETE")
                    ledger.setdefault("technical_reruns", []).append(
                        {
                            "run_id": "R2A-HLC-01-HLC_MONOTONIC_REFERENCE-TECHRERUN01",
                            "technical_rerun_of": "R2A-HLC-01-HLC_MONOTONIC_REFERENCE",
                            "attempt": 2,
                            "status": "TECHNICAL_BOOKKEEPING_FAILURE_AFTER_SUCCESSFUL_NUPLAN_RUN",
                            "technical_failure": "ValueError:RELATIVE_OUTPUT_ROOT_COULD_NOT_BE_RELATIVIZED",
                            "run_root": str(prior_root.relative_to(ROOT)),
                            "telemetry_counts": prior_counts,
                            "excluded_from_transfer_analysis": True,
                        }
                    )
                    ledger["status"] = "TECHNICAL_FAILURE_STOPPED_REMAINING_SCHEDULE"
                    ledger["counts"].update({"executed": 0, "technical_reruns": 1, "actual_engineering_runs": 2})
                    write_json(LEDGER, ledger)
                if ledger.get("status") != "TECHNICAL_FAILURE_STOPPED_REMAINING_SCHEDULE":
                    raise PermissionError("R2_A_NO_TECHNICAL_FAILURE_AVAILABLE_FOR_RESUME")
                failed = [row for row in runs if row.get("status") == "TECHNICAL_FAILURE"]
                if len(failed) != 1 or failed[0]["run_order"] != 1:
                    raise PermissionError("R2_A_TECHNICAL_RECOVERY_EXPECTED_ONLY_FIRST_RUN_FAILURE")
                original = failed[0]
                prior_rerun_count = len(ledger.get("technical_reruns", []))
                recovery_number = prior_rerun_count + 1
                recovery_run = {key: original[key] for key in STATIC_RUN_KEYS}
                recovery_run.update(
                    {
                        "run_id": f"{original['run_id']}-TECHRERUN{recovery_number:02d}",
                        "attempt": recovery_number + 1,
                        "technical_rerun_of": original["run_id"],
                        "status": "TECHNICAL_RERUN_AUTHORIZED",
                    }
                )
                entry = entries[str(recovery_run["scenario_token"])]
                excitation = excitations[str(recovery_run["excitation_id"])]
                recovery_audit = _construct(recovery_run, entry, excitation, root, True)
                recovery_audit["technical_rerun_reason"] = original["technical_failure"]
                ledger.setdefault("technical_reruns", []).append(recovery_audit)
                original.update(
                    {
                        "status": "TECHNICAL_COMPLETE_AFTER_FRESH_RERUN",
                        "effective_run_id": recovery_run["run_id"],
                        "effective_run_root": recovery_audit["run_root"],
                        "effective_trace_path": recovery_audit["trace_path"],
                        "effective_planner_telemetry_path": recovery_audit["planner_telemetry_path"],
                        "effective_controller_command_path": recovery_audit["controller_command_path"],
                    }
                )
                audits.append(recovery_audit)
                ledger["counts"].update(
                    {
                        "executed": 1,
                        "technical_reruns": recovery_number,
                        "actual_engineering_runs": recovery_number + 1,
                    }
                )
                write_json(LEDGER, ledger)
                print(json.dumps({"progress": "R2_A_TECHNICAL_RERUN", "run_id": recovery_run["run_id"], "status": "TECHNICAL_COMPLETE"}), flush=True)
                runs_to_execute = runs[1:]
                recovery_count = recovery_number
            else:
                if ledger.get("status") != "FROZEN_PRE_EXECUTION":
                    raise PermissionError("R2_A_FRESH_EXECUTION_REQUIRES_FROZEN_PRE_EXECUTION_LEDGER")
                runs_to_execute = runs
                recovery_count = 0
            ledger["status"] = "AUTHORIZED_ENGINEERING_EXECUTION_RUNNING"
            write_json(LEDGER, ledger)
        else:
            runs_to_execute = runs
        completed_offset = 1 if execute and resume_technical_failure else 0
        for index, run_row in enumerate(runs_to_execute, completed_offset + 1):
            entry = entries[str(run_row["scenario_token"])]
            excitation = excitations[str(run_row["excitation_id"])]
            try:
                audit = _construct(run_row, entry, excitation, root, execute)
            except Exception as exc:
                if execute:
                    run_row["status"] = "TECHNICAL_FAILURE"
                    run_row["technical_failure"] = f"{type(exc).__name__}:{exc}"
                    ledger["status"] = "TECHNICAL_FAILURE_STOPPED_REMAINING_SCHEDULE"
                    ledger["counts"]["executed"] = sum(row.get("status") == "TECHNICAL_COMPLETE" for row in runs)
                    write_json(LEDGER, ledger)
                raise
            audits.append(audit)
            if execute:
                run_row.update(audit)
                ledger["counts"]["executed"] = index
                ledger["counts"]["actual_engineering_runs"] = index + recovery_count
                write_json(LEDGER, ledger)
                print(json.dumps({"progress": "R2_A_ENGINEERING_SIMULATION", "completed": index, "total": len(runs), "run_id": run_row["run_id"]}), flush=True)
        status = "80_OF_80_ZERO_RUN_CONSTRUCTION_PASS" if not execute else "80_OF_80_AUTHORIZED_ENGINEERING_RUNS_TECHNICAL_COMPLETE"
        result = {
            "schema_version": "r2_a_controller_transfer_execution_audit_v1.0",
            "status": status,
            "runs": audits,
            "counts": {
                "exact_resolution": len(audits),
                "runner_construction": len(audits),
                "Primary80_controller": len(audits),
                "TwoStageController_LQR": len(audits),
                "actual_engineering_runs": (80 + recovery_count) if execute else 0,
                "technical_reruns": recovery_count if execute else 0,
            },
            "scientific_simulation": False,
            "confirmatory_roster_selected": False,
            "RBR_started": False,
        }
        if execute:
            ledger["status"] = status
            ledger["counts"] = {
                "planned": 80,
                "executed": 80,
                "technical_reruns": recovery_count,
                "actual_engineering_runs": 80 + recovery_count,
            }
            ledger["output_root"] = str(output_root.relative_to(ROOT)) if output_root.is_relative_to(ROOT) else str(output_root)
            ledger["actual_engineering_simulations"] = 80 + recovery_count
            ledger["scientific_simulations"] = 0
            write_json(LEDGER, ledger)
        return result
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the authorized frozen DEV design")
    parser.add_argument("--resume-technical-failure", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.resume_technical_failure and not args.execute:
        raise ValueError("TECHNICAL_FAILURE_RESUME_REQUIRES_EXECUTE")
    result = run(
        execute=args.execute,
        output_root=args.output_root,
        resume_technical_failure=args.resume_technical_failure,
    )
    if args.audit_output:
        if args.audit_output.exists():
            raise FileExistsError(f"VERSIONED_AUDIT_EXISTS:{args.audit_output}")
        write_json(args.audit_output, result)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
