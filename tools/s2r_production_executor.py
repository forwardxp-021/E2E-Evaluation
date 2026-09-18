#!/usr/bin/env python3
"""Canonical S2R production execution binding.

The module is import-safe and zero-run by default.  It binds the one future
TSB production path, constructs every arm from fresh objects, captures the
precontext identity in the execution manifest, and exposes the sole
authorization-gated runner call.  Scientific execution remains closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEVKIT = ROOT.parent / "nuplan-devkit"
CONFIG_ROOT = DEVKIT / "nuplan/planning/script/config/simulation"
PARAMETERS = ROOT / "docs/stageR/r2/r2_b_calibration_rounds/r2_b_tsb_round_0_parameters_v1.0.json"
PLANNER = ROOT / "tools/r2_b_controller_aware_planner_v1.py"
RECORDER = ROOT / "tools/r2_bj_b0_2_passive_actual_lqr_recorder.py"
ANALYZER = ROOT / "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py"
SERIALIZER = ROOT / "tools/s1_protocol_schema.py"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
SCHEMA_VERSION = "s2r_execution_manifest_v1"
CONFIG_SCHEMA_VERSION = "s2r_production_stack_v1"
SEED = 2026091801
ARMS = ("BASELINE", "TREATMENT")
MANIFEST_FIELDS = (
    "schema_version", "run_id", "pair_id", "arm", "session_id", "log_id",
    "scenario_token", "exposure_class", "role", "git_sha", "executor_sha",
    "planner_sha", "config_hash", "schema_hash", "runtime_fingerprint",
    "precontext_id", "route_id", "arm_instance_id", "planner_instance_id",
    "controller_instance_id", "recorder_instance_id", "random_state_declaration",
    "start_timestamp_us", "expected_output_paths", "execution_status",
)


class S2RExecutionError(RuntimeError):
    """Fail-closed canonical executor error."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def precontext_id(payload: Mapping[str, Any]) -> str:
    if payload.get("schema_version") != "s2r_precontext_v1":
        raise S2RExecutionError("PRECONTEXT_SCHEMA_VERSION_MISMATCH")
    return canonical_sha256(payload)


def route_id(map_name: str, route_roadblock_ids: list[str]) -> str:
    if not map_name or not route_roadblock_ids:
        raise S2RExecutionError("ROUTE_ID_INPUT_INCOMPLETE")
    return canonical_sha256({"schema_version": "s2r_route_v1", "map_name": map_name, "route_roadblock_ids": route_roadblock_ids})


def runtime_fingerprint() -> Mapping[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "nuplan_devkit_root": str(DEVKIT),
        "seed": SEED,
        "thread_limits": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
    }


def resolved_override_contract(spec: Mapping[str, Any], raw_root: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r2_b_controller_aware_dev_v1",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{spec['db_path']}]",
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_tokens=[{spec['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential",
        "disable_callback_parallelization=true",
        "scenario_builder.max_workers=1",
        "max_callback_workers=1",
        "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0",
        "gpu=false",
        f"seed={SEED}",
        "run_metric=true",
        "enable_simulation_progress_bar=false",
        "experiment_name=s2r_canonical_production",
        f"job_name={spec['run_id']}",
        f"output_dir={raw_root}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


@dataclass
class FreshArm:
    spec: Mapping[str, Any]
    runner: Any
    planner: Any
    simulation: Any
    controller: Any
    tracker: Any
    motion_model: Any
    callbacks: Any
    recorder: Any
    random_state: random.Random
    run_root: Path
    config_hash: str
    precontext_id: str
    route_id: str
    common_builder: Any = None
    cfg: Any = None


def _validate_spec(spec: Mapping[str, Any]) -> None:
    required = {
        "run_id", "pair_id", "arm", "session_id", "log_id", "scenario_token",
        "exposure_class", "role", "db_path", "map_name", "route_roadblock_ids",
        "precontext", "start_timestamp_us",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise S2RExecutionError(f"ARM_SPEC_FIELDS_MISSING:{missing}")
    if spec["arm"] not in ARMS or spec["role"] not in {"B", "V", "C"}:
        raise S2RExecutionError("ARM_OR_ROLE_INVALID")
    if precontext_id(spec["precontext"]) != spec.get("precontext_id"):
        raise S2RExecutionError("PRECONTEXT_ID_MISMATCH")
    expected_route = route_id(str(spec["map_name"]), [str(x) for x in spec["route_roadblock_ids"]])
    if expected_route != spec.get("route_id"):
        raise S2RExecutionError("ROUTE_ID_MISMATCH")


def verify_scenario_precontext(scenario: Any, spec: Mapping[str, Any]) -> None:
    """Re-capture authoritative scenario identity before any runner call."""
    expected = spec["precontext"]["ego_initial_state"]
    tokens = list(scenario._lidarpc_tokens)
    if not tokens or str(tokens[0]) != str(expected["official_simulation_initial_lidar_token"]):
        raise S2RExecutionError("PRODUCTION_INITIAL_LIDAR_TOKEN_MISMATCH")
    ego = scenario.get_ego_state_at_iteration(0)
    observed = {
        "initial_x": round(float(ego.rear_axle.x), 6),
        "initial_y": round(float(ego.rear_axle.y), 6),
        "initial_heading": round(float(ego.rear_axle.heading), 8),
        "initial_speed_mps": round(float(ego.dynamic_car_state.speed), 6),
        "initial_time_us": int(ego.time_us),
    }
    for key, value in observed.items():
        if value != expected[key]:
            raise S2RExecutionError(f"PRODUCTION_PRECONTEXT_EGO_MISMATCH:{key}")
    observed_route = [str(x) for x in scenario.get_route_roadblock_ids()]
    if observed_route != [str(x) for x in spec["route_roadblock_ids"]]:
        raise S2RExecutionError("PRODUCTION_ROUTE_BINDING_MISMATCH")
    if str(scenario.token) != str(spec["scenario_token"]):
        raise S2RExecutionError("PRODUCTION_SCENARIO_TOKEN_MISMATCH")
    if str(scenario.log_name) != str(spec["log_id"]):
        raise S2RExecutionError("PRODUCTION_LOG_ID_MISMATCH")


def build_official_tsb_arm(spec: Mapping[str, Any], run_root: Path) -> FreshArm:
    """Construct one official arm without calling runner.run or planner.step."""
    _validate_spec(spec)
    if sha256_file(PROTECTED) != PROTECTED_SHA:
        raise S2RExecutionError("PROTECTED_ASSET_SHA_MISMATCH")
    if run_root.exists():
        raise S2RExecutionError("FRESH_RUN_ROOT_REQUIRED")

    from tools.r1_b2_8_r3_prospective_selector import official_count, official_env
    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r2_b_controller_aware_planner_v1 import R2BControllerAwarePlannerV1
    from tools.r2_bj_b0_2_passive_actual_lqr_recorder import PassiveActualLQRRecorderV1

    if official_count(str(spec["db_path"]), str(spec["scenario_token"])) != 1:
        raise S2RExecutionError("EXACT_SCENARIO_RESOLUTION_NOT_ONE")
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))["parameters"]
    trace_root, telemetry_root, raw_root = run_root / "trace", run_root / "telemetry", run_root / "raw"
    trace_root.mkdir(parents=True)
    roster_row = {
        "scenario_token": str(spec["scenario_token"]),
        "log_id": str(spec["log_id"]),
        "db_path": str(spec["db_path"]),
        "map_name": str(spec["map_name"]),
        "route_roadblock_ids": [str(x) for x in spec["route_roadblock_ids"]],
        "initial_state": dict(spec["precontext"]["ego_initial_state"]),
    }
    planner = R2BControllerAwarePlannerV1(roster_row, "R-TSB", str(spec["arm"]), parameters, str(trace_root), str(telemetry_root))
    os.environ.update({
        "R2_B_RUN_ID": str(spec["run_id"]),
        "R2_B_TRACE_DIR": str(trace_root),
        "R2_B_TELEMETRY_DIR": str(telemetry_root),
        "R2_B_PARAMETER_FILE": str(PARAMETERS),
    })
    overrides = resolved_override_contract(spec, raw_root)
    with initialize_config_dir(config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="default_simulation", overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if "${" in canonical_json(resolved):
        raise S2RExecutionError("UNRESOLVED_HYDRA_CONFIG")
    common = set_up_common_builder(cfg, "s2r_canonical_zero_run_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise S2RExecutionError("RUNNER_COUNT_NOT_ONE")
    runner = runners[0]
    simulation = runner._simulation
    verify_scenario_precontext(simulation._scenario, spec)
    controller = simulation._ego_controller
    tracker, motion_model = controller._tracker, controller._motion_model
    recorder = PassiveActualLQRRecorderV1(
        telemetry_root / "actual_lqr_controller_telemetry.jsonl",
        spec,
        {"executor_sha256": sha256_file(Path(__file__)), "planner_sha256": sha256_file(PLANNER), "config_hash": canonical_sha256(resolved)},
    )
    recorder.install(controller, simulation._time_controller)
    return FreshArm(
        spec=dict(spec), runner=runner, planner=planner, simulation=simulation,
        controller=controller, tracker=tracker, motion_model=motion_model,
        callbacks=callbacks, recorder=recorder, random_state=random.Random(SEED),
        run_root=run_root, config_hash=canonical_sha256(resolved),
        precontext_id=str(spec["precontext_id"]), route_id=str(spec["route_id"]),
        common_builder=common, cfg=cfg,
    )


def build_fresh_arm(
    spec: Mapping[str, Any], run_root: Path,
    factory: Callable[[Mapping[str, Any], Path], FreshArm] = build_official_tsb_arm,
) -> FreshArm:
    return factory(spec, run_root)


def component_identity(arm: FreshArm) -> Mapping[str, int]:
    return {
        "runner": id(arm.runner), "planner": id(arm.planner), "simulation": id(arm.simulation),
        "controller": id(arm.controller), "tracker": id(arm.tracker), "motion_model": id(arm.motion_model),
        "callbacks": id(arm.callbacks), "recorder": id(arm.recorder), "random_state": id(arm.random_state),
    }


def assert_fresh_pair(baseline: FreshArm, treatment: FreshArm) -> Mapping[str, Any]:
    if baseline.spec["arm"] != "BASELINE" or treatment.spec["arm"] != "TREATMENT":
        raise S2RExecutionError("PAIR_ARM_ORDER_INVALID")
    if baseline.spec["pair_id"] != treatment.spec["pair_id"]:
        raise S2RExecutionError("PAIR_ID_MISMATCH")
    if baseline.precontext_id != treatment.precontext_id:
        raise S2RExecutionError("PAIR_INVALID_PRECONTEXT")
    if baseline.route_id != treatment.route_id:
        raise S2RExecutionError("PAIR_INVALID_ROUTE")
    left, right = component_identity(baseline), component_identity(treatment)
    shared = sorted(key for key in left if left[key] == right[key])
    if shared:
        raise S2RExecutionError(f"CROSS_ARM_OBJECT_REUSE:{shared}")
    if baseline.run_root == treatment.run_root:
        raise S2RExecutionError("CROSS_ARM_ARTIFACT_NAMESPACE_REUSE")
    return {
        "status": "NO_CROSS_ARM_STATE_LEAKAGE_PROVEN_BY_CONSTRUCTION",
        "independent_components": sorted(left),
        "baseline_component_ids": left,
        "treatment_component_ids": right,
        "precontext_equal": True,
        "route_equal": True,
        "isolated_process_required_for_execution": True,
    }


def execution_manifest(spec: Mapping[str, Any], arm: FreshArm, git_sha: str, status: str) -> Mapping[str, Any]:
    ids = component_identity(arm)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": spec["run_id"], "pair_id": spec["pair_id"], "arm": spec["arm"],
        "session_id": spec["session_id"], "log_id": spec["log_id"], "scenario_token": spec["scenario_token"],
        "exposure_class": spec["exposure_class"], "role": spec["role"], "git_sha": git_sha,
        "executor_sha": sha256_file(Path(__file__)), "planner_sha": sha256_file(PLANNER),
        "config_hash": arm.config_hash, "schema_hash": canonical_sha256(MANIFEST_FIELDS),
        "runtime_fingerprint": runtime_fingerprint(), "precontext_id": arm.precontext_id,
        "route_id": arm.route_id, "arm_instance_id": canonical_sha256({"run_id": spec["run_id"], "component_ids": ids}),
        "planner_instance_id": str(ids["planner"]), "controller_instance_id": str(ids["controller"]),
        "recorder_instance_id": str(ids["recorder"]),
        "random_state_declaration": {"seed": SEED, "fresh_python_random_instance": True, "execution_process": "ONE_FRESH_PROCESS_PER_ARM"},
        "start_timestamp_us": int(spec["start_timestamp_us"]),
        "expected_output_paths": {
            "root": str(arm.run_root), "trace": str(arm.run_root / "trace/realized_current_ego.jsonl"),
            "planner_telemetry": str(arm.run_root / "telemetry/planner_transfer.jsonl"),
            "controller_telemetry": str(arm.run_root / "telemetry/actual_lqr_controller_telemetry.jsonl"),
            "raw": str(arm.run_root / "raw"), "manifest": str(arm.run_root / "execution_manifest.json"),
        },
        "execution_status": status,
    }
    if tuple(payload) != MANIFEST_FIELDS:
        raise S2RExecutionError("EXECUTION_MANIFEST_SCHEMA_ORDER_MISMATCH")
    return payload


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def isolated_arm_command(spec_path: Path, authorization_path: Path, output_root: Path) -> list[str]:
    return [sys.executable, "-B", str(Path(__file__).resolve()), "--execute-authorized-arm", str(spec_path), str(authorization_path), str(output_root)]


def validate_authorization(authorization: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if authorization.get("S2R_SCIENTIFIC_EXECUTION_AUTHORIZED") is not True:
        raise S2RExecutionError("SCIENTIFIC_EXECUTION_NOT_AUTHORIZED")
    if authorization.get("authorized_executor_sha256") != sha256_file(Path(__file__)):
        raise S2RExecutionError("AUTHORIZED_EXECUTOR_SHA_MISMATCH")
    if authorization.get("authorized_run_id") != spec["run_id"]:
        raise S2RExecutionError("AUTHORIZED_RUN_ID_MISMATCH")
    if int(authorization.get("new_run_budget", 0)) != 1:
        raise S2RExecutionError("ONE_ARM_PROCESS_BUDGET_REQUIRED")


def execute_authorized_arm(spec: Mapping[str, Any], authorization: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    """Sole future runner.run call; unreachable with the committed closed gate."""
    validate_authorization(authorization, spec)
    arm = build_fresh_arm(spec, output_root / str(spec["run_id"]))
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest_path = arm.run_root / "execution_manifest.json"
    write_manifest(manifest_path, execution_manifest(spec, arm, git_sha, "READY_BEFORE_RUNNER_RUN"))
    report = arm.runner.run()
    if not bool(getattr(report, "succeeded", False)):
        write_manifest(manifest_path, execution_manifest(spec, arm, git_sha, "TECHNICAL_FAILURE_RETAIN_NO_REPLACEMENT"))
        raise S2RExecutionError("RUNNER_REPORT_NOT_SUCCEEDED")
    arm.recorder.validate_complete()
    write_manifest(manifest_path, execution_manifest(spec, arm, git_sha, "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS"))
    return {"run_id": spec["run_id"], "status": "TECHNICAL_COMPLETE_PENDING_PAIR_ANALYSIS", "manifest": str(manifest_path)}


def _main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--execute-authorized-arm":
        spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        authorization = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
        print(json.dumps(execute_authorized_arm(spec, authorization, Path(sys.argv[4])), sort_keys=True))
        return 0
    raise SystemExit("This executor is closed by default; an exact Owner authorization record is required.")


if __name__ == "__main__":
    raise SystemExit(_main())
