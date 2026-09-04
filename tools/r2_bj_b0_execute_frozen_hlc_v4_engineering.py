#!/usr/bin/env python3
"""Construct all frozen BJ-B0 runners and stop before any simulation starts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_generator_v4 import ARM_BASELINE, ARM_TREATMENT  # noqa: E402
from tools.r2_bj_a_hlc_morphology_feasible_planner_v4 import _states  # noqa: E402
from tools.r2_bj_b0_hlc_v4_engineering_planner import R2BJB0HLCV4EngineeringPlanner  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_bj_b0_hlc_v4_engineering_roster_v1.0.json"
SCHEDULE = R2 / "r2_bj_b0_hlc_v4_pair_schedule_v1.0.json"
BINDINGS = R2 / "r2_bj_b0_exact_pair_binding_manifest_v1.0.json"
AUTHORIZATION = R2 / "r2_bj_b0_execution_authorization_gate_v1.0.json"
PARAMETERS = R2 / "r2_bj_a_hlc_global_parameter_space_v4.0.json"
CENSUS = R2 / "r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json"
PROTECTED = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_planner_from_environment(run_id: str, trace_dir: str, telemetry_dir: str) -> R2BJB0HLCV4EngineeringPlanner:
    run = next(row for row in read(SCHEDULE)["runs"] if row["run_id"] == run_id)
    entry = next(row for row in read(ROSTER)["entries"] if row["scenario_token"] == run["scenario_token"])
    parameters = read(PARAMETERS)["global_parameters"]
    return R2BJB0HLCV4EngineeringPlanner(entry, run["arm"], parameters, trace_dir, telemetry_dir)


def authorize_before_simulator_start(
    gate: Mapping[str, Any], component_sha: str, schedule_sha: str, binding_sha: str,
    requested_runs: int, exact_schedule: bool,
) -> None:
    """The mandatory last boundary; every failure occurs before simulator start."""
    if not gate.get("BJ_B_ENGINEERING_SIMULATION_AUTHORIZED", False):
        raise PermissionError("R2_BJ_B0_NOT_AUTHORIZED_BEFORE_SIMULATOR_START")
    expected = gate.get("authorized", {})
    if component_sha != expected.get("component_manifest_sha256"):
        raise PermissionError("R2_BJ_B0_COMPONENT_SHA_MISMATCH_BEFORE_SIMULATOR_START")
    if schedule_sha != expected.get("schedule_sha256") or binding_sha != expected.get("pair_binding_sha256"):
        raise PermissionError("R2_BJ_B0_FROZEN_PACKAGE_SHA_MISMATCH_BEFORE_SIMULATOR_START")
    if requested_runs > int(gate.get("NEW_RUN_BUDGET", 0)):
        raise PermissionError("R2_BJ_B0_BUDGET_EXCEEDED_BEFORE_SIMULATOR_START")
    if not exact_schedule:
        raise PermissionError("R2_BJ_B0_SCHEDULE_MISMATCH_BEFORE_SIMULATOR_START")


def _decode(value: Mapping[str, Any]) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(value["base64"]), dtype=np.dtype(value["dtype"]))
    result = data.reshape(value["shape"])
    if hashlib.sha256(result.tobytes(order="C")).hexdigest() != value["sha256"]:
        raise RuntimeError("R2_BJ_B0_REFERENCE_ARRAY_SHA_MISMATCH")
    return result


def _prediv(binding: Mapping[str, Any], census_by_index: Mapping[int, Mapping[str, Any]], parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    source = census_by_index[int(binding["shared_binding"]["reference_geometry_locator"]["census_index"])]
    closure = source["predicate_result"]["closure"]
    reference = closure["reference_geometry"]
    route = closure["route_coverage"]
    corridor = {
        "source_reference_xy": _decode(reference["source"]), "target_reference_xy": _decode(reference["target"]),
        "source_current_arc_m": route["source_current_arc_m"], "target_current_arc_m": route["target_current_arc_m"],
    }
    initial = binding["shared_binding"]["initial_state"]
    current = {
        "time_us": initial["initial_time_us"], "speed_mps": initial["initial_speed_mps"],
        "rear_axle": {"x": initial["initial_x"], "y": initial["initial_y"], "heading": initial["initial_heading"]},
    }
    checks = []
    for absolute in np.arange(0.0, 1.1, 0.1):
        left, _, _ = _states(current, float(absolute), corridor, ARM_BASELINE, parameters, True)
        right, _, _ = _states(current, float(absolute), corridor, ARM_TREATMENT, parameters, True)
        checks.append({"absolute_episode_time_s": round(float(absolute), 1), "exact_equal": left == right})
    return {"planner_call_count": 11, "checks": checks, "exact_equal": all(row["exact_equal"] for row in checks)}


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents", "planner=r2_bj_b0_hlc_v4_engineering",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential", "disable_callback_parallelization=true", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026090401",
        "run_metric=true", "enable_simulation_progress_bar=false", "experiment_name=r2_bj_b0_hlc_v4_engineering",
        f"job_name={run['run_id']}", f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def zero_run_preflight() -> Mapping[str, Any]:
    gate, roster, schedule, binding_doc = read(AUTHORIZATION), read(ROSTER), read(SCHEDULE), read(BINDINGS)
    if gate["BJ_B_ENGINEERING_SIMULATION_AUTHORIZED"] or gate["NEW_RUN_BUDGET"] != 0:
        raise PermissionError("R2_BJ_B0_ZERO_RUN_GATE_NOT_CLOSED")
    if sha(PROTECTED) != PROTECTED_SHA:
        raise RuntimeError("R2_BJ_B0_PROTECTED_CSV_SHA_CHANGED")
    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1

    by_token = {row["scenario_token"]: row for row in roster["entries"]}
    binding_by_pair = {row["pair_id"]: row for row in binding_doc["bindings"]}
    census_by_index = {row["census_index"]: row for row in read(CENSUS)["entries"]}
    parameters = read(PARAMETERS)["global_parameters"]
    prediv = [_prediv(row, census_by_index, parameters) for row in binding_doc["bindings"]]
    rows, output_paths = [], []
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with tempfile.TemporaryDirectory(prefix="r2_bj_b0_zero_run_") as directory:
        temp = Path(directory)
        for run in schedule["runs"]:
            entry = by_token[run["scenario_token"]]
            binding = binding_by_pair[run["pair_id"]]
            if binding["shared_binding"]["scenario_token"] != run["scenario_token"]:
                raise RuntimeError("R2_BJ_B0_PAIR_BINDING_LOOKUP_FAIL")
            run_root = temp / run["run_id"]
            trace, telemetry, raw = run_root / "trace", run_root / "telemetry", run_root / "raw"
            if run_root.exists():
                raise FileExistsError("R2_BJ_B0_OUTPUT_PATH_COLLISION")
            output_paths.append(str(run_root))
            if official_count(entry["db_path"], run["scenario_token"]) != 1:
                raise RuntimeError("R2_BJ_B0_EXACT_SCENARIO_RESOLUTION_NOT_ONE")
            trace.mkdir(parents=True)
            planner = R2BJB0HLCV4EngineeringPlanner(entry, run["arm"], parameters, str(trace), str(telemetry))
            os.environ.update({"R2_BJ_B0_RUN_ID": run["run_id"], "R2_BJ_B0_TRACE_DIR": str(trace), "R2_BJ_B0_TELEMETRY_DIR": str(telemetry)})
            with initialize_config_dir(config_dir=str(config_root)):
                cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
            resolved = OmegaConf.to_container(cfg, resolve=True)
            if "${" in json.dumps(resolved, sort_keys=True):
                raise RuntimeError("R2_BJ_B0_UNRESOLVED_HYDRA")
            common = set_up_common_builder(cfg, "r2_bj_b0_zero_run_build")
            callback_worker = build_callbacks_worker(cfg)
            callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
            runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
            if len(runners) != 1:
                raise RuntimeError("R2_BJ_B0_RUNNER_COUNT_NOT_ONE")
            controller = runners[0]._simulation._time_controller
            if controller.__class__ is not R1Primary80ScientificTimeControllerV1 or controller.number_of_iterations() != 81:
                raise RuntimeError("R2_BJ_B0_PRIMARY80_BINDING_FAIL")
            rows.append({
                "run_order": run["run_order"], "run_id": run["run_id"], "pair_id": run["pair_id"],
                "arm": run["arm"], "exact_scenario_resolution_count": 1, "full_Hydra_composition": "PASS",
                "pair_binding_lookup": "PASS", "output_path_fresh": True,
                "planner_class": planner.__class__.__name__, "time_controller_class": controller.__class__.__name__,
                "time_controller_iterations": controller.number_of_iterations(), "runner_count": 1,
                "simulation_started": False, "runner_run_calls": 0,
            })
    return {
        "schema_version": "r2_bj_b0_zero_run_integration_preflight_audit_v1.0",
        "status": "R2_BJ_B0_ZERO_RUN_EXECUTION_PACKAGE_FROZEN_READY_FOR_CANARY_OWNER_REVIEW",
        "A5_APPLICABLE_POOL": 34, "BJ_B_ROSTER": 8, "UNSELECTED_POOL": 26,
        "ROSTER_TOKEN_UNIQUE": f"{len({row['scenario_token'] for row in roster['entries']})}/8",
        "ROSTER_LOG_UNIQUE": f"{len({row['log_id'] for row in roster['entries']})}/8",
        "history_and_permanent_exclusion_overlap": 0, "PAIR_BINDINGS": len(binding_doc["bindings"]),
        "INTENDED_RUNS": len(rows), "hydra_compositions": sum(row["full_Hydra_composition"] == "PASS" for row in rows),
        "exact_scenario_resolutions": sum(row["exact_scenario_resolution_count"] == 1 for row in rows),
        "runner_constructions": sum(row["runner_count"] == 1 for row in rows),
        "output_path_unique": len(set(output_paths)) == 16, "predivergence_exact_pair_count": sum(row["exact_equal"] for row in prediv),
        "A5_component_provenance_complete": "8/8", "runs": rows, "predivergence": prediv,
        "RUNNER_RUN": 0, "NEW_RUN_BUDGET": 0, "CANARY_AUTHORIZED": False,
        "engineering_simulation": 0, "scientific_simulation": 0, "TSB_simulation": 0,
        "R2_C_STARTED": False, "CONFIRMATORY_SMOKE_STARTED": False, "RBR_STARTED": False,
        "protected_CSV_sha256": sha(PROTECTED),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = zero_run_preflight()
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"R2_BJ_B0_VERSIONED_OUTPUT_EXISTS:{args.output}")
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "INTENDED_RUNS", "hydra_compositions", "exact_scenario_resolutions", "runner_constructions", "RUNNER_RUN")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
