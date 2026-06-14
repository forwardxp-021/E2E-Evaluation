#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import importlib
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SENTINEL = -9999.0
EGO_STATE_CHANNELS = [
    "x",
    "y",
    "yaw",
    "speed",
    "velocity_y",
    "acceleration",
    "acceleration_y",
    "time_s",
]
DISCOVERY_MODULES = [
    "nuplan.planning.simulation.planner.abstract_planner",
    "nuplan.planning.simulation.planner.simple_planner",
    "nuplan.planning.simulation.planner.log_future_planner",
    "nuplan.planning.simulation.planner.idm_planner",
    "nuplan.planning.script.run_simulation",
    "nuplan.planning.scenario_builder",
    "nuplan.planning.simulation.runner",
    "nuplan.planning.simulation.simulation",
]
SCENARIO_KEYS = ["db_name", "scene_token", "scenario_id", "sample_id", "start_frame_index", "end_frame_index"]
CSV_COLUMNS = [
    "scenario_index", "planner_id", "planner_name", "timestep_index", "time_s", "x", "y", "yaw",
    "speed", "acceleration", "steering_angle_or_curvature_if_available", "db_name", "scene_token",
    "scenario_id", "sample_id",
]

PLANNER_PROFILES = {
    "expert_or_log_replay": {
        "planner_type": "expert_replay",
        "policy_style": "reference",
        "preferred_classes": ["LogFuturePlanner", "LogPlaybackPlanner", "SimplePlanner"],
        "parameters": {"purpose": "expert/log replay baseline when available"},
    },
    "idm_conservative": {
        "planner_type": "idm",
        "policy_style": "conservative",
        "preferred_classes": ["IDMPlanner", "SimplePlanner"],
        "parameters": {"target_velocity_mps": 7.0, "headway_time_s": 2.0, "accel_max_mps2": 1.0, "decel_max_mps2": 3.5},
    },
    "idm_aggressive": {
        "planner_type": "idm",
        "policy_style": "aggressive",
        "preferred_classes": ["IDMPlanner", "SimplePlanner"],
        "parameters": {"target_velocity_mps": 13.5, "headway_time_s": 0.8, "accel_max_mps2": 2.8, "decel_max_mps2": 5.0},
    },
    "idm_comfort": {
        "planner_type": "idm",
        "policy_style": "comfort",
        "preferred_classes": ["IDMPlanner", "SimplePlanner"],
        "parameters": {"target_velocity_mps": 9.5, "headway_time_s": 1.4, "accel_max_mps2": 1.2, "decel_max_mps2": 2.5},
    },
}



def write_empty_float32_npy(path: Path, shape: Tuple[int, ...]) -> None:
    """Write an empty NumPy .npy v1.0 float32 array without requiring numpy at import time."""
    header = {"descr": "<f4", "fortran_order": False, "shape": shape}
    header_text = str(header).replace('False', 'False') + "\n"
    magic = b"\x93NUMPY"
    version = b"\x01\x00"
    header_len = len(header_text.encode("latin1"))
    pad = (16 - ((len(magic) + len(version) + 2 + header_len) % 16)) % 16
    header_text = header_text[:-1] + (" " * pad) + "\n"
    header_bytes = header_text.encode("latin1")
    path.write_bytes(magic + version + len(header_bytes).to_bytes(2, "little") + header_bytes)

def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_modules() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in DISCOVERY_MODULES:
        item: Dict[str, Any] = {"available": False, "classes": [], "error": ""}
        try:
            spec = importlib.util.find_spec(name)
            item["available"] = spec is not None
            if spec is not None:
                mod = importlib.import_module(name)
                item["classes"] = sorted(k for k, v in vars(mod).items() if isinstance(v, type))
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        out[name] = item
    return out


def choose_planner_class(planner_name: str, discovery: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    profile = PLANNER_PROFILES[planner_name]
    for preferred in profile["preferred_classes"]:
        for module, info in discovery.items():
            if preferred in info.get("classes", []):
                return preferred, module
    return "UNAVAILABLE", ""


def validate_inputs(context_dir: Path, db_root: Path, map_root: Path) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    if not context_dir.is_dir():
        warnings.append({"type": "missing_context_dir", "scenario_id": "", "planner_name": "", "message": f"context_dir does not exist: {context_dir}"})
    if not (context_dir / "merged_metadata.csv").is_file():
        warnings.append({"type": "missing_metadata", "scenario_id": "", "planner_name": "", "message": f"missing Stage 7B.4 metadata: {context_dir / 'merged_metadata.csv'}"})
    if not db_root.is_dir():
        warnings.append({"type": "missing_nuplan_db_root", "scenario_id": "", "planner_name": "", "message": f"nuplan_db_root does not exist: {db_root}"})
    if not map_root.is_dir():
        warnings.append({"type": "missing_nuplan_map_root", "scenario_id": "", "planner_name": "", "message": f"nuplan_map_root does not exist: {map_root}"})
    return warnings


def run_official_nuplan_cli(command_template: str, planner_name: str, scenario: Dict[str, str], out_dir: Path, timeout_s: int) -> Tuple[bool, str]:
    replacements = {"planner_name": planner_name, "output_dir": str(out_dir)}
    for key, value in scenario.items():
        replacements[key] = value
    command = command_template.format(**replacements)
    proc = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    log_path = out_dir / f"nuplan_cli_{planner_name}_{scenario.get('scenario_index', '')}.log"
    log_path.write_text("$ " + command + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8")
    return proc.returncode == 0, str(log_path)


def fail_outputs(out_dir: Path, args: argparse.Namespace, metadata: List[Dict[str, str]], planners: List[str], discovery: Dict[str, Any], warnings: List[Dict[str, str]], planner_rows: List[Dict[str, Any]]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "simulated_ego_trajectory.csv", [], CSV_COLUMNS)
    write_empty_float32_npy(out_dir / "simulated_ego_seq.npy", (0, 0, 0, len(EGO_STATE_CHANNELS)))
    write_csv(out_dir / "simulated_planner_metadata.csv", planner_rows, ["planner_id", "planner_name", "planner_class", "planner_type", "policy_style", "parameters_json", "nuplan_api_used"])
    write_csv(out_dir / "scenario_planner_index.csv", [], ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
    write_csv(out_dir / "simulation_summary.csv", [], ["planner_name", "num_scenarios_attempted", "num_scenarios_succeeded", "success_ratio", "mean_num_timesteps", "mean_final_displacement", "mean_speed", "mean_acceleration", "mean_abs_acceleration"])
    schema = {
        "stage": "7C.1",
        "feature_type": "nuplan_closed_loop_simulated_ego_trajectory",
        "input_stage": "7B.4",
        "uses_official_nuplan_simulation": False,
        "pseudo_rollout": False,
        "num_input_scenarios": len(metadata),
        "num_simulated_scenarios": 0,
        "num_planners": len(planners),
        "planner_names": planners,
        "ego_state_channels": EGO_STATE_CHANNELS,
        "sentinel_value": SENTINEL,
        "simulation_api": "official nuPlan API discovery only; no simulation succeeded",
        "planner_api": "nuPlan planner discovery; unavailable planners are reported in warnings.json",
        "scenario_selection_keys": SCENARIO_KEYS,
        "notes": ["This stage refuses pseudo rollout.", "No fake simulated trajectory was generated.", "Resolve warnings and rerun with official nuPlan simulation available."],
    }
    write_json(out_dir / "simulation_schema.json", schema)
    write_json(out_dir / "warnings.json", {"warnings": warnings, "simulation_api_discovery": discovery, "planner_api_discovery": planner_rows, "scenario_selection": {"metadata_rows": len(metadata), "max_scenarios": args.max_scenarios}, "validation": {"pass": False, "reason": "no official nuPlan closed-loop simulation output was produced"}})
    report = f"""# Stage 7C.1 nuPlan Closed-loop Simulation Report

## Purpose
Run official nuPlan closed-loop simulation for the Stage 7B.4 selected scenarios and export simulated ego trajectories.

## PASS/FAIL summary
FAIL — no official nuPlan closed-loop simulation completed. This script did not create pseudo rollout data.

## nuPlan simulation API used
Discovery result only. Official modules may be available, but no completed closed-loop trajectory export was produced in this run.

## Input dirs
- context_dir: `{args.context_dir}`
- nuplan_db_root: `{args.nuplan_db_root}`
- nuplan_map_root: `{args.nuplan_map_root}`

## Output dir
`{args.output_dir}`

## Planner variants
{', '.join(planners)}

## Scenario selection method
Rows are read from `merged_metadata.csv` and order is preserved. Keys: {', '.join(SCENARIO_KEYS)}.

## Number of attempted scenarios
0

## Number of successful simulations
0

## Output shapes
- simulated_ego_seq.npy: `(0, 0, 0, {len(EGO_STATE_CHANNELS)})`

## Warning summary
See `warnings.json` for structured diagnostics.
"""
    (out_dir / "simulation_report.md").write_text(report, encoding="utf-8")
    return 2


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir exists and is not empty: {out_dir}. Use --overwrite.")
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_dir = Path(args.context_dir)
    metadata_path = context_dir / "merged_metadata.csv"
    db_root = Path(args.nuplan_db_root).expanduser()
    map_root = Path(args.nuplan_map_root).expanduser()
    planners = list(args.planners)
    warnings = validate_inputs(context_dir, db_root, map_root)
    metadata = read_csv(metadata_path) if metadata_path.is_file() else []
    if args.max_scenarios > 0:
        metadata = metadata[: args.max_scenarios]
    for i, row in enumerate(metadata):
        row["scenario_index"] = str(i)

    discovery = discover_modules()
    planner_rows: List[Dict[str, Any]] = []
    for planner_id, planner_name in enumerate(planners):
        if planner_name not in PLANNER_PROFILES:
            warnings.append({"type": "unknown_planner", "scenario_id": "", "planner_name": planner_name, "message": "planner name is not one of the configured Stage 7C.1 planner profiles"})
            continue
        klass, module = choose_planner_class(planner_name, discovery)
        if klass == "UNAVAILABLE":
            warnings.append({"type": "planner_class_unavailable", "scenario_id": "", "planner_name": planner_name, "message": f"No preferred nuPlan class found among {PLANNER_PROFILES[planner_name]['preferred_classes']}"})
        planner_rows.append({
            "planner_id": planner_id,
            "planner_name": planner_name,
            "planner_class": klass,
            "planner_type": PLANNER_PROFILES[planner_name]["planner_type"],
            "policy_style": PLANNER_PROFILES[planner_name]["policy_style"],
            "parameters_json": json.dumps(PLANNER_PROFILES[planner_name]["parameters"], ensure_ascii=False),
            "nuplan_api_used": module,
        })

    run_sim_available = bool(discovery.get("nuplan.planning.script.run_simulation", {}).get("available"))
    runner_available = bool(discovery.get("nuplan.planning.simulation.runner", {}).get("available"))
    if warnings or not metadata or not (run_sim_available or runner_available):
        if not (run_sim_available or runner_available):
            warnings.append({"type": "nuplan_simulation_api_unavailable", "scenario_id": "", "planner_name": "", "message": "Official nuPlan simulation entry points are unavailable in this Python environment."})
        return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows)

    # The safe default is to require an explicit official nuPlan command template because Hydra configs differ by devkit version.
    # This prevents accidental pseudo rollouts or brittle hard-coded config assumptions.
    if not args.nuplan_simulation_command_template:
        warnings.append({"type": "missing_official_simulation_command", "scenario_id": "", "planner_name": "", "message": "Provide --nuplan_simulation_command_template to call the installed official nuPlan run_simulation configuration. No pseudo fallback is allowed."})
        return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows)

    # Command execution hook: official nuPlan writes its own simulation artifacts. This script records status and refuses
    # to synthesize trajectories unless a future parser for those artifacts is added.
    index_rows: List[Dict[str, Any]] = []
    for scenario in metadata:
        for prow in planner_rows:
            ok, log_path = run_official_nuplan_cli(args.nuplan_simulation_command_template, str(prow["planner_name"]), scenario, out_dir, args.command_timeout_s)
            index_rows.append({"scenario_index": scenario.get("scenario_index", ""), "planner_id": prow["planner_id"], "planner_name": prow["planner_name"], "status": "nuplan_cli_succeeded_parser_not_configured" if ok else "nuplan_cli_failed", "num_timesteps": 0, "warning_count": 1, "db_name": scenario.get("db_name", ""), "scene_token": scenario.get("scene_token", ""), "scenario_id": scenario.get("scenario_id", ""), "sample_id": scenario.get("sample_id", "")})
            warnings.append({"type": "trajectory_parser_not_configured" if ok else "nuplan_cli_failed", "scenario_id": scenario.get("scenario_id", ""), "planner_name": str(prow["planner_name"]), "message": f"official nuPlan command log: {log_path}"})
    write_csv(out_dir / "scenario_planner_index.csv", index_rows, ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
    return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 7C.1 official nuPlan closed-loop simulation runner and trajectory export.")
    p.add_argument("--context_dir", default="outputs/stage7b4_nuplan_context_merged")
    p.add_argument("--nuplan_db_root", required=True)
    p.add_argument("--nuplan_map_root", required=True)
    p.add_argument("--output_dir", default="outputs/stage7c1_nuplan_simulation")
    p.add_argument("--planners", nargs="+", default=["expert_or_log_replay", "idm_conservative", "idm_aggressive", "idm_comfort"])
    p.add_argument("--max_scenarios", type=int, default=5, help="0 means all Stage 7B.4 metadata rows.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--nuplan_simulation_command_template", default="", help="Optional official nuPlan command template. Placeholders include {planner_name}, {scenario_id}, {db_name}, {scene_token}, {sample_id}, {output_dir}.")
    p.add_argument("--command_timeout_s", type=int, default=3600)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
