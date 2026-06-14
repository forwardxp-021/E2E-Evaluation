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



def _finite_float(value: Any, default: float = SENTINEL) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _first_value(record: Dict[str, Any], names: List[str], default: Any = "") -> Any:
    lower = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
        lname = name.lower()
        if lname in lower and lower[lname] not in (None, ""):
            return lower[lname]
    return default


def _flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_json(v, key))
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
                out.update(_flatten_json(v[0], key))
            else:
                out[key] = v
    return out


def discover_simulation_artifacts(root: Path, allow_unsafe_pickle: bool = False) -> List[Path]:
    suffixes = {".csv", ".json", ".jsonl", ".parquet"}
    if allow_unsafe_pickle:
        suffixes.update({".pkl", ".pickle", ".msgpack", ".msg"})
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        name = path.name.lower()
        if name.startswith("nuplan_cli_") or name in {"warnings.json", "simulation_schema.json"}:
            continue
        score = sum(token in str(path).lower() for token in ["simulation", "trajectory", "ego", "planner", "runner", "history"])
        if score > 0 or path.suffix.lower() in {".parquet", ".jsonl"}:
            candidates.append(path)
    return sorted(candidates, key=lambda x: (x.suffix.lower() != ".parquet", len(str(x)), str(x)))


def _records_from_artifact(path: Path, warnings: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return [dict(r) for r in read_csv(path)]
        if suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(_flatten_json(obj))
            return rows
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return [_flatten_json(x) for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                for key in ["ego_trajectory", "trajectory", "trajectories", "simulation_history", "data", "rows"]:
                    val = obj.get(key)
                    if isinstance(val, list):
                        return [_flatten_json(x) for x in val if isinstance(x, dict)]
                return [_flatten_json(obj)]
        if suffix == ".parquet":
            pd = importlib.import_module("pandas")
            return pd.read_parquet(path).to_dict(orient="records")
        if suffix in {".pkl", ".pickle"}:
            pickle = importlib.import_module("pickle")
            obj = pickle.loads(path.read_bytes())
            warnings.append({"type": "unsafe_pickle_artifact_parsed", "scenario_id": "", "planner_name": "", "message": f"Parsed trusted pickle artifact after explicit --allow_unsafe_pickle_artifacts: {path}"})
            if isinstance(obj, list):
                return [_flatten_json(x) for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return [_flatten_json(obj)]
        if suffix in {".msgpack", ".msg"}:
            msgpack = importlib.import_module("msgpack")
            obj = msgpack.unpackb(path.read_bytes(), raw=False)
            if isinstance(obj, list):
                return [_flatten_json(x) for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return [_flatten_json(obj)]
    except Exception as exc:
        warnings.append({"type": "artifact_parse_error", "scenario_id": "", "planner_name": "", "message": f"{path}: {type(exc).__name__}: {exc}"})
    return []


def _row_has_trajectory(record: Dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in record}
    has_x = any(k.endswith("x") or k in {"ego_x", "pose_x", "x"} for k in keys)
    has_y = any(k.endswith("y") or k in {"ego_y", "pose_y", "y"} for k in keys)
    has_yaw = any("yaw" in k or "heading" in k for k in keys)
    return has_x and has_y and has_yaw


def parse_official_trajectory_outputs(search_dir: Path, scenario: Dict[str, str], planner_row: Dict[str, Any], warnings: List[Dict[str, str]], allow_unsafe_pickle: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    artifacts = discover_simulation_artifacts(search_dir, allow_unsafe_pickle=allow_unsafe_pickle)
    parsed_rows: List[Dict[str, Any]] = []
    used: List[str] = []
    for artifact in artifacts:
        for rec in _records_from_artifact(artifact, warnings):
            if not _row_has_trajectory(rec):
                continue
            t = _finite_float(_first_value(rec, ["time_s", "time", "timestamp_s", "relative_time_s", "ego_state.time_s"], len(parsed_rows)))
            row = {
                "scenario_index": scenario.get("scenario_index", _first_value(rec, ["scenario_index"], "")),
                "planner_id": planner_row.get("planner_id", _first_value(rec, ["planner_id"], "")),
                "planner_name": planner_row.get("planner_name", _first_value(rec, ["planner_name", "planner"], "")),
                "timestep_index": int(_finite_float(_first_value(rec, ["timestep_index", "iteration", "step", "index"], len(parsed_rows)), len(parsed_rows))),
                "time_s": t,
                "x": _finite_float(_first_value(rec, ["x", "ego_x", "pose_x", "ego_state.x", "center.x", "rear_axle.x"])),
                "y": _finite_float(_first_value(rec, ["y", "ego_y", "pose_y", "ego_state.y", "center.y", "rear_axle.y"])),
                "yaw": _finite_float(_first_value(rec, ["yaw", "heading", "ego_yaw", "ego_state.heading", "center.heading", "rear_axle.heading"])),
                "speed": _finite_float(_first_value(rec, ["speed", "velocity", "v", "ego_speed", "dynamic_car_state.speed", "velocity_x"])),
                "acceleration": _finite_float(_first_value(rec, ["acceleration", "accel", "a", "ego_acceleration", "dynamic_car_state.acceleration", "acceleration_x"])),
                "steering_angle_or_curvature_if_available": _finite_float(_first_value(rec, ["steering_angle", "curvature", "tire_steering_angle"], SENTINEL)),
                "db_name": scenario.get("db_name", _first_value(rec, ["db_name", "database", "log_name"], "")),
                "scene_token": scenario.get("scene_token", _first_value(rec, ["scene_token", "token"], "")),
                "scenario_id": scenario.get("scenario_id", _first_value(rec, ["scenario_id", "scenario_name"], "")),
                "sample_id": scenario.get("sample_id", _first_value(rec, ["sample_id", "sample_token", "lidar_pc_token"], "")),
            }
            if all(math.isfinite(float(row[c])) for c in ["time_s", "x", "y", "yaw", "speed", "acceleration"]):
                parsed_rows.append(row)
                used.append(str(artifact))
    parsed_rows.sort(key=lambda r: (int(r["timestep_index"]), float(r["time_s"])))
    if not parsed_rows:
        return [], ""
    parser_name = "recursive_official_artifact_parser:" + ";".join(sorted(set(used))[:5])
    return parsed_rows, parser_name


def build_simulated_seq(rows: List[Dict[str, Any]], out_path: Path) -> Tuple[Tuple[int, int, int, int], bool]:
    np = importlib.import_module("numpy")
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario_index"]), str(row["planner_id"])), []).append(row)
    n = len(groups)
    t_max = max((len(v) for v in groups.values()), default=0)
    arr = np.full((n, 1, t_max, len(EGO_STATE_CHANNELS)), SENTINEL, dtype=np.float32)
    for i, key in enumerate(sorted(groups, key=lambda x: (int(x[0]) if str(x[0]).isdigit() else 10**9, int(x[1]) if str(x[1]).isdigit() else 10**9))):
        for t, row in enumerate(sorted(groups[key], key=lambda r: int(r["timestep_index"]))):
            values = [row["x"], row["y"], row["yaw"], row["speed"], SENTINEL, row["acceleration"], SENTINEL, row["time_s"]]
            arr[i, 0, t, :] = np.asarray(values, dtype=np.float32)
    finite = bool(arr.size > 0 and np.isfinite(arr).all())
    np.save(out_path, arr)
    return tuple(arr.shape), finite


def fail_outputs(out_dir: Path, args: argparse.Namespace, metadata: List[Dict[str, str]], planners: List[str], discovery: Dict[str, Any], warnings: List[Dict[str, str]], planner_rows: List[Dict[str, Any]]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "simulated_ego_trajectory.csv", [], CSV_COLUMNS)
    write_empty_float32_npy(out_dir / "simulated_ego_seq.npy", (0, 0, 0, len(EGO_STATE_CHANNELS)))
    write_csv(out_dir / "simulated_planner_metadata.csv", planner_rows, ["planner_id", "planner_name", "planner_class", "planner_type", "policy_style", "parameters_json", "nuplan_api_used"])
    scenario_index_path = out_dir / "scenario_planner_index.csv"
    if not scenario_index_path.is_file():
        write_csv(scenario_index_path, [], ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
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

    index_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    parser_names: List[str] = []
    official_success_count = 0
    for scenario in metadata:
        for prow in planner_rows:
            before_warning_count = len(warnings)
            run_dir = out_dir / "official_nuplan_runs" / f"scenario_{scenario.get('scenario_index', '')}" / str(prow["planner_name"])
            run_dir.mkdir(parents=True, exist_ok=True)
            ok, log_path = run_official_nuplan_cli(args.nuplan_simulation_command_template, str(prow["planner_name"]), scenario, run_dir, args.command_timeout_s)
            if not ok:
                warnings.append({"type": "nuplan_cli_failed", "scenario_id": scenario.get("scenario_id", ""), "planner_name": str(prow["planner_name"]), "message": f"official nuPlan command failed; log: {log_path}"})
                status = "failed"
                parsed: List[Dict[str, Any]] = []
            else:
                official_success_count += 1
                parsed, parser_name = parse_official_trajectory_outputs(run_dir, scenario, prow, warnings, allow_unsafe_pickle=args.allow_unsafe_pickle_artifacts)
                if parsed:
                    status = "succeeded"
                    trajectory_rows.extend(parsed)
                    parser_names.append(parser_name)
                else:
                    new_warning_types = {w.get("type", "") for w in warnings[before_warning_count:]}
                    status = "parser_failed" if "artifact_parse_error" in new_warning_types else "no_trajectory_found"
                    warnings.append({"type": "no_trajectory_found", "scenario_id": scenario.get("scenario_id", ""), "planner_name": str(prow["planner_name"]), "message": f"official nuPlan command succeeded but no supported trajectory artifact was parsed under {run_dir}; log: {log_path}"})
            index_rows.append({"scenario_index": scenario.get("scenario_index", ""), "planner_id": prow["planner_id"], "planner_name": prow["planner_name"], "status": status, "num_timesteps": len(parsed), "warning_count": len(warnings) - before_warning_count, "db_name": scenario.get("db_name", ""), "scene_token": scenario.get("scene_token", ""), "scenario_id": scenario.get("scenario_id", ""), "sample_id": scenario.get("sample_id", "")})

    if not trajectory_rows:
        write_csv(out_dir / "scenario_planner_index.csv", index_rows, ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
        return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows)

    if importlib.util.find_spec("numpy") is None:
        warnings.append({"type": "missing_numpy", "scenario_id": "", "planner_name": "", "message": "Parsed official trajectories, but NumPy is required to write non-empty simulated_ego_seq.npy."})
        write_csv(out_dir / "scenario_planner_index.csv", index_rows, ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
        return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows)

    write_csv(out_dir / "simulated_ego_trajectory.csv", trajectory_rows, CSV_COLUMNS)
    shape, arrays_finite = build_simulated_seq(trajectory_rows, out_dir / "simulated_ego_seq.npy")
    write_csv(out_dir / "simulated_planner_metadata.csv", planner_rows, ["planner_id", "planner_name", "planner_class", "planner_type", "policy_style", "parameters_json", "nuplan_api_used"])
    write_csv(out_dir / "scenario_planner_index.csv", index_rows, ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])

    summary_rows: List[Dict[str, Any]] = []
    for prow in planner_rows:
        pname = str(prow["planner_name"])
        attempted = [r for r in index_rows if r["planner_name"] == pname]
        succeeded = [r for r in attempted if r["status"] == "succeeded"]
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in trajectory_rows:
            if row["planner_name"] == pname:
                groups.setdefault(str(row["scenario_index"]), []).append(row)
        final_displacements = []
        speeds = []
        accels = []
        for grows in groups.values():
            grows = sorted(grows, key=lambda r: int(r["timestep_index"]))
            if len(grows) >= 2:
                final_displacements.append(math.hypot(float(grows[-1]["x"]) - float(grows[0]["x"]), float(grows[-1]["y"]) - float(grows[0]["y"])))
            speeds.extend(float(r["speed"]) for r in grows)
            accels.extend(float(r["acceleration"]) for r in grows if float(r["acceleration"]) != SENTINEL)
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        summary_rows.append({"planner_name": pname, "num_scenarios_attempted": len(attempted), "num_scenarios_succeeded": len(succeeded), "success_ratio": len(succeeded) / len(attempted) if attempted else 0.0, "mean_num_timesteps": mean([float(r["num_timesteps"]) for r in succeeded]), "mean_final_displacement": mean(final_displacements), "mean_speed": mean(speeds), "mean_acceleration": mean(accels), "mean_abs_acceleration": mean([abs(x) for x in accels])})
    write_csv(out_dir / "simulation_summary.csv", summary_rows, ["planner_name", "num_scenarios_attempted", "num_scenarios_succeeded", "success_ratio", "mean_num_timesteps", "mean_final_displacement", "mean_speed", "mean_acceleration", "mean_abs_acceleration"])

    pass_ok = official_success_count > 0 and bool(trajectory_rows) and arrays_finite
    schema = {"stage": "7C.1", "feature_type": "nuplan_closed_loop_simulated_ego_trajectory", "input_stage": "7B.4", "uses_official_nuplan_simulation": True, "pseudo_rollout": False, "trajectory_parser": sorted(set(parser_names)), "num_input_scenarios": len(metadata), "num_simulated_scenarios": len({r["scenario_index"] for r in trajectory_rows}), "num_planners": len(planners), "planner_names": planners, "ego_state_channels": EGO_STATE_CHANNELS, "sentinel_value": SENTINEL, "scenario_selection_keys": SCENARIO_KEYS, "simulated_ego_seq_shape": list(shape)}
    write_json(out_dir / "simulation_schema.json", schema)
    write_json(out_dir / "warnings.json", {"warnings": warnings, "simulation_api_discovery": discovery, "planner_api_discovery": planner_rows, "validation": {"pass": pass_ok, "official_success_count": official_success_count, "trajectory_rows": len(trajectory_rows), "arrays_finite": arrays_finite}})
    report_status = "PASS" if pass_ok else "FAIL"
    report = f"""# Stage 7C.1 nuPlan Closed-loop Simulation Report

## PASS/FAIL summary
{report_status} — official nuPlan simulation commands succeeded and trajectory export {'is valid' if pass_ok else 'failed validation'}. No pseudo rollout data was generated.

## Output shapes
- simulated_ego_seq.npy: `{shape}`

## Parsed trajectories
- official command successes: {official_success_count}
- parsed trajectory rows: {len(trajectory_rows)}
- parser: `{'; '.join(sorted(set(parser_names)))}`

## Output dir
`{args.output_dir}`

## Warning summary
See `warnings.json` for structured diagnostics.
"""
    (out_dir / "simulation_report.md").write_text(report, encoding="utf-8")
    return 0 if pass_ok else 2


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
    p.add_argument("--allow_unsafe_pickle_artifacts", action="store_true", help="Parse trusted pickle/msgpack nuPlan artifacts. Pickle is unsafe and remains disabled by default.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
