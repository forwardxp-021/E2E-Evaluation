#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import importlib
import importlib.util
import json
import lzma
import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
ALIGNMENT_FIELDS = [
    "scenario_index", "planner_name", "target_db_name", "target_log_name", "target_scene_token",
    "target_scenario_id", "actual_planner_class", "actual_scenario_type", "actual_log_name",
    "actual_scene_token", "actual_nuplan_scenario_token", "actual_msgpack_path", "runner_report_log_name", "runner_report_scenario_name",
    "runner_report_planner_name", "runner_report_succeeded", "runner_report_error_message",
    "db_name_match", "target_log_name_match", "scene_token_match", "stage7b_scene_token_match",
    "scenario_id_match", "aligned", "same_log_alignment_passed", "strict_nuplan_token_alignment_passed",
    "alignment_status",
]

SHELL_UNSAFE_RE = re.compile(r"[|/\\:;&()\[\]{}<> \t\n'\"`$]")
RAW_COMMAND_PLACEHOLDERS = ["scenario_id", "db_name", "scene_token", "sample_id", "planner_name", "target_log_name", "target_scene_token", "target_db_name"]


def shell_safe_slug(value: Any) -> str:
    """Return a shell/path-safe slug for command-template placeholder substitution."""
    text = str(value)
    safe = SHELL_UNSAFE_RE.sub("_", text)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "empty"


def _template_uses_placeholder(template: str, placeholder: str) -> bool:
    return "{" + placeholder + "}" in template


def build_command_replacements(planner_name: str, scenario: Dict[str, str], out_dir: Path) -> Dict[str, str]:
    replacements = {"planner_name": planner_name, "planner_name_safe": shell_safe_slug(planner_name), "output_dir": str(out_dir)}
    for key, value in scenario.items():
        replacements[key] = str(value)
    target = normalize_target_scenario(scenario)
    replacements.update({
        "target_db_name": target["target_db_name"],
        "target_log_name": target["target_log_name"],
        "target_scene_token": target["target_scene_token"],
    })
    for key in ["scenario_id", "db_name", "scene_token", "sample_id", "target_db_name", "target_log_name", "target_scene_token"]:
        replacements[f"{key}_safe"] = shell_safe_slug(scenario.get(key, ""))
    replacements["target_db_name_safe"] = shell_safe_slug(target["target_db_name"])
    replacements["target_log_name_safe"] = shell_safe_slug(target["target_log_name"])
    replacements["target_scene_token_safe"] = shell_safe_slug(target["target_scene_token"])
    return replacements


def add_unsafe_placeholder_warnings(command_template: str, replacements: Dict[str, str], warnings: List[Dict[str, str]]) -> None:
    for placeholder in RAW_COMMAND_PLACEHOLDERS:
        if not _template_uses_placeholder(command_template, placeholder):
            continue
        raw_value = replacements.get(placeholder, "")
        if SHELL_UNSAFE_RE.search(raw_value):
            warnings.append({
                "type": "unsafe_command_placeholder",
                "placeholder": placeholder,
                "raw_value": raw_value,
                "recommended_placeholder": f"{placeholder}_safe",
                "message": f"Command template uses raw {{{placeholder}}} with shell/path-unsafe characters; use {{{placeholder}_safe}} instead.",
            })

CSV_COLUMNS = [
    "scenario_index", "planner_id", "planner_name", "timestep_index", "time_s", "x", "y", "yaw",
    "speed", "acceleration", "steering_angle_or_curvature_if_available", "db_name", "scene_token",
    "scenario_id", "sample_id",
]

PLANNER_PROFILES = {
    "simple_planner": {
        "planner_type": "simple_baseline",
        "policy_style": "simple_baseline",
        "preferred_classes": ["SimplePlanner"],
        "parameters": {
            "purpose": "nuPlan built-in simple planner baseline"
        },
    },
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


def _strip_db_suffix(value: Any) -> str:
    text = str(value or "").strip()
    return text[:-3] if text.endswith(".db") else text


def normalize_target_scenario(scenario: Dict[str, Any]) -> Dict[str, str]:
    scenario_id = str(scenario.get("scenario_id", "") or "")
    scenario_id_db_part = ""
    scenario_id_token_part = ""
    if "|" in scenario_id:
        scenario_id_db_part, scenario_id_token_part = scenario_id.split("|", 1)
    target_db_name = str(scenario.get("db_name", "") or scenario_id_db_part or "").strip()
    target_scene_token = str(scenario.get("scene_token", "") or scenario_id_token_part or "").strip()
    return {
        "scenario_index": str(scenario.get("scenario_index", "")),
        "target_db_name": target_db_name,
        "target_log_name": _strip_db_suffix(target_db_name),
        "target_scene_token": target_scene_token,
        "target_scenario_id": scenario_id,
        "scenario_id_db_part": scenario_id_db_part,
        "scenario_id_log_part": _strip_db_suffix(scenario_id_db_part),
        "scenario_id_token_part": scenario_id_token_part,
    }


def _extract_identity_from_msgpack_path(path: Path) -> Dict[str, str]:
    parts = path.parts
    if "simulation_log" not in parts:
        return {}
    i = parts.index("simulation_log")
    if len(parts) <= i + 4:
        return {}
    return {
        "actual_planner_class": parts[i + 1],
        "actual_scenario_type": parts[i + 2],
        "actual_log_name": parts[i + 3],
        "actual_scene_token": parts[i + 4],
        "actual_msgpack_path": str(path),
    }


def _read_runner_report(run_dir: Path, warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    reports = sorted(run_dir.rglob("runner_report.parquet"))
    if not reports:
        return {}
    try:
        pd = importlib.import_module("pandas")
        rows = pd.read_parquet(reports[0]).to_dict(orient="records")
    except Exception as exc:
        warnings.append({"type": "runner_report_parse_error", "scenario_id": "", "planner_name": "", "message": f"{reports[0]}: {type(exc).__name__}: {exc}"})
        return {}
    return dict(rows[0]) if rows else {}


def build_alignment_record(scenario: Dict[str, str], planner_name: str, run_dir: Path, command_succeeded: bool, warnings: List[Dict[str, str]]) -> Dict[str, Any]:
    target = normalize_target_scenario(scenario)
    msgpacks = discover_msgpack_simulation_logs(run_dir) if run_dir.exists() else []
    actual = _extract_identity_from_msgpack_path(msgpacks[0]) if msgpacks else {}
    runner = _read_runner_report(run_dir, warnings) if run_dir.exists() else {}
    actual_log_name = str(actual.get("actual_log_name", "") or runner.get("log_name", "") or "")
    actual_nuplan_scenario_token = str(actual.get("actual_scene_token", "") or runner.get("scenario_name", "") or "")
    actual_scene_token = actual_nuplan_scenario_token
    target_log_name_match = bool(target["target_log_name"] and actual_log_name and target["target_log_name"] == actual_log_name)
    db_name_match = target_log_name_match
    stage7b_scene_token_match = bool(target["target_scene_token"] and actual_nuplan_scenario_token and target["target_scene_token"] == actual_nuplan_scenario_token)
    scene_token_match = stage7b_scene_token_match
    scenario_id_log_match = bool(target["scenario_id_log_part"] and actual_log_name and target["scenario_id_log_part"] == actual_log_name)
    scenario_id_token_match = bool(target["scenario_id_token_part"] and actual_nuplan_scenario_token and target["scenario_id_token_part"] == actual_nuplan_scenario_token)
    scenario_id_match = scenario_id_log_match and scenario_id_token_match if (target["scenario_id_log_part"] or target["scenario_id_token_part"]) else False
    same_log_alignment_passed = target_log_name_match
    strict_nuplan_token_alignment_passed = target_log_name_match and stage7b_scene_token_match
    aligned = same_log_alignment_passed
    if not command_succeeded:
        status = "NOT_RUN"
    elif not actual_log_name or not actual_nuplan_scenario_token:
        status = "UNKNOWN"
    elif strict_nuplan_token_alignment_passed:
        status = "PASS_STRICT"
    elif same_log_alignment_passed:
        status = "PASS_LOG_ONLY"
    else:
        status = "FAIL"
    return {
        "scenario_index": target["scenario_index"],
        "planner_name": planner_name,
        "target_db_name": target["target_db_name"],
        "target_log_name": target["target_log_name"],
        "target_scene_token": target["target_scene_token"],
        "target_scenario_id": target["target_scenario_id"],
        "actual_planner_class": actual.get("actual_planner_class", ""),
        "actual_scenario_type": actual.get("actual_scenario_type", ""),
        "actual_log_name": actual_log_name,
        "actual_scene_token": actual_scene_token,
        "actual_nuplan_scenario_token": actual_nuplan_scenario_token,
        "actual_msgpack_path": actual.get("actual_msgpack_path", ""),
        "runner_report_log_name": runner.get("log_name", ""),
        "runner_report_scenario_name": runner.get("scenario_name", ""),
        "runner_report_planner_name": runner.get("planner_name", ""),
        "runner_report_succeeded": runner.get("succeeded", ""),
        "runner_report_error_message": runner.get("error_message", ""),
        "db_name_match": db_name_match,
        "target_log_name_match": target_log_name_match,
        "scene_token_match": scene_token_match,
        "stage7b_scene_token_match": stage7b_scene_token_match,
        "scenario_id_match": scenario_id_match,
        "aligned": aligned,
        "same_log_alignment_passed": same_log_alignment_passed,
        "strict_nuplan_token_alignment_passed": strict_nuplan_token_alignment_passed,
        "alignment_status": status,
    }


def write_alignment_outputs(out_dir: Path, metadata: List[Dict[str, str]], records: List[Dict[str, Any]], official_success_count: int) -> Dict[str, Any]:
    num_aligned = sum(1 for r in records if r.get("same_log_alignment_passed") is True)
    num_strict_aligned = sum(1 for r in records if r.get("strict_nuplan_token_alignment_passed") is True)
    num_actual = sum(1 for r in records if r.get("actual_log_name") and r.get("actual_nuplan_scenario_token"))
    summary = {
        "stage": "7C.1C",
        "num_target_scenarios": len(metadata),
        "num_official_successes": official_success_count,
        "num_actual_scenarios_extracted": num_actual,
        "num_aligned": num_aligned,
        "num_same_log_aligned": num_aligned,
        "num_strict_nuplan_token_aligned": num_strict_aligned,
        "alignment_pass_ratio": num_aligned / len(records) if records else 0.0,
        "strict_nuplan_token_alignment_pass_ratio": num_strict_aligned / len(records) if records else 0.0,
        "records": records,
    }
    write_json(out_dir / "scenario_alignment.json", summary)
    write_csv(out_dir / "scenario_alignment.csv", records, ALIGNMENT_FIELDS)
    first = records[0] if records else {}
    if records and all(r.get("alignment_status") == "PASS_STRICT" for r in records):
        status = "PASS_STRICT"
    elif records and all(r.get("alignment_status") in {"PASS_STRICT", "PASS_LOG_ONLY"} for r in records):
        status = "PASS_LOG_ONLY"
    elif not records or all(r.get("alignment_status") == "NOT_RUN" for r in records):
        status = "NOT_RUN"
    elif any(r.get("alignment_status") == "FAIL" for r in records):
        status = "FAIL"
    else:
        status = "UNKNOWN"
    interpretation = (
        "official simulation/export pipeline works and strict Stage 7B.4 scene_token to nuPlan scenario token alignment passed."
        if status == "PASS_STRICT"
        else (
            "official simulation/export pipeline works and the target log matched; Stage 7B.4 scene_token differs from the actual nuPlan scenario token, so use actual_nuplan_scenario_token for exact future reruns."
            if status == "PASS_LOG_ONLY"
            else ("official simulation/export pipeline works, but same-log alignment failed." if status == "FAIL" else "actual scenario identity was not fully available; same-log alignment is not proven.")
        )
    )
    report = f"""# Stage 7C.1C Same-Scenario Alignment Report

## target scenario
- db_name: `{first.get('target_db_name', '')}`
- target_log_name: `{first.get('target_log_name', '')}`
- scene_token: `{first.get('target_scene_token', '')}`
- scenario_id: `{first.get('target_scenario_id', '')}`

## actual simulated scenario
- actual_log_name: `{first.get('actual_log_name', '')}`
- actual_nuplan_scenario_token: `{first.get('actual_nuplan_scenario_token', '')}`
- actual_scenario_type: `{first.get('actual_scenario_type', '')}`
- actual_msgpack_path: `{first.get('actual_msgpack_path', '')}`

## comparison
- official_simulation_export_pipeline_works: `{official_success_count > 0}`
- target_log_name_match: `{first.get('target_log_name_match', False)}`
- stage7b_scene_token_match: `{first.get('stage7b_scene_token_match', False)}`
- same_log_alignment_passed: `{first.get('same_log_alignment_passed', False)}`
- strict_nuplan_token_alignment_passed: `{first.get('strict_nuplan_token_alignment_passed', False)}`
- future_exact_rerun_scenario_token: `{first.get('actual_nuplan_scenario_token', '')}`

## status
{status}

## interpretation
- {interpretation}
- Stage 7B.4 `scene_token` is not necessarily equal to nuPlan `scenario_filter.scenario_tokens`.
- Do not fail Stage 7C.1 smoke only because strict Stage 7B.4 scene-token matching failed when same-log alignment passed.

## aggregate counts
- num_target_scenarios: `{summary['num_target_scenarios']}`
- num_official_successes: `{summary['num_official_successes']}`
- num_actual_scenarios_extracted: `{summary['num_actual_scenarios_extracted']}`
- num_aligned: `{summary['num_aligned']}`
- num_strict_nuplan_token_aligned: `{summary['num_strict_nuplan_token_aligned']}`
- alignment_pass_ratio: `{summary['alignment_pass_ratio']}`
- strict_nuplan_token_alignment_pass_ratio: `{summary['strict_nuplan_token_alignment_pass_ratio']}`
"""
    (out_dir / "scenario_alignment_report.md").write_text(report, encoding="utf-8")
    return summary


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
    context_dir_resolved = context_dir.expanduser().resolve()
    metadata_resolved = (context_dir / "merged_metadata.csv").expanduser().resolve()
    if not context_dir.is_dir():
        message = f"context_dir does not exist: input={context_dir}, resolved={context_dir_resolved}"
        print(f"ERROR: {message}", file=sys.stderr)
        warnings.append({"type": "missing_context_dir", "scenario_id": "", "planner_name": "", "message": message, "context_dir_input": str(context_dir), "context_dir_resolved": str(context_dir_resolved), "merged_metadata_resolved": str(metadata_resolved)})
    if not (context_dir / "merged_metadata.csv").is_file():
        message = f"missing Stage 7B.4 metadata: input={context_dir / 'merged_metadata.csv'}, resolved={metadata_resolved}"
        print(f"ERROR: {message}", file=sys.stderr)
        warnings.append({"type": "missing_metadata", "scenario_id": "", "planner_name": "", "message": message, "context_dir_input": str(context_dir), "context_dir_resolved": str(context_dir_resolved), "merged_metadata_resolved": str(metadata_resolved)})
    if not db_root.is_dir():
        warnings.append({"type": "missing_nuplan_db_root", "scenario_id": "", "planner_name": "", "message": f"nuplan_db_root does not exist: {db_root}"})
    if not map_root.is_dir():
        warnings.append({"type": "missing_nuplan_map_root", "scenario_id": "", "planner_name": "", "message": f"nuplan_map_root does not exist: {map_root}"})
    return warnings


def run_official_nuplan_cli(command_template: str, planner_name: str, scenario: Dict[str, str], out_dir: Path, timeout_s: int, warnings: List[Dict[str, str]], use_shell: bool = False) -> Tuple[bool, str]:
    replacements = build_command_replacements(planner_name, scenario, out_dir)
    add_unsafe_placeholder_warnings(command_template, replacements, warnings)
    command = command_template.format(**replacements)
    argv = shlex.split(command)
    proc = subprocess.run(command if use_shell else argv, shell=use_shell, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    log_path = out_dir / f"nuplan_cli_{shell_safe_slug(planner_name)}_{scenario.get('scenario_index', '')}.log"
    log_path.write_text("$ " + command + "\nargv: " + json.dumps(argv, ensure_ascii=False) + "\nshell: " + str(use_shell) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8")
    return proc.returncode == 0, str(log_path)



def _finite_float(value: Any, default: float = SENTINEL) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _required_float(record: Dict[str, Any], candidate_names: List[str], field_name: str) -> Optional[float]:
    value = _first_value(record, candidate_names, None)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def _empty_parser_validation() -> Dict[str, Any]:
    return {
        "num_candidate_artifact_rows": 0,
        "num_valid_trajectory_rows": 0,
        "num_rejected_rows_invalid_required_pose": 0,
        "msgpack_simulation_log_files_found": 0,
        "msgpack_simulation_log_files_parsed": 0,
        "msgpack_trajectory_rows_extracted": 0,
        "msgpack_parse_errors": [],
        "required_pose_valid_ratio": 0.0,
        "x_non_sentinel_ratio": 0.0,
        "y_non_sentinel_ratio": 0.0,
        "yaw_non_sentinel_ratio": 0.0,
        "min_timesteps_per_trajectory": 0,
        "mean_timesteps_per_trajectory": 0.0,
        "num_trajectories_with_too_few_steps": 0,
        "num_trajectories_with_zero_motion": 0,
    }


def _obj_value(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _obj_path(obj: Any, paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            cur = _obj_value(cur, [part], None)
            if cur is None:
                ok = False
                break
        if ok:
            return cur
    return default


def _time_seconds(ego_state: Any, timestep_index: int) -> float:
    time_point = _obj_path(ego_state, ["time_point", "car_footprint.time_point"], None)
    if time_point is not None:
        seconds = _obj_value(time_point, ["time_s", "seconds"], None)
        if seconds is not None:
            return _finite_float(seconds, float(timestep_index))
        us = _obj_value(time_point, ["time_us", "microseconds"], None)
        if us is not None:
            return _finite_float(us, float(timestep_index) * 1e6) / 1e6
    return float(timestep_index)


def _ego_state_to_row(ego_state: Any, timestep_index: int, scenario: Dict[str, str], planner_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    x = _required_float({"x": _obj_path(ego_state, ["rear_axle.x", "center.x", "car_footprint.rear_axle.x", "car_footprint.center.x"], None)}, ["x"], "x")
    y = _required_float({"y": _obj_path(ego_state, ["rear_axle.y", "center.y", "car_footprint.rear_axle.y", "car_footprint.center.y"], None)}, ["y"], "y")
    yaw = _required_float({"yaw": _obj_path(ego_state, ["rear_axle.heading", "center.heading", "car_footprint.rear_axle.heading", "car_footprint.center.heading"], None)}, ["yaw"], "yaw")
    if x is None or y is None or yaw is None:
        return None
    dynamic = _obj_value(ego_state, ["dynamic_car_state"], None)
    speed = _finite_float(_obj_path(dynamic, ["speed", "rear_axle_velocity_2d.x", "center_velocity_2d.x"], None))
    accel = _finite_float(_obj_path(dynamic, ["acceleration", "rear_axle_acceleration_2d.x", "center_acceleration_2d.x"], None))
    return {
        "scenario_index": scenario.get("scenario_index", ""),
        "planner_id": planner_row.get("planner_id", ""),
        "planner_name": planner_row.get("planner_name", ""),
        "timestep_index": timestep_index,
        "time_s": _time_seconds(ego_state, timestep_index),
        "x": x,
        "y": y,
        "yaw": yaw,
        "speed": speed,
        "acceleration": accel,
        "steering_angle_or_curvature_if_available": _finite_float(_obj_path(ego_state, ["tire_steering_angle", "car_footprint.vehicle_parameters"], SENTINEL)),
        "db_name": scenario.get("db_name", ""),
        "scene_token": scenario.get("scene_token", ""),
        "scenario_id": scenario.get("scenario_id", ""),
        "sample_id": scenario.get("sample_id", ""),
    }


def _load_nuplan_simulation_log(path: Path) -> Any:
    sim_log_mod = importlib.import_module("nuplan.planning.simulation.simulation_log")
    simulation_log_cls = getattr(sim_log_mod, "SimulationLog")
    for method in ["load_data", "deserialize", "load"]:
        if hasattr(simulation_log_cls, method):
            loader = getattr(simulation_log_cls, method)
            try:
                return loader(path)
            except Exception:
                try:
                    return loader(str(path))
                except Exception:
                    pass
    for kwargs in [{"file_path": path}, {"file_path": str(path)}, {"log_file": path}, {"log_file": str(path)}]:
        try:
            obj = simulation_log_cls(**kwargs)
        except Exception:
            continue
        if hasattr(obj, "load_data"):
            return obj.load_data()
        return obj
    raise AttributeError("SimulationLog has no supported load_data/deserialize/load method")


def _parse_msgpack_simulation_log(path: Path, scenario: Dict[str, str], planner_row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    try:
        sim_log = _load_nuplan_simulation_log(path)
    except Exception:
        msgpack = importlib.import_module("msgpack")
        sim_log = msgpack.unpackb(lzma.open(path, "rb").read(), raw=False, strict_map_key=False)
    history = _obj_value(sim_log, ["simulation_history"], sim_log)
    data = _obj_value(history, ["data"], history if isinstance(history, list) else [])
    rows: List[Dict[str, Any]] = []
    rejected = 0
    for i, sample in enumerate(data):
        ego_state = _obj_value(sample, ["ego_state"], None)
        if ego_state is None:
            ego_state = _obj_path(sample, ["ego_state", "sample.ego_state"], None)
        if ego_state is None:
            rejected += 1
            continue
        row = _ego_state_to_row(ego_state, i, scenario, planner_row)
        if row is not None:
            rows.append(row)
        else:
            rejected += 1
    return rows, rejected


def discover_msgpack_simulation_logs(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("simulation_log/**/*.msgpack.xz") if path.is_file())


def parse_official_trajectory_outputs(search_dir: Path, scenario: Dict[str, str], planner_row: Dict[str, Any], warnings: List[Dict[str, str]], min_timesteps: int, allow_unsafe_pickle: bool = False) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    artifacts = discover_simulation_artifacts(search_dir, allow_unsafe_pickle=allow_unsafe_pickle)
    parsed_rows: List[Dict[str, Any]] = []
    used: List[str] = []
    validation = _empty_parser_validation()
    msgpack_logs = discover_msgpack_simulation_logs(search_dir)
    validation["msgpack_simulation_log_files_found"] = len(msgpack_logs)
    for artifact in msgpack_logs:
        try:
            rows, rejected = _parse_msgpack_simulation_log(artifact, scenario, planner_row)
            validation["msgpack_simulation_log_files_parsed"] += 1
            validation["msgpack_trajectory_rows_extracted"] += len(rows)
            validation["num_candidate_artifact_rows"] += len(rows) + rejected
            validation["num_rejected_rows_invalid_required_pose"] += rejected
            parsed_rows.extend(rows)
            if rows:
                used.append(str(artifact))
        except Exception as exc:
            message = f"{artifact}: {type(exc).__name__}: {exc}"
            validation["msgpack_parse_errors"].append(message)
            warnings.append({"type": "msgpack_simulation_log_parse_error", "scenario_id": scenario.get("scenario_id", ""), "planner_name": str(planner_row.get("planner_name", "")), "message": message})
    for artifact in artifacts:
        for rec in _records_from_artifact(artifact, warnings):
            if not _row_has_trajectory(rec):
                continue
            validation["num_candidate_artifact_rows"] += 1
            x = _required_float(rec, ["x", "ego_x", "pose_x", "ego_state.x", "center.x", "rear_axle.x"], "x")
            y = _required_float(rec, ["y", "ego_y", "pose_y", "ego_state.y", "center.y", "rear_axle.y"], "y")
            yaw = _required_float(rec, ["yaw", "heading", "ego_yaw", "ego_state.heading", "center.heading", "rear_axle.heading"], "yaw")
            time_value = _required_float(rec, ["time_s", "time", "timestamp_s", "relative_time_s", "ego_state.time_s"], "time_s")
            timestep_value = _required_float(rec, ["timestep_index", "iteration", "step", "index"], "timestep_index")
            if time_value is None and timestep_value is None:
                validation["num_rejected_rows_invalid_required_pose"] += 1
                continue
            if x is None or y is None or yaw is None:
                validation["num_rejected_rows_invalid_required_pose"] += 1
                continue
            if time_value is None:
                time_value = float(timestep_value)
            if timestep_value is None:
                timestep_value = float(len(parsed_rows))
            row = {
                "scenario_index": scenario.get("scenario_index", _first_value(rec, ["scenario_index"], "")),
                "planner_id": planner_row.get("planner_id", _first_value(rec, ["planner_id"], "")),
                "planner_name": planner_row.get("planner_name", _first_value(rec, ["planner_name", "planner"], "")),
                "timestep_index": int(timestep_value),
                "time_s": time_value,
                "x": x,
                "y": y,
                "yaw": yaw,
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
    validation["num_valid_trajectory_rows"] = len(parsed_rows)
    candidates = validation["num_candidate_artifact_rows"]
    validation["required_pose_valid_ratio"] = len(parsed_rows) / candidates if candidates else 0.0
    for field in ["x", "y", "yaw"]:
        validation[f"{field}_non_sentinel_ratio"] = sum(float(r[field]) != SENTINEL for r in parsed_rows) / len(parsed_rows) if parsed_rows else 0.0
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in parsed_rows:
        groups.setdefault((str(row["scenario_index"]), str(row["planner_id"])), []).append(row)
    lengths = [len(v) for v in groups.values()]
    validation["min_timesteps_per_trajectory"] = min(lengths) if lengths else 0
    validation["mean_timesteps_per_trajectory"] = sum(lengths) / len(lengths) if lengths else 0.0
    validation["num_trajectories_with_too_few_steps"] = sum(n < min_timesteps for n in lengths)
    zero_motion = 0
    valid_keys = set()
    for key, grows in groups.items():
        ordered = sorted(grows, key=lambda r: (int(r["timestep_index"]), float(r["time_s"])))
        has_motion = any(math.hypot(float(r["x"]) - float(ordered[0]["x"]), float(r["y"]) - float(ordered[0]["y"])) > 1e-6 or abs(float(r["yaw"]) - float(ordered[0]["yaw"])) > 1e-6 for r in ordered[1:])
        has_distinct_timestamps = len({float(r["time_s"]) for r in ordered}) > 1 or len({int(r["timestep_index"]) for r in ordered}) > 1
        if not has_motion and not has_distinct_timestamps:
            zero_motion += 1
        if len(ordered) >= min_timesteps and (has_motion or has_distinct_timestamps):
            valid_keys.add(key)
    validation["num_trajectories_with_zero_motion"] = zero_motion
    parsed_rows = [r for r in parsed_rows if (str(r["scenario_index"]), str(r["planner_id"])) in valid_keys]
    if not parsed_rows:
        return [], "", validation
    parser_name = "recursive_official_artifact_parser:" + ";".join(sorted(set(used))[:5])
    return parsed_rows, parser_name, validation


def _axis_sort_key(value: Any) -> Tuple[int, float, str]:
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return (1, 0.0, text)
    return (0, number, text)


def build_simulated_seq(rows: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    np = importlib.import_module("numpy")
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    planner_names_by_id: Dict[str, str] = {}
    for row in rows:
        scenario_key = str(row["scenario_index"])
        planner_key = str(row["planner_id"])
        groups.setdefault((scenario_key, planner_key), []).append(row)
        planner_names_by_id.setdefault(planner_key, str(row.get("planner_name", "")))

    scenario_axis = sorted({key[0] for key in groups}, key=_axis_sort_key)
    planner_axis = sorted({key[1] for key in groups}, key=_axis_sort_key)
    scenario_lookup = {value: i for i, value in enumerate(scenario_axis)}
    planner_lookup = {value: i for i, value in enumerate(planner_axis)}
    t_max = max((len(v) for v in groups.values()), default=0)

    shape = (len(scenario_axis), len(planner_axis), t_max, len(EGO_STATE_CHANNELS))
    arr = np.full(shape, SENTINEL, dtype=np.float32)
    mask = np.zeros(shape[:3], dtype=np.uint8)
    for key in sorted(groups, key=lambda x: (_axis_sort_key(x[0]), _axis_sort_key(x[1]))):
        scenario_i = scenario_lookup[key[0]]
        planner_i = planner_lookup[key[1]]
        for t, row in enumerate(sorted(groups[key], key=lambda r: int(r["timestep_index"]))):
            values = [row["x"], row["y"], row["yaw"], row["speed"], SENTINEL, row["acceleration"], SENTINEL, row["time_s"]]
            arr[scenario_i, planner_i, t, :] = np.asarray(values, dtype=np.float32)
            mask[scenario_i, planner_i, t] = 1

    mask_path = out_path.with_name("simulated_ego_seq_mask.npy")
    index_path = out_path.with_name("simulated_ego_seq_index.json")
    planner_axis_names = [planner_names_by_id.get(planner_id, "") for planner_id in planner_axis]
    np.save(out_path, arr)
    np.save(mask_path, mask)
    index = {
        "scenario_axis": scenario_axis,
        "planner_axis": planner_axis,
        "planner_axis_names": planner_axis_names,
        "ego_state_channels": EGO_STATE_CHANNELS,
        "sentinel_value": SENTINEL,
        "shape": list(shape),
    }
    write_json(index_path, index)
    missing_pair_count = len(scenario_axis) * len(planner_axis) - len(groups)
    return {
        "shape": tuple(shape),
        "mask_shape": tuple(mask.shape),
        "scenario_axis": scenario_axis,
        "planner_axis": planner_axis,
        "planner_axis_names": planner_axis_names,
        "valid_timestep_count": int(mask.sum()),
        "missing_pair_count": int(missing_pair_count),
        "arrays_finite": bool(arr.size > 0 and np.isfinite(arr).all()),
        "mask_path": str(mask_path),
        "index_path": str(index_path),
    }



def merge_parser_validation(total: Dict[str, Any], item: Dict[str, Any]) -> None:
    for key in ["num_candidate_artifact_rows", "num_valid_trajectory_rows", "num_rejected_rows_invalid_required_pose", "num_trajectories_with_too_few_steps", "num_trajectories_with_zero_motion", "msgpack_simulation_log_files_found", "msgpack_simulation_log_files_parsed", "msgpack_trajectory_rows_extracted"]:
        total[key] = int(total.get(key, 0)) + int(item.get(key, 0))
    total.setdefault("msgpack_parse_errors", []).extend(item.get("msgpack_parse_errors", []))
    mins = total.setdefault("_trajectory_mins", [])
    means = total.setdefault("_trajectory_means", [])
    if int(item.get("min_timesteps_per_trajectory", 0)) > 0:
        mins.append(int(item["min_timesteps_per_trajectory"]))
    if float(item.get("mean_timesteps_per_trajectory", 0.0)) > 0:
        means.append(float(item["mean_timesteps_per_trajectory"]))


def finalize_parser_validation(total: Dict[str, Any], rows: List[Dict[str, Any]], min_timesteps: int) -> Dict[str, Any]:
    out = _empty_parser_validation()
    out.update({k: total.get(k, out[k]) for k in out})
    candidates = int(out["num_candidate_artifact_rows"])
    valid = int(out["num_valid_trajectory_rows"])
    out["required_pose_valid_ratio"] = valid / candidates if candidates else 0.0
    for field in ["x", "y", "yaw"]:
        out[f"{field}_non_sentinel_ratio"] = sum(float(r[field]) != SENTINEL for r in rows) / len(rows) if rows else 0.0
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["scenario_index"]), str(row["planner_id"])), []).append(row)
    lengths = [len(v) for v in groups.values()]
    out["min_timesteps_per_trajectory"] = min(lengths) if lengths else 0
    out["mean_timesteps_per_trajectory"] = sum(lengths) / len(lengths) if lengths else 0.0
    out["num_trajectories_with_too_few_steps"] = sum(n < min_timesteps for n in lengths)
    return out

def fail_outputs(out_dir: Path, args: argparse.Namespace, metadata: List[Dict[str, str]], planners: List[str], discovery: Dict[str, Any], warnings: List[Dict[str, str]], planner_rows: List[Dict[str, Any]], parser_validation: Optional[Dict[str, Any]] = None, official_success_count: int = 0, alignment_records: Optional[List[Dict[str, Any]]] = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "simulated_ego_trajectory.csv", [], CSV_COLUMNS)
    write_empty_float32_npy(out_dir / "simulated_ego_seq.npy", (0, 0, 0, len(EGO_STATE_CHANNELS)))
    write_empty_float32_npy(out_dir / "simulated_ego_seq_mask.npy", (0, 0, 0))
    write_json(out_dir / "simulated_ego_seq_index.json", {"scenario_axis": [], "planner_axis": [], "planner_axis_names": [], "ego_state_channels": EGO_STATE_CHANNELS, "sentinel_value": SENTINEL, "shape": [0, 0, 0, len(EGO_STATE_CHANNELS)]})
    write_csv(out_dir / "simulated_planner_metadata.csv", planner_rows, ["planner_id", "planner_name", "planner_class", "planner_type", "policy_style", "parameters_json", "nuplan_api_used"])
    scenario_index_path = out_dir / "scenario_planner_index.csv"
    if not scenario_index_path.is_file():
        write_csv(scenario_index_path, [], ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
    write_csv(out_dir / "simulation_summary.csv", [], ["planner_name", "num_scenarios_attempted", "num_scenarios_succeeded", "success_ratio", "mean_num_timesteps", "mean_final_displacement", "mean_speed", "mean_acceleration", "mean_abs_acceleration"])
    schema = {
        "stage": "7C.1",
        "feature_type": "nuplan_closed_loop_simulated_ego_trajectory",
        "input_stage": "7B.4",
        "uses_official_nuplan_simulation": official_success_count > 0,
        "pseudo_rollout": False,
        "num_input_scenarios": len(metadata),
        "num_simulated_scenarios": 0,
        "num_planners": len(planners),
        "planner_names": planners,
        "ego_state_channels": EGO_STATE_CHANNELS,
        "sentinel_value": SENTINEL,
        "trajectory_parser": [],
        "required_pose_fields": ["x", "y", "yaw"],
        "optional_sentinel_fields": ["speed", "acceleration", "steering_angle_or_curvature_if_available"],
        "min_timesteps": args.min_timesteps,
        "simulation_api": "official nuPlan command template executed" if official_success_count > 0 else "official nuPlan API discovery only; no simulation succeeded",
        "planner_api": "nuPlan planner discovery; unavailable planners are reported in warnings.json",
        "scenario_selection_keys": SCENARIO_KEYS,
        "simulated_ego_seq_shape": [0, 0, 0, len(EGO_STATE_CHANNELS)],
        "scenario_axis": [],
        "planner_axis": [],
        "scenario_axis_key": "scenario_index",
        "planner_axis_key": "planner_id",
        "planner_axis_names": [],
        "notes": ["This stage refuses pseudo rollout.", "No fake simulated trajectory was generated.", "Resolve warnings and rerun with official nuPlan simulation available."],
    }
    parser_validation = parser_validation or _empty_parser_validation()
    if alignment_records is None:
        alignment_records = []
        for scenario in metadata:
            for planner_name in planners:
                alignment_records.append(build_alignment_record(scenario, planner_name, out_dir / "official_nuplan_runs" / f"scenario_{scenario.get('scenario_index', '')}" / str(planner_name), False, warnings))
    alignment_summary = write_alignment_outputs(out_dir, metadata, alignment_records, official_success_count)
    alignment_passed = bool(alignment_records) and all(r.get("same_log_alignment_passed") is True for r in alignment_records)
    strict_alignment_passed = bool(alignment_records) and all(r.get("strict_nuplan_token_alignment_passed") is True for r in alignment_records)
    schema.update({
        "same_scenario_alignment_checked": True,
        "same_scenario_alignment_passed": alignment_passed,
        "strict_nuplan_token_alignment_passed": strict_alignment_passed,
        "same_scenario_alignment_report": "scenario_alignment_report.md",
    })
    write_json(out_dir / "simulation_schema.json", schema)
    write_json(out_dir / "warnings.json", {"warnings": warnings, "simulation_api_discovery": discovery, "planner_api_discovery": planner_rows, "scenario_selection": {"metadata_rows": len(metadata), "max_scenarios": args.max_scenarios}, "scenario_alignment": {"num_target_scenarios": alignment_summary["num_target_scenarios"], "num_actual_scenarios_extracted": alignment_summary["num_actual_scenarios_extracted"], "num_aligned": alignment_summary["num_aligned"], "alignment_pass_ratio": alignment_summary["alignment_pass_ratio"], "passed": alignment_passed, "strict_nuplan_token_alignment_passed": strict_alignment_passed}, "validation": {"pass": False, "reason": "no official nuPlan closed-loop simulation output was produced", "official_success_count": official_success_count, "pseudo_rollout": False, "uses_official_nuplan_simulation": official_success_count > 0, "tensor_validation": {"shape": [0, 0, 0, len(EGO_STATE_CHANNELS)], "mask_shape": [0, 0, 0], "valid_timestep_count": 0, "missing_pair_count": 0, "passed": False}}, "trajectory_parser_validation": parser_validation or _empty_parser_validation()})
    report = f"""# Stage 7C.1 nuPlan Closed-loop Simulation Report

## Purpose
Run official nuPlan closed-loop simulation for the Stage 7B.4 selected scenarios and export simulated ego trajectories.

## PASS/FAIL summary
FAIL — no valid official nuPlan closed-loop trajectory was parsed. This script did not create pseudo rollout data.

## nuPlan simulation API used
Official command successes: `{official_success_count}`. Official modules may be available, but no valid required-pose closed-loop trajectory export was produced in this run.

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

## Number of successful official commands
{official_success_count}

## Output shapes
- simulated_ego_seq.npy: `(0, 0, 0, {len(EGO_STATE_CHANNELS)})`

## Trajectory parser validation
- min_timesteps requirement: `{args.min_timesteps}`
- num_candidate_artifact_rows: `{parser_validation['num_candidate_artifact_rows']}`
- num_valid_trajectory_rows: `{parser_validation['num_valid_trajectory_rows']}`
- num_rejected_rows_invalid_required_pose: `{parser_validation['num_rejected_rows_invalid_required_pose']}`
- msgpack_simulation_log_files_found: `{parser_validation['msgpack_simulation_log_files_found']}`
- msgpack_simulation_log_files_parsed: `{parser_validation['msgpack_simulation_log_files_parsed']}`
- msgpack_trajectory_rows_extracted: `{parser_validation['msgpack_trajectory_rows_extracted']}`
- msgpack_parse_errors: `{len(parser_validation.get('msgpack_parse_errors', []))}`
- required_pose_valid_ratio: `{parser_validation['required_pose_valid_ratio']}`
- x_non_sentinel_ratio: `{parser_validation['x_non_sentinel_ratio']}`
- y_non_sentinel_ratio: `{parser_validation['y_non_sentinel_ratio']}`
- yaw_non_sentinel_ratio: `{parser_validation['yaw_non_sentinel_ratio']}`
- min_timesteps_per_trajectory: `{parser_validation['min_timesteps_per_trajectory']}`
- mean_timesteps_per_trajectory: `{parser_validation['mean_timesteps_per_trajectory']}`
- num_trajectories_with_too_few_steps: `{parser_validation['num_trajectories_with_too_few_steps']}`
- num_trajectories_with_zero_motion: `{parser_validation['num_trajectories_with_zero_motion']}`

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
    parser_validation_total = _empty_parser_validation()
    alignment_records: List[Dict[str, Any]] = []
    for scenario in metadata:
        for prow in planner_rows:
            before_warning_count = len(warnings)
            run_dir = out_dir / "official_nuplan_runs" / f"scenario_{scenario.get('scenario_index', '')}" / str(prow["planner_name"])
            run_dir.mkdir(parents=True, exist_ok=True)
            ok, log_path = run_official_nuplan_cli(args.nuplan_simulation_command_template, str(prow["planner_name"]), scenario, run_dir, args.command_timeout_s, warnings, use_shell=args.nuplan_simulation_command_use_shell)
            if not ok:
                warnings.append({"type": "nuplan_cli_failed", "scenario_id": scenario.get("scenario_id", ""), "planner_name": str(prow["planner_name"]), "message": f"official nuPlan command failed; log: {log_path}"})
                status = "failed"
                parsed: List[Dict[str, Any]] = []
            else:
                official_success_count += 1
                parsed, parser_name, parser_validation = parse_official_trajectory_outputs(run_dir, scenario, prow, warnings, args.min_timesteps, allow_unsafe_pickle=args.allow_unsafe_pickle_artifacts)
                merge_parser_validation(parser_validation_total, parser_validation)
                if parsed:
                    status = "succeeded"
                    trajectory_rows.extend(parsed)
                    parser_names.append(parser_name)
                else:
                    new_warning_types = {w.get("type", "") for w in warnings[before_warning_count:]}
                    status = "parser_failed" if "artifact_parse_error" in new_warning_types else "no_trajectory_found"
                    warnings.append({"type": "no_trajectory_found", "scenario_id": scenario.get("scenario_id", ""), "planner_name": str(prow["planner_name"]), "message": f"official nuPlan command succeeded but no supported trajectory artifact was parsed under {run_dir}; log: {log_path}"})
            alignment_records.append(build_alignment_record(scenario, str(prow["planner_name"]), run_dir, ok, warnings))
            index_rows.append({"scenario_index": scenario.get("scenario_index", ""), "planner_id": prow["planner_id"], "planner_name": prow["planner_name"], "status": status, "num_timesteps": len(parsed), "warning_count": len(warnings) - before_warning_count, "db_name": scenario.get("db_name", ""), "scene_token": scenario.get("scene_token", ""), "scenario_id": scenario.get("scenario_id", ""), "sample_id": scenario.get("sample_id", "")})

    if not trajectory_rows:
        write_csv(out_dir / "scenario_planner_index.csv", index_rows, ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
        trajectory_parser_validation = finalize_parser_validation(parser_validation_total, trajectory_rows, args.min_timesteps)
        return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows, trajectory_parser_validation, official_success_count, alignment_records)

    if importlib.util.find_spec("numpy") is None:
        warnings.append({"type": "missing_numpy", "scenario_id": "", "planner_name": "", "message": "Parsed official trajectories, but NumPy is required to write non-empty simulated_ego_seq.npy."})
        write_csv(out_dir / "scenario_planner_index.csv", index_rows, ["scenario_index", "planner_id", "planner_name", "status", "num_timesteps", "warning_count", "db_name", "scene_token", "scenario_id", "sample_id"])
        trajectory_parser_validation = finalize_parser_validation(parser_validation_total, trajectory_rows, args.min_timesteps)
        return fail_outputs(out_dir, args, metadata, planners, discovery, warnings, planner_rows, trajectory_parser_validation, official_success_count, alignment_records)

    trajectory_parser_validation = finalize_parser_validation(parser_validation_total, trajectory_rows, args.min_timesteps)
    write_csv(out_dir / "simulated_ego_trajectory.csv", trajectory_rows, CSV_COLUMNS)
    tensor_info = build_simulated_seq(trajectory_rows, out_dir / "simulated_ego_seq.npy")
    shape = tensor_info["shape"]
    mask_shape = tensor_info["mask_shape"]
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
    alignment_summary = write_alignment_outputs(out_dir, metadata, alignment_records, official_success_count)
    alignment_passed = bool(alignment_records) and all(r.get("same_log_alignment_passed") is True for r in alignment_records)
    strict_alignment_passed = bool(alignment_records) and all(r.get("strict_nuplan_token_alignment_passed") is True for r in alignment_records)

    smoke_pass_ok = (
        official_success_count > 0
        and bool(trajectory_rows)
        and len(shape) == 4
        and shape[0] == len(tensor_info["scenario_axis"])
        and shape[1] == len(tensor_info["planner_axis"])
        and shape[3] == len(EGO_STATE_CHANNELS)
        and (out_dir / "simulated_ego_seq_mask.npy").is_file()
        and mask_shape == shape[:3]
        and tensor_info["valid_timestep_count"] > 0
        and trajectory_parser_validation["required_pose_valid_ratio"] > 0
        and trajectory_parser_validation["x_non_sentinel_ratio"] > 0
        and trajectory_parser_validation["y_non_sentinel_ratio"] > 0
        and trajectory_parser_validation["yaw_non_sentinel_ratio"] > 0
        and trajectory_parser_validation["num_trajectories_with_too_few_steps"] == 0
    )
    pass_ok = smoke_pass_ok and (alignment_passed if args.require_same_scenario_alignment else True) and (strict_alignment_passed if args.require_strict_nuplan_token_alignment else True)
    schema = {
        "stage": "7C.1",
        "feature_type": "nuplan_closed_loop_simulated_ego_trajectory",
        "input_stage": "7B.4",
        "uses_official_nuplan_simulation": True,
        "pseudo_rollout": False,
        "trajectory_parser": sorted(set(parser_names)),
        "required_pose_fields": ["x", "y", "yaw"],
        "optional_sentinel_fields": ["speed", "acceleration", "steering_angle_or_curvature_if_available"],
        "min_timesteps": args.min_timesteps,
        "num_input_scenarios": len(metadata),
        "num_simulated_scenarios": len(tensor_info["scenario_axis"]),
        "num_planners": len(planners),
        "planner_names": planners,
        "ego_state_channels": EGO_STATE_CHANNELS,
        "sentinel_value": SENTINEL,
        "scenario_selection_keys": SCENARIO_KEYS,
        "simulated_ego_seq_shape": list(shape),
        "scenario_axis": tensor_info["scenario_axis"],
        "planner_axis": tensor_info["planner_axis"],
        "scenario_axis_key": "scenario_index",
        "planner_axis_key": "planner_id",
        "planner_axis_names": tensor_info["planner_axis_names"],
        "same_scenario_alignment_checked": True,
        "same_scenario_alignment_passed": alignment_passed,
        "strict_nuplan_token_alignment_passed": strict_alignment_passed,
        "same_scenario_alignment_report": "scenario_alignment_report.md",
    }
    write_json(out_dir / "simulation_schema.json", schema)
    write_json(out_dir / "warnings.json", {"warnings": warnings, "simulation_api_discovery": discovery, "planner_api_discovery": planner_rows, "scenario_alignment": {"num_target_scenarios": alignment_summary["num_target_scenarios"], "num_actual_scenarios_extracted": alignment_summary["num_actual_scenarios_extracted"], "num_aligned": alignment_summary["num_aligned"], "alignment_pass_ratio": alignment_summary["alignment_pass_ratio"], "passed": alignment_passed, "strict_nuplan_token_alignment_passed": strict_alignment_passed}, "validation": {"pass": pass_ok, "official_success_count": official_success_count, "trajectory_rows": len(trajectory_rows), "pseudo_rollout": False, "uses_official_nuplan_simulation": True, "same_scenario_alignment_required": bool(args.require_same_scenario_alignment), "strict_nuplan_token_alignment_required": bool(args.require_strict_nuplan_token_alignment), "smoke_pass": smoke_pass_ok, "tensor_validation": {"shape": list(shape), "mask_shape": list(mask_shape), "valid_timestep_count": tensor_info["valid_timestep_count"], "missing_pair_count": tensor_info["missing_pair_count"], "passed": smoke_pass_ok}}, "trajectory_parser_validation": trajectory_parser_validation})
    report_status = "PASS" if pass_ok else "FAIL"
    report = f"""# Stage 7C.1 nuPlan Closed-loop Simulation Report

## PASS/FAIL summary
{report_status} — official nuPlan simulation commands succeeded and trajectory export {'is valid' if pass_ok else 'failed validation'}. No pseudo rollout data was generated.

## Output shapes
- simulated_ego_seq.npy: `{shape}`
- simulated_ego_seq_mask.npy: `{mask_shape}`
- scenario axis size: `{len(tensor_info["scenario_axis"])}`
- planner axis size: `{len(tensor_info["planner_axis"])}`
- T_sim: `{shape[2] if len(shape) == 4 else 0}`
- C: `{shape[3] if len(shape) == 4 else 0}`
- mask valid timestep count: `{tensor_info["valid_timestep_count"]}`
- missing scenario-planner pair count: `{tensor_info["missing_pair_count"]}`

## Parsed trajectories
- official command successes: {official_success_count}
- parsed trajectory rows: {len(trajectory_rows)}
- parser: `{'; '.join(sorted(set(parser_names)))}`
- min_timesteps requirement: `{args.min_timesteps}`
- num_candidate_artifact_rows: `{trajectory_parser_validation['num_candidate_artifact_rows']}`
- num_valid_trajectory_rows: `{trajectory_parser_validation['num_valid_trajectory_rows']}`
- num_rejected_rows_invalid_required_pose: `{trajectory_parser_validation['num_rejected_rows_invalid_required_pose']}`
- msgpack_simulation_log_files_found: `{trajectory_parser_validation['msgpack_simulation_log_files_found']}`
- msgpack_simulation_log_files_parsed: `{trajectory_parser_validation['msgpack_simulation_log_files_parsed']}`
- msgpack_trajectory_rows_extracted: `{trajectory_parser_validation['msgpack_trajectory_rows_extracted']}`
- msgpack_parse_errors: `{len(trajectory_parser_validation.get('msgpack_parse_errors', []))}`
- required_pose_valid_ratio: `{trajectory_parser_validation['required_pose_valid_ratio']}`
- x_non_sentinel_ratio: `{trajectory_parser_validation['x_non_sentinel_ratio']}`
- y_non_sentinel_ratio: `{trajectory_parser_validation['y_non_sentinel_ratio']}`
- yaw_non_sentinel_ratio: `{trajectory_parser_validation['yaw_non_sentinel_ratio']}`
- min_timesteps_per_trajectory: `{trajectory_parser_validation['min_timesteps_per_trajectory']}`
- mean_timesteps_per_trajectory: `{trajectory_parser_validation['mean_timesteps_per_trajectory']}`
- num_trajectories_with_too_few_steps: `{trajectory_parser_validation['num_trajectories_with_too_few_steps']}`
- num_trajectories_with_zero_motion: `{trajectory_parser_validation['num_trajectories_with_zero_motion']}`

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
    p.add_argument("--nuplan_simulation_command_template", default="", help="Optional official nuPlan command template. Placeholders include {planner_name}, {scenario_id}, {db_name}, {scene_token}, {sample_id}, {output_dir}, plus target placeholders {target_log_name}, {target_scene_token}, {target_db_name}. Prefer shell/path-safe variants such as {target_log_name_safe}, {target_scene_token_safe}, {target_db_name_safe}; exact same-scenario nuPlan commands should use target placeholders, not raw {scenario_id}, because Hydra filter keys may need log/token values separately.")
    p.add_argument("--nuplan_simulation_command_use_shell", action="store_true", help="Run the formatted official nuPlan command through the shell. Default is false: shlex.split(command) and subprocess.run(argv, shell=False) to avoid shell metacharacter interpretation.")
    p.add_argument("--require_same_scenario_alignment", action="store_true", help="Require Stage 7C.1C same-log alignment PASS for the final Stage 7C.1 PASS. Default preserves smoke behavior and allows alignment FAIL.")
    p.add_argument("--require_strict_nuplan_token_alignment", action="store_true", help="Require Stage 7B.4 scene_token to match the actual nuPlan scenario token. Default false because Stage 7B.4 scene_token may differ from nuPlan scenario_filter.scenario_tokens.")
    p.add_argument("--command_timeout_s", type=int, default=3600)
    p.add_argument("--min_timesteps", type=int, default=2, help="Minimum parsed timesteps required for each successful scenario-planner trajectory.")
    p.add_argument("--allow_unsafe_pickle_artifacts", action="store_true", help="Parse trusted pickle/msgpack nuPlan artifacts. Pickle is unsafe and remains disabled by default.")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
