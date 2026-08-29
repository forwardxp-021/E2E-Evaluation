#!/usr/bin/env python3
"""Execute or zero-budget-preflight the one-time compliant R1 Phase-B2 smoke.

The executor is intentionally separate from the historical core-only smoke
tool.  It uses the V3-validated official nuPlan runtime and canonical official
Parquet metrics.  In execute mode it claims exactly one of the frozen 48 runs
immediately before each simulation and stops the whole batch on any technical
failure.  It does not read representation, BDD, probes, checkpoints, or RBR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

from tools.r1_context_mechanism_core import (
    assert_pair_context_identity,
    build_canonical_context_record,
    calculate_hlc_option_b,
    calculate_tsb_option_a,
    qualify_hlc_pair,
    qualify_tsb_pair,
)
from tools.r1_official_metric_canonicalizer import MetricCanonicalizationError, canonicalize_official_metrics, sha256_file
from tools.r1_official_technical_smoke_planner import (
    ALLOWED_ARMS,
    EPISODE_DURATION_SECONDS,
    EPISODE_FRAME_COUNT,
    HLC_BASELINE,
    HLC_TREATMENT,
    R1OfficialTechnicalSmokePlanner,
    SAMPLING_TIME_SECONDS,
    TSB_BASELINE,
    TSB_TREATMENT,
    _episode_times,
    hlc_progress,
    tsb_profile,
)
from tools.r1_run_runtime_determinism_validation import _read_trace, read_json, write_json
from tools.r1_runtime_determinism_planner import _extended_polyline, canonical_sha256
from tools.stage7_m6_4b_run_locked_rollouts import stage7c_environment
from tools.stage7l_pure_lateral_execution_planner import derive_trajectory_states


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_ROSTER = R1_DIR / "r1_official_technical_smoke_roster_v1.0.json"
DEFAULT_SCOPE = R1_DIR / "r1_official_technical_smoke_scope_amendment_v1.0.json"
DEFAULT_SELECTOR = R1_DIR / "r1_future_compliant_smoke_selector_contract_v0.3.json"
DEFAULT_REPLAY = R1_DIR / "r1_official_nuplan_replay_contract_v1.0.json"
DEFAULT_PRELIGHT = R1_DIR / "r1_official_technical_smoke_preflight_v1.0.json"
DEFAULT_AUTHORIZATION = R1_DIR / "r1_official_compliant_technical_smoke_authorization_v1.0.json"
DEFAULT_OUTPUT = ROOT / "outputs/r1_official_compliant_technical_smoke_v1"
DEFAULT_LEDGER = R1_DIR / "r1_official_technical_smoke_run_ledger_v1.0.csv"
DEFAULT_PAIR = R1_DIR / "r1_official_technical_smoke_pair_metrics_v1.0.csv"
DEFAULT_CONTEXT = R1_DIR / "r1_official_technical_smoke_context_identity_v1.0.csv"
DEFAULT_SAFETY = R1_DIR / "r1_official_technical_smoke_safety_v1.0.csv"
DEFAULT_MANIFEST = R1_DIR / "r1_official_technical_smoke_execution_manifest_v1.0.json"
RUN_CAP = 48
WINDOW_CONVENTION = "first planner-output trajectory: np.arange(0.0,8.0,0.1), 80 samples [0.0,8.0)"

LEDGER_FIELDS = (
    "run_id", "scenario_token", "log_id", "family", "smoke_arm", "claim_status", "actual_run_number", "execution_status",
    "official_command_return_code", "technical_failure_status", "technical_failure_reasons", "trace_sha256", "planner_binding_sha256", "canonical_metric_payload_sha256", "command_log_sha256",
)
PAIR_FIELDS = (
    "scenario_token", "log_id", "family", "baseline_run_id", "treatment_run_id", "technical_execution_pass", "context_identity_pass", "mechanism_pair_status", "mechanism_pair_pass", "primary_f_match_status", "primary_f_match_pass", "secondary_heading_change_abs_total_delta", "endpoint_status", "endpoint_pass", "engineering_status", "engineering_pass", "pair_readiness",
)
CONTEXT_FIELDS = ("scenario_token", "log_id", "family", "baseline_run_id", "treatment_run_id", "raw_history_canonical_hash_equal", "canonical_context_json_hash_equal", "pair_context_identity_pass", "baseline_raw_history_canonical_hash", "treatment_raw_history_canonical_hash", "baseline_canonical_context_json_hash", "treatment_canonical_context_json_hash")
SAFETY_FIELDS = ("scenario_token", "log_id", "family", "baseline_run_id", "treatment_run_id", "baseline_at_fault_collision_count", "treatment_at_fault_collision_count", "baseline_drivable_area_compliance", "treatment_drivable_area_compliance", "baseline_safety_pass", "treatment_safety_pass", "pair_safety_pass")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(command: Sequence[str]) -> str:
    return subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()


def _path_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required bound artifact is missing: {path}")
    return sha256_file(path)


def _schedule(roster: Mapping[str, Any]) -> List[Dict[str, str]]:
    entries = list(roster.get("entries", []))
    if len(entries) != 24:
        raise ValueError("frozen smoke roster must contain exactly 24 identities")
    schedule: List[Dict[str, str]] = []
    for entry in entries:
        family = str(entry["family"])
        arms = [str(value) for value in entry.get("arms", [])]
        if family not in ALLOWED_ARMS or set(arms) != ALLOWED_ARMS[family] or len(arms) != 2:
            raise ValueError(f"roster arms are not the frozen two-arm set: {entry.get('scenario_token')}")
        for arm in arms:
            schedule.append({
                "run_id": f"{family}__{entry['scenario_token']}__{arm}", "scenario_token": str(entry["scenario_token"]), "log_id": str(entry["log_id"]),
                "family": family, "smoke_arm": arm, "db_path": str(entry["db_path"]),
            })
    if len(schedule) != RUN_CAP or len({row["run_id"] for row in schedule}) != RUN_CAP:
        raise ValueError("exact 48-run schedule with unique run IDs is required")
    if len({(row["scenario_token"], row["smoke_arm"]) for row in schedule}) != RUN_CAP:
        raise ValueError("duplicate scenario arm in frozen smoke schedule")
    for token in {row["scenario_token"] for row in schedule}:
        if sum(row["scenario_token"] == token for row in schedule) != 2:
            raise ValueError("each frozen scenario must have exactly two arms")
    return schedule


@dataclass
class OfficialSmokeBudget:
    schedule: Sequence[Mapping[str, str]]
    path: Path
    records: List[Dict[str, Any]]

    @classmethod
    def create(cls, schedule: Sequence[Mapping[str, str]], path: Path) -> "OfficialSmokeBudget":
        budget = cls(schedule=schedule, path=path, records=[])
        budget.flush()
        return budget

    def flush(self) -> None:
        write_json(self.path, {"schema_version": "r1_official_technical_smoke_budget_v1.0", "unit": "OFFICIAL_CLOSED_LOOP_RUN", "authorized_cap": RUN_CAP, "claimed_count": len(self.records), "records": self.records})

    def claim(self, run: Mapping[str, str]) -> None:
        expected = {str(item["run_id"]) for item in self.schedule}
        run_id = str(run["run_id"])
        if len(self.records) >= RUN_CAP:
            raise RuntimeError("OFFICIAL_CLOSED_LOOP_RUN cap 48 reached before simulation start")
        if run_id not in expected:
            raise RuntimeError(f"refusing unplanned official run: {run_id}")
        if any(str(row["run_id"]) == run_id for row in self.records):
            raise RuntimeError(f"refusing duplicate official run: {run_id}")
        record = dict(run)
        record.update({"claim_status": "CLAIMED_BEFORE_SIMULATION", "actual_run_number": len(self.records) + 1, "execution_status": "CLAIMED_NOT_STARTED", "official_command_return_code": "", "technical_failure_status": "", "technical_failure_reasons": "", "trace_sha256": "", "planner_binding_sha256": "", "canonical_metric_payload_sha256": "", "command_log_sha256": ""})
        self.records.append(record)
        self.flush()

    def complete(self, run_id: str, summary: Mapping[str, Any]) -> None:
        row = next(item for item in self.records if str(item["run_id"]) == run_id)
        row.update({"execution_status": "EXECUTED", "official_command_return_code": summary["official_command_return_code"], "technical_failure_status": summary["technical_failure_status"], "technical_failure_reasons": " | ".join(summary["technical_failure_reasons"]), "trace_sha256": summary.get("trace_sha256") or "", "planner_binding_sha256": summary.get("planner_binding_sha256") or "", "canonical_metric_payload_sha256": summary.get("metrics", {}).get("canonical_payload_sha256") or "", "command_log_sha256": summary.get("command_log_sha256") or ""})
        self.flush()

    def assert_49th_rejected(self) -> Dict[str, str]:
        saved = list(self.records)
        self.records = [dict(item) for item in self.schedule]
        try:
            self.claim({"run_id": "R1_FORBIDDEN_49TH_PRE_RUN_CLAIM"})
        except RuntimeError as exc:
            self.records = saved
            return {"status": "REJECTED_BEFORE_SIMULATION", "reason": str(exc)}
        finally:
            self.records = saved
        raise RuntimeError("49th pre-run claim did not fail closed")


def _command_for(args: argparse.Namespace, run: Mapping[str, str], run_dir: Path) -> List[str]:
    hydra = f"[file://{(args.project_root / 'configs/r1_official_technical_smoke_hydra').resolve()},pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
    return [
        str(args.python_executable.resolve()), str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()),
        "+simulation=closed_loop_nonreactive_agents", "planner=r1_official_technical_smoke", "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{Path(run['db_path']).resolve()}]", "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "worker=single_machine_thread_pool", "worker.max_workers=1", "scenario_builder.max_workers=1", "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1", "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701", "run_metric=true", "enable_simulation_progress_bar=false",
        "experiment_name=r1_official_compliant_technical_smoke_v1", f"job_name={run['run_id']}", f"output_dir={run_dir / 'nuplan_output'}", f"hydra.searchpath={hydra}",
    ]


def _context_record(entry: Mapping[str, Any], first_trace: Mapping[str, Any]) -> Dict[str, Any]:
    history = list(first_trace.get("initial_history_canonical", []))
    observations = list(first_trace.get("pre_context_raw", []))
    if len(history) < 11 or len(observations) < 11:
        raise ValueError("official history buffer cannot provide the frozen ten-frame pre-context")
    pre_ego, pre_obs = history[-11:-1], observations[-11:-1]
    if len(pre_ego) != 10 or len(pre_obs) != 10:
        raise ValueError("pre-context extraction is not exactly ten frames")
    frames: List[Dict[str, Any]] = []
    for index, (ego, observation) in enumerate(zip(pre_ego, pre_obs)):
        frame: Dict[str, Any] = {
            "time_s": round(index * SAMPLING_TIME_SECONDS, 6), "ego_valid": True, "map_valid": True, "current_required_lane_valid": True,
            "speed_mps": float(ego["speed_mps"]), "lane_offset_m": 0.0, "legal_projected_dynamic_vehicle_count": len(observation),
            "slots": {name: {"valid": False} for name in ("front", "left_front", "left_rear", "right_front", "right_rear")},
        }
        if entry["family"] == "R-HLC":
            frame.update({"target_front": {"valid": False}, "target_rear": {"valid": False}})
        else:
            frame.update({"front": {"valid": False}})
        frames.append(frame)
    payload: Dict[str, Any] = {
        "family": str(entry["family"]), "scenario_token": str(entry["scenario_token"]), "map_version": str(entry["map_name"]), "route_fingerprint": str(entry["route_fingerprint"]), "initial_state_fingerprint": str(entry["initial_state"]["initial_state_fingerprint"]),
        "map_location": str(entry["map_name"]), "road_class": "OFFICIAL_MAP_ROAD_CLASS_NOT_REQUIRED_FOR_SMOKE_ELIGIBILITY", "log_id": str(entry["log_id"]), "query_version": "r1_official_technical_smoke_context_adapter_v1", "history_source": "OFFICIAL_HISTORY_BUFFER", "t_anchor_s": 1.0, "frames": frames,
        "map_source_ids": dict(entry.get("relevant_context_ids", {})),
    }
    if entry["family"] == "R-HLC":
        payload["intended_lane_change_direction"] = str(entry["direction"]).upper()
    else:
        payload["hazard_multi_hot"] = ["NONE_OBSERVED"]
    canonical = build_canonical_context_record(payload)
    raw = {"initial_history": pre_ego, "observations": pre_obs}
    return {"pre_context_raw_hash": canonical_sha256(raw), "canonical_context_json_hash": canonical["canonical_context_json_hash"], "canonical_context": canonical, "raw_history_frame_count": len(pre_ego)}


def _planner_window(first_trace: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(first_trace.get("planner_output_trajectory", []))
    if len(rows) != EPISODE_FRAME_COUNT:
        raise ValueError(f"planner output must contain exactly frozen 80 samples, got {len(rows)}")
    xy = np.asarray([[row["rear_axle"]["x"], row["rear_axle"]["y"]] for row in rows], dtype=np.float64)
    heading = np.asarray([row["rear_axle"]["heading"] for row in rows], dtype=np.float64)
    speed = np.asarray([row["speed_mps"] for row in rows], dtype=np.float64)
    if not np.isfinite(xy).all() or not np.isfinite(heading).all() or not np.isfinite(speed).all():
        raise ValueError("planner output has non-finite values")
    return _episode_times(), xy, heading, speed


def _ego13_descriptors(time_s: np.ndarray, xy: np.ndarray, speed: np.ndarray) -> Dict[str, float]:
    accel = np.diff(speed, prepend=speed[0]) / SAMPLING_TIME_SECONDS
    heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1], time_s, edge_order=2), np.gradient(xy[:, 0], time_s, edge_order=2)))
    return {"mean_speed": round(float(np.mean(speed)), 6), "end_minus_start_speed": round(float(speed[-1] - speed[0]), 6), "mean_abs_accel": round(float(np.mean(np.abs(accel))), 6), "heading_change_abs_total": round(float(np.sum(np.abs(np.diff(heading)))), 6), "path_length": round(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))), 6)}


def _f_match(baseline: Mapping[str, float], treatment: Mapping[str, float], family: str) -> Dict[str, Any]:
    calipers = {"mean_speed": 0.708203939, "end_minus_start_speed": 0.978755681, "path_length": 5.38423459}
    if family == "R-TSB":
        calipers["mean_abs_accel"] = 0.11777666
    delta = {key: round(abs(float(treatment[key]) - float(baseline[key])), 6) for key in calipers}
    pass_by_feature = {key: value <= calipers[key] + 1e-12 for key, value in delta.items()}
    return {"status": "F_MATCH_PASS" if all(pass_by_feature.values()) else "F_MATCH_FAIL", "pass": all(pass_by_feature.values()), "calipers": calipers, "absolute_delta": delta, "pass_by_feature": pass_by_feature}


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _hlc_measurement(entry: Mapping[str, Any], time_s: np.ndarray, xy: np.ndarray, speed: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    initial = entry["initial_state"]
    arc = float(initial["initial_speed_mps"]) * time_s
    source = _extended_polyline(entry["source_reference_xy"], float(entry["source_start_arc_m"]) + arc)
    target = _extended_polyline(entry["target_reference_xy"], float(entry["target_start_arc_m"]) + arc)
    delta = target - source
    denominator = np.sum(delta * delta, axis=1)
    if np.any(denominator <= 1e-9):
        raise ValueError("HLC native lane separation is not evaluable")
    progress = np.sum((xy - source) * delta, axis=1) / denominator
    mechanism = calculate_hlc_option_b(time_s, progress, speed, map_valid=True)
    target_heading = float(np.arctan2(target[-1, 1] - target[-2, 1], target[-1, 0] - target[-2, 0]))
    final_velocity = (xy[-1] - xy[-2]) / SAMPLING_TIME_SECONDS
    normal = np.asarray([-np.sin(target_heading), np.cos(target_heading)])
    endpoint = {
        "terminal_lateral_offset_to_target_center_m": round(float(np.linalg.norm(xy[-1] - target[-1])), 6),
        "terminal_heading_error_rad": round(abs(_wrap_angle(float(np.arctan2(xy[-1, 1] - xy[-2, 1], xy[-1, 0] - xy[-2, 0])) - target_heading)), 6),
        "terminal_lateral_velocity_mps": round(abs(float(np.dot(final_velocity, normal))), 6),
        "complete_target_lane_transition": bool(progress[-1] >= 0.75),
        "endpoint_limits": {"offset_m_max": 0.25, "heading_error_rad_max": 0.05, "lateral_velocity_mps_max": 0.25},
    }
    states = derive_trajectory_states(xy, time_s, wheel_base_m=3.0)
    engineering = {"max_abs_lateral_accel_mps2": round(float(np.max(np.abs(states["lateral_accel"]))), 6), "max_abs_yaw_rate_radps": round(float(np.max(np.abs(states["yaw_rate"]))), 6), "max_abs_curvature_inv_m": round(float(np.max(np.abs(states["curvature"]))), 6)}
    return mechanism, endpoint, engineering


def _technical_summary(run: Mapping[str, str], return_code: int, run_dir: Path) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        trace = _read_trace(run_dir / "trace" / "planner_trace.jsonl")
        first_trace = trace[0]
        _planner_window(first_trace)
    except Exception as exc:
        trace, first_trace = [], {}
        failures.append(f"TRACE_OR_80_FRAME_WINDOW_UNAVAILABLE:{type(exc).__name__}:{exc}")
    try:
        binding = read_json(run_dir / "trace" / "planner_binding.json")
        if binding.get("episode_window", {}).get("frame_count") != EPISODE_FRAME_COUNT:
            raise ValueError("planner binding does not declare 80 frozen samples")
    except Exception as exc:
        binding = {}
        failures.append(f"BINDING_UNAVAILABLE_OR_INVALID:{type(exc).__name__}:{exc}")
    try:
        metrics = canonicalize_official_metrics(run_dir)
    except MetricCanonicalizationError as exc:
        metrics = {"canonical_payload": {}, "canonical_payload_sha256": None, "artifact_provenance": {}}
        failures.append(f"CANONICAL_METRIC_PARSER_FAILURE:{exc}")
    if return_code != 0:
        failures.append(f"OFFICIAL_COMMAND_RETURN_CODE_{return_code}")
    return {"run_id": run["run_id"], "scenario_token": run["scenario_token"], "log_id": run["log_id"], "family": run["family"], "smoke_arm": run["smoke_arm"], "official_command_return_code": return_code, "technical_failure_status": "NO_TECHNICAL_FAILURE" if not failures else "TECHNICAL_FAILURE", "technical_failure_reasons": failures, "trace": trace, "first_trace": first_trace, "binding": binding, "planner_binding_sha256": canonical_sha256(binding) if binding else None, "metrics": metrics, "trace_sha256": sha256_file(run_dir / "trace" / "planner_trace.jsonl") if (run_dir / "trace" / "planner_trace.jsonl").is_file() else None, "command_log_sha256": sha256_file(run_dir / "official_run.log") if (run_dir / "official_run.log").is_file() else None}


def _evaluate_pair(entry: Mapping[str, Any], baseline: Mapping[str, Any], treatment: Mapping[str, Any]) -> Dict[str, Any]:
    technical = baseline["technical_failure_status"] == "NO_TECHNICAL_FAILURE" and treatment["technical_failure_status"] == "NO_TECHNICAL_FAILURE"
    if not technical:
        return {"technical_execution_pass": False, "pair_readiness": "NOT_EVALUABLE_TECHNICAL_FAILURE"}
    base_context, treatment_context = _context_record(entry, baseline["first_trace"]), _context_record(entry, treatment["first_trace"])
    identity = assert_pair_context_identity(base_context, treatment_context)
    time_s, base_xy, _, base_speed = _planner_window(baseline["first_trace"])
    _, treatment_xy, _, treatment_speed = _planner_window(treatment["first_trace"])
    base_desc, treatment_desc = _ego13_descriptors(time_s, base_xy, base_speed), _ego13_descriptors(time_s, treatment_xy, treatment_speed)
    fmatch = _f_match(base_desc, treatment_desc, str(entry["family"]))
    if entry["family"] == "R-HLC":
        base_mech, base_endpoint, base_engineering = _hlc_measurement(entry, time_s, base_xy, base_speed)
        treatment_mech, treatment_endpoint, treatment_engineering = _hlc_measurement(entry, time_s, treatment_xy, treatment_speed)
        mechanism = qualify_hlc_pair(base_mech, treatment_mech)
        route_delta = abs(base_desc["path_length"] - treatment_desc["path_length"])
        endpoint_pass = all([base_endpoint["complete_target_lane_transition"], treatment_endpoint["complete_target_lane_transition"], base_endpoint["terminal_lateral_offset_to_target_center_m"] <= 0.25, treatment_endpoint["terminal_lateral_offset_to_target_center_m"] <= 0.25, base_endpoint["terminal_heading_error_rad"] <= 0.05, treatment_endpoint["terminal_heading_error_rad"] <= 0.05, base_endpoint["terminal_lateral_velocity_mps"] <= 0.25, treatment_endpoint["terminal_lateral_velocity_mps"] <= 0.25, route_delta <= 1.5])
        endpoint = {"status": "HLC_ENDPOINT_PASS" if endpoint_pass else "HLC_ENDPOINT_FAIL", "pass": endpoint_pass, "baseline": base_endpoint, "treatment": treatment_endpoint, "paired_route_progress_delta_m": round(route_delta, 6)}
        engineering_pass = all(value <= limit + 1e-12 for record in (base_engineering, treatment_engineering) for value, limit in ((record["max_abs_lateral_accel_mps2"], 6.0), (record["max_abs_yaw_rate_radps"], 1.0), (record["max_abs_curvature_inv_m"], 0.5)))
        engineering = {"status": "ENGINEERING_PASS" if engineering_pass else "ENGINEERING_FAIL", "pass": engineering_pass, "baseline": base_engineering, "treatment": treatment_engineering}
        secondary = round(abs(treatment_desc["heading_change_abs_total"] - base_desc["heading_change_abs_total"]), 6)
    else:
        base_mech = calculate_tsb_option_a(time_s, base_speed)
        treatment_mech = calculate_tsb_option_a(time_s, treatment_speed)
        mechanism = qualify_tsb_pair(base_mech, treatment_mech)
        endpoint, engineering, secondary = {"status": "NOT_APPLICABLE_TSB", "pass": True}, {"status": "DIAGNOSTIC_ONLY_TSB", "pass": True}, None
    payload = {"scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "family": entry["family"], "baseline_run_id": baseline["run_id"], "treatment_run_id": treatment["run_id"], "technical_execution_pass": technical, "context": {"baseline": base_context, "treatment": treatment_context, "identity": identity}, "descriptors": {"baseline": base_desc, "treatment": treatment_desc}, "mechanism": {"baseline": base_mech, "treatment": treatment_mech, "pair": mechanism}, "f_match": fmatch, "secondary_heading_change_abs_total_delta": secondary, "endpoint": endpoint, "engineering": engineering, "baseline_safety": baseline["metrics"]["canonical_payload"], "treatment_safety": treatment["metrics"]["canonical_payload"]}
    safety = payload["baseline_safety"]["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0 and payload["treatment_safety"]["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0 and payload["baseline_safety"]["drivable_area"]["drivable_area_compliance_stat_value"] and payload["treatment_safety"]["drivable_area"]["drivable_area_compliance_stat_value"]
    payload["safety"] = {"pass": bool(safety)}
    all_gate = technical and identity["pair_context_identity_pass"] and mechanism["pass"] and fmatch["pass"] and endpoint["pass"] and engineering["pass"] and safety
    payload["pair_readiness"] = "PAIR_ALL_REQUIRED_GATES_PASS" if all_gate else "PAIR_REQUIRED_GATE_FAILURE"
    return payload


def _preflight(args: argparse.Namespace) -> Dict[str, Any]:
    required = [args.roster, args.scope_amendment, args.selector_contract, args.replay_contract]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preflight required artifact missing: {missing}")
    roster, scope, selector, replay = (read_json(path) for path in required)
    schedule = _schedule(roster)
    if scope.get("only_legal_scope", {}).get("total_official_closed_loop_runs") != RUN_CAP:
        raise ValueError("scope amendment does not bind 48 official runs")
    if selector.get("selection_scope", {}).get("scenarios_per_family") != 12 or selector.get("selection_scope", {}).get("arms_per_scenario") != 2:
        raise ValueError("selector contract does not bind 12 scenarios x 2 arms")
    if replay.get("official_replay") != "READY_FOR_TECHNICAL_SMOKE_REVIEW":
        raise ValueError("V3 replay contract is not ready for technical smoke review")
    if not issubclass(R1OfficialTechnicalSmokePlanner, AbstractPlanner):
        raise TypeError("official smoke planner is not a legal AbstractPlanner")
    entry = roster["entries"][0]
    dummy = type("FrozenScenario", (), {"token": entry["scenario_token"]})()
    planner = R1OfficialTechnicalSmokePlanner(dummy, str(args.roster), str(entry["family"]), str(entry["arms"][0]), str(args.output.parent / "synthetic_trace_not_executed"))
    if planner.name() == "":
        raise RuntimeError("official smoke planner failed construction")
    t = _episode_times()
    hlc_baseline, hlc_treatment = hlc_progress(t, HLC_BASELINE), hlc_progress(t, HLC_TREATMENT)
    if not qualify_hlc_pair(calculate_hlc_option_b(t, hlc_baseline, np.full_like(t, 10.0)), calculate_hlc_option_b(t, hlc_treatment, np.full_like(t, 10.0)))["pass"]:
        raise RuntimeError("frozen HLC generator cannot satisfy its frozen synthetic mechanism preflight")
    _, tsb_base_speed, _ = tsb_profile(t, TSB_BASELINE, 10.0)
    _, tsb_treatment_speed, _ = tsb_profile(t, TSB_TREATMENT, 10.0)
    if not qualify_tsb_pair(calculate_tsb_option_a(t, tsb_base_speed), calculate_tsb_option_a(t, tsb_treatment_speed))["pass"]:
        raise RuntimeError("frozen TSB generator cannot satisfy its frozen synthetic mechanism preflight")
    synthetic_context = {"family": "R-TSB", "scenario_token": "synthetic", "map_version": "synthetic", "route_fingerprint": "synthetic", "initial_state_fingerprint": "synthetic", "map_location": "synthetic", "road_class": "synthetic", "log_id": "synthetic", "query_version": "preflight", "t_anchor_s": 1.0, "frames": [{"time_s": round(i * 0.1, 6), "ego_valid": True, "map_valid": True, "current_required_lane_valid": True, "speed_mps": 10.0, "lane_offset_m": 0.0, "legal_projected_dynamic_vehicle_count": 0, "slots": {}, "front": {"valid": False}} for i in range(10)], "hazard_multi_hot": ["NONE_OBSERVED"]}
    context = build_canonical_context_record(synthetic_context)
    if not context["eligible"] or not callable(canonicalize_official_metrics) or not inspect.isfunction(_f_match):
        raise RuntimeError("required canonicalizer/evaluator is not callable")
    budget = OfficialSmokeBudget.create(schedule, args.output.parent / "r1_preflight_synthetic_budget_not_official.json")
    cap = budget.assert_49th_rejected()
    (args.output.parent / "r1_preflight_synthetic_budget_not_official.json").unlink()
    result = {"schema_version": "r1_official_technical_smoke_preflight_v1.0", "status": "PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED", "actual_official_run_count": 0, "official_run_budget_claimed": 0, "planner_abstract_planner_pass": True, "frozen_contract_hashes": {"roster": _path_hash(args.roster), "scope_amendment": _path_hash(args.scope_amendment), "selector_contract": _path_hash(args.selector_contract), "replay_contract": _path_hash(args.replay_contract), "metric_canonicalizer": _path_hash(ROOT / "tools/r1_official_metric_canonicalizer.py"), "context_mechanism_contract": _path_hash(R1_DIR / "r1_context_contract_v1.0.json"), "hlc_generator": _path_hash(R1_DIR / "r1_hlc_generator_v2_contract_v1.0.json"), "hlc_endpoint": _path_hash(R1_DIR / "r1_hlc_generator_endpoint_validity_v1.0.json"), "tsb_generator": _path_hash(R1_DIR / "r1_tsb_generator_v2_contract_v1.0.json"), "planner": _path_hash(ROOT / "tools/r1_official_technical_smoke_planner.py"), "executor": _path_hash(Path(__file__)), "simulation_config": _path_hash(ROOT / "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke.yaml")}, "roster_checks": {"entries": len(roster["entries"]), "schedule": len(schedule), "unique_run_ids": len({row["run_id"] for row in schedule}), "unique_scenario_arms": len({(row["scenario_token"], row["smoke_arm"]) for row in schedule}), "each_scenario_exactly_two_arms": True}, "temporal_window": {"duration_seconds": EPISODE_DURATION_SECONDS, "dt_seconds": SAMPLING_TIME_SECONDS, "frame_count": EPISODE_FRAME_COUNT, "convention": WINDOW_CONVENTION}, "evaluator_checks": {"metric_canonicalizer_callable": True, "context_canonicalizer_callable": True, "mechanism_evaluator_callable": True, "f_match_evaluator_callable": True, "hlc_endpoint_evaluator_callable": True}, "frozen_generator_synthetic_mechanism_preflight": {"HLC_OPTION_B": "PASS", "TSB_OPTION_A": "PASS"}, "49th_pre_run_claim": cap, "git": {"commit": _git(["rev-parse", "HEAD"]), "tree": _git(["rev-parse", "HEAD^{tree}"])} }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight output: {args.output}")
    write_json(args.output, result)
    return result


def _validate_authorization(args: argparse.Namespace, roster: Mapping[str, Any]) -> Dict[str, Any]:
    authority = read_json(args.authorization)
    if authority.get("status") != "AUTHORIZED_ONCE_AFTER_ZERO_BUDGET_PREFLIGHT":
        raise ValueError("one-time technical-smoke authorization is absent or invalid")
    binding = authority.get("binding", {})
    expected = {"roster_sha256": _path_hash(args.roster), "scope_amendment_sha256": _path_hash(args.scope_amendment), "selector_contract_sha256": _path_hash(args.selector_contract), "replay_contract_sha256": _path_hash(args.replay_contract), "metric_canonicalizer_sha256": _path_hash(ROOT / "tools/r1_official_metric_canonicalizer.py"), "planner_sha256": _path_hash(ROOT / "tools/r1_official_technical_smoke_planner.py"), "smoke_executor_sha256": _path_hash(Path(__file__)), "simulation_config_sha256": _path_hash(ROOT / "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke.yaml"), "preflight_sha256": _path_hash(DEFAULT_PRELIGHT)}
    mismatches = {key: {"expected": value, "actual": binding.get(key)} for key, value in expected.items() if binding.get(key) != value}
    if mismatches:
        raise ValueError(f"authorization binding mismatch: {mismatches}")
    if binding.get("run_cap") != RUN_CAP or binding.get("master_seed") != 2026082701:
        raise ValueError("authorization cap or master seed differs from frozen value")
    if len(roster.get("entries", [])) != 24:
        raise ValueError("authorization cannot execute an invalid roster")
    return authority


def _execute(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output.exists() or any(path.exists() for path in (args.ledger, args.pair_metrics, args.context_identity, args.safety, args.manifest)):
        raise FileExistsError("refusing to overwrite official smoke output or result artifacts")
    roster = read_json(args.roster)
    authority = _validate_authorization(args, roster)
    schedule = _schedule(roster)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = OfficialSmokeBudget.create(schedule, args.output / "official_run_budget_v1.0.json")
    summaries: Dict[str, Dict[str, Any]] = {}
    stopped_reason: str | None = None
    for run in schedule:
        budget.claim(run)
        run_dir = args.output / "runs" / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        environment = stage7c_environment()
        environment.update({"R1_OFFICIAL_TECHNICAL_SMOKE_ROSTER": str(args.roster.resolve()), "R1_OFFICIAL_TECHNICAL_SMOKE_FAMILY": run["family"], "R1_OFFICIAL_TECHNICAL_SMOKE_ARM": run["smoke_arm"], "R1_OFFICIAL_TECHNICAL_SMOKE_TRACE_DIR": str((run_dir / "trace").resolve())})
        command = _command_for(args, run, run_dir)
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (run_dir / "official_run.log").write_text(completed.stdout, encoding="utf-8")
        summary = _technical_summary(run, completed.returncode, run_dir)
        summaries[run["run_id"]] = summary
        budget.complete(run["run_id"], summary)
        if summary["technical_failure_status"] != "NO_TECHNICAL_FAILURE":
            stopped_reason = f"TECHNICAL_FAILURE:{run['run_id']}"
            break
    pair_results: List[Dict[str, Any]] = []
    if stopped_reason is None:
        by_token = {str(row["scenario_token"]): row for row in roster["entries"]}
        for token, entry in by_token.items():
            run_rows = [row for row in schedule if row["scenario_token"] == token]
            baseline = summaries[next(row["run_id"] for row in run_rows if "BASELINE" in row["smoke_arm"])]
            treatment = summaries[next(row["run_id"] for row in run_rows if "TREATMENT" in row["smoke_arm"])]
            pair_results.append(_evaluate_pair(entry, baseline, treatment))
    ledger_rows = budget.records
    _write_csv(args.ledger, ledger_rows, LEDGER_FIELDS)
    pair_rows, context_rows, safety_rows = [], [], []
    for item in pair_results:
        pair_rows.append({"scenario_token": item["scenario_token"], "log_id": item["log_id"], "family": item["family"], "baseline_run_id": item["baseline_run_id"], "treatment_run_id": item["treatment_run_id"], "technical_execution_pass": item["technical_execution_pass"], "context_identity_pass": item["context"]["identity"]["pair_context_identity_pass"], "mechanism_pair_status": item["mechanism"]["pair"]["status"], "mechanism_pair_pass": item["mechanism"]["pair"]["pass"], "primary_f_match_status": item["f_match"]["status"], "primary_f_match_pass": item["f_match"]["pass"], "secondary_heading_change_abs_total_delta": item["secondary_heading_change_abs_total_delta"], "endpoint_status": item["endpoint"]["status"], "endpoint_pass": item["endpoint"]["pass"], "engineering_status": item["engineering"]["status"], "engineering_pass": item["engineering"]["pass"], "pair_readiness": item["pair_readiness"]})
        context_rows.append({"scenario_token": item["scenario_token"], "log_id": item["log_id"], "family": item["family"], "baseline_run_id": item["baseline_run_id"], "treatment_run_id": item["treatment_run_id"], "raw_history_canonical_hash_equal": item["context"]["identity"]["fields"]["pre_context_raw_hash"], "canonical_context_json_hash_equal": item["context"]["identity"]["fields"]["canonical_context_json_hash"], "pair_context_identity_pass": item["context"]["identity"]["pair_context_identity_pass"], "baseline_raw_history_canonical_hash": item["context"]["baseline"]["pre_context_raw_hash"], "treatment_raw_history_canonical_hash": item["context"]["treatment"]["pre_context_raw_hash"], "baseline_canonical_context_json_hash": item["context"]["baseline"]["canonical_context_json_hash"], "treatment_canonical_context_json_hash": item["context"]["treatment"]["canonical_context_json_hash"]})
        bs, ts = item["baseline_safety"], item["treatment_safety"]
        safety_rows.append({"scenario_token": item["scenario_token"], "log_id": item["log_id"], "family": item["family"], "baseline_run_id": item["baseline_run_id"], "treatment_run_id": item["treatment_run_id"], "baseline_at_fault_collision_count": bs["collision"]["number_of_all_at_fault_collisions_stat_value"], "treatment_at_fault_collision_count": ts["collision"]["number_of_all_at_fault_collisions_stat_value"], "baseline_drivable_area_compliance": bs["drivable_area"]["drivable_area_compliance_stat_value"], "treatment_drivable_area_compliance": ts["drivable_area"]["drivable_area_compliance_stat_value"], "baseline_safety_pass": bs["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0 and bs["drivable_area"]["drivable_area_compliance_stat_value"], "treatment_safety_pass": ts["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0 and ts["drivable_area"]["drivable_area_compliance_stat_value"], "pair_safety_pass": item["safety"]["pass"]})
    _write_csv(args.pair_metrics, pair_rows, PAIR_FIELDS)
    _write_csv(args.context_identity, context_rows, CONTEXT_FIELDS)
    _write_csv(args.safety, safety_rows, SAFETY_FIELDS)
    manifest = {"schema_version": "r1_official_technical_smoke_execution_manifest_v1.0", "status": "COMPLETE" if stopped_reason is None else "STOPPED_TECHNICAL_FAILURE", "stopped_reason": stopped_reason, "actual_official_run_count": len(budget.records), "authorized_run_cap": RUN_CAP, "authorization_sha256": _path_hash(args.authorization), "roster_sha256": _path_hash(args.roster), "run_budget": {"path": str((args.output / "official_run_budget_v1.0.json").resolve()), "49th_pre_run_claim": budget.assert_49th_rejected() if len(budget.records) == RUN_CAP else "NOT_REACHED_DUE_TO_TECHNICAL_FAILURE"}, "window": {"duration_seconds": EPISODE_DURATION_SECONDS, "dt_seconds": SAMPLING_TIME_SECONDS, "frame_count": EPISODE_FRAME_COUNT, "convention": WINDOW_CONVENTION}, "technical_failure_count": sum(row["technical_failure_status"] != "NO_TECHNICAL_FAILURE" for row in budget.records), "pair_results": pair_results, "authority_status": authority["status"], "git": {"commit": _git(["rev-parse", "HEAD"]), "tree": _git(["rev-parse", "HEAD^{tree}"])} }
    write_json(args.manifest, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), required=True)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--scope-amendment", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--selector-contract", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--replay-contract", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRELIGHT)
    parser.add_argument("--official-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--pair-metrics", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--context-identity", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--nuplan-devkit-root", type=Path, default=ROOT.parent / "nuplan-devkit")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        result = _preflight(args)
    else:
        args.output = args.official_output
        result = _execute(args)
    print(json.dumps({"status": result["status"], "actual_official_run_count": result.get("actual_official_run_count"), "stopped_reason": result.get("stopped_reason")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
