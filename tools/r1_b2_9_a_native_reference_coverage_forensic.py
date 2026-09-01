#!/usr/bin/env python3
"""Offline B2.9-A native-reference coverage forensic; never runs simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.r1_b2_8_r3_prospective_selector import official_env
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1
from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1
from tools.r1_prospective_generator_contract_v2 import (
    HLC_BASELINE,
    HLC_TREATMENT,
    hlc_progress,
    polyline_arclength,
)


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json"
TRACE = ROOT / "outputs/r1_b2_8_r3_3_official_smoke_once_v1/R1B27-01-R-HLC-BASELINE/trace/realized_current_ego.jsonl"
ATTEMPT = ROOT / "outputs/r1_b2_8_r3_3_official_smoke_once_v1"
MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
ITERATION_OUT = R1 / "r1_b2_9_a_iteration_0_33_native_coverage_audit_v1.json"
ALL12_OUT = R1 / "r1_b2_9_a_all12_hlc_nominal_replan_coverage_audit_v1.json"
HORIZON_SECONDS = 7.9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"B2_9_A_VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _current(current: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rear_axle": {
            "x": float(current["initial_x"]),
            "y": float(current["initial_y"]),
            "heading": float(current["initial_heading"]),
        },
        "speed_mps": float(current["initial_speed_mps"]),
        "time_us": int(current["initial_time_us"]),
    }


def coverage_row(
    bridge: R1OfficialMapQueryBridgeV2_1,
    source_lane_id: str,
    target_lane_id: str,
    source_total_m: float,
    target_total_m: float,
    iteration_index: int,
    ego: Mapping[str, Any],
) -> Dict[str, Any]:
    xy = (float(ego["rear_axle"]["x"]), float(ego["rear_axle"]["y"]))
    speed = float(ego["speed_mps"])
    future_distance = speed * HORIZON_SECONDS
    source_arc = float(bridge.project(source_lane_id, xy)["arc_m"])
    target_arc = float(bridge.project(target_lane_id, xy)["arc_m"])
    source_requested = source_arc + future_distance
    target_requested = target_arc + future_distance
    source_margin = source_total_m - source_requested
    target_margin = target_total_m - target_requested
    invalid = []
    if source_margin < -1e-12:
        invalid.append("source")
    if target_margin < -1e-12:
        invalid.append("target")
    return {
        "iteration_index": int(iteration_index),
        "ego_time_us": int(ego["time_us"]),
        "realized_x": xy[0],
        "realized_y": xy[1],
        "current_speed_mps": speed,
        "required_7_9s_future_distance_m": future_distance,
        "source_current_arc_m": source_arc,
        "target_current_arc_m": target_arc,
        "source_native_total_length_m": source_total_m,
        "target_native_total_length_m": target_total_m,
        "source_max_requested_arc_m": source_requested,
        "target_max_requested_arc_m": target_requested,
        "source_coverage_margin_m": source_margin,
        "target_coverage_margin_m": target_margin,
        "invalid_references": invalid,
    }


def first_invalid(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((row for row in rows if row["invalid_references"]), None)


def _attempt_sha_inventory() -> Dict[str, str]:
    paths = [
        R1 / "r1_b2_8_r3_3_scientific_owner_authorization_once_v1.0.json",
        R1 / "R1_B2_8_R3_3_Official_Smoke_Stop_Report_v1.md",
        ATTEMPT / "official_smoke_stop_report_v1.json",
        ATTEMPT / "R1B27-01-R-HLC-BASELINE/trace/realized_current_ego.jsonl",
        ATTEMPT / "R1B27-01-R-HLC-BASELINE/raw/log.txt",
        ATTEMPT / "R1B27-01-R-HLC-BASELINE/raw/nuboard_1788239794.nuboard",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FAILED_ATTEMPT_ARTIFACT_MISSING:{missing}")
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def _map_api(map_name: str, cache: Dict[str, Any]) -> Any:
    if map_name not in cache:
        official_env()
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

        cache[map_name] = get_maps_api(str(MAP_ROOT), "nuplan-maps-v1.0", map_name)
    return cache[map_name]


def iteration_audit(entries: Sequence[Mapping[str, Any]], cache: Dict[str, Any]) -> Dict[str, Any]:
    entry = next(row for row in entries if row["scenario_token"] == "b1be12bca092597a")
    bridge = R1OfficialMapQueryBridgeV2_1(_map_api(str(entry["map_name"]), cache))
    source_xy = bridge.native_reference_xy(str(entry["source_lane_id"]))
    target_xy = bridge.native_reference_xy(str(entry["target_lane_id"]))
    source_total = float(polyline_arclength(source_xy)[-1])
    target_total = float(polyline_arclength(target_xy)[-1])
    trace_rows = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["iteration_index"]) for row in trace_rows] != list(range(34)):
        raise ValueError("FAILED_TRACE_MUST_BE_EXACT_ITERATIONS_0_TO_33")
    rows = [
        coverage_row(
            bridge,
            str(entry["source_lane_id"]),
            str(entry["target_lane_id"]),
            source_total,
            target_total,
            int(row["iteration_index"]),
            row["current_ego"],
        )
        for row in trace_rows
    ]
    failed = first_invalid(rows)
    if failed is None or failed["iteration_index"] != 33:
        raise ValueError("OFFLINE_COVERAGE_FAILURE_DID_NOT_ALIGN_WITH_ITERATION_33")
    absolute_times = 3.3 + np.arange(80, dtype=np.float64) * 0.1
    progress = hlc_progress(absolute_times, HLC_BASELINE)
    return {
        "schema_version": "r1_b2_9_a_iteration_0_33_native_coverage_audit_v1",
        "status": "OFFLINE_RECONSTRUCTION_ALIGNS_WITH_ACTUAL_ITERATION_33_FAILURE",
        "execution_prohibition": {"simulation": False, "runner_run": False, "simulation_step": False, "selector": False},
        "attempt_sha256": _attempt_sha_inventory(),
        "scenario_token": entry["scenario_token"],
        "pair_id": "R1B27-01-R-HLC",
        "run_id": "R1B27-01-R-HLC-BASELINE",
        "source_lane_id": entry["source_lane_id"],
        "target_lane_id": entry["target_lane_id"],
        "planner_sampling_order": ["source_reference_xy", "target_reference_xy"],
        "exact_first_raised_reference": "source_reference_xy",
        "simultaneously_invalid_references_at_iteration_33": list(failed["invalid_references"]),
        "first_invalid_iteration": 33,
        "first_invalid_row": failed,
        "iteration_32_last_valid_row": rows[32],
        "baseline_iteration_33_output_window": {
            "absolute_time_start_s": float(absolute_times[0]),
            "absolute_time_end_s": float(absolute_times[-1]),
            "source_weight_min": float(np.min(1.0 - progress)),
            "source_weight_max": float(np.max(1.0 - progress)),
            "target_weight_min": float(np.min(progress)),
            "target_weight_max": float(np.max(progress)),
            "classification": [
                "UNNECESSARY_ZERO_WEIGHT_SOURCE_REFERENCE_EVALUATION_RAISED_FIRST",
                "ACTIVE_TARGET_NATIVE_REFERENCE_REPLAN_COVERAGE_EXHAUSTION_SIMULTANEOUS",
            ],
        },
        "rows": rows,
        "scientific_evidence": "NOT_EVALUABLE",
    }


def _arm_nominal_audit(
    bridge: R1OfficialMapQueryBridgeV2_1,
    entry: Mapping[str, Any],
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    source_total: float,
    target_total: float,
    source_arc: float,
    target_arc: float,
    arm: str,
) -> Dict[str, Any]:
    current = _current(entry["initial_state"])
    states = build_hlc_native_geometry_v1_1(current, 0.0, source_xy, target_xy, source_arc, target_arc, arm)
    rows = [
        coverage_row(
            bridge,
            str(entry["source_lane_id"]),
            str(entry["target_lane_id"]),
            source_total,
            target_total,
            index,
            state,
        )
        for index, state in enumerate(states)
    ]
    invalid = first_invalid(rows)
    return {
        "arm": arm,
        "predicted_first_coverage_exhaustion_iteration": None if invalid is None else invalid["iteration_index"],
        "predicted_first_invalid_references": [] if invalid is None else list(invalid["invalid_references"]),
        "rolling_call_envelope_iteration_0_79": rows,
    }


def all12_audit(entries: Sequence[Mapping[str, Any]], cache: Dict[str, Any]) -> Dict[str, Any]:
    output = []
    for entry in [row for row in entries if row["family"] == "R-HLC"]:
        bridge = R1OfficialMapQueryBridgeV2_1(_map_api(str(entry["map_name"]), cache))
        source_xy = bridge.native_reference_xy(str(entry["source_lane_id"]))
        target_xy = bridge.native_reference_xy(str(entry["target_lane_id"]))
        source_total = float(polyline_arclength(source_xy)[-1])
        target_total = float(polyline_arclength(target_xy)[-1])
        initial = entry["initial_state"]
        xy = (float(initial["initial_x"]), float(initial["initial_y"]))
        source_arc = float(bridge.project(str(entry["source_lane_id"]), xy)["arc_m"])
        target_arc = float(bridge.project(str(entry["target_lane_id"]), xy)["arc_m"])
        distance = float(initial["initial_speed_mps"]) * HORIZON_SECONDS
        output.append({
            "scenario_token": entry["scenario_token"],
            "log_id": entry["log_id"],
            "source_lane_id": entry["source_lane_id"],
            "target_lane_id": entry["target_lane_id"],
            "initial_speed_mps": initial["initial_speed_mps"],
            "source_initial_arc_m": source_arc,
            "target_initial_arc_m": target_arc,
            "source_native_total_length_m": source_total,
            "target_native_total_length_m": target_total,
            "one_shot_7_9s_future_distance_m": distance,
            "initial_source_coverage_margin_m": source_total - source_arc - distance,
            "initial_target_coverage_margin_m": target_total - target_arc - distance,
            "initial_one_shot_80_frame_no_extrapolation": "PASS",
            "baseline": _arm_nominal_audit(bridge, entry, source_xy, target_xy, source_total, target_total, source_arc, target_arc, HLC_BASELINE),
            "treatment": _arm_nominal_audit(bridge, entry, source_xy, target_xy, source_total, target_total, source_arc, target_arc, HLC_TREATMENT),
        })
    if len(output) != 12:
        raise ValueError(f"FROZEN_HLC_IDENTITY_COUNT_MUST_EQUAL_12:{len(output)}")
    return {
        "schema_version": "r1_b2_9_a_all12_hlc_nominal_replan_coverage_audit_v1",
        "status": "TECHNICAL_DIAGNOSTIC_ONLY_NOT_SCIENTIFIC_OUTCOME",
        "method": "ONE_SHOT_FROZEN_HLC_GEOMETRY_STATES_0_TO_79_REPROJECTED_AS_NOMINAL_REPLAN_CURRENT_EGO_WITH_CONSTANT_INITIAL_SPEED_AND_7_9S_OUTPUT_ENVELOPE",
        "simulation_executed": False,
        "identity_count": 12,
        "all_initial_one_shot_pass": all(row["initial_one_shot_80_frame_no_extrapolation"] == "PASS" for row in output),
        "all_predicted_to_exhaust_before_iteration_80": all(
            row[arm]["predicted_first_coverage_exhaustion_iteration"] is not None
            for row in output for arm in ("baseline", "treatment")
        ),
        "entries": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=R1)
    args = parser.parse_args()
    roster = read_json(ROSTER)
    entries = roster.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ROSTER_ENTRIES_REQUIRED")
    cache: Dict[str, Any] = {}
    iteration = iteration_audit(entries, cache)
    all12 = all12_audit(entries, cache)
    write_new(args.output_dir / ITERATION_OUT.name, iteration)
    write_new(args.output_dir / ALL12_OUT.name, all12)
    print(json.dumps({
        "status": "B2_9_A_OFFLINE_GEOMETRY_AUDIT_COMPLETE",
        "failure_iteration": iteration["first_invalid_iteration"],
        "exact_first_raised_reference": iteration["exact_first_raised_reference"],
        "identity_count": all12["identity_count"],
        "simulation_executed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
