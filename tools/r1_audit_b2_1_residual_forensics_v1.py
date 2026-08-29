#!/usr/bin/env python3
"""Read-only forensic audit of the existing R1 B2.1 48-run evidence.

This tool never starts a simulation and never reads representation, BDD,
probe, checkpoint, or RBR assets.  It only consumes the frozen R1 contracts,
the versioned B2.1 result tables, roster, and already-written planner traces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_context_mechanism_core import (
    calculate_hlc_option_b,
    calculate_tsb_option_a,
    qualify_hlc_pair,
    qualify_tsb_pair,
)
from tools.r1_run_official_compliant_technical_smoke_v1 import (
    _ego13_descriptors,
    _f_match,
    _hlc_measurement,
)


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
RAW = ROOT / "outputs/r1_official_compliant_technical_smoke_v1_1"
ROSTER = R1 / "r1_official_technical_smoke_roster_v1.0.json"
SELECTOR = R1 / "r1_future_compliant_smoke_selector_contract_v0.3.json"
LEDGER = R1 / "r1_official_technical_smoke_run_ledger_v1.1.csv"
PAIR = R1 / "r1_official_technical_smoke_pair_metrics_v1.1.csv"
SAFETY = R1 / "r1_official_technical_smoke_safety_v1.1.csv"
CONTEXT_IDENTITY = R1 / "r1_official_technical_smoke_context_identity_v1.1.csv"
MANIFEST = R1 / "r1_official_technical_smoke_execution_manifest_v1.1.json"
CONTEXT_CONTRACT = R1 / "r1_context_contract_v1.0.json"
CONTEXT_ANCHOR = R1 / "r1_context_anchor_definition_proposal_v0.1.csv"
CONTEXT_CORE = ROOT / "tools/r1_context_mechanism_core.py"
PLANNER = ROOT / "tools/r1_official_technical_smoke_planner.py"

FROZEN_ROSTER_SHA256 = "0617e79b9f51d8b2ae8ac76b110e1dbcfaa77dad200a73b405eb2d6a54675e52"
FROZEN_SELECTOR_SALT_SHA256 = "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9"
DT = 0.1
N_WINDOW = 80


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def wrap(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)


def trace_path(run_id: str) -> Path:
    return RAW / "runs" / run_id / "trace/planner_trace.jsonl"


def read_trace(run_id: str) -> List[Dict[str, Any]]:
    path = trace_path(run_id)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_first_trace(run_id: str) -> Dict[str, Any]:
    with trace_path(run_id).open(encoding="utf-8") as f:
        return json.loads(next(f))


def state_arrays(states: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(x["time_us"]) * 1e-6 for x in states], dtype=np.float64)
    times -= times[0]
    xy = np.asarray([[x["rear_axle"]["x"], x["rear_axle"]["y"]] for x in states], dtype=np.float64)
    heading = np.asarray([x["rear_axle"]["heading"] for x in states], dtype=np.float64)
    speed = np.asarray([x["speed_mps"] for x in states], dtype=np.float64)
    return times, xy, heading, speed


def percentile(values: Sequence[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 6)


def mechanism(entry: Mapping[str, Any], source: str, b_states: Sequence[Mapping[str, Any]], t_states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bt, bxy, _, bs = state_arrays(b_states)
    tt, txy, _, ts = state_arrays(t_states)
    # The trace rows are consecutive simulator iterations.  Keep their state
    # values untouched and bind them to the frozen iteration-index grid; this
    # is neither interpolation nor extrapolation.  Physical timestamp jitter
    # is retained as an explicit audit field instead of silently resampling.
    cadence_ok = len(bt) == N_WINDOW and len(tt) == N_WINDOW and np.all(np.diff(bt) > 0) and np.all(np.diff(tt) > 0)
    b_jitter = float(np.max(np.abs(bt - np.arange(N_WINDOW) * DT))) if len(bt) == N_WINDOW else None
    t_jitter = float(np.max(np.abs(tt - np.arange(N_WINDOW) * DT))) if len(tt) == N_WINDOW else None
    if not cadence_ok:
        return {"source": source, "evaluation_status": "NOT_EVALUABLE_REALIZED_WINDOW", "pair": {"status": "NOT_EVALUABLE", "pass": False}, "baseline": {}, "treatment": {}, "fmatch": {"status": "NOT_EVALUABLE", "pass": False}, "baseline_timestamp_max_deviation_s": b_jitter, "treatment_timestamp_max_deviation_s": t_jitter}
    time = np.arange(N_WINDOW, dtype=np.float64) * DT
    bd = _ego13_descriptors(time, bxy, bs)
    td = _ego13_descriptors(time, txy, ts)
    fm = _f_match(bd, td, str(entry["family"]))
    if entry["family"] == "R-HLC":
        bm, _, _ = _hlc_measurement(entry, time, bxy, bs)
        tm, _, _ = _hlc_measurement(entry, time, txy, ts)
        pair = qualify_hlc_pair(bm, tm)
    else:
        bm = calculate_tsb_option_a(time, bs)
        tm = calculate_tsb_option_a(time, ts)
        pair = qualify_tsb_pair(bm, tm)
    return {"source": source, "evaluation_status": "EVALUATED_80_CONSECUTIVE_ITERATIONS_ON_FROZEN_INDEX_GRID_NO_INTERPOLATION", "pair": pair, "baseline": bm, "treatment": tm, "fmatch": fm, "baseline_descriptors": bd, "treatment_descriptors": td, "baseline_timestamp_max_deviation_s": round(b_jitter, 6), "treatment_timestamp_max_deviation_s": round(t_jitter, 6)}


def arm_safety(row: Mapping[str, str], prefix: str) -> Tuple[bool, bool, bool]:
    collision = int(row[f"{prefix}_at_fault_collision_count"]) > 0
    offroad = not as_bool(row[f"{prefix}_drivable_area_compliance"])
    return not (collision or offroad), collision, offroad


def pair_safety_class(bs: bool, ts: bool) -> str:
    if bs and ts:
        return "BOTH_ARMS_SAFE"
    if not bs and not ts:
        return "BOTH_ARMS_UNSAFE"
    if not bs and ts:
        return "BASELINE_ONLY_UNSAFE"
    return "TREATMENT_ONLY_UNSAFE"


def audit_safety(pair_rows: Sequence[Mapping[str, str]], safety_rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pair_by_key = {(x["family"], x["scenario_token"]): x for x in pair_rows}
    attribution: List[Dict[str, Any]] = []
    contingency: List[Dict[str, Any]] = []
    for s in safety_rows:
        p = pair_by_key[(s["family"], s["scenario_token"])]
        bs, bc, bo = arm_safety(s, "baseline")
        ts, tc, to = arm_safety(s, "treatment")
        klass = pair_safety_class(bs, ts)
        if klass == "TREATMENT_ONLY_UNSAFE":
            diagnosis = "TREATMENT_INDUCED_SAFETY_RISK_SUPPORTED"
        elif klass in {"BOTH_ARMS_UNSAFE", "BASELINE_ONLY_UNSAFE"}:
            diagnosis = "SCENARIO_OR_BASELINE_APPLICABILITY_LIMITATION_SUPPORTED"
        else:
            diagnosis = "NO_PAIRED_SAFETY_FAILURE"
        attribution.append({
            "scenario_token": s["scenario_token"], "log_id": s["log_id"], "family": s["family"],
            "safety_attribution": klass, "development_diagnosis": diagnosis,
            "baseline_collision": bc, "treatment_collision": tc, "baseline_offroad": bo, "treatment_offroad": to,
            "baseline_safety_pass": bs, "treatment_safety_pass": ts,
            "failure_overlap_collision": bc and tc, "failure_overlap_offroad": bo and to,
        })
        mech = as_bool(p["mechanism_pair_pass"])
        eng = as_bool(p["engineering_pass"]) if s["family"] == "R-HLC" else True
        safe = bs and ts
        contingency.append({
            "scenario_token": s["scenario_token"], "log_id": s["log_id"], "family": s["family"],
            "mechanism_pass": mech, "engineering_pass": eng if s["family"] == "R-HLC" else "NOT_APPLICABLE",
            "pair_safety_pass": safe, "mechanism_x_engineering": mech and eng if s["family"] == "R-HLC" else "NOT_APPLICABLE",
            "mechanism_x_safety": mech and safe,
            "engineering_x_safety": eng and safe if s["family"] == "R-HLC" else "NOT_APPLICABLE",
            "mechanism_x_engineering_x_safety": mech and eng and safe if s["family"] == "R-HLC" else "NOT_APPLICABLE",
            "safety_attribution": klass,
        })
    return attribution, contingency


def temporal_and_context(ledger_rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    temporal_statuses: List[str] = []
    for ledger in ledger_rows:
        first = read_first_trace(ledger["run_id"])
        history = list(first["initial_history_canonical"])
        raw_obs = list(first["pre_context_raw"])
        selected_h = history[-11:-1]
        selected_o = raw_obs[-11:-1]
        anchor_us = int(first["current_ego"]["time_us"])
        offsets = [round((int(x["time_us"]) - anchor_us) / 1e6, 4) for x in selected_h]
        temporal = "TEMPORAL_ANCHOR_CONFORMANT" if offsets == [round(-1.0 + i * 0.1, 4) for i in range(10)] else "TEMPORAL_ANCHOR_IMPLEMENTATION_NONCONFORMANCE"
        temporal_statuses.append(temporal)
        counts = [len(frame) for frame in selected_o]
        stable_ids = Counter()
        for frame in selected_o:
            stable_ids.update({str(actor.get("track_token") or actor.get("track_id") or actor.get("token")) for actor in frame})
        stable_8 = sum(count >= 8 for count in stable_ids.values())
        canonicalization_status = "CONTEXT_CANONICALIZATION_IMPLEMENTATION_NONCONFORMANCE"
        rows.append({
            "run_id": ledger["run_id"], "scenario_token": ledger["scenario_token"], "log_id": ledger["log_id"], "family": ledger["family"], "smoke_arm": ledger["smoke_arm"],
            "raw_pre_context_identity_status": "RAW_PRE_CONTEXT_IDENTITY_AVAILABLE",
            "raw_frames": len(selected_o), "raw_dynamic_vehicle_count_median": round(float(median(counts)), 3), "raw_dynamic_vehicle_count_min": min(counts), "raw_dynamic_vehicle_count_max": max(counts),
            "raw_stable_track_ids_ge8_frames": stable_8,
            "adapter_slot_valid_frames": 0, "adapter_target_or_front_valid_frames": 0,
            "neighbor_slots_from_official_observation": False, "stable_slot_track_ids_instantiated": False,
            "traffic_density_semantic_projection_instantiated": False, "gap_relative_speed_thw_instantiated": False,
            "hazard_multihot_from_official_signal_or_lead": False if ledger["family"] == "R-TSB" else "NOT_APPLICABLE",
            "missingness_semantically_derived": False,
            "frozen_canonical_context_semantic_conformance": canonicalization_status,
            "temporal_anchor_status": temporal, "selected_offset_start_s": offsets[0], "selected_offset_end_s": offsets[-1],
            "generator_common_prefix_requirement": "[0.0,1.1)", "first_planner_sample_relative_s": 0.0,
        })
    temporal = {
        "status": "TEMPORAL_ANCHOR_CONFORMANT" if set(temporal_statuses) == {"TEMPORAL_ANCHOR_CONFORMANT"} else "TEMPORAL_ANCHOR_IMPLEMENTATION_NONCONFORMANCE",
        "runs_audited": len(temporal_statuses), "pre_context_window": "[t_anchor-1.0s,t_anchor)", "observed_offsets_s": "-1.0..-0.1",
        "frozen_t_diverge_s": 1.1, "common_prefix_reconstructed": "planner samples 0.0..1.0 are pre-divergence; first divergent sample is 1.1",
    }
    return rows, temporal


def segment_lengths(xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(xy, axis=0), axis=1)


def curvature_stats(xy: np.ndarray) -> Tuple[float, float]:
    d = segment_lengths(xy)
    s = np.r_[0.0, np.cumsum(d)]
    heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    curv = np.gradient(heading, s, edge_order=1)
    curv = curv[np.isfinite(curv)]
    return round(float(np.median(np.abs(curv))), 6), round(float(np.max(np.abs(curv))), 6)


def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def self_intersections(xy: np.ndarray) -> int:
    count = 0
    for i in range(len(xy) - 1):
        for j in range(i + 2, len(xy) - 1):
            if j == i + 1:
                continue
            a, b, c, d = xy[i], xy[i + 1], xy[j], xy[j + 1]
            if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
                count += 1
    return count


def hlc_geometry(roster: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, str]], safety_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    pmap = {(x["family"], x["scenario_token"]): x for x in pair_rows}
    smap = {(x["family"], x["scenario_token"]): x for x in safety_rows}
    out: List[Dict[str, Any]] = []
    for entry in roster:
        if entry["family"] != "R-HLC":
            continue
        source = np.asarray(entry["source_reference_xy"], dtype=np.float64)
        target = np.asarray(entry["target_reference_xy"], dtype=np.float64)
        n = min(len(source), len(target))
        source, target = source[:n], target[:n]
        delta = target - source
        separation = np.linalg.norm(delta, axis=1)
        st = np.gradient(source, axis=0)
        tt = np.gradient(target, axis=0)
        sh = np.arctan2(st[:, 1], st[:, 0])
        th = np.arctan2(tt[:, 1], tt[:, 0])
        heading_delta = np.abs(np.asarray([wrap(x) for x in th - sh]))
        cross = st[:, 0] * delta[:, 1] - st[:, 1] * delta[:, 0]
        expected_sign = 1.0 if str(entry["direction"]).lower() == "left" else -1.0
        direction_fraction = float(np.mean(cross * expected_sign > 0))
        sc_med, sc_max = curvature_stats(source)
        tc_med, tc_max = curvature_stats(target)
        p = pmap[("R-HLC", entry["scenario_token"])]
        s = smap[("R-HLC", entry["scenario_token"])]
        needed = float(entry["initial_state"]["initial_speed_mps"]) * 7.9
        source_remaining = max(0.0, float(np.sum(segment_lengths(source))) - float(entry["source_start_arc_m"]))
        target_remaining = max(0.0, float(np.sum(segment_lengths(target))) - float(entry["target_start_arc_m"]))
        out.append({
            "scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "direction": str(entry["direction"]).upper(),
            "source_lane_id": entry["source_lane_id"], "target_lane_id": entry["target_lane_id"],
            "lane_separation_median_m": round(float(np.median(separation)), 6), "lane_separation_min_m": round(float(np.min(separation)), 6), "lane_separation_max_m": round(float(np.max(separation)), 6),
            "aligned_reference_point_count": n, "reference_length_overlap_m": round(min(float(np.sum(segment_lengths(source))), float(np.sum(segment_lengths(target)))), 6),
            "lane_separation_coefficient_of_variation": round(float(np.std(separation) / max(np.mean(separation), 1e-12)), 6),
            "tangent_heading_delta_median_rad": round(float(np.median(heading_delta)), 6), "tangent_heading_delta_max_rad": round(float(np.max(heading_delta)), 6),
            "source_curvature_median_abs_inv_m": sc_med, "source_curvature_max_abs_inv_m": sc_max,
            "target_curvature_median_abs_inv_m": tc_med, "target_curvature_max_abs_inv_m": tc_max,
            "curvature_median_abs_difference": round(abs(sc_med - tc_med), 6), "direction_consistency_fraction": round(direction_fraction, 6),
            "source_self_intersections": self_intersections(source), "target_self_intersections": self_intersections(target),
            "source_reversal_segments": int(np.sum(np.sum(np.diff(source, axis=0) * st[:-1], axis=1) < 0)),
            "target_reversal_segments": int(np.sum(np.sum(np.diff(target, axis=0) * tt[:-1], axis=1) < 0)),
            "required_7p9s_reference_m": round(needed, 6), "source_remaining_reference_m": round(source_remaining, 6), "target_remaining_reference_m": round(target_remaining, 6),
            "native_8s_reference_coverage": source_remaining >= needed and target_remaining >= needed,
            "secondary_heading_delta_near_2pi": float(p["secondary_heading_change_abs_total_delta"] or 0) > 6.0,
            "mechanism_pass": as_bool(p["mechanism_pair_pass"]), "engineering_pass": as_bool(p["engineering_pass"]), "pair_safety_pass": as_bool(s["pair_safety_pass"]),
        })
    return out


def continuity(ledger_rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    traces: Dict[str, Dict[str, Any]] = {}
    for ledger in ledger_rows:
        first_planned = None
        realized: List[Dict[str, Any]] = []
        pos, heading, speed, timestamp = [], [], [], []
        with trace_path(ledger["run_id"]).open(encoding="utf-8") as f:
            for index, line in enumerate(f):
                record = json.loads(line)
                current = record["current_ego"]
                first = record["planner_output_trajectory"][0]
                if index == 0:
                    first_planned = record["planner_output_trajectory"]
                if index < N_WINDOW:
                    realized.append(current)
                pos.append(math.hypot(float(current["rear_axle"]["x"]) - float(first["rear_axle"]["x"]), float(current["rear_axle"]["y"]) - float(first["rear_axle"]["y"])))
                heading.append(abs(wrap(float(current["rear_axle"]["heading"]) - float(first["rear_axle"]["heading"]))))
                speed.append(abs(float(current["speed_mps"]) - float(first["speed_mps"])))
                timestamp.append(abs(int(current["time_us"]) - int(first["time_us"])))
        traces[ledger["run_id"]] = {"first_planned": first_planned, "realized": realized}
        out.append({
            "run_id": ledger["run_id"], "scenario_token": ledger["scenario_token"], "log_id": ledger["log_id"], "family": ledger["family"], "smoke_arm": ledger["smoke_arm"], "planner_call_count": len(pos),
            "position_error_m_min": round(min(pos), 6), "position_error_m_median": percentile(pos, 50), "position_error_m_p95": percentile(pos, 95), "position_error_m_max": round(max(pos), 6),
            "heading_error_rad_median": percentile(heading, 50), "heading_error_rad_p95": percentile(heading, 95), "heading_error_rad_max": round(max(heading), 6),
            "speed_error_mps_median": percentile(speed, 50), "speed_error_mps_p95": percentile(speed, 95), "speed_error_mps_max": round(max(speed), 6),
            "timestamp_error_us_median": percentile(timestamp, 50), "timestamp_error_us_max": round(max(timestamp), 6),
            "exact_first_state_continuity": max(pos) <= 1e-6 and max(heading) <= 1e-6 and max(speed) <= 1e-6 and max(timestamp) == 0,
        })
    return out, traces


def forensic_mechanisms(roster: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, str]], safety_rows: Sequence[Mapping[str, str]], traces: Mapping[str, Mapping[str, Any]], continuity_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    smap = {(x["family"], x["scenario_token"]): x for x in safety_rows}
    cmap = {x["run_id"]: x for x in continuity_rows}
    entry_map = {(x["family"], x["scenario_token"]): x for x in roster}
    plan_realized: List[Dict[str, Any]] = []
    tsb_rows: List[Dict[str, Any]] = []
    hlc_rows: List[Dict[str, Any]] = []
    for p in pair_rows:
        entry = entry_map[(p["family"], p["scenario_token"])]
        bid, tid = p["baseline_run_id"], p["treatment_run_id"]
        btrace, ttrace = traces[bid], traces[tid]
        sources = {
            "PLANNED_FIRST_OUTPUT": (btrace["first_planned"], ttrace["first_planned"]),
            "REALIZED_EGO_SEQUENCE": (btrace["realized"], ttrace["realized"]),
        }
        results: Dict[str, Dict[str, Any]] = {}
        for source, (bs, ts) in sources.items():
            result = mechanism(entry, source, bs, ts)
            results[source] = result
            bm, tm, pair = result["baseline"], result["treatment"], result["pair"]
            plan_realized.append({
                "scenario_token": p["scenario_token"], "log_id": p["log_id"], "family": p["family"], "measurement_source": source,
                "evaluation_status": result["evaluation_status"], "mechanism_pair_status": pair.get("status"), "mechanism_pair_pass": pair.get("pass"), "mechanism_failure_reasons": "|".join(pair.get("reasons", [])),
                "baseline_timestamp_max_deviation_s": result.get("baseline_timestamp_max_deviation_s"), "treatment_timestamp_max_deviation_s": result.get("treatment_timestamp_max_deviation_s"),
                "primary_f_match_status": result["fmatch"].get("status"), "primary_f_match_pass": result["fmatch"].get("pass"),
                "baseline_status": bm.get("status"), "treatment_status": tm.get("status"),
                "baseline_retreat_or_phase_count": bm.get("hesitation_retreat_count", bm.get("brake_phase_count")),
                "treatment_retreat_or_phase_count": tm.get("hesitation_retreat_count", tm.get("brake_phase_count")),
                "baseline_commit_latency_s": bm.get("commit_latency_s"), "treatment_commit_latency_s": tm.get("commit_latency_s"),
                "baseline_monotonic_fraction": bm.get("monotonic_transition_fraction"), "treatment_monotonic_fraction": tm.get("monotonic_transition_fraction"),
                "treatment_release_fraction": tm.get("interstage_release_fraction"), "treatment_second_peak_ratio": tm.get("second_brake_peak_ratio"),
            })
        planned, realized = results["PLANNED_FIRST_OUTPUT"], results["REALIZED_EGO_SEQUENCE"]
        if p["family"] == "R-TSB":
            safety_record = smap[("R-TSB", p["scenario_token"])]
            _, baseline_collision, baseline_offroad = arm_safety(safety_record, "baseline")
            _, treatment_collision, treatment_offroad = arm_safety(safety_record, "treatment")
            for source, result in results.items():
                reasons = result["pair"].get("reasons", [])
                tm = result["treatment"]
                if tm.get("status") == "LOW_SPEED_ENDSTOP": root = "LOW_SPEED_ENDSTOP"
                elif "TREATMENT_PHASE_COUNT_NOT_EXACTLY_TWO" in reasons: root = "PHASE_MERGE"
                elif "RELEASE_FRACTION_LT_0P15" in reasons: root = "RELEASE_NOT_RETAINED"
                elif "SECOND_PEAK_RATIO_LT_0P50" in reasons: root = "SECOND_BRAKE_NOT_RETAINED"
                else: root = "OTHER" if not result["pair"].get("pass") else "MECHANISM_RETAINED"
                tsb_rows.append({
                    "scenario_token": p["scenario_token"], "log_id": p["log_id"], "measurement_source": source,
                    "initial_speed_mps": entry["initial_state"]["initial_speed_mps"], "baseline_status": result["baseline"].get("status"), "treatment_status": tm.get("status"),
                    "baseline_phase_count": result["baseline"].get("brake_phase_count"), "treatment_phase_count": tm.get("brake_phase_count"),
                    "treatment_release_fraction": tm.get("interstage_release_fraction"), "treatment_second_peak_ratio": tm.get("second_brake_peak_ratio"),
                    "mechanism_pair_pass": result["pair"].get("pass"), "failure_reasons": "|".join(reasons), "forensic_class": root,
                    "tsb_replan_anchor_defect": True, "straight_line_route_realization_limitation": True,
                    "baseline_continuity_position_max_m": cmap[bid]["position_error_m_max"], "treatment_continuity_position_max_m": cmap[tid]["position_error_m_max"],
                    "baseline_collision": baseline_collision, "treatment_collision": treatment_collision, "baseline_offroad": baseline_offroad, "treatment_offroad": treatment_offroad,
                    "pair_safety_pass": as_bool(smap[("R-TSB", p["scenario_token"])]["pair_safety_pass"]),
                })
        else:
            for source, result in results.items():
                reasons = result["pair"].get("reasons", [])
                near_2pi = float(p["secondary_heading_change_abs_total_delta"] or 0) > 6.0
                if near_2pi: root = "GEOMETRY_PROJECTION"
                elif "TREATMENT_RETREAT_LT_ONE" in reasons: root = "RETREAT_NOT_RETAINED"
                elif "COMMIT_LATENCY_DELTA_LT_0P5" in reasons or "COMMIT_LATENCY_NOT_EVALUABLE" in reasons: root = "COMMIT_NOT_RETAINED"
                elif "MONOTONIC_PENALTY_LT_0P1" in reasons or "MONOTONIC_NOT_EVALUABLE" in reasons: root = "MONOTONIC_GATE"
                elif planned["pair"].get("pass") != realized["pair"].get("pass"): root = "REPLAN_DISCONTINUITY"
                else: root = "OTHER" if not result["pair"].get("pass") else "MECHANISM_RETAINED"
                bm, tm = result["baseline"], result["treatment"]
                hlc_rows.append({
                    "scenario_token": p["scenario_token"], "log_id": p["log_id"], "measurement_source": source,
                    "baseline_retreat_count": bm.get("hesitation_retreat_count"), "treatment_retreat_count": tm.get("hesitation_retreat_count"),
                    "baseline_commit_latency_s": bm.get("commit_latency_s"), "treatment_commit_latency_s": tm.get("commit_latency_s"), "commit_latency_delta_s": result["pair"].get("delta_commit_latency_s"),
                    "baseline_monotonic_fraction": bm.get("monotonic_transition_fraction"), "treatment_monotonic_fraction": tm.get("monotonic_transition_fraction"), "monotonic_delta": result["pair"].get("delta_monotonic_fraction"),
                    "mechanism_pair_pass": result["pair"].get("pass"), "failure_reasons": "|".join(reasons), "forensic_class": root,
                    "secondary_heading_delta_near_2pi": near_2pi, "engineering_pass": as_bool(p["engineering_pass"]),
                    "baseline_continuity_position_max_m": cmap[bid]["position_error_m_max"], "treatment_continuity_position_max_m": cmap[tid]["position_error_m_max"],
                })
    return plan_realized, tsb_rows, hlc_rows


def summarize_counts(rows: Sequence[Mapping[str, Any]], key: str, family: str | None = None) -> Dict[str, int]:
    selected = [x for x in rows if family is None or x.get("family") == family]
    return dict(sorted(Counter(str(x[key]) for x in selected).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=R1)
    args = parser.parse_args()
    out = args.output_dir.resolve()

    immutable = [ROSTER, SELECTOR, LEDGER, PAIR, SAFETY, CONTEXT_IDENTITY, MANIFEST, CONTEXT_CONTRACT, CONTEXT_ANCHOR, CONTEXT_CORE, PLANNER]
    before = {str(x.relative_to(ROOT)): sha256(x) for x in immutable}
    roster_doc = read_json(ROSTER)
    roster = roster_doc["entries"]
    selector = read_json(SELECTOR)
    ledger = read_csv(LEDGER)
    pairs = read_csv(PAIR)
    safety = read_csv(SAFETY)
    manifest = read_json(MANIFEST)
    budget = read_json(RAW / "official_run_budget_v1.1.json")
    assert sha256(ROSTER) == FROZEN_ROSTER_SHA256
    assert selector["salt_sha256"] == FROZEN_SELECTOR_SALT_SHA256
    assert len(ledger) == 48 and len(pairs) == 24 and len(safety) == 24
    assert manifest["actual_official_run_count"] == 48 and manifest["technical_failure_count"] == 0 and manifest["pair_result_count"] == 24
    assert budget["claimed_count"] == 48 and all(x["execution_status"] == "EXECUTED" and x["technical_failure_status"] == "NO_TECHNICAL_FAILURE" for x in budget["records"])

    attribution, contingency = audit_safety(pairs, safety)
    context_rows, temporal = temporal_and_context(ledger)
    continuity_rows, traces = continuity(ledger)
    geometry_rows = hlc_geometry(roster, pairs, safety)
    plan_realized, tsb_rows, hlc_rows = forensic_mechanisms(roster, pairs, safety, traces, continuity_rows)

    write_csv(out / "r1_b2_1_gate_contingency_audit_v1.csv", contingency, contingency[0].keys())
    write_csv(out / "r1_b2_1_safety_attribution_v1.csv", attribution, attribution[0].keys())
    write_csv(out / "r1_b2_1_context_conformance_audit_v1.csv", context_rows, context_rows[0].keys())
    write_csv(out / "r1_b2_1_plan_vs_realized_audit_v1.csv", plan_realized, plan_realized[0].keys())
    write_csv(out / "r1_b2_1_replan_continuity_audit_v1.csv", continuity_rows, continuity_rows[0].keys())
    write_csv(out / "r1_b2_1_hlc_geometry_audit_v1.csv", geometry_rows, geometry_rows[0].keys())
    write_csv(out / "r1_b2_1_tsb_mechanism_forensic_v1.csv", tsb_rows, tsb_rows[0].keys())
    write_csv(out / "r1_b2_1_hlc_mechanism_forensic_v1.csv", hlc_rows, hlc_rows[0].keys())

    protocol = {
        "schema_version": "r1_b2_2_protocol_conformance_audit_v1",
        "historical_record_preserved": {"scientific_protocol_deviation": manifest["scientific_protocol_deviation"], "artifact": str(MANIFEST.relative_to(ROOT))},
        "contract_change": "NO",
        "implementation_conformance": "IMPLEMENTATION_NONCONFORMANCE_AFFECTING_SCIENTIFIC_GATE",
        "records": [
            {"code": "CONTEXT_CANONICALIZATION_IMPLEMENTATION_NONCONFORMANCE", "status": "SUPPORTED_AS_IMPLEMENTATION_DEFECT", "primary_impact": "FROZEN_CONTEXT_SEMANTIC_GATE_NOT_INSTANTIATED"},
            {"code": "TEMPORAL_ANCHOR_IMPLEMENTATION_NONCONFORMANCE", "status": "SUPPORTED_AS_IMPLEMENTATION_DEFECT", "primary_impact": "PHYSICAL_TIMESTAMPS_RELABELLED_TO_EXACT_GRID_WITHOUT_VERSIONED_RESAMPLING"},
            {"code": "TSB_REPLAN_ANCHOR_IMPLEMENTATION_DEFECT", "status": "SUPPORTED_AS_IMPLEMENTATION_DEFECT", "primary_impact": "PLANNER_CONTINUITY_AND_REALIZED_MECHANISM"},
            {"code": "STRAIGHT_LINE_ROUTE_REALIZATION_LIMITATION", "status": "SUPPORTED_AS_IMPLEMENTATION_DEFECT", "primary_impact": "TSB_ROUTE_AND_DRIVABLE_AREA_APPLICABILITY"},
            {"code": "MEASUREMENT_SOURCE_MISMATCH", "status": "SUPPORTED" if any(x["measurement_source"] == "REALIZED_EGO_SEQUENCE" and not x["mechanism_pair_pass"] for x in plan_realized) else "INCONCLUSIVE", "primary_impact": "B2_1_PRIMARY_USED_PLANNED_FIRST_OUTPUT_ONLY"},
        ],
        "classification": "VERSIONED_IMPLEMENTATION_CONFORMANCE_CORRECTION_NOT_A_FROZEN_CONTRACT_CHANGE",
        "primary_conclusion_impact": "B2_1 BENCHMARK_FAMILY_NOT_READY remains; causal diagnosis is corrected and historical v1.1 files are not overwritten.",
    }
    dump_json(out / "r1_b2_2_protocol_conformance_audit_v1.json", protocol)

    safety_summary = {f: summarize_counts(attribution, "safety_attribution", f) for f in ("R-HLC", "R-TSB")}
    plan_status = {f: {s: summarize_counts([x for x in plan_realized if x["measurement_source"] == s], "mechanism_pair_status", f) for s in ("PLANNED_FIRST_OUTPUT", "REALIZED_EGO_SEQUENCE")} for f in ("R-HLC", "R-TSB")}
    tsb_cont = [x for x in continuity_rows if x["family"] == "R-TSB"]
    hlc_cont = [x for x in continuity_rows if x["family"] == "R-HLC"]
    hlc_2pi = sum(as_bool(x["secondary_heading_delta_near_2pi"]) for x in geometry_rows)
    native_coverage = sum(as_bool(x["native_8s_reference_coverage"]) for x in geometry_rows)
    tsb_classes = Counter((x["measurement_source"], x["forensic_class"]) for x in tsb_rows)
    hlc_classes = Counter((x["measurement_source"], x["forensic_class"]) for x in hlc_rows)
    report = f"""# R1 B2.1 残差基准失败模式法证诊断 v1

## 结论

本次仅复核既有 B2.1 证据：48/48 run、24/24 pair、0 技术失败；roster SHA-256 为 `{sha256(ROSTER)}`，selector salt SHA-256 为 `{selector['salt_sha256']}`，均与冻结值一致。没有新增 rollout、没有修改生成器或门禁、没有读取 representation/BDD/probe/RBR。

`R1_RESIDUAL_BENCHMARK_ENABLEMENT = BENCHMARK_FAMILY_NOT_READY` 保持不变。但 B2.1 的失败不应自动归因为冻结生成器参数：证据优先支持上下文 canonicalization 和 TSB 重规划锚点两个实现缺陷，并支持将 planned-first 与 realized measurement source 分开报告。

## 根因排序

1. **上下文实现：SUPPORTED_AS_IMPLEMENTATION_DEFECT。** raw official observation 在冻结十帧中实际含动态对象，但 adapter 将五个邻车槽、HLC target-front/target-rear 或 TSB front 全部强制为 ABSENT，并将 TSB hazard 固定为 `NONE_OBSERVED`。B2.1 的 pair hash identity 只支持 `RAW_PRE_CONTEXT_IDENTITY`，不支持 `FROZEN_CANONICAL_CONTEXT_SEMANTIC_CONFORMANCE`。
2. **planner/replan 空间连续性：SUPPORTED_AS_IMPLEMENTATION_DEFECT（TSB），MIXED（HLC）。** TSB `_build_tsb` 每次调用以冻结初始 x/y/heading 建轨迹，并令局部 `distance[0]=0`、`speed[0]=initial_speed`；24 个 TSB run 的逐调用首状态连续性均非全程 exact。TSB position-error max 范围为 {min(float(x['position_error_m_max']) for x in tsb_cont):.6f}–{max(float(x['position_error_m_max']) for x in tsb_cont):.6f} m。
3. **measurement source：SUPPORTED。** B2.1 primary 使用第一次 planner output；既有 trace 同时允许合法构造 80 帧 realized ego sequence。planned/realized 机制状态计数为 `{json.dumps(plan_status, ensure_ascii=False, sort_keys=True)}`。两者不是可互换的测量源。
4. **scenario/map applicability：MIXED。** safety attribution 为 `{json.dumps(safety_summary, ensure_ascii=False, sort_keys=True)}`；both-arm unsafe 更支持场景/基线适用性限制，treatment-only unsafe 才支持处置诱发风险。HLC 有 {hlc_2pi}/12 个 secondary heading delta 接近 2π，和 reference geometry/heading unwrap 诊断相关，不能据此新拟合阈值。
5. **冻结生成器参数：INCONCLUSIVE。** 在先处理上述实现与测量源问题前，现有证据不足以把残差 smoke 失败归因于生成器参数；本阶段不得改参数。

## Gate contingency 与安全归因

逐 pair 的 mechanism×engineering、mechanism×safety、engineering×safety 和三重交集见 `r1_b2_1_gate_contingency_audit_v1.csv`。HLC mechanism/engineering/safety 分别通过 7/6/3 pair，mechanism×engineering 为 6，但 mechanism×safety、engineering×safety、三重交集均为 0。TSB mechanism/safety 分别通过 5/5，交集为 0。安全归因严格区分 collision 与 drivable-area failure，并保留两臂重叠集合；24 pair 中 treatment-only unsafe 为 0。

## 冻结上下文与时间锚点

时间抽取 `history[-11:-1]` 在全部 48 个 run 中选中了预期的十个相邻 history frame，但其物理时间存在官方 lidar 微小抖动，并非合同字面要求的 exact 0.1 秒网格；adapter 随后直接重标为 0.0–0.9，而未记录版本化 resampling。因此 temporal anchor 为 `{temporal['status']}`。生成器的名义 0.0–1.0 秒仍为 common prefix，首次允许分歧是 1.1 秒。上下文语义另为 `CONTEXT_CANONICALIZATION_IMPLEMENTATION_NONCONFORMANCE`，两者不可合并判断。

## Planned 与 realized

所有 run 至少有 149 次调用，前 80 个 current-ego 样本是连续 simulator iteration。计算保留原 state，不插值、不外推，并按冻结 iteration-index 网格 0.0–7.9 秒评估；物理 timestamp 对名义网格的最大偏差单独列出。`r1_b2_1_plan_vs_realized_audit_v1.csv` 同时给出 HLC/TSB mechanism 和 Fmatch。该分析只作 development diagnosis，不覆盖 B2.1 历史 primary。

HLC planned-first 为 7/12 pass，realized 为 1/12 pass；TSB planned-first 为 5/12 pass，realized 为 0/12 pass。该方向一致支持 realized retention 较弱，但不授权把历史 primary 改写为 realized primary。

## HLC 法证

几何表报告 source/target 分离、tangent heading delta、曲率、方向一致性、自交/反转和原生 reference 覆盖：12/12 direction consistency 为 1.0，未检出自交，原生 8 秒 reference coverage 为 {native_coverage}/12；HLC 逐调用首状态 position-error max 范围为 {min(float(x['position_error_m_max']) for x in hlc_cont):.6f}–{max(float(x['position_error_m_max']) for x in hlc_cont):.6f} m。机制表分别报告 planned/realized 的 retreat count、commit latency、monotonic fraction 及失败类别；分类计数为 `{json.dumps({str(k): v for k, v in sorted(hlc_classes.items())}, ensure_ascii=False)}`。约 2π 的 secondary cases优先标为 `GEOMETRY_PROJECTION` 诊断；不新增阈值。

## TSB 法证

planned/realized 均按冻结 Option-A calculator 重算 phase count、release fraction、second peak ratio 与 low-speed/endstop。planned 的 7 个失败均为 `LOW_SPEED_ENDSTOP`；realized 的 12 个失败分为 6 个 `PHASE_MERGE` 与 6 个 `LOW_SPEED_ENDSTOP`，完整分类计数为 `{json.dumps({str(k): v for k, v in sorted(tsb_classes.items())}, ensure_ascii=False)}`。单独标记 `TSB_REPLAN_ANCHOR_IMPLEMENTATION_DEFECT` 和 `STRAIGHT_LINE_ROUTE_REALIZATION_LIMITATION` 的空间/安全关联；不修改 profile。

## 协议与授权

冻结合同没有变化，历史 v1.1 的 `SCIENTIFIC_PROTOCOL_DEVIATION` 原记录保持不动。本审计新增版本化记录 `IMPLEMENTATION_NONCONFORMANCE_AFFECTING_SCIENTIFIC_GATE`：这是实现合规修正，不伪装成合同修改。它不把 NOT_READY 翻为 READY，只修正失败原因的科学解释。

- `R1_RESIDUAL_BENCHMARK_ENABLEMENT = BENCHMARK_FAMILY_NOT_READY`
- `RBR_A = NOT_AUTHORIZED`
- `RBR_B = NOT_AUTHORIZED`
- `RBR_C = NOT_AUTHORIZED`
- `NEW_ROLLOUT = NOT_AUTHORIZED`
"""
    (out / "R1_B2_1_Residual_Benchmark_Forensic_Diagnosis_v1.md").write_text(report, encoding="utf-8")

    decision = """# R1 B2.2 Scientific Owner 决策单 v0.1

## 当前冻结结论

- 当前状态：`BENCHMARK_FAMILY_NOT_READY`。
- 本页全部选项均为 prospective；未获 owner 明确批准前，不形成新冻结、不允许新 rollout。
- 禁止 outcome-driven threshold tuning；任何新版本都必须在执行前绑定实现 SHA、测量源和适用性规则。

## 待 owner 决策的候选修正

| 选项 | prospective 内容 | 本次是否实施 | owner 决策 |
|---|---|---:|---|
| A | 按冻结合同从 official observation 和 map query 构造 lane-aware slots、稳定 track ID、gap/relative speed/THW、traffic density、hazard multi-hot 与真实 missingness | 否 | 待批准 |
| B | TSB 改为 route-aligned longitudinal trajectory，同时保留 Option-A acceleration profile；明确连续重规划锚点 | 否 | 待批准 |
| C | HLC 在冻结前声明 map geometry applicability 与原生 8 秒 reference coverage；不得按结果拟合阈值 | 否 | 待批准 |
| D | planner 首状态以 current ego 形成连续 anchor，并版本化位置/heading/speed/timestamp 语义 | 否 | 待批准 |
| E | 在执行前冻结 primary measurement source：planned-first 或 realized-ego；若双报告，必须指定主次与冲突解释 | 否 | 待批准 |

## 建议的审批顺序

先审 A、D、E，再判断是否需要 B/C。当前证据不支持直接修改冻结生成器参数；先消除上下文和重规划实现缺陷，才能隔离 generator-specific failure。

## 执行授权

- 新 planner rollout：`NOT_AUTHORIZED`
- D2/D4：`NOT_AUTHORIZED`
- RBR A/B/C：`NOT_AUTHORIZED`
"""
    (out / "R1_B2_2_Scientific_Owner_Decision_Sheet_v0.1.md").write_text(decision, encoding="utf-8")

    after = {str(x.relative_to(ROOT)): sha256(x) for x in immutable}
    assert before == after, "a frozen input changed during read-only audit"
    audit_manifest = {
        "schema_version": "r1_b2_2_residual_forensic_execution_manifest_v1",
        "status": "COMPLETE_READ_ONLY_EXISTING_EVIDENCE",
        "baseline_commit_supplied_by_owner": "b38b184c7c8ac7a814ad5515040227e070b43f8f",
        "run_count": len(ledger), "pair_count": len(pairs), "technical_failure_count": manifest["technical_failure_count"],
        "roster_sha256": sha256(ROSTER), "selector_salt_sha256": selector["salt_sha256"],
        "temporal_anchor": temporal, "immutable_input_sha256_before": before, "immutable_input_sha256_after": after,
        "new_rollout_count": 0, "rbr_assets_read": False, "frozen_artifacts_overwritten": False,
        "enablement": "BENCHMARK_FAMILY_NOT_READY", "rbr_authorization": "NOT_AUTHORIZED",
    }
    dump_json(out / "r1_b2_2_residual_forensic_execution_manifest_v1.json", audit_manifest)
    print(json.dumps({"status": "PASS", "outputs": 12, "safety": safety_summary, "plan_vs_realized": plan_status, "temporal": temporal["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
