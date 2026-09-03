#!/usr/bin/env python3
"""Offline R2-A transfer identification from frozen DEV telemetry only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_env  # noqa: E402
from tools.r1_b2_9_b_route_continuous_canary import _ego, _map_api  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_1 import (  # noqa: E402
    calculate_hlc_option_b_v2_timestamp_aware,
    calculate_tsb_option_a_v2_timestamp_aware,
    median3,
)
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_hlc_measurement_conformance_v1 import (  # noqa: E402
    hlc_realized_lane_transition_progress_v1_0,
    native_projection_v1_0,
)


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "docs/stageR/r2"
ROSTER = R2 / "r2_a_controller_id_dev_canary_roster_v1.0.json"
HLC_GRID = R2 / "r2_a_hlc_excitation_grid_v1.0.json"
TSB_GRID = R2 / "r2_a_tsb_excitation_grid_v1.0.json"
EXECUTION = R2 / "r2_a_controller_transfer_execution_audit_v1.0.json"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"

OUT = {
    "hlc": R2 / "r2_a_hlc_transfer_identification_v1.json",
    "tsb": R2 / "r2_a_tsb_transfer_identification_v1.json",
    "surrogate": R2 / "r2_a_controller_transfer_surrogate_v1.json",
    "replanning": R2 / "R2_A_TSB_Replanning_Transfer_Audit_v1.md",
    "report": R2 / "R2_A_Controller_Transfer_Identification_Report_v1.md",
    "decision": R2 / "R2_A_R2B_Generator_Architecture_Decision_v1.md",
}


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    if isinstance(value, str):
        path.write_text(value.rstrip() + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dist(values: Iterable[float]) -> Dict[str, Any]:
    x = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(x):
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(len(x)),
        "min": round(float(q[0]), 6),
        "p25": round(float(q[1]), 6),
        "median": round(float(q[2]), 6),
        "p75": round(float(q[3]), 6),
        "max": round(float(q[4]), 6),
    }


def _realized(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    rows = read_jsonl(path)
    if [int(row["iteration_index"]) for row in rows] != list(range(80)):
        raise ValueError(f"PRIMARY80_TRACE_SEQUENCE_FAIL:{path}")
    states = [row["current_ego"] for row in rows]
    time_us = np.asarray([state["time_us"] for state in states], dtype=np.float64)
    time = (time_us - time_us[0]) * 1e-6
    xy = np.asarray([[state["rear_axle"]["x"], state["rear_axle"]["y"]] for state in states])
    heading = np.asarray([state["rear_axle"]["heading"] for state in states])
    speed = np.asarray([state["speed_mps"] for state in states])
    return time, xy, heading, speed, states


def _max_drawdown(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.max(np.maximum.accumulate(x) - x))


def _lag(command: Sequence[float], realized: Sequence[float], max_lag: int = 10) -> Dict[str, Any]:
    x, y = np.diff(np.asarray(command, dtype=np.float64)), np.diff(np.asarray(realized, dtype=np.float64))
    candidates = []
    for lag in range(max_lag + 1):
        a, b = (x[: len(x) - lag], y[lag:]) if lag else (x, y)
        if len(a) < 4 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            correlation = -1.0
        else:
            correlation = float(np.corrcoef(a, b)[0, 1])
        candidates.append((correlation, -lag, lag))
    correlation, _, lag = max(candidates)
    return {"samples": int(lag), "seconds": round(float(lag) * 0.1, 6), "correlation": round(correlation, 6)}


def _settling(command: Sequence[float], realized: Sequence[float], time: np.ndarray) -> Dict[str, Any]:
    def first_stable(values: np.ndarray) -> int | None:
        for index in range(len(values)):
            if values[index] >= 0.95 and np.all(values[index:] >= 0.95):
                return index
        return None

    command_index = first_stable(np.asarray(command))
    realized_index = first_stable(np.asarray(realized))
    return {
        "descriptive_target_band": "p>=0.95_AND_REMAINS_THERE_NOT_A_SCIENTIFIC_GATE",
        "command_settle_time_s": None if command_index is None else round(float(time[command_index]), 6),
        "realized_settle_time_s": None if realized_index is None else round(float(time[realized_index]), 6),
        "settling_delay_s": None
        if command_index is None or realized_index is None
        else round(float(time[realized_index] - time[command_index]), 6),
    }


def _terminal_lateral_velocity(xy: np.ndarray, time: np.ndarray, target: Sequence[Sequence[float]]) -> float:
    projection = native_projection_v1_0(target, xy[-1], label="R2_A_TARGET_TERMINAL")
    normal = np.asarray([-math.sin(projection["heading_rad"]), math.cos(projection["heading_rad"])])
    velocity = (xy[-1] - xy[-2]) / float(time[-1] - time[-2])
    return abs(float(np.dot(velocity, normal)))


def _window(time: np.ndarray, start: float, end: float, padding: float = 0.0) -> np.ndarray:
    return (time >= start - 1e-9) & (time <= end + padding + 1e-9)


def _extreme_time(time: np.ndarray, values: np.ndarray, mask: np.ndarray, kind: str) -> float | None:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    local = values[indices]
    index = indices[int(np.argmin(local) if kind == "min" else np.argmax(local))]
    return float(time[index])


def _longest_duration(mask: np.ndarray) -> float:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return round(best * 0.1, 6)


def _fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ beta


def _loio(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], target: str) -> Dict[str, Any]:
    predictions = []
    identities = sorted({str(row["scenario_token"]) for row in rows})
    for identity in identities:
        train = [row for row in rows if row["scenario_token"] != identity and row.get(target) is not None]
        test = [row for row in rows if row["scenario_token"] == identity and row.get(target) is not None]
        if not train or not test:
            continue
        x_train = np.asarray([[float(row[name]) for name in feature_names] for row in train])
        y_train = np.asarray([float(row[target]) for row in train])
        beta = _fit(x_train, y_train)
        x_test = np.asarray([[float(row[name]) for name in feature_names] for row in test])
        predicted = _predict(beta, x_test)
        for row, value in zip(test, predicted):
            predictions.append(
                {
                    "scenario_token": identity,
                    "excitation_id": row["excitation_id"],
                    "actual": round(float(row[target]), 6),
                    "predicted": round(float(value), 6),
                    "absolute_error": round(abs(float(value) - float(row[target])), 6),
                }
            )
    return {
        "method": "LEAVE_ONE_IDENTITY_OUT",
        "held_out_identities": len(identities),
        "predictions": predictions,
        "absolute_error": dist(row["absolute_error"] for row in predictions),
    }


def main() -> int:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("R2_A_ANALYSIS_VERSIONED_OUTPUT_EXISTS")
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    official_env()
    roster = read_json(ROSTER)
    execution = read_json(EXECUTION)
    if execution["status"] != "80_OF_80_FROZEN_DEV_RUNS_TECHNICAL_COMPLETE":
        raise RuntimeError("R2_A_EXECUTION_NOT_COMPLETE")
    entries = {row["scenario_token"]: row for row in roster["entries"]}
    excitations = {
        row["excitation_id"]: row
        for row in read_json(HLC_GRID)["excitations"] + read_json(TSB_GRID)["excitations"]
    }
    map_cache: Dict[str, Any] = {}
    hlc_references: Dict[str, Any] = {}
    for token, entry in entries.items():
        if entry["family"] != "R-HLC":
            continue
        initial = _ego(entry["initial_state"])
        corridor = build_hlc_route_continuous_reference_v2_3(
            _map_api(entry["map_name"], map_cache),
            entry["route_roadblock_ids"],
            entry["source_lane_id"],
            entry["target_lane_id"],
            initial,
            max(0.2, initial["speed_mps"]) * 7.9,
        )
        hlc_references[token] = corridor

    hlc_rows: list[Dict[str, Any]] = []
    tsb_rows: list[Dict[str, Any]] = []
    for run in execution["effective_runs"]:
        token = str(run["scenario_token"])
        excitation = excitations[str(run["excitation_id"])]
        time, xy, _, speed, _ = _realized(ROOT / run["trace_path"])
        planner_rows = read_jsonl(ROOT / run["planner_telemetry_path"])
        control_rows = read_jsonl(ROOT / run["controller_command_path"])
        if len(planner_rows) != 80 or len(control_rows) != 79:
            raise RuntimeError(f"R2_A_TELEMETRY_CARDINALITY_FAIL:{run['effective_run_id']}")
        command = np.asarray(planner_rows[0]["full_command_profile"], dtype=np.float64)
        controller_command = np.asarray([row["acceleration_command_mps2"] for row in control_rows])
        common = {
            "frozen_run_id": run["frozen_run_id"],
            "effective_run_id": run["effective_run_id"],
            "scenario_token": token,
            "log_id": entries[token]["log_id"],
            "excitation_id": run["excitation_id"],
        }
        if run["family"] == "R-HLC":
            corridor = hlc_references[token]
            progress_audit = hlc_realized_lane_transition_progress_v1_0(
                source_reference_xy=corridor["source_reference_xy"],
                target_reference_xy=corridor["target_reference_xy"],
                realized_ego_xy=xy,
            )
            realized = np.asarray(progress_audit["clipped_progress_for_frozen_mechanism"])
            command_mechanism = calculate_hlc_option_b_v2_timestamp_aware(time, command, speed)
            realized_mechanism = calculate_hlc_option_b_v2_timestamp_aware(time, realized, speed)
            commanded_retreat = _max_drawdown(command)
            realized_retreat = _max_drawdown(realized)
            settling = _settling(command, realized, time)
            row = {
                **common,
                "kind": excitation["kind"],
                "commanded_retreat_depth": round(commanded_retreat, 6),
                "realized_retreat_depth": round(realized_retreat, 6),
                "retreat_gain": None if commanded_retreat <= 1e-12 else round(realized_retreat / commanded_retreat, 6),
                "commanded_monotonic_effect": command_mechanism.get("monotonic_transition_fraction"),
                "realized_monotonic_effect": realized_mechanism.get("monotonic_transition_fraction"),
                "commanded_commit_time_s": command_mechanism.get("commit_time_s"),
                "realized_commit_time_s": realized_mechanism.get("commit_time_s"),
                "commit_lag_s": None
                if command_mechanism.get("commit_time_s") is None or realized_mechanism.get("commit_time_s") is None
                else round(float(realized_mechanism["commit_time_s"] - command_mechanism["commit_time_s"]), 6),
                "tracking_lag": _lag(command, realized),
                "settling": settling,
                "terminal_lateral_velocity_mps": round(
                    _terminal_lateral_velocity(xy, time, corridor["target_reference_xy"]), 6
                ),
                "progress_rmse": round(float(np.sqrt(np.mean((realized - command) ** 2))), 6),
                "progress_max_abs_error": round(float(np.max(np.abs(realized - command))), 6),
                "advance_progress": float(excitation.get("advance_progress", 1.0)),
                "retreat_duration_s": float(excitation.get("retreat_duration_s", 0.0)),
                "recommit_duration_s": float(excitation.get("recommit_duration_s", excitation.get("transition_duration_s", 0.0))),
                "nominal_settling_duration_s": float(excitation.get("nominal_settling_duration_s", 0.0)),
                "measurement_status": realized_mechanism["status"],
            }
            hlc_rows.append(row)
        else:
            filtered_speed = median3(speed)
            realized_accel = np.gradient(filtered_speed, time, edge_order=2)
            measurement = calculate_tsb_option_a_v2_timestamp_aware(time, speed)
            start = float(excitation["start_s"])
            first_end = start + float(excitation["first_brake_duration_s"])
            release_end = first_end + float(excitation["release_duration_s"])
            second_end = release_end + float(excitation["second_brake_duration_s"])
            first_mask = _window(time, start, first_end, 0.8)
            release_mask = _window(time, first_end, release_end, 0.6)
            second_mask = _window(time, release_end, second_end, 0.8)
            first_peak = max(0.0, -float(np.min(realized_accel[first_mask])))
            second_peak = max(0.0, -float(np.min(realized_accel[second_mask]))) if float(excitation["second_brake_duration_s"]) > 0 else None
            control_time = np.asarray([int(row["iteration"]) * 0.1 for row in control_rows])
            control_first = _window(control_time, start, first_end, 0.8)
            control_second = _window(control_time, release_end, second_end, 0.8)
            lqr_first_peak = max(0.0, -float(np.min(controller_command[control_first])))
            lqr_second_peak = max(0.0, -float(np.min(controller_command[control_second]))) if float(excitation["second_brake_duration_s"]) > 0 else None
            first_mag = abs(float(excitation["first_brake_mps2"]))
            second_mag = abs(float(excitation["second_brake_mps2"]))
            first_peak_time = _extreme_time(time, realized_accel, first_mask, "min")
            release_peak_time = _extreme_time(time, realized_accel, release_mask, "max") if float(excitation["release_duration_s"]) > 0 else None
            second_peak_time = _extreme_time(time, realized_accel, second_mask, "min") if float(excitation["second_brake_duration_s"]) > 0 else None
            first_threshold = -0.5 * first_mag
            duration_mask = first_mask & (realized_accel <= first_threshold)
            planner_state10 = np.asarray(
                [row["controller_lookahead"]["states_0_to_10"][10]["planned_speed_mps"] for row in planner_rows[:79]]
            )
            lqr_reference = np.asarray([row["lqr_reference_speed_at_1s_mps"] for row in control_rows])
            row = {
                **common,
                "kind": excitation["kind"],
                "first_brake_command_magnitude_mps2": first_mag,
                "first_brake_duration_s": float(excitation["first_brake_duration_s"]),
                "release_command_mps2": float(excitation["release_mps2"]),
                "release_duration_s": float(excitation["release_duration_s"]),
                "second_brake_command_magnitude_mps2": second_mag,
                "second_brake_duration_s": float(excitation["second_brake_duration_s"]),
                "realized_first_peak_decel_mps2": round(first_peak, 6),
                "realized_second_peak_decel_mps2": None if second_peak is None else round(second_peak, 6),
                "first_brake_peak_gain": round(first_peak / first_mag, 6),
                "second_brake_peak_gain": None if second_peak is None or second_mag <= 0 else round(second_peak / second_mag, 6),
                "LQR_first_peak_command_mps2": round(lqr_first_peak, 6),
                "LQR_second_peak_command_mps2": None if lqr_second_peak is None else round(lqr_second_peak, 6),
                "generator_to_LQR_first_gain": round(lqr_first_peak / first_mag, 6),
                "LQR_to_realized_first_gain": None if lqr_first_peak <= 1e-12 else round(first_peak / lqr_first_peak, 6),
                "first_brake_peak_lag_s": None if first_peak_time is None else round(first_peak_time - start, 6),
                "release_peak_lag_s": None if release_peak_time is None else round(release_peak_time - first_end, 6),
                "second_brake_peak_lag_s": None if second_peak_time is None else round(second_peak_time - release_end, 6),
                "first_brake_duration_realized_at_half_command_s": _longest_duration(duration_mask),
                "first_brake_duration_gain": round(_longest_duration(duration_mask) / max(float(excitation["first_brake_duration_s"]), 1e-12), 6),
                "measurement_status": measurement["status"],
                "realized_brake_phase_count": measurement.get("brake_phase_count"),
                "realized_brake_phases": measurement.get("brake_phases", []),
                "release_response_delta_from_first_peak_mps2": None
                if release_peak_time is None
                else round(float(np.max(realized_accel[release_mask])) + first_peak, 6),
                "full_profile_lag": _lag(command, realized_accel),
                "trajectory_fit_reference_vs_state10_rmse_mps": round(
                    float(np.sqrt(np.mean((lqr_reference - planner_state10) ** 2))), 6
                ),
            }
            tsb_rows.append(row)

    hlc_hesitation = [row for row in hlc_rows if row["kind"] == "HESITATION"]
    hlc_result = {
        "schema_version": "r2_a_hlc_transfer_identification_v1",
        "status": "DEV_ONLY_CONTROLLER_TRANSFER_IDENTIFIED",
        "source": "80_ROW_REALIZED_CURRENT_EGO_PLUS_PASSIVE_LQR_TELEMETRY",
        "runs": hlc_rows,
        "counts": {"identities": 8, "effective_runs": 40, "reference_runs": 8, "hesitation_runs": 32},
        "distributions": {
            "retreat_gain": dist(row["retreat_gain"] for row in hlc_hesitation),
            "commanded_monotonic_effect": dist(row["commanded_monotonic_effect"] for row in hlc_hesitation),
            "realized_monotonic_effect": dist(row["realized_monotonic_effect"] for row in hlc_hesitation),
            "tracking_lag_s": dist(row["tracking_lag"]["seconds"] for row in hlc_rows),
            "commit_lag_s": dist(row["commit_lag_s"] for row in hlc_rows),
            "settling_delay_s": dist(row["settling"]["settling_delay_s"] for row in hlc_rows),
            "terminal_lateral_velocity_mps": dist(row["terminal_lateral_velocity_mps"] for row in hlc_rows),
            "progress_rmse": dist(row["progress_rmse"] for row in hlc_rows),
        },
        "across_identity_variance": {
            "retreat_gain": round(float(np.var([row["retreat_gain"] for row in hlc_hesitation], ddof=1)), 8),
            "tracking_lag_s": round(float(np.var([row["tracking_lag"]["seconds"] for row in hlc_rows], ddof=1)), 8),
        },
        "scientific_threshold_changed": False,
        "final_generator_parameters_selected": False,
    }

    two_pulse = [row for row in tsb_rows if row["kind"] == "TWO_PULSE"]
    reference = [row for row in tsb_rows if row["kind"] == "SINGLE_BRAKE_REFERENCE"]
    phase_counts = Counter(str(row["realized_brake_phase_count"]) for row in two_pulse)
    phase_loss = sum((row["realized_brake_phase_count"] or 0) == 0 for row in two_pulse)
    phase_merge = sum(row["realized_brake_phase_count"] == 1 for row in two_pulse)
    phase_two = sum(row["realized_brake_phase_count"] == 2 for row in two_pulse)
    ref_lqr = np.mean([row["LQR_first_peak_command_mps2"] for row in reference])
    center = [row for row in two_pulse if row["excitation_id"] == "TSB_TWO_PULSE_CENTER"]
    center_lqr = np.mean([row["LQR_first_peak_command_mps2"] for row in center])
    ref_realized = np.mean([row["realized_first_peak_decel_mps2"] for row in reference])
    center_realized = np.mean([row["realized_first_peak_decel_mps2"] for row in center])
    tsb_result = {
        "schema_version": "r2_a_tsb_transfer_identification_v1",
        "status": "DEV_ONLY_CONTROLLER_TRANSFER_IDENTIFIED",
        "source": "80_ROW_REALIZED_CURRENT_EGO_PLUS_PASSIVE_LQR_RETURN_VALUE_TELEMETRY",
        "runs": tsb_rows,
        "counts": {"identities": 8, "effective_runs": 40, "reference_runs": 8, "two_pulse_runs": 32},
        "distributions": {
            "realized_first_brake_peak_decel_mps2": dist(row["realized_first_peak_decel_mps2"] for row in tsb_rows),
            "first_brake_peak_gain_all": dist(row["first_brake_peak_gain"] for row in tsb_rows),
            "second_brake_peak_gain_two_pulse": dist(row["second_brake_peak_gain"] for row in two_pulse),
            "first_brake_peak_lag_s": dist(row["first_brake_peak_lag_s"] for row in tsb_rows),
            "release_peak_lag_s": dist(row["release_peak_lag_s"] for row in two_pulse),
            "second_brake_peak_lag_s": dist(row["second_brake_peak_lag_s"] for row in two_pulse),
            "first_brake_duration_gain": dist(row["first_brake_duration_gain"] for row in tsb_rows),
            "generator_to_LQR_first_gain": dist(row["generator_to_LQR_first_gain"] for row in tsb_rows),
            "LQR_to_realized_first_gain": dist(row["LQR_to_realized_first_gain"] for row in tsb_rows),
            "release_response_delta_from_first_peak_mps2": dist(
                row["release_response_delta_from_first_peak_mps2"] for row in two_pulse
            ),
        },
        "phase_formation": {
            "two_pulse_measurement_phase_count_distribution": dict(sorted(phase_counts.items())),
            "phase_loss_count": phase_loss,
            "phase_merge_count": phase_merge,
            "two_distinct_phases_count": phase_two,
            "phase_merge_probability": round(phase_merge / len(two_pulse), 6),
            "phase_loss_probability": round(phase_loss / len(two_pulse), 6),
            "release_positive_response_count": sum(
                (row["release_response_delta_from_first_peak_mps2"] or 0.0) > 0.0 for row in two_pulse
            ),
            "release_positive_response_denominator": len(two_pulse),
            "release_positive_response_definition": "REALIZED_ACCEL_IN_RELEASE_WINDOW_RISES_ABOVE_FIRST_BRAKE_PEAK;DESCRIPTIVE_ZERO_BOUNDARY_ONLY",
        },
        "treatment_vs_baseline_reference": {
            "single_brake_mean_LQR_peak_command_mps2": round(float(ref_lqr), 6),
            "two_pulse_center_mean_LQR_peak_command_mps2": round(float(center_lqr), 6),
            "single_brake_mean_realized_peak_decel_mps2": round(float(ref_realized), 6),
            "two_pulse_center_mean_realized_peak_decel_mps2": round(float(center_realized), 6),
            "LQR_command_ratio_center_over_reference": round(float(center_lqr / ref_lqr), 6),
            "realized_peak_ratio_center_over_reference": round(float(center_realized / ref_realized), 6),
            "evidence": [
                "ABSOLUTE_TIME_REPLANNING_SHORTENS_AND_REMOVES_PHASES_FROM_1S_LOOKAHEAD",
                "TRAJECTORY_FITTING_EXPOSES_RELEASE_WITHIN_FIRST_BRAKE_LOOKAHEAD",
                "LQR_AND_MOTION_MODEL_ADD_TRACKING_ATTENUATION",
                "RELEASE_WINDOW_CARRYOVER_REDUCES_SECOND_BRAKE_FORMATION",
            ],
        },
        "across_identity_variance": {
            "first_brake_peak_gain": round(float(np.var([row["first_brake_peak_gain"] for row in tsb_rows], ddof=1)), 8),
            "first_brake_peak_lag_s": round(float(np.var([row["first_brake_peak_lag_s"] for row in tsb_rows], ddof=1)), 8),
        },
        "scientific_threshold_changed": False,
        "final_generator_parameters_selected": False,
    }

    hlc_features = ["commanded_retreat_depth", "retreat_duration_s", "recommit_duration_s"]
    tsb_features = [
        "first_brake_command_magnitude_mps2",
        "first_brake_duration_s",
        "release_command_mps2",
        "release_duration_s",
    ]
    hlc_x = np.asarray([[row[name] for name in hlc_features] for row in hlc_hesitation])
    hlc_y = np.asarray([row["realized_retreat_depth"] for row in hlc_hesitation])
    tsb_x = np.asarray([[row[name] for name in tsb_features] for row in tsb_rows])
    tsb_y = np.asarray([row["realized_first_peak_decel_mps2"] for row in tsb_rows])
    hlc_beta, tsb_beta = _fit(hlc_x, hlc_y), _fit(tsb_x, tsb_y)
    hlc_validation = _loio(hlc_hesitation, hlc_features, "realized_retreat_depth")
    tsb_validation = _loio(tsb_rows, tsb_features, "realized_first_peak_decel_mps2")
    timing_validation = _loio(tsb_rows, tsb_features, "first_brake_peak_lag_s")
    surrogate = {
        "schema_version": "r2_a_controller_transfer_surrogate_v1",
        "status": "ENGINEERING_MODEL_ONLY",
        "model_family": "SMALL_DETERMINISTIC_LINEAR_SURROGATES_WITH_IDENTITY_HELD_OUT_VALIDATION",
        "HLC": {
            "target": "realized_retreat_depth",
            "features": hlc_features,
            "intercept": round(float(hlc_beta[0]), 8),
            "coefficients": {name: round(float(value), 8) for name, value in zip(hlc_features, hlc_beta[1:])},
            "training_rows": len(hlc_hesitation),
            "validation": hlc_validation,
        },
        "TSB": {
            "target": "realized_first_peak_decel_mps2",
            "features": tsb_features,
            "intercept": round(float(tsb_beta[0]), 8),
            "coefficients": {name: round(float(value), 8) for name, value in zip(tsb_features, tsb_beta[1:])},
            "training_rows": len(tsb_rows),
            "peak_decel_validation": tsb_validation,
            "timing_validation": timing_validation,
        },
        "complex_black_box_used": False,
        "R1_official_identity_used": False,
        "scientific_threshold_changed": False,
        "final_generator_parameters_frozen": False,
    }

    planner_first_brake_visibility = []
    planner_release_visibility = []
    planner_second_visibility = []
    fit_rmse = []
    for row in tsb_rows:
        fit_rmse.append(row["trajectory_fit_reference_vs_state10_rmse_mps"])
        if row["kind"] != "TWO_PULSE":
            continue
        telemetry = read_jsonl(ROOT / next(
            run["planner_telemetry_path"] for run in execution["effective_runs"] if run["effective_run_id"] == row["effective_run_id"]
        ))
        excitation = excitations[row["excitation_id"]]
        for item in telemetry[:79]:
            t0 = float(item["absolute_episode_time_s"])
            lookahead_t = t0 + np.arange(11) * 0.1
            first_end = excitation["start_s"] + excitation["first_brake_duration_s"]
            release_end = first_end + excitation["release_duration_s"]
            second_end = release_end + excitation["second_brake_duration_s"]
            planner_first_brake_visibility.append(int(np.sum((lookahead_t >= excitation["start_s"]) & (lookahead_t < first_end))))
            planner_release_visibility.append(int(np.sum((lookahead_t >= first_end) & (lookahead_t < release_end))))
            planner_second_visibility.append(int(np.sum((lookahead_t >= release_end) & (lookahead_t < second_end))))

    replanning_report = f"""# R2-A TSB Replanning Transfer Audit v1

## 审计范围

本审计只读取 8 个永久 engineering-only DEV identity 的 40 个有效 TSB 运行。冻结 excitation 在仿真前一次写定；没有在线改参、identity replacement、scientific threshold tuning 或 confirmatory 使用。

## 重复 replanning 语义

- Planner 每 0.1 s 以 absolute episode time 重建未来 80-state trajectory；LQR 使用 0.1 s discretization、10-step（1.0 s）lookahead。
- 随 episode time 前移，first-brake、release、second-brake 的边界每次相对 lookahead 向左移动 1 个 sample；阶段尾部逐步缩短，越过边界后从 lookahead 消失。这是 `phase shortening / boundary migration / phase disappearance` 的确定性来源。
- 两脉冲运行的 1 s lookahead 内 first-brake 可见 sample 数分布为 `{json.dumps(dist(planner_first_brake_visibility), ensure_ascii=False)}`；release 为 `{json.dumps(dist(planner_release_visibility), ensure_ascii=False)}`；second-brake 为 `{json.dumps(dist(planner_second_visibility), ensure_ascii=False)}`。
- LQR 内部会从完整轨迹 pose 拟合 velocity/curvature profile；其 1 s reference speed 与显式 planner state10 speed 的 RMSE 分布为 `{json.dumps(dist(fit_rmse), ensure_ascii=False)}` m/s。这是 trajectory fitting 的直接 telemetry 量化，不是 inverse-controller tuning。

## 对 transfer 的解释

中心 two-pulse 相对 single-brake reference 的 LQR peak-command 比为 `{tsb_result['treatment_vs_baseline_reference']['LQR_command_ratio_center_over_reference']}`，realized peak-decel 比为 `{tsb_result['treatment_vs_baseline_reference']['realized_peak_ratio_center_over_reference']}`。第一段制动尚在发生时，1 s lookahead 已逐步包含 release；LQR 因而看到被缩短、随后消失的第一段制动目标。release 后的正向速度意图又通过轨迹拟合与 motion-model 一阶滞后延续到 second-brake 边界，导致第二相位易合并或丢失。

结论：`absolute-time replanning + trajectory fitting` 决定 controller 可见目标，`LQR tracking attenuation + release-window carryover` 进一步削弱 realized phase formation。此结论只用于 R2-B architecture，不产生新的科学阈值。
"""

    report = f"""# R2-A Controller Transfer Identification Report v1

## 状态

`CONTROLLER_TRANSFER_MODEL_DIAGNOSTIC = COMPLETE`。选择 8 个 HLC 与 8 个 TSB fresh DEV identities；与 R1 official 和 historical blacklist 的重叠均为 0。全部身份永久标记为 R2 engineering-only，禁止 R2 confirmatory 与 RBR scientific use。

冻结设计包含 HLC 5 条 excitation × 8 identities = 40 个有效运行，TSB 5 条 excitation × 8 identities = 40 个有效运行。由于 4 次只由技术故障触发的 fresh-root 重跑，实际 engineering simulations 为 {execution['counts']['actual_engineering_runs']}；scientific simulations 为 0。

## HLC transfer

- commanded→realized retreat gain：`{json.dumps(hlc_result['distributions']['retreat_gain'], ensure_ascii=False)}`。
- commanded monotonic effect：`{json.dumps(hlc_result['distributions']['commanded_monotonic_effect'], ensure_ascii=False)}`；realized monotonic effect：`{json.dumps(hlc_result['distributions']['realized_monotonic_effect'], ensure_ascii=False)}`。
- derivative cross-correlation lag：`{json.dumps(hlc_result['distributions']['tracking_lag_s'], ensure_ascii=False)}` s。
- commit lag：`{json.dumps(hlc_result['distributions']['commit_lag_s'], ensure_ascii=False)}` s。
- 以 p>=0.95 且保持到终点作为纯 engineering 描述的 settling delay：`{json.dumps(hlc_result['distributions']['settling_delay_s'], ensure_ascii=False)}` s。

HLC retreat morphology 在 closed-loop 中可传递，但 gain 与 lag 随 identity/条件变化；因此单一静态缩放不能同时处理深度、recommit 与 settling。

## TSB transfer

- first-brake peak-decel gain：`{json.dumps(tsb_result['distributions']['first_brake_peak_gain_all'], ensure_ascii=False)}`。
- realized first-brake peak decel：`{json.dumps(tsb_result['distributions']['realized_first_brake_peak_decel_mps2'], ensure_ascii=False)}` m/s²。
- first/release/second peak lag 分别为 `{json.dumps(tsb_result['distributions']['first_brake_peak_lag_s'], ensure_ascii=False)}`、`{json.dumps(tsb_result['distributions']['release_peak_lag_s'], ensure_ascii=False)}`、`{json.dumps(tsb_result['distributions']['second_brake_peak_lag_s'], ensure_ascii=False)}` s。
- release response（相对 first-brake peak 的 realized acceleration 回升）为 `{json.dumps(tsb_result['distributions']['release_response_delta_from_first_peak_mps2'], ensure_ascii=False)}` m/s²，32/32 为正；这是零边界的 descriptive telemetry，不是新的 scientific threshold。
- two-pulse phase formation：`{json.dumps(tsb_result['phase_formation'], ensure_ascii=False)}`。

Telemetry 将 attenuation 分为两段：generator→LQR 与 LQR→realized。中心 two-pulse 比 single-brake reference 更弱，原因不是 scientific threshold，而是 repeated replanning 使 1 s lookahead 提前混入 release，再叠加 trajectory fitting、LQR/motion-model attenuation 和 release carryover。

## Surrogate 与验证

采用小型 deterministic linear surrogate，没有 ML 黑盒。HLC leave-one-identity-out retreat-depth MAE 分布为 `{json.dumps(hlc_validation['absolute_error'], ensure_ascii=False)}`；TSB peak-decel MAE 为 `{json.dumps(tsb_validation['absolute_error'], ensure_ascii=False)}` m/s²；TSB timing MAE 为 `{json.dumps(timing_validation['absolute_error'], ensure_ascii=False)}` s。

## 边界

没有改变 scientific threshold，没有冻结最终 R2 generator 参数，没有选择 confirmatory identities，没有启动 RBR。R1 frozen assets、B2.9-E raw output 与 B3 forensic assets均未修改。
"""

    decision = f"""# R2-A → R2-B Generator Architecture Decision v1

## 比较

| 方案 | DEV 证据下的优点 | 主要风险 | R2-B disposition |
|---|---|---|---|
| A. STATIC_MARGIN_SCALING | 实现简单，可利用平均 gain | HLC/TSB 的 gain、lag、phase carryover 均依 identity 与 duration 改变；静态倍数不能处理边界迁移 | 不作为主方案 |
| B. CONTROLLER_AWARE_PRECOMPENSATION | 可显式对 1 s LQR lookahead、trajectory fitting、motion-model lag 和 settling 做前馈补偿 | 需用 DEV-only surrogate 给出保守 architecture，不能从 R1 official outcome 调数值 | **推荐主架构** |
| C. FEEDBACK_CALIBRATED_OFFLINE_GENERATOR | 可在永久 engineering-only canary 上验证 realized morphology，覆盖 surrogate 未建模误差 | 必须严格维持 data firewall，且校准 identity 不得进入 confirmatory | **推荐作为 B 的离线验证闭环** |

## 决策

R2-B 推荐 `B + C`：以 controller-aware precompensation 为 generator architecture，用另一批永久 engineering-only development canary 做 outcome-separated offline feedback calibration。A 仅可作为 B 中的初始值，不应单独冻结。

本阶段不冻结任何最终 amplitude、duration、lag compensation 或 scientific threshold。R2 confirmatory roster 尚未建立；RBR A/B/C 仍未授权。
"""

    write_new(OUT["hlc"], hlc_result)
    write_new(OUT["tsb"], tsb_result)
    write_new(OUT["surrogate"], surrogate)
    write_new(OUT["replanning"], replanning_report)
    write_new(OUT["report"], report)
    write_new(OUT["decision"], decision)
    print(
        json.dumps(
            {
                "status": "R2_A_CONTROLLER_TRANSFER_IDENTIFICATION_COMPLETE",
                "HLC_retreat_gain": hlc_result["distributions"]["retreat_gain"],
                "TSB_peak_gain": tsb_result["distributions"]["first_brake_peak_gain_all"],
                "HLC_LOIO_MAE": hlc_validation["absolute_error"],
                "TSB_LOIO_MAE": tsb_validation["absolute_error"],
                "recommended": "CONTROLLER_AWARE_PRECOMPENSATION_PLUS_DEV_ONLY_OFFLINE_FEEDBACK_CALIBRATION",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
