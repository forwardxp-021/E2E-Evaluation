#!/usr/bin/env python3
"""Evaluate Stage7L lane/Frenet mechanisms without reading representations or BDD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from shapely.geometry import LineString, Point


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2))) if len(values) else float("nan")


def first_sustained(mask: np.ndarray, minimum_samples: int) -> int | None:
    count = 0
    for index, value in enumerate(mask):
        count = count + 1 if value else 0
        if count >= minimum_samples:
            return index - count + 1
    return None


def read_official_safety_metrics(root: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    """Read official collision/drivable-area parquet outputs when available."""
    import pandas as pd

    result: Dict[tuple[str, str], Dict[str, Any]] = {}
    for planner_dir in sorted(root.glob("scenario_*/*")):
        metrics_dir = planner_dir / "metrics"
        collision_path = metrics_dir / "no_ego_at_fault_collisions.parquet"
        drivable_path = metrics_dir / "drivable_area_compliance.parquet"
        if not collision_path.is_file() or not drivable_path.is_file():
            continue
        collision = pd.read_parquet(collision_path).iloc[0]
        drivable = pd.read_parquet(drivable_path).iloc[0]
        no_collision = bool(collision["no_ego_at_fault_collisions_stat_value"])
        drivable_compliant = bool(drivable["drivable_area_compliance_stat_value"])
        scenario_token = str(collision["scenario_name"])
        result[(scenario_token, planner_dir.name)] = {
            "collision": not no_collision,
            "at_fault_collision_count": int(collision["number_of_all_at_fault_collisions_stat_value"]),
            "offroad": not drivable_compliant,
            "drivable_area_compliant": drivable_compliant,
            "official_safety_metrics_available": True,
        }
    return result


def evaluate_one(rows: Sequence[Mapping[str, str]], maneuver: Mapping[str, Any]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["timestep_index"]))
    t = np.asarray([float(row["time_s"]) for row in ordered], dtype=np.float64)
    xy = np.asarray([[float(row["x"]), float(row["y"])] for row in ordered], dtype=np.float64)
    yaw = np.unwrap(np.asarray([float(row["yaw"]) for row in ordered], dtype=np.float64))
    speed = np.asarray([float(row["speed"]) for row in ordered], dtype=np.float64)
    if len(t) < 5 or not np.isfinite(np.c_[t, xy, yaw, speed]).all() or np.any(np.diff(t) <= 0):
        raise ValueError("trajectory is non-finite, too short, or non-monotonic")
    source = LineString(maneuver["source_reference_xy"])
    target = LineString(maneuver["target_reference_xy"])
    source_distance = np.asarray([Point(*point).distance(source) for point in xy])
    target_distance = np.asarray([Point(*point).distance(target) for point in xy])
    target_progress = np.asarray([target.project(Point(*point)) for point in xy])
    lane_half_width = max(1.3, float(maneuver["nominal_lane_width_m"]) * 0.5)
    departure = first_sustained(source_distance > lane_half_width, 2)
    entry = first_sustained(target_distance < lane_half_width, 2)
    sample_dt = float(np.median(np.diff(t)))
    settle_samples = max(2, int(round(1.0 / sample_dt)))
    settling = first_sustained(target_distance <= 0.5, settle_samples)
    acceleration = np.gradient(speed, t, edge_order=2)
    jerk = np.gradient(acceleration, t, edge_order=2)
    yaw_rate = np.gradient(yaw, t, edge_order=2)
    lateral_accel = speed * yaw_rate
    lateral_jerk = np.gradient(lateral_accel, t, edge_order=2)
    curvature = np.divide(yaw_rate, speed, out=np.zeros_like(speed), where=speed > 0.2)
    curvature_rate = np.gradient(curvature, t, edge_order=2)
    completion = settling is not None
    return {
        "scenario_token": maneuver["scenario_token"], "planner_name": ordered[0]["planner_name"],
        "valid": True, "lane_change_completion": completion,
        "source_lane_departure_time_s": None if departure is None else float(t[departure] - t[0]),
        "target_lane_entry_time_s": None if entry is None else float(t[entry] - t[0]),
        "target_center_settling_time_s": None if settling is None else float(t[settling] - t[0]),
        "lane_change_duration_s": None if departure is None or settling is None else float(t[settling] - t[departure]),
        "rms_lateral_accel_mps2": rms(lateral_accel), "peak_lateral_accel_mps2": float(np.max(np.abs(lateral_accel))),
        "rms_yaw_rate_radps": rms(yaw_rate), "peak_yaw_rate_radps": float(np.max(np.abs(yaw_rate))),
        "rms_lateral_jerk_mps3": rms(lateral_jerk), "peak_lateral_jerk_mps3": float(np.max(np.abs(lateral_jerk))),
        "rms_curvature_inv_m": rms(curvature), "peak_curvature_inv_m": float(np.max(np.abs(curvature))),
        "rms_curvature_rate_inv_ms": rms(curvature_rate),
        "final_target_lane_center_offset_m": float(target_distance[-1]),
        "mean_speed_mps": float(np.mean(speed)), "rms_longitudinal_accel_mps2": rms(acceleration),
        "rms_longitudinal_jerk_mps3": rms(jerk),
        "route_progress_m": float(target_progress[-1] - target_progress[0]),
        "collision": "N/A_NOT_IN_TRAJECTORY_EXPORT", "offroad": "N/A_NOT_IN_TRAJECTORY_EXPORT",
        "invalid_or_incomplete": not completion,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = json.loads(args.maneuver_manifest.read_text(encoding="utf-8"))
    by_token = {row["scenario_token"]: row for row in manifest["maneuvers"]}
    rows = read_csv(args.trajectory_csv)
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        token = str(row.get("scene_token") or row.get("scenario_token") or "")
        if token in by_token:
            grouped.setdefault((token, row["planner_name"]), []).append(row)
    metrics = [evaluate_one(group, by_token[token]) for (token, _), group in sorted(grouped.items())]
    if not metrics:
        raise ValueError("no Stage7L trajectory row matched the maneuver manifest")
    official_safety = read_official_safety_metrics(args.official_runs_root) if args.official_runs_root else {}
    for row in metrics:
        safety = official_safety.get((row["scenario_token"], row["planner_name"]))
        if safety:
            row.update(safety)
    dose0 = {row["scenario_token"]: row for row in metrics if row["planner_name"].endswith("dose0")}
    for row in metrics:
        baseline = dose0.get(row["scenario_token"])
        for key in ("mean_speed_mps", "rms_longitudinal_accel_mps2", "rms_longitudinal_jerk_mps3", "route_progress_m"):
            row[f"delta_vs_dose0_{key}"] = None if baseline is None else float(row[key]) - float(baseline[key])
    scenario_ordering: List[bool] = []
    for token in sorted({row["scenario_token"] for row in metrics}):
        token_rows = [row for row in metrics if row["scenario_token"] == token]
        peak_lateral = []
        for dose in (0, 25, 50, 75, 100):
            matches = [row for row in token_rows if row["planner_name"].endswith(f"dose{dose}")]
            if len(matches) == 1:
                peak_lateral.append(float(matches[0]["peak_lateral_accel_mps2"]))
        scenario_ordering.append(
            len(peak_lateral) == 5
            and all(current < following for current, following in zip(peak_lateral, peak_lateral[1:]))
        )
    lateral_dose_ordering = bool(scenario_ordering) and all(scenario_ordering)
    nuisance_keys = (
        "delta_vs_dose0_mean_speed_mps",
        "delta_vs_dose0_rms_longitudinal_accel_mps2",
        "delta_vs_dose0_rms_longitudinal_jerk_mps3",
        "delta_vs_dose0_route_progress_m",
    )
    max_abs_nuisance = {
        key: max(abs(float(row[key])) for row in metrics if row[key] is not None) for key in nuisance_keys
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in metrics for key in row})
    csv_path = args.output_dir / "stage7l_a2_lateral_mechanism_metrics.csv"
    write_csv(csv_path, metrics, fields)
    scenario_count = len({row["scenario_token"] for row in metrics})
    summary = {
        "schema_version": "stage7l_lateral_mechanism_v2",
        "status": "DEVELOPMENT_MECHANISM_EVALUATED_NO_BDD" if scenario_count > 1 else "A2_SMOKE_MECHANISM_EVALUATED_NO_BDD",
        "scenario_count": scenario_count,
        "trajectory_group_count": len(metrics),
        "all_valid": all(row["valid"] for row in metrics),
        "all_complete": all(row["lane_change_completion"] for row in metrics),
        "lateral_peak_accel_strict_dose_ordering": lateral_dose_ordering,
        "lateral_peak_accel_strict_dose_ordering_scenario_fraction": float(np.mean(scenario_ordering)),
        "official_safety_metrics_available": len(official_safety) == len(metrics),
        "all_collision_free": all(row.get("collision") is False for row in metrics),
        "all_drivable_area_compliant": all(row.get("offroad") is False for row in metrics),
        "max_abs_realized_longitudinal_nuisance": max_abs_nuisance,
        "embedding_or_bdd_read": False,
        "metrics_csv": str(csv_path.resolve()),
        "metrics": metrics,
    }
    (args.output_dir / "stage7l_a2_lateral_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_csv", type=Path, required=True)
    parser.add_argument("--maneuver_manifest", type=Path, required=True)
    parser.add_argument("--official_runs_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
