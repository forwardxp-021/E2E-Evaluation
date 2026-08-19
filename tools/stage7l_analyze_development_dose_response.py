#!/usr/bin/env python3
"""Summarize Stage7L-B physical dose response without representations or BDD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


DOSES = (0, 25, 50, 75, 100)
LATERAL_METRICS = (
    "lane_change_duration_s", "rms_lateral_accel_mps2", "peak_lateral_accel_mps2",
    "rms_yaw_rate_radps", "peak_yaw_rate_radps", "rms_lateral_jerk_mps3",
    "peak_lateral_jerk_mps3", "target_center_settling_time_s", "final_target_lane_center_offset_m",
)
NUISANCE_METRICS = (
    "delta_vs_dose0_mean_speed_mps", "delta_vs_dose0_rms_longitudinal_accel_mps2",
    "delta_vs_dose0_rms_longitudinal_jerk_mps3", "delta_vs_dose0_route_progress_m",
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dose_of(planner: str) -> int:
    match = re.search(r"dose(0|25|50|75|100)$", planner)
    if not match:
        raise ValueError(f"unrecognized planner dose: {planner}")
    return int(match.group(1))


def finite_values(rows: List[Dict[str, Any]], metric: str) -> np.ndarray:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(replicates, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = read_csv(args.mechanism_metrics_csv)
    rows: List[Dict[str, Any]] = []
    for row in raw:
        item: Dict[str, Any] = dict(row)
        item["dose"] = dose_of(row["planner_name"])
        rows.append(item)
    tokens = sorted({row["scenario_token"] for row in rows})
    if len(tokens) != 24 or len(rows) != 120:
        raise ValueError(f"expected 24x5 complete metrics, got {len(tokens)} tokens and {len(rows)} rows")
    by_token_dose = {(row["scenario_token"], row["dose"]): row for row in rows}
    if len(by_token_dose) != 120:
        raise ValueError("duplicate or missing token-dose metrics")
    rng = np.random.default_rng(args.bootstrap_seed)
    dose_summary: List[Dict[str, Any]] = []
    for dose in DOSES:
        group = [row for row in rows if row["dose"] == dose]
        for metric in LATERAL_METRICS:
            values = finite_values(group, metric)
            lo, hi = bootstrap_median_ci(values, rng, args.bootstrap_replicates)
            dose_summary.append({
                "dose": dose, "metric": metric, "n": len(values), "mean": float(np.mean(values)),
                "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)), "bootstrap_median_ci_low": lo,
                "bootstrap_median_ci_high": hi,
            })
    paired: List[Dict[str, Any]] = []
    expected_negative = {"lane_change_duration_s", "target_center_settling_time_s"}
    for token in tokens:
        reference = by_token_dose[(token, 0)]
        for dose in DOSES[1:]:
            target = by_token_dose[(token, dose)]
            for metric in LATERAL_METRICS:
                delta = float(target[metric]) - float(reference[metric])
                expected = "negative" if metric in expected_negative else (
                    "absolute_small" if metric == "final_target_lane_center_offset_m" else "positive"
                )
                consistent = delta < 0 if expected == "negative" else (delta > 0 if expected == "positive" else abs(float(target[metric])) <= 0.25)
                paired.append({
                    "scenario_token": token, "dose": dose, "reference_dose": 0, "metric": metric,
                    "reference_value": float(reference[metric]), "target_value": float(target[metric]),
                    "paired_delta": delta, "expected_direction": expected, "directionally_consistent": consistent,
                })
    paired_summary: List[Dict[str, Any]] = []
    for dose in DOSES[1:]:
        for metric in LATERAL_METRICS:
            group = [row for row in paired if row["dose"] == dose and row["metric"] == metric]
            values = np.asarray([row["paired_delta"] for row in group], dtype=np.float64)
            lo, hi = bootstrap_median_ci(values, rng, args.bootstrap_replicates)
            paired_summary.append({
                "dose": dose, "metric": metric, "n": len(values), "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)), "q25_delta": float(np.quantile(values, 0.25)),
                "q75_delta": float(np.quantile(values, 0.75)), "bootstrap_median_delta_ci_low": lo,
                "bootstrap_median_delta_ci_high": hi,
                "directional_consistency_fraction": float(np.mean([row["directionally_consistent"] for row in group])),
            })
    nuisance: List[Dict[str, Any]] = []
    for dose in DOSES[1:]:
        group = [row for row in rows if row["dose"] == dose]
        for metric in NUISANCE_METRICS:
            values = finite_values(group, metric); absolute = np.abs(values)
            nuisance.append({
                "dose": dose, "metric": metric, "n": len(values), "mean_signed": float(np.mean(values)),
                "median_signed": float(np.median(values)), "p90_absolute": float(np.quantile(absolute, 0.9)),
                "max_absolute": float(np.max(absolute)),
            })
    safety: List[Dict[str, Any]] = []
    for dose in DOSES:
        group = [row for row in rows if row["dose"] == dose]
        safety.append({
            "dose": dose, "rollout_count": len(group),
            "valid_count": sum(str(row["valid"]).lower() == "true" for row in group),
            "completion_count": sum(str(row["lane_change_completion"]).lower() == "true" for row in group),
            "collision_count": sum(str(row["collision"]).lower() == "true" for row in group),
            "responsible_collision_count": sum(int(float(row["at_fault_collision_count"])) > 0 for row in group),
            "offroad_count": sum(str(row["offroad"]).lower() == "true" for row in group),
            "drivable_area_compliant_count": sum(str(row["drivable_area_compliant"]).lower() == "true" for row in group),
            "invalid_or_incomplete_count": sum(str(row["invalid_or_incomplete"]).lower() == "true" for row in group),
        })
    roster = read_csv(args.roster_csv)
    geometry_fields = (
        "initial_speed_mps", "nominal_lane_width_m", "paired_reference_remaining_m",
        "minimum_target_lane_object_gap_m", "source_curvature_p90_inv_m",
        "target_curvature_p90_inv_m", "source_target_separation_m",
    )
    geometry = {}
    for field in geometry_fields:
        values = np.asarray([float(row[field]) for row in roster if row.get(field, "") not in ("", None)], dtype=float)
        geometry[field] = {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))} if len(values) else None
    write_csv(args.output_dir / "lateral_mechanism_long.csv", rows, list(rows[0]))
    write_csv(args.output_dir / "lateral_dose_summary.csv", dose_summary, list(dose_summary[0]))
    write_csv(args.output_dir / "scenario_level_paired_deltas.csv", paired, list(paired[0]))
    write_csv(args.output_dir / "scenario_level_paired_summary.csv", paired_summary, list(paired_summary[0]))
    write_csv(args.output_dir / "longitudinal_nuisance_summary.csv", nuisance, list(nuisance[0]))
    write_csv(args.output_dir / "safety_validity_summary.csv", safety, list(safety[0]))
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    freeze_summary = json.loads(args.freeze_summary.read_text(encoding="utf-8"))
    endpoint = {(row["metric"]): row for row in paired_summary if row["dose"] == 100}
    mechanism_pass = (
        endpoint["lane_change_duration_s"]["median_delta"] < 0
        and endpoint["rms_lateral_accel_mps2"]["median_delta"] > 0
        and endpoint["peak_yaw_rate_radps"]["median_delta"] > 0
        and endpoint["rms_lateral_jerk_mps3"]["median_delta"] > 0
    )
    safety_pass = all(row["completion_count"] == 24 and row["collision_count"] == 0 and row["offroad_count"] == 0 and row["invalid_or_incomplete_count"] == 0 for row in safety)
    final = {
        "schema_version": "stage7l_b_development_analysis_v1",
        "status": "STAGE7L_B_DEVELOPMENT_COMPLETE" if (
            run_summary["status"] == "PASS" and mechanism_pass and safety_pass
            and freeze_summary["stage7l_c_target_80_token_inventory_feasible"]
        ) else "STAGE7L_B_DEVELOPMENT_NOT_READY_FOR_FREEZE",
        "role": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "scenario_count": 24, "log_count": 24,
        "official_rollout_count": 120, "official_run_pass": run_summary["status"] == "PASS",
        "canonical_longitudinal_identity_pass": run_summary["canonical_identity_all_pass"],
        "mechanism_direction_pass": mechanism_pass, "safety_feasibility_pass": safety_pass,
        "geometry_coverage": geometry,
        "remaining_confirmation_inventory": {
            key: freeze_summary[key] for key in (
                "remaining_fresh_eligible_tokens", "remaining_fresh_eligible_logs", "remaining_left",
                "remaining_right", "stage7l_c_target_80_token_inventory_feasible"
            )
        },
        "proposed_stage7l_c_primary_mechanism_metrics": [
            "lane_change_duration_s", "rms_lateral_accel_mps2", "peak_yaw_rate_radps"
        ],
        "proposed_longitudinal_nuisance_gates": {
            "abs_delta_mean_speed_mps": 0.02,
            "abs_delta_rms_longitudinal_accel_mps2": 0.05,
            "abs_delta_rms_longitudinal_jerk_mps3": 0.10,
            "abs_delta_route_progress_m": 0.25,
            "basis": "engineering-scale margins, proposed only; not fitted to exact development maxima",
        },
        "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed,
        "embedding_or_bdd_read": False,
        "input_sha256": {
            "mechanism_metrics": sha256_file(args.mechanism_metrics_csv), "roster": sha256_file(args.roster_csv),
            "run_summary": sha256_file(args.run_summary), "freeze_summary": sha256_file(args.freeze_summary),
        },
    }
    (args.output_dir / "stage7l_b_development_analysis_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism_metrics_csv", type=Path, required=True)
    parser.add_argument("--roster_csv", type=Path, required=True)
    parser.add_argument("--run_summary", type=Path, required=True)
    parser.add_argument("--freeze_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=3407)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
