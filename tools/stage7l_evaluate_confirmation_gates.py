#!/usr/bin/env python3
"""Evaluate the frozen Stage7L-D planner-level confirmation gates.

Inputs are limited to the frozen protocol/roster plus official execution,
trajectory mechanism, nuisance, safety, and canonical-identity outputs.  The
module intentionally has no representation, checkpoint, BDD, or MMD path.
"""

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


DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")
DOSE_NUMBER = {"dose0": 0, "dose25": 25, "dose50": 50, "dose75": 75, "dose100": 100}
MECHANISM_METRICS = (
    "lane_change_duration_s", "rms_lateral_accel_mps2", "peak_yaw_rate_radps",
    "peak_lateral_accel_mps2", "rms_yaw_rate_radps", "rms_lateral_jerk_mps3",
    "target_center_settling_time_s", "final_target_lane_center_offset_m",
)
NUISANCE_METRICS = (
    "mean_speed_mps", "rms_longitudinal_accel_mps2", "rms_longitudinal_jerk_mps3", "route_progress_m",
)
PRIMARY_DIRECTION = {
    "lane_change_duration_s": "negative",
    "rms_lateral_accel_mps2": "positive",
    "peak_yaw_rate_radps": "positive",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def optional_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("nan")


def cluster_bootstrap_median_ci(
    rows: List[Dict[str, Any]], rng: np.random.Generator, replicates: int
) -> tuple[float, float]:
    by_log: Dict[str, np.ndarray] = {}
    for log_name in sorted({str(row["log_name"]) for row in rows}):
        by_log[log_name] = np.asarray(
            [float(row["paired_delta"]) for row in rows if row["log_name"] == log_name], dtype=np.float64
        )
    logs = sorted(by_log)
    if not logs:
        return float("nan"), float("nan")
    medians = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(logs), size=len(logs))
        sample = np.concatenate([by_log[logs[item]] for item in selected])
        medians[index] = np.median(sample)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def complete(summary: Mapping[str, Any]) -> bool:
    return summary.get("official_run_status") == "SUCCEEDED" and as_bool(summary.get("trajectory_available"))


def build_completeness(
    roster: List[Dict[str, str]], summary: List[Dict[str, str]]
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_token_dose = {(row["scenario_token"], row["dose"]): row for row in summary}
    rows: List[Dict[str, Any]] = []
    for roster_row in sorted(roster, key=lambda item: int(item["collection_order"])):
        token = roster_row["scenario_token"]
        flags = {dose: complete(by_token_dose.get((token, dose), {})) for dose in DOSES}
        rows.append({
            "collection_order": int(roster_row["collection_order"]), "scenario_token": token,
            "log_name": roster_row["log_name"], "direction": roster_row["direction"],
            **{f"{dose}_complete": flags[dose] for dose in DOSES},
            "all_five_doses_complete": all(flags.values()),
        })
    dose_counts = {dose: sum(row[f"{dose}_complete"] for row in rows) for dose in DOSES}
    all_five = sum(row["all_five_doses_complete"] for row in rows)
    execution = {
        "N_design": 80, "planned_rollout_cells": 400,
        "successful_official_rollout_cells": sum(dose_counts.values()),
        "failed_rollout_cells": 400 - sum(dose_counts.values()),
        "dose_success_count": dose_counts,
        "N_complete_all_five_doses": all_five,
        "minimum_completed_scenarios": 76,
        "execution_gate_pass": all_five >= 76,
    }
    return rows, execution


def build_pairwise(metrics: List[Dict[str, str]], roster: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    log_by_token = {row["scenario_token"]: row["log_name"] for row in roster}
    direction_by_token = {row["scenario_token"]: row["direction"] for row in roster}
    by_token_dose = {(row["scenario_token"], row["dose"]): row for row in metrics}
    pairs: List[Dict[str, Any]] = []
    for token in sorted({row["scenario_token"] for row in metrics}):
        reference = by_token_dose.get((token, "dose0"), {})
        for dose in DOSES[1:]:
            target = by_token_dose.get((token, dose), {})
            for metric in (*MECHANISM_METRICS, *NUISANCE_METRICS):
                ref_value = finite(reference.get(metric))
                target_value = finite(target.get(metric))
                if ref_value is None or target_value is None:
                    continue
                delta = target_value - ref_value
                expected = PRIMARY_DIRECTION.get(metric, "descriptive")
                consistent = delta < 0 if expected == "negative" else (delta > 0 if expected == "positive" else None)
                pairs.append({
                    "scenario_token": token, "log_name": log_by_token[token], "direction": direction_by_token[token],
                    "dose": dose, "dose_number": DOSE_NUMBER[dose], "reference_dose": "dose0", "metric": metric,
                    "reference_value": ref_value, "target_value": target_value, "paired_delta": delta,
                    "absolute_paired_delta": abs(delta), "expected_direction": expected,
                    "directionally_consistent": consistent,
                })
    return pairs


def mechanism_summary(
    pairs: List[Dict[str, Any]], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(protocol["semantic_uncertainty_reporting"]["seed"]))
    replicates = int(protocol["semantic_uncertainty_reporting"]["replicates"])
    dose_response: List[Dict[str, Any]] = []
    for dose in DOSES[1:]:
        for metric in MECHANISM_METRICS:
            group = [row for row in pairs if row["dose"] == dose and row["metric"] == metric]
            values = np.asarray([row["paired_delta"] for row in group], dtype=np.float64)
            lo, hi = cluster_bootstrap_median_ci(group, rng, replicates)
            expected = PRIMARY_DIRECTION.get(metric)
            consistency = None
            if expected:
                consistency = float(np.mean([bool(row["directionally_consistent"]) for row in group])) if group else float("nan")
            dose_response.append({
                "dose": dose, "metric": metric, "n_pair": len(group),
                "paired_median_delta": float(np.median(values)) if len(values) else float("nan"),
                "directional_consistency": consistency,
                "log_cluster_bootstrap_median_ci_95": [lo, hi],
            })
    primary = {row["metric"]: row for row in dose_response if row["dose"] == "dose100" and row["metric"] in PRIMARY_DIRECTION}
    criteria = protocol["mechanism"]["pass_criteria"]
    gate_items: Dict[str, Any] = {}
    for metric, direction in PRIMARY_DIRECTION.items():
        row = primary[metric]
        threshold = float(criteria[metric]["directional_consistency_at_least"])
        median_direction_pass = row["paired_median_delta"] < 0 if direction == "negative" else row["paired_median_delta"] > 0
        consistency_pass = row["directional_consistency"] >= threshold
        gate_items[metric] = {
            **row, "expected_direction": direction, "consistency_threshold": threshold,
            "median_direction_pass": median_direction_pass, "consistency_pass": consistency_pass,
            "pass": median_direction_pass and consistency_pass,
        }
    gate_pass = all(row["pass"] for row in gate_items.values())
    return {
        "schema_version": "stage7l_d_mechanism_summary_v1", "primary_contrast": "dose100_minus_dose0",
        "primary": gate_items, "secondary_dose_response": dose_response,
        "bootstrap": {"cluster": "log_name", "replicates": replicates, "seed": int(protocol["semantic_uncertainty_reporting"]["seed"]), "interval": "percentile_95", "gate_use": False},
        "mechanism_gate_pass": gate_pass,
    }


def nuisance_summary(pairs: List[Dict[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    rng = np.random.default_rng(int(protocol["semantic_uncertainty_reporting"]["seed"]))
    replicates = int(protocol["semantic_uncertainty_reporting"]["replicates"])
    median_thresholds = protocol["nuisance_gate"]["absolute_paired_median_thresholds"]
    p90_thresholds = protocol["nuisance_gate"]["absolute_p90_thresholds"]
    items: Dict[str, Any] = {}
    for metric in NUISANCE_METRICS:
        group = [row for row in pairs if row["dose"] == "dose100" and row["metric"] == metric]
        signed = np.asarray([row["paired_delta"] for row in group], dtype=np.float64)
        absolute = np.abs(signed)
        lo, hi = cluster_bootstrap_median_ci(group, rng, replicates)
        median_abs = float(np.median(absolute)) if len(absolute) else float("nan")
        p90_abs = quantile(absolute, 0.9)
        item = {
            "n_pair": len(group), "paired_median_signed_delta": float(np.median(signed)) if len(signed) else float("nan"),
            "paired_median_absolute_delta": median_abs, "p90_absolute_delta": p90_abs,
            "max_absolute_delta_diagnostic": float(np.max(absolute)) if len(absolute) else float("nan"),
            "log_cluster_bootstrap_signed_median_ci_95": [lo, hi],
            "median_threshold": float(median_thresholds[metric]), "p90_threshold": float(p90_thresholds[metric]),
        }
        item["median_pass"] = median_abs <= item["median_threshold"]
        item["p90_pass"] = p90_abs <= item["p90_threshold"]
        item["pass"] = item["median_pass"] and item["p90_pass"]
        items[metric] = item
    return {
        "schema_version": "stage7l_d_longitudinal_nuisance_summary_v1",
        "contrast": "dose100_minus_dose0", "metrics": items,
        "nuisance_gate_pass": all(item["pass"] for item in items.values()),
    }


def safety_summary(
    roster: List[Dict[str, str]], summary: List[Dict[str, str]], metrics: List[Dict[str, str]],
    protocol: Mapping[str, Any], execution_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    by_summary = {(row["scenario_token"], row["dose"]): row for row in summary}
    by_metric = {(row["scenario_token"], row["dose"]): row for row in metrics}
    scenario_rows: List[Dict[str, Any]] = []
    for roster_row in roster:
        token = roster_row["scenario_token"]
        run_rows = [by_summary.get((token, dose), {}) for dose in DOSES]
        metric_rows = [by_metric.get((token, dose), {}) for dose in DOSES]
        official_success = all(complete(row) for row in run_rows)
        completion = official_success and all(as_bool(row.get("lane_change_completion")) for row in metric_rows)
        offroad = any(optional_bool(row.get("offroad")) is True for row in metric_rows)
        responsible_collision = any(optional_bool(row.get("responsible_collision")) is True for row in metric_rows)
        route_failure = any(optional_bool(row.get("route_failure")) is True for row in metric_rows)
        invalid = (not official_success) or any(not as_bool(row.get("valid")) for row in metric_rows)
        scenario_rows.append({
            "scenario_token": token, "log_name": roster_row["log_name"], "direction": roster_row["direction"],
            "official_success": official_success, "lane_change_completion": completion, "offroad": offroad,
            "responsible_collision": responsible_collision, "route_failure": route_failure, "invalid": invalid,
            "incomplete": not completion,
        })
    n = len(roster)
    rates = {
        "official_success_rate": sum(row["official_success"] for row in scenario_rows) / n,
        "lane_change_completion_rate": sum(row["lane_change_completion"] for row in scenario_rows) / n,
        "offroad_rate": sum(row["offroad"] for row in scenario_rows) / n,
        "responsible_collision_rate": sum(row["responsible_collision"] for row in scenario_rows) / n,
    }
    criteria = protocol["safety_validity_gate"]
    checks = {
        "official_success": rates["official_success_rate"] >= float(criteria["official_success_rate_at_least"]),
        "lane_change_completion": rates["lane_change_completion_rate"] >= float(criteria["lane_change_completion_rate_at_least"]),
        "offroad": rates["offroad_rate"] <= float(criteria["offroad_rate_at_most"]),
        "responsible_collision": rates["responsible_collision_rate"] <= float(criteria["responsible_collision_rate_at_most"]),
    }
    return {
        "schema_version": "stage7l_d_safety_validity_summary_v1",
        "denominator": n, "aggregation": execution_contract["safety_aggregation"],
        "counts": {
            "official_success": sum(row["official_success"] for row in scenario_rows),
            "lane_change_completion": sum(row["lane_change_completion"] for row in scenario_rows),
            "offroad": sum(row["offroad"] for row in scenario_rows),
            "responsible_collision": sum(row["responsible_collision"] for row in scenario_rows),
            "any_collision": "N/A_OFFICIAL_BUNDLE_ONLY_EXPOSES_AT_FAULT_COLLISION",
            "invalid": sum(row["invalid"] for row in scenario_rows),
            "incomplete": sum(row["incomplete"] for row in scenario_rows),
            "route_failure": sum(row["route_failure"] for row in scenario_rows),
        },
        "rates": rates, "checks": checks, "safety_gate_pass": all(checks.values()),
        "scenario_outcomes": scenario_rows,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    protocol = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    roster = read_csv(args.roster_csv)
    planned = read_csv(args.planned_ledger)
    attempts = read_csv(args.attempt_ledger)
    summary = read_csv(args.official_run_summary)
    metrics = read_csv(args.mechanism_metrics)
    identity = json.loads(args.canonical_identity_audit.read_text(encoding="utf-8"))
    contract = json.loads(args.execution_contract.read_text(encoding="utf-8"))
    if len(roster) != 80 or len(planned) != 400 or len(summary) != 400 or len(metrics) != 400:
        raise ValueError("gate inputs must preserve the exact 80-scenario / 400-cell design")
    completeness_rows, execution = build_completeness(roster, summary)
    write_csv(args.output_dir / "scenario_five_dose_completeness.csv", completeness_rows, list(completeness_rows[0]))
    pairs = build_pairwise(metrics, roster)
    write_csv(args.output_dir / "mechanism_pairwise_deltas.csv", pairs, list(pairs[0]) if pairs else ["scenario_token"])
    mechanism = mechanism_summary(pairs, protocol)
    nuisance = nuisance_summary(pairs, protocol)
    safety = safety_summary(roster, summary, metrics, protocol, contract)
    identity_pass = (
        identity.get("mismatch_count") == 0
        and identity.get("canonical_identity_pass_count") == execution["N_complete_all_five_doses"]
    )
    final_pass = bool(
        execution["execution_gate_pass"] and identity_pass and mechanism["mechanism_gate_pass"]
        and nuisance["nuisance_gate_pass"] and safety["safety_gate_pass"]
    )
    failed_gate = next((name for name, value in (
        ("execution", execution["execution_gate_pass"]), ("canonical_identity", identity_pass),
        ("mechanism", mechanism["mechanism_gate_pass"]), ("longitudinal_nuisance", nuisance["nuisance_gate_pass"]),
        ("safety_validity", safety["safety_gate_pass"]),
    ) if not value), None)
    infrastructure_failed_cells = sum(
        row.get("official_run_status") != "SUCCEEDED" and row.get("failure_category") == "INFRASTRUCTURE_RUNTIME"
        for row in summary
    )
    treatment_failed_cells = sum(
        complete(row) and any(
            optional_bool(by.get(key)) is True for key in ("offroad", "responsible_collision", "route_failure")
        )
        for row in summary
        for by in [next((item for item in metrics if item["cell_id"] == row["cell_id"]), {})]
    )
    missingness = {
        "schema_version": "stage7l_d_missingness_summary_v1", **execution,
        "attempt_count": len(attempts), "infrastructure_failed_cells": infrastructure_failed_cells,
        "treatment_outcome_failed_cells": treatment_failed_cells,
        "failed_cell_details": [row for row in summary if row.get("official_run_status") != "SUCCEEDED"],
        "replacement_count": 0,
    }
    decision = {
        "schema_version": "stage7l_d_gate_decision_v1",
        "status": "STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_PASSED" if final_pass else "STAGE7L_D_PLANNER_LEVEL_CONFIRMATION_FAILED",
        "failed_gate": failed_gate,
        "execution_gate_pass": execution["execution_gate_pass"],
        "canonical_identity_pass": identity_pass,
        "mechanism_gate_pass": mechanism["mechanism_gate_pass"],
        "nuisance_gate_pass": nuisance["nuisance_gate_pass"],
        "safety_gate_pass": safety["safety_gate_pass"],
        "representation_unlock": final_pass,
        "representation_status": "STAGE7L_E_REPRESENTATION_EVALUATION_UNLOCKED" if final_pass else "STAGE7L_E_REPRESENTATION_EVALUATION_NOT_UNLOCKED",
        "gate_statuses": {
            "execution": "STAGE7L_D_CONFIRMATION_EXECUTION_SUFFICIENT" if execution["execution_gate_pass"] else "STAGE7L_D_CONFIRMATION_EXECUTION_INSUFFICIENT",
            "mechanism": "STAGE7L_D_MECHANISM_GATE_PASSED" if mechanism["mechanism_gate_pass"] else "STAGE7L_D_MECHANISM_GATE_FAILED",
            "nuisance": "STAGE7L_D_LONGITUDINAL_NUISANCE_GATE_PASSED" if nuisance["nuisance_gate_pass"] else "STAGE7L_D_LONGITUDINAL_NUISANCE_GATE_FAILED",
            "safety": "STAGE7L_D_SAFETY_VALIDITY_GATE_PASSED" if safety["safety_gate_pass"] else "STAGE7L_D_SAFETY_VALIDITY_GATE_FAILED",
        },
        "embedding_read": False, "checkpoint_read": False, "bdd_computed": False, "mmd_computed": False,
        "stage7l_e_executed": False,
    }
    for path, value in (
        (args.output_dir / "mechanism_summary.json", mechanism),
        (args.output_dir / "longitudinal_nuisance_summary.json", nuisance),
        (args.output_dir / "safety_validity_summary.json", safety),
        (args.output_dir / "missingness_summary.json", missingness),
        (args.output_dir / "stage7l_d_gate_decision.json", decision),
    ):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/stage7l_d_one_time_confirmation_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol_config", type=Path, default=root / "configs/stage7l_c_prospective_confirmation_protocol_v1.json")
    parser.add_argument("--roster_csv", type=Path, default=root / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_roster.csv")
    parser.add_argument("--planned_ledger", type=Path, default=output / "planned_rollout_ledger.csv")
    parser.add_argument("--attempt_ledger", type=Path, default=output / "attempt_ledger.csv")
    parser.add_argument("--official_run_summary", type=Path, default=output / "official_run_summary.csv")
    parser.add_argument("--mechanism_metrics", type=Path, default=output / "mechanism_metrics_long.csv")
    parser.add_argument("--canonical_identity_audit", type=Path, default=output / "canonical_identity_audit.json")
    parser.add_argument("--execution_contract", type=Path, default=output / "stage7l_d_execution_contract.json")
    parser.add_argument("--output_dir", type=Path, default=output)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
