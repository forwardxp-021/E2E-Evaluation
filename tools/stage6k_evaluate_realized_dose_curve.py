#!/usr/bin/env python3
"""Evaluate nominal versus realized Stage 6K longitudinal dose before BDD read."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6j_evaluate_kinematic_gate import (  # noqa: E402
    METRICS,
    build_pair_rows,
    cluster_bootstrap_mean_ci,
    contrast_rows,
    read_csv,
)
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402


SCHEMA_VERSION = "stage6k_realized_longitudinal_dose_curve_v1"
ADDENDUM_STATUS = "FROZEN_BEFORE_NEW_DOSE_EMBEDDING_OR_BDD_READ"
DOSES = [
    ("dose25", 0.25, "pdm_closed_assertive_longitudinal_dose25_v1"),
    ("dose50", 0.50, "pdm_closed_assertive_longitudinal_dose50_v1"),
    ("dose75", 0.75, "pdm_closed_assertive_longitudinal_dose75_v1"),
]
BASELINE = "pdm_closed_conservative_longitudinal_v1"
TASK_COUNTS = {"following_interaction": 60, "longitudinal_high_motion": 56, "stop_go_control": 67}
PAIR_FIELDS = [
    "dose_label", "nominal_dose", "scenario_index", "collection_order", "scenario_token", "log_name",
    "task", "scenario_type", "mean_speed_A", "mean_speed_B", "delta_mean_speed", "rms_accel_A",
    "rms_accel_B", "delta_rms_accel", "rms_jerk_A", "rms_jerk_B", "delta_rms_jerk",
    "mean_abs_yaw_rate_A", "mean_abs_yaw_rate_B", "delta_mean_abs_yaw_rate", "mean_thw_A",
    "mean_thw_B", "delta_mean_thw", "mean_front_distance_A", "mean_front_distance_B",
    "delta_mean_front_distance", "front_valid_ratio_A", "front_valid_ratio_B", "delta_front_valid_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 6K realized kinematic dose curve without reading embedding/BDD.")
    parser.add_argument("--addendum_manifest", type=Path, required=True)
    parser.add_argument("--views_dir", type=Path, required=True)
    parser.add_argument("--contexts_dir", type=Path, required=True)
    parser.add_argument("--stage6j_kinematic_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    """Represent undefined descriptive statistics as strict-JSON null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_common(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    addendum = read_json(args.addendum_manifest.resolve())
    if addendum.get("status") != ADDENDUM_STATUS or addendum.get("new_dose_embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6K addendum is not frozen before new-dose embedding/BDD read")
    spec = addendum.get("analysis_specification", {})
    gate = spec.get("realized_kinematic_gate", {})
    if gate.get("cluster_unit") != "log_name" or int(gate.get("bootstrap_repetitions", 0)) != 10000:
        raise ValueError("Stage 6K realized-dose gate differs from the frozen addendum")
    views = read_json(args.views_dir.resolve() / "stage6k_views_summary.json")
    if views.get("status") != "STAGE6K_ALL_DOSE_VIEWS_READY" or views.get("full_embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6K dose views are not complete and BDD-blind")
    stage6j = read_json(args.stage6j_kinematic_dir.resolve() / "stage6j_kinematic_gate_summary.json")
    if stage6j.get("status") != "KINEMATIC_GATE_PASS" or int(stage6j.get("pair_count", 0)) != 183:
        raise ValueError("Stage 6J 100% endpoint kinematic evidence is not valid")
    return addendum, views


def load_new_dose(
    label: str, dose: float, planner_a: str, args: argparse.Namespace, gate: Mapping[str, Any]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    view_dir = args.views_dir.resolve() / label
    context_dir = args.contexts_dir.resolve() / label
    view = read_json(view_dir / "stage6k_dose_view_summary.json")
    warnings = read_json(context_dir / "warnings.json")
    if view.get("status") != "STAGE6K_DOSE_VIEW_READY" or view.get("scenario_count") != 183:
        raise ValueError(f"Stage 6K {label} view is incomplete")
    if warnings.get("validation", {}).get("pass") is not True:
        raise ValueError(f"Stage 6K {label} context validation failed")
    metadata = read_csv(context_dir / "metadata.csv")
    ledger = read_csv(view_dir / "stage6k_scenario_ledger.csv")
    if len(metadata) != 366 or len(ledger) != 183:
        raise ValueError(f"Stage 6K {label} expected 366 metadata and 183 ledger rows")
    if dict(Counter(row["task"] for row in ledger)) != TASK_COUNTS:
        raise ValueError(f"Stage 6K {label} task composition changed")
    config = {
        "planner_a": planner_a, "planner_b": BASELINE, "expected_pair_count": 183,
        "expected_task_counts": TASK_COUNTS,
        "bootstrap": {
            "repetitions": int(gate["bootstrap_repetitions"]), "seed": int(gate["random_seed"]),
            "confidence_level": 0.95,
        },
    }
    pair_rows = build_pair_rows(config, context_dir, metadata, ledger)
    contrasts = contrast_rows(pair_rows, config)
    for row in pair_rows:
        row.update({"dose_label": label, "nominal_dose": dose})
    for row in contrasts:
        row.update({"dose_label": label, "nominal_dose": dose})
    hashes = {
        "view_summary_sha256": sha256_file(view_dir / "stage6k_dose_view_summary.json"),
        "context_warnings_sha256": sha256_file(context_dir / "warnings.json"),
        "metadata_sha256": sha256_file(context_dir / "metadata.csv"),
    }
    return pair_rows, contrasts, hashes


def load_stage6j_endpoint(path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    pair_path = path / "stage6j_pair_kinematics.csv"
    contrast_path = path / "stage6j_kinematic_contrasts.csv"
    pairs: List[Dict[str, Any]] = []
    for source in read_csv(pair_path):
        row: Dict[str, Any] = {"dose_label": "dose100", "nominal_dose": 1.0}
        for key, value in source.items():
            if key in {"scenario_token", "log_name", "task", "scenario_type"}:
                row[key] = value
            elif key in {"scenario_index", "collection_order"}:
                row[key] = int(value)
            else:
                row[key] = float(value) if value else math.nan
        pairs.append(row)
    contrasts: List[Dict[str, Any]] = []
    for source in read_csv(contrast_path):
        row = {"dose_label": "dose100", "nominal_dose": 1.0}
        for key, value in source.items():
            if key in {"scope", "metric", "label_zh", "unit"}:
                row[key] = value
            elif key in {"pair_count", "finite_pair_count", "distinct_log_count"}:
                row[key] = int(value)
            else:
                row[key] = float(value) if value else math.nan
        contrasts.append(row)
    return pairs, contrasts, {"pair_kinematics_sha256": sha256_file(pair_path), "contrasts_sha256": sha256_file(contrast_path)}


def descriptive_zero_rows() -> List[Dict[str, Any]]:
    return [{
        "dose_label": "dose0", "nominal_dose": 0.0, "scope": "overall", "metric": metric,
        "label_zh": label, "unit": unit, "pair_count": 183, "finite_pair_count": 183,
        "distinct_log_count": 156, "mean_delta_A_minus_B": 0.0,
        "cluster_bootstrap_ci95_low": 0.0, "cluster_bootstrap_ci95_high": 0.0,
        "median_delta_A_minus_B": 0.0, "positive_fraction": 0.0,
        "analysis_role": "descriptive_origin_no_inference",
    } for metric, label, unit in METRICS]


def gate_decisions(contrasts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lookup = {(row["dose_label"], row["scope"], row["metric"]): row for row in contrasts}
    decisions: List[Dict[str, Any]] = []
    for label, dose, _ in [*DOSES, ("dose100", 1.0, "")]:
        rows = [lookup[(label, "overall", metric)] for metric in ["delta_mean_speed", "delta_rms_accel"]]
        metrics = [{
            "metric": row["metric"], "mean_delta": float(row["mean_delta_A_minus_B"]),
            "one_sided95_low": float(row["cluster_bootstrap_one_sided95_low"]),
            "two_sided95_ci_low": float(row["cluster_bootstrap_ci95_low"]),
            "ci95_high": float(row["cluster_bootstrap_ci95_high"]),
            "pass": float(row["cluster_bootstrap_one_sided95_low"]) > 0.0,
        } for row in rows]
        decisions.append({
            "dose_label": label, "nominal_dose": dose, "required_metrics": metrics,
            "kinematic_gate_passed": all(row["pass"] for row in metrics),
            "rule": "both one-sided log-cluster bootstrap 95% lower bounds > 0",
        })
    return decisions


def add_frozen_one_sided_bounds(
    pairs: Sequence[Mapping[str, Any]], contrasts: Sequence[Dict[str, Any]], gate: Mapping[str, Any]
) -> None:
    """Annotate required overall metrics with the frozen 5th-percentile cluster bound."""
    by_key = {(row["dose_label"], row["scope"], row["metric"]): row for row in contrasts}
    for dose_index, (label, _, _) in enumerate([*DOSES, ("dose100", 1.0, "")]):
        current = [row for row in pairs if row["dose_label"] == label]
        clusters = np.asarray([row["log_name"] for row in current], dtype=str)
        for metric_index, metric in enumerate(["delta_mean_speed", "delta_rms_accel"]):
            values = np.asarray([float(row[metric]) for row in current])
            low, _, _, _ = cluster_bootstrap_mean_ci(
                values, clusters, repetitions=int(gate["bootstrap_repetitions"]),
                seed=int(gate["random_seed"]) + dose_index * 100 + metric_index,
                confidence_level=0.90,
            )
            by_key[(label, "overall", metric)]["cluster_bootstrap_one_sided95_low"] = low


def trend_rows(contrasts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    overall = [row for row in contrasts if row["scope"] == "overall" and float(row["nominal_dose"]) > 0]
    for metric, label, unit in METRICS:
        current = sorted([row for row in overall if row["metric"] == metric], key=lambda row: float(row["nominal_dose"]))
        doses = np.asarray([float(row["nominal_dose"]) for row in current])
        values = np.asarray([float(row["mean_delta_A_minus_B"]) for row in current])
        finite = np.isfinite(values)
        rho, p_value = spearmanr(doses[finite], values[finite]) if finite.sum() >= 3 else (math.nan, math.nan)
        rows.append({"metric": metric, "label_zh": label, "unit": unit, "finite_dose_count": int(finite.sum()), "spearman_rho": float(rho), "raw_p_descriptive": float(p_value)})
    return rows


def plot_curve(contrasts: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    metrics = [
        ("delta_mean_speed", "Δ mean speed (m/s)"),
        ("delta_rms_accel", "Δ RMS acceleration (m/s²)"),
        ("delta_rms_jerk", "Δ RMS jerk (m/s³)"),
        ("delta_mean_thw", "Δ mean THW (s)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    overall = [row for row in contrasts if row["scope"] == "overall"]
    for axis, (metric, title) in zip(axes.flat, metrics):
        current = sorted([row for row in overall if row["metric"] == metric], key=lambda row: float(row["nominal_dose"]))
        x = np.asarray([float(row["nominal_dose"]) * 100 for row in current])
        y = np.asarray([float(row["mean_delta_A_minus_B"]) for row in current])
        low = np.asarray([float(row["cluster_bootstrap_ci95_low"]) for row in current])
        high = np.asarray([float(row["cluster_bootstrap_ci95_high"]) for row in current])
        axis.errorbar(x, y, yerr=np.vstack([y - low, high - y]), marker="o", capsize=4)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=title, xlabel="Nominal longitudinal-parameter dose (%)", ylabel="A − B")
        axis.grid(alpha=0.25)
    fig.suptitle("Stage 6K: nominal parameter dose versus realized longitudinal behavior")
    fig.savefig(output_dir / "stage6k_realized_dose_curve.png", dpi=180)
    fig.savefig(output_dir / "stage6k_realized_dose_curve.pdf")
    plt.close(fig)


def build_report(decisions: Sequence[Mapping[str, Any]], trends: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Stage 6K 名义剂量—实现行为剂量报告", "", "## 结论", "",
        "25%、50%、75%表示六个纵向IDM参数的线性插值比例，不假定运动学响应线性。",
        "本报告只读取rollout/context运动学，不读取新增剂量embedding或BDD。", "", "## 实现运动学门禁", "",
        "| 名义剂量 | Δ平均速度 | 速度CI下界 | ΔRMS加速度 | 加速度CI下界 | 门禁 |", "|---:|---:|---:|---:|---:|---|",
    ]
    for item in decisions:
        speed, accel = item["required_metrics"]
        lines.append(f"| {100 * item['nominal_dose']:.0f}% | {speed['mean_delta']:.4f} | {speed['one_sided95_low']:.4f} | {accel['mean_delta']:.4f} | {accel['one_sided95_low']:.4f} | {'PASS' if item['kinematic_gate_passed'] else 'FAIL'} |")
    lines.extend(["", "## 有序趋势（描述性）", "", "| 指标 | Spearman rho | raw p |", "|---|---:|---:|"])
    for item in trends:
        lines.append(f"| {item['label_zh']} | {item['spearman_rho']:.4f} | {item['raw_p_descriptive']:.6g} |")
    lines.extend(["", "运动学门禁不等于embedding检出；最小可检出剂量仍须按addendum使用四剂量overall BDD Holm p<0.05。", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    addendum, views = validate_common(args)
    gate = addendum["analysis_specification"]["realized_kinematic_gate"]
    all_pairs: List[Dict[str, Any]] = []
    all_contrasts: List[Dict[str, Any]] = descriptive_zero_rows()
    input_hashes: Dict[str, Any] = {
        "addendum_manifest_sha256": sha256_file(args.addendum_manifest.resolve()),
        "views_summary_sha256": sha256_file(args.views_dir.resolve() / "stage6k_views_summary.json"),
    }
    canonical_tokens: List[str] | None = None
    for label, dose, planner in DOSES:
        pairs, contrasts, hashes = load_new_dose(label, dose, planner, args, gate)
        tokens = [str(row["scenario_token"]) for row in pairs]
        if canonical_tokens is None:
            canonical_tokens = tokens
        elif tokens != canonical_tokens:
            raise ValueError(f"Stage 6K {label} pair order differs")
        all_pairs.extend(pairs)
        all_contrasts.extend(contrasts)
        input_hashes[label] = hashes
    pairs100, contrasts100, hashes100 = load_stage6j_endpoint(args.stage6j_kinematic_dir.resolve())
    if [str(row["scenario_token"]) for row in pairs100] != canonical_tokens:
        raise ValueError("Stage 6J 100% endpoint token order differs from Stage 6K")
    all_pairs.extend(pairs100)
    all_contrasts.extend(contrasts100)
    input_hashes["dose100"] = hashes100
    add_frozen_one_sided_bounds(all_pairs, all_contrasts, gate)
    decisions = gate_decisions(all_contrasts)
    trends = trend_rows(all_contrasts)
    contrast_fields = ["dose_label", "nominal_dose", "analysis_role", "scope", "metric", "label_zh", "unit", "pair_count", "finite_pair_count", "distinct_log_count", "mean_delta_A_minus_B", "cluster_bootstrap_one_sided95_low", "cluster_bootstrap_ci95_low", "cluster_bootstrap_ci95_high", "median_delta_A_minus_B", "positive_fraction"]
    write_csv(output_dir / "stage6k_pair_kinematics.csv", all_pairs, PAIR_FIELDS)
    write_csv(output_dir / "stage6k_kinematic_contrasts.csv", all_contrasts, contrast_fields)
    write_csv(output_dir / "stage6k_kinematic_gate_decisions.csv", [{"dose_label": row["dose_label"], "nominal_dose": row["nominal_dose"], "kinematic_gate_passed": row["kinematic_gate_passed"], "rule": row["rule"]} for row in decisions], ["dose_label", "nominal_dose", "kinematic_gate_passed", "rule"])
    write_csv(output_dir / "stage6k_kinematic_dose_trends.csv", trends, ["metric", "label_zh", "unit", "finite_dose_count", "spearman_rho", "raw_p_descriptive"])
    plot_curve(all_contrasts, output_dir)
    report_path = output_dir / "stage6k_realized_dose_report_zh.md"
    report_path.write_text(build_report(decisions, trends), encoding="utf-8")
    result = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE6K_REALIZED_DOSE_CURVE_COMPLETE", "new_dose_embedding_or_bdd_read": False,
        "pair_count_per_nonzero_dose": 183, "distinct_log_count": 156,
        "nominal_dose_is_not_assumed_linear_in_realized_behavior": True,
        "gate_decisions": decisions, "ordered_trends_descriptive": trends,
        "input_hashes": input_hashes, "tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "pair_kinematics": "stage6k_pair_kinematics.csv", "contrasts": "stage6k_kinematic_contrasts.csv",
            "gate_decisions": "stage6k_kinematic_gate_decisions.csv", "trends": "stage6k_kinematic_dose_trends.csv",
            "figure_png": "stage6k_realized_dose_curve.png", "figure_pdf": "stage6k_realized_dose_curve.pdf",
            "report_zh": report_path.name,
        },
    }
    write_json(output_dir / "stage6k_realized_dose_summary.json", result)
    return result


def main() -> None:
    print(json.dumps(json_safe(run(parse_args())), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
