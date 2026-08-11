#!/usr/bin/env python3
"""Descriptive post-treatment lane-quality sensitivity for Stage 6K."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6k_run_longitudinal_dose_bdd import DOSES, build_pairs, read_json
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file


SCHEMA_VERSION = "stage6k_lane_quality_sensitivity_v1"
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260812
QUALITY_METRICS = ["max_pair_fallback_rate", "max_pair_ambiguity_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 6K descriptive lane-quality sensitivity.")
    parser.add_argument("--addendum_manifest", type=Path, required=True)
    parser.add_argument("--new_contexts_dir", type=Path, required=True)
    parser.add_argument("--new_embeddings_dir", type=Path, required=True)
    parser.add_argument("--stage6j_context_dir", type=Path, required=True)
    parser.add_argument("--stage6j_embedding_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or len(np.unique(x[finite])) < 2 or len(np.unique(y[finite])) < 2:
        return 0.0
    return float(spearmanr(x[finite], y[finite]).statistic)


def task_adjusted_rank_correlation(x: np.ndarray, y: np.ndarray, tasks: Sequence[str]) -> float:
    x_rank = rankdata(np.asarray(x, dtype=np.float64), method="average")
    y_rank = rankdata(np.asarray(y, dtype=np.float64), method="average")
    task_values = np.asarray(tasks, dtype=str)
    for task in np.unique(task_values):
        mask = task_values == task
        x_rank[mask] -= float(np.mean(x_rank[mask]))
        y_rank[mask] -= float(np.mean(y_rank[mask]))
    if np.std(x_rank) <= 0 or np.std(y_rank) <= 0:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def log_cluster_bootstrap_interval(
    x: np.ndarray,
    y: np.ndarray,
    tasks: Sequence[str],
    logs: Sequence[str],
    *,
    adjusted: bool,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    log_values = np.asarray(logs, dtype=str)
    unique_logs = np.unique(log_values)
    by_log = {log_name: np.flatnonzero(log_values == log_name) for log_name in unique_logs}
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    task_values = np.asarray(tasks, dtype=str)
    for position in range(repetitions):
        selected_logs = rng.choice(unique_logs, size=len(unique_logs), replace=True)
        indices = np.concatenate([by_log[log_name] for log_name in selected_logs])
        samples[position] = (
            task_adjusted_rank_correlation(x[indices], y[indices], task_values[indices])
            if adjusted
            else finite_spearman(x[indices], y[indices])
        )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_pair_quality_rows(
    label: str,
    dose: float,
    planner_a: str,
    context_dir: Path,
    embedding_dir: Path,
) -> List[Dict[str, Any]]:
    metadata = pd.read_csv(embedding_dir / "metadata.csv")
    quality = pd.read_csv(context_dir / "nuplan_lane_assignment_by_row.csv")
    embedding = np.asarray(np.load(embedding_dir / "embedding.npy", mmap_mode="r"), dtype=np.float64)
    pair_indices, tokens, logs = build_pairs(metadata, planner_a)
    if len(quality) != 366 or set(quality["global_row"].astype(int)) != set(range(366)):
        raise ValueError(f"{label} lane-quality rows are not exhaustive")
    quality_by_row = quality.set_index("global_row", drop=False)
    metadata_by_row = metadata.set_index("global_row", drop=False)
    rows: List[Dict[str, Any]] = []
    for position, ((row_a, row_b), token, log_name) in enumerate(zip(pair_indices, tokens, logs)):
        qa, qb = quality_by_row.loc[int(row_a)], quality_by_row.loc[int(row_b)]
        ma, mb = metadata_by_row.loc[int(row_a)], metadata_by_row.loc[int(row_b)]
        if str(ma["scenario_type"]) != str(mb["scenario_type"]):
            raise ValueError(f"{label} pair {position} scenario_type differs")
        rows.append({
            "dose_label": label,
            "nominal_dose": dose,
            "pair_position": position,
            "scenario_token": token,
            "log_name": log_name,
            "scenario_type": str(ma["scenario_type"]),
            "task": str(ma["scenario_type"]),
            "row_A": int(row_a),
            "row_B": int(row_b),
            "fallback_rate_A": float(qa["fallback_rate"]),
            "fallback_rate_B": float(qb["fallback_rate"]),
            "max_pair_fallback_rate": max(float(qa["fallback_rate"]), float(qb["fallback_rate"])),
            "ambiguity_rate_A": float(qa["ambiguous_frame_rate"]),
            "ambiguity_rate_B": float(qb["ambiguous_frame_rate"]),
            "max_pair_ambiguity_rate": max(float(qa["ambiguous_frame_rate"]), float(qb["ambiguous_frame_rate"])),
            "embedding_pair_l2_distance": float(np.linalg.norm(embedding[int(row_a)] - embedding[int(row_b)])),
        })
    return rows


def assign_frozen_tasks(rows: Sequence[Dict[str, Any]], definitions: Mapping[str, Sequence[str]]) -> None:
    type_to_task = {scenario_type: task for task, values in definitions.items() for scenario_type in values}
    for row in rows:
        scenario_type = str(row["scenario_type"])
        if scenario_type not in type_to_task:
            raise ValueError(f"Unmapped scenario_type: {scenario_type}")
        row["task"] = type_to_task[scenario_type]


def association_rows(pair_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for dose_index, (label, dose, _) in enumerate(DOSES):
        current = [row for row in pair_rows if row["dose_label"] == label]
        outcome = np.asarray([float(row["embedding_pair_l2_distance"]) for row in current])
        tasks = [str(row["task"]) for row in current]
        logs = [str(row["log_name"]) for row in current]
        for metric_index, metric in enumerate(QUALITY_METRICS):
            quality = np.asarray([float(row[metric]) for row in current])
            for adjusted in [False, True]:
                estimate = (
                    task_adjusted_rank_correlation(quality, outcome, tasks)
                    if adjusted
                    else finite_spearman(quality, outcome)
                )
                low, high = log_cluster_bootstrap_interval(
                    quality, outcome, tasks, logs, adjusted=adjusted,
                    repetitions=BOOTSTRAP_REPETITIONS,
                    seed=BOOTSTRAP_SEED + dose_index * 100 + metric_index * 10 + int(adjusted),
                )
                output.append({
                    "dose_label": label,
                    "nominal_dose": dose,
                    "quality_metric": metric,
                    "analysis": "task_adjusted_rank_residual" if adjusted else "overall_spearman_rank",
                    "n_pairs": len(current),
                    "n_logs": len(set(logs)),
                    "estimate": estimate,
                    "log_cluster_bootstrap_ci95_low": low,
                    "log_cluster_bootstrap_ci95_high": high,
                    "interval_excludes_zero": bool(low > 0 or high < 0),
                    "role": "post_treatment_descriptive_not_causal_adjustment",
                })
    return output


def build_report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Stage 6K lane-quality与embedding距离敏感性（中文）", "", "## 结论", "",
        "本分析仅描述fallback/ambiguity与pair embedding L2距离的关联。它们是rollout后变量，不用于删样本、重加权或替代primary BDD。", "",
        "| 剂量 | 质量指标 | 分析 | 相关系数 | log-cluster 95% CI | 是否跨0 |", "|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {100*float(row['nominal_dose']):.0f}% | {row['quality_metric']} | {row['analysis']} | {float(row['estimate']):.4f} | [{float(row['log_cluster_bootstrap_ci95_low']):.4f}, {float(row['log_cluster_bootstrap_ci95_high']):.4f}] | {'否' if row['interval_excludes_zero'] else '是'} |")
    lines += ["", "任何显著关联都只能提示测量质量可能参与embedding距离变化，不能解释为因果混杂，也不能据此重做确认性样本选择。", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    addendum = read_json(args.addendum_manifest.resolve())
    quality_spec = addendum.get("analysis_specification", {}).get("lane_quality_sensitivity", {})
    if quality_spec.get("role") != "descriptive_exploratory_post_treatment_not_causal_adjustment":
        raise ValueError("Stage 6K lane-quality sensitivity is not frozen")
    task_definitions = read_json(Path("configs/stage6j_paired_bdd_analysis.json"))["task_conditioned_secondary"]["tasks"]
    pair_rows: List[Dict[str, Any]] = []
    input_hashes: Dict[str, Any] = {}
    for label, dose, planner_a in DOSES:
        context_dir = args.stage6j_context_dir.resolve() if label == "dose100" else args.new_contexts_dir.resolve() / label
        embedding_dir = args.stage6j_embedding_dir.resolve() if label == "dose100" else args.new_embeddings_dir.resolve() / label
        current = build_pair_quality_rows(label, dose, planner_a, context_dir, embedding_dir)
        assign_frozen_tasks(current, task_definitions)
        pair_rows.extend(current)
        input_hashes[label] = {
            "lane_quality_sha256": sha256_file(context_dir / "nuplan_lane_assignment_by_row.csv"),
            "embedding_sha256": sha256_file(embedding_dir / "embedding.npy"),
            "metadata_sha256": sha256_file(embedding_dir / "metadata.csv"),
        }
    associations = association_rows(pair_rows)
    write_csv(output_dir / "stage6k_pair_lane_quality.csv", pair_rows, list(pair_rows[0]))
    write_csv(output_dir / "stage6k_lane_quality_associations.csv", associations, list(associations[0]))
    report_path = output_dir / "stage6k_lane_quality_report_zh.md"
    report_path.write_text(build_report(associations), encoding="utf-8")
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE6K_LANE_QUALITY_SENSITIVITY_COMPLETE",
        "role": quality_spec["role"],
        "pair_count": len(pair_rows),
        "association_count": len(associations),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "no_filtering_or_reweighting": True,
        "associations": associations,
        "input_hashes": input_hashes,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "pair_quality": "stage6k_pair_lane_quality.csv",
            "associations": "stage6k_lane_quality_associations.csv",
            "report_zh": report_path.name,
        },
    }
    write_json(output_dir / "stage6k_lane_quality_summary.json", result)
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
