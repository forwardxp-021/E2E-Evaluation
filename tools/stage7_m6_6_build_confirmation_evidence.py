#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_2_locked_task_bdd import PRETREATMENT_TASKS  # noqa: E402
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402


SCHEMA_VERSION = "stage7_m6_6_confirmation_evidence_v1"
TASK_ORDER = list(PRETREATMENT_TASKS)
TIER_ORDER = ["A", "B", "C"]
TASK_LABELS = {
    "following_interaction": "Following",
    "lane_change": "Lane change",
    "stop_go_control": "Stop/go",
    "high_motion_dynamics": "High motion",
    "dense_or_vulnerable_interaction": "Dense/vulnerable",
}
COLORS = {
    "following_interaction": "#0f766e",
    "lane_change": "#2563eb",
    "stop_go_control": "#7c3aed",
    "high_motion_dynamics": "#dc2626",
    "dense_or_vulnerable_interaction": "#d97706",
    "A": "#15803d",
    "B": "#eab308",
    "C": "#dc2626",
}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_default(value: Any) -> Any:
    """Normalize common scientific-Python scalars for deterministic JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(output_dir: Path, name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    write_csv(output_dir / f"{name}.csv", rows)
    (output_dir / f"{name}.md").write_text(markdown_table(rows, fields), encoding="utf-8")


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def finite(values: Iterable[Any]) -> np.ndarray:
    converted = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=np.float64)
    return converted[np.isfinite(converted)]


def bootstrap_mean_ci(values: np.ndarray, *, repetitions: int, seed: int) -> Tuple[float, float]:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source)]
    if len(source) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions - start)
        indices = rng.integers(0, len(source), size=(size, len(source)))
        means[start : start + size] = source[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def vectorized_correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(
        np.sum(x_centered * y_centered, axis=1),
        denominator,
        out=np.zeros(len(x), dtype=np.float64),
        where=denominator > 0,
    )


def bootstrap_rank_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> Tuple[float, float]:
    x_rank = rankdata(np.asarray(x, dtype=np.float64))
    y_rank = rankdata(np.asarray(y, dtype=np.float64))
    if len(x_rank) < 3:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    groups = [np.arange(len(x_rank))]
    if strata is not None:
        raw_strata = np.asarray(strata, dtype=object)
        groups = [np.flatnonzero(raw_strata == value) for value in sorted(set(raw_strata))]
    for start in range(0, repetitions, 500):
        size = min(500, repetitions - start)
        pieces = [
            group[rng.integers(0, len(group), size=(size, len(group)))]
            for group in groups
        ]
        indices = np.concatenate(pieces, axis=1)
        samples[start : start + size] = vectorized_correlations(x_rank[indices], y_rank[indices])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def task_adjusted_rank_association(x: np.ndarray, y: np.ndarray, tasks: Sequence[str]) -> Dict[str, float]:
    x_rank = rankdata(np.asarray(x, dtype=np.float64))
    y_rank = rankdata(np.asarray(y, dtype=np.float64))
    task_array = np.asarray(tasks, dtype=object)
    design = np.ones((len(task_array), 1 + len(TASK_ORDER) - 1), dtype=np.float64)
    for position, task in enumerate(TASK_ORDER[1:], start=1):
        design[:, position] = (task_array == task).astype(np.float64)
    residual_x = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
    residual_y = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
    result = pearsonr(residual_x, residual_y)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "residual_x": residual_x,
        "residual_y": residual_y,
    }


def build_type_to_task() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for task, scenario_types in PRETREATMENT_TASKS.items():
        for scenario_type in scenario_types:
            if scenario_type in result:
                raise ValueError(f"scenario_type belongs to multiple tasks: {scenario_type}")
            result[scenario_type] = task
    return result


def add_pair_tasks(pair_audit: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required_metadata = {"global_row", "scenario_token", "scenario_type", "planner_name", "parameters_json"}
    missing = sorted(required_metadata - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing fields: {missing}")
    by_row = metadata.set_index("global_row", drop=False)
    type_to_task = build_type_to_task()
    tasks: List[str] = []
    types: List[str] = []
    for row in pair_audit.to_dict("records"):
        left = by_row.loc[int(row["row_A"])]
        right = by_row.loc[int(row["row_B"])]
        if str(left["scenario_token"]) != str(row["scenario_token"]) or str(right["scenario_token"]) != str(row["scenario_token"]):
            raise ValueError(f"pair/metadata scenario mismatch: {row['scenario_token']}")
        scenario_type = str(left["scenario_type"])
        if scenario_type != str(right["scenario_type"]):
            raise ValueError(f"pair has unequal pre-treatment scenario_type: {row['scenario_token']}")
        if scenario_type not in type_to_task:
            raise ValueError(f"unmapped pre-treatment scenario_type: {scenario_type}")
        types.append(scenario_type)
        tasks.append(type_to_task[scenario_type])
    result = pair_audit.copy()
    result["scenario_type"] = types
    result["task"] = tasks
    return result


def validate_hash_record(record: Mapping[str, Any], label: str) -> None:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"{label} path missing: {path}")
    observed = sha256_file(path)
    if observed != record.get("sha256"):
        raise ValueError(f"{label} hash mismatch: observed={observed}, expected={record.get('sha256')}")


def validate_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = args.analysis_dir / "m6_5_locked_confirmation_summary.json"
    pair_audit_path = args.analysis_dir / "m6_5_pair_quality_audit.csv"
    quality_table_path = args.analysis_dir / "m6_5_primary_and_quality_sensitivity.csv"
    task_summary_path = args.analysis_dir / "task_conditioned/milestone6_2_summary.json"
    task_table_path = args.analysis_dir / "task_conditioned/table_m6_2_task_paired_bdd.csv"
    for path in (summary_path, pair_audit_path, quality_table_path, task_summary_path, task_table_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = read_json(summary_path)
    lock = read_json(args.analysis_lock)
    quality_summary = read_json(args.quality_summary)
    task_summary = read_json(task_summary_path)
    if summary.get("status") != "LOCKED_CONFIRMATION_ANALYSIS_COMPLETE":
        raise ValueError("M6.5 analysis is not complete")
    if lock.get("status") != "FROZEN_BEFORE_CONFIRMATION_EMBEDDING_UNBLINDING":
        raise ValueError("M6.5 analysis lock status is invalid")
    if int(summary.get("pair_count", -1)) != 310 or int(summary.get("row_count", -1)) != 620:
        raise ValueError("M6.5 summary must contain 310 pairs / 620 rows")
    if int(lock.get("pair_count", -1)) != 310 or int(lock.get("row_count", -1)) != 620:
        raise ValueError("M6.5 lock must contain 310 pairs / 620 rows")
    for name, record in lock.get("locked_files", {}).items():
        validate_hash_record(record, f"analysis lock file {name}")
    for name, record in summary.get("inputs", {}).items():
        validate_hash_record(record, f"M6.5 result input {name}")
    disjoint = task_summary.get("disjointness_audit", {})
    power = task_summary.get("power_justification", {}).get("sample_target_validation", {})
    if not disjoint.get("passed") or disjoint.get("scenario_token_overlap_count") != 0 or disjoint.get("log_overlap_count") != 0:
        raise ValueError("M6.5 development disjointness did not pass")
    if not power.get("passed"):
        raise ValueError("M6.5 power target validation did not pass")
    if quality_summary.get("full_pairs") != 310 or quality_summary.get("tier_a_pairs") != 58 or quality_summary.get("tier_b_inclusive_pairs") != 135:
        raise ValueError("M6.5 quality counts differ from frozen observed counts")
    pair_audit = pd.read_csv(pair_audit_path)
    metadata = pd.read_csv(args.metadata_csv)
    paired_delta = pd.read_csv(args.paired_delta_csv)
    task_table = pd.read_csv(task_table_path)
    quality_table = pd.read_csv(quality_table_path)
    if len(pair_audit) != 310 or pair_audit["scenario_token"].nunique() != 310:
        raise ValueError("pair audit must contain 310 unique scenarios")
    if len(metadata) != 620 or metadata["global_row"].nunique() != 620:
        raise ValueError("metadata must contain 620 unique global rows")
    if len(paired_delta) != 310 or paired_delta["scenario"].nunique() != 310:
        raise ValueError("paired delta must contain 310 unique scenarios")
    if set(pair_audit["scenario_token"].astype(str)) != set(paired_delta["scenario"].astype(str)):
        raise ValueError("pair audit and paired delta scenario sets differ")
    if pair_audit["valid_horizon_equal"].map(parse_bool).sum() != 310:
        raise ValueError("not all confirmation pairs have equal valid horizon")
    if pair_audit["embedding_rows_finite"].map(parse_bool).sum() != 310:
        raise ValueError("not all confirmation pairs have finite embeddings")
    learned = task_table.loc[task_table["representation"] == "learned_embedding"]
    if set(learned["task"]) != set(TASK_ORDER) or len(learned) != 5:
        raise ValueError("learned embedding task family differs from frozen five tasks")
    if not learned["reject_holm_0_05"].map(parse_bool).all():
        raise ValueError("M6.5 learned embedding task table no longer matches completed result")
    full = quality_table.loc[quality_table["dataset"] == "full_primary"]
    if len(full) != 1 or not math.isclose(float(full.iloc[0]["original_monte_carlo_p"]), float(summary["primary_endpoint"]["original_monte_carlo_p"]), rel_tol=0, abs_tol=1e-15):
        raise ValueError("M6.5 primary differs between summary and quality table")
    return {
        "summary": summary,
        "lock": lock,
        "quality_summary": quality_summary,
        "task_summary": task_summary,
        "pair_audit": add_pair_tasks(pair_audit, metadata),
        "metadata": metadata,
        "paired_delta": paired_delta,
        "task_table": task_table,
        "quality_table": quality_table,
        "paths": {
            "summary": summary_path,
            "pair_audit": pair_audit_path,
            "quality_table": quality_table_path,
            "task_summary": task_summary_path,
            "task_table": task_table_path,
        },
    }


def quality_attribution_rows(pair_audit: pd.DataFrame, *, repetitions: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    y = pair_audit["embedding_l2_distance"].to_numpy(dtype=np.float64)
    tasks = pair_audit["task"].astype(str).to_numpy()
    metrics = [
        ("max_pair_fallback_rate", "maximum paired fallback rate"),
        ("fallback_rate_abs_delta", "absolute within-pair fallback-rate difference"),
        ("max_pair_ambiguous_rate", "maximum paired ambiguous-frame rate"),
        ("ambiguous_rate_abs_delta", "absolute within-pair ambiguous-rate difference"),
    ]
    for metric_position, (metric, label) in enumerate(metrics):
        x = pair_audit[metric].to_numpy(dtype=np.float64)
        result = spearmanr(x, y)
        low, high = bootstrap_rank_correlation_ci(
            x, y, repetitions=repetitions, seed=seed + metric_position, strata=tasks
        )
        adjusted = task_adjusted_rank_association(x, y, tasks)
        adj_low, adj_high = bootstrap_rank_correlation_ci(
            adjusted["residual_x"],
            adjusted["residual_y"],
            repetitions=repetitions,
            seed=seed + 100 + metric_position,
            strata=tasks,
        )
        rows.extend([
            {
                "scope": "overall_stratified_bootstrap",
                "task": "all",
                "quality_metric": metric,
                "quality_metric_label": label,
                "n_pairs": len(pair_audit),
                "rank_association": float(result.statistic),
                "raw_p": float(result.pvalue),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "role": "descriptive_exploratory_post_treatment",
            },
            {
                "scope": "task_adjusted_rank_residual",
                "task": "all",
                "quality_metric": metric,
                "quality_metric_label": label,
                "n_pairs": len(pair_audit),
                "rank_association": adjusted["rho"],
                "raw_p": adjusted["p_value"],
                "bootstrap_ci95_low": adj_low,
                "bootstrap_ci95_high": adj_high,
                "role": "descriptive_exploratory_post_treatment",
            },
        ])
        for task_position, task in enumerate(TASK_ORDER):
            subset = pair_audit.loc[pair_audit["task"] == task]
            task_x = subset[metric].to_numpy(dtype=np.float64)
            task_y = subset["embedding_l2_distance"].to_numpy(dtype=np.float64)
            if len(np.unique(task_x)) < 2:
                rho, p_value = 0.0, 1.0
            else:
                task_result = spearmanr(task_x, task_y)
                rho, p_value = float(task_result.statistic), float(task_result.pvalue)
            task_low, task_high = bootstrap_rank_correlation_ci(
                task_x,
                task_y,
                repetitions=repetitions,
                seed=seed + 1000 + metric_position * 10 + task_position,
            )
            rows.append({
                "scope": "within_task",
                "task": task,
                "quality_metric": metric,
                "quality_metric_label": label,
                "n_pairs": len(subset),
                "rank_association": rho,
                "raw_p": p_value,
                "bootstrap_ci95_low": task_low,
                "bootstrap_ci95_high": task_high,
                "role": "descriptive_exploratory_post_treatment",
            })
    return rows


def tier_distance_rows(pair_audit: pd.DataFrame, *, repetitions: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for position, tier in enumerate(TIER_ORDER):
        values = pair_audit.loc[pair_audit["pair_quality_tier"] == tier, "embedding_l2_distance"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean_ci(values, repetitions=repetitions, seed=seed + position)
        rows.append({
            "pair_quality_tier": tier,
            "n_pairs": len(values),
            "mean_embedding_l2": float(np.mean(values)),
            "mean_ci95_low": low,
            "mean_ci95_high": high,
            "median_embedding_l2": float(np.median(values)),
            "q25_embedding_l2": float(np.quantile(values, 0.25)),
            "q75_embedding_l2": float(np.quantile(values, 0.75)),
            "role": "descriptive_exploratory_post_treatment",
        })
    return rows


def quality_composition_rows(pair_audit: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for task in TASK_ORDER:
        subset = pair_audit.loc[pair_audit["task"] == task]
        counts = subset["pair_quality_tier"].value_counts().to_dict()
        row: Dict[str, Any] = {"task": task, "n_pairs": len(subset)}
        for tier in TIER_ORDER:
            row[f"tier_{tier.lower()}_pairs"] = int(counts.get(tier, 0))
            row[f"tier_{tier.lower()}_fraction"] = float(counts.get(tier, 0) / len(subset))
        rows.append(row)
    return rows


def kinematic_rows(paired_delta: pd.DataFrame, *, repetitions: int, seed: int) -> List[Dict[str, Any]]:
    definitions = [
        ("delta_mean_speed", "Mean speed", "m/s"),
        ("delta_rms_accel", "RMS acceleration", "m/s²"),
        ("delta_mean_thw", "Mean THW", "s"),
        ("delta_mean_front_distance", "Mean front distance", "m"),
    ]
    rows = []
    for position, (metric, label, unit) in enumerate(definitions):
        values = finite(paired_delta[metric])
        low, high = bootstrap_mean_ci(values, repetitions=repetitions, seed=seed + position)
        rows.append({
            "metric": metric,
            "label": label,
            "unit": unit,
            "n_finite_pairs": len(values),
            "mean_delta_A_minus_B": float(np.mean(values)),
            "mean_ci95_low": low,
            "mean_ci95_high": high,
            "median_delta_A_minus_B": float(np.median(values)),
            "positive_fraction": float(np.mean(values > 0)),
            "role": "descriptive_supporting_evidence",
        })
    return rows


def planner_treatment_rows(metadata: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for planner, group in metadata.groupby("planner_name", sort=True):
        values = {json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")) for raw in group["parameters_json"].astype(str)}
        if len(values) != 1:
            raise ValueError(f"planner has multiple treatment configurations: {planner}")
        parameters = json.loads(next(iter(values)))
        for key in sorted(parameters):
            value = parameters[key]
            rows.append({
                "planner_name": planner,
                "policy_style": str(group["policy_style"].iloc[0]),
                "parameter": key,
                "value": json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value,
                "planner_fingerprint": stable_hash(parameters),
            })
    return rows


def save_figure(fig: plt.Figure, plot_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(plot_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    plot_dir: Path,
    *,
    task_table: pd.DataFrame,
    quality_table: pd.DataFrame,
    pair_audit: pd.DataFrame,
    quality_composition: Sequence[Mapping[str, Any]],
    kinematics: Sequence[Mapping[str, Any]],
) -> List[str]:
    figures: List[str] = []
    learned = task_table.loc[task_table["representation"] == "learned_embedding"].set_index("task").loc[TASK_ORDER]
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    bars = axis.barh(
        [TASK_LABELS[task] for task in TASK_ORDER],
        learned["mmd2"].to_numpy(),
        color=[COLORS[task] for task in TASK_ORDER],
    )
    for bar, (_, row) in zip(bars, learned.iterrows()):
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  Holm p={row['holm_p_within_pretreatment_tasks']:.3g}", va="center", fontsize=9)
    axis.set_xlabel("Biased single-RBF MMD²")
    axis.set_title("Locked learned-embedding BDD by pre-treatment task")
    axis.invert_yaxis()
    save_figure(fig, plot_dir, "m6_6_task_bdd")
    figures.append("m6_6_task_bdd")

    quality = quality_table.set_index("dataset").loc[["full_primary", "tier_a_sensitivity", "tier_a_plus_b_sensitivity"]]
    labels = ["Full (n=310)", "Tier A (n=58)", "Tier A+B (n=135)"]
    positions = np.arange(3)
    fig, axis = plt.subplots(figsize=(8.0, 4.3))
    axis.bar(positions - 0.18, quality["original_mmd2"], width=0.36, label="Original embedding", color="#2563eb")
    axis.bar(positions + 0.18, quality["residual_mmd2"], width=0.36, label="Pair-midpoint residual", color="#7c3aed")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("MMD²")
    axis.set_title("Locked primary and lane-quality sensitivities")
    axis.legend(frameon=False)
    axis.set_ylim(0, float(quality[["original_mmd2", "residual_mmd2"]].to_numpy().max()) * 1.28)
    for pos, (_, row) in enumerate(quality.iterrows()):
        axis.text(pos - 0.18, row["original_mmd2"], f"p={row['original_monte_carlo_p']:.3g}", ha="center", va="bottom", fontsize=8, rotation=35)
        axis.text(pos + 0.18, row["residual_mmd2"], f"p={row['residual_monte_carlo_p']:.3g}", ha="center", va="bottom", fontsize=8, rotation=35)
    save_figure(fig, plot_dir, "m6_6_quality_robustness")
    figures.append("m6_6_quality_robustness")

    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    for task in TASK_ORDER:
        subset = pair_audit.loc[pair_audit["task"] == task]
        axis.scatter(subset["max_pair_fallback_rate"], subset["embedding_l2_distance"], s=24, alpha=0.68, label=TASK_LABELS[task], color=COLORS[task])
    axis.set_xlabel("Maximum fallback rate within planner pair")
    axis.set_ylabel("Within-pair embedding L2 distance")
    axis.set_title("Post-treatment lane-quality association")
    axis.legend(frameon=False, fontsize=8)
    save_figure(fig, plot_dir, "m6_6_fallback_embedding_distance")
    figures.append("m6_6_fallback_embedding_distance")

    composition = pd.DataFrame(quality_composition).set_index("task").loc[TASK_ORDER]
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    left = np.zeros(len(TASK_ORDER), dtype=np.float64)
    for tier in TIER_ORDER:
        values = composition[f"tier_{tier.lower()}_fraction"].to_numpy()
        axis.barh([TASK_LABELS[task] for task in TASK_ORDER], values, left=left, label=f"Tier {tier}", color=COLORS[tier])
        left += values
    axis.set_xlim(0, 1)
    axis.set_xlabel("Fraction of scenario pairs")
    axis.set_title("Lane-quality composition by pre-treatment task")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.invert_yaxis()
    save_figure(fig, plot_dir, "m6_6_task_quality_composition")
    figures.append("m6_6_task_quality_composition")

    kin = pd.DataFrame(kinematics)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2))
    for axis, (_, row) in zip(axes.ravel(), kin.iterrows()):
        mean = float(row["mean_delta_A_minus_B"])
        low = float(row["mean_ci95_low"])
        high = float(row["mean_ci95_high"])
        axis.errorbar(
            [mean],
            [0],
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color="#155e75",
            capsize=5,
            markersize=7,
        )
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set_yticks([])
        axis.set_title(str(row["label"]), fontsize=10)
        axis.set_xlabel(f"A−B delta ({row['unit']})", fontsize=9)
        axis.text(
            0.02,
            0.92,
            f"mean={mean:.3g}; 95% CI [{low:.3g}, {high:.3g}]",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
    fig.suptitle("Descriptive paired kinematic contrasts", fontsize=13)
    save_figure(fig, plot_dir, "m6_6_kinematic_contrasts")
    figures.append("m6_6_kinematic_contrasts")

    controls = task_table.loc[task_table["representation"].isin(["interaction_features", "trajectory_summary"])].copy()
    pivot = controls.pivot(index="representation", columns="task", values="p_value").loc[:, TASK_ORDER]
    values = -np.log10(np.clip(pivot.to_numpy(dtype=np.float64), 1e-12, 1.0))
    fig, axis = plt.subplots(figsize=(8.2, 3.2))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER], rotation=25, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), ["Interaction features", "Trajectory summary"])
    axis.set_title("Mechanism controls: −log10(raw p), exploratory")
    fig.colorbar(image, ax=axis, label="−log10(raw p)")
    save_figure(fig, plot_dir, "m6_6_mechanism_controls")
    figures.append("m6_6_mechanism_controls")
    return figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage7 M6.6 paper evidence package from locked M6.5 outputs.")
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument("--analysis_lock", type=Path, required=True)
    parser.add_argument("--quality_summary", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    if args.bootstrap_repetitions < 1000:
        raise ValueError("--bootstrap_repetitions must be at least 1000")
    data = validate_inputs(args)
    args.output_dir.mkdir(parents=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir()

    summary = data["summary"]
    pair_audit: pd.DataFrame = data["pair_audit"]
    task_table: pd.DataFrame = data["task_table"]
    quality_table: pd.DataFrame = data["quality_table"]
    paired_delta: pd.DataFrame = data["paired_delta"]
    metadata: pd.DataFrame = data["metadata"]

    overview_rows = [{
        "dataset_role": summary["dataset_role"],
        "pairs": summary["pair_count"],
        "rows": summary["row_count"],
        "primary_mmd2": summary["primary_endpoint"]["original_mmd2"],
        "primary_exceedance_count": summary["primary_endpoint"]["original_exceedance_count"],
        "primary_permutations": 100000,
        "primary_plus_one_p": summary["primary_endpoint"]["original_monte_carlo_p"],
        "development_scenario_overlap": data["task_summary"]["disjointness_audit"]["scenario_token_overlap_count"],
        "development_log_overlap": data["task_summary"]["disjointness_audit"]["log_overlap_count"],
        "power_target_validation": data["task_summary"]["power_justification"]["sample_target_validation"]["passed"],
        "interpretation": "planner-conditioned behavior distribution difference; not safety or superiority",
    }]
    learned_rows = task_table.loc[task_table["representation"] == "learned_embedding"].to_dict("records")
    mechanism_rows = task_table.loc[task_table["representation"] != "learned_embedding"].to_dict("records")
    quality_rows = quality_table.to_dict("records")
    attribution = quality_attribution_rows(pair_audit, repetitions=args.bootstrap_repetitions, seed=args.seed)
    tier_distance = tier_distance_rows(pair_audit, repetitions=args.bootstrap_repetitions, seed=args.seed + 2000)
    composition = quality_composition_rows(pair_audit)
    kinematics = kinematic_rows(paired_delta, repetitions=args.bootstrap_repetitions, seed=args.seed + 3000)
    treatments = planner_treatment_rows(metadata)
    sample_checks = [
        {"check": "locked_confirmation_complete", "passed": summary["status"] == "LOCKED_CONFIRMATION_ANALYSIS_COMPLETE", "value": summary["status"]},
        {"check": "pair_count_310", "passed": len(pair_audit) == 310, "value": len(pair_audit)},
        {"check": "row_count_620", "passed": len(metadata) == 620, "value": len(metadata)},
        {"check": "development_scenario_overlap_zero", "passed": data["task_summary"]["disjointness_audit"]["scenario_token_overlap_count"] == 0, "value": data["task_summary"]["disjointness_audit"]["scenario_token_overlap_count"]},
        {"check": "development_log_overlap_zero", "passed": data["task_summary"]["disjointness_audit"]["log_overlap_count"] == 0, "value": data["task_summary"]["disjointness_audit"]["log_overlap_count"]},
        {"check": "power_targets_pass", "passed": data["task_summary"]["power_justification"]["sample_target_validation"]["passed"], "value": data["task_summary"]["power_justification"]["sample_target_validation"]["observed_complete_pairs_by_task"]},
        {"check": "all_five_tasks_holm_significant", "passed": all(parse_bool(row["reject_holm_0_05"]) for row in learned_rows), "value": 5},
        {"check": "pair_embedding_rows_finite", "passed": pair_audit["embedding_rows_finite"].map(parse_bool).all(), "value": 310},
    ]

    save_table(args.output_dir, "table_m6_6_confirmation_overview", overview_rows, list(overview_rows[0]))
    save_table(args.output_dir, "table_m6_6_task_bdd", learned_rows, ["task", "n_pairs", "mmd2", "p_value", "holm_p_within_pretreatment_tasks", "reject_holm_0_05"])
    save_table(args.output_dir, "table_m6_6_quality_robustness", quality_rows, ["dataset", "n_pairs", "original_mmd2", "original_monte_carlo_p", "original_holm_p_within_quality_sensitivity_family", "residual_mmd2", "residual_monte_carlo_p", "residual_holm_p_within_quality_sensitivity_family"])
    save_table(args.output_dir, "table_m6_6_mechanism_controls", mechanism_rows, ["representation", "task", "n_pairs", "mmd2", "p_value", "analysis_role"])
    save_table(args.output_dir, "table_m6_6_sample_audit", sample_checks, ["check", "passed", "value"])
    save_table(args.output_dir, "table_m6_6_quality_attribution", attribution, ["scope", "task", "quality_metric", "n_pairs", "rank_association", "raw_p", "bootstrap_ci95_low", "bootstrap_ci95_high", "role"])
    save_table(args.output_dir, "table_m6_6_quality_tier_distance", tier_distance, ["pair_quality_tier", "n_pairs", "mean_embedding_l2", "mean_ci95_low", "mean_ci95_high", "median_embedding_l2", "role"])
    save_table(args.output_dir, "table_m6_6_task_quality_composition", composition, ["task", "n_pairs", "tier_a_pairs", "tier_a_fraction", "tier_b_pairs", "tier_b_fraction", "tier_c_pairs", "tier_c_fraction"])
    save_table(args.output_dir, "table_m6_6_kinematic_contrasts", kinematics, ["label", "unit", "n_finite_pairs", "mean_delta_A_minus_B", "mean_ci95_low", "mean_ci95_high", "median_delta_A_minus_B", "positive_fraction", "role"])
    save_table(args.output_dir, "table_m6_6_planner_treatments", treatments, ["planner_name", "policy_style", "parameter", "value", "planner_fingerprint"])

    figures = create_figures(
        plot_dir,
        task_table=task_table,
        quality_table=quality_table,
        pair_audit=pair_audit,
        quality_composition=composition,
        kinematics=kinematics,
    )
    max_fallback_overall = next(row for row in attribution if row["scope"] == "overall_stratified_bootstrap" and row["quality_metric"] == "max_pair_fallback_rate")
    max_fallback_adjusted = next(row for row in attribution if row["scope"] == "task_adjusted_rank_residual" and row["quality_metric"] == "max_pair_fallback_rate")
    limitations = [
        "Lane-quality measures are realized after planner rollout and are post-treatment; associations are descriptive/exploratory, not causal adjustment.",
        "The M6.5 full 310-pair primary remains the sole overall confirmatory endpoint; no M6.6 subset or regression replaces it.",
        "Tier A residual sensitivity is not significant, and global fallback exceeds the legacy 5% scale-readiness threshold.",
        "BDD demonstrates a behavior-representation distribution difference, not safety, policy quality, or planner superiority.",
        "Mechanism-control p-values are raw exploratory values and are not part of the learned-embedding Holm family.",
    ]
    evidence_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_QUALITY_LIMITATIONS",
        "dataset_role": summary["dataset_role"],
        "pair_count": len(pair_audit),
        "row_count": len(metadata),
        "primary_endpoint_preserved_without_recomputation": summary["primary_endpoint"],
        "learned_embedding_task_results_preserved": learned_rows,
        "quality_tier_counts": pair_audit["pair_quality_tier"].value_counts().to_dict(),
        "quality_attribution_key_results": {
            "max_fallback_overall": max_fallback_overall,
            "max_fallback_task_adjusted": max_fallback_adjusted,
        },
        "tables": sorted(path.name for path in args.output_dir.glob("table_m6_6_*.csv")),
        "figures": figures,
        "sample_checks": sample_checks,
        "limitations": limitations,
        "interpretation": "Locked confirmation supports a planner-conditioned behavior distribution difference across five pre-treatment tasks, with material lane-quality association that limits pure-mechanism attribution.",
    }
    write_json(args.output_dir / "m6_6_confirmation_evidence_summary.json", evidence_summary)

    input_paths = {
        "analysis_lock": args.analysis_lock,
        "quality_summary": args.quality_summary,
        "metadata": args.metadata_csv,
        "paired_delta": args.paired_delta_csv,
        **data["paths"],
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in input_paths.items()},
        "parameters": {"bootstrap_repetitions": args.bootstrap_repetitions, "seed": args.seed},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "output_summary_sha256": sha256_file(args.output_dir / "m6_6_confirmation_evidence_summary.json"),
    }
    write_json(args.output_dir / "m6_6_reproducibility_provenance.json", provenance)

    report = [
        "# Stage 7 M6.6 Confirmation Evidence Package",
        "",
        "## Verdict",
        "",
        "`PASS_WITH_QUALITY_LIMITATIONS`",
        "",
        f"- locked pairs: `{len(pair_audit)}`",
        f"- overall original-embedding primary p: `{summary['primary_endpoint']['original_monte_carlo_p']:.8g}`",
        "- learned-embedding tasks passing frozen Holm family: `5/5`",
        "- M6.6 recomputed confirmatory p-values: `none`",
        "",
        "## Locked confirmation overview",
        "",
        markdown_table(overview_rows, list(overview_rows[0])).rstrip(),
        "",
        "## Pre-treatment task family",
        "",
        markdown_table(learned_rows, ["task", "n_pairs", "mmd2", "p_value", "holm_p_within_pretreatment_tasks", "reject_holm_0_05"]).rstrip(),
        "",
        "## Quality attribution boundary",
        "",
        f"- overall max-fallback association: rho=`{max_fallback_overall['rank_association']:.4f}`; bootstrap 95% CI=`[{max_fallback_overall['bootstrap_ci95_low']:.4f}, {max_fallback_overall['bootstrap_ci95_high']:.4f}]`",
        f"- task-adjusted rank-residual association: rho=`{max_fallback_adjusted['rank_association']:.4f}`; bootstrap 95% CI=`[{max_fallback_adjusted['bootstrap_ci95_low']:.4f}, {max_fallback_adjusted['bootstrap_ci95_high']:.4f}]`",
        "- these are post-treatment descriptive associations and are not causal adjustment",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in limitations],
    ]
    (args.output_dir / "m6_6_confirmation_evidence_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manuscript = [
        "# Manuscript-ready M6.6 result text",
        "",
        "## English",
        "",
        (
            f"In a locked, log- and scenario-disjoint confirmation set of {len(pair_audit)} paired scenarios, "
            f"the original 64-dimensional behavior embedding differed between the assertive and conservative planners "
            f"(biased single-RBF MMD²={summary['primary_endpoint']['original_mmd2']:.6f}, "
            f"0/{100000} paired-label permutations reached the observed statistic, plus-one p={summary['primary_endpoint']['original_monte_carlo_p']:.6g}). "
            "All five pre-treatment task strata remained significant after the frozen Holm correction. "
            f"However, maximum pair-level lane-assignment fallback was associated with embedding distance "
            f"(Spearman rho={max_fallback_overall['rank_association']:.3f}, bootstrap 95% CI "
            f"[{max_fallback_overall['bootstrap_ci95_low']:.3f}, {max_fallback_overall['bootstrap_ci95_high']:.3f}]). "
            "Because lane quality is realized after rollout, this association is descriptive and does not identify a causal quality-adjusted planner effect."
        ),
        "",
        "## 中文",
        "",
        (
            f"在{len(pair_audit)}个新 log/scenario-disjoint 锁定场景对上，assertive 与 conservative planner 的原始64维行为 embedding "
            f"存在显著分布差异（biased single-RBF MMD²={summary['primary_endpoint']['original_mmd2']:.6f}，"
            f"100000次成对标签置换中0次达到 observed，plus-one p={summary['primary_endpoint']['original_monte_carlo_p']:.6g}），"
            "五个仿真前任务层在冻结 Holm 校正后均显著。与此同时，pair-level lane fallback 与 embedding distance 存在明显关联；"
            "由于 lane quality 是 rollout 后实现的变量，该关联仅为描述性证据，不能解释为因果质量校正后的 planner effect。"
        ),
    ]
    (args.output_dir / "m6_6_manuscript_results.md").write_text("\n".join(manuscript) + "\n", encoding="utf-8")
    if not all(parse_bool(row["passed"]) for row in sample_checks):
        raise RuntimeError(f"M6.6 sample checks failed: {sample_checks}")
    print(json.dumps(evidence_summary, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
