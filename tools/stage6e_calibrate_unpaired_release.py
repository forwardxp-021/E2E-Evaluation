#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tools import stage6d_unpaired_version_bdd as stage6d


SCHEMA_VERSION = "stage6e_unpaired_release_calibration_v1"
PASS_STATUS = "PASS_PUBLIC_FIELD_RELEASE_EMULATION"
INSUFFICIENT_STATUS = "INSUFFICIENT_COMPARABLE_TRIALS"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    required = [
        "row_id_column",
        "pair_id_column",
        "cluster_column",
        "planner_column",
        "planners",
        "covariates",
    ]
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"config missing required fields: {missing}")
    planners = raw["planners"]
    if not isinstance(planners, dict) or set(planners) != {"assertive", "conservative"}:
        raise ValueError("config.planners must contain exactly assertive and conservative")
    if str(planners["assertive"]) == str(planners["conservative"]):
        raise ValueError("assertive and conservative planner labels must differ")
    config = dict(raw)
    config.setdefault("seed", 20260809)
    config.setdefault("alpha", 0.05)
    config.setdefault("calibration_trials_per_planner", 100)
    config.setdefault("evaluation_trials_per_planner", 100)
    config.setdefault("ab_trials_per_direction", 100)
    config.setdefault("split_search_candidates", 32)
    config.setdefault("max_mmd_samples", 2000)
    config.setdefault("minimum_valid_trials", 20)
    config.setdefault("invariant_pair_columns", [])
    if not 0 < float(config["alpha"]) < 0.5:
        raise ValueError("config.alpha must be in (0, 0.5)")
    for name in [
        "calibration_trials_per_planner",
        "evaluation_trials_per_planner",
        "ab_trials_per_direction",
        "split_search_candidates",
        "max_mmd_samples",
        "minimum_valid_trials",
    ]:
        if int(config[name]) < 1:
            raise ValueError(f"config.{name} must be >=1")
    design = {
        "group_column": "_release_group",
        "groups": {"A": "A", "B": "B"},
        "row_id_column": str(config["row_id_column"]),
        "cluster_column": str(config["cluster_column"]),
        "reference_distribution": "equal_group_pooled_common_support",
        "covariates": list(config["covariates"]),
        "tasks": list(config.get("tasks", [])),
        "post_treatment_columns": list(config.get("post_treatment_columns", [])),
        "thresholds": dict(config.get("support_thresholds", {})),
    }
    config["stage6d_design"] = stage6d.validate_design(design)
    return config


def validate_paired_metadata(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    embedding_rows: int,
) -> pd.DataFrame:
    row_col = str(config["row_id_column"])
    pair_col = str(config["pair_id_column"])
    cluster_col = str(config["cluster_column"])
    planner_col = str(config["planner_column"])
    required = {
        row_col,
        pair_col,
        cluster_col,
        planner_col,
        *(str(item["name"]) for item in config["covariates"]),
        *(str(item["column"]) for item in config.get("tasks", [])),
        *(str(name) for name in config.get("invariant_pair_columns", [])),
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    result = metadata.copy()
    rows = pd.to_numeric(result[row_col], errors="raise").astype(np.int64)
    if rows.isna().any() or rows.duplicated().any():
        raise ValueError(f"row id column must be non-missing and unique: {row_col}")
    if len(rows) and (int(rows.min()) < 0 or int(rows.max()) >= embedding_rows):
        raise ValueError(
            f"row ids exceed embedding shape: min={int(rows.min())}, max={int(rows.max())}, "
            f"embedding_rows={embedding_rows}"
        )
    expected = {str(config["planners"]["assertive"]), str(config["planners"]["conservative"])}
    result[planner_col] = result[planner_col].astype(str)
    result = result.loc[result[planner_col].isin(expected)].copy()
    if not len(result):
        raise ValueError(f"metadata contains no configured planner rows: {sorted(expected)}")
    if result[pair_col].isna().any() or result[cluster_col].isna().any():
        raise ValueError("pair or cluster column contains missing values")
    result[pair_col] = result[pair_col].astype(str)
    result[cluster_col] = result[cluster_col].astype(str)
    result[row_col] = rows.loc[result.index].to_numpy(dtype=np.int64)
    failures: List[str] = []
    invariant_columns = [cluster_col] + [str(name) for name in config.get("invariant_pair_columns", [])]
    for pair_id, group in result.groupby(pair_col, sort=False):
        observed = set(group[planner_col].astype(str))
        if len(group) != 2 or observed != expected:
            failures.append(f"{pair_id}: rows={len(group)}, planners={sorted(observed)}")
            continue
        bad_columns = [name for name in invariant_columns if group[name].astype(str).nunique(dropna=False) != 1]
        if bad_columns:
            failures.append(f"{pair_id}: invariant mismatch={bad_columns}")
    if failures:
        raise ValueError(f"paired metadata audit failed; examples={failures[:5]}")
    pair_count = result[pair_col].nunique()
    if len(result) != 2 * pair_count:
        raise ValueError(f"expected exactly two rows per pair: rows={len(result)}, pairs={pair_count}")
    return result.reset_index(drop=True)


def build_pair_inventory(metadata: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    pair_col = str(config["pair_id_column"])
    planner_col = str(config["planner_column"])
    assertive = str(config["planners"]["assertive"])
    inventory = metadata.loc[metadata[planner_col].astype(str) == assertive].copy()
    if inventory[pair_col].duplicated().any():
        raise ValueError("assertive planner inventory has duplicate pair ids")
    inventory, _ = stage6d.coarsen_covariates(inventory, config["stage6d_design"])
    return inventory.reset_index(drop=True)


def split_score(inventory: pd.DataFrame, cluster_col: str, logs_a: Sequence[str]) -> float:
    mask_a = inventory[cluster_col].astype(str).isin(set(logs_a)).to_numpy()
    if not mask_a.any() or mask_a.all():
        return math.inf
    counts_a = inventory.loc[mask_a, "_support_cell"].value_counts(normalize=True)
    counts_b = inventory.loc[~mask_a, "_support_cell"].value_counts(normalize=True)
    cells = sorted(set(counts_a.index.astype(str)) | set(counts_b.index.astype(str)))
    composition = sum(abs(float(counts_a.get(cell, 0.0)) - float(counts_b.get(cell, 0.0))) for cell in cells)
    row_imbalance = abs(int(mask_a.sum()) - int((~mask_a).sum())) / len(inventory)
    cluster_imbalance = abs(len(set(logs_a)) - (inventory[cluster_col].nunique() - len(set(logs_a)))) / inventory[cluster_col].nunique()
    return float(composition + row_imbalance + 0.25 * cluster_imbalance)


def choose_log_split(
    inventory: pd.DataFrame,
    cluster_col: str,
    rng: np.random.Generator,
    search_candidates: int,
) -> Tuple[List[str], List[str], float]:
    logs = np.asarray(sorted(inventory[cluster_col].astype(str).unique()))
    if len(logs) < 4:
        raise ValueError(f"at least four independent clusters are required, observed={len(logs)}")
    n_a = len(logs) // 2
    best: Tuple[List[str], List[str], float] | None = None
    for _ in range(search_candidates):
        shuffled = rng.permutation(logs)
        logs_a = sorted(shuffled[:n_a].tolist())
        logs_b = sorted(shuffled[n_a:].tolist())
        score = split_score(inventory, cluster_col, logs_a)
        if best is None or score < best[2]:
            best = (logs_a, logs_b, score)
    assert best is not None
    return best


def build_trial(
    metadata: pd.DataFrame,
    embeddings_all: np.ndarray,
    config: Mapping[str, Any],
    logs_a: Sequence[str],
    logs_b: Sequence[str],
    planner_a: str,
    planner_b: str,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    planner_col = str(config["planner_column"])
    cluster_col = str(config["cluster_column"])
    pair_col = str(config["pair_id_column"])
    row_col = str(config["row_id_column"])
    group_a = metadata.loc[
        metadata[cluster_col].astype(str).isin(set(logs_a))
        & (metadata[planner_col].astype(str) == str(planner_a))
    ].copy()
    group_b = metadata.loc[
        metadata[cluster_col].astype(str).isin(set(logs_b))
        & (metadata[planner_col].astype(str) == str(planner_b))
    ].copy()
    group_a["_release_group"] = "A"
    group_b["_release_group"] = "B"
    overlap_logs = set(group_a[cluster_col].astype(str)) & set(group_b[cluster_col].astype(str))
    overlap_pairs = set(group_a[pair_col].astype(str)) & set(group_b[pair_col].astype(str))
    if overlap_logs or overlap_pairs:
        raise AssertionError(
            f"release split overlap: logs={sorted(overlap_logs)[:3]}, pairs={sorted(overlap_pairs)[:3]}"
        )
    trial = pd.concat([group_a, group_b], ignore_index=True)
    rows = trial[row_col].to_numpy(dtype=np.int64)
    embeddings = np.asarray(embeddings_all[rows], dtype=np.float64)
    trial, _ = stage6d.coarsen_covariates(trial, config["stage6d_design"])
    audit = {
        "n_A": int(len(group_a)),
        "n_B": int(len(group_b)),
        "clusters_A": int(group_a[cluster_col].nunique()),
        "clusters_B": int(group_b[cluster_col].nunique()),
        "log_overlap_count": int(len(overlap_logs)),
        "scenario_overlap_count": int(len(overlap_pairs)),
    }
    return trial, embeddings, audit


def fixed_scope_bandwidths(
    metadata: pd.DataFrame,
    embeddings_all: np.ndarray,
    config: Mapping[str, Any],
) -> Dict[str, float]:
    row_col = str(config["row_id_column"])
    embeddings = np.asarray(embeddings_all[metadata[row_col].to_numpy(dtype=np.int64)], dtype=np.float64)
    rng = np.random.default_rng(int(config["seed"]) + 17)
    values = {"overall": stage6d.median_bandwidth(embeddings, embeddings[:0], rng)}
    for task in config.get("tasks", []):
        mask = stage6d.task_mask(metadata, task)
        selected = embeddings[mask]
        if len(selected) < 2:
            raise ValueError(f"task has fewer than two rows for bandwidth freeze: {task['name']}")
        values[str(task["name"])] = stage6d.median_bandwidth(selected, selected[:0], rng)
    return values


def evaluate_trial_scopes(
    trial: pd.DataFrame,
    embeddings: np.ndarray,
    config: Mapping[str, Any],
    bandwidths: Mapping[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    design = config["stage6d_design"]
    max_samples = int(config["max_mmd_samples"])
    scopes: List[Tuple[str, np.ndarray]] = [("overall", np.ones(len(trial), dtype=bool))]
    scopes.extend((str(task["name"]), stage6d.task_mask(trial, task)) for task in config.get("tasks", []))
    rows: List[Dict[str, Any]] = []
    for position, (scope, mask) in enumerate(scopes):
        frame = trial.loc[mask].copy().reset_index(drop=True)
        scope_embeddings = embeddings[mask]
        mask_a = stage6d._group_mask(frame, design, "A") if len(frame) else np.zeros(0, dtype=bool)
        mask_b = stage6d._group_mask(frame, design, "B") if len(frame) else np.zeros(0, dtype=bool)
        base = {
            "scope": scope,
            "n_A": int(mask_a.sum()),
            "n_B": int(mask_b.sum()),
            "bandwidth": float(bandwidths[scope]),
        }
        if not mask_a.any() or not mask_b.any():
            rows.append({
                **base,
                "status": stage6d.NOT_COMPARABLE_STATUS,
                "raw_mmd2": math.nan,
                "standardized_mmd2": math.nan,
                "support_fraction_A": 0.0,
                "support_fraction_B": 0.0,
                "ess_ratio_A": 0.0,
                "ess_ratio_B": 0.0,
                "max_weight_ratio_A": math.inf,
                "max_weight_ratio_B": math.inf,
            })
            continue
        rng = np.random.default_rng(seed + position * 1009)
        raw = stage6d.evaluate_mmd(
            frame,
            scope_embeddings,
            design,
            stage6d.uniform_group_weights(frame, design),
            float(bandwidths[scope]),
            rng,
            max_samples,
        )
        standardization = stage6d.build_standardization(frame, design)
        standardized = math.nan
        if standardization["passed"]:
            standardized = stage6d.evaluate_mmd(
                frame,
                scope_embeddings,
                design,
                standardization["weights"],
                float(bandwidths[scope]),
                rng,
                max_samples,
            )
        rows.append({
            **base,
            "status": standardization["status"],
            "raw_mmd2": raw,
            "standardized_mmd2": standardized,
            "support_fraction_A": standardization["group_A"]["support_fraction"],
            "support_fraction_B": standardization["group_B"]["support_fraction"],
            "ess_ratio_A": standardization["group_A"]["ess_ratio"],
            "ess_ratio_B": standardization["group_B"]["ess_ratio"],
            "max_weight_ratio_A": standardization["group_A"]["max_weight_ratio"],
            "max_weight_ratio_B": standardization["group_B"]["max_weight_ratio"],
        })
    return rows


def experiment_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    planners = {key: str(value) for key, value in config["planners"].items()}
    return [
        {
            "experiment_set": "AA_CALIBRATION_ASSERTIVE",
            "family": "AA_CALIBRATION",
            "planner_A": planners["assertive"],
            "planner_B": planners["assertive"],
            "trials": int(config["calibration_trials_per_planner"]),
            "seed_offset": 100_000,
        },
        {
            "experiment_set": "AA_CALIBRATION_CONSERVATIVE",
            "family": "AA_CALIBRATION",
            "planner_A": planners["conservative"],
            "planner_B": planners["conservative"],
            "trials": int(config["calibration_trials_per_planner"]),
            "seed_offset": 200_000,
        },
        {
            "experiment_set": "AA_EVALUATION_ASSERTIVE",
            "family": "AA_EVALUATION",
            "planner_A": planners["assertive"],
            "planner_B": planners["assertive"],
            "trials": int(config["evaluation_trials_per_planner"]),
            "seed_offset": 300_000,
        },
        {
            "experiment_set": "AA_EVALUATION_CONSERVATIVE",
            "family": "AA_EVALUATION",
            "planner_A": planners["conservative"],
            "planner_B": planners["conservative"],
            "trials": int(config["evaluation_trials_per_planner"]),
            "seed_offset": 400_000,
        },
        {
            "experiment_set": "AB_ASSERTIVE_TO_CONSERVATIVE",
            "family": "AB_EVALUATION",
            "planner_A": planners["assertive"],
            "planner_B": planners["conservative"],
            "trials": int(config["ab_trials_per_direction"]),
            "seed_offset": 500_000,
        },
        {
            "experiment_set": "AB_CONSERVATIVE_TO_ASSERTIVE",
            "family": "AB_EVALUATION",
            "planner_A": planners["conservative"],
            "planner_B": planners["assertive"],
            "trials": int(config["ab_trials_per_direction"]),
            "seed_offset": 600_000,
        },
    ]


def empirical_threshold(values: Sequence[float], alpha: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan
    return float(np.quantile(finite, 1.0 - alpha, method="higher"))


def summarize_trials(
    trials: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = float(config["alpha"])
    minimum = int(config["minimum_valid_trials"])
    threshold_rows: List[Dict[str, Any]] = []
    scopes = ["overall"] + [str(task["name"]) for task in config.get("tasks", [])]
    for scope in scopes:
        subset = trials.loc[
            (trials["family"] == "AA_CALIBRATION")
            & (trials["scope"] == scope)
            & (trials["status"] == stage6d.PASS_STATUS)
        ]
        threshold_rows.append({
            "scope": scope,
            "threshold_source": "POOLED_ASSERTIVE_AND_CONSERVATIVE_AA_CALIBRATION",
            "alpha": alpha,
            "quantile": 1.0 - alpha,
            "quantile_method": "higher",
            "valid_calibration_trials": int(len(subset)),
            "minimum_valid_trials": minimum,
            "standardized_mmd2_threshold": empirical_threshold(subset["standardized_mmd2"], alpha),
            "status": "PASS" if len(subset) >= minimum else INSUFFICIENT_STATUS,
        })
    thresholds = pd.DataFrame(threshold_rows)
    threshold_map = dict(zip(thresholds["scope"], thresholds["standardized_mmd2_threshold"]))
    evaluated = trials.copy()
    evaluated["threshold"] = evaluated["scope"].map(threshold_map)
    evaluated["exceeds_threshold"] = (
        (evaluated["status"] == stage6d.PASS_STATUS)
        & np.isfinite(evaluated["standardized_mmd2"])
        & np.isfinite(evaluated["threshold"])
        & (evaluated["standardized_mmd2"] > evaluated["threshold"])
    )
    summary_rows: List[Dict[str, Any]] = []
    for (experiment_set, family, scope), group in evaluated.groupby(
        ["experiment_set", "family", "scope"], sort=False
    ):
        valid = group.loc[
            (group["status"] == stage6d.PASS_STATUS)
            & np.isfinite(group["standardized_mmd2"])
        ]
        values = valid["standardized_mmd2"].to_numpy(dtype=float)
        summary_rows.append({
            "experiment_set": experiment_set,
            "family": family,
            "scope": scope,
            "total_trials": int(len(group)),
            "valid_trials": int(len(valid)),
            "not_comparable_trials": int(len(group) - len(valid)),
            "threshold": float(threshold_map[scope]),
            "exceedance_count": int(valid["exceeds_threshold"].sum()),
            "exceedance_rate": float(valid["exceeds_threshold"].mean()) if len(valid) else math.nan,
            "standardized_mmd2_median": float(np.median(values)) if len(values) else math.nan,
            "standardized_mmd2_q05": float(np.quantile(values, 0.05)) if len(values) else math.nan,
            "standardized_mmd2_q95": float(np.quantile(values, 0.95)) if len(values) else math.nan,
            "status": "PASS" if len(valid) >= minimum else INSUFFICIENT_STATUS,
        })
    return thresholds, evaluated, pd.DataFrame(summary_rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    rate = successes / total
    denominator = 1.0 + (z * z / total)
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def operating_characteristics(evaluated: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scope, group in evaluated.groupby("scope", sort=False):
        aa = group.loc[
            (group["family"] == "AA_EVALUATION")
            & (group["status"] == stage6d.PASS_STATUS)
            & np.isfinite(group["standardized_mmd2"])
        ]
        ab = group.loc[
            (group["family"] == "AB_EVALUATION")
            & (group["status"] == stage6d.PASS_STATUS)
            & np.isfinite(group["standardized_mmd2"])
        ]
        aa_success = int(aa["exceeds_threshold"].sum())
        ab_success = int(ab["exceeds_threshold"].sum())
        aa_rate = float(aa_success / len(aa)) if len(aa) else math.nan
        ab_rate = float(ab_success / len(ab)) if len(ab) else math.nan
        aa_low, aa_high = wilson_interval(aa_success, len(aa))
        ab_low, ab_high = wilson_interval(ab_success, len(ab))
        rows.append({
            "scope": scope,
            "scope_role": "PRIMARY_OVERALL" if scope == "overall" else "DIAGNOSTIC_TASK_NO_MULTIPLICITY_CONTROL",
            "threshold": float(group["threshold"].dropna().iloc[0]) if group["threshold"].notna().any() else math.nan,
            "aa_evaluation_valid_trials": int(len(aa)),
            "aa_false_positive_count": aa_success,
            "aa_false_positive_rate": aa_rate,
            "aa_false_positive_wilson95_low": aa_low,
            "aa_false_positive_wilson95_high": aa_high,
            "ab_evaluation_valid_trials": int(len(ab)),
            "ab_detection_count": ab_success,
            "ab_detection_rate": ab_rate,
            "ab_detection_wilson95_low": ab_low,
            "ab_detection_wilson95_high": ab_high,
            "detection_minus_false_positive_rate": ab_rate - aa_rate,
            "detection_to_false_positive_ratio": (
                ab_rate / aa_rate if np.isfinite(aa_rate) and aa_rate > 0 else math.inf
            ),
        })
    return pd.DataFrame(rows)


def descriptive_overall_conclusion(operating: pd.DataFrame) -> str:
    row = operating.loc[operating["scope"] == "overall"]
    if len(row) != 1:
        return "INSUFFICIENT_OVERALL_OPERATING_CHARACTERISTICS"
    value = row.iloc[0]
    separated = value["ab_detection_wilson95_low"] > value["aa_false_positive_wilson95_high"]
    if not separated:
        return "AB_NOT_CLEARLY_SEPARATED_FROM_AA"
    if value["ab_detection_rate"] >= 0.8:
        return "AB_SEPARATED_FROM_AA_WITH_STRONG_SINGLE_RELEASE_SENSITIVITY"
    return "AB_SEPARATED_FROM_AA_BUT_SINGLE_RELEASE_SENSITIVITY_LIMITED"


def extract_oracle(path: Path | None) -> Dict[str, Any] | None:
    if path is None:
        return None
    value = read_json(path)
    primary = value.get("primary_endpoint", {})
    tasks = value.get("learned_embedding_task_results", [])
    return {
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "dataset_role": value.get("dataset_role"),
        "pair_count": value.get("pair_count"),
        "overall_original_mmd2": primary.get("original_mmd2"),
        "overall_original_monte_carlo_p": primary.get("original_monte_carlo_p"),
        "task_results": [
            {
                "task": row.get("task"),
                "mmd2": row.get("mmd2"),
                "holm_p": row.get("holm_p_within_pretreatment_tasks"),
                "reject_holm_0_05": row.get("reject_holm_0_05"),
            }
            for row in tasks
            if row.get("representation") == "learned_embedding"
        ],
        "interpretation": "REFERENCE_ONLY_PAIRED_ESTIMAND_NOT_AN_UNPAIRED_THRESHOLD_OR_P_VALUE",
    }


def write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    detection: pd.DataFrame,
    operating: pd.DataFrame,
) -> None:
    overall = detection.loc[detection["scope"] == "overall"].copy()
    lines = [
        "# Stage 6E public field-release emulation",
        "",
        f"- status: `{summary['status']}`",
        f"- independent log clusters: `{summary['input_audit']['cluster_count']}`",
        f"- paired scenarios available only as oracle: `{summary['input_audit']['pair_count']}`",
        f"- A/A threshold alpha: `{summary['alpha']}`",
        f"- descriptive evidence conclusion: `{summary['descriptive_evidence_conclusion']}`",
        "",
        "## Overall empirical rates",
        "",
        "| experiment | role | valid/total | threshold | median standardized MMD² | exceedance rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in overall.to_dict("records"):
        lines.append(
            f"| {row['experiment_set']} | {row['family']} | {row['valid_trials']}/{row['total_trials']} | "
            f"{row['threshold']:.6g} | {row['standardized_mmd2_median']:.6g} | {row['exceedance_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Pooled operating characteristics",
        "",
        "| scope | role | A/A false-positive rate (95% Wilson CI) | A/B detection rate (95% Wilson CI) | rate difference |",
        "| --- | --- | --- | --- | ---: |",
    ])
    for row in operating.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['scope_role']} | {row['aa_false_positive_rate']:.3f} "
            f"[{row['aa_false_positive_wilson95_low']:.3f}, {row['aa_false_positive_wilson95_high']:.3f}] | "
            f"{row['ab_detection_rate']:.3f} [{row['ab_detection_wilson95_low']:.3f}, "
            f"{row['ab_detection_wilson95_high']:.3f}] | "
            f"{row['detection_minus_false_positive_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The A/A threshold is an empirical public-data calibration for this frozen embedding, planner family, scenario pool, split design, and sample size. It is not a universal MMD cutoff and cannot be transferred directly to company road tests. A/B exceedance estimates whether known planner-style changes remain detectable after discarding scenario pairing and enforcing log-disjoint releases.",
        "",
        "Repeated trials reuse the same finite public scenario pool. Calibration and evaluation use separate seed streams, but they are not independent real-world collections. Task rows are diagnostic and have no multiplicity control. This experiment supports a public field-release emulation claim, not a claim of validation on proprietary software releases, safety, causality, or planner superiority.",
    ])
    (output_dir / "stage6e_release_emulation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = validate_config(read_json(args.config_json))
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    embedding_mmap = np.load(args.embedding_path, mmap_mode="r")
    if embedding_mmap.ndim != 2:
        raise ValueError(f"embedding must be 2D, observed={embedding_mmap.shape}")
    metadata = validate_paired_metadata(
        pd.read_csv(args.metadata_csv), config, int(embedding_mmap.shape[0])
    )
    selected_rows = metadata[str(config["row_id_column"])].to_numpy(dtype=np.int64)
    selected_embeddings = np.asarray(embedding_mmap[selected_rows], dtype=np.float64)
    if not np.isfinite(selected_embeddings).all():
        raise ValueError("selected embeddings contain non-finite values")
    embeddings_all = embedding_mmap
    inventory = build_pair_inventory(metadata, config)
    bandwidths = fixed_scope_bandwidths(metadata, embeddings_all, config)
    args.output_dir.mkdir(parents=True)

    specs = experiment_specs(config)
    total_trials = sum(int(spec["trials"]) for spec in specs)
    try:
        from tqdm import tqdm

        progress = tqdm(total=total_trials, desc="Stage 6E pseudo releases")
    except Exception:
        progress = None
    trial_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    base_seed = int(config["seed"])
    cluster_col = str(config["cluster_column"])
    for spec in specs:
        for repetition in range(int(spec["trials"])):
            split_seed = base_seed + int(spec["seed_offset"]) + repetition
            rng = np.random.default_rng(split_seed)
            logs_a, logs_b, score = choose_log_split(
                inventory,
                cluster_col,
                rng,
                int(config["split_search_candidates"]),
            )
            trial, trial_embeddings, audit = build_trial(
                metadata,
                embeddings_all,
                config,
                logs_a,
                logs_b,
                str(spec["planner_A"]),
                str(spec["planner_B"]),
            )
            identity = {
                "experiment_set": str(spec["experiment_set"]),
                "family": str(spec["family"]),
                "repetition": repetition,
                "split_seed": split_seed,
                "planner_A": str(spec["planner_A"]),
                "planner_B": str(spec["planner_B"]),
                "split_balance_score": score,
            }
            audit_rows.append({**identity, **audit})
            assignment_rows.extend(
                {**identity, "release_group": "A", "log_name": log_name} for log_name in logs_a
            )
            assignment_rows.extend(
                {**identity, "release_group": "B", "log_name": log_name} for log_name in logs_b
            )
            for scope_row in evaluate_trial_scopes(
                trial,
                trial_embeddings,
                config,
                bandwidths,
                seed=split_seed + 10_000_000,
            ):
                trial_rows.append({**identity, **audit, **scope_row})
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    trials = pd.DataFrame(trial_rows)
    thresholds, evaluated, detection = summarize_trials(trials, config)
    operating = operating_characteristics(evaluated)
    threshold_pass = bool((thresholds["status"] == "PASS").all())
    detection_pass = bool((detection["status"] == "PASS").all())
    status = PASS_STATUS if threshold_pass and detection_pass else INSUFFICIENT_STATUS
    evaluated.to_csv(args.output_dir / "release_trial_bdd.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.output_dir / "release_trial_split_audit.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(args.output_dir / "release_trial_log_assignments.csv", index=False)
    thresholds.to_csv(args.output_dir / "aa_empirical_thresholds.csv", index=False)
    detection.to_csv(args.output_dir / "release_detection_summary.csv", index=False)
    operating.to_csv(args.output_dir / "release_operating_characteristics.csv", index=False)
    pd.DataFrame(
        [{"scope": scope, "fixed_bandwidth": bandwidth} for scope, bandwidth in bandwidths.items()]
    ).to_csv(args.output_dir / "fixed_scope_bandwidths.csv", index=False)

    overall_detection = detection.loc[detection["scope"] == "overall"].to_dict("records")
    evidence_conclusion = descriptive_overall_conclusion(operating)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_role": "PUBLIC_LOG_DISJOINT_FIELD_RELEASE_EMULATION",
        "interpretation_role": "AA_CALIBRATED_UNPAIRED_DETECTION_NOT_PROPRIETARY_DEPLOYMENT_VALIDATION",
        "alpha": float(config["alpha"]),
        "config": {key: value for key, value in config.items() if key != "stage6d_design"},
        "stage6d_design": config["stage6d_design"],
        "input_audit": {
            "row_count": int(len(metadata)),
            "pair_count": int(metadata[str(config["pair_id_column"])].nunique()),
            "cluster_count": int(metadata[str(config["cluster_column"])].nunique()),
            "planner_counts": metadata[str(config["planner_column"])].value_counts().to_dict(),
            "all_trial_log_overlap_zero": bool((pd.DataFrame(audit_rows)["log_overlap_count"] == 0).all()),
            "all_trial_scenario_overlap_zero": bool((pd.DataFrame(audit_rows)["scenario_overlap_count"] == 0).all()),
        },
        "fixed_bandwidths": bandwidths,
        "thresholds": thresholds.to_dict("records"),
        "overall_detection": overall_detection,
        "operating_characteristics": operating.to_dict("records"),
        "descriptive_evidence_conclusion": evidence_conclusion,
        "descriptive_evidence_conclusion_rule": (
            "Wilson intervals separated; strong single-release sensitivity additionally requires detection rate >=0.8. "
            "This label is descriptive and not a preregistered hypothesis test."
        ),
        "paired_oracle": extract_oracle(args.paired_oracle_json),
        "limitations": [
            "Repeated trials reuse one finite public scenario pool; separate seed streams do not create independent real-world collections.",
            "Thresholds are specific to this embedding, sample size, planner family, ODD design, and split protocol.",
            "Public field-release emulation is not validation on proprietary company software releases.",
            "BDD detects behavior-distribution difference and is not a safety, causality, or planner-superiority metric.",
            "Company A/A data must recalibrate thresholds before operational deployment.",
            "Task-specific exceedance rates are diagnostic and are not multiplicity controlled.",
        ],
    }
    write_json(args.output_dir / "stage6e_release_emulation_summary.json", summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "embedding": {"path": str(args.embedding_path.resolve()), "sha256": sha256_file(args.embedding_path)},
            "metadata": {"path": str(args.metadata_csv.resolve()), "sha256": sha256_file(args.metadata_csv)},
            "config": {"path": str(args.config_json.resolve()), "sha256": sha256_file(args.config_json)},
            "paired_oracle": (
                {"path": str(args.paired_oracle_json.resolve()), "sha256": sha256_file(args.paired_oracle_json)}
                if args.paired_oracle_json is not None
                else None
            ),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(args.output_dir / "stage6e_reproducibility_provenance.json", provenance)
    write_report(args.output_dir, summary, detection, operating)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate log-disjoint unpaired version BDD with repeated same-version A/A pseudo releases."
    )
    parser.add_argument("--embedding_path", type=Path, required=True, help="Aligned flat 2D embedding .npy.")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Paired planner metadata used only to emulate releases.")
    parser.add_argument("--config_json", type=Path, required=True, help="Frozen Stage 6E calibration and split design.")
    parser.add_argument("--output_dir", type=Path, required=True, help="New output directory; overwrite is forbidden.")
    parser.add_argument(
        "--paired_oracle_json",
        type=Path,
        default=None,
        help="Optional frozen paired confirmation summary; read as reference only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
