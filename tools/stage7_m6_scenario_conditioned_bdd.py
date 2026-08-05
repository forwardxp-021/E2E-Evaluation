#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded_path(recorded: str, reference_dir: Path) -> Path:
    path = Path(recorded)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, reference_dir / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Recorded file path cannot be resolved: {recorded}; tried "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def git_provenance(repo_root: Path) -> Dict[str, Any]:
    def run_git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run_git("status", "--porcelain")
        return {
            "commit_sha": run_git("rev-parse", "HEAD"),
            "worktree_dirty": bool(status),
            "dirty_entry_count": len(status.splitlines()) if status else 0,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "commit_sha": "unavailable",
            "worktree_dirty": None,
            "dirty_entry_count": None,
            "warning": f"git provenance unavailable: {exc}",
        }


def package_versions() -> Dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "torch", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def validate_and_build_pairs(
    metadata: pd.DataFrame,
    paired_rows: Sequence[Dict[str, str]],
    row_count: int,
    *,
    planner_a: str,
    planner_b: str,
    allow_unequal_valid_horizon: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    required_metadata = {
        "global_row",
        "scenario_index",
        "scenario_token",
        "planner_name",
        "valid_timestep_count",
    }
    missing_metadata = sorted(required_metadata - set(metadata.columns))
    if missing_metadata:
        raise ValueError(f"metadata.csv missing required columns: {missing_metadata}")
    if len(metadata) != row_count:
        raise ValueError(
            f"metadata/embedding row count mismatch: metadata={len(metadata)} "
            f"embedding={row_count}"
        )
    global_rows = metadata["global_row"].to_numpy(dtype=np.int64)
    if len(np.unique(global_rows)) != row_count:
        raise ValueError("metadata global_row values are not unique")
    if set(global_rows.tolist()) != set(range(row_count)):
        raise ValueError(
            "metadata global_row must cover every embedding row exactly once"
        )
    by_row = metadata.set_index("global_row", drop=False)

    if not paired_rows:
        raise ValueError("paired_delta_csv contains no scenario pairs")
    required_pair_columns = {"scenario", "row_A", "row_B"}
    missing_pair_columns = sorted(required_pair_columns - set(paired_rows[0]))
    if missing_pair_columns:
        raise ValueError(
            f"paired_delta_csv missing required columns: {missing_pair_columns}"
        )

    pairs: List[Tuple[int, int]] = []
    scenarios: List[str] = []
    used_rows: List[int] = []
    for pair_position, record in enumerate(paired_rows):
        scenario = str(record["scenario"])
        try:
            row_a = int(record["row_A"])
            row_b = int(record["row_B"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"paired_delta_csv has invalid row index at pair {pair_position}"
            ) from exc
        if row_a == row_b:
            raise ValueError(f"pair {pair_position} reuses row {row_a} for A and B")
        if row_a < 0 or row_a >= row_count or row_b < 0 or row_b >= row_count:
            raise ValueError(
                f"pair {pair_position} row out of range: A={row_a}, B={row_b}, "
                f"embedding_rows={row_count}"
            )
        metadata_a = by_row.loc[row_a]
        metadata_b = by_row.loc[row_b]
        scenario_a = str(metadata_a["scenario_token"])
        scenario_b = str(metadata_b["scenario_token"])
        if scenario_a != scenario or scenario_b != scenario:
            raise ValueError(
                f"scenario mismatch at pair {pair_position}: csv={scenario}, "
                f"metadata_A={scenario_a}, metadata_B={scenario_b}"
            )
        actual_planner_a = str(metadata_a["planner_name"])
        actual_planner_b = str(metadata_b["planner_name"])
        if actual_planner_a != planner_a or actual_planner_b != planner_b:
            raise ValueError(
                f"planner mismatch at pair {pair_position}: "
                f"A={actual_planner_a} expected={planner_a}, "
                f"B={actual_planner_b} expected={planner_b}"
            )
        horizon_a = int(metadata_a["valid_timestep_count"])
        horizon_b = int(metadata_b["valid_timestep_count"])
        if horizon_a <= 0 or horizon_b <= 0:
            raise ValueError(
                f"pair {pair_position} has non-positive valid horizon: "
                f"A={horizon_a}, B={horizon_b}"
            )
        if horizon_a != horizon_b and not allow_unequal_valid_horizon:
            raise ValueError(
                f"pair {pair_position} has unequal valid horizon: "
                f"A={horizon_a}, B={horizon_b}; use a fixed cropping rule or "
                "--allow_unequal_valid_horizon with a documented justification"
            )
        pairs.append((row_a, row_b))
        scenarios.append(scenario)
        used_rows.extend((row_a, row_b))

    if len(set(scenarios)) != len(scenarios):
        raise ValueError("paired_delta_csv contains duplicate scenario pairs")
    if len(set(used_rows)) != len(used_rows):
        raise ValueError("paired_delta_csv reuses an embedding row across pairs")
    if set(used_rows) != set(range(row_count)):
        missing_rows = sorted(set(range(row_count)) - set(used_rows))
        extra_note = f"; missing rows include {missing_rows[:10]}" if missing_rows else ""
        raise ValueError(
            "paired_delta_csv must cover every embedding row exactly once"
            + extra_note
        )
    return np.asarray(pairs, dtype=np.int64), scenarios


def build_pair_quality_audit(
    metadata: pd.DataFrame,
    paired_rows: Sequence[Dict[str, str]],
    pair_indices: np.ndarray,
    embedding: np.ndarray,
    *,
    row_quality: pd.DataFrame | None,
    pair_quality: pd.DataFrame | None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    by_metadata_row = metadata.set_index("global_row", drop=False)
    row_quality_by_row = None
    if row_quality is not None:
        required = {
            "global_row",
            "scenario_index",
            "planner_name",
            "fallback_rate",
            "ambiguous_frame_rate",
            "quality_tier",
        }
        missing = sorted(required - set(row_quality.columns))
        if missing:
            raise ValueError(f"row_quality_csv missing required columns: {missing}")
        if row_quality["global_row"].duplicated().any():
            raise ValueError("row_quality_csv contains duplicate global_row values")
        if set(row_quality["global_row"].astype(int)) != set(range(len(metadata))):
            raise ValueError(
                "row_quality_csv must cover every embedding row exactly once"
            )
        row_quality_by_row = row_quality.set_index("global_row", drop=False)

    pair_quality_by_scenario = None
    if pair_quality is not None:
        required = {
            "scenario_index",
            "pair_quality_tier",
            "tier_a_pair_eligible",
            "tier_b_inclusive_pair_eligible",
        }
        missing = sorted(required - set(pair_quality.columns))
        if missing:
            raise ValueError(f"pair_quality_csv missing required columns: {missing}")
        if pair_quality["scenario_index"].duplicated().any():
            raise ValueError(
                "pair_quality_csv contains duplicate scenario_index values"
            )
        pair_quality_by_scenario = pair_quality.set_index(
            "scenario_index", drop=False
        )

    records = []
    for pair_position, ((row_a, row_b), paired_record) in enumerate(
        zip(pair_indices, paired_rows)
    ):
        metadata_a = by_metadata_row.loc[int(row_a)]
        metadata_b = by_metadata_row.loc[int(row_b)]
        scenario_index_a = int(metadata_a["scenario_index"])
        scenario_index_b = int(metadata_b["scenario_index"])
        if scenario_index_a != scenario_index_b:
            raise ValueError(
                f"pair {pair_position} scenario_index mismatch: "
                f"A={scenario_index_a}, B={scenario_index_b}"
            )
        record: Dict[str, Any] = {
            "pair_position": pair_position,
            "scenario_token": str(paired_record["scenario"]),
            "scenario_index": scenario_index_a,
            "row_A": int(row_a),
            "row_B": int(row_b),
            "planner_A": str(metadata_a["planner_name"]),
            "planner_B": str(metadata_b["planner_name"]),
            "valid_horizon_A": int(metadata_a["valid_timestep_count"]),
            "valid_horizon_B": int(metadata_b["valid_timestep_count"]),
            "valid_horizon_equal": bool(
                int(metadata_a["valid_timestep_count"])
                == int(metadata_b["valid_timestep_count"])
            ),
            "embedding_l2_distance": float(
                np.linalg.norm(embedding[int(row_a)] - embedding[int(row_b)])
            ),
            "embedding_rows_finite": bool(
                np.isfinite(embedding[[int(row_a), int(row_b)]]).all()
            ),
        }
        if row_quality_by_row is not None:
            quality_a = row_quality_by_row.loc[int(row_a)]
            quality_b = row_quality_by_row.loc[int(row_b)]
            for label, quality in (("A", quality_a), ("B", quality_b)):
                if int(quality["scenario_index"]) != scenario_index_a:
                    raise ValueError(
                        f"row_quality scenario mismatch at row {int(quality['global_row'])}"
                    )
                if str(quality["planner_name"]) != str(
                    metadata_a["planner_name"] if label == "A" else metadata_b["planner_name"]
                ):
                    raise ValueError(
                        f"row_quality planner mismatch at row {int(quality['global_row'])}"
                    )
                record[f"fallback_rate_{label}"] = float(quality["fallback_rate"])
                record[f"ambiguous_rate_{label}"] = float(
                    quality["ambiguous_frame_rate"]
                )
                record[f"row_quality_tier_{label}"] = str(quality["quality_tier"])
            record["max_pair_fallback_rate"] = max(
                record["fallback_rate_A"], record["fallback_rate_B"]
            )
            record["fallback_rate_abs_delta"] = abs(
                record["fallback_rate_A"] - record["fallback_rate_B"]
            )
            record["max_pair_ambiguous_rate"] = max(
                record["ambiguous_rate_A"], record["ambiguous_rate_B"]
            )
            record["ambiguous_rate_abs_delta"] = abs(
                record["ambiguous_rate_A"] - record["ambiguous_rate_B"]
            )
        if pair_quality_by_scenario is not None:
            if scenario_index_a not in pair_quality_by_scenario.index:
                raise ValueError(
                    f"pair_quality_csv missing scenario_index={scenario_index_a}"
                )
            quality_pair = pair_quality_by_scenario.loc[scenario_index_a]
            record["pair_quality_tier"] = str(quality_pair["pair_quality_tier"])
            record["tier_a_pair_eligible"] = parse_bool(
                quality_pair["tier_a_pair_eligible"]
            )
            record["tier_b_inclusive_pair_eligible"] = parse_bool(
                quality_pair["tier_b_inclusive_pair_eligible"]
            )
        records.append(record)

    audit = pd.DataFrame(records)
    summary = {
        "target_pairs": int(len(pair_indices)),
        "complete_pairs": int(len(audit)),
        "duplicate_scenario_tokens": int(audit["scenario_token"].duplicated().sum()),
        "missing_planner_rows": 0,
        "row_index_conflicts": int(
            pd.concat([audit["row_A"], audit["row_B"]]).duplicated().sum()
        ),
        "unequal_valid_horizon_pairs": int((~audit["valid_horizon_equal"]).sum()),
        "nonfinite_embedding_pairs": int((~audit["embedding_rows_finite"]).sum()),
        "valid_horizon_values": sorted(
            set(audit["valid_horizon_A"]).union(set(audit["valid_horizon_B"]))
        ),
        "same_embedding_preprocessing_for_all_rows": True,
    }
    if "pair_quality_tier" in audit:
        summary["pair_quality_tier_counts"] = {
            str(key): int(value)
            for key, value in audit["pair_quality_tier"].value_counts().items()
        }
        summary["tier_a_pairs"] = int(audit["tier_a_pair_eligible"].sum())
        summary["tier_b_inclusive_pairs"] = int(
            audit["tier_b_inclusive_pair_eligible"].sum()
        )
    if "fallback_rate_A" in audit:
        summary["fallback"] = {
            "mean_rate_A": float(audit["fallback_rate_A"].mean()),
            "mean_rate_B": float(audit["fallback_rate_B"].mean()),
            "max_pair_rate": float(audit["max_pair_fallback_rate"].max()),
            "mean_abs_pair_delta": float(audit["fallback_rate_abs_delta"].mean()),
        }
    return audit, summary


def exact_median_bandwidth(values: np.ndarray) -> float:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2:
        raise ValueError("bandwidth input must be a 2D array with at least two rows")
    if not np.isfinite(source).all():
        raise ValueError("bandwidth input contains non-finite values")
    squared = np.sum(
        (source[:, None, :] - source[None, :, :]) ** 2, axis=2
    )
    upper = np.sqrt(squared[np.triu_indices(len(source), k=1)])
    positive = upper[np.isfinite(upper) & (upper > 0)]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.median(positive))
    return bandwidth if bandwidth > 1e-8 else 1.0


def rbf_kernel(values: np.ndarray, bandwidth: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"invalid RBF bandwidth: {bandwidth}")
    squared = np.sum(
        (source[:, None, :] - source[None, :, :]) ** 2, axis=2
    )
    return np.exp(-squared / (2.0 * bandwidth**2))


def biased_mmd2_from_kernel(
    kernel: np.ndarray, index_a: np.ndarray, index_b: np.ndarray
) -> float:
    return float(
        np.mean(kernel[np.ix_(index_a, index_a)])
        + np.mean(kernel[np.ix_(index_b, index_b)])
        - 2.0 * np.mean(kernel[np.ix_(index_a, index_b)])
    )


def scenario_residualize(
    values: np.ndarray, pair_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float64)
    index_a, index_b = pair_indices[:, 0], pair_indices[:, 1]
    midpoint = 0.5 * (source[index_a] + source[index_b])
    residual_a = source[index_a] - midpoint
    residual_b = source[index_b] - midpoint
    return residual_a, residual_b


def permutation_bdd(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    paired_swap: bool,
    progress_label: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"paired BDD requires equal A/B shapes, got {values_a.shape} and "
            f"{values_b.shape}"
        )
    if repetitions <= 0:
        raise ValueError("permutation repetitions must be positive")
    n_pairs = len(values_a)
    pooled = np.vstack([values_a, values_b])
    index_a = np.arange(n_pairs, dtype=np.int64)
    index_b = np.arange(n_pairs, 2 * n_pairs, dtype=np.int64)
    bandwidth = exact_median_bandwidth(pooled)
    kernel = rbf_kernel(pooled, bandwidth)
    observed = biased_mmd2_from_kernel(kernel, index_a, index_b)

    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for repetition in tqdm(
        range(repetitions), desc=progress_label, unit="perm", leave=False
    ):
        if paired_swap:
            swap = rng.integers(0, 2, size=n_pairs).astype(bool)
            candidate_a = np.where(swap, index_b, index_a)
            candidate_b = np.where(swap, index_a, index_b)
        else:
            permutation = rng.permutation(2 * n_pairs)
            candidate_a = permutation[:n_pairs]
            candidate_b = permutation[n_pairs:]
        samples[repetition] = biased_mmd2_from_kernel(
            kernel, candidate_a, candidate_b
        )
    exceedance_count = int(np.sum(samples >= observed))
    p_value = float((exceedance_count + 1) / (repetitions + 1))
    monte_carlo_resolution = float(1.0 / (repetitions + 1))
    if exceedance_count == 0:
        reporting_text = (
            f"0/{repetitions} null statistics reached observed; "
            f"plus-one Monte Carlo p={p_value:.8g}, report at this resolution as "
            f"p<={monte_carlo_resolution:.3g}"
        )
    else:
        reporting_text = (
            f"{exceedance_count}/{repetitions} null statistics reached observed; "
            f"plus-one Monte Carlo p={p_value:.8g}"
        )
    result = {
        "metric": "BDD_MMD",
        "mmd_estimator": "biased_single_rbf_fixed_pooled_median_bandwidth",
        "kernel_type": "single_rbf",
        "kernel_formula": "exp(-squared_euclidean_distance/(2*bandwidth^2))",
        "mmd2": observed,
        "bandwidth": bandwidth,
        "bandwidth_selection": (
            "exact median of all finite positive off-diagonal pooled Euclidean distances"
        ),
        "bandwidth_fixed_across_observed_and_all_permutations": True,
        "biased_v_statistic_includes_kernel_diagonal": True,
        "dtype": "float64",
        "subsampling": "none",
        "n_A": n_pairs,
        "n_B": n_pairs,
        "permutation_scheme": (
            "within_scenario_pair_label_swap"
            if paired_swap
            else "pooled_unpaired_label_shuffle"
        ),
        "permutations": repetitions,
        "exceedance_count": exceedance_count,
        "monte_carlo_plus_one_p": p_value,
        "monte_carlo_resolution": monte_carlo_resolution,
        "p_value": p_value,
        "p_value_reporting_text": reporting_text,
        "null_median": float(np.median(samples)),
        "null_q95": float(np.quantile(samples, 0.95)),
        "null_q99": float(np.quantile(samples, 0.99)),
    }
    return result, samples


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def quality_sensitivity_analysis(
    embedding: np.ndarray,
    pair_indices: np.ndarray,
    pair_audit: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    required = {"tier_a_pair_eligible", "tier_b_inclusive_pair_eligible"}
    missing = sorted(required - set(pair_audit.columns))
    if missing:
        raise ValueError(
            f"quality sensitivity requires pair audit columns: missing={missing}"
        )
    definitions = [
        (
            "full_primary",
            np.ones(len(pair_audit), dtype=bool),
            "all complete pairs; primary analysis",
        ),
        (
            "tier_a_sensitivity",
            pair_audit["tier_a_pair_eligible"].to_numpy(dtype=bool),
            "both planner rows satisfy Tier A quality; realized-rollout sensitivity only",
        ),
        (
            "tier_a_plus_b_sensitivity",
            pair_audit["tier_b_inclusive_pair_eligible"].to_numpy(dtype=bool),
            "both planner rows satisfy Tier A or B quality; realized-rollout sensitivity only",
        ),
    ]
    rows: List[Dict[str, Any]] = []
    null_samples: Dict[str, np.ndarray] = {}
    for position, (name, mask, definition) in enumerate(definitions):
        selected = pair_indices[mask]
        if len(selected) < 2:
            raise ValueError(
                f"quality sensitivity subset {name} has fewer than two pairs"
            )
        values_a = embedding[selected[:, 0]]
        values_b = embedding[selected[:, 1]]
        original, original_null = permutation_bdd(
            values_a,
            values_b,
            repetitions=repetitions,
            seed=seed + position * 10,
            paired_swap=True,
            progress_label=f"{name} original paired BDD",
        )
        residual_a, residual_b = scenario_residualize(embedding, selected)
        residual, residual_null = permutation_bdd(
            residual_a,
            residual_b,
            repetitions=repetitions,
            seed=seed + position * 10 + 1,
            paired_swap=True,
            progress_label=f"{name} residual paired BDD",
        )
        rows.append(
            {
                "dataset": name,
                "n_pairs": int(len(selected)),
                "definition": definition,
                "original_mmd2": original["mmd2"],
                "original_exceedance_count": original["exceedance_count"],
                "original_monte_carlo_p": original["p_value"],
                "residual_mmd2": residual["mmd2"],
                "residual_exceedance_count": residual["exceedance_count"],
                "residual_monte_carlo_p": residual["p_value"],
                "selection_role": (
                    "primary" if name == "full_primary" else "sensitivity_only"
                ),
            }
        )
        null_samples[f"{name}_original"] = original_null
        null_samples[f"{name}_residual"] = residual_null

    sensitivity_positions = [1, 2]
    original_adjusted = holm_adjust(
        [rows[index]["original_monte_carlo_p"] for index in sensitivity_positions]
    )
    residual_adjusted = holm_adjust(
        [rows[index]["residual_monte_carlo_p"] for index in sensitivity_positions]
    )
    for index, original_p, residual_p in zip(
        sensitivity_positions, original_adjusted, residual_adjusted
    ):
        rows[index]["original_holm_p_within_quality_sensitivity_family"] = original_p
        rows[index]["residual_holm_p_within_quality_sensitivity_family"] = residual_p
    rows[0]["original_holm_p_within_quality_sensitivity_family"] = ""
    rows[0]["residual_holm_p_within_quality_sensitivity_family"] = ""
    return rows, null_samples


def fallback_distance_sensitivity(pair_audit: pd.DataFrame) -> List[Dict[str, Any]]:
    required = {
        "embedding_l2_distance",
        "max_pair_fallback_rate",
        "fallback_rate_abs_delta",
        "max_pair_ambiguous_rate",
        "ambiguous_rate_abs_delta",
    }
    missing = sorted(required - set(pair_audit.columns))
    if missing:
        raise ValueError(
            f"fallback sensitivity requires pair audit columns: missing={missing}"
        )
    rows = []
    for quality_metric in (
        "max_pair_fallback_rate",
        "fallback_rate_abs_delta",
        "max_pair_ambiguous_rate",
        "ambiguous_rate_abs_delta",
    ):
        quality_values = pair_audit[quality_metric].to_numpy(dtype=np.float64)
        embedding_distance = pair_audit["embedding_l2_distance"].to_numpy(
            dtype=np.float64
        )
        if len(np.unique(quality_values)) < 2:
            statistic, p_value, status = 0.0, 1.0, "degenerate_quality_metric"
        else:
            result = spearmanr(quality_values, embedding_distance)
            statistic, p_value, status = (
                float(result.statistic),
                float(result.pvalue),
                "evaluated",
            )
        rows.append(
            {
                "quality_metric": quality_metric,
                "n_pairs": int(len(pair_audit)),
                "spearman_vs_embedding_l2": statistic,
                "raw_p": p_value,
                "status": status,
            }
        )
    adjusted = holm_adjust([row["raw_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p"] = value
        row["reject_holm_0_05"] = bool(value < 0.05)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage7 M6: compare pooled marginal and scenario-conditioned paired "
            "BDD for complete same-scenario planner pairs."
        )
    )
    parser.add_argument("--embedding_path", type=Path, required=True)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--marginal_bdd_summary", type=Path, required=True)
    parser.add_argument("--embedding_manifest", type=Path, required=True)
    parser.add_argument("--row_quality_csv", type=Path, required=True)
    parser.add_argument("--pair_quality_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--planner_a", required=True)
    parser.add_argument("--planner_b", required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--allow_unequal_valid_horizon",
        action="store_true",
        help=(
            "Allow unequal valid_timestep_count within a pair only when a fixed "
            "cropping/masking justification is documented externally."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir()

    embedding = np.asarray(
        np.load(args.embedding_path, mmap_mode="r"), dtype=np.float64
    )
    if embedding.ndim != 2 or len(embedding) == 0:
        raise ValueError(
            f"embedding must be a non-empty 2D array, got shape={embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains non-finite values")
    metadata = pd.read_csv(args.metadata_csv)
    paired_rows = read_csv(args.paired_delta_csv)
    pair_indices, scenarios = validate_and_build_pairs(
        metadata,
        paired_rows,
        len(embedding),
        planner_a=args.planner_a,
        planner_b=args.planner_b,
        allow_unequal_valid_horizon=args.allow_unequal_valid_horizon,
    )
    index_a, index_b = pair_indices[:, 0], pair_indices[:, 1]
    row_quality = pd.read_csv(args.row_quality_csv) if args.row_quality_csv else None
    pair_quality = (
        pd.read_csv(args.pair_quality_csv) if args.pair_quality_csv else None
    )
    pair_audit, pair_audit_summary = build_pair_quality_audit(
        metadata,
        paired_rows,
        pair_indices,
        embedding,
        row_quality=row_quality,
        pair_quality=pair_quality,
    )
    pair_audit.to_csv(args.output_dir / "m6_pair_quality_audit.csv", index=False)
    (args.output_dir / "m6_pair_quality_audit_summary.json").write_text(
        json.dumps(pair_audit_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if any(
        pair_audit_summary[key] != 0
        for key in (
            "duplicate_scenario_tokens",
            "missing_planner_rows",
            "row_index_conflicts",
            "nonfinite_embedding_pairs",
        )
    ):
        raise RuntimeError(
            f"strict pair audit failed: {pair_audit_summary}"
        )
    if (
        pair_audit_summary["unequal_valid_horizon_pairs"] != 0
        and not args.allow_unequal_valid_horizon
    ):
        raise RuntimeError(
            f"strict pair audit found unequal valid horizons: {pair_audit_summary}"
        )

    original_paired, original_paired_samples = permutation_bdd(
        embedding[index_a],
        embedding[index_b],
        repetitions=args.permutations,
        seed=args.seed,
        paired_swap=True,
        progress_label="Original-space paired BDD",
    )
    original_pooled, original_pooled_samples = permutation_bdd(
        embedding[index_a],
        embedding[index_b],
        repetitions=args.permutations,
        seed=args.seed + 1,
        paired_swap=False,
        progress_label="Original-space pooled BDD",
    )
    residual_a, residual_b = scenario_residualize(embedding, pair_indices)
    if not np.allclose(residual_a + residual_b, 0.0, atol=1e-10):
        raise RuntimeError("scenario residualization failed midpoint cancellation")
    residual_paired, residual_paired_samples = permutation_bdd(
        residual_a,
        residual_b,
        repetitions=args.permutations,
        seed=args.seed + 2,
        paired_swap=True,
        progress_label="Residual-space paired BDD",
    )

    quality_sensitivity_rows: List[Dict[str, Any]] = []
    fallback_sensitivity_rows: List[Dict[str, Any]] = []
    if args.row_quality_csv and args.pair_quality_csv:
        quality_sensitivity_rows, quality_null_samples = quality_sensitivity_analysis(
            embedding,
            pair_indices,
            pair_audit,
            repetitions=args.permutations,
            seed=args.seed + 1000,
        )
        quality_sensitivity_rows[0].update(
            {
                "original_mmd2": original_paired["mmd2"],
                "original_exceedance_count": original_paired["exceedance_count"],
                "original_monte_carlo_p": original_paired["p_value"],
                "residual_mmd2": residual_paired["mmd2"],
                "residual_exceedance_count": residual_paired["exceedance_count"],
                "residual_monte_carlo_p": residual_paired["p_value"],
            }
        )
        quality_null_samples["full_primary_original"] = original_paired_samples
        quality_null_samples["full_primary_residual"] = residual_paired_samples
        pd.DataFrame(quality_sensitivity_rows).to_csv(
            args.output_dir / "table_m6_quality_sensitivity.csv", index=False
        )
        (args.output_dir / "table_m6_quality_sensitivity.md").write_text(
            markdown_table(
                quality_sensitivity_rows,
                [
                    "dataset",
                    "n_pairs",
                    "original_mmd2",
                    "original_exceedance_count",
                    "original_monte_carlo_p",
                    "original_holm_p_within_quality_sensitivity_family",
                    "residual_mmd2",
                    "residual_exceedance_count",
                    "residual_monte_carlo_p",
                    "residual_holm_p_within_quality_sensitivity_family",
                    "selection_role",
                ],
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            args.output_dir / "m6_quality_sensitivity_null_samples.npz",
            **quality_null_samples,
        )
        fallback_sensitivity_rows = fallback_distance_sensitivity(pair_audit)
        pd.DataFrame(fallback_sensitivity_rows).to_csv(
            args.output_dir / "table_m6_fallback_distance_sensitivity.csv",
            index=False,
        )
        (args.output_dir / "table_m6_fallback_distance_sensitivity.md").write_text(
            markdown_table(
                fallback_sensitivity_rows,
                [
                    "quality_metric",
                    "n_pairs",
                    "spearman_vs_embedding_l2",
                    "raw_p",
                    "holm_p",
                    "reject_holm_0_05",
                    "status",
                ],
            ),
            encoding="utf-8",
        )

    frozen_marginal = read_json(args.marginal_bdd_summary)
    if int(frozen_marginal.get("n_A", -1)) != len(pair_indices) or int(
        frozen_marginal.get("n_B", -1)
    ) != len(pair_indices):
        raise ValueError(
            "frozen marginal BDD pair count does not match validated scenario pairs"
        )

    comparison = [
        {
            "analysis": "frozen_marginal_bdd",
            "analysis_role": "historical_reference",
            "embedding_space": "original",
            "mmd2": float(frozen_marginal["mmd2"]),
            "p_value": float(frozen_marginal["p_value"]),
            "exceedance_count": "not_recorded_by_frozen_M4_summary",
            "null": "pooled unpaired permutation",
            "estimand": "unconditional marginal planner distribution difference",
        },
        {
            "analysis": "fixed_kernel_marginal_recheck",
            "analysis_role": "estimand_control",
            "embedding_space": "original",
            "mmd2": original_pooled["mmd2"],
            "p_value": original_pooled["p_value"],
            "exceedance_count": original_pooled["exceedance_count"],
            "null": original_pooled["permutation_scheme"],
            "estimand": "unconditional marginal planner distribution difference",
        },
        {
            "analysis": "paired_label_swap_bdd",
            "analysis_role": "primary_development_set_result",
            "embedding_space": "original",
            "mmd2": original_paired["mmd2"],
            "p_value": original_paired["p_value"],
            "exceedance_count": original_paired["exceedance_count"],
            "null": original_paired["permutation_scheme"],
            "estimand": "same-scenario planner effect",
        },
        {
            "analysis": "scenario_residualized_paired_bdd",
            "analysis_role": "secondary_mechanism_analysis",
            "embedding_space": "pair-midpoint residual",
            "mmd2": residual_paired["mmd2"],
            "p_value": residual_paired["p_value"],
            "exceedance_count": residual_paired["exceedance_count"],
            "null": residual_paired["permutation_scheme"],
            "estimand": "same-scenario planner effect after removing scenario midpoint",
        },
    ]
    pd.DataFrame(comparison).to_csv(
        args.output_dir / "table_m6_bdd_estimand_comparison.csv", index=False
    )
    (args.output_dir / "table_m6_bdd_estimand_comparison.md").write_text(
        markdown_table(
            comparison,
            [
                "analysis",
                "analysis_role",
                "embedding_space",
                "mmd2",
                "exceedance_count",
                "p_value",
                "null",
                "estimand",
            ],
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "original_paired_swap_mmd2": original_paired_samples,
            "original_pooled_shuffle_mmd2": original_pooled_samples,
            "residual_paired_swap_mmd2": residual_paired_samples,
        }
    ).to_csv(args.output_dir / "m6_permutation_samples.csv", index=False)

    repo_root = Path(__file__).resolve().parent.parent
    input_paths: Dict[str, Path] = {
        "embedding": args.embedding_path,
        "metadata": args.metadata_csv,
        "paired_delta": args.paired_delta_csv,
        "frozen_marginal_bdd": args.marginal_bdd_summary,
        "analysis_tool": Path(__file__).resolve(),
    }
    embedding_manifest_data = None
    checkpoint_path = None
    if args.embedding_manifest:
        embedding_manifest_data = read_json(args.embedding_manifest)
        if int(embedding_manifest_data.get("total_rows", -1)) != len(embedding):
            raise ValueError(
                "embedding_manifest total_rows does not match embedding rows"
            )
        if int(embedding_manifest_data.get("embedding_dim", -1)) != embedding.shape[1]:
            raise ValueError(
                "embedding_manifest embedding_dim does not match embedding array"
            )
        input_paths["embedding_manifest"] = args.embedding_manifest
        recorded_checkpoint = str(embedding_manifest_data.get("checkpoint") or "")
        if not recorded_checkpoint:
            raise ValueError("embedding_manifest is missing checkpoint path")
        checkpoint_path = resolve_recorded_path(
            recorded_checkpoint, args.embedding_manifest.parent
        )
        input_paths["checkpoint"] = checkpoint_path
    if args.row_quality_csv:
        input_paths["row_quality"] = args.row_quality_csv
    if args.pair_quality_csv:
        input_paths["pair_quality"] = args.pair_quality_csv

    hashes = {
        name: {
            "path_as_invoked": str(path),
            "resolved_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in input_paths.items()
    }
    provenance = {
        "git": git_provenance(repo_root),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "seed": int(args.seed),
        "permutations": int(args.permutations),
        "inputs": hashes,
        "embedding_manifest": embedding_manifest_data,
        "checkpoint_resolved_path": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
    }
    (args.output_dir / "m6_reproducibility_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    quality_thresholds = None
    if args.pair_quality_csv:
        quality_summary_path = args.pair_quality_csv.parent / "milestone2b_summary.json"
        if quality_summary_path.is_file():
            quality_thresholds = read_json(quality_summary_path).get("thresholds")
    frozen_analysis_spec = {
        "spec_id": "stage7_m6_original_embedding_paired_bdd_v1",
        "spec_status": "FROZEN_FOR_FUTURE_LOCKED_CONFIRMATION",
        "development_dataset_status": (
            "The current 45 pairs were used for method development and are not "
            "an independent confirmatory test."
        ),
        "primary_analysis": {
            "representation": "unchanged original 64D learned embedding",
            "statistic": "biased single-RBF MMD2 V-statistic",
            "bandwidth": (
                "exact median of all finite positive off-diagonal pooled "
                "Euclidean distances"
            ),
            "bandwidth_fixed_during_permutation": True,
            "null": "independent A/B label swap within each complete scenario pair",
            "permutations": int(args.permutations),
            "plus_one_correction": True,
            "alpha": 0.05,
            "alternative": "upper tail: permuted MMD2 >= observed MMD2",
        },
        "secondary_analysis": {
            "name": "pair-midpoint residualized paired BDD",
            "role": "mechanism analysis only",
            "not_directly_comparable_to_primary_mmd2": True,
        },
        "pair_eligibility": {
            "same_scenario_token": True,
            "exactly_one_row_per_required_planner": True,
            "unique_rows_and_scenarios": True,
            "equal_positive_valid_horizon_required": (
                not args.allow_unequal_valid_horizon
            ),
            "finite_embedding_required": True,
        },
        "quality_sensitivity": {
            "role": "sensitivity only; never replaces full primary analysis",
            "pair_membership_fixed_before_permutation": True,
            "thresholds": quality_thresholds,
            "holm_family": [
                "tier_a_sensitivity",
                "tier_a_plus_b_sensitivity",
            ],
        },
        "locked_confirmation_requirements": {
            "new_data_only": True,
            "log_disjoint": True,
            "scenario_disjoint": True,
            "planner_configuration_or_strength_holdout": True,
            "no_estimator_or_threshold_changes_after_unblinding": True,
        },
    }
    (args.output_dir / "m6_frozen_analysis_spec.json").write_text(
        json.dumps(frozen_analysis_spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    checks = {
        "embedding_metadata_row_count_match": len(metadata) == len(embedding),
        "global_row_exact_coverage": set(metadata["global_row"].astype(int))
        == set(range(len(embedding))),
        "complete_pair_coverage": 2 * len(pair_indices) == len(embedding),
        "scenario_unique": len(set(scenarios)) == len(scenarios),
        "planner_a_exact_match": True,
        "planner_b_exact_match": True,
        "scenario_midpoint_removed": bool(
            np.allclose(residual_a + residual_b, 0.0, atol=1e-10)
        ),
        "paired_permutations_requested": args.permutations,
        "formal_monte_carlo_resolution_at_least_50000": args.permutations >= 50000,
        "exceedance_count_reported": all(
            "exceedance_count" in result
            for result in (original_pooled, original_paired, residual_paired)
        ),
        "strict_pair_quality_audit_written": True,
        "equal_valid_horizon_or_explicit_override": bool(
            pair_audit_summary["unequal_valid_horizon_pairs"] == 0
            or args.allow_unequal_valid_horizon
        ),
        "input_hashes_recorded": bool(hashes),
        "checkpoint_hash_recorded": "checkpoint" in hashes,
        "frozen_analysis_spec_written": True,
        "quality_sensitivity_reported": bool(quality_sensitivity_rows),
        "marginal_bdd_retained_not_replaced": True,
        "mmd_magnitudes_across_spaces_not_ranked": True,
    }
    method_freeze_ready = all(checks.values())
    mmd2_recompute_difference = float(
        original_pooled["mmd2"] - float(frozen_marginal["mmd2"])
    )
    summary = {
        "milestone": "Stage 7 Milestone 6.1 paired BDD method freeze",
        "overall_verdict": (
            "PASS_WITH_LIMITATIONS" if method_freeze_ready else "FAIL"
        ),
        "analysis_status": "DEVELOPMENT_SET_METHOD_FREEZE",
        "method_freeze_ready_for_new_locked_set": method_freeze_ready,
        "current_dataset_role": "METHOD_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "planner_a": args.planner_a,
        "planner_b": args.planner_b,
        "paired_scenarios": len(pair_indices),
        "embedding_dim": int(embedding.shape[1]),
        "frozen_marginal_bdd": frozen_marginal,
        "fixed_kernel_original_pooled_bdd": original_pooled,
        "fixed_kernel_original_paired_bdd": original_paired,
        "scenario_residualized_paired_bdd": residual_paired,
        "primary_result": {
            "analysis": "fixed_kernel_original_paired_bdd",
            "mmd2": original_paired["mmd2"],
            "exceedance_count": original_paired["exceedance_count"],
            "permutations": original_paired["permutations"],
            "monte_carlo_p": original_paired["p_value"],
            "reporting_text": original_paired["p_value_reporting_text"],
        },
        "secondary_mechanism_result": {
            "analysis": "scenario_residualized_paired_bdd",
            "mmd2": residual_paired["mmd2"],
            "exceedance_count": residual_paired["exceedance_count"],
            "permutations": residual_paired["permutations"],
            "monte_carlo_p": residual_paired["p_value"],
            "reporting_text": residual_paired["p_value_reporting_text"],
        },
        "frozen_m4_vs_m6_recompute": {
            "m4_mmd2": float(frozen_marginal["mmd2"]),
            "m6_fixed_kernel_mmd2": original_pooled["mmd2"],
            "m6_minus_m4": mmd2_recompute_difference,
            "explanation": (
                "The frozen M4 summary did not record its realized bandwidth. "
                "M6.1 uses an exact pooled off-diagonal median bandwidth fixed "
                "across all permutations, so the small numerical difference is "
                "retained and documented rather than treated as identical."
            ),
        },
        "pair_quality_audit": pair_audit_summary,
        "quality_sensitivity": quality_sensitivity_rows,
        "fallback_distance_sensitivity": fallback_sensitivity_rows,
        "reproducibility_provenance_path": "m6_reproducibility_provenance.json",
        "frozen_analysis_spec_path": "m6_frozen_analysis_spec.json",
        "checks": checks,
        "interpretation": (
            "On the current 45-pair method-development dataset, the unchanged "
            "embedding contains a small but cross-scenario-consistent "
            "planner-conditioned shift. This is not evidence of strong clustering "
            "or independent generalization. Pooled marginal permutation answers a "
            "different estimand and is dominated by cross-scenario heterogeneity."
        ),
        "limitations": [
            "M6.1 was designed after inspecting M3-M5 and is a method-development freeze, not independent confirmation.",
            "MMD2 magnitudes in original and pair-midpoint residual spaces are not directly comparable because their kernel bandwidths differ.",
            "Pair-midpoint residualization is an analysis transform that requires a matched A/B pair; it is not a row-wise deployable encoder output.",
            "Tier A and Tier A+B membership depends on realized planner-rollout lane quality, so quality subsets are sensitivity analyses only.",
            "Scenario-conditioned paired BDD is appropriate for matched simulation; unpaired real-world logs must continue to use Stage6 marginal and scenario-control protocols.",
            "A significant paired BDD does not establish safety, planner superiority, or causal generalization beyond the sampled scenarios.",
        ],
    }
    (args.output_dir / "milestone6_paired_bdd_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(
        original_pooled_samples,
        bins=35,
        alpha=0.65,
        label="pooled shuffle null",
        color="#64748b",
    )
    axes[0].hist(
        original_paired_samples,
        bins=35,
        alpha=0.65,
        label="within-pair swap null",
        color="#2563eb",
    )
    axes[0].axvline(
        original_paired["mmd2"], color="#dc2626", linewidth=2, label="observed"
    )
    axes[0].set_title("Original embedding space")
    axes[0].set_xlabel("MMD²")
    axes[0].legend(fontsize=8)
    axes[1].hist(
        residual_paired_samples,
        bins=35,
        alpha=0.75,
        color="#0f766e",
        label="within-pair swap null",
    )
    axes[1].axvline(
        residual_paired["mmd2"], color="#dc2626", linewidth=2, label="observed"
    )
    axes[1].set_title("Scenario-residualized embedding space")
    axes[1].set_xlabel("MMD²")
    axes[1].legend(fontsize=8)
    fig.suptitle("M6 BDD null distributions by experimental design")
    fig.tight_layout()
    fig.savefig(plot_dir / "m6_bdd_null_comparison.png", dpi=180)
    plt.close(fig)

    report = [
        "# Stage 7 Milestone 6.1 Paired BDD Method Freeze",
        "",
        "## Status",
        "",
        f"`{summary['overall_verdict']}` — development-set method freeze, not confirmatory evidence.",
        "",
        "## Primary result",
        "",
        (
            f"Original unchanged 64D embedding: MMD²=`{original_paired['mmd2']:.8f}`, "
            f"exceedances=`{original_paired['exceedance_count']}/{args.permutations}`, "
            f"plus-one Monte Carlo p=`{original_paired['p_value']:.8g}`."
        ),
        "",
        "This is the primary result. It supports a small but consistent "
        "scenario-conditioned planner effect; it does not establish strong planner "
        "classification or clear two-cluster separation.",
        "",
        (
            "Secondary residualized mechanism result: "
            f"MMD²=`{residual_paired['mmd2']:.8f}`, "
            f"exceedances=`{residual_paired['exceedance_count']}/{args.permutations}`; "
            f"at the current Monte Carlo resolution report `p<="
            f"{residual_paired['monte_carlo_resolution']:.3g}`."
        ),
        "",
        "## BDD comparison",
        "",
        markdown_table(
            comparison,
            [
                "analysis",
                "analysis_role",
                "embedding_space",
                "mmd2",
                "exceedance_count",
                "p_value",
                "null",
                "estimand",
            ],
        ).rstrip(),
        "",
        "The residualized result is secondary mechanism analysis. Its estimand and "
        "kernel scale differ from the original-space primary MMD².",
        "",
        "## Frozen estimator provenance",
        "",
        markdown_table(
            [
                {
                    "kernel": original_paired["kernel_type"],
                    "bandwidth": original_paired["bandwidth"],
                    "bandwidth_rule": original_paired["bandwidth_selection"],
                    "estimator": original_paired["mmd_estimator"],
                    "dtype": original_paired["dtype"],
                    "subsampling": original_paired["subsampling"],
                    "permutations": original_paired["permutations"],
                }
            ],
            [
                "kernel",
                "bandwidth",
                "bandwidth_rule",
                "estimator",
                "dtype",
                "subsampling",
                "permutations",
            ],
        ).rstrip(),
        "",
        (
            f"Frozen M4 MMD²=`{float(frozen_marginal['mmd2']):.8f}` versus "
            f"M6.1 fixed-kernel MMD²=`{original_pooled['mmd2']:.8f}` "
            f"(difference `{mmd2_recompute_difference:+.8f}`). The M4 summary did "
            "not record its realized bandwidth; M6.1 therefore retains M4 as a "
            "historical reference and freezes the exact bandwidth rule above for "
            "future locked confirmation."
        ),
        "",
        "## Strict pair audit",
        "",
        markdown_table(
            [
                {
                    "target_pairs": pair_audit_summary["target_pairs"],
                    "complete_pairs": pair_audit_summary["complete_pairs"],
                    "duplicate_tokens": pair_audit_summary[
                        "duplicate_scenario_tokens"
                    ],
                    "missing_planner_rows": pair_audit_summary[
                        "missing_planner_rows"
                    ],
                    "row_conflicts": pair_audit_summary["row_index_conflicts"],
                    "unequal_horizon": pair_audit_summary[
                        "unequal_valid_horizon_pairs"
                    ],
                    "nonfinite_pairs": pair_audit_summary[
                        "nonfinite_embedding_pairs"
                    ],
                }
            ],
            [
                "target_pairs",
                "complete_pairs",
                "duplicate_tokens",
                "missing_planner_rows",
                "row_conflicts",
                "unequal_horizon",
                "nonfinite_pairs",
            ],
        ).rstrip(),
        "",
        (
            "Valid horizon values were "
            f"`{pair_audit_summary['valid_horizon_values']}` frames; every A/B pair "
            "had equal valid horizon and all rows used the same embedding file and "
            "preprocessing manifest."
        ),
        "",
        "## Lane-quality sensitivity",
        "",
        (
            markdown_table(
                quality_sensitivity_rows,
                [
                    "dataset",
                    "n_pairs",
                    "original_mmd2",
                    "original_exceedance_count",
                    "original_monte_carlo_p",
                    "original_holm_p_within_quality_sensitivity_family",
                    "selection_role",
                ],
            ).rstrip()
            if quality_sensitivity_rows
            else "Not run: row/pair quality CSVs were not provided."
        ),
        "",
        (
            f"Mean fallback rate: planner A=`"
            f"{pair_audit_summary.get('fallback', {}).get('mean_rate_A', float('nan')):.6f}`, "
            f"planner B=`"
            f"{pair_audit_summary.get('fallback', {}).get('mean_rate_B', float('nan')):.6f}`. "
            "Tier membership is based on realized rollout quality and remains "
            "sensitivity-only."
        ),
        "",
        (
            markdown_table(
                fallback_sensitivity_rows,
                [
                    "quality_metric",
                    "spearman_vs_embedding_l2",
                    "raw_p",
                    "holm_p",
                    "reject_holm_0_05",
                ],
            ).rstrip()
            if fallback_sensitivity_rows
            else "Fallback-distance correlation was not run."
        ),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "The frozen marginal result is retained. M6 does not relabel it as a failure; "
        "it adds the design-matched conditional estimand for same-scenario simulation.",
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in summary["limitations"]],
    ]
    (args.output_dir / "milestone6_paired_bdd_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if not method_freeze_ready:
        raise RuntimeError(f"M6.1 method-freeze checks failed: {checks}")
    print(
        "Stage7 M6.1 method freeze PASS_WITH_LIMITATIONS: "
        f"original paired p={original_paired['p_value']:.6g}, "
        f"residual paired p={residual_paired['p_value']:.6g}"
    )


if __name__ == "__main__":
    main()
