#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import stage6d_unpaired_version_bdd as stage6d
from tools import stage6e_calibrate_unpaired_release as stage6e


SCHEMA_VERSION = "stage6f_unpaired_power_curve_v1"
COMPLETE_STATUS = "POWER_CURVE_COMPLETE"
TARGET_REACHED = "TARGET_REACHED_WITH_AVAILABLE_PUBLIC_LOGS"
TARGET_NOT_REACHED = "TARGET_NOT_REACHED_WITH_AVAILABLE_PUBLIC_LOGS"
LOG_SPLIT_STRATEGIES = {
    "half_log_pool_prefix_v1",
    "sequential_full_log_pool_v1",
}


def validate_power_config(raw: Mapping[str, Any]) -> Dict[str, Any]:
    config = stage6e.validate_config(raw)
    sample_sizes = raw.get("sample_sizes_per_release")
    if not isinstance(sample_sizes, list) or not sample_sizes:
        raise ValueError("config.sample_sizes_per_release must be a non-empty list")
    sizes = [int(value) for value in sample_sizes]
    if any(value < 10 for value in sizes):
        raise ValueError("every sample size must be >=10")
    if sizes != sorted(set(sizes)):
        raise ValueError("sample sizes must be unique and strictly increasing")
    config["sample_sizes_per_release"] = sizes
    config.setdefault("target_detection_rate", 0.8)
    config.setdefault("target_false_positive_rate", float(config["alpha"]))
    config.setdefault("log_split_strategy", "half_log_pool_prefix_v1")
    if config["log_split_strategy"] not in LOG_SPLIT_STRATEGIES:
        raise ValueError(
            f"log_split_strategy must be one of {sorted(LOG_SPLIT_STRATEGIES)}"
        )
    if not 0 < float(config["target_detection_rate"]) <= 1:
        raise ValueError("target_detection_rate must be in (0,1]")
    if not 0 < float(config["target_false_positive_rate"]) < 0.5:
        raise ValueError("target_false_positive_rate must be in (0,0.5)")
    return config


def _select_log_prefix(
    pool: Sequence[str],
    scenario_counts: Mapping[str, int],
    target: int,
) -> Tuple[List[str], int]:
    selected: List[str] = []
    total = 0
    for position, log_name in enumerate(pool):
        count = int(scenario_counts[str(log_name)])
        if total >= target:
            break
        if total + count > target:
            remaining_need = target - total
            later_exact = next(
                (str(value) for value in pool[position + 1 :] if int(scenario_counts[str(value)]) == remaining_need),
                None,
            )
            if later_exact is not None:
                selected.append(later_exact)
                total += int(scenario_counts[later_exact])
                break
        selected.append(str(log_name))
        total += count
    return sorted(set(selected)), total


def _sample_size_split_score(
    inventory: pd.DataFrame,
    cluster_col: str,
    logs_a: Sequence[str],
    logs_b: Sequence[str],
    target: int,
) -> float:
    selected_a = inventory[cluster_col].astype(str).isin(set(logs_a))
    selected_b = inventory[cluster_col].astype(str).isin(set(logs_b))
    if not selected_a.any() or not selected_b.any():
        return math.inf
    counts_a = inventory.loc[selected_a, "_support_cell"].value_counts(normalize=True)
    counts_b = inventory.loc[selected_b, "_support_cell"].value_counts(normalize=True)
    cells = sorted(set(counts_a.index.astype(str)) | set(counts_b.index.astype(str)))
    composition = sum(
        abs(float(counts_a.get(cell, 0.0)) - float(counts_b.get(cell, 0.0))) for cell in cells
    )
    n_a = int(selected_a.sum())
    n_b = int(selected_b.sum())
    target_error = (abs(n_a - target) + abs(n_b - target)) / target
    group_imbalance = abs(n_a - n_b) / target
    return float(composition + target_error + group_imbalance)


def choose_disjoint_logs_for_size(
    inventory: pd.DataFrame,
    cluster_col: str,
    target: int,
    rng: np.random.Generator,
    search_candidates: int,
    strategy: str = "half_log_pool_prefix_v1",
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    logs = np.asarray(sorted(inventory[cluster_col].astype(str).unique()))
    scenario_counts = inventory.groupby(cluster_col).size().astype(int).to_dict()
    if int(sum(scenario_counts.values())) < 2 * target:
        raise ValueError(
            f"target requires at least {2 * target} scenarios, available={sum(scenario_counts.values())}"
        )
    best: Tuple[List[str], List[str], Dict[str, Any]] | None = None
    for _ in range(search_candidates):
        shuffled = rng.permutation(logs)
        if strategy == "half_log_pool_prefix_v1":
            half = len(shuffled) // 2
            pools = [shuffled[:half].tolist(), shuffled[half:].tolist()]
            logs_a, n_a = _select_log_prefix(pools[0], scenario_counts, target)
            logs_b, n_b = _select_log_prefix(pools[1], scenario_counts, target)
        elif strategy == "sequential_full_log_pool_v1":
            logs_a, n_a = _select_log_prefix(shuffled.tolist(), scenario_counts, target)
            remaining = [str(value) for value in shuffled if str(value) not in set(logs_a)]
            logs_b, n_b = _select_log_prefix(remaining, scenario_counts, target)
        else:
            raise ValueError(f"unsupported log split strategy: {strategy}")
        if abs(n_a - target) > 1 or abs(n_b - target) > 1:
            continue
        if set(logs_a) & set(logs_b):
            continue
        score = _sample_size_split_score(inventory, cluster_col, logs_a, logs_b, target)
        audit = {
            "target_scenarios_per_release": target,
            "selected_scenarios_A": n_a,
            "selected_scenarios_B": n_b,
            "selected_logs_A": len(logs_a),
            "selected_logs_B": len(logs_b),
            "split_balance_score": score,
            "log_split_strategy": strategy,
        }
        if best is None or score < best[2]["split_balance_score"]:
            best = (logs_a, logs_b, audit)
    if best is None:
        raise ValueError(
            f"could not construct two disjoint log sets within ±1 of target={target}; "
            f"increase split_search_candidates or add logs"
        )
    return best


def summarize_by_sample_size(
    trials: pd.DataFrame,
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_parts = []
    evaluated_parts = []
    detection_parts = []
    operating_parts = []
    for sample_size in config["sample_sizes_per_release"]:
        subset = trials.loc[trials["target_scenarios_per_release"] == sample_size].copy()
        thresholds, evaluated, detection = stage6e.summarize_trials(subset, config)
        thresholds.insert(0, "target_scenarios_per_release", sample_size)
        evaluated["target_scenarios_per_release"] = sample_size
        detection.insert(0, "target_scenarios_per_release", sample_size)
        operating = stage6e.operating_characteristics(evaluated)
        operating.insert(0, "target_scenarios_per_release", sample_size)
        actual = subset.loc[subset["scope"] == "overall"].groupby(
            ["experiment_set", "repetition"], sort=False
        )[["n_A", "n_B"]].first()
        operating["mean_actual_n_A"] = float(actual["n_A"].mean())
        operating["mean_actual_n_B"] = float(actual["n_B"].mean())
        threshold_parts.append(thresholds)
        evaluated_parts.append(evaluated)
        detection_parts.append(detection)
        operating_parts.append(operating)
    return (
        pd.concat(threshold_parts, ignore_index=True),
        pd.concat(evaluated_parts, ignore_index=True),
        pd.concat(detection_parts, ignore_index=True),
        pd.concat(operating_parts, ignore_index=True),
    )


def build_sufficiency_summary(operating: pd.DataFrame, config: Mapping[str, Any]) -> Dict[str, Any]:
    overall = operating.loc[operating["scope"] == "overall"].sort_values(
        "target_scenarios_per_release"
    )
    detection_target = float(config["target_detection_rate"])
    false_positive_target = float(config["target_false_positive_rate"])
    point_pass = overall.loc[
        (overall["ab_detection_rate"] >= detection_target)
        & (overall["aa_false_positive_rate"] <= false_positive_target)
    ]
    confidence_pass = overall.loc[
        (overall["ab_detection_wilson95_low"] >= detection_target)
        & (overall["aa_false_positive_wilson95_high"] <= false_positive_target)
    ]
    point_n = int(point_pass.iloc[0]["target_scenarios_per_release"]) if len(point_pass) else None
    confidence_n = (
        int(confidence_pass.iloc[0]["target_scenarios_per_release"]) if len(confidence_pass) else None
    )
    status = TARGET_REACHED if confidence_n is not None else TARGET_NOT_REACHED
    max_row = overall.iloc[-1]
    return {
        "status": status,
        "target_detection_rate": detection_target,
        "target_false_positive_rate": false_positive_target,
        "gate_definition": (
            "Wilson lower bound of A/B detection >= target detection and Wilson upper bound of A/A "
            "false-positive <= target false-positive"
        ),
        "minimum_observed_sample_size_meeting_point_targets": point_n,
        "minimum_observed_sample_size_meeting_confidence_targets": confidence_n,
        "maximum_observed_target_scenarios_per_release": int(
            max_row["target_scenarios_per_release"]
        ),
        "maximum_observed_detection_rate": float(max_row["ab_detection_rate"]),
        "maximum_observed_detection_wilson95": [
            float(max_row["ab_detection_wilson95_low"]),
            float(max_row["ab_detection_wilson95_high"]),
        ],
        "maximum_observed_false_positive_rate": float(max_row["aa_false_positive_rate"]),
        "maximum_observed_false_positive_wilson95": [
            float(max_row["aa_false_positive_wilson95_low"]),
            float(max_row["aa_false_positive_wilson95_high"]),
        ],
        "extrapolation": "FORBIDDEN_OUTSIDE_OBSERVED_SAMPLE_SIZE_RANGE",
        "recommendation": (
            "Collect additional public log-disjoint scenarios and extend the empirical curve. "
            "The exact sample size required above the observed maximum is unknown."
            if status == TARGET_NOT_REACHED
            else "Target reached inside the observed public-data range; confirm on a new independent pool."
        ),
    }


def plot_power_curve(output_dir: Path, operating: pd.DataFrame, config: Mapping[str, Any]) -> None:
    overall = operating.loc[operating["scope"] == "overall"].sort_values(
        "target_scenarios_per_release"
    )
    x = overall["target_scenarios_per_release"].to_numpy(dtype=float)
    aa = overall["aa_false_positive_rate"].to_numpy(dtype=float)
    ab = overall["ab_detection_rate"].to_numpy(dtype=float)
    aa_low = overall["aa_false_positive_wilson95_low"].to_numpy(dtype=float)
    aa_high = overall["aa_false_positive_wilson95_high"].to_numpy(dtype=float)
    ab_low = overall["ab_detection_wilson95_low"].to_numpy(dtype=float)
    ab_high = overall["ab_detection_wilson95_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, ab, marker="o", linewidth=2, label="A/B detection rate")
    ax.fill_between(x, ab_low, ab_high, alpha=0.2)
    ax.plot(x, aa, marker="s", linewidth=2, label="A/A false-positive rate")
    ax.fill_between(x, aa_low, aa_high, alpha=0.2)
    ax.axhline(float(config["target_detection_rate"]), color="tab:green", linestyle="--", label="detection target")
    ax.axhline(float(config["target_false_positive_rate"]), color="tab:red", linestyle=":", label="false-positive target")
    ax.set_xlabel("Target scenarios per software release")
    ax.set_ylabel("Empirical rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Stage 6F log-disjoint unpaired BDD power curve")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_unpaired_power_curve.png", dpi=180)
    fig.savefig(output_dir / "overall_unpaired_power_curve.pdf")
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    operating: pd.DataFrame,
) -> None:
    overall = operating.loc[operating["scope"] == "overall"].sort_values(
        "target_scenarios_per_release"
    )
    lines = [
        "# Stage 6F empirical unpaired BDD power curve",
        "",
        f"- execution status: `{summary['status']}`",
        f"- public-data sufficiency: `{summary['sufficiency']['status']}`",
        f"- detection target: `{summary['sufficiency']['target_detection_rate']:.1%}`",
        f"- false-positive target: `{summary['sufficiency']['target_false_positive_rate']:.1%}`",
        f"- insufficient diagnostic task thresholds: `{summary['threshold_audit']['insufficient_diagnostic_threshold_count']}`",
        "",
        "## Overall primary curve",
        "",
        "| target scenarios/release | mean actual A/B | A/A threshold | false-positive rate (95% CI) | detection rate (95% CI) |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for row in overall.to_dict("records"):
        lines.append(
            f"| {int(row['target_scenarios_per_release'])} | "
            f"{row['mean_actual_n_A']:.1f}/{row['mean_actual_n_B']:.1f} | {row['threshold']:.6g} | "
            f"{row['aa_false_positive_rate']:.3f} [{row['aa_false_positive_wilson95_low']:.3f}, "
            f"{row['aa_false_positive_wilson95_high']:.3f}] | {row['ab_detection_rate']:.3f} "
            f"[{row['ab_detection_wilson95_low']:.3f}, {row['ab_detection_wilson95_high']:.3f}] |"
        )
    lines.extend([
        "",
        "## Decision boundary",
        "",
        summary["sufficiency"]["recommendation"],
        "",
        "No sample-size estimate is extrapolated beyond the largest observed public-data release. Each sample size has its own A/A threshold. The curve reuses a finite public scenario pool, so it is an empirical development result rather than independent production releases. Task curves are diagnostic without multiplicity control.",
    ])
    (output_dir / "stage6f_power_curve_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = validate_power_config(stage6e.read_json(args.config_json))
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    embedding_mmap = np.load(args.embedding_path, mmap_mode="r")
    if embedding_mmap.ndim != 2:
        raise ValueError(f"embedding must be 2D, observed={embedding_mmap.shape}")
    metadata = stage6e.validate_paired_metadata(
        pd.read_csv(args.metadata_csv), config, int(embedding_mmap.shape[0])
    )
    selected = np.asarray(
        embedding_mmap[metadata[str(config["row_id_column"])].to_numpy(dtype=np.int64)],
        dtype=np.float64,
    )
    if not np.isfinite(selected).all():
        raise ValueError("selected embeddings contain non-finite values")
    inventory = stage6e.build_pair_inventory(metadata, config)
    bandwidths = stage6e.fixed_scope_bandwidths(metadata, embedding_mmap, config)
    max_size = max(config["sample_sizes_per_release"])
    if 2 * max_size > len(inventory):
        raise ValueError(
            f"largest sample size requires {2 * max_size} scenarios, available={len(inventory)}"
        )
    args.output_dir.mkdir(parents=True)

    specs = stage6e.experiment_specs(config)
    total_trials = len(config["sample_sizes_per_release"]) * sum(int(spec["trials"]) for spec in specs)
    try:
        from tqdm import tqdm

        progress = tqdm(total=total_trials, desc="Stage 6F power curve")
    except Exception:
        progress = None
    trial_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    cluster_col = str(config["cluster_column"])
    for size_position, sample_size in enumerate(config["sample_sizes_per_release"]):
        for spec in specs:
            for repetition in range(int(spec["trials"])):
                split_seed = (
                    int(config["seed"])
                    + (size_position + 1) * 10_000_000
                    + int(spec["seed_offset"])
                    + repetition
                )
                logs_a, logs_b, size_audit = choose_disjoint_logs_for_size(
                    inventory,
                    cluster_col,
                    int(sample_size),
                    np.random.default_rng(split_seed),
                    int(config["split_search_candidates"]),
                    str(config["log_split_strategy"]),
                )
                trial, trial_embeddings, overlap_audit = stage6e.build_trial(
                    metadata,
                    embedding_mmap,
                    config,
                    logs_a,
                    logs_b,
                    str(spec["planner_A"]),
                    str(spec["planner_B"]),
                )
                identity = {
                    "target_scenarios_per_release": int(sample_size),
                    "experiment_set": str(spec["experiment_set"]),
                    "family": str(spec["family"]),
                    "repetition": repetition,
                    "split_seed": split_seed,
                    "planner_A": str(spec["planner_A"]),
                    "planner_B": str(spec["planner_B"]),
                }
                audit_rows.append({**identity, **size_audit, **overlap_audit})
                assignment_rows.extend(
                    {**identity, "release_group": "A", "log_name": log_name} for log_name in logs_a
                )
                assignment_rows.extend(
                    {**identity, "release_group": "B", "log_name": log_name} for log_name in logs_b
                )
                for scope_row in stage6e.evaluate_trial_scopes(
                    trial,
                    trial_embeddings,
                    config,
                    bandwidths,
                    seed=split_seed + 1_000_000_000,
                ):
                    trial_rows.append({**identity, **size_audit, **overlap_audit, **scope_row})
                if progress is not None:
                    progress.update(1)
    if progress is not None:
        progress.close()

    trials = pd.DataFrame(trial_rows)
    thresholds, evaluated, detection, operating = summarize_by_sample_size(trials, config)
    sufficiency = build_sufficiency_summary(operating, config)
    audits = pd.DataFrame(audit_rows)
    assignments = pd.DataFrame(assignment_rows)
    evaluated.to_csv(args.output_dir / "power_curve_trial_bdd.csv", index=False)
    audits.to_csv(args.output_dir / "power_curve_split_audit.csv", index=False)
    assignments.to_csv(args.output_dir / "power_curve_log_assignments.csv", index=False)
    thresholds.to_csv(args.output_dir / "power_curve_aa_thresholds.csv", index=False)
    detection.to_csv(args.output_dir / "power_curve_detection_summary.csv", index=False)
    operating.to_csv(args.output_dir / "power_curve_operating_characteristics.csv", index=False)
    pd.DataFrame(
        [{"scope": scope, "fixed_bandwidth": bandwidth} for scope, bandwidth in bandwidths.items()]
    ).to_csv(args.output_dir / "fixed_scope_bandwidths.csv", index=False)
    plot_power_curve(args.output_dir, operating, config)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": COMPLETE_STATUS,
        "dataset_role": "PUBLIC_LOG_DISJOINT_EMPIRICAL_POWER_CURVE",
        "interpretation_role": "SAMPLE_SIZE_SENSITIVITY_WITHOUT_OUT_OF_RANGE_EXTRAPOLATION",
        "config": {key: value for key, value in config.items() if key != "stage6d_design"},
        "input_audit": {
            "row_count": int(len(metadata)),
            "pair_count": int(metadata[str(config["pair_id_column"])].nunique()),
            "cluster_count": int(metadata[cluster_col].nunique()),
            "all_trial_log_overlap_zero": bool((audits["log_overlap_count"] == 0).all()),
            "all_trial_scenario_overlap_zero": bool((audits["scenario_overlap_count"] == 0).all()),
            "all_actual_sample_sizes_within_one": bool(
                (
                    (audits["selected_scenarios_A"] - audits["target_scenarios_per_release"]).abs() <= 1
                ).all()
                and (
                    (audits["selected_scenarios_B"] - audits["target_scenarios_per_release"]).abs() <= 1
                ).all()
            ),
        },
        "fixed_bandwidths": bandwidths,
        "threshold_audit": {
            "all_overall_thresholds_pass": bool(
                (thresholds.loc[thresholds["scope"] == "overall", "status"] == "PASS").all()
            ),
            "insufficient_diagnostic_threshold_count": int(
                (
                    (thresholds["scope"] != "overall")
                    & (thresholds["status"] != "PASS")
                ).sum()
            ),
            "insufficient_diagnostic_thresholds": thresholds.loc[
                (thresholds["scope"] != "overall") & (thresholds["status"] != "PASS"),
                [
                    "target_scenarios_per_release",
                    "scope",
                    "valid_calibration_trials",
                    "minimum_valid_trials",
                    "status",
                ],
            ].to_dict("records"),
        },
        "sufficiency": sufficiency,
        "overall_power_curve": operating.loc[operating["scope"] == "overall"].to_dict("records"),
        "paired_oracle": stage6e.extract_oracle(args.paired_oracle_json),
        "limitations": [
            "Repeated trials reuse one finite public scenario pool and are not independent road-test releases.",
            "Each sample size has a separately calibrated A/A threshold; thresholds are not universal.",
            "No result is extrapolated beyond the largest observed scenarios-per-release value.",
            "Task curves are diagnostic and are not multiplicity controlled.",
            "Company A/A data must recalibrate the curve before operational use.",
        ],
    }
    stage6e.write_json(args.output_dir / "stage6f_power_curve_summary.json", summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": stage6e.sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "embedding": {
                "path": str(args.embedding_path.resolve()),
                "sha256": stage6e.sha256_file(args.embedding_path),
            },
            "metadata": {
                "path": str(args.metadata_csv.resolve()),
                "sha256": stage6e.sha256_file(args.metadata_csv),
            },
            "config": {
                "path": str(args.config_json.resolve()),
                "sha256": stage6e.sha256_file(args.config_json),
            },
            "paired_oracle": (
                {
                    "path": str(args.paired_oracle_json.resolve()),
                    "sha256": stage6e.sha256_file(args.paired_oracle_json),
                }
                if args.paired_oracle_json is not None
                else None
            ),
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    stage6e.write_json(args.output_dir / "stage6f_reproducibility_provenance.json", provenance)
    write_report(args.output_dir, summary, operating)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical log-disjoint unpaired BDD power curve with per-size A/A calibration."
    )
    parser.add_argument("--embedding_path", type=Path, required=True, help="Aligned flat 2D embedding .npy.")
    parser.add_argument("--metadata_csv", type=Path, required=True, help="Paired planner metadata for public emulation.")
    parser.add_argument("--config_json", type=Path, required=True, help="Frozen Stage 6F sample-size design.")
    parser.add_argument("--output_dir", type=Path, required=True, help="New output directory; overwrite is forbidden.")
    parser.add_argument(
        "--paired_oracle_json",
        type=Path,
        default=None,
        help="Optional paired confirmation summary, read as reference only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
