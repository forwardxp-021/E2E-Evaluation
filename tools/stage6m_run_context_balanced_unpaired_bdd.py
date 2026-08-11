#!/usr/bin/env python3
"""Aggregate frozen Stage 6H trials into four Stage 6M release-level methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


STATUS = "STAGE6M_CONTEXT_BALANCED_UNPAIRED_RELIABILITY_COMPLETE"
PASS_STATUS = "PASS_DESCRIPTIVE_STANDARDIZED_VERSION_DRIFT"
IDENTITY = [
    "target_scenarios_per_release",
    "experiment_set",
    "family",
    "repetition",
    "split_seed",
    "planner_A",
    "planner_B",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def higher_quantile(values: Iterable[float], quantile: float) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile, method="higher")) if len(finite) else math.nan


def aggregate_trials(trials: pd.DataFrame, design: dict[str, Any]) -> pd.DataFrame:
    tasks = [row["name"] for row in design["tasks"]]
    task_weights = {row["name"]: float(row["weight"]) for row in design["tasks"]}
    rows: list[dict[str, Any]] = []
    for identity, group in trials.groupby(IDENTITY, sort=False, dropna=False):
        base = dict(zip(IDENTITY, identity))
        by_scope = group.set_index("scope")
        if set(by_scope.index) != {"overall", *tasks}:
            raise ValueError(f"incomplete trial scopes: {base}")
        overall = by_scope.loc["overall"]
        task_rows = by_scope.loc[tasks]
        definitions = [
            ("raw_marginal", float(overall["raw_mmd2"]), np.isfinite(overall["raw_mmd2"]), [overall]),
            (
                "task_conditioned",
                float(sum(task_weights[t] * float(task_rows.loc[t, "raw_mmd2"]) for t in tasks)),
                bool(np.isfinite(task_rows["raw_mmd2"].to_numpy(dtype=float)).all()),
                [task_rows.loc[t] for t in tasks],
            ),
            (
                "context_balanced",
                float(overall["standardized_mmd2"]),
                bool(overall["status"] == PASS_STATUS and np.isfinite(overall["standardized_mmd2"])),
                [overall],
            ),
            (
                "task_context_balanced",
                float(sum(task_weights[t] * float(task_rows.loc[t, "standardized_mmd2"]) for t in tasks)),
                bool(
                    (task_rows["status"] == PASS_STATUS).all()
                    and np.isfinite(task_rows["standardized_mmd2"].to_numpy(dtype=float)).all()
                ),
                [task_rows.loc[t] for t in tasks],
            ),
        ]
        for method, statistic, valid, diagnostic_rows in definitions:
            diagnostic = pd.DataFrame(diagnostic_rows)
            balanced = "balanced" in method
            rows.append(
                {
                    **base,
                    "method": method,
                    "statistic": statistic if valid else math.nan,
                    "valid": bool(valid),
                    "n_A": int(overall["n_A"]),
                    "n_B": int(overall["n_B"]),
                    "support_fraction_A_min": (
                        float(diagnostic["support_fraction_A"].min()) if balanced else 1.0
                    ),
                    "support_fraction_B_min": (
                        float(diagnostic["support_fraction_B"].min()) if balanced else 1.0
                    ),
                    "ess_ratio_A_min": float(diagnostic["ess_ratio_A"].min()) if balanced else 1.0,
                    "ess_ratio_B_min": float(diagnostic["ess_ratio_B"].min()) if balanced else 1.0,
                    "max_weight_ratio_A_max": (
                        float(diagnostic["max_weight_ratio_A"].max()) if balanced else 1.0
                    ),
                    "max_weight_ratio_B_max": (
                        float(diagnostic["max_weight_ratio_B"].max()) if balanced else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def calibrate_and_evaluate(aggregated: pd.DataFrame, design: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, Any]] = []
    evaluated_parts: list[pd.DataFrame] = []
    alpha = float(design["alpha"])
    minimum = int(design["minimum_valid_calibration_trials"])
    for (size, method), group in aggregated.groupby(
        ["target_scenarios_per_release", "method"], sort=False
    ):
        calibration = group.loc[(group["family"] == "AA_CALIBRATION") & group["valid"]]
        threshold = higher_quantile(calibration["statistic"], 1.0 - alpha)
        status = "PASS" if len(calibration) >= minimum and np.isfinite(threshold) else "INSUFFICIENT"
        threshold_rows.append(
            {
                "target_scenarios_per_release": int(size),
                "method": method,
                "alpha": alpha,
                "quantile": 1.0 - alpha,
                "quantile_method": design["calibration_quantile_method"],
                "valid_calibration_trials": int(len(calibration)),
                "minimum_valid_calibration_trials": minimum,
                "threshold": threshold,
                "status": status,
            }
        )
        current = group.copy()
        current["threshold"] = threshold
        current["calibration_status"] = status
        current["alert"] = current["valid"] & (current["statistic"] > threshold)
        current["statistic_to_threshold_ratio"] = current["statistic"] / threshold
        evaluated_parts.append(current)
    return pd.DataFrame(threshold_rows), pd.concat(evaluated_parts, ignore_index=True)


def operating_characteristics(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for (size, method), group in evaluated.groupby(
        ["target_scenarios_per_release", "method"], sort=False
    ):
        aa = group.loc[(group["family"] == "AA_EVALUATION") & group["valid"]]
        ab = group.loc[(group["family"] == "AB_EVALUATION") & group["valid"]]
        aa_count, ab_count = int(aa["alert"].sum()), int(ab["alert"].sum())
        aa_low, aa_high = wilson_interval(aa_count, len(aa))
        ab_low, ab_high = wilson_interval(ab_count, len(ab))
        pooled_rows.append(
            {
                "target_scenarios_per_release": int(size),
                "method": method,
                "threshold": float(group["threshold"].iloc[0]),
                "aa_valid_trials": int(len(aa)),
                "aa_false_positive_count": aa_count,
                "aa_false_positive_rate": aa_count / len(aa) if len(aa) else math.nan,
                "aa_wilson95_low": aa_low,
                "aa_wilson95_high": aa_high,
                "ab_valid_trials": int(len(ab)),
                "ab_detection_count": ab_count,
                "ab_detection_rate": ab_count / len(ab) if len(ab) else math.nan,
                "ab_wilson95_low": ab_low,
                "ab_wilson95_high": ab_high,
                "detection_minus_false_positive": (
                    ab_count / len(ab) - aa_count / len(aa) if len(ab) and len(aa) else math.nan
                ),
            }
        )
        for experiment_set, direction in ab.groupby("experiment_set", sort=False):
            count = int(direction["alert"].sum())
            low, high = wilson_interval(count, len(direction))
            direction_rows.append(
                {
                    "target_scenarios_per_release": int(size),
                    "method": method,
                    "experiment_set": experiment_set,
                    "valid_trials": int(len(direction)),
                    "detection_count": count,
                    "detection_rate": count / len(direction) if len(direction) else math.nan,
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )
    return pd.DataFrame(pooled_rows), pd.DataFrame(direction_rows)


def diagnostic_summary(evaluated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "support_fraction_A_min",
        "support_fraction_B_min",
        "ess_ratio_A_min",
        "ess_ratio_B_min",
        "max_weight_ratio_A_max",
        "max_weight_ratio_B_max",
    ]
    rows: list[dict[str, Any]] = []
    for (size, method), group in evaluated.groupby(
        ["target_scenarios_per_release", "method"], sort=False
    ):
        row: dict[str, Any] = {
            "target_scenarios_per_release": int(size),
            "method": method,
            "total_trials": int(len(group)),
            "valid_trials": int(group["valid"].sum()),
            "not_comparable_trials": int((~group["valid"]).sum()),
        }
        for column in columns:
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_worst"] = (
                float(group[column].min()) if "min" in column else float(group[column].max())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_method_comparisons(evaluated: pd.DataFrame) -> pd.DataFrame:
    ab = evaluated.loc[(evaluated["family"] == "AB_EVALUATION") & evaluated["valid"]]
    pivot = ab.pivot(index=IDENTITY, columns="method", values="alert").reset_index()
    rows: list[dict[str, Any]] = []
    for size, group in pivot.groupby("target_scenarios_per_release", sort=True):
        for method in ["task_conditioned", "context_balanced", "task_context_balanced"]:
            paired = group.loc[group[["raw_marginal", method]].notna().all(axis=1)]
            raw = paired["raw_marginal"].astype(bool)
            candidate = paired[method].astype(bool)
            gained = int((candidate & ~raw).sum())
            lost = int((~candidate & raw).sum())
            discordant = gained + lost
            exact_p = float(binomtest(gained, discordant, 0.5).pvalue) if discordant else 1.0
            rows.append(
                {
                    "target_scenarios_per_release": int(size),
                    "reference_method": "raw_marginal",
                    "candidate_method": method,
                    "paired_valid_trials": int(len(paired)),
                    "reference_detection_rate": float(raw.mean()),
                    "candidate_detection_rate": float(candidate.mean()),
                    "candidate_minus_reference_rate": float(candidate.mean() - raw.mean()),
                    "candidate_only_alerts": gained,
                    "reference_only_alerts": lost,
                    "mcnemar_exact_two_sided_p": exact_p,
                }
            )
    return pd.DataFrame(rows)


def plot_reliability(output_dir: Path, operating: pd.DataFrame) -> None:
    labels = {
        "raw_marginal": "Raw marginal",
        "task_conditioned": "Task-conditioned",
        "context_balanced": "Context-balanced",
        "task_context_balanced": "Task + context",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for method, group in operating.groupby("method", sort=False):
        group = group.sort_values("target_scenarios_per_release")
        x = group["target_scenarios_per_release"].to_numpy()
        axes[0].plot(x, group["aa_false_positive_rate"], marker="o", label=labels[method])
        axes[1].plot(x, group["ab_detection_rate"], marker="o", label=labels[method])
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("A/A false-positive rate")
    axes[1].set_title("A/B detection rate")
    for ax in axes:
        ax.set_xlabel("Scenarios per release")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical rate")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "stage6m_four_method_reliability.png", dpi=180)
    fig.savefig(output_dir / "stage6m_four_method_reliability.pdf")
    plt.close(fig)


def plot_distributions(output_dir: Path, evaluated: pd.DataFrame) -> None:
    subset = evaluated.loc[
        (evaluated["target_scenarios_per_release"] == 400)
        & evaluated["family"].isin(["AA_EVALUATION", "AB_EVALUATION"])
        & evaluated["valid"]
    ].copy()
    methods = ["raw_marginal", "task_conditioned", "context_balanced", "task_context_balanced"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, method in zip(axes.flat, methods):
        current = subset.loc[subset["method"] == method]
        values = [
            current.loc[current["family"] == family, "statistic_to_threshold_ratio"].to_numpy()
            for family in ["AA_EVALUATION", "AB_EVALUATION"]
        ]
        ax.boxplot(values, tick_labels=["A/A", "A/B"], showfliers=False)
        ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=1)
        ax.set_title(method)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Statistic / matched A/A threshold")
    axes[1, 0].set_ylabel("Statistic / matched A/A threshold")
    fig.suptitle("Stage 6M n=400 A/A and A/B calibrated distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "stage6m_aa_ab_distributions_n400.png", dpi=180)
    fig.savefig(output_dir / "stage6m_aa_ab_distributions_n400.pdf")
    plt.close(fig)


def write_report(
    output_dir: Path,
    operating: pd.DataFrame,
    directions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    labels = {
        "raw_marginal": "raw marginal",
        "task_conditioned": "task-conditioned",
        "context_balanced": "context-balanced",
        "task_context_balanced": "task + context-balanced",
    }
    lines = [
        "# Stage 6M context-balanced unpaired BDD 中文主报告",
        "",
        "## 四方法可靠性",
        "",
        "| 每版本场景数 | 方法 | A/A FPR (Wilson 95%) | A/B detection (Wilson 95%) | 有效A/A / A/B |",
        "|---:|---|---|---|---:|",
    ]
    for _, row in operating.sort_values(["target_scenarios_per_release", "method"]).iterrows():
        lines.append(
            f"| {int(row['target_scenarios_per_release'])} | {labels[row['method']]} | "
            f"{row['aa_false_positive_rate']:.3f} [{row['aa_wilson95_low']:.3f}, {row['aa_wilson95_high']:.3f}] | "
            f"{row['ab_detection_rate']:.3f} [{row['ab_wilson95_low']:.3f}, {row['ab_wilson95_high']:.3f}] | "
            f"{int(row['aa_valid_trials'])} / {int(row['ab_valid_trials'])} |"
        )
    best400 = operating.loc[operating["target_scenarios_per_release"] == 400].sort_values(
        "ab_detection_rate", ascending=False
    )
    best = best400.iloc[0]
    raw = best400.loc[best400["method"] == "raw_marginal"].iloc[0]
    comparison400 = comparisons.loc[
        (comparisons["target_scenarios_per_release"] == 400)
        & (comparisons["candidate_method"] == "context_balanced")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## 核心解释",
            "",
            f"- n=400时检出率最高的方法为 `{labels[best['method']]}`：{best['ab_detection_rate']:.3f}；raw marginal为 {raw['ab_detection_rate']:.3f}。",
            f"- context-balanced相对raw为 {comparison400['candidate_minus_reference_rate']:+.3f}；同一release split配对McNemar exact p={comparison400['mcnemar_exact_two_sided_p']:.4g}，不支持稳定提升。",
            "- 每种方法和样本量均使用其匹配的独立A/A calibration阈值；没有定义通用raw BDD阈值。",
            "- task-conditioned统计按冻结800-pair池中的五类task比例聚合，不由A/B结果重新选择task或权重。",
            "- context balance仅使用pre-treatment的map_name、scenario_type与冻结task；不使用任何rollout后行为、fallback、embedding或BDD进行matching。",
            "",
            "## 支持度与边界",
            "",
            "- common support、ESS、最大权重及不可比trial见 `stage6m_support_ess_diagnostics.csv`。",
            "- 两个release严格log-disjoint且scenario-token不重叠；有限public pool的重复release模拟不等于真实独立量产发布。",
            "- 结果用于版本风格漂移检测，不证明安全性、planner优劣或因果效应。",
            "- 双向A/B结果见 `stage6m_direction_specific_detection.csv`；方向差异只作诊断。",
        ]
    )
    (output_dir / "stage6m_context_balanced_unpaired_bdd_report_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    freeze = read_json(args.freeze_manifest.resolve())
    if freeze.get("status") != "FROZEN_BEFORE_STAGE6M_AGGREGATED_RELIABILITY_RESULTS":
        raise ValueError("Stage 6M freeze manifest is not authoritative")
    design = freeze["design"]
    expected_trial_sha = freeze["sources"]["trial_bdd"]["sha256"]
    if sha256_file(args.trial_bdd.resolve()) != expected_trial_sha:
        raise ValueError("trial BDD input differs from frozen SHA-256")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    trials = pd.read_csv(args.trial_bdd)
    aggregated = aggregate_trials(trials, design)
    thresholds, evaluated = calibrate_and_evaluate(aggregated, design)
    operating, directions = operating_characteristics(evaluated)
    diagnostics = diagnostic_summary(evaluated)
    comparisons = paired_method_comparisons(evaluated)
    aggregated.to_csv(output_dir / "stage6m_aggregated_trial_statistics.csv", index=False)
    evaluated.to_csv(output_dir / "stage6m_evaluated_trials.csv", index=False)
    thresholds.to_csv(output_dir / "stage6m_method_specific_aa_thresholds.csv", index=False)
    operating.to_csv(output_dir / "stage6m_four_method_operating_characteristics.csv", index=False)
    directions.to_csv(output_dir / "stage6m_direction_specific_detection.csv", index=False)
    comparisons.to_csv(output_dir / "stage6m_paired_method_comparisons.csv", index=False)
    diagnostics.to_csv(output_dir / "stage6m_support_ess_diagnostics.csv", index=False)
    plot_reliability(output_dir, operating)
    plot_distributions(output_dir, evaluated)
    write_report(output_dir, operating, directions, diagnostics, comparisons)
    summary = {
        "schema_version": "stage6m_context_balanced_unpaired_bdd_results_v1",
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 254,
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "source_stage6h_results_modified": False,
        "method_definitions": design["methods"],
        "operating_characteristics": operating.to_dict("records"),
        "direction_specific_detection": directions.to_dict("records"),
        "paired_method_comparisons": comparisons.to_dict("records"),
        "support_ess_diagnostics": diagnostics.to_dict("records"),
    }
    (output_dir / "stage6m_context_balanced_unpaired_bdd_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": STATUS, "output_dir": str(output_dir)}, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--trial_bdd", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
