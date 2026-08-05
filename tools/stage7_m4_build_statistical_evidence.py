#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PRIMARY_ENDPOINTS = (
    {
        "metric": "delta_mean_speed",
        "label": "Mean speed",
        "unit": "m/s",
        "direction": 1,
        "hypothesis": "assertive > conservative",
    },
    {
        "metric": "delta_rms_accel",
        "label": "RMS acceleration",
        "unit": "m/s²",
        "direction": 1,
        "hypothesis": "assertive > conservative",
    },
    {
        "metric": "delta_mean_thw",
        "label": "Mean THW",
        "unit": "s",
        "direction": -1,
        "hypothesis": "assertive < conservative",
    },
)


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank_value, index in enumerate(order):
        candidate = min(1.0, (count - rank_value) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(
    values: np.ndarray, *, repetitions: int, seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def hodges_lehmann(values: np.ndarray) -> float:
    walsh = [
        (float(values[left]) + float(values[right])) / 2.0
        for left in range(len(values))
        for right in range(left, len(values))
    ]
    return float(np.median(walsh))


def endpoint_statistics(
    values: Iterable[float],
    *,
    direction: int,
    bootstrap_repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    raw = np.asarray(list(values), dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    if len(raw) < 2:
        raise ValueError(f"endpoint needs at least two finite paired values, got {len(raw)}")
    oriented = raw * int(direction)
    nonzero = oriented[np.abs(oriented) > 1e-12]
    if len(nonzero):
        wilcoxon_p = float(
            wilcoxon(nonzero, alternative="greater", zero_method="wilcox").pvalue
        )
        ranks = rankdata(np.abs(nonzero))
        rank_biserial = float(
            (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
        )
        positive = int(np.sum(nonzero > 0))
        negative = int(np.sum(nonzero < 0))
        sign_p = float(binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue)
    else:
        wilcoxon_p, rank_biserial, positive, negative, sign_p = 1.0, 0.0, 0, 0, 1.0
    std = float(np.std(oriented, ddof=1))
    raw_ci_low, raw_ci_high = bootstrap_mean_ci(
        raw, repetitions=bootstrap_repetitions, seed=seed
    )
    return {
        "n": int(len(raw)),
        "mean_delta": float(np.mean(raw)),
        "mean_ci95_low": raw_ci_low,
        "mean_ci95_high": raw_ci_high,
        "median_delta": float(np.median(raw)),
        "hodges_lehmann_delta": hodges_lehmann(raw),
        "paired_cohen_dz_oriented": float(np.mean(oriented) / std) if std > 0 else 0.0,
        "rank_biserial_oriented": rank_biserial,
        "positive_direction_count": positive,
        "opposite_direction_count": negative,
        "zero_count": int(len(raw) - len(nonzero)),
        "wilcoxon_one_sided_p": wilcoxon_p,
        "sign_test_one_sided_p": sign_p,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage7 M4 paired inference and paper-ready evidence artifacts."
    )
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--task_bdd_csv", type=Path, required=True)
    parser.add_argument("--m3_summary", type=Path, required=True)
    parser.add_argument("--bdd_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--alpha", type=float, default=0.05)
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

    paired_rows = read_csv(args.paired_delta_csv)
    task_rows = read_csv(args.task_bdd_csv)
    m3 = read_json(args.m3_summary)
    if m3.get("thesis_scale_status") != "MINIMUM_USEFUL_SCALE_REACHED":
        raise ValueError("M3 minimum useful scale was not reached")
    if len(paired_rows) != int(m3["complete_paired_scenarios"]):
        raise ValueError(
            f"paired row count mismatch: {len(paired_rows)} != "
            f"{m3['complete_paired_scenarios']}"
        )

    primary_rows: List[Dict[str, Any]] = []
    for index, endpoint in enumerate(PRIMARY_ENDPOINTS):
        metric = endpoint["metric"]
        values = []
        missing = []
        for row in paired_rows:
            value = row.get(metric, "")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            if not math.isfinite(numeric):
                missing.append(row.get("scenario", ""))
            else:
                values.append(numeric)
        stats = endpoint_statistics(
            values,
            direction=int(endpoint["direction"]),
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed + index,
        )
        primary_rows.append({
            **endpoint,
            **stats,
            "missing_pair_count": len(missing),
            "missing_scenario_examples": missing[:10],
        })
    wilcoxon_adjusted = holm_adjust(
        [float(row["wilcoxon_one_sided_p"]) for row in primary_rows]
    )
    sign_adjusted = holm_adjust(
        [float(row["sign_test_one_sided_p"]) for row in primary_rows]
    )
    for row, wilcoxon_p, sign_p in zip(primary_rows, wilcoxon_adjusted, sign_adjusted):
        row["wilcoxon_holm_p"] = wilcoxon_p
        row["sign_test_holm_p"] = sign_p
        row["wilcoxon_reject_holm"] = wilcoxon_p < args.alpha
        row["sign_test_reject_holm"] = sign_p < args.alpha

    bdd_rows = []
    expected_counts = {
        "full": int(m3["quality_pair_counts"]["full"]),
        "tier_a": int(m3["quality_pair_counts"]["tier_a"]),
        "tier_b_inclusive": int(m3["quality_pair_counts"]["tier_b_inclusive"]),
    }
    for name, expected in expected_counts.items():
        bdd = read_json(args.bdd_root / name / "bdd_summary.json")
        runtime = read_json(args.bdd_root / name / "runtime_stats.json")
        if int(bdd["n_A"]) != expected or int(bdd["n_B"]) != expected:
            raise ValueError(f"{name} BDD count mismatch")
        if int(runtime["num_bootstrap"]) != 1000 or int(runtime["num_permutation"]) != 1000:
            raise ValueError(f"{name} BDD is not the frozen 1000/1000 robustness run")
        bdd_rows.append({
            "dataset": name,
            "pairs": expected,
            "mmd2": float(bdd["mmd2"]),
            "permutation_p": float(bdd["p_value"]),
            "permutations": int(runtime["num_permutation"]),
            "bootstrap_repetitions": int(runtime["num_bootstrap"]),
            "significant_at_0_05": float(bdd["p_value"]) < args.alpha,
        })

    task_p = [float(row["p_value"]) for row in task_rows]
    task_adjusted = holm_adjust(task_p)
    task_table = []
    for row, adjusted in zip(task_rows, task_adjusted):
        task_table.append({
            "task_key": row["task_key"],
            "n_A": int(row["n_A"]),
            "n_B": int(row["n_B"]),
            "mmd2": float(row["bdd_mmd"]),
            "raw_p": float(row["p_value"]),
            "holm_p": adjusted,
            "reject_holm_0_05": adjusted < args.alpha,
            "dominant_detector_strength": row["dominant_detector_strength"],
        })

    primary_csv_rows = [
        {
            key: row[key]
            for key in (
                "metric", "label", "hypothesis", "unit", "n", "mean_delta",
                "missing_pair_count",
                "mean_ci95_low", "mean_ci95_high", "median_delta",
                "hodges_lehmann_delta", "paired_cohen_dz_oriented",
                "rank_biserial_oriented", "positive_direction_count",
                "opposite_direction_count", "wilcoxon_one_sided_p",
                "wilcoxon_holm_p", "sign_test_one_sided_p", "sign_test_holm_p",
                "wilcoxon_reject_holm", "sign_test_reject_holm",
            )
        }
        for row in primary_rows
    ]
    write_csv(args.output_dir / "table_m4_paired_primary.csv", primary_csv_rows)
    write_csv(args.output_dir / "table_m4_bdd_robustness.csv", bdd_rows)
    write_csv(args.output_dir / "table_m4_task_multiplicity.csv", task_table)
    (args.output_dir / "table_m4_paired_primary.md").write_text(
        markdown_table(
            primary_csv_rows,
            [
                "label", "hypothesis", "n", "mean_delta", "mean_ci95_low",
                "mean_ci95_high", "paired_cohen_dz_oriented",
                "wilcoxon_holm_p", "sign_test_holm_p",
            ],
        ),
        encoding="utf-8",
    )
    (args.output_dir / "table_m4_bdd_robustness.md").write_text(
        markdown_table(
            bdd_rows, ["dataset", "pairs", "mmd2", "permutation_p", "permutations"]
        ),
        encoding="utf-8",
    )
    (args.output_dir / "table_m4_task_multiplicity.md").write_text(
        markdown_table(
            task_table,
            ["task_key", "n_A", "n_B", "mmd2", "raw_p", "holm_p",
             "dominant_detector_strength"],
        ),
        encoding="utf-8",
    )

    labels = [row["label"] for row in primary_rows]
    standardized = []
    for row in primary_rows:
        direction = int(row["direction"])
        scale = abs(float(row["mean_delta"]) / float(row["paired_cohen_dz_oriented"]))
        standardized.append({
            "mean": direction * float(row["mean_delta"]) / scale,
            "low": direction * float(row["mean_ci95_low"]) / scale,
            "high": direction * float(row["mean_ci95_high"]) / scale,
        })
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(labels))
    means = np.asarray([row["mean"] for row in standardized])
    lows = np.asarray([min(row["low"], row["high"]) for row in standardized])
    highs = np.asarray([max(row["low"], row["high"]) for row in standardized])
    axis.errorbar(
        means, positions, xerr=np.vstack([means - lows, highs - means]),
        fmt="o", color="#155e75", capsize=4,
    )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Oriented standardized paired mean difference (95% bootstrap CI)")
    axis.set_title("M4 primary paired endpoints")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plot_dir / "m4_primary_effects_forest.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    bars = axis.bar(
        [row["dataset"] for row in bdd_rows],
        [row["mmd2"] for row in bdd_rows],
        color=["#155e75", "#0f766e", "#14b8a6"],
    )
    for bar, row in zip(bars, bdd_rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={row['permutation_p']:.3f}",
            ha="center", va="bottom",
        )
    axis.set_ylabel("MMD²")
    axis.set_title("BDD robustness across lane-context quality tiers")
    fig.tight_layout()
    fig.savefig(plot_dir / "m4_bdd_quality_sensitivity.png", dpi=180)
    plt.close(fig)

    checks = {
        "m3_minimum_useful_scale_reached": True,
        "paired_rows_equal_45": len(paired_rows) == 45,
        "three_primary_endpoints_complete": len(primary_rows) == 3,
        "paired_bootstrap_10000": args.bootstrap_repetitions == 10000,
        "bdd_bootstrap_1000": all(row["bootstrap_repetitions"] == 1000 for row in bdd_rows),
        "bdd_permutation_1000": all(row["permutations"] == 1000 for row in bdd_rows),
        "holm_applied_within_primary_family": len(wilcoxon_adjusted) == 3,
        "holm_applied_across_six_tasks": len(task_adjusted) == 6,
        "bdd_conclusion_stable_across_tiers": len({
            row["significant_at_0_05"] for row in bdd_rows
        }) == 1,
    }
    verdict = "PASS_WITH_LIMITATIONS" if all(checks.values()) else "FAIL"
    limitations = [
        "M4 endpoint plan was frozen after observing M3 exploratory results; this is not an independent preregistered confirmation.",
        (
            "Mean-THW inference is an available-case paired analysis because "
            f"{next(row for row in primary_rows if row['metric'] == 'delta_mean_thw')['missing_pair_count']} "
            "scenario pairs have no finite front-agent THW contrast."
        ),
        "Task-conditioned tests are exploratory, overlap across tasks, and include proxy-dominant detectors.",
        "BDD resampling intervals describe bootstrap variability and are not null intervals; permutation p-values determine significance.",
    ]
    summary = {
        "milestone": "Stage 7 Milestone 4 formal statistical evidence package",
        "overall_verdict": verdict,
        "analysis_status": "RETROSPECTIVE_FORMALIZATION_OF_M3_EXPLORATORY_RESULTS",
        "planner_contrast": "pdm_closed_assertive_v1 minus pdm_closed_conservative_v1",
        "paired_scenarios": len(paired_rows),
        "alpha": args.alpha,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "random_seed": args.seed,
        "primary_endpoints": primary_rows,
        "bdd_robustness": bdd_rows,
        "task_multiplicity": task_table,
        "checks": checks,
        "limitations": limitations,
    }
    (args.output_dir / "milestone4_statistical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Stage 7 Milestone 4 Statistical Evidence",
        "",
        f"## Verdict: `{verdict}`",
        "",
        f"- paired scenarios: `{len(paired_rows)}`",
        f"- paired bootstrap repetitions: `{args.bootstrap_repetitions}`",
        "- Full/Tier BDD: `1000 bootstrap + 1000 permutation`",
        "- multiplicity: Holm within the 3-endpoint primary family and across 6 task BDDs",
        "",
        "## Primary paired endpoints",
        "",
        markdown_table(
            primary_csv_rows,
            [
                "label", "hypothesis", "mean_delta", "mean_ci95_low",
                "mean_ci95_high", "paired_cohen_dz_oriented",
                "wilcoxon_holm_p", "sign_test_holm_p",
            ],
        ).rstrip(),
        "",
        "## BDD robustness",
        "",
        markdown_table(
            bdd_rows, ["dataset", "pairs", "mmd2", "permutation_p", "permutations"]
        ).rstrip(),
        "",
        "## Interpretation boundary",
        "",
        *[f"- {value}" for value in limitations],
    ]
    (args.output_dir / "milestone4_statistical_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if verdict == "FAIL":
        raise RuntimeError(f"M4 evidence checks failed: {checks}")
    print(f"Stage7 Milestone 4 evidence {verdict}: {args.output_dir}")


if __name__ == "__main__":
    main()
