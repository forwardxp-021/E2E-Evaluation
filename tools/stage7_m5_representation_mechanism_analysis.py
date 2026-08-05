#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, spearmanr, t
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6_compare_unpaired_style import compute_mmd2
from tools.stage7_m4_build_statistical_evidence import holm_adjust, markdown_table, write_csv


TRAJECTORY_FEATURES = (
    "front_valid_ratio",
    "max_abs_accel",
    "max_abs_jerk",
    "max_speed",
    "mean_abs_yaw_rate",
    "mean_front_distance",
    "mean_speed",
    "mean_thw",
    "min_front_distance",
    "min_thw",
    "rms_accel",
    "rms_jerk",
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


def robust_standardize(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(source, axis=0)
    filled = np.where(np.isfinite(source), source, median)
    q25, q75 = np.percentile(filled, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.std(filled, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return (filled - median) / scale


def build_trajectory_summary(
    paired_rows: Sequence[Dict[str, str]], row_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((row_count, len(TRAJECTORY_FEATURES)), np.nan, dtype=np.float64)
    labels = np.full(row_count, -1, dtype=np.int64)
    groups = np.full(row_count, -1, dtype=np.int64)
    pair_indices = []
    for group, row in enumerate(paired_rows):
        index_a, index_b = int(row["row_A"]), int(row["row_B"])
        pair_indices.append((index_a, index_b))
        labels[index_a], labels[index_b] = 1, 0
        groups[index_a] = groups[index_b] = group
        for feature_index, feature in enumerate(TRAJECTORY_FEATURES):
            for row_index, suffix in ((index_a, "A"), (index_b, "B")):
                try:
                    values[row_index, feature_index] = float(row[f"{feature}_{suffix}"])
                except (TypeError, ValueError):
                    values[row_index, feature_index] = np.nan
    if set(labels.tolist()) != {0, 1} or np.any(groups < 0):
        raise ValueError("paired delta rows do not cover every representation row exactly")
    return values, labels, groups, np.asarray(pair_indices, dtype=np.int64)


def grouped_probe_score(
    values: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, folds: int
) -> Dict[str, float]:
    splitter = GroupKFold(n_splits=folds)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    for train, test in splitter.split(values, labels, groups):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=0),
        )
        model.fit(values[train], labels[train])
        probabilities[test] = model.predict_proba(values[test])[:, 1]
        predictions[test] = (probabilities[test] >= 0.5).astype(np.int64)
    return {
        "grouped_roc_auc": float(roc_auc_score(labels, probabilities)),
        "grouped_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def paired_label_permutation_pvalue(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    pair_indices: np.ndarray,
    *,
    folds: int,
    repetitions: int,
    seed: int,
) -> Tuple[Dict[str, float], float, np.ndarray]:
    observed = grouped_probe_score(values, labels, groups, folds=folds)
    rng = np.random.default_rng(seed)
    null_auc = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        permuted = labels.copy()
        swap = rng.integers(0, 2, size=len(pair_indices)).astype(bool)
        for should_swap, (index_a, index_b) in zip(swap, pair_indices):
            if should_swap:
                permuted[index_a], permuted[index_b] = permuted[index_b], permuted[index_a]
        null_auc[repetition] = grouped_probe_score(
            values, permuted, groups, folds=folds
        )["grouped_roc_auc"]
    p_value = float(
        (np.sum(null_auc >= observed["grouped_roc_auc"]) + 1) / (repetitions + 1)
    )
    return observed, p_value, null_auc


def paired_sign_flip_test(
    values: np.ndarray,
    pair_indices: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    standardized = robust_standardize(values)
    differences = standardized[pair_indices[:, 0]] - standardized[pair_indices[:, 1]]
    observed = float(np.linalg.norm(np.mean(differences, axis=0)))
    denominator = float(np.sum(np.linalg.norm(differences, axis=1)))
    concentration = (
        float(np.linalg.norm(np.sum(differences, axis=0)) / denominator)
        if denominator > 0
        else 0.0
    )
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        null_means = signs @ differences / len(differences)
        exceed += int(np.sum(np.linalg.norm(null_means, axis=1) >= observed))
    return {
        "standardized_mean_shift_norm": observed,
        "paired_direction_concentration": concentration,
        "sign_flip_p": float((exceed + 1) / (repetitions + 1)),
        "sign_flip_repetitions": repetitions,
    }


def marginal_mmd_test(
    values: np.ndarray,
    pair_indices: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    source = np.asarray(values, dtype=np.float64)
    index_a, index_b = pair_indices[:, 0], pair_indices[:, 1]
    rng = np.random.default_rng(seed)
    observed = compute_mmd2(source[index_a], source[index_b], rng, 2000, 512)
    pooled = np.concatenate([index_a, index_b])
    exceed = 0
    for _ in range(repetitions):
        permutation = rng.permutation(pooled)
        candidate = compute_mmd2(
            source[permutation[: len(index_a)]],
            source[permutation[len(index_a) :]],
            rng,
            2000,
            512,
        )
        exceed += int(candidate >= observed)
    return {
        "mmd2": float(observed),
        "permutation_p": float((exceed + 1) / (repetitions + 1)),
        "permutations": repetitions,
    }


def minimum_detectable_paired_dz(n: int, *, alpha: float, power: float) -> float:
    degrees = n - 1
    critical = t.ppf(1.0 - alpha, degrees)

    def objective(effect: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return 1.0 - nct.cdf(critical, degrees, effect * math.sqrt(n)) - power

    return float(brentq(objective, 1e-6, 5.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain paired-vs-marginal representation sensitivity for Stage7 M5."
    )
    parser.add_argument("--embedding_path", type=Path, required=True)
    parser.add_argument("--interaction_feature_path", type=Path, required=True)
    parser.add_argument("--paired_delta_csv", type=Path, required=True)
    parser.add_argument("--m4_summary", type=Path, required=True)
    parser.add_argument("--m4_full_bdd_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--probe_permutations", type=int, default=1000)
    parser.add_argument("--sign_flip_repetitions", type=int, default=10000)
    parser.add_argument("--mmd_permutations", type=int, default=1000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
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
    representation_dir = args.output_dir / "representations"
    plot_dir.mkdir()
    representation_dir.mkdir()

    m4 = read_json(args.m4_summary)
    m4_full_bdd = read_json(args.m4_full_bdd_summary)
    if m4.get("overall_verdict") != "PASS_WITH_LIMITATIONS":
        raise ValueError("M4 evidence package is not frozen PASS_WITH_LIMITATIONS")
    paired_rows = read_csv(args.paired_delta_csv)
    embedding = np.asarray(np.load(args.embedding_path, mmap_mode="r"), dtype=np.float64)
    interaction = np.asarray(
        np.load(args.interaction_feature_path, mmap_mode="r"), dtype=np.float64
    )
    if embedding.shape[0] != interaction.shape[0]:
        raise ValueError("embedding and interaction feature row counts differ")
    trajectory, labels, groups, pair_indices = build_trajectory_summary(
        paired_rows, embedding.shape[0]
    )
    if len(pair_indices) != 45:
        raise ValueError(f"M5 requires frozen 45 pairs, got {len(pair_indices)}")
    representations = {
        "learned_embedding": {
            "analysis": embedding,
            "probe": embedding,
        },
        "interaction_features": {
            "analysis": robust_standardize(interaction),
            "probe": interaction,
        },
        "trajectory_summary": {
            "analysis": robust_standardize(trajectory),
            "probe": trajectory,
        },
    }
    for name, payload in representations.items():
        np.save(
            representation_dir / f"{name}.npy",
            payload["analysis"].astype(np.float32),
        )
    (representation_dir / "representation_schema.json").write_text(
        json.dumps(
            {
                "learned_embedding": {"dim": embedding.shape[1], "source": str(args.embedding_path)},
                "interaction_features": {
                    "dim": interaction.shape[1],
                    "source": str(args.interaction_feature_path),
                    "paired_mmd_transform": "pooled median/IQR with finite median imputation",
                    "probe_transform": "median imputation and standard scaling fit on each training fold only",
                },
                "trajectory_summary": {
                    "dim": len(TRAJECTORY_FEATURES),
                    "features": list(TRAJECTORY_FEATURES),
                    "paired_mmd_transform": "pooled median/IQR with finite median imputation",
                    "probe_transform": "median imputation and standard scaling fit on each training fold only",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    comparison_rows = []
    null_distributions = {}
    for index, (name, payload) in enumerate(
        tqdm(representations.items(), desc="M5 representations")
    ):
        analysis_values = payload["analysis"]
        probe_values = payload["probe"]
        sign_flip = paired_sign_flip_test(
            analysis_values,
            pair_indices,
            repetitions=args.sign_flip_repetitions,
            seed=args.seed + index,
        )
        probe, probe_p, null_auc = paired_label_permutation_pvalue(
            probe_values,
            labels,
            groups,
            pair_indices,
            folds=args.folds,
            repetitions=args.probe_permutations,
            seed=args.seed + 100 + index,
        )
        mmd = marginal_mmd_test(
            analysis_values,
            pair_indices,
            repetitions=args.mmd_permutations,
            seed=args.seed + 200 + index,
        )
        marginal_source = "M5 representation-control permutation"
        if name == "learned_embedding":
            if int(m4_full_bdd["n_A"]) != 45 or int(m4_full_bdd["n_B"]) != 45:
                raise ValueError("M4 Full BDD does not contain the frozen 45 planner pairs")
            mmd["mmd2"] = float(m4_full_bdd["mmd2"])
            mmd["permutation_p"] = float(m4_full_bdd["p_value"])
            marginal_source = "frozen M4 Full 1000-permutation BDD"
        comparison_rows.append({
            "representation": name,
            "dimension": int(analysis_values.shape[1]),
            **sign_flip,
            **probe,
            "grouped_probe_permutation_p": probe_p,
            **mmd,
            "marginal_mmd_source": marginal_source,
        })
        null_distributions[name] = null_auc
    write_csv(args.output_dir / "table_m5_representation_mechanism.csv", comparison_rows)
    (args.output_dir / "table_m5_representation_mechanism.md").write_text(
        markdown_table(
            comparison_rows,
            [
                "representation", "dimension", "standardized_mean_shift_norm",
                "paired_direction_concentration", "sign_flip_p", "grouped_roc_auc",
                "grouped_balanced_accuracy", "grouped_probe_permutation_p",
                "mmd2", "permutation_p",
            ],
        ),
        encoding="utf-8",
    )
    np.savez(
        args.output_dir / "grouped_probe_null_auc.npz",
        **{name: values for name, values in null_distributions.items()},
    )

    correlation_rows = []
    embedding_distance = np.asarray(
        [float(row["embedding_l2_distance"]) for row in paired_rows]
    )
    for metric in ("delta_mean_speed", "delta_rms_accel", "delta_mean_thw"):
        delta = np.asarray([
            float(row[metric]) if row.get(metric, "") not in ("", "nan", "NaN") else np.nan
            for row in paired_rows
        ])
        valid = np.isfinite(delta) & np.isfinite(embedding_distance)
        result = spearmanr(embedding_distance[valid], np.abs(delta[valid]))
        correlation_rows.append({
            "trajectory_contrast": metric,
            "n": int(np.sum(valid)),
            "spearman_embedding_distance_vs_abs_delta": float(result.statistic),
            "raw_p": float(result.pvalue),
        })
    adjusted = holm_adjust([row["raw_p"] for row in correlation_rows])
    for row, value in zip(correlation_rows, adjusted):
        row["holm_p"] = value
        row["reject_holm_0_05"] = value < 0.05
    write_csv(args.output_dir / "table_m5_distance_sensitivity.csv", correlation_rows)
    (args.output_dir / "table_m5_distance_sensitivity.md").write_text(
        markdown_table(
            correlation_rows,
            [
                "trajectory_contrast", "n",
                "spearman_embedding_distance_vs_abs_delta", "raw_p", "holm_p",
            ],
        ),
        encoding="utf-8",
    )

    mde_rows = []
    for n in (30, 45, 60, 90, 120):
        mde_rows.append({
            "paired_n": n,
            "mde_dz_one_sided_alpha_0_05_power_0_80": minimum_detectable_paired_dz(
                n, alpha=0.05, power=0.80
            ),
            "mde_dz_one_sided_bonferroni_0_05_over_3_power_0_80": minimum_detectable_paired_dz(
                n, alpha=0.05 / 3.0, power=0.80
            ),
        })
    write_csv(args.output_dir / "table_m5_paired_mde.csv", mde_rows)
    (args.output_dir / "table_m5_paired_mde.md").write_text(
        markdown_table(mde_rows, list(mde_rows[0])), encoding="utf-8"
    )

    names = [row["representation"] for row in comparison_rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    axes[0].bar(x, [row["paired_direction_concentration"] for row in comparison_rows])
    axes[0].set_title("Paired shift concentration")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, [row["grouped_roc_auc"] for row in comparison_rows])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Grouped planner ROC-AUC")
    axes[1].set_ylim(0, 1)
    axes[2].bar(x, [row["permutation_p"] for row in comparison_rows])
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Marginal MMD permutation p")
    axes[2].set_ylim(0, 1)
    for axis in axes:
        axis.set_xticks(x, ["embedding", "interaction", "trajectory"], rotation=25)
    fig.suptitle("M5 paired, predictive, and marginal representation sensitivity")
    fig.tight_layout()
    fig.savefig(plot_dir / "m5_representation_mechanism.png", dpi=180)
    plt.close(fig)

    standardized_embedding = robust_standardize(embedding)
    coordinates = PCA(n_components=2, random_state=0).fit_transform(standardized_embedding)
    fig, axis = plt.subplots(figsize=(6.5, 5.2))
    for index_a, index_b in pair_indices:
        axis.plot(
            [coordinates[index_b, 0], coordinates[index_a, 0]],
            [coordinates[index_b, 1], coordinates[index_a, 1]],
            color="#94a3b8",
            alpha=0.35,
            linewidth=0.8,
        )
    axis.scatter(
        coordinates[pair_indices[:, 1], 0],
        coordinates[pair_indices[:, 1], 1],
        label="conservative",
        s=24,
        color="#2563eb",
    )
    axis.scatter(
        coordinates[pair_indices[:, 0], 0],
        coordinates[pair_indices[:, 0], 1],
        label="assertive",
        s=24,
        color="#dc2626",
    )
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.set_title("Learned embedding: same-scenario paired shifts")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "m5_embedding_paired_pca.png", dpi=180)
    plt.close(fig)

    embedding_result = next(
        row for row in comparison_rows if row["representation"] == "learned_embedding"
    )
    checks = {
        "m4_frozen_input": True,
        "complete_pair_count_45": len(pair_indices) == 45,
        "three_representations_compared": len(comparison_rows) == 3,
        "grouped_cv_scenario_disjoint": True,
        "probe_imputation_and_scaling_fit_within_training_folds": True,
        "probe_pairwise_label_permutation_1000": args.probe_permutations == 1000,
        "paired_sign_flip_10000": args.sign_flip_repetitions == 10000,
        "marginal_mmd_permutation_1000": args.mmd_permutations == 1000,
        "embedding_paired_and_marginal_questions_both_reported": (
            "sign_flip_p" in embedding_result and "permutation_p" in embedding_result
        ),
        "paired_mde_reported_without_observed_effect_power_claim": len(mde_rows) == 5,
    }
    verdict = "PASS_WITH_LIMITATIONS" if all(checks.values()) else "FAIL"
    limitations = [
        "Grouped linear probes diagnose representation separability; they are not new planner-performance metrics.",
        "Interaction and trajectory baselines are derived from the same realized rollouts and are descriptive mechanism controls.",
        "MDE is a paired-t sensitivity calculation, not post-hoc achieved power and not an MMD sample-size guarantee.",
        "M5 was designed after M3/M4 results and remains explanatory rather than independent confirmation.",
    ]
    summary = {
        "milestone": "Stage 7 Milestone 5 representation mechanism analysis",
        "overall_verdict": verdict,
        "analysis_status": "EXPLANATORY_POST_M4_MECHANISM_ANALYSIS",
        "paired_scenarios": len(pair_indices),
        "representation_comparison": comparison_rows,
        "embedding_distance_sensitivity": correlation_rows,
        "paired_minimum_detectable_effect": mde_rows,
        "checks": checks,
        "limitations": limitations,
        "interpretation": (
            "Paired sign-flip, scenario-grouped probing, and marginal MMD answer different "
            "questions. A representation can encode systematic within-scenario planner shifts "
            "without producing a large marginal distribution difference."
        ),
    }
    (args.output_dir / "milestone5_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Stage 7 Milestone 5 Representation Mechanism",
        "",
        f"## Verdict: `{verdict}`",
        "",
        "## Representation comparison",
        "",
        markdown_table(
            comparison_rows,
            [
                "representation", "paired_direction_concentration", "sign_flip_p",
                "grouped_roc_auc", "grouped_probe_permutation_p", "mmd2",
                "permutation_p",
            ],
        ).rstrip(),
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Limitations",
        "",
        *[f"- {value}" for value in limitations],
    ]
    (args.output_dir / "milestone5_mechanism_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if verdict == "FAIL":
        raise RuntimeError(f"M5 checks failed: {checks}")
    print(f"Stage7 Milestone 5 mechanism {verdict}: {args.output_dir}")


if __name__ == "__main__":
    main()
