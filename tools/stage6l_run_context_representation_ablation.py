#!/usr/bin/env python3
"""Run frozen Stage 6L A-D paired-BDD and context-quality diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6j_run_paired_bdd import build_task_masks
from tools.stage6k_run_longitudinal_dose_bdd import build_pairs, null_diagnostics
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file
from tools.stage7_m6_scenario_conditioned_bdd import exact_median_bandwidth, holm_adjust, rbf_kernel


REPRESENTATIONS = [
    "learned64_full_context",
    "learned64_neighbor_zero_input",
    "ego_kinematic_13d",
    "handcrafted_interaction_trajectory_46d",
]
REP_LABELS_ZH = {
    "learned64_full_context": "A 完整上下文64D",
    "learned64_neighbor_zero_input": "B 邻车置零64D",
    "ego_kinematic_13d": "C ego运动学13D",
    "handcrafted_interaction_trajectory_46d": "D 交互+轨迹46D",
}
DOSES = [
    ("dose25", 0.25, "pdm_closed_assertive_longitudinal_dose25_v1"),
    ("dose50", 0.50, "pdm_closed_assertive_longitudinal_dose50_v1"),
    ("dose75", 0.75, "pdm_closed_assertive_longitudinal_dose75_v1"),
    ("dose100", 1.00, "pdm_closed_assertive_longitudinal_v1"),
]
QUALITY_METRICS = ["max_pair_fallback_rate", "max_pair_ambiguity_rate"]
QUALITY_OUTCOMES = ["pair_l2_distance", "absolute_kernel_contribution"]


def configure_cjk_font() -> None:
    candidates = [
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(path)).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--decision_addendum_manifest", type=Path, required=True)
    parser.add_argument("--representation_dir", type=Path, required=True)
    parser.add_argument("--stage6j_context_dir", type=Path, required=True)
    parser.add_argument("--stage6k_contexts_dir", type=Path, required=True)
    parser.add_argument("--stage6j_bdd_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def signed_quadratic_null(
    contrast_kernel: np.ndarray,
    repetitions: int,
    seed: int,
    *,
    cluster_inverse: np.ndarray | None = None,
    chunk_size: int = 2000,
) -> np.ndarray:
    contrast = np.asarray(contrast_kernel, dtype=np.float64)
    n_pairs = len(contrast)
    if contrast.shape != (n_pairs, n_pairs):
        raise ValueError("contrast kernel must be square")
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    n_units = n_pairs if cluster_inverse is None else int(np.max(cluster_inverse)) + 1
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        unit_signs = rng.integers(0, 2, size=(stop - start, n_units), dtype=np.int8)
        unit_signs = unit_signs.astype(np.float64) * 2.0 - 1.0
        signs = unit_signs if cluster_inverse is None else unit_signs[:, cluster_inverse]
        samples[start:stop] = np.einsum("bi,ij,bj->b", signs, contrast, signs, optimize=True) / (n_pairs**2)
    return samples


def kernel_analysis(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    logs: Sequence[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"Invalid paired representation arrays: {a.shape}/{b.shape}")
    n_pairs = len(a)
    pooled = np.vstack([a, b])
    bandwidth = exact_median_bandwidth(pooled)
    kernel = rbf_kernel(pooled, bandwidth)
    kaa = kernel[:n_pairs, :n_pairs]
    kbb = kernel[n_pairs:, n_pairs:]
    kab = kernel[:n_pairs, n_pairs:]
    contrast = kaa + kbb - kab - kab.T
    observed = float(np.sum(contrast) / (n_pairs**2))
    cluster_inverse = None
    n_clusters = n_pairs
    if logs is not None:
        unique_logs, cluster_inverse = np.unique(np.asarray(logs, dtype=str), return_inverse=True)
        n_clusters = len(unique_logs)
    samples = signed_quadratic_null(contrast, repetitions, seed, cluster_inverse=cluster_inverse)
    exceedance = int(np.sum(samples >= observed))
    result = {
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "mmd2": observed,
        "bandwidth": bandwidth,
        "permutations": repetitions,
        "exceedance_count": exceedance,
        "raw_p": float((exceedance + 1) / (repetitions + 1)),
    }
    result.update(null_diagnostics(observed, samples))
    contribution = np.sum(contrast, axis=1) / n_pairs
    if not np.isclose(float(np.mean(contribution)), observed, rtol=1e-10, atol=1e-12):
        raise AssertionError("Per-pair kernel contributions do not average to observed MMD²")
    return result, samples, contribution


def residualized_ranks(values: np.ndarray, tasks: Sequence[str] | None) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average").astype(np.float64)
    if tasks is not None:
        labels = np.asarray(tasks, dtype=str)
        for label in np.unique(labels):
            mask = labels == label
            ranks[mask] -= float(np.mean(ranks[mask]))
    return ranks


def weighted_correlations(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights, axis=1)
    mean_x = weights @ x / total
    mean_y = weights @ y / total
    centered_x = x[None, :] - mean_x[:, None]
    centered_y = y[None, :] - mean_y[:, None]
    cov = np.sum(weights * centered_x * centered_y, axis=1)
    var_x = np.sum(weights * centered_x**2, axis=1)
    var_y = np.sum(weights * centered_y**2, axis=1)
    denom = np.sqrt(var_x * var_y)
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)


def fixed_rank_log_cluster_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    logs: Sequence[str],
    tasks: Sequence[str] | None,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    xr = residualized_ranks(x, tasks)
    yr = residualized_ranks(y, tasks)
    estimate = float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 0 and np.std(yr) > 0 else 0.0
    unique_logs, inverse = np.unique(np.asarray(logs, dtype=str), return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        draws = rng.integers(0, len(unique_logs), size=(stop - start, len(unique_logs)))
        counts = np.zeros((stop - start, len(unique_logs)), dtype=np.float64)
        row_ids = np.repeat(np.arange(stop - start), len(unique_logs))
        np.add.at(counts, (row_ids, draws.ravel()), 1.0)
        samples[start:stop] = weighted_correlations(xr, yr, counts[:, inverse])
    return estimate, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def context_dir(args: argparse.Namespace, label: str) -> Path:
    return args.stage6j_context_dir.resolve() if label == "dose100" else args.stage6k_contexts_dir.resolve() / label


def pair_quality(
    args: argparse.Namespace,
    label: str,
    metadata: pd.DataFrame,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quality = pd.read_csv(context_dir(args, label) / "nuplan_lane_assignment_by_row.csv").set_index("global_row")
    fallback = np.asarray([
        max(float(quality.loc[int(a), "fallback_rate"]), float(quality.loc[int(b), "fallback_rate"]))
        for a, b in pairs
    ])
    ambiguity = np.asarray([
        max(float(quality.loc[int(a), "ambiguous_frame_rate"]), float(quality.loc[int(b), "ambiguous_frame_rate"]))
        for a, b in pairs
    ])
    return fallback, ambiguity


def apply_holm(rows: list[dict[str, Any]], cluster_rows: list[dict[str, Any]]) -> None:
    for rep in REPRESENTATIONS:
        current = [row for row in rows if row["representation"] == rep]
        overall = [row for row in current if row["scope"] == "overall"]
        tasks = [row for row in current if row["scope"] != "overall"]
        clusters = [row for row in cluster_rows if row["representation"] == rep]
        if len(overall) != 4 or len(tasks) != 12 or len(clusters) != 4:
            raise ValueError(f"Unexpected multiplicity family for {rep}: {len(overall)}/{len(tasks)}/{len(clusters)}")
        for family in [overall, tasks, clusters]:
            adjusted = holm_adjust([float(row["raw_p"]) for row in family])
            for row, value in zip(family, adjusted):
                row["holm_p"] = float(value)
                row["reject_holm_0_05"] = bool(value < 0.05)


def make_plots(rows: list[dict[str, Any]], associations: list[dict[str, Any]], output_dir: Path) -> None:
    configure_cjk_font()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for rep, axis in zip(REPRESENTATIONS, axes.ravel()):
        current = sorted([r for r in rows if r["representation"] == rep and r["scope"] == "overall"], key=lambda r: r["nominal_dose"])
        x = [100 * float(r["nominal_dose"]) for r in current]
        axis.plot(x, [float(r["null_standardized_z_bdd"]) for r in current], marker="o", label="Z_BDD")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=REP_LABELS_ZH[rep], xlabel="名义纵向剂量 (%)", ylabel="各自null标准化 Z_BDD")
        axis.grid(alpha=0.25)
    fig.savefig(output_dir / "stage6l_representation_dose_z_bdd.png", dpi=180)
    fig.savefig(output_dir / "stage6l_representation_dose_z_bdd.pdf")
    plt.close(fig)

    task_names = sorted({str(r["scope"]) for r in rows if r["scope"] != "overall"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for rep, axis in zip(REPRESENTATIONS, axes.ravel()):
        matrix = np.zeros((len(task_names), 4), dtype=float)
        for i, task in enumerate(task_names):
            for j, (label, _, _) in enumerate(DOSES):
                row = next(r for r in rows if r["representation"] == rep and r["scope"] == task and r["dose_label"] == label)
                matrix[i, j] = min(-math.log10(max(float(row["holm_p"]), 1e-12)), 12.0)
        im = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=max(2.0, float(matrix.max())))
        axis.set(title=REP_LABELS_ZH[rep], xticks=range(4), xticklabels=["25", "50", "75", "100"], yticks=range(len(task_names)), yticklabels=task_names, xlabel="剂量 (%)")
        fig.colorbar(im, ax=axis, label="-log10(Holm p)")
    fig.savefig(output_dir / "stage6l_task_detection_heatmap.png", dpi=180)
    fig.savefig(output_dir / "stage6l_task_detection_heatmap.pdf")
    plt.close(fig)

    selected = [a for a in associations if a["quality_metric"] == "max_pair_fallback_rate" and a["analysis"] == "task_adjusted_fixed_rank_log_cluster_bootstrap"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for outcome, axis in zip(QUALITY_OUTCOMES, axes):
        matrix = np.zeros((4, 4), dtype=float)
        for i, rep in enumerate(REPRESENTATIONS):
            for j, (label, _, _) in enumerate(DOSES):
                matrix[i, j] = float(next(a["estimate"] for a in selected if a["representation"] == rep and a["dose_label"] == label and a["outcome"] == outcome))
        im = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-0.6, vmax=0.6)
        axis.set(title=outcome, xticks=range(4), xticklabels=["25", "50", "75", "100"], yticks=range(4), yticklabels=[REP_LABELS_ZH[r] for r in REPRESENTATIONS], xlabel="剂量 (%)")
        fig.colorbar(im, ax=axis, label="task-adjusted rank correlation")
    fig.savefig(output_dir / "stage6l_context_quality_association.png", dpi=180)
    fig.savefig(output_dir / "stage6l_context_quality_association.pdf")
    plt.close(fig)


def decision_summary(
    rows: list[dict[str, Any]], associations: list[dict[str, Any]], addendum: dict[str, Any]
) -> dict[str, Any]:
    overall_z = {
        rep: [float(r["null_standardized_z_bdd"]) for r in rows if r["representation"] == rep and r["scope"] == "overall"]
        for rep in REPRESENTATIONS
    }
    median_z = {rep: float(np.median(values)) for rep, values in overall_z.items()}
    task_pass = {
        rep: sum(bool(r["reject_holm_0_05"]) for r in rows if r["representation"] == rep and r["scope"] != "overall")
        for rep in REPRESENTATIONS
    }
    positive_doses: dict[str, list[str]] = {}
    for rep in ["learned64_full_context", "learned64_neighbor_zero_input"]:
        positive = []
        for label, _, _ in DOSES:
            candidates = [
                a for a in associations
                if a["representation"] == rep and a["dose_label"] == label
                and a["quality_metric"] == "max_pair_fallback_rate"
                and a["outcome"] in QUALITY_OUTCOMES
                and a["analysis"] == "task_adjusted_fixed_rank_log_cluster_bootstrap"
            ]
            if any(bool(a["interval_excludes_zero"]) and float(a["estimate"]) > 0 for a in candidates):
                positive.append(label)
        positive_doses[rep] = positive
    context_rule = addendum["context_v2_go"]
    context_go = len(positive_doses["learned64_full_context"]) >= int(context_rule["minimum_positive_doses"])
    neighbor_reduction = len(positive_doses["learned64_full_context"]) - len(positive_doses["learned64_neighbor_zero_input"])

    rule = addendum["retraining_go"]
    base_z = median_z["learned64_full_context"]
    ego_condition = (
        median_z["ego_kinematic_13d"] >= float(rule["ego_z_ratio"]) * base_z
        and task_pass["ego_kinematic_13d"] >= task_pass["learned64_full_context"] + int(rule["ego_extra_task_cells"])
    )
    hand_condition = (
        median_z["handcrafted_interaction_trajectory_46d"] >= float(rule["handcrafted_z_ratio"]) * base_z
        and task_pass["handcrafted_interaction_trajectory_46d"] >= task_pass["learned64_full_context"] + int(rule["handcrafted_extra_task_cells"])
    )
    neighbor_condition = (
        median_z["learned64_neighbor_zero_input"] >= float(rule["neighbor_zero_min_z_retention"]) * base_z
        and task_pass["learned64_neighbor_zero_input"] >= task_pass["learned64_full_context"] - int(rule["neighbor_zero_max_task_cell_loss"])
        and context_go
    )
    retraining_go = bool(ego_condition or hand_condition or neighbor_condition)
    min_dose = {}
    for rep in REPRESENTATIONS:
        detected = sorted(float(r["nominal_dose"]) for r in rows if r["representation"] == rep and r["scope"] == "overall" and r["reject_holm_0_05"])
        min_dose[rep] = detected[0] if detected else None
    return {
        "median_overall_z_bdd": median_z,
        "task_dose_holm_pass_cells_out_of_12": task_pass,
        "minimum_detectable_nominal_dose": min_dose,
        "positive_fallback_association_doses": positive_doses,
        "context_v2_decision": "GO_PREPARE_SEPARATELY_VERSIONED_CONTEXT_V2" if context_go else "NO_GO_CURRENT_EVIDENCE_INSUFFICIENT",
        "neighbor_zero_positive_dose_reduction": neighbor_reduction,
        "neighbor_zero_reduction_threshold_met": neighbor_reduction >= int(context_rule["neighbor_zero_reduction_doses"]),
        "retraining_rule_components": {"ego_condition": ego_condition, "handcrafted_condition": hand_condition, "neighbor_zero_condition": neighbor_condition},
        "retraining_decision": "GO_PREPARE_SEPARATELY_VERSIONED_TRAINING_PROTOCOL" if retraining_go else str(rule["otherwise"]),
        "go_does_not_authorize_training": True,
    }


def build_report(rows: list[dict[str, Any]], associations: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# Stage 6L context-quality 表示消融中文主报告", "", "## 结论", "",
        f"- context-v2：`{decision['context_v2_decision']}`",
        f"- 立即扩大Waymo重训：`{decision['retraining_decision']}`",
        "- GO仅表示准备独立版本协议，不授权覆盖当前checkpoint。", "",
        "## Overall剂量曲线", "",
        "| 表示 | 剂量 | MMD² | null q95 | ratio | Z_BDD | raw p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rep in REPRESENTATIONS:
        for row in sorted([r for r in rows if r["representation"] == rep and r["scope"] == "overall"], key=lambda r: r["nominal_dose"]):
            lines.append(
                f"| {REP_LABELS_ZH[rep]} | {100*float(row['nominal_dose']):.0f}% | {float(row['mmd2']):.8f} | "
                f"{float(row['paired_null_q95']):.8f} | {float(row['bdd_to_null_q95_ratio']):.3f} | "
                f"{float(row['null_standardized_z_bdd']):.3f} | {float(row['raw_p']):.8g} | {float(row['holm_p']):.8g} |"
            )
    lines += [
        "", "不同表示各自使用独立bandwidth与null；上表raw MMD²不能跨表示比较。可比较的是各自null标准化证据、Holm结论和检出剂量。", "",
        "## Task检出单元", "",
        "| 表示 | 12个task×dose中Holm通过数 | 最小overall检出剂量 | median overall Z_BDD |",
        "|---|---:|---:|---:|",
    ]
    for rep in REPRESENTATIONS:
        minimum = decision["minimum_detectable_nominal_dose"][rep]
        minimum_text = "未检出" if minimum is None else f"{100*minimum:.0f}%"
        lines.append(f"| {REP_LABELS_ZH[rep]} | {decision['task_dose_holm_pass_cells_out_of_12'][rep]} | {minimum_text} | {decision['median_overall_z_bdd'][rep]:.3f} |")
    lines += [
        "", "## Context-quality诊断", "",
        "fallback/ambiguity均为rollout后变量。以下关联只描述测量质量与表示距离/BDD pair贡献共同变化，不用于删样本、重加权或因果调整。", "",
        "| 表示 | fallback正关联剂量（task-adjusted CI不跨0） |",
        "|---|---|",
    ]
    for rep in ["learned64_full_context", "learned64_neighbor_zero_input"]:
        values = decision["positive_fallback_association_doses"][rep]
        lines.append(f"| {REP_LABELS_ZH[rep]} | {', '.join(values) if values else '无'} |")
    lines += [
        "", "## 解释边界", "",
        "- B是同checkpoint邻车通道置零，不是重新训练的ego-only learned model。",
        "- C/D使用dose100保守planner参考行拟合median/IQR scaler。",
        "- 结果不建立跨数据集通用BDD阈值。",
        "- 本报告不推断安全性、planner优劣或真实发布可靠性。",
        "- geometric-only context重建不属于本次A–D primary，需在本诊断后另行冻结为supplement。", "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    freeze = read_json(args.freeze_manifest.resolve())
    addendum_manifest = read_json(args.decision_addendum_manifest.resolve())
    rep_manifest = read_json(args.representation_dir.resolve() / "stage6l_representation_manifest.json")
    if freeze.get("status") != "FROZEN_BEFORE_STAGE6L_REPRESENTATION_ABLATION":
        raise ValueError("Invalid Stage 6L freeze")
    if addendum_manifest.get("status") != "FROZEN_BEFORE_STAGE6L_REPRESENTATION_BDD_READ" or addendum_manifest.get("representation_bdd_read") is not False:
        raise ValueError("Invalid Stage 6L decision addendum")
    if addendum_manifest.get("representation_manifest_sha256") != sha256_file(args.representation_dir.resolve() / "stage6l_representation_manifest.json"):
        raise ValueError("Representation manifest differs from decision freeze")
    if rep_manifest.get("status") != "STAGE6L_A_D_REPRESENTATIONS_READY":
        raise ValueError("Stage 6L representations are not ready")
    design = freeze["design"]
    stats = design["statistics"]
    paired_repetitions = int(stats["paired_permutations"])
    cluster_repetitions = int(stats["log_cluster_permutations"])
    task_config = read_json(args.stage6j_bdd_config.resolve())["task_conditioned_secondary"]["tasks"]
    diagnostic_spec = design["pair_diagnostics"]

    results: list[dict[str, Any]] = []
    cluster_results: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    null_samples: dict[str, np.ndarray] = {}
    canonical_tokens: list[str] | None = None
    for rep_index, rep in enumerate(REPRESENTATIONS):
        for dose_index, (label, dose, planner_a) in enumerate(DOSES):
            values_path = args.representation_dir.resolve() / "representations" / rep / f"{label}.npy"
            metadata_path = args.representation_dir.resolve() / "metadata" / f"{label}.csv"
            values = np.asarray(np.load(values_path, mmap_mode="r"), dtype=np.float64)
            metadata = pd.read_csv(metadata_path)
            pairs, tokens, logs = build_pairs(metadata, planner_a)
            if canonical_tokens is None:
                canonical_tokens = tokens
            elif tokens != canonical_tokens:
                raise ValueError(f"Scenario token order differs: {rep}/{label}")
            task_masks, task_names = build_task_masks(metadata, pairs, task_config)
            scopes = [("overall", np.ones(len(pairs), dtype=bool)), *task_masks.items()]
            overall_contribution = None
            for scope_index, (scope, mask) in enumerate(scopes):
                selected = pairs[np.asarray(mask, dtype=bool)]
                result, samples, contribution = kernel_analysis(
                    values[selected[:, 0]], values[selected[:, 1]],
                    repetitions=paired_repetitions,
                    seed=int(stats["paired_seed"]) + rep_index * 1000 + dose_index * 100 + scope_index,
                )
                row = {
                    "representation": rep,
                    "dose_label": label,
                    "nominal_dose": dose,
                    "scope": scope,
                    "role": "overall_primary_within_representation" if scope == "overall" else "task_conditioned_secondary_within_representation",
                    **result,
                    "holm_p": math.nan,
                    "reject_holm_0_05": False,
                }
                results.append(row)
                null_samples[f"{rep}__{label}__{scope}__paired"] = samples
                if scope == "overall":
                    overall_contribution = contribution
            cluster, cluster_samples, _ = kernel_analysis(
                values[pairs[:, 0]], values[pairs[:, 1]],
                repetitions=cluster_repetitions,
                seed=int(stats["log_cluster_seed"]) + rep_index * 100 + dose_index,
                logs=logs,
            )
            cluster_results.append({
                "representation": rep, "dose_label": label, "nominal_dose": dose,
                "role": "log_cluster_sensitivity_within_representation", **cluster,
                "holm_p": math.nan, "reject_holm_0_05": False,
            })
            null_samples[f"{rep}__{label}__overall__log_cluster"] = cluster_samples
            if overall_contribution is None:
                raise AssertionError("Missing overall contribution")
            fallback, ambiguity = pair_quality(args, label, metadata, pairs)
            tasks_by_pair = []
            for position in range(len(pairs)):
                matched = [name for name, mask in task_masks.items() if bool(mask[position])]
                if len(matched) != 1:
                    raise ValueError(f"Pair does not map to exactly one frozen task: {label}/{position}/{matched}")
                tasks_by_pair.append(matched[0])
            for position, ((row_a, row_b), token, log_name, task) in enumerate(zip(pairs, tokens, logs, tasks_by_pair)):
                pair_rows.append({
                    "representation": rep,
                    "dose_label": label,
                    "nominal_dose": dose,
                    "pair_position": position,
                    "scenario_token": token,
                    "log_name": log_name,
                    "task": task,
                    "row_A": int(row_a),
                    "row_B": int(row_b),
                    "pair_l2_distance": float(np.linalg.norm(values[row_a] - values[row_b])),
                    "signed_kernel_contribution": float(overall_contribution[position]),
                    "absolute_kernel_contribution": float(abs(overall_contribution[position])),
                    "max_pair_fallback_rate": float(fallback[position]),
                    "max_pair_ambiguity_rate": float(ambiguity[position]),
                })
    apply_holm(results, cluster_results)

    associations: list[dict[str, Any]] = []
    for rep_index, rep in enumerate(REPRESENTATIONS):
        for dose_index, (label, dose, _) in enumerate(DOSES):
            current = [row for row in pair_rows if row["representation"] == rep and row["dose_label"] == label]
            logs = [str(row["log_name"]) for row in current]
            tasks = [str(row["task"]) for row in current]
            for quality_index, quality_metric in enumerate(QUALITY_METRICS):
                x = np.asarray([float(row[quality_metric]) for row in current])
                for outcome_index, outcome in enumerate(QUALITY_OUTCOMES):
                    y = np.asarray([float(row[outcome]) for row in current])
                    for adjusted in [False, True]:
                        estimate, low, high = fixed_rank_log_cluster_bootstrap(
                            x, y, logs, tasks if adjusted else None,
                            repetitions=int(diagnostic_spec["bootstrap_repetitions"]),
                            seed=int(diagnostic_spec["bootstrap_seed"]) + rep_index * 10000 + dose_index * 1000 + quality_index * 100 + outcome_index * 10 + int(adjusted),
                        )
                        associations.append({
                            "representation": rep,
                            "dose_label": label,
                            "nominal_dose": dose,
                            "quality_metric": quality_metric,
                            "outcome": outcome,
                            "analysis": "task_adjusted_fixed_rank_log_cluster_bootstrap" if adjusted else "overall_fixed_rank_log_cluster_bootstrap",
                            "n_pairs": len(current),
                            "n_logs": len(set(logs)),
                            "estimate": estimate,
                            "ci95_low": low,
                            "ci95_high": high,
                            "interval_excludes_zero": bool(low > 0 or high < 0),
                            "role": diagnostic_spec["role"],
                        })
    decision = decision_summary(results, associations, addendum_manifest["addendum"])
    write_csv(output_dir / "stage6l_representation_bdd_results.csv", results, list(results[0]))
    write_csv(output_dir / "stage6l_log_cluster_results.csv", cluster_results, list(cluster_results[0]))
    write_csv(output_dir / "stage6l_pair_diagnostics.csv", pair_rows, list(pair_rows[0]))
    write_csv(output_dir / "stage6l_context_quality_associations.csv", associations, list(associations[0]))
    np.savez_compressed(output_dir / "stage6l_null_samples.npz", **null_samples)
    make_plots(results, associations, output_dir)
    report = build_report(results, associations, decision)
    (output_dir / "stage6l_context_representation_ablation_report_zh.md").write_text(report, encoding="utf-8")
    summary = {
        "schema_version": "stage6l_context_representation_ablation_results_v1",
        "status": "STAGE6L_CONTEXT_REPRESENTATION_ABLATION_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 253,
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest.resolve()),
        "decision_addendum_manifest_sha256": sha256_file(args.decision_addendum_manifest.resolve()),
        "representation_manifest_sha256": sha256_file(args.representation_dir.resolve() / "stage6l_representation_manifest.json"),
        "cross_representation_raw_mmd_comparison_forbidden": True,
        "source_stage6j_k_outputs_modified": False,
        "results": results,
        "log_cluster_results": cluster_results,
        "context_quality_associations": associations,
        "decision": decision,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json(output_dir / "stage6l_context_representation_ablation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
