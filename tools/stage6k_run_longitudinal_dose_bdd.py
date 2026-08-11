#!/usr/bin/env python3
"""Run the frozen Stage 6K paired-BDD longitudinal dose-response analysis."""

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
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6j_run_paired_bdd import build_task_masks  # noqa: E402
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import (  # noqa: E402
    biased_mmd2_from_kernel,
    exact_median_bandwidth,
    holm_adjust,
    permutation_bdd,
    rbf_kernel,
)


SCHEMA_VERSION = "stage6k_longitudinal_dose_paired_bdd_v1"
ADDENDUM_STATUS = "FROZEN_BEFORE_NEW_DOSE_EMBEDDING_OR_BDD_READ"
BASELINE = "pdm_closed_conservative_longitudinal_v1"
DOSES = [
    ("dose25", 0.25, "pdm_closed_assertive_longitudinal_dose25_v1"),
    ("dose50", 0.50, "pdm_closed_assertive_longitudinal_dose50_v1"),
    ("dose75", 0.75, "pdm_closed_assertive_longitudinal_dose75_v1"),
    ("dose100", 1.00, "pdm_closed_assertive_longitudinal_v1"),
]
EXPECTED_CHECKPOINT_SHA256 = "909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 6K frozen four-dose paired BDD analysis.")
    parser.add_argument("--addendum_manifest", type=Path, required=True)
    parser.add_argument("--realized_dose_summary", type=Path, required=True)
    parser.add_argument("--new_embeddings_dir", type=Path, required=True)
    parser.add_argument("--stage6j_embedding_dir", type=Path, required=True)
    parser.add_argument("--stage6j_bdd_dir", type=Path, required=True)
    parser.add_argument("--stage6j_bdd_config", type=Path, default=Path("configs/stage6j_paired_bdd_analysis.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_embedding_dir(path: Path) -> tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    manifest = read_json(path / "embedding_manifest.json")
    expected = {
        "total_rows": 366,
        "embedding_dim": 64,
        "context_layout_used": "stage5d_context_dataset_direct",
        "stage5d_schema_matched": True,
        "context_padded_to_checkpoint_dim": False,
        "nonfinite_embedding_values": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path} embedding manifest violates {key}={value!r}")
    embedding = np.asarray(np.load(path / "embedding.npy", mmap_mode="r"), dtype=np.float64)
    metadata = pd.read_csv(path / "metadata.csv")
    if embedding.shape != (366, 64) or len(metadata) != 366 or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid embedding/metadata in {path}: {embedding.shape}/{len(metadata)}")
    return embedding, metadata, manifest


def build_pairs(metadata: pd.DataFrame, planner_a: str) -> tuple[np.ndarray, List[str], List[str]]:
    required = {"global_row", "scenario_token", "planner_name", "log_name", "scenario_type"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing pair columns: {missing}")
    pairs: List[tuple[int, int]] = []
    tokens: List[str] = []
    logs: List[str] = []
    for token, frame in metadata.groupby("scenario_token", sort=False):
        a = frame.loc[frame["planner_name"].astype(str) == planner_a]
        b = frame.loc[frame["planner_name"].astype(str) == BASELINE]
        if len(a) != 1 or len(b) != 1 or len(frame) != 2:
            raise ValueError(f"Scenario {token} is not one complete {planner_a}/{BASELINE} pair")
        if str(a.iloc[0]["log_name"]) != str(b.iloc[0]["log_name"]):
            raise ValueError(f"Scenario {token} has unequal log_name")
        if str(a.iloc[0]["scenario_type"]) != str(b.iloc[0]["scenario_type"]):
            raise ValueError(f"Scenario {token} has unequal scenario_type")
        pairs.append((int(a.iloc[0]["global_row"]), int(b.iloc[0]["global_row"])))
        tokens.append(str(token))
        logs.append(str(a.iloc[0]["log_name"]))
    result = np.asarray(pairs, dtype=np.int64)
    if result.shape != (183, 2) or set(result.ravel()) != set(range(366)):
        raise ValueError(f"Expected 183 exhaustive pairs, got {result.shape}")
    return result, tokens, logs


def null_diagnostics(observed: float, samples: np.ndarray) -> Dict[str, float]:
    source = np.asarray(samples, dtype=np.float64)
    mean = float(np.mean(source))
    std = float(np.std(source, ddof=1))
    q95 = float(np.quantile(source, 0.95))
    return {
        "paired_null_mean": mean,
        "paired_null_sd": std,
        "paired_null_q95": q95,
        "bdd_to_null_q95_ratio": float(observed / q95) if q95 > 0 else math.inf,
        "null_standardized_z_bdd": float((observed - mean) / std) if std > 0 else math.inf,
    }


def cluster_randomization_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    logs: Sequence[str],
    *,
    repetitions: int,
    seed: int,
    progress_label: str,
) -> tuple[Dict[str, Any], np.ndarray]:
    if values_a.shape != values_b.shape or len(values_a) != len(logs):
        raise ValueError("cluster randomization inputs differ in shape")
    unique_logs, inverse = np.unique(np.asarray(logs, dtype=str), return_inverse=True)
    n_pairs = len(values_a)
    pooled = np.vstack([values_a, values_b]).astype(np.float64, copy=False)
    bandwidth = exact_median_bandwidth(pooled)
    kernel = rbf_kernel(pooled, bandwidth)
    index_a = np.arange(n_pairs, dtype=np.int64)
    index_b = np.arange(n_pairs, 2 * n_pairs, dtype=np.int64)
    observed = biased_mmd2_from_kernel(kernel, index_a, index_b)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for position in tqdm(range(repetitions), desc=progress_label, unit="perm", leave=False):
        swap = rng.integers(0, 2, size=len(unique_logs)).astype(bool)[inverse]
        candidate_a = np.where(swap, index_b, index_a)
        candidate_b = np.where(swap, index_a, index_b)
        samples[position] = biased_mmd2_from_kernel(kernel, candidate_a, candidate_b)
    exceedance = int(np.sum(samples >= observed))
    result = {
        "mmd2": observed,
        "bandwidth": bandwidth,
        "n_pairs": n_pairs,
        "n_clusters": len(unique_logs),
        "permutations": repetitions,
        "exceedance_count": exceedance,
        "raw_p": float((exceedance + 1) / (repetitions + 1)),
        "randomization": "all scenario pairs sharing log_name flip together",
    }
    result.update(null_diagnostics(observed, samples))
    return result, samples


def primary_row(
    label: str, dose: float, scope: str, role: str, result: Mapping[str, Any], samples: np.ndarray
) -> Dict[str, Any]:
    observed = float(result["mmd2"])
    raw_p = float(result.get("p_value", result.get("raw_p")))
    row = {
        "dose_label": label,
        "nominal_dose": dose,
        "scope": scope,
        "role": role,
        "n_pairs": int(result.get("n_A", result.get("n_pairs"))),
        "mmd2": observed,
        "bandwidth": float(result["bandwidth"]),
        "exceedance_count": int(result["exceedance_count"]),
        "raw_p": raw_p,
        "holm_p": math.nan,
        "reject_holm_0_05": False,
        "permutations": int(result.get("permutations", len(samples))),
    }
    row.update(null_diagnostics(observed, samples))
    return row


def apply_frozen_multiplicity(
    rows: Sequence[Dict[str, Any]], gate_by_label: Mapping[str, bool]
) -> tuple[float | None, List[Dict[str, Any]]]:
    overall = [row for row in rows if row["scope"] == "overall"]
    tasks = [row for row in rows if row["scope"] != "overall"]
    if len(overall) != 4 or len(tasks) != 12:
        raise ValueError(f"Expected 4 overall and 12 task-dose rows, got {len(overall)}/{len(tasks)}")
    for family in (overall, tasks):
        for row, adjusted in zip(family, holm_adjust([float(item["raw_p"]) for item in family])):
            row["holm_p"] = float(adjusted)
            row["reject_holm_0_05"] = bool(adjusted < 0.05)
    detectable = sorted(
        float(row["nominal_dose"])
        for row in overall
        if row["reject_holm_0_05"] and gate_by_label.get(str(row["dose_label"]), False)
    )
    return (detectable[0] if detectable else None), list(rows)


def plot_results(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    overall = sorted([row for row in rows if row["scope"] == "overall"], key=lambda row: float(row["nominal_dose"]))
    x = np.asarray([100 * float(row["nominal_dose"]) for row in overall])
    mmd = np.asarray([float(row["mmd2"]) for row in overall])
    q95 = np.asarray([float(row["paired_null_q95"]) for row in overall])
    z = np.asarray([float(row["null_standardized_z_bdd"]) for row in overall])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    axes[0].plot(x, mmd, marker="o", label="Observed BDD / MMD²")
    axes[0].plot(x, q95, marker="s", linestyle="--", label="Paired-null q95")
    axes[0].set(xlabel="Nominal longitudinal dose (%)", ylabel="MMD²", title="Raw BDD relative to its dose-specific null")
    axes[0].legend()
    axes[1].plot(x, z, marker="o")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Nominal longitudinal dose (%)", ylabel="Z_BDD", title="Null-standardized BDD")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output_dir / "stage6k_bdd_dose_curve.png", dpi=180)
    fig.savefig(output_dir / "stage6k_bdd_dose_curve.pdf")
    plt.close(fig)


def build_report(
    rows: Sequence[Mapping[str, Any]], cluster_rows: Sequence[Mapping[str, Any]], minimum: float | None
) -> str:
    overall = sorted([row for row in rows if row["scope"] == "overall"], key=lambda row: float(row["nominal_dose"]))
    tasks = sorted([row for row in rows if row["scope"] != "overall"], key=lambda row: (float(row["nominal_dose"]), str(row["scope"])))
    conclusion = "未找到同时通过运动学门禁和BDD Holm检验的剂量" if minimum is None else f"最小可检出名义剂量为 {minimum * 100:.0f}%"
    lines = [
        "# Stage 6K 纵向风格 BDD 剂量—响应中文报告", "", f"## 主要结论：{conclusion}", "",
        "BDD绝对值不采用通用阈值；每一档均相对于其预冻结pair-swap零分布解释，并对四档overall统一Holm校正。", "",
        "| 剂量 | BDD/MMD² | null q95 | BDD/q95 | Z_BDD | raw p | Holm p | 结论 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in overall:
        lines.append(f"| {100*float(row['nominal_dose']):.0f}% | {float(row['mmd2']):.8f} | {float(row['paired_null_q95']):.8f} | {float(row['bdd_to_null_q95_ratio']):.3f} | {float(row['null_standardized_z_bdd']):.3f} | {float(row['raw_p']):.8g} | {float(row['holm_p']):.8g} | {'检出' if row['reject_holm_0_05'] else '未检出'} |")
    lines += ["", "## Task×dose次要检验（12项统一Holm）", "", "| 剂量 | task | pair数 | BDD | raw p | Holm p | 结论 |", "|---:|---|---:|---:|---:|---:|---|"]
    for row in tasks:
        lines.append(f"| {100*float(row['nominal_dose']):.0f}% | {row['scope']} | {row['n_pairs']} | {float(row['mmd2']):.8f} | {float(row['raw_p']):.8g} | {float(row['holm_p']):.8g} | {'检出' if row['reject_holm_0_05'] else '未检出'} |")
    lines += ["", "## 同log整体翻转敏感性", "", "| 剂量 | log数 | raw p | Holm p | 结论 |", "|---:|---:|---:|---:|---|"]
    for row in sorted(cluster_rows, key=lambda item: float(item["nominal_dose"])):
        lines.append(f"| {100*float(row['nominal_dose']):.0f}% | {row['n_clusters']} | {float(row['raw_p']):.8g} | {float(row['holm_p']):.8g} | {'稳健' if row['reject_holm_0_05'] else '不稳健'} |")
    lines += [
        "", "## 解释边界", "",
        "- 最小可检出剂量是本checkpoint、本场景池、本planner干预和本统计设计下的实验结果，不是通用BDD阈值。",
        "- nominal dose是IDM参数插值；不能直接等同于真实车辆风格差异强度。",
        "- overall pair-swap是primary；task和log-cluster结果分别是secondary与supplementary。",
        "- 结果支持受控同场景纯纵向风格检出，不直接证明异场景整车发布可靠性。", "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    addendum = read_json(args.addendum_manifest.resolve())
    if addendum.get("status") != ADDENDUM_STATUS or addendum.get("new_dose_embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6K addendum was not frozen before new-dose BDD read")
    spec = addendum["analysis_specification"]
    if int(spec["paired_bdd_primary"]["permutations"]) != 100000:
        raise ValueError("Frozen paired BDD repetitions changed")
    if sha256_file(args.checkpoint.resolve()) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Waymo checkpoint SHA-256 differs")
    realized = read_json(args.realized_dose_summary.resolve())
    if realized.get("status") != "STAGE6K_REALIZED_DOSE_CURVE_COMPLETE" or realized.get("new_dose_embedding_or_bdd_read") is not False:
        raise ValueError("Realized-dose analysis is incomplete or not BDD-blind")
    gate_by_label = {str(row["dose_label"]): bool(row["kinematic_gate_passed"]) for row in realized["gate_decisions"]}
    task_config = read_json(args.stage6j_bdd_config.resolve())["task_conditioned_secondary"]
    repetitions = int(spec["paired_bdd_primary"]["permutations"])
    seed = int(spec["paired_bdd_primary"]["random_seed"])
    cluster_repetitions = int(spec["log_cluster_sensitivity"]["permutations"])
    cluster_seed = int(spec["log_cluster_sensitivity"]["random_seed"])
    endpoint_summary = read_json(args.stage6j_bdd_dir.resolve() / "stage6j_paired_bdd_summary.json")
    endpoint_null = np.load(args.stage6j_bdd_dir.resolve() / "stage6j_paired_bdd_null_samples.npz")
    endpoint_by_scope = {"overall": endpoint_summary["primary_result"], **{row["scope"]: row for row in endpoint_summary["task_results"]}}
    rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    null_samples: Dict[str, np.ndarray] = {}
    canonical_tokens: List[str] | None = None
    input_hashes: Dict[str, Any] = {}
    for label, dose, planner_a in DOSES:
        embedding_dir = args.stage6j_embedding_dir.resolve() if label == "dose100" else args.new_embeddings_dir.resolve() / label
        embedding, metadata, _ = validate_embedding_dir(embedding_dir)
        pair_indices, tokens, logs = build_pairs(metadata, planner_a)
        if canonical_tokens is None:
            canonical_tokens = tokens
        elif tokens != canonical_tokens:
            raise ValueError(f"Scenario token order differs for {label}")
        task_masks, _ = build_task_masks(metadata, pair_indices, task_config["tasks"])
        scopes = [("overall", np.ones(len(pair_indices), dtype=bool)), *task_masks.items()]
        for scope_index, (scope, mask) in enumerate(scopes):
            selected = pair_indices[np.asarray(mask, dtype=bool)]
            if label == "dose100":
                result = endpoint_by_scope[scope]
                samples = np.asarray(endpoint_null[scope], dtype=np.float64)
            else:
                result, samples = permutation_bdd(
                    embedding[selected[:, 0]], embedding[selected[:, 1]], repetitions=repetitions,
                    seed=seed + scope_index, paired_swap=True,
                    progress_label=f"Stage6K {label} {scope} paired BDD",
                )
            rows.append(primary_row(label, dose, scope, "primary" if scope == "overall" else "task_conditioned_secondary", result, samples))
            null_samples[f"{label}_{scope}"] = samples
        cluster_result, cluster_samples = cluster_randomization_test(
            embedding[pair_indices[:, 0]], embedding[pair_indices[:, 1]], logs,
            repetitions=cluster_repetitions, seed=cluster_seed,
            progress_label=f"Stage6K {label} log-cluster BDD",
        )
        cluster_rows.append({"dose_label": label, "nominal_dose": dose, **cluster_result})
        null_samples[f"{label}_log_cluster_overall"] = cluster_samples
        input_hashes[label] = {
            "embedding_manifest_sha256": sha256_file(embedding_dir / "embedding_manifest.json"),
            "embedding_sha256": sha256_file(embedding_dir / "embedding.npy"),
            "metadata_sha256": sha256_file(embedding_dir / "metadata.csv"),
        }
    minimum, rows = apply_frozen_multiplicity(rows, gate_by_label)
    for row, adjusted in zip(cluster_rows, holm_adjust([float(item["raw_p"]) for item in cluster_rows])):
        row["holm_p"] = float(adjusted)
        row["reject_holm_0_05"] = bool(adjusted < 0.05)
    result_fields = list(rows[0])
    cluster_fields = list(cluster_rows[0])
    write_csv(output_dir / "stage6k_bdd_dose_results.csv", rows, result_fields)
    write_csv(output_dir / "stage6k_log_cluster_sensitivity.csv", cluster_rows, cluster_fields)
    np.savez_compressed(output_dir / "stage6k_bdd_null_samples.npz", **null_samples)
    plot_results(rows, output_dir)
    report_path = output_dir / "stage6k_bdd_dose_report_zh.md"
    report_path.write_text(build_report(rows, cluster_rows, minimum), encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE6K_BDD_DOSE_RESPONSE_COMPLETE",
        "minimum_detectable_nominal_dose": minimum,
        "minimum_detectable_rule": spec["primary_overall_dose_family"]["minimum_detectable_nominal_dose_rule"],
        "kinematic_gate_by_dose": gate_by_label,
        "overall_results": [row for row in rows if row["scope"] == "overall"],
        "task_results": [row for row in rows if row["scope"] != "overall"],
        "log_cluster_sensitivity": cluster_rows,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "input_hashes": input_hashes,
        "addendum_manifest_sha256": sha256_file(args.addendum_manifest.resolve()),
        "realized_dose_summary_sha256": sha256_file(args.realized_dose_summary.resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "results_csv": "stage6k_bdd_dose_results.csv",
            "cluster_csv": "stage6k_log_cluster_sensitivity.csv",
            "null_samples": "stage6k_bdd_null_samples.npz",
            "figure_png": "stage6k_bdd_dose_curve.png",
            "figure_pdf": "stage6k_bdd_dose_curve.pdf",
            "report_zh": report_path.name,
        },
    }
    write_json(output_dir / "stage6k_bdd_dose_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
