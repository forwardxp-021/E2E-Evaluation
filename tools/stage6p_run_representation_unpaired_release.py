#!/usr/bin/env python3
"""Run the frozen Stage 6P representation x unpaired-release experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6l_prepare_context_representation_ablation import (
    apply_scaler,
    ego_kinematic_features,
)
from tools.stage7e_embed_stage6_dataset import embed_context, load_checkpoint


STATUS = "STAGE6P_REPRESENTATION_UNPAIRED_RELEASE_COMPLETE"
IDENTITY = [
    "target_scenarios_per_release",
    "experiment_set",
    "family",
    "repetition",
    "split_seed",
    "planner_A",
    "planner_B",
]
REPRESENTATIONS = ["full64", "ego13", "handcrafted46", "neighbor_zero64"]


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def higher_quantile(values: Iterable[float], quantile: float) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan
    return float(np.quantile(finite, quantile, method="higher"))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def validate_design(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "stage6p_representation_unpaired_release_design_v1":
        raise ValueError("not a Stage6P v1 config")
    if [row["id"] for row in config["representations"]] != REPRESENTATIONS:
        raise ValueError("frozen representation order changed")
    if config["statistic"].get("cross_representation_raw_mmd2_comparison_forbidden") is not True:
        raise ValueError("cross-representation raw MMD prohibition missing")


def source_context_for_pool(source_pool: str, args: argparse.Namespace) -> Path:
    if source_pool == "existing_310":
        return args.context_existing
    if source_pool == "stage6g_490":
        return args.context_expanded
    raise ValueError(f"unknown source_pool: {source_pool}")


def build_representations(args: argparse.Namespace, pool: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    full = np.asarray(np.load(args.embedding_pool / "embedding.npy", mmap_mode="r"), dtype=np.float32)
    if full.shape != (1600, 64):
        raise ValueError(f"unexpected full embedding shape: {full.shape}")
    scaler = np.load(args.scaler)
    ego_features = np.empty((len(pool), 13), dtype=np.float64)
    interaction = np.empty((len(pool), 33), dtype=np.float64)
    neighbor_zero = np.empty((len(pool), 64), dtype=np.float32)
    checkpoint = load_checkpoint(args.checkpoint)
    alignment_rows: list[dict[str, Any]] = []
    for source_pool in ["existing_310", "stage6g_490"]:
        positions = np.flatnonzero(pool["source_pool"].astype(str).to_numpy() == source_pool)
        context_dir = source_context_for_pool(source_pool, args)
        source_meta = pd.read_csv(context_dir / "metadata.csv").sort_values("global_row").reset_index(drop=True)
        source_rows = pool.iloc[positions]["source_global_row"].astype(int).to_numpy()
        selected_meta = source_meta.iloc[source_rows].reset_index(drop=True)
        selected_pool = pool.iloc[positions].reset_index(drop=True)
        for column in ["scenario_token", "planner_name", "log_name"]:
            if selected_meta[column].astype(str).tolist() != selected_pool[column].astype(str).tolist():
                raise ValueError(f"row alignment failed for {source_pool}/{column}")
        context = np.asarray(np.load(context_dir / "context_traj.npy", mmap_mode="r")[source_rows], dtype=np.float32)
        ego = np.asarray(np.load(context_dir / "ego_seq.npy", mmap_mode="r")[source_rows], dtype=np.float32)
        mask = np.asarray(np.load(context_dir / "ego_seq_mask.npy", mmap_mode="r")[source_rows], dtype=bool)
        current_interaction = np.asarray(
            np.load(context_dir / "interaction_feat_style.npy", mmap_mode="r")[source_rows], dtype=np.float64
        )
        if context.shape[1:] != (150, 83) or current_interaction.shape[1] != 33:
            raise ValueError(f"unexpected context shape for {source_pool}: {context.shape}/{current_interaction.shape}")
        ego_features[positions] = ego_kinematic_features(ego, mask)
        interaction[positions] = current_interaction
        context[:, :, 8:83] = 0.0
        embedded, metadata = embed_context(context, checkpoint, args.batch_size, args.device)
        neighbor_zero[positions] = embedded
        alignment_rows.append(
            {
                "source_pool": source_pool,
                "rows": int(len(positions)),
                "source_metadata": str(context_dir / "metadata.csv"),
                "encoder_metadata": metadata,
            }
        )
    ego13 = apply_scaler(ego_features, scaler["ego_median"], scaler["ego_scale"])
    combined = np.concatenate([ego_features, interaction], axis=1)
    handcrafted46 = apply_scaler(combined, scaler["combined_median"], scaler["combined_scale"])
    arrays = {
        "full64": full,
        "ego13": ego13,
        "handcrafted46": handcrafted46,
        "neighbor_zero64": neighbor_zero,
    }
    for rep, expected_dim in [("full64", 64), ("ego13", 13), ("handcrafted46", 46), ("neighbor_zero64", 64)]:
        if arrays[rep].shape != (1600, expected_dim) or not np.isfinite(arrays[rep]).all():
            raise ValueError(f"invalid representation {rep}: {arrays[rep].shape}")
    return arrays, {"row_alignment": alignment_rows}


def median_bandwidth(values: np.ndarray, seed: int, max_pairs: int) -> float:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(values), size=max_pairs)
    right = rng.integers(0, len(values), size=max_pairs)
    keep = left != right
    distances = np.linalg.norm(values[left[keep]].astype(np.float64) - values[right[keep]].astype(np.float64), axis=1)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if not len(distances):
        raise ValueError("no positive distances for bandwidth")
    return float(np.median(distances))


def rbf_kernel(values: np.ndarray, bandwidth: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    squared = np.sum(x * x, axis=1, keepdims=True) + np.sum(x * x, axis=1)[None, :] - 2.0 * (x @ x.T)
    np.maximum(squared, 0.0, out=squared)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth)).astype(np.float32)


def build_trials(assignments: pd.DataFrame) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for identity, group in assignments.groupby(IDENTITY, sort=False, dropna=False):
        base = dict(zip(IDENTITY, identity))
        releases = {
            label: sorted(set(group.loc[group["release_group"] == label, "log_name"].astype(str)))
            for label in ["A", "B"]
        }
        if not releases["A"] or not releases["B"] or set(releases["A"]) & set(releases["B"]):
            raise ValueError(f"invalid log split: {base}")
        trials.append({**base, "logs_A": releases["A"], "logs_B": releases["B"]})
    if len(trials) != 2400:
        raise ValueError(f"expected 2400 frozen trials, got {len(trials)}")
    return trials


def row_indices(pool: pd.DataFrame, planner: str, logs: list[str]) -> np.ndarray:
    mask = (pool["planner_name"].astype(str).to_numpy() == str(planner)) & pool["log_name"].astype(str).isin(logs).to_numpy()
    return np.flatnonzero(mask)


def compute_trials(rep: str, kernel: np.ndarray, pool: pd.DataFrame, trials: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tokens = pool["scenario_token"].astype(str).to_numpy()
    for trial in tqdm(trials, desc=f"Stage6P {rep}", unit="trial"):
        ia = row_indices(pool, str(trial["planner_A"]), trial["logs_A"])
        ib = row_indices(pool, str(trial["planner_B"]), trial["logs_B"])
        if not len(ia) or not len(ib):
            raise ValueError(f"empty release group: {trial}")
        overlap = set(tokens[ia]) & set(tokens[ib])
        if overlap:
            raise ValueError(f"scenario token overlap in frozen trial: {len(overlap)}")
        statistic = float(max(0.0, kernel[np.ix_(ia, ia)].mean() + kernel[np.ix_(ib, ib)].mean() - 2.0 * kernel[np.ix_(ia, ib)].mean()))
        rows.append(
            {
                **{column: trial[column] for column in IDENTITY},
                "representation": rep,
                "n_A": int(len(ia)),
                "n_B": int(len(ib)),
                "log_count_A": int(len(trial["logs_A"])),
                "log_count_B": int(len(trial["logs_B"])),
                "scenario_overlap": 0,
                "raw_mmd2_within_representation_only": statistic,
            }
        )
    return pd.DataFrame(rows)


def calibrate(trials: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, Any]] = []
    evaluated: list[pd.DataFrame] = []
    for (rep, size), group in trials.groupby(["representation", "target_scenarios_per_release"], sort=False):
        calibration = group.loc[group["family"] == "AA_CALIBRATION", "raw_mmd2_within_representation_only"]
        threshold = higher_quantile(calibration, 1.0 - alpha)
        threshold_rows.append(
            {
                "representation": rep,
                "target_scenarios_per_release": int(size),
                "calibration_trials": int(len(calibration)),
                "alpha": alpha,
                "q95_threshold_within_representation_only": threshold,
                "quantile_method": "higher",
            }
        )
        current = group.copy()
        current["threshold_within_representation_only"] = threshold
        current["statistic_to_own_threshold_ratio"] = current["raw_mmd2_within_representation_only"] / threshold
        current["alert"] = current["raw_mmd2_within_representation_only"] > threshold
        evaluated.append(current)
    return pd.DataFrame(threshold_rows), pd.concat(evaluated, ignore_index=True)


def operating_characteristics(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    for (rep, size), group in evaluated.groupby(["representation", "target_scenarios_per_release"], sort=False):
        aa = group.loc[group["family"] == "AA_EVALUATION"]
        ab = group.loc[group["family"] == "AB_EVALUATION"]
        fp, det = int(aa["alert"].sum()), int(ab["alert"].sum())
        fp_ci, det_ci = wilson_interval(fp, len(aa)), wilson_interval(det, len(ab))
        pooled.append(
            {
                "representation": rep,
                "target_scenarios_per_release": int(size),
                "aa_holdout_trials": int(len(aa)),
                "aa_false_positive_count": fp,
                "aa_false_positive_rate": fp / len(aa),
                "aa_wilson95_low": fp_ci[0],
                "aa_wilson95_high": fp_ci[1],
                "ab_trials": int(len(ab)),
                "ab_detection_count": det,
                "ab_detection_rate": det / len(ab),
                "ab_wilson95_low": det_ci[0],
                "ab_wilson95_high": det_ci[1],
                "detection_minus_false_positive": det / len(ab) - fp / len(aa),
            }
        )
        for experiment_set, current in ab.groupby("experiment_set", sort=False):
            count = int(current["alert"].sum())
            low, high = wilson_interval(count, len(current))
            directions.append(
                {
                    "representation": rep,
                    "target_scenarios_per_release": int(size),
                    "experiment_set": experiment_set,
                    "trials": int(len(current)),
                    "detection_count": count,
                    "detection_rate": count / len(current),
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )
    return pd.DataFrame(pooled), pd.DataFrame(directions)


def paired_comparisons(evaluated: pd.DataFrame) -> pd.DataFrame:
    ab = evaluated.loc[evaluated["family"] == "AB_EVALUATION"]
    pivot = ab.pivot(index=IDENTITY, columns="representation", values="alert").reset_index()
    rows: list[dict[str, Any]] = []
    for size, group in pivot.groupby("target_scenarios_per_release", sort=True):
        reference = group["full64"].astype(bool)
        for candidate_name in ["ego13", "handcrafted46", "neighbor_zero64"]:
            candidate = group[candidate_name].astype(bool)
            gained = int((candidate & ~reference).sum())
            lost = int((~candidate & reference).sum())
            discordant = gained + lost
            rows.append(
                {
                    "target_scenarios_per_release": int(size),
                    "reference": "full64",
                    "candidate": candidate_name,
                    "paired_trials": int(len(group)),
                    "reference_detection_rate": float(reference.mean()),
                    "candidate_detection_rate": float(candidate.mean()),
                    "candidate_minus_reference": float(candidate.mean() - reference.mean()),
                    "candidate_only_alerts": gained,
                    "reference_only_alerts": lost,
                    "mcnemar_exact_two_sided_p": float(binomtest(gained, discordant, 0.5).pvalue) if discordant else 1.0,
                }
            )
    return pd.DataFrame(rows)


def plot_results(output_dir: Path, operating: pd.DataFrame) -> None:
    labels = {"full64": "full learned64", "ego13": "ego kinematic 13D", "handcrafted46": "handcrafted 46D", "neighbor_zero64": "neighbor-zero 64D (diagnostic)"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)
    for rep, group in operating.groupby("representation", sort=False):
        group = group.sort_values("target_scenarios_per_release")
        x = group["target_scenarios_per_release"]
        axes[0].plot(x, group["aa_false_positive_rate"], marker="o", label=labels[rep])
        axes[1].plot(x, group["ab_detection_rate"], marker="o", label=labels[rep])
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("A/A holdout false-positive rate")
    axes[1].set_title("A/B detection rate")
    for ax in axes:
        ax.set_xlabel("Scenarios per release")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical rate")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "stage6p_representation_unpaired_reliability.png", dpi=180)
    fig.savefig(output_dir / "stage6p_representation_unpaired_reliability.pdf")
    plt.close(fig)


def write_report(output_dir: Path, operating: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    lines = [
        "# Stage 6P Representation × Unpaired Release 中文报告",
        "",
        "## 结论口径",
        "",
        "本实验只比较每种representation经自身A/A校准后的FPR、A/B检出率和同trial告警；禁止跨representation比较raw MMD²。",
        "",
        "## 可靠性结果",
        "",
        "| representation | n/版本 | A/A holdout FPR (Wilson 95%) | A/B detection (Wilson 95%) | detection-FPR |",
        "|---|---:|---|---|---:|",
    ]
    for _, row in operating.sort_values(["target_scenarios_per_release", "representation"]).iterrows():
        lines.append(
            f"| {row['representation']} | {int(row['target_scenarios_per_release'])} | "
            f"{row['aa_false_positive_rate']:.3f} [{row['aa_wilson95_low']:.3f}, {row['aa_wilson95_high']:.3f}] | "
            f"{row['ab_detection_rate']:.3f} [{row['ab_wilson95_low']:.3f}, {row['ab_wilson95_high']:.3f}] | "
            f"{row['detection_minus_false_positive']:+.3f} |"
        )
    lines.extend(["", "## ego13 与 full64 的同release配对比较", "", "| n/版本 | full64 detection | ego13 detection | 差值 | ego13-only / full64-only | McNemar exact p |", "|---:|---:|---:|---:|---:|---:|"])
    ego = comparisons.loc[comparisons["candidate"] == "ego13"].sort_values("target_scenarios_per_release")
    for _, row in ego.iterrows():
        lines.append(
            f"| {int(row['target_scenarios_per_release'])} | {row['reference_detection_rate']:.3f} | "
            f"{row['candidate_detection_rate']:.3f} | {row['candidate_minus_reference']:+.3f} | "
            f"{int(row['candidate_only_alerts'])} / {int(row['reference_only_alerts'])} | {row['mcnemar_exact_two_sided_p']:.4g} |"
        )
    row400 = ego.loc[ego["target_scenarios_per_release"] == 400].iloc[0]
    verdict = "优于" if row400["candidate_minus_reference"] > 0 and row400["mcnemar_exact_two_sided_p"] < 0.05 else "未证明优于"
    lines.extend(
        [
            "",
            "## 核心回答",
            "",
            f"- 在n=400的真正unpaired release条件下，ego13 **{verdict}** full64：检出率差 {row400['candidate_minus_reference']:+.3f}，配对McNemar p={row400['mcnemar_exact_two_sided_p']:.4g}。",
            "- 是否‘更可靠’同时看A/A holdout FPR、A/B detection与detection-FPR，不能只看检出率，更不能看跨representation raw MMD²。",
            "- ego13若更强，只说明当前checkpoint的64D表示在这一受控纵向版本差异上未充分保留ego运动学信号；不代表neighbor/context无用。",
            "- 论文定义保持不变：behavior style = ego response conditioned on traffic / interaction context。",
            "- neighbor-zero64仅为同checkpoint输入消融诊断，不是ego-only训练模型，也不能替代正式候选表示。",
        ]
    )
    (output_dir / "stage6p_representation_unpaired_release_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config.resolve())
    validate_design(config)
    if sha256_file(args.assignments.resolve()) != config["source_contract"]["log_assignments_sha256"]:
        raise ValueError("frozen log assignment SHA-256 mismatch")
    if sha256_file(args.embedding_pool / "metadata.csv") != config["source_contract"]["pool_metadata_sha256"]:
        raise ValueError("frozen pool metadata SHA-256 mismatch")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    rep_dir = output_dir / "representations"
    rep_dir.mkdir()

    pool = pd.read_csv(args.embedding_pool / "metadata.csv").sort_values("global_row").reset_index(drop=True)
    if len(pool) != 1600 or pool["log_name"].nunique() != 489:
        raise ValueError("frozen 800-pair pool contract failed")
    arrays, preparation = build_representations(args, pool)
    assignments = pd.read_csv(args.assignments)
    frozen_trials = build_trials(assignments)
    bandwidth_rows: list[dict[str, Any]] = []
    result_parts: list[pd.DataFrame] = []
    for rep in REPRESENTATIONS:
        path = rep_dir / f"{rep}.npy"
        np.save(path, arrays[rep])
        bandwidth = median_bandwidth(
            arrays[rep], int(config["statistic"]["bandwidth_seed"]), int(config["statistic"]["median_pair_draws"])
        )
        bandwidth_rows.append(
            {
                "representation": rep,
                "dimension": int(arrays[rep].shape[1]),
                "bandwidth": bandwidth,
                "representation_sha256": sha256_file(path),
            }
        )
        kernel = rbf_kernel(arrays[rep], bandwidth)
        result_parts.append(compute_trials(rep, kernel, pool, frozen_trials))
        del kernel
    raw_trials = pd.concat(result_parts, ignore_index=True)
    if len(raw_trials) != 9600:
        raise ValueError(f"expected 9600 representation-trials, got {len(raw_trials)}")
    thresholds, evaluated = calibrate(raw_trials, float(config["calibration"]["alpha"]))
    operating, directions = operating_characteristics(evaluated)
    comparisons = paired_comparisons(evaluated)
    bandwidths = pd.DataFrame(bandwidth_rows)
    raw_trials.to_csv(output_dir / "stage6p_trial_mmd_within_representation.csv", index=False)
    thresholds.to_csv(output_dir / "stage6p_representation_specific_aa_thresholds.csv", index=False)
    evaluated.to_csv(output_dir / "stage6p_evaluated_trials.csv", index=False)
    operating.to_csv(output_dir / "stage6p_operating_characteristics.csv", index=False)
    directions.to_csv(output_dir / "stage6p_direction_specific_detection.csv", index=False)
    comparisons.to_csv(output_dir / "stage6p_paired_representation_comparisons.csv", index=False)
    bandwidths.to_csv(output_dir / "stage6p_representation_bandwidths_do_not_compare_raw_mmd.csv", index=False)
    plot_results(output_dir, operating)
    write_report(output_dir, operating, comparisons)
    summary = {
        "schema_version": "stage6p_representation_unpaired_release_results_v1",
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 257,
        "config_sha256": sha256_file(args.config.resolve()),
        "source_log_assignments_sha256": sha256_file(args.assignments.resolve()),
        "source_pool_metadata_sha256": sha256_file(args.embedding_pool / "metadata.csv"),
        "source_outputs_modified": False,
        "nuplan_simulation_rerun": False,
        "training_performed": False,
        "cross_representation_raw_mmd2_comparison_forbidden": True,
        "preparation": preparation,
        "bandwidths": bandwidth_rows,
        "operating_characteristics": operating.to_dict("records"),
        "paired_representation_comparisons": comparisons.to_dict("records"),
    }
    write_json(output_dir / "stage6p_representation_unpaired_release_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embedding_pool", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--context_existing", type=Path, required=True)
    parser.add_argument("--context_expanded", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"]}, ensure_ascii=False))
