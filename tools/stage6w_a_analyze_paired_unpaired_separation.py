#!/usr/bin/env python3
"""Explain paired/unpaired separation on the same frozen 800-pair pool.

The analysis never trains a model and never compares raw MMD² across
representations.  For every frozen n=400 A/B release, it constructs two
same-support paired contrasts (release-A support and release-B support), so
sample size, scenario pool, and representation are held fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6p_run_representation_unpaired_release import build_trials, median_bandwidth, rbf_kernel


ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = ROOT / "outputs/stage6h_expanded_800_embedding_pool_v1"
ASSIGNMENTS = ROOT / "outputs/stage6h_nuplan_power_curve_800_v1/power_curve_log_assignments.csv"
STAGE6P = ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1"
STAGE6JK = ROOT / "outputs/stage6v_stage6jk_paired_blind_v1"
STAGE6V_FINAL = ROOT / "outputs/stage6v_one_time_blind_evaluation_final_v1/stage6v_blind_evaluation_final_manifest.json"
DEFAULT_OUT = ROOT / "outputs/stage6w_a_paired_unpaired_mechanism_v1"

REPRESENTATIONS = ["old64", "A_3407", "B_3407", "C_3407", "ego13"]
ASSERTIVE = "pdm_closed_assertive_v1"
CONSERVATIVE = "pdm_closed_conservative_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def row_indices(pool: pd.DataFrame, planner: str, logs: list[str]) -> np.ndarray:
    mask = (
        (pool["planner_name"].astype(str).to_numpy() == planner)
        & pool["log_name"].astype(str).isin(logs).to_numpy()
    )
    return np.flatnonzero(mask)


def unit_direction_coherence(displacements: np.ndarray) -> tuple[float, float, float]:
    norms = np.linalg.norm(displacements, axis=1)
    valid = norms > 1e-12
    units = displacements[valid] / norms[valid, None]
    mean_unit = np.mean(units, axis=0)
    resultant = float(np.linalg.norm(mean_unit))
    global_shift = np.mean(displacements, axis=0)
    global_norm = float(np.linalg.norm(global_shift))
    if global_norm <= 1e-12:
        return resultant, math.nan, 0.0
    cosines = (displacements @ global_shift) / np.maximum(norms * global_norm, 1e-12)
    return resultant, float(np.mean(cosines > 0.0)), global_norm


def variance_decomposition(displacements: np.ndarray, logs: np.ndarray) -> dict[str, float]:
    n = len(displacements)
    mean = np.mean(displacements, axis=0)
    total_energy = float(np.sum(displacements * displacements))
    planner = float(n * np.sum(mean * mean))
    log_ss = 0.0
    residual = 0.0
    for log in np.unique(logs):
        current = displacements[logs == log]
        log_mean = np.mean(current, axis=0)
        log_ss += float(len(current) * np.sum((log_mean - mean) ** 2))
        residual += float(np.sum((current - log_mean) ** 2))
    denominator = max(total_energy, 1e-12)
    return {
        "planner_signal_energy_fraction": planner / denominator,
        "log_heterogeneity_energy_fraction": log_ss / denominator,
        "scenario_residual_energy_fraction": residual / denominator,
        "decomposition_closure_error": abs(total_energy - planner - log_ss - residual) / denominator,
    }


def analytic_paired_null(contrast: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    selected = np.asarray(contrast[np.ix_(indices, indices)], dtype=np.float64)
    n = len(indices)
    observed = float(np.sum(selected) / (n * n))
    mean = float(np.trace(selected) / (n * n))
    off_diagonal = selected.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    variance = float(2.0 * np.sum(off_diagonal * off_diagonal) / (n**4))
    sd = math.sqrt(max(variance, 0.0))
    q95 = mean + 1.6448536269514722 * sd
    return {
        "paired_observed_within_rep_only": observed,
        "paired_null_mean_within_rep_only": mean,
        "paired_null_sd_within_rep_only": sd,
        "paired_null_q95_normal_approx_within_rep_only": q95,
        "paired_z": (observed - mean) / sd if sd > 0 else math.inf,
        "paired_observed_to_q95": observed / q95 if q95 > 0 else math.inf,
        "paired_signal_excess_over_q95": (observed - mean) / q95 if q95 > 0 else math.inf,
        "paired_null_sd_over_q95": sd / q95 if q95 > 0 else math.inf,
    }


def validate_analytic_null(
    contrast: np.ndarray, indices: np.ndarray, analytic: dict[str, float], seed: int, repetitions: int = 10000
) -> dict[str, float]:
    selected = np.asarray(contrast[np.ix_(indices, indices)], dtype=np.float64)
    n = len(indices)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, n))
        samples[start:stop] = np.einsum("bi,ij,bj->b", signs, selected, signs, optimize=True) / (n * n)
    empirical_mean = float(np.mean(samples))
    empirical_sd = float(np.std(samples, ddof=1))
    return {
        "empirical_mean": empirical_mean,
        "empirical_sd": empirical_sd,
        "analytic_mean": analytic["paired_null_mean_within_rep_only"],
        "analytic_sd": analytic["paired_null_sd_within_rep_only"],
        "relative_mean_error": abs(empirical_mean - analytic["paired_null_mean_within_rep_only"])
        / max(abs(empirical_mean), 1e-12),
        "relative_sd_error": abs(empirical_sd - analytic["paired_null_sd_within_rep_only"])
        / max(empirical_sd, 1e-12),
    }


def release_geometry(values: np.ndarray, ia: np.ndarray, ib: np.ndarray, bandwidth: float) -> dict[str, Any]:
    a = values[ia].astype(np.float64)
    b = values[ib].astype(np.float64)
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    shift = mean_a - mean_b
    within = 0.5 * (float(np.mean(np.sum((a - mean_a) ** 2, axis=1))) + float(np.mean(np.sum((b - mean_b) ** 2, axis=1))))
    shift_norm = float(np.linalg.norm(shift))
    return {
        "release_shift": shift,
        "release_shift_norm_over_bandwidth": shift_norm / bandwidth,
        "within_release_rms_over_bandwidth": math.sqrt(max(within, 0.0)) / bandwidth,
        "centroid_shift_to_within_rms": shift_norm / math.sqrt(max(within, 1e-12)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    final = read_json(STAGE6V_FINAL)
    if final.get("status") != "FROZEN_STAGE6V_ONE_TIME_BLIND_EVALUATION_COMPLETE":
        raise RuntimeError("Stage6V is not frozen complete")
    if final.get("training_or_protocol_modified") is not False:
        raise RuntimeError("Stage6V immutability contract changed")
    pool = pd.read_csv(POOL_DIR / "metadata.csv").sort_values("global_row").reset_index(drop=True)
    if len(pool) != 1600 or pool["scenario_index"].nunique() != 800:
        raise RuntimeError("frozen 800-pair pool contract failed")
    pair_table = pool.pivot(index="scenario_index", columns="planner_name", values="global_row").sort_index()
    if list(pair_table.columns) != [ASSERTIVE, CONSERVATIVE] or pair_table.isna().any().any():
        raise RuntimeError("pool is not 800 exhaustive assertive/conservative pairs")
    assertive_rows = pair_table[ASSERTIVE].astype(int).to_numpy()
    conservative_rows = pair_table[CONSERVATIVE].astype(int).to_numpy()
    pair_logs = pool.iloc[assertive_rows]["log_name"].astype(str).to_numpy()

    assignments = pd.read_csv(ASSIGNMENTS)
    trials = [row for row in build_trials(assignments) if int(row["target_scenarios_per_release"]) == 400 and row["family"] == "AB_EVALUATION"]
    if len(trials) != 200:
        raise RuntimeError(f"expected 200 frozen n=400 A/B trials, got {len(trials)}")
    unpaired_raw = pd.read_csv(STAGE6P / "stage6v_stage6p_trial_statistics.csv")
    unpaired_raw = unpaired_raw[
        (unpaired_raw["target_scenarios_per_release"] == 400)
        & (unpaired_raw["family"].isin(["AA_CALIBRATION", "AA_EVALUATION", "AB_EVALUATION"]))
        & (unpaired_raw["method"] == "raw_marginal")
    ].copy()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    release_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    null_validation: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    rep_summaries: list[dict[str, Any]] = []

    for rep_index, representation in enumerate(REPRESENTATIONS):
        values = np.asarray(np.load(STAGE6P / "representations" / f"{representation}.npy"), dtype=np.float64)
        bandwidth = median_bandwidth(values, 620272, 100000)
        kernel = rbf_kernel(values, bandwidth).astype(np.float64)
        kaa = kernel[np.ix_(assertive_rows, assertive_rows)]
        kbb = kernel[np.ix_(conservative_rows, conservative_rows)]
        kab = kernel[np.ix_(assertive_rows, conservative_rows)]
        contrast = kaa + kbb - kab - kab.T
        displacement = values[assertive_rows] - values[conservative_rows]
        norms = np.linalg.norm(displacement, axis=1)
        coherence, positive_fraction, global_shift_norm = unit_direction_coherence(displacement)
        decomposition = variance_decomposition(displacement, pair_logs)
        decomposition_rows.append({
            "representation": representation,
            "pair_count": 800,
            "median_pair_displacement_over_bandwidth": float(np.median(norms) / bandwidth),
            "mean_pair_displacement_over_bandwidth": float(np.mean(norms) / bandwidth),
            "global_mean_shift_over_bandwidth": global_shift_norm / bandwidth,
            "pair_direction_resultant_length": coherence,
            "pair_positive_projection_fraction": positive_fraction,
            **decomposition,
        })

        calibration = unpaired_raw[(unpaired_raw["representation"] == representation) & (unpaired_raw["family"] == "AA_CALIBRATION")]
        calibration_values = calibration["statistic"].astype(float).to_numpy()
        unpaired_null_mean = float(np.mean(calibration_values))
        unpaired_null_sd = float(np.std(calibration_values, ddof=1))
        unpaired_null_q95 = float(np.quantile(calibration_values, 0.95, method="higher"))
        release_vectors = []
        paired_z_values = []
        unpaired_z_values = []
        unpaired_signal_q_values = []
        paired_noise_q_values = []

        rep_ab = unpaired_raw[(unpaired_raw["representation"] == representation) & (unpaired_raw["family"] == "AB_EVALUATION")]
        keyed_ab = rep_ab.set_index(["experiment_set", "repetition", "split_seed", "planner_A", "planner_B"])
        for trial_index, trial in enumerate(trials):
            ia = row_indices(pool, str(trial["planner_A"]), trial["logs_A"])
            ib = row_indices(pool, str(trial["planner_B"]), trial["logs_B"])
            support_a = np.sort(pool.iloc[ia]["scenario_index"].astype(int).unique())
            support_b = np.sort(pool.iloc[ib]["scenario_index"].astype(int).unique())
            if len(ia) != 400 or len(ib) != 400 or len(support_a) != 400 or len(support_b) != 400:
                raise RuntimeError("n=400 frozen release does not contain exactly 400 scenarios per group")
            key = (trial["experiment_set"], int(trial["repetition"]), int(trial["split_seed"]), trial["planner_A"], trial["planner_B"])
            unpaired_observed = float(keyed_ab.loc[key, "statistic"])
            unpaired_z = (unpaired_observed - unpaired_null_mean) / unpaired_null_sd
            unpaired_signal_q = (unpaired_observed - unpaired_null_mean) / unpaired_null_q95
            geometry = release_geometry(values, ia, ib, bandwidth)
            oriented_shift = geometry.pop("release_shift")
            if str(trial["planner_A"]) == CONSERVATIVE:
                oriented_shift = -oriented_shift
            release_vectors.append(oriented_shift)
            unpaired_z_values.append(unpaired_z)
            unpaired_signal_q_values.append(unpaired_signal_q)
            for support_name, support in (("release_A_support", support_a), ("release_B_support", support_b)):
                paired = analytic_paired_null(contrast, support)
                paired_z_values.append(paired["paired_z"])
                paired_noise_q_values.append(paired["paired_null_sd_over_q95"])
                pair_rows.append({
                    "representation": representation,
                    "trial_index": trial_index,
                    "experiment_set": trial["experiment_set"],
                    "repetition": int(trial["repetition"]),
                    "split_seed": int(trial["split_seed"]),
                    "support": support_name,
                    "n_pairs": len(support),
                    **paired,
                })
                if trial_index == 0 and support_name == "release_A_support":
                    null_validation.append({
                        "representation": representation,
                        **validate_analytic_null(contrast, support, paired, 620290 + rep_index),
                    })
            release_rows.append({
                "representation": representation,
                "trial_index": trial_index,
                "experiment_set": trial["experiment_set"],
                "repetition": int(trial["repetition"]),
                "split_seed": int(trial["split_seed"]),
                "planner_A": trial["planner_A"],
                "planner_B": trial["planner_B"],
                "n_A": len(ia),
                "n_B": len(ib),
                "unpaired_observed_within_rep_only": unpaired_observed,
                "unpaired_null_mean_within_rep_only": unpaired_null_mean,
                "unpaired_null_sd_within_rep_only": unpaired_null_sd,
                "unpaired_null_q95_within_rep_only": unpaired_null_q95,
                "unpaired_z": unpaired_z,
                "unpaired_observed_to_q95": unpaired_observed / unpaired_null_q95,
                "unpaired_signal_excess_over_q95": unpaired_signal_q,
                "unpaired_null_sd_over_q95": unpaired_null_sd / unpaired_null_q95,
                **geometry,
            })

        release_vectors_array = np.asarray(release_vectors)
        release_coherence, release_positive, release_shift_norm = unit_direction_coherence(release_vectors_array)
        mean_pair_shift = np.mean(displacement, axis=0)
        denominator = np.linalg.norm(release_vectors_array, axis=1) * max(float(np.linalg.norm(mean_pair_shift)), 1e-12)
        alignment = (release_vectors_array @ mean_pair_shift) / np.maximum(denominator, 1e-12)
        rep_summaries.append({
            "representation": representation,
            "same_pool_sample_size": 400,
            "ab_trial_count": 200,
            "median_paired_z_same_support": float(np.median(paired_z_values)),
            "median_unpaired_z": float(np.median(unpaired_z_values)),
            "median_paired_null_sd_over_q95": float(np.median(paired_noise_q_values)),
            "unpaired_null_sd_over_q95": unpaired_null_sd / unpaired_null_q95,
            "median_unpaired_signal_excess_over_q95": float(np.median(unpaired_signal_q_values)),
            "release_direction_resultant_length": release_coherence,
            "release_positive_projection_fraction": release_positive,
            "release_mean_shift_over_bandwidth": release_shift_norm / bandwidth,
            "median_release_alignment_to_pair_shift": float(np.median(alignment)),
            "median_centroid_shift_to_within_rms": float(np.median([r["centroid_shift_to_within_rms"] for r in release_rows if r["representation"] == representation])),
        })

    summary_df = pd.DataFrame(rep_summaries)
    decomposition_df = pd.DataFrame(decomposition_rows)
    old = summary_df.set_index("representation").loc["old64"]
    for representation in ("A_3407", "B_3407", "C_3407", "ego13"):
        current = summary_df.set_index("representation").loc[representation]
        signal_ratio = float(current["median_unpaired_signal_excess_over_q95"] / old["median_unpaired_signal_excess_over_q95"])
        noise_ratio = float(current["unpaired_null_sd_over_q95"] / old["unpaired_null_sd_over_q95"])
        signal_log = math.log(max(signal_ratio, 1e-12))
        noise_log = -math.log(max(noise_ratio, 1e-12))
        positive_total = max(signal_log, 0.0) + max(noise_log, 0.0)
        driver_rows.append({
            "representation": representation,
            "relative_to": "old64",
            "standardized_signal_ratio": signal_ratio,
            "relative_null_noise_ratio": noise_ratio,
            "log_z_gain_from_signal": signal_log,
            "log_z_gain_from_noise_reduction": noise_log,
            "positive_gain_share_signal": max(signal_log, 0.0) / positive_total if positive_total > 0 else math.nan,
            "positive_gain_share_noise_reduction": max(noise_log, 0.0) / positive_total if positive_total > 0 else math.nan,
            "primary_driver": "signal_enhancement" if signal_log > noise_log else "null_variance_reduction",
        })

    pair_df = pd.DataFrame(pair_rows)
    release_df = pd.DataFrame(release_rows)
    driver_df = pd.DataFrame(driver_rows)
    validation_df = pd.DataFrame(null_validation)
    pair_df.to_csv(output / "stage6w_a_same_support_paired_trials.csv", index=False)
    release_df.to_csv(output / "stage6w_a_unpaired_release_geometry.csv", index=False)
    summary_df.to_csv(output / "stage6w_a_representation_summary.csv", index=False)
    decomposition_df.to_csv(output / "stage6w_a_pair_displacement_decomposition.csv", index=False)
    driver_df.to_csv(output / "stage6w_a_signal_noise_driver_decomposition.csv", index=False)
    validation_df.to_csv(output / "stage6w_a_analytic_null_validation.csv", index=False)

    historical = pd.read_csv(STAGE6JK / "stage6v_stage6jk_decisions.csv")
    historical = historical[["representation", "median_overall_z_bdd", "task_dose_holm_pass_cells_out_of_12"]]
    historical.to_csv(output / "stage6w_a_historical_stage6jk_context.csv", index=False)
    files = [
        "stage6w_a_same_support_paired_trials.csv", "stage6w_a_unpaired_release_geometry.csv",
        "stage6w_a_representation_summary.csv", "stage6w_a_pair_displacement_decomposition.csv",
        "stage6w_a_signal_noise_driver_decomposition.csv", "stage6w_a_analytic_null_validation.csv",
        "stage6w_a_historical_stage6jk_context.csv",
    ]
    manifest = safe({
        "schema_version": "stage6w_a_paired_unpaired_mechanism_v1",
        "status": "FROZEN_STAGE6W_A_PAIRED_UNPAIRED_MECHANISM_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "same_pool_pair_count": 800,
        "same_sample_size_per_group": 400,
        "ab_release_trial_count": 200,
        "paired_supports_per_trial": 2,
        "representations": REPRESENTATIONS,
        "comparison_rule": "within-representation null standardization only; no cross-representation raw MMD2 comparison",
        "analytic_paired_null": "exact Rademacher quadratic-form mean and variance; 10000-swap validation per representation",
        "training_or_checkpoint_write": False,
        "nuplan_simulation_rerun": False,
        "source_sha256": {
            "stage6v_final": sha256(STAGE6V_FINAL),
            "pool_metadata": sha256(POOL_DIR / "metadata.csv"),
            "assignments": sha256(ASSIGNMENTS),
            "stage6p_manifest": sha256(STAGE6P / "stage6v_stage6p_result_manifest.json"),
            "stage6jk_manifest": sha256(STAGE6JK / "stage6v_stage6jk_result_manifest.json"),
        },
        "result_files": {name: sha256(output / name) for name in files},
    })
    (output / "stage6w_a_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "summary": safe(rep_summaries), "drivers": safe(driver_rows), "decomposition": safe(decomposition_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, allow_nan=False))
