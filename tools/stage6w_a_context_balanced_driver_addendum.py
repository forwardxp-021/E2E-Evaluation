#!/usr/bin/env python3
"""Add context-balanced signal/noise attribution to frozen Stage6W-A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "outputs/stage6w_a_paired_unpaired_mechanism_v1/stage6w_a_manifest.json"
TRIALS = ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_trial_statistics.csv"
STAGE6P_MANIFEST = ROOT / "outputs/stage6v_stage6p_unpaired_blind_v1/stage6v_stage6p_result_manifest.json"
REPS = ["old64", "A_3407", "B_3407", "C_3407", "ego13"]
METHODS = ["raw_marginal", "context_balanced"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = read_json(BASE)
    if base.get("status") != "FROZEN_STAGE6W_A_PAIRED_UNPAIRED_MECHANISM_COMPLETE":
        raise RuntimeError("Stage6W-A base analysis is not frozen complete")
    trials = pd.read_csv(TRIALS)
    trials = trials[
        (trials.target_scenarios_per_release == 400)
        & trials.representation.isin(REPS)
        & trials.method.isin(METHODS)
        & trials.valid.astype(bool)
    ].copy()
    rows = []
    for representation in REPS:
        for method in METHODS:
            current = trials[(trials.representation == representation) & (trials.method == method)]
            calibration = current[current.family == "AA_CALIBRATION"].statistic.astype(float).to_numpy()
            evaluation = current[current.family == "AB_EVALUATION"].statistic.astype(float).to_numpy()
            if len(calibration) != 200 or len(evaluation) != 200:
                raise RuntimeError(f"incomplete Stage6P n=400 trials: {representation}/{method}")
            null_mean = float(np.mean(calibration))
            null_sd = float(np.std(calibration, ddof=1))
            null_q95 = float(np.quantile(calibration, 0.95, method="higher"))
            observed = float(np.median(evaluation))
            rows.append({
                "representation": representation, "method": method,
                "calibration_trials": len(calibration), "ab_trials": len(evaluation),
                "median_ab_z_within_representation": (observed - null_mean) / null_sd,
                "median_ab_signal_excess_over_q95": (observed - null_mean) / null_q95,
                "null_sd_over_q95": null_sd / null_q95,
                "ab_detection_at_own_q95": float(np.mean(evaluation > null_q95)),
            })
    summary = pd.DataFrame(rows)
    drivers = []
    indexed = summary.set_index(["representation", "method"])
    for method in METHODS:
        old = indexed.loc[("old64", method)]
        for representation in REPS[1:]:
            current = indexed.loc[(representation, method)]
            signal_ratio = float(current.median_ab_signal_excess_over_q95 / old.median_ab_signal_excess_over_q95)
            noise_ratio = float(current.null_sd_over_q95 / old.null_sd_over_q95)
            signal_gain = math.log(max(signal_ratio, 1e-12))
            noise_gain = -math.log(max(noise_ratio, 1e-12))
            drivers.append({
                "representation": representation, "method": method, "relative_to": "old64",
                "standardized_signal_ratio": signal_ratio, "relative_null_noise_ratio": noise_ratio,
                "log_gain_from_signal": signal_gain, "log_gain_from_noise_reduction": noise_gain,
                "primary_driver": "signal_enhancement" if signal_gain > noise_gain else "null_variance_reduction",
            })
    drivers_frame = pd.DataFrame(drivers)
    if not (drivers_frame.primary_driver == "signal_enhancement").all():
        raise RuntimeError("driver classification requires review")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "stage6w_a_unpaired_signal_noise_by_method.csv"
    driver_path = output / "stage6w_a_unpaired_driver_by_method.csv"
    summary.to_csv(summary_path, index=False)
    drivers_frame.to_csv(driver_path, index=False)
    manifest = {
        "schema_version": "stage6w_a_context_balanced_driver_addendum_v2",
        "status": "FROZEN_STAGE6W_A_CONTEXT_BALANCED_DRIVER_ADDENDUM_V2_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_stage6w_a_manifest_sha256": sha256(BASE),
        "stage6p_trial_statistics_sha256": sha256(TRIALS),
        "stage6p_manifest_sha256": sha256(STAGE6P_MANIFEST),
        "methods": METHODS, "representations": REPS,
        "primary_result": "B/C improvement is primarily signal enhancement under both methods; context balancing adds only a smaller secondary null-noise reduction.",
        "cross_representation_raw_mmd2_comparison_performed": False,
        "training_or_checkpoint_write": False,
        "result_files": {summary_path.name: sha256(summary_path), driver_path.name: sha256(driver_path)},
    }
    manifest_path = output / "stage6w_a_context_balanced_driver_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "summary": rows, "drivers": drivers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path,
        default=ROOT / "outputs/stage6w_a_context_balanced_driver_addendum_v2",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False, allow_nan=False))
