#!/usr/bin/env python3
"""Evaluate robust Stage6S-v2 development mechanisms without embeddings/BDD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PLANNERS = ["pdm_closed_interaction_short_headway_v2", "pdm_closed_interaction_long_headway_v2"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def mean(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.mean(values)) if values.size else math.nan


def median(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.median(values)) if values.size else math.nan


def rms(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.sqrt(np.mean(values * values))) if values.size else math.nan


def metrics(ego: np.ndarray, neighbor: np.ndarray, mask: np.ndarray, thw_cfg: dict[str, Any]) -> dict[str, float]:
    ego = ego[mask]
    front = neighbor[0, mask]
    front_valid = front[:, 0] > 0.5
    speed, accel = ego[:, 5], ego[:, 6]
    gap = np.where(front_valid, front[:, 5], np.nan)
    closing = np.where(front_valid, front[:, 8], np.nan)
    raw_thw = np.where(front_valid, front[:, 10], np.nan)
    thw = np.where(
        np.isfinite(raw_thw)
        & (raw_thw > float(thw_cfg["minimum_seconds_exclusive"]))
        & (raw_thw < float(thw_cfg["maximum_seconds_exclusive"])), raw_thw, np.nan,
    )
    closing_mask = np.isfinite(closing) & (closing > 0.5)
    pressure_mask = closing_mask & np.isfinite(gap) & (gap <= 40.0)
    return {
        "mean_speed": mean(speed), "rms_accel": rms(accel),
        "median_front_gap": median(gap), "median_finite_thw": median(thw),
        "mean_accel_during_closing": mean(np.where(closing_mask, accel, np.nan)),
        "mean_accel_during_following_pressure": mean(np.where(pressure_mask, accel, np.nan)),
        "front_valid_ratio": float(np.mean(front_valid)),
        "finite_thw_ratio": float(np.mean(np.isfinite(thw))),
        "closing_response_valid_ratio": float(np.mean(closing_mask)),
        "following_pressure_valid_ratio": float(np.mean(pressure_mask)),
    }


def directional_fraction(pairs: list[dict[str, Any]], key: str, direction: int) -> float:
    values = finite(np.asarray([row[key] for row in pairs], dtype=float))
    if not values.size:
        return math.nan
    return float(np.mean(values * direction > 0.0))


def json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite diagnostics to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    freeze = read_json(args.freeze_manifest)
    view = read_json(args.view_dir / "stage6s_v2_development_view_summary.json")
    validation = read_json(args.context_dir / "warnings.json").get("validation", {})
    expected = int(freeze["scenario_count"])
    if freeze.get("embedding_or_bdd_read") is not False or freeze.get("confirmation_roster_read") is not False:
        raise ValueError("Stage6S-v2 development analysis is not blinded")
    if view.get("status") != "STAGE6S_V2_DEVELOPMENT_VIEW_READY" or view.get("scenario_count") != expected:
        raise ValueError("Stage6S-v2 development view is incomplete")
    if view.get("full_embedding_or_bdd_read") is not False or validation.get("pass") is not True:
        raise ValueError("Stage6S-v2 development view/context validation failed")
    meta = read_csv(args.context_dir / "metadata.csv")
    ledger = read_csv(args.view_dir / "stage6s_v2_development_scenario_ledger.csv")
    ego = np.load(args.context_dir / "ego_seq.npy", mmap_mode="r")
    mask = np.load(args.context_dir / "ego_seq_mask.npy", mmap_mode="r").astype(bool)
    neighbor = np.load(args.context_dir / "neighbor_seq.npy", mmap_mode="r")
    by_pair = {(int(row["scenario_index"]), row["planner_name"]): index for index, row in enumerate(meta)}
    by_scenario = {int(row["global_scenario_index"]): row for row in ledger}
    pairs = []
    for scenario in range(expected):
        row: dict[str, Any] = {"scenario_index": scenario, **by_scenario[scenario]}
        values = []
        for planner in PLANNERS:
            index = by_pair[(scenario, planner)]
            values.append(metrics(ego[index], neighbor[index], mask[index], config["thw_definition"]))
        for name in values[0]:
            row[f"short_{name}"] = values[0][name]; row[f"long_{name}"] = values[1][name]
            row[f"delta_{name}"] = values[0][name] - values[1][name]
        pairs.append(row)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    fields = list(pairs[0])
    with (output / "stage6s_v2_development_pair_mechanisms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(pairs)
    aggregate = {
        "delta_mean_speed_mps": median(np.asarray([r["delta_mean_speed"] for r in pairs])),
        "delta_rms_accel_mps2": median(np.asarray([r["delta_rms_accel"] for r in pairs])),
        "delta_median_front_gap_m": median(np.asarray([r["delta_median_front_gap"] for r in pairs])),
        "delta_median_finite_thw_s": median(np.asarray([r["delta_median_finite_thw"] for r in pairs])),
        "delta_mean_accel_during_closing_mps2": median(np.asarray([r["delta_mean_accel_during_closing"] for r in pairs])),
        "delta_mean_accel_during_following_pressure_mps2": median(np.asarray([r["delta_mean_accel_during_following_pressure"] for r in pairs])),
        "pairs_with_valid_front": int(sum(
            np.isfinite(r["short_median_front_gap"]) and np.isfinite(r["long_median_front_gap"])
            for r in pairs
        )),
    }
    gate = config["mechanism_gate"]
    directional = {
        "front_gap": directional_fraction(pairs, "delta_median_front_gap", -1),
        "finite_thw": directional_fraction(pairs, "delta_median_finite_thw", -1),
        "closing_accel": max(
            directional_fraction(pairs, "delta_mean_accel_during_closing", 1),
            directional_fraction(pairs, "delta_mean_accel_during_closing", -1),
        ),
        "following_accel": max(
            directional_fraction(pairs, "delta_mean_accel_during_following_pressure", 1),
            directional_fraction(pairs, "delta_mean_accel_during_following_pressure", -1),
        ),
    }
    interaction_checks = {
        "front_gap": aggregate["delta_median_front_gap_m"] <= gate["short_minus_long_median_front_gap_m_max"] and directional["front_gap"] >= gate["minimum_directional_pair_fraction"],
        "finite_thw": aggregate["delta_median_finite_thw_s"] <= gate["short_minus_long_median_finite_thw_s_max"] and directional["finite_thw"] >= gate["minimum_directional_pair_fraction"],
        "closing_accel": abs(aggregate["delta_mean_accel_during_closing_mps2"]) >= gate["short_minus_long_closing_accel_abs_min_mps2"] and directional["closing_accel"] >= gate["minimum_directional_pair_fraction"],
        "following_accel": abs(aggregate["delta_mean_accel_during_following_pressure_mps2"]) >= gate["short_minus_long_following_accel_abs_min_mps2"] and directional["following_accel"] >= gate["minimum_directional_pair_fraction"],
    }
    checks = {
        "complete_pairs": len(pairs) >= gate["minimum_complete_pairs"],
        "mean_speed_small": abs(aggregate["delta_mean_speed_mps"]) <= gate["absolute_delta_mean_speed_mps_max"],
        "rms_accel_small": abs(aggregate["delta_rms_accel_mps2"]) <= gate["absolute_delta_rms_accel_mps2_max"],
        "valid_front_pairs": aggregate["pairs_with_valid_front"] >= gate["minimum_pairs_with_valid_front"],
        "interaction_metric_count": sum(interaction_checks.values()) >= gate["required_interaction_metric_count"],
    }
    passed = all(checks.values())
    result = {
        "schema_version": "stage6s_v2_development_mechanism_v1",
        "status": "DEVELOPMENT_MECHANISM_PASS_CONFIRMATION_FREEZE_ALLOWED" if passed else "DEVELOPMENT_MECHANISM_NOT_STABLE",
        "aggregate": aggregate, "directional_pair_fractions": directional,
        "interaction_checks": {k: bool(v) for k, v in interaction_checks.items()},
        "gate_checks": {k: bool(v) for k, v in checks.items()},
        "thw_definition": config["thw_definition"], "embedding_or_bdd_read": False,
        "confirmation_roster_read": False, "checkpoint_training_launched": False,
    }
    result = json_safe(result)
    (output / "stage6s_v2_development_mechanism_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--view_dir", type=Path, required=True)
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
