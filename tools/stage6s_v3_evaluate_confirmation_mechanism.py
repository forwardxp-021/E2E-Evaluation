#!/usr/bin/env python3
"""Evaluate the frozen Stage6S-v3 mechanism gate; never reads embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage6s_v2_evaluate_development_mechanism import (  # noqa: E402
    PLANNERS, directional_fraction, median, metrics,
)

FREEZE_SHA = "7105940bd822f02d643ed4f5cb9a8321b3827ca6117be289914057e3fe8a26c6"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def bootstrap(pairs: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    logs = np.asarray([row["log_name"] for row in pairs], dtype=str)
    unique = np.unique(logs)
    reps = int(design["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(design["statistics"]["bootstrap_seed"]))
    keys = [
        "delta_mean_speed", "delta_rms_accel", "delta_median_front_gap",
        "delta_median_finite_thw", "delta_mean_accel_during_closing",
        "delta_mean_accel_during_following_pressure",
    ]
    samples = {key: np.empty(reps, dtype=float) for key in keys}
    for repetition in range(reps):
        selected_logs = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(logs == log) for log in selected_logs])
        selected = [pairs[index] for index in indices]
        for key in keys:
            samples[key][repetition] = median(np.asarray([row[key] for row in selected], dtype=float))
    return {
        key: {
            "bootstrap95_low": float(np.nanquantile(values, 0.025)),
            "bootstrap95_high": float(np.nanquantile(values, 0.975)),
            "finite_replicates": int(np.isfinite(values).sum()),
        }
        for key, values in samples.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256(args.freeze_manifest) != FREEZE_SHA:
        raise ValueError("Stage6S-v3 freeze manifest changed")
    design = read_json(args.design)
    freeze = read_json(args.freeze_manifest)
    view = read_json(args.view_dir / "stage6s_v3_confirmation_view_summary.json")
    validation = read_json(args.context_dir / "warnings.json").get("validation", {})
    expected = int(freeze["scenario_count"])
    if freeze.get("status") != "STAGE6S_V3_CONFIRMATION_ROSTER_FROZEN_NOT_RUN" or freeze.get("embedding_or_bdd_read") is not False:
        raise ValueError("Stage6S-v3 freeze/blind state changed")
    if sha256(args.design) != freeze["confirmation_design_sha256"]:
        raise ValueError("Stage6S-v3 design changed")
    if view.get("status") != "STAGE6S_V3_CONFIRMATION_VIEW_READY" or view.get("scenario_count") != expected:
        raise ValueError("Stage6S-v3 view incomplete")
    if view.get("embedding_or_bdd_read") is not False or validation.get("pass") is not True:
        raise ValueError("Stage6S-v3 view/context validation failed")
    meta = read_csv(args.context_dir / "metadata.csv")
    ledger = read_csv(args.view_dir / "stage6s_v3_confirmation_scenario_ledger.csv")
    ego = np.load(args.context_dir / "ego_seq.npy", mmap_mode="r")
    mask = np.load(args.context_dir / "ego_seq_mask.npy", mmap_mode="r").astype(bool)
    neighbor = np.load(args.context_dir / "neighbor_seq.npy", mmap_mode="r")
    by_pair = {(int(row["scenario_index"]), row["planner_name"]): index for index, row in enumerate(meta)}
    by_scenario = {int(row["global_scenario_index"]): row for row in ledger}
    pairs: list[dict[str, Any]] = []
    for scenario in range(expected):
        row: dict[str, Any] = {"scenario_index": scenario, **by_scenario[scenario]}
        values = []
        for planner in PLANNERS:
            index = by_pair[(scenario, planner)]
            values.append(metrics(ego[index], neighbor[index], mask[index], design["thw_definition"]))
        for name in values[0]:
            row[f"short_{name}"] = values[0][name]
            row[f"long_{name}"] = values[1][name]
            row[f"delta_{name}"] = values[0][name] - values[1][name]
        pairs.append(row)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pair_path = args.output_dir / "stage6s_v3_confirmation_pair_mechanisms.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    aggregate = {
        "delta_mean_speed_mps": median(np.asarray([r["delta_mean_speed"] for r in pairs])),
        "delta_rms_accel_mps2": median(np.asarray([r["delta_rms_accel"] for r in pairs])),
        "delta_median_front_gap_m": median(np.asarray([r["delta_median_front_gap"] for r in pairs])),
        "delta_median_finite_thw_s": median(np.asarray([r["delta_median_finite_thw"] for r in pairs])),
        "delta_mean_accel_during_closing_mps2": median(np.asarray([r["delta_mean_accel_during_closing"] for r in pairs])),
        "delta_mean_accel_during_following_pressure_mps2": median(np.asarray([r["delta_mean_accel_during_following_pressure"] for r in pairs])),
        "pairs_with_valid_front": int(sum(np.isfinite(r["short_median_front_gap"]) and np.isfinite(r["long_median_front_gap"]) for r in pairs)),
    }
    directional = {
        "front_gap": directional_fraction(pairs, "delta_median_front_gap", -1),
        "finite_thw": directional_fraction(pairs, "delta_median_finite_thw", -1),
        "closing_accel": max(directional_fraction(pairs, "delta_mean_accel_during_closing", 1), directional_fraction(pairs, "delta_mean_accel_during_closing", -1)),
        "following_accel": max(directional_fraction(pairs, "delta_mean_accel_during_following_pressure", 1), directional_fraction(pairs, "delta_mean_accel_during_following_pressure", -1)),
    }
    gate = design["mechanism_gate"]
    interactions = {
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
        "interaction_metric_count": sum(interactions.values()) >= gate["required_interaction_metric_count"],
    }
    passed = all(checks.values())
    result = safe({
        "schema_version": "stage6s_v3_confirmation_mechanism_v1",
        "status": "STAGE6S_V3_MECHANISM_GATE_PASS_REPRESENTATION_EVALUATION_AUTHORIZED" if passed else "STAGE6S_V3_MECHANISM_GATE_FAILED_STOP_NO_EMBEDDING",
        "mechanism_gate_passed": passed, "complete_pairs": len(pairs),
        "distinct_logs": len(set(r["log_name"] for r in pairs)),
        "aggregate": aggregate, "directional_pair_fractions": directional,
        "interaction_checks": {key: bool(value) for key, value in interactions.items()},
        "gate_checks": {key: bool(value) for key, value in checks.items()},
        "log_cluster_bootstrap": bootstrap(pairs, design),
        "thw_definition": design["thw_definition"], "embedding_or_bdd_read": False,
        "training_or_protocol_modified": False, "design_sha256": sha256(args.design),
        "freeze_manifest_sha256": sha256(args.freeze_manifest), "pair_mechanisms_sha256": sha256(pair_path),
    })
    (args.output_dir / "stage6s_v3_confirmation_mechanism_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--view_dir", type=Path, required=True)
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
