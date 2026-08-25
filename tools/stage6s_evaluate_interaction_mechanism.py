#!/usr/bin/env python3
"""Evaluate the pre-frozen Stage 6S interaction mechanism gate (no embeddings)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PLANNERS = [
    "pdm_closed_interaction_short_headway_v1",
    "pdm_closed_interaction_long_headway_v1",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def finite_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if values.size else math.nan


def metrics(ego: np.ndarray, neighbor: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ego = ego[mask]
    front = neighbor[0, mask]
    front_valid = front[:, 0] > 0.5
    speed = ego[:, 5]
    accel = ego[:, 6]
    closing = np.where(front_valid, front[:, 8], np.nan)
    closing_positive = np.isfinite(closing) & (closing > 0.5)
    response = np.where(closing_positive, accel, np.nan)
    thw = np.where(front_valid, front[:, 10], np.nan)
    thw_uncapped = np.where(front_valid & (front[:, 10] < 998.999), front[:, 10], np.nan)
    return {
        "mean_speed": finite_mean(speed),
        "rms_accel": finite_rms(accel),
        "mean_thw": finite_mean(thw),
        "median_thw_uncapped_diagnostic": float(np.nanmedian(thw_uncapped)) if np.isfinite(thw_uncapped).any() else math.nan,
        "thw_cap_fraction_diagnostic": float(np.mean(front_valid & (front[:, 10] >= 998.999))),
        "mean_front_gap": finite_mean(np.where(front_valid, front[:, 5], np.nan)),
        "mean_closing": finite_mean(closing),
        "mean_accel_during_closing": finite_mean(response),
        "front_valid_ratio": float(np.mean(front_valid)),
        "closing_response_valid_ratio": float(np.mean(closing_positive)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    freeze = read_json(args.freeze_manifest)
    view = read_json(args.view_dir / "stage6s_view_summary.json")
    context_validation = read_json(args.context_dir / "warnings.json").get("validation", {})
    if freeze.get("embedding_or_bdd_read") is not False or view.get("full_embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6S mechanism analysis is not certified blinded")
    if view.get("status") != "INTERACTION_DOMINANT_VIEW_READY" or view.get("scenario_count") != 24:
        raise ValueError("Stage 6S view is incomplete")
    if context_validation.get("pass") is not True:
        raise ValueError("Stage 6S context validation failed")
    meta = read_csv(args.context_dir / "metadata.csv")
    ledger = read_csv(args.view_dir / "stage6s_scenario_ledger.csv")
    ego = np.load(args.context_dir / "ego_seq.npy", mmap_mode="r")
    mask = np.load(args.context_dir / "ego_seq_mask.npy", mmap_mode="r").astype(bool)
    neighbor = np.load(args.context_dir / "neighbor_seq.npy", mmap_mode="r")
    by_pair = {(int(row["scenario_index"]), row["planner_name"]): index for index, row in enumerate(meta)}
    by_scenario = {int(row["global_scenario_index"]): row for row in ledger}
    pairs = []
    for scenario in range(24):
        row = {"scenario_index": scenario, **by_scenario[scenario]}
        values = []
        for planner in PLANNERS:
            index = by_pair[(scenario, planner)]
            values.append(metrics(ego[index], neighbor[index], mask[index]))
        for name in values[0]:
            row[f"short_{name}"] = values[0][name]
            row[f"long_{name}"] = values[1][name]
            row[f"delta_{name}"] = values[0][name] - values[1][name]
        pairs.append(row)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    fields = list(pairs[0])
    with (output / "stage6s_pair_mechanism_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(pairs)
    gate = freeze["mechanism_gate"]
    mean = lambda key: finite_mean(np.asarray([row[key] for row in pairs], dtype=float))
    valid_front_pairs = sum(np.isfinite(row["short_mean_thw"]) and np.isfinite(row["long_mean_thw"]) for row in pairs)
    aggregate = {
        "delta_mean_speed_mps": mean("delta_mean_speed"),
        "delta_rms_accel_mps2": mean("delta_rms_accel"),
        "delta_mean_thw_s": mean("delta_mean_thw"),
        "delta_median_thw_uncapped_diagnostic_s": mean("delta_median_thw_uncapped_diagnostic"),
        "delta_thw_cap_fraction_diagnostic": mean("delta_thw_cap_fraction_diagnostic"),
        "delta_mean_front_gap_m": mean("delta_mean_front_gap"),
        "delta_mean_closing_mps": mean("delta_mean_closing"),
        "delta_mean_accel_during_closing_mps2": mean("delta_mean_accel_during_closing"),
        "pairs_with_valid_front": int(valid_front_pairs),
    }
    scenario_type_diagnostics = {}
    for scenario_type in sorted({row["scenario_type"] for row in pairs}):
        scoped = [row for row in pairs if row["scenario_type"] == scenario_type]
        scoped_mean = lambda key: finite_mean(np.asarray([row[key] for row in scoped], dtype=float))
        scenario_type_diagnostics[scenario_type] = {
            "pair_count": len(scoped),
            "delta_mean_speed_mps": scoped_mean("delta_mean_speed"),
            "delta_rms_accel_mps2": scoped_mean("delta_rms_accel"),
            "delta_mean_front_gap_m": scoped_mean("delta_mean_front_gap"),
            "delta_mean_closing_mps": scoped_mean("delta_mean_closing"),
            "delta_mean_accel_during_closing_mps2": scoped_mean("delta_mean_accel_during_closing"),
            "short_front_valid_ratio": scoped_mean("short_front_valid_ratio"),
            "long_front_valid_ratio": scoped_mean("long_front_valid_ratio"),
            "short_closing_response_valid_ratio": scoped_mean("short_closing_response_valid_ratio"),
            "long_closing_response_valid_ratio": scoped_mean("long_closing_response_valid_ratio"),
        }
    interaction_checks = {
        "thw": math.isfinite(aggregate["delta_mean_thw_s"]) and aggregate["delta_mean_thw_s"] <= gate["short_minus_long_mean_thw_max"],
        "front_gap": math.isfinite(aggregate["delta_mean_front_gap_m"]) and aggregate["delta_mean_front_gap_m"] <= gate["short_minus_long_mean_front_gap_m_max"],
    }
    diagnostics = {
        "closing_response_absolute_delta_mps2": aggregate["delta_mean_accel_during_closing_mps2"],
        "closing_response_has_prefrozen_threshold": False,
    }
    checks = {
        "mean_speed_small": abs(aggregate["delta_mean_speed_mps"]) <= gate["absolute_delta_mean_speed_mps_max"],
        "rms_accel_small": abs(aggregate["delta_rms_accel_mps2"]) <= gate["absolute_delta_rms_accel_mps2_max"],
        "valid_front_pairs": valid_front_pairs >= gate["minimum_pairs_with_valid_front"],
        "interaction_metric_count": sum(interaction_checks.values()) >= gate["required_interaction_metric_count"],
    }
    interaction_checks = {key: bool(value) for key, value in interaction_checks.items()}
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    result = {
        "schema_version": "stage6s_interaction_mechanism_evidence_v1",
        "status": "INTERACTION_DOMINANT_BENCHMARK_PASS" if passed else "PDM_INTERACTION_BENCHMARK_LIMITATION",
        "aggregate": aggregate,
        "interaction_checks": interaction_checks,
        "diagnostics": diagnostics,
        "scenario_type_diagnostics": scenario_type_diagnostics,
        "gate_checks": checks,
        "embedding_or_bdd_read": False,
        "planner_tuning_after_outcome": False,
    }
    (output / "stage6s_mechanism_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Stage 6S interaction-dominant nuPlan benchmark机制报告",
        "",
        f"## 结论：{result['status']}",
        "",
        "本报告只读取realized trajectory与逐帧front语义上下文；未读取embedding、BDD或effect size，也未按结果回调planner参数。",
        "",
        f"- short-long平均速度差：{aggregate['delta_mean_speed_mps']:.4f} m/s。",
        f"- short-long RMS accel差：{aggregate['delta_rms_accel_mps2']:.4f} m/s²。",
        f"- short-long THW差：{aggregate['delta_mean_thw_s']:.4f} s。",
        f"- THW存在999 s cap；cap-excluded逐pair median的short-long诊断差为{aggregate['delta_median_thw_uncapped_diagnostic_s']:.4f} s，cap比例差为{aggregate['delta_thw_cap_fraction_diagnostic']:.4f}。这两项为结果后稳健性诊断，不替代预冻结门禁。",
        f"- short-long front gap差：{aggregate['delta_mean_front_gap_m']:.4f} m。",
        f"- short-long closing期加速度响应差：{aggregate['delta_mean_accel_during_closing_mps2']:.4f} m/s²。",
        f"- 有效front配对：{valid_front_pairs}/24。",
        f"- 门禁：{checks}；预冻结interaction指标：{interaction_checks}。closing-response没有预冻结数值阈值，只作诊断。",
        "",
        "若结论为PDM limitation，则按预冻结规则如实记录，不继续为了模型获胜而调参。",
    ]
    (output / "stage6s_mechanism_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--view_dir", type=Path, required=True)
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
