#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402


SCHEMA_VERSION = "stage6j_pure_longitudinal_kinematic_evidence_v1"
EGO = {"speed": 5, "accel": 6, "yaw_rate": 7}
FRONT = {"valid": 0, "distance": 5, "thw": 10}
METRICS = [
    ("delta_mean_speed", "平均速度", "m/s"),
    ("delta_rms_accel", "RMS加速度", "m/s²"),
    ("delta_rms_jerk", "RMS加加速度", "m/s³"),
    ("delta_mean_abs_yaw_rate", "平均绝对横摆角速度", "rad/s"),
    ("delta_mean_thw", "平均车头时距", "s"),
    ("delta_mean_front_distance", "平均前车距离", "m"),
    ("delta_front_valid_ratio", "前车有效帧比例", "ratio"),
]


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def finite_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if values.size else math.nan


def row_metrics(
    ego: np.ndarray, neighbor: np.ndarray, valid_mask: np.ndarray
) -> Dict[str, float]:
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 1 or not np.any(mask):
        raise ValueError("row has no valid ego timestep")
    current_ego = np.asarray(ego, dtype=float)[mask]
    current_neighbor = np.asarray(neighbor, dtype=float)[:, mask]
    speed = current_ego[:, EGO["speed"]]
    accel = current_ego[:, EGO["accel"]]
    yaw_rate = current_ego[:, EGO["yaw_rate"]]
    jerk = np.diff(accel) / 0.1
    front = current_neighbor[0]
    front_valid = front[:, FRONT["valid"]] > 0.5
    distance = np.where(front_valid, front[:, FRONT["distance"]], np.nan)
    thw = np.where(front_valid, front[:, FRONT["thw"]], np.nan)
    return {
        "mean_speed": finite_mean(speed),
        "rms_accel": finite_rms(accel),
        "rms_jerk": finite_rms(jerk),
        "mean_abs_yaw_rate": finite_mean(np.abs(yaw_rate)),
        "mean_thw": finite_mean(thw),
        "mean_front_distance": finite_mean(distance),
        "front_valid_ratio": float(np.mean(front_valid)),
    }


def cluster_bootstrap_mean_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float, int, int]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    valid = np.isfinite(values) & (np.char.str_len(clusters) > 0)
    values = values[valid]
    clusters = clusters[valid]
    if not values.size:
        return math.nan, math.nan, 0, 0
    names = np.unique(clusters)
    sums = np.asarray([values[clusters == name].sum() for name in names], dtype=float)
    counts = np.asarray([np.sum(clusters == name) for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(repetitions, len(names)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high), int(values.size), int(len(names))


def validate_inputs(
    config: Mapping[str, Any],
    context_dir: Path,
    view_dir: Path,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if config.get("frozen_before_context_result_read") is not True:
        raise ValueError("kinematic gate was not frozen before context result read")
    prohibitions = set(config.get("prohibitions", []))
    required_prohibitions = {
        "do_not_read_embedding",
        "do_not_read_bdd",
        "do_not_read_effect_size_before_gate",
        "do_not_change_thresholds_after_observing_stage6j_kinematics",
    }
    if not required_prohibitions.issubset(prohibitions):
        raise ValueError("kinematic gate config is missing required anti-unblinding rules")
    warnings = read_json(context_dir / "warnings.json")
    if warnings.get("validation", {}).get("pass") is not True:
        raise ValueError("Stage5D context validation.pass is not true")
    view = read_json(view_dir / "stage6j_view_summary.json")
    if (
        view.get("status") != "PURE_LONGITUDINAL_VIEW_READY"
        or view.get("scenario_count") != config.get("expected_pair_count")
        or view.get("reaudit_failure_count") != 0
        or view.get("full_embedding_or_bdd_read") is not False
    ):
        raise ValueError("Stage6J unified rollout view is incomplete or untrusted")
    metadata = read_csv(context_dir / "metadata.csv")
    ledger = read_csv(view_dir / "stage6j_scenario_ledger.csv")
    expected_pairs = int(config["expected_pair_count"])
    if len(metadata) != expected_pairs * 2 or len(ledger) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs * 2} context rows and {expected_pairs} ledger rows; "
            f"got {len(metadata)} and {len(ledger)}"
        )
    task_counts = dict(Counter(row["task"] for row in ledger))
    if task_counts != config.get("expected_task_counts"):
        raise ValueError(f"task composition differs from frozen gate: {task_counts}")
    distinct_logs = len({row["log_name"] for row in ledger})
    if distinct_logs != int(config["expected_distinct_log_count"]):
        raise ValueError(f"expected 156 distinct logs, got {distinct_logs}")
    return metadata, ledger


def build_pair_rows(
    config: Mapping[str, Any],
    context_dir: Path,
    metadata: Sequence[Mapping[str, str]],
    ledger: Sequence[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    ego = np.load(context_dir / "ego_seq.npy", mmap_mode="r")
    mask = np.load(context_dir / "ego_seq_mask.npy", mmap_mode="r")
    neighbor = np.load(context_dir / "neighbor_seq.npy", mmap_mode="r")
    if ego.shape[:2] != mask.shape or ego.shape[0] != neighbor.shape[0]:
        raise ValueError(
            f"context array alignment mismatch: ego={ego.shape}, mask={mask.shape}, "
            f"neighbor={neighbor.shape}"
        )
    if ego.shape[-1] != 8 or neighbor.ndim != 4 or neighbor.shape[1:] != (5, ego.shape[1], 15):
        raise ValueError(
            f"unexpected Stage5D shapes: ego={ego.shape}, neighbor={neighbor.shape}"
        )
    by_scenario = {int(row["global_scenario_index"]): row for row in ledger}
    by_pair: Dict[tuple[int, str], int] = {}
    for row_index, row in enumerate(metadata):
        key = (int(row["scenario_index"]), row["planner_name"])
        if key in by_pair:
            raise ValueError(f"duplicate context scenario/planner pair: {key}")
        by_pair[key] = row_index

    planner_a = str(config["planner_a"])
    planner_b = str(config["planner_b"])
    pair_rows: List[Dict[str, Any]] = []
    for scenario_index in range(int(config["expected_pair_count"])):
        if scenario_index not in by_scenario:
            raise ValueError(f"ledger missing global_scenario_index={scenario_index}")
        ledger_row = by_scenario[scenario_index]
        try:
            row_a = by_pair[(scenario_index, planner_a)]
            row_b = by_pair[(scenario_index, planner_b)]
        except KeyError as exc:
            raise ValueError(f"incomplete planner pair for scenario_index={scenario_index}") from exc
        metrics_a = row_metrics(ego[row_a], neighbor[row_a], mask[row_a])
        metrics_b = row_metrics(ego[row_b], neighbor[row_b], mask[row_b])
        result: Dict[str, Any] = {
            "scenario_index": scenario_index,
            "collection_order": int(ledger_row["collection_order"]),
            "scenario_token": ledger_row["scenario_token"],
            "log_name": ledger_row["log_name"],
            "task": ledger_row["task"],
            "scenario_type": ledger_row["scenario_type"],
        }
        for metric in metrics_a:
            result[f"{metric}_A"] = metrics_a[metric]
            result[f"{metric}_B"] = metrics_b[metric]
            result[f"delta_{metric}"] = metrics_a[metric] - metrics_b[metric]
        pair_rows.append(result)
    return pair_rows


def contrast_rows(
    pair_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    bootstrap = config["bootstrap"]
    scopes = [("overall", list(pair_rows))]
    for task in config["expected_task_counts"]:
        scopes.append((task, [row for row in pair_rows if row["task"] == task]))
    results: List[Dict[str, Any]] = []
    for scope_index, (scope, rows) in enumerate(scopes):
        clusters = np.asarray([row["log_name"] for row in rows], dtype=str)
        for metric_index, (metric, label, unit) in enumerate(METRICS):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            low, high, finite_pairs, cluster_count = cluster_bootstrap_mean_ci(
                values,
                clusters,
                repetitions=int(bootstrap["repetitions"]),
                seed=int(bootstrap["seed"]) + scope_index * 100 + metric_index,
                confidence_level=float(bootstrap["confidence_level"]),
            )
            finite = values[np.isfinite(values)]
            results.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "label_zh": label,
                    "unit": unit,
                    "pair_count": len(rows),
                    "finite_pair_count": finite_pairs,
                    "distinct_log_count": cluster_count,
                    "mean_delta_A_minus_B": float(np.mean(finite)) if finite.size else math.nan,
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "median_delta_A_minus_B": float(np.median(finite)) if finite.size else math.nan,
                    "positive_fraction": float(np.mean(finite > 0)) if finite.size else math.nan,
                }
            )
    return results


def evaluate_gate(
    contrasts: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[bool, List[Dict[str, Any]]]:
    overall = {
        row["metric"]: row for row in contrasts if row["scope"] == "overall"
    }
    decisions: List[Dict[str, Any]] = []
    for metric, rule in config["primary_gate"]["metrics"].items():
        row = overall[metric]
        direction = rule["expected_direction"]
        threshold = float(rule["minimum_one_sided_ci_bound"])
        if direction == "positive":
            observed_bound = float(row["cluster_bootstrap_ci95_low"])
            passed = math.isfinite(observed_bound) and observed_bound >= threshold
        elif direction == "negative":
            observed_bound = float(row["cluster_bootstrap_ci95_high"])
            passed = math.isfinite(observed_bound) and observed_bound <= -threshold
        else:
            raise ValueError(f"unsupported gate direction for {metric}: {direction}")
        decisions.append(
            {
                "metric": metric,
                "expected_direction": direction,
                "threshold": threshold,
                "observed_one_sided_bound": observed_bound,
                "mean_delta_A_minus_B": row["mean_delta_A_minus_B"],
                "pass": bool(passed),
            }
        )
    return all(item["pass"] for item in decisions), decisions


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_chinese_report(
    path: Path,
    gate_passed: bool,
    decisions: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    overall = [row for row in contrasts if row["scope"] == "overall"]
    decision = "通过" if gate_passed else "未通过"
    lines = [
        "# Stage 6J 纯纵向运动学门禁报告",
        "",
        f"## 结论：{decision}",
        "",
        "本报告只读取official nuPlan rollout生成的Stage5D运动学与邻车上下文；未读取embedding、BDD或effect size。",
        "",
        "## 冻结主门禁",
        "",
        "| 指标 | A-B均值 | log-cluster bootstrap 95% CI | 冻结边界 | 结果 |",
        "|---|---:|---:|---:|---|",
    ]
    overall_by_metric = {row["metric"]: row for row in overall}
    for item in decisions:
        row = overall_by_metric[item["metric"]]
        lines.append(
            f"| {row['label_zh']} | {row['mean_delta_A_minus_B']:.4f} | "
            f"[{row['cluster_bootstrap_ci95_low']:.4f}, {row['cluster_bootstrap_ci95_high']:.4f}] | "
            f"下界≥{item['threshold']:.3f} | {'PASS' if item['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "主门禁要求平均速度和RMS加速度两个预冻结指标全部PASS。THW、前车距离、jerk和横摆角速度只作支持性诊断，不参与主门禁，避免在前车暴露不完整时过度解释。",
        "",
        "## 总体支持性指标",
        "",
        "| 指标 | 有限pair | A-B均值 | log-cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['label_zh']} ({row['unit']}) | {row['finite_pair_count']} | "
            f"{row['mean_delta_A_minus_B']:.4f} | "
            f"[{row['cluster_bootstrap_ci95_low']:.4f}, {row['cluster_bootstrap_ci95_high']:.4f}] |"
        )
    lines += [
        "",
        "## 后续规则",
        "",
        "- 门禁通过：说明仿真确实实现了足够强的纯纵向运动学对比，随后才允许计算Waymo模型embedding与paired BDD。",
        "- 门禁未通过：先增强或修正PDM纵向处置，不把弱BDD归因于Waymo模型。",
        "- 前车暴露诊断阈值为0.20，但不替代两个主运动学指标。",
        "",
        f"Bootstrap：{config['bootstrap']['repetitions']}次，抽样单位为log_name cluster。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pre-frozen Stage 6J pure-longitudinal kinematic gate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage6j_kinematic_gate.json"),
    )
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--view_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output_dir already exists: {args.output_dir}; use --overwrite"
            )
        shutil.rmtree(args.output_dir)
    config = read_json(args.config)
    metadata, ledger = validate_inputs(config, args.context_dir, args.view_dir)
    pair_rows = build_pair_rows(config, args.context_dir, metadata, ledger)
    contrasts = contrast_rows(pair_rows, config)
    gate_passed, decisions = evaluate_gate(contrasts, config)
    args.output_dir.mkdir(parents=True)
    pair_fields = list(pair_rows[0])
    contrast_fields = list(contrasts[0])
    write_csv(args.output_dir / "stage6j_pair_kinematics.csv", pair_rows, pair_fields)
    write_csv(
        args.output_dir / "stage6j_kinematic_contrasts.csv", contrasts, contrast_fields
    )
    write_csv(
        args.output_dir / "stage6j_primary_gate_decisions.csv",
        decisions,
        list(decisions[0]),
    )
    write_chinese_report(
        args.output_dir / "stage6j_kinematic_gate_report_zh.md",
        gate_passed,
        decisions,
        contrasts,
        config,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "KINEMATIC_GATE_PASS" if gate_passed else "KINEMATIC_GATE_FAIL",
        "kinematic_gate_passed": gate_passed,
        "embedding_and_bdd_analysis_allowed": gate_passed,
        "embedding_or_bdd_read": False,
        "pair_count": len(pair_rows),
        "distinct_log_count": len({row["log_name"] for row in pair_rows}),
        "task_counts": dict(Counter(row["task"] for row in pair_rows)),
        "primary_gate_decisions": decisions,
        "input_files": {
            "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
            "context_warnings": {
                "path": str((args.context_dir / "warnings.json").resolve()),
                "sha256": sha256_file(args.context_dir / "warnings.json"),
            },
            "view_summary": {
                "path": str((args.view_dir / "stage6j_view_summary.json").resolve()),
                "sha256": sha256_file(args.view_dir / "stage6j_view_summary.json"),
            },
        },
        "outputs": {
            "pair_kinematics": "stage6j_pair_kinematics.csv",
            "contrasts": "stage6j_kinematic_contrasts.csv",
            "gate_decisions": "stage6j_primary_gate_decisions.csv",
            "chinese_report": "stage6j_kinematic_gate_report_zh.md",
        },
    }
    write_json(args.output_dir / "stage6j_kinematic_gate_summary.json", json_safe(summary))
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
