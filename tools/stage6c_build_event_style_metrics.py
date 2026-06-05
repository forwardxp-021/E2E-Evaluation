#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tools.stage6c_build_dynamic_event_bins import ALIASES
from tools.stage6c_common import (
    FeatureResolver,
    iter_progress,
    load_feature_rows,
    load_schema_names,
    min_available,
    max_available,
    robust_score,
    score_from_parts,
    safe_array,
    write_json,
)


def get_features(X, resolver):
    return {k: resolver.get(X, k, v) for k, v in ALIASES.items()}


def negative_if_available(x):
    if x is None:
        return None
    return -np.asarray(x, dtype=float)


def abs_if_available(x):
    if x is None:
        return None
    return np.abs(np.asarray(x, dtype=float))


def build_metrics(X, dynamic_bins, meta, resolver, progress_enabled=True):
    n = len(meta)
    f = get_features(X, resolver)
    rows = meta.copy()
    if len(dynamic_bins) != n:
        raise ValueError(f"dynamic_event_bins row count mismatch: bins={len(dynamic_bins)}, features={n}")
    if "global_row" not in dynamic_bins.columns:
        raise ValueError("dynamic_event_bins_path must contain global_row")
    if not np.array_equal(dynamic_bins["global_row"].to_numpy(), rows["global_row"].to_numpy()):
        raise ValueError("dynamic_event_bins_path is not row-aligned with shard_manifest global_row order")

    # Base derived proxies. Missing inputs propagate to NaN rather than zero.
    front_min_gap = min_available([f["min_front_distance"], f["mean_front_distance"]], n)
    side_min_gap = min_available([f["left_front_min_gap"], f["right_front_min_gap"], f["left_rear_min_gap"], f["right_rear_min_gap"]], n)
    rear_min_gap = min_available([f["rear_min_gap"], f["left_rear_min_gap"], f["right_rear_min_gap"]], n)
    peak_decel = max_available([negative_if_available(f["min_acc"]), f["max_abs_accel"]], n)
    peak_accel = max_available([f["max_accel"], f["max_abs_accel"]], n)
    jerk = max_available([f["jerk_p95"], f["max_abs_jerk"]], n)
    gap_pressure_score = score_from_parts([
        robust_score(front_min_gap, higher_is_more=False),
        robust_score(side_min_gap, higher_is_more=False),
        robust_score(rear_min_gap, higher_is_more=False),
        robust_score(f["front_pressure_score"], higher_is_more=True),
    ], n)
    lateral_sharpness = score_from_parts([
        robust_score(f["rms_yaw_rate"], True),
        robust_score(f["rms_curvature"], True),
        robust_score(f["heading_change_total"], True),
    ], n)
    overtake_opportunity_score = score_from_parts([
        robust_score(f["front_pressure_score"], True),
        robust_score(front_min_gap, False),
        robust_score(negative_if_available(f["front_rel_speed"]), True),
        robust_score(f["ego_speed"], True),
    ], n)
    overtake_execution_score = score_from_parts([
        robust_score(f["lane_change_count_proxy"], True),
        robust_score(f["speed_gain"], True),
        robust_score(peak_accel, True),
    ], n)
    assertiveness_score = score_from_parts([
        gap_pressure_score,
        robust_score(f["ego_accel"], True),
        robust_score(f["speed_gain"], True),
        robust_score(f["yielding_score_proxy"], False),
    ], n)
    hesitation_score = score_from_parts([
        robust_score(f["lane_change_duration_proxy"], True),
        robust_score(f["yaw_rate_sign_changes"], True),
        robust_score(f["speed_oscillation"], True),
    ], n)
    hard_brake_score = score_from_parts([
        robust_score(peak_decel, True),
        robust_score(jerk, True),
        robust_score(f["brake_count"], True),
    ], n)
    brake_comfort_score = None if hard_brake_score is None else -hard_brake_score
    cruise_stability_score = score_from_parts([
        robust_score(f["speed_oscillation"], False),
        robust_score(f["jerk_p95"], False),
        robust_score(f["rms_yaw_rate"], False),
    ], n)

    metric_map = {
        "following_mean_thw": f["mean_thw"],
        "following_min_thw": f["min_thw"],
        "following_mean_front_distance": f["mean_front_distance"],
        "following_min_front_distance": f["min_front_distance"],
        "following_peak_decel": peak_decel,
        "following_jerk_p95": f["jerk_p95"],
        "following_front_pressure": f["front_pressure_score"],
        "cutin_reaction_delay_proxy": f["yielding_score_proxy"],
        "cutin_peak_decel_proxy": peak_decel,
        "cutin_min_ttc_proxy": f["min_ttc"],
        "cutin_min_front_gap": front_min_gap,
        "cutin_jerk_after_proxy": jerk,
        "lc_max_yaw_rate": abs_if_available(f["rms_yaw_rate"]),
        "lc_rms_yaw_rate": f["rms_yaw_rate"],
        "lc_rms_curvature": f["rms_curvature"],
        "lc_heading_change_total": f["heading_change_total"],
        "lc_duration_proxy": f["lane_change_duration_proxy"],
        "lc_min_front_gap": front_min_gap,
        "lc_min_rear_gap": rear_min_gap,
        "lc_gap_acceptance_score": gap_pressure_score,
        "lc_lateral_sharpness_score": lateral_sharpness,
        "overtake_opportunity_score": overtake_opportunity_score,
        "overtake_execution_score": overtake_execution_score,
        "overtake_peak_accel": peak_accel,
        "overtake_jerk": jerk,
        "overtake_speed_gain_proxy": f["speed_gain"],
        "yielding_score": f["yielding_score_proxy"],
        "gap_pressure_score": gap_pressure_score,
        "assertiveness_score": assertiveness_score,
        "conflict_accel_score": score_from_parts([gap_pressure_score, robust_score(f["ego_accel"], True)], n),
        "small_gap_speed_maintain_score": score_from_parts([gap_pressure_score, robust_score(f["ego_speed"], True)], n),
        "hesitation_score": hesitation_score,
        "yaw_oscillation_proxy": f["yaw_rate_sign_changes"],
        "speed_oscillation_proxy": f["speed_oscillation"],
        "abort_like_proxy": hesitation_score,
        "hard_brake_score": hard_brake_score,
        "peak_decel": peak_decel,
        "jerk_p95": f["jerk_p95"],
        "max_abs_jerk": f["max_abs_jerk"],
        "brake_comfort_score": brake_comfort_score,
        "cruise_speed_std": f["speed_oscillation"],
        "cruise_acc_std": f["max_abs_accel"],
        "cruise_jerk": jerk,
        "cruise_yaw_rate": f["rms_yaw_rate"],
        "cruise_stability_score": cruise_stability_score,
    }

    for name, arr in iter_progress(list(metric_map.items()), enabled=progress_enabled, desc="computing event style metrics", unit="metric"):
        rows[name] = safe_array(arr, n)
    return rows, metric_map


def main(args):
    t0 = time.time()
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"output_dir exists: {out}; use --overwrite")
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    progress_enabled = not args.no_progress
    X, meta, total_shards = load_feature_rows(args.shard_manifest, progress_enabled=progress_enabled)
    names = load_schema_names(args.feature_schema_path)
    if X.shape[1] != len(names):
        raise ValueError(f"feature shape/schema mismatch: interaction_feat_style dim={X.shape[1]}, schema names={len(names)}")
    dynamic_bins = pd.read_csv(args.dynamic_event_bins_path)
    resolver = FeatureResolver(names)
    df, metric_map = build_metrics(X, dynamic_bins, meta, resolver, progress_enabled=progress_enabled)

    df.to_csv(out / "event_style_metrics.csv", index=False)
    np.save(out / "event_style_metrics.npy", df.to_records(index=False))
    metric_cols = [c for c in df.columns if c not in {"global_row", "shard_id", "local_row"}]
    unavailable = {k: "missing_required_proxy" for k, v in metric_map.items() if v is None}
    write_json(out / "event_style_metric_schema.json", {
        "description": "Stage 6C row-aligned event-specific style metrics. NaN means the required proxy feature was unavailable or non-finite.",
        "row_alignment": "global_row matches dynamic_event_bins.csv and shard_manifest order.",
        "metric_columns": metric_cols,
        "columns": df.columns.tolist(),
    })
    metric_valid_stats = {}
    score_scale_warnings = []
    low_valid_rate_warnings = []
    for c in metric_cols:
        arr = df[c].to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        valid_count = int(len(finite))
        valid_rate = float(valid_count / len(df)) if len(df) else 0.0
        if valid_count:
            stats = {
                "valid_count": valid_count,
                "valid_rate": valid_rate,
                "min": float(np.min(finite)),
                "p01": float(np.quantile(finite, 0.01)),
                "p50": float(np.quantile(finite, 0.50)),
                "p99": float(np.quantile(finite, 0.99)),
                "max": float(np.max(finite)),
            }
        else:
            stats = {"valid_count": 0, "valid_rate": valid_rate, "min": float("nan"), "p01": float("nan"), "p50": float("nan"), "p99": float("nan"), "max": float("nan")}
        metric_valid_stats[c] = stats
        if valid_rate < 0.01:
            low_valid_rate_warnings.append({"metric_name": c, "warning": "low_valid_rate", "valid_count": valid_count, "valid_rate": valid_rate})
        if (np.isfinite(stats["p99"]) and abs(stats["p99"]) > 100) or (np.isfinite(stats["p01"]) and abs(stats["p01"]) > 100):
            score_scale_warnings.append({"metric_name": c, "warning": "score_scale_exploded", "p01": stats["p01"], "p99": stats["p99"]})
    valid_counts = {c: stats["valid_count"] for c, stats in metric_valid_stats.items()}
    write_json(out / "event_style_metric_warnings.json", {
        "resolved_features": resolver.resolved,
        "missing_feature_aliases": resolver.missing,
        "unavailable_metrics": unavailable,
        "metric_valid_stats": metric_valid_stats,
        "score_distribution_diagnostics": metric_valid_stats,
        "low_valid_rate_warnings": low_valid_rate_warnings,
        "score_scale_warnings": score_scale_warnings,
    })
    report = "# Stage 6C event style metrics\n\n"
    report += f"- total shards: {total_shards}\n- total rows: {len(df)}\n- feature dim: {X.shape[1]}\n- runtime seconds: {time.time() - t0:.3f}\n\n"
    report += "## 解释原则\n\nEmbedding/BDD 是统一的行为分布测量层；event-specific metrics 是语义解释层。缺失代理特征时指标保持 NaN，不填 0。\n\n"
    report += "## 有效样本数\n\n```json\n" + __import__("json").dumps(valid_counts, ensure_ascii=False, indent=2) + "\n```\n"
    report += "\n## 有效率与低有效率告警\n\n- valid_rate < 0.01 会写入 `low_valid_rate` 告警，但默认不失败。\n\n```json\n" + __import__("json").dumps({"metric_valid_stats": metric_valid_stats, "low_valid_rate_warnings": low_valid_rate_warnings}, ensure_ascii=False, indent=2) + "\n```\n"
    report += "\n## 不可计算指标\n\n```json\n" + __import__("json").dumps(unavailable, ensure_ascii=False, indent=2) + "\n```\n"
    (out / "event_style_metric_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Stage 6C row-aligned event-specific style metrics.")
    p.add_argument("--shard_manifest", required=True)
    p.add_argument("--feature_schema_path", required=True)
    p.add_argument("--dynamic_event_bins_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    main(p.parse_args())
