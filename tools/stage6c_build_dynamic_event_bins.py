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

from tools.stage6c_common import (
    FeatureResolver,
    combine_and,
    combine_or,
    high_mask,
    iter_progress,
    label_from_mask,
    load_feature_rows,
    load_schema_names,
    low_mask,
    present_mask,
    score_from_parts,
    robust_score,
    write_json,
)

ALIASES = {
    "mean_thw": ["mean_thw", "thw_mean", "time_headway_mean"],
    "min_thw": ["min_thw", "thw_min", "time_headway_min"],
    "mean_front_distance": ["mean_front_distance", "front_distance_mean", "front_gap_mean", "mean_front_gap"],
    "min_front_distance": ["min_front_distance", "front_distance_min", "front_gap_min", "min_front_gap"],
    "front_pressure_score": ["front_pressure_score", "front_pressure", "front_gap_pressure"],
    "front_rel_speed": ["front_rel_speed", "rel_speed", "front_relative_speed", "relative_speed_front"],
    "yielding_score_proxy": ["yielding_score_proxy", "yielding_score", "yield_score_proxy"],
    "cutin_count_proxy": ["cutin_count_proxy", "cut_in_count_proxy", "cutin_count", "cut_in_count"],
    "left_front_min_gap": ["left_front_min_gap", "left_front_gap_min", "target_left_front_gap_min"],
    "right_front_min_gap": ["right_front_min_gap", "right_front_gap_min", "target_right_front_gap_min"],
    "left_rear_min_gap": ["left_rear_min_gap", "left_rear_gap_min", "target_left_rear_gap_min"],
    "right_rear_min_gap": ["right_rear_min_gap", "right_rear_gap_min", "target_right_rear_gap_min"],
    "rear_min_gap": ["rear_min_gap", "min_rear_distance", "rear_gap_min"],
    "neighbor_count": ["neighbor_count", "num_neighbors", "valid_neighbor_count", "interaction_neighbor_count"],
    "interaction_density": ["interaction_density", "traffic_density", "neighbor_density"],
    "ego_speed": ["ego_speed", "ego_speed_mean", "speed_mean", "speed_norm", "speed_norm_mean"],
    "speed_gain": ["speed_gain_proxy", "speed_delta", "delta_speed", "speed_norm_change"],
    "ego_accel": ["ego_accel", "accel_mean", "mean_accel", "acceleration_mean"],
    "max_accel": ["max_accel", "peak_accel", "accel_max", "max_positive_accel"],
    "min_acc": ["min_acc", "min_accel", "max_decel", "peak_decel", "max_deceleration"],
    "max_abs_accel": ["max_abs_accel", "abs_accel_max", "max_abs_acceleration"],
    "jerk_p95": ["jerk_p95", "abs_jerk_p95", "jerk_95p"],
    "max_abs_jerk": ["max_abs_jerk", "abs_jerk_max", "jerk_max_abs"],
    "brake_count": ["brake_count", "hard_brake_count", "braking_count_proxy"],
    "min_ttc": ["min_ttc", "ttc_min", "minimum_ttc"],
    "lane_change_count_proxy": ["lane_change_count_proxy", "lane_change_count", "lc_count_proxy"],
    "lane_change_left_count_proxy": ["lane_change_left_count_proxy", "left_lane_change_count_proxy"],
    "lane_change_right_count_proxy": ["lane_change_right_count_proxy", "right_lane_change_count_proxy"],
    "lane_change_rate_proxy": ["lane_change_rate_proxy", "lc_rate_proxy", "lane_change_rate"],
    "lane_change_duration_proxy": ["lane_change_duration_proxy", "lc_duration_proxy", "lane_change_duration"],
    "rms_yaw_rate": ["rms_yaw_rate", "yaw_rate_rms", "yaw_rate_std"],
    "heading_change_total": ["heading_change_total", "total_heading_change", "heading_delta_total"],
    "rms_curvature": ["rms_curvature", "curvature_rms"],
    "yaw_rate_sign_changes": ["yaw_rate_sign_changes", "yaw_sign_changes", "yaw_oscillation_proxy"],
    "speed_oscillation": ["speed_oscillation_proxy", "speed_oscillation", "speed_std", "speed_norm_std"],
    "stop_count": ["stop_count", "stop_count_proxy", "num_stops"],
}


def get_features(X, resolver):
    return {k: resolver.get(X, k, v) for k, v in ALIASES.items()}


def gt_zero(x):
    if x is None:
        return None
    return np.isfinite(x) & (x > 0)


def build_bins(X, meta, resolver, progress_enabled=True):
    n = len(meta)
    f = get_features(X, resolver)
    df = meta.copy()

    front_present = combine_or([
        present_mask(f["min_thw"], 0.67, True),
        present_mask(f["mean_thw"], 0.67, True),
        present_mask(f["min_front_distance"], 0.67, True),
        present_mask(f["mean_front_distance"], 0.67, True),
        high_mask(f["front_pressure_score"], 0.67),
    ], n)
    following = front_present

    cut_in = combine_or([
        gt_zero(f["cutin_count_proxy"]),
        combine_and([high_mask(f["front_pressure_score"], 0.67), present_mask(f["min_front_distance"], 0.67, True)], n),
        combine_and([present_mask(f["left_front_min_gap"], 0.67, True), present_mask(f["right_front_min_gap"], 0.67, True)], n),
        high_mask(f["yielding_score_proxy"], 0.67),
    ], n)

    ego_speed_ok = high_mask(f["ego_speed"], 0.33)
    slower_front = None
    if f["front_rel_speed"] is not None:
        slower_front = np.isfinite(f["front_rel_speed"]) & (f["front_rel_speed"] < np.nanmedian(f["front_rel_speed"]))
    lateral_context = combine_or([gt_zero(f["lane_change_count_proxy"]), high_mask(f["rms_yaw_rate"], 0.67)], n)
    overtake_opp = combine_and([front_present, combine_or([slower_front, high_mask(f["front_pressure_score"], 0.67)], n), ego_speed_ok], n)
    if lateral_context is not None and overtake_opp is not None:
        overtake_opp = overtake_opp | (overtake_opp & lateral_context)

    dense = combine_or([
        high_mask(f["neighbor_count"], 0.67),
        high_mask(f["interaction_density"], 0.67),
        combine_and([present_mask(f["min_front_distance"], 0.67, True), present_mask(f["left_front_min_gap"], 0.67, True)], n),
        combine_and([present_mask(f["right_front_min_gap"], 0.67, True), present_mask(f["rear_min_gap"], 0.67, True)], n),
    ], n)

    front_pressure = combine_or([high_mask(f["front_pressure_score"], 0.67), present_mask(f["min_front_distance"], 0.67, True), present_mask(f["min_thw"], 0.67, True)], n)
    side_pressure = combine_or([
        present_mask(f["left_front_min_gap"], 0.67, True),
        present_mask(f["right_front_min_gap"], 0.67, True),
        present_mask(f["left_rear_min_gap"], 0.67, True),
        present_mask(f["right_rear_min_gap"], 0.67, True),
    ], n)
    gap_pressure = combine_or([front_pressure, side_pressure, present_mask(f["rear_min_gap"], 0.67, True)], n)
    yield_conflict = combine_or([high_mask(f["yielding_score_proxy"], 0.67), combine_and([front_pressure, side_pressure], n), cut_in], n)

    lane_change = combine_or([gt_zero(f["lane_change_count_proxy"]), gt_zero(f["lane_change_left_count_proxy"]), gt_zero(f["lane_change_right_count_proxy"]), high_mask(f["lane_change_rate_proxy"], 0.67), high_mask(f["rms_yaw_rate"], 0.90), high_mask(f["heading_change_total"], 0.90)], n)

    free_cruising = combine_and([
        None if following is None else ~following,
        None if cut_in is None else ~cut_in,
        None if front_pressure is None else ~front_pressure,
        None if side_pressure is None else ~side_pressure,
        None if lane_change is None else ~lane_change,
    ], n)

    hard_brake = combine_or([
        low_mask(f["min_acc"], 0.10),
        high_mask(f["max_abs_accel"], 0.90),
        high_mask(f["jerk_p95"], 0.90),
        high_mask(f["max_abs_jerk"], 0.90),
        gt_zero(f["brake_count"]),
    ], n)
    late_brake = combine_and([hard_brake, combine_or([present_mask(f["min_thw"], 0.67, True), present_mask(f["min_front_distance"], 0.67, True), high_mask(f["front_pressure_score"], 0.67), present_mask(f["min_ttc"], 0.67, True)], n)], n)

    speed_gain = high_mask(f["speed_gain"], 0.67)
    overtake_executed = combine_and([lane_change, combine_or([speed_gain, high_mask(f["max_accel"], 0.67)], n)], n)

    hesitation = combine_or([high_mask(f["lane_change_duration_proxy"], 0.75), high_mask(f["yaw_rate_sign_changes"], 0.75), high_mask(f["speed_oscillation"], 0.75)], n)

    acc_or_speed = combine_or([high_mask(f["ego_accel"], 0.67), high_mask(f["speed_gain"], 0.67), high_mask(f["ego_speed"], 0.67)], n)
    low_yield = None if f["yielding_score_proxy"] is None else low_mask(f["yielding_score_proxy"], 0.33)
    assertive = combine_and([combine_or([gap_pressure, front_pressure, yield_conflict], n), combine_or([acc_or_speed, low_yield], n)], n)

    stop_go = combine_or([gt_zero(f["stop_count"]), combine_and([low_mask(f["ego_speed"], 0.25), combine_or([high_mask(f["jerk_p95"], 0.67), high_mask(f["speed_oscillation"], 0.67)], n)], n)], n)

    lateral_unstable = combine_or([high_mask(f["rms_yaw_rate"], 0.90), high_mask(f["rms_curvature"], 0.90), high_mask(f["heading_change_total"], 0.90), combine_and([high_mask(f["lane_change_duration_proxy"], 0.75), high_mask(f["yaw_rate_sign_changes"], 0.75)], n)], n)

    specs = [
        ("exposure_following", following, "following", "not_following"),
        ("exposure_cut_in", cut_in, "cut_in_exposure", "no_cut_in_exposure"),
        ("exposure_overtake_opportunity", overtake_opp, "overtake_opportunity", "no_overtake_opportunity"),
        ("exposure_dense_traffic", dense, "dense_traffic", "normal_traffic"),
        ("exposure_front_pressure", front_pressure, "high_front_pressure", "low_front_pressure"),
        ("exposure_side_pressure", side_pressure, "high_side_pressure", "low_side_pressure"),
        ("exposure_gap_pressure", gap_pressure, "small_gap", "normal_gap"),
        ("exposure_yield_conflict", yield_conflict, "yield_conflict", "no_yield_conflict"),
        ("exposure_free_cruising", free_cruising, "free_cruising", "not_free_cruising"),
        ("outcome_ego_lane_change", lane_change, "lane_change", "no_lane_change"),
        ("outcome_overtake_executed", overtake_executed, "overtake_executed", "no_overtake_executed"),
        ("outcome_hard_brake", hard_brake, "hard_brake", "no_hard_brake"),
        ("outcome_late_brake", late_brake, "late_brake", "not_late_brake"),
        ("outcome_hesitation", hesitation, "hesitation", "no_hesitation"),
        ("outcome_assertive_interaction", assertive, "assertive_interaction", "non_assertive_interaction"),
        ("outcome_stop_go", stop_go, "stop_go", "not_stop_go"),
        ("outcome_lateral_unstable", lateral_unstable, "lateral_unstable", "lateral_stable_or_low_activity"),
    ]
    for name, mask, true_label, false_label in iter_progress(specs, enabled=progress_enabled, desc="building dynamic event bins", unit="bin"):
        df[name] = label_from_mask(mask, true_label, false_label, n)

    available = np.zeros(n, dtype=int)
    missing = np.zeros(n, dtype=int)
    for arr in f.values():
        if arr is None:
            missing += 1
        else:
            available += np.isfinite(arr).astype(int)
            missing += (~np.isfinite(arr)).astype(int)
    df["available_feature_count"] = available
    df["missing_feature_count"] = missing
    unknown_counts = df[[s[0] for s in specs]].eq("unknown").sum(axis=1)
    df["event_quality_flag"] = np.where(unknown_counts >= len(specs) // 2, "low_feature_coverage", "ok")
    return df, specs


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
    resolver = FeatureResolver(names)
    df, specs = build_bins(X, meta, resolver, progress_enabled=progress_enabled)

    df.to_csv(out / "dynamic_event_bins.csv", index=False)
    np.save(out / "dynamic_event_bins.npy", df.to_records(index=False))
    schema = {
        "description": "Stage 6C row-aligned dynamic interaction exposure bins and behavior outcome bins.",
        "row_alignment": "global_row is zero-based across shard_manifest order; local_row is zero-based within shard_id.",
        "exposure_columns": [s[0] for s in specs if s[0].startswith("exposure_")],
        "outcome_columns": [s[0] for s in specs if s[0].startswith("outcome_")],
        "columns": df.columns.tolist(),
    }
    write_json(out / "dynamic_event_bin_schema.json", schema)
    warnings = {
        "resolved_features": resolver.resolved,
        "missing_feature_aliases": resolver.missing,
        "optional_inputs": {
            "odd_bins_path": args.odd_bins_path,
            "behavior_bins_path": args.behavior_bins_path,
            "note": "Optional Stage 6B bins are accepted for workflow compatibility but first-pass Stage 6C bins are computed from feature_schema-resolved dynamic proxies.",
        },
    }
    write_json(out / "dynamic_event_bin_warnings.json", warnings)
    count_cols = [s[0] for s in specs]
    counts = {c: df[c].value_counts(dropna=False).to_dict() for c in count_cols}
    report = "# Stage 6C dynamic event bins\n\n"
    report += f"- total shards: {total_shards}\n- total rows: {len(df)}\n- feature dim: {X.shape[1]}\n- runtime seconds: {time.time() - t0:.3f}\n\n"
    report += "## 设计边界\n\n- exposure_* 表示动态交互暴露，可用于后续 matching/control 的候选变量。\n- outcome_* 表示行为结果/风格，主要用于报告和定位，不应被当作纯场景控制变量。\n- 缺失代理特征时输出 `unknown`，不伪造分箱。\n\n"
    report += "## 分箱计数\n\n```json\n" + __import__("json").dumps(counts, ensure_ascii=False, indent=2) + "\n```\n"
    report += "\n## 缺失特征摘要\n\n```json\n" + __import__("json").dumps(resolver.missing, ensure_ascii=False, indent=2) + "\n```\n"
    (out / "dynamic_event_bin_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build Stage 6C dynamic interaction exposure and behavior outcome bins.")
    p.add_argument("--shard_manifest", required=True)
    p.add_argument("--feature_schema_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--odd_bins_path")
    p.add_argument("--behavior_bins_path")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    main(p.parse_args())
