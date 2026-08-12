#!/usr/bin/env python3
"""Freeze Dynamic Builder v2 full51 training readiness without training a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6q_audit_waymo_raw_interaction_coverage import event_flags


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    merged = read_json(args.dynamic_full51_manifest)
    old = read_json(args.stage6o_v1_manifest)
    if old.get("status") != "FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING":
        raise ValueError("Stage6O v1 is not the immutable blocked artifact")
    if sha256_file(args.stage6o_v1_manifest) != args.expected_stage6o_v1_sha256:
        raise ValueError("Stage6O v1 changed before v2 readiness freeze")
    if merged.get("status") != "DYNAMIC_FULL51_FINALIZED_PENDING_STAGE6O_V2":
        raise ValueError("Dynamic full51 has not been globally finalized")
    part_manifests = [read_json(Path(root) / "dynamic_builder_v2_manifest.json") for root in merged["part_roots"]]
    semantic_strict_parts = all(
        manifest.get("assignment_mode_required") == "lane_aware_only"
        and manifest.get("geometric_adjacent_lane_inference") is False
        for manifest in part_manifests
    )
    counts = Counter()
    split_scenarios: dict[str, set[str]] = {name: set() for name in ["train", "val", "test"]}
    finite_violations = 0
    shape_violations = 0
    derivative_violations = 0
    switch_counts = np.zeros(5, dtype=np.int64)
    transition_counts = np.zeros(5, dtype=np.int64)
    raw_abs: list[list[np.ndarray]] = [[], [], []]
    longitudinal_window_rms: list[list[np.ndarray]] = [[], []]
    normalized_abs_max = 0.0
    rows = 0
    slot_frame_valid = np.zeros(5, dtype=np.int64)
    slot_window_occupied = np.zeros(5, dtype=np.int64)
    total_frames = 0
    for raw_shard in merged["shard_paths"]:
        shard = Path(raw_shard)
        meta = np.load(shard / "meta.npy", allow_pickle=True)
        split = np.load(shard / "split.npy", allow_pickle=True).astype(str)
        ids = np.load(shard / "slot_track_id_timeline.npy").astype(str)
        valid = np.load(shard / "slot_valid_mask.npy")
        switches = np.load(shard / "slot_identity_switch_mask.npy")
        derivative = np.load(shard / "slot_derivative_valid_mask.npy")
        neighbor = np.load(shard / "neighbor_seq.npy", mmap_mode="r")
        raw_target = np.load(shard / "longitudinal_supervision_v2_raw.npy", mmap_mode="r")
        target = np.load(shard / "longitudinal_supervision_v2.npy", mmap_mode="r")
        rows += len(meta)
        total_frames += int(valid.shape[0] * valid.shape[2])
        slot_frame_valid += valid.sum(axis=(0, 2))
        slot_window_occupied += valid.any(axis=2).sum(axis=0)
        expected = (len(meta), 5, neighbor.shape[2])
        shape_violations += int(ids.shape != expected or valid.shape != expected or switches.shape != expected or derivative.shape != expected)
        finite_violations += int(not np.isfinite(neighbor).all()) + int(not np.isfinite(raw_target).all()) + int(not np.isfinite(target).all())
        derivative_violations += int(np.sum((~derivative) & ((neighbor[:, :, :, 12] != 0) | (neighbor[:, :, :, 14] != 0))))
        switch_counts += switches.sum(axis=(0, 2))
        transition_counts += (valid[:, :, 1:] & valid[:, :, :-1]).sum(axis=(0, 2))
        normalized_abs_max = max(normalized_abs_max, float(np.max(np.abs(target))))
        for column in range(3):
            raw_abs[column].append(np.abs(np.asarray(raw_target[:, :, column])).reshape(-1))
        for output_index, column in enumerate([1, 2]):
            values = np.asarray(raw_target[:, :, column], dtype=np.float64)
            longitudinal_window_rms[output_index].append(np.sqrt(np.mean(np.square(values), axis=1)))
        for index, row in enumerate(meta):
            split_name = str(split[index])
            split_scenarios[split_name].add(str(row["scenario_id"]))
            front_valid = valid[index, 0]
            lead = [None if value == "-1" else value for value in ids[index, 0].tolist()]
            flags = event_flags(
                lead,
                np.where(front_valid, neighbor[index, 0, :, 5], np.nan),
                np.where(front_valid, neighbor[index, 0, :, 8], np.nan),
                5,
            )
            if split_name == "train":
                for name in ["lead_entry", "lead_exit", "intermittent_following_primary", "front_identity_switch", "free_flow_to_closing_to_following", "following_to_free_flow"]:
                    counts[name] += int(bool(flags[name]))
    overlap = (split_scenarios["train"] & split_scenarios["val"]) | (split_scenarios["train"] & split_scenarios["test"]) | (split_scenarios["val"] & split_scenarios["test"])
    raw_q99 = [float(np.quantile(np.concatenate(values), 0.99)) for values in raw_abs]
    window_rms_quantiles = {}
    for name, values in zip(["rms_accel_mps2", "rms_jerk_mps3"], longitudinal_window_rms):
        merged_values = np.concatenate(values)
        window_rms_quantiles[name] = {
            f"q{int(quantile * 100):02d}": float(np.quantile(merged_values, quantile))
            for quantile in [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        }
    gate = config["full51_readiness_gate"]
    switch_rates = (switch_counts / np.maximum(1, transition_counts)).tolist()
    slot_frame_coverage = (slot_frame_valid / max(1, total_frames)).tolist()
    slot_window_coverage = (slot_window_occupied / max(1, rows)).tolist()
    checks = {
        "semantic_strict_assignment": semantic_strict_parts,
        "intermittent_train": counts["intermittent_following_primary"] >= gate["intermittent_train_min"],
        "split_no_leakage": len(overlap) <= gate["scenario_cross_split_overlap_max"],
        "finite_shape": finite_violations <= gate["nonfinite_values_max"] and shape_violations == 0,
        "derivative_cross_identity": derivative_violations <= gate["derivative_cross_identity_violations_max"],
        "slot_continuity": max(switch_rates) <= gate["slot_switch_rate_max"],
        "longitudinal_raw_quality": all(value <= limit for value, limit in zip(raw_q99, gate["longitudinal_raw_abs_q99_max"])),
        "longitudinal_normalized_quality": normalized_abs_max <= gate["normalized_target_abs_max"],
    }
    passed = all(checks.values())
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    result = {
        "schema_version": "stage6o_v2_dynamic_interaction_training_readiness_v1",
        "status": "FROZEN_READY_FOR_INTERACTION_AWARE_V2_PREPARATION" if passed else "FROZEN_BLOCKED_DYNAMIC_V2_DATA_QUALITY",
        "checks": checks,
        "train_event_counts": dict(counts),
        "scenario_cross_split_overlap_count": len(overlap),
        "nonfinite_violation_count": finite_violations,
        "shape_violation_count": shape_violations,
        "derivative_cross_identity_violation_count": derivative_violations,
        "slot_switch_rates": switch_rates,
        "slot_valid_frame_ratios": slot_frame_coverage,
        "slot_occupied_window_ratios": slot_window_coverage,
        "longitudinal_raw_abs_q99": raw_q99,
        "longitudinal_window_rms_quantiles": window_rms_quantiles,
        "longitudinal_normalized_abs_max": normalized_abs_max,
        "row_count": rows,
        "dynamic_full51_manifest_sha256": sha256_file(args.dynamic_full51_manifest),
        "stage6o_v1_sha256_unchanged": sha256_file(args.stage6o_v1_manifest),
        "checkpoint_training_launched": False,
        "waymo_source_expanded": False,
    }
    (output / "stage6o_v2_training_readiness_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Stage 6O-v2 Dynamic Interaction训练准备度冻结",
        "",
        f"## 结论：{result['status']}",
        "",
        f"- train intermittent-following：{counts['intermittent_following_primary']}（冻结门槛{gate['intermittent_train_min']}）。",
        f"- split跨集合scenario重叠：{len(overlap)}。",
        f"- finite/shape违规：{finite_violations}/{shape_violations}。",
        f"- 跨identity导数违规：{derivative_violations}；五slot switch rate：{switch_rates}。",
        f"- 五slot帧覆盖率：{slot_frame_coverage}；窗口覆盖率：{slot_window_coverage}。",
        f"- longitudinal raw |q99|：{raw_q99}；normalized max abs：{normalized_abs_max:.4f}。",
        f"- longitudinal窗口RMS分位数：{window_rms_quantiles}。",
        f"- 全部门禁：{checks}。",
        "- Stage6O v1保持原SHA256且永久BLOCKED；本阶段未训练checkpoint、未扩大Waymo。",
    ]
    (output / "stage6o_v2_training_readiness_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dynamic_full51_manifest", type=Path, required=True)
    parser.add_argument("--stage6o_v1_manifest", type=Path, required=True)
    parser.add_argument("--expected_stage6o_v1_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
