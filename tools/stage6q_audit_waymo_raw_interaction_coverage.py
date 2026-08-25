#!/usr/bin/env python3
"""Audit dynamic lead interactions directly from the raw Waymo full51 source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


STATUS = "STAGE6Q_WAYMO_RAW_INTERACTION_COVERAGE_AUDIT_COMPLETE"
FUNNELS = ["raw_all_vehicle_windows", "formal_target_sampling", "formal_builder_retained"]
EVENTS = [
    "lead_entry",
    "lead_exit",
    "intermittent_following_primary",
    "intermittent_following_strict",
    "front_identity_switch",
    "free_flow_to_closing_to_following",
    "following_to_free_flow",
]


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


def resolve_builder_shards(manifest_path: Path) -> list[Path]:
    manifest = read_json(manifest_path)
    candidates: list[Path] = []
    repo_parent = manifest_path.resolve().parents[3]
    for original in manifest["shard_paths"]:
        source = Path(original)
        if source.exists():
            candidates.append(source)
            continue
        repaired = repo_parent / "outputs" / source.parent.parent.name / "shards" / source.name
        if not repaired.exists():
            raise FileNotFoundError(f"cannot resolve builder shard: {original}")
        candidates.append(repaired)
    return candidates


def load_retained_keys(manifest_path: Path) -> tuple[set[tuple[str, str, int]], dict[str, Any]]:
    keys: set[tuple[str, str, int]] = set()
    shards = resolve_builder_shards(manifest_path)
    for shard in shards:
        meta = np.load(shard / "meta.npy", allow_pickle=True)
        for row in meta:
            keys.add((str(row["scenario_id"]), str(row["target_agent_id"]), int(row["start"])))
    return keys, {"shards": len(shards), "rows": len(keys), "unique_keys": len(keys)}


def raw_files(config: dict[str, Any], override: Path | None, max_files: int | None) -> list[Path]:
    source = config["source"]
    root = (override or Path(source["waymo_dir"])).resolve()
    pattern = re.compile(r"tfrecord-(\d{5})-of-01000$")
    start, end = source["file_indices_inclusive"]
    selected = []
    for path in sorted(root.glob(source["file_glob"])):
        match = pattern.search(path.name)
        if match and start <= int(match.group(1)) <= end:
            selected.append(path)
    if max_files is None and len(selected) != int(source["expected_files"]):
        raise ValueError(f"expected {source['expected_files']} raw files, got {len(selected)}")
    return selected[:max_files] if max_files else selected


def run_segments(values: list[Any]) -> list[tuple[int, int, Any]]:
    if not values:
        return []
    segments: list[tuple[int, int, Any]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            segments.append((start, index, values[start]))
            start = index
    return segments


def has_run(mask: np.ndarray, minimum: int) -> bool:
    return any(end - start >= minimum and bool(value) for start, end, value in run_segments(mask.astype(bool).tolist()))


def ordered_runs(first: np.ndarray, second: np.ndarray, third: np.ndarray | None, minimum: int) -> bool:
    first_runs = [(a, b) for a, b, value in run_segments(first.astype(bool).tolist()) if value and b - a >= minimum]
    second_runs = [(a, b) for a, b, value in run_segments(second.astype(bool).tolist()) if value and b - a >= minimum]
    third_runs = [] if third is None else [(a, b) for a, b, value in run_segments(third.astype(bool).tolist()) if value and b - a >= minimum]
    for _, first_end in first_runs:
        for second_start, second_end in second_runs:
            if second_start < first_end:
                continue
            if third is None:
                return True
            if any(third_start >= second_end for third_start, _ in third_runs):
                return True
    return False


def event_flags(lead_ids: list[str | None], gaps: np.ndarray, closing: np.ndarray, minimum: int) -> dict[str, bool | float | int]:
    occupied = np.asarray([value is not None for value in lead_ids], dtype=bool)
    segments = run_segments(lead_ids)
    entry = any(
        segments[index - 1][2] is None
        and segments[index][2] is not None
        and segments[index - 1][1] - segments[index - 1][0] >= minimum
        and segments[index][1] - segments[index][0] >= minimum
        for index in range(1, len(segments))
    )
    exit_event = any(
        segments[index - 1][2] is not None
        and segments[index][2] is None
        and segments[index - 1][1] - segments[index - 1][0] >= minimum
        and segments[index][1] - segments[index][0] >= minimum
        for index in range(1, len(segments))
    )
    identity_switch = any(
        segments[index - 1][2] is not None
        and segments[index][2] is not None
        and segments[index - 1][2] != segments[index][2]
        and segments[index - 1][1] - segments[index - 1][0] >= minimum
        and segments[index][1] - segments[index][0] >= minimum
        for index in range(1, len(segments))
    )
    valid_count = int(occupied.sum())
    missing_count = int((~occupied).sum())
    intermittent_primary = minimum <= valid_count < int(math.ceil(0.8 * len(occupied))) and missing_count >= minimum
    intermittent_strict = minimum <= valid_count < int(math.ceil(0.5 * len(occupied))) and missing_count >= minimum
    free = ~occupied
    closing_state = occupied & np.isfinite(closing) & (closing > 0.5)
    following = occupied & np.isfinite(gaps) & (gaps <= 40.0)
    return {
        "lead_valid_frames": valid_count,
        "lead_valid_ratio": valid_count / len(occupied),
        "lead_entry": entry,
        "lead_exit": exit_event,
        "intermittent_following_primary": intermittent_primary,
        "intermittent_following_strict": intermittent_strict,
        "front_identity_switch": identity_switch,
        "free_flow_to_closing_to_following": ordered_runs(free, closing_state, following, minimum),
        "following_to_free_flow": ordered_runs(following, free, None, minimum),
    }


def dynamic_lead(
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
    valid: np.ndarray,
    ids: list[str],
    ego_index: int,
    lateral_max: float,
    longitudinal_max: float,
    heading_max_rad: float,
) -> tuple[list[str | None], np.ndarray, np.ndarray]:
    ego_pos = positions[ego_index]
    ego_vel = velocities[ego_index]
    ego_heading = headings[ego_index].copy()
    fallback = np.arctan2(ego_vel[:, 1], ego_vel[:, 0])
    ego_heading = np.where(np.isfinite(ego_heading), ego_heading, fallback)
    ego_heading = np.where(np.isfinite(ego_heading), ego_heading, 0.0)
    delta = positions - ego_pos[None, :, :]
    cosine, sine = np.cos(ego_heading)[None, :], np.sin(ego_heading)[None, :]
    longitudinal = delta[:, :, 0] * cosine + delta[:, :, 1] * sine
    lateral = -delta[:, :, 0] * sine + delta[:, :, 1] * cosine
    heading_delta = (headings - ego_heading[None, :] + np.pi) % (2.0 * np.pi) - np.pi
    mask = (
        valid
        & valid[ego_index][None, :]
        & np.isfinite(longitudinal)
        & np.isfinite(lateral)
        & (longitudinal > 0.0)
        & (longitudinal <= longitudinal_max)
        & (np.abs(lateral) <= lateral_max)
        & (np.abs(heading_delta) <= heading_max_rad)
    )
    mask[ego_index] = False
    candidate_distance = np.where(mask, longitudinal, np.inf)
    chosen = np.argmin(candidate_distance, axis=0)
    best = candidate_distance[chosen, np.arange(candidate_distance.shape[1])]
    occupied = np.isfinite(best)
    lead_ids: list[str | None] = [ids[int(chosen[t])] if occupied[t] else None for t in range(len(occupied))]
    gaps = np.where(occupied, best, np.nan)
    ego_speed = np.linalg.norm(ego_vel, axis=1)
    lead_speed = np.linalg.norm(velocities[chosen, np.arange(len(chosen))], axis=1)
    closing = np.where(occupied, ego_speed - lead_speed, np.nan)
    return lead_ids, gaps, closing


def extract_vehicle_arrays(scenario: Any, window_len: int) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks = [track for track in scenario.tracks if int(track.object_type) == 1 and len(track.states) >= window_len]
    ids = [str(track.id) for track in tracks]
    n = len(tracks)
    positions = np.full((n, window_len, 2), np.nan, dtype=np.float32)
    velocities = np.full((n, window_len, 2), np.nan, dtype=np.float32)
    headings = np.full((n, window_len), np.nan, dtype=np.float32)
    valid = np.zeros((n, window_len), dtype=bool)
    for row, track in enumerate(tracks):
        for frame, state in enumerate(track.states[:window_len]):
            if not state.valid:
                continue
            positions[row, frame] = [state.center_x, state.center_y]
            velocities[row, frame] = [state.velocity_x, state.velocity_y]
            headings[row, frame] = state.heading
            valid[row, frame] = True
    return ids, positions, velocities, headings, valid


def aggregate_row(counts: dict[tuple[str, float], Counter], funnel: str, lateral: float, flags: dict[str, Any]) -> None:
    counter = counts[(funnel, lateral)]
    counter["eligible_windows"] += 1
    counter["lead_occupied_windows"] += int(flags["lead_valid_frames"] > 0)
    for event in EVENTS:
        counter[event] += int(bool(flags[event]))


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    primary = [row for row in summary["coverage"] if row["lateral_max_m"] == 3.0]
    by_funnel = {row["funnel"]: row for row in primary}
    raw = by_funnel["raw_all_vehicle_windows"]
    formal = by_funnel["formal_target_sampling"]
    retained = by_funnel["formal_builder_retained"]
    lines = [
        "# Stage 6Q Waymo full51 原始交互覆盖率审计（中文）",
        "",
        "## 核心结论",
        "",
        f"- intermittent-following=0 的根因：**{summary['root_cause_zh']}**",
        f"- 原始全部合格vehicle窗口中，primary intermittent为 `{raw['intermittent_following_primary']}`；正式前64 target sampling中为 `{formal['intermittent_following_primary']}`；这些窗口在正式builder retained集合中仍有 `{retained['intermittent_following_primary']}` 个动态几何proxy命中。",
        f"- Stage6O v1正式静态front-slot统计仍为 `0`，其状态保持 `{summary['stage6o_status']}`，本审计未修改或覆盖该产物。",
        "- 正式builder只在窗口参考帧选择一次固定front，并对该track调用整窗有效率>=0.8的sanitize；这会把动态entry/exit压缩成空槽或>=0.8持续槽，是代码级结构性机制。",
        "",
        "## 3.0m主规则覆盖率漏斗",
        "",
        "| 漏斗 | 合格窗口 | lead entry | lead exit | intermittent <0.8 | intermittent <0.5 | identity switch | free→closing→following | following→free |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['funnel']} | {row['eligible_windows']} | {row['lead_entry']} | {row['lead_exit']} | "
            f"{row['intermittent_following_primary']} | {row['intermittent_following_strict']} | {row['front_identity_switch']} | "
            f"{row['free_flow_to_closing_to_following']} | {row['following_to_free_flow']} |"
        )
    lines.extend(
        [
            "",
            "## 判定与下一步",
            "",
            f"- Stage6O冻结门槛为5000，raw primary intermittent计数为 {raw['intermittent_following_primary']}；没有事后降低threshold。",
            f"- 推荐：**{summary['next_step_zh']}**",
            "- 先让builder支持逐帧动态front identity与显式mask，再重新生成一个新版本数据集并重新执行Stage6O；旧Stage6O v1必须继续作为blocked证据保留。",
            "- 在builder修复并通过冻结coverage门槛后，才继续准备Interaction-aware v2；本审计不授权训练。",
            "",
            "## 解释边界",
            "",
            "- raw动态lead使用冻结的ego坐标几何proxy，目的是覆盖率漏斗审计，不替代lane-aware正式语义。",
            "- 2/3/4m lateral敏感性结果见CSV；主结论不得依赖单一几何阈值偶然性。",
            "- 论文定义保持：behavior style = ego response conditioned on traffic / interaction context。",
        ]
    )
    (output_dir / "stage6q_waymo_raw_interaction_coverage_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_coverage(output_dir: Path, coverage: list[dict[str, Any]]) -> None:
    primary = [row for row in coverage if row["lateral_max_m"] == 3.0]
    labels = [row["funnel"] for row in primary]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, event in enumerate(["lead_entry", "lead_exit", "intermittent_following_primary"]):
        ax.bar(x + (offset - 1) * width, [row[event] for row in primary], width, label=event)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Window count")
    ax.set_title("Waymo full51 dynamic interaction coverage funnel (3.0m proxy)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "stage6q_interaction_coverage_funnel.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config.resolve())
    if config.get("schema_version") != "stage6q_waymo_raw_interaction_coverage_audit_design_v1":
        raise ValueError("not a Stage6Q v1 config")
    stage6o = read_json(args.stage6o_manifest.resolve())
    if stage6o.get("status") != "FROZEN_BLOCKED_WAYMO_COVERAGE_NOT_TRAINING":
        raise ValueError("Stage6O v1 is not the expected blocked artifact")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    files = raw_files(config, args.waymo_dir, args.max_files)
    retained_keys, retained_summary = load_retained_keys(args.builder_manifest.resolve())
    source_manifest = [
        {"index": index, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for index, path in enumerate(tqdm(files, desc="Hash raw Waymo files", unit="file"))
    ]
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    source = config["source"]
    dynamic = config["dynamic_lead_rule"]
    minimum = int(config["events"]["minimum_state_run_frames"])
    lateral_values = [float(value) for value in dynamic["sensitivity_lateral_max_m"]]
    counts: dict[tuple[str, float], Counter] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    total_scenarios = 0
    total_vehicle_tracks = 0
    for file_index, path in enumerate(files):
        dataset = tf.data.TFRecordDataset(str(path), num_parallel_reads=1)
        for raw_record in tqdm(dataset, desc=f"Raw audit {file_index + 1}/{len(files)}", unit="scenario", leave=False):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(raw_record.numpy()))
            total_scenarios += 1
            ids, positions, velocities, headings, valid = extract_vehicle_arrays(scenario, int(source["window_len"]))
            total_vehicle_tracks += len(ids)
            if not ids:
                continue
            speed = np.linalg.norm(velocities, axis=2)
            valid_ratio = valid.mean(axis=1)
            mean_speed = np.divide(
                np.where(valid, speed, 0.0).sum(axis=1),
                np.maximum(valid.sum(axis=1), 1),
            )
            eligible = np.flatnonzero((valid_ratio >= 0.8) & (mean_speed >= 1.0))
            eligible_set = set(int(value) for value in eligible)
            retained_indices = {
                index
                for index, agent_id in enumerate(ids)
                if (str(scenario.scenario_id), agent_id, 0) in retained_keys
            }
            audit_indices = sorted(eligible_set | retained_indices)
            for ego_index in audit_indices:
                key = (str(scenario.scenario_id), ids[int(ego_index)], 0)
                funnels: list[str] = []
                if int(ego_index) in eligible_set:
                    funnels.append("raw_all_vehicle_windows")
                    if int(ego_index) < 64:
                        funnels.append("formal_target_sampling")
                if key in retained_keys:
                    funnels.append("formal_builder_retained")
                primary_flags: dict[str, Any] | None = None
                for lateral in lateral_values:
                    lead_ids, gaps, closing = dynamic_lead(
                        positions,
                        velocities,
                        headings,
                        valid,
                        ids,
                        int(ego_index),
                        lateral,
                        float(dynamic["longitudinal_max_m"]),
                        math.radians(float(dynamic["absolute_heading_difference_max_deg"])),
                    )
                    flags = event_flags(lead_ids, gaps, closing, minimum)
                    for funnel in funnels:
                        aggregate_row(counts, funnel, lateral, flags)
                    if lateral == 3.0:
                        primary_flags = flags
                if primary_flags and any(bool(primary_flags[event]) for event in EVENTS) and len(examples) < args.max_examples:
                    examples.append(
                        {
                            "scenario_id": str(scenario.scenario_id),
                            "target_agent_id": ids[int(ego_index)],
                            "formal_target_sampling": int(ego_index) < 64,
                            "formal_builder_retained": key in retained_keys,
                            **primary_flags,
                        }
                    )
        print(
            f"[stage6q] files={file_index + 1}/{len(files)} scenarios={total_scenarios} "
            f"raw_windows={counts[('raw_all_vehicle_windows', 3.0)]['eligible_windows']} "
            f"raw_intermittent={counts[('raw_all_vehicle_windows', 3.0)]['intermittent_following_primary']}",
            flush=True,
        )
    coverage: list[dict[str, Any]] = []
    for lateral in lateral_values:
        for funnel in FUNNELS:
            current = counts[(funnel, lateral)]
            coverage.append(
                {
                    "funnel": funnel,
                    "lateral_max_m": lateral,
                    **{column: int(current[column]) for column in ["eligible_windows", "lead_occupied_windows", *EVENTS]},
                }
            )
    raw_primary = next(row for row in coverage if row["funnel"] == "raw_all_vehicle_windows" and row["lateral_max_m"] == 3.0)
    enough = raw_primary["intermittent_following_primary"] >= 5000
    root_cause = (
        "原始Waymo full51并不缺少动态intermittent/transition；正式builder的首帧固定槽位与>=0.8整窗有效率规则造成结构性过滤"
        if enough
        else "原始Waymo full51在冻结几何proxy下的intermittent覆盖也不足，不能仅归因于builder"
    )
    next_step = (
        "优先修复dataset builder的逐帧动态front assignment与mask语义，不扩大Waymo；修复版通过coverage后再准备Interaction-aware v2"
        if enough
        else "先设计扩大Waymo source的预冻结抽样方案，同时保留builder动态槽位修复要求；当前不启动训练"
    )
    summary = {
        "schema_version": "stage6q_waymo_raw_interaction_coverage_audit_results_v1",
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 258,
        "config_sha256": sha256_file(args.config.resolve()),
        "stage6o_manifest_sha256_before": sha256_file(args.stage6o_manifest.resolve()),
        "stage6o_status": stage6o["status"],
        "stage6o_modified": False,
        "training_performed": False,
        "waymo_source_expanded": False,
        "threshold_lowered": False,
        "files_processed": len(files),
        "scenarios_processed": total_scenarios,
        "vehicle_tracks_seen": total_vehicle_tracks,
        "raw_source_manifest": source_manifest,
        "formal_builder_retained_inventory": retained_summary,
        "coverage": coverage,
        "stage6o_intermittent_following_count": 0,
        "stage6o_gate_threshold": 5000,
        "root_cause_zh": root_cause,
        "next_step_zh": next_step,
        "builder_code_mechanism": "assign_stage5d_slots is called once at the reference frame; the selected fixed track then passes sanitize_track_window with min_valid_ratio=0.8",
        "geometric_proxy_not_lane_semantic_replacement": True,
    }
    with (output_dir / "stage6q_coverage_by_funnel_and_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0].keys()))
        writer.writeheader()
        writer.writerows(coverage)
    if examples:
        with (output_dir / "stage6q_dynamic_interaction_examples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(examples[0].keys()))
            writer.writeheader()
            writer.writerows(examples)
    write_json(output_dir / "stage6q_waymo_raw_source_manifest.json", {"files": source_manifest})
    write_json(output_dir / "stage6q_waymo_raw_interaction_coverage_summary.json", summary)
    plot_coverage(output_dir, coverage)
    write_report(output_dir, summary)
    if sha256_file(args.stage6o_manifest.resolve()) != summary["stage6o_manifest_sha256_before"]:
        raise RuntimeError("Stage6O v1 changed during Stage6Q audit")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--waymo_dir", type=Path)
    parser.add_argument("--builder_manifest", type=Path, required=True)
    parser.add_argument("--stage6o_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--max_examples", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "root_cause_zh": result["root_cause_zh"]}, ensure_ascii=False))
