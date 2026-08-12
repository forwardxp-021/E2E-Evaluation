#!/usr/bin/env python3
"""Versioned Waymo builder with per-frame semantic slots and v2 longitudinal targets."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import build_waymo_5neighbor_context_dataset as legacy
from tools.interaction_context_features import aggregate_interaction_features
from tools.lane_aware_assignment import assign_neighbors_lane_aware
from tools.stage5d_context_core import (
    SLOT_NAMES,
    build_context_traj_from_standard_tracks,
    build_ego_features_8d,
    build_neighbor_features_15d,
)
from tools.trajectory_context_utils import localize, sanitize_track_window
from tools.waymo_lane_utils import find_best_lane_for_agent


SCHEMA_VERSION = "waymo_dynamic_interaction_builder_v2"
ORIGINAL_FLUSH = legacy.flush_shard
ORIGINAL_MAIN = legacy.main


def wrap_angle(values: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def assignment_config(args) -> dict:
    names = [
        "lane_max_lateral_distance", "lane_max_heading_diff_deg", "adjacent_lane_min_offset",
        "adjacent_lane_max_offset", "adjacent_lane_max_heading_diff_deg", "lane_search_radius",
        "lane_topk_candidates", "front_max_distance", "side_front_max_distance",
        "side_rear_max_distance", "lane_lateral_tolerance", "slot_heading_diff_deg",
        "static_speed_threshold", "disable_lane_spatial_index",
    ]
    return {
        **{name: getattr(args, name) for name in names},
        "ego_projection_precomputed": True,
        "candidate_projections_complete": True,
        # Dynamic v2 is a semantic dataset. Missing proto adjacency must yield an
        # empty side slot, not a geometry-only lane guess near intersections.
        "allow_geometric_adjacent_lane_inference": False,
    }


def track_projection_cache(tracks: dict[str, np.ndarray], lane_infos: dict, args) -> dict[tuple[str, int], dict]:
    if not lane_infos or args.assignment_mode == "geometric_only":
        return {}
    cache: dict[tuple[str, int], dict] = {}
    for agent_id, track in tracks.items():
        for frame, state in enumerate(track):
            if state[5] <= 0.5 or not np.isfinite(state[:2]).all():
                continue
            heading = float(state[4]) if np.isfinite(state[4]) else float(np.arctan2(state[3], state[2]))
            projection, _, _ = find_best_lane_for_agent(
                state[:2], heading, lane_infos,
                args.lane_max_lateral_distance, np.deg2rad(args.lane_max_heading_diff_deg),
                search_radius=args.lane_search_radius, topk_candidates=args.lane_topk_candidates,
                disable_spatial_index=args.disable_lane_spatial_index,
            )
            if projection is not None:
                cache[(agent_id, frame)] = projection
    return cache


def process_one_scenario(sid, tracks, lane_infos, args, cnt, timing, global_row_start):
    """Drop-in replacement for the legacy static-window assignment."""
    batch = legacy.ScenarioOutputBatch()
    slot_valid_counts = defaultdict(int)
    assignment_counts = {slot: {"lane_aware": 0, "geometric_fallback": 0, "empty": 0, "sanitize_failed": 0} for slot in SLOT_NAMES}
    split = legacy.split_of_sid(sid)
    ids = list(tracks.keys())[: args.max_agents_per_scenario]
    cnt["targets"] += len(ids)
    projections = track_projection_cache(tracks, lane_infos, args)
    cfg = assignment_config(args)
    feature_names = None
    for agent_id in ids:
        raw_ego = tracks[agent_id]
        for start in range(0, max(0, len(raw_ego) - args.window_len + 1), args.stride):
            cnt["windows_total"] += 1
            ego_window, ego_valid, _ = sanitize_track_window(
                raw_ego[start : start + args.window_len], args.dt, agent_id, 1.0 - args.min_valid_ratio
            )
            if ego_window is None:
                cnt["f_invalid"] += 1
                cnt["ego_windows_dropped_sanitize_failed"] += 1
                continue
            speed = np.hypot(ego_window[:, 2], ego_window[:, 3])
            if np.nanmean(speed) < args.min_speed:
                cnt["f_static"] += 1
                continue
            reference = int(np.flatnonzero((ego_valid > 0.5) & (speed > 1e-3))[0])
            origin = ego_window[reference, :2]
            base_heading = float(ego_window[reference, 4])
            ego, ego_heading, _ = build_ego_features_8d(ego_window, origin, base_heading, args.dt)
            neighbor = np.zeros((len(SLOT_NAMES), args.window_len, 15), dtype=np.float32)
            slot_ids = [["-1"] * args.window_len for _ in SLOT_NAMES]
            derivative_valid = np.zeros((len(SLOT_NAMES), args.window_len), dtype=bool)
            switch_mask = np.zeros((len(SLOT_NAMES), args.window_len), dtype=bool)
            previous_id: list[str | None] = [None] * len(SLOT_NAMES)
            previous_speed: list[float | None] = [None] * len(SLOT_NAMES)
            previous_heading: list[float | None] = [None] * len(SLOT_NAMES)
            window_laneaware = [False] * len(SLOT_NAMES)
            window_fallback = [False] * len(SLOT_NAMES)
            for local_t in range(args.window_len):
                global_t = start + local_t
                if ego_valid[local_t] <= 0.5:
                    previous_id = [None] * len(SLOT_NAMES)
                    previous_speed = [None] * len(SLOT_NAMES)
                    previous_heading = [None] * len(SLOT_NAMES)
                    continue
                states = {}
                candidate_projections = {}
                for other_id, track in tracks.items():
                    if other_id == agent_id or global_t >= len(track):
                        continue
                    state = track[global_t]
                    if state[5] <= 0.5 or not np.isfinite(state[:4]).all():
                        continue
                    heading = float(state[4]) if np.isfinite(state[4]) else float(np.arctan2(state[3], state[2]))
                    states[other_id] = {
                        "x": float(state[0]), "y": float(state[1]), "velocity_x": float(state[2]),
                        "velocity_y": float(state[3]), "heading": heading,
                        "speed": float(np.hypot(state[2], state[3])), "valid": True,
                    }
                    projection = projections.get((other_id, global_t))
                    if projection is not None:
                        candidate_projections[other_id] = projection
                ego_state = {
                    "x": float(ego_window[local_t, 0]), "y": float(ego_window[local_t, 1]),
                    "heading": float(ego_heading[local_t]), "velocity_x": float(ego_window[local_t, 2]),
                    "velocity_y": float(ego_window[local_t, 3]), "speed": float(speed[local_t]),
                }
                result = assign_neighbors_lane_aware(
                    ego_state, states, lane_infos=lane_infos, assignment_mode=args.assignment_mode,
                    config=cfg, ego_projection=projections.get((agent_id, global_t)),
                    candidate_projections=candidate_projections,
                )
                batch.debug_rows.extend(result.per_slot_debug)
                for slot_index, slot in enumerate(SLOT_NAMES):
                    selected = result.slot_to_agent.get(slot)
                    method = legacy.get_slot_method(result, slot)
                    window_laneaware[slot_index] |= method == "lane_aware"
                    window_fallback[slot_index] |= method == "geometric_fallback"
                    if not selected or selected not in states:
                        previous_id[slot_index] = None
                        previous_speed[slot_index] = None
                        previous_heading[slot_index] = None
                        continue
                    state = states[selected]
                    delta_xy = localize(
                        np.asarray([[state["x"], state["y"]]], dtype=np.float32),
                        ego_window[local_t, :2], float(ego_heading[local_t]),
                    )[0]
                    relative_velocity = localize(
                        np.asarray([[state["velocity_x"] - ego_window[local_t, 2], state["velocity_y"] - ego_window[local_t, 3]]], dtype=np.float32),
                        np.asarray([0.0, 0.0], dtype=np.float32), float(ego_heading[local_t]),
                    )[0]
                    heading_relative = float(wrap_angle(state["heading"] - ego_heading[local_t]))
                    same_identity = previous_id[slot_index] == selected
                    switch_mask[slot_index, local_t] = previous_id[slot_index] is not None and not same_identity
                    derivative_valid[slot_index, local_t] = bool(same_identity)
                    acceleration = (
                        (state["speed"] - float(previous_speed[slot_index])) / args.dt
                        if same_identity and previous_speed[slot_index] is not None else 0.0
                    )
                    yaw_rate = (
                        float(wrap_angle(heading_relative - float(previous_heading[slot_index]))) / args.dt
                        if same_identity and previous_heading[slot_index] is not None else 0.0
                    )
                    neighbor[slot_index, local_t] = build_neighbor_features_15d(
                        rel_x=float(delta_xy[0]), rel_y=float(delta_xy[1]),
                        rel_vx=float(relative_velocity[0]), rel_vy=float(relative_velocity[1]),
                        ego_forward_speed=float(ego[local_t, 2]), neighbor_speed=state["speed"],
                        neighbor_accel=acceleration, heading_rel=heading_relative,
                        neighbor_yaw_rate=yaw_rate, ttc_cap=args.ttc_cap, thw_cap=args.thw_cap,
                    )
                    slot_ids[slot_index][local_t] = selected
                    slot_valid_counts[slot] += 1
                    previous_id[slot_index] = selected
                    previous_speed[slot_index] = state["speed"]
                    previous_heading[slot_index] = heading_relative
            for slot_index, slot in enumerate(SLOT_NAMES):
                method = "lane_aware" if window_laneaware[slot_index] else "geometric_fallback" if window_fallback[slot_index] else "empty"
                assignment_counts[slot][method] += 1
            context = build_context_traj_from_standard_tracks(ego, neighbor)
            interaction, feature_names = aggregate_interaction_features(ego, neighbor, args.dt)
            row_index = global_row_start + len(batch.ego_seq)
            batch.ego_seq.append(ego)
            batch.neighbor_seq.append(neighbor)
            batch.context_traj.append(context)
            batch.context_mask.append((neighbor[:, :, 0] > 0.5).T)
            batch.context_mask_window.append(np.max(neighbor[:, :, 0], axis=1) > 0.5)
            batch.neighbor_slot_ids.append(slot_ids)
            batch.splits.append(split)
            batch.interaction_raw.append(np.nan_to_num(interaction, nan=0.0, posinf=1e6, neginf=-1e6))
            lane_assignment_success = bool(any(window_laneaware))
            fallback_used = bool(any(window_fallback))
            lane_context_quality = (
                "dynamic_lane_aware"
                if lane_assignment_success and not fallback_used
                else "dynamic_mixed"
                if lane_assignment_success
                else "dynamic_geometric_fallback"
                if fallback_used
                else "dynamic_empty"
            )
            batch.meta_rows.append(
                (
                    row_index,
                    str(sid),
                    str(agent_id),
                    int(start),
                    int(args.window_len),
                    split,
                    args.assignment_mode,
                    lane_assignment_success,
                    fallback_used,
                    lane_context_quality,
                )
            )
            # Stored as dynamic attributes and recovered by the main-loop-compatible wrapper below.
            if not hasattr(batch, "dynamic_derivative_valid"):
                batch.dynamic_derivative_valid = []
                batch.dynamic_switch_mask = []
            batch.dynamic_derivative_valid.append(derivative_valid)
            batch.dynamic_switch_mask.append(switch_mask)
            cnt["kept"] += 1
    return batch, slot_valid_counts, assignment_counts, feature_names


def flush_shard(batch, shard_idx, out_dir):
    shard = ORIGINAL_FLUSH(batch, shard_idx, out_dir)
    slot_ids = np.asarray(batch["neighbor_slot_ids"], dtype=str)
    valid = slot_ids != "-1"
    switches = np.zeros(valid.shape, dtype=bool)
    derivative_valid = np.zeros(valid.shape, dtype=bool)
    switches[:, :, 1:] = valid[:, :, 1:] & valid[:, :, :-1] & (slot_ids[:, :, 1:] != slot_ids[:, :, :-1])
    derivative_valid[:, :, 1:] = valid[:, :, 1:] & valid[:, :, :-1] & (slot_ids[:, :, 1:] == slot_ids[:, :, :-1])
    np.save(shard / "slot_track_id_timeline.npy", slot_ids)
    np.save(shard / "slot_valid_mask.npy", valid)
    np.save(shard / "slot_identity_switch_mask.npy", switches)
    np.save(shard / "slot_derivative_valid_mask.npy", derivative_valid)
    neighbor = np.load(shard / "neighbor_seq.npy", mmap_mode="r")
    if np.any(neighbor[:, :, :, 12][~derivative_valid] != 0.0) or np.any(neighbor[:, :, :, 14][~derivative_valid] != 0.0):
        raise RuntimeError("cross-identity temporal derivative was not reset")
    return shard


def median5(values: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(values, dtype=np.float64), (2, 2), mode="edge")
    return np.asarray([np.median(padded[index : index + 5]) for index in range(len(values))], dtype=np.float64)


def build_longitudinal_supervision(out_dir: Path) -> dict:
    summary = json.loads((out_dir / "build_summary.json").read_text())
    shards = [Path(path) for path in summary["shard_paths"]]
    train_values = []
    for shard in shards:
        ego = np.load(shard / "ego_seq.npy", mmap_mode="r")
        raw = np.empty((len(ego), ego.shape[1], 3), dtype=np.float32)
        for row in range(len(ego)):
            speed = median5(ego[row, :, 5])
            accel = np.diff(speed, prepend=speed[0]) / 0.1
            jerk = np.diff(accel, prepend=accel[0]) / 0.1
            raw[row] = np.stack([speed, accel, jerk], axis=1)
        np.save(shard / "longitudinal_supervision_v2_raw.npy", raw)
        split = np.load(shard / "split.npy", allow_pickle=True).astype(str)
        if np.any(split == "train"):
            train_values.append(raw[split == "train"].reshape(-1, 3))
    train = np.concatenate(train_values, axis=0).astype(np.float64)
    q01, q99 = np.quantile(train, [0.01, 0.99], axis=0)
    clipped = np.clip(train, q01, q99)
    median = np.median(clipped, axis=0)
    q25, q75 = np.quantile(clipped, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    for shard in shards:
        raw = np.load(shard / "longitudinal_supervision_v2_raw.npy", mmap_mode="r")
        normalized = ((np.clip(raw, q01, q99) - median) / scale).astype(np.float32)
        np.save(shard / "longitudinal_supervision_v2.npy", normalized)
    result = {
        "feature_names": ["ego_speed_smoothed", "ego_longitudinal_accel", "ego_longitudinal_jerk"],
        "median_filter_frames": 5, "train_q01": q01.tolist(), "train_q99": q99.tolist(),
        "train_median_after_winsorize": median.tolist(), "train_iqr_after_winsorize": scale.tolist(),
        "train_frame_count": int(len(train)), "finite": True,
    }
    (out_dir / "longitudinal_supervision_v2_schema.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_dynamic_summary(out_dir: Path) -> dict:
    path = out_dir / "build_summary.json"
    summary = json.loads(path.read_text())
    shards = [Path(value) for value in summary["shard_paths"]]
    total_rows = 0
    total_frames = 0
    split_counts: dict[str, int] = defaultdict(int)
    valid_frames = np.zeros(len(SLOT_NAMES), dtype=np.int64)
    occupied_windows = np.zeros(len(SLOT_NAMES), dtype=np.int64)
    switches = np.zeros(len(SLOT_NAMES), dtype=np.int64)
    derivative_violations = 0
    for shard in shards:
        valid = np.load(shard / "slot_valid_mask.npy", mmap_mode="r")
        switch = np.load(shard / "slot_identity_switch_mask.npy", mmap_mode="r")
        derivative = np.load(shard / "slot_derivative_valid_mask.npy", mmap_mode="r")
        neighbor = np.load(shard / "neighbor_seq.npy", mmap_mode="r")
        split = np.load(shard / "split.npy", allow_pickle=True).astype(str)
        total_rows += int(valid.shape[0])
        total_frames += int(valid.shape[0] * valid.shape[2])
        valid_frames += valid.sum(axis=(0, 2)).astype(np.int64)
        occupied_windows += valid.any(axis=2).sum(axis=0).astype(np.int64)
        switches += switch.sum(axis=(0, 2)).astype(np.int64)
        for name, count in zip(*np.unique(split, return_counts=True)):
            split_counts[str(name)] += int(count)
        derivative_violations += int(np.sum((~derivative) & ((neighbor[:, :, :, 12] != 0.0) | (neighbor[:, :, :, 14] != 0.0))))
    summary["dataset_type"] = "waymo_dynamic_interaction_v2"
    summary["slot_valid_ratio_legacy_not_applicable"] = summary.pop("slot_valid_ratio", {})
    summary["slot_valid_frame_ratio"] = {slot: float(valid_frames[index] / max(1, total_frames)) for index, slot in enumerate(SLOT_NAMES)}
    summary["slot_occupied_window_ratio"] = {slot: float(occupied_windows[index] / max(1, total_rows)) for index, slot in enumerate(SLOT_NAMES)}
    summary["slot_identity_switch_count"] = {slot: int(switches[index]) for index, slot in enumerate(SLOT_NAMES)}
    summary["split_counts"] = dict(sorted(split_counts.items()))
    summary["derivative_cross_identity_violation_count"] = derivative_violations
    summary["dynamic_summary_validation_pass"] = bool(
        total_rows == int(summary["n_windows_kept"])
        and derivative_violations == 0
        and all(0.0 <= value <= 1.0 for value in summary["slot_valid_frame_ratio"].values())
        and all(0.0 <= value <= 1.0 for value in summary["slot_occupied_window_ratio"].values())
    )
    if not summary["dynamic_summary_validation_pass"]:
        raise RuntimeError("dynamic builder summary validation failed")
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    legacy.process_one_scenario = process_one_scenario
    legacy.flush_shard = flush_shard
    args = legacy.parse_args()
    if args.assignment_mode != "lane_aware_only":
        raise ValueError(
            "Dynamic Interaction Builder v2 requires --assignment_mode lane_aware_only; "
            "geometric fallback cannot establish semantic-slot correctness"
        )
    ORIGINAL_MAIN()
    out_dir = Path(args.out_dir)
    supervision = build_longitudinal_supervision(out_dir)
    summary = rewrite_dynamic_summary(out_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "semantic_slots": SLOT_NAMES,
        "slot_assignment": "per_frame_dynamic",
        "assignment_mode_required": "lane_aware_only",
        "geometric_adjacent_lane_inference": False,
        "semantic_correctness_over_track_continuity": True,
        "identity_switch_derivative_policy": "reset_zero_and_mask_false",
        "ego_min_valid_ratio": args.min_valid_ratio,
        "neighbor_min_window_valid_ratio": None,
        "old_33d_supervision_preserved": True,
        "longitudinal_supervision_v2": supervision,
        "build_summary_sha256": sha256_file(out_dir / "build_summary.json"),
        "dynamic_summary_validation_pass": summary["dynamic_summary_validation_pass"],
        "source_selection": {
            "waymo_dir": str(Path(args.waymo_dir).resolve()),
            "file_start": int(args.file_start),
            "file_end_exclusive": int(args.file_end) if args.file_end is not None else None,
            "max_files": int(args.max_files) if args.max_files is not None else None,
            "n_files_processed": int(summary["n_files_processed"]),
        },
        "artifact_sha256": {
            "feature_schema.json": sha256_file(out_dir / "feature_schema.json"),
            "longitudinal_supervision_v2_schema.json": sha256_file(out_dir / "longitudinal_supervision_v2_schema.json"),
        },
    }
    (out_dir / "dynamic_builder_v2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
