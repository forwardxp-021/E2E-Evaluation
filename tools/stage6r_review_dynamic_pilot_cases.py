#!/usr/bin/env python3
"""Reconstruct 20 Stage 6R pilot cases before independent visual sign-off."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.build_waymo_dynamic_interaction_dataset_v2 import assignment_config, track_projection_cache
from tools.lane_aware_assignment import assign_neighbors_lane_aware
from tools.waymo_lane_utils import extract_lane_polylines


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def args_contract() -> Any:
    return argparse.Namespace(
        assignment_mode="lane_aware_only",
        lane_max_lateral_distance=3.0, lane_max_heading_diff_deg=45.0,
        adjacent_lane_min_offset=2.0, adjacent_lane_max_offset=5.5,
        adjacent_lane_max_heading_diff_deg=35.0, lane_search_radius=20.0,
        lane_topk_candidates=32, front_max_distance=120.0,
        side_front_max_distance=80.0, side_rear_max_distance=120.0,
        lane_lateral_tolerance=2.0, slot_heading_diff_deg=45.0,
        static_speed_threshold=0.5, disable_lane_spatial_index=False,
    )


def scenario_tracks(scenario: Any) -> dict[str, np.ndarray]:
    result = {}
    for track in scenario.tracks:
        if int(track.object_type) != 1:
            continue
        result[str(track.id)] = np.asarray([
            [state.center_x, state.center_y, state.velocity_x, state.velocity_y, state.heading, 1.0]
            if state.valid else [np.nan, np.nan, np.nan, np.nan, np.nan, 0.0]
            for state in track.states
        ], dtype=np.float32)
    return result


def review_case(case: dict[str, str], scenario: Any) -> dict[str, Any]:
    contract = args_contract()
    tracks = scenario_tracks(scenario)
    lanes = extract_lane_polylines(scenario)
    projections = track_projection_cache(tracks, lanes, contract)
    cfg = assignment_config(contract)
    target = case["target_agent_id"]
    slot = case["slot"]
    start = int(case["start"])
    expected_ids = ["-1"] * 80
    for segment in case["slot_identity_segments"].split(";"):
        begin_text, end_text, identity = segment.split(":", 2)
        expected_ids[int(begin_text) : int(end_text)] = [identity] * (int(end_text) - int(begin_text))
    target_track = tracks[target]
    laneaware = 0
    fallback = 0
    mismatches = 0
    topology_failures = 0
    valid_frames = 0
    for local_t in range(80):
        frame = start + local_t
        ego_raw = target_track[frame]
        if ego_raw[5] <= 0.5:
            continue
        states = {}
        candidates = {}
        for agent_id, track in tracks.items():
            if agent_id == target or frame >= len(track):
                continue
            state = track[frame]
            if state[5] <= 0.5 or not np.isfinite(state[:4]).all():
                continue
            heading = float(state[4]) if np.isfinite(state[4]) else float(np.arctan2(state[3], state[2]))
            states[agent_id] = {
                "x": float(state[0]), "y": float(state[1]), "velocity_x": float(state[2]),
                "velocity_y": float(state[3]), "heading": heading,
                "speed": float(np.hypot(state[2], state[3])), "valid": True,
            }
            if (agent_id, frame) in projections:
                candidates[agent_id] = projections[(agent_id, frame)]
        ego_heading = float(ego_raw[4]) if np.isfinite(ego_raw[4]) else float(np.arctan2(ego_raw[3], ego_raw[2]))
        ego_state = {
            "x": float(ego_raw[0]), "y": float(ego_raw[1]), "heading": ego_heading,
            "velocity_x": float(ego_raw[2]), "velocity_y": float(ego_raw[3]),
            "speed": float(np.hypot(ego_raw[2], ego_raw[3])),
        }
        result = assign_neighbors_lane_aware(
            ego_state, states, lane_infos=lanes,
            assignment_mode=contract.assignment_mode, config=cfg,
            ego_projection=projections.get((target, frame)), candidate_projections=candidates,
        )
        selected = result.slot_to_agent.get(slot)
        if str(selected or "-1") != expected_ids[local_t]:
            mismatches += 1
        if not selected:
            continue
        valid_frames += 1
        debug = next(row for row in result.per_slot_debug if row.get("slot_name") == slot)
        method = debug.get("assignment_method")
        if method == "lane_aware":
            laneaware += 1
            ds = float(debug.get("delta_s", math.nan))
            lane_ok = bool(debug.get("neighbor_lane_id")) and debug.get("neighbor_lane_id") == debug.get("slot_lane_id")
            direction_ok = ds > 0 if slot in {"front", "left_front", "right_front"} else ds < 0
            if not (lane_ok and direction_ok):
                topology_failures += 1
        elif method == "geometric_fallback":
            fallback += 1
            lon = float(debug.get("longitudinal_gap", math.nan))
            lat = float(debug.get("lateral_gap", math.nan))
            expected = {
                "front": lon > 0,
                "left_front": lon > 0 and lat > 0,
                "left_rear": lon <= 0 and lat > 0,
                "right_front": lon > 0 and lat <= 0,
                "right_rear": lon <= 0 and lat <= 0,
            }[slot]
            topology_failures += int(not expected)
        else:
            topology_failures += 1
    passed = valid_frames > 0 and mismatches == 0 and topology_failures == 0
    return {
        **case,
        "reconstructed_valid_frames": valid_frames,
        "laneaware_frames": laneaware,
        "geometric_fallback_frames": fallback,
        "reconstruction_mismatch_count": mismatches,
        "topology_failure_count": topology_failures,
        "topology_reconstruction": "PASS" if passed else "FAIL",
        "topology_reconstruction_basis": "raw_TFRecord_reconstruction_and_lane_topology",
    }


def write_case_overview(path: Path, cases: list[dict[str, str]], scenarios: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 5, figsize=(22, 17))
    for axis, case in zip(axes.flat, cases):
        scenario = scenarios[case["scenario_id"]]
        tracks = scenario_tracks(scenario)
        lanes = extract_lane_polylines(scenario)
        target = case["target_agent_id"]
        start = int(case["start"])
        stop = start + 80
        ego = tracks[target][start:stop]
        for lane in lanes.values():
            axis.plot(lane.centerline_xy[:, 0], lane.centerline_xy[:, 1], color="0.82", linewidth=0.45, zorder=1)
        valid_ego = ego[:, 5] > 0.5
        axis.plot(ego[valid_ego, 0], ego[valid_ego, 1], color="#1769aa", linewidth=2.2, label="ego", zorder=3)
        identities = []
        for segment in case["slot_identity_segments"].split(";"):
            _, _, identity = segment.split(":", 2)
            if identity != "-1" and identity not in identities:
                identities.append(identity)
        colors = plt.cm.autumn(np.linspace(0.15, 0.85, max(1, len(identities))))
        plotted = [ego[valid_ego, :2]]
        for color, identity in zip(colors, identities):
            other = tracks[identity][start:stop]
            valid = other[:, 5] > 0.5
            axis.plot(other[valid, 0], other[valid, 1], color=color, linewidth=1.7, marker=".", markersize=2.5, label=f"slot:{identity}", zorder=4)
            plotted.append(other[valid, :2])
        # Draw start/end direction arrows so a visual reviewer can distinguish
        # parallel adjacent motion from crossing/perpendicular trajectories.
        for points, color in [(ego[valid_ego, :2], "#1769aa")]:
            if len(points) >= 2:
                delta = points[min(5, len(points) - 1)] - points[0]
                axis.arrow(points[0, 0], points[0, 1], delta[0], delta[1], color=color,
                           width=0.08, head_width=0.8, length_includes_head=True, zorder=5)
        points = np.concatenate([value for value in plotted if len(value)], axis=0)
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        span = max(float(np.max(hi - lo)), 30.0) * 0.65
        axis.set_xlim(center[0] - span, center[0] + span)
        axis.set_ylim(center[1] - span, center[1] + span)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{case['slot']} | {case['scenario_id'][:6]} | {case['events'][:28]}", fontsize=8)
        axis.tick_params(labelsize=6)
        axis.grid(alpha=0.12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, max(1, len(labels))), fontsize=7)
    fig.suptitle("Stage6R pilot: 20 fixed-seed semantic-slot cases", fontsize=14)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = read_csv(args.cases_csv)
    needed = {row["scenario_id"] for row in cases}
    scenarios = {}
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    files = sorted(args.waymo_dir.glob("*.tfrecord*"))[args.file_start : args.file_end]
    for path in files:
        for record in tf.data.TFRecordDataset(str(path)):
            scenario = scenario_pb2.Scenario(); scenario.ParseFromString(bytes(record.numpy()))
            if scenario.scenario_id in needed:
                scenarios[scenario.scenario_id] = scenario
        if len(scenarios) == len(needed):
            break
    missing = sorted(needed - set(scenarios))
    if missing:
        raise ValueError(f"pilot review scenarios missing from frozen TFRecords: {missing}")
    reviewed = [review_case(case, scenarios[case["scenario_id"]]) for case in cases]
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_csv(output / "stage6r_manual_semantic_cases_reviewed.csv", reviewed)
    counts = {slot: sum(row["slot"] == slot and row["topology_reconstruction"] == "PASS" for row in reviewed) for slot in ["front", "left_front", "left_rear", "right_front", "right_rear"]}
    passed = len(reviewed) == 20 and all(value == 4 for value in counts.values())
    result = {
        "schema_version": "stage6r_pilot_topology_reconstruction_v2",
        "status": "TOPOLOGY_RECONSTRUCTION_PASS_PENDING_VISUAL_REVIEW" if passed else "TOPOLOGY_RECONSTRUCTION_FAIL",
        "reviewed_case_count": len(reviewed),
        "pass_count_by_slot": counts,
        "laneaware_frame_count": sum(int(row["laneaware_frames"]) for row in reviewed),
        "geometric_fallback_frame_count": sum(int(row["geometric_fallback_frames"]) for row in reviewed),
        "topology_failure_count": sum(int(row["topology_failure_count"]) for row in reviewed),
        "reconstruction_mismatch_count": sum(int(row["reconstruction_mismatch_count"]) for row in reviewed),
        "review_basis": "automatic raw-TFRecord reconstruction only; this is not a visual semantic sign-off",
    }
    (output / "stage6r_manual_review_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_case_overview(output / "stage6r_manual_semantic_cases_overview.png", cases, scenarios)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_csv", type=Path, required=True)
    parser.add_argument("--waymo_dir", type=Path, required=True)
    parser.add_argument("--file_start", type=int, default=0)
    parser.add_argument("--file_end", type=int, default=3)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
