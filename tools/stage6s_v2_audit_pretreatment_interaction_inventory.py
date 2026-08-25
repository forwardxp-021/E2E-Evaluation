#!/usr/bin/env python3
"""Audit nuPlan following tags using only pre-treatment SQLite trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FIELDS = [
    "db_file", "log_name", "scenario_token", "scene_token", "db_scene_token", "scenario_type",
    "scenario_tag_token", "agent_track_token", "raw_frame_count", "valid_front_frame_count",
    "front_exposure_ratio", "front_exposure_seconds", "initial_front_gap_m", "median_front_gap_m",
    "ego_median_speed_mps", "closing_median_mps", "closing_p75_mps", "closing_pressure_ratio",
    "median_abs_lateral_m", "median_abs_heading_diff_deg", "eligible", "ineligible_reasons",
    "pre_treatment_only", "planner_outcome_read", "embedding_or_bdd_read",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def angle_delta(value: float, reference: float) -> float:
    return (value - reference + math.pi) % (2.0 * math.pi) - math.pi


def token(value: str) -> bytes:
    return bytes.fromhex(value)


def finite_percentile(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else math.nan


def load_db_cache(
    conn: sqlite3.Connection,
    allowed_types: set[str],
) -> tuple[
    dict[tuple[bytes, str], sqlite3.Row],
    dict[bytes, tuple[bytes, int]],
    dict[bytes, list[sqlite3.Row]],
    dict[tuple[bytes, bytes], sqlite3.Row],
]:
    placeholders = ",".join("?" for _ in allowed_types)
    tag_rows = conn.execute(
        f"SELECT token, lidar_pc_token, type, agent_track_token FROM scenario_tag "
        f"WHERE type IN ({placeholders}) AND agent_track_token IS NOT NULL ORDER BY token",
        tuple(sorted(allowed_types)),
    ).fetchall()
    tags: dict[tuple[bytes, str], sqlite3.Row] = {}
    track_tokens: set[bytes] = set()
    for item in tag_rows:
        key = (bytes(item["lidar_pc_token"]), str(item["type"]))
        tags.setdefault(key, item)
        track_tokens.add(bytes(item["agent_track_token"]))
    frame_rows = conn.execute(
        """
        SELECT lp.token AS lidar_pc_token, lp.timestamp, lp.scene_token,
               ep.x AS ex, ep.y AS ey, ep.qw, ep.qx, ep.qy, ep.qz,
               ep.vx AS evx, ep.vy AS evy
        FROM lidar_pc AS lp JOIN ego_pose AS ep ON ep.token=lp.ego_pose_token
        ORDER BY lp.scene_token, lp.timestamp
        """
    ).fetchall()
    frames_by_scene: dict[bytes, list[sqlite3.Row]] = defaultdict(list)
    frame_location: dict[bytes, tuple[bytes, int]] = {}
    for frame in frame_rows:
        scene = bytes(frame["scene_token"])
        location = len(frames_by_scene[scene])
        frames_by_scene[scene].append(frame)
        frame_location[bytes(frame["lidar_pc_token"])] = (scene, location)
    boxes: dict[tuple[bytes, bytes], sqlite3.Row] = {}
    ordered_tracks = sorted(track_tokens)
    for offset in range(0, len(ordered_tracks), 500):
        chunk = ordered_tracks[offset : offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT lidar_pc_token, track_token, x, y, vx, vy, yaw, length "
            f"FROM lidar_box WHERE track_token IN ({placeholders})",
            tuple(chunk),
        ).fetchall()
        for item in rows:
            boxes[(bytes(item["lidar_pc_token"]), bytes(item["track_token"]))] = item
    return tags, frame_location, frames_by_scene, boxes


def audit_candidate(
    row: dict[str, str],
    cfg: dict[str, Any],
    cache: tuple[
        dict[tuple[bytes, str], sqlite3.Row],
        dict[bytes, tuple[bytes, int]],
        dict[bytes, list[sqlite3.Row]],
        dict[tuple[bytes, bytes], sqlite3.Row],
    ],
) -> dict[str, Any]:
    scenario = token(row["scenario_token"])
    tags, frame_location, frames_by_scene, boxes = cache
    tagged = tags.get((scenario, row["scenario_type"]))
    base = {key: row.get(key, "") for key in FIELDS if key in row}
    base.update({
        "pre_treatment_only": True, "planner_outcome_read": False, "embedding_or_bdd_read": False,
        "agent_track_token": bytes(tagged["agent_track_token"]).hex() if tagged else "",
    })
    if tagged is None:
        return {**base, "eligible": False, "ineligible_reasons": "missing_agent_track_token"}
    location = frame_location.get(scenario)
    if location is None:
        return {**base, "eligible": False, "ineligible_reasons": "missing_scenario_lidar_pc"}
    scene, start_index = location
    scene_frames = frames_by_scene[scene]
    start_timestamp = int(scene_frames[start_index]["timestamp"])
    end_timestamp = int(start_timestamp + float(cfg["horizon_seconds"]) * 1_000_000)
    frames = []
    for frame in scene_frames[start_index:]:
        if int(frame["timestamp"]) > end_timestamp:
            break
        box = boxes.get((bytes(frame["lidar_pc_token"]), bytes(tagged["agent_track_token"])))
        frames.append((frame, box))
    valid = []
    gaps: list[float] = []
    speeds: list[float] = []
    closings: list[float] = []
    laterals: list[float] = []
    headings: list[float] = []
    initial_gaps: list[float] = []
    first_timestamp = start_timestamp
    for frame, box in frames:
        heading = yaw(frame["qw"], frame["qx"], frame["qy"], frame["qz"])
        c, s = math.cos(heading), math.sin(heading)
        # nuPlan ego_pose vx/vy are provided in the ego/body convention in these DBs;
        # their Euclidean norm is the invariant physical speed. Projecting them by
        # the global quaternion a second time incorrectly collapses speed on N/S roads.
        ego_forward = float(math.hypot(frame["evx"], frame["evy"]))
        speeds.append(ego_forward)
        frame_valid = box is not None
        if frame_valid:
            dx, dy = float(box["x"] - frame["ex"]), float(box["y"] - frame["ey"])
            longitudinal = dx * c + dy * s
            lateral = -dx * s + dy * c
            heading_diff = abs(math.degrees(angle_delta(float(box["yaw"]), heading)))
            frame_valid = (
                longitudinal > 0.0
                and abs(lateral) <= float(cfg["front_lateral_abs_max_m"])
                and heading_diff <= float(cfg["front_heading_abs_max_deg"])
            )
            if frame_valid:
                box_length = float(box["length"] or 4.5)
                gap = longitudinal - 0.5 * (float(cfg["ego_vehicle_length_m"]) + box_length)
                object_forward = float(box["vx"] * c + box["vy"] * s)
                closing = ego_forward - object_forward
                gaps.append(gap); closings.append(closing); laterals.append(abs(lateral)); headings.append(heading_diff)
                if int(frame["timestamp"]) - first_timestamp <= float(cfg["initial_window_seconds"]) * 1_000_000:
                    initial_gaps.append(gap)
        valid.append(bool(frame_valid))
    raw_count = len(frames)
    valid_count = sum(valid)
    duration = 0.0
    if raw_count >= 2:
        duration = max(0.0, (int(frames[-1][0]["timestamp"]) - int(frames[0][0]["timestamp"])) / 1_000_000)
    frame_hz = (raw_count - 1) / duration if duration > 0 else 0.0
    exposure_seconds = valid_count / frame_hz if frame_hz > 0 else 0.0
    exposure_ratio = valid_count / max(1, raw_count)
    initial_gap = finite_percentile(initial_gaps, 50)
    closing_pressure = float(np.mean(np.asarray(closings) > float(cfg["closing_positive_threshold_mps"]))) if closings else 0.0
    reasons = []
    checks = [
        (raw_count >= int(cfg["minimum_raw_frames"]), "insufficient_raw_frames"),
        (exposure_ratio >= float(cfg["front_exposure_ratio_min"]), "low_front_exposure_ratio"),
        (exposure_seconds >= float(cfg["front_exposure_seconds_min"]), "short_front_exposure"),
        (math.isfinite(initial_gap) and float(cfg["initial_front_gap_min_m"]) <= initial_gap <= float(cfg["initial_front_gap_max_m"]), "invalid_initial_gap"),
        (finite_percentile(speeds, 50) >= float(cfg["ego_median_speed_min_mps"]), "low_ego_speed"),
        (closing_pressure >= float(cfg["closing_pressure_ratio_min"]), "low_closing_pressure_ratio"),
        (finite_percentile(closings, 75) >= float(cfg["closing_p75_min_mps"]), "low_closing_p75"),
    ]
    reasons.extend(label for passed, label in checks if not passed)
    return {
        **base,
        "raw_frame_count": raw_count,
        "valid_front_frame_count": valid_count,
        "front_exposure_ratio": exposure_ratio,
        "front_exposure_seconds": exposure_seconds,
        "initial_front_gap_m": initial_gap,
        "median_front_gap_m": finite_percentile(gaps, 50),
        "ego_median_speed_mps": finite_percentile(speeds, 50),
        "closing_median_mps": finite_percentile(closings, 50),
        "closing_p75_mps": finite_percentile(closings, 75),
        "closing_pressure_ratio": closing_pressure,
        "median_abs_lateral_m": finite_percentile(laterals, 50),
        "median_abs_heading_diff_deg": finite_percentile(headings, 50),
        "eligible": not reasons,
        "ineligible_reasons": "|".join(reasons),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    cfg = config["inventory"]
    inventory = read_csv(args.inventory_csv)
    inputs = {row["db_file"]: Path(row["db_path"]) for row in read_csv(args.inventory_inputs_csv)}
    allowed = set(cfg["allowed_scenario_types"])
    candidates = [row for row in inventory if row["scenario_type"] in allowed]
    by_db: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_db[row["db_file"]].append(row)
    results = []
    for db_index, db_file in enumerate(sorted(by_db), start=1):
        path = inputs.get(db_file)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"inventory DB is unavailable: {db_file} -> {path}")
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cache = load_db_cache(conn, allowed)
            for row in by_db[db_file]:
                results.append(audit_candidate(row, cfg, cache))
        if db_index % max(1, args.progress_every_dbs) == 0:
            print(f"[Stage6S-v2] db={db_index}/{len(by_db)} candidates={len(results)} eligible={sum(bool(r['eligible']) for r in results)}", flush=True)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    csv_path = output / "stage6s_v2_pretreatment_interaction_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(results)
    eligible = [row for row in results if bool(row["eligible"])]
    summary = {
        "schema_version": "stage6s_v2_pretreatment_interaction_inventory_v1",
        "status": "PRETREATMENT_INTERACTION_INVENTORY_READY",
        "candidate_count": len(results),
        "eligible_count": len(eligible),
        "eligible_unique_log_count": len({row["log_name"] for row in eligible}),
        "eligible_scenario_type_counts": dict(Counter(row["scenario_type"] for row in eligible)),
        "ineligible_reason_counts": dict(Counter(reason for row in results for reason in str(row["ineligible_reasons"]).split("|") if reason)),
        "config_sha256": sha256_file(args.config),
        "inventory_csv_sha256": sha256_file(args.inventory_csv),
        "inventory_inputs_sha256": sha256_file(args.inventory_inputs_csv),
        "output_csv_sha256": sha256_file(csv_path),
        "pre_treatment_only": True,
        "planner_outcome_read": False,
        "embedding_or_bdd_read": False,
        "checkpoint_training_launched": False,
    }
    (output / "stage6s_v2_pretreatment_inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--inventory_inputs_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--progress_every_dbs", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
