#!/usr/bin/env python3
"""Build an outcome-blind, map-based Stage7L lane-change opportunity inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.stage7l_pure_lateral_execution_planner import (
    canonical_json_sha256,
    initial_state_fingerprint,
)


BACKGROUND_MODE = "closed_loop_nonreactive_agents"
BACKGROUND_AGENT_MODEL = "nuplan.planning.simulation.observation.tracks_observation.TracksObservation"
BACKGROUND_CONFIG = {
    "simulation": BACKGROUND_MODE,
    "observation_target": BACKGROUND_AGENT_MODEL,
    "reactive": False,
    "source": "nuPlan official simulation config",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def token_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value or "")


def historical_exclusions(stage7_roster: Path, stage7p_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if stage7_roster.is_file():
        for row in read_csv(stage7_roster):
            if row.get("task") == "lane_change" and row.get("scenario_token"):
                rows.append({
                    "scenario_token": row["scenario_token"], "log_name": row.get("log_name", ""),
                    "exclusion_reason": "STAGE7_FROZEN_LANE_CHANGE_60",
                    "source_path": str(stage7_roster.resolve()),
                })
    for path in sorted(stage7p_root.rglob("scenario_alignment.csv")) if stage7p_root.is_dir() else []:
        try:
            source_rows = read_csv(path)
        except (OSError, csv.Error):
            continue
        for row in source_rows:
            for key in ("actual_nuplan_token", "actual_nuplan_scenario_token", "scenario_token", "target_scene_token"):
                token = str(row.get(key, "")).strip().replace('\\"', "").replace('"', "")
                if len(token) == 16:
                    rows.append({
                        "scenario_token": token,
                        "log_name": str(row.get("target_log_name") or row.get("log_name") or ""),
                        "exclusion_reason": "STAGE7P_TUNING_OR_SMOKE",
                        "source_path": str(path.resolve()),
                    })
    unique: Dict[str, Dict[str, str]] = {}
    for row in rows:
        token = row["scenario_token"]
        if token not in unique:
            unique[token] = row
        elif row["exclusion_reason"] not in unique[token]["exclusion_reason"]:
            unique[token]["exclusion_reason"] += "|" + row["exclusion_reason"]
            unique[token]["source_path"] += "|" + row["source_path"]
    return sorted(unique.values(), key=lambda row: row["scenario_token"])


def candidate_rows_from_db(db_path: Path, per_db: int, minimum_speed_mps: float) -> List[Dict[str, Any]]:
    query = """
        WITH ordered_scenes AS (
          SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS scene_row_num FROM scene
        ), scene_count AS (SELECT COUNT(*) AS n FROM scene),
        candidates AS (
          SELECT lp.token, lp.timestamp, lp.scene_token, l.map_version, l.logfile,
                 ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy,
                 os.scene_row_num, sc.n AS scene_count,
                 ROW_NUMBER() OVER (ORDER BY lp.timestamp ASC) AS candidate_order
          FROM lidar_pc lp
          JOIN ego_pose ep ON ep.token = lp.ego_pose_token
          JOIN lidar ld ON ld.token = lp.lidar_token
          JOIN log l ON l.token = ld.log_token
          JOIN ordered_scenes os ON os.token = lp.scene_token
          CROSS JOIN scene_count sc
          WHERE os.scene_row_num >= 3 AND os.scene_row_num < sc.n - 1
            AND (ep.vx * ep.vx + ep.vy * ep.vy) >= ?
        )
        SELECT * FROM candidates
        WHERE (candidate_order % MAX(1, (SELECT COUNT(*) FROM candidates) / ?)) = 0
        ORDER BY candidate_order LIMIT ?
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        raw = connection.execute(query, (minimum_speed_mps**2, per_db, per_db)).fetchall()
    result = []
    for row in raw:
        item = dict(row)
        item["scenario_token"] = token_text(item.pop("token"))
        item["db_scene_token"] = token_text(item.pop("scene_token"))
        item["db_file"] = db_path.name
        item["log_name"] = str(item.pop("logfile"))
        item["map_name"] = str(item.pop("map_version"))
        result.append(item)
    return result


def candidate_row_for_token(db_path: Path, token: str) -> Dict[str, Any]:
    query = """
        WITH ordered_scenes AS (
          SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS scene_row_num FROM scene
        ), scene_count AS (SELECT COUNT(*) AS n FROM scene)
        SELECT lp.token, lp.timestamp, lp.scene_token, l.map_version, l.logfile,
               ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy,
               os.scene_row_num, sc.n AS scene_count
        FROM lidar_pc lp
        JOIN ego_pose ep ON ep.token = lp.ego_pose_token
        JOIN lidar ld ON ld.token = lp.lidar_token
        JOIN log l ON l.token = ld.log_token
        JOIN ordered_scenes os ON os.token = lp.scene_token
        CROSS JOIN scene_count sc
        WHERE lp.token = ? AND os.scene_row_num >= 3 AND os.scene_row_num < sc.n - 1
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        raw = connection.execute(query, (bytes.fromhex(token),)).fetchone()
    if raw is None:
        raise ValueError(f"token is not officially runnable in {db_path.name}: {token}")
    item = dict(raw)
    item["scenario_token"] = token_text(item.pop("token"))
    item["db_scene_token"] = token_text(item.pop("scene_token"))
    item["db_file"] = db_path.name
    item["log_name"] = str(item.pop("logfile"))
    item["map_name"] = str(item.pop("map_version"))
    return item


def heading_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def select_source_lane(map_api: Any, x: float, y: float, heading: float) -> Tuple[Optional[Any], str]:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    point = Point2D(float(x), float(y))
    try:
        lanes = list(map_api.get_all_map_objects(point, SemanticMapLayer.LANE))
    except Exception as exc:
        return None, f"MAP_LANE_QUERY_ERROR:{type(exc).__name__}"
    if not lanes:
        return None, "NO_SOURCE_LANE_AT_INITIAL_EGO"
    ranked = []
    for lane in lanes:
        pose = lane.baseline_path.get_nearest_pose_from_position(point)
        ranked.append((heading_error(heading, pose.heading), lane))
    ranked.sort(key=lambda item: item[0])
    if ranked[0][0] > math.radians(35.0):
        return None, "SOURCE_LANE_HEADING_MISMATCH"
    return ranked[0][1], "PASS"


def route_ids(db_path: Path, token: str) -> List[str]:
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_roadblock_ids_for_lidarpc_token_from_db
    return [str(x) for x in (get_roadblock_ids_for_lidarpc_token_from_db(str(db_path), token) or [])]


def initial_objects(db_path: Path, token: str) -> List[Tuple[float, float]]:
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_tracked_objects_for_lidarpc_token_from_db
    result: List[Tuple[float, float]] = []
    for obj in get_tracked_objects_for_lidarpc_token_from_db(str(db_path), token):
        center = obj.center
        result.append((float(center.x), float(center.y)))
    return result


def official_simulation_initial_token(
    db_path: Path, anchor_token: str, anchor_timestamp_us: int
) -> Tuple[str, int, Tuple[str, ...]]:
    """Resolve nuPlan's true first lidar token for tagged and untagged scenarios.

    The frozen official ``nuplan_scenario_mapping`` applies a -3 s extraction
    offset to every tagged scenario.  Untagged/default scenarios are expanded
    from their anchor token without that offset.  We resolve the actual sampled
    lidar token first and subsequently construct the ego state through the
    official query helper, avoiding independent quaternion reconstruction.
    """
    query = """
        SELECT lp.token, lp.timestamp
        FROM lidar_pc lp
        WHERE lp.timestamp >= ?
        ORDER BY lp.timestamp ASC LIMIT 1
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        tag_rows = connection.execute(
            "SELECT type FROM scenario_tag WHERE lidar_pc_token = ? ORDER BY type",
            (bytes.fromhex(anchor_token),),
        ).fetchall()
        scenario_types = tuple(str(row["type"]) for row in tag_rows)
        offset_s = -3.0 if scenario_types else 0.0
        target = int(anchor_timestamp_us + offset_s * 1e6)
        row = connection.execute(query, (target,)).fetchone()
    if row is None:
        raise ValueError("official simulation initial lidar row is unavailable")
    return token_text(row["token"]), int(row["timestamp"]), scenario_types


def line_values(lane: Any, x: float, y: float) -> Tuple[List[List[float]], float, float]:
    from nuplan.common.actor_state.state_representation import Point2D
    line = lane.baseline_path.linestring
    points = [[float(px), float(py)] for px, py in line.coords]
    start_arc = float(lane.baseline_path.get_nearest_arc_length_from_position(Point2D(x, y)))
    remaining = float(line.length - start_arc)
    return points, start_arc, remaining


def minimum_target_gap(objects_xy: Sequence[Tuple[float, float]], target_lane: Any, ego_x: float, ego_y: float) -> float:
    from shapely.geometry import Point
    line = target_lane.baseline_path.linestring
    ego_s = float(line.project(Point(ego_x, ego_y)))
    gaps = []
    for x, y in objects_xy:
        point = Point(x, y)
        if float(point.distance(line)) <= 2.5:
            gaps.append(abs(float(line.project(point)) - ego_s))
    return min(gaps) if gaps else 999.0


def evaluate_candidate(
    row: Mapping[str, Any], db_root: Path, map_root: Path, map_version: str,
    minimum_remaining_m: float, minimum_target_gap_m: float, minimum_speed_mps: float,
) -> Dict[str, Any]:
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db
    db_path = db_root / str(row["db_file"])
    official_token, official_timestamp_us, scenario_types = official_simulation_initial_token(
        db_path, str(row["scenario_token"]), int(row["timestamp"])
    )
    official_initial = get_ego_state_for_lidarpc_token_from_db(str(db_path), official_token)
    if official_initial is None:
        raise ValueError("official initial ego state is unavailable")
    speed = float(official_initial.dynamic_car_state.speed)
    yaw = float(official_initial.rear_axle.heading)
    result: Dict[str, Any] = {
        "db_file": row["db_file"], "log_name": row["log_name"], "scenario_token": row["scenario_token"],
        "db_scene_token": row["db_scene_token"], "map_name": row["map_name"],
        "initial_x": float(official_initial.rear_axle.x), "initial_y": float(official_initial.rear_axle.y), "initial_heading": yaw,
        "initial_speed_mps": speed, "initial_time_us": int(official_initial.time_us),
        "official_scenario_anchor_time_us": int(row["timestamp"]),
        "official_simulation_initial_lidar_token": official_token,
        "official_simulation_initial_timestamp_us": official_timestamp_us,
        "official_scenario_types_json": json.dumps(scenario_types, separators=(",", ":")),
        "official_query_runnable": True, "eligible": False, "reason_code": "UNSET",
    }
    if speed < minimum_speed_mps:
        result["reason_code"] = "OFFICIAL_INITIAL_SPEED_BELOW_MINIMUM"
        return result
    api = get_maps_api(str(map_root), map_version, str(row["map_name"]))
    source, reason = select_source_lane(api, result["initial_x"], result["initial_y"], yaw)
    if source is None:
        result["reason_code"] = reason
        return result
    route = route_ids(db_path, str(row["scenario_token"]))
    if not route:
        result["reason_code"] = "MISSING_ROUTE"
        return result
    source_roadblock = str(source.get_roadblock_id())
    if source_roadblock not in route:
        result["reason_code"] = "SOURCE_ROADBLOCK_NOT_ON_ROUTE"
        return result
    left, right = source.adjacent_edges
    objects = initial_objects(db_path, official_token)
    result["minimum_source_lane_object_gap_m"] = round(
        float(minimum_target_gap(objects, source, result["initial_x"], result["initial_y"])), 3
    )
    options = []
    for direction, target in (("left", left), ("right", right)):
        if target is None:
            continue
        target_roadblock = str(target.get_roadblock_id())
        if target_roadblock != source_roadblock:
            continue
        try:
            source_xy, source_arc, source_remaining = line_values(source, result["initial_x"], result["initial_y"])
            target_xy, target_arc, target_remaining = line_values(target, result["initial_x"], result["initial_y"])
        except Exception:
            continue
        remaining = min(source_remaining, target_remaining)
        gap = minimum_target_gap(objects, target, result["initial_x"], result["initial_y"])
        if remaining >= minimum_remaining_m and gap >= minimum_target_gap_m:
            options.append((remaining, gap, direction, target, source_xy, target_xy, source_arc, target_arc))
    if not options:
        result["reason_code"] = "NO_NATIVE_ADJACENT_WITH_ROUTE_LENGTH_AND_CLEARANCE"
        return result
    options.sort(key=lambda item: (item[0], item[1], item[2] == "left"), reverse=True)
    remaining, gap, direction, target, source_xy, target_xy, source_arc, target_arc = options[0]
    route_tuple = tuple(route)
    result.update({
        "eligible": True, "reason_code": "ELIGIBLE_NATIVE_ADJACENT_PRETREATMENT",
        "source_lane_id": str(source.id), "target_lane_id": str(target.id), "direction": direction,
        "source_roadblock_id": source_roadblock, "target_roadblock_id": str(target.get_roadblock_id()),
        "route_roadblock_ids_json": json.dumps(route_tuple, separators=(",", ":")),
        "route_fingerprint": canonical_json_sha256(route_tuple),
        "source_start_arc_m": round(float(source_arc), 6), "target_start_arc_m": round(float(target_arc), 6),
        "paired_reference_remaining_m": round(float(remaining), 3),
        "minimum_target_lane_object_gap_m": round(float(gap), 3),
        "nominal_lane_width_m": round(float(np.median(np.linalg.norm(np.asarray(target_xy)[:min(len(target_xy), len(source_xy))] - np.asarray(source_xy)[:min(len(target_xy), len(source_xy))], axis=1))), 3),
        "source_reference_xy_json": json.dumps(source_xy, separators=(",", ":")),
        "target_reference_xy_json": json.dumps(target_xy, separators=(",", ":")),
        "initial_state_fingerprint": initial_state_fingerprint(result["initial_x"], result["initial_y"], yaw, speed, int(official_initial.time_us)),
    })
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusions = historical_exclusions(args.stage7_lane_change_roster, args.stage7p_root)
    if args.additional_exclusion_ledger and args.additional_exclusion_ledger.is_file():
        merged = {row["scenario_token"]: row for row in exclusions}
        merged.update({row["scenario_token"]: row for row in read_csv(args.additional_exclusion_ledger)})
        exclusions = sorted(merged.values(), key=lambda row: row["scenario_token"])
    existing_ledger = args.output_dir / "stage7l_prior_exclusion_ledger.csv"
    if existing_ledger.is_file():
        preserved = [
            row for row in read_csv(existing_ledger)
            if row.get("exclusion_reason", "").startswith("STAGE7L_A2_")
        ]
        merged = {row["scenario_token"]: row for row in exclusions}
        merged.update({row["scenario_token"]: row for row in preserved})
        exclusions = sorted(merged.values(), key=lambda row: row["scenario_token"])
    excluded_tokens = {row["scenario_token"] for row in exclusions}
    inputs = read_csv(args.inventory_inputs)
    results: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for db_index, input_row in enumerate(inputs, start=1):
        if args.max_dbs and db_index > args.max_dbs:
            break
        db_path = args.nuplan_db_root / input_row["db_file"]
        if not db_path.is_file():
            counts["missing_db"] += 1
            continue
        try:
            candidates = candidate_rows_from_db(db_path, args.candidates_per_db, args.minimum_speed_mps)
        except sqlite3.Error:
            counts["db_query_error"] += 1
            continue
        for candidate in candidates:
            if candidate["scenario_token"] in excluded_tokens:
                counts["historically_excluded"] += 1
                continue
            try:
                evaluated = evaluate_candidate(
                    candidate, args.nuplan_db_root, args.nuplan_map_root, args.map_version,
                    args.minimum_paired_reference_remaining_m, args.minimum_target_gap_m, args.minimum_speed_mps,
                )
            except Exception as exc:
                evaluated = {
                    "db_file": candidate["db_file"], "log_name": candidate["log_name"],
                    "scenario_token": candidate["scenario_token"], "eligible": False,
                    "reason_code": f"EVALUATION_ERROR:{type(exc).__name__}", "error": str(exc),
                }
            results.append(evaluated)
            counts[str(evaluated["reason_code"])] += 1
        eligible_now = sum(bool(row.get("eligible")) for row in results)
        if eligible_now >= args.stop_after_eligible:
            break
        if db_index % 50 == 0:
            print(f"[Stage7L inventory] db={db_index}/{len(inputs)} evaluated={len(results)} eligible={eligible_now}", flush=True)
    if not results:
        raise ValueError("Stage7L inventory evaluated zero candidates")
    fields = sorted({key for row in results for key in row})
    inventory_path = args.output_dir / "stage7l_lane_change_opportunity_inventory.csv"
    write_csv(inventory_path, results, fields)
    exclusion_path = args.output_dir / "stage7l_prior_exclusion_ledger.csv"
    write_csv(exclusion_path, exclusions, ["scenario_token", "log_name", "exclusion_reason", "source_path"])
    eligible = [row for row in results if row.get("eligible")]
    summary = {
        "schema_version": "stage7l_a2_lane_change_opportunity_inventory_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_is_pre_treatment": True,
        "expert_execution_outcome_used": False,
        "rollout_outcome_used": False,
        "embedding_or_bdd_read": False,
        "candidate_source": str(args.inventory_inputs.resolve()),
        "candidate_source_sha256": sha256_file(args.inventory_inputs),
        "db_root": str(args.nuplan_db_root.resolve()),
        "map_root": str(args.nuplan_map_root.resolve()),
        "map_version": args.map_version,
        "evaluated_count": len(results),
        "eligible_unique_token_count": len({row["scenario_token"] for row in eligible}),
        "eligible_unique_log_count": len({row["log_name"] for row in eligible}),
        "eligible_left_count": sum(row.get("direction") == "left" for row in eligible),
        "eligible_right_count": sum(row.get("direction") == "right" for row in eligible),
        "eligible_map_counts": dict(Counter(str(row.get("map_name", "")) for row in eligible)),
        "eligible_roadblock_count": len({row.get("source_roadblock_id") for row in eligible}),
        "fresh_minimum_required": 104,
        "fresh_target_preferred": 150,
        "fresh_supply_gate_pass": len({row["scenario_token"] for row in eligible}) >= 104,
        "strict_log_disjoint_partition_feasible": len({row["log_name"] for row in eligible}) >= 104,
        "history_excluded_unique_token_count": len(excluded_tokens),
        "reason_counts": dict(counts),
        "background_mode": BACKGROUND_MODE,
        "background_agent_model": BACKGROUND_AGENT_MODEL,
        "background_config_sha256": canonical_json_sha256(BACKGROUND_CONFIG),
        "inventory_csv": str(inventory_path.resolve()),
        "inventory_csv_sha256": sha256_file(inventory_path),
        "prior_exclusion_ledger": str(exclusion_path.resolve()),
        "prior_exclusion_ledger_sha256": sha256_file(exclusion_path),
    }
    summary_path = args.output_dir / "stage7l_lane_change_opportunity_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory_inputs", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--map_version", default="nuplan-maps-v1.0")
    parser.add_argument("--stage7_lane_change_roster", type=Path, required=True)
    parser.add_argument("--stage7p_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--additional_exclusion_ledger", type=Path)
    parser.add_argument("--candidates_per_db", type=int, default=4)
    parser.add_argument("--max_dbs", type=int, default=0)
    parser.add_argument("--stop_after_eligible", type=int, default=160)
    parser.add_argument("--minimum_speed_mps", type=float, default=3.0)
    parser.add_argument("--minimum_paired_reference_remaining_m", type=float, default=90.0)
    parser.add_argument("--minimum_target_gap_m", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
