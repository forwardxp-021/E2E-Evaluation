#!/usr/bin/env python3
"""Freeze the Stage7L-C pre-treatment protocol and 80-scenario roster.

This tool is deliberately an offline freeze operation.  It validates frozen
source assets, selects only from Stage7L-B2 Pool B using pre-treatment fields,
rechecks official nuPlan scene runnability, and writes immutable Stage7L-C
artifacts.  It does not import the planner, read a rollout, export an
embedding, or calculate BDD/MMD.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def stable_rank(seed: int, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{int(seed)}:{row['direction']}:{row['log_name']}:{row['scenario_token']}".encode("utf-8")
    ).hexdigest()


def git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def validate_file_sha(root: Path, name: str, spec: Mapping[str, Any]) -> Path:
    path = root / str(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"frozen source asset missing: {name}: {path}")
    actual = sha256_file(path)
    if actual != str(spec["sha256"]):
        raise ValueError(f"frozen source asset SHA mismatch: {name}: {actual} != {spec['sha256']}")
    return path


def validate_authority_assets(
    root: Path, protocol: Mapping[str, Any], require_authorized_head: bool = True
) -> Dict[str, Path]:
    if require_authorized_head and git_head(root) != str(protocol["authorized_base_commit"]):
        raise ValueError(
            "Stage7L-C must freeze from the authorized B2 HEAD; source branch HEAD has changed"
        )
    resolved: Dict[str, Path] = {}
    for name, spec in protocol["source_assets"].items():
        resolved[name] = validate_file_sha(root, name, spec)
        if "required_status" in spec:
            status = read_json(resolved[name]).get("status")
            if status != spec["required_status"]:
                raise ValueError(f"authority status mismatch for {name}: {status!r}")
    b2 = read_json(resolved["b2_dynamic_clearance_manifest"])
    pool_b_sha = b2["sha256"]["pool_b"]
    if pool_b_sha != protocol["source_assets"]["pool_b"]["sha256"]:
        raise ValueError("B2 manifest does not attest the configured Pool B")
    if b2["pittsburgh_expansion"]["pool_b_strict_stage7l_b_log_disjoint"]["tokens"] != 152:
        raise ValueError("B2 Pool B source inventory count is not the frozen 152")
    return resolved


def curvature_p90(raw_reference: str) -> float:
    xy = np.asarray(json.loads(raw_reference), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] != 2:
        return 0.0
    delta = np.diff(xy, axis=0)
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1e-6
    if int(np.sum(valid)) < 3:
        return 0.0
    heading = np.unwrap(np.arctan2(delta[valid, 1], delta[valid, 0]))
    curve_length = length[valid]
    curvature = np.abs(np.diff(heading)) / np.maximum(curve_length[1:], 1e-6)
    return float(np.quantile(curvature, 0.9)) if len(curvature) else 0.0


def enrich_geometry(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = dict(row)
        item.update(
            {
                "initial_speed_mps": float(row["initial_speed_mps"]),
                "source_curvature_p90_inv_m": curvature_p90(row["source_reference_xy_json"]),
                "target_curvature_p90_inv_m": curvature_p90(row["target_reference_xy_json"]),
                "nominal_lane_width_m": float(row["nominal_lane_width_m"]),
                "traffic_density_replay_track_count": float(row["dynamic_replay_track_count"]),
                "paired_reference_remaining_m": float(row["paired_reference_remaining_m"]),
                "selection_seed_rank": stable_rank(620271, row),
            }
        )
        enriched.append(item)
    return enriched


def quantile_label(value: float, values: Sequence[float]) -> str:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    if value <= float(low):
        return "LOW"
    if value <= float(high):
        return "MID"
    return "HIGH"


def add_selection_strata(rows: Sequence[MutableMapping[str, Any]]) -> None:
    features = [
        "initial_speed_mps",
        "source_curvature_p90_inv_m",
        "target_curvature_p90_inv_m",
        "nominal_lane_width_m",
        "traffic_density_replay_track_count",
        "paired_reference_remaining_m",
    ]
    for direction in ("left", "right"):
        subset = [row for row in rows if row["direction"] == direction]
        for row in subset:
            labels = [
                f"{feature.replace('_p90_inv_m', '').replace('_mps', '').replace('_m', '')}="
                f"{quantile_label(float(row[feature]), [float(peer[feature]) for peer in subset])}"
                for feature in features
            ]
            row["selection_stratum"] = ";".join(labels)


def normalized_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    features = [
        "initial_speed_mps",
        "source_curvature_p90_inv_m",
        "target_curvature_p90_inv_m",
        "nominal_lane_width_m",
        "traffic_density_replay_track_count",
        "paired_reference_remaining_m",
    ]
    matrix = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)
    minimum = np.min(matrix, axis=0)
    span = np.maximum(np.max(matrix, axis=0) - minimum, 1e-9)
    return (matrix - minimum) / span


def select_direction(
    rows: Sequence[Dict[str, Any]], count: int, globally_used_logs: Set[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select a direction deterministically, maximizing geometric diversity and log coverage."""
    if len(rows) < count:
        raise ValueError(f"insufficient {rows[0]['direction'] if rows else 'unknown'} candidates: {len(rows)} < {count}")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["selection_seed_rank"])
    geometry = normalized_matrix(ordered)
    selected_indices: List[int] = []
    selected_direction_logs: Set[str] = set()
    traces: List[Dict[str, Any]] = []
    for order in range(1, count + 1):
        remaining = [index for index in range(len(ordered)) if index not in selected_indices]
        unseen_direction = [index for index in remaining if ordered[index]["log_name"] not in selected_direction_logs]
        eligible = unseen_direction if unseen_direction else remaining
        unseen_global = [index for index in eligible if ordered[index]["log_name"] not in globally_used_logs]
        if unseen_global:
            eligible = unseen_global
        if not selected_indices:
            chosen = min(eligible, key=lambda index: ordered[index]["selection_seed_rank"])
            diversity_score = None
        else:
            def score(index: int) -> Tuple[float, str]:
                distance = float(np.min(np.linalg.norm(geometry[index] - geometry[selected_indices], axis=1)))
                roadblock_bonus = 0.04 if all(
                    ordered[index]["source_roadblock_id"] != ordered[prior]["source_roadblock_id"]
                    for prior in selected_indices
                ) else 0.0
                return distance + roadblock_bonus, ordered[index]["selection_seed_rank"]
            chosen, (diversity_score, _) = max(((index, score(index)) for index in eligible), key=lambda item: item[1])
        row = ordered[chosen]
        new_direction_log = row["log_name"] not in selected_direction_logs
        new_global_log = row["log_name"] not in globally_used_logs
        action = "NEW_DIRECTION_AND_GLOBAL_LOG" if new_direction_log and new_global_log else (
            "NEW_DIRECTION_LOG" if new_direction_log else "REQUIRED_DIRECTION_LOG_REUSE"
        )
        selected_indices.append(chosen)
        selected_direction_logs.add(row["log_name"])
        globally_used_logs.add(row["log_name"])
        selected = dict(row)
        selected.update(
            {
                "selection_order_within_direction": order,
                "selection_global_log_action": action,
                "selection_diversity_score": diversity_score,
            }
        )
        traces.append(selected)
    return traces, ordered


def select_roster_rows(rows: Sequence[Mapping[str, str]], protocol: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    enriched = enrich_geometry(rows)
    add_selection_strata(enriched)
    quotas = protocol["selection"]["direction_quotas"]
    global_logs: Set[str] = set()
    selected_left, left_candidates = select_direction(
        [row for row in enriched if row["direction"] == "left"], int(quotas["left"]), global_logs
    )
    selected_right, right_candidates = select_direction(
        [row for row in enriched if row["direction"] == "right"], int(quotas["right"]), global_logs
    )
    selected = selected_left + selected_right
    selected_by_token = {row["scenario_token"]: row for row in selected}
    trace: List[Dict[str, Any]] = []
    for row in sorted(left_candidates + right_candidates, key=lambda item: (item["direction"], item["selection_seed_rank"])):
        output = dict(row)
        chosen = selected_by_token.get(row["scenario_token"])
        output["selected"] = chosen is not None
        output["selection_order_within_direction"] = chosen.get("selection_order_within_direction", "") if chosen else ""
        output["selection_global_log_action"] = chosen.get("selection_global_log_action", "NOT_SELECTED") if chosen else "NOT_SELECTED"
        output["selection_diversity_score"] = chosen.get("selection_diversity_score", "") if chosen else ""
        trace.append(output)
    return selected, trace


def validate_pool_rows(rows: Sequence[Mapping[str, str]], protocol: Mapping[str, Any]) -> None:
    if len(rows) != 152:
        raise ValueError(f"Pool B must contain exactly 152 frozen candidates, got {len(rows)}")
    if len({row["scenario_token"] for row in rows}) != len(rows):
        raise ValueError("Pool B contains duplicate scenario tokens")
    dynamic = protocol["eligibility"]["dynamic_requirements"]
    violations: List[str] = []
    for row in rows:
        if row.get("map_name") != protocol["eligibility"]["map_name"]:
            violations.append(f"unsupported map {row['scenario_token']}")
        if not all(as_bool(row.get(name)) for name in ("eligible", "static_reference_coverage_pass", "dynamic_clearance_pass", "official_query_runnable")):
            violations.append(f"static/dynamic/runnability failure {row['scenario_token']}")
        if row.get("dynamic_reason_code") != "DYNAMIC_CLEAR":
            violations.append(f"dynamic reason not clear {row['scenario_token']}")
        if not as_bool(row.get("dynamic_dose_independent")) or not as_bool(row.get("dynamic_eligibility_pre_treatment")):
            violations.append(f"pre-treatment/dose-independence failure {row['scenario_token']}")
        if float(row["minimum_target_lane_object_gap_m"]) < 15.0:
            violations.append(f"initial target gap failure {row['scenario_token']}")
        if float(row["dynamic_horizon_seconds"]) != float(dynamic["horizon_s"]):
            violations.append(f"dynamic horizon mismatch {row['scenario_token']}")
        if float(row["dynamic_time_step_seconds"]) != float(dynamic["time_step_s"]):
            violations.append(f"dynamic time grid mismatch {row['scenario_token']}")
    if violations:
        raise ValueError("; ".join(violations[:8]))


def development_sets(ledger_rows: Sequence[Mapping[str, str]]) -> Tuple[Set[str], Set[str]]:
    tokens = {str(row["scenario_token"]) for row in ledger_rows if row.get("scenario_token")}
    logs = {
        str(row["log_name"])
        for row in ledger_rows
        if row.get("log_name") and "STAGE7L_B_" in str(row.get("exclusion_reason", ""))
    }
    if len(logs) != 26:
        raise ValueError(f"expected 26 Stage7L-B development logs in permanent ledger, got {len(logs)}")
    return tokens, logs


def scene_boundary(db_path: Path, token: str) -> Dict[str, Any]:
    query = """
        WITH ordered_scenes AS (
            SELECT token, name, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num FROM scene
        ), num_scenes AS (SELECT COUNT(*) AS cnt FROM scene)
        SELECT o.row_num, n.cnt AS scene_count, lower(hex(lp.scene_token)) AS db_scene_token
        FROM lidar_pc AS lp
        INNER JOIN ordered_scenes AS o ON o.token = lp.scene_token
        CROSS JOIN num_scenes AS n
        WHERE lp.token = ?
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(query, (bytes.fromhex(token),)).fetchone()
    if row is None:
        raise ValueError(f"scenario token absent from DB: {db_path.name}:{token}")
    return dict(row)


def official_runnability(
    rows: Sequence[Mapping[str, Any]], db_root: Path, devkit_root: Path
) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]]]:
    """Recheck official nuPlan query semantics without constructing a planner rollout."""
    if str(devkit_root) not in sys.path:
        sys.path.insert(0, str(devkit_root))
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["db_file"])].append(row)
    runnable: Dict[str, bool] = {}
    boundaries: Dict[str, Dict[str, Any]] = {}
    for db_file, group in grouped.items():
        db_path = db_root / db_file
        if not db_path.is_file():
            raise FileNotFoundError(f"confirmation DB missing: {db_path}")
        requested = [str(row["scenario_token"]) for row in group]
        official = {
            result["token"].hex()
            for result in get_scenarios_from_db(
                str(db_path), requested, None, None,
                include_invalid_mission_goals=True, include_cameras=False,
            )
        }
        for row in group:
            token = str(row["scenario_token"])
            boundary = scene_boundary(db_path, token)
            expected = int(boundary["row_num"]) >= 3 and int(boundary["row_num"]) < int(boundary["scene_count"]) - 1
            runnable[token] = token in official
            boundaries[token] = boundary
            if runnable[token] != expected:
                raise ValueError(f"official query/boundary disagreement: {db_file}:{token}")
    return runnable, boundaries


def reference_sha(raw: str) -> str:
    return canonical_sha(json.loads(raw))


def roster_row(row: Mapping[str, Any], order: int, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    treatment = protocol["treatment"]
    canonical_generator = {
        "initial_speed_mps": float(row["initial_speed_mps"]),
        "target_speed_mps": treatment["target_speed_mps"],
        "accel_limit_mps2": treatment["accel_limit_mps2"],
        "sampling_interval_s": treatment["sampling_interval_s"],
        "scenario_horizon_s": treatment["scenario_horizon_s"],
    }
    return {
        "collection_order": order,
        "selection_role": "STAGE7L_C_CONFIRMATION_FROZEN_NOT_RUN",
        "scenario_token": row["scenario_token"], "log_name": row["log_name"], "db_file": row["db_file"],
        "map_name": row["map_name"], "direction": row["direction"],
        "source_lane_id": row["source_lane_id"], "target_lane_id": row["target_lane_id"],
        "source_roadblock_id": row["source_roadblock_id"], "target_roadblock_id": row["target_roadblock_id"],
        "initial_state_fingerprint": row["initial_state_fingerprint"], "route_fingerprint": row["route_fingerprint"],
        "initial_speed_mps": float(row["initial_speed_mps"]), "nominal_lane_width_m": float(row["nominal_lane_width_m"]),
        "paired_reference_remaining_m": float(row["paired_reference_remaining_m"]),
        "dynamic_replay_track_count": int(float(row["dynamic_replay_track_count"])),
        "trigger_s_route_m": treatment["trigger_s_route_m"],
        "dose_family_sha256": canonical_sha(treatment["dose_transition_length_m"]),
        "canonical_generator_config_sha256": canonical_sha(canonical_generator),
        "dynamic_clearance_config_sha256": row["dynamic_clearance_config_sha256"],
        "dynamic_clearance_status": row["dynamic_reason_code"],
        "official_query_runnable": True,
        "source_reference_sha256": reference_sha(row["source_reference_xy_json"]),
        "target_reference_sha256": reference_sha(row["target_reference_xy_json"]),
        "selection_stratum": row["selection_stratum"], "selection_seed_rank": row["selection_seed_rank"],
        "selection_order_within_direction": row["selection_order_within_direction"],
        "selection_global_log_action": row["selection_global_log_action"],
    }


def maneuver_row(row: Mapping[str, Any], roster: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **dict(roster),
        "initial_x": float(row["initial_x"]), "initial_y": float(row["initial_y"]),
        "initial_heading": float(row["initial_heading"]),
        "source_start_arc_m": float(row["source_start_arc_m"]), "target_start_arc_m": float(row["target_start_arc_m"]),
        "route_roadblock_ids": json.loads(row["route_roadblock_ids_json"]),
        "source_reference_xy": json.loads(row["source_reference_xy_json"]),
        "target_reference_xy": json.loads(row["target_reference_xy_json"]),
        "background_mode": protocol["treatment"]["background_mode"],
        "planner_horizon_s": protocol["treatment"]["planner_horizon_s"],
        "scenario_horizon_s": protocol["treatment"]["scenario_horizon_s"],
    }


def write_geometry_summary(path: Path, selected: Sequence[Mapping[str, Any]], trace: Sequence[Mapping[str, Any]]) -> None:
    metrics = [
        "initial_speed_mps", "source_curvature_p90_inv_m", "target_curvature_p90_inv_m",
        "nominal_lane_width_m", "traffic_density_replay_track_count", "paired_reference_remaining_m",
    ]
    output: Dict[str, Any] = {"schema_version": "stage7l_c_geometry_summary_v1", "directions": {}}
    for direction in ("left", "right"):
        candidates = [row for row in trace if row["direction"] == direction]
        chosen = [row for row in selected if row["direction"] == direction]
        summary: Dict[str, Any] = {"candidate_count": len(candidates), "selected_count": len(chosen), "metrics": {}}
        for metric in metrics:
            summary["metrics"][metric] = {
                "candidate_q10_q50_q90": [float(value) for value in np.quantile([float(row[metric]) for row in candidates], [0.1, 0.5, 0.9])],
                "selected_q10_q50_q90": [float(value) for value in np.quantile([float(row[metric]) for row in chosen], [0.1, 0.5, 0.9])],
            }
        output["directions"][direction] = summary
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.repo_root.resolve()
    protocol = read_json(args.protocol_config)
    if args.protocol_config.resolve() != root / "configs/stage7l_c_prospective_confirmation_protocol_v1.json":
        raise ValueError("Stage7L-C only accepts the fixed prospective protocol config")
    assets = validate_authority_assets(root, protocol)
    if args.pool_b.resolve() != assets["pool_b"].resolve() or args.development_ledger.resolve() != assets["development_exclusion_ledger"].resolve():
        raise ValueError("Pool B and development ledger must match the frozen protocol source assets")
    pool_rows = read_csv(args.pool_b)
    validate_pool_rows(pool_rows, protocol)
    ledger_rows = read_csv(args.development_ledger)
    excluded_tokens, development_logs = development_sets(ledger_rows)
    pool_tokens = {row["scenario_token"] for row in pool_rows}
    pool_logs = {row["log_name"] for row in pool_rows}
    if pool_tokens & excluded_tokens:
        raise ValueError("Pool B overlaps a permanently excluded historical scenario token")
    if pool_logs & development_logs:
        raise ValueError("Pool B overlaps a Stage7L-B development log")

    selected, trace = select_roster_rows(pool_rows, protocol)
    quotas = protocol["selection"]["direction_quotas"]
    if len(selected) != int(protocol["selection"]["scenario_count"]):
        raise AssertionError("roster selection did not produce 80 scenarios")
    if sum(row["direction"] == "left" for row in selected) != int(quotas["left"]):
        raise AssertionError("frozen left quota is not 15")
    if sum(row["direction"] == "right" for row in selected) != int(quotas["right"]):
        raise AssertionError("frozen right quota is not 65")
    selected_tokens = {row["scenario_token"] for row in selected}
    selected_logs = {row["log_name"] for row in selected}
    if len(selected_tokens) != 80:
        raise AssertionError("confirmation roster has duplicate token")
    if selected_tokens & excluded_tokens or selected_logs & development_logs:
        raise AssertionError("confirmation roster overlaps development")

    runnable, boundaries = official_runnability(selected, args.nuplan_db_root, args.nuplan_devkit_root)
    if sum(runnable.values()) != 80:
        raise ValueError("official runnability is not 80/80")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite a frozen confirmation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    roster = [roster_row(row, index, protocol) for index, row in enumerate(selected, start=1)]
    roster_by_token = {row["scenario_token"]: row for row in roster}
    roster_path = output_dir / "confirmation_roster.csv"
    write_csv(roster_path, roster, list(roster[0].keys()))
    maneuver_payload = {
        "schema_version": "stage7l_c_confirmation_maneuver_manifest_v1",
        "status": "STAGE7L_C_CONFIRMATION_MANEUVER_MANIFEST_FROZEN_NOT_RUN",
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "dose_transition_length_m": protocol["treatment"]["dose_transition_length_m"],
        "maneuvers": [maneuver_row(row, roster_by_token[row["scenario_token"]], protocol) for row in selected],
    }
    maneuver_path = output_dir / "confirmation_maneuver_manifest.json"
    maneuver_path.write_text(json.dumps(maneuver_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace_path = output_dir / "confirmation_selection_trace.csv"
    trace_fields = [
        "scenario_token", "log_name", "db_file", "direction", "selection_seed_rank", "selected",
        "selection_order_within_direction", "selection_global_log_action", "selection_diversity_score",
        "selection_stratum", "initial_speed_mps", "source_curvature_p90_inv_m", "target_curvature_p90_inv_m",
        "nominal_lane_width_m", "traffic_density_replay_track_count", "paired_reference_remaining_m",
        "source_roadblock_id", "route_fingerprint",
    ]
    write_csv(trace_path, trace, trace_fields)
    geometry_path = output_dir / "confirmation_geometry_summary.json"
    write_geometry_summary(geometry_path, selected, trace)
    runnability_rows = []
    for row in roster:
        boundary = boundaries[row["scenario_token"]]
        runnability_rows.append({
            "scenario_token": row["scenario_token"], "log_name": row["log_name"], "db_file": row["db_file"],
            "scene_token": boundary["db_scene_token"], "scene_row_num": boundary["row_num"],
            "scene_count": boundary["scene_count"], "official_query_runnable": runnable[row["scenario_token"]],
            "selected_for_confirmation": True,
        })
    runnability_path = output_dir / "confirmation_runnability_audit.csv"
    write_csv(runnability_path, runnability_rows, list(runnability_rows[0].keys()))
    exclusion_rows = []
    for row in pool_rows:
        exclusion_rows.append({
            "scenario_token": row["scenario_token"], "log_name": row["log_name"], "direction": row["direction"],
            "selected": row["scenario_token"] in selected_tokens,
            "selection_outcome": "SELECTED" if row["scenario_token"] in selected_tokens else "REMAINING_RESERVE_NOT_A_REPLACEMENT_POOL",
            "historical_scenario_disjoint": row["scenario_token"] not in excluded_tokens,
            "stage7l_b_development_log_disjoint": row["log_name"] not in development_logs,
            "official_query_runnable": as_bool(row["official_query_runnable"]),
            "dynamic_clearance_pass": as_bool(row["dynamic_clearance_pass"]),
        })
    exclusion_path = output_dir / "confirmation_exclusion_audit.csv"
    write_csv(exclusion_path, exclusion_rows, list(exclusion_rows[0].keys()))
    reserve_path = output_dir / "remaining_reserve_inventory.csv"
    write_csv(reserve_path, [row for row in exclusion_rows if not row["selected"]], list(exclusion_rows[0].keys()))

    log_counts = Counter(row["log_name"] for row in roster)
    required_dynamic_sha = read_json(assets["b2_dynamic_clearance_manifest"])["dynamic_clearance"]["config_fingerprint"]
    all_dynamic_config_match = all(row["dynamic_clearance_config_sha256"] == required_dynamic_sha for row in roster)
    assertions = {
        "scenario_count_equals_80": len(roster) == 80,
        "left_equals_15": sum(row["direction"] == "left" for row in roster) == 15,
        "right_equals_65": sum(row["direction"] == "right" for row in roster) == 65,
        "duplicate_token_count_equals_0": len(roster) == len(selected_tokens),
        "historical_scenario_overlap_count_equals_0": len(selected_tokens & excluded_tokens) == 0,
        "stage7l_b_development_log_overlap_count_equals_0": len(selected_logs & development_logs) == 0,
        "official_runnability_80_of_80": sum(runnable.values()) == 80,
        "dynamic_clearance_80_of_80": all(row["dynamic_clearance_status"] == "DYNAMIC_CLEAR" for row in roster),
        "static_eligibility_80_of_80": all(as_bool(row["official_query_runnable"]) for row in roster),
        "manifest_source_target_trigger_complete_80_of_80": all(
            row["source_reference_sha256"] and row["target_reference_sha256"] and row["trigger_s_route_m"] == 12.0
            for row in roster
        ),
        "unsupported_map_count_equals_0": all(row["map_name"] == "us-pa-pittsburgh-hazelwood" for row in roster),
        "dynamic_config_sha_matches_b2": all_dynamic_config_match,
    }
    if not all(assertions.values()):
        failed = [key for key, value in assertions.items() if not value]
        raise AssertionError(f"Stage7L-C freeze assertions failed: {failed}")
    summary = {
        "schema_version": "stage7l_c_confirmation_freeze_summary_v1",
        "status": "STAGE7L_C_PROSPECTIVE_PROTOCOL_FROZEN_AND_CONFIRMATION_ROSTER_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_repository_head": git_head(root), "protocol_config_sha256": sha256_file(args.protocol_config),
        "scenario_count": len(roster), "distinct_log_count": len(log_counts), "scenarios_per_log": dict(sorted(log_counts.items())),
        "left": 15, "right": 65, "left_log_count": len({row["log_name"] for row in roster if row["direction"] == "left"}),
        "right_log_count": len({row["log_name"] for row in roster if row["direction"] == "right"}),
        "reused_log_count": sum(count > 1 for count in log_counts.values()),
        "reused_log_reason": "one left-stratum reuse is unavoidable because Pool B has 19 left candidates across 14 logs; all reuse is recorded before outcome access",
        "development_scenario_overlap_count": 0, "development_log_overlap_count": 0,
        "official_runnability_selected_count": 80, "dynamic_clearance_selected_count": 80,
        "selection_is_pre_treatment": True, "selection_used_rollout_outcomes": False,
        "selection_used_embedding_or_bdd": False, "confirmation_rollout_started": False,
        "stage7l_d_started": False, "assertions": assertions,
        "sha256": {
            "protocol_config": sha256_file(args.protocol_config), "pool_b": sha256_file(args.pool_b),
            "development_ledger": sha256_file(args.development_ledger), "confirmation_roster": sha256_file(roster_path),
            "confirmation_maneuver_manifest": sha256_file(maneuver_path), "selection_trace": sha256_file(trace_path),
            "geometry_summary": sha256_file(geometry_path), "runnability_audit": sha256_file(runnability_path),
            "exclusion_audit": sha256_file(exclusion_path), "reserve_inventory": sha256_file(reserve_path),
        },
    }
    summary_path = output_dir / "confirmation_freeze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    authorization = {
        "schema_version": "stage7l_c_blind_confirmation_authorization_manifest_v1",
        "status": "STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED",
        "authorization_scope": "Stage7L-D planner-level one-time confirmation only: 80 frozen scenarios x 5 frozen doses = 400 official rollouts",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_repository_head": git_head(root),
        "stage7l_d_started": False,
        "stage7l_e_representation_evaluation_authorized_now": False,
        "representation_unlock_condition": "ONLY_IF_STAGE7L_D_MECHANISM_AND_SAFETY_GATES_PASS",
        "frozen_protocol": {
            "path": str(args.protocol_config.resolve()),
            "sha256": sha256_file(args.protocol_config),
            "treatment": protocol["treatment"],
            "eligibility": protocol["eligibility"],
            "mechanism": protocol["mechanism"],
            "nuisance_gate": protocol["nuisance_gate"],
            "safety_validity_gate": protocol["safety_validity_gate"],
            "failure_policy": protocol["failure_policy"],
            "paired_bdd": protocol["paired_bdd"],
            "representation_lock": protocol["representation_lock"],
        },
        "frozen_confirmation_artifacts": {
            "roster": {"path": str(roster_path.resolve()), "sha256": summary["sha256"]["confirmation_roster"]},
            "maneuver_manifest": {"path": str(maneuver_path.resolve()), "sha256": summary["sha256"]["confirmation_maneuver_manifest"]},
            "selection_trace": {"path": str(trace_path.resolve()), "sha256": summary["sha256"]["selection_trace"]},
            "runnability_audit": {"path": str(runnability_path.resolve()), "sha256": summary["sha256"]["runnability_audit"]},
            "freeze_summary": {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)},
        },
        "hard_assertions": assertions,
        "no_replacement_rule": True,
        "forbidden_before_stage7l_d_e_complete": protocol["immutability"]["after_freeze_changes_forbidden"],
        "forbidden_actions": ["new_development", "training", "checkpoint_change", "embedding_export", "BDD_or_MMD_before_mechanism_unlock", "scenario_replacement"],
        "claim_boundary": protocol["claim_boundary"],
    }
    if args.authorization_manifest.exists():
        raise FileExistsError(f"refusing to overwrite blind authorization manifest: {args.authorization_manifest}")
    args.authorization_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.authorization_manifest.write_text(json.dumps(authorization, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=ROOT)
    parser.add_argument("--protocol_config", type=Path, required=True)
    parser.add_argument("--pool_b", type=Path, required=True)
    parser.add_argument("--development_ledger", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--authorization_manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
