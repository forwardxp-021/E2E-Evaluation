#!/usr/bin/env python3
"""Freeze the R1 Phase-B2 fresh official technical-smoke roster.

This is an outcome-blind, read-only DB/map selector.  It may read only
pre-treatment identity, official initial state, route, and native-map data.
It never starts nuPlan, opens a historical smoke executor, or reads treatment
outcomes, representation, BDD, probes, checkpoints, or RBR artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.stage7l_build_lane_change_opportunity_inventory import (
    line_values,
    official_simulation_initial_token,
    route_ids,
)
from tools.stage7l_pure_lateral_execution_planner import initial_state_fingerprint


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_CACHE_ROOT = ROOT.parent / "nuplan/dataset/data/cache"
DEFAULT_MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
SELECTOR = R1_DIR / "r1_future_compliant_smoke_selector_contract_v0.3.json"
SCOPE_AMENDMENT = R1_DIR / "r1_official_technical_smoke_scope_amendment_v1.0.json"
SOURCE_UNIVERSE = R1_DIR / "r1_fresh_smoke_source_universe_v0.1.json"
OLD_SMOKE_BLACKLIST = R1_DIR / "r1_technical_smoke_v1_permanent_blacklist_v1.json"
RUNTIME_ROSTER = R1_DIR / "r1_runtime_determinism_validation_roster_v1.0.json"
FUTURE_R4 = ROOT / "docs/stageR/r0/manifests/r0_future_r4_reserved_pool_freeze_v0.1.json"
OUTPUT = R1_DIR / "r1_official_technical_smoke_roster_v1.0.json"
BLACKLIST_OUTPUT = R1_DIR / "r1_official_technical_smoke_permanent_blacklist_v1.0.json"

FAMILIES = ("R-HLC", "R-TSB")
RANK_PREFIX_SIZE = 32768
OLD_SMOKE_LABEL = "HISTORICAL_NONCOMPLIANT_TECHNICAL_SMOKE"
RUNTIME_LABEL = "RUNTIME_DETERMINISM_VALIDATION_ONLY"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen Phase-B2 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def rank_digest(salt_sha256: str, family: str, scenario_token: str, log_id: str) -> str:
    return hashlib.sha256(f"{salt_sha256}|{family}|{scenario_token}|{log_id}".encode("utf-8")).hexdigest()


def _candidate_rows(db_path: Path, source_partition: str) -> Iterable[Dict[str, Any]]:
    """Yield only official-runnable non-edge anchors, without any speed/gap threshold."""
    query = """
        WITH ordered_scenes AS (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS scene_row_num FROM scene
        ), scene_count AS (SELECT COUNT(*) AS n FROM scene)
        SELECT lower(hex(lp.token)) AS scenario_token, lp.timestamp,
               l.logfile AS log_id, l.map_version AS map_name
        FROM lidar_pc lp
        JOIN lidar ld ON ld.token = lp.lidar_token
        JOIN log l ON l.token = ld.log_token
        JOIN ordered_scenes os ON os.token = lp.scene_token
        CROSS JOIN scene_count sc
        WHERE os.scene_row_num >= 3 AND os.scene_row_num < sc.n - 1
        ORDER BY lp.timestamp ASC
    """
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for token, timestamp, log_id, map_name in connection.execute(query):
            yield {
                "scenario_token": str(token),
                "timestamp": int(timestamp),
                "log_id": str(log_id),
                "map_name": str(map_name),
                "db_file": db_path.name,
                "db_path": str(db_path.resolve()),
                "source_partition": source_partition,
            }


def ranked_prefix(
    cache_root: Path,
    family: str,
    salt_sha256: str,
    forbidden_tokens: set[str],
    forbidden_logs: set[str],
    prefix_size: int,
) -> List[Dict[str, Any]]:
    """Return a complete deterministic rank prefix without adding eligibility cutoffs."""
    heap: List[Tuple[int, str, str, int, str, int, Dict[str, Any]]] = []
    enumeration_index = 0
    for partition in ("mini", "train_pittsburgh"):
        for db_path in sorted((cache_root / partition).glob("*.db")):
            for row in _candidate_rows(db_path, partition):
                enumeration_index += 1
                token, log_id = row["scenario_token"], row["log_id"]
                if token in forbidden_tokens or log_id in forbidden_logs:
                    continue
                digest = rank_digest(salt_sha256, family, token, log_id)
                rank_int = int(digest, 16)
                row["selector_rank_sha256"] = digest
                # The same lidar identity can occur in more than one source
                # partition.  Keep an explicit deterministic enumeration
                # tie-break before the dictionary so heap comparisons never
                # depend on non-orderable mapping objects.
                item = (-rank_int, token, log_id, int(row["timestamp"]), str(row["db_file"]), enumeration_index, row)
                if len(heap) < prefix_size:
                    heapq.heappush(heap, item)
                elif rank_int < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    rows = [item[-1] for item in heap]
    rows.sort(key=lambda row: (str(row["selector_rank_sha256"]), str(row["scenario_token"]), str(row["log_id"]), int(row["timestamp"])))
    return rows


def _official_initial(candidate: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db

    db_path = Path(str(candidate["db_path"]))
    token, timestamp_us, scenario_types = official_simulation_initial_token(
        db_path, str(candidate["scenario_token"]), int(candidate["timestamp"])
    )
    ego = get_ego_state_for_lidarpc_token_from_db(str(db_path), token)
    if ego is None:
        raise ValueError("official initial ego state is unavailable")
    speed = float(ego.dynamic_car_state.speed)
    if not np.isfinite(speed):
        raise ValueError("official initial ego speed is non-finite")
    result = {
        "official_simulation_initial_lidar_token": str(token),
        "official_simulation_initial_timestamp_us": int(timestamp_us),
        "official_scenario_types": sorted(str(value) for value in scenario_types),
        "initial_x": round(float(ego.rear_axle.x), 6),
        "initial_y": round(float(ego.rear_axle.y), 6),
        "initial_heading": round(float(ego.rear_axle.heading), 8),
        "initial_speed_mps": round(speed, 6),
        "initial_time_us": int(ego.time_us),
    }
    result["initial_state_fingerprint"] = initial_state_fingerprint(
        result["initial_x"], result["initial_y"], result["initial_heading"], result["initial_speed_mps"], result["initial_time_us"]
    )
    route = route_ids(db_path, str(candidate["scenario_token"]))
    if not route:
        raise ValueError("official route roadblocks are unavailable")
    return result, route


def _map_api(map_root: Path, map_name: str, cache: MutableMapping[str, Any]) -> Any:
    if map_name not in cache:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        cache[map_name] = get_maps_api(str(map_root), "nuplan-maps-v1.0", map_name)
    return cache[map_name]


def _heading_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _hlc_entry(candidate: Mapping[str, Any], map_root: Path, map_cache: MutableMapping[str, Any]) -> Dict[str, Any]:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    initial, route = _official_initial(candidate)
    api = _map_api(map_root, str(candidate["map_name"]), map_cache)
    point = Point2D(float(initial["initial_x"]), float(initial["initial_y"]))
    lanes = list(api.get_all_map_objects(point, SemanticMapLayer.LANE))
    if not lanes:
        raise ValueError("official map has no source lane at initial ego state")
    lanes.sort(key=lambda lane: (_heading_error(float(initial["initial_heading"]), float(lane.baseline_path.get_nearest_pose_from_position(point).heading)), str(lane.id)))
    source = lanes[0]
    source_roadblock = str(source.get_roadblock_id())
    if source_roadblock not in route:
        raise ValueError("native source-lane roadblock is absent from official route")
    options: List[Tuple[str, str, Any, List[List[float]], List[List[float]], float, float]] = []
    for direction, target in (("left", source.adjacent_edges[0]), ("right", source.adjacent_edges[1])):
        if target is None or str(target.get_roadblock_id()) != source_roadblock:
            continue
        source_xy, source_arc, _ = line_values(source, float(initial["initial_x"]), float(initial["initial_y"]))
        target_xy, target_arc, _ = line_values(target, float(initial["initial_x"]), float(initial["initial_y"]))
        source_array, target_array = np.asarray(source_xy, dtype=np.float64), np.asarray(target_xy, dtype=np.float64)
        if source_array.ndim != 2 or target_array.ndim != 2 or source_array.shape[1] != 2 or target_array.shape[1] != 2 or len(source_array) < 2 or len(target_array) < 2:
            continue
        if not np.isfinite(source_array).all() or not np.isfinite(target_array).all():
            continue
        options.append((direction, str(target.id), target, source_xy, target_xy, source_arc, target_arc))
    if not options:
        raise ValueError("no native same-roadblock adjacent target lane with finite references")
    direction, _, target, source_xy, target_xy, source_arc, target_arc = sorted(options, key=lambda item: (item[0], item[1]))[0]
    return {
        "family": "R-HLC",
        "scenario_token": str(candidate["scenario_token"]),
        "log_id": str(candidate["log_id"]),
        "db_file": str(candidate["db_file"]),
        "db_path": str(candidate["db_path"]),
        "source_partition": str(candidate["source_partition"]),
        "map_name": str(candidate["map_name"]),
        "selector_rank_sha256": str(candidate["selector_rank_sha256"]),
        "scenario_anchor_timestamp_us": int(candidate["timestamp"]),
        "initial_state": initial,
        "route_roadblock_ids": route,
        "route_fingerprint": canonical_sha256(route),
        "source_lane_id": str(source.id),
        "target_lane_id": str(target.id),
        "source_roadblock_id": source_roadblock,
        "target_roadblock_id": str(target.get_roadblock_id()),
        "direction": direction,
        "source_start_arc_m": round(float(source_arc), 6),
        "target_start_arc_m": round(float(target_arc), 6),
        "source_reference_xy": source_xy,
        "target_reference_xy": target_xy,
        "pre_treatment_eligibility": [
            "OFFICIAL_INITIAL_STATE_AVAILABLE", "OFFICIAL_ROUTE_NONEMPTY", "NATIVE_SOURCE_LANE_RESOLVED", "SOURCE_ROADBLOCK_ON_ROUTE", "NATIVE_SAME_ROADBLOCK_TARGET_LANE", "FINITE_SOURCE_TARGET_LANE_REFERENCES"
        ],
        "relevant_context_ids": {"source_lane_id": str(source.id), "target_lane_id": str(target.id), "source_roadblock_id": source_roadblock, "target_roadblock_id": str(target.get_roadblock_id())},
        "arms": ["HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE", "HLC_TREATMENT_HLC_GEN_V2_OPTION_B"],
        "isolation_labels": ["FRESH_OFFICIAL_TECHNICAL_SMOKE_ONLY", "EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER", "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION"],
    }


def _tsb_entry(candidate: Mapping[str, Any], map_root: Path, map_cache: MutableMapping[str, Any]) -> Dict[str, Any]:
    initial, route = _official_initial(candidate)
    _map_api(map_root, str(candidate["map_name"]), map_cache)
    return {
        "family": "R-TSB",
        "scenario_token": str(candidate["scenario_token"]),
        "log_id": str(candidate["log_id"]),
        "db_file": str(candidate["db_file"]),
        "db_path": str(candidate["db_path"]),
        "source_partition": str(candidate["source_partition"]),
        "map_name": str(candidate["map_name"]),
        "selector_rank_sha256": str(candidate["selector_rank_sha256"]),
        "scenario_anchor_timestamp_us": int(candidate["timestamp"]),
        "initial_state": initial,
        "route_roadblock_ids": route,
        "route_fingerprint": canonical_sha256(route),
        "source_lane_id": "NOT_APPLICABLE_TSB",
        "target_lane_id": "NOT_APPLICABLE_TSB",
        "pre_treatment_eligibility": ["OFFICIAL_INITIAL_STATE_AVAILABLE", "OFFICIAL_ROUTE_NONEMPTY", "OFFICIAL_MAP_AVAILABLE"],
        "relevant_context_ids": {"route_roadblock_ids": route},
        "arms": ["TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING", "TSB_TREATMENT_TSB_GEN_V2_OPTION_A"],
        "isolation_labels": ["FRESH_OFFICIAL_TECHNICAL_SMOKE_ONLY", "EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER", "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION"],
    }


def _select_family(
    family: str,
    candidates: Sequence[Mapping[str, Any]],
    required: int,
    used_tokens: set[str],
    used_logs: set[str],
    map_root: Path,
    map_cache: MutableMapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    selected: List[Dict[str, Any]] = []
    rejections: List[Dict[str, str]] = []
    builder = _hlc_entry if family == "R-HLC" else _tsb_entry
    for candidate in candidates:
        token, log_id = str(candidate["scenario_token"]), str(candidate["log_id"])
        if token in used_tokens or log_id in used_logs:
            rejections.append({"scenario_token": token, "log_id": log_id, "reason": "GLOBAL_TOKEN_OR_LOG_DIVERSITY_EXCLUSION"})
            continue
        try:
            entry = builder(candidate, map_root, map_cache)
        except Exception as exc:
            rejections.append({"scenario_token": token, "log_id": log_id, "reason": f"PRETREATMENT_INELIGIBLE:{type(exc).__name__}"})
            continue
        selected.append(entry)
        used_tokens.add(token)
        used_logs.add(log_id)
        if len(selected) == required:
            return selected, rejections
    raise RuntimeError(f"{family} has fewer than {required} eligible identities in the complete inspected rank prefix")


def build_exclusion_blacklist() -> Dict[str, Any]:
    old = read_json(OLD_SMOKE_BLACKLIST)
    runtime = read_json(RUNTIME_ROSTER)
    future_r4 = read_json(FUTURE_R4)
    entries: List[Dict[str, str]] = []
    for row in old.get("entries", []):
        entries.append({"scenario_token": str(row["scenario_token"]), "log_id": str(row["log_id"]), "reason": OLD_SMOKE_LABEL})
    for row in runtime.get("entries", []):
        entries.append({"scenario_token": str(row["scenario_token"]), "log_id": str(row["log_id"]), "reason": RUNTIME_LABEL})
    if bool(future_r4.get("identity_roster_frozen", False)):
        for row in future_r4.get("identities", []):
            entries.append({"scenario_token": str(row["scenario_token"]), "log_id": str(row.get("log_id", "")), "reason": "FUTURE_R4_RESERVED"})
    merged: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in entries:
        key = (row["scenario_token"].lower(), row["log_id"])
        if key not in merged:
            merged[key] = dict(row)
        elif row["reason"] not in merged[key]["reason"].split("|"):
            merged[key]["reason"] += f"|{row['reason']}"
    return {
        "schema_version": "r1_official_technical_smoke_permanent_blacklist_v1.0",
        "status": "PERMANENT_EXCLUSION_ACTIVE_BEFORE_FRESH_SELECTION",
        "match_keys": ["scenario_token", "log_id"],
        "sources": {
            "historical_noncompliant_smoke": {"path": str(OLD_SMOKE_BLACKLIST.relative_to(ROOT)), "sha256": sha256_file(OLD_SMOKE_BLACKLIST)},
            "runtime_determinism_validation": {"path": str(RUNTIME_ROSTER.relative_to(ROOT)), "sha256": sha256_file(RUNTIME_ROSTER)},
            "future_r4_reserved_pool": {"path": str(FUTURE_R4.relative_to(ROOT)), "sha256": sha256_file(FUTURE_R4), "identity_roster_frozen": bool(future_r4.get("identity_roster_frozen", False))},
            "formal_r1_development_roster": {"status": "NONE_CREATED_AT_SELECTION"}
        },
        "entries": sorted(merged.values(), key=lambda row: (row["scenario_token"], row["log_id"])),
        "selected_identity_labels": ["FRESH_OFFICIAL_TECHNICAL_SMOKE_ONLY", "EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER", "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--blacklist-output", type=Path, default=BLACKLIST_OUTPUT)
    parser.add_argument("--rank-prefix-size", type=int, default=RANK_PREFIX_SIZE)
    args = parser.parse_args()
    required_paths = [SELECTOR, SCOPE_AMENDMENT, SOURCE_UNIVERSE, OLD_SMOKE_BLACKLIST, RUNTIME_ROSTER, FUTURE_R4, args.cache_root / "mini", args.cache_root / "train_pittsburgh", args.map_root]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required frozen selector input(s): {missing}")
    if args.output.exists() or args.blacklist_output.exists():
        raise FileExistsError("refusing to overwrite a frozen roster or blacklist")
    if args.rank_prefix_size < 24:
        raise ValueError("rank-prefix-size is an algorithmic completeness guard and must be at least 24")
    selector, scope, source = read_json(SELECTOR), read_json(SCOPE_AMENDMENT), read_json(SOURCE_UNIVERSE)
    if selector.get("status") != "FROZEN_BEFORE_FRESH_OFFICIAL_SMOKE_CANDIDATE_ENUMERATION":
        raise RuntimeError("selector v0.3 is not prospectively frozen")
    if scope.get("status") != "FROZEN_PROSPECTIVE_BEFORE_FRESH_CANDIDATE_ENUMERATION":
        raise RuntimeError("scope amendment is not prospectively frozen")
    if source.get("status") != "READY_FOR_OUTCOME_BLIND_SELECTION":
        raise RuntimeError("fresh source universe is not ready")
    if selector["selection_scope"] != {"families": ["R-HLC", "R-TSB"], "scenarios_per_family": 12, "arms_per_scenario": 2, "planned_official_closed_loop_runs": 48, "authorized_cap": 48}:
        raise RuntimeError("selector scope differs from the prospective 12x2 amendment")
    blacklist = build_exclusion_blacklist()
    forbidden_tokens = {str(row["scenario_token"]).lower() for row in blacklist["entries"]}
    forbidden_logs = {str(row["log_id"]) for row in blacklist["entries"] if str(row["log_id"])}
    salt = str(selector["salt_sha256"])
    used_tokens, used_logs = set(forbidden_tokens), set(forbidden_logs)
    map_cache: Dict[str, Any] = {}
    selections: Dict[str, List[Dict[str, Any]]] = {}
    rejection_counts: Dict[str, int] = {}
    for family in FAMILIES:
        prefix = ranked_prefix(args.cache_root, family, salt, forbidden_tokens, forbidden_logs, args.rank_prefix_size)
        rows, rejected = _select_family(family, prefix, 12, used_tokens, used_logs, args.map_root, map_cache)
        selections[family] = rows
        rejection_counts[family] = len(rejected)
    entries = selections["R-HLC"] + selections["R-TSB"]
    tokens = [str(row["scenario_token"]) for row in entries]
    logs = [str(row["log_id"]) for row in entries]
    if len(entries) != 24 or len(set(tokens)) != 24 or len(set(logs)) != 24:
        raise RuntimeError("SOURCE_DIVERSITY_REQUIREMENT_NOT_MET: 24 unique fresh scenario tokens and logs are required")
    if any(set(row["isolation_labels"]) != {"FRESH_OFFICIAL_TECHNICAL_SMOKE_ONLY", "EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER", "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION"} for row in entries):
        raise RuntimeError("selected isolation labels differ from frozen contract")
    write_new_json(args.blacklist_output, blacklist)
    roster = {
        "schema_version": "r1_official_technical_smoke_roster_v1.0",
        "status": "FROZEN_BEFORE_OFFICIAL_CLOSED_LOOP_EXECUTION",
        "selection_method": "full_pre_treatment_sqlite_rank_prefix_then_native_official_map_and_initial_state_validation",
        "rank_prefix_size": int(args.rank_prefix_size),
        "selector_contract_sha256": sha256_file(SELECTOR),
        "scope_amendment_sha256": sha256_file(SCOPE_AMENDMENT),
        "source_universe_sha256": sha256_file(SOURCE_UNIVERSE),
        "blacklist_sha256": sha256_file(args.blacklist_output),
        "db_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"],
        "map_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"],
        "master_seed": int(selector["master_seed"]),
        "selection_inputs_allowlist": selector["selection_inputs_allowlist"],
        "forbidden_inputs_not_opened": selector["forbidden_selection_inputs"],
        "permanent_selected_identity_labels": selector["isolation_labels_for_selected_identities"],
        "entries": entries,
        "counts": {"R-HLC": len(selections["R-HLC"]), "R-TSB": len(selections["R-TSB"]), "total_scenarios": len(entries), "unique_scenario_tokens": len(set(tokens)), "unique_logs": len(set(logs)), "planned_official_closed_loop_runs": 48},
        "selection_rejection_counts_diagnostic_only": rejection_counts,
        "execution_authorized": False,
    }
    write_new_json(args.output, roster)
    print(json.dumps({"output": str(args.output), "roster_sha256": sha256_file(args.output), "blacklist_output": str(args.blacklist_output), "blacklist_sha256": sha256_file(args.blacklist_output), "counts": roster["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
