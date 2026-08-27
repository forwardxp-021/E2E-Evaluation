#!/usr/bin/env python3
"""Freeze the R1 runtime-determinism roster from pre-treatment nuPlan inputs.

This selector is deliberately independent of the future 48-call smoke selector.
It enumerates only DB/map/initial-state information after the selector contract
has frozen its salt.  It never starts simulation and never opens treatment,
representation, BDD, probe, checkpoint, or RBR artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_CACHE_ROOT = ROOT.parent / "nuplan/dataset/data/cache"
DEFAULT_MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
SELECTOR = R1_DIR / "r1_runtime_determinism_selector_contract_v1.0.json"
SOURCE_UNIVERSE = R1_DIR / "r1_fresh_smoke_source_universe_v0.1.json"
BLACKLIST = R1_DIR / "r1_technical_smoke_v1_permanent_blacklist_v1.json"
HISTORICAL_SMOKE_ROSTER = R1_DIR / "r1_technical_smoke_roster_v1.json"
FUTURE_R4 = ROOT / "docs/stageR/r0/manifests/r0_future_r4_reserved_pool_freeze_v0.1.json"
OUTPUT = R1_DIR / "r1_runtime_determinism_validation_roster_v1.0.json"

FAMILIES = ("R-HLC", "R-TSB")
POOL_SIZE = 256
MINIMUM_SPEED_MPS = 5.0


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_digest(salt_sha256: str, family: str, scenario_token: str, log_id: str) -> str:
    return hashlib.sha256(
        f"{salt_sha256}|{family}|{scenario_token}|{log_id}".encode("utf-8")
    ).hexdigest()


def _candidate_rows(db_path: Path, source_partition: str) -> Iterable[Dict[str, Any]]:
    """Yield every non-edge, moving lidar anchor using SQLite read-only access."""
    query = """
        WITH ordered_scenes AS (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS scene_row_num FROM scene
        ), scene_count AS (SELECT COUNT(*) AS n FROM scene)
        SELECT lower(hex(lp.token)) AS scenario_token, lp.timestamp,
               l.logfile AS log_id, l.map_version AS map_name,
               sqrt(ep.vx * ep.vx + ep.vy * ep.vy) AS speed_mps
        FROM lidar_pc lp
        JOIN ego_pose ep ON ep.token = lp.ego_pose_token
        JOIN lidar ld ON ld.token = lp.lidar_token
        JOIN log l ON l.token = ld.log_token
        JOIN ordered_scenes os ON os.token = lp.scene_token
        CROSS JOIN scene_count sc
        WHERE os.scene_row_num >= 3 AND os.scene_row_num < sc.n - 1
          AND (ep.vx * ep.vx + ep.vy * ep.vy) >= ?
        ORDER BY lp.timestamp ASC
    """
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for token, timestamp, log_id, map_name, speed_mps in connection.execute(
            query, (MINIMUM_SPEED_MPS**2,)
        ):
            yield {
                "scenario_token": str(token),
                "timestamp": int(timestamp),
                "log_id": str(log_id),
                "map_name": str(map_name),
                "initial_speed_pre_treatment_mps": round(float(speed_mps), 6),
                "db_file": db_path.name,
                "db_path": str(db_path.resolve()),
                "source_partition": source_partition,
            }


def stable_candidate_pool(
    cache_root: Path,
    family: str,
    salt_sha256: str,
    forbidden_tokens: set[str],
    forbidden_logs: set[str],
    pool_size: int = POOL_SIZE,
) -> List[Dict[str, Any]]:
    """Keep the globally lowest deterministic ranks without materializing DB tensors."""
    # One scenario token has many lidar timestamps and therefore shares the
    # same rank digest.  Keep timestamp as an explicit deterministic tie-break
    # so heapq never attempts to compare the row dictionaries.
    heap: List[Tuple[int, str, int, str, int, Dict[str, Any]]] = []
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
                entry = (-rank_int, token, int(row["timestamp"]), str(row["db_file"]), enumeration_index, row)
                if len(heap) < pool_size:
                    heapq.heappush(heap, entry)
                elif rank_int < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
    best_by_token: Dict[str, Dict[str, Any]] = {}
    for _, _, _, _, _, row in heap:
        token = row["scenario_token"]
        current = best_by_token.get(token)
        if current is None or row["selector_rank_sha256"] < current["selector_rank_sha256"]:
            best_by_token[token] = row
    return sorted(best_by_token.values(), key=lambda row: (row["selector_rank_sha256"], row["scenario_token"], row["log_id"]))


def _official_initial(db_path: Path, token: str, anchor_timestamp: int) -> Dict[str, Any]:
    """Use nuPlan's official query helpers, never a custom pose reconstruction."""
    from tools.stage7l_build_lane_change_opportunity_inventory import official_simulation_initial_token, route_ids
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db

    initial_token, initial_timestamp, scenario_types = official_simulation_initial_token(
        db_path, token, anchor_timestamp
    )
    ego = get_ego_state_for_lidarpc_token_from_db(str(db_path), initial_token)
    if ego is None:
        raise ValueError("official initial ego state is unavailable")
    speed = float(ego.dynamic_car_state.speed)
    if speed < MINIMUM_SPEED_MPS:
        raise ValueError("official initial speed below frozen runtime-validation minimum")
    return {
        "official_simulation_initial_lidar_token": initial_token,
        "official_simulation_initial_timestamp_us": int(initial_timestamp),
        "official_scenario_types": sorted(str(value) for value in scenario_types),
        "initial_x": round(float(ego.rear_axle.x), 6),
        "initial_y": round(float(ego.rear_axle.y), 6),
        "initial_heading": round(float(ego.rear_axle.heading), 8),
        "initial_speed_mps": round(speed, 6),
        "initial_time_us": int(ego.time_us),
        "route_roadblock_ids": route_ids(db_path, token),
    }


def select_hlc(
    pool: Sequence[Mapping[str, Any]], map_root: Path, used_tokens: set[str], used_logs: set[str]
) -> List[Dict[str, Any]]:
    """Select two native-adjacent map opportunities from deterministic ranked rows."""
    from tools.stage7l_build_lane_change_opportunity_inventory import evaluate_candidate_options

    selected: List[Dict[str, Any]] = []
    for candidate in pool:
        if candidate["scenario_token"] in used_tokens or candidate["log_id"] in used_logs:
            continue
        db_path = Path(str(candidate["db_path"]))
        row = {
            "db_file": db_path.name,
            "log_name": candidate["log_id"],
            "scenario_token": candidate["scenario_token"],
            "timestamp": candidate["timestamp"],
            "map_name": candidate["map_name"],
            "db_scene_token": "",
        }
        try:
            options = evaluate_candidate_options(
                row=row,
                db_root=db_path.parent,
                map_root=map_root,
                map_version="nuplan-maps-v1.0",
                minimum_remaining_m=70.0,
                minimum_target_gap_m=8.0,
                minimum_speed_mps=MINIMUM_SPEED_MPS,
            )
        except Exception:
            continue
        eligible = [item for item in options if item.get("eligible")]
        if not eligible:
            continue
        eligible.sort(
            key=lambda item: (
                -float(item["paired_reference_remaining_m"]),
                -float(item["minimum_target_lane_object_gap_m"]),
                str(item["direction"]),
            )
        )
        item = eligible[0]
        selected.append(
            {
                "family": "R-HLC",
                "selection_features_scope": "PRE_TREATMENT_DB_MAP_AND_OFFICIAL_INITIAL_STATE_ONLY",
                "scenario_token": candidate["scenario_token"],
                "log_id": candidate["log_id"],
                "db_file": db_path.name,
                "db_path": str(db_path),
                "source_partition": candidate["source_partition"],
                "map_name": candidate["map_name"],
                "selector_rank_sha256": candidate["selector_rank_sha256"],
                "scenario_anchor_timestamp_us": int(candidate["timestamp"]),
                "pre_treatment_initial_speed_mps": candidate["initial_speed_pre_treatment_mps"],
                "initial_state": {
                    key: item[key]
                    for key in (
                        "official_simulation_initial_lidar_token",
                        "official_simulation_initial_timestamp_us",
                        "initial_x",
                        "initial_y",
                        "initial_heading",
                        "initial_speed_mps",
                        "initial_time_us",
                        "initial_state_fingerprint",
                    )
                },
                "route_roadblock_ids": json.loads(item["route_roadblock_ids_json"]),
                "route_fingerprint": item["route_fingerprint"],
                "source_lane_id": item["source_lane_id"],
                "target_lane_id": item["target_lane_id"],
                "source_roadblock_id": item["source_roadblock_id"],
                "target_roadblock_id": item["target_roadblock_id"],
                "direction": item["direction"],
                "source_start_arc_m": item["source_start_arc_m"],
                "target_start_arc_m": item["target_start_arc_m"],
                "source_reference_xy": json.loads(item["source_reference_xy_json"]),
                "target_reference_xy": json.loads(item["target_reference_xy_json"]),
                "pre_treatment_native_adjacency": True,
                "paired_reference_remaining_m": item["paired_reference_remaining_m"],
                "minimum_target_lane_object_gap_m": item["minimum_target_lane_object_gap_m"],
                "runtime_arm": "HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE",
                "isolation_labels": [
                    "RUNTIME_DETERMINISM_VALIDATION_ONLY",
                    "EXCLUDED_FROM_FRESH_TECHNICAL_SMOKE",
                    "EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER",
                    "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION",
                ],
            }
        )
        used_tokens.add(candidate["scenario_token"])
        used_logs.add(candidate["log_id"])
        if len(selected) == 2:
            return selected
    raise RuntimeError("could not obtain two deterministic HLC map-eligible validation scenarios")


def select_tsb(
    pool: Sequence[Mapping[str, Any]], used_tokens: set[str], used_logs: set[str]
) -> List[Dict[str, Any]]:
    """Select two speed-valid official baseline-braking scenarios from ranked rows."""
    selected: List[Dict[str, Any]] = []
    for candidate in pool:
        if candidate["scenario_token"] in used_tokens or candidate["log_id"] in used_logs:
            continue
        try:
            initial = _official_initial(
                Path(str(candidate["db_path"])), candidate["scenario_token"], int(candidate["timestamp"])
            )
        except Exception:
            continue
        if not initial["route_roadblock_ids"]:
            continue
        initial_fp = canonical_sha256(
            {
                "x": initial["initial_x"],
                "y": initial["initial_y"],
                "heading": initial["initial_heading"],
                "speed_mps": initial["initial_speed_mps"],
                "time_us": initial["initial_time_us"],
            }
        )
        selected.append(
            {
                "family": "R-TSB",
                "selection_features_scope": "PRE_TREATMENT_DB_ROUTE_AND_OFFICIAL_INITIAL_STATE_ONLY",
                "scenario_token": candidate["scenario_token"],
                "log_id": candidate["log_id"],
                "db_file": Path(str(candidate["db_path"])).name,
                "db_path": candidate["db_path"],
                "source_partition": candidate["source_partition"],
                "map_name": candidate["map_name"],
                "selector_rank_sha256": candidate["selector_rank_sha256"],
                "scenario_anchor_timestamp_us": int(candidate["timestamp"]),
                "pre_treatment_initial_speed_mps": candidate["initial_speed_pre_treatment_mps"],
                "initial_state": {**initial, "initial_state_fingerprint": initial_fp},
                "route_fingerprint": canonical_sha256(initial["route_roadblock_ids"]),
                "runtime_arm": "TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING",
                "isolation_labels": [
                    "RUNTIME_DETERMINISM_VALIDATION_ONLY",
                    "EXCLUDED_FROM_FRESH_TECHNICAL_SMOKE",
                    "EXCLUDED_FROM_R1_DEVELOPMENT_ROSTER",
                    "EXCLUDED_FROM_FUTURE_R4_CONFIRMATION",
                ],
            }
        )
        used_tokens.add(candidate["scenario_token"])
        used_logs.add(candidate["log_id"])
        if len(selected) == 2:
            return selected
    raise RuntimeError("could not obtain two deterministic TSB baseline validation scenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    required = (
        SELECTOR, SOURCE_UNIVERSE, BLACKLIST, HISTORICAL_SMOKE_ROSTER, FUTURE_R4,
        args.cache_root / "mini", args.cache_root / "train_pittsburgh", args.map_root,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required frozen selector input(s): {missing}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite runtime validation roster: {args.output}")
    selector = read_json(SELECTOR)
    if selector["status"] != "FROZEN_BEFORE_CANDIDATE_ENUMERATION":
        raise RuntimeError("runtime validation selector is not frozen before enumeration")
    source_universe = read_json(SOURCE_UNIVERSE)
    if source_universe["status"] != "READY_FOR_OUTCOME_BLIND_SELECTION":
        raise RuntimeError("fresh source universe is not ready for outcome-blind selection")
    blacklist = read_json(BLACKLIST)
    historical_smoke = read_json(HISTORICAL_SMOKE_ROSTER)
    future_r4 = read_json(FUTURE_R4)
    forbidden_tokens = {str(row["scenario_token"]).lower() for row in blacklist["entries"]}
    forbidden_logs = {str(row["log_id"]) for row in blacklist["entries"]}
    # This legacy technical-smoke roster must never become a validation input,
    # even if a future edit accidentally makes it diverge from the old12 ledger.
    for rows in historical_smoke.get("families", {}).values():
        for row in rows:
            forbidden_tokens.add(str(row["scenario_token"]).lower())
            forbidden_logs.add(str(row["log_id"]))
    if future_r4.get("identity_roster_frozen"):
        forbidden_tokens.update(str(row["scenario_token"]).lower() for row in future_r4.get("identities", []))
    salt = str(selector["salt_sha256"])
    hlc_pool = stable_candidate_pool(args.cache_root, "R-HLC", salt, forbidden_tokens, forbidden_logs)
    tsb_pool = stable_candidate_pool(args.cache_root, "R-TSB", salt, forbidden_tokens, forbidden_logs)
    used_tokens: set[str] = set()
    used_logs: set[str] = set()
    hlc = select_hlc(hlc_pool, args.map_root, used_tokens, used_logs)
    tsb = select_tsb(tsb_pool, used_tokens, used_logs)
    entries = hlc + tsb
    if len(entries) != 4 or len({row["scenario_token"] for row in entries}) != 4 or len({row["log_id"] for row in entries}) != 4:
        raise RuntimeError("runtime validation roster must contain four unique scenario tokens from four unique logs")
    roster = {
        "schema_version": "r1_runtime_determinism_validation_roster_v1.0",
        "status": "FROZEN_RUNTIME_DETERMINISM_VALIDATION_ONLY",
        "actual_fresh_smoke_roster_selected": False,
        "actual_r1_development_roster_selected": False,
        "future_r4_identity_roster_used": False,
        "selector_contract_sha256": sha256_file(SELECTOR),
        "source_universe_sha256": sha256_file(SOURCE_UNIVERSE),
        "old12_blacklist_sha256": sha256_file(BLACKLIST),
        "historical_technical_smoke_roster_sha256": sha256_file(HISTORICAL_SMOKE_ROSTER),
        "future_r4_freeze_sha256": sha256_file(FUTURE_R4),
        "selection_method": "full_pre_treatment_sqlite_stream_then_frozen_sha256_rank_then_pre_treatment_official_map_validation",
        "selection_inputs": [
            "scenario_token", "log_id", "DB timestamp", "DB speed", "official initial state",
            "official route roadblocks", "native map adjacency", "map reference length", "initial object clearance",
        ],
        "forbidden_inputs_not_opened": [
            "treatment outcome", "HLC/TSB mechanism success", "F_match result", "representation", "BDD", "probe", "checkpoint", "RBR",
        ],
        "entries": entries,
        "counts": {"R-HLC": len(hlc), "R-TSB": len(tsb), "total": len(entries), "unique_logs": len(used_logs)},
        "run_budget": {"unit": "OFFICIAL_CLOSED_LOOP_RUN", "planned": 8, "authorized_cap": 8, "per_scenario": 2},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(roster, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": roster["counts"], "roster_sha256": sha256_file(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
