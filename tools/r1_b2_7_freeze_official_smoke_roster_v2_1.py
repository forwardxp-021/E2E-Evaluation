#!/usr/bin/env python3
"""B2.7-R1 globally-deduplicated, lazy, outcome-blind roster freeze.

This is a zero-rollout entry point.  It scans frozen source metadata globally,
then evaluates frozen pre-treatment eligibility only in SHA rank order until
the fixed roster is mathematically determined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r1_b2_7_freeze_official_smoke_roster_v2 as attempt1
from tools import r1_future_compliant_smoke_selector_v1_1 as selector


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
SOURCE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
CONTRACT = R1 / "r1_future_compliant_smoke_selector_contract_v1.1.json"
DEFAULT_CACHE_ROOT = ROOT.parent / "nuplan/dataset/data/cache"
DEFAULT_MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
FAMILIES = ("R-HLC", "R-TSB")
OUTPUTS = {
    "blacklist": "r1_b2_7_effective_permanent_blacklist_audit_v1.0.json",
    "leakage": "r1_b2_7_selector_information_leakage_audit_v1.0.json",
    "dedup": "r1_b2_7_global_dedup_audit_v1.0.json",
    "eligibility": "r1_b2_7_family_eligibility_audit_v1.0.json",
    "summary": "r1_b2_7_enumeration_summary_v1.0.json",
    "roster": "r1_official_compliant_technical_smoke_roster_v2.0.json",
    "schedule": "r1_official_compliant_technical_smoke_schedule_v2.0.json",
    "preflight": "r1_b2_7_zero_run_roster_preflight_v1.0.json",
    "status": "r1_b2_7_status_v1.0.json",
}


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _db_inventory(cache_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for partition in ("mini", "train_pittsburgh"):
        for path in sorted((cache_root / partition).glob("*.db")):
            rows.append({"partition": partition, "db_path": str(path.resolve()), "db_file": path.name, "db_sha256": attempt1.sha256_file(path)})
    if len(rows) != 1624:
        raise RuntimeError(f"FROZEN_DB_INVENTORY_COUNT_MISMATCH:{len(rows)}")
    return rows


def _occurrence_count(path: Path) -> int:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        return int(connection.execute("SELECT COUNT(DISTINCT lidar_pc_token) FROM scenario_tag").fetchone()[0])


def _inventory_logs(path: Path) -> set[str]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        return {str(row[0]) for row in connection.execute("SELECT DISTINCT logfile FROM log")}


def _metadata_hash(row: Mapping[str, Any]) -> str:
    immutable = {
        "scenario_token": str(row["scenario_token"]), "timestamp": int(row["timestamp"]),
        "log_id": str(row["log_id"]), "map_name": str(row["map_name"]),
        "pre_initial_speed_mps": float(row["pre_initial_speed_mps"]),
    }
    return hashlib.sha256(json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _create_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        CREATE TABLE candidates (
            scenario_token TEXT PRIMARY KEY, log_id TEXT NOT NULL, map_name TEXT NOT NULL,
            timestamp INTEGER NOT NULL, pre_initial_speed_mps REAL NOT NULL,
            db_path TEXT NOT NULL, db_file TEXT NOT NULL, source_partition TEXT NOT NULL,
            db_sha256 TEXT NOT NULL, metadata_sha256 TEXT NOT NULL, conflict INTEGER NOT NULL DEFAULT 0,
            rank_hlc TEXT NOT NULL, rank_tsb TEXT NOT NULL
        );
        CREATE TABLE blacklist_token (value TEXT PRIMARY KEY);
        CREATE TABLE blacklist_log (value TEXT PRIMARY KEY);
    """)
    return connection


def global_source_preflight(cache_root: Path, salt: str, blacklist: Mapping[str, Any], store_path: Path, reuse_existing_store: bool) -> tuple[sqlite3.Connection, Dict[str, Any]]:
    """Globally deduplicate source identities before any costly eligibility call."""
    inventory = _db_inventory(cache_root)
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        by_sha[row["db_sha256"]].append(row)
    canonical_paths = {rows[0]["db_path"] for rows in by_sha.values()}
    duplicate_groups = []
    for digest, rows in sorted(by_sha.items()):
        if len(rows) > 1:
            counts = [_occurrence_count(Path(row["db_path"])) for row in rows]
            if len(set(counts)) != 1:
                raise RuntimeError(f"BYTE_IDENTICAL_DB_OCCURRENCE_COUNT_MISMATCH:{digest}")
            duplicate_groups.append({
                "sha256": digest, "canonical_representative": rows[0], "duplicate_occurrences": rows[1:],
                "token_occurrence_count_per_db": counts[0], "collapsed_duplicate_token_occurrences": sum(counts[1:]),
                "canonicalization_rule": "FROZEN_INVENTORY_ORDER_FIRST_OCCURRENCE",
            })
    occurrence_count = sum(_occurrence_count(Path(row["db_path"])) for row in inventory)
    inventory_logs = {log_id for row in inventory for log_id in _inventory_logs(Path(row["db_path"]))}
    if reuse_existing_store:
        if not store_path.exists():
            raise FileNotFoundError(f"REQUESTED_SOURCE_METADATA_STORE_REUSE_MISSING:{store_path}")
        connection = sqlite3.connect(store_path)
        expected_columns = {"scenario_token", "log_id", "metadata_sha256", "conflict", "rank_hlc", "rank_tsb"}
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(candidates)")}
        if not expected_columns <= columns:
            raise RuntimeError("SOURCE_METADATA_STORE_SCHEMA_INVALID_FOR_REUSE")
    else:
        connection = _create_store(store_path)
    insert_sql = """
        INSERT INTO candidates (scenario_token,log_id,map_name,timestamp,pre_initial_speed_mps,db_path,db_file,source_partition,db_sha256,metadata_sha256,rank_hlc,rank_tsb)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(scenario_token) DO UPDATE SET
          conflict=CASE WHEN candidates.metadata_sha256=excluded.metadata_sha256 THEN candidates.conflict ELSE 1 END
    """
    if not reuse_existing_store:
        for index, db in enumerate(inventory, 1):
            if db["db_path"] not in canonical_paths:
                continue
            batch = []
            for row in attempt1.source_rows(Path(db["db_path"]), db["partition"]):
                token, log_id = str(row["scenario_token"]), str(row["log_id"])
                batch.append((token, log_id, str(row["map_name"]), int(row["timestamp"]), float(row["pre_initial_speed_mps"]), db["db_path"], db["db_file"], db["partition"], db["db_sha256"], _metadata_hash(row), selector.rank_digest(salt, "R-HLC", token, log_id), selector.rank_digest(salt, "R-TSB", token, log_id)))
            connection.executemany(insert_sql, batch)
            if index % 100 == 0:
                print(json.dumps({"progress": "B2_7_R1_GLOBAL_SOURCE_PREFLIGHT", "scanned_db_count": index}, ensure_ascii=False), flush=True)
        connection.commit()
    conflicts = int(connection.execute("SELECT COUNT(*) FROM candidates WHERE conflict=1").fetchone()[0])
    unique_tokens = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    token_bearing_logs = int(connection.execute("SELECT COUNT(DISTINCT log_id) FROM candidates").fetchone()[0])
    unique_logs = len(inventory_logs)
    duplicate_occurrences = occurrence_count - unique_tokens
    source = _read_json(SOURCE)
    expected_tokens = int(source["unfiltered_universe"]["unique_scenario_token_count"])
    expected_logs = int(source["unfiltered_universe"]["unique_log_count"])
    if conflicts:
        raise RuntimeError(f"CONFLICTING_DUPLICATE_IDENTITY_FAIL_CLOSED:{conflicts}")
    if (occurrence_count, duplicate_occurrences, unique_tokens, unique_logs) != (5405672, 19097, expected_tokens, expected_logs):
        raise RuntimeError(f"FROZEN_GLOBAL_SOURCE_PREFLIGHT_MISMATCH:{occurrence_count}:{duplicate_occurrences}:{unique_tokens}:{unique_logs}")
    tokens = [(str(row["scenario_token"]),) for row in blacklist["entries"]]
    logs = [(str(row["log_id"]),) for row in blacklist["entries"]]
    connection.executemany("INSERT OR IGNORE INTO blacklist_token VALUES (?)", tokens)
    connection.executemany("INSERT OR IGNORE INTO blacklist_log VALUES (?)", logs)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_candidate_hlc_rank ON candidates(rank_hlc, scenario_token, log_id, timestamp)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_candidate_tsb_rank ON candidates(rank_tsb, scenario_token, log_id, timestamp)")
    connection.commit()
    post_blacklist = int(connection.execute("SELECT COUNT(*) FROM candidates c LEFT JOIN blacklist_token t ON c.scenario_token=t.value LEFT JOIN blacklist_log l ON c.log_id=l.value WHERE t.value IS NULL AND l.value IS NULL").fetchone()[0])
    audit = {
        "schema_version": "r1_b2_7_global_dedup_audit_v1.0", "status": "PASS_GLOBAL_UNIQUE_SOURCE_PREFLIGHT",
        "identity_unit": "GLOBAL_UNIQUE_SCENARIO_TOKEN", "occurrence_token_count": occurrence_count,
        "duplicate_token_occurrence_count": duplicate_occurrences, "global_unique_scenario_token_count": unique_tokens,
        "global_unique_log_count": unique_logs, "token_bearing_candidate_log_count": token_bearing_logs,
        "closure": "5405672 - 19097 = 5386575",
        "duplicate_db_groups": duplicate_groups, "conflicting_duplicate_identity_count": conflicts,
        "post_blacklist_candidate_count": post_blacklist,
        "source_contract_path": str(SOURCE.relative_to(ROOT)), "source_contract_sha256": attempt1.sha256_file(SOURCE),
        "source_root_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"],
        "source_metadata_store_reused_after_pre_eligibility_fail_closed_correction": reuse_existing_store,
    }
    return connection, audit


def _ranked_rows(connection: sqlite3.Connection, family: str) -> Iterator[selector.RankedIdentity]:
    rank_column = "rank_hlc" if family == "R-HLC" else "rank_tsb"
    query = f"""SELECT c.scenario_token,c.log_id,c.map_name,c.timestamp,c.pre_initial_speed_mps,c.db_path,c.db_file,c.source_partition,c.db_sha256,c.{rank_column}
        FROM candidates c LEFT JOIN blacklist_token t ON c.scenario_token=t.value
        LEFT JOIN blacklist_log l ON c.log_id=l.value WHERE t.value IS NULL AND l.value IS NULL
        ORDER BY c.{rank_column},c.scenario_token,c.log_id,c.timestamp"""
    for row in connection.execute(query):
        token, log_id, map_name, timestamp, speed, db_path, db_file, partition, db_sha, rank = row
        payload = {"scenario_token": str(token), "log_id": str(log_id), "map_name": str(map_name), "timestamp": int(timestamp), "pre_initial_speed_mps": float(speed), "db_path": str(db_path), "db_file": str(db_file), "source_partition": str(partition), "db_sha256": str(db_sha)}
        yield selector.RankedIdentity(family, str(token), str(log_id), int(timestamp), str(rank), payload)


def _lazy_family(connection: sqlite3.Connection, family: str, map_root: Path, used_tokens: set[str], used_logs: set[str]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    map_cache: Dict[str, Any] = {}
    def eligible(candidate: selector.RankedIdentity) -> tuple[bool, Mapping[str, Any]]:
        try:
            entry, audit = attempt1.evaluate_candidate(candidate.payload, family, map_root, map_cache)
            entry["selector_rank_sha256"] = candidate.rank_sha256
            audit["selector_rank_sha256"] = candidate.rank_sha256
            return True, {"entry": entry, "audit": audit}
        except attempt1.EligibilityError as exc:
            return False, {"failure_reason": str(exc)}
    selected, accounting = selector.lazy_rank_ordered_select(_ranked_rows(connection, family), eligible, 12, used_tokens, used_logs)
    if len(selected) != 12:
        raise RuntimeError(f"INSUFFICIENT_ELIGIBLE_FRESH_IDENTITIES:{family}:PASS={len(selected)}")
    return [dict(value["entry"]) for _, value in selected], [dict(value["audit"]) for _, value in selected], accounting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=R1)
    parser.add_argument("--reuse-pre-eligibility-source-store", action="store_true")
    args = parser.parse_args()
    targets = {key: args.output_dir / value for key, value in OUTPUTS.items()}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"B2_7_R1_OUTPUT_ALREADY_EXISTS:{existing}")
    contract = _read_json(CONTRACT)
    if contract.get("status") != "FROZEN_FOR_B2_7_R1_ONE_TIME_RERUN":
        raise RuntimeError("B2_7_R1_CONTRACT_NOT_FROZEN")
    if attempt1.sha256_file(ROOT / "tools/r1_future_compliant_smoke_selector_v1_1.py") != contract["corrected_selector_implementation_sha256"]:
        raise RuntimeError("CORRECTED_SELECTOR_IMPLEMENTATION_SHA_MISMATCH")
    if contract["master_seed"] != 2026082701 or contract["salt_sha256"] != "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9":
        raise RuntimeError("FROZEN_SEED_OR_SALT_MISMATCH")
    blacklist = attempt1.resolve_effective_blacklist()
    ephemeral_store = args.output_dir / ".r1_b2_7_r1_global_source.sqlite"
    if ephemeral_store.exists() and not args.reuse_pre_eligibility_source_store:
        raise FileExistsError(f"STALE_B2_7_R1_EPHEMERAL_STORE_REQUIRES_MANUAL_REVIEW:{ephemeral_store}")
    connection, dedup = global_source_preflight(args.cache_root, contract["salt_sha256"], blacklist, ephemeral_store, args.reuse_pre_eligibility_source_store)
    # The preflight store is intentionally ephemeral; retain only audited JSON summaries.
    used_tokens: set[str] = set()
    used_logs: set[str] = set()
    selections: Dict[str, list[Dict[str, Any]]] = {}
    audits: Dict[str, list[Dict[str, Any]]] = {}
    accounting: Dict[str, Dict[str, Any]] = {}
    try:
        for family in FAMILIES:
            selections[family], audits[family], accounting[family] = _lazy_family(connection, family, args.map_root, used_tokens, used_logs)
    finally:
        connection.close()
    if not ephemeral_store.exists():
        raise RuntimeError("B2_7_R1_EPHEMERAL_STORE_MISSING_BEFORE_CLEANUP")
    ephemeral_store.unlink()
    entries = selections["R-HLC"] + selections["R-TSB"]
    if len(entries) != 24 or len({row["scenario_token"] for row in entries}) != 24 or len({row["log_id"] for row in entries}) != 24:
        raise RuntimeError("ROSTER_UNIQUENESS_FAIL_NO_MANUAL_REPLACEMENT")
    source = _read_json(SOURCE)
    selector_sha = attempt1.sha256_file(CONTRACT)
    roster = {"schema_version": "r1_official_compliant_technical_smoke_roster_v2.0", "status": "FROZEN_PENDING_SMOKE_AUTHORIZATION", "selector_contract_path": str(CONTRACT.relative_to(ROOT)), "selector_contract_sha256": selector_sha, "selector_implementation_sha256": contract["corrected_selector_implementation_sha256"], "source_universe_path": str(SOURCE.relative_to(ROOT)), "source_universe_sha256": attempt1.sha256_file(SOURCE), "effective_blacklist_sha256": attempt1.canonical_sha256(blacklist), "db_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"], "map_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"], "master_seed": 2026082701, "salt_sha256": contract["salt_sha256"], "entries": entries, "counts": {"R-HLC": 12, "R-TSB": 12, "total": 24, "unique_scenario_tokens": 24, "unique_logs": 24}, "identity_replacement_allowed": False, "manual_identity_replacement_performed": False, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0}
    roster_sha = attempt1.artifact_sha256(roster)
    bindings = {"OFFICIAL_SMOKE_PLANNER_V2_1": attempt1.sha256_file(ROOT / "tools/r1_official_technical_smoke_planner_v2_1.py"), "OFFICIAL_SMOKE_EVALUATOR_V2_1": attempt1.sha256_file(ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py"), "ABSOLUTE_EPISODE_CLOCK_BINDING": attempt1.sha256_file(R1 / "r1_absolute_episode_clock_binding_v1.0.json"), "HLC_REALIZED_PROGRESS_V1": attempt1.sha256_file(R1 / "r1_hlc_realized_progress_contract_v1.0.json"), "HLC_TERMINAL_ROUTE_PROGRESS_V1": attempt1.sha256_file(R1 / "r1_hlc_terminal_route_progress_contract_v1.0.json"), "CONTEXT_V2_1": attempt1.sha256_file(ROOT / "tools/r1_closed_loop_context_adapter_v2_1.py"), "HLC_CLEARANCE_V1_1": attempt1.sha256_file(ROOT / "tools/r1_hlc_dynamic_clearance_v1_1.py"), "HLC_APPLICABILITY_V1_0": attempt1.sha256_file(R1 / "r1_hlc_map_geometry_applicability_contract_v1.0.json"), "TSB_APPLICABILITY_V1_0": attempt1.sha256_file(R1 / "r1_tsb_mechanism_applicability_contract_v1.0.json"), "OFFICIAL_MAP_BRIDGE": attempt1.sha256_file(ROOT / "tools/r1_official_map_query_bridge_v2_1.py"), "OFFICIAL_EGO_FOOTPRINT": attempt1.sha256_file(ROOT / "tools/r1_official_ego_vehicle_binding_v1.py"), "B2_6_FINAL_EXECUTION_CONFORMANCE_MANIFEST": attempt1.sha256_file(R1 / "r1_b2_6_final_execution_conformance_sha_manifest_v1.0.json")}
    schedule = attempt1.build_schedule(entries, bindings, roster_sha)
    preflight = attempt1.zero_run_preflight(entries, schedule, args.map_root)
    eligibility = {"schema_version": "r1_b2_7_family_eligibility_audit_v1.0", "status": "PASS_FROZEN_PRETREATMENT_ONLY_LAZY_RANK_ORDERED_ELIGIBILITY", "execution_mode": "LAZY_RANK_ORDERED_ELIGIBILITY", "TOTAL_ELIGIBLE_COUNT": "NOT_EXHAUSTIVELY_COUNTED_BY_DESIGN", "order": ["GLOBAL_UNIQUE_SOURCE_UNIVERSE", "PERMANENT_BLACKLIST", "FROZEN_SHA_RANKING", "RANK_ORDERED_FROZEN_FAMILY_ELIGIBILITY", "TOP_12_CLOSURE"], "audited_selected_candidates": audits, "accounting": accounting, "threshold_changed": False, "manual_identity_replacement": False, "outcome_input_used": False}
    leakage = {"schema_version": "r1_b2_7_selector_information_leakage_audit_v1.0", "status": "PASS_NO_OUTCOME_INPUTS", "allowed_inputs": contract["selection_inputs_allowlist"], "forbidden_inputs_not_read": attempt1.FORBIDDEN_TERMS, "historical_outcome_file_open_count": 0, "future_outcome_available": False, "representation_bdd_probe_checkpoint_rbr_open_count": 0, "simulation_launched": False}
    summary = {"schema_version": "r1_b2_7_enumeration_summary_v1.0", "status": "B2_7_ENUMERATION_ATTEMPT_2_COMPLETE_AUTHORIZED_ONCE", "source_universe": dedup, "effective_blacklist": blacklist["counts"], "execution_mode": "LAZY_RANK_ORDERED_ELIGIBILITY", "family": {family: {"selected_count": len(selections[family]), **accounting[family], "TOTAL_ELIGIBLE_COUNT": "NOT_EXHAUSTIVELY_COUNTED_BY_DESIGN"} for family in FAMILIES}, "roster": {"sha256": roster_sha, "unique_tokens": 24, "unique_logs": 24}, "schedule": {"sha256": attempt1.artifact_sha256(schedule), "runs": 48, "pairs": 24}, "manual_identity_replacement": False, "threshold_changed": False, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0}
    status = {"schema_version": "r1_b2_7_status_v1.0", "B2_7_SELECTOR_DEDUP_CORRECTION": "COMPLETE", "B2_7_ENUMERATION_ATTEMPT_2": "COMPLETE_AUTHORIZED_ONCE", "FRESH_ROSTER": "FROZEN_PENDING_SMOKE_AUTHORIZATION", "OFFICIAL_SMOKE": "NOT_AUTHORIZED", "NEW_RUN_BUDGET": 0, "RBR_A": "NOT_AUTHORIZED", "RBR_B": "NOT_AUTHORIZED", "RBR_C": "NOT_AUTHORIZED", "simulation_launched": False, "run_simulation_called": False}
    for key, value in (("blacklist", blacklist), ("leakage", leakage), ("dedup", dedup), ("eligibility", eligibility), ("summary", summary), ("roster", roster), ("schedule", schedule), ("preflight", preflight), ("status", status)):
        attempt1.write_new(targets[key], value)
    print(json.dumps({"status": status["B2_7_ENUMERATION_ATTEMPT_2"], "roster_sha256": attempt1.sha256_file(targets["roster"]), "schedule_sha256": attempt1.sha256_file(targets["schedule"]), "preflight": preflight["status"], "simulation_launched": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
