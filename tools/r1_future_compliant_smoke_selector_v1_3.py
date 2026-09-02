#!/usr/bin/env python3
"""Outcome-blind B2.9-D selector and prospective roster/schedule freezer.

The implementation reuses the frozen v0.1 source universe and the completed
v2.1 rank prefix.  It only enumerates the rank suffix needed to replace an
identity made permanently ineligible after v2.1.  It never imports a
SimulationRunner and never calls planner or simulator rollout methods.
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
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r1_b2_7_freeze_official_smoke_roster_v2 as frozen  # noqa: E402
from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_b2_9_b_route_continuous_canary import _ego, _map_api  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1  # noqa: E402
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3  # noqa: E402
from tools.r1_prospective_generator_contract_v2 import (  # noqa: E402
    HLC_BASELINE,
    HLC_TREATMENT,
)


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
CACHE = ROOT.parent / "nuplan/dataset/data/cache"
MAPS = ROOT.parent / "nuplan/dataset/maps"
SOURCE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
OLD_CONTRACT = R1 / "r1_future_compliant_smoke_selector_contract_v1.2.json"
OLD_ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json"
OLD_SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v2.1.json"
OLD_ELIGIBILITY = R1 / "r1_b2_7_family_eligibility_audit_v1.0.json"
OLD_INCREMENTAL_AUDIT = R1 / "r1_b2_8_r3_roster_schedule_delta_audit_v1.0.json"
SOURCE_SUMMARY = R1 / "r1_b2_7_enumeration_summary_v1.0.json"
RUNTIME_CANDIDATE = R1 / "r1_b2_9_c_scientific_runtime_candidate_manifest_v1.0.json"
ROUTE_AUDIT_C = R1 / "r1_b2_9_c_route_progression_invariant_audit_v1.json"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
MASTER_SEED = 2026082701
SALT = "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9"
FAMILIES = ("R-HLC", "R-TSB")
TAIL_CAPACITY = 15000

OUTPUTS = {
    "exclusion": R1 / "r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json",
    "contract": R1 / "r1_future_compliant_smoke_selector_contract_v1.3.json",
    "audit": R1 / "r1_b2_9_d_selector_eligibility_audit_v1.0.json",
    "roster": R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json",
    "comparison": R1 / "r1_b2_9_d_roster_v2_1_to_v3_0_comparison_v1.json",
    "schedule": R1 / "r1_official_compliant_technical_smoke_schedule_v3.0.json",
}

EXCLUSION_SOURCES = (
    R1 / "r1_b2_7_effective_permanent_blacklist_audit_v1.0.json",
    R1 / "r1_official_execution_ineligible_identity_ledger_v1.0.json",
    R1 / "r1_b2_9_b_engineering_canary_exclusion_ledger_v1.0.json",
    R1 / "r1_b2_9_c_cross_family_engineering_canary_roster_v1.0.json",
)
ATTEMPT1_TOKEN = "b1be12bca092597a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def rank_digest(family: str, token: str, log_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{family}|{token}|{log_id}".encode("utf-8")).hexdigest()


def effective_exclusion_ledger() -> Dict[str, Any]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    source_bindings = []
    for path in EXCLUSION_SOURCES:
        payload = read_json(path)
        source_bindings.append(
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "entry_count": len(payload["entries"])}
        )
        for row in payload["entries"]:
            key = (str(row["scenario_token"]).lower(), str(row["log_id"]))
            target = merged.setdefault(
                key,
                {
                    "scenario_token": key[0],
                    "log_id": key[1],
                    "source_ledgers": [],
                    "reasons": [],
                    "PERMANENT_FUTURE_SELECTOR_EXCLUSION": True,
                    "OFFICIAL_ATTEMPT_CONSUMED": key[0] == ATTEMPT1_TOKEN,
                },
            )
            source_name = str(path.relative_to(ROOT))
            if source_name not in target["source_ledgers"]:
                target["source_ledgers"].append(source_name)
            reason = str(
                row.get("reason")
                or row.get("canary_source_ledger")
                or ("B2_9_C_PERMANENT_ENGINEERING_CANARY" if row.get("PERMANENT_FUTURE_SELECTOR_EXCLUSION") else payload.get("status"))
            )
            if reason not in target["reasons"]:
                target["reasons"].append(reason)
    if ATTEMPT1_TOKEN not in {key[0] for key in merged}:
        raise RuntimeError("ATTEMPT1_CONSUMED_IDENTITY_MISSING_FROM_EFFECTIVE_EXCLUSION")
    entries = sorted(merged.values(), key=lambda row: (row["scenario_token"], row["log_id"]))
    for row in entries:
        row["source_ledgers"].sort()
        row["reasons"].sort()
        if row["OFFICIAL_ATTEMPT_CONSUMED"]:
            row["reasons"].append("OFFICIAL_ATTEMPT_CONSUMED_PERMANENTLY")
    if len(entries) != 45 or len({row["scenario_token"] for row in entries}) != 45 or len({row["log_id"] for row in entries}) != 45:
        raise RuntimeError("EFFECTIVE_EXCLUSION_EXPECTED_45_UNIQUE_IDENTITIES")
    return {
        "schema_version": "r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0",
        "status": "FROZEN_ADDITIVE_PERMANENT_EXCLUSION_CLOSURE",
        "match_rule": "EXCLUDE_IF_SCENARIO_TOKEN_OR_LOG_ID_MATCHES",
        "sources": source_bindings,
        "entries": entries,
        "counts": {"entries": 45, "unique_scenario_tokens": 45, "unique_logs": 45},
        "attempt1": {
            "scenario_token": ATTEMPT1_TOKEN,
            "OFFICIAL_ATTEMPT_CONSUMED": True,
            "future_scientific_selection_allowed": False,
            "history_modified": False,
        },
        "entry_removal_reduction_or_replacement_allowed": False,
    }


def selector_contract(exclusion: Mapping[str, Any]) -> Dict[str, Any]:
    old = read_json(OLD_CONTRACT)
    if old["master_seed"] != MASTER_SEED or old["salt_sha256"] != SALT:
        raise RuntimeError("FROZEN_SEED_OR_SALT_MISMATCH")
    return {
        "schema_version": "r1_future_compliant_smoke_selector_contract_v1.3",
        "status": "FROZEN_PROSPECTIVE_PRIMARY80_ROUTE_CONTINUOUS_SELECTION",
        "inherits": {"path": str(OLD_CONTRACT.relative_to(ROOT)), "sha256": sha256(OLD_CONTRACT)},
        "master_seed": old["master_seed"],
        "salt_sha256": old["salt_sha256"],
        "salt_regeneration_allowed": False,
        "rank_rule": old["rank_rule"],
        "sort_and_tie_break": old["sort_and_tie_break"],
        "source_universe": old["source_universe"],
        "selection_inputs_allowlist": old["selection_inputs_allowlist"],
        "selection_scope": old["selection_scope"],
        "effective_exclusion_ledger": {
            "path": str(OUTPUTS["exclusion"].relative_to(ROOT)),
            "canonical_sha256": canonical_sha(exclusion),
            "match_rule": exclusion["match_rule"],
        },
        "incremental_eligibility_only": {
            "all_existing_v1_2_gates_preserved": True,
            "global": [
                "OFFICIAL_EXACT_SINGLE_SCENARIO_RESOLUTION_UNDER_BOUND_NUPLAN_1_2_2",
                "OFFICIAL_SCENARIO_CONTROLLER_ITERATIONS_AT_LEAST_81_NO_PADDING_NO_EXTENSION",
            ],
            "R-HLC": [
                "V2_3_ROLLING_REPLAN_PRIMARY_0_TO_79_ROUTE_CONTINUOUS_NATIVE_COVERAGE",
                "SOURCE_TARGET_SHARED_FROZEN_ROUTE_PROGRESSION",
                "TARGET_ROUTE_CONSISTENCY_INVARIANT",
            ],
            "R-TSB": "EXISTING_FROZEN_ELIGIBILITY_UNCHANGED_EXCEPT_GLOBAL_PRIMARY80_GATES",
        },
        "unchanged": [
            "family_quotas", "context_gates", "mechanism_independent_applicability_gates",
            "scientific_thresholds", "HLC_mechanism", "TSB_mechanism", "F_match",
            "endpoint", "engineering_limits", "safety_thresholds", "outcome_blindness",
            "safety_blindness",
        ],
        "manual_identity_selection_or_replacement_allowed": False,
        "OFFICIAL_SMOKE_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
        "RBR_A_B_C_AUTHORIZED": False,
    }


def _route_continuous_primary80_audit(entry: Mapping[str, Any], map_cache: MutableMapping[str, Any]) -> Dict[str, Any]:
    api = _map_api(str(entry["map_name"]), map_cache)
    arm_rows = []
    selected_transitions = 0
    for arm in (HLC_BASELINE, HLC_TREATMENT):
        initial = _ego(entry["initial_state"])
        first = build_hlc_route_continuous_reference_v2_3(
            api,
            entry["route_roadblock_ids"],
            entry["source_lane_id"],
            entry["target_lane_id"],
            initial,
            float(initial["speed_mps"]) * 7.9,
        )
        states = build_hlc_native_geometry_v1_1(
            initial,
            0.0,
            first["source_reference_xy"],
            first["target_reference_xy"],
            first["source_current_arc_m"],
            first["target_current_arc_m"],
            arm,
        )
        if len(states) != 80:
            raise ValueError("HLC_NOMINAL_PRIMARY_NOT_EXACT_80")
        arm_transitions = 0
        for index, state in enumerate(states):
            corridor = build_hlc_route_continuous_reference_v2_3(
                api,
                entry["route_roadblock_ids"],
                entry["source_lane_id"],
                entry["target_lane_id"],
                state,
                float(state["speed_mps"]) * 7.9,
            )
            invariant = corridor.get("route_progression_invariant", {})
            if invariant.get("status") != "TARGET_AND_SOURCE_EXACT_SAME_FROZEN_ROUTE_PROGRESSION":
                raise ValueError(f"ROUTE_PROGRESSION_INVARIANT_FAIL:{index}")
            arm_transitions += len(corridor["transitions"])
        selected_transitions += arm_transitions
        arm_rows.append({"arm": arm, "iterations": 80, "selected_transitions": arm_transitions, "status": "PASS"})
    return {
        "status": "PASS",
        "nominal_primary_iterations": list(range(80)),
        "arms": arm_rows,
        "rolling_calls": 160,
        "selected_transitions": selected_transitions,
        "target_route_consistency_violations": 0,
        "native_only": True,
        "extrapolation_used": False,
        "manual_points_used": False,
    }


def _existing_selected_prefix(
    exclusion: Mapping[str, Any], map_cache: MutableMapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    forbidden_tokens = {row["scenario_token"] for row in exclusion["entries"]}
    forbidden_logs = {row["log_id"] for row in exclusion["entries"]}
    old_entries = read_json(OLD_ROSTER)["entries"]
    selected: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    removed = []
    for row in old_entries:
        token, log_id = str(row["scenario_token"]), str(row["log_id"])
        if token in forbidden_tokens or log_id in forbidden_logs:
            removed.append({"family": row["family"], "scenario_token": token, "log_id": log_id, "reason": "EFFECTIVE_PERMANENT_EXCLUSION"})
            continue
        resolution = official_count(str(row["db_path"]), token)
        if resolution != 1:
            removed.append({"family": row["family"], "scenario_token": token, "log_id": log_id, "reason": f"EXACT_RESOLUTION_{resolution}"})
            continue
        copied = dict(row)
        copied["official_exact_single_scenario_resolution"] = "PASS"
        copied["Primary80_execution_eligibility"] = "PASS_PENDING_EXACT_CONTROLLER_CONFIRMATION_IN_ZERO_RUN"
        if row["family"] == "R-HLC":
            route = _route_continuous_primary80_audit(row, map_cache)
            copied["HLC_route_continuous_Primary80_applicability"] = "PASS"
            audits.append({"family": row["family"], "scenario_token": token, "selector_rank_sha256": row["selector_rank_sha256"], "existing_v1_2_gates": "PASS_INHERITED_FROM_V2_1", "official_resolution_count": resolution, "route_continuous": route})
        else:
            copied["TSB_frozen_eligibility_semantics"] = "UNCHANGED_PASS_INHERITED_FROM_V2_1"
            audits.append({"family": row["family"], "scenario_token": token, "selector_rank_sha256": row["selector_rank_sha256"], "existing_v1_2_gates": "PASS_INHERITED_FROM_V2_1", "official_resolution_count": resolution, "route_continuous": "NOT_APPLICABLE"})
        selected.append(copied)
    return selected, audits, {"removed": removed, "source": "FROZEN_V2_1_SELECTED_PREFIX_WITH_ADDITIVE_GATES_ONLY"}


def _duplicate_db_paths() -> set:
    summary = read_json(SOURCE_SUMMARY)
    return {
        str(row["db_path"])
        for group in summary["source_universe"]["duplicate_db_groups"]
        for row in group["duplicate_occurrences"]
    }


def _tail_candidates(
    family: str,
    cutoff: str,
    exclusion: Mapping[str, Any],
    used_tokens: set,
    used_logs: set,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    forbidden_tokens = {row["scenario_token"] for row in exclusion["entries"]}
    forbidden_logs = {row["log_id"] for row in exclusion["entries"]}
    skip = _duplicate_db_paths()
    heap: List[Tuple[Any, ...]] = []
    query = """SELECT DISTINCT lower(hex(st.lidar_pc_token)),lp.timestamp,l.logfile,l.map_version
               FROM scenario_tag st JOIN lidar_pc lp ON lp.token=st.lidar_pc_token
               JOIN scene s ON s.token=lp.scene_token JOIN log l ON l.token=s.log_token"""
    scanned_db_count = 0
    source_rows_seen = 0
    for partition in ("mini", "train_pittsburgh"):
        for path in sorted((CACHE / partition).glob("*.db")):
            if str(path.resolve()) in skip:
                continue
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
                connection.execute("PRAGMA query_only=ON")
                for token, timestamp, log_id, map_name in connection.execute(query):
                    source_rows_seen += 1
                    token, log_id = str(token), str(log_id)
                    if token in forbidden_tokens or log_id in forbidden_logs or token in used_tokens or log_id in used_logs:
                        continue
                    rank = rank_digest(family, token, log_id)
                    if rank <= cutoff:
                        continue
                    item = (
                        -int(rank, 16), rank, token, log_id, int(timestamp), str(map_name),
                        str(path.resolve()), path.name, partition,
                    )
                    if len(heap) < TAIL_CAPACITY:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
            scanned_db_count += 1
            if scanned_db_count % 200 == 0:
                print(json.dumps({"progress": "B2_9_D_FROZEN_UNIVERSE_RANK_SUFFIX", "scanned_db_count": scanned_db_count}), flush=True)
    source = read_json(SOURCE)
    expected = int(source["unfiltered_universe"]["unique_scenario_token_count"])
    if scanned_db_count != 1621 or source_rows_seen != expected:
        raise RuntimeError(f"FROZEN_SOURCE_UNIVERSE_REUSE_MISMATCH:{scanned_db_count}:{source_rows_seen}:{expected}")
    ordered = sorted(heap, key=lambda row: (row[1], row[2], row[3], row[4]))
    rows = [
        {
            "selector_rank_sha256": row[1], "scenario_token": row[2], "log_id": row[3],
            "timestamp": row[4], "map_name": row[5], "db_path": row[6], "db_file": row[7],
            "source_partition": row[8],
        }
        for row in ordered
    ]
    return rows, {
        "source_universe_reused": True,
        "family": family,
        "source_universe_path": str(SOURCE.relative_to(ROOT)),
        "source_universe_sha256": sha256(SOURCE),
        "raw_source_universe_replacement": False,
        "rank_suffix_metadata_recomputed_from_bound_read_only_DB": True,
        "canonical_db_count": scanned_db_count,
        "global_unique_source_rows_seen": source_rows_seen,
        "tail_capacity": TAIL_CAPACITY,
        "cutoff_exclusive": cutoff,
    }


def _hydrate_candidate(row: Mapping[str, Any]) -> Dict[str, Any]:
    query = """SELECT iep.vx,iep.vy FROM scenario_tag st
               JOIN lidar_pc lp ON lp.token=st.lidar_pc_token
               JOIN lidar_pc ilp ON ilp.timestamp=(SELECT MIN(lp2.timestamp) FROM lidar_pc lp2 WHERE lp2.timestamp>=lp.timestamp-3000000)
               JOIN ego_pose iep ON iep.token=ilp.ego_pose_token
               WHERE lower(hex(st.lidar_pc_token))=? LIMIT 1"""
    with sqlite3.connect(f"file:{row['db_path']}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        values = connection.execute(query, (row["scenario_token"],)).fetchone()
    if values is None:
        raise RuntimeError(f"FROZEN_CANDIDATE_METADATA_MISSING:{row['scenario_token']}")
    return {
        "scenario_token": row["scenario_token"], "log_id": row["log_id"], "map_name": row["map_name"],
        "timestamp": row["timestamp"], "pre_initial_speed_mps": math.hypot(float(values[0]), float(values[1])),
        "db_path": row["db_path"], "db_file": row["db_file"], "source_partition": row["source_partition"],
    }


def _select_family_suffix(
    family: str,
    needed: int,
    cutoff: str,
    exclusion: Mapping[str, Any],
    map_cache: MutableMapping[str, Any],
    used_tokens: set,
    used_logs: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if needed <= 0:
        return [], [], {"evaluated": 0, "failure_counts": {}, "selected": 0}
    candidates, source_audit = _tail_candidates(family, cutoff, exclusion, used_tokens, used_logs)
    selected: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    failures: Counter = Counter()
    evaluated = 0
    for row in candidates:
        evaluated += 1
        candidate = _hydrate_candidate(row)
        resolution = official_count(str(candidate["db_path"]), str(candidate["scenario_token"]))
        if resolution != 1:
            failures["OFFICIAL_EXACT_RESOLUTION_COUNT_NOT_ONE"] += 1
            continue
        try:
            entry, legacy_audit = frozen.evaluate_candidate(candidate, family, MAPS, map_cache)
        except frozen.EligibilityError as exc:
            failures[str(exc)] += 1
            continue
        if family == "R-HLC":
            try:
                route: Any = _route_continuous_primary80_audit(entry, map_cache)
            except Exception as exc:
                failures[f"HLC_ROUTE_CONTINUOUS_PRIMARY80_FAIL:{type(exc).__name__}:{exc}"] += 1
                continue
        else:
            route = "NOT_APPLICABLE_TSB_FROZEN_SEMANTICS_UNCHANGED"
        entry["selector_rank_sha256"] = row["selector_rank_sha256"]
        entry["official_exact_single_scenario_resolution"] = "PASS"
        entry["Primary80_execution_eligibility"] = "PASS_PENDING_EXACT_CONTROLLER_CONFIRMATION_IN_ZERO_RUN"
        if family == "R-HLC":
            entry["HLC_route_continuous_Primary80_applicability"] = "PASS"
        else:
            entry["TSB_frozen_eligibility_semantics"] = "UNCHANGED_PASS"
        selected.append(entry)
        audits.append(
            {
                "family": family, "scenario_token": entry["scenario_token"],
                "selector_rank_sha256": entry["selector_rank_sha256"],
                "legacy_frozen_eligibility": legacy_audit,
                "official_resolution_count": resolution,
                "route_continuous": route,
                "deterministic_rank_position_after_v2_1_stop": evaluated,
            }
        )
        if len(selected) == needed:
            source_audit.update(
                {
                    "evaluated_after_v2_1_stop": evaluated,
                    "failure_counts": dict(sorted(failures.items())),
                    "selected_count": len(selected),
                    "stopping_rank_sha256": entry["selector_rank_sha256"],
                    "tail_window_proof": "selected_position_below_capacity_and_every_lower_rank_in_suffix_evaluated",
                }
            )
            return selected, audits, source_audit
    raise RuntimeError(f"INSUFFICIENT_{family}_AFTER_V1_3_INCREMENTAL_GATES:{len(selected)}")


def _schedule(entries: Sequence[Mapping[str, Any]], roster_sha: str) -> Dict[str, Any]:
    runs = []
    for pair_index, entry in enumerate(entries, 1):
        pair_id = f"R1B29D-{pair_index:02d}-{entry['family']}"
        for arm_index, arm in enumerate(entry["arms"]):
            arm_name = "BASELINE" if arm_index == 0 else "TREATMENT"
            runs.append(
                {
                    "run_id": f"{pair_id}-{arm_name}", "pair_id": pair_id, "family": entry["family"],
                    "scenario_token": entry["scenario_token"], "log_id": entry["log_id"],
                    "arm": arm, "run_order": len(runs) + 1,
                }
            )
    if len(runs) != 48 or [row["run_order"] for row in runs] != list(range(1, 49)):
        raise RuntimeError("B2_9_D_SCHEDULE_CARDINALITY_OR_ORDER_FAIL")
    old_ids = {row["run_id"] for row in read_json(OLD_SCHEDULE)["runs"]}
    if old_ids & {row["run_id"] for row in runs}:
        raise RuntimeError("B2_9_D_RUN_ID_REUSE")
    return {
        "schema_version": "r1_official_compliant_technical_smoke_schedule_v3.0",
        "status": "FROZEN_PENDING_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION",
        "roster_sha256": roster_sha,
        "runtime_bindings": {
            "planner": "tools.r1_official_technical_smoke_planner_v3_1.R1OfficialTechnicalSmokePlannerV3_1",
            "time_controller": "tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
            "planner_reference_semantics": "ROUTE_CONTINUOUS_V2_3",
            "measurement_reference_semantics": "FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT",
        },
        "runs": runs,
        "audit": {"pairs": 24, "runs": 48, "run_order": "EXACT_1_TO_48_NO_SHUFFLE", "new_namespace": "R1B29D"},
        "OFFICIAL_SMOKE_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
        "simulation_launched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-tail-capacity", type=int, default=TAIL_CAPACITY)
    args = parser.parse_args()
    if args.rank_tail_capacity != TAIL_CAPACITY:
        raise ValueError("FROZEN_IMPLEMENTATION_CAPACITY_OVERRIDE_FORBIDDEN")
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"B2_9_D_VERSIONED_OUTPUT_EXISTS:{existing}")
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    if sha256(RUNTIME_CANDIDATE) != "9af891c87c951494b382840154ef39036b019815cc8923875f41b0e24a632434":
        raise ValueError("APPROVED_RUNTIME_CANDIDATE_SHA_MISMATCH")
    official_env()
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    exclusion = effective_exclusion_ledger()
    contract = selector_contract(exclusion)
    map_cache: Dict[str, Any] = {}
    retained, retained_audits, prefix = _existing_selected_prefix(exclusion, map_cache)
    family_retained = {family: [row for row in retained if row["family"] == family] for family in FAMILIES}
    if any(len(family_retained[family]) > 12 for family in FAMILIES):
        raise RuntimeError("RETAINED_COUNT_EXCEEDS_FAMILY_QUOTA")
    old_hlc_stop = str(read_json(OLD_INCREMENTAL_AUDIT)["lazy_accounting"]["family"]["R-HLC"]["stopping_rank_sha256"])
    old_tsb_stop = str(read_json(OLD_INCREMENTAL_AUDIT)["lazy_accounting"]["family"]["R-TSB"]["stopping_rank_sha256"])
    used_tokens = {row["scenario_token"] for row in retained}
    used_logs = {row["log_id"] for row in retained}
    hlc_replacements, hlc_replacement_audits, hlc_suffix = _select_family_suffix(
        "R-HLC", 12 - len(family_retained["R-HLC"]), old_hlc_stop, exclusion, map_cache,
        used_tokens, used_logs,
    )
    used_tokens.update(row["scenario_token"] for row in hlc_replacements)
    used_logs.update(row["log_id"] for row in hlc_replacements)
    tsb_replacements, tsb_replacement_audits, tsb_suffix = _select_family_suffix(
        "R-TSB", 12 - len(family_retained["R-TSB"]), old_tsb_stop, exclusion, map_cache,
        used_tokens, used_logs,
    )
    replacement_audits = hlc_replacement_audits + tsb_replacement_audits
    hlc_entries = sorted(family_retained["R-HLC"] + hlc_replacements, key=lambda row: (row["selector_rank_sha256"], row["scenario_token"], row["log_id"]))
    tsb_entries = sorted(family_retained["R-TSB"] + tsb_replacements, key=lambda row: (row["selector_rank_sha256"], row["scenario_token"], row["log_id"]))
    entries = hlc_entries + tsb_entries
    if len(entries) != 24 or len({row["scenario_token"] for row in entries}) != 24 or len({row["log_id"] for row in entries}) != 24:
        raise RuntimeError("B2_9_D_ROSTER_GLOBAL_UNIQUENESS_FAIL")
    forbidden_tokens = {row["scenario_token"] for row in exclusion["entries"]}
    forbidden_logs = {row["log_id"] for row in exclusion["entries"]}
    if any(row["scenario_token"] in forbidden_tokens or row["log_id"] in forbidden_logs for row in entries):
        raise RuntimeError("B2_9_D_ROSTER_CONTAINS_EFFECTIVE_EXCLUSION")
    audit = {
        "schema_version": "r1_b2_9_d_selector_eligibility_audit_v1.0",
        "status": "PASS_OUTCOME_BLIND_TOP_12_PER_FAMILY",
        "source_universe": {"R-HLC": hlc_suffix, "R-TSB": tsb_suffix},
        "prefix_reuse": prefix,
        "selected_candidate_audits": retained_audits + replacement_audits,
        "counts": {"R-HLC_eligible_selected": 12, "R-TSB_eligible_selected": 12, "total": 24},
        "forbidden_inputs_opened": [],
        "scientific_or_canary_outcome_used": False,
        "threshold_mechanism_F_match_safety_changed": False,
        "simulation_started": False,
    }
    roster = {
        "schema_version": "r1_official_compliant_technical_smoke_roster_v3.0",
        "status": "FROZEN_PROSPECTIVE_PRIMARY80_SELECTION_PENDING_FINAL_ZERO_RUN_CONFIRMATION",
        "selector_contract_path": str(OUTPUTS["contract"].relative_to(ROOT)),
        "selector_contract_sha256": canonical_sha(contract),
        "selector_implementation_path": str(Path(__file__).relative_to(ROOT)),
        "selector_implementation_sha256": sha256(Path(__file__)),
        "source_universe_path": str(SOURCE.relative_to(ROOT)),
        "source_universe_sha256": sha256(SOURCE),
        "effective_exclusion_ledger_path": str(OUTPUTS["exclusion"].relative_to(ROOT)),
        "effective_exclusion_ledger_sha256": canonical_sha(exclusion),
        "master_seed": MASTER_SEED,
        "salt_sha256": SALT,
        "entries": entries,
        "counts": {"R-HLC": 12, "R-TSB": 12, "total": 24, "unique_scenario_tokens": 24, "unique_logs": 24},
        "manual_identity_replacement_performed": False,
        "outcome_information_used": False,
        "threshold_changed": False,
        "OFFICIAL_SMOKE_AUTHORIZED": False,
        "NEW_RUN_BUDGET": 0,
    }
    roster_sha = canonical_sha(roster)
    schedule = _schedule(entries, roster_sha)
    old_entries = read_json(OLD_ROSTER)["entries"]
    old_by_token = {row["scenario_token"]: row for row in old_entries}
    new_by_token = {row["scenario_token"]: row for row in entries}
    replacements_rows = []
    for row in old_entries:
        if row["scenario_token"] not in new_by_token:
            replacements_rows.append(
                {
                    "old_identity": {"scenario_token": row["scenario_token"], "log_id": row["log_id"], "family": row["family"]},
                    "old_disposition": "OFFICIAL_ATTEMPT_CONSUMED_AND_PERMANENT_ENGINEERING_CANARY_EXCLUDED" if row["scenario_token"] == ATTEMPT1_TOKEN else "ADDITIVE_ELIGIBILITY_EXCLUDED",
                    "new_identity": None,
                }
            )
    new_only = [row for row in entries if row["scenario_token"] not in old_by_token]
    if len(replacements_rows) != len(new_only):
        raise RuntimeError("OLD_NEW_REPLACEMENT_CARDINALITY_MISMATCH")
    for target, replacement in zip(replacements_rows, new_only):
        target["new_identity"] = {
            "scenario_token": replacement["scenario_token"], "log_id": replacement["log_id"],
            "family": replacement["family"], "selector_rank_sha256": replacement["selector_rank_sha256"],
            "deterministic_reason": "FIRST_COMPLETE_PASS_IN_FROZEN_V1_3_RANK_ORDER_AFTER_EFFECTIVE_EXCLUSIONS",
        }
    comparison = {
        "schema_version": "r1_b2_9_d_roster_v2_1_to_v3_0_comparison_v1",
        "status": "PASS_DETERMINISTIC_OUTCOME_BLIND_COMPARISON",
        "old_roster": {"path": str(OLD_ROSTER.relative_to(ROOT)), "sha256": sha256(OLD_ROSTER)},
        "new_roster_canonical_sha256": roster_sha,
        "counts": {
            "HLC_retained": sum(row["family"] == "R-HLC" and row["scenario_token"] in old_by_token for row in entries),
            "HLC_replaced": sum(row["family"] == "R-HLC" and row["scenario_token"] not in old_by_token for row in entries),
            "TSB_retained": sum(row["family"] == "R-TSB" and row["scenario_token"] in old_by_token for row in entries),
            "TSB_replaced": sum(row["family"] == "R-TSB" and row["scenario_token"] not in old_by_token for row in entries),
        },
        "replacements": replacements_rows,
        "scientific_rollout_outcome_used": False,
    }
    for key, payload in (
        ("exclusion", exclusion), ("contract", contract), ("audit", audit),
        ("roster", roster), ("comparison", comparison), ("schedule", schedule),
    ):
        write_new(OUTPUTS[key], payload)
    print(
        json.dumps(
            {
                "status": audit["status"], "exclusion_count": 45,
                "roster_sha256": sha256(OUTPUTS["roster"]),
                "schedule_sha256": sha256(OUTPUTS["schedule"]),
                "HLC_retained": comparison["counts"]["HLC_retained"],
                "HLC_replaced": comparison["counts"]["HLC_replaced"],
                "TSB_retained": comparison["counts"]["TSB_retained"],
                "TSB_replaced": comparison["counts"]["TSB_replaced"],
                "simulation_started": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
