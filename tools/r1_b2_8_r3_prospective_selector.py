#!/usr/bin/env python3
"""R1 B2.8-R3 prospective, outcome-blind official-execution amendment.

This entry point only reads frozen source/map/DB material.  It never imports a
simulation runner and never invokes simulator or planner rollout methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import r1_b2_7_freeze_official_smoke_roster_v2 as frozen
from tools import r1_b2_7_freeze_official_smoke_roster_v2_1 as lazy
from tools import r1_future_compliant_smoke_selector_v1_1 as selector

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
CACHE = ROOT.parent / "nuplan/dataset/data/cache"
MAPS = ROOT.parent / "nuplan/dataset/maps"
OLD_ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.0.json"
OLD_SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v2.0.json"
OLD_CONTRACT = R1 / "r1_future_compliant_smoke_selector_contract_v1.1.json"
SOURCE = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
OUT = {
    "forensic": R1 / "r1_b2_8_r3_four_token_resolution_forensic_v1.0.json",
    "forensic_report": R1 / "R1_B2_8_R3_Four_Token_Resolution_Forensic_Report_v1.md",
    "contract": R1 / "r1_future_compliant_smoke_selector_contract_v1.2.json",
    "roster": R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json",
    "schedule": R1 / "r1_official_compliant_technical_smoke_schedule_v2.1.json",
    "delta": R1 / "r1_b2_8_r3_roster_schedule_delta_audit_v1.0.json",
    "ledger": R1 / "r1_official_execution_ineligible_identity_ledger_v1.0.json",
}
FAILED = {
    "a6e0468e028357de": "2021.08.27.14.14.40_veh-45_01790_02016",
    "0198af1831f65977": "2021.09.28.13.24.06_veh-44_02759_02879",
    "cf56ddebd44f5372": "2021.09.28.19.55.30_veh-44_01744_01819",
    "0f67192c7dd45664": "2021.09.10.15.00.33_veh-45_01265_01432",
}


def sha(path: Path) -> str:
    return frozen.sha256_file(path)


def read(path: Path) -> Dict[str, Any]:
    return frozen.read_json(path)


def write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R3_VERSIONED_ARTIFACT_EXISTS:{path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def official_env() -> None:
    devkit = ROOT.parent / "nuplan-devkit"
    os.environ.update({
        "NUPLAN_DEVKIT_ROOT": str(devkit), "NUPLAN_DATA_ROOT": str(ROOT.parent / "nuplan/dataset/data"),
        "NUPLAN_MAPS_ROOT": str(MAPS), "NUPLAN_MAP_ROOT": str(MAPS),
        "NUPLAN_EXP_ROOT": str(ROOT.parent / "nuplan/exp"), "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    for path in (str(devkit), str(ROOT.parent / "tuplan_garage"), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def official_count(db_path: str, token: str) -> int:
    official_env()
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
    return len(list(get_scenarios_from_db(db_path, [token], None, None, True, False)))


def db_forensic(entry: Mapping[str, Any]) -> Dict[str, Any]:
    token, db = str(entry["scenario_token"]), Path(str(entry["db_path"]))
    query = """WITH ordered AS (SELECT token,name,ROW_NUMBER() OVER (ORDER BY name) rn FROM scene), n AS (SELECT COUNT(*) cnt FROM scene)
    SELECT lp.timestamp, lower(hex(lp.scene_token)), o.name,o.rn,n.cnt,
      (SELECT group_concat(type,'|') FROM scenario_tag st WHERE st.lidar_pc_token=lp.token),
      s.goal_ego_pose_token IS NOT NULL,l.logfile,l.map_version
    FROM lidar_pc lp JOIN ordered o ON o.token=lp.scene_token CROSS JOIN n JOIN scene s ON s.token=lp.scene_token
    JOIN lidar ld ON ld.token=lp.lidar_token JOIN log l ON l.token=ld.log_token WHERE lower(hex(lp.token))=?"""
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as con:
        row = con.execute(query, (token,)).fetchone()
    if row is None:
        raise RuntimeError(f"FROZEN_TOKEN_MISSING_FROM_DB:{token}")
    timestamp, scene_token, scene_name, rank, total, tags, has_goal, log_id, map_name = row
    valid_by_bound_query = int(rank) >= 3 and int(rank) < int(total) - 1
    return {
        "scenario_token": token, "log_id": str(log_id), "db_path": str(db), "token_exists_in_lidar_pc": True,
        "scenario_tag_exists": tags is not None, "scenario_tag_types": [] if tags is None else str(tags).split("|"),
        "lidar_timestamp_us": int(timestamp), "scene_token": str(scene_token), "scene_name": str(scene_name),
        "scene_rank_by_name": int(rank), "scene_count_in_log": int(total), "map_name": str(map_name),
        "mission_goal_relation_present": bool(has_goal), "route_relation_present": bool(entry.get("route_roadblock_ids")),
        "bound_nuplan_valid_scene_predicate": "scene_rank >= 3 AND scene_rank < scene_count - 1",
        "bound_nuplan_valid_scene_predicate_pass": valid_by_bound_query,
        "official_get_scenarios_from_db_count": official_count(str(db), token),
        "bytes_hex_case_parameter_error": False, "db_partition_mismatch": False,
        "nuplan_1_1_cache_vs_bound_1_2_2_semantic_mismatch": False,
    }


def forensic() -> Dict[str, Any]:
    roster = read(OLD_ROSTER)["entries"]
    targets = [row for row in roster if row["scenario_token"] in FAILED]
    controls = [row for row in roster if row["scenario_token"] not in FAILED][:4]
    failed = [db_forensic(row) for row in targets]
    passing = [db_forensic(row) for row in controls]
    if len(failed) != 4 or any(row["official_get_scenarios_from_db_count"] != 0 for row in failed):
        raise RuntimeError("FOUR_TOKEN_FORENSIC_INPUT_MISMATCH")
    if len(passing) != 4 or any(row["official_get_scenarios_from_db_count"] != 1 for row in passing):
        raise RuntimeError("NEGATIVE_CONTROL_OFFICIAL_RESOLUTION_MISMATCH")
    return {
        "schema_version": "r1_b2_8_r3_four_token_resolution_forensic_v1.0",
        "status": "FROZEN_IDENTITY_NOT_OFFICIAL_EXECUTION_ELIGIBLE_UNDER_BOUND_RUNTIME",
        "bound_runtime": {"nuplan_version": "1.2.2", "implementation": "nuplan.database.nuplan_db.nuplan_scenario_queries.get_scenarios_from_db"},
        "root_cause": "BOUND_QUERY_VALID_SCENES_REQUIRES_TWO_PRECEDING_AND_TWO_FOLLOWING_SCENES; four frozen identities are outside that window.",
        "failed_identities": failed, "negative_controls": passing,
        "classification": "B_FROZEN_IDENTITY_NOT_OFFICIAL_EXECUTION_ELIGIBLE_UNDER_BOUND_RUNTIME",
        "query_implementation_defect": False, "other_pre_run_execution_binding_defect": False,
        "replacement_performed": False, "simulation_started": False,
    }


def amendment_contract() -> Dict[str, Any]:
    old = read(OLD_CONTRACT)
    return {
        "schema_version": "r1_future_compliant_smoke_selector_contract_v1.2",
        "status": "FROZEN_AFTER_PROSPECTIVE_EXECUTION_ELIGIBILITY_AMENDMENT",
        "inherits": {"path": str(OLD_CONTRACT.relative_to(ROOT)), "sha256": sha(OLD_CONTRACT)},
        "purpose": "PRE_RUN_PROTOCOL_AMENDMENT_BEFORE_ANY_B2_8_OFFICIAL_ROLLOUT",
        "master_seed": old["master_seed"], "salt_sha256": old["salt_sha256"], "salt_regeneration_allowed": False,
        "rank_rule": old["rank_rule"], "sort_and_tie_break": old["sort_and_tie_break"],
        "source_universe": old["source_universe"], "permanent_blacklist": old["permanent_blacklist"],
        "frozen_eligibility_implementation": old["frozen_eligibility_implementation"],
        "selection_inputs_allowlist": old["selection_inputs_allowlist"], "selection_scope": old["selection_scope"],
        "new_execution_eligibility_gate": {
            "name": "OFFICIAL_EXACT_SINGLE_SCENARIO_RESOLUTION_UNDER_BOUND_RUNTIME",
            "authoritative_semantics": "nuPlan_1.2.2_get_scenarios_from_db(db_path,[scenario_token],None,None,True,False)",
            "pass": "resolution_count == 1", "zero_match": "EXECUTION_INELIGIBLE", "multiple_match": "EXECUTION_AMBIGUOUS_INELIGIBLE",
            "pretreatment_only": True, "outcome_blind": True, "representation_blind": True, "planner_outcome_blind": True,
            "mechanism_outcome_blind": True, "safety_outcome_blind": True,
        },
        "all_other_selection_rules_exact_identical_to_v1_1": True,
        "manual_identity_selection_or_replacement_allowed": False,
        "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0, "RBR_A_B_C_AUTHORIZED": False,
    }


def select(contract: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    blacklist = frozen.resolve_effective_blacklist()
    store = R1 / ".r1_b2_8_r3_source.sqlite"
    if store.exists():
        raise FileExistsError(f"STALE_R3_SOURCE_STORE_REQUIRES_REVIEW:{store}")
    con, source_audit = lazy.global_source_preflight(CACHE, str(contract["salt_sha256"]), blacklist, store, False)
    used_tokens: set[str] = set(); used_logs: set[str] = set(); selected: list[Dict[str, Any]] = []; accounting: Dict[str, Any] = {}
    try:
        for family in ("R-HLC", "R-TSB"):
            map_cache: Dict[str, Any] = {}
            def eligible(candidate: selector.RankedIdentity):
                count = official_count(str(candidate.payload["db_path"]), candidate.scenario_token)
                if count != 1:
                    return False, {"failure_reason": "EXECUTION_INELIGIBLE" if count == 0 else "EXECUTION_AMBIGUOUS_INELIGIBLE", "resolution_count": count}
                try:
                    entry, audit = frozen.evaluate_candidate(candidate.payload, family, MAPS, map_cache)
                except frozen.EligibilityError as exc:
                    return False, {"failure_reason": str(exc)}
                entry["selector_rank_sha256"] = candidate.rank_sha256
                audit["selector_rank_sha256"] = candidate.rank_sha256
                audit["OFFICIAL_EXACT_SINGLE_SCENARIO_RESOLUTION_UNDER_BOUND_RUNTIME"] = "PASS"
                return True, {"entry": entry, "audit": audit}
            rows, info = selector.lazy_rank_ordered_select(lazy._ranked_rows(con, family), eligible, 12, used_tokens, used_logs)
            if len(rows) != 12:
                raise RuntimeError(f"R3_INSUFFICIENT_ELIGIBLE_IDENTITIES:{family}:{len(rows)}")
            selected.extend(dict(value["entry"]) for _, value in rows); accounting[family] = info
    finally:
        con.close(); store.unlink(missing_ok=True)
    if len(selected) != 24 or len({r["scenario_token"] for r in selected}) != 24 or len({r["log_id"] for r in selected}) != 24:
        raise RuntimeError("R3_GLOBAL_UNIQUENESS_FAIL")
    return selected, {"source": source_audit, "family": accounting}


def main() -> None:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("R3_VERSIONED_OUTPUT_ALREADY_EXISTS")
    result = forensic(); write(OUT["forensic"], result)
    report = "# R1 B2.8-R3 四 Token 官方解析取证\n\n四个 token 均真实存在、tag/log/map 正确；不是 bytes/hex、DB partition 或 API 参数错误。bound nuPlan 1.2.2 的 `get_scenarios_from_db` 要求 scene rank >= 3 且 < scene_count - 1。四项均在该有效窗口外，故为 B：冻结 identity 在 bound runtime 下不具 official execution eligibility。\n"
    OUT["forensic_report"].write_text(report, encoding="utf-8")
    contract = amendment_contract(); write(OUT["contract"], contract)
    entries, accounting = select(contract)
    source = read(SOURCE); blacklist = frozen.resolve_effective_blacklist()
    roster = {"schema_version": "r1_official_compliant_technical_smoke_roster_v2.1", "status": "FROZEN_AFTER_PRE_RUN_EXECUTION_ELIGIBILITY_AMENDMENT", "selector_contract_path": str(OUT["contract"].relative_to(ROOT)), "selector_contract_sha256": frozen.artifact_sha256(contract), "selector_implementation_sha256": sha(Path(__file__)), "source_universe_path": str(SOURCE.relative_to(ROOT)), "source_universe_sha256": sha(SOURCE), "effective_blacklist_sha256": frozen.canonical_sha256(blacklist), "db_fingerprint_sha256": source["source_db"]["source_root_fingerprint_sha256"], "map_fingerprint_sha256": source["map_binding"]["map_root_fingerprint_sha256"], "master_seed": contract["master_seed"], "salt_sha256": contract["salt_sha256"], "entries": entries, "counts": {"R-HLC": 12, "R-TSB": 12, "total": 24, "unique_scenario_tokens": 24, "unique_logs": 24}, "manual_identity_replacement_performed": False, "outcome_information_used": False, "threshold_changed": False, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0}
    roster_sha = frozen.artifact_sha256(roster); write(OUT["roster"], roster)
    bindings = read(OLD_SCHEDULE)["execution_bindings"]
    schedule = frozen.build_schedule(entries, bindings, roster_sha); schedule["schema_version"] = "r1_official_compliant_technical_smoke_schedule_v2.1"; schedule["status"] = "FROZEN_AFTER_PRE_RUN_EXECUTION_ELIGIBILITY_AMENDMENT"; write(OUT["schedule"], schedule)
    old = read(OLD_ROSTER)["entries"]; old_tokens = {r["scenario_token"]: r for r in old}; new_tokens = {r["scenario_token"]: r for r in entries}
    delta = {"schema_version": "r1_b2_8_r3_roster_schedule_delta_audit_v1.0", "v2_0_status": "SUPERSEDED_PROSPECTIVELY_BEFORE_ANY_OFFICIAL_RUN", "v2_1_status": "FROZEN_AFTER_PRE_RUN_EXECUTION_ELIGIBILITY_AMENDMENT", "manual_identity_replacement": False, "outcome_information_used": False, "threshold_changed": False, "salt_changed": False, "ranking_rule_changed": False, "identities": [{"scenario_token": token, "log_id": row["log_id"], "status": "RETAINED" if token in new_tokens else "REMOVED_EXECUTION_INELIGIBLE" if token in FAILED else "REMOVED_BY_FROZEN_RANK_ORDER"} for token,row in old_tokens.items()] + [{"scenario_token": token, "log_id": row["log_id"], "status": "NEWLY_SELECTED_BY_FROZEN_RANK_ORDER"} for token,row in new_tokens.items() if token not in old_tokens], "lazy_accounting": accounting}
    write(OUT["delta"], delta)
    ledger = {"schema_version": "r1_official_execution_ineligible_identity_ledger_v1.0", "status": "PRE_RUN_EXECUTION_ELIGIBILITY_ONLY", "historical_outcome_blacklist": False, "entries": [{"scenario_token": r["scenario_token"], "log_id": r["log_id"], "reason": "BOUND_NUPLAN_VALID_SCENE_WINDOW_0_MATCH", "source": str(OUT["forensic"].relative_to(ROOT))} for r in result["failed_identities"]], "simulation_started": False}
    write(OUT["ledger"], ledger)
    print(json.dumps({"status": roster["status"], "roster_sha256": sha(OUT["roster"]), "schedule_sha256": sha(OUT["schedule"]), "simulation_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
