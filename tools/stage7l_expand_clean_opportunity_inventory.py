#!/usr/bin/env python3
"""Expand Stage7L-B2 clean opportunities with pre-treatment dynamic clearance.

This offline tool reads only original nuPlan database/map/replay-track assets.
It never imports the Stage7L planner or a rollout, so dynamic eligibility is
identical for every future lateral dose.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_audit_dynamic_lane_change_clearance import config_from_args
from tools.stage7l_build_lane_change_opportunity_inventory import (
    candidate_rows_from_db,
    evaluate_candidate_options,
    historical_exclusions,
    read_csv,
    sha256_file,
    write_csv,
)
from tools.stage7l_dynamic_clearance import dynamic_clearance_audit
from tools.stage7l_freeze_development_roster import canonical_progress


def merge_exclusions(stage7_roster: Path, stage7p_root: Path, additional_ledger: Path) -> List[Dict[str, str]]:
    rows = historical_exclusions(stage7_roster, stage7p_root)
    if additional_ledger.is_file():
        rows.extend(read_csv(additional_ledger))
    merged: Dict[str, Dict[str, str]] = {}
    for row in rows:
        token = str(row.get("scenario_token", ""))
        if not token:
            continue
        if token not in merged:
            merged[token] = dict(row)
        else:
            prior = merged[token]
            if str(row.get("exclusion_reason", "")) not in str(prior.get("exclusion_reason", "")):
                prior["exclusion_reason"] = str(prior.get("exclusion_reason", "")) + "|" + str(row.get("exclusion_reason", ""))
                prior["source_path"] = str(prior.get("source_path", "")) + "|" + str(row.get("source_path", ""))
            if not prior.get("log_name") and row.get("log_name"):
                prior["log_name"] = row["log_name"]
    return sorted(merged.values(), key=lambda row: row["scenario_token"])


def reference_coverage_pass(candidate: Mapping[str, Any]) -> bool:
    required = canonical_progress(float(candidate["initial_speed_mps"]), 15.4)
    return float(candidate["paired_reference_remaining_m"]) >= required


def choose_one_option_per_token(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministically choose one clean direction per token without outcomes."""
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        token = str(row["scenario_token"])
        if token not in selected:
            selected[token] = row
            continue
        current = selected[token]
        candidate_key = (
            float(row["paired_reference_remaining_m"]), float(row["minimum_target_lane_object_gap_m"]), row["direction"] == "left"
        )
        current_key = (
            float(current["paired_reference_remaining_m"]), float(current["minimum_target_lane_object_gap_m"]), current["direction"] == "left"
        )
        if candidate_key > current_key:
            selected[token] = row
    return sorted(selected.values(), key=lambda row: (row["log_name"], row["scenario_token"]))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = config_from_args(args)
    exclusions = merge_exclusions(args.stage7_lane_change_roster, args.stage7p_root, args.additional_exclusion_ledger)
    excluded_tokens = {row["scenario_token"] for row in exclusions}
    # Pool A excludes every historical token.  Pool B additionally excludes
    # only logs consumed by Stage7L-B development itself; older Stage7/P token
    # exclusions must not silently erase hundreds of unrelated Pittsburgh logs.
    development_logs = {
        row.get("log_name", "") for row in read_csv(args.additional_exclusion_ledger)
        if row.get("log_name") and "STAGE7L_B_" in str(row.get("exclusion_reason", ""))
    }
    inputs = read_csv(args.inventory_inputs)
    rows: List[Dict[str, Any]] = []
    static_reason_counts: Counter[str] = Counter()
    dynamic_reason_counts: Counter[str] = Counter()
    scanned_anchor_candidates = 0
    scanned_db_count = 0
    for db_index, input_row in enumerate(inputs, start=1):
        if args.max_dbs and db_index > args.max_dbs:
            break
        db_path = args.nuplan_db_root / input_row["db_file"]
        if not db_path.is_file():
            static_reason_counts["MISSING_DB"] += 1
            continue
        scanned_db_count += 1
        try:
            anchors = candidate_rows_from_db(db_path, args.candidates_per_db, args.minimum_speed_mps)
        except sqlite3.Error:
            static_reason_counts["DB_QUERY_ERROR"] += 1
            continue
        scanned_anchor_candidates += len(anchors)
        for anchor in anchors:
            if args.required_map_name and str(anchor.get("map_name", "")) != args.required_map_name:
                static_reason_counts["MAP_NOT_IN_SCOPE"] += 1
                continue
            if anchor["scenario_token"] in excluded_tokens:
                static_reason_counts["HISTORIC_TOKEN_EXCLUDED"] += 1
                continue
            try:
                options = evaluate_candidate_options(
                    anchor, args.nuplan_db_root, args.nuplan_map_root, args.map_version,
                    args.minimum_paired_reference_remaining_m, args.minimum_target_gap_m, args.minimum_speed_mps,
                )
            except Exception as exc:
                options = [{
                    "db_file": anchor["db_file"], "log_name": anchor["log_name"],
                    "scenario_token": anchor["scenario_token"], "eligible": False,
                    "reason_code": f"STATIC_EVALUATION_ERROR:{type(exc).__name__}", "error": str(exc),
                }]
            for option in options:
                static_reason_counts[str(option.get("reason_code", "UNSET"))] += 1
                row = dict(option)
                row["development_log_disjoint"] = row.get("log_name", "") not in development_logs
                row["static_reference_coverage_pass"] = (
                    bool(row.get("eligible")) and reference_coverage_pass(row)
                )
                row["candidate_key"] = f"{row.get('scenario_token', '')}:{row.get('direction', 'none')}"
                if bool(row.get("eligible")) and row["static_reference_coverage_pass"]:
                    try:
                        dynamic = dynamic_clearance_audit(row, args.nuplan_db_root, config)
                    except Exception as exc:
                        dynamic = {
                            "dynamic_clearance_pass": False,
                            "dynamic_reason_code": f"DYNAMIC_AUDIT_ERROR:{type(exc).__name__}",
                            "dynamic_error": str(exc), "dynamic_clearance_config_sha256": config.fingerprint(),
                            "dynamic_eligibility_pre_treatment": True, "dynamic_dose_independent": True,
                        }
                    row.update(dynamic)
                    dynamic_reason_counts[str(row.get("dynamic_reason_code", "UNSET"))] += 1
                else:
                    row.update({
                        "dynamic_clearance_pass": False,
                        "dynamic_reason_code": "STATIC_PRECONDITION_FAIL",
                        "dynamic_eligibility_pre_treatment": True, "dynamic_dose_independent": True,
                        "dynamic_clearance_config_sha256": config.fingerprint(),
                    })
                rows.append(row)
        if db_index % 50 == 0:
            clean = sum(bool(row.get("dynamic_clearance_pass")) for row in rows)
            print(f"[Stage7L-B2 expand] db={db_index}/{len(inputs)} anchors={scanned_anchor_candidates} dynamic_clean_options={clean}", flush=True)
    if not rows:
        raise ValueError("expanded inventory evaluated zero candidates")
    fields = sorted({key for row in rows for key in row})
    expanded_path = args.output_dir / "expanded_candidate_inventory.csv"
    write_csv(expanded_path, rows, fields)
    clean_options = [row for row in rows if bool(row.get("dynamic_clearance_pass"))]
    pool_a = choose_one_option_per_token(clean_options)
    pool_b = [row for row in pool_a if bool(row["development_log_disjoint"])]
    pool_a_path = args.output_dir / "pool_a_scenario_disjoint_dynamic_clean.csv"
    pool_b_path = args.output_dir / "pool_b_strict_development_log_disjoint_dynamic_clean.csv"
    pool_fields = sorted({key for row in pool_a for key in row}) if pool_a else fields
    write_csv(pool_a_path, pool_a, pool_fields)
    write_csv(pool_b_path, pool_b, pool_fields)
    rejection_rows = [
        {"stage": "static", "reason_code": key, "count": value} for key, value in sorted(static_reason_counts.items())
    ] + [
        {"stage": "dynamic", "reason_code": key, "count": value} for key, value in sorted(dynamic_reason_counts.items())
    ]
    rejection_path = args.output_dir / "rejection_reason_summary.csv"
    write_csv(rejection_path, rejection_rows, ["stage", "reason_code", "count"])
    def direction_counts(pool: List[Dict[str, Any]]) -> Dict[str, int]:
        return {"left": sum(row.get("direction") == "left" for row in pool), "right": sum(row.get("direction") == "right" for row in pool)}
    summary = {
        "schema_version": "stage7l_b2_dynamic_clearance_inventory_expansion_v1",
        "status": "PRETREATMENT_DYNAMIC_INVENTORY_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_is_pre_treatment": True, "rollout_outcome_read": False,
        "embedding_or_bdd_read": False, "dose_dependent_input": False,
        "scanned_db_count": scanned_db_count, "scanned_anchor_candidate_count": scanned_anchor_candidates,
        "expanded_option_count": len(rows),
        "static_eligible_option_count": sum(bool(row.get("eligible")) and bool(row.get("static_reference_coverage_pass")) for row in rows),
        "dynamic_clean_option_count": len(clean_options),
        "pool_a_scenario_disjoint": {
            "unique_tokens": len(pool_a), "unique_logs": len({row["log_name"] for row in pool_a}), **direction_counts(pool_a),
        },
        "pool_b_strict_development_log_disjoint": {
            "unique_tokens": len(pool_b), "unique_logs": len({row["log_name"] for row in pool_b}), **direction_counts(pool_b),
        },
        "official_runnability_all_pool_b": all(bool(row.get("official_query_runnable")) for row in pool_b),
        "development_excluded_token_count": len(excluded_tokens), "development_excluded_log_count": len(development_logs),
        "dynamic_config": vars(config), "dynamic_config_sha256": config.fingerprint(),
        "static_reason_counts": dict(static_reason_counts), "dynamic_reason_counts": dict(dynamic_reason_counts),
        "sha256": {
            "inventory_inputs": sha256_file(args.inventory_inputs),
            "additional_exclusion_ledger": sha256_file(args.additional_exclusion_ledger),
            "expanded_candidate_inventory": sha256_file(expanded_path),
            "pool_a": sha256_file(pool_a_path), "pool_b": sha256_file(pool_b_path),
            "rejection_summary": sha256_file(rejection_path),
        },
    }
    (args.output_dir / "fresh_inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory_inputs", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--map_version", default="nuplan-maps-v1.0")
    parser.add_argument("--stage7_lane_change_roster", type=Path, required=True)
    parser.add_argument("--stage7p_root", type=Path, required=True)
    parser.add_argument("--additional_exclusion_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--candidates_per_db", type=int, default=24)
    parser.add_argument("--max_dbs", type=int, default=0)
    parser.add_argument("--minimum_speed_mps", type=float, default=3.0)
    parser.add_argument("--minimum_paired_reference_remaining_m", type=float, default=90.0)
    parser.add_argument("--minimum_target_gap_m", type=float, default=15.0)
    parser.add_argument("--required_map_name", default="us-pa-pittsburgh-hazelwood")
    parser.add_argument("--horizon_seconds", type=float, default=15.0)
    parser.add_argument("--time_step_seconds", type=float, default=0.1)
    parser.add_argument("--maximum_track_interpolation_gap_seconds", type=float, default=0.25)
    parser.add_argument("--trigger_route_progress_m", type=float, default=12.0)
    parser.add_argument("--gentle_transition_length_m", type=float, default=60.0)
    parser.add_argument("--settling_margin_m", type=float, default=10.0)
    parser.add_argument("--target_speed_mps", type=float, default=5.0)
    parser.add_argument("--accel_limit_mps2", type=float, default=1.0)
    parser.add_argument("--ego_length_m", type=float, default=5.0)
    parser.add_argument("--ego_width_m", type=float, default=2.0)
    parser.add_argument("--longitudinal_buffer_m", type=float, default=3.0)
    parser.add_argument("--lateral_buffer_m", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
