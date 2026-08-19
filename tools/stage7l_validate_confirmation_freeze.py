#!/usr/bin/env python3
"""Read-only validation of a prospective Stage7L-C confirmation freeze."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_freeze_confirmation_roster import (
    ROOT,
    as_bool,
    development_sets,
    read_csv,
    read_json,
    select_roster_rows,
    sha256_file,
    validate_authority_assets,
    validate_pool_rows,
)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = args.repo_root.resolve()
    protocol = read_json(args.protocol_config)
    assets = validate_authority_assets(root, protocol, require_authorized_head=False)
    if args.pool_b.resolve() != assets["pool_b"].resolve() or args.development_ledger.resolve() != assets["development_exclusion_ledger"].resolve():
        raise ValueError("validation inputs do not match the frozen authority assets")
    pool = read_csv(args.pool_b)
    validate_pool_rows(pool, protocol)
    expected, expected_trace = select_roster_rows(pool, protocol)
    roster_path = args.freeze_dir / "confirmation_roster.csv"
    maneuver_path = args.freeze_dir / "confirmation_maneuver_manifest.json"
    trace_path = args.freeze_dir / "confirmation_selection_trace.csv"
    runnability_path = args.freeze_dir / "confirmation_runnability_audit.csv"
    summary_path = args.freeze_dir / "confirmation_freeze_summary.json"
    for path in (roster_path, maneuver_path, trace_path, runnability_path, summary_path, args.authorization_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"freeze artifact is missing: {path}")
    roster = read_csv(roster_path)
    trace = read_csv(trace_path)
    runnability = read_csv(runnability_path)
    summary = read_json(summary_path)
    authorization = read_json(args.authorization_manifest)
    ledger_tokens, ledger_logs = development_sets(read_csv(args.development_ledger))
    expected_tokens = [row["scenario_token"] for row in expected]
    actual_tokens = [row["scenario_token"] for row in roster]
    if actual_tokens != expected_tokens:
        raise ValueError("roster selection cannot be replayed from Pool B + frozen seed")
    expected_trace_by_token = {row["scenario_token"]: row for row in expected_trace}
    for row in trace:
        source = expected_trace_by_token.get(row["scenario_token"])
        if source is None or str(bool(source["selected"])) != str(row["selected"]):
            raise ValueError("selection trace is not reproducible")
    token_set = set(actual_tokens)
    log_set = {row["log_name"] for row in roster}
    checks = {
        "scenario_count_equals_80": len(roster) == 80,
        "left_equals_15": sum(row["direction"] == "left" for row in roster) == 15,
        "right_equals_65": sum(row["direction"] == "right" for row in roster) == 65,
        "duplicate_token_count_equals_0": len(token_set) == 80,
        "development_scenario_overlap_count_equals_0": not bool(token_set & ledger_tokens),
        "development_log_overlap_count_equals_0": not bool(log_set & ledger_logs),
        "official_runnability_80_of_80": len(runnability) == 80 and all(as_bool(row["official_query_runnable"]) for row in runnability),
        "dynamic_clearance_80_of_80": all(row["dynamic_clearance_status"] == "DYNAMIC_CLEAR" for row in roster),
        "source_target_trigger_complete_80_of_80": all(
            row.get("source_reference_sha256") and row.get("target_reference_sha256") and row.get("trigger_s_route_m") == "12.0"
            for row in roster
        ),
        "summary_roster_sha_matches": summary["sha256"]["confirmation_roster"] == sha256_file(roster_path),
        "summary_maneuver_sha_matches": summary["sha256"]["confirmation_maneuver_manifest"] == sha256_file(maneuver_path),
        "authorization_status_correct": authorization.get("status") == "STAGE7L_D_ONE_TIME_CONFIRMATION_AUTHORIZED",
        "authorization_roster_sha_matches": authorization["frozen_confirmation_artifacts"]["roster"]["sha256"] == sha256_file(roster_path),
        "authorization_protocol_sha_matches": authorization["frozen_protocol"]["sha256"] == sha256_file(args.protocol_config),
        "freeze_happened_on_authorized_base_head": summary.get("freeze_repository_head") == protocol["authorized_base_commit"],
        "rollout_not_started": summary.get("confirmation_rollout_started") is False and summary.get("stage7l_d_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage7L-C freeze validation failed: {[name for name, value in checks.items() if not value]}")
    result = {
        "schema_version": "stage7l_c_confirmation_freeze_validation_v1",
        "status": "STAGE7L_C_FREEZE_VALIDATED",
        "freeze_dir": str(args.freeze_dir.resolve()),
        "checks": checks,
        "validation_read_embedding_or_bdd": False,
        "validation_launched_rollout": False,
    }
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=ROOT)
    parser.add_argument("--protocol_config", type=Path, required=True)
    parser.add_argument("--pool_b", type=Path, required=True)
    parser.add_argument("--development_ledger", type=Path, required=True)
    parser.add_argument("--freeze_dir", type=Path, required=True)
    parser.add_argument("--authorization_manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
