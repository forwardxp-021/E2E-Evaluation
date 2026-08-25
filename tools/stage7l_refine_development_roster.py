#!/usr/bin/env python3
"""Freeze the Stage7L-B final development roster after mechanism-only refinement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_freeze_development_roster import canonical_progress, maneuver


DOSE_LENGTHS_V2 = {"dose0": 60.0, "dose25": 58.5, "dose50": 57.0, "dose75": 55.5, "dose100": 54.0}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_pass(row: Mapping[str, str]) -> bool:
    required = canonical_progress(float(row["initial_speed_mps"]), 15.4)
    return float(row["paired_reference_remaining_m"]) >= required


def write_manifest(path: Path, rows: List[Dict[str, Any]], inventory_sha: str) -> None:
    payload = {
        "schema_version": "stage7l_b_development_maneuver_manifest_v2",
        "status": "FROZEN_FINAL_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "role": "FULL_24_REFINED_BEFORE_EXECUTION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "maneuvers": [dict(maneuver(row), planner_profile_ids=list(DOSE_LENGTHS_V2)) for row in rows],
        "dose_transition_length_m": DOSE_LENGTHS_V2,
        "planner_horizon_s": 0.4,
        "parameter_version": "safe_development_dose_set_v2",
        "uniform_minimum_target_lane_object_gap_m": 15.0,
        "inventory_csv_sha256": inventory_sha,
        "embedding_or_bdd_read": False,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = read_csv(args.original_roster)
    if len(original) != 24 or len({row["log_name"] for row in original}) != 24:
        raise ValueError("original development roster must contain 24 unique logs")
    retained = [row for row in original if float(row["minimum_target_lane_object_gap_m"]) >= 15.0]
    removed = [row for row in original if row not in retained]
    deficits = Counter(row["direction"] for row in removed)
    inventory = [row for row in read_csv(args.refined_inventory) if row.get("eligible", "").lower() == "true"]
    inventory = [row for row in inventory if reference_pass(row)]
    original_logs = {row["log_name"] for row in original}
    original_tokens = {row["scenario_token"] for row in original}
    candidates = [row for row in inventory if row["log_name"] not in original_logs and row["scenario_token"] not in original_tokens]
    log_counts = Counter(row["log_name"] for row in candidates)
    replacements: List[Dict[str, str]] = []
    replacement_logs = set()
    for direction in ("left", "right"):
        pool = [row for row in candidates if row["direction"] == direction]
        pool.sort(key=lambda row: (
            log_counts[row["log_name"]],
            -min(float(row["minimum_target_lane_object_gap_m"]), 999.0),
            -float(row["paired_reference_remaining_m"]),
            row["scenario_token"],
        ))
        for row in pool:
            if len([x for x in replacements if x["direction"] == direction]) >= deficits[direction]:
                break
            if row["log_name"] not in replacement_logs:
                replacements.append(row); replacement_logs.add(row["log_name"])
    if len(replacements) != len(removed):
        raise ValueError("insufficient replacements for direction quotas")
    final = retained + replacements
    final.sort(key=lambda row: (row["direction"] != "left", row["scenario_token"]))
    if len(final) != 24 or len({row["log_name"] for row in final}) != 24:
        raise AssertionError("refined roster must contain 24 unique logs")
    if Counter(row["direction"] for row in final) != Counter({"left": 6, "right": 18}):
        raise AssertionError("refined roster direction quota changed")
    if len(original_tokens | {row["scenario_token"] for row in replacements}) > 32:
        raise AssertionError("development token budget exceeded")
    for order, row in enumerate(final, start=1):
        row["development_order"] = order
        row["selection_status"] = "FINAL_DEVELOPMENT_ONLY_PRETREATMENT_REFINED"
        row["replacement_after_safety_refinement"] = row in replacements
    fields = sorted({key for row in final for key in row})
    roster_path = args.output_dir / "final_development_roster.csv"
    write_csv(roster_path, final, fields)
    manifest_path = args.output_dir / "final_development_maneuver_manifest.json"
    write_manifest(manifest_path, final, sha256_file(args.refined_inventory))
    ledger = read_csv(args.prior_exclusion_ledger)
    for row in replacements:
        ledger.append({
            "scenario_token": row["scenario_token"], "log_name": row["log_name"],
            "exclusion_reason": "STAGE7L_B_REFINED_DEVELOPMENT_REPLACEMENT_PERMANENT_EXCLUSION",
            "source_path": str(roster_path.resolve()),
        })
    ledger = list({row["scenario_token"]: row for row in ledger}.values())
    ledger_path = args.output_dir / "stage7l_b_final_prior_exclusion_ledger.csv"
    write_csv(ledger_path, sorted(ledger, key=lambda row: row["scenario_token"]),
              ["scenario_token", "log_name", "exclusion_reason", "source_path"])
    all_development_logs = original_logs | {row["log_name"] for row in replacements}
    remaining = [row for row in inventory if row["log_name"] not in all_development_logs]
    summary = {
        "schema_version": "stage7l_b_refined_development_roster_freeze_v1",
        "status": "FROZEN_FINAL_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "selection_inputs": "pre-treatment inventory plus development-only mechanism/safety gate",
        "embedding_or_bdd_read": False,
        "original_unique_tokens": 24,
        "replacement_unique_tokens": len(replacements),
        "total_unique_development_tokens": len(original_tokens | {row["scenario_token"] for row in replacements}),
        "maximum_total_unique_development_tokens": 32,
        "removed_tokens": [row["scenario_token"] for row in removed],
        "replacement_tokens": [row["scenario_token"] for row in replacements],
        "uniform_minimum_target_lane_object_gap_m": 15.0,
        "final_scenarios": 24, "final_logs": 24, "left": 6, "right": 18,
        "remaining_fresh_eligible_tokens": len(remaining),
        "remaining_fresh_eligible_logs": len({row["log_name"] for row in remaining}),
        "remaining_left": sum(row["direction"] == "left" for row in remaining),
        "remaining_right": sum(row["direction"] == "right" for row in remaining),
        "stage7l_c_target_80_token_inventory_feasible": len(remaining) >= 80,
        "roster_sha256": sha256_file(roster_path),
        "manifest_sha256": sha256_file(manifest_path),
        "exclusion_ledger_sha256": sha256_file(ledger_path),
        "refined_inventory_sha256": sha256_file(args.refined_inventory),
    }
    (args.output_dir / "refined_development_roster_freeze_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_roster", type=Path, required=True)
    parser.add_argument("--refined_inventory", type=Path, required=True)
    parser.add_argument("--prior_exclusion_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
