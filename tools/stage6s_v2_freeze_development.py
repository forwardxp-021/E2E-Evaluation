#!/usr/bin/env python3
"""Freeze Stage6S-v2 development scenarios from pre-treatment inventory only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


LOCKED_FIELDS = [
    "collection_order", "source_global_scenario_index", "task", "source_task", "scenario_type",
    "log_name", "scenario_token", "scene_token", "db_file", "selection_role",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_rank(salt: str, row: dict[str, str]) -> str:
    return hashlib.sha256(f"{salt}:{row['log_name']}:{row['scenario_token']}".encode()).hexdigest()


def validate_planners(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    planners = [config["planner_a"], config["planner_b"]]
    allowed_differences = {"idm_policies.min_gap_to_lead_agent", "idm_policies.headway_time"}
    parameters = {}
    for planner in planners:
        parameters[planner] = {
            key: value for key, value in PLANNER_PROFILES[planner]["parameters"].items()
            if key not in {"source", "checkpoint_required"}
        }
        for key, expected in config["shared_parameters"].items():
            if parameters[planner].get(key) != expected:
                raise ValueError(f"shared planner parameter differs: {planner}/{key}")
        for key, expected in config["treatment_parameters"][planner].items():
            if parameters[planner].get(key) != expected:
                raise ValueError(f"treatment parameter differs: {planner}/{key}")
    differences = sorted(
        key for key in set(parameters[planners[0]]) | set(parameters[planners[1]])
        if parameters[planners[0]].get(key) != parameters[planners[1]].get(key)
    )
    if set(differences) != allowed_differences:
        raise ValueError(f"Stage6S-v2 treatment differs outside frozen interaction parameters: {differences}")
    return planners, differences


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    summary = read_json(args.inventory_summary)
    if summary.get("status") != "PRETREATMENT_INTERACTION_INVENTORY_READY":
        raise ValueError("pre-treatment interaction inventory is not ready")
    if summary.get("planner_outcome_read") is not False or summary.get("embedding_or_bdd_read") is not False:
        raise ValueError("pre-treatment inventory blinding contract failed")
    rows = [row for row in read_csv(args.inventory_csv) if row["eligible"].lower() == "true"]
    old_tokens = {row["scenario_token"] for row in read_csv(args.stage6s_v1_roster)}
    rows = [row for row in rows if row["scenario_token"] not in old_tokens]
    dev = config["development"]
    priority = {value: index for index, value in enumerate(config["inventory"]["allowed_scenario_types"])}
    rows.sort(key=lambda row: (
        priority[row["scenario_type"]], -float(row["front_exposure_ratio"]),
        -float(row["closing_pressure_ratio"]), stable_rank(dev["stable_rank_salt"], row),
    ))
    selected = []
    per_log = Counter()
    for row in rows:
        if per_log[row["log_name"]] >= int(dev["maximum_scenarios_per_log"]):
            continue
        selected.append(row); per_log[row["log_name"]] += 1
        if len(selected) == int(dev["pair_count"]):
            break
    if len(selected) != int(dev["pair_count"]):
        raise ValueError(f"only {len(selected)} development scenarios satisfy frozen selection")
    if len(per_log) < int(dev["minimum_distinct_logs"]):
        raise ValueError(f"development log diversity failed: {dict(per_log)}")
    planners, differences = validate_planners(config)
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    locked = output / "stage6s_v2_development_roster.csv"
    locked_rows = []
    for index, row in enumerate(selected, start=1):
        locked_rows.append({
            "collection_order": index, "source_global_scenario_index": index - 1,
            "task": "following_interaction_v2", "source_task": "following_interaction_v2",
            "scenario_type": row["scenario_type"], "log_name": row["log_name"],
            "scenario_token": row["scenario_token"], "scene_token": row["scenario_token"],
            "db_file": row["db_file"], "selection_role": "STAGE6S_V2_DEVELOPMENT",
        })
    with locked.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCKED_FIELDS)
        writer.writeheader(); writer.writerows(locked_rows)
    manifest = {
        "schema_version": "stage6s_v2_development_freeze_v1",
        "status": "DEVELOPMENT_FROZEN_MECHANISM_TUNING_ALLOWED_NO_EMBEDDING",
        "created_utc": datetime.now(timezone.utc).isoformat(), "issue": 261,
        "scenario_count": len(selected), "distinct_log_count": len(per_log),
        "scenario_count_by_log": dict(sorted(per_log.items())),
        "scenario_type_counts": dict(Counter(row["scenario_type"] for row in selected)),
        "excluded_stage6s_v1_token_count": len(old_tokens),
        "stage6s_v1_token_overlap_count": len({row["scenario_token"] for row in selected} & old_tokens),
        "config_sha256": sha256_file(args.config), "inventory_summary_sha256": sha256_file(args.inventory_summary),
        "inventory_csv_sha256": sha256_file(args.inventory_csv), "development_roster_sha256": sha256_file(locked),
        "planners": planners,
        "planner_fingerprints": {planner: canonical_hash(PLANNER_PROFILES[planner]["parameters"]) for planner in planners},
        "only_different_parameters": differences, "mechanism_gate": config["mechanism_gate"],
        "thw_definition": config["thw_definition"], "mechanism_metrics": config["mechanism_metrics"],
        "pre_treatment_selection_only": True, "planner_outcome_read_for_selection": False,
        "embedding_or_bdd_read": False, "checkpoint_training_launched": False,
        "confirmation_roster_read": False,
    }
    (output / "stage6s_v2_development_freeze_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory_summary", type=Path, required=True)
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--stage6s_v1_roster", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
