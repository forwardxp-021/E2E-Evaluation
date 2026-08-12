#!/usr/bin/env python3
"""Freeze an outcome-blind Stage6S-v2 confirmation roster after development PASS."""

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

from tools.stage6s_v2_freeze_development import LOCKED_FIELDS, validate_planners
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_rank(salt: str, row: dict[str, str]) -> str:
    return hashlib.sha256(f"{salt}:{row['log_name']}:{row['scenario_token']}".encode()).hexdigest()


def select_confirmation(
    eligible_rows: list[dict[str, str]],
    development_rows: list[dict[str, str]],
    stage6s_v1_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    confirmation = config["confirmation"]
    development_logs = {row["log_name"] for row in development_rows}
    development_tokens = {row["scenario_token"] for row in development_rows}
    old_tokens = {row["scenario_token"] for row in stage6s_v1_rows}
    rows = [row for row in eligible_rows if row.get("eligible", "").lower() == "true"]
    counts = {"eligible_before_exclusions": len(rows)}
    rows = [row for row in rows if row["log_name"] not in development_logs]
    counts["after_development_log_exclusion"] = len(rows)
    rows = [row for row in rows if row["scenario_token"] not in development_tokens]
    counts["after_development_scenario_exclusion"] = len(rows)
    # Stage6S-v1 tokens are unconditionally excluded: the v2 benchmark must not
    # reuse the 24 historical development outcomes even though confirmation
    # selection itself remains pre-treatment-only.
    rows = [row for row in rows if row["scenario_token"] not in old_tokens]
    counts["after_stage6s_v1_token_exclusion"] = len(rows)
    rows.sort(key=lambda row: stable_rank(confirmation["stable_rank_salt"], row))
    target = int(confirmation["target_pair_count"])
    selected = rows[:target]
    if len(selected) < int(confirmation["minimum_pair_count"]):
        raise ValueError(
            f"confirmation inventory below frozen minimum: selected={len(selected)}, "
            f"minimum={confirmation['minimum_pair_count']}"
        )
    if len(selected) > int(confirmation["maximum_pair_count"]):
        raise ValueError("confirmation selection exceeds frozen maximum")
    return selected, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.config)
    inventory_summary = read_json(args.inventory_summary)
    development_manifest = read_json(args.development_manifest)
    mechanism = read_json(args.development_mechanism)
    if inventory_summary.get("status") != "PRETREATMENT_INTERACTION_INVENTORY_READY":
        raise ValueError("pre-treatment inventory is not ready")
    if inventory_summary.get("planner_outcome_read") is not False or inventory_summary.get("embedding_or_bdd_read") is not False:
        raise ValueError("inventory is not outcome blind")
    if development_manifest.get("embedding_or_bdd_read") is not False:
        raise ValueError("development freeze blinding contract failed")
    if mechanism.get("status") != "DEVELOPMENT_MECHANISM_PASS_CONFIRMATION_FREEZE_ALLOWED":
        raise ValueError("development mechanism has not passed")
    if mechanism.get("embedding_or_bdd_read") is not False:
        raise ValueError("development mechanism read a forbidden representation result")
    planners, differences = validate_planners(config)
    development_rows = read_csv(args.development_roster)
    old_rows = read_csv(args.stage6s_v1_roster)
    selected, exclusion_counts = select_confirmation(
        read_csv(args.inventory_csv), development_rows, old_rows, config
    )
    development_logs = {row["log_name"] for row in development_rows}
    development_tokens = {row["scenario_token"] for row in development_rows}
    old_tokens = {row["scenario_token"] for row in old_rows}
    selected_logs = {row["log_name"] for row in selected}
    selected_tokens = {row["scenario_token"] for row in selected}
    if selected_logs & development_logs or selected_tokens & development_tokens:
        raise ValueError("confirmation is not development-disjoint")
    if selected_tokens & old_tokens:
        raise ValueError("confirmation reuses a Stage6S-v1 scenario")

    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    roster_path = output / "stage6s_v2_confirmation_roster.csv"
    locked_rows = []
    for index, row in enumerate(selected, start=1):
        locked_rows.append({
            "collection_order": index,
            "source_global_scenario_index": index - 1,
            "task": "following_interaction_confirmation_v2",
            "source_task": "following_interaction_confirmation_v2",
            "scenario_type": row["scenario_type"],
            "log_name": row["log_name"],
            "scenario_token": row["scenario_token"],
            "scene_token": row["scenario_token"],
            "db_file": row["db_file"],
            "selection_role": "STAGE6S_V2_CONFIRMATION_FROZEN_NOT_RUN",
        })
    with roster_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCKED_FIELDS)
        writer.writeheader()
        writer.writerows(locked_rows)

    design = {
        "planner_a": config["planner_a"],
        "planner_b": config["planner_b"],
        "shared_parameters": config["shared_parameters"],
        "treatment_parameters": config["treatment_parameters"],
        "inventory_rule": config["inventory"],
        "confirmation_selection": config["confirmation"],
        "confirmation_selection_exclusions": [
            "development_logs", "development_scenario_tokens", "stage6s_v1_scenario_tokens"
        ],
        "mechanism_metrics": config["mechanism_metrics"],
        "mechanism_gate": config["mechanism_gate"],
        "thw_definition": config["thw_definition"],
        "statistics": {
            "paired_mechanism_summary": "pair-level statistic then across-pair median",
            "directional_stability": "fraction of complete pairs with predeclared direction",
            "confirmation_decision_rule": "apply the frozen mechanism_gate unchanged; all control gates and at least two interaction metrics must pass",
            "uncertainty": "log-cluster bootstrap percentile 95% interval",
            "cluster_unit": "log_name",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 620261,
            "missingness": "metric-specific complete pairs; never impute THW sentinel/cap",
            "embedding_or_bdd_evaluation": "not part of benchmark freeze",
        },
    }
    design_path = output / "stage6s_v2_confirmation_frozen_design.json"
    design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_log = Counter(row["log_name"] for row in selected)
    manifest = {
        "schema_version": "stage6s_v2_confirmation_freeze_v1",
        "status": "CONFIRMATION_ROSTER_FROZEN_NOT_RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": 261,
        "scenario_count": len(selected),
        "target_pair_count": int(config["confirmation"]["target_pair_count"]),
        "minimum_pair_count": int(config["confirmation"]["minimum_pair_count"]),
        "maximum_pair_count": int(config["confirmation"]["maximum_pair_count"]),
        "sample_size_basis": "predeclared midpoint target of 80 within the frozen 60-100 range; 243 eligible candidates remained before Stage6S-v1 token exclusion",
        "distinct_log_count": len(per_log),
        "scenario_count_by_log": dict(sorted(per_log.items())),
        "scenario_type_counts": dict(Counter(row["scenario_type"] for row in selected)),
        "development_log_overlap_count": len(selected_logs & development_logs),
        "development_scenario_overlap_count": len(selected_tokens & development_tokens),
        "stage6s_v1_scenario_overlap_count": len(selected_tokens & old_tokens),
        "exclusion_counts": exclusion_counts,
        "planners": planners,
        "planner_fingerprints": {
            planner: hashlib.sha256(
                json.dumps(PLANNER_PROFILES[planner]["parameters"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for planner in planners
        },
        "only_different_parameters": differences,
        "config_sha256": sha256_file(args.config),
        "inventory_summary_sha256": sha256_file(args.inventory_summary),
        "inventory_csv_sha256": sha256_file(args.inventory_csv),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "development_roster_sha256": sha256_file(args.development_roster),
        "development_mechanism_sha256": sha256_file(args.development_mechanism),
        "confirmation_design_sha256": sha256_file(design_path),
        "confirmation_roster_sha256": sha256_file(roster_path),
        "pre_treatment_selection_only": True,
        "development_mechanism_used_only_as_freeze_authorization": True,
        "planner_outcome_read_for_confirmation_selection": False,
        "confirmation_rollouts_launched": False,
        "embedding_or_bdd_read": False,
        "checkpoint_training_launched": False,
        "new_model_evaluation_launched": False,
        "immutable_after_freeze": True,
    }
    (output / "stage6s_v2_confirmation_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory_summary", type=Path, required=True)
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--development_manifest", type=Path, required=True)
    parser.add_argument("--development_roster", type=Path, required=True)
    parser.add_argument("--development_mechanism", type=Path, required=True)
    parser.add_argument("--stage6s_v1_roster", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
