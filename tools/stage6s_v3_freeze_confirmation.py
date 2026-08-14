#!/usr/bin/env python3
"""Prospectively freeze Stage6S-v3 after applying only official scene runnability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage6s_v2_freeze_development import LOCKED_FIELDS, validate_planners  # noqa: E402
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def stable_rank(salt: str, row: dict[str, str]) -> str:
    return hashlib.sha256(f"{salt}:{row['log_name']}:{row['scenario_token']}".encode()).hexdigest()


def token_set(rows: list[dict[str, str]]) -> set[str]:
    return {row["scenario_token"] for row in rows}


def log_set(rows: list[dict[str, str]]) -> set[str]:
    return {row["log_name"] for row in rows}


def scene_boundary(db_path: Path, token: str) -> dict[str, Any]:
    query = """
        WITH ordered_scenes AS (
            SELECT token, name, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num FROM scene
        ), num_scenes AS (SELECT COUNT(*) AS cnt FROM scene)
        SELECT o.row_num, n.cnt AS scene_count, lower(hex(lp.scene_token)) AS db_scene_token
        FROM lidar_pc AS lp
        INNER JOIN ordered_scenes AS o ON o.token = lp.scene_token
        CROSS JOIN num_scenes AS n
        WHERE lp.token = ?
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(query, (bytes.fromhex(token),)).fetchone()
    if row is None:
        raise ValueError(f"scenario token is absent from DB: {db_path.name}:{token}")
    return dict(row)


def official_runnability(
    rows: list[dict[str, str]], db_root: Path, devkit_root: Path
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    sys.path.insert(0, str(devkit_root))
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["db_file"]].append(row)
    runnable: dict[str, bool] = {}
    boundaries: dict[str, dict[str, Any]] = {}
    for db_file, group in grouped.items():
        db_path = db_root / db_file
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        requested = [row["scenario_token"] for row in group]
        official = {
            result["token"].hex()
            for result in get_scenarios_from_db(
                str(db_path), requested, None, None,
                include_invalid_mission_goals=True, include_cameras=False,
            )
        }
        for row in group:
            token = row["scenario_token"]
            runnable[token] = token in official
            boundaries[token] = scene_boundary(db_path, token)
            boundary = boundaries[token]
            expected = boundary["row_num"] >= 3 and boundary["row_num"] < boundary["scene_count"] - 1
            if expected != runnable[token]:
                raise ValueError(f"official query/boundary disagreement for {token}")
    return runnable, boundaries


def round_robin_select(rows: list[dict[str, str]], salt: str, target: int) -> list[dict[str, str]]:
    by_log: dict[str, deque[dict[str, str]]] = {}
    for log_name in sorted({row["log_name"] for row in rows}, key=lambda log: hashlib.sha256(f"{salt}:{log}".encode()).hexdigest()):
        ranked = sorted((row for row in rows if row["log_name"] == log_name), key=lambda row: stable_rank(salt, row))
        by_log[log_name] = deque(ranked)
    selected: list[dict[str, str]] = []
    while len(selected) < target and any(by_log.values()):
        for log_name in list(by_log):
            if by_log[log_name] and len(selected) < target:
                selected.append(by_log[log_name].popleft())
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = read_json(args.stage6s_v2_config)
    repair = read_json(args.repair_config)
    if sha256_file(args.stage6s_v2_config) != repair["base_stage6s_v2_config_sha256"]:
        raise ValueError("Stage6S-v2 base config changed")
    inventory_summary = read_json(args.inventory_summary)
    development_manifest = read_json(args.development_manifest)
    development_mechanism = read_json(args.development_mechanism)
    v2_confirmation_manifest = read_json(args.stage6s_v2_confirmation_manifest)
    v2_execution_failure = read_json(args.stage6s_v2_execution_failure)
    if inventory_summary.get("status") != "PRETREATMENT_INTERACTION_INVENTORY_READY":
        raise ValueError("pre-treatment inventory is not ready")
    if inventory_summary.get("planner_outcome_read") is not False or inventory_summary.get("embedding_or_bdd_read") is not False:
        raise ValueError("inventory is not outcome blind")
    if development_manifest.get("embedding_or_bdd_read") is not False:
        raise ValueError("development freeze blinding contract failed")
    if development_mechanism.get("status") != "DEVELOPMENT_MECHANISM_PASS_CONFIRMATION_FREEZE_ALLOWED":
        raise ValueError("Stage6S-v2 development mechanism did not pass")
    if development_mechanism.get("embedding_or_bdd_read") is not False:
        raise ValueError("development mechanism used forbidden representation results")
    if v2_confirmation_manifest.get("status") != "CONFIRMATION_ROSTER_FROZEN_NOT_RUN":
        raise ValueError("Stage6S-v2 confirmation freeze changed")
    if v2_execution_failure.get("status") != "CONFIRMATION_EXECUTION_INCOMPLETE_STOP_NO_MECHANISM_OR_EMBEDDING":
        raise ValueError("Stage6S-v2 failure record changed")
    planners, differences = validate_planners(base)

    inventory = [row for row in read_csv(args.inventory_csv) if row.get("eligible", "").lower() == "true"]
    v1_rows = read_csv(args.stage6s_v1_roster)
    development_rows = read_csv(args.development_roster)
    v2_confirmation_rows = read_csv(args.stage6s_v2_confirmation_roster)
    v1_tokens = token_set(v1_rows)
    development_tokens = token_set(development_rows)
    development_logs = log_set(development_rows)
    v2_confirmation_tokens = token_set(v2_confirmation_rows)
    v2_confirmation_logs = log_set(v2_confirmation_rows)
    counts = {"eligible_before_exclusions": len(inventory)}
    candidates = [row for row in inventory if row["scenario_token"] not in v1_tokens]
    counts["after_stage6s_v1_token_exclusion"] = len(candidates)
    candidates = [row for row in candidates if row["scenario_token"] not in development_tokens]
    counts["after_stage6s_v2_development_token_exclusion"] = len(candidates)
    candidates = [row for row in candidates if row["log_name"] not in development_logs]
    counts["after_stage6s_v2_development_log_exclusion"] = len(candidates)
    candidates = [row for row in candidates if row["scenario_token"] not in v2_confirmation_tokens]
    counts["after_all_stage6s_v2_confirmation_token_exclusion"] = len(candidates)

    runnable, boundaries = official_runnability(candidates, args.nuplan_db_root, args.nuplan_devkit_root)
    runnable_rows = [row for row in candidates if runnable[row["scenario_token"]]]
    counts["after_official_scene_runnability"] = len(runnable_rows)
    target = int(repair["selection"]["target_pair_count"])
    selected = round_robin_select(runnable_rows, base["confirmation"]["stable_rank_salt"], target)
    if len(selected) < int(repair["selection"]["minimum_pair_count"]):
        raise ValueError("runnable inventory is below the frozen minimum")
    if len(selected) != target:
        raise ValueError("could not freeze the target of 80 pairs")
    selected_tokens = token_set(selected)
    selected_logs = log_set(selected)
    if selected_tokens & (v1_tokens | development_tokens | v2_confirmation_tokens):
        raise ValueError("new confirmation roster reuses an excluded scenario")
    if selected_logs & development_logs:
        raise ValueError("new confirmation roster overlaps development logs")
    if not all(runnable[token] for token in selected_tokens):
        raise ValueError("new confirmation roster is not 100% officially runnable")

    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    audit_path = output / "stage6s_v3_official_runnability_audit.csv"
    audit_rows = []
    for row in candidates:
        boundary = boundaries[row["scenario_token"]]
        audit_rows.append({
            "log_name": row["log_name"], "db_file": row["db_file"],
            "scenario_token": row["scenario_token"], "scenario_type": row["scenario_type"],
            "db_scene_token": boundary["db_scene_token"], "scene_row_num": boundary["row_num"],
            "scene_count": boundary["scene_count"], "official_query_runnable": runnable[row["scenario_token"]],
            "selected_for_stage6s_v3": row["scenario_token"] in selected_tokens,
        })
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0])); writer.writeheader(); writer.writerows(audit_rows)

    roster_path = output / "stage6s_v3_confirmation_roster.csv"
    roster_rows = []
    for index, row in enumerate(selected, start=1):
        roster_rows.append({
            "collection_order": index, "source_global_scenario_index": index - 1,
            "task": "following_interaction_confirmation_v3",
            "source_task": "following_interaction_confirmation_v3",
            "scenario_type": row["scenario_type"], "log_name": row["log_name"],
            "scenario_token": row["scenario_token"],
            "scene_token": boundaries[row["scenario_token"]]["db_scene_token"],
            "db_file": row["db_file"],
            "selection_role": "STAGE6S_V3_CONFIRMATION_FROZEN_NOT_RUN",
        })
    with roster_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCKED_FIELDS); writer.writeheader(); writer.writerows(roster_rows)

    v2_design = read_json(args.stage6s_v2_confirmation_design)
    frozen_core_keys = [
        "planner_a", "planner_b", "shared_parameters", "treatment_parameters", "inventory_rule",
        "mechanism_metrics", "mechanism_gate", "thw_definition", "statistics",
    ]
    frozen_core = {key: v2_design[key] for key in frozen_core_keys}
    design = {
        **frozen_core,
        "schema_version": "stage6s_v3_interaction_confirmation_frozen_design_v1",
        "prospective_repair": repair,
        "confirmation_selection_exclusions": [
            "stage6s_v1_scenario_tokens", "stage6s_v2_development_scenario_tokens",
            "stage6s_v2_development_logs", "all_stage6s_v2_confirmation_scenario_tokens",
        ],
        "representation_primary_endpoint": "C full-context minus C neighbor-zero null-standardized delta-Z; log-cluster bootstrap 95% CI lower bound > 0",
        "representation_evaluation_condition": "ONLY_IF_CONFIRMATION_MECHANISM_GATE_PASSES",
    }
    design_path = output / "stage6s_v3_confirmation_frozen_design.json"
    design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_log = Counter(row["log_name"] for row in selected)
    available_outside_v2_logs = sum(row["log_name"] not in v2_confirmation_logs for row in runnable_rows)
    manifest = {
        "schema_version": "stage6s_v3_confirmation_freeze_v1",
        "status": "STAGE6S_V3_CONFIRMATION_ROSTER_FROZEN_NOT_RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": repair["issue"], "scenario_count": len(selected),
        "distinct_log_count": len(per_log), "scenario_count_by_log": dict(sorted(per_log.items())),
        "scenario_type_counts": dict(Counter(row["scenario_type"] for row in selected)),
        "exclusion_counts": counts,
        "official_runnable_candidate_count": len(runnable_rows),
        "official_nonrunnable_candidate_count": len(candidates) - len(runnable_rows),
        "official_runnability_selected_count": sum(runnable[row["scenario_token"]] for row in selected),
        "official_runnability_selected_fraction": 1.0,
        "stage6s_v1_scenario_overlap_count": len(selected_tokens & v1_tokens),
        "stage6s_v2_development_scenario_overlap_count": len(selected_tokens & development_tokens),
        "stage6s_v2_development_log_overlap_count": len(selected_logs & development_logs),
        "stage6s_v2_confirmation_scenario_overlap_count": len(selected_tokens & v2_confirmation_tokens),
        "stage6s_v2_confirmation_log_overlap_count": len(selected_logs & v2_confirmation_logs),
        "runnable_candidates_outside_stage6s_v2_confirmation_logs": available_outside_v2_logs,
        "stage6s_v2_confirmation_log_disjointness_feasible": available_outside_v2_logs >= int(repair["selection"]["minimum_pair_count"]),
        "log_disjointness_limitation": "No eligible runnable candidates remain outside Stage6S-v2 confirmation logs after mandatory development-log exclusion; all Stage6S-v2 confirmation tokens are nevertheless excluded.",
        "selection_used_planner_outcomes": False, "selection_used_embedding_or_bdd": False,
        "pre_treatment_selection_only": True,
        "planners": planners,
        "planner_fingerprints": {planner: canonical_sha(PLANNER_PROFILES[planner]["parameters"]) for planner in planners},
        "only_different_parameters": differences,
        "unchanged_stage6s_v2_core_sha256": canonical_sha(frozen_core),
        "stage6s_v2_config_sha256": sha256_file(args.stage6s_v2_config),
        "repair_config_sha256": sha256_file(args.repair_config),
        "inventory_summary_sha256": sha256_file(args.inventory_summary),
        "inventory_csv_sha256": sha256_file(args.inventory_csv),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "development_roster_sha256": sha256_file(args.development_roster),
        "development_mechanism_sha256": sha256_file(args.development_mechanism),
        "stage6s_v2_confirmation_manifest_sha256": sha256_file(args.stage6s_v2_confirmation_manifest),
        "stage6s_v2_confirmation_roster_sha256": sha256_file(args.stage6s_v2_confirmation_roster),
        "stage6s_v2_execution_failure_sha256": sha256_file(args.stage6s_v2_execution_failure),
        "stage6s_v2_permanent_status": "confirmation execution failure due to roster runnability omission",
        "nuplan_scenario_query_source_sha256": sha256_file(args.nuplan_scenario_query_source),
        "nuplan_devkit_commit": os.popen(f"git -C {args.nuplan_devkit_root} rev-parse HEAD").read().strip(),
        "confirmation_design_sha256": sha256_file(design_path),
        "confirmation_roster_sha256": sha256_file(roster_path),
        "official_runnability_audit_sha256": sha256_file(audit_path),
        "confirmation_rollouts_launched": False, "mechanism_evaluated": False,
        "embedding_or_bdd_read": False, "checkpoint_training_launched": False,
        "immutable_after_freeze": True,
    }
    manifest_path = output / "stage6s_v3_confirmation_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6s_v2_config", type=Path, required=True)
    parser.add_argument("--repair_config", type=Path, required=True)
    parser.add_argument("--inventory_summary", type=Path, required=True)
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--development_manifest", type=Path, required=True)
    parser.add_argument("--development_roster", type=Path, required=True)
    parser.add_argument("--development_mechanism", type=Path, required=True)
    parser.add_argument("--stage6s_v1_roster", type=Path, required=True)
    parser.add_argument("--stage6s_v2_confirmation_manifest", type=Path, required=True)
    parser.add_argument("--stage6s_v2_confirmation_roster", type=Path, required=True)
    parser.add_argument("--stage6s_v2_confirmation_design", type=Path, required=True)
    parser.add_argument("--stage6s_v2_execution_failure", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--nuplan_scenario_query_source", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
