#!/usr/bin/env python3
"""Freeze the Stage6S-v2 confirmation execution outcome without reading embeddings.

This deliberately treats a partially runnable locked roster as an execution failure.
It never creates a post-hoc confirmation subset and never unlocks representation
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def classify_failure(attempt_dir: Path) -> str:
    texts: list[str] = []
    for path in sorted(attempt_dir.rglob("*.log")):
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    merged = "\n".join(texts)
    if "No scenarios found to simulate!" in merged:
        return "NUPLAN_NO_SCENARIOS_FOUND"
    if "timed out" in merged.lower() or "timeout" in merged.lower():
        return "TIMEOUT"
    if "out of memory" in merged.lower() or "oom" in merged.lower():
        return "OOM"
    return "OTHER_OFFICIAL_COMMAND_FAILURE"


def scene_boundary_audit(db_file: Path, scenario_token: str) -> dict[str, Any]:
    """Mirror nuPlan's valid_scenes rank rule used by get_scenarios_from_db."""
    with sqlite3.connect(db_file) as connection:
        row = connection.execute(
            "SELECT scene_token FROM lidar_pc WHERE lower(hex(token)) = ?", (scenario_token.lower(),)
        ).fetchone()
        if row is None:
            return {"scenario_token_found": False}
        scenes = [item[0] for item in connection.execute("SELECT token FROM scene ORDER BY name ASC")]
    rank = scenes.index(row[0]) + 1
    count = len(scenes)
    # nuPlan: row_num >= 3 AND row_num < n.cnt - 1.
    valid = rank >= 3 and rank < count - 1
    return {
        "scenario_token_found": True,
        "scene_rank_1_based": rank,
        "scene_count": count,
        "passes_nuplan_valid_scenes_rule": valid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256(AUTH) != AUTH_SHA:
        raise ValueError("blind authorization changed")
    authorization = read_json(AUTH)
    if authorization.get("stage6s_v2_representation_evaluation_condition") != "ONLY_IF_CONFIRMATION_MECHANISM_GATE_PASSES":
        raise ValueError("conditional representation rule changed")

    freeze = read_json(args.freeze_manifest)
    batch = read_json(args.batch_manifest)
    state = read_json(args.batch_state)
    roster = read_csv(args.locked_scenarios_csv)
    statuses = read_csv(args.batch_status_csv)
    expected = int(freeze["scenario_count"])
    if expected != len(roster) or expected != len(statuses):
        raise ValueError("locked roster/status cardinality mismatch")
    if batch.get("locked_scenarios_sha256") != sha256(args.locked_scenarios_csv):
        raise ValueError("locked confirmation roster changed")
    if int(state.get("total_scenarios", -1)) != expected or int(state.get("counts", {}).get("PENDING", -1)) != 0:
        raise ValueError("confirmation execution has not reached a terminal state")

    failure_rows: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for row in statuses:
        if row["status"] == "SUCCEEDED":
            continue
        rollout_dir = args.run_dir / "rollouts" / f"order_{int(row['collection_order']):04d}_{row['scenario_token']}"
        attempts = sorted(rollout_dir.glob("attempt_*"))
        latest = attempts[-1] if attempts else rollout_dir
        category = classify_failure(latest)
        boundary = scene_boundary_audit(args.nuplan_db_root / row["db_file"], row["scenario_token"])
        if category == "NUPLAN_NO_SCENARIOS_FOUND" and boundary.get("passes_nuplan_valid_scenes_rule") is False:
            category = "NUPLAN_VALID_SCENES_BOUNDARY_EXCLUSION"
        categories[category] += 1
        failure_rows.append({
            "collection_order": int(row["collection_order"]),
            "scenario_token": row["scenario_token"],
            "log_name": row["log_name"],
            "scenario_type": row["scenario_type"],
            "attempt_count": len(attempts),
            "failure_category": category,
            **boundary,
        })

    succeeded = int(state["counts"]["SUCCEEDED"])
    failed = int(state["counts"]["FAILED_REVIEW_REQUIRED"])
    complete = succeeded == expected and failed == 0
    result: dict[str, Any] = {
        "schema_version": "stage6v_stage6s_v2_confirmation_execution_freeze_v1",
        "status": (
            "CONFIRMATION_EXECUTION_COMPLETE_MECHANISM_EVALUATION_AUTHORIZED"
            if complete else
            "CONFIRMATION_EXECUTION_INCOMPLETE_STOP_NO_MECHANISM_OR_EMBEDDING"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_roster_count": expected,
        "succeeded": succeeded,
        "failed_review_required": failed,
        "pending": 0,
        "complete_fraction": succeeded / expected,
        "failure_categories": dict(sorted(categories.items())),
        "failure_rows": failure_rows,
        "retry_performed_for_all_failures": bool(failure_rows) and all(row["attempt_count"] >= 2 for row in failure_rows),
        "post_hoc_complete_case_subset_created": False,
        "mechanism_evaluation_authorized": complete,
        "embedding_or_bdd_evaluation_authorized": False,
        "embedding_or_bdd_read": False,
        "training_or_protocol_modified": False,
        "benchmark_or_roster_modified": False,
        "decision_note": (
            "The frozen 80-pair roster must complete as locked. Successful cases are not redefined post hoc as a new confirmation set."
        ),
        "authorization_sha256": AUTH_SHA,
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "locked_scenarios_sha256": sha256(args.locked_scenarios_csv),
        "batch_manifest_sha256": sha256(args.batch_manifest),
        "batch_state_sha256": sha256(args.batch_state),
        "batch_status_sha256": sha256(args.batch_status_csv),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "stage6s_v2_confirmation_execution_freeze.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["manifest_sha256"] = sha256(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--locked_scenarios_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--batch_state", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
