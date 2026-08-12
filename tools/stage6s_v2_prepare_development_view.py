#!/usr/bin/env python3
"""Prepare a validated Stage6S-v2 development trajectory view."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6j_prepare_pure_longitudinal_view import prepare_view, read_csv, read_json
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file


PLANNERS = ["pdm_closed_interaction_short_headway_v2", "pdm_closed_interaction_long_headway_v2"]
LEDGER_FIELDS = [
    "global_scenario_index", "collection_order", "source_global_scenario_index", "task",
    "source_task", "scenario_type", "log_name", "scenario_token", "db_file", "attempt",
    "stage7c_output_dir", "simulated_ego_seq_sha256", "simulated_ego_seq_mask_sha256",
    "scenario_planner_index_sha256", "official_msgpack_count", "official_msgpack_size_bytes",
    "stage7c_audit_pass",
]


def run(args: argparse.Namespace) -> dict:
    freeze = read_json(args.freeze_manifest)
    batch = read_json(args.batch_manifest)
    state = read_json(args.batch_state)
    locked = read_csv(args.locked_scenarios_csv)
    statuses = read_csv(args.batch_status_csv)
    expected = int(freeze["scenario_count"])
    if freeze.get("status") != "DEVELOPMENT_FROZEN_MECHANISM_TUNING_ALLOWED_NO_EMBEDDING":
        raise ValueError("Stage6S-v2 development freeze status changed")
    if freeze.get("embedding_or_bdd_read") is not False or freeze.get("confirmation_roster_read") is not False:
        raise ValueError("Stage6S-v2 development blinding contract failed")
    if batch.get("schema_version") != "stage6s_v2_development_batch_v1":
        raise ValueError("unexpected Stage6S-v2 batch schema")
    if batch.get("planners") != PLANNERS or batch.get("full_embedding_or_bdd_read") is not False:
        raise ValueError("Stage6S-v2 batch planner/blinding contract changed")
    if batch.get("locked_scenarios_sha256") != sha256_file(args.locked_scenarios_csv):
        raise ValueError("development roster changed after batch freeze")
    if state.get("counts") != {"SUCCEEDED": expected, "FAILED_REVIEW_REQUIRED": 0, "PENDING": 0}:
        raise ValueError(f"Stage6S-v2 rollout batch is incomplete: {state.get('counts')}")
    by_order = {int(row["collection_order"]): row for row in statuses}
    rows = []
    for frozen in locked:
        status = by_order[int(frozen["collection_order"])]
        if status["status"] != "SUCCEEDED":
            raise ValueError("non-successful Stage6S-v2 development rollout")
        rows.append({**frozen, **status})
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    summary = prepare_view(
        rows, args.output_dir, expected_planners=PLANNERS,
        ledger_filename="stage6s_v2_development_scenario_ledger.csv", ledger_fields=LEDGER_FIELDS,
        stage_label="6S-v2 development trajectory view", schema_version="stage6s_v2_development_view_v1",
    )
    result = {
        **summary, "status": "STAGE6S_V2_DEVELOPMENT_VIEW_READY", "full_embedding_or_bdd_read": False,
        "confirmation_roster_read": False, "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "batch_manifest_sha256": sha256_file(args.batch_manifest), "batch_state_sha256": sha256_file(args.batch_state),
        "batch_status_sha256": sha256_file(args.batch_status_csv),
    }
    (args.output_dir / "stage6s_v2_development_view_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--locked_scenarios_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--batch_state", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
