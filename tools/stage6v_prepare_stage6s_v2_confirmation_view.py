#!/usr/bin/env python3
"""Prepare the locked Stage6S-v2 confirmation trajectory view."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage6j_prepare_pure_longitudinal_view import prepare_view, read_csv, read_json  # noqa: E402
from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402

PLANNERS = ["pdm_closed_interaction_short_headway_v2", "pdm_closed_interaction_long_headway_v2"]
AUTH = ROOT / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"
LEDGER_FIELDS = [
    "global_scenario_index", "collection_order", "source_global_scenario_index", "task",
    "source_task", "scenario_type", "log_name", "scenario_token", "db_file", "attempt",
    "stage7c_output_dir", "simulated_ego_seq_sha256", "simulated_ego_seq_mask_sha256",
    "scenario_planner_index_sha256", "official_msgpack_count", "official_msgpack_size_bytes",
    "stage7c_audit_pass",
]


def run(args: argparse.Namespace) -> dict:
    if sha256_file(AUTH) != AUTH_SHA:
        raise ValueError("blind authorization changed")
    freeze = read_json(args.freeze_manifest)
    batch = read_json(args.batch_manifest)
    state = read_json(args.batch_state)
    locked = read_csv(args.locked_scenarios_csv)
    statuses = read_csv(args.batch_status_csv)
    expected = int(freeze["scenario_count"])
    if freeze.get("status") != "CONFIRMATION_ROSTER_FROZEN_NOT_RUN" or freeze.get("embedding_or_bdd_read") is not False:
        raise ValueError("confirmation freeze/blind state changed")
    if batch.get("schema_version") != "stage6s_v2_confirmation_batch_v1" or batch.get("planners") != PLANNERS:
        raise ValueError("unexpected confirmation batch")
    if batch.get("locked_scenarios_sha256") != sha256_file(args.locked_scenarios_csv):
        raise ValueError("confirmation roster changed")
    if state.get("counts") != {"SUCCEEDED": expected, "FAILED_REVIEW_REQUIRED": 0, "PENDING": 0}:
        raise ValueError(f"confirmation batch incomplete: {state.get('counts')}")
    by_order = {int(row["collection_order"]): row for row in statuses}
    rows = []
    for frozen in locked:
        status = by_order[int(frozen["collection_order"])]
        if status["status"] != "SUCCEEDED":
            raise ValueError("non-successful confirmation rollout")
        rows.append({**frozen, **status})
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    summary = prepare_view(
        rows, args.output_dir, expected_planners=PLANNERS,
        ledger_filename="stage6s_v2_confirmation_scenario_ledger.csv", ledger_fields=LEDGER_FIELDS,
        stage_label="6S-v2 confirmation trajectory view", schema_version="stage6s_v2_confirmation_view_v1",
    )
    result = {
        **summary, "status": "STAGE6S_V2_CONFIRMATION_VIEW_READY", "embedding_or_bdd_read": False,
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest), "batch_manifest_sha256": sha256_file(args.batch_manifest),
        "batch_state_sha256": sha256_file(args.batch_state), "batch_status_sha256": sha256_file(args.batch_status_csv),
        "blind_authorization_sha256": AUTH_SHA,
    }
    (args.output_dir / "stage6s_v2_confirmation_view_summary.json").write_text(json.dumps(result, indent=2) + "\n")
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
