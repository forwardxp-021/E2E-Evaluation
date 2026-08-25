#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools import stage7_m6_5_prepare_locked_confirmation as m65


SCHEMA_VERSION = "stage6h_expanded_rollout_view_v1"
READY_STATUS = "EXPANDED_490_PAIR_VIEW_READY"
EXPECTED_TASK_COUNTS = {
    "following_interaction": 122,
    "lane_change": 11,
    "stop_go_control": 115,
    "high_motion_dynamics": 122,
    "dense_or_vulnerable_interaction": 120,
}


def validate_inputs(args: argparse.Namespace) -> List[Dict[str, str]]:
    freeze = m65.read_json(args.freeze_manifest)
    if freeze.get("status") != "FROZEN_BEFORE_STAGE6G_ROLLOUTS":
        raise ValueError("Stage6G freeze manifest is not ready")
    if int(freeze.get("planned_primary_additions", -1)) != 490:
        raise ValueError("Stage6G freeze manifest does not contain exactly 490 primary additions")
    if any(bool(value) for value in freeze.get("forbidden_inputs_read", {}).values()):
        raise ValueError("Stage6G freeze reports forbidden post-treatment input use")
    expected_primary_sha = str(freeze.get("hashes", {}).get("primary_csv_sha256", ""))
    if batch.sha256_file(args.primary_csv) != expected_primary_sha:
        raise ValueError("Stage6G primary CSV hash differs from freeze manifest")
    frozen_rows = m65.index_by_token(m65.read_csv(args.primary_csv), "Stage6G primary")
    status_rows = m65.read_csv(args.batch_status_csv)
    if len(status_rows) != 490:
        raise ValueError(f"Stage6G status must contain 490 rows, observed={len(status_rows)}")
    if any(row.get("status") != "SUCCEEDED" for row in status_rows):
        counts = Counter(row.get("status", "") for row in status_rows)
        raise ValueError(f"Stage6G primary run is not fully successful: {dict(counts)}")
    rows: List[Dict[str, str]] = []
    for status in sorted(status_rows, key=lambda row: int(row["collection_order"])):
        token = status.get("scenario_token", "")
        if token not in frozen_rows:
            raise ValueError(f"successful Stage6G token is absent from frozen primary: {token}")
        frozen = frozen_rows[token]
        for key in ("collection_order", "task", "log_name", "db_file"):
            if str(status.get(key, "")) != str(frozen.get(key, "")):
                raise ValueError(f"Stage6G status/freeze mismatch for token={token}, field={key}")
        output_dir = Path(status.get("stage7c_output_dir", "")).resolve()
        rows.append(
            {
                **frozen,
                "source_group": "stage6g_primary",
                "source_collection_order": status["collection_order"],
                "stage7c_output_dir": str(output_dir),
            }
        )
    counts = {task: int(Counter(row["task"] for row in rows)[task]) for task in EXPECTED_TASK_COUNTS}
    if counts != EXPECTED_TASK_COUNTS:
        raise ValueError(f"Stage6G task composition mismatch: {counts}")
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    rows = validate_inputs(args)
    old_ledger = m65.read_csv(args.existing_ledger_csv)
    old_tokens = {row["scenario_token"] for row in old_ledger}
    new_tokens = {row["scenario_token"] for row in rows}
    overlap = sorted(old_tokens & new_tokens)
    if overlap:
        raise ValueError(f"Stage6G and existing 310 scenario tokens overlap: {overlap}")
    if len(old_tokens) != 310 or len(new_tokens) != 490:
        raise ValueError("expected 310 existing and 490 new unique scenario tokens")
    try:
        view = m65.prepare_view(rows, args.output_dir)
        schema_path = args.output_dir / "simulation_schema.json"
        schema = m65.read_json(schema_path)
        schema.update(
            {
                "stage": "Stage 6H expanded 490-pair official nuPlan rollout view",
                "schema_version": SCHEMA_VERSION,
                "selection_is_outcome_blind_and_unchanged": True,
            }
        )
        m65.write_json(schema_path, schema)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": READY_STATUS,
            "issue": "https://github.com/forwardxp-021/E2E-Evaluation/issues/246",
            "selection_is_outcome_blind_and_unchanged": True,
            "existing_pool_scenario_count": 310,
            "new_pool_scenario_count": 490,
            "combined_target_scenario_count": 800,
            "existing_new_token_overlap_count": 0,
            "expected_task_counts": EXPECTED_TASK_COUNTS,
            "planner_parameter_fingerprints": batch.current_planner_fingerprints(batch.EXPECTED_PLANNERS),
            "preparation_tool_sha256": batch.sha256_file(Path(__file__).resolve()),
            "input_files": {
                "freeze_manifest": {"path": str(args.freeze_manifest.resolve()), "sha256": batch.sha256_file(args.freeze_manifest)},
                "primary_csv": {"path": str(args.primary_csv.resolve()), "sha256": batch.sha256_file(args.primary_csv)},
                "batch_status_csv": {"path": str(args.batch_status_csv.resolve()), "sha256": batch.sha256_file(args.batch_status_csv)},
                "existing_ledger_csv": {"path": str(args.existing_ledger_csv.resolve()), "sha256": batch.sha256_file(args.existing_ledger_csv)},
            },
            **view,
        }
        m65.write_json(args.output_dir / "stage6h_expanded_view_summary.json", summary)
    except Exception:
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        raise
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audited Stage6H 490-pair official rollout view.")
    parser.add_argument("--freeze_manifest", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
    parser.add_argument("--existing_ledger_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
