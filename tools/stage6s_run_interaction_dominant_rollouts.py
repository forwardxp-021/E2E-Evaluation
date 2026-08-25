#!/usr/bin/env python3
"""Run the frozen Stage 6S interaction-dominant pilot without reading embeddings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage6j_run_pure_longitudinal_rollouts as runner


PLANNERS = [
    "pdm_closed_interaction_short_headway_v1",
    "pdm_closed_interaction_long_headway_v1",
]
SCHEMA_VERSION = "stage6s_interaction_dominant_batch_v1"
FREEZE_STATUS = "FROZEN_BEFORE_INTERACTION_PILOT_ROLLOUTS"


def validate_inputs(args):
    manifest = runner.read_json(args.freeze_manifest.resolve())
    rows = runner.read_csv(args.locked_scenarios_csv.resolve())
    if manifest.get("schema_version") != "stage6s_interaction_dominant_freeze_v1":
        raise ValueError("unexpected Stage 6S freeze schema")
    if manifest.get("status") != FREEZE_STATUS:
        raise ValueError("Stage 6S is not frozen before rollout")
    if manifest.get("embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6S freeze does not certify embedding_or_bdd_read=false")
    locked_sha = runner.sha256_file(args.locked_scenarios_csv.resolve())
    if manifest.get("locked_scenarios_sha256") != locked_sha:
        raise ValueError("Stage 6S locked scenario CSV changed after freeze")
    if manifest.get("planners") != PLANNERS:
        raise ValueError("Stage 6S planner order changed")
    if manifest.get("planner_fingerprints") != runner.planner_fingerprints():
        raise ValueError("Stage 6S planner parameters changed after freeze")
    if len(rows) != int(manifest.get("scenario_count", -1)):
        raise ValueError("Stage 6S frozen scenario count changed")
    tokens = [row["scenario_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Stage 6S scenario tokens are not unique")
    for row in rows:
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError(f"DB/log mismatch for {row['scenario_token']}")
        if not (args.nuplan_db_root.resolve() / row["db_file"]).is_file():
            raise FileNotFoundError(args.nuplan_db_root.resolve() / row["db_file"])
    for path in [args.python_executable, args.stage7c_tool]:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    for path in [
        args.nuplan_map_root,
        args.nuplan_data_root,
        args.nuplan_exp_root,
        args.nuplan_devkit_root,
        args.tuplan_garage_root,
    ]:
        if not path.resolve().is_dir():
            raise FileNotFoundError(path.resolve())
    if runner.git_commit(args.nuplan_devkit_root.resolve()) != args.expected_nuplan_commit:
        raise ValueError("nuPlan devkit commit differs from frozen runtime")
    if runner.git_commit(args.tuplan_garage_root.resolve()) != args.expected_tuplan_commit:
        raise ValueError("tuPlan Garage commit differs from frozen runtime")
    return manifest, rows, locked_sha


def main() -> None:
    runner.PLANNERS = PLANNERS
    runner.SCHEMA_VERSION = SCHEMA_VERSION
    runner.FREEZE_STATUS = FREEZE_STATUS
    runner.validate_inputs = validate_inputs
    args = runner.parse_args()
    args.experiment_name = "stage6s_interaction_dominant_batch_v1"
    print(json.dumps(runner.run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
