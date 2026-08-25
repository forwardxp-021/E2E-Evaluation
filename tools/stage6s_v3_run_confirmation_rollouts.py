#!/usr/bin/env python3
"""Run the prospective Stage6S-v3 roster; never reads representations."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage6j_run_pure_longitudinal_rollouts as runner


PLANNERS = ["pdm_closed_interaction_short_headway_v2", "pdm_closed_interaction_long_headway_v2"]
SCHEMA_VERSION = "stage6s_v3_confirmation_batch_v1"
FREEZE_STATUS = "STAGE6S_V3_CONFIRMATION_ROSTER_FROZEN_NOT_RUN"
FREEZE_SHA = "7105940bd822f02d643ed4f5cb9a8321b3827ca6117be289914057e3fe8a26c6"
V2_FAILURE = Path(__file__).resolve().parents[1] / "outputs/stage6v_stage6s_v2_confirmation_execution_freeze_v1/stage6s_v2_confirmation_execution_freeze.json"
V2_FAILURE_SHA = "e092ee198d412c0fcc830649ae7b22031d09a4284197131b9d0f2733c61faea8"


def official_boundary(db_path: Path, token: str) -> bool:
    query = """
        WITH ordered_scenes AS (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num FROM scene
        ), num_scenes AS (SELECT COUNT(*) AS cnt FROM scene)
        SELECT o.row_num, n.cnt FROM lidar_pc AS lp
        INNER JOIN ordered_scenes AS o ON o.token = lp.scene_token CROSS JOIN num_scenes AS n
        WHERE lp.token = ?
    """
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(query, (bytes.fromhex(token),)).fetchone()
    return bool(row and row[0] >= 3 and row[0] < row[1] - 1)


def validate_inputs(args):
    if runner.sha256_file(args.freeze_manifest.resolve()) != FREEZE_SHA:
        raise ValueError("Stage6S-v3 freeze manifest changed")
    if runner.sha256_file(V2_FAILURE) != V2_FAILURE_SHA:
        raise ValueError("Stage6S-v2 permanent failure record changed")
    manifest = runner.read_json(args.freeze_manifest.resolve())
    rows = runner.read_csv(args.locked_scenarios_csv.resolve())
    if manifest.get("schema_version") != "stage6s_v3_confirmation_freeze_v1" or manifest.get("status") != FREEZE_STATUS:
        raise ValueError("unexpected Stage6S-v3 confirmation freeze")
    if manifest.get("confirmation_rollouts_launched") is not False or manifest.get("embedding_or_bdd_read") is not False:
        raise ValueError("Stage6S-v3 blind state changed before launch")
    locked_sha = runner.sha256_file(args.locked_scenarios_csv.resolve())
    if locked_sha != manifest.get("confirmation_roster_sha256"):
        raise ValueError("Stage6S-v3 roster changed")
    if manifest.get("planners") != PLANNERS or manifest.get("planner_fingerprints") != runner.planner_fingerprints():
        raise ValueError("Stage6S-v3 planners changed")
    if len(rows) != 80 or int(manifest.get("scenario_count", -1)) != 80:
        raise ValueError("Stage6S-v3 roster must contain exactly 80 scenarios")
    if [int(row["collection_order"]) for row in rows] != list(range(1, 81)):
        raise ValueError("Stage6S-v3 collection order is not frozen 1..80")
    tokens = [row["scenario_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Stage6S-v3 scenario tokens are not unique")
    for row in rows:
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError("Stage6S-v3 DB/log mismatch")
        db_path = args.nuplan_db_root.resolve() / row["db_file"]
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        if not official_boundary(db_path, row["scenario_token"]):
            raise ValueError(f"selected token is outside official scene boundary: {row['scenario_token']}")
    for path in [args.python_executable, args.stage7c_tool]:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    for path in [args.nuplan_map_root, args.nuplan_data_root, args.nuplan_exp_root, args.nuplan_devkit_root, args.tuplan_garage_root]:
        if not path.resolve().is_dir():
            raise FileNotFoundError(path.resolve())
    if runner.git_commit(args.nuplan_devkit_root.resolve()) != args.expected_nuplan_commit:
        raise ValueError("nuPlan commit differs from freeze")
    if runner.git_commit(args.tuplan_garage_root.resolve()) != args.expected_tuplan_commit:
        raise ValueError("tuPlan commit differs from freeze")
    return manifest, rows, locked_sha


def main() -> None:
    runner.PLANNERS = PLANNERS
    runner.SCHEMA_VERSION = SCHEMA_VERSION
    runner.FREEZE_STATUS = FREEZE_STATUS
    runner.validate_inputs = validate_inputs
    args = runner.parse_args()
    args.experiment_name = "stage6s_v3_confirmation_batch_v1"
    result = runner.run(args)
    result["embedding_or_bdd_read"] = False
    result["stage6s_v3_freeze_sha256"] = FREEZE_SHA
    result["stage6s_v2_permanent_failure_sha256"] = V2_FAILURE_SHA
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
