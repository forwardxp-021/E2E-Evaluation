#!/usr/bin/env python3
"""Run the locked Stage6S-v2 confirmation roster without representations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage6j_run_pure_longitudinal_rollouts as runner


PLANNERS = ["pdm_closed_interaction_short_headway_v2", "pdm_closed_interaction_long_headway_v2"]
SCHEMA_VERSION = "stage6s_v2_confirmation_batch_v1"
FREEZE_STATUS = "CONFIRMATION_ROSTER_FROZEN_NOT_RUN"
AUTH = Path(__file__).resolve().parents[1] / "outputs/stage6v_blind_evaluation_authorization_v1/stage6v_blind_evaluation_authorization_manifest.json"
AUTH_SHA = "c7f945b3236856b4bb0ee9c8e888c2eca83856dd6201d4c4c957fae9dacef5bd"


def validate_inputs(args):
    if runner.sha256_file(AUTH) != AUTH_SHA:
        raise ValueError("blind authorization changed")
    auth = runner.read_json(AUTH)
    if auth.get("stage6s_v2_confirmation_rollout_authorized") is not True:
        raise ValueError("Stage6S-v2 confirmation rollout not authorized")
    manifest = runner.read_json(args.freeze_manifest.resolve())
    rows = runner.read_csv(args.locked_scenarios_csv.resolve())
    if manifest.get("schema_version") != "stage6s_v2_confirmation_freeze_v1" or manifest.get("status") != FREEZE_STATUS:
        raise ValueError("unexpected Stage6S-v2 confirmation freeze")
    if manifest.get("confirmation_rollouts_launched") is not False or manifest.get("embedding_or_bdd_read") is not False:
        raise ValueError("confirmation blind state changed before launch")
    locked_sha = runner.sha256_file(args.locked_scenarios_csv.resolve())
    if locked_sha != manifest.get("confirmation_roster_sha256"):
        raise ValueError("confirmation roster changed")
    if manifest.get("planners") != PLANNERS or manifest.get("planner_fingerprints") != runner.planner_fingerprints():
        raise ValueError("confirmation planners changed")
    if len(rows) != int(manifest.get("scenario_count", -1)) or len(rows) != 80:
        raise ValueError("confirmation roster must contain 80 scenarios")
    orders = [int(row["collection_order"]) for row in rows]
    if orders != list(range(1, 81)):
        raise ValueError("confirmation collection order is not frozen 1..80")
    tokens = [row["scenario_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("confirmation scenario tokens are not unique")
    for row in rows:
        if row["db_file"] != f"{row['log_name']}.db":
            raise ValueError("confirmation DB/log mismatch")
        if not (args.nuplan_db_root.resolve() / row["db_file"]).is_file():
            raise FileNotFoundError(args.nuplan_db_root.resolve() / row["db_file"])
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
    args.experiment_name = "stage6s_v2_confirmation_batch_v1"
    result = runner.run(args)
    result["embedding_or_bdd_read"] = False
    result["blind_authorization_sha256"] = AUTH_SHA
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
