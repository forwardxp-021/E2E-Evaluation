#!/usr/bin/env python3
"""Freeze A2 smoke-only Stage7L maneuver manifests from eligible inventory rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_build_lane_change_opportunity_inventory import (
    BACKGROUND_AGENT_MODEL, BACKGROUND_CONFIG, BACKGROUND_MODE,
    candidate_row_for_token, evaluate_candidate,
)
from tools.stage7l_pure_lateral_execution_planner import DOSE_TRANSITION_LENGTH_M, canonical_json_sha256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    with args.inventory_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if str(row.get("eligible", "")).lower() == "true"]
    if args.scenario_token:
        rows = [row for row in rows if row["scenario_token"] == args.scenario_token]
    if not rows and args.refresh_smoke_db_file:
        if not args.scenario_token or args.nuplan_db_root is None or args.nuplan_map_root is None:
            raise ValueError("refreshing a permanently excluded smoke token requires token, DB root, and map root")
        db_path = args.nuplan_db_root / args.refresh_smoke_db_file
        candidate = candidate_row_for_token(db_path, args.scenario_token)
        refreshed = evaluate_candidate(candidate, args.nuplan_db_root, args.nuplan_map_root, args.map_version, 90.0, 10.0, 3.0)
        if not refreshed.get("eligible"):
            raise ValueError(f"refreshed smoke token is no longer eligible: {refreshed.get('reason_code')}")
        rows = [{key: (json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, tuple)) else str(value)) for key, value in refreshed.items()}]
    rows = rows[: args.max_scenarios]
    if not rows:
        raise ValueError("no eligible Stage7L maneuver selected")
    dose_ids = tuple(DOSE_TRANSITION_LENGTH_M)
    maneuvers: List[Dict[str, Any]] = []
    for row in rows:
        maneuver = {
            "scenario_token": row["scenario_token"], "log_name": row["log_name"], "db_file": row["db_file"],
            "initial_state_fingerprint": row["initial_state_fingerprint"],
            "initial_x": float(row["initial_x"]), "initial_y": float(row["initial_y"]),
            "initial_heading": float(row["initial_heading"]), "initial_speed_mps": float(row["initial_speed_mps"]),
            "source_lane_id": row["source_lane_id"], "target_lane_id": row["target_lane_id"],
            "source_roadblock_id": row["source_roadblock_id"], "target_roadblock_id": row["target_roadblock_id"],
            "direction": row["direction"], "route_roadblock_ids": json.loads(row["route_roadblock_ids_json"]),
            "route_fingerprint": row["route_fingerprint"], "trigger_s_route_m": float(args.trigger_s_route_m),
            "source_start_arc_m": float(row["source_start_arc_m"]), "target_start_arc_m": float(row["target_start_arc_m"]),
            "nominal_lane_width_m": float(row["nominal_lane_width_m"]), "horizon_s": float(args.scenario_horizon_s),
            "background_mode": BACKGROUND_MODE, "background_agent_model": BACKGROUND_AGENT_MODEL,
            "background_config_sha256": canonical_json_sha256(BACKGROUND_CONFIG),
            "source_reference_xy": json.loads(row["source_reference_xy_json"]),
            "target_reference_xy": json.loads(row["target_reference_xy_json"]),
            "planner_profile_ids": dose_ids,
        }
        maneuvers.append(maneuver)
    payload = {
        "schema_version": "stage7l_a2_frozen_maneuver_manifest_v1",
        "status": "A2_SMOKE_ONLY_NOT_FROZEN_FOR_CONFIRMATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "maneuvers": maneuvers,
        "dose_transition_length_m": DOSE_TRANSITION_LENGTH_M,
        "dose_invariant_fields": [
            "scenario_token", "initial_state_fingerprint", "source_lane_id", "target_lane_id", "direction",
            "route_fingerprint", "trigger_s_route_m", "source_start_arc_m", "target_start_arc_m",
            "background_config_sha256",
        ],
        "embedding_or_bdd_read": False,
        "inventory_csv_sha256": sha256_file(args.inventory_csv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"output": str(args.output.resolve()), "sha256": sha256_file(args.output), "scenario_count": len(maneuvers)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario_token", default="")
    parser.add_argument("--max_scenarios", type=int, default=1)
    parser.add_argument("--trigger_s_route_m", type=float, default=12.0)
    parser.add_argument("--scenario_horizon_s", type=float, default=15.0)
    parser.add_argument("--refresh_smoke_db_file", default="")
    parser.add_argument("--nuplan_db_root", type=Path)
    parser.add_argument("--nuplan_map_root", type=Path)
    parser.add_argument("--map_version", default="nuplan-maps-v1.0")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
