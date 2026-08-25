#!/usr/bin/env python3
"""Freeze the interaction-dominant nuPlan pilot before trajectory outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text())
    source = list(csv.DictReader(args.source_csv.open()))
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    selection = config["selection"]
    candidates = [row for row in source if row["task"] == selection["allowed_task"]]
    selected = []
    for scenario_type in selection["scenario_type_priority"]:
        selected.extend(sorted((row for row in candidates if row["scenario_type"] == scenario_type), key=lambda row: int(row["collection_order"])))
    selected = selected[: int(config["pilot_pair_count"])]
    if len(selected) != int(config["pilot_pair_count"]) or len({row["scenario_token"] for row in selected}) != len(selected):
        raise ValueError("unable to freeze 24 unique pre-treatment scenarios")
    planners = [config["planner_a"], config["planner_b"]]
    shared = config["shared_parameters"]
    audits = []
    allowed_difference = {"idm_policies.min_gap_to_lead_agent", "idm_policies.headway_time"}
    clean = {}
    for planner in planners:
        params = {key: value for key, value in PLANNER_PROFILES[planner]["parameters"].items() if key not in {"source", "checkpoint_required"}}
        clean[planner] = params
        for key, expected in shared.items():
            if params.get(key) != expected:
                raise ValueError(f"shared planner parameter differs: {planner}/{key}")
        for key, expected in config["treatment_parameters"][planner].items():
            if params.get(key) != expected:
                raise ValueError(f"treatment parameter differs: {planner}/{key}")
    actual_differences = {key for key in set(clean[planners[0]]) | set(clean[planners[1]]) if clean[planners[0]].get(key) != clean[planners[1]].get(key)}
    if actual_differences != allowed_difference:
        raise ValueError(f"treatment is not interaction-dominant: {actual_differences}")
    for key in sorted(set(clean[planners[0]]) | set(clean[planners[1]])):
        audits.append({"parameter": key, "planner_a": json.dumps(clean[planners[0]][key]), "planner_b": json.dumps(clean[planners[1]][key]), "different": key in actual_differences})
    locked = output / "stage6s_locked_scenarios.csv"
    fields = list(selected[0].keys())
    with locked.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    with (output / "stage6s_planner_parameter_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "planner_a", "planner_b", "different"])
        writer.writeheader(); writer.writerows(audits)
    manifest = {
        "schema_version": "stage6s_interaction_dominant_freeze_v1", "status": "FROZEN_BEFORE_INTERACTION_PILOT_ROLLOUTS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "issue": 260,
        "config_sha256": sha256_file(args.config), "source_csv_sha256": sha256_file(args.source_csv),
        "locked_scenarios_sha256": sha256_file(locked), "scenario_count": len(selected),
        "scenario_type_counts": {value: sum(row["scenario_type"] == value for row in selected) for value in selection["scenario_type_priority"]},
        "planners": planners, "planner_fingerprints": {planner: canonical_hash(PLANNER_PROFILES[planner]["parameters"]) for planner in planners},
        "only_different_parameters": sorted(actual_differences), "embedding_or_bdd_read": False,
        "rollouts_launched": False, "mechanism_gate": config["mechanism_gate"],
    }
    (output / "stage6s_freeze_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "stage6s_freeze_report_zh.md").write_text(
        "# Stage 6S interaction-dominant nuPlan benchmark冻结\n\n"
        f"- 场景：{len(selected)}对；只使用pre-treatment following场景。\n"
        f"- 类型：{manifest['scenario_type_counts']}\n"
        "- 两planner只允许headway与minimum gap不同；desired speed、accel/decel和lateral完全一致。\n"
        "- 冻结未读取embedding、BDD或trajectory outcome。\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
