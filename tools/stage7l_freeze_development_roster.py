#!/usr/bin/env python3
"""Freeze an outcome-blind Stage7L-B development roster and permanent exclusions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.stage7l_build_lane_change_opportunity_inventory import (
    BACKGROUND_AGENT_MODEL,
    BACKGROUND_CONFIG,
    BACKGROUND_MODE,
)
from tools.stage7l_pure_lateral_execution_planner import canonical_json_sha256


DOSE_LENGTHS = {"dose0": 60.0, "dose25": 54.0, "dose50": 48.0, "dose75": 42.0, "dose100": 36.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def canonical_progress(initial_speed: float, time_s: float, target_speed: float = 5.0, accel: float = 1.0) -> float:
    delta = target_speed - initial_speed
    ramp = abs(delta) / accel
    ramp_t = min(time_s, ramp)
    signed_accel = accel if delta > 0 else (-accel if delta < 0 else 0.0)
    return initial_speed * ramp_t + 0.5 * signed_accel * ramp_t**2 + target_speed * max(time_s - ramp, 0.0)


def geometry_features(row: Mapping[str, str]) -> Dict[str, float]:
    def curvature_p90(raw: str) -> float:
        xy = np.asarray(json.loads(raw), dtype=np.float64)
        delta = np.diff(xy, axis=0)
        length = np.linalg.norm(delta, axis=1)
        valid = length > 1e-6
        heading = np.unwrap(np.arctan2(delta[valid, 1], delta[valid, 0]))
        local = np.abs(np.diff(heading)) / np.maximum(length[valid][1:], 1e-6)
        return float(np.quantile(local, 0.9)) if len(local) else 0.0

    source = np.asarray(json.loads(row["source_reference_xy_json"]), dtype=np.float64)
    target = np.asarray(json.loads(row["target_reference_xy_json"]), dtype=np.float64)
    n = min(len(source), len(target))
    separation = np.linalg.norm(source[:n] - target[:n], axis=1)
    return {
        "initial_speed_mps": float(row["initial_speed_mps"]),
        "lane_width_m": float(row["nominal_lane_width_m"]),
        "paired_reference_remaining_m": float(row["paired_reference_remaining_m"]),
        "target_clearance_m_capped": min(float(row["minimum_target_lane_object_gap_m"]), 100.0),
        "source_curvature_p90_inv_m": curvature_p90(row["source_reference_xy_json"]),
        "target_curvature_p90_inv_m": curvature_p90(row["target_reference_xy_json"]),
        "source_target_separation_m": float(np.median(separation)),
    }


def diverse_select(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"insufficient candidates for requested stratum: {len(rows)} < {count}")
    keys = [
        "initial_speed_mps", "lane_width_m", "paired_reference_remaining_m", "target_clearance_m_capped",
        "source_curvature_p90_inv_m", "target_curvature_p90_inv_m", "source_target_separation_m",
    ]
    matrix = np.asarray([[float(row[key]) for key in keys] for row in rows], dtype=np.float64)
    low = np.min(matrix, axis=0); span = np.maximum(np.max(matrix, axis=0) - low, 1e-9)
    normalized = (matrix - low) / span
    seed = int(np.argmax(normalized[:, 0] + normalized[:, 4] + normalized[:, 5]))
    chosen = [seed]
    while len(chosen) < count:
        best_index = None; best_score = -1.0
        for index, row in enumerate(rows):
            if index in chosen:
                continue
            distance = float(np.min(np.linalg.norm(normalized[index] - normalized[chosen], axis=1)))
            categorical = 0.08 * all(row["source_roadblock_id"] != rows[j]["source_roadblock_id"] for j in chosen)
            categorical += 0.04 * all(row["map_name"] != rows[j]["map_name"] for j in chosen)
            score = distance + categorical
            if score > best_score or (score == best_score and row["scenario_token"] < rows[best_index]["scenario_token"]):
                best_index, best_score = index, score
        assert best_index is not None
        chosen.append(best_index)
    return [rows[index] for index in chosen]


def maneuver(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_token": row["scenario_token"], "log_name": row["log_name"], "db_file": row["db_file"],
        "initial_state_fingerprint": row["initial_state_fingerprint"],
        "initial_x": float(row["initial_x"]), "initial_y": float(row["initial_y"]),
        "initial_heading": float(row["initial_heading"]), "initial_speed_mps": float(row["initial_speed_mps"]),
        "source_lane_id": row["source_lane_id"], "target_lane_id": row["target_lane_id"],
        "source_roadblock_id": row["source_roadblock_id"], "target_roadblock_id": row["target_roadblock_id"],
        "direction": row["direction"], "route_roadblock_ids": json.loads(row["route_roadblock_ids_json"]),
        "route_fingerprint": row["route_fingerprint"], "trigger_s_route_m": 12.0,
        "source_start_arc_m": float(row["source_start_arc_m"]), "target_start_arc_m": float(row["target_start_arc_m"]),
        "nominal_lane_width_m": float(row["nominal_lane_width_m"]), "horizon_s": 15.0,
        "background_mode": BACKGROUND_MODE, "background_agent_model": BACKGROUND_AGENT_MODEL,
        "background_config_sha256": canonical_json_sha256(BACKGROUND_CONFIG),
        "source_reference_xy": json.loads(row["source_reference_xy_json"]),
        "target_reference_xy": json.loads(row["target_reference_xy_json"]),
        "planner_profile_ids": list(DOSE_LENGTHS),
    }


def write_manifest(path: Path, rows: List[Dict[str, Any]], inventory_sha: str, role: str) -> None:
    payload = {
        "schema_version": "stage7l_b_development_maneuver_manifest_v1",
        "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY", "role": role,
        "created_utc": datetime.now(timezone.utc).isoformat(), "maneuvers": [maneuver(row) for row in rows],
        "dose_transition_length_m": DOSE_LENGTHS, "planner_horizon_s": 0.4,
        "parameter_version": "candidate_development_dose_set_v1",
        "inventory_csv_sha256": inventory_sha, "embedding_or_bdd_read": False,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw = [row for row in read_csv(args.inventory_csv) if row.get("eligible", "").lower() == "true"]
    enriched: List[Dict[str, Any]] = []
    for row in raw:
        item: Dict[str, Any] = dict(row); item.update(geometry_features(row))
        required = canonical_progress(float(row["initial_speed_mps"]), 15.0 + 0.4)
        item["required_reference_remaining_m"] = required
        item["stage7l_b_reference_coverage_pass"] = float(row["paired_reference_remaining_m"]) >= required
        if item["stage7l_b_reference_coverage_pass"]:
            enriched.append(item)
    log_counts = Counter(row["log_name"] for row in enriched)
    single_log = [row for row in enriched if log_counts[row["log_name"]] == 1]
    left = diverse_select([row for row in single_log if row["direction"] == "left"], 6)
    right = diverse_select([row for row in single_log if row["direction"] == "right"], 18)
    selected = left + right
    if len({row["log_name"] for row in selected}) != 24:
        raise AssertionError("development roster must use 24 unique logs")
    sanity = diverse_select(left, 2) + diverse_select(right, 6)
    sanity_tokens = {row["scenario_token"] for row in sanity}
    roster_rows = []
    for order, row in enumerate(selected, start=1):
        out = dict(row); out["development_order"] = order
        out["sanity_subset"] = row["scenario_token"] in sanity_tokens
        out["selection_status"] = "PRETREATMENT_FROZEN_DEVELOPMENT_ONLY"
        roster_rows.append(out)
    fields = sorted({key for row in roster_rows for key in row})
    roster_path = args.output_dir / "development_roster.csv"
    write_csv(roster_path, roster_rows, fields)
    sanity_path = args.output_dir / "sanity_subset_roster.csv"
    write_csv(sanity_path, [row for row in roster_rows if row["sanity_subset"]], fields)
    inventory_sha = sha256_file(args.inventory_csv)
    full_manifest = args.output_dir / "development_maneuver_manifest.json"
    sanity_manifest = args.output_dir / "sanity_maneuver_manifest.json"
    write_manifest(full_manifest, roster_rows, inventory_sha, "FULL_24")
    write_manifest(sanity_manifest, [row for row in roster_rows if row["sanity_subset"]], inventory_sha, "SANITY_8")
    exclusion_rows = read_csv(args.a2_exclusion_ledger)
    for row in roster_rows:
        exclusion_rows.append({
            "scenario_token": row["scenario_token"], "log_name": row["log_name"],
            "exclusion_reason": "STAGE7L_B_DEVELOPMENT_ROSTER_PERMANENT_EXCLUSION",
            "source_path": str(roster_path.resolve()),
        })
    exclusion_by_token = {row["scenario_token"]: row for row in exclusion_rows}
    exclusion_path = args.output_dir / "stage7l_b_prior_exclusion_ledger.csv"
    write_csv(exclusion_path, sorted(exclusion_by_token.values(), key=lambda row: row["scenario_token"]),
              ["scenario_token", "log_name", "exclusion_reason", "source_path"])
    dev_logs = {row["log_name"] for row in roster_rows}
    remaining = [row for row in enriched if row["log_name"] not in dev_logs]
    remaining_summary = {
        "schema_version": "stage7l_b_remaining_confirmation_inventory_v1",
        "eligibility_rule": "paired reference covers canonical progress through 15.4 s",
        "eligible_before_development_tokens": len(enriched),
        "eligible_before_development_logs": len({row["log_name"] for row in enriched}),
        "remaining_fresh_eligible_tokens": len(remaining),
        "remaining_fresh_eligible_logs": len({row["log_name"] for row in remaining}),
        "remaining_left": sum(row["direction"] == "left" for row in remaining),
        "remaining_right": sum(row["direction"] == "right" for row in remaining),
        "strict_log_disjoint_80_confirmation_feasible": len(remaining) >= 80,
        "development_log_overlap": 0,
    }
    remaining_path = args.output_dir / "remaining_confirmation_inventory_summary.json"
    remaining_path.write_text(json.dumps(remaining_summary, indent=2) + "\n", encoding="utf-8")
    versions_path = args.output_dir / "treatment_versions.csv"
    write_csv(versions_path, [{
        "version": "candidate_development_dose_set_v1", "dose0_m": 60, "dose25_m": 54,
        "dose50_m": 48, "dose75_m": 42, "dose100_m": 36, "trigger_m": 12,
        "planner_horizon_s": 0.4, "status": "CANDIDATE_BEFORE_SANITY",
    }], ["version", "dose0_m", "dose25_m", "dose50_m", "dose75_m", "dose100_m", "trigger_m", "planner_horizon_s", "status"])
    summary = {
        "schema_version": "stage7l_b_development_roster_freeze_v1",
        "status": "FROZEN_DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "selection_is_pre_treatment": True, "rollout_outcome_used": False, "embedding_or_bdd_read": False,
        "development_scenarios": 24, "development_logs": 24, "left": 6, "right": 18,
        "sanity_scenarios": 8, "sanity_left": 2, "sanity_right": 6,
        "reference_coverage_eligible_count": len(enriched), "all_selected_from_single_candidate_logs": True,
        "roster_sha256": sha256_file(roster_path), "sanity_roster_sha256": sha256_file(sanity_path),
        "full_manifest_sha256": sha256_file(full_manifest), "sanity_manifest_sha256": sha256_file(sanity_manifest),
        "exclusion_ledger_sha256": sha256_file(exclusion_path), "inventory_sha256": inventory_sha,
        "remaining_confirmation": remaining_summary,
    }
    (args.output_dir / "development_roster_freeze_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--a2_exclusion_ledger", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
