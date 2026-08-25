#!/usr/bin/env python3
"""Extract frozen Stage7L-D trajectory, nuisance, safety, and purity metrics.

The tool only reads official planner outputs and frozen maneuver geometry.  It
does not load representations/checkpoints or calculate BDD/MMD.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tools.stage7l_evaluate_lateral_mechanism import evaluate_one


DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def first_metric_row(root: Path, filename: str) -> Mapping[str, Any] | None:
    paths = sorted(root.glob(f"scenario_*/*/metrics/{filename}"))
    if len(paths) != 1:
        return None
    table = pd.read_parquet(paths[0])
    return None if table.empty else table.iloc[0]


def official_metrics(root: Path) -> Dict[str, Any]:
    collision = first_metric_row(root, "no_ego_at_fault_collisions.parquet")
    drivable = first_metric_row(root, "drivable_area_compliance.parquet")
    progress = first_metric_row(root, "ego_is_making_progress.parquet")
    result: Dict[str, Any] = {
        "official_safety_metrics_available": collision is not None and drivable is not None,
        "responsible_collision": None,
        "responsible_collision_count": None,
        "offroad": None,
        "drivable_area_compliant": None,
        "any_collision": "N/A_OFFICIAL_BUNDLE_ONLY_EXPOSES_AT_FAULT_COLLISION",
        "any_collision_evidence_status": "NOT_AVAILABLE",
        "route_failure": None,
    }
    if collision is not None:
        count = int(collision["number_of_all_at_fault_collisions_stat_value"])
        result.update({"responsible_collision": count > 0, "responsible_collision_count": count})
    if drivable is not None:
        compliant = bool(drivable["drivable_area_compliance_stat_value"])
        result.update({"offroad": not compliant, "drivable_area_compliant": compliant})
    if progress is not None:
        candidates = [name for name in progress.index if name.endswith("stat_value") and "making_progress" in name]
        if candidates:
            result["route_failure"] = not bool(progress[candidates[0]])
    return result


def empty_metric_row(summary: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "cell_id": summary["cell_id"], "collection_order": int(summary["collection_order"]),
        "scenario_token": summary["scenario_token"], "log_name": summary["log_name"],
        "direction": summary["direction"], "dose": summary["dose"],
        "transition_length_m": float(summary["transition_length_m"]),
        "official_run_status": summary["official_run_status"],
        "trajectory_available": as_bool(summary["trajectory_available"]),
        "valid": False, "lane_change_completion": False,
        "metric_missing_reason": summary.get("failure_category") or "TRAJECTORY_METRIC_NOT_AVAILABLE",
        "source_target_projection_anomaly": None, "invalid_reference": None, "map_failure": None,
        "official_safety_metrics_available": False, "responsible_collision": None,
        "responsible_collision_count": None, "offroad": None, "drivable_area_compliant": None,
        "any_collision": "N/A_OFFICIAL_BUNDLE_ONLY_EXPOSES_AT_FAULT_COLLISION",
        "any_collision_evidence_status": "NOT_AVAILABLE", "route_failure": None,
    }


def extract_one(summary: Mapping[str, str], maneuver: Mapping[str, Any]) -> Dict[str, Any]:
    row = empty_metric_row(summary)
    if summary.get("official_run_status") != "SUCCEEDED" or not as_bool(summary.get("trajectory_available")):
        return row
    trajectory_path = Path(summary["trajectory_csv"])
    trajectories = [item for item in read_csv(trajectory_path) if item.get("scene_token") == summary["scenario_token"]]
    if not trajectories:
        row["metric_missing_reason"] = "SUCCESS_SUMMARY_WITHOUT_MATCHING_TRAJECTORY"
        return row
    evaluated = evaluate_one(trajectories, maneuver)
    row.update(evaluated)
    row["planner_name"] = trajectories[0]["planner_name"]
    row["metric_missing_reason"] = "" if evaluated["valid"] else "INVALID_TRAJECTORY"
    row["source_target_projection_anomaly"] = bool(
        not math.isfinite(float(evaluated["route_progress_m"])) or float(evaluated["route_progress_m"]) < -0.01
    )
    row["invalid_reference"] = False
    row["map_failure"] = False
    row.update(official_metrics(Path(summary["official_runs_root"])))
    return row


def canonical_identity(summary: List[Dict[str, str]]) -> Dict[str, Any]:
    by_token: Dict[str, List[Dict[str, str]]] = {}
    for row in summary:
        if row.get("official_run_status") == "SUCCEEDED" and as_bool(row.get("trajectory_available")):
            by_token.setdefault(row["scenario_token"], []).append(row)
    details: Dict[str, Any] = {}
    mismatch_count = 0
    complete_count = 0
    for token, group in sorted(by_token.items()):
        complete = len(group) == 5 and {row["dose"] for row in group} == set(DOSES)
        if not complete:
            continue
        complete_count += 1
        audits = []
        missing = []
        for row in sorted(group, key=lambda item: DOSES.index(item["dose"])):
            path = Path(row["planner_audit_path"]) if row.get("planner_audit_path") else Path("__missing__")
            if not path.is_file():
                missing.append(row["dose"])
                continue
            audits.append(json.loads(path.read_text(encoding="utf-8")))
        arrays = [np.asarray(item["s_route_initial_plan_m"], dtype=np.float64) for item in audits]
        item = {
            "dose_count": len(group), "audit_count": len(audits), "missing_audit_doses": missing,
            "s_route_pointwise_identical": len(arrays) == 5 and all(np.array_equal(arrays[0], other) for other in arrays[1:]),
            "manifest_sha_identical": len(audits) == 5 and len({item["dose_invariant_manifest_sha256"] for item in audits}) == 1,
            "longitudinal_generator_sha_identical": len(audits) == 5 and len({item["canonical_longitudinal_generator_sha256"] for item in audits}) == 1,
            "trigger_background_identity": len(audits) == 5 and len({(item.get("background_mode"), item.get("background_config_sha256")) for item in audits}) == 1,
        }
        item["pass"] = all(item[key] for key in (
            "s_route_pointwise_identical", "manifest_sha_identical", "longitudinal_generator_sha_identical",
            "trigger_background_identity",
        ))
        mismatch_count += int(not item["pass"])
        details[token] = item
    return {
        "schema_version": "stage7l_d_canonical_identity_audit_v1",
        "status": "PASS" if mismatch_count == 0 else "FAIL_CANONICAL_TREATMENT_PURITY",
        "complete_five_dose_scenario_count": complete_count,
        "canonical_identity_pass_count": complete_count - mismatch_count,
        "mismatch_count": mismatch_count,
        "details": details,
        "embedding_read": False, "checkpoint_read": False, "bdd_computed": False, "mmd_computed": False,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    summary = read_csv(args.official_run_summary)
    if len(summary) != 400:
        raise ValueError(f"official_run_summary must contain all 400 planned cells, got {len(summary)}")
    manifest = json.loads(args.maneuver_manifest.read_text(encoding="utf-8"))
    by_token = {row["scenario_token"]: row for row in manifest["maneuvers"]}
    rows = [extract_one(item, by_token[item["scenario_token"]]) for item in summary]
    fields = sorted({key for row in rows for key in row})
    write_csv(args.output_dir / "mechanism_metrics_long.csv", rows, fields)
    identity = canonical_identity(summary)
    (args.output_dir / "canonical_identity_audit.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {
        "schema_version": "stage7l_d_metric_extraction_v1",
        "status": "STAGE7L_D_PLANNER_METRICS_EXTRACTED_NO_REPRESENTATION",
        "planned_rows": 400, "valid_trajectory_metric_rows": sum(bool(row["valid"]) for row in rows),
        "completion_rows": sum(bool(row["lane_change_completion"]) for row in rows),
        "canonical_identity_status": identity["status"],
        "embedding_read": False, "checkpoint_read": False, "bdd_computed": False, "mmd_computed": False,
    }
    (args.output_dir / "metric_extraction_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official_run_summary", type=Path, default=root / "outputs/stage7l_d_one_time_confirmation_v1/official_run_summary.csv")
    parser.add_argument("--maneuver_manifest", type=Path, default=root / "outputs/stage7l_c_confirmation_freeze_v1/confirmation_maneuver_manifest.json")
    parser.add_argument("--output_dir", type=Path, default=root / "outputs/stage7l_d_one_time_confirmation_v1")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
