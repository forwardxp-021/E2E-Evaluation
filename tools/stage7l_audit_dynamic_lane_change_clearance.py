#!/usr/bin/env python3
"""Audit a fixed Stage7L candidate table using original replay traffic only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7l_dynamic_clearance import DynamicClearanceConfig, dynamic_clearance_audit, sha256_file


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def config_from_args(args: argparse.Namespace) -> DynamicClearanceConfig:
    return DynamicClearanceConfig(
        horizon_seconds=args.horizon_seconds, time_step_seconds=args.time_step_seconds,
        maximum_track_interpolation_gap_seconds=args.maximum_track_interpolation_gap_seconds,
        trigger_route_progress_m=args.trigger_route_progress_m,
        gentle_transition_length_m=args.gentle_transition_length_m,
        settling_margin_m=args.settling_margin_m, target_speed_mps=args.target_speed_mps,
        accel_limit_mps2=args.accel_limit_mps2, ego_length_m=args.ego_length_m,
        ego_width_m=args.ego_width_m, longitudinal_buffer_m=args.longitudinal_buffer_m,
        lateral_buffer_m=args.lateral_buffer_m,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = config_from_args(args)
    source_rows = read_csv(args.candidate_csv)
    if args.only_static_eligible:
        source_rows = [row for row in source_rows if row.get("eligible", "").lower() == "true"]
    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(source_rows, start=1):
        try:
            audit = dynamic_clearance_audit(candidate, args.nuplan_db_root, config)
        except Exception as exc:
            audit = {
                "dynamic_clearance_pass": False,
                "dynamic_reason_code": f"DYNAMIC_AUDIT_ERROR:{type(exc).__name__}",
                "dynamic_error": str(exc), "dynamic_eligibility_pre_treatment": True,
                "dynamic_dose_independent": True, "dynamic_clearance_config_sha256": config.fingerprint(),
            }
        row = dict(candidate); row.update(audit); row["audit_order"] = index
        rows.append(row)
        if index % 25 == 0:
            print(f"[Stage7L-B2 dynamic audit] {index}/{len(source_rows)}", flush=True)
    fields = sorted({key for row in rows for key in row})
    audit_path = args.output_dir / "dynamic_clearance_audit.csv"
    write_csv(audit_path, rows, fields)
    reasons = Counter(str(row.get("dynamic_reason_code", "UNSET")) for row in rows)
    summary = {
        "schema_version": "stage7l_b2_dynamic_clearance_audit_v1",
        "status": "PRETREATMENT_DYNAMIC_AUDIT_COMPLETE",
        "candidate_count": len(rows),
        "dynamic_clear_count": sum(bool(row.get("dynamic_clearance_pass")) for row in rows),
        "dynamic_rejected_count": sum(not bool(row.get("dynamic_clearance_pass")) for row in rows),
        "reason_counts": dict(reasons),
        "config": vars(config), "config_sha256": config.fingerprint(),
        "candidate_csv_sha256": sha256_file(args.candidate_csv), "audit_csv_sha256": sha256_file(audit_path),
        "rollout_outcome_read": False, "embedding_or_bdd_read": False, "dose_dependent_input": False,
    }
    (args.output_dir / "dynamic_clearance_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--only_static_eligible", action="store_true")
    parser.add_argument("--horizon_seconds", type=float, default=15.0)
    parser.add_argument("--time_step_seconds", type=float, default=0.1)
    parser.add_argument("--maximum_track_interpolation_gap_seconds", type=float, default=0.25)
    parser.add_argument("--trigger_route_progress_m", type=float, default=12.0)
    parser.add_argument("--gentle_transition_length_m", type=float, default=60.0)
    parser.add_argument("--settling_margin_m", type=float, default=10.0)
    parser.add_argument("--target_speed_mps", type=float, default=5.0)
    parser.add_argument("--accel_limit_mps2", type=float, default=1.0)
    parser.add_argument("--ego_length_m", type=float, default=5.0)
    parser.add_argument("--ego_width_m", type=float, default=2.0)
    parser.add_argument("--longitudinal_buffer_m", type=float, default=3.0)
    parser.add_argument("--lateral_buffer_m", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
