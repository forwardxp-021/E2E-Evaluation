#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from tools.compare_lane_aware_diagnostics import summarize_waymo, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export comparable Stage5D Waymo lane-aware diagnostics without changing assignment logic.")
    p.add_argument("--waymo_dir", type=Path, required=True, help="Existing Waymo Stage5D context output directory.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory for waymo_lane_aware_diagnostics.json/.md and optional CSV.")
    p.add_argument("--max_rows", type=int, default=5000, help="Maximum rows to scan from arrays for bounded coverage/switch diagnostics.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def render_report(metrics: Dict[str, Any]) -> str:
    return "\n".join([
        "# Waymo Stage5D Lane-Aware Diagnostics",
        "",
        "Stage5D CORE remains the only lane-aware assignment implementation; this script only exports existing Waymo diagnostics.",
        "",
        f"- source path: `{metrics.get('path')}`",
        f"- lane_assignment_available_rate: `{metrics.get('lane_assignment_available_rate')}`",
        f"- fallback_assignment_used_rate: `{metrics.get('fallback_assignment_used_rate')}`",
        f"- candidate_projection_success_rate: `{metrics.get('candidate_projection_success_rate')}`",
        f"- adjacency_source_counts: `{metrics.get('adjacency_source_counts')}`",
        f"- lane_context_quality_counts: `{metrics.get('lane_context_quality_counts')}`",
        f"- rejection_reason_counts: `{metrics.get('rejection_reason_counts')}`",
        f"- slot_coverage_by_slot: `{metrics.get('slot_coverage_by_slot')}`",
        f"- slot_switch_rate_by_slot: `{metrics.get('slot_switch_rate_by_slot')}`",
        f"- assignment_method_counts_by_slot: `{metrics.get('assignment_method_counts_by_slot_from_debug_csv')}`",
        f"- slot_coverage_metric_source: `{metrics.get('slot_coverage_metric_source')}`",
    ]) + "\n"


def write_flat_csv(path: Path, metrics: Dict[str, Any]) -> None:
    rows = []
    for slot, coverage in (metrics.get("slot_coverage_by_slot") or {}).items():
        rows.append({"metric_group": "slot_coverage_by_slot", "key": slot, "value": coverage})
    for slot, rate in (metrics.get("slot_switch_rate_by_slot") or {}).items():
        rows.append({"metric_group": "slot_switch_rate_by_slot", "key": slot, "value": rate})
    for reason, count in (metrics.get("rejection_reason_counts") or {}).items():
        rows.append({"metric_group": "rejection_reason_counts", "key": reason, "value": count})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric_group", "key", "value"])
        w.writeheader(); w.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.waymo_dir.exists():
        raise FileNotFoundError(f"Waymo directory does not exist: {args.waymo_dir}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {args.output_dir}. Use --overwrite.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    metrics = summarize_waymo(args.waymo_dir, max_rows=args.max_rows)
    metrics["diagnostic_export_source_dir"] = str(args.waymo_dir)
    metrics["max_rows"] = args.max_rows
    write_json(args.output_dir / "waymo_lane_aware_diagnostics.json", metrics)
    (args.output_dir / "waymo_lane_aware_diagnostics.md").write_text(render_report(metrics), encoding="utf-8")
    write_flat_csv(args.output_dir / "waymo_lane_aware_diagnostics.csv", metrics)
    print(f"Wrote Waymo diagnostics: {args.output_dir}")


if __name__ == "__main__":
    main()
