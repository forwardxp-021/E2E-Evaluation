#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

SLOT_NAMES = ["front", "left_front", "left_rear", "right_front", "right_rear"]

CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("default", {}),
    ("loose_projection", {
        "lane_search_radius": 80,
        "lane_topk_candidates": 64,
        "lane_max_lateral_distance": 4.0,
        "lane_max_heading_diff_deg": 60,
    }),
    ("loose_adjacency_v1", {
        "lane_search_radius": 80,
        "lane_topk_candidates": 64,
        "lane_max_lateral_distance": 4.0,
        "lane_max_heading_diff_deg": 60,
        "adjacent_lane_min_offset": 1.5,
        "adjacent_lane_max_offset": 8.0,
        "adjacent_lane_max_heading_diff_deg": 60,
        "lane_lateral_tolerance": 3.0,
        "slot_heading_diff_deg": 60,
    }),
    ("loose_adjacency_v2", {
        "lane_search_radius": 100,
        "lane_topk_candidates": 96,
        "lane_max_lateral_distance": 5.0,
        "lane_max_heading_diff_deg": 70,
        "adjacent_lane_min_offset": 1.2,
        "adjacent_lane_max_offset": 10.0,
        "adjacent_lane_max_heading_diff_deg": 70,
        "lane_lateral_tolerance": 3.5,
        "slot_heading_diff_deg": 70,
    }),
]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def flatten_rejections(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for slot_counts in (row.get("slot_rejection_reason_counts") or {}).values():
            if isinstance(slot_counts, dict):
                counts.update({str(k): int(v) for k, v in slot_counts.items()})
    return dict(counts)


def relation_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter({"lane_relation_unknown_count": 0, "same_lane_count": 0, "left_adjacent_count": 0, "right_adjacent_count": 0})
    for row in rows:
        if row.get("fallback_assignment_used") or not row.get("lane_assignment_available"):
            counts["lane_relation_unknown_count"] += 1
            continue
        cur = str(row.get("current_lane_id") or "")
        left = str(row.get("left_lane_id") or "")
        right = str(row.get("right_lane_id") or "")
        for slot in row.get("per_slot_debug", []) or []:
            if slot.get("assignment_method") != "lane_aware":
                continue
            lane_id = str(slot.get("slot_lane_id") or slot.get("neighbor_lane_id") or "")
            if lane_id and lane_id == cur:
                counts["same_lane_count"] += 1
            elif lane_id and left and lane_id == left:
                counts["left_adjacent_count"] += 1
            elif lane_id and right and lane_id == right:
                counts["right_adjacent_count"] += 1
            else:
                counts["lane_relation_unknown_count"] += 1
    return dict(counts)


def summarize_run(name: str, params: Dict[str, Any], run_dir: Path, returncode: int) -> Dict[str, Any]:
    warnings_payload = read_json(run_dir / "warnings.json", {})
    validation = warnings_payload.get("validation", {}) if isinstance(warnings_payload, dict) else {}
    rows = read_json(run_dir / "assignment_debug.json", [])
    rows = rows if isinstance(rows, list) else []
    rel = relation_counts(rows)
    slot_coverage = validation.get("slot_coverage_by_slot", {}) or {}
    slot_switch = validation.get("slot_id_switch_rate_by_slot", {}) or {}
    slot_sanity = {k: bool(v) for k, v in validation.items() if k.startswith("stage5d_slot") or k == "slot_sanity_passed"}
    slot_sanity_passed = bool(validation.get("slot_sanity_passed", False))
    fallback = safe_float(validation.get("fallback_assignment_used_rate"), 0.0)
    summary = {
        "config_name": name,
        "run_dir": str(run_dir),
        "returncode": int(returncode),
        "valid_config": bool(returncode == 0 and slot_sanity_passed),
        "decision_status": "valid" if returncode == 0 and slot_sanity_passed else "invalid_slot_sanity_or_build_failed",
        "parameters": params,
        "fallback_assignment_used_rate": fallback,
        "lane_assignment_available_rate": safe_float(validation.get("ego_lane_projection_success_rate"), 0.0),
        "ego_projection_success_rate": safe_float(validation.get("ego_lane_projection_success_rate"), 0.0),
        "candidate_projection_success_rate": safe_float(validation.get("candidate_lane_projection_success_rate"), 0.0),
        "lane_relation_unknown_count": int(rel.get("lane_relation_unknown_count", 0)),
        "same_lane_count": int(rel.get("same_lane_count", 0)),
        "left_adjacent_count": int(rel.get("left_adjacent_count", 0)),
        "right_adjacent_count": int(rel.get("right_adjacent_count", 0)),
        "rejection_reason_counts": flatten_rejections(rows),
        "slot_coverage_by_slot": slot_coverage,
        "slot_sanity_checks": slot_sanity,
        "slot_switch_rate_by_slot": slot_switch,
    }
    for slot in SLOT_NAMES:
        summary[f"slot_coverage_{slot}"] = safe_float(slot_coverage.get(slot), 0.0)
        summary[f"slot_switch_rate_{slot}"] = safe_float(slot_switch.get(slot), 0.0)
    return summary


def verdict(rows: List[Dict[str, Any]], substantial_drop: float) -> Dict[str, Any]:
    if not rows:
        return {"decision": "inconclusive_no_runs", "reason": "No sweep rows were collected."}
    default = next((r for r in rows if r["config_name"] == "default"), rows[0])
    valid = [r for r in rows if r.get("valid_config")]
    if not valid:
        return {"decision": "investigate_nuplan_topology_adapter", "reason": "No threshold config both completed and passed slot sanity."}
    best = min(valid, key=lambda r: r["fallback_assignment_used_rate"])
    drop = default["fallback_assignment_used_rate"] - best["fallback_assignment_used_rate"]
    if drop >= substantial_drop and best.get("valid_config"):
        return {"decision": "prefer_tuned_thresholds_stage5_style", "best_config": best["config_name"], "fallback_drop": drop, "reason": "Fallback dropped substantially while slot sanity remained true."}
    return {"decision": "investigate_nuplan_topology_adapter", "best_config": best["config_name"], "fallback_drop": drop, "reason": "Fallback did not drop substantially under valid threshold-only configs."}


def render_report(payload: Dict[str, Any]) -> str:
    lines = ["# Stage7E nuPlan Lane-Aware Threshold Sweep", "", "## Scope", "", "- Reuses existing `build_nuplan_5neighbor_context_dataset.py` CLI thresholds.", "- Does not modify `tools/lane_aware_assignment.py` and does not add a Stage7-specific assignment algorithm.", "", "## Decision", "", f"- decision: `{payload['decision']['decision']}`", f"- reason: {payload['decision']['reason']}"]
    if "best_config" in payload["decision"]:
        lines.append(f"- best_config: `{payload['decision']['best_config']}`")
        lines.append(f"- fallback_drop_vs_default: `{payload['decision']['fallback_drop']}`")
    lines += ["", "## Sweep Results", ""]
    for row in payload["runs"]:
        lines += [f"### {row['config_name']}", "", f"- valid_config: `{row['valid_config']}`", f"- fallback_assignment_used_rate: `{row['fallback_assignment_used_rate']}`", f"- lane_assignment_available_rate: `{row['lane_assignment_available_rate']}`", f"- ego_projection_success_rate: `{row['ego_projection_success_rate']}`", f"- candidate_projection_success_rate: `{row['candidate_projection_success_rate']}`", f"- lane_relation_unknown_count: `{row['lane_relation_unknown_count']}`", f"- same_lane_count: `{row['same_lane_count']}`", f"- left_adjacent_count: `{row['left_adjacent_count']}`", f"- right_adjacent_count: `{row['right_adjacent_count']}`", f"- rejection_reason_counts: `{row['rejection_reason_counts']}`", f"- slot coverage by slot: `{row['slot_coverage_by_slot']}`", f"- slot sanity checks: `{row['slot_sanity_checks']}`", f"- slot switch rate by slot: `{row['slot_switch_rate_by_slot']}`", ""]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Stage5-style threshold sweep for the nuPlan lane-aware adapter without changing assignment logic.")
    p.add_argument("--sim_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True, help="Sweep root. Each config writes into a subdirectory; summary files are written here.")
    p.add_argument("--nuplan_map_root", type=Path, default=None)
    p.add_argument("--map_name", default="")
    p.add_argument("--scenario_map_metadata_csv", type=Path, default=None)
    p.add_argument("--assignment_mode", default="lane_aware_with_geometric_fallback", choices=["lane_aware_only", "lane_aware_with_geometric_fallback", "geometric_only"])
    p.add_argument("--substantial_fallback_drop", type=float, default=0.20)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--write_projection_debug", action="store_true")
    p.add_argument("build_args", nargs=argparse.REMAINDER, help="Optional extra args passed to build_nuplan_5neighbor_context_dataset.py after --, e.g. -- --required_planners planner_a")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    extra = args.build_args[1:] if args.build_args[:1] == ["--"] else args.build_args
    for name, params in CONFIGS:
        run_dir = args.output_dir / name
        cmd = [sys.executable, "tools/build_nuplan_5neighbor_context_dataset.py", "--sim_dir", str(args.sim_dir), "--output_dir", str(run_dir), "--assignment_mode", args.assignment_mode]
        if args.nuplan_map_root is not None:
            cmd += ["--nuplan_map_root", str(args.nuplan_map_root)]
        if args.map_name:
            cmd += ["--map_name", args.map_name]
        if args.scenario_map_metadata_csv is not None:
            cmd += ["--scenario_map_metadata_csv", str(args.scenario_map_metadata_csv)]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.write_projection_debug:
            cmd.append("--write_projection_debug")
        for key, value in params.items():
            cmd += [f"--{key}", str(value)]
        cmd += extra
        print(f"[sweep] running {name}: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
        runs.append(summarize_run(name, params, run_dir, result.returncode))
    payload = {"runs": runs, "decision": verdict(runs, args.substantial_fallback_drop)}
    write_json(args.output_dir / "laneaware_threshold_sweep_summary.json", payload)
    csv_fields = ["config_name", "run_dir", "returncode", "valid_config", "decision_status", "fallback_assignment_used_rate", "lane_assignment_available_rate", "ego_projection_success_rate", "candidate_projection_success_rate", "lane_relation_unknown_count", "same_lane_count", "left_adjacent_count", "right_adjacent_count"] + [f"slot_coverage_{s}" for s in SLOT_NAMES] + [f"slot_switch_rate_{s}" for s in SLOT_NAMES]
    with (args.output_dir / "laneaware_threshold_sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in runs:
            writer.writerow({k: row.get(k, "") for k in csv_fields})
    (args.output_dir / "laneaware_threshold_sweep_report.md").write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote sweep summary to {args.output_dir}")


if __name__ == "__main__":
    main()
