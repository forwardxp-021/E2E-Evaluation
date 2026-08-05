#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validation_of(warnings: Dict[str, Any]) -> Dict[str, Any]:
    validation = warnings.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("warnings.json is missing object field validation")
    return validation


def evaluate_m2a(
    baseline_validation: Dict[str, Any],
    repaired_validation: Dict[str, Any],
    assignment_rows: List[Dict[str, str]],
    *,
    max_fallback_rate: float,
    min_absolute_improvement: float,
) -> Dict[str, Any]:
    baseline_rate = float(baseline_validation["fallback_assignment_used_rate"])
    repaired_rate = float(repaired_validation["fallback_assignment_used_rate"])
    improvement = baseline_rate - repaired_rate
    scenario_planners: Dict[int, set[str]] = {}
    for row in assignment_rows:
        scenario_planners.setdefault(int(row["scenario_index"]), set()).add(str(row["planner_name"]))
    planner_sets = list(scenario_planners.values())
    expected_planners = set().union(*planner_sets) if planner_sets else set()
    fallback_reasons = repaired_validation.get("lane_assignment_fallback_reason_counts", {})
    map_names = [str(value) for value in repaired_validation.get("map_names_used", [])]
    checks = {
        "repaired_context_validation_pass": repaired_validation.get("pass") is True,
        "scenario_local_lane_cache": repaired_validation.get("lane_cache_scope") == "map_name_plus_source_scenario",
        "lane_cache_entry_per_source_scenario": int(repaired_validation.get("lane_cache_entry_count", -1)) == len(scenario_planners),
        "map_api_cache_nonempty": int(repaired_validation.get("map_api_cache_entry_count", 0)) > 0,
        "canonical_map_names_used": bool(map_names) and "las_vegas" not in map_names,
        "log_db_map_resolution_used": int(repaired_validation.get("log_db_map_resolution_count", 0)) > 0,
        "no_lane_map_unavailable_fallback": int(fallback_reasons.get("lane_map_unavailable", 0)) == 0,
        "assignment_diagnostics_cover_all_rows": len(assignment_rows) > 0
        and len(assignment_rows) == int(repaired_validation.get("ego_seq_mask_shape", [0])[0]),
        "every_scenario_has_all_planners": bool(planner_sets)
        and all(planners == expected_planners for planners in planner_sets),
        "fallback_below_threshold": repaired_rate < max_fallback_rate,
        "fallback_improvement_sufficient": improvement >= min_absolute_improvement,
        "map_query_success": repaired_validation.get("map_query_success") is True,
        "ego_projection_majority_success": float(repaired_validation.get("ego_lane_projection_success_rate", 0.0)) > 0.5,
        "candidate_projection_majority_success": float(repaired_validation.get("candidate_lane_projection_success_rate", 0.0)) > 0.5,
    }
    return {
        "milestone": "Stage 7 Milestone 2A lane projection and adjacency fallback repair",
        "baseline_fallback_assignment_used_rate": baseline_rate,
        "repaired_fallback_assignment_used_rate": repaired_rate,
        "absolute_fallback_rate_improvement": improvement,
        "relative_fallback_rate_reduction": improvement / baseline_rate if baseline_rate else 0.0,
        "max_fallback_rate": max_fallback_rate,
        "min_absolute_improvement": min_absolute_improvement,
        "source_scenario_count": len(scenario_planners),
        "planner_names": sorted(expected_planners),
        "assignment_diagnostic_row_count": len(assignment_rows),
        "lane_cache_entry_count": repaired_validation.get("lane_cache_entry_count"),
        "map_api_cache_entry_count": repaired_validation.get("map_api_cache_entry_count"),
        "ego_lane_projection_success_rate": repaired_validation.get("ego_lane_projection_success_rate"),
        "candidate_lane_projection_success_rate": repaired_validation.get("candidate_lane_projection_success_rate"),
        "map_names_used": map_names,
        "fallback_reason_counts": fallback_reasons,
        "checks": checks,
        "overall_verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Stage7 Milestone 2A scenario-local lane-cache repair.")
    parser.add_argument("--baseline_context_dir", type=Path, required=True)
    parser.add_argument("--repaired_context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_fallback_rate", type=float, default=0.5)
    parser.add_argument("--min_absolute_improvement", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.max_fallback_rate <= 1.0:
        raise ValueError("--max_fallback_rate must be in [0,1]")
    if not 0.0 <= args.min_absolute_improvement <= 1.0:
        raise ValueError("--min_absolute_improvement must be in [0,1]")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    baseline = validation_of(read_json(args.baseline_context_dir / "warnings.json"))
    repaired = validation_of(read_json(args.repaired_context_dir / "warnings.json"))
    assignment_rows = read_csv(args.repaired_context_dir / "nuplan_lane_assignment_by_row.csv")
    summary = evaluate_m2a(
        baseline,
        repaired,
        assignment_rows,
        max_fallback_rate=args.max_fallback_rate,
        min_absolute_improvement=args.min_absolute_improvement,
    )
    (args.output_dir / "milestone2a_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 7 Milestone 2A Lane Assignment Audit",
        "",
        f"## Verdict: `{summary['overall_verdict']}`",
        "",
        f"- baseline fallback rate: `{summary['baseline_fallback_assignment_used_rate']}`",
        f"- repaired fallback rate: `{summary['repaired_fallback_assignment_used_rate']}`",
        f"- absolute improvement: `{summary['absolute_fallback_rate_improvement']}`",
        f"- relative reduction: `{summary['relative_fallback_rate_reduction']}`",
        f"- source scenarios / diagnostic rows: `{summary['source_scenario_count']}` / `{summary['assignment_diagnostic_row_count']}`",
        f"- ego projection success rate: `{summary['ego_lane_projection_success_rate']}`",
        f"- candidate projection success rate: `{summary['candidate_lane_projection_success_rate']}`",
        f"- fallback reasons: `{summary['fallback_reason_counts']}`",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in summary["checks"].items()],
    ]
    (args.output_dir / "milestone2a_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if summary["overall_verdict"] != "PASS":
        raise RuntimeError(f"Milestone 2A audit failed: {summary['checks']}")
    print(f"Stage7 Milestone 2A audit PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
