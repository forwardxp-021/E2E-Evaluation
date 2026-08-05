#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TIER_ORDER = {"A": 0, "B": 1, "C": 2}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_quality_tier(
    row: Dict[str, Any],
    *,
    tier_a_max_fallback: float,
    tier_a_max_ambiguous: float,
    tier_b_max_fallback: float,
    tier_b_max_ambiguous: float,
) -> Tuple[str, List[str]]:
    fallback = float(row["fallback_rate"])
    ambiguous = float(row["ambiguous_frame_rate"])
    bad = float(row["bad_frame_rate"])
    if bad > 0:
        return "C", ["bad_lane_context_present"]
    if fallback <= tier_a_max_fallback and ambiguous <= tier_a_max_ambiguous:
        return "A", []
    if fallback <= tier_b_max_fallback and ambiguous <= tier_b_max_ambiguous:
        reasons = []
        if fallback > tier_a_max_fallback:
            reasons.append("fallback_above_tier_a")
        if ambiguous > tier_a_max_ambiguous:
            reasons.append("ambiguity_above_tier_a")
        return "B", reasons
    reasons = []
    if fallback > tier_b_max_fallback:
        reasons.append("fallback_above_tier_b")
    if ambiguous > tier_b_max_ambiguous:
        reasons.append("ambiguity_above_tier_b")
    return "C", reasons or ["quality_outside_tier_b"]


def build_quality_gate(
    assignment_rows: List[Dict[str, str]],
    *,
    tier_a_max_fallback: float,
    tier_a_max_ambiguous: float,
    tier_b_max_fallback: float,
    tier_b_max_ambiguous: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[int]]]:
    required = {
        "global_row", "scenario_index", "planner_name", "fallback_rate",
        "ambiguous_frame_rate", "bad_frame_rate", "quality_eligible_frame_rate",
    }
    row_quality: List[Dict[str, Any]] = []
    for source in assignment_rows:
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"assignment diagnostic row missing fields {missing}")
        tier, reasons = classify_quality_tier(
            source,
            tier_a_max_fallback=tier_a_max_fallback,
            tier_a_max_ambiguous=tier_a_max_ambiguous,
            tier_b_max_fallback=tier_b_max_fallback,
            tier_b_max_ambiguous=tier_b_max_ambiguous,
        )
        row_quality.append({
            **source,
            "quality_tier": tier,
            "quality_tier_reasons": json.dumps(reasons),
        })

    by_scenario: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    planner_names = sorted({str(row["planner_name"]) for row in row_quality})
    for row in row_quality:
        by_scenario[int(row["scenario_index"])].append(row)
    pair_rows: List[Dict[str, Any]] = []
    indices = {f"full_{planner}": [] for planner in planner_names}
    indices.update({f"tier_a_{planner}": [] for planner in planner_names})
    indices.update({f"tier_b_inclusive_{planner}": [] for planner in planner_names})
    for scenario_index in sorted(by_scenario):
        rows = by_scenario[scenario_index]
        observed = sorted(str(row["planner_name"]) for row in rows)
        if observed != planner_names:
            raise ValueError(
                f"scenario {scenario_index} planner set mismatch: observed={observed}, expected={planner_names}"
            )
        worst = max((str(row["quality_tier"]) for row in rows), key=TIER_ORDER.__getitem__)
        row_by_planner = {str(row["planner_name"]): row for row in rows}
        pair_rows.append({
            "scenario_index": scenario_index,
            "pair_quality_tier": worst,
            "tier_a_pair_eligible": worst == "A",
            "tier_b_inclusive_pair_eligible": TIER_ORDER[worst] <= TIER_ORDER["B"],
            "planner_row_tiers": json.dumps(
                {planner: row_by_planner[planner]["quality_tier"] for planner in planner_names},
                sort_keys=True,
            ),
            "planner_fallback_rates": json.dumps(
                {planner: float(row_by_planner[planner]["fallback_rate"]) for planner in planner_names},
                sort_keys=True,
            ),
            "planner_ambiguous_rates": json.dumps(
                {planner: float(row_by_planner[planner]["ambiguous_frame_rate"]) for planner in planner_names},
                sort_keys=True,
            ),
        })
        for planner, row in row_by_planner.items():
            index = int(row["global_row"])
            indices[f"full_{planner}"].append(index)
            if worst == "A":
                indices[f"tier_a_{planner}"].append(index)
            if TIER_ORDER[worst] <= TIER_ORDER["B"]:
                indices[f"tier_b_inclusive_{planner}"].append(index)
    return row_quality, pair_rows, indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage7 M2B symmetric planner-paired lane-context quality gates.")
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tier_a_max_fallback", type=float, default=0.05)
    parser.add_argument("--tier_a_max_ambiguous", type=float, default=0.05)
    parser.add_argument("--tier_b_max_fallback", type=float, default=0.20)
    parser.add_argument("--tier_b_max_ambiguous", type=float, default=0.20)
    parser.add_argument("--min_full_pairs_for_scale", type=int, default=15)
    parser.add_argument("--min_tier_a_pairs_for_analysis", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "tier_a_max_fallback", "tier_a_max_ambiguous",
        "tier_b_max_fallback", "tier_b_max_ambiguous",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0,1], got {value}")
    if args.tier_a_max_fallback > args.tier_b_max_fallback or args.tier_a_max_ambiguous > args.tier_b_max_ambiguous:
        raise ValueError("Tier A thresholds must be no looser than Tier B thresholds")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    assignment_rows = read_csv(args.context_dir / "nuplan_lane_assignment_by_row.csv")
    warnings = read_json(args.context_dir / "warnings.json")
    projection = read_json(args.context_dir / "nuplan_lane_projection_debug_summary.json")
    row_quality, pair_rows, indices = build_quality_gate(
        assignment_rows,
        tier_a_max_fallback=args.tier_a_max_fallback,
        tier_a_max_ambiguous=args.tier_a_max_ambiguous,
        tier_b_max_fallback=args.tier_b_max_fallback,
        tier_b_max_ambiguous=args.tier_b_max_ambiguous,
    )
    validation = warnings.get("validation", {})
    pair_tiers = Counter(str(row["pair_quality_tier"]) for row in pair_rows)
    row_tiers = Counter(str(row["quality_tier"]) for row in row_quality)
    row_tiers_by_planner = {
        planner: dict(Counter(str(row["quality_tier"]) for row in row_quality if row["planner_name"] == planner))
        for planner in sorted({str(row["planner_name"]) for row in row_quality})
    }
    tier_a_pairs = int(pair_tiers.get("A", 0))
    tier_b_pairs = int(pair_tiers.get("A", 0) + pair_tiers.get("B", 0))
    full_pairs = len(pair_rows)
    structural_checks = {
        "context_validation_pass": validation.get("pass") is True,
        "all_assignment_rows_covered": len(row_quality) == int(validation.get("ego_seq_mask_shape", [0])[0]),
        "all_rows_have_exactly_one_tier": sum(row_tiers.values()) == len(row_quality),
        "full_pair_count_sufficient": full_pairs >= args.min_full_pairs_for_scale,
        "fallback_rate_below_0_05": float(validation.get("fallback_assignment_used_rate", 1.0)) <= 0.05,
        "lane_map_unavailable_absent": int(validation.get("lane_assignment_fallback_reason_counts", {}).get("lane_map_unavailable", 0)) == 0,
        "paired_indices_symmetric": all(
            len(indices[f"{prefix}_{planner}"]) == len(indices[f"{prefix}_{other}"])
            for prefix in ("full", "tier_a", "tier_b_inclusive")
            for planner in row_tiers_by_planner
            for other in row_tiers_by_planner
        ),
    }
    if not all(structural_checks.values()):
        scale_readiness = "BLOCKED"
    elif tier_a_pairs >= args.min_tier_a_pairs_for_analysis:
        scale_readiness = "READY_TO_SCALE"
    else:
        scale_readiness = "READY_TO_SCALE_WITH_TIER_A_SAMPLE_GROWTH"
    unknown = projection.get("lane_relation_unknown_breakdown", {})
    unknown_total = int(sum(int(value) for value in unknown.values()))
    summary = {
        "milestone": "Stage 7 Milestone 2B lane-context quality and scale readiness",
        "overall_verdict": "PASS" if all(structural_checks.values()) else "FAIL",
        "scale_readiness": scale_readiness,
        "primary_analysis_dataset": f"full_{full_pairs}_planner_paired_scenarios",
        "sensitivity_analysis_datasets": ["tier_a_paired", "tier_b_inclusive_paired"],
        "selection_bias_policy": "Quality gates are symmetric at scenario-pair level and sensitivity-only because lane quality is measured on realized planner rollouts.",
        "thresholds": {
            "tier_a_max_fallback": args.tier_a_max_fallback,
            "tier_a_max_ambiguous": args.tier_a_max_ambiguous,
            "tier_b_max_fallback": args.tier_b_max_fallback,
            "tier_b_max_ambiguous": args.tier_b_max_ambiguous,
        },
        "full_pairs": full_pairs,
        "tier_a_pairs": tier_a_pairs,
        "tier_b_inclusive_pairs": tier_b_pairs,
        "row_tier_counts": dict(row_tiers),
        "row_tier_counts_by_planner": row_tiers_by_planner,
        "pair_tier_counts": dict(pair_tiers),
        "fallback_assignment_used_rate": validation.get("fallback_assignment_used_rate"),
        "relation_unknown_count": projection.get("lane_relation_unknown_count"),
        "relation_unknown_breakdown": unknown,
        "relation_unknown_breakdown_fraction": {
            key: int(value) / unknown_total if unknown_total else 0.0 for key, value in unknown.items()
        },
        "structural_checks": structural_checks,
        "limitations": (
            []
            if tier_a_pairs >= args.min_tier_a_pairs_for_analysis
            else ["tier_a_paired_sample_below_analysis_target"]
        ),
    }
    write_csv(args.output_dir / "row_quality_tiers.csv", row_quality)
    write_csv(args.output_dir / "paired_quality_gate.csv", pair_rows)
    index_dir = args.output_dir / "indices"
    index_dir.mkdir()
    for name, values in indices.items():
        np.save(index_dir / f"{name}.npy", np.asarray(values, dtype=np.int64))
    (args.output_dir / "milestone2b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 7 Milestone 2B Quality Stratification",
        "",
        f"## Verdict: `{summary['overall_verdict']}`",
        "",
        f"- scale readiness: `{scale_readiness}`",
        f"- full / Tier A / Tier B-inclusive pairs: `{full_pairs}` / `{tier_a_pairs}` / `{tier_b_pairs}`",
        f"- row tiers: `{dict(row_tiers)}`",
        f"- row tiers by planner: `{row_tiers_by_planner}`",
        f"- relation-unknown breakdown: `{unknown}`",
        "",
        "## Analysis policy",
        "",
        "- Primary thesis analysis uses all complete planner-paired scenarios.",
        "- Tier A and Tier B-inclusive subsets are symmetric scenario-pair sensitivity analyses only.",
        "- Do not select individual planner rows after observing realized lane quality; that would break pairing and can induce post-treatment selection bias.",
        "",
        "## Structural checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in structural_checks.items()],
    ]
    (args.output_dir / "milestone2b_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if summary["overall_verdict"] != "PASS":
        raise RuntimeError(f"Milestone 2B quality gate failed: {structural_checks}")
    print(f"Stage7 Milestone 2B quality gate PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
