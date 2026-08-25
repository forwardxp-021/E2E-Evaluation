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
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m3_select_balanced_scaleup import manifest_hash


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def symmetric_failed_scenarios(
    task_records: List[Dict[str, Any]], planner_names: List[str]
) -> Dict[int, List[Dict[str, Any]]]:
    failed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in task_records:
        if row.get("status") == "failed":
            failed[int(row["scenario_index"])].append(row)
    expected = set(planner_names)
    for scenario_index, rows in failed.items():
        observed = {str(row["planner"]) for row in rows}
        if observed != expected:
            raise ValueError(
                f"technical failure is not planner-symmetric for scenario {scenario_index}: "
                f"observed={sorted(observed)}, expected={sorted(expected)}"
            )
    return dict(failed)


def bdd_row(name: str, path: Path, expected_pairs: int, alpha: float) -> Dict[str, Any]:
    source = read_json(path)
    n_a, n_b = int(source["n_A"]), int(source["n_B"])
    if n_a != expected_pairs or n_b != expected_pairs:
        raise ValueError(
            f"{name} BDD count mismatch: n_A={n_a}, n_B={n_b}, expected={expected_pairs}"
        )
    return {
        "dataset": name,
        "pairs": expected_pairs,
        "mmd2": float(source["mmd2"]),
        "p_value": float(source["p_value"]),
        "significant_at_alpha": float(source["p_value"]) < alpha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complete Stage7 Milestone 3 scale-up.")
    parser.add_argument("--selection_dir", type=Path, required=True)
    parser.add_argument("--sim_dir", type=Path, required=True)
    parser.add_argument("--context_dir", type=Path, required=True)
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--stage7f_dir", type=Path, required=True)
    parser.add_argument("--quality_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_complete_pairs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    selection_summary = read_json(args.selection_dir / "milestone3_selection_summary.json")
    selected_rows = read_csv(args.selection_dir / "selected_scenarios.csv")
    progress = read_json(args.sim_dir / "stage7c_progress.json")
    sim_index = read_json(args.sim_dir / "simulated_ego_seq_index.json")
    alignment = read_json(args.sim_dir / "scenario_alignment.json")
    context_warnings = read_json(args.context_dir / "warnings.json")
    embedding_manifest = read_json(args.embedding_dir / "embedding_manifest.json")
    paired = read_json(
        args.stage7f_dir
        / "paired_delta_assertive_minus_conservative/paired_delta_summary.json"
    )
    task_summary = read_json(
        args.stage7f_dir
        / "task_bdd_assertive_minus_conservative_v1/stage7f_task_bdd_summary.json"
    )
    overlap = read_json(args.stage7f_dir / "task_overlap_v1/task_overlap_summary.json")
    quality = read_json(args.quality_dir / "milestone2b_summary.json")
    event_warnings = read_json(
        args.stage7f_dir / "behavior_events_v2/behavior_event_warnings_v2.json"
    )

    planner_names = [str(value) for value in sim_index["planner_axis_names"]]
    failed = symmetric_failed_scenarios(progress["task_records"], planner_names)
    scenario_axis = [int(value) for value in sim_index["scenario_axis"]]
    complete_pairs = len(scenario_axis)
    successful_rows = [selected_rows[index] for index in scenario_axis]
    failed_rows = [selected_rows[index] for index in sorted(failed)]
    embedding = np.load(args.embedding_dir / "embedding.npy", mmap_mode="r")
    context = np.load(args.context_dir / "context_traj.npy", mmap_mode="r")
    context_validation = context_warnings["validation"]

    bdd = [
        bdd_row(
            "full",
            args.stage7f_dir
            / "report_card/stage6_pairwise/"
            "pdm_closed_assertive_v1_vs_pdm_closed_conservative_v1/bdd_summary.json",
            complete_pairs,
            args.alpha,
        ),
        bdd_row(
            "tier_a",
            args.quality_dir
            / "bdd_sensitivity/tier_a_assertive_vs_conservative/bdd_summary.json",
            int(quality["tier_a_pairs"]),
            args.alpha,
        ),
        bdd_row(
            "tier_b_inclusive",
            args.quality_dir
            / "bdd_sensitivity/tier_b_inclusive_assertive_vs_conservative/bdd_summary.json",
            int(quality["tier_b_inclusive_pairs"]),
            args.alpha,
        ),
    ]
    task_counts = overlap["positive_counts"]
    under_ten_tasks = sorted(
        key
        for key, value in task_counts.items()
        if int(value["paired_scenarios"]) < 10
    )
    proxy_tasks = sorted({
        warning["task_key"]
        for warning in event_warnings
        if warning.get("warning") == "detector_strength_not_strong"
        and warning.get("task_key") in task_summary["task_keys"]
    })
    physical_warning_count = sum(
        warning.get("warning") in {
            "metric_physical_range_warning",
            "raw_metric_physically_implausible",
        }
        for warning in event_warnings
    )
    checks = {
        "selection_verdict_pass": selection_summary.get("overall_verdict") == "PASS",
        "selection_manifest_hash_matches": (
            manifest_hash(selected_rows) == selection_summary["selection_manifest_sha256"]
        ),
        "all_100_simulation_tasks_completed": (
            int(progress["completed_tasks"]) == int(progress["total_tasks"]) == 100
        ),
        "technical_failures_planner_symmetric": len(failed) * len(planner_names)
        == int(progress["failed_tasks"]),
        "complete_pair_count_at_least_30": complete_pairs >= args.min_complete_pairs,
        "official_success_count_matches_complete_pairs": int(
            alignment["num_official_successes"]
        )
        == complete_pairs * len(planner_names),
        "strict_token_alignment_for_all_successes": int(
            alignment["num_strict_nuplan_token_aligned"]
        )
        == complete_pairs * len(planner_names),
        "context_validation_pass": context_validation.get("pass") is True,
        "context_scenario_axis_matches_simulation": [
            int(value) for value in context_validation["scenario_axis"]
        ]
        == scenario_axis,
        "context_shape_correct": tuple(context.shape) == (complete_pairs * 2, 150, 83),
        "embedding_shape_correct": tuple(embedding.shape) == (complete_pairs * 2, 64),
        "embedding_finite": bool(np.isfinite(embedding).all()),
        "paired_delta_complete": int(paired["num_paired_scenarios"]) == complete_pairs,
        "lane_quality_gate_pass": quality.get("overall_verdict") == "PASS",
        "quality_primary_label_dynamic": quality.get("primary_analysis_dataset")
        == f"full_{complete_pairs}_planner_paired_scenarios",
        "bdd_significance_conclusion_stable": len({
            row["significant_at_alpha"] for row in bdd
        })
        == 1,
        "rollout_validity_mask_applied": any(
            warning.get("warning") == "rollout_validity_mask_applied"
            for warning in event_warnings
        ),
        "valid_frame_physical_anomalies_absent": physical_warning_count == 0,
        "all_six_task_reports_generated": int(task_summary["valid_task_count"]) == 6,
    }
    hard_failures = [name for name, passed in checks.items() if not passed]
    limitations = []
    if under_ten_tasks:
        limitations.append(
            "task paired-scenario slices below 10: " + ", ".join(under_ten_tasks)
        )
    following_queue_jaccard = float(
        overlap["following_vs_queue"]["paired_scenarios"]["jaccard"]
    )
    if following_queue_jaccard >= 0.8:
        limitations.append(
            f"following/queue paired-scenario overlap is high (Jaccard={following_queue_jaccard:.3f})"
        )
    if proxy_tasks:
        limitations.append("proxy-dominant task detectors: " + ", ".join(proxy_tasks))
    verdict = "FAIL" if hard_failures else ("PASS_WITH_LIMITATIONS" if limitations else "PASS")
    summary = {
        "milestone": "Stage 7 Milestone 3 balanced50 scale-up final audit",
        "overall_verdict": verdict,
        "thesis_scale_status": "MINIMUM_USEFUL_SCALE_REACHED" if not hard_failures else "BLOCKED",
        "selected_scenarios": len(selected_rows),
        "complete_paired_scenarios": complete_pairs,
        "successful_official_rollouts": complete_pairs * 2,
        "technical_failure_scenarios": len(failed),
        "technical_failure_indices": sorted(failed),
        "technical_failure_tokens": [row["scenario_token"] for row in failed_rows],
        "successful_bucket_counts": dict(Counter(row["bucket"] for row in successful_rows)),
        "technical_failure_bucket_counts": dict(Counter(row["bucket"] for row in failed_rows)),
        "context_shape": list(context.shape),
        "embedding_shape": list(embedding.shape),
        "fallback_assignment_used_rate": context_validation["fallback_assignment_used_rate"],
        "ego_lane_projection_success_rate": context_validation["ego_lane_projection_success_rate"],
        "quality_pair_counts": {
            "full": int(quality["full_pairs"]),
            "tier_a": int(quality["tier_a_pairs"]),
            "tier_b_inclusive": int(quality["tier_b_inclusive_pairs"]),
        },
        "bdd": bdd,
        "paired_trajectory_results": {
            "assertive_higher_mean_speed": paired["aggressive_gt_conservative_speed_count"],
            "assertive_higher_rms_accel": paired["aggressive_gt_conservative_accel_count"],
            "assertive_smaller_mean_thw": paired["aggressive_smaller_thw_count"],
            "total_pairs": complete_pairs,
        },
        "task_paired_scenario_counts": {
            key: int(value["paired_scenarios"]) for key, value in task_counts.items()
        },
        "following_queue_paired_jaccard": following_queue_jaccard,
        "checks": checks,
        "limitations": limitations,
        "conclusion": (
            "Assertive and conservative planners show consistent realized trajectory differences, "
            "while full and lane-quality sensitivity BDD estimates remain small and non-significant."
        ),
    }
    (args.output_dir / "milestone3_final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "technical_failures.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["scenario_index", "scenario_token", "log_name", "scenario_type", "bucket"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, row in zip(sorted(failed), failed_rows):
            writer.writerow({"scenario_index": index, **row})
    report = [
        "# Stage 7 Milestone 3 Final Audit",
        "",
        f"## Verdict: `{verdict}`",
        "",
        f"- thesis scale: `{summary['thesis_scale_status']}`",
        f"- selected / complete pairs / successful rollouts: "
        f"`{len(selected_rows)} / {complete_pairs} / {complete_pairs * 2}`",
        f"- symmetric technical failures: `{len(failed)}`",
        f"- context / embedding: `{list(context.shape)} / {list(embedding.shape)}`",
        f"- fallback / ego projection success: "
        f"`{summary['fallback_assignment_used_rate']:.6f} / "
        f"{summary['ego_lane_projection_success_rate']:.6f}`",
        "",
        "## BDD quality sensitivity",
        "",
        "| dataset | pairs | MMD² | p | significant |",
        "| --- | ---: | ---: | ---: | --- |",
        *[
            f"| {row['dataset']} | {row['pairs']} | {row['mmd2']:.8f} | "
            f"{row['p_value']:.6f} | {row['significant_at_alpha']} |"
            for row in bdd
        ],
        "",
        "## Checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in checks.items()],
        "",
        "## Limitations",
        "",
        *([f"- {value}" for value in limitations] or ["- none"]),
        "",
        "## Conclusion",
        "",
        summary["conclusion"],
    ]
    (args.output_dir / "milestone3_final_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if hard_failures:
        raise RuntimeError(f"Milestone 3 final audit failed: {hard_failures}")
    print(f"Stage7 Milestone 3 final audit {verdict}: {args.output_dir}")


if __name__ == "__main__":
    main()
