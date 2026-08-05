#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BUCKET_TYPES: Dict[str, Tuple[str, ...]] = {
    "following_interaction": (
        "following_lane_with_slow_lead",
        "following_lane_with_lead",
        "near_long_vehicle",
        "behind_long_vehicle",
    ),
    "stop_go_signal": (
        "stopping_with_lead",
        "stopping_at_traffic_light_with_lead",
        "stopping_at_traffic_light_without_lead",
        "stopping_at_stop_sign_with_lead",
        "stopping_at_stop_sign_without_lead",
        "stationary_in_traffic",
        "stationary_at_traffic_light_with_lead",
        "stationary_at_traffic_light_without_lead",
        "accelerating_at_traffic_light_with_lead",
        "accelerating_at_traffic_light_without_lead",
        "accelerating_at_traffic_light",
        "accelerating_at_stop_sign",
    ),
    "dense_interaction": (
        "near_multiple_vehicles",
        "near_pedestrian_on_crosswalk_with_ego",
        "waiting_for_pedestrian_to_cross",
        "near_multiple_pedestrians",
        "near_pedestrian_on_crosswalk",
    ),
    "lateral_turning": (
        "high_lateral_acceleration",
        "starting_left_turn",
        "starting_right_turn",
        "starting_high_speed_turn",
        "starting_protected_cross_turn",
        "starting_unprotected_cross_turn",
        "starting_protected_noncross_turn",
        "starting_unprotected_noncross_turn",
    ),
    "speed_context": (
        "near_high_speed_vehicle",
        "high_magnitude_speed",
        "medium_magnitude_speed",
    ),
}

DEFAULT_QUOTAS = {
    "actual_verified_lane_change": 8,
    "following_interaction": 10,
    "stop_go_signal": 10,
    "dense_interaction": 8,
    "lateral_turning": 7,
    "speed_context": 7,
}

SEED_BUCKET_MAP = {
    "actual_verified_lane_change": "actual_verified_lane_change",
    "following_slow_lead": "following_interaction",
    "following_with_lead": "following_interaction",
    "stopping_with_lead": "stop_go_signal",
    "stationary_in_traffic": "stop_go_signal",
    "accelerating_signal": "stop_go_signal",
    "stopping_signal": "stop_go_signal",
    "near_multiple_vehicles": "dense_interaction",
    "high_lateral_acceleration": "lateral_turning",
    "near_high_speed_vehicle": "speed_context",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_rank(token: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{token}".encode("utf-8")).hexdigest()


def load_successful_seed_rows(seed_context: Path, prior_sim_dir: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    seed_rows = read_csv(seed_context)
    index_path = prior_sim_dir / "simulated_ego_seq_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    successful_indices = [int(value) for value in index.get("scenario_axis", [])]
    if not successful_indices:
        raise ValueError(f"no successful scenario axis in {index_path}")
    if max(successful_indices) >= len(seed_rows):
        raise ValueError(
            f"successful scenario index exceeds seed context: max={max(successful_indices)}, "
            f"rows={len(seed_rows)}"
        )
    successful = []
    for source_index in successful_indices:
        source = seed_rows[source_index]
        old_bucket = source.get("bucket", "")
        if old_bucket not in SEED_BUCKET_MAP:
            raise ValueError(f"seed row {source_index} has unknown bucket {old_bucket!r}")
        successful.append({
            **source,
            "bucket": SEED_BUCKET_MAP[old_bucket],
            "selection_origin": "milestone2b_successful_pair",
            "source_scenario_index": str(source_index),
        })
    successful_set = set(successful_indices)
    failed_tokens = [
        row["scenario_token"]
        for index_value, row in enumerate(seed_rows)
        if index_value not in successful_set
    ]
    return successful, failed_tokens


def inventory_candidates(path: Path, excluded_tokens: Iterable[str]) -> Dict[str, List[Dict[str, str]]]:
    excluded = set(excluded_tokens)
    type_to_bucket = {
        scenario_type: bucket
        for bucket, scenario_types in BUCKET_TYPES.items()
        for scenario_type in scenario_types
    }
    by_token: Dict[str, Dict[str, str]] = {}
    bucket_priority = {bucket: index for index, bucket in enumerate(BUCKET_TYPES)}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            token = row.get("scenario_token", "")
            bucket = type_to_bucket.get(row.get("scenario_type", ""))
            if not token or not bucket or token in excluded:
                continue
            existing = by_token.get(token)
            if existing is None or bucket_priority[bucket] < bucket_priority[existing["bucket"]]:
                by_token[token] = {
                    **row,
                    "bucket": bucket,
                    "selection_origin": "db_tag_scaleup_candidate",
                    "source": "db_tag_scaleup_candidate",
                    "scene_token": token,
                }
    candidates: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in by_token.values():
        candidates[row["bucket"]].append(row)
    for bucket, rows in candidates.items():
        rows.sort(key=lambda row: stable_rank(row["scenario_token"], f"stage7-m3-{bucket}"))
    return dict(candidates)


def fill_quotas(
    seeds: Sequence[Dict[str, str]],
    candidates: Dict[str, List[Dict[str, str]]],
    quotas: Dict[str, int],
    *,
    max_per_log: int,
) -> List[Dict[str, str]]:
    selected = [dict(row) for row in seeds]
    token_set = {row["scenario_token"] for row in selected}
    log_counts = Counter(row["log_name"] for row in selected)
    bucket_counts = Counter(row["bucket"] for row in selected)
    cursors = Counter()
    while any(bucket_counts[bucket] < quota for bucket, quota in quotas.items()):
        progress = False
        for bucket, quota in quotas.items():
            if bucket == "actual_verified_lane_change" or bucket_counts[bucket] >= quota:
                continue
            rows = candidates.get(bucket, [])
            while cursors[bucket] < len(rows):
                row = rows[cursors[bucket]]
                cursors[bucket] += 1
                token = row["scenario_token"]
                if token in token_set or log_counts[row["log_name"]] >= max_per_log:
                    continue
                selected.append(dict(row))
                token_set.add(token)
                log_counts[row["log_name"]] += 1
                bucket_counts[bucket] += 1
                progress = True
                break
        if not progress:
            deficits = {
                bucket: quota - bucket_counts[bucket]
                for bucket, quota in quotas.items()
                if bucket_counts[bucket] < quota
            }
            raise RuntimeError(f"could not fill balanced quotas under max_per_log={max_per_log}: {deficits}")
    return selected


def build_reserve(
    candidates: Dict[str, List[Dict[str, str]]],
    selected: Sequence[Dict[str, str]],
    *,
    reserve_size: int,
    max_per_log: int,
) -> List[Dict[str, str]]:
    used_tokens = {row["scenario_token"] for row in selected}
    combined_log_counts = Counter(row["log_name"] for row in selected)
    pools = {
        bucket: [row for row in rows if row["scenario_token"] not in used_tokens]
        for bucket, rows in candidates.items()
    }
    cursors = Counter()
    reserve: List[Dict[str, str]] = []
    bucket_order = list(BUCKET_TYPES)
    while len(reserve) < reserve_size:
        progress = False
        for bucket in bucket_order:
            rows = pools.get(bucket, [])
            while cursors[bucket] < len(rows):
                row = rows[cursors[bucket]]
                cursors[bucket] += 1
                if (
                    row["scenario_token"] in used_tokens
                    or combined_log_counts[row["log_name"]] >= max_per_log
                ):
                    continue
                reserve.append({
                    **row,
                    "reserve_rank": str(len(reserve) + 1),
                })
                used_tokens.add(row["scenario_token"])
                combined_log_counts[row["log_name"]] += 1
                progress = True
                break
            if len(reserve) >= reserve_size:
                break
        if not progress:
            raise RuntimeError(f"could only build {len(reserve)} of {reserve_size} reserve rows")
    return reserve


def manifest_hash(rows: Sequence[Dict[str, str]]) -> str:
    canonical = [
        {
            "selection_index": index,
            "log_name": row["log_name"],
            "scenario_token": row["scenario_token"],
            "scenario_type": row["scenario_type"],
            "bucket": row["bucket"],
        }
        for index, row in enumerate(rows)
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a balanced 50-scenario Stage7 Milestone 3 scale-up manifest."
    )
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--seed_context", type=Path, required=True)
    parser.add_argument("--prior_sim_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_scenarios", type=int, default=50)
    parser.add_argument("--reserve_size", type=int, default=20)
    parser.add_argument("--max_per_log", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_scenarios != sum(DEFAULT_QUOTAS.values()):
        raise ValueError(
            f"default frozen quotas total {sum(DEFAULT_QUOTAS.values())}; "
            f"--target_scenarios must match, got {args.target_scenarios}"
        )
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    seeds, prior_failed_tokens = load_successful_seed_rows(args.seed_context, args.prior_sim_dir)
    if len(seeds) != 17:
        raise ValueError(f"expected 17 M2B successful seed scenarios, got {len(seeds)}")
    excluded = {row["scenario_token"] for row in seeds} | set(prior_failed_tokens)
    candidates = inventory_candidates(args.inventory_csv, excluded)
    selected = fill_quotas(
        seeds,
        candidates,
        DEFAULT_QUOTAS,
        max_per_log=args.max_per_log,
    )
    reserve = build_reserve(
        candidates,
        selected,
        reserve_size=args.reserve_size,
        max_per_log=args.max_per_log,
    )
    selected_tokens = [row["scenario_token"] for row in selected]
    checks = {
        "target_count_met": len(selected) == args.target_scenarios,
        "tokens_unique": len(set(selected_tokens)) == len(selected_tokens),
        "prior_failed_tokens_excluded": not (set(prior_failed_tokens) & set(selected_tokens)),
        "bucket_quotas_exact": Counter(row["bucket"] for row in selected) == Counter(DEFAULT_QUOTAS),
        "max_per_log_respected": max(Counter(row["log_name"] for row in selected).values()) <= args.max_per_log,
        "reserve_count_met": len(reserve) == args.reserve_size,
        "selected_reserve_disjoint": not (
            set(selected_tokens) & {row["scenario_token"] for row in reserve}
        ),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    fields = [
        "log_name", "scenario_token", "scene_token", "scenario_type", "source", "bucket",
        "selection_origin", "source_scenario_index", "db_scene_token", "db_file",
    ]
    context_dir = args.output_dir / "stage7c_candidate_context"
    context_dir.mkdir()
    write_csv(context_dir / "merged_metadata.csv", selected, fields)
    write_csv(args.output_dir / "selected_scenarios.csv", selected, fields)
    write_csv(
        args.output_dir / "technical_failure_reserve.csv",
        reserve,
        fields + ["reserve_rank"],
    )
    summary = {
        "milestone": "Stage 7 Milestone 3 balanced scale-up selection freeze",
        "overall_verdict": verdict,
        "target_scenarios": args.target_scenarios,
        "target_official_rollouts": args.target_scenarios * 2,
        "seed_successful_pairs_reused_in_selection": len(seeds),
        "new_candidate_pairs": len(selected) - len(seeds),
        "prior_technical_failure_tokens_excluded": prior_failed_tokens,
        "bucket_quotas": DEFAULT_QUOTAS,
        "bucket_counts": dict(Counter(row["bucket"] for row in selected)),
        "scenario_type_counts": dict(Counter(row["scenario_type"] for row in selected)),
        "log_count": len({row["log_name"] for row in selected}),
        "max_per_log": args.max_per_log,
        "reserve_size": len(reserve),
        "selection_manifest_sha256": manifest_hash(selected),
        "selection_policy": (
            "Manifest is frozen before M3 planner outcomes. Reserve rows may replace only "
            "documented technical scenario-extraction failures, in reserve_rank order, "
            "never based on planner behavior or BDD results."
        ),
        "checks": checks,
    }
    (args.output_dir / "milestone3_selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage 7 Milestone 3 Scale-up Selection",
        "",
        f"## Verdict: `{verdict}`",
        "",
        f"- target scenarios / rollouts: `{args.target_scenarios}` / "
        f"`{args.target_scenarios * 2}`",
        f"- M2B successful seeds / new candidates: `{len(seeds)}` / "
        f"`{len(selected) - len(seeds)}`",
        f"- buckets: `{dict(Counter(row['bucket'] for row in selected))}`",
        f"- distinct logs: `{summary['log_count']}`; max per log: `{args.max_per_log}`",
        f"- frozen manifest SHA-256: `{summary['selection_manifest_sha256']}`",
        "",
        "## Outcome-independent replacement policy",
        "",
        "- The selected 50-scenario manifest is frozen before M3 planner outcomes.",
        "- Reserve rows can replace only documented technical extraction failures.",
        "- Replacements must follow reserve_rank and cannot use planner behavior or BDD results.",
        "- Complete planner pairs remain the analysis unit.",
        "",
        "## Checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in checks.items()],
    ]
    (args.output_dir / "milestone3_selection_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if verdict != "PASS":
        raise RuntimeError(f"Milestone 3 selection failed: {checks}")
    print(f"Stage7 Milestone 3 selection PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
