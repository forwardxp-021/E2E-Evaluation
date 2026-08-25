#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools import stage7_m6_4c_audit_locked_recovery as recovery


SCHEMA_VERSION = "stage7_m6_4d_high_motion_supplement_freeze_v1"
READY_STATUS = "FROZEN_BEFORE_M6_4D_SUPPLEMENT_ROLLOUTS"
TASK = "high_motion_dynamics"
DEFAULT_SALT = "stage7-m6.4d-high-motion-supplement-v1"
OUTPUT_FIELDS = batch.PRIMARY_FIELDS
PROBE_FIELDS = [
    "probe_rank",
    "stable_rank_sha256",
    "log_name",
    "scenario_token",
    "scenario_type",
    "db_file",
    "token_found",
    "scene_position",
    "scene_count",
    "official_scene_position_valid",
    "hydra_requires_quoted_token",
    "decision",
]


def read_csv(path: Path, required: Iterable[str] = ()) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = recovery.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} actual={actual}")
    return actual


def stable_rank(row: Mapping[str, str], salt: str) -> str:
    value = f"{salt}:{TASK}:{row['log_name']}:{row['scenario_token']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def successful_recovery_results(path: Path) -> List[Dict[str, Any]]:
    state = recovery.read_json(path)
    results = list(state.get("results", []))
    if int(state.get("selected_action_count", -1)) != len(results):
        raise ValueError(f"recovery state result count mismatch: {path}")
    if int(state.get("failed", -1)) != 0:
        raise ValueError(f"recovery state contains failures: {path}")
    if int(state.get("succeeded", -1)) != len(results):
        raise ValueError(f"recovery state success count mismatch: {path}")
    if any(row.get("status") != "SUCCEEDED" or row.get("pass") is not True for row in results):
        raise ValueError(f"recovery state contains a non-PASS result: {path}")
    return results


def validate_inputs(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, int]]:
    manifest = recovery.read_json(args.locked_manifest)
    if manifest.get("status") != recovery.READY_STATUS:
        raise ValueError("original M6.4 manifest is not frozen before locked rollouts")
    validate_sha(
        args.eligible_inventory,
        str(manifest.get("candidate_pool_sha256", "")),
        "eligible candidate inventory",
    )
    validate_sha(
        args.primary_csv,
        str(manifest.get("primary_collection_csv_sha256", "")),
        "original primary CSV",
    )
    validate_sha(
        args.reserve_csv,
        str(manifest.get("reserve_collection_csv_sha256", "")),
        "original reserve CSV",
    )
    validate_sha(
        args.development_metadata_csv,
        str(manifest.get("development_metadata_sha256", "")),
        "development metadata",
    )
    validate_sha(
        args.stage7c_tool,
        str(manifest.get("stage7c_tool_sha256", "")),
        "frozen Stage7C tool",
    )

    audit_summary = recovery.read_json(args.m6_4c_audit_summary)
    if audit_summary.get("schema_version") != recovery.SCHEMA_VERSION:
        raise ValueError("M6.4C audit summary schema mismatch")
    if any(bool(value) for value in audit_summary.get("forbidden_inputs_read", {}).values()):
        raise ValueError("M6.4C audit reports forbidden post-treatment input use")
    audit_inputs = audit_summary.get("input_audit", {})
    validate_sha(
        args.batch_status_csv,
        str(audit_inputs.get("batch_status_csv_sha256", "")),
        "M6.4B status CSV",
    )
    validate_sha(
        args.batch_manifest,
        str(audit_inputs.get("batch_manifest_sha256", "")),
        "M6.4B batch manifest",
    )
    validate_sha(
        args.batch_tool,
        str(audit_inputs.get("batch_tool_sha256", "")),
        "frozen M6.4B tool",
    )

    original_status = read_csv(
        args.batch_status_csv, {"task", "status", "scenario_token", "collection_order"}
    )
    if len(original_status) != int(manifest.get("planned_primary_scenarios", -1)):
        raise ValueError("M6.4B status row count differs from the frozen primary count")
    if any(row["status"] not in {"SUCCEEDED", "FAILED_REVIEW_REQUIRED"} for row in original_status):
        raise ValueError("M6.4B full batch is not in a terminal technical state")

    recovery_results = successful_recovery_results(args.quoted_recovery_state)
    recovery_results += successful_recovery_results(args.reserve_recovery_state)
    counts = Counter(row["task"] for row in original_status if row["status"] == "SUCCEEDED")
    counts.update(str(row["task"]) for row in recovery_results)
    required = {
        str(task): int(value)
        for task, value in manifest.get("required_complete_pairs_by_task", {}).items()
    }
    if TASK not in required:
        raise ValueError(f"original manifest has no required quota for {TASK}")
    deficit = max(0, required[TASK] - int(counts[TASK]))
    if deficit != args.primary_count:
        raise ValueError(
            f"supplement primary_count must equal the audited high-motion deficit: {deficit}"
        )
    if any(counts[task] < target for task, target in required.items() if task != TASK):
        raise ValueError("a non-high-motion task remains below its frozen quota")

    batch_manifest = recovery.read_json(args.batch_manifest)
    if batch_manifest.get("planners") != batch.EXPECTED_PLANNERS:
        raise ValueError("frozen M6.4B planner list mismatch")
    if batch_manifest.get("batch_tool_sha256") != recovery.sha256_file(args.batch_tool):
        raise ValueError("batch manifest does not match the frozen batch tool")
    frozen_runtime = batch_manifest.get("frozen_input_audit", {})
    if batch.current_planner_fingerprints(batch.EXPECTED_PLANNERS) != frozen_runtime.get(
        "planner_parameter_fingerprints"
    ):
        raise ValueError("planner fingerprints differ from the frozen M6.4B runtime")
    if batch.resolve_git_commit(args.nuplan_devkit_root) != frozen_runtime.get(
        "nuplan_devkit_commit"
    ):
        raise ValueError("nuPlan commit differs from the frozen M6.4B runtime")
    if batch.resolve_git_commit(args.tuplan_garage_root) != frozen_runtime.get(
        "tuplan_garage_commit"
    ):
        raise ValueError("tuPlan Garage commit differs from the frozen M6.4B runtime")
    return manifest, {task: int(counts[task]) for task in required}


def top_ranked_candidates(
    path: Path,
    *,
    salt: str,
    probe_limit: int,
    excluded_tokens: set[str],
    excluded_logs: set[str],
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    required = {
        "task",
        "log_name",
        "scenario_token",
        "scenario_type",
        "db_file",
        "db_scene_token",
        "scenario_tag_token",
    }
    heap: List[Tuple[int, str, Dict[str, str]]] = []
    counters: Counter = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"eligible inventory missing columns: {missing}")
        for raw in reader:
            counters["inventory_rows"] += 1
            if raw.get("task") != TASK:
                continue
            counters["high_motion_rows"] += 1
            token = str(raw.get("scenario_token", "")).strip()
            log_name = str(raw.get("log_name", "")).strip()
            if token in excluded_tokens:
                counters["excluded_original_or_development_token"] += 1
                continue
            if log_name in excluded_logs:
                counters["excluded_original_or_development_log"] += 1
                continue
            row = dict(raw)
            rank = stable_rank(row, salt)
            row["stable_rank_sha256"] = rank
            item = (-int(rank, 16), token, row)
            if len(heap) < probe_limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            counters["eligible_after_overlap_exclusion"] += 1
    rows = [item[2] for item in heap]
    rows.sort(key=lambda row: (row["stable_rank_sha256"], row["scenario_token"]))
    return rows, dict(counters)


def inspect_and_select(
    candidates: Sequence[Dict[str, str]],
    *,
    db_root: Path,
    required_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    used_logs: set[str] = set()
    for probe_rank, row in enumerate(candidates, start=1):
        technical = recovery.inspect_token_scene_position(
            db_root / row["db_file"], row["scenario_token"]
        )
        if row["log_name"] in used_logs:
            decision = "EXCLUDED_DUPLICATE_SUPPLEMENT_LOG"
        elif not technical["token_found"]:
            decision = "EXCLUDED_TOKEN_NOT_FOUND"
        elif not technical["official_scene_position_valid"]:
            decision = "EXCLUDED_INVALID_SCENE_POSITION"
        else:
            decision = "SELECTED_TECHNICALLY_RUNNABLE"
            selected.append(dict(row))
            used_logs.add(row["log_name"])
        audit.append(
            {
                **row,
                **technical,
                "probe_rank": probe_rank,
                "decision": decision,
            }
        )
        if len(selected) == required_count:
            break
    return selected, audit


def number_rows(
    rows: Sequence[Mapping[str, Any]], role: str, salt: str
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        result.append(
            {
                **source,
                "collection_order": index,
                "task": TASK,
                "task_rank": index,
                "scene_token": source["scenario_token"],
                "selection_role": role,
                "selection_salt": salt,
            }
        )
    return result


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    manifest, complete_counts = validate_inputs(args)
    original_primary = read_csv(args.primary_csv, {"scenario_token", "log_name"})
    original_reserve = read_csv(args.reserve_csv, {"scenario_token", "log_name"})
    development = read_csv(
        args.development_metadata_csv, {"scenario_token", "log_name"}
    )
    excluded_tokens = {
        row["scenario_token"] for row in original_primary + original_reserve + development
    }
    excluded_logs = {row["log_name"] for row in original_primary + original_reserve + development}
    ranked, scan_audit = top_ranked_candidates(
        args.eligible_inventory,
        salt=args.selection_salt,
        probe_limit=args.candidate_probe_limit,
        excluded_tokens=excluded_tokens,
        excluded_logs=excluded_logs,
    )
    total_required = args.primary_count + args.reserve_count
    selected, probe_audit = inspect_and_select(
        ranked, db_root=args.nuplan_db_root, required_count=total_required
    )
    if len(selected) != total_required:
        raise ValueError(
            f"candidate probe found only {len(selected)} technically runnable distinct-log "
            f"rows; required={total_required}, probe_limit={args.candidate_probe_limit}"
        )
    primary = number_rows(
        selected[: args.primary_count], "supplemental_primary", args.selection_salt
    )
    reserve = number_rows(
        selected[args.primary_count :], "supplemental_technical_reserve", args.selection_salt
    )
    all_logs = [row["log_name"] for row in primary + reserve]
    all_tokens = [row["scenario_token"] for row in primary + reserve]
    if len(all_logs) != len(set(all_logs)) or len(all_tokens) != len(set(all_tokens)):
        raise AssertionError("supplement selection is not token/log unique")

    args.output_dir.mkdir(parents=True)
    primary_path = args.output_dir / "m6_4d_locked_primary_collection.csv"
    reserve_path = args.output_dir / "m6_4d_locked_reserve_collection.csv"
    write_csv(primary_path, primary, OUTPUT_FIELDS)
    write_csv(reserve_path, reserve, OUTPUT_FIELDS)
    write_csv(args.output_dir / "m6_4d_candidate_probe_audit.csv", probe_audit, PROBE_FIELDS)
    input_hashes = {
        "locked_manifest_sha256": recovery.sha256_file(args.locked_manifest),
        "eligible_inventory_sha256": recovery.sha256_file(args.eligible_inventory),
        "development_metadata_sha256": recovery.sha256_file(args.development_metadata_csv),
        "original_primary_csv_sha256": recovery.sha256_file(args.primary_csv),
        "original_reserve_csv_sha256": recovery.sha256_file(args.reserve_csv),
        "batch_status_csv_sha256": recovery.sha256_file(args.batch_status_csv),
        "batch_manifest_sha256": recovery.sha256_file(args.batch_manifest),
        "m6_4c_audit_summary_sha256": recovery.sha256_file(args.m6_4c_audit_summary),
        "quoted_recovery_state_sha256": recovery.sha256_file(args.quoted_recovery_state),
        "reserve_recovery_state_sha256": recovery.sha256_file(args.reserve_recovery_state),
        "stage7c_tool_sha256": recovery.sha256_file(args.stage7c_tool),
        "batch_tool_sha256": recovery.sha256_file(args.batch_tool),
        "selection_tool_sha256": recovery.sha256_file(Path(__file__)),
    }
    supplement_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": READY_STATUS,
        "ready_to_launch_supplemental_rollouts": True,
        "analysis_role": "PRETREATMENT_METADATA_AND_TECHNICAL_RUNNABILITY_ONLY_OUTCOME_BLIND",
        "task": TASK,
        "selection_salt": args.selection_salt,
        "candidate_probe_limit": args.candidate_probe_limit,
        "input_hashes": input_hashes,
        "planner_parameter_fingerprints": manifest["planner_parameter_fingerprints"],
        "planners": manifest["planners"],
        "required_complete_pairs": int(manifest["required_complete_pairs_by_task"][TASK]),
        "complete_pairs_before_supplement_by_task": complete_counts,
        "high_motion_deficit_before_supplement": args.primary_count,
        "planned_primary_scenarios": len(primary),
        "maximum_reserve_scenarios": len(reserve),
        "primary_manifest_sha256": recovery.canonical_rows_hash(primary),
        "reserve_manifest_sha256": recovery.canonical_rows_hash(reserve),
        "primary_collection_csv_sha256": recovery.sha256_file(primary_path),
        "reserve_collection_csv_sha256": recovery.sha256_file(reserve_path),
        "supplement_token_overlap_with_prior_or_development": 0,
        "supplement_log_overlap_with_prior_or_development": 0,
        "supplement_internal_token_duplicates": 0,
        "supplement_internal_log_duplicates": 0,
        "candidate_scan_audit": scan_audit,
        "candidate_probe_count": len(probe_audit),
        "candidate_probe_decisions": dict(Counter(row["decision"] for row in probe_audit)),
        "forbidden_inputs_read": {
            "embedding": False,
            "bdd": False,
            "effect_size": False,
            "trajectory_metrics": False,
            "planner_outcome_for_selection": False,
        },
        "freeze_rules": [
            "Only the original eligible pre-treatment inventory and SQLite technical runnability are used for selection.",
            "Every development and original M6.4 primary/reserve token and log is excluded.",
            "The fixed supplemental salt and probe limit determine a reproducible order.",
            "Five supplemental primary rows are all attempted without inspecting effect size.",
            "Frozen supplemental reserve rows may be used only for documented technical failures.",
        ],
    }
    write_json(args.output_dir / "m6_4d_locked_supplement_manifest.json", supplement_manifest)
    report = [
        "# Stage 7 M6.4D High-Motion Supplemental Freeze",
        "",
        f"Status: `{READY_STATUS}`",
        "",
        f"High-motion complete pairs before supplement: `{complete_counts[TASK]}`.",
        f"Frozen target: `{supplement_manifest['required_complete_pairs']}`; deficit: `{args.primary_count}`.",
        f"Selected supplemental primary/reserve: `{len(primary)}` / `{len(reserve)}`.",
        f"Candidate rows technically probed: `{len(probe_audit)}`.",
        "",
        "No embedding, BDD, effect size, trajectory metric, or planner outcome was used for selection.",
    ]
    (args.output_dir / "m6_4d_supplement_freeze_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": READY_STATUS,
                "primary": len(primary),
                "reserve": len(reserve),
                "primary_manifest_sha256": supplement_manifest["primary_manifest_sha256"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an outcome-blind M6.4D high-motion supplemental collection."
    )
    parser.add_argument("--locked_manifest", type=Path, required=True)
    parser.add_argument("--eligible_inventory", type=Path, required=True)
    parser.add_argument("--development_metadata_csv", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--reserve_csv", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--m6_4c_audit_summary", type=Path, required=True)
    parser.add_argument("--quoted_recovery_state", type=Path, required=True)
    parser.add_argument("--reserve_recovery_state", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--tuplan_garage_root", type=Path, required=True)
    parser.add_argument("--stage7c_tool", type=Path, default=Path("tools/stage7c1_run_nuplan_simulation.py"))
    parser.add_argument("--batch_tool", type=Path, default=Path("tools/stage7_m6_4b_run_locked_rollouts.py"))
    parser.add_argument("--selection_salt", default=DEFAULT_SALT)
    parser.add_argument("--primary_count", type=int, default=5)
    parser.add_argument("--reserve_count", type=int, default=5)
    parser.add_argument("--candidate_probe_limit", type=int, default=2048)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.primary_count < 1 or args.reserve_count < 0:
        parser.error("--primary_count must be >= 1 and --reserve_count must be >= 0")
    if args.candidate_probe_limit < args.primary_count + args.reserve_count:
        parser.error("--candidate_probe_limit must cover primary_count + reserve_count")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
