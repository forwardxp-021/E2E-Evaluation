#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_2_locked_task_bdd import (
    PRETREATMENT_TASKS,
    planner_fingerprints,
)
from tools.stage7_m6_scenario_conditioned_bdd import sha256_file
from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


STATUS_READY = "FROZEN_BEFORE_LOCKED_ROLLOUTS"
STATUS_BLOCKED = "BLOCKED_INSUFFICIENT_PRETREATMENT_INVENTORY"
REQUIRED_INVENTORY_COLUMNS = {
    "db_file",
    "log_name",
    "scenario_token",
    "scenario_type",
}
OUTPUT_FIELDS = [
    "collection_order",
    "task",
    "task_rank",
    "log_name",
    "scenario_token",
    "scene_token",
    "scenario_type",
    "db_file",
    "db_scene_token",
    "scenario_tag_token",
    "selection_role",
    "selection_salt",
]


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_rank(row: Mapping[str, str], task: str, salt: str) -> str:
    value = f"{salt}:{task}:{row['log_name']}:{row['scenario_token']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def current_planner_fingerprints(planners: Sequence[str]) -> Dict[str, str]:
    rows: List[Dict[str, str]] = []
    for planner in planners:
        if planner not in PLANNER_PROFILES:
            raise ValueError(f"planner profile is not registered in Stage7C: {planner}")
        rows.append(
            {
                "planner_name": planner,
                "parameters_json": json.dumps(
                    PLANNER_PROFILES[planner]["parameters"], ensure_ascii=False
                ),
            }
        )
    return planner_fingerprints(pd.DataFrame(rows), planners)


def validate_frozen_inputs(
    *,
    development_metadata: Path,
    m6_2_lock_path: Path,
    power_path: Path,
    planner_a: str,
    planner_b: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, Dict[str, str]]:
    lock = read_json(m6_2_lock_path)
    power = read_json(power_path)
    if lock.get("status") != "FROZEN_BEFORE_NEW_CONFIRMATION_DATA":
        raise ValueError("M6.2 lock is not frozen before new confirmation data")
    if power.get("status") != "FROZEN_BEFORE_LOCKED_CONFIRMATION":
        raise ValueError("M6.3 power justification is not frozen")
    if power.get("m6_2_lock_spec_sha256") != sha256_file(m6_2_lock_path):
        raise ValueError("M6.3 power justification does not match the M6.2 lock SHA256")
    m6_2_tool = Path(__file__).with_name("stage7_m6_2_locked_task_bdd.py")
    m6_3_tool = Path(__file__).with_name("stage7_m6_3_simulation_power_analysis.py")
    if lock.get("analysis_tool_sha256") != sha256_file(m6_2_tool):
        raise ValueError("M6.2 lock analysis_tool_sha256 does not match current code")
    if power.get("power_analysis_tool_sha256") != sha256_file(m6_3_tool):
        raise ValueError("M6.3 power analysis tool SHA256 does not match current code")
    if lock.get("development_metadata_sha256") != sha256_file(development_metadata):
        raise ValueError("development metadata SHA256 does not match the M6.2 lock")

    frozen_tasks = lock.get("task_conditioned_secondary", {}).get("task_definitions", {})
    expected_tasks = {key: list(value) for key, value in PRETREATMENT_TASKS.items()}
    if frozen_tasks != expected_tasks:
        raise ValueError("M6.2 task definitions differ from current frozen task mapping")
    power_tasks = set(power.get("required_complete_pairs_by_task", {}))
    if power_tasks != set(expected_tasks):
        raise ValueError("M6.3 task family differs from the M6.2 frozen mapping")

    metadata = pd.read_csv(development_metadata)
    required_metadata = {"scenario_token", "log_name", "planner_name", "parameters_json"}
    missing = sorted(required_metadata - set(metadata.columns))
    if missing:
        raise ValueError(f"development metadata missing columns: {missing}")
    planners = [planner_a, planner_b]
    development_fp = planner_fingerprints(metadata, planners)
    current_fp = current_planner_fingerprints(planners)
    frozen_fp = lock.get("planner_parameter_fingerprints", {})
    if development_fp != frozen_fp or current_fp != frozen_fp:
        raise ValueError(
            "planner treatment fingerprint mismatch among development data, M6.2 lock, "
            "and current Stage7C profiles"
        )
    return lock, power, metadata, current_fp


def inventory_candidates(
    inventory_csv: Path,
    *,
    development_tokens: Iterable[str],
    development_logs: Iterable[str],
    db_root: Path,
    selection_salt: str,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any], List[Dict[str, Any]]]:
    if not inventory_csv.is_file():
        raise FileNotFoundError(inventory_csv)
    type_to_task: Dict[str, str] = {}
    for task, scenario_types in PRETREATMENT_TASKS.items():
        for scenario_type in scenario_types:
            if scenario_type in type_to_task:
                raise ValueError(f"scenario_type belongs to multiple frozen tasks: {scenario_type}")
            type_to_task[scenario_type] = task

    dev_tokens = {str(value) for value in development_tokens}
    dev_logs = {str(value) for value in development_logs}
    records_by_token: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    token_locations: Dict[str, set] = defaultdict(set)
    inventory_rows = 0
    frozen_type_rows = 0
    with inventory_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_INVENTORY_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"inventory missing columns: {missing}")
        for raw in reader:
            inventory_rows += 1
            scenario_type = str(raw.get("scenario_type", "")).strip()
            task = type_to_task.get(scenario_type)
            if task is None:
                continue
            frozen_type_rows += 1
            token = str(raw.get("scenario_token", "")).strip()
            log_name = str(raw.get("log_name", "")).strip()
            db_file = str(raw.get("db_file", "")).strip()
            if not token or not log_name or not db_file:
                continue
            token_locations[token].add((log_name, db_file))
            records_by_token[token].setdefault(scenario_type, dict(raw))

    pools: Dict[str, List[Dict[str, str]]] = {task: [] for task in PRETREATMENT_TASKS}
    exclusion_counts: Counter = Counter()
    exclusions: List[Dict[str, Any]] = []
    for token, by_type in records_by_token.items():
        reasons: List[str] = []
        if len(token_locations[token]) != 1:
            reasons.append("conflicting_log_or_db_for_token")
        if len(by_type) != 1:
            reasons.append("ambiguous_multiple_frozen_scenario_types")
        row = next(iter(by_type.values()))
        log_name = str(row["log_name"]).strip()
        if token in dev_tokens:
            reasons.append("development_scenario_overlap")
        if log_name in dev_logs:
            reasons.append("development_log_overlap")
        db_path = db_root / str(row["db_file"]).strip()
        if not db_path.is_file():
            reasons.append("db_file_missing")
        if reasons:
            for reason in reasons:
                exclusion_counts[reason] += 1
            exclusions.append(
                {
                    "scenario_token": token,
                    "log_name": log_name,
                    "scenario_types": "|".join(sorted(by_type)),
                    "reasons": "|".join(sorted(set(reasons))),
                }
            )
            continue
        scenario_type = next(iter(by_type))
        task = type_to_task[scenario_type]
        candidate = {
            "task": task,
            "log_name": log_name,
            "scenario_token": token,
            "scene_token": token,
            "scenario_type": scenario_type,
            "db_file": str(row["db_file"]).strip(),
            "db_scene_token": str(row.get("db_scene_token", "")).strip(),
            "scenario_tag_token": str(row.get("scenario_tag_token", "")).strip(),
            "selection_salt": selection_salt,
        }
        pools[task].append(candidate)

    for task, rows in pools.items():
        rows.sort(key=lambda row: stable_rank(row, task, selection_salt))
        for rank, row in enumerate(rows, 1):
            row["candidate_rank"] = str(rank)
    audit = {
        "inventory_rows": inventory_rows,
        "frozen_scenario_type_rows": frozen_type_rows,
        "unique_tokens_with_frozen_types": len(records_by_token),
        "eligible_unique_candidates": sum(len(rows) for rows in pools.values()),
        "eligible_distinct_logs": len(
            {row["log_name"] for rows in pools.values() for row in rows}
        ),
        "eligible_by_task_before_log_cap": {task: len(rows) for task, rows in pools.items()},
        "excluded_unique_tokens": len(exclusions),
        "exclusion_reason_counts": dict(exclusion_counts),
    }
    return pools, audit, exclusions


def select_round_robin(
    pools: Mapping[str, Sequence[Mapping[str, str]]],
    quotas: Mapping[str, int],
    *,
    max_per_log: int,
    already_used_tokens: Iterable[str] = (),
    initial_log_counts: Mapping[str, int] | None = None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    used = set(already_used_tokens)
    log_counts = Counter(initial_log_counts or {})
    selected: List[Dict[str, str]] = []
    task_counts: Counter = Counter()
    cursors: Counter = Counter()
    tasks = list(quotas)
    while any(task_counts[task] < int(quotas[task]) for task in tasks):
        progress = False
        for task in tasks:
            if task_counts[task] >= int(quotas[task]):
                continue
            rows = pools.get(task, ())
            while cursors[task] < len(rows):
                row = dict(rows[cursors[task]])
                cursors[task] += 1
                token = row["scenario_token"]
                if token in used or log_counts[row["log_name"]] >= max_per_log:
                    continue
                selected.append(row)
                used.add(token)
                log_counts[row["log_name"]] += 1
                task_counts[task] += 1
                progress = True
                break
        if not progress:
            break
    deficits = {
        task: max(0, int(quota) - int(task_counts[task]))
        for task, quota in quotas.items()
    }
    return selected, deficits


def canonical_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "collection_order": int(row["collection_order"]),
            "task": row["task"],
            "task_rank": int(row["task_rank"]),
            "log_name": row["log_name"],
            "scenario_token": row["scenario_token"],
            "scenario_type": row["scenario_type"],
            "db_file": row["db_file"],
            "selection_role": row["selection_role"],
        }
        for row in rows
    ]
    return canonical_hash(canonical)


def number_selected(
    rows: Sequence[Mapping[str, str]], role: str
) -> List[Dict[str, Any]]:
    task_counts: Counter = Counter()
    numbered: List[Dict[str, Any]] = []
    for order, source in enumerate(rows, 1):
        row = dict(source)
        task_counts[row["task"]] += 1
        row.update(
            {
                "collection_order": order,
                "task_rank": task_counts[row["task"]],
                "selection_role": role,
            }
        )
        numbered.append(row)
    return numbered


def freeze_collection(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    lock, power, metadata, planner_fp = validate_frozen_inputs(
        development_metadata=args.development_metadata_csv,
        m6_2_lock_path=args.m6_2_lock_spec,
        power_path=args.power_justification_file,
        planner_a=args.planner_a,
        planner_b=args.planner_b,
    )
    development_tokens = set(metadata["scenario_token"].astype(str))
    development_logs = set(metadata["log_name"].astype(str))
    pools, inventory_audit, exclusions = inventory_candidates(
        args.inventory_csv,
        development_tokens=development_tokens,
        development_logs=development_logs,
        db_root=args.nuplan_db_root,
        selection_salt=args.selection_salt,
    )

    gross_per_task = int(power["planned_gross_pairs_per_task_with_attrition"])
    primary_quotas = {task: gross_per_task for task in PRETREATMENT_TASKS}
    primary, primary_deficits = select_round_robin(
        pools, primary_quotas, max_per_log=args.max_per_log
    )
    primary_counts = Counter(row["task"] for row in primary)
    primary_log_counts = Counter(row["log_name"] for row in primary)

    reserve_quotas = {task: args.reserve_per_task for task in PRETREATMENT_TASKS}
    reserve, reserve_deficits = select_round_robin(
        pools,
        reserve_quotas,
        max_per_log=args.max_per_log,
        already_used_tokens=(row["scenario_token"] for row in primary),
        initial_log_counts=primary_log_counts,
    )
    reserve_counts = Counter(row["task"] for row in reserve)
    ready = not any(primary_deficits.values()) and not any(reserve_deficits.values())
    status = STATUS_READY if ready else STATUS_BLOCKED

    eligible_rows: List[Dict[str, Any]] = []
    for task, rows in pools.items():
        eligible_rows.extend(rows)
    eligible_rows.sort(key=lambda row: (row["task"], int(row["candidate_rank"])))
    candidate_fields = [
        "task",
        "candidate_rank",
        "log_name",
        "scenario_token",
        "scene_token",
        "scenario_type",
        "db_file",
        "db_scene_token",
        "scenario_tag_token",
        "selection_salt",
    ]
    write_csv(args.output_dir / "m6_4_eligible_candidate_inventory.csv", eligible_rows, candidate_fields)
    write_csv(
        args.output_dir / "m6_4_excluded_candidate_audit.csv",
        exclusions,
        ["scenario_token", "log_name", "scenario_types", "reasons"],
    )

    capacity_rows = []
    for task in PRETREATMENT_TASKS:
        capacity_rows.append(
            {
                "task": task,
                "eligible_before_log_cap": len(pools[task]),
                "required_primary_gross": primary_quotas[task],
                "selected_primary_under_log_cap": primary_counts[task],
                "primary_deficit": primary_deficits[task],
                "required_reserve": reserve_quotas[task],
                "selected_reserve_under_log_cap": reserve_counts[task],
                "reserve_deficit": reserve_deficits[task],
            }
        )
    write_csv(
        args.output_dir / "m6_4_task_capacity.csv",
        capacity_rows,
        [
            "task",
            "eligible_before_log_cap",
            "required_primary_gross",
            "selected_primary_under_log_cap",
            "primary_deficit",
            "required_reserve",
            "selected_reserve_under_log_cap",
            "reserve_deficit",
        ],
    )

    candidate_pool_sha = sha256_file(args.output_dir / "m6_4_eligible_candidate_inventory.csv")
    planned_primary_total = sum(primary_quotas.values())
    planned_reserve_total = sum(reserve_quotas.values())
    readiness = {
        "milestone": "Stage 7 Milestone 6.4 locked collection preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ready_to_launch_locked_rollouts": ready,
        "analysis_role": "PRETREATMENT_METADATA_ONLY_NO_PLANNER_OUTCOMES",
        "inventory_csv": str(args.inventory_csv),
        "inventory_csv_sha256": sha256_file(args.inventory_csv),
        "development_metadata_sha256": sha256_file(args.development_metadata_csv),
        "m6_2_lock_spec_sha256": sha256_file(args.m6_2_lock_spec),
        "power_justification_sha256": sha256_file(args.power_justification_file),
        "selection_tool_sha256": sha256_file(Path(__file__)),
        "stage7c_tool_sha256": sha256_file(Path(__file__).with_name("stage7c1_run_nuplan_simulation.py")),
        "candidate_pool_sha256": candidate_pool_sha,
        "planner_parameter_fingerprints": planner_fp,
        "selection_salt": args.selection_salt,
        "max_scenarios_per_log_across_primary_and_reserve": args.max_per_log,
        "development_scenario_count": len(development_tokens),
        "development_log_count": len(development_logs),
        "required_complete_pairs_by_task": power["required_complete_pairs_by_task"],
        "required_overall_complete_pairs": power["required_complete_pairs_overall"],
        "planned_primary_gross_pairs_by_task": primary_quotas,
        "planned_reserve_pairs_by_task": reserve_quotas,
        "planned_primary_gross_pairs_total": planned_primary_total,
        "planned_reserve_pairs_total": planned_reserve_total,
        "minimum_distinct_logs_for_primary_under_cap": int(
            math.ceil(planned_primary_total / args.max_per_log)
        ),
        "minimum_distinct_logs_for_primary_plus_reserve_under_cap": int(
            math.ceil((planned_primary_total + planned_reserve_total) / args.max_per_log)
        ),
        "inventory_audit": inventory_audit,
        "primary_selected_by_task": {task: primary_counts[task] for task in PRETREATMENT_TASKS},
        "primary_deficits_by_task": primary_deficits,
        "reserve_selected_by_task": {task: reserve_counts[task] for task in PRETREATMENT_TASKS},
        "reserve_deficits_by_task": reserve_deficits,
        "freeze_rules": [
            "Only pre-treatment nuPlan scenario_type metadata may be used for selection.",
            "Development scenario tokens and every development log are excluded.",
            "Tokens with multiple frozen scenario types are excluded as ambiguous.",
            "Primary rows are all attempted in frozen order without inspecting effect sizes.",
            "Reserve rows are consumed in task-specific rank order only for documented technical or quality failures and only until the frozen complete-pair quota is reached.",
            "Planner treatment fingerprints must match the M6.2 lock.",
        ],
    }
    write_json(args.output_dir / "m6_4_inventory_readiness.json", readiness)

    if ready:
        primary_numbered = number_selected(primary, "primary_gross")
        reserve_numbered = number_selected(reserve, "technical_quality_reserve")
        write_csv(args.output_dir / "m6_4_locked_primary_collection.csv", primary_numbered, OUTPUT_FIELDS)
        write_csv(args.output_dir / "m6_4_locked_reserve_collection.csv", reserve_numbered, OUTPUT_FIELDS)
        context_dir = args.output_dir / "stage7c_primary_context"
        context_dir.mkdir()
        write_csv(context_dir / "merged_metadata.csv", primary_numbered, OUTPUT_FIELDS)
        manifest = {
            **readiness,
            "status": STATUS_READY,
            "primary_manifest_sha256": canonical_rows_hash(primary_numbered),
            "reserve_manifest_sha256": canonical_rows_hash(reserve_numbered),
            "primary_collection_csv_sha256": sha256_file(
                args.output_dir / "m6_4_locked_primary_collection.csv"
            ),
            "reserve_collection_csv_sha256": sha256_file(
                args.output_dir / "m6_4_locked_reserve_collection.csv"
            ),
            "planned_primary_scenarios": len(primary_numbered),
            "planned_primary_rollouts": len(primary_numbered) * 2,
            "maximum_reserve_scenarios": len(reserve_numbered),
            "planners": [args.planner_a, args.planner_b],
        }
        write_json(args.output_dir / "m6_4_locked_collection_manifest.json", manifest)

    report = [
        "# Stage 7 Milestone 6.4 Locked Collection Preflight",
        "",
        f"## Status: `{status}`",
        "",
        "This preflight used only pre-treatment scenario metadata. No new planner outcome or embedding was read.",
        f"Eligible log count: `{inventory_audit['eligible_distinct_logs']}`; minimum for the primary manifest under max_per_log={args.max_per_log}: `{math.ceil(planned_primary_total / args.max_per_log)}`.",
        "",
        "## Capacity",
        "",
        "| task | eligible | primary selected / required | reserve selected / required |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in capacity_rows:
        report.append(
            f"| {row['task']} | {row['eligible_before_log_cap']} | "
            f"{row['selected_primary_under_log_cap']} / {row['required_primary_gross']} | "
            f"{row['selected_reserve_under_log_cap']} / {row['required_reserve']} |"
        )
    report.extend(
        [
            "",
            "## Decision",
            "",
            (
                "The primary and reserve manifests are frozen. Locked rollouts may start."
                if ready
                else "The inventory cannot satisfy the frozen quotas. No locked collection manifest was emitted and rollouts must not start."
            ),
            "",
            "The task mapping, diversity cap, planner fingerprints, power justification, and selection salt must not be changed after planner outcomes are observed.",
        ]
    )
    (args.output_dir / "milestone6_4_collection_preflight_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return readiness, 0 if ready else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an outcome-blind Stage7 M6.4 locked collection manifest."
    )
    parser.add_argument("--inventory_csv", type=Path, required=True)
    parser.add_argument("--development_metadata_csv", type=Path, required=True)
    parser.add_argument("--m6_2_lock_spec", type=Path, required=True)
    parser.add_argument("--power_justification_file", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--planner_a", required=True)
    parser.add_argument("--planner_b", required=True)
    parser.add_argument("--max_per_log", type=int, default=2)
    parser.add_argument("--reserve_per_task", type=int, default=15)
    parser.add_argument("--selection_salt", default="stage7-m6.4-locked-v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_per_log < 1:
        parser.error("--max_per_log must be >= 1")
    if args.reserve_per_task < 0:
        parser.error("--reserve_per_task must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    readiness, exit_code = freeze_collection(args)
    print(
        f"M6.4 preflight: status={readiness['status']}, "
        f"ready={readiness['ready_to_launch_locked_rollouts']}"
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
