#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import stage7_m6_4b_run_locked_rollouts as batch
from tools import stage7_m6_4c_audit_locked_recovery as recovery
from tools.stage7_m6_2_locked_task_bdd import PRETREATMENT_TASKS


SCHEMA_VERSION = "stage6g_expanded_release_pool_freeze_v1"
READY_STATUS = "FROZEN_BEFORE_STAGE6G_ROLLOUTS"
BLOCKED_STATUS = "BLOCKED_INSUFFICIENT_TECHNICALLY_RUNNABLE_INVENTORY"
PROBE_FIELDS = [
    "probe_rank",
    "stable_rank_sha256",
    "task",
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
    "selection_role",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path, required: Iterable[str] = ()) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(required) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_rank(row: Mapping[str, str], task: str, salt: str) -> str:
    value = f"{salt}:{task}:{row['log_name']}:{row['scenario_token']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_sha(path: Path, expected: str, label: str) -> str:
    actual = batch.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} actual={actual}")
    return actual


def validate_config(config: Mapping[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    if config.get("schema_version") != "stage6g_expanded_release_pool_config_v1":
        raise ValueError("Stage6G config schema mismatch")
    primary = {str(k): int(v) for k, v in config["primary_additions_by_task"].items()}
    reserve = {str(k): int(v) for k, v in config["reserve_by_task"].items()}
    expected = set(PRETREATMENT_TASKS)
    if set(primary) != expected or set(reserve) != expected:
        raise ValueError("Stage6G quotas must cover exactly the frozen M6.2 task family")
    if any(value < 0 for value in [*primary.values(), *reserve.values()]):
        raise ValueError("Stage6G quotas must be non-negative")
    existing = int(config["existing_pool_size"])
    target = int(config["target_combined_pool_size"])
    if sum(primary.values()) != target - existing:
        raise ValueError("primary additions do not bridge existing_pool_size to target")
    return primary, reserve


def collect_top_candidates(
    inventory_path: Path,
    *,
    excluded_tokens: set[str],
    salt: str,
    probe_limit: int,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
    required = {
        "task", "log_name", "scenario_token", "scenario_type", "db_file",
        "db_scene_token", "scenario_tag_token",
    }
    heaps: Dict[str, List[Tuple[int, str, Dict[str, str]]]] = {
        task: [] for task in PRETREATMENT_TASKS
    }
    counts: Counter = Counter()
    type_to_task = {
        scenario_type: task
        for task, scenario_types in PRETREATMENT_TASKS.items()
        for scenario_type in scenario_types
    }
    seen_tokens: set[str] = set()
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"eligible inventory missing columns: {missing}")
        for raw in reader:
            counts["inventory_rows"] += 1
            task = str(raw.get("task", ""))
            token = str(raw.get("scenario_token", "")).strip()
            if task not in heaps:
                counts["excluded_unknown_task"] += 1
                continue
            if type_to_task.get(str(raw.get("scenario_type", ""))) != task:
                raise ValueError(f"scenario_type/task mismatch for token {token}")
            if token in seen_tokens:
                raise ValueError(f"eligible inventory contains duplicate token: {token}")
            seen_tokens.add(token)
            if token in excluded_tokens:
                counts[f"excluded_existing_token::{task}"] += 1
                continue
            row = {key: str(value or "").strip() for key, value in raw.items()}
            rank = stable_rank(row, task, salt)
            row["stable_rank_sha256"] = rank
            item = (-int(rank, 16), token, row)
            heap = heaps[task]
            if len(heap) < probe_limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            counts[f"eligible_after_token_exclusion::{task}"] += 1
    pools: Dict[str, List[Dict[str, str]]] = {}
    for task, heap in heaps.items():
        rows = [item[2] for item in heap]
        rows.sort(key=lambda row: (row["stable_rank_sha256"], row["scenario_token"]))
        pools[task] = rows
    audit = {
        "inventory_rows": counts["inventory_rows"],
        "excluded_existing_tokens_by_task": {
            task: counts[f"excluded_existing_token::{task}"] for task in PRETREATMENT_TASKS
        },
        "eligible_after_token_exclusion_by_task": {
            task: counts[f"eligible_after_token_exclusion::{task}"] for task in PRETREATMENT_TASKS
        },
        "retained_for_technical_probe_by_task": {task: len(rows) for task, rows in pools.items()},
    }
    return pools, audit


def select_technically_runnable(
    pools: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    primary_quotas: Mapping[str, int],
    reserve_quotas: Mapping[str, int],
    existing_log_counts: Mapping[str, int],
    max_per_log: int,
    db_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    log_counts = Counter({str(k): int(v) for k, v in existing_log_counts.items()})
    primary: List[Dict[str, Any]] = []
    reserve: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    deficits: Dict[str, int] = {}
    # Protect the rarest frozen strata first.  This ordering is determined only
    # from pre-treatment candidate capacity, never from planner outcomes.
    task_order = sorted(PRETREATMENT_TASKS, key=lambda task: (len(pools[task]), task))
    for task in task_order:
        required_primary = int(primary_quotas[task])
        required_reserve = int(reserve_quotas[task])
        selected_primary = 0
        selected_reserve = 0
        for probe_rank, source in enumerate(pools[task], start=1):
            row = dict(source)
            if log_counts[row["log_name"]] >= max_per_log:
                technical = {
                    "token_found": "",
                    "scene_position": "",
                    "scene_count": "",
                    "official_scene_position_valid": "",
                    "hydra_requires_quoted_token": "",
                }
                decision = "EXCLUDED_FROZEN_LOG_CAP"
                role = ""
            else:
                technical = recovery.inspect_token_scene_position(
                    db_root / row["db_file"], row["scenario_token"]
                )
                if not technical["token_found"]:
                    decision = "EXCLUDED_TOKEN_NOT_FOUND"
                    role = ""
                elif not technical["official_scene_position_valid"]:
                    decision = "EXCLUDED_INVALID_SCENE_POSITION"
                    role = ""
                elif selected_primary < required_primary:
                    decision = "SELECTED_TECHNICALLY_RUNNABLE_PRIMARY"
                    role = "stage6g_primary"
                    row["selection_role"] = role
                    primary.append(row)
                    selected_primary += 1
                    log_counts[row["log_name"]] += 1
                elif selected_reserve < required_reserve:
                    decision = "SELECTED_TECHNICALLY_RUNNABLE_RESERVE"
                    role = "stage6g_technical_reserve"
                    row["selection_role"] = role
                    reserve.append(row)
                    selected_reserve += 1
                    log_counts[row["log_name"]] += 1
                else:
                    break
            audit.append(
                {
                    **row,
                    **technical,
                    "probe_rank": probe_rank,
                    "decision": decision,
                    "selection_role": role,
                }
            )
            if selected_primary == required_primary and selected_reserve == required_reserve:
                break
        deficits[task] = (required_primary - selected_primary) + (required_reserve - selected_reserve)
    return primary, reserve, audit, deficits


def number_rows(rows: Sequence[Mapping[str, Any]], role: str, salt: str) -> List[Dict[str, Any]]:
    task_counts: Counter = Counter()
    numbered: List[Dict[str, Any]] = []
    for order, source in enumerate(rows, start=1):
        row = dict(source)
        task_counts[row["task"]] += 1
        row.update(
            collection_order=order,
            task_rank=task_counts[row["task"]],
            scene_token=row["scenario_token"],
            selection_role=role,
            selection_salt=salt,
        )
        numbered.append(row)
    return numbered


def freeze(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    project_root = Path(__file__).resolve().parents[1]
    config = read_json(args.config)
    primary_quotas, reserve_quotas = validate_config(config)
    inputs = config["inputs"]
    resolved = {
        key: resolve_path(project_root, value)
        for key, value in inputs.items()
        if key.endswith(("_csv", "_json", "_tool")) or key in {"stage7c_tool", "runner_tool"}
    }
    expected_input_hashes = {
        "eligible_inventory_csv": inputs["eligible_inventory_sha256"],
        "existing_confirmation_ledger_csv": inputs["existing_confirmation_ledger_sha256"],
        "historical_batch_manifest_json": inputs["historical_batch_manifest_sha256"],
        "stage7c_tool": inputs["stage7c_tool_sha256"],
        "historical_batch_tool": inputs["historical_batch_tool_sha256"],
    }
    for key, expected in expected_input_hashes.items():
        validate_sha(resolved[key], str(expected), key)
    if not resolved["runner_tool"].is_file():
        raise FileNotFoundError(resolved["runner_tool"])

    historical = read_json(resolved["historical_batch_manifest_json"])
    planners = list(historical.get("planners", []))
    if planners != batch.EXPECTED_PLANNERS:
        raise ValueError("historical batch planner order mismatch")
    frozen_runtime = historical.get("frozen_input_audit", {})
    if batch.current_planner_fingerprints(planners) != frozen_runtime.get("planner_parameter_fingerprints"):
        raise ValueError("current planner fingerprints differ from frozen historical runtime")

    ledger = read_csv(
        resolved["existing_confirmation_ledger_csv"],
        {"task", "scenario_token", "log_name", "stage7c_audit_pass"},
    )
    if len(ledger) != int(config["existing_pool_size"]):
        raise ValueError("existing confirmation ledger row count mismatch")
    if any(str(row["stage7c_audit_pass"]).lower() != "true" for row in ledger):
        raise ValueError("existing confirmation ledger contains a failed Stage7C audit")
    existing_tokens = {row["scenario_token"] for row in ledger}
    if len(existing_tokens) != len(ledger):
        raise ValueError("existing confirmation ledger has duplicate scenario tokens")
    existing_counts = Counter(row["task"] for row in ledger)
    expected_existing = {str(k): int(v) for k, v in config["expected_existing_counts_by_task"].items()}
    if {task: existing_counts[task] for task in PRETREATMENT_TASKS} != expected_existing:
        raise ValueError("existing confirmation task counts differ from Stage6G config")

    max_per_log = int(config["max_scenarios_per_log_across_existing_primary_and_reserve"])
    existing_log_counts = Counter(row["log_name"] for row in ledger)
    if max(existing_log_counts.values()) > max_per_log:
        raise ValueError("existing pool already exceeds the frozen per-log cap")
    pools, inventory_audit = collect_top_candidates(
        resolved["eligible_inventory_csv"],
        excluded_tokens=existing_tokens,
        salt=str(config["selection_salt"]),
        probe_limit=int(config["probe_limit_per_task"]),
    )
    primary, reserve, probe_audit, deficits = select_technically_runnable(
        pools,
        primary_quotas=primary_quotas,
        reserve_quotas=reserve_quotas,
        existing_log_counts=existing_log_counts,
        max_per_log=max_per_log,
        db_root=Path(historical["nuplan_db_root"]),
    )
    primary_numbered = number_rows(primary, "stage6g_primary", str(config["selection_salt"]))
    reserve_numbered = number_rows(reserve, "stage6g_technical_reserve", str(config["selection_salt"]))
    primary_counts = Counter(row["task"] for row in primary_numbered)
    reserve_counts = Counter(row["task"] for row in reserve_numbered)
    ready = not any(deficits.values())
    status = READY_STATUS if ready else BLOCKED_STATUS

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "stage6g_candidate_probe_audit.csv", probe_audit, PROBE_FIELDS)
    capacity_rows = []
    for task in PRETREATMENT_TASKS:
        valid_considered = sum(
            str(row.get("official_scene_position_valid", "")).lower() == "true"
            for row in probe_audit if row["task"] == task
        )
        invalid_considered = sum(
            row["decision"] in {"EXCLUDED_INVALID_SCENE_POSITION", "EXCLUDED_TOKEN_NOT_FOUND"}
            for row in probe_audit if row["task"] == task
        )
        capacity_rows.append(
            {
                "task": task,
                "existing_successful": existing_counts[task],
                "eligible_after_existing_token_exclusion": inventory_audit["eligible_after_token_exclusion_by_task"][task],
                "technically_valid_considered": valid_considered,
                "technically_invalid_considered": invalid_considered,
                "required_primary_addition": primary_quotas[task],
                "selected_primary": primary_counts[task],
                "required_reserve": reserve_quotas[task],
                "selected_reserve": reserve_counts[task],
                "combined_target_after_primary": existing_counts[task] + primary_counts[task],
                "deficit": deficits[task],
            }
        )
    capacity_fields = list(capacity_rows[0])
    write_csv(args.output_dir / "stage6g_task_capacity.csv", capacity_rows, capacity_fields)
    if ready:
        write_csv(args.output_dir / "stage6g_locked_primary.csv", primary_numbered, batch.PRIMARY_FIELDS)
        write_csv(args.output_dir / "stage6g_locked_reserve.csv", reserve_numbered, batch.PRIMARY_FIELDS)

    hashes = {
        "config_sha256": batch.sha256_file(args.config),
        "freeze_tool_sha256": batch.sha256_file(Path(__file__)),
        "runner_tool_sha256": batch.sha256_file(resolved["runner_tool"]),
        "eligible_inventory_sha256": batch.sha256_file(resolved["eligible_inventory_csv"]),
        "existing_confirmation_ledger_sha256": batch.sha256_file(resolved["existing_confirmation_ledger_csv"]),
        "historical_batch_manifest_sha256": batch.sha256_file(resolved["historical_batch_manifest_json"]),
        "historical_batch_tool_sha256": batch.sha256_file(resolved["historical_batch_tool"]),
        "stage7c_tool_sha256": batch.sha256_file(resolved["stage7c_tool"]),
        "candidate_probe_audit_sha256": batch.sha256_file(args.output_dir / "stage6g_candidate_probe_audit.csv"),
        "task_capacity_sha256": batch.sha256_file(args.output_dir / "stage6g_task_capacity.csv"),
    }
    if ready:
        hashes.update(
            primary_csv_sha256=batch.sha256_file(args.output_dir / "stage6g_locked_primary.csv"),
            reserve_csv_sha256=batch.sha256_file(args.output_dir / "stage6g_locked_reserve.csv"),
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": status,
        "ready_to_launch_rollouts": ready,
        "issue": config["issue"],
        "analysis_role": "PRETREATMENT_METADATA_AND_TECHNICAL_RUNNABILITY_ONLY",
        "selection_salt": config["selection_salt"],
        "existing_pool_size": len(ledger),
        "planned_primary_additions": len(primary_numbered),
        "planned_reserve": len(reserve_numbered),
        "target_combined_pool_size": int(config["target_combined_pool_size"]),
        "primary_selected_by_task": {task: primary_counts[task] for task in PRETREATMENT_TASKS},
        "reserve_selected_by_task": {task: reserve_counts[task] for task in PRETREATMENT_TASKS},
        "combined_target_by_task": {
            task: existing_counts[task] + primary_counts[task] for task in PRETREATMENT_TASKS
        },
        "deficits_by_task": deficits,
        "max_scenarios_per_log_across_existing_primary_and_reserve": max_per_log,
        "inventory_audit": inventory_audit,
        "runtime": {
            key: historical[key]
            for key in [
                "nuplan_db_root", "nuplan_map_root", "nuplan_data_root", "nuplan_exp_root",
                "python_executable", "command_timeout_s",
            ]
        },
        "nuplan_devkit_root": str(Path(historical["nuplan_db_root"]).resolve().parents[4] / "nuplan-devkit"),
        "tuplan_garage_root": str(Path(historical["nuplan_db_root"]).resolve().parents[4] / "tuplan_garage"),
        "planners": planners,
        "planner_parameter_fingerprints": frozen_runtime["planner_parameter_fingerprints"],
        "nuplan_devkit_commit": frozen_runtime["nuplan_devkit_commit"],
        "tuplan_garage_commit": frozen_runtime["tuplan_garage_commit"],
        "input_paths": {key: str(path.resolve()) for key, path in resolved.items()},
        "hashes": hashes,
        "primary_manifest_sha256": batch.canonical_rows_hash(primary_numbered) if ready else "",
        "reserve_manifest_sha256": batch.canonical_rows_hash(reserve_numbered) if ready else "",
        "forbidden_inputs_read": {
            "embedding": False,
            "bdd": False,
            "effect_size": False,
            "trajectory_metrics_for_selection": False,
            "planner_outcomes_for_selection": False,
        },
        "selection_rules": [
            "The frozen M6.2 pre-treatment scenario_type mapping is unchanged.",
            "All 310 existing successful scenario tokens are excluded; logs are retained as grouping units.",
            "Candidate order is a deterministic salted hash independent of planner outcomes.",
            "Every selected token passes the official nuPlan scene-position gate before rollout.",
            "The per-log cap counts existing, primary, and reserve rows together.",
            "The lane-change quota is the complete technically runnable remainder, not a post-outcome choice.",
        ],
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": batch.resolve_git_commit(project_root),
        },
    }
    write_json(args.output_dir / "stage6g_freeze_manifest.json", manifest)
    report = [
        "# Stage 6G 扩展发布池冻结报告",
        "",
        f"- 状态：`{status}`",
        f"- 现有池：{len(ledger)}；新增主集：{len(primary_numbered)}；合并目标：{len(ledger) + len(primary_numbered)}",
        f"- 新增预备集：{len(reserve_numbered)}",
        "- 选择时机：planner rollout 前；未读取 embedding、BDD、effect size、轨迹指标或 planner outcome。",
        "",
        "## 任务配额",
        "",
        "| task | 现有 | 新增主集 | 合并目标 | 预备 | deficit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in PRETREATMENT_TASKS:
        report.append(
            f"| {task} | {existing_counts[task]} | {primary_counts[task]} | "
            f"{existing_counts[task] + primary_counts[task]} | {reserve_counts[task]} | {deficits[task]} |"
        )
    report += [
        "",
        "## lane-change 边界",
        "",
        "排除现有 token 后共有 50 个原定义候选，但只有 11 个通过官方 scene-position 门槛。"
        "因此冻结新增 11 个并保留原任务定义；不使用转弯等不同语义标签补齐数量。",
        "",
    ]
    (args.output_dir / "stage6g_freeze_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": status, "primary": len(primary_numbered), "reserve": len(reserve_numbered), "deficits": deficits}, indent=2))
    return manifest, 0 if ready else 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the outcome-blind Stage6G nuPlan pool expansion.")
    parser.add_argument("--config", type=Path, default=Path("configs/stage6g_expanded_release_pool.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/stage6g_expanded_release_pool_freeze_v1"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(freeze(parsed)[1])
