#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMA_VERSION = "stage7_m6_4c_locked_recovery_audit_v2"
READY_STATUS = "FROZEN_BEFORE_LOCKED_ROLLOUTS"
PRIMARY_FIELDS = [
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
STATUS_REQUIRED_FIELDS = {
    "collection_order",
    "task",
    "scenario_token",
    "status",
    "failure_category",
}
PRIMARY_AUDIT_FIELDS = [
    "collection_order",
    "task",
    "task_rank",
    "log_name",
    "scenario_token",
    "scenario_type",
    "db_file",
    "batch_status",
    "failure_category",
    "token_found",
    "scene_position",
    "scene_count",
    "official_scene_position_valid",
    "numeric_only_token",
    "hydra_requires_quoted_token",
    "technical_classification",
]
RESERVE_AUDIT_FIELDS = [
    "collection_order",
    "task",
    "task_rank",
    "log_name",
    "scenario_token",
    "scenario_type",
    "db_file",
    "token_found",
    "scene_position",
    "scene_count",
    "official_scene_position_valid",
    "numeric_only_token",
    "hydra_requires_quoted_token",
    "technical_classification",
]
RECOVERY_PLAN_FIELDS = [
    "plan_order",
    "task",
    "action",
    "source_role",
    "collection_order",
    "task_rank",
    "scenario_token",
    "log_name",
    "db_file",
    "rationale",
    "approval_status",
]
QUOTA_FIELDS = [
    "task",
    "required_complete_pairs",
    "current_succeeded_pairs",
    "initial_deficit",
    "quoted_primary_retries_proposed",
    "runnable_reserves_available",
    "runnable_reserves_proposed",
    "projected_complete_pairs",
    "remaining_deficit_after_frozen_recovery",
    "quota_status",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, required_fields: Sequence[str]) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(required_fields) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_rows_hash(rows: Sequence[Mapping[str, str]]) -> str:
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


def validate_locked_inputs(
    manifest_path: Path,
    primary_csv: Path,
    reserve_csv: Path,
    batch_status_csv: Path,
    batch_manifest_path: Path,
    stage7c_tool: Path,
    batch_tool: Path,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, str]],
    List[Dict[str, str]],
    List[Dict[str, str]],
    Dict[str, str],
]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != READY_STATUS:
        raise ValueError(f"locked manifest status is not {READY_STATUS}")
    if manifest.get("ready_to_launch_locked_rollouts") is not True:
        raise ValueError("locked manifest ready_to_launch_locked_rollouts is not true")

    primary = read_csv(primary_csv, PRIMARY_FIELDS)
    reserve = read_csv(reserve_csv, PRIMARY_FIELDS)
    statuses = read_csv(batch_status_csv, sorted(STATUS_REQUIRED_FIELDS))
    if len(primary) != int(manifest.get("planned_primary_scenarios", -1)):
        raise ValueError("primary row count differs from locked manifest")
    if len(reserve) != int(manifest.get("maximum_reserve_scenarios", -1)):
        raise ValueError("reserve row count differs from locked manifest")
    if len(statuses) != len(primary):
        raise ValueError("batch status row count differs from locked primary collection")

    expected_hashes = {
        primary_csv: manifest.get("primary_collection_csv_sha256"),
        reserve_csv: manifest.get("reserve_collection_csv_sha256"),
        stage7c_tool: manifest.get("stage7c_tool_sha256"),
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected={expected}, actual={actual}"
            )
    if canonical_rows_hash(primary) != manifest.get("primary_manifest_sha256"):
        raise ValueError("primary canonical manifest hash mismatch")
    if canonical_rows_hash(reserve) != manifest.get("reserve_manifest_sha256"):
        raise ValueError("reserve canonical manifest hash mismatch")

    batch_manifest = read_json(batch_manifest_path)
    expected_batch_tool_hash = batch_manifest.get("batch_tool_sha256")
    actual_batch_tool_hash = sha256_file(batch_tool)
    if actual_batch_tool_hash != expected_batch_tool_hash:
        raise ValueError(
            "M6.4B batch tool SHA-256 differs from the immutable batch manifest: "
            f"expected={expected_batch_tool_hash}, actual={actual_batch_tool_hash}"
        )

    primary_by_order = {int(row["collection_order"]): row for row in primary}
    seen_orders = set()
    for status in statuses:
        order = int(status["collection_order"])
        if order in seen_orders or order not in primary_by_order:
            raise ValueError(f"duplicate or unknown collection_order in batch status: {order}")
        seen_orders.add(order)
        locked = primary_by_order[order]
        if status["task"] != locked["task"] or status["scenario_token"] != locked["scenario_token"]:
            raise ValueError(f"batch status identity differs from locked primary row: order={order}")

    audit = {
        "locked_manifest_sha256": sha256_file(manifest_path),
        "primary_csv_sha256": sha256_file(primary_csv),
        "reserve_csv_sha256": sha256_file(reserve_csv),
        "batch_status_csv_sha256": sha256_file(batch_status_csv),
        "batch_manifest_sha256": sha256_file(batch_manifest_path),
        "stage7c_tool_sha256": sha256_file(stage7c_tool),
        "batch_tool_sha256": actual_batch_tool_hash,
    }
    return manifest, primary, reserve, statuses, audit


def inspect_token_scene_position(db_path: Path, scenario_token: str) -> Dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(f"locked DB file is missing: {db_path}")
    try:
        token_bytes = bytes.fromhex(scenario_token)
    except ValueError as exc:
        raise ValueError(f"scenario_token is not hexadecimal: {scenario_token}") from exc
    if len(token_bytes) != 8:
        raise ValueError(f"scenario_token must decode to 8 bytes: {scenario_token}")

    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        token_row = connection.execute(
            "SELECT scene_token FROM lidar_pc WHERE token = ?", (token_bytes,)
        ).fetchone()
        scenes = connection.execute(
            "SELECT token FROM scene ORDER BY name ASC"
        ).fetchall()
    finally:
        connection.close()

    scene_count = len(scenes)
    scene_position = 0
    if token_row is not None:
        scene_token = token_row[0]
        scene_position = next(
            (index for index, item in enumerate(scenes, start=1) if item[0] == scene_token),
            0,
        )
    valid = bool(scene_position >= 3 and scene_position < scene_count - 1)
    return {
        "token_found": token_row is not None,
        "scene_position": scene_position,
        "scene_count": scene_count,
        "official_scene_position_valid": valid,
        "numeric_only_token": scenario_token.isdigit(),
        "hydra_requires_quoted_token": hydra_requires_quoted_token(scenario_token),
    }


def hydra_requires_quoted_token(scenario_token: str) -> bool:
    """Return whether an unquoted token is parsed by Hydra/OmegaConf as numeric."""
    return bool(
        re.fullmatch(
            r"[+-]?[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][+-]?[0-9_]+)?",
            scenario_token,
        )
    )


def classify_primary(status: Mapping[str, str], technical: Mapping[str, Any]) -> str:
    if not technical["token_found"]:
        return "TOKEN_NOT_FOUND"
    if not technical["official_scene_position_valid"]:
        return "INVALID_SCENE_POSITION"
    if status["status"] == "SUCCEEDED":
        return "RUNNABLE_SUCCEEDED"
    if technical["hydra_requires_quoted_token"]:
        return "RETRY_WITH_QUOTED_HYDRA_TOKEN"
    return "RUNNABLE_FAILED_REVIEW_REQUIRED"


def audit_rows(
    primary: Sequence[Dict[str, str]],
    reserve: Sequence[Dict[str, str]],
    statuses: Sequence[Dict[str, str]],
    db_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    status_by_order = {int(row["collection_order"]): row for row in statuses}
    primary_audit: List[Dict[str, Any]] = []
    reserve_audit: List[Dict[str, Any]] = []
    for row in primary:
        status = status_by_order[int(row["collection_order"])]
        technical = inspect_token_scene_position(db_root / row["db_file"], row["scenario_token"])
        primary_audit.append(
            {
                **row,
                "batch_status": status["status"],
                "failure_category": status["failure_category"],
                **technical,
                "technical_classification": classify_primary(status, technical),
            }
        )
    for row in reserve:
        technical = inspect_token_scene_position(db_root / row["db_file"], row["scenario_token"])
        if not technical["token_found"]:
            classification = "TOKEN_NOT_FOUND"
        elif not technical["official_scene_position_valid"]:
            classification = "INVALID_SCENE_POSITION"
        else:
            classification = "RUNNABLE_RESERVE"
        reserve_audit.append(
            {**row, **technical, "technical_classification": classification}
        )
    return primary_audit, reserve_audit


def build_recovery_plan(
    manifest: Mapping[str, Any],
    primary_audit: Sequence[Mapping[str, Any]],
    reserve_audit: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    required = {
        str(task): int(value)
        for task, value in manifest.get("required_complete_pairs_by_task", {}).items()
    }
    if not required:
        raise ValueError("locked manifest has no required_complete_pairs_by_task")

    successes = Counter(
        str(row["task"])
        for row in primary_audit
        if row["technical_classification"] == "RUNNABLE_SUCCEEDED"
    )
    retries: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    reserves: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in primary_audit:
        if row["technical_classification"] == "RETRY_WITH_QUOTED_HYDRA_TOKEN":
            retries[str(row["task"])].append(row)
    for row in reserve_audit:
        if row["technical_classification"] == "RUNNABLE_RESERVE":
            reserves[str(row["task"])].append(row)
    for rows in retries.values():
        rows.sort(key=lambda row: int(row["collection_order"]))
    for rows in reserves.values():
        rows.sort(key=lambda row: int(row["task_rank"]))

    plan: List[Dict[str, Any]] = []
    quota_rows: List[Dict[str, Any]] = []
    for task in required:
        target = required[task]
        current = int(successes.get(task, 0))
        initial_deficit = max(0, target - current)
        selected_retries = retries.get(task, [])[:initial_deficit]
        after_retries = current + len(selected_retries)
        reserve_need = max(0, target - after_retries)
        selected_reserves = reserves.get(task, [])[:reserve_need]
        projected = after_retries + len(selected_reserves)
        remaining = max(0, target - projected)

        for row in selected_retries:
            plan.append(
                {
                    "task": task,
                    "action": "RETRY_PRIMARY_QUOTED_TOKEN",
                    "source_role": "primary_gross",
                    "collection_order": row["collection_order"],
                    "task_rank": row["task_rank"],
                    "scenario_token": row["scenario_token"],
                    "log_name": row["log_name"],
                    "db_file": row["db_file"],
                    "rationale": "valid official scene position; Hydra must receive a quoted string token",
                    "approval_status": "PROPOSED_NOT_EXECUTED",
                }
            )
        for row in selected_reserves:
            plan.append(
                {
                    "task": task,
                    "action": "RUN_FROZEN_RESERVE",
                    "source_role": "technical_quality_reserve",
                    "collection_order": row["collection_order"],
                    "task_rank": row["task_rank"],
                    "scenario_token": row["scenario_token"],
                    "log_name": row["log_name"],
                    "db_file": row["db_file"],
                    "rationale": "task remains below its frozen complete-pair quota and reserve is technically runnable",
                    "approval_status": "PROPOSED_NOT_EXECUTED",
                }
            )
        quota_rows.append(
            {
                "task": task,
                "required_complete_pairs": target,
                "current_succeeded_pairs": current,
                "initial_deficit": initial_deficit,
                "quoted_primary_retries_proposed": len(selected_retries),
                "runnable_reserves_available": len(reserves.get(task, [])),
                "runnable_reserves_proposed": len(selected_reserves),
                "projected_complete_pairs": projected,
                "remaining_deficit_after_frozen_recovery": remaining,
                "quota_status": "PROJECTED_COMPLETE" if remaining == 0 else "SUPPLEMENTAL_PROTOCOL_REQUIRED",
            }
        )
    for index, row in enumerate(plan, start=1):
        row["plan_order"] = index
    return plan, quota_rows


def render_report(
    primary_audit: Sequence[Mapping[str, Any]],
    reserve_audit: Sequence[Mapping[str, Any]],
    quota_rows: Sequence[Mapping[str, Any]],
    recovery_plan: Sequence[Mapping[str, Any]],
) -> str:
    primary_counts = Counter(str(row["technical_classification"]) for row in primary_audit)
    reserve_counts = Counter(str(row["technical_classification"]) for row in reserve_audit)
    lines = [
        "# Stage 7 M6.4C Locked Recovery Technical Audit",
        "",
        "## Scope",
        "",
        "This audit reads only frozen collection metadata, SQLite structure, and M6.4B technical status. ",
        "It does not read embedding, BDD, effect size, trajectory metrics, or planner outcome metrics.",
        "",
        "## Primary technical classifications",
        "",
    ]
    for key in sorted(primary_counts):
        lines.append(f"- {key}: `{primary_counts[key]}`")
    lines += ["", "## Reserve technical classifications", ""]
    for key in sorted(reserve_counts):
        lines.append(f"- {key}: `{reserve_counts[key]}`")
    lines += [
        "",
        "## Frozen quota recovery projection",
        "",
        "| task | required | current | quoted retries | runnable reserves proposed | projected | remaining | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in quota_rows:
        lines.append(
            f"| {row['task']} | {row['required_complete_pairs']} | {row['current_succeeded_pairs']} | "
            f"{row['quoted_primary_retries_proposed']} | {row['runnable_reserves_proposed']} | "
            f"{row['projected_complete_pairs']} | {row['remaining_deficit_after_frozen_recovery']} | "
            f"{row['quota_status']} |"
        )
    lines += [
        "",
        f"Recovery actions proposed: `{len(recovery_plan)}`.",
        "All actions remain `PROPOSED_NOT_EXECUTED`.",
        "No sample outside the frozen primary/reserve collections is selected by this tool.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(
            f"output_dir already exists: {args.output_dir}. Use a new immutable audit directory."
        )
    manifest, primary, reserve, statuses, input_audit = validate_locked_inputs(
        args.locked_manifest,
        args.primary_csv,
        args.reserve_csv,
        args.batch_status_csv,
        args.batch_manifest,
        args.stage7c_tool,
        args.batch_tool,
    )
    primary_audit, reserve_audit = audit_rows(
        primary, reserve, statuses, args.nuplan_db_root
    )
    recovery_plan, quota_rows = build_recovery_plan(
        manifest, primary_audit, reserve_audit
    )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "primary_technical_audit.csv", primary_audit, PRIMARY_AUDIT_FIELDS)
    write_csv(args.output_dir / "reserve_technical_audit.csv", reserve_audit, RESERVE_AUDIT_FIELDS)
    write_csv(args.output_dir / "recovery_plan.csv", recovery_plan, RECOVERY_PLAN_FIELDS)
    write_csv(args.output_dir / "quota_recovery_projection.csv", quota_rows, QUOTA_FIELDS)
    report = render_report(primary_audit, reserve_audit, quota_rows, recovery_plan)
    (args.output_dir / "m6_4c_recovery_audit_report.md").write_text(
        report, encoding="utf-8"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "analysis_role": "PRETREATMENT_AND_TECHNICAL_STATUS_ONLY_OUTCOME_BLIND",
        "execution_enabled": False,
        "input_audit": input_audit,
        "primary_classification_counts": dict(
            sorted(Counter(row["technical_classification"] for row in primary_audit).items())
        ),
        "reserve_classification_counts": dict(
            sorted(Counter(row["technical_classification"] for row in reserve_audit).items())
        ),
        "quota_projection": quota_rows,
        "recovery_action_count": len(recovery_plan),
        "forbidden_inputs_read": {
            "embedding": False,
            "bdd": False,
            "effect_size": False,
            "trajectory_metrics": False,
            "planner_outcome_metrics": False,
        },
    }
    (args.output_dir / "m6_4c_recovery_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "M6.4C audit: "
        f"primary={summary['primary_classification_counts']} "
        f"reserve={summary['reserve_classification_counts']} "
        f"actions={len(recovery_plan)} output={args.output_dir}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-blind technical runnability audit for Stage 7 M6.4C locked recovery."
    )
    parser.add_argument("--locked_manifest", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--reserve_csv", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument(
        "--stage7c_tool", type=Path, default=Path("tools/stage7c1_run_nuplan_simulation.py")
    )
    parser.add_argument(
        "--batch_tool", type=Path, default=Path("tools/stage7_m6_4b_run_locked_rollouts.py")
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
