#!/usr/bin/env python3
"""Freeze a same-scenario pure-longitudinal PDM confirmation design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES  # noqa: E402


SCHEMA_VERSION = "stage6j_pure_longitudinal_confirmation_freeze_v1"
DESIGN_SCHEMA_VERSION = "stage6j_pure_longitudinal_confirmation_design_v1"
OUTPUT_FIELDS = [
    "collection_order",
    "source_global_scenario_index",
    "task",
    "source_task",
    "scenario_type",
    "log_name",
    "scenario_token",
    "scene_token",
    "db_file",
    "selection_role",
]
IGNORED_PARAMETER_KEYS = {"source", "checkpoint_required", "note"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a pure-longitudinal PDM confirmation without reading embeddings or BDD."
    )
    parser.add_argument("--design_json", type=Path, required=True)
    parser.add_argument("--confirmation_ledger_csv", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage 6J input does not exist: {path}")
    return path


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(require_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with require_file(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: str(value or "") for key, value in row.items()} for row in reader], list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def audit_planner_treatment(design: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    planners = [str(design.get("planner_a", "")), str(design.get("planner_b", ""))]
    if not all(planner in PLANNER_PROFILES for planner in planners):
        missing = [planner for planner in planners if planner not in PLANNER_PROFILES]
        raise ValueError(f"Stage 6J planners are not registered in Stage7C: {missing}")
    profiles = [PLANNER_PROFILES[planner] for planner in planners]
    for planner, profile in zip(planners, profiles):
        if profile.get("style_scope") != "pure_longitudinal_closed_loop_planner":
            raise ValueError(f"Planner {planner} is not marked pure_longitudinal_closed_loop_planner")

    params = [dict(profile["parameters"]) for profile in profiles]
    shared_lateral = dict(design.get("shared_lateral_parameters", {}))
    allowed_different = set(design.get("allowed_different_longitudinal_parameters", []))
    if not shared_lateral or not allowed_different:
        raise ValueError("Stage 6J requires non-empty shared lateral and allowed longitudinal parameter sets")

    keys = sorted((set(params[0]) | set(params[1])) - IGNORED_PARAMETER_KEYS)
    rows: List[Dict[str, Any]] = []
    unexpected_differences: List[str] = []
    for key in keys:
        value_a, value_b = params[0].get(key), params[1].get(key)
        same = value_a == value_b
        if key in shared_lateral:
            dimension = "lateral"
            expected = shared_lateral[key]
            if value_a != expected or value_b != expected:
                raise ValueError(
                    f"Shared lateral parameter {key} differs from frozen design: "
                    f"A={value_a}, B={value_b}, expected={expected}"
                )
        elif key in allowed_different:
            dimension = "longitudinal"
            if same:
                raise ValueError(f"Frozen longitudinal contrast parameter unexpectedly matches: {key}")
        else:
            dimension = "unclassified"
            if not same:
                unexpected_differences.append(key)
        rows.append(
            {
                "parameter": key,
                "dimension": dimension,
                "planner_a_value": json.dumps(value_a, ensure_ascii=False),
                "planner_b_value": json.dumps(value_b, ensure_ascii=False),
                "same_value": same,
                "difference_allowed": same or key in allowed_different,
            }
        )
    if unexpected_differences:
        raise ValueError(f"Non-longitudinal planner differences are not allowed: {unexpected_differences}")

    lateral_differences = [row["parameter"] for row in rows if row["dimension"] == "lateral" and not row["same_value"]]
    longitudinal_differences = [
        row["parameter"] for row in rows if row["dimension"] == "longitudinal" and not row["same_value"]
    ]
    summary = {
        "planner_a": planners[0],
        "planner_b": planners[1],
        "planner_a_parameter_sha256": canonical_hash(params[0]),
        "planner_b_parameter_sha256": canonical_hash(params[1]),
        "lateral_difference_count": len(lateral_differences),
        "lateral_differences": lateral_differences,
        "longitudinal_difference_count": len(longitudinal_differences),
        "longitudinal_differences": longitudinal_differences,
        "pure_longitudinal_treatment": not lateral_differences and bool(longitudinal_differences),
    }
    if not summary["pure_longitudinal_treatment"]:
        raise ValueError(f"Planner treatment is not pure longitudinal: {summary}")
    return rows, summary


def select_scenarios(
    design: Mapping[str, Any], ledger_rows: Sequence[Mapping[str, str]], db_root: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    required = {"global_scenario_index", "task", "scenario_type", "log_name", "scenario_token", "db_file"}
    if ledger_rows and not required.issubset(ledger_rows[0]):
        raise ValueError(f"Confirmation ledger is missing columns: {sorted(required - set(ledger_rows[0]))}")
    included = {str(task): set(map(str, types)) for task, types in design.get("included_scenario_types", {}).items()}
    type_to_task: Dict[str, str] = {}
    for task, scenario_types in included.items():
        for scenario_type in scenario_types:
            if scenario_type in type_to_task:
                raise ValueError(f"Scenario type is assigned to multiple Stage 6J tasks: {scenario_type}")
            type_to_task[scenario_type] = task
    excluded_types = set(map(str, design.get("excluded_scenario_types", [])))
    if excluded_types & set(type_to_task):
        raise ValueError("A Stage 6J scenario type is both included and excluded")

    selected: List[Dict[str, Any]] = []
    for row in ledger_rows:
        scenario_type = str(row["scenario_type"])
        if scenario_type not in type_to_task:
            continue
        selected.append(
            {
                "collection_order": 0,
                "source_global_scenario_index": int(row["global_scenario_index"]),
                "task": type_to_task[scenario_type],
                "source_task": row["task"],
                "scenario_type": scenario_type,
                "log_name": row["log_name"],
                "scenario_token": row["scenario_token"],
                "scene_token": row["scenario_token"],
                "db_file": row["db_file"],
                "selection_role": "FROZEN_PURE_LONGITUDINAL_PAIRED_PRIMARY",
            }
        )
    selected.sort(key=lambda row: (row["task"], row["source_global_scenario_index"], row["scenario_token"]))
    for index, row in enumerate(selected, start=1):
        row["collection_order"] = index

    tokens = [row["scenario_token"] for row in selected]
    if len(tokens) != len(set(tokens)):
        duplicates = sorted(token for token, count in Counter(tokens).items() if count > 1)
        raise ValueError(f"Stage 6J selected duplicate scenario tokens: {duplicates}")
    missing_db = sorted({row["db_file"] for row in selected if not (db_root / row["db_file"]).is_file()})
    if missing_db:
        raise FileNotFoundError(f"Stage 6J selected DB files are missing under {db_root}: {missing_db[:10]}")

    counts = Counter(row["task"] for row in selected)
    expected_counts = {str(key): int(value) for key, value in design.get("expected_selected_counts", {}).items()}
    if dict(sorted(counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(f"Stage 6J selected task counts changed: actual={dict(counts)}, expected={expected_counts}")
    expected_total = int(design.get("expected_selected_total", -1))
    if len(selected) != expected_total:
        raise ValueError(f"Stage 6J selected total changed: actual={len(selected)}, expected={expected_total}")

    summary = {
        "selected_scenario_count": len(selected),
        "selected_rollout_count": len(selected) * 2,
        "task_counts": dict(sorted(counts.items())),
        "distinct_log_count": len({row["log_name"] for row in selected}),
        "distinct_db_count": len({row["db_file"] for row in selected}),
        "duplicate_scenario_token_count": 0,
        "missing_db_count": 0,
        "excluded_tasks": list(design.get("excluded_tasks", [])),
        "excluded_scenario_types": list(design.get("excluded_scenario_types", [])),
    }
    return selected, summary


def build_report(
    design: Mapping[str, Any], treatment: Mapping[str, Any], selection: Mapping[str, Any]
) -> str:
    lines = [
        "# Stage 6J 纯纵向 PDM A/B 冻结报告",
        "",
        "## 结论",
        "",
        "冻结审计通过：两个 planner 使用完全相同的横向参数，仅允许六个纵向 IDM 参数不同。",
        "本步骤只冻结场景和处置，不读取 embedding/BDD，也不启动全量仿真。",
        "",
        "## Planner 处置",
        "",
        f"- Planner A：`{treatment['planner_a']}`",
        f"- Planner B：`{treatment['planner_b']}`",
        f"- 横向差异数量：{treatment['lateral_difference_count']}",
        f"- 纵向差异数量：{treatment['longitudinal_difference_count']}",
        f"- pure_longitudinal_treatment：{str(treatment['pure_longitudinal_treatment']).lower()}",
        "",
        "## 冻结场景",
        "",
        f"- 同场景配对数量：{selection['selected_scenario_count']}",
        f"- 计划 rollout 数量：{selection['selected_rollout_count']}",
        f"- 独立 log 数量：{selection['distinct_log_count']}",
        f"- task 数量：`{json.dumps(selection['task_counts'], ensure_ascii=False)}`",
        "- 排除 lane-change、dense/vulnerable 和 high_lateral_acceleration。",
        "",
        "## 后续门槛",
        "",
        "1. 先运行 1 场景 × 2 planners 的 official nuPlan smoke。",
        "2. smoke 必须满足双 planner 成功、same-log 与 strict-token alignment PASS。",
        "3. 全量运行后先检查 realized speed/accel/jerk/THW/gap，再计算 paired BDD。",
        "4. 只有 paired 纵向敏感性通过后，才进入异 log/异场景 release emulation。",
        "",
        f"设计 Issue：{design.get('issue', '')}",
        "",
    ]
    return "\n".join(lines)


def freeze(args: argparse.Namespace) -> Dict[str, Any]:
    design_path = args.design_json.resolve()
    ledger_path = args.confirmation_ledger_csv.resolve()
    db_root = args.nuplan_db_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    design = read_json(design_path)
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError(f"Unexpected Stage 6J design schema: {design.get('schema_version')!r}")
    ledger_rows, ledger_fields = read_csv(ledger_path)
    forbidden = {field for field in ledger_fields if "embedding" in field.lower() or "bdd" in field.lower()}
    if forbidden:
        raise ValueError(f"Stage 6J freeze ledger must not expose embedding/BDD columns: {sorted(forbidden)}")

    treatment_rows, treatment_summary = audit_planner_treatment(design)
    selected, selection_summary = select_scenarios(design, ledger_rows, db_root)
    smoke_rows = [next(row for row in selected if row["task"] == task) for task in design["included_scenario_types"]]

    selected_path = output_dir / "stage6j_locked_scenarios.csv"
    smoke_path = output_dir / "stage6j_smoke_scenarios.csv"
    treatment_path = output_dir / "stage6j_planner_parameter_audit.csv"
    context_path = output_dir / "stage7c_context" / "merged_metadata.csv"
    report_path = output_dir / "stage6j_freeze_report.md"
    write_csv(selected_path, selected, OUTPUT_FIELDS)
    write_csv(smoke_path, smoke_rows, OUTPUT_FIELDS)
    write_csv(treatment_path, treatment_rows, [
        "parameter", "dimension", "planner_a_value", "planner_b_value", "same_value", "difference_allowed"
    ])
    write_csv(context_path, selected, OUTPUT_FIELDS)
    report_path.write_text(build_report(design, treatment_summary, selection_summary), encoding="utf-8")

    tool_path = Path(__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_PURE_LONGITUDINAL_ROLLOUTS",
        "issue": design.get("issue"),
        "git_commit": git_commit(tool_path.parent.parent),
        "tool": str(tool_path),
        "tool_sha256": sha256_file(tool_path),
        "design_json": str(design_path),
        "design_sha256": sha256_file(design_path),
        "confirmation_ledger_csv": str(ledger_path),
        "confirmation_ledger_sha256": sha256_file(ledger_path),
        "nuplan_db_root": str(db_root),
        "treatment_audit": treatment_summary,
        "selection_audit": selection_summary,
        "new_treatment_outcome_blind_freeze": True,
        "source_ledger_conditioned_on_prior_technical_success": True,
        "embedding_or_bdd_read": False,
        "full_rollouts_launched": False,
        "outputs": {
            "locked_scenarios": {"path": selected_path.name, "sha256": sha256_file(selected_path)},
            "smoke_scenarios": {"path": smoke_path.name, "sha256": sha256_file(smoke_path)},
            "planner_parameter_audit": {"path": treatment_path.name, "sha256": sha256_file(treatment_path)},
            "stage7c_context": {"path": str(context_path.relative_to(output_dir)), "sha256": sha256_file(context_path)},
            "report": {"path": report_path.name, "sha256": sha256_file(report_path)},
        },
    }
    manifest_path = output_dir / "stage6j_freeze_manifest.json"
    write_json(manifest_path, result)
    return result


def main() -> None:
    result = freeze(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
