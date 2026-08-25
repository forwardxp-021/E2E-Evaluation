#!/usr/bin/env python3
"""Freeze the Stage 6K graded pure-longitudinal treatment before simulation."""

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
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES  # noqa: E402


SCHEMA_VERSION = "stage6k_longitudinal_dose_response_freeze_v1"
DESIGN_SCHEMA_VERSION = "stage6k_longitudinal_dose_response_design_v1"
FREEZE_STATUS = "FROZEN_BEFORE_LONGITUDINAL_DOSE_ROLLOUTS"
IGNORED_PARAMETER_KEYS = {"source", "checkpoint_required", "note"}
SOURCE_FIELDS = [
    "collection_order", "source_global_scenario_index", "task", "source_task",
    "scenario_type", "log_name", "scenario_token", "scene_token", "db_file", "selection_role",
]
JOB_FIELDS = [
    "collection_order", "source_collection_order", "dose", "dose_label", "planner_a", "planner_b",
] + [field for field in SOURCE_FIELDS if field != "collection_order"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Stage 6K without reading embeddings or BDD.")
    parser.add_argument("--design_json", type=Path, required=True)
    parser.add_argument("--stage6j_locked_scenarios_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: str(value or "") for key, value in row.items()} for row in reader], list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable"


def clean_parameters(planner: str) -> Dict[str, Any]:
    if planner not in PLANNER_PROFILES:
        raise ValueError(f"Stage 6K planner is not registered: {planner}")
    profile = PLANNER_PROFILES[planner]
    if profile.get("style_scope") != "pure_longitudinal_closed_loop_planner":
        raise ValueError(f"Stage 6K planner is not pure longitudinal: {planner}")
    return {key: value for key, value in profile["parameters"].items() if key not in IGNORED_PARAMETER_KEYS}


def interpolate(baseline: Any, anchor: Any, dose: float) -> Any:
    if isinstance(baseline, list) and isinstance(anchor, list):
        if len(baseline) != len(anchor):
            raise ValueError("Cannot interpolate parameter lists with different lengths")
        return [interpolate(left, right, dose) for left, right in zip(baseline, anchor)]
    if isinstance(baseline, (int, float)) and isinstance(anchor, (int, float)):
        return float(baseline) + dose * (float(anchor) - float(baseline))
    if baseline != anchor:
        raise ValueError(f"Non-numeric parameter differs across endpoints: {baseline!r} vs {anchor!r}")
    return baseline


def values_close(actual: Any, expected: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(values_close(a, e, tolerance) for a, e in zip(actual, expected))
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def audit_profiles(design: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    baseline_name = str(design["baseline_planner"])
    anchor_name = str(design["assertive_anchor_planner"])
    baseline = clean_parameters(baseline_name)
    anchor = clean_parameters(anchor_name)
    longitudinal = set(map(str, design["interpolated_longitudinal_parameters"]))
    shared = dict(design["shared_lateral_parameters"])
    all_keys = (set(baseline) | set(anchor))
    if all_keys != longitudinal | set(shared):
        raise ValueError(f"Stage 6K parameters are not fully classified: {sorted(all_keys - longitudinal - set(shared))}")
    rows: List[Dict[str, Any]] = []
    fingerprints: Dict[str, str] = {
        baseline_name: canonical_hash(PLANNER_PROFILES[baseline_name]["parameters"]),
        anchor_name: canonical_hash(PLANNER_PROFILES[anchor_name]["parameters"]),
    }
    for item in design["dose_profiles"]:
        dose = float(item["dose"])
        planner = str(item["planner"])
        actual = clean_parameters(planner)
        if set(actual) != all_keys:
            raise ValueError(f"Stage 6K profile parameter keys changed for {planner}")
        fingerprints[planner] = canonical_hash(PLANNER_PROFILES[planner]["parameters"])
        for key in sorted(all_keys):
            expected = shared[key] if key in shared else interpolate(baseline[key], anchor[key], dose)
            passed = values_close(actual[key], expected)
            rows.append({
                "dose": dose, "dose_label": item["label"], "planner": planner, "parameter": key,
                "dimension": "lateral" if key in shared else "longitudinal",
                "baseline_value": json.dumps(baseline[key], ensure_ascii=False),
                "anchor_value": json.dumps(anchor[key], ensure_ascii=False),
                "expected_value": json.dumps(expected, ensure_ascii=False),
                "actual_value": json.dumps(actual[key], ensure_ascii=False), "interpolation_passed": passed,
            })
            if not passed:
                raise ValueError(f"Stage 6K interpolation mismatch for {planner} {key}")
    return rows, {
        "baseline_planner": baseline_name, "assertive_anchor_planner": anchor_name,
        "dose_profile_count": len(design["dose_profiles"]), "interpolation_passed": True,
        "planner_parameter_fingerprints": fingerprints,
    }


def validate_scenarios(design: Mapping[str, Any], rows: Sequence[Mapping[str, str]], fields: Sequence[str], path: Path) -> Dict[str, Any]:
    missing = sorted(set(SOURCE_FIELDS) - set(fields))
    if missing:
        raise ValueError(f"Stage 6J locked scenarios are missing fields: {missing}")
    forbidden = sorted(field for field in fields if "embedding" in field.lower() or "bdd" in field.lower())
    if forbidden:
        raise ValueError(f"Stage 6K freeze cannot read embedding/BDD columns: {forbidden}")
    source_sha = sha256_file(path)
    if source_sha != design["source_stage6j_locked_scenarios_sha256"]:
        raise ValueError("Stage 6J locked scenario SHA-256 differs from the Stage 6K design")
    if len(rows) != int(design["expected_scenario_count"]):
        raise ValueError(f"Stage 6K scenario count changed: {len(rows)}")
    orders = [int(row["collection_order"]) for row in rows]
    if orders != list(range(1, len(rows) + 1)):
        raise ValueError("Stage 6J collection order is not contiguous")
    tokens = [row["scenario_token"] for row in rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Stage 6K source scenarios contain duplicate tokens")
    counts = dict(sorted(Counter(row["task"] for row in rows).items()))
    expected = dict(sorted((str(key), int(value)) for key, value in design["expected_task_counts"].items()))
    if counts != expected:
        raise ValueError(f"Stage 6K task counts changed: actual={counts}, expected={expected}")
    return {"scenario_count": len(rows), "task_counts": counts, "distinct_log_count": len({row["log_name"] for row in rows}), "source_sha256": source_sha}


def build_jobs(design: Mapping[str, Any], scenarios: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    order = 0
    for item in design["dose_profiles"]:
        for source in scenarios:
            order += 1
            jobs.append({
                **source, "collection_order": order,
                "source_collection_order": int(source["collection_order"]),
                "dose": float(item["dose"]), "dose_label": item["label"],
                "planner_a": item["planner"], "planner_b": design["baseline_planner"],
                "selection_role": "FROZEN_STAGE6K_GRADED_PURE_LONGITUDINAL_PAIR",
            })
    return jobs


def build_report(audit: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(row["dose_label"]) for row in jobs)
    return "\n".join([
        "# Stage 6K 纯纵向处置剂量曲线冻结报告", "", "## 结论", "",
        "冻结通过。25%、50%、75% 三档均由保守端点向激进端点逐参数线性插值，横向参数完全不变。",
        "冻结时未读取新增剂量的 embedding、BDD 或 effect size。", "", "## 固定任务", "",
        f"- 同场景数量：{len(jobs) // 3}", f"- 新增场景×剂量任务：{len(jobs)}",
        f"- official rollout：{len(jobs) * 2}", f"- 分档数量：`{json.dumps(dict(counts), ensure_ascii=False)}`",
        f"- 插值审计：{str(audit['interpolation_passed']).lower()}", "", "## 预冻结判定", "",
        "- 不用跨数据集通用 raw BDD 阈值定义检出。",
        "- 每一剂量同时报告实现的纵向运动学差异与配对随机化 BDD。",
        "- 最小可检出剂量是同时通过运动学门禁且 paired BDD p<0.05 的最小非零剂量。",
        "- 固定183个场景、checkpoint、kernel与100000次配对交换；不得看结果后删剂量、换场景或提前停止。", "",
    ])


def freeze(args: argparse.Namespace) -> Dict[str, Any]:
    design_path = args.design_json.resolve()
    scenarios_path = args.stage6j_locked_scenarios_csv.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to rebuild")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design = read_json(design_path)
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError(f"Unexpected Stage 6K design schema: {design.get('schema_version')!r}")
    scenarios, fields = read_csv(scenarios_path)
    scenario_audit = validate_scenarios(design, scenarios, fields, scenarios_path)
    parameter_rows, profile_audit = audit_profiles(design)
    jobs = build_jobs(design, scenarios)
    jobs_path = output_dir / "stage6k_locked_jobs.csv"
    parameters_path = output_dir / "stage6k_profile_parameter_audit.csv"
    report_path = output_dir / "stage6k_freeze_report.md"
    write_csv(jobs_path, jobs, JOB_FIELDS)
    write_csv(parameters_path, parameter_rows, [
        "dose", "dose_label", "planner", "parameter", "dimension", "baseline_value", "anchor_value",
        "expected_value", "actual_value", "interpolation_passed",
    ])
    report_path.write_text(build_report(profile_audit, jobs), encoding="utf-8")
    tool_path = Path(__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": FREEZE_STATUS, "issue": design["issue"], "git_commit": git_commit(tool_path.parent.parent),
        "tool": str(tool_path), "tool_sha256": sha256_file(tool_path),
        "design_json": str(design_path), "design_sha256": sha256_file(design_path),
        "source_stage6j_locked_scenarios_csv": str(scenarios_path),
        "scenario_audit": scenario_audit, "profile_audit": profile_audit,
        "job_audit": {"job_count": len(jobs), "planned_rollout_count": len(jobs) * 2, "dose_count": len(design["dose_profiles"])},
        "analysis_freeze": design["analysis_freeze"], "embedding_or_bdd_read": False, "full_rollouts_launched": False,
        "outputs": {
            "locked_jobs": {"path": jobs_path.name, "sha256": sha256_file(jobs_path)},
            "profile_parameter_audit": {"path": parameters_path.name, "sha256": sha256_file(parameters_path)},
            "report": {"path": report_path.name, "sha256": sha256_file(report_path)},
        },
    }
    write_json(output_dir / "stage6k_freeze_manifest.json", result)
    return result


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
