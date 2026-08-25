#!/usr/bin/env python3
"""Freeze the Stage 6K pre-analysis addendum after rollouts and before BDD read."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMA_VERSION = "stage6k_preanalysis_addendum_freeze_v1"
DESIGN_SCHEMA_VERSION = "stage6k_preanalysis_addendum_design_v1"
OUTPUT_STATUS = "FROZEN_BEFORE_NEW_DOSE_EMBEDDING_OR_BDD_READ"
EXPECTED_DOSES = {"dose25": 0.25, "dose50": 0.5, "dose75": 0.75}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Stage 6K analysis addendum without reading embeddings or BDD.")
    parser.add_argument("--design_json", type=Path, required=True)
    parser.add_argument("--rollout_freeze_manifest", type=Path, required=True)
    parser.add_argument("--locked_jobs_csv", type=Path, required=True)
    parser.add_argument("--batch_manifest", type=Path, required=True)
    parser.add_argument("--batch_state", type=Path, required=True)
    parser.add_argument("--batch_status_csv", type=Path, required=True)
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: str(value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_design(design: Mapping[str, Any]) -> None:
    if design.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ValueError(f"Unexpected addendum design schema: {design.get('schema_version')!r}")
    primary = design.get("primary_overall_dose_family", {})
    if primary.get("hypothesis_count") != 4 or primary.get("correction") != "Holm":
        raise ValueError("Stage 6K addendum must use Holm across four nonzero overall doses")
    secondary = design.get("secondary_task_dose_family", {})
    if secondary.get("hypothesis_count") != 12 or secondary.get("correction") != "Holm_across_all_4_doses_x_3_tasks":
        raise ValueError("Stage 6K addendum must freeze one 12-hypothesis task-dose Holm family")
    gate = design.get("realized_kinematic_gate", {})
    if gate.get("cluster_unit") != "log_name" or gate.get("required_metrics") != ["delta_mean_speed", "delta_rms_accel"]:
        raise ValueError("Stage 6K realized gate must be log-clustered speed plus RMS-acceleration")
    if design.get("dose_zero_role", "").startswith("descriptive origin") is False:
        raise ValueError("Stage 6K dose zero must be descriptive only")


def audit_completed_rollouts(
    rollout_freeze: Mapping[str, Any], batch: Mapping[str, Any], state: Mapping[str, Any],
    jobs: Sequence[Mapping[str, str]], statuses: Sequence[Mapping[str, str]], jobs_path: Path,
) -> Dict[str, Any]:
    if rollout_freeze.get("status") != "FROZEN_BEFORE_LONGITUDINAL_DOSE_ROLLOUTS":
        raise ValueError("Original Stage 6K rollout freeze is not valid")
    if rollout_freeze.get("embedding_or_bdd_read") is not False:
        raise ValueError("Original rollout freeze does not certify embedding_or_bdd_read=false")
    if batch.get("schema_version") != "stage6k_longitudinal_dose_batch_v1":
        raise ValueError("Unexpected Stage 6K batch schema")
    if batch.get("full_embedding_or_bdd_read") is not False:
        raise ValueError("Batch manifest does not certify full_embedding_or_bdd_read=false")
    locked_sha = sha256_file(jobs_path)
    if locked_sha != rollout_freeze.get("outputs", {}).get("locked_jobs", {}).get("sha256"):
        raise ValueError("Locked jobs changed after rollout freeze")
    if locked_sha != batch.get("locked_jobs_sha256"):
        raise ValueError("Locked jobs differ from batch manifest")
    counts = state.get("counts", {})
    if counts != {"SUCCEEDED": 549, "FAILED_REVIEW_REQUIRED": 0, "PENDING": 0}:
        raise ValueError(f"Stage 6K rollout batch is incomplete: {counts}")
    if len(jobs) != 549 or len(statuses) != 549:
        raise ValueError(f"Expected 549 jobs/statuses, got {len(jobs)}/{len(statuses)}")
    jobs_by_order = {int(row["collection_order"]): row for row in jobs}
    statuses_by_order = {int(row["collection_order"]): row for row in statuses}
    if set(jobs_by_order) != set(range(1, 550)) or set(statuses_by_order) != set(range(1, 550)):
        raise ValueError("Stage 6K job/status order is not contiguous 1..549")
    dose_counts: Counter[str] = Counter()
    tokens_by_dose: Dict[str, List[str]] = defaultdict(list)
    distinct_logs: set[str] = set()
    retry_orders: List[int] = []
    for order in range(1, 550):
        job, status = jobs_by_order[order], statuses_by_order[order]
        for field in ["dose", "dose_label", "planner_a", "planner_b", "task", "log_name", "scenario_token"]:
            if job.get(field) != status.get(field):
                raise ValueError(f"Order {order} differs between locked job and status for {field}")
        label = job["dose_label"]
        if label not in EXPECTED_DOSES or abs(float(job["dose"]) - EXPECTED_DOSES[label]) > 1e-12:
            raise ValueError(f"Unexpected dose at order {order}: {job['dose_label']} {job['dose']}")
        if status.get("status") != "SUCCEEDED" or int(status.get("official_success_count", "0")) != 2:
            raise ValueError(f"Order {order} is not a 2/2 official success")
        if not truth(status.get("same_log_alignment_passed", "")) or not truth(status.get("strict_token_alignment_passed", "")):
            raise ValueError(f"Order {order} failed frozen alignment")
        if not Path(status.get("stage7c_output_dir", "")).is_dir():
            raise FileNotFoundError(f"Order {order} Stage7C output is missing: {status.get('stage7c_output_dir')}")
        dose_counts[label] += 1
        tokens_by_dose[label].append(job["scenario_token"])
        distinct_logs.add(job["log_name"])
        if int(status.get("attempt", "0")) > 1:
            retry_orders.append(order)
    if dose_counts != Counter({label: 183 for label in EXPECTED_DOSES}):
        raise ValueError(f"Unexpected completed dose counts: {dose_counts}")
    canonical_tokens = tokens_by_dose["dose25"]
    for label in ["dose50", "dose75"]:
        if tokens_by_dose[label] != canonical_tokens:
            raise ValueError(f"Scenario token order changed for {label}")
    return {
        "pass": True, "job_count": 549, "rollout_count": 1098,
        "dose_counts": dict(sorted(dose_counts.items())), "shared_scenario_count": len(canonical_tokens),
        "distinct_log_count": len(distinct_logs), "retry_orders": retry_orders,
        "same_scenarios_and_order_across_doses": True, "all_2_of_2_official_success": True,
        "all_same_log_and_strict_token_alignment": True,
    }


def report(audit: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Stage 6K 解盲前统计补充冻结报告", "", "## 结论", "",
        "补充规则已在读取25%/50%/75%新增剂量embedding或BDD之前冻结。原rollout freeze、场景、planner和输出均未改变。", "",
        "## Rollout完成审计", "",
        f"- scene-dose任务：{audit['job_count']}", f"- official rollout：{audit['rollout_count']}",
        f"- 三档完成数：`{json.dumps(audit['dose_counts'], ensure_ascii=False)}`",
        f"- 相同场景：{audit['shared_scenario_count']}", f"- 独立log：{audit['distinct_log_count']}",
        f"- 产生新attempt后成功的order：`{audit['retry_orders']}`", "", "## 新冻结规则", "",
        "1. 25/50/75/100四个overall剂量构成一个Holm family。",
        "2. 最小可检出名义剂量必须同时通过实现运动学门禁和overall Holm p<0.05。",
        "3. 4剂量×3 tasks共12个secondary检验统一Holm。",
        "4. 区分IDM参数插值的名义剂量与speed/accel/jerk/THW/gap的实现剂量。",
        "5. 同log整体label flip仅作为cluster sensitivity，不替代scenario-pair primary。",
        "6. fallback/ambiguity关联是post-treatment描述性敏感性，不得用于删样本或因果调整。", "",
    ])


def freeze(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: getattr(args, name).resolve() for name in [
        "design_json", "rollout_freeze_manifest", "locked_jobs_csv", "batch_manifest", "batch_state", "batch_status_csv"
    ]}
    design = read_json(paths["design_json"])
    validate_design(design)
    rollout_freeze = read_json(paths["rollout_freeze_manifest"])
    batch = read_json(paths["batch_manifest"])
    state = read_json(paths["batch_state"])
    jobs = read_csv(paths["locked_jobs_csv"])
    statuses = read_csv(paths["batch_status_csv"])
    audit = audit_completed_rollouts(rollout_freeze, batch, state, jobs, statuses, paths["locked_jobs_csv"])
    report_path = output_dir / "stage6k_preanalysis_addendum_report_zh.md"
    report_path.write_text(report(audit), encoding="utf-8")
    tool_path = Path(__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION, "status": OUTPUT_STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(), "issue": design["issue"],
        "reason": design["reason"], "new_dose_embedding_or_bdd_read": False,
        "original_rollout_freeze_overwritten": False, "rollout_audit": audit,
        "analysis_specification": {key: value for key, value in design.items() if key not in {"schema_version", "issue", "reason"}},
        "input_files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "tool": str(tool_path), "tool_sha256": sha256_file(tool_path),
        "outputs": {"report": {"path": report_path.name, "sha256": sha256_file(report_path)}},
    }
    manifest_path = output_dir / "stage6k_preanalysis_addendum_manifest.json"
    write_json(manifest_path, result)
    return result


def main() -> None:
    print(json.dumps(freeze(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
