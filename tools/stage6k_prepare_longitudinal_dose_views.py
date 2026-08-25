#!/usr/bin/env python3
"""Re-audit Stage 6K outputs and build one unified Stage7C view per dose."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6j_prepare_pure_longitudinal_view import prepare_view  # noqa: E402
from tools.stage7_m6_4b_run_locked_rollouts import (  # noqa: E402
    current_planner_fingerprints,
    sha256_file,
)


SCHEMA_VERSION = "stage6k_longitudinal_dose_views_v1"
ADDENDUM_STATUS = "FROZEN_BEFORE_NEW_DOSE_EMBEDDING_OR_BDD_READ"
DOSE_PLANNERS = {
    "dose25": "pdm_closed_assertive_longitudinal_dose25_v1",
    "dose50": "pdm_closed_assertive_longitudinal_dose50_v1",
    "dose75": "pdm_closed_assertive_longitudinal_dose75_v1",
}
BASELINE_PLANNER = "pdm_closed_conservative_longitudinal_v1"
EXPECTED_TASK_COUNTS = {
    "following_interaction": 60,
    "longitudinal_high_motion": 56,
    "stop_go_control": 67,
}
LEDGER_FIELDS = [
    "global_scenario_index", "collection_order", "source_collection_order", "dose", "dose_label",
    "planner_a", "planner_b", "source_global_scenario_index", "task", "source_task",
    "scenario_type", "log_name", "scenario_token", "db_file", "attempt", "stage7c_output_dir",
    "simulated_ego_seq_sha256", "simulated_ego_seq_mask_sha256", "scenario_planner_index_sha256",
    "official_msgpack_count", "official_msgpack_size_bytes", "stage7c_audit_pass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 6K dose-specific unified views without reading embedding/BDD.")
    parser.add_argument("--addendum_manifest", type=Path, required=True)
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_sources(args: argparse.Namespace) -> tuple[Dict[str, List[Dict[str, str]]], Dict[str, Any]]:
    paths = {name: getattr(args, name).resolve() for name in [
        "rollout_freeze_manifest", "locked_jobs_csv", "batch_manifest", "batch_state", "batch_status_csv"
    ]}
    addendum_path = args.addendum_manifest.resolve()
    addendum = read_json(addendum_path)
    if addendum.get("status") != ADDENDUM_STATUS or addendum.get("new_dose_embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6K pre-analysis addendum is not valid")
    for name, path in paths.items():
        expected = addendum.get("input_files", {}).get(name, {}).get("sha256")
        if expected != sha256_file(path):
            raise ValueError(f"Stage 6K {name} changed after pre-analysis addendum freeze")
    rollout_freeze = read_json(paths["rollout_freeze_manifest"])
    batch = read_json(paths["batch_manifest"])
    state = read_json(paths["batch_state"])
    jobs = read_csv(paths["locked_jobs_csv"])
    statuses = read_csv(paths["batch_status_csv"])
    if state.get("counts") != {"SUCCEEDED": 549, "FAILED_REVIEW_REQUIRED": 0, "PENDING": 0}:
        raise ValueError("Stage 6K batch is not complete")
    if batch.get("full_embedding_or_bdd_read") is not False or rollout_freeze.get("embedding_or_bdd_read") is not False:
        raise ValueError("Stage 6K source manifests do not certify BDD-blind rollout collection")
    if len(jobs) != 549 or len(statuses) != 549:
        raise ValueError("Stage 6K expected 549 job and status rows")
    jobs_by_order = {int(row["collection_order"]): row for row in jobs}
    status_by_order = {int(row["collection_order"]): row for row in statuses}
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for order in range(1, 550):
        job, status = jobs_by_order[order], status_by_order[order]
        if status.get("status") != "SUCCEEDED":
            raise ValueError(f"Stage 6K order {order} is not successful")
        for field in ["dose", "dose_label", "planner_a", "planner_b", "task", "log_name", "scenario_token"]:
            if job.get(field) != status.get(field):
                raise ValueError(f"Stage 6K order {order} changed field {field}")
        label = job["dose_label"]
        if job["planner_a"] != DOSE_PLANNERS.get(label) or job["planner_b"] != BASELINE_PLANNER:
            raise ValueError(f"Unexpected Stage 6K planner pair at order {order}")
        local_order = int(job["source_collection_order"])
        grouped[label].append({**job, **status, "collection_order": str(local_order), "source_collection_order": str(local_order)})
    canonical_tokens: List[str] | None = None
    for label in DOSE_PLANNERS:
        rows = sorted(grouped[label], key=lambda row: int(row["collection_order"]))
        grouped[label] = rows
        if len(rows) != 183 or [int(row["collection_order"]) for row in rows] != list(range(1, 184)):
            raise ValueError(f"Stage 6K {label} is not a complete local 1..183 view")
        task_counts = dict(Counter(row["task"] for row in rows))
        if task_counts != EXPECTED_TASK_COUNTS:
            raise ValueError(f"Stage 6K {label} task composition changed: {task_counts}")
        tokens = [row["scenario_token"] for row in rows]
        if canonical_tokens is None:
            canonical_tokens = tokens
        elif tokens != canonical_tokens:
            raise ValueError(f"Stage 6K {label} token order differs from other doses")
    planners = list(batch.get("planner_fingerprints", {}))
    required_planners = {*DOSE_PLANNERS.values(), BASELINE_PLANNER, "pdm_closed_assertive_longitudinal_v1"}
    if set(planners) != required_planners:
        raise ValueError(f"Unexpected planner fingerprint set in Stage 6K batch: {planners}")
    fingerprints = current_planner_fingerprints(planners)
    if batch.get("planner_fingerprints") != fingerprints:
        raise ValueError("Current Stage 6K planner fingerprints differ from batch manifest")
    return grouped, {
        "pass": True, "addendum_manifest_sha256": sha256_file(addendum_path),
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "planner_parameter_fingerprints": fingerprints, "dose_count": 3,
        "scenario_count_per_dose": 183, "job_count": 549, "rollout_count": 1098,
        "same_scenarios_and_order_across_doses": True,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir already exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    grouped, source_audit = validate_sources(args)
    output_dir.mkdir(parents=True)
    dose_summaries: Dict[str, Any] = {}
    try:
        for label, planner_a in DOSE_PLANNERS.items():
            dose_dir = output_dir / label
            view = prepare_view(
                grouped[label], dose_dir, [planner_a, BASELINE_PLANNER],
                ledger_filename="stage6k_scenario_ledger.csv", ledger_fields=LEDGER_FIELDS,
                stage_label=f"Stage 6K {label} unified longitudinal-only Stage7C view",
                schema_version=SCHEMA_VERSION,
            )
            dose_summary = {
                "schema_version": SCHEMA_VERSION, "status": "STAGE6K_DOSE_VIEW_READY",
                "dose_label": label, "dose": float(grouped[label][0]["dose"]),
                "planner_a": planner_a, "planner_b": BASELINE_PLANNER,
                "full_embedding_or_bdd_read": False, **view,
            }
            write_json(dose_dir / "stage6k_dose_view_summary.json", dose_summary)
            dose_summaries[label] = dose_summary
        summary = {
            "schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "STAGE6K_ALL_DOSE_VIEWS_READY", "full_embedding_or_bdd_read": False,
            "selection_changed_after_outcome_review": False, "source_audit": source_audit,
            "preparation_tool_sha256": sha256_file(Path(__file__).resolve()),
            "dose_summaries": dose_summaries,
        }
        write_json(output_dir / "stage6k_views_summary.json", summary)
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
