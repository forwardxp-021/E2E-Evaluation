#!/usr/bin/env python3
"""Run one development-only Stage7L token through all five official nuPlan doses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.stage7_m6_4b_run_locked_rollouts import stage7c_environment
from tools.stage7l_pure_lateral_execution_planner import DOSE_TRANSITION_LENGTH_M


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def permanently_exclude_smoke_tokens(
    path: Path, maneuvers: List[Dict[str, Any]], maneuver_manifest: Path
) -> str:
    fields = ["scenario_token", "log_name", "exclusion_reason", "source_path"]
    existing: List[Dict[str, str]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    by_token = {row["scenario_token"]: row for row in existing}
    for maneuver in maneuvers:
        by_token[maneuver["scenario_token"]] = {
            "scenario_token": maneuver["scenario_token"], "log_name": maneuver["log_name"],
            "exclusion_reason": "STAGE7L_A2_DEVELOPMENT_ONLY_OFFICIAL_SMOKE",
            "source_path": str(maneuver_manifest.resolve()),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows(sorted(by_token.values(), key=lambda row: row["scenario_token"]))
    return sha256_file(path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = json.loads(args.maneuver_manifest.read_text(encoding="utf-8"))
    maneuvers = manifest["maneuvers"]
    if not 1 <= len(maneuvers) <= 2:
        raise ValueError("A2 official smoke permits only one or two development-only scenarios")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusion_sha = permanently_exclude_smoke_tokens(
        args.prior_exclusion_ledger, maneuvers, args.maneuver_manifest
    )
    context = args.output_dir / "context"
    context.mkdir(exist_ok=True)
    fields = ["collection_order", "source_global_scenario_index", "task", "source_task", "scenario_type", "log_name", "scenario_token", "scene_token", "db_file", "selection_role", "actual_nuplan_token"]
    rows = []
    for index, maneuver in enumerate(maneuvers, start=1):
        rows.append({
            "collection_order": index, "source_global_scenario_index": index - 1,
            "task": "stage7l_a2_smoke", "source_task": "pre_treatment_map_opportunity",
            "scenario_type": "unknown", "log_name": maneuver["log_name"],
            "scenario_token": maneuver["scenario_token"], "scene_token": maneuver["scenario_token"],
            "db_file": maneuver["db_file"], "selection_role": "A2_SMOKE_ONLY_PERMANENTLY_EXCLUDED",
            "actual_nuplan_token": maneuver["scenario_token"],
        })
    with (context / "merged_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    planners = [f"stage7l_pure_lateral_{dose}" for dose in DOSE_TRANSITION_LENGTH_M]
    hydra_searchpath = (
        f"[file://{(args.project_root / 'configs/stage7l_hydra').resolve()},"
        "pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
    )
    command_template = " ".join([
        str(args.python_executable.resolve()), str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()),
        "+simulation=closed_loop_nonreactive_agents", "{planner_hydra_overrides}",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{args.nuplan_db_root.resolve()}/{{target_log_name}}.db]",
        "scenario_filter=all_scenarios", "{scenario_hydra_overrides}", "worker=single_machine_thread_pool",
        "worker.max_workers=1", "scenario_builder.max_workers=1", "max_callback_workers=1", "gpu=false",
        "experiment_name=stage7l_a2_official_smoke_v1", "job_name=closed_loop_nonreactive_agents_{planner_name_safe}",
        "output_dir={output_dir}",
    ])
    output = args.output_dir / "stage7c_output"
    command: List[str] = [
        str(args.python_executable.resolve()), str(args.stage7c_tool.resolve()),
        "--context_dir", str(context.resolve()), "--nuplan_db_root", str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root", str(args.nuplan_map_root.resolve()), "--output_dir", str(output.resolve()),
        "--planners", *planners, "--max_scenarios", str(len(rows)), "--min_timesteps", "20",
        "--require_same_scenario_alignment", "--require_strict_nuplan_token_alignment",
        "--allow_unsafe_pickle_artifacts",
        "--allow_external_planner_name", "--hydra_searchpath", hydra_searchpath,
        "--command_timeout_s", str(args.command_timeout_s),
        "--nuplan_simulation_command_template", command_template,
    ]
    env_args = argparse.Namespace(
        nuplan_devkit_root=args.nuplan_devkit_root, tuplan_garage_root=args.tuplan_garage_root,
        nuplan_data_root=args.nuplan_data_root, nuplan_map_root=args.nuplan_map_root,
        nuplan_exp_root=args.nuplan_exp_root,
    )
    env = stage7c_environment(env_args)
    env["STAGE7L_MANEUVER_MANIFEST"] = str(args.maneuver_manifest.resolve())
    env["STAGE7L_PLANNER_AUDIT_DIR"] = str((args.output_dir / "planner_audits").resolve())
    log_path = args.output_dir / "stage7l_a2_official_smoke.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv: " + json.dumps(command, ensure_ascii=False) + "\n\n"); log.flush()
        process = subprocess.run(command, cwd=args.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    progress_path = output / "stage7c_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    task_records = progress.get("task_records", [])
    official_success_count = sum(row.get("status") == "succeeded" for row in task_records)
    expected = len(rows) * len(planners)
    audit_files = sorted((args.output_dir / "planner_audits").glob("planner_audit_*.json"))
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_files]
    by_token: Dict[str, List[Dict[str, Any]]] = {}
    for audit in audits:
        by_token.setdefault(audit["scenario_token"], []).append(audit)
    identity = {}
    for token, group in by_token.items():
        s_arrays = [np.asarray(row["s_route_initial_plan_m"], dtype=float) for row in group]
        identity[token] = {
            "dose_count": len(group),
            "s_route_pointwise_identical": len(group) == 5 and all(np.array_equal(s_arrays[0], arr) for arr in s_arrays[1:]),
            "manifest_sha_identical": len({row["dose_invariant_manifest_sha256"] for row in group}) == 1,
            "longitudinal_generator_sha_identical": len({row["canonical_longitudinal_generator_sha256"] for row in group}) == 1,
        }
    result = {
        "schema_version": "stage7l_a2_official_smoke_v1",
        "status": "PASS" if process.returncode == 0 and len(audits) == expected and all(v["s_route_pointwise_identical"] and v["manifest_sha_identical"] and v["longitudinal_generator_sha_identical"] for v in identity.values()) else "FAIL",
        "return_code": process.returncode, "scenario_count": len(rows), "dose_count": len(planners),
        "expected_rollout_count": expected, "planner_audit_count": len(audits),
        "official_success_count": official_success_count,
        "canonical_identity_audit": identity, "background_mode": "closed_loop_nonreactive_agents",
        "smoke_tokens_permanently_excluded": [row["scenario_token"] for row in maneuvers],
        "prior_exclusion_ledger": str(args.prior_exclusion_ledger.resolve()),
        "prior_exclusion_ledger_sha256": exclusion_sha,
        "embedding_or_bdd_read": False, "log": str(log_path.resolve()), "stage7c_output": str(output.resolve()),
    }
    (args.output_dir / "stage7l_a2_official_smoke_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maneuver_manifest", type=Path, required=True)
    parser.add_argument("--prior_exclusion_ledger", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--nuplan_data_root", type=Path, required=True)
    parser.add_argument("--nuplan_exp_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--tuplan_garage_root", type=Path, required=True)
    parser.add_argument("--stage7c_tool", type=Path, required=True)
    parser.add_argument("--python_executable", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--command_timeout_s", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
