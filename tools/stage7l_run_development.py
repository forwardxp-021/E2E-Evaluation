#!/usr/bin/env python3
"""Run a frozen Stage7L-B development manifest through five official doses."""

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


DOSES = (0, 25, 50, 75, 100)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = json.loads(args.maneuver_manifest.read_text(encoding="utf-8"))
    maneuvers = manifest["maneuvers"]
    if not maneuvers or len(maneuvers) > 24:
        raise ValueError("Stage7L-B run requires 1..24 frozen development scenarios")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    context = args.output_dir / "context"; context.mkdir()
    fields = ["collection_order", "source_global_scenario_index", "task", "source_task", "scenario_type",
              "log_name", "scenario_token", "scene_token", "db_file", "selection_role", "actual_nuplan_token"]
    rows = []
    for index, item in enumerate(maneuvers, start=1):
        rows.append({
            "collection_order": index, "source_global_scenario_index": index - 1,
            "task": "stage7l_b_development", "source_task": "pre_treatment_map_opportunity",
            "scenario_type": "unknown", "log_name": item["log_name"], "scenario_token": item["scenario_token"],
            "scene_token": item["scenario_token"], "db_file": item["db_file"],
            "selection_role": "DEVELOPMENT_ONLY_PERMANENTLY_EXCLUDED",
            "actual_nuplan_token": item["scenario_token"],
        })
    with (context / "merged_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    planners = [f"{args.planner_prefix}_dose{dose}" for dose in DOSES]
    searchpath = (
        f"[file://{(args.project_root / 'configs/stage7l_hydra').resolve()},"
        "pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
    )
    template = " ".join([
        str(args.python_executable.resolve()), str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()),
        "+simulation=closed_loop_nonreactive_agents", "{planner_hydra_overrides}",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{args.nuplan_db_root.resolve()}/{{target_log_name}}.db]",
        "+scenario_builder.scenario_mapping.scenario_map.unknown=[15.0,0.0]",
        "scenario_filter=all_scenarios", "{scenario_hydra_overrides}", "worker=single_machine_thread_pool",
        "worker.max_workers=1", "scenario_builder.max_workers=1", "max_callback_workers=1", "gpu=false",
        "experiment_name=stage7l_b_development_v1", "job_name=closed_loop_nonreactive_agents_{planner_name_safe}",
        "output_dir={output_dir}",
    ])
    output = args.output_dir / "stage7c_output"
    command: List[str] = [
        str(args.python_executable.resolve()), str(args.stage7c_tool.resolve()),
        "--context_dir", str(context.resolve()), "--nuplan_db_root", str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root", str(args.nuplan_map_root.resolve()), "--output_dir", str(output.resolve()),
        "--planners", *planners, "--max_scenarios", str(len(rows)), "--min_timesteps", "20",
        "--require_same_scenario_alignment", "--require_strict_nuplan_token_alignment",
        "--allow_unsafe_pickle_artifacts", "--allow_external_planner_name", "--hydra_searchpath", searchpath,
        "--command_timeout_s", str(args.command_timeout_s), "--nuplan_simulation_command_template", template,
    ]
    env_args = argparse.Namespace(
        nuplan_devkit_root=args.nuplan_devkit_root, tuplan_garage_root=args.tuplan_garage_root,
        nuplan_data_root=args.nuplan_data_root, nuplan_map_root=args.nuplan_map_root, nuplan_exp_root=args.nuplan_exp_root,
    )
    env = stage7c_environment(env_args)
    env["STAGE7L_MANEUVER_MANIFEST"] = str(args.maneuver_manifest.resolve())
    env["STAGE7L_PLANNER_AUDIT_DIR"] = str((args.output_dir / "planner_audits").resolve())
    log_path = args.output_dir / "stage7l_b_development.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv: " + json.dumps(command, ensure_ascii=False) + "\n\n"); log.flush()
        process = subprocess.run(command, cwd=args.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    progress_path = output / "stage7c_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    records = progress.get("task_records", [])
    success = sum(row.get("status") == "succeeded" for row in records)
    expected = len(rows) * 5
    audit_files = sorted((args.output_dir / "planner_audits").glob("planner_audit_*.json"))
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_files]
    by_token: Dict[str, List[Dict[str, Any]]] = {}
    for audit in audits:
        by_token.setdefault(audit["scenario_token"], []).append(audit)
    identity: Dict[str, Any] = {}
    for token, group in by_token.items():
        arrays = [np.asarray(row["s_route_initial_plan_m"], dtype=float) for row in group]
        identity[token] = {
            "dose_count": len(group),
            "s_route_pointwise_identical": len(group) == 5 and all(np.array_equal(arrays[0], arr) for arr in arrays[1:]),
            "manifest_sha_identical": len({row["dose_invariant_manifest_sha256"] for row in group}) == 1,
            "longitudinal_generator_sha_identical": len({row["canonical_longitudinal_generator_sha256"] for row in group}) == 1,
        }
    purity = len(identity) == len(rows) and all(
        row["s_route_pointwise_identical"] and row["manifest_sha_identical"]
        and row["longitudinal_generator_sha_identical"] for row in identity.values()
    )
    result = {
        "schema_version": "stage7l_b_development_run_v1", "status": "PASS" if success == expected and purity else "FAIL",
        "role": manifest.get("role"), "return_code": process.returncode, "scenario_count": len(rows),
        "dose_count": 5, "expected_rollout_count": expected, "official_success_count": success,
        "failed_rollout_count": expected - success, "planner_audit_count": len(audits),
        "canonical_identity_all_pass": purity, "canonical_identity_audit": identity,
        "background_mode": "closed_loop_nonreactive_agents", "manifest_sha256": sha256_file(args.maneuver_manifest),
        "embedding_or_bdd_read": False, "stage7c_output": str(output.resolve()), "log": str(log_path.resolve()),
    }
    summary_path = args.output_dir / "stage7l_b_development_run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maneuver_manifest", type=Path, required=True)
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
    parser.add_argument("--planner_prefix", default="stage7l_b_pure_lateral")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
