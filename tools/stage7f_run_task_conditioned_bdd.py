#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, shutil, subprocess
from pathlib import Path
import pandas as pd

from tools.stage7f_idm_diagnostic_common import find_planner_index, idm_parameter_markdown, require_file

DEFAULT_TASK_KEYS = "task_following,task_lead_brake_response,task_queue_approach,task_cutin_response,task_yield_conflict"


def build_commands(args, events_dir: Path):
    emb_manifest = require_file(Path(args.embedding_dir) / "embedding_manifest.json", "embedding_manifest.json")
    shard_manifest = require_file(Path(args.context_dataset_dir) / "shard_manifest.json", "shard_manifest.json")
    schema = require_file(Path(args.context_dataset_dir) / "feature_schema.json", "feature_schema.json")
    a_idx = find_planner_index(Path(args.stage7f_dir), args.planner_a)
    b_idx = find_planner_index(Path(args.stage7f_dir), args.planner_b)
    build_cmd = [sys.executable, "tools/stage6c_build_behavior_events_v2.py", "--shard_manifest", str(shard_manifest), "--feature_schema_path", str(schema), "--output_dir", str(events_dir)]
    if args.overwrite_events:
        build_cmd.append("--overwrite")
    report_cmd = [sys.executable, "tools/stage6c_task_conditioned_bdd_report.py", "--embedding_manifest", str(emb_manifest), "--shard_manifest", str(shard_manifest), "--feature_schema_path", str(schema), "--a_indices_path", str(a_idx), "--b_indices_path", str(b_idx), "--behavior_event_bins_path", str(events_dir / "behavior_event_bins_v2.csv"), "--behavior_event_metrics_path", str(events_dir / "behavior_event_metrics_v2.csv"), "--output_dir", str(args.output_dir), "--task_keys", args.task_keys, "--min_bin_size", str(args.min_bin_size), "--num_bootstrap", str(args.num_bootstrap), "--num_permutation", str(args.num_permutation)]
    if args.overwrite:
        report_cmd.append("--overwrite")
    return build_cmd, report_cmd, {"embedding_manifest": str(emb_manifest), "shard_manifest": str(shard_manifest), "feature_schema": str(schema), "a_indices_path": str(a_idx), "b_indices_path": str(b_idx), "events_dir": str(events_dir)}


def write_stage7_summary(out: Path, args, resolved):
    bdd_csv = out / "task_bdd_summary.csv"
    rows = pd.read_csv(bdd_csv).to_dict("records") if bdd_csv.exists() else []
    summary = {"stage": "7F", "diagnostic": "task_conditioned_bdd_same_scenario_planner_pair", "planner_a": args.planner_a, "planner_b": args.planner_b, "min_bin_size": args.min_bin_size, "task_keys": args.task_keys.split(","), "resolved_inputs": resolved, "valid_task_count": len(rows), "outputs": ["task_report_card.md", "task_bdd_summary.csv", "task_style_delta.csv", "top_task_drift_cases.csv", "warnings.json", "plots/task_bdd_bar.png", "plots/task_style_delta_bar.png"]}
    (out / "stage7f_task_bdd_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Stage7F task-conditioned BDD summary", "", "This is task-conditioned BDD for same scenario planner pair A/B.", f"* A = `{args.planner_a}`", f"* B = `{args.planner_b}`", "* Overall Stage7F BDD was small, so this report checks task slices.", "* following and yield_conflict are strongest detectors; cutin/lead/queue may be proxy-based.", "* If task bins have low n_A/n_B, interpret as exploratory only.", "", idm_parameter_markdown(args.planner_a, args.planner_b), "", "## Stage6C task BDD rows", ""]
    if rows:
        lines += ["| task_key | n_A | n_B | BDD_MMD | p_value |", "|---|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r.get('task_key')} | {r.get('n_A')} | {r.get('n_B')} | {r.get('bdd_mmd')} | {r.get('p_value')} |")
    else:
        lines.append("No task passed the filters; inspect warnings.json.")
    (out / "stage7f_task_bdd_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    events_dir = Path(args.behavior_events_dir) if args.behavior_events_dir else out.parent / "stage7f_idm_20scenes_stage6c_behavior_events_v2"
    build_cmd, report_cmd, resolved = build_commands(args, events_dir)
    if not (events_dir / "behavior_event_bins_v2.csv").exists() or not (events_dir / "behavior_event_metrics_v2.csv").exists():
        subprocess.run(build_cmd + (["--overwrite"] if args.overwrite and "--overwrite" not in build_cmd else []), check=True)
    subprocess.run(report_cmd, check=True)
    write_stage7_summary(out, args, resolved)


def parse_args():
    p = argparse.ArgumentParser(description="Stage7F wrapper that reuses Stage6C v2 task-conditioned BDD for one planner pair.")
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--context_dataset_dir", required=True)
    p.add_argument("--stage7f_dir", required=True)
    p.add_argument("--planner_a", required=True)
    p.add_argument("--planner_b", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--task_keys", default=DEFAULT_TASK_KEYS)
    p.add_argument("--min_bin_size", type=int, default=2, help="Small default for Stage7 20-scenario pilot.")
    p.add_argument("--num_bootstrap", type=int, default=100)
    p.add_argument("--num_permutation", type=int, default=200)
    p.add_argument("--behavior_events_dir", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite_events", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
