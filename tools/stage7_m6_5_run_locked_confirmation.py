#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import sha256_file  # noqa: E402
from tools.stage7_m6_scenario_conditioned_bdd import (  # noqa: E402
    build_pair_quality_audit,
    fallback_distance_sensitivity,
    quality_sensitivity_analysis,
    validate_and_build_pairs,
)


SCHEMA_VERSION = "stage7_m6_5_locked_confirmation_analysis_v1"
PLANNER_A = "pdm_closed_assertive_v1"
PLANNER_B = "pdm_closed_conservative_v1"
PERMUTATIONS = 100000
SEED = 20260726
EXPECTED_HASHES = {
    "m6_1_analysis_tool": "fc22984f6a93fc1e625dc7040369ec2fd143e4829224304c61c419da01189e1f",
    "m6_2_analysis_tool": "3f9f2432d09ed44513b2f65d0f3207be6c88be61ec585604e9446d509c802bd2",
    "checkpoint": "909022f5df03a3f01c2149da6c9b44c613e955a4d816e8ec4d5862f39f8bf0cc",
}


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def path_record(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def freeze(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    locked_files = {
        "m6_1_frozen_spec": args.m6_1_frozen_spec,
        "m6_1_analysis_tool": args.m6_1_analysis_tool,
        "m6_2_lock_spec": args.m6_2_lock_spec,
        "m6_2_analysis_tool": args.m6_2_analysis_tool,
        "m6_3_power_justification": args.m6_3_power_justification,
        "checkpoint": args.checkpoint,
        "context_builder": args.context_builder,
        "embedding_tool": args.embedding_tool,
        "paired_delta_tool": args.paired_delta_tool,
        "quality_gate_tool": args.quality_gate_tool,
        "view_preparation_tool": args.view_preparation_tool,
        "representation_preparation_tool": args.representation_preparation_tool,
        "confirmation_analysis_tool": Path(__file__).resolve(),
        "confirmation_view_summary": args.confirmation_view_summary,
        "confirmation_ledger": args.confirmation_ledger,
        "development_metadata": args.development_metadata_csv,
    }
    records = {name: path_record(path) for name, path in locked_files.items()}
    for name, expected in EXPECTED_HASHES.items():
        if records[name]["sha256"] != expected:
            raise ValueError(f"{name} hash differs from frozen expected hash")
    view = read_json(args.confirmation_view_summary)
    if view.get("status") != "LOCKED_CONFIRMATION_VIEW_READY" or view.get("scenario_count") != 310:
        raise ValueError("confirmation view is not the exact ready 310-pair view")
    if not view.get("development_disjoint_audit", {}).get("pass"):
        raise ValueError("confirmation view development-disjoint audit did not pass")
    m6_1 = read_json(args.m6_1_frozen_spec)
    primary = m6_1.get("primary_analysis", {})
    if primary.get("permutations") != PERMUTATIONS:
        raise ValueError("M6.1 frozen permutation count differs from locked M6.5 count")
    lock = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_CONFIRMATION_EMBEDDING_UNBLINDING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "issue": "https://github.com/forwardxp-021/E2E-Evaluation/issues/240",
        "dataset_role": "NEW_LOG_AND_SCENARIO_DISJOINT_CONFIRMATION",
        "pair_count": 310,
        "row_count": 620,
        "task_counts": view["task_counts"],
        "planner_a": PLANNER_A,
        "planner_b": PLANNER_B,
        "primary_analysis": primary,
        "quality_sensitivity": m6_1["quality_sensitivity"],
        "task_conditioned_analysis": {
            "implementation": "unchanged frozen M6.2 tool",
            "permutations": PERMUTATIONS,
            "multiplicity": "Holm across five pre-treatment tasks for learned embedding",
            "representations": ["learned_embedding", "interaction_features", "trajectory_summary"],
        },
        "no_estimator_threshold_or_selection_changes_after_this_freeze": True,
        "quality_is_sensitivity_only": True,
        "locked_files": records,
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "m6_5_confirmation_analysis_lock.json", lock)
    print(json.dumps(lock, indent=2, ensure_ascii=False))


def validate_lock(args: argparse.Namespace) -> Dict[str, Any]:
    lock = read_json(args.lock_manifest)
    if lock.get("status") != "FROZEN_BEFORE_CONFIRMATION_EMBEDDING_UNBLINDING":
        raise ValueError("M6.5 analysis lock has invalid status")
    if lock.get("pair_count") != 310 or lock.get("row_count") != 620:
        raise ValueError("M6.5 analysis lock has wrong sample count")
    for name, record in lock.get("locked_files", {}).items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"locked file changed or disappeared: {name} ({path})")
    if sha256_file(Path(__file__).resolve()) != lock["locked_files"]["confirmation_analysis_tool"]["sha256"]:
        raise ValueError("confirmation analysis tool changed after freeze")
    return lock


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {args.output_dir}")
    lock = validate_lock(args)
    metadata = pd.read_csv(args.metadata_csv)
    paired_rows = read_csv(args.paired_delta_csv)
    embedding = np.asarray(np.load(args.embedding_path, mmap_mode="r"), dtype=np.float64)
    if embedding.shape != (620, 64) or not np.isfinite(embedding).all():
        raise ValueError(f"locked primary requires finite [620,64] embedding, got {embedding.shape}")
    pair_indices, scenarios = validate_and_build_pairs(
        metadata,
        paired_rows,
        len(embedding),
        planner_a=PLANNER_A,
        planner_b=PLANNER_B,
    )
    if pair_indices.shape != (310, 2) or len(scenarios) != 310:
        raise ValueError("locked confirmation pair validation did not yield exactly 310 pairs")
    row_quality = pd.read_csv(args.row_quality_csv)
    pair_quality = pd.read_csv(args.pair_quality_csv)
    pair_audit, pair_summary = build_pair_quality_audit(
        metadata,
        paired_rows,
        pair_indices,
        embedding,
        row_quality=row_quality,
        pair_quality=pair_quality,
    )
    args.output_dir.mkdir(parents=True)
    pair_audit.to_csv(args.output_dir / "m6_5_pair_quality_audit.csv", index=False)
    quality_rows, null_samples = quality_sensitivity_analysis(
        embedding,
        pair_indices,
        pair_audit,
        repetitions=PERMUTATIONS,
        seed=SEED,
    )
    pd.DataFrame(quality_rows).to_csv(args.output_dir / "m6_5_primary_and_quality_sensitivity.csv", index=False)
    fallback_rows = fallback_distance_sensitivity(pair_audit)
    pd.DataFrame(fallback_rows).to_csv(args.output_dir / "m6_5_quality_distance_sensitivity.csv", index=False)
    np.savez_compressed(args.output_dir / "m6_5_primary_quality_null_samples.npz", **null_samples)

    task_dir = args.output_dir / "task_conditioned"
    command = [
        sys.executable,
        str(Path(lock["locked_files"]["m6_2_analysis_tool"]["path"])),
        "--metadata_csv", str(args.metadata_csv),
        "--paired_delta_csv", str(args.paired_delta_csv),
        "--development_metadata_csv", str(Path(lock["locked_files"]["development_metadata"]["path"])),
        "--m6_frozen_spec", str(Path(lock["locked_files"]["m6_1_frozen_spec"]["path"])),
        "--representation", f"learned_embedding={args.embedding_path}",
        "--representation", f"interaction_features={args.interaction_representation_path}",
        "--representation", f"trajectory_summary={args.trajectory_representation_path}",
        "--output_dir", str(task_dir),
        "--analysis_role", "locked_confirmation",
        "--lock_manifest", str(Path(lock["locked_files"]["m6_2_lock_spec"]["path"])),
        "--power_justification_file", str(Path(lock["locked_files"]["m6_3_power_justification"]["path"])),
        "--planner_a", PLANNER_A,
        "--planner_b", PLANNER_B,
        "--minimum_overall_pairs", "80",
        "--minimum_task_pairs", "12",
        "--task_monte_carlo_permutations", str(PERMUTATIONS),
        "--seed", "20260729",
    ]
    completed = subprocess.run(command, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"frozen M6.2 task analysis failed with return code {completed.returncode}")
    task_summary = read_json(task_dir / "milestone6_2_summary.json")
    task_table = pd.read_csv(task_dir / "table_m6_2_task_paired_bdd.csv")
    primary = next(row for row in quality_rows if row["dataset"] == "full_primary")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCKED_CONFIRMATION_ANALYSIS_COMPLETE",
        "dataset_role": lock["dataset_role"],
        "pair_count": len(pair_indices),
        "row_count": len(embedding),
        "primary_endpoint": primary,
        "quality_sensitivity": quality_rows[1:],
        "pair_quality_audit": pair_summary,
        "task_conditioned_summary": task_summary,
        "learned_embedding_task_results": task_table.loc[
            task_table["representation"] == "learned_embedding"
        ].to_dict("records"),
        "inputs": {
            name: path_record(path)
            for name, path in {
                "analysis_lock": args.lock_manifest,
                "embedding": args.embedding_path,
                "metadata": args.metadata_csv,
                "paired_delta": args.paired_delta_csv,
                "row_quality": args.row_quality_csv,
                "pair_quality": args.pair_quality_csv,
                "interaction_representation": args.interaction_representation_path,
                "trajectory_representation": args.trajectory_representation_path,
            }.items()
        },
    }
    write_json(args.output_dir / "m6_5_locked_confirmation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def add_frozen_path_args(parser: argparse.ArgumentParser) -> None:
    base = Path("outputs")
    parser.add_argument("--m6_1_frozen_spec", type=Path, default=base / "stage7_m6_1_paired_bdd_method_freeze_v1/m6_frozen_analysis_spec.json")
    parser.add_argument("--m6_1_analysis_tool", type=Path, default=Path("tools/stage7_m6_scenario_conditioned_bdd.py"))
    parser.add_argument("--m6_2_lock_spec", type=Path, default=base / "stage7_m6_2_locked_task_bdd_development_v1/m6_2_locked_confirmation_spec.json")
    parser.add_argument("--m6_2_analysis_tool", type=Path, default=Path("tools/stage7_m6_2_locked_task_bdd.py"))
    parser.add_argument("--m6_3_power_justification", type=Path, default=base / "stage7_m6_3_simulation_power_v1/m6_3_locked_power_justification.json")
    parser.add_argument("--checkpoint", type=Path, default=base / "waymo_5neighbor_context_laneaware_clean_v1_full51_merged/context_gru_stage5d_balanced_v2/best_model.pt")
    parser.add_argument("--context_builder", type=Path, default=Path("tools/build_nuplan_5neighbor_context_dataset.py"))
    parser.add_argument("--embedding_tool", type=Path, default=Path("tools/stage7e_embed_stage6_dataset.py"))
    parser.add_argument("--paired_delta_tool", type=Path, default=Path("tools/stage7f_aggressive_conservative_paired_delta.py"))
    parser.add_argument("--quality_gate_tool", type=Path, default=Path("tools/stage7_m2b_build_paired_quality_gate.py"))
    parser.add_argument("--view_preparation_tool", type=Path, default=Path("tools/stage7_m6_5_prepare_locked_confirmation.py"))
    parser.add_argument("--representation_preparation_tool", type=Path, default=Path("tools/stage7_m6_5_prepare_confirmatory_representations.py"))
    parser.add_argument("--confirmation_view_summary", type=Path, default=base / "stage7_m6_5_locked_confirmation_view_v1/m6_5_confirmation_view_summary.json")
    parser.add_argument("--confirmation_ledger", type=Path, default=base / "stage7_m6_5_locked_confirmation_view_v1/confirmation_scenario_ledger.csv")
    parser.add_argument("--development_metadata_csv", type=Path, default=base / "stage7e_pdm_v1_balanced50_paired45_embeddings_v1_m3/metadata.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or run the Stage7 M6.5 locked 310-pair confirmation analysis.")
    sub = parser.add_subparsers(dest="mode", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--output_dir", type=Path, required=True)
    add_frozen_path_args(freeze_parser)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--lock_manifest", type=Path, required=True)
    run_parser.add_argument("--embedding_path", type=Path, required=True)
    run_parser.add_argument("--metadata_csv", type=Path, required=True)
    run_parser.add_argument("--paired_delta_csv", type=Path, required=True)
    run_parser.add_argument("--row_quality_csv", type=Path, required=True)
    run_parser.add_argument("--pair_quality_csv", type=Path, required=True)
    run_parser.add_argument("--interaction_representation_path", type=Path, required=True)
    run_parser.add_argument("--trajectory_representation_path", type=Path, required=True)
    run_parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
