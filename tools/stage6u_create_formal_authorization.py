#!/usr/bin/env python3
"""Create the fail-closed Stage6U A/B/C formal-training authorization."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6u_unified_abc_trainer import (  # noqa: E402
    read_json,
    resolve_repo_path,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def ordered_candidate_seed_pairs(candidates: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    return [(candidate, int(seed)) for candidate in candidates for seed in seeds]


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    freeze_path = args.implementation_freeze_manifest.resolve()
    config = read_json(config_path)
    freeze = read_json(freeze_path)
    freeze_sha = sha256_file(freeze_path)
    if freeze.get("status") != "FROZEN_READY_FOR_ABC_FORMAL_TRAINING":
        raise ValueError(f"Implementation freeze is not ready: {freeze.get('status')}")
    if not freeze.get("validation") or not all(freeze["validation"].values()):
        raise ValueError("Implementation freeze has a failed validation gate")
    if freeze.get("formal_checkpoint_count") != 0:
        raise ValueError("Implementation freeze was not created at checkpoint count 0/9")
    current_sources = {
        "trainer": REPO_ROOT / "tools/stage6u_unified_abc_trainer.py",
        "stage6u_config": config_path,
    }
    for name, path in current_sources.items():
        actual = sha256_file(path)
        expected = freeze["source_records"][name]["sha256"]
        if actual != expected:
            raise ValueError(f"{name} changed after implementation freeze: expected={expected}, actual={actual}")

    stage6t_path = resolve_repo_path(config["stage6t_protocol"]["config_path"])
    stage6t = read_json(stage6t_path)
    guard = config["formal_training_guard"]
    candidates = list(guard["authorized_candidates_required"])
    seeds = [int(value) for value in guard["authorized_seeds_required"]]
    training_order = []
    order = 0
    for candidate, seed in ordered_candidate_seed_pairs(candidates, seeds):
        order += 1
        output_dir = resolve_repo_path(stage6t["candidates"][candidate]["output_root"]) / f"seed_{seed}"
        if output_dir.exists():
            raise FileExistsError(f"Formal output must be absent before authorization: {output_dir}")
        training_order.append(
            {
                "order": order,
                "candidate": candidate,
                "seed": seed,
                "output_dir": str(output_dir),
            }
        )

    support_paths = {
        "authorization_creator": REPO_ROOT / "tools/stage6u_create_formal_authorization.py",
        "trainer": current_sources["trainer"],
        "stage6u_config": config_path,
        "serial_orchestrator": REPO_ROOT / "tools/stage6u_run_formal_abc_serial.py",
        "monitor": REPO_ROOT / "tools/stage6u_monitor_formal_training.py",
        "checkpoint_locker": REPO_ROOT / "tools/stage6u_lock_formal_checkpoints.py",
        "stage6t_config": stage6t_path,
    }
    manifest = {
        "schema_version": "stage6u_formal_training_authorization_v1",
        "status": "AUTHORIZED_STAGE6U_ABC_FORMAL_TRAINING",
        "created_at_utc": utc_now_iso(),
        "authorization_scope": "A_B_C_X_3_SEEDS_FORMAL_TRAINING_ONLY",
        "training_authorized": True,
        "checkpoint_write_authorized": True,
        "implementation_freeze_manifest_path": str(freeze_path),
        "implementation_freeze_sha256": freeze_sha,
        "implementation_fingerprint_sha256": freeze["implementation_fingerprint_sha256"],
        "stage6t_protocol_id": stage6t["protocol_id"],
        "stage6t_protocol_fingerprint_sha256": freeze["stage6t_protocol_fingerprint_sha256"],
        "authorized_candidates": candidates,
        "authorized_seeds": seeds,
        "primary_seed": int(stage6t["common_optimization"]["primary_seed"]),
        "training_order": training_order,
        "single_device_serial_execution": True,
        "device_policy": stage6t["common_optimization"]["device_policy"],
        "concurrent_training_processes_max": 1,
        "training_constraints": {
            "waymo_train_only_for_optimization": True,
            "waymo_val_only_for_checkpoint_selection_and_early_stopping": True,
            "architecture_loss_sampling_hyperparameters_frozen": True,
            "maximum_epochs": int(stage6t["common_optimization"]["max_epochs"]),
            "maximum_optimizer_steps_per_seed": int(
                stage6t["common_optimization"]["max_total_optimizer_steps_per_seed"]
            ),
            "early_stopping_patience_epochs": int(
                stage6t["common_optimization"]["early_stopping_patience_epochs"]
            ),
            "best_epoch_rule": stage6t["checkpoint_selection"]["best_epoch_rule"],
        },
        "checkpoint_rules": {
            "best_checkpoint": "best_model.pt_selected_by_frozen_waymo_val_objective",
            "last_checkpoint": "last_model.pt_written_at_each_completed_epoch",
            "resume_checkpoint": "resume_model.pt_atomically_written_at_epoch_start_every_100_steps_and_epoch_boundary",
            "resume_must_restore": [
                "epoch",
                "next_batch_index",
                "epoch_accumulator",
                "global_step",
                "optimizer",
                "scheduler",
                "python_numpy_torch_rng",
                "random_plan_ledger_when_mid_epoch",
                "best_state",
                "patience_state",
            ],
            "overwrite_existing_seed_directory": False,
        },
        "logging_rules": {
            "tqdm": "train_and_validation_each_epoch",
            "jsonl_heartbeat_every_optimizer_steps": 100,
            "epoch_csv": "train_log.csv",
            "per_task_process_log": True,
            "hourly_read_only_monitor": True,
        },
        "forbidden_evaluation_boundary": {
            "waymo_test": True,
            "stage6j_k_p": True,
            "nuplan": True,
            "embedding_bdd_mmd": True,
            "stage6s_v2_confirmation": True,
        },
        "completion_rule": "LOCK_ALL_9_VALIDATION_SELECTED_BEST_CHECKPOINTS_THEN_STOP",
        "source_records": {name: source_record(path) for name, path in support_paths.items()},
        "environment_at_authorization": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
    }
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "stage6u_formal_training_authorization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--implementation_freeze_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "implementation_freeze_sha256": result["implementation_freeze_sha256"],
                "task_count": len(result["training_order"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
