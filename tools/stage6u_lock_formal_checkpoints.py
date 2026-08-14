#!/usr/bin/env python3
"""Lock the nine Stage6U validation-selected checkpoints without reading test or nuPlan."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6u_unified_abc_trainer import read_json, resolve_repo_path, sha256_file  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args: argparse.Namespace) -> dict[str, Any]:
    authorization_path = args.authorization_manifest.resolve()
    authorization = read_json(authorization_path)
    authorization_sha = sha256_file(authorization_path)
    if authorization.get("status") != "AUTHORIZED_STAGE6U_ABC_FORMAL_TRAINING":
        raise ValueError("Formal authorization status is invalid")
    for name in ("trainer", "stage6u_config", "checkpoint_locker"):
        record = authorization.get("source_records", {}).get(name)
        if not record or sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Authorized source changed before checkpoint lock: {name}")
    rows = []
    for run_record in authorization["training_order"]:
        candidate = str(run_record["candidate"])
        seed = int(run_record["seed"])
        root = resolve_repo_path(run_record["output_dir"])
        summary_path = root / "formal_training_summary.json"
        best_path = root / "best_model.pt"
        last_path = root / "last_model.pt"
        config_path = root / "formal_training_config.json"
        for path in (summary_path, best_path, last_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = read_json(summary_path)
        if summary.get("candidate") != candidate or int(summary.get("seed", -1)) != seed:
            raise ValueError(f"Summary identity mismatch for {candidate}/{seed}")
        if summary.get("training_complete") is not True:
            raise ValueError(f"Training is incomplete for {candidate}/{seed}")
        forbidden_flags = {
            "waymo_test_read": summary.get("waymo_test_read"),
            "stage6j_k_p_read_or_run": summary.get("stage6j_k_p_read_or_run"),
            "nuplan_read_or_run": summary.get("nuplan_read_or_run"),
            "stage6s_v2_confirmation_read_or_run": summary.get("stage6s_v2_confirmation_read_or_run"),
            "embedding_bdd_mmd_read": summary.get("embedding_bdd_mmd_read"),
        }
        if any(value is not False for value in forbidden_flags.values()):
            raise ValueError(f"Blind boundary failed for {candidate}/{seed}: {forbidden_flags}")
        best_sha = sha256_file(best_path)
        last_sha = sha256_file(last_path)
        if summary.get("best_checkpoint_sha256") != best_sha or summary.get("last_checkpoint_sha256") != last_sha:
            raise ValueError(f"Checkpoint SHA mismatch for {candidate}/{seed}")
        if summary.get("authorization_manifest_sha256") != authorization_sha:
            raise ValueError(f"Authorization binding mismatch for {candidate}/{seed}")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("checkpoint_metadata", {})
        if checkpoint.get("candidate") != candidate or int(checkpoint.get("seed", -1)) != seed:
            raise ValueError(f"Best checkpoint identity mismatch for {candidate}/{seed}")
        if checkpoint.get("authorization_manifest_sha256") != authorization_sha:
            raise ValueError(f"Best checkpoint authorization mismatch for {candidate}/{seed}")
        rows.append(
            {
                "order": int(run_record["order"]),
                "candidate": candidate,
                "seed": seed,
                "primary_seed": seed == int(authorization["primary_seed"]),
                "best_epoch": int(summary["best_epoch"]),
                "best_waymo_val_loss": float(summary["best_val_loss"]),
                "validation_selection_rule": metadata["validation_objective"],
                "stopped_reason": summary["stopped_reason"],
                "global_step": int(summary["global_step"]),
                "best_checkpoint_path": str(best_path),
                "best_checkpoint_sha256": best_sha,
                "last_checkpoint_path": str(last_path),
                "last_checkpoint_sha256": last_sha,
                "training_config_path": str(config_path),
                "training_config_sha256": sha256_file(config_path),
                "trainer_sha256": summary["trainer_sha256"],
                "stage6u_config_sha256": summary["stage6u_config_sha256"],
                "implementation_freeze_sha256": summary["implementation_freeze_sha256"],
                "authorization_manifest_sha256": authorization_sha,
                "dataset_content_signature_sha256": metadata["dataset_content_signature_sha256"],
                "architecture_id": metadata["architecture_id"],
                "sampling_package_id": metadata["sampling_package_id"],
                "objective_package_id": metadata["objective_package_id"],
                "git_commit": metadata["git_commit"],
                "training_environment": metadata["training_environment"],
                "training_complete": True,
                "resume_history": summary.get("resume_history", []),
            }
        )
    if len(rows) != 9 or len({(row["candidate"], row["seed"]) for row in rows}) != 9:
        raise ValueError("Checkpoint ledger must contain exactly nine unique candidate/seed rows")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema_version": "stage6u_formal_checkpoint_ledger_v1",
        "status": "LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK",
        "locked_at_utc": utc_now_iso(),
        "authorization_manifest_path": str(authorization_path),
        "authorization_manifest_sha256": authorization_sha,
        "implementation_freeze_sha256": authorization["implementation_freeze_sha256"],
        "primary_seed": int(authorization["primary_seed"]),
        "checkpoint_count": 9,
        "rows": rows,
        "waymo_test_read": False,
        "stage6j_k_p_read_or_run": False,
        "nuplan_read_or_run": False,
        "stage6s_v2_confirmation_read_or_run": False,
        "embedding_bdd_mmd_read": False,
        "formal_evaluation_unlocked": False,
        "next_action_requires_separate_authorization": True,
        "environment_at_lock": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    json_path = output_dir / "stage6u_formal_checkpoint_ledger.json"
    json_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output_dir / "stage6u_formal_checkpoint_ledger.csv"
    csv_fields = [
        "order", "candidate", "seed", "primary_seed", "best_epoch", "best_waymo_val_loss",
        "stopped_reason", "global_step", "best_checkpoint_path", "best_checkpoint_sha256",
        "last_checkpoint_path", "last_checkpoint_sha256", "training_config_sha256", "trainer_sha256",
        "stage6u_config_sha256", "implementation_freeze_sha256", "authorization_manifest_sha256",
        "training_complete",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Stage6U A/B/C正式训练checkpoint锁定报告",
        "",
        f"状态：`{ledger['status']}`",
        "",
        "9个任务全部仅用Waymo train优化、Waymo val选best epoch。Primary seed固定为3407。",
        "本锁定没有读取Waymo test、Stage6J/K/P、nuPlan、BDD/MMD或Stage6S-v2 confirmation。",
        "",
        "| Candidate | Seed | Best epoch | Waymo val loss | Best SHA256 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['candidate']} | {row['seed']} | {row['best_epoch']} | "
            f"{row['best_waymo_val_loss']:.8f} | `{row['best_checkpoint_sha256']}` |"
        )
    report.extend(
        [
            "",
            "已达到可另行授权一次性Waymo test与后续nuPlan盲测的前置条件，但本阶段没有解锁或运行评估。",
        ]
    )
    (output_dir / "stage6u_formal_checkpoint_lock_report_zh.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "checkpoint_count": result["checkpoint_count"]}, indent=2))
