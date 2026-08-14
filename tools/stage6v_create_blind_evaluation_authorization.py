#!/usr/bin/env python3
"""Create the one-time Stage6 blind-evaluation authorization.

This command reads only frozen protocol/training/roster metadata and checkpoint
bytes for SHA verification.  It does not open Waymo test rows, nuPlan
representations, BDD/MMD results, or launch any evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LEDGER_STATUS = "LOCKED_9_OF_9_READY_FOR_BLIND_EVALUATION_UNLOCK"
AUTHORIZATION_STATUS = "AUTHORIZED_STAGE6_ONE_TIME_BLIND_EVALUATION"
IMMUTABILITY_STATEMENT = "evaluation results cannot trigger retraining or protocol changes"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: expected={expected}, actual={actual}, path={path}")
    return {"path": str(path.resolve()), "sha256": actual, "size_bytes": path.stat().st_size}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def validate_checkpoint_ledger(
    ledger_path: Path,
    implementation_freeze_sha256: str,
    training_authorization_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ledger = read_json(ledger_path)
    if ledger.get("status") != REQUIRED_LEDGER_STATUS:
        raise ValueError(f"Checkpoint ledger is not ready: {ledger.get('status')!r}")
    if int(ledger.get("checkpoint_count", -1)) != 9 or len(ledger.get("rows", [])) != 9:
        raise ValueError("Checkpoint ledger must contain exactly 9 locked rows")
    if int(ledger.get("primary_seed", -1)) != 3407:
        raise ValueError("Primary seed must remain fixed at 3407")
    if ledger.get("implementation_freeze_sha256") != implementation_freeze_sha256:
        raise ValueError("Checkpoint ledger implementation-freeze binding changed")
    if ledger.get("authorization_manifest_sha256") != training_authorization_sha256:
        raise ValueError("Checkpoint ledger training-authorization binding changed")
    forbidden_flags = (
        "waymo_test_read",
        "stage6j_k_p_read_or_run",
        "nuplan_read_or_run",
        "stage6s_v2_confirmation_read_or_run",
        "embedding_bdd_mmd_read",
    )
    if any(ledger.get(key) is not False for key in forbidden_flags):
        raise ValueError("Pre-evaluation blind boundary was already crossed")
    if ledger.get("formal_evaluation_unlocked") is not False:
        raise ValueError("Checkpoint ledger unexpectedly reports prior evaluation unlock")
    expected_order = [(candidate, seed) for candidate in "ABC" for seed in (3407, 3408, 3409)]
    observed_order = [(str(row.get("candidate")), int(row.get("seed", -1))) for row in ledger["rows"]]
    if observed_order != expected_order:
        raise ValueError(f"Checkpoint order changed: {observed_order}")
    checkpoint_records: list[dict[str, Any]] = []
    for row in ledger["rows"]:
        if row.get("training_complete") is not True or row.get("resume_history") != []:
            raise ValueError(f"Training integrity failed for {row.get('candidate')}/{row.get('seed')}")
        best_path = resolve(row["best_checkpoint_path"])
        best_record = require_sha(
            best_path,
            str(row["best_checkpoint_sha256"]),
            f"best checkpoint {row['candidate']}/{row['seed']}",
        )
        checkpoint_records.append(
            {
                "order": int(row["order"]),
                "candidate": str(row["candidate"]),
                "seed": int(row["seed"]),
                "primary_seed": bool(row["primary_seed"]),
                "best_epoch": int(row["best_epoch"]),
                "best_waymo_val_loss": float(row["best_waymo_val_loss"]),
                "validation_selection_rule": str(row["validation_selection_rule"]),
                "checkpoint": best_record,
                "training_config_sha256": str(row["training_config_sha256"]),
                "trainer_sha256": str(row["trainer_sha256"]),
                "implementation_freeze_sha256": str(row["implementation_freeze_sha256"]),
                "training_authorization_sha256": str(row["authorization_manifest_sha256"]),
            }
        )
    return ledger, checkpoint_records


def build_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Stage6 一次性盲测授权报告",
        "",
        f"状态：`{manifest['status']}`",
        "",
        "本授权严格绑定Stage6T协议、Stage6U训练实现与训练授权、9个已锁定best checkpoint，以及Stage6S-v2 80-pair confirmation roster。",
        "",
        f"不可变规则：`{manifest['immutability_statement']}`。",
        "",
        "## 固定执行顺序",
        "",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(manifest["evaluation_sequence"], start=1))
    lines += [
        "",
        "## 训练边界",
        "",
        "- 禁止重新训练、换seed、换epoch、改loss、改architecture或根据盲测结果修改协议。",
        "- Primary seed固定为3407；3408/3409只用于seed stability。",
        "- Stage6S-v2必须先通过trajectory mechanism gate，才允许读取interaction representation。",
        "- 本授权生成步骤没有读取Waymo test、nuPlan embedding或BDD/MMD结果，也没有启动rollout。",
        "",
        "## 锁定checkpoint",
        "",
        "| Candidate | Seed | Best epoch | Best SHA256 |",
        "|---|---:|---:|---|",
    ]
    for row in manifest["locked_best_checkpoints"]:
        lines.append(
            f"| {row['candidate']} | {row['seed']} | {row['best_epoch']} | `{row['checkpoint']['sha256']}` |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage6t_config_path = args.stage6t_config.resolve()
    stage6t_freeze_path = args.stage6t_freeze.resolve()
    implementation_freeze_path = args.implementation_freeze.resolve()
    training_authorization_path = args.training_authorization.resolve()
    checkpoint_ledger_path = args.checkpoint_ledger.resolve()
    confirmation_manifest_path = args.confirmation_manifest.resolve()
    confirmation_design_path = args.confirmation_design.resolve()
    confirmation_roster_path = args.confirmation_roster.resolve()

    stage6t_config = read_json(stage6t_config_path)
    stage6t_freeze = read_json(stage6t_freeze_path)
    implementation_freeze = read_json(implementation_freeze_path)
    training_authorization = read_json(training_authorization_path)
    confirmation_manifest = read_json(confirmation_manifest_path)

    protocol_fingerprint = str(stage6t_freeze.get("protocol_content_fingerprint_sha256", ""))
    if not protocol_fingerprint or stage6t_config.get("protocol_id") != stage6t_freeze.get("protocol_id"):
        raise ValueError("Stage6T protocol/freeze binding is invalid")
    implementation_freeze_sha = sha256_file(implementation_freeze_path)
    if implementation_freeze.get("status") != "FROZEN_READY_FOR_ABC_FORMAL_TRAINING":
        raise ValueError("Stage6U implementation freeze is not final")
    training_authorization_sha = sha256_file(training_authorization_path)
    if training_authorization.get("status") != "AUTHORIZED_STAGE6U_ABC_FORMAL_TRAINING":
        raise ValueError("Formal training authorization status is invalid")
    if training_authorization.get("implementation_freeze_sha256") != implementation_freeze_sha:
        raise ValueError("Formal training authorization does not bind the final implementation freeze")
    ledger, checkpoints = validate_checkpoint_ledger(
        checkpoint_ledger_path,
        implementation_freeze_sha,
        training_authorization_sha,
    )
    if confirmation_manifest.get("status") != "CONFIRMATION_ROSTER_FROZEN_NOT_RUN":
        raise ValueError("Stage6S-v2 confirmation roster is not in the required blind frozen state")
    confirmation_records = {
        "manifest": require_sha(
            confirmation_manifest_path,
            "4ada675278e0e634da2f552f69abf04b73c9b0f90e114b3f05dda8eaaadf6472",
            "Stage6S-v2 confirmation manifest",
        ),
        "design": require_sha(
            confirmation_design_path,
            str(confirmation_manifest["confirmation_design_sha256"]),
            "Stage6S-v2 confirmation design",
        ),
        "roster": require_sha(
            confirmation_roster_path,
            str(confirmation_manifest["confirmation_roster_sha256"]),
            "Stage6S-v2 confirmation roster",
        ),
    }
    if int(confirmation_manifest.get("scenario_count", -1)) != 80:
        raise ValueError("Stage6S-v2 confirmation roster must contain exactly 80 scenarios")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output_dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "stage6v_blind_evaluation_authorization_v1",
        "status": AUTHORIZATION_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_scope": "ONE_TIME_WAYMO_TEST_STAGE6J_K_STAGE6P_STAGE6S_V2_BLIND_EVALUATION",
        "immutability_statement": IMMUTABILITY_STATEMENT,
        "retraining_or_protocol_change_from_evaluation_results_allowed": False,
        "training_authorized": False,
        "checkpoint_write_authorized": False,
        "waymo_dynamic_v2_test_authorized": True,
        "stage6j_k_existing_rollout_evaluation_authorized": True,
        "stage6p_existing_release_split_evaluation_authorized": True,
        "stage6s_v2_confirmation_rollout_authorized": True,
        "stage6s_v2_representation_evaluation_condition": "ONLY_IF_CONFIRMATION_MECHANISM_GATE_PASSES",
        "primary_seed": 3407,
        "secondary_seed_role": "seed_stability_only_no_model_or_seed_selection",
        "stage6t": {
            "protocol_id": stage6t_config["protocol_id"],
            "protocol_fingerprint_sha256": protocol_fingerprint,
            "config": {"path": str(stage6t_config_path), "sha256": sha256_file(stage6t_config_path)},
            "freeze": {"path": str(stage6t_freeze_path), "sha256": sha256_file(stage6t_freeze_path)},
        },
        "stage6u": {
            "implementation_freeze": {"path": str(implementation_freeze_path), "sha256": implementation_freeze_sha},
            "formal_training_authorization": {"path": str(training_authorization_path), "sha256": training_authorization_sha},
            "checkpoint_ledger": {"path": str(checkpoint_ledger_path), "sha256": sha256_file(checkpoint_ledger_path)},
            "checkpoint_ledger_status": ledger["status"],
        },
        "locked_best_checkpoints": checkpoints,
        "stage6s_v2_confirmation": {
            "scenario_count": 80,
            "distinct_log_count": int(confirmation_manifest["distinct_log_count"]),
            **confirmation_records,
        },
        "evaluation_sequence": [
            "Waymo Dynamic v2 test: old64 and A/B/C x 3 seeds; freeze results",
            "Stage6J/K existing-rollout longitudinal paired blind evaluation; freeze results",
            "Stage6P existing 800-pair/489-log/2400-split unpaired blind evaluation; freeze results",
            "Stage6S-v2 80-pair official rollout and trajectory mechanism gate without representation",
            "Only after mechanism pass: locked interaction representation evaluation and C neighbor-zero diagnostic",
            "Apply the pre-frozen final-model decision rule and stop",
        ],
        "prohibited_actions": [
            "retraining",
            "changing_seed_or_primary_seed",
            "changing_epoch_or_checkpoint",
            "changing_loss_or_architecture",
            "changing_stage6t_protocol",
            "changing_stage6j_k_p_statistical_methods",
            "changing_stage6s_v2_roster_planner_metrics_or_gates",
            "using_results_to_create_a_second_confirmation_on_the_same_data",
        ],
        "preauthorization_blind_state": {
            "waymo_test_read": False,
            "stage6j_k_p_new_checkpoint_evaluation_run": False,
            "stage6s_v2_confirmation_rollout_run": False,
            "stage6s_v2_confirmation_embedding_or_bdd_read": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
    }
    manifest_path = output_dir / "stage6v_blind_evaluation_authorization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "stage6v_blind_evaluation_authorization_report_zh.md").write_text(
        build_report(manifest), encoding="utf-8"
    )
    result = {
        "status": AUTHORIZATION_STATUS,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_count": len(checkpoints),
        "primary_seed": 3407,
        "waymo_test_or_nuplan_result_read_during_authorization": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage6t_config", type=Path, default=Path("configs/stage6t_training_evaluation_protocol.json"))
    parser.add_argument("--stage6t_freeze", type=Path, default=Path("outputs/stage6t_training_evaluation_protocol_freeze_v1/stage6t_training_evaluation_protocol_freeze_manifest.json"))
    parser.add_argument("--implementation_freeze", type=Path, default=Path("outputs/stage6u_trainer_implementation_freeze_v2_preformal/stage6u_trainer_implementation_freeze_manifest.json"))
    parser.add_argument("--training_authorization", type=Path, default=Path("outputs/stage6u_formal_training_authorization_v1/stage6u_formal_training_authorization_manifest.json"))
    parser.add_argument("--checkpoint_ledger", type=Path, default=Path("outputs/stage6u_abc_formal_training_v1/checkpoint_lock/stage6u_formal_checkpoint_ledger.json"))
    parser.add_argument("--confirmation_manifest", type=Path, default=Path("outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_freeze_manifest.json"))
    parser.add_argument("--confirmation_design", type=Path, default=Path("outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_frozen_design.json"))
    parser.add_argument("--confirmation_roster", type=Path, default=Path("outputs/stage6s_v2_confirmation_freeze_v1/stage6s_v2_confirmation_roster.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/stage6v_blind_evaluation_authorization_v1"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
