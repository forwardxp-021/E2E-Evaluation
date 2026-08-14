#!/usr/bin/env python3
"""Run the nine authorized Stage6U tasks serially on one MPS device."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6u_unified_abc_trainer import read_json, resolve_repo_path, sha256_file  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def active_formal_training_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "ax", "-o", "pid=,command="], check=True, text=True, stdout=subprocess.PIPE
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "stage6u_unified_abc_trainer.py" in line and "--mode formal" in line
    ]


def validate_authorized_sources(authorization: dict[str, Any]) -> None:
    for name in ("trainer", "stage6u_config", "serial_orchestrator", "checkpoint_locker"):
        record = authorization.get("source_records", {}).get(name)
        if not record:
            raise ValueError(f"Authorization is missing source record: {name}")
        path = Path(record["path"])
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(
                f"Authorized source changed after authorization: {name}, "
                f"expected={record['sha256']}, actual={actual}"
            )


def validate_completed_summary(path: Path, candidate: str, seed: int, authorization_sha: str) -> dict[str, Any]:
    summary = read_json(path)
    checks = {
        "candidate": summary.get("candidate") == candidate,
        "seed": int(summary.get("seed", -1)) == seed,
        "complete": summary.get("training_complete") is True,
        "authorization": summary.get("authorization_manifest_sha256") == authorization_sha,
        "waymo_test": summary.get("waymo_test_read") is False,
        "stage6j_k_p": summary.get("stage6j_k_p_read_or_run") is False,
        "nuplan": summary.get("nuplan_read_or_run") is False,
        "stage6s_v2_confirmation": summary.get("stage6s_v2_confirmation_read_or_run") is False,
        "bdd_mmd": summary.get("embedding_bdd_mmd_read") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Completed summary failed validation: {checks}, path={path}")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    authorization_path = args.authorization_manifest.resolve()
    authorization = read_json(authorization_path)
    authorization_sha = sha256_file(authorization_path)
    if authorization.get("status") != "AUTHORIZED_STAGE6U_ABC_FORMAL_TRAINING":
        raise ValueError("Formal authorization status is invalid")
    if authorization.get("single_device_serial_execution") is not True:
        raise ValueError("Authorization does not require serial execution")
    validate_authorized_sources(authorization)
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "stage6u_formal_training_state.json"
    lock_path = run_dir / "orchestrator.pid"
    if lock_path.is_file():
        previous_pid = int(lock_path.read_text(encoding="utf-8").strip())
        if pid_alive(previous_pid):
            raise RuntimeError(f"Another Stage6U orchestrator is active: pid={previous_pid}")
    active = active_formal_training_processes()
    if active:
        raise RuntimeError(f"A formal trainer is already active; refusing concurrency: {active}")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    if state_path.is_file():
        state = read_json(state_path)
        if state.get("authorization_manifest_sha256") != authorization_sha:
            raise ValueError("Existing orchestrator state has a different authorization SHA")
    else:
        state = {
            "schema_version": "stage6u_formal_training_orchestrator_state_v1",
            "status": "RUNNING_ABC_FORMAL_TRAINING",
            "started_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
            "orchestrator_pid": os.getpid(),
            "authorization_manifest_path": str(authorization_path),
            "authorization_manifest_sha256": authorization_sha,
            "implementation_freeze_sha256": authorization["implementation_freeze_sha256"],
            "single_device_serial_execution": True,
            "tasks": [
                {
                    **row,
                    "status": "PENDING",
                    "attempts": [],
                    "summary_path": str(resolve_repo_path(row["output_dir"]) / "formal_training_summary.json"),
                }
                for row in authorization["training_order"]
            ],
            "completed_tasks": 0,
            "current_task": None,
            "waymo_test_read": False,
            "nuplan_read_or_run": False,
            "embedding_bdd_mmd_read": False,
        }
    state["orchestrator_pid"] = os.getpid()
    state["status"] = "RUNNING_ABC_FORMAL_TRAINING"
    atomic_json(state_path, state)
    try:
        for task in state["tasks"]:
            validate_authorized_sources(authorization)
            candidate = str(task["candidate"])
            seed = int(task["seed"])
            output_dir = resolve_repo_path(task["output_dir"])
            summary_path = output_dir / "formal_training_summary.json"
            if summary_path.is_file():
                validate_completed_summary(summary_path, candidate, seed, authorization_sha)
                task["status"] = "COMPLETED"
                continue
            resume_path = output_dir / "resume_model.pt"
            resume = output_dir.is_dir()
            if resume and not resume_path.is_file():
                raise FileNotFoundError(f"Incomplete formal output has no resume checkpoint: {output_dir}")
            active = active_formal_training_processes()
            if active:
                raise RuntimeError(f"A formal trainer is already active; refusing concurrency: {active}")
            log_path = run_dir / "logs" / f"{int(task['order']):02d}_{candidate}_{seed}.log"
            command = [
                sys.executable,
                str(REPO_ROOT / "tools/stage6u_unified_abc_trainer.py"),
                "--config",
                str(REPO_ROOT / "configs/stage6u_unified_abc_trainer.json"),
                "--candidate",
                candidate,
                "--mode",
                "formal",
                "--seed",
                str(seed),
                "--output_dir",
                str(output_dir),
                "--authorization_manifest",
                str(authorization_path),
                "--implementation_freeze_sha256",
                authorization["implementation_freeze_sha256"],
            ]
            if resume:
                command.extend(["--resume_checkpoint", str(resume_path)])
            attempt = {
                "started_at_utc": utc_now_iso(),
                "resume": resume,
                "resume_checkpoint": str(resume_path) if resume else None,
                "log_path": str(log_path),
                "command": command,
            }
            task["attempts"].append(attempt)
            task["status"] = "RUNNING"
            state["current_task"] = {"order": task["order"], "candidate": candidate, "seed": seed}
            state["updated_at_utc"] = utc_now_iso()
            atomic_json(state_path, state)
            with log_path.open("a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                attempt["trainer_pid"] = process.pid
                state["current_trainer_pid"] = process.pid
                state["updated_at_utc"] = utc_now_iso()
                atomic_json(state_path, state)
                exit_code = process.wait()
            attempt["ended_at_utc"] = utc_now_iso()
            attempt["exit_code"] = exit_code
            state["current_trainer_pid"] = None
            if exit_code != 0:
                task["status"] = "FAILED_RESUMABLE" if resume_path.is_file() else "FAILED"
                state["status"] = "STOPPED_ON_TRAINING_FAILURE"
                state["updated_at_utc"] = utc_now_iso()
                atomic_json(state_path, state)
                raise RuntimeError(f"Formal task {candidate}/{seed} failed with exit code {exit_code}")
            validate_completed_summary(summary_path, candidate, seed, authorization_sha)
            task["status"] = "COMPLETED"
            state["completed_tasks"] = sum(row["status"] == "COMPLETED" for row in state["tasks"])
            state["current_task"] = None
            state["updated_at_utc"] = utc_now_iso()
            atomic_json(state_path, state)
            time.sleep(1)

        validate_authorized_sources(authorization)
        lock_dir = run_dir / "checkpoint_lock"
        lock_command = [
            sys.executable,
            str(REPO_ROOT / "tools/stage6u_lock_formal_checkpoints.py"),
            "--authorization_manifest",
            str(authorization_path),
            "--output_dir",
            str(lock_dir),
        ]
        subprocess.run(lock_command, cwd=REPO_ROOT, check=True)
        ledger_path = lock_dir / "stage6u_formal_checkpoint_ledger.json"
        ledger = read_json(ledger_path)
        state["status"] = ledger["status"]
        state["completed_tasks"] = 9
        state["current_task"] = None
        state["completed_at_utc"] = utc_now_iso()
        state["checkpoint_ledger_path"] = str(ledger_path)
        state["checkpoint_ledger_sha256"] = sha256_file(ledger_path)
        state["updated_at_utc"] = utc_now_iso()
        atomic_json(state_path, state)
        return state
    except KeyboardInterrupt:
        state["status"] = "INTERRUPTED_RESUMABLE"
        state["updated_at_utc"] = utc_now_iso()
        atomic_json(state_path, state)
        raise
    finally:
        if lock_path.is_file() and lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization_manifest", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "completed_tasks": result["completed_tasks"]}, indent=2))
