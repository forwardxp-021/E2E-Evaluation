#!/usr/bin/env python3
"""Read-only status and ETA monitor for Stage6U formal A/B/C training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage6u_unified_abc_trainer import read_json, resolve_repo_path  # noqa: E402


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def hours_text(hours: float | None) -> str:
    if hours is None:
        return "尚无法估算"
    if hours < 1:
        return f"约{hours * 60:.0f}分钟"
    return f"约{hours:.1f}小时"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def read_epoch_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_last_heartbeat(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "stage6u_formal_training_state.json"
    state = read_json(state_path)
    authorization = read_json(Path(state["authorization_manifest_path"]))
    freeze = read_json(Path(authorization["implementation_freeze_manifest_path"]))
    timing = freeze["training_time_estimate"]["per_seed_max30_hours"]
    completed = sum(row["status"] == "COMPLETED" for row in state["tasks"])
    current = state.get("current_task")
    current_detail = None
    measured_hours: dict[str, float] = {}
    for task in state["tasks"]:
        root = resolve_repo_path(task["output_dir"])
        epochs = read_epoch_rows(root / "train_log.csv")
        if epochs:
            elapsed = float(epochs[-1]["elapsed_seconds"])
            completed_epochs = int(epochs[-1]["epoch"])
            if completed_epochs > 0:
                measured_hours.setdefault(str(task["candidate"]), elapsed / completed_epochs / 3600.0)
        if current and int(task["order"]) == int(current["order"]):
            heartbeat = read_last_heartbeat(root / "progress.jsonl")
            current_detail = {
                "candidate": task["candidate"],
                "seed": int(task["seed"]),
                "completed_epochs": int(epochs[-1]["epoch"]) if epochs else 0,
                "last_epoch_val_loss": float(epochs[-1]["val_loss"]) if epochs else None,
                "last_heartbeat": heartbeat,
                "trainer_pid": state.get("current_trainer_pid"),
                "trainer_process_alive": pid_alive(state.get("current_trainer_pid")),
            }
    remaining_hours = 0.0
    for task in state["tasks"]:
        if task["status"] == "COMPLETED":
            continue
        candidate = str(task["candidate"])
        root = resolve_repo_path(task["output_dir"])
        epochs = read_epoch_rows(root / "train_log.csv")
        completed_epochs = int(epochs[-1]["epoch"]) if epochs else 0
        if candidate in measured_hours:
            remaining_hours += measured_hours[candidate] * max(30 - completed_epochs, 0)
        else:
            remaining_hours += float(timing[candidate]) * max(30 - completed_epochs, 0) / 30.0
    now = datetime.now(timezone.utc)
    eta = now.timestamp() + remaining_hours * 3600.0
    status = {
        "checked_at_utc": now.isoformat(),
        "status": state["status"],
        "orchestrator_pid": state.get("orchestrator_pid"),
        "orchestrator_alive": pid_alive(state.get("orchestrator_pid")),
        "completed_tasks": completed,
        "total_tasks": 9,
        "completion_percent": completed / 9.0 * 100.0,
        "current_task": current_detail,
        "measured_epoch_hours_by_candidate": measured_hours,
        "estimated_remaining_hours_max30": remaining_hours,
        "estimated_finish_local": datetime.fromtimestamp(eta).astimezone().isoformat(),
        "blind_boundary": {
            "waymo_test_read": state.get("waymo_test_read", False),
            "nuplan_read_or_run": state.get("nuplan_read_or_run", False),
            "embedding_bdd_mmd_read": state.get("embedding_bdd_mmd_read", False),
        },
    }
    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        current_text = "当前无运行任务"
        if current_detail:
            heartbeat = current_detail["last_heartbeat"] or {}
            current_text = (
                f"当前 {current_detail['candidate']}/{current_detail['seed']}，已完成"
                f"{current_detail['completed_epochs']}个epoch"
            )
            if heartbeat:
                current_text += (
                    f"，正在epoch {int(heartbeat.get('epoch', 0)) + 1}、"
                    f"batch {int(heartbeat.get('batch_index', 0)) + 1}、step {heartbeat.get('global_step')}"
                )
        print(
            f"Stage6U状态：{status['status']}；9任务完成 {completed}/9（{status['completion_percent']:.1f}%）。"
            f"{current_text}。保守剩余{hours_text(remaining_hours)}，预计本地完成时间"
            f"{status['estimated_finish_local']}。"
        )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
