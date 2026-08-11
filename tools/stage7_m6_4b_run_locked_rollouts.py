#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7c1_run_nuplan_simulation import PLANNER_PROFILES


SCHEMA_VERSION = "stage7_m6_4b_locked_batch_v1"
READY_STATUS = "FROZEN_BEFORE_LOCKED_ROLLOUTS"
EXPECTED_PLANNERS = ["pdm_closed_assertive_v1", "pdm_closed_conservative_v1"]
PRIMARY_FIELDS = [
    "collection_order",
    "task",
    "task_rank",
    "log_name",
    "scenario_token",
    "scene_token",
    "scenario_type",
    "db_file",
    "db_scene_token",
    "scenario_tag_token",
    "selection_role",
    "selection_salt",
]
STATUS_FIELDS = [
    "collection_order",
    "task",
    "task_rank",
    "log_name",
    "scenario_token",
    "db_file",
    "status",
    "failure_category",
    "attempt",
    "return_code",
    "official_success_count",
    "trajectory_rows",
    "same_log_alignment_passed",
    "strict_token_alignment_passed",
    "started_utc",
    "ended_utc",
    "duration_seconds",
    "stage7c_output_dir",
]
RESERVE_ELIGIBLE_FAILURES = {
    "OFFICIAL_COMMAND_FAILED",
    "COMMAND_TIMEOUT",
    "TRAJECTORY_EXPORT_FAILED",
    "INCOMPLETE_PLANNER_PAIR",
    "ALIGNMENT_FAILED",
    "INVALID_STAGE7C_OUTPUT",
    "QUALITY_GATE_FAILED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(PRIMARY_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing locked collection columns: {missing}")
        return [{key: str(value or "") for key, value in row.items()} for row in reader]


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, payload: Any) -> None:
    next_path = path.with_name(path.name + ".next")
    next_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(next_path, path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_rows_hash(rows: Sequence[Mapping[str, str]]) -> str:
    canonical = [
        {
            "collection_order": int(row["collection_order"]),
            "task": row["task"],
            "task_rank": int(row["task_rank"]),
            "log_name": row["log_name"],
            "scenario_token": row["scenario_token"],
            "scenario_type": row["scenario_type"],
            "db_file": row["db_file"],
            "selection_role": row["selection_role"],
        }
        for row in rows
    ]
    return canonical_hash(canonical)


def current_planner_fingerprints(planners: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for planner in planners:
        if planner not in PLANNER_PROFILES:
            raise ValueError(f"planner is not registered in frozen Stage7C: {planner}")
        canonical = json.dumps(
            PLANNER_PROFILES[planner]["parameters"],
            sort_keys=True,
            separators=(",", ":"),
        )
        result[planner] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def resolve_git_commit(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"cannot resolve git commit for {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def validate_locked_inputs(
    *,
    manifest_path: Path,
    primary_csv: Path,
    reserve_csv: Path,
    stage7c_tool: Path,
    db_root: Path,
    nuplan_devkit_root: Path,
    tuplan_garage_root: Path,
    expected_nuplan_commit: str,
    expected_tuplan_commit: str,
    commit_resolver: Callable[[Path], str] = resolve_git_commit,
) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != READY_STATUS:
        raise ValueError(f"locked manifest status is not {READY_STATUS}")
    if manifest.get("ready_to_launch_locked_rollouts") is not True:
        raise ValueError("locked manifest ready_to_launch_locked_rollouts is not true")
    planners = list(manifest.get("planners", []))
    if planners != EXPECTED_PLANNERS:
        raise ValueError(f"unexpected locked planner order: {planners}")
    expected_hashes = {
        primary_csv: manifest.get("primary_collection_csv_sha256"),
        reserve_csv: manifest.get("reserve_collection_csv_sha256"),
        stage7c_tool: manifest.get("stage7c_tool_sha256"),
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected={expected}, actual={actual}")

    primary = read_csv(primary_csv)
    reserve = read_csv(reserve_csv)
    if len(primary) != int(manifest.get("planned_primary_scenarios", -1)):
        raise ValueError("primary row count differs from locked manifest")
    if len(primary) * len(planners) != int(manifest.get("planned_primary_rollouts", -1)):
        raise ValueError("planned primary rollout count differs from locked manifest")
    if len(reserve) != int(manifest.get("maximum_reserve_scenarios", -1)):
        raise ValueError("reserve row count differs from locked manifest")
    if canonical_rows_hash(primary) != manifest.get("primary_manifest_sha256"):
        raise ValueError("primary canonical manifest hash mismatch")
    if canonical_rows_hash(reserve) != manifest.get("reserve_manifest_sha256"):
        raise ValueError("reserve canonical manifest hash mismatch")

    all_tokens: set = set()
    for label, rows, expected_role in [
        ("primary", primary, "primary_gross"),
        ("reserve", reserve, "technical_quality_reserve"),
    ]:
        expected_orders = list(range(1, len(rows) + 1))
        orders = [int(row["collection_order"]) for row in rows]
        if orders != expected_orders:
            raise ValueError(f"{label} collection_order must be contiguous and frozen")
        ranks: Dict[str, List[int]] = defaultdict(list)
        for row in rows:
            token = row["scenario_token"].strip()
            if not token or token in all_tokens:
                raise ValueError(f"duplicate or empty scenario token across collections: {token}")
            all_tokens.add(token)
            if row["selection_role"] != expected_role:
                raise ValueError(f"unexpected {label} selection_role: {row['selection_role']}")
            ranks[row["task"]].append(int(row["task_rank"]))
            db_file = row["db_file"].strip()
            if Path(db_file).name != db_file:
                raise ValueError(f"db_file must be a basename: {db_file}")
            if not (db_root / db_file).is_file():
                raise FileNotFoundError(f"locked DB file is missing: {db_root / db_file}")
        for task, values in ranks.items():
            if values != list(range(1, len(values) + 1)):
                raise ValueError(f"{label} task_rank is not contiguous for task {task}")

    primary_counts = dict(Counter(row["task"] for row in primary))
    reserve_counts = dict(Counter(row["task"] for row in reserve))
    if primary_counts != manifest.get("primary_selected_by_task"):
        raise ValueError("primary task counts differ from locked manifest")
    if reserve_counts != manifest.get("reserve_selected_by_task"):
        raise ValueError("reserve task counts differ from locked manifest")
    salts = {row["selection_salt"] for row in primary + reserve}
    if len(salts) != 1 or next(iter(salts)) != manifest.get("selection_salt"):
        raise ValueError("selection salt differs within collection or from locked manifest")

    fingerprints = current_planner_fingerprints(planners)
    if fingerprints != manifest.get("planner_parameter_fingerprints"):
        raise ValueError("current Stage7C planner fingerprints differ from locked manifest")
    nuplan_commit = commit_resolver(nuplan_devkit_root)
    tuplan_commit = commit_resolver(tuplan_garage_root)
    if nuplan_commit != expected_nuplan_commit:
        raise ValueError(
            f"nuPlan commit mismatch: expected={expected_nuplan_commit}, actual={nuplan_commit}"
        )
    if tuplan_commit != expected_tuplan_commit:
        raise ValueError(
            f"tuPlan Garage commit mismatch: expected={expected_tuplan_commit}, actual={tuplan_commit}"
        )
    required_paths = [
        nuplan_devkit_root / "nuplan/planning/script/run_simulation.py",
        tuplan_garage_root / "tuplan_garage/planning/simulation/planner/pdm_planner/pdm_closed_planner.py",
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required external planner path is missing: {path}")
    audit = {
        "manifest_sha256": sha256_file(manifest_path),
        "primary_csv_sha256": sha256_file(primary_csv),
        "reserve_csv_sha256": sha256_file(reserve_csv),
        "stage7c_tool_sha256": sha256_file(stage7c_tool),
        "planner_parameter_fingerprints": fingerprints,
        "nuplan_devkit_commit": nuplan_commit,
        "tuplan_garage_commit": tuplan_commit,
    }
    return manifest, primary, reserve, audit


class BatchLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> "BatchLock":
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            detail = self.path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError(f"batch lock already exists: {self.path}; owner={detail}") from exc
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_utc": utc_now(),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()


def invocation_selection(
    rows: Sequence[Dict[str, str]], start_order: int, end_order: int, max_scenarios: int
) -> List[Dict[str, str]]:
    selected = [
        row
        for row in rows
        if int(row["collection_order"]) >= start_order
        and (end_order == 0 or int(row["collection_order"]) <= end_order)
    ]
    if max_scenarios > 0:
        selected = selected[:max_scenarios]
    return selected


def scenario_slug(row: Mapping[str, str]) -> str:
    return f"order_{int(row['collection_order']):04d}_{row['scenario_token']}"


def successful_attempts(scenario_dir: Path, planners: Sequence[str], row: Mapping[str, str]) -> List[Tuple[Path, Dict[str, Any]]]:
    valid: List[Tuple[Path, Dict[str, Any]]] = []
    if not scenario_dir.is_dir():
        return valid
    for attempt in sorted(scenario_dir.glob("attempt_*")):
        output = attempt / "stage7c_output"
        audit = audit_stage7c_output(output, planners, row)
        if audit["pass"]:
            valid.append((attempt, audit))
    return valid


def audit_stage7c_output(
    output_dir: Path, planners: Sequence[str], row: Mapping[str, str]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "pass": False,
        "failure_category": "INVALID_STAGE7C_OUTPUT",
        "official_success_count": 0,
        "trajectory_rows": 0,
        "same_log_alignment_passed": False,
        "strict_token_alignment_passed": False,
    }
    required = [
        output_dir / "warnings.json",
        output_dir / "simulation_schema.json",
        output_dir / "scenario_planner_index.csv",
        output_dir / "scenario_alignment.csv",
        output_dir / "simulated_ego_seq.npy",
        output_dir / "simulated_ego_seq_mask.npy",
    ]
    if not all(path.is_file() for path in required):
        return result
    try:
        warnings = read_json(output_dir / "warnings.json")
        schema = read_json(output_dir / "simulation_schema.json")
        validation = warnings.get("validation", {})
        tensor = validation.get("tensor_validation", {})
        result["official_success_count"] = int(validation.get("official_success_count", 0))
        result["trajectory_rows"] = int(validation.get("trajectory_rows", 0))
        result["same_log_alignment_passed"] = bool(schema.get("same_log_alignment_passed"))
        result["strict_token_alignment_passed"] = bool(
            schema.get("strict_nuplan_token_alignment_passed")
        )
        with (output_dir / "scenario_planner_index.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            index_rows = list(csv.DictReader(handle))
        with (output_dir / "scenario_alignment.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            alignment_rows = list(csv.DictReader(handle))
        index_planners = [item.get("planner_name", "") for item in index_rows]
        all_index_success = len(index_rows) == len(planners) and set(index_planners) == set(planners) and all(
            item.get("status") == "succeeded" for item in index_rows
        )
        identity_ok = len(alignment_rows) == len(planners) and all(
            item.get("target_log_name") == row["log_name"]
            and item.get("actual_log_name") == row["log_name"]
            and item.get("target_nuplan_scenario_token") == row["scenario_token"]
            and item.get("actual_nuplan_scenario_token") == row["scenario_token"]
            for item in alignment_rows
        )
        shape = schema.get("simulated_ego_seq_shape", [])
        shape_ok = (
            isinstance(shape, list)
            and len(shape) == 4
            and shape[0] == 1
            and shape[1] == len(planners)
            and shape[2] >= 2
            and shape[3] == 8
        )
        pair_ok = (
            tensor.get("expected_pair_count") == len(planners)
            and tensor.get("observed_pair_count") == len(planners)
            and tensor.get("missing_pair_count") == 0
            and tensor.get("passed") is True
        )
        result["pass"] = bool(
            validation.get("pass") is True
            and validation.get("pseudo_rollout") is False
            and result["official_success_count"] == len(planners)
            and result["trajectory_rows"] > 0
            and schema.get("planner_names") == list(planners)
            and schema.get("pseudo_rollout") is False
            and result["same_log_alignment_passed"]
            and result["strict_token_alignment_passed"]
            and all_index_success
            and identity_ok
            and shape_ok
            and pair_ok
        )
        if result["pass"]:
            result["failure_category"] = ""
        elif result["official_success_count"] < len(planners):
            result["failure_category"] = "OFFICIAL_COMMAND_FAILED"
        elif result["trajectory_rows"] <= 0:
            result["failure_category"] = "TRAJECTORY_EXPORT_FAILED"
        elif not pair_ok:
            result["failure_category"] = "INCOMPLETE_PLANNER_PAIR"
        elif not (result["same_log_alignment_passed"] and result["strict_token_alignment_passed"] and identity_ok):
            result["failure_category"] = "ALIGNMENT_FAILED"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return result
    return result


def classify_process_failure(return_code: int, log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    environment_markers = [
        "ModuleNotFoundError",
        "ImportError",
        "Could not override",
        "Cannot find primary config",
        "No module named",
    ]
    if any(marker in text for marker in environment_markers):
        return "ENVIRONMENT_OR_CONFIG_FAILURE"
    if return_code == 124:
        return "COMMAND_TIMEOUT"
    return "OFFICIAL_COMMAND_FAILED"


def build_command_template(args: argparse.Namespace, db_path: Path) -> str:
    python = shlex.quote(str(args.python_executable.resolve()))
    run_simulation = shlex.quote(
        str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve())
    )
    db_override = f"scenario_builder.db_files=[{db_path.resolve()}]"
    return " ".join(
        [
            python,
            run_simulation,
            "+simulation=closed_loop_nonreactive_agents",
            "{planner_hydra_overrides}",
            "scenario_builder=nuplan_mini",
            db_override,
            "scenario_filter=all_scenarios",
            "{scenario_hydra_overrides}",
            "worker=single_machine_thread_pool",
            "worker.max_workers=1",
            "scenario_builder.max_workers=1",
            "max_callback_workers=1",
            "gpu=false",
            "experiment_name=stage7_m6_4b_locked_primary_mac_v1",
            "job_name=closed_loop_nonreactive_agents_stage7c_{planner_name_safe}",
            "output_dir='{output_dir}'",
        ]
    )


def stage7c_environment(args: argparse.Namespace) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "NUPLAN_DEVKIT_ROOT": str(args.nuplan_devkit_root.resolve()),
            "NUPLAN_DATA_ROOT": str(args.nuplan_data_root.resolve()),
            "NUPLAN_MAPS_ROOT": str(args.nuplan_map_root.resolve()),
            "NUPLAN_MAP_ROOT": str(args.nuplan_map_root.resolve()),
            "NUPLAN_EXP_ROOT": str(args.nuplan_exp_root.resolve()),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    roots = [
        str(args.nuplan_devkit_root.resolve()),
        str(args.tuplan_garage_root.resolve()),
        str(Path(__file__).resolve().parents[1]),
    ]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return env


def run_stage7c(args: argparse.Namespace, row: Mapping[str, str], attempt_dir: Path) -> int:
    context_dir = attempt_dir / "context"
    output_dir = attempt_dir / "stage7c_output"
    context_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    write_csv(context_dir / "merged_metadata.csv", [row], PRIMARY_FIELDS)
    db_path = args.nuplan_db_root / row["db_file"]
    command_template = build_command_template(args, db_path)
    command = [
        str(args.python_executable.resolve()),
        str(args.stage7c_tool.resolve()),
        "--context_dir",
        str(context_dir.resolve()),
        "--nuplan_db_root",
        str(args.nuplan_db_root.resolve()),
        "--nuplan_map_root",
        str(args.nuplan_map_root.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--planners",
        *EXPECTED_PLANNERS,
        "--max_scenarios",
        "1",
        "--min_timesteps",
        "2",
        "--require_same_scenario_alignment",
        "--require_strict_nuplan_token_alignment",
        "--allow_external_planner_name",
        "--hydra_searchpath",
        "[pkg://tuplan_garage.planning.script.config.common,pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
        "--command_timeout_s",
        str(args.command_timeout_s),
        "--nuplan_simulation_command_template",
        command_template,
    ]
    log_path = attempt_dir / "stage7c_driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("argv: " + json.dumps(command, ensure_ascii=False) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=stage7c_environment(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(proc.wait())


def status_from_existing(
    primary: Sequence[Dict[str, str]], output_dir: Path, planners: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in primary:
        scenario_dir = output_dir / "rollouts" / scenario_slug(source)
        attempts = sorted(scenario_dir.glob("attempt_*")) if scenario_dir.is_dir() else []
        successes = successful_attempts(scenario_dir, planners, source)
        status = "SUCCEEDED" if successes else ("FAILED_REVIEW_REQUIRED" if attempts else "PENDING")
        summary: Dict[str, Any] = {}
        if attempts and (attempts[-1] / "attempt_summary.json").is_file():
            try:
                summary = read_json(attempts[-1] / "attempt_summary.json")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                summary = {}
        if successes:
            audit = {**summary, **successes[-1][1]}
        elif attempts:
            audit = summary or audit_stage7c_output(
                attempts[-1] / "stage7c_output", planners, source
            )
        else:
            audit = {}
        rows.append(
            {
                **source,
                "status": status,
                "failure_category": audit.get("failure_category", ""),
                "attempt": len(attempts),
                "return_code": audit.get("return_code", ""),
                "official_success_count": audit.get("official_success_count", 0),
                "trajectory_rows": audit.get("trajectory_rows", 0),
                "same_log_alignment_passed": audit.get("same_log_alignment_passed", False),
                "strict_token_alignment_passed": audit.get("strict_token_alignment_passed", False),
                "started_utc": audit.get("started_utc", ""),
                "ended_utc": audit.get("ended_utc", ""),
                "duration_seconds": audit.get("duration_seconds", ""),
                "stage7c_output_dir": str(successes[-1][0] / "stage7c_output") if successes else (
                    str(attempts[-1] / "stage7c_output") if attempts else ""
                ),
            }
        )
    return rows


def reserve_proposal(
    status_rows: Sequence[Mapping[str, Any]], reserve: Sequence[Dict[str, str]]
) -> List[Dict[str, Any]]:
    failures_by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in status_rows:
        if row.get("failure_category") in RESERVE_ELIGIBLE_FAILURES:
            failures_by_task[str(row["task"])].append(row)
    reserves_by_task: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in reserve:
        reserves_by_task[row["task"]].append(row)
    proposal: List[Dict[str, Any]] = []
    for task in sorted(failures_by_task):
        failures = sorted(failures_by_task[task], key=lambda item: int(item["collection_order"]))
        candidates = sorted(reserves_by_task.get(task, []), key=lambda item: int(item["task_rank"]))
        for index, failure in enumerate(failures):
            if index >= len(candidates):
                break
            replacement = candidates[index]
            proposal.append(
                {
                    "proposal_order": len(proposal) + 1,
                    "task": task,
                    "failed_primary_order": failure["collection_order"],
                    "failed_primary_token": failure["scenario_token"],
                    "failure_category": failure["failure_category"],
                    "reserve_collection_order": replacement["collection_order"],
                    "reserve_task_rank": replacement["task_rank"],
                    "reserve_token": replacement["scenario_token"],
                    "reserve_log_name": replacement["log_name"],
                    "approval_status": "PROPOSED_NOT_APPROVED_NOT_EXECUTED",
                }
            )
    return proposal


def write_batch_outputs(
    args: argparse.Namespace,
    primary: Sequence[Dict[str, str]],
    reserve: Sequence[Dict[str, str]],
    audit: Mapping[str, Any],
    selected_orders: Sequence[int],
) -> Dict[str, Any]:
    status_rows = status_from_existing(primary, args.output_dir, EXPECTED_PLANNERS)
    write_csv(args.output_dir / "batch_scenario_status.csv", status_rows, STATUS_FIELDS)
    proposal = reserve_proposal(status_rows, reserve)
    proposal_fields = [
        "proposal_order",
        "task",
        "failed_primary_order",
        "failed_primary_token",
        "failure_category",
        "reserve_collection_order",
        "reserve_task_rank",
        "reserve_token",
        "reserve_log_name",
        "approval_status",
    ]
    write_csv(args.output_dir / "reserve_replacement_proposal.csv", proposal, proposal_fields)
    counts = Counter(str(row["status"]) for row in status_rows)
    state = {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": utc_now(),
        "execution_mode": "execute" if args.execute else "dry_run",
        "batch_status": (
            "COMPLETE_PRIMARY"
            if counts["SUCCEEDED"] == len(primary)
            else ("PARTIAL_WITH_FAILURES" if counts["FAILED_REVIEW_REQUIRED"] else "PARTIAL_OR_PLANNED")
        ),
        "total_primary_scenarios": len(primary),
        "total_primary_rollouts": len(primary) * len(EXPECTED_PLANNERS),
        "selected_orders_this_invocation": list(selected_orders),
        "status_counts": dict(sorted(counts.items())),
        "reserve_proposals": len(proposal),
        "reserve_execution_enabled": False,
        "frozen_input_audit": dict(audit),
    }
    atomic_write_json(args.output_dir / "batch_state.json", state)
    report = [
        "# Stage 7 M6.4B Locked Rollout Batch Report",
        "",
        f"- mode: `{state['execution_mode']}`",
        f"- batch status: `{state['batch_status']}`",
        f"- primary scenarios: `{len(primary)}`",
        f"- primary rollouts: `{len(primary) * len(EXPECTED_PLANNERS)}`",
        f"- succeeded scenarios: `{counts['SUCCEEDED']}`",
        f"- failed review required: `{counts['FAILED_REVIEW_REQUIRED']}`",
        f"- pending: `{counts['PENDING']}`",
        f"- reserve proposals: `{len(proposal)}`",
        "",
        "Reserve proposals are audit-only. This tool does not execute reserve rows.",
        "No embedding, BDD, effect size, or planner outcome is used for stopping or selection.",
    ]
    (args.output_dir / "batch_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return state


def immutable_batch_manifest(args: argparse.Namespace, audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_tool_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "primary_csv": str(args.primary_csv.resolve()),
        "reserve_csv": str(args.reserve_csv.resolve()),
        "stage7c_tool": str(args.stage7c_tool.resolve()),
        "nuplan_db_root": str(args.nuplan_db_root.resolve()),
        "nuplan_map_root": str(args.nuplan_map_root.resolve()),
        "nuplan_data_root": str(args.nuplan_data_root.resolve()),
        "nuplan_exp_root": str(args.nuplan_exp_root.resolve()),
        "python_executable": str(args.python_executable.resolve()),
        "command_timeout_s": int(args.command_timeout_s),
        "planners": EXPECTED_PLANNERS,
        "frozen_input_audit": dict(audit),
    }


def prepare_output_dir(args: argparse.Namespace, batch_manifest: Mapping[str, Any]) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(f"output_dir is not empty; use --resume: {args.output_dir}")
        existing = read_json(args.output_dir / "batch_manifest.json")
        if existing != batch_manifest:
            raise ValueError("resume batch_manifest differs from current immutable configuration")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "batch_manifest.json", batch_manifest)


def run(args: argparse.Namespace) -> int:
    manifest, primary, reserve, audit = validate_locked_inputs(
        manifest_path=args.manifest_path,
        primary_csv=args.primary_csv,
        reserve_csv=args.reserve_csv,
        stage7c_tool=args.stage7c_tool,
        db_root=args.nuplan_db_root,
        nuplan_devkit_root=args.nuplan_devkit_root,
        tuplan_garage_root=args.tuplan_garage_root,
        expected_nuplan_commit=args.expected_nuplan_commit,
        expected_tuplan_commit=args.expected_tuplan_commit,
    )
    batch_manifest = immutable_batch_manifest(args, audit)
    prepare_output_dir(args, batch_manifest)
    selected = invocation_selection(primary, args.start_order, args.end_order, args.max_scenarios)
    if not selected:
        raise ValueError("invocation selected zero primary scenarios")
    selected_orders = [int(row["collection_order"]) for row in selected]
    plan_rows = [
        {**row, "selected_this_invocation": int(row["collection_order"]) in selected_orders}
        for row in primary
    ]
    write_csv(args.output_dir / "batch_plan.csv", plan_rows, PRIMARY_FIELDS + ["selected_this_invocation"])
    if not args.execute:
        write_batch_outputs(args, primary, reserve, audit, selected_orders)
        print(json.dumps({"mode": "dry_run", "selected_scenarios": len(selected), "output_dir": str(args.output_dir)}, indent=2))
        return 0
    if args.confirm_primary_manifest_sha256 != manifest.get("primary_manifest_sha256"):
        raise ValueError("--confirm_primary_manifest_sha256 does not match locked manifest")
    args.nuplan_exp_root.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    with BatchLock(args.output_dir / "batch.lock"):
        for position, row in enumerate(selected, 1):
            scenario_dir = args.output_dir / "rollouts" / scenario_slug(row)
            successes = successful_attempts(scenario_dir, EXPECTED_PLANNERS, row)
            if successes:
                print(
                    f"[M6.4B batch] SKIP {position}/{len(selected)} order={row['collection_order']} "
                    f"token={row['scenario_token']} validated_success={successes[-1][0]}",
                    flush=True,
                )
                continue
            attempts = sorted(scenario_dir.glob("attempt_*")) if scenario_dir.is_dir() else []
            if attempts and not args.retry_failed:
                print(
                    f"[M6.4B batch] BLOCKED order={row['collection_order']} has failed/corrupt attempt; "
                    "inspect it and use --retry_failed to create a new attempt directory",
                    file=sys.stderr,
                )
                exit_code = 2
                continue
            attempt_number = len(attempts) + 1
            attempt_dir = scenario_dir / f"attempt_{attempt_number:03d}"
            attempt_dir.mkdir(parents=True)
            started = utc_now()
            start_time = time.monotonic()
            append_jsonl(
                args.output_dir / "batch_events.jsonl",
                {
                    "event": "attempt_started",
                    "time_utc": started,
                    "collection_order": int(row["collection_order"]),
                    "scenario_token": row["scenario_token"],
                    "attempt": attempt_number,
                },
            )
            print(
                f"[M6.4B batch] START {position}/{len(selected)} order={row['collection_order']} "
                f"task={row['task']} token={row['scenario_token']} attempt={attempt_number}",
                flush=True,
            )
            orchestration_error = ""
            try:
                return_code = run_stage7c(args, row, attempt_dir)
            except Exception as exc:
                return_code = 125
                orchestration_error = f"{type(exc).__name__}: {exc}"
                (attempt_dir / "orchestration_error.txt").write_text(
                    orchestration_error + "\n", encoding="utf-8"
                )
            duration = time.monotonic() - start_time
            output_audit = audit_stage7c_output(
                attempt_dir / "stage7c_output", EXPECTED_PLANNERS, row
            )
            if return_code != 0 and output_audit["failure_category"] == "INVALID_STAGE7C_OUTPUT":
                output_audit["failure_category"] = (
                    "ORCHESTRATION_FAILURE"
                    if orchestration_error
                    else classify_process_failure(return_code, attempt_dir / "stage7c_driver.log")
                )
            attempt_summary = {
                "schema_version": SCHEMA_VERSION,
                "collection_order": int(row["collection_order"]),
                "scenario_token": row["scenario_token"],
                "attempt": attempt_number,
                "started_utc": started,
                "ended_utc": utc_now(),
                "duration_seconds": duration,
                "return_code": return_code,
                "orchestration_error": orchestration_error,
                **output_audit,
            }
            atomic_write_json(attempt_dir / "attempt_summary.json", attempt_summary)
            append_jsonl(
                args.output_dir / "batch_events.jsonl",
                {"event": "attempt_finished", "time_utc": utc_now(), **attempt_summary},
            )
            print(
                f"[M6.4B batch] DONE order={row['collection_order']} return_code={return_code} "
                f"pass={output_audit['pass']} duration={duration:.1f}s",
                flush=True,
            )
            if not output_audit["pass"]:
                exit_code = 2
            write_batch_outputs(args, primary, reserve, audit, selected_orders)
    state = write_batch_outputs(args, primary, reserve, audit, selected_orders)
    print(json.dumps({"mode": "execute", "batch_status": state["batch_status"], "status_counts": state["status_counts"], "output_dir": str(args.output_dir)}, indent=2))
    return exit_code


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed Stage 7 M6.4B locked primary rollout batch orchestrator."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--primary_csv", type=Path, required=True)
    parser.add_argument("--reserve_csv", type=Path, required=True)
    parser.add_argument("--nuplan_db_root", type=Path, required=True)
    parser.add_argument("--nuplan_map_root", type=Path, required=True)
    parser.add_argument("--nuplan_data_root", type=Path, required=True)
    parser.add_argument("--nuplan_exp_root", type=Path, required=True)
    parser.add_argument("--nuplan_devkit_root", type=Path, required=True)
    parser.add_argument("--tuplan_garage_root", type=Path, required=True)
    parser.add_argument("--stage7c_tool", type=Path, default=Path("tools/stage7c1_run_nuplan_simulation.py"))
    parser.add_argument("--python_executable", type=Path, required=True)
    parser.add_argument("--expected_nuplan_commit", required=True)
    parser.add_argument("--expected_tuplan_commit", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_order", type=int, default=1)
    parser.add_argument("--end_order", type=int, default=0, help="0 means no upper bound")
    parser.add_argument("--max_scenarios", type=int, default=0, help="0 means all rows in the selected order range")
    parser.add_argument("--command_timeout_s", type=int, default=3600)
    parser.add_argument("--execute", action="store_true", help="Actually run official rollouts; default is dry-run only")
    parser.add_argument("--confirm_primary_manifest_sha256", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true", help="Create a new attempt directory for failed/corrupt rows; never overwrites prior attempts")
    args = parser.parse_args(argv)
    if args.start_order < 1:
        parser.error("--start_order must be >= 1")
    if args.end_order < 0 or (args.end_order and args.end_order < args.start_order):
        parser.error("--end_order must be 0 or >= --start_order")
    if args.max_scenarios < 0:
        parser.error("--max_scenarios must be >= 0")
    if args.command_timeout_s < 1:
        parser.error("--command_timeout_s must be >= 1")
    if args.retry_failed and not args.execute:
        parser.error("--retry_failed requires --execute")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
