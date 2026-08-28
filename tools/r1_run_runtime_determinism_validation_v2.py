#!/usr/bin/env python3
"""Run the owner-authorized, bound R1 runtime determinism validation V2.

V2 is intentionally a narrow replacement execution: it reuses the four frozen
outcome-blind roster rows, claims exactly one official closed-loop unit before
each call, and compares only reproducibility artifacts.  It does not evaluate
scientific outcomes or authorize any treatment, smoke, selector, or RBR work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_run_runtime_determinism_validation import _metric_groups, _read_trace, canonical_sha256, read_json, sha256_file, write_json
from tools.stage7_m6_4b_run_locked_rollouts import stage7c_environment


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_AUTHORIZATION = R1_DIR / "r1_runtime_determinism_validation_v2_authorization_v1.0.json"
DEFAULT_ROSTER = R1_DIR / "r1_runtime_determinism_validation_roster_v1.0.json"
DEFAULT_PREFLIGHT = R1_DIR / "r1_runtime_determinism_v2_interface_preflight_v1.1.json"
DEFAULT_OUTPUT = ROOT / "outputs/r1_runtime_determinism_validation_v2"
DEFAULT_RESULT = R1_DIR / "r1_runtime_determinism_result_v2.0.json"
DEFAULT_COMPARISON = R1_DIR / "r1_runtime_determinism_comparison_v2.0.csv"
DEFAULT_LEDGER = R1_DIR / "r1_runtime_determinism_run_ledger_v2.0.csv"
TRACE_FILE = "planner_trace.jsonl"
COMPARISON_FIELDS = (
    "scenario_token_log",
    "map_fingerprint",
    "route_roadblock_sequence",
    "initial_history_canonical",
    "pre_context_raw",
    "canonical_context_json",
    "simulation_step_count",
    "simulation_timestamps",
    "traffic_light_state_sequence",
    "background_tracked_object_state_sequence",
    "ego_state_trajectory",
    "planner_output_trajectory",
    "collision_metric",
    "offroad_drivable_area_metric",
    "technical_failure_status",
)
LEDGER_FIELDS = (
    "run_id", "scenario_token", "log_id", "family", "repeat", "runtime_arm", "claim_status", "actual_run_number",
    "execution_status", "official_command_return_code", "technical_failure_status", "technical_failure_reasons", "trace_sha256",
    "planner_binding_sha256", "command_log_sha256",
)
COMPARISON_CSV_FIELDS = (
    "scenario_token", "log_id", "family", "run_a", "run_b", "category", "exact_canonical_equality",
    "run_a_sha256", "run_b_sha256", "max_abs_difference_diagnostic_only", "first_difference_step", "affected_fields",
)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


@dataclass
class OfficialRunBudgetV2:
    cap: int
    expected_run_ids: set[str]
    raw_ledger_path: Path
    public_ledger_path: Path
    records: List[Dict[str, Any]]

    @classmethod
    def create(cls, schedule: Sequence[Mapping[str, Any]], raw_ledger_path: Path, public_ledger_path: Path) -> "OfficialRunBudgetV2":
        run_ids = {str(row["run_id"]) for row in schedule}
        if len(schedule) != 8 or len(run_ids) != 8:
            raise ValueError("V2 schedule must contain exactly eight unique official run IDs")
        budget = cls(8, run_ids, raw_ledger_path, public_ledger_path, [])
        budget.flush()
        return budget

    def flush(self) -> None:
        write_json(
            self.raw_ledger_path,
            {
                "schema_version": "r1_runtime_determinism_v2_official_run_budget_v1.0",
                "unit": "OFFICIAL_CLOSED_LOOP_RUN",
                "authorized_cap": self.cap,
                "claimed_count": len(self.records),
                "records": self.records,
            },
        )
        _write_csv(self.public_ledger_path, self.records, LEDGER_FIELDS)

    def claim(self, run: Mapping[str, Any]) -> Dict[str, Any]:
        if len(self.records) >= self.cap:
            raise RuntimeError(f"OFFICIAL_CLOSED_LOOP_RUN cap {self.cap} reached before simulation start")
        run_id = str(run["run_id"])
        if run_id not in self.expected_run_ids:
            raise RuntimeError(f"refusing unplanned official run: {run_id}")
        if any(str(row["run_id"]) == run_id for row in self.records):
            raise RuntimeError(f"refusing duplicate official run: {run_id}")
        record = dict(run)
        record.update(
            {
                "claim_status": "V2_CLAIMED_BEFORE_SIMULATION",
                "actual_run_number": len(self.records) + 1,
                "execution_status": "CLAIMED_NOT_STARTED",
                "official_command_return_code": "",
                "technical_failure_status": "",
                "technical_failure_reasons": "",
                "trace_sha256": "",
                "planner_binding_sha256": "",
                "command_log_sha256": "",
            }
        )
        self.records.append(record)
        self.flush()
        return record

    def complete(self, run_id: str, summary: Mapping[str, Any]) -> None:
        record = next(row for row in self.records if str(row["run_id"]) == run_id)
        record.update(
            {
                "execution_status": "EXECUTED",
                "official_command_return_code": summary["official_command_return_code"],
                "technical_failure_status": summary["technical_failure_status"],
                "technical_failure_reasons": " | ".join(summary["technical_failure_reasons"]),
                "trace_sha256": summary["trace_sha256"] or "",
                "planner_binding_sha256": summary["planner_binding_sha256"] or "",
                "command_log_sha256": summary["command_log_sha256"] or "",
            }
        )
        self.flush()

    def assert_ninth_rejected(self) -> Dict[str, str]:
        try:
            self.claim({"run_id": "V2_FORBIDDEN_NINTH_PRE_RUN_CLAIM"})
        except RuntimeError as exc:
            return {"status": "REJECTED_BEFORE_SIMULATION", "reason": str(exc)}
        raise RuntimeError("ninth official run was not rejected before simulation")


def _validate_roster(roster: Mapping[str, Any], roster_path: Path, authority: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if roster.get("status") != "FROZEN_RUNTIME_DETERMINISM_VALIDATION_ONLY":
        raise ValueError("runtime roster is not frozen validation-only")
    if sha256_file(roster_path) != authority["binding"]["original_frozen_runtime_roster_sha256"]:
        raise ValueError("V2 roster hash differs from the original frozen roster")
    entries = list(roster.get("entries", []))
    if len(entries) != 4 or [row.get("family") for row in entries].count("R-HLC") != 2 or [row.get("family") for row in entries].count("R-TSB") != 2:
        raise ValueError("V2 requires exactly two R-HLC and two R-TSB frozen rows")
    if len({str(row.get("scenario_token")) for row in entries}) != 4 or len({str(row.get("log_id")) for row in entries}) != 4:
        raise ValueError("V2 roster token/log identities are not unique")
    required_labels = set(authority["roster_reuse"]["required_isolation_labels"])
    for row in entries:
        if not required_labels.issubset(set(row.get("isolation_labels", []))):
            raise ValueError(f"V2 roster row is missing required permanent isolation labels: {row.get('scenario_token')}")
    expected_arm = authority["authorization"]["permitted_arms"]
    for row in entries:
        if row.get("runtime_arm") != expected_arm[row["family"]]:
            raise ValueError(f"frozen runtime arm is not permitted for {row['family']}")
    return entries


def _build_schedule(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    for entry in entries:
        for repeat in ("V2_RUN_A", "V2_RUN_B"):
            schedule.append(
                {
                    "run_id": f"{entry['family']}__{entry['scenario_token']}__{repeat}",
                    "scenario_token": str(entry["scenario_token"]),
                    "log_id": str(entry["log_id"]),
                    "db_path": str(entry["db_path"]),
                    "family": str(entry["family"]),
                    "repeat": repeat,
                    "runtime_arm": str(entry["runtime_arm"]),
                }
            )
    return schedule


def _command_for(args: argparse.Namespace, run: Mapping[str, Any], run_dir: Path) -> List[str]:
    hydra_searchpath = (
        f"[file://{(args.project_root / 'configs/r1_runtime_determinism_hydra').resolve()},"
        "pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
    )
    return [
        str(args.python_executable.resolve()),
        str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()),
        "+simulation=closed_loop_nonreactive_agents", "planner=r1_runtime_determinism", "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{Path(str(run['db_path'])).resolve()}]", "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_tokens=[{run['scenario_token']}]", "worker=single_machine_thread_pool", "worker.max_workers=1",
        "scenario_builder.max_workers=1", "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701", "run_metric=true",
        "enable_simulation_progress_bar=false", "experiment_name=r1_runtime_determinism_validation_v2",
        f"job_name={run['run_id']}", f"output_dir={run_dir / 'nuplan_output'}", f"hydra.searchpath={hydra_searchpath}",
    ]


def _summarise_run(run: Mapping[str, Any], return_code: int, run_dir: Path) -> Dict[str, Any]:
    trace_path = run_dir / "trace" / TRACE_FILE
    binding_path = run_dir / "trace" / "planner_binding.json"
    failures: List[str] = []
    records: List[Dict[str, Any]] = []
    binding: Dict[str, Any] = {}
    try:
        records = _read_trace(trace_path)
    except Exception as exc:
        failures.append(f"TRACE_UNAVAILABLE:{type(exc).__name__}:{exc}")
    try:
        binding = read_json(binding_path)
    except Exception as exc:
        failures.append(f"BINDING_UNAVAILABLE:{type(exc).__name__}:{exc}")
    metrics = _metric_groups(run_dir)
    if not metrics["collision_metric_present"]:
        failures.append("COLLISION_METRIC_UNAVAILABLE")
    if not metrics["offroad_drivable_metric_present"]:
        failures.append("OFFROAD_DRIVABLE_METRIC_UNAVAILABLE")
    if return_code != 0:
        failures.append(f"OFFICIAL_COMMAND_RETURN_CODE_{return_code}")
    sequence = {
        "initial_history_canonical": [row["initial_history_canonical"] for row in records],
        "pre_context_raw": [row["pre_context_raw"] for row in records],
        "canonical_context": [row["canonical_context"] for row in records],
        "traffic_light_states": [row["traffic_light_states"] for row in records],
        "background_tracked_object_states": [row["canonical_context"] for row in records],
        "ego_state_trajectory": [row["current_ego"] for row in records],
        "planner_output_trajectory": [row["planner_output_trajectory"] for row in records],
        "timestamps_us": [row["iteration_time_us"] for row in records],
    }
    return {
        "run_id": run["run_id"], "scenario_token": run["scenario_token"], "log_id": run["log_id"], "family": run["family"], "repeat": run["repeat"],
        "official_command_return_code": return_code,
        "technical_failure_status": "NO_TECHNICAL_FAILURE" if not failures else "TECHNICAL_FAILURE",
        "technical_failure_reasons": failures, "planner_binding": binding,
        "planner_binding_sha256": canonical_sha256(binding) if binding else None, "simulation_step_count": len(records),
        "sequence": sequence, "sequence_hashes": {key: canonical_sha256(value) for key, value in sequence.items()}, "metrics": metrics,
        "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
        "command_log_sha256": sha256_file(run_dir / "official_run.log") if (run_dir / "official_run.log").is_file() else None,
    }


def _difference(left: Any, right: Any, path: str = "$") -> Dict[str, Any]:
    if left == right:
        return {"max_abs": None, "first_step": None, "fields": []}
    fields: List[str] = []
    maximum: float | None = None
    first_step: int | None = None

    def record(a: Any, b: Any, here: str) -> None:
        nonlocal maximum, first_step
        if a == b:
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
            delta = abs(float(a) - float(b))
            maximum = delta if maximum is None else max(maximum, delta)
        if not fields:
            matched = re.search(r"\[(\d+)\]", here)
            first_step = None if matched is None else int(matched.group(1))
        if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
            for key in sorted(a):
                record(a[key], b[key], f"{here}.{key}")
            return
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                record(item_a, item_b, f"{here}[{index}]")
            return
        fields.append(here)

    record(left, right, path)
    return {"max_abs": maximum, "first_step": first_step, "fields": fields[:100]}


def _compare_pair(run_a: Mapping[str, Any], run_b: Mapping[str, Any], map_fingerprint: str) -> Dict[str, Any]:
    categories = {
        "scenario_token_log": ({"scenario_token": run_a["scenario_token"], "log_id": run_a["log_id"]}, {"scenario_token": run_b["scenario_token"], "log_id": run_b["log_id"]}),
        "map_fingerprint": (map_fingerprint, map_fingerprint),
        "route_roadblock_sequence": (run_a["planner_binding"].get("route_roadblock_ids"), run_b["planner_binding"].get("route_roadblock_ids")),
        "initial_history_canonical": (run_a["sequence"]["initial_history_canonical"], run_b["sequence"]["initial_history_canonical"]),
        "pre_context_raw": (run_a["sequence"]["pre_context_raw"], run_b["sequence"]["pre_context_raw"]),
        "canonical_context_json": (run_a["sequence"]["canonical_context"], run_b["sequence"]["canonical_context"]),
        "simulation_step_count": (run_a["simulation_step_count"], run_b["simulation_step_count"]),
        "simulation_timestamps": (run_a["sequence"]["timestamps_us"], run_b["sequence"]["timestamps_us"]),
        "traffic_light_state_sequence": (run_a["sequence"]["traffic_light_states"], run_b["sequence"]["traffic_light_states"]),
        "background_tracked_object_state_sequence": (run_a["sequence"]["background_tracked_object_states"], run_b["sequence"]["background_tracked_object_states"]),
        "ego_state_trajectory": (run_a["sequence"]["ego_state_trajectory"], run_b["sequence"]["ego_state_trajectory"]),
        "planner_output_trajectory": (run_a["sequence"]["planner_output_trajectory"], run_b["sequence"]["planner_output_trajectory"]),
        "collision_metric": (run_a["metrics"]["collision_metric_payloads"], run_b["metrics"]["collision_metric_payloads"]),
        "offroad_drivable_area_metric": (run_a["metrics"]["offroad_drivable_metric_payloads"], run_b["metrics"]["offroad_drivable_metric_payloads"]),
        "technical_failure_status": ({"status": run_a["technical_failure_status"], "reasons": run_a["technical_failure_reasons"]}, {"status": run_b["technical_failure_status"], "reasons": run_b["technical_failure_reasons"]}),
    }
    results: Dict[str, Dict[str, Any]] = {}
    for name in COMPARISON_FIELDS:
        left, right = categories[name]
        left_hash, right_hash = canonical_sha256(left), canonical_sha256(right)
        exact = left_hash == right_hash
        diagnostic = _difference(left, right) if not exact else {"max_abs": None, "first_step": None, "fields": []}
        results[name] = {
            "exact_canonical_equality": exact, "run_a_sha256": left_hash, "run_b_sha256": right_hash,
            "max_abs_difference_diagnostic_only": diagnostic["max_abs"], "first_difference_step": diagnostic["first_step"],
            "affected_fields": diagnostic["fields"],
        }
    ready = run_a["technical_failure_status"] == "NO_TECHNICAL_FAILURE" and run_b["technical_failure_status"] == "NO_TECHNICAL_FAILURE"
    exact_all = ready and all(row["exact_canonical_equality"] for row in results.values())
    return {
        "scenario_token": run_a["scenario_token"], "log_id": run_a["log_id"], "family": run_a["family"], "run_a": run_a["run_id"], "run_b": run_b["run_id"],
        "comparison_rule": "EXACT_CANONICAL_EQUALITY_NO_TOLERANCE", "categories": results,
        "status": "EXACTLY_EQUAL" if exact_all else "DETERMINISM_NOT_VERIFIED",
    }


def _write_comparison_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        for category in COMPARISON_FIELDS:
            row = pair["categories"][category]
            rows.append(
                {
                    "scenario_token": pair["scenario_token"], "log_id": pair["log_id"], "family": pair["family"], "run_a": pair["run_a"], "run_b": pair["run_b"],
                    "category": category, "exact_canonical_equality": row["exact_canonical_equality"], "run_a_sha256": row["run_a_sha256"], "run_b_sha256": row["run_b_sha256"],
                    "max_abs_difference_diagnostic_only": row["max_abs_difference_diagnostic_only"], "first_difference_step": row["first_difference_step"],
                    "affected_fields": json.dumps(row["affected_fields"], ensure_ascii=False, separators=(",", ":")),
                }
            )
    _write_csv(path, rows, COMPARISON_CSV_FIELDS)


def _result_run_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: summary[key]
        for key in ("run_id", "scenario_token", "log_id", "family", "repeat", "official_command_return_code", "technical_failure_status", "technical_failure_reasons", "planner_binding_sha256", "simulation_step_count", "sequence_hashes", "metrics", "trace_sha256", "command_log_sha256")
    }


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    required = (args.authorization, args.roster, args.preflight, args.nuplan_devkit_root, args.tuplan_garage_root, args.nuplan_data_root, args.nuplan_map_root, args.python_executable)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing bound V2 input(s): {missing}")
    protected = (args.output_dir, args.result_path, args.comparison_path, args.ledger_path)
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite V2 evidence/output path(s): {existing}")
    authority, roster, preflight = read_json(args.authorization), read_json(args.roster), read_json(args.preflight)
    if authority.get("status") != "AUTHORIZED_ONCE":
        raise ValueError("V2 authorization status is not AUTHORIZED_ONCE")
    if authority["authorization"].get("maximum_new_runs") != 8:
        raise ValueError("V2 authorization does not bind the exact eight-run cap")
    if authority["binding"].get("current_local_git_commit_sha") != _git_head():
        raise ValueError("working HEAD differs from the V2 authorization binding")
    if authority["binding"].get("repaired_planner_sha256") != sha256_file(ROOT / "tools/r1_runtime_determinism_planner.py"):
        raise ValueError("repaired planner hash differs from V2 authorization binding")
    if authority["binding"].get("v2_execution_tool_sha256") != sha256_file(Path(__file__)):
        raise ValueError("V2 execution tool hash differs from V2 authorization binding")
    if preflight.get("status") != "PASS_NO_OFFICIAL_RUN_BUDGET_CONSUMED" or preflight.get("budget_consumed") != 0:
        raise ValueError("V2 interface preflight is not a zero-budget pass")
    if preflight.get("authorization_sha256") != sha256_file(args.authorization):
        raise ValueError("V2 interface preflight authorization binding differs")
    if preflight.get("planner_sha256") != sha256_file(ROOT / "tools/r1_runtime_determinism_planner.py"):
        raise ValueError("V2 interface preflight planner binding differs")
    entries = _validate_roster(roster, args.roster, authority)
    schedule = _build_schedule(entries)
    if len(schedule) != 8:
        raise RuntimeError("V2 schedule construction did not yield exactly eight runs")

    args.output_dir.mkdir(parents=True)
    write_json(
        args.output_dir / "execution_manifest_v2.json",
        {
            "schema_version": "r1_runtime_determinism_v2_execution_manifest_v1.0", "purpose": "BOUND_RUNTIME_REPRODUCIBILITY_ONLY_NO_SCIENTIFIC_OUTCOME_ANALYSIS",
            "authorization_sha256": sha256_file(args.authorization), "preflight_sha256": sha256_file(args.preflight), "roster_sha256": sha256_file(args.roster),
            "planner_sha256": sha256_file(ROOT / "tools/r1_runtime_determinism_planner.py"), "execution_tool_sha256": sha256_file(Path(__file__)),
            "official_run_cap": 8, "schedule": schedule, "treatment_runs": 0, "new_48_call_smoke_runs": 0,
        },
    )
    budget = OfficialRunBudgetV2.create(schedule, args.output_dir / "official_run_budget_v2.json", args.ledger_path)
    env_args = argparse.Namespace(nuplan_devkit_root=args.nuplan_devkit_root, tuplan_garage_root=args.tuplan_garage_root, nuplan_data_root=args.nuplan_data_root, nuplan_map_root=args.nuplan_map_root, nuplan_exp_root=args.nuplan_exp_root)
    base_env = stage7c_environment(env_args)
    summaries: Dict[str, Dict[str, Any]] = {}
    pairs: List[Dict[str, Any]] = []
    stopped_reason: str | None = None
    for run in schedule:
        budget.claim(run)
        run_dir = args.output_dir / "runs" / run["run_id"]
        run_dir.mkdir(parents=True)
        env = dict(base_env)
        env.update({"R1_RUNTIME_DETERMINISM_ROSTER": str(args.roster.resolve()), "R1_RUNTIME_DETERMINISM_FAMILY": str(run["family"]), "R1_RUNTIME_DETERMINISM_TRACE_DIR": str((run_dir / "trace").resolve())})
        command = _command_for(args, run, run_dir)
        write_json(run_dir / "official_command.json", {"command": command})
        return_code = -999
        try:
            with (run_dir / "official_run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=args.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=args.command_timeout_s)
                return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = -124
        summary = _summarise_run(run, return_code, run_dir)
        summaries[run["run_id"]] = summary
        write_json(run_dir / "run_summary.json", summary)
        budget.complete(str(run["run_id"]), summary)
        if summary["technical_failure_status"] != "NO_TECHNICAL_FAILURE":
            stopped_reason = f"RUN_FAILURE:{run['run_id']}"
            break
        if run["repeat"] == "V2_RUN_B":
            first = summaries[f"{run['family']}__{run['scenario_token']}__V2_RUN_A"]
            pair = _compare_pair(first, summary, authority["binding"]["map_fingerprint_sha256"])
            pairs.append(pair)
            write_json(run_dir.parent / f"pair_{run['family']}__{run['scenario_token']}.json", pair)
            _write_comparison_csv(args.comparison_path, pairs)
            if pair["status"] != "EXACTLY_EQUAL":
                stopped_reason = f"PAIR_MISMATCH:{run['family']}:{run['scenario_token']}"
                break
    if not args.comparison_path.exists():
        _write_comparison_csv(args.comparison_path, pairs)
    ninth_preflight: Dict[str, str] = {"status": "NOT_ATTEMPTED_DUE_TO_EARLY_STOP"}
    if stopped_reason is None and len(budget.records) == 8:
        ninth_preflight = budget.assert_ninth_rejected()
    passed = stopped_reason is None and len(summaries) == 8 and len(pairs) == 4 and all(pair["status"] == "EXACTLY_EQUAL" for pair in pairs) and ninth_preflight["status"] == "REJECTED_BEFORE_SIMULATION"
    result = {
        "schema_version": "r1_runtime_determinism_result_v2.0", "status": "PASS" if passed else "FAIL",
        "scope": "BOUND_RUNTIME_REPRODUCIBILITY_ONLY_NO_SCIENTIFIC_OUTCOME_ANALYSIS",
        "background_replay_determinism": "VERIFIED_ON_BOUND_RUNTIME" if passed else "NOT_VERIFIED",
        "official_replay": "READY_FOR_TECHNICAL_SMOKE_REVIEW" if passed else "NOT_READY",
        "authorization_sha256": sha256_file(args.authorization), "preflight_sha256": sha256_file(args.preflight), "roster_sha256": sha256_file(args.roster),
        "planner_sha256": sha256_file(ROOT / "tools/r1_runtime_determinism_planner.py"), "execution_tool_sha256": sha256_file(Path(__file__)),
        "map_fingerprint_sha256": authority["binding"]["map_fingerprint_sha256"], "official_run_unit": "OFFICIAL_CLOSED_LOOP_RUN", "authorized_cap": 8,
        "actual_official_run_count": len(budget.records), "ninth_preflight": ninth_preflight, "stopped_reason": stopped_reason,
        "runs": {run_id: _result_run_summary(summary) for run_id, summary in summaries.items()}, "pairs": pairs,
        "raw_output_directory": str(args.output_dir.resolve()), "raw_output_directory_committed": False, "scientific_outcome_analysis_performed": False,
    }
    write_json(args.result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--nuplan-devkit-root", type=Path, default=ROOT.parent / "nuplan-devkit")
    parser.add_argument("--tuplan-garage-root", type=Path, default=ROOT.parent / "tuplan_garage")
    parser.add_argument("--nuplan-data-root", type=Path, default=ROOT.parent / "nuplan/dataset/data")
    parser.add_argument("--nuplan-map-root", type=Path, default=ROOT.parent / "nuplan/dataset/maps")
    parser.add_argument("--nuplan-exp-root", type=Path, default=ROOT.parent / "nuplan/exp")
    parser.add_argument("--python-executable", type=Path, default=Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9"))
    parser.add_argument("--command-timeout-s", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    result = execute(parse_args())
    print(json.dumps({key: result[key] for key in ("status", "actual_official_run_count", "stopped_reason", "background_replay_determinism", "official_replay")}, ensure_ascii=False, indent=2))
