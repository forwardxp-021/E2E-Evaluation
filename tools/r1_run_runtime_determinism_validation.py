#!/usr/bin/env python3
"""Run the one-time R1 bound-runtime replay determinism validation.

This controller permits exactly four frozen scenarios and two repetitions per
scenario.  It claims each official closed-loop run before starting nuPlan and
records only reproducibility hashes; it never evaluates scientific outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.stage7_m6_4b_run_locked_rollouts import stage7c_environment


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
DEFAULT_ROSTER = R1_DIR / "r1_runtime_determinism_validation_roster_v1.0.json"
DEFAULT_SELECTOR = R1_DIR / "r1_runtime_determinism_selector_contract_v1.0.json"
DEFAULT_OWNER = R1_DIR / "r1_phaseb1_scientific_owner_approval_v0.1.json"
DEFAULT_OUTPUT = ROOT / "outputs/r1_runtime_determinism_validation_v1"
DEFAULT_RESULT = R1_DIR / "r1_runtime_determinism_result_v1.0.json"
TRACE_FILE = "planner_trace.jsonl"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class OfficialRunBudget:
    """Fail-closed ledger for the only budget unit in this phase."""

    cap: int
    expected_run_ids: set[str]
    ledger_path: Path
    records: List[Dict[str, Any]]

    @classmethod
    def create(cls, cap: int, expected: Sequence[Mapping[str, Any]], ledger_path: Path) -> "OfficialRunBudget":
        run_ids = {str(row["run_id"]) for row in expected}
        if len(run_ids) != cap or len(expected) != cap:
            raise ValueError("runtime validation schedule must contain exactly eight unique official run IDs")
        ledger = cls(cap=cap, expected_run_ids=run_ids, ledger_path=ledger_path, records=[])
        ledger._flush()
        return ledger

    def _flush(self) -> None:
        write_json(
            self.ledger_path,
            {
                "schema_version": "r1_runtime_determinism_official_run_budget_v1.0",
                "unit": "OFFICIAL_CLOSED_LOOP_RUN",
                "authorized_cap": self.cap,
                "claimed_count": len(self.records),
                "records": self.records,
            },
        )

    def claim(self, run: Mapping[str, Any]) -> None:
        run_id = str(run["run_id"])
        if run_id not in self.expected_run_ids:
            raise RuntimeError(f"refusing unplanned official run: {run_id}")
        if any(record["run_id"] == run_id for record in self.records):
            raise RuntimeError(f"refusing duplicate official run: {run_id}")
        if len(self.records) >= self.cap:
            raise RuntimeError(f"OFFICIAL_CLOSED_LOOP_RUN cap {self.cap} reached before simulation start")
        record = dict(run)
        record.update({"claim_status": "CLAIMED_BEFORE_SIMULATION", "actual_run_number": len(self.records) + 1})
        self.records.append(record)
        self._flush()


def _state_sequence_hash(records: Sequence[Mapping[str, Any]], field: str) -> str:
    return canonical_sha256([record[field] for record in records])


def _read_trace(trace_path: Path) -> List[Dict[str, Any]]:
    if not trace_path.is_file():
        raise FileNotFoundError(f"planner trace was not produced: {trace_path}")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"planner trace is empty: {trace_path}")
    expected = list(range(len(rows)))
    actual = [int(row["iteration_index"]) for row in rows]
    if actual != expected:
        raise RuntimeError(f"planner trace iterations are not a single exact sequence: {actual[:5]} ...")
    return rows


def _metric_groups(run_dir: Path) -> Dict[str, Any]:
    """Collect the official metric JSON payload hashes without interpreting them."""
    all_metrics: List[Dict[str, str]] = []
    for path in sorted(run_dir.rglob("*.json")):
        relative = str(path.relative_to(run_dir))
        if "metric" not in relative.lower() and "metrics" not in relative.lower():
            continue
        all_metrics.append({"relative_path": relative, "sha256": sha256_file(path)})
    collision = [row for row in all_metrics if "collision" in row["relative_path"].lower()]
    drivable = [row for row in all_metrics if any(marker in row["relative_path"].lower() for marker in ("drivable", "offroad", "off_road"))]
    return {
        "all_metric_payloads": all_metrics,
        "collision_metric_payloads": collision,
        "offroad_drivable_metric_payloads": drivable,
        "collision_metric_present": bool(collision),
        "offroad_drivable_metric_present": bool(drivable),
    }


def summarise_run(run: Mapping[str, Any], return_code: int, run_dir: Path) -> Dict[str, Any]:
    trace_path = run_dir / "trace" / TRACE_FILE
    binding_path = run_dir / "trace" / "planner_binding.json"
    failures: List[str] = []
    records: List[Dict[str, Any]] = []
    binding: Dict[str, Any] = {}
    try:
        records = _read_trace(trace_path)
    except Exception as exc:  # Preserve the exact failed prerequisite in the deterministic status.
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
        "initial_history_canonical": [] if not records else [row["initial_history_canonical"] for row in records],
        "pre_context_raw": [] if not records else [row["pre_context_raw"] for row in records],
        "canonical_context": [] if not records else [row["canonical_context"] for row in records],
        "traffic_light_states": [] if not records else [row["traffic_light_states"] for row in records],
        "background_tracked_object_states": [] if not records else [row["canonical_context"] for row in records],
        "ego_state_trajectory": [] if not records else [row["current_ego"] for row in records],
        "planner_output_trajectory": [] if not records else [row["planner_output_trajectory"] for row in records],
        "timestamps_us": [] if not records else [row["iteration_time_us"] for row in records],
    }
    return {
        "run_id": run["run_id"],
        "scenario_token": run["scenario_token"],
        "log_id": run["log_id"],
        "family": run["family"],
        "repeat": run["repeat"],
        "official_command_return_code": return_code,
        "technical_failure_status": "NO_TECHNICAL_FAILURE" if not failures else "TECHNICAL_FAILURE",
        "technical_failure_reasons": failures,
        "planner_binding": binding,
        "planner_binding_sha256": canonical_sha256(binding) if binding else None,
        "simulation_step_count": len(records),
        "sequence": sequence,
        "sequence_hashes": {key: canonical_sha256(value) for key, value in sequence.items()},
        "metrics": metrics,
        "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
        "command_log_sha256": sha256_file(run_dir / "official_run.log") if (run_dir / "official_run.log").is_file() else None,
    }


def _numeric_max_abs_difference(left: Any, right: Any) -> float | None:
    """Report a diagnostic maximum only; this never becomes an acceptance tolerance."""
    maximum: float | None = None
    if isinstance(left, (float, int)) and isinstance(right, (float, int)) and not isinstance(left, bool) and not isinstance(right, bool):
        return abs(float(left) - float(right))
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for a, b in zip(left, right):
            delta = _numeric_max_abs_difference(a, b)
            maximum = delta if maximum is None else (max(maximum, delta) if delta is not None else maximum)
    elif isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        for key in sorted(left):
            delta = _numeric_max_abs_difference(left[key], right[key])
            maximum = delta if maximum is None else (max(maximum, delta) if delta is not None else maximum)
    return maximum


def compare_pair(run_a: Mapping[str, Any], run_b: Mapping[str, Any], map_fingerprint: str) -> Dict[str, Any]:
    categories = {
        "scenario_token_log": ("identity", {"scenario_token": run_a["scenario_token"], "log_id": run_a["log_id"]}, {"scenario_token": run_b["scenario_token"], "log_id": run_b["log_id"]}),
        "map_fingerprint": ("identity", map_fingerprint, map_fingerprint),
        "route_roadblock_sequence": ("binding", run_a.get("planner_binding", {}).get("route_roadblock_ids"), run_b.get("planner_binding", {}).get("route_roadblock_ids")),
        "initial_history_canonical_hash": ("sequence", run_a["sequence"]["initial_history_canonical"], run_b["sequence"]["initial_history_canonical"]),
        "pre_context_raw_hash": ("sequence", run_a["sequence"]["pre_context_raw"], run_b["sequence"]["pre_context_raw"]),
        "canonical_context_json_hash": ("sequence", run_a["sequence"]["canonical_context"], run_b["sequence"]["canonical_context"]),
        "simulation_step_count": ("identity", run_a["simulation_step_count"], run_b["simulation_step_count"]),
        "simulation_timestamps": ("sequence", run_a["sequence"]["timestamps_us"], run_b["sequence"]["timestamps_us"]),
        "traffic_light_state_sequence": ("sequence", run_a["sequence"]["traffic_light_states"], run_b["sequence"]["traffic_light_states"]),
        "background_tracked_object_state_sequence": ("sequence", run_a["sequence"]["background_tracked_object_states"], run_b["sequence"]["background_tracked_object_states"]),
        "ego_state_trajectory": ("sequence", run_a["sequence"]["ego_state_trajectory"], run_b["sequence"]["ego_state_trajectory"]),
        "planner_output_trajectory": ("sequence", run_a["sequence"]["planner_output_trajectory"], run_b["sequence"]["planner_output_trajectory"]),
        "collision_metric": ("metric", run_a["metrics"]["collision_metric_payloads"], run_b["metrics"]["collision_metric_payloads"]),
        "offroad_drivable_area_metric": ("metric", run_a["metrics"]["offroad_drivable_metric_payloads"], run_b["metrics"]["offroad_drivable_metric_payloads"]),
        "technical_failure_status": ("identity", {"status": run_a["technical_failure_status"], "reasons": run_a["technical_failure_reasons"]}, {"status": run_b["technical_failure_status"], "reasons": run_b["technical_failure_reasons"]}),
    }
    results: Dict[str, Dict[str, Any]] = {}
    for name, (_, left, right) in categories.items():
        left_hash, right_hash = canonical_sha256(left), canonical_sha256(right)
        exact = left_hash == right_hash
        results[name] = {
            "exact_canonical_equality": exact,
            "run_a_sha256": left_hash,
            "run_b_sha256": right_hash,
            "max_abs_difference_diagnostic_only": None if exact else _numeric_max_abs_difference(left, right),
        }
    technical_ready = run_a["technical_failure_status"] == "NO_TECHNICAL_FAILURE" and run_b["technical_failure_status"] == "NO_TECHNICAL_FAILURE"
    exact_all = technical_ready and all(row["exact_canonical_equality"] for row in results.values())
    return {
        "scenario_token": run_a["scenario_token"],
        "log_id": run_a["log_id"],
        "family": run_a["family"],
        "run_a": run_a["run_id"],
        "run_b": run_b["run_id"],
        "comparison_rule": "EXACT_CANONICAL_EQUALITY_NO_TOLERANCE",
        "categories": results,
        "status": "EXACTLY_EQUAL" if exact_all else "DETERMINISM_NOT_VERIFIED",
    }


def build_schedule(roster: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries = list(roster.get("entries", []))
    if len(entries) != 4 or {row.get("family") for row in entries} != {"R-HLC", "R-TSB"}:
        raise ValueError("frozen runtime roster must contain exactly two HLC and two TSB entries")
    schedule: List[Dict[str, Any]] = []
    for entry in entries:
        for repeat in ("RUN_A", "RUN_B"):
            schedule.append(
                {
                    "run_id": f"{entry['family']}__{entry['scenario_token']}__{repeat}",
                    "scenario_token": entry["scenario_token"],
                    "log_id": entry["log_id"],
                    "db_path": entry["db_path"],
                    "family": entry["family"],
                    "repeat": repeat,
                    "runtime_arm": entry["runtime_arm"],
                }
            )
    return schedule


def command_for(args: argparse.Namespace, run: Mapping[str, Any], run_dir: Path) -> List[str]:
    hydra_searchpath = (
        f"[file://{(args.project_root / 'configs/r1_runtime_determinism_hydra').resolve()},"
        "pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
    )
    return [
        str(args.python_executable.resolve()),
        str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()),
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r1_runtime_determinism",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{Path(str(run['db_path'])).resolve()}]",
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "worker=single_machine_thread_pool",
        "worker.max_workers=1",
        "scenario_builder.max_workers=1",
        "max_callback_workers=1",
        "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0",
        "gpu=false",
        "seed=2026082701",
        "run_metric=true",
        "enable_simulation_progress_bar=false",
        "experiment_name=r1_runtime_determinism_validation_v1",
        f"job_name={run['run_id']}",
        f"output_dir={run_dir / 'nuplan_output'}",
        f"hydra.searchpath={hydra_searchpath}",
    ]


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    required = (args.roster, args.selector_contract, args.owner_approval, args.nuplan_devkit_root, args.tuplan_garage_root, args.nuplan_data_root, args.nuplan_map_root, args.python_executable)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing bound runtime input(s): {missing}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime-validation output: {args.output_dir}")
    if args.result_path.exists():
        raise FileExistsError(f"refusing to overwrite runtime determinism result: {args.result_path}")
    roster, selector, owner = read_json(args.roster), read_json(args.selector_contract), read_json(args.owner_approval)
    if roster.get("status") != "FROZEN_RUNTIME_DETERMINISM_VALIDATION_ONLY":
        raise RuntimeError("runtime roster is not frozen for validation only")
    if selector.get("status") != "FROZEN_BEFORE_CANDIDATE_ENUMERATION":
        raise RuntimeError("runtime selector was not frozen before enumeration")
    if owner.get("decisions", {}).get("R1_OFFICIAL_RUNTIME_DETERMINISM_VALIDATION") != "AUTHORIZED_ONCE":
        raise RuntimeError("scientific owner did not authorize the one-time runtime validation")
    schedule = build_schedule(roster)
    if len(schedule) != 8 or int(roster["run_budget"]["authorized_cap"]) != 8:
        raise RuntimeError("official runtime validation must schedule exactly eight runs")
    args.output_dir.mkdir(parents=True)
    manifest_path = args.output_dir / "execution_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "r1_runtime_determinism_execution_manifest_v1.0",
            "purpose": "BOUND_RUNTIME_REPRODUCIBILITY_ONLY_NO_SCIENTIFIC_OUTCOME_ANALYSIS",
            "roster_sha256": sha256_file(args.roster),
            "selector_contract_sha256": sha256_file(args.selector_contract),
            "owner_approval_sha256": sha256_file(args.owner_approval),
            "schedule": schedule,
            "official_run_cap": 8,
            "treatment_runs": 0,
        },
    )
    budget = OfficialRunBudget.create(8, schedule, args.output_dir / "official_run_budget.json")
    env_args = argparse.Namespace(
        nuplan_devkit_root=args.nuplan_devkit_root,
        tuplan_garage_root=args.tuplan_garage_root,
        nuplan_data_root=args.nuplan_data_root,
        nuplan_map_root=args.nuplan_map_root,
        nuplan_exp_root=args.nuplan_exp_root,
    )
    base_env = stage7c_environment(env_args)
    summaries: Dict[str, Dict[str, Any]] = {}
    stopped_reason: str | None = None
    for run in schedule:
        budget.claim(run)
        run_dir = args.output_dir / "runs" / run["run_id"]
        run_dir.mkdir(parents=True)
        trace_dir = run_dir / "trace"
        env = dict(base_env)
        env.update(
            {
                "R1_RUNTIME_DETERMINISM_ROSTER": str(args.roster.resolve()),
                "R1_RUNTIME_DETERMINISM_FAMILY": str(run["family"]),
                "R1_RUNTIME_DETERMINISM_TRACE_DIR": str(trace_dir.resolve()),
            }
        )
        command = command_for(args, run, run_dir)
        (run_dir / "official_command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (run_dir / "official_run.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=args.project_root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=args.command_timeout_s)
        summary = summarise_run(run, process.returncode, run_dir)
        summaries[str(run["run_id"])] = summary
        write_json(run_dir / "run_summary.json", summary)
        if summary["technical_failure_status"] != "NO_TECHNICAL_FAILURE":
            stopped_reason = f"RUN_FAILURE:{run['run_id']}"
            break
        if run["repeat"] == "RUN_B":
            prior = summaries[f"{run['family']}__{run['scenario_token']}__RUN_A"]
            pair = compare_pair(prior, summary, args.map_fingerprint_sha256)
            write_json(run_dir.parent / f"pair_{run['family']}__{run['scenario_token']}.json", pair)
            if pair["status"] != "EXACTLY_EQUAL":
                stopped_reason = f"PAIR_MISMATCH:{run['family']}:{run['scenario_token']}"
                break
    pairs: List[Dict[str, Any]] = []
    for entry in roster["entries"]:
        run_a = summaries.get(f"{entry['family']}__{entry['scenario_token']}__RUN_A")
        run_b = summaries.get(f"{entry['family']}__{entry['scenario_token']}__RUN_B")
        if run_a is not None and run_b is not None:
            pairs.append(compare_pair(run_a, run_b, args.map_fingerprint_sha256))
    ninth_preflight = {"status": "NOT_ATTEMPTED_DUE_TO_EARLY_STOP"}
    if len(budget.records) == 8 and stopped_reason is None:
        try:
            budget.claim({**schedule[0], "run_id": "R1_FORBIDDEN_NINTH_PRE_RUN_CLAIM"})
        except RuntimeError as exc:
            ninth_preflight = {"status": "REJECTED_BEFORE_SIMULATION", "reason": str(exc)}
        else:  # pragma: no cover - the branch is a safety failure, not an alternate protocol path.
            raise RuntimeError("ninth official run was not rejected before simulation")
    pass_all = (
        stopped_reason is None
        and len(summaries) == 8
        and len(pairs) == 4
        and all(pair["status"] == "EXACTLY_EQUAL" for pair in pairs)
        and ninth_preflight["status"] == "REJECTED_BEFORE_SIMULATION"
    )
    result = {
        "schema_version": "r1_runtime_determinism_result_v1.0",
        "status": "PASS" if pass_all else "FAIL",
        "scope": "BOUND_RUNTIME_REPRODUCIBILITY_ONLY_NO_SCIENTIFIC_OUTCOME_ANALYSIS",
        "background_replay_determinism": "VERIFIED_ON_BOUND_RUNTIME" if pass_all else "NOT_VERIFIED",
        "official_replay": "READY_FOR_TECHNICAL_SMOKE_REVIEW" if pass_all else "NOT_READY",
        "roster_sha256": sha256_file(args.roster),
        "selector_contract_sha256": sha256_file(args.selector_contract),
        "owner_approval_sha256": sha256_file(args.owner_approval),
        "map_fingerprint_sha256": args.map_fingerprint_sha256,
        "official_run_unit": "OFFICIAL_CLOSED_LOOP_RUN",
        "authorized_cap": 8,
        "actual_official_run_count": len(budget.records),
        "ninth_preflight": ninth_preflight,
        "stopped_reason": stopped_reason,
        "runs": summaries,
        "pairs": pairs,
        "raw_output_directory": str(args.output_dir.resolve()),
        "raw_output_directory_committed": False,
        "scientific_outcome_analysis_performed": False,
    }
    write_json(args.result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--selector-contract", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--owner-approval", type=Path, default=DEFAULT_OWNER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--nuplan-devkit-root", type=Path, default=ROOT.parent / "nuplan-devkit")
    parser.add_argument("--tuplan-garage-root", type=Path, default=ROOT.parent / "tuplan_garage")
    parser.add_argument("--nuplan-data-root", type=Path, default=ROOT.parent / "nuplan/dataset/data")
    parser.add_argument("--nuplan-map-root", type=Path, default=ROOT.parent / "nuplan/dataset/maps")
    parser.add_argument("--nuplan-exp-root", type=Path, default=ROOT.parent / "nuplan/exp")
    parser.add_argument("--python-executable", type=Path, default=Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9"))
    parser.add_argument("--command-timeout-s", type=int, default=1200)
    parser.add_argument("--map-fingerprint-sha256", default="a85e17eba18e5fdd65148705844b8f189bb4d4373a1d82805e1f8ffd4ae8afb3")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(execute(parse_args()), ensure_ascii=False, indent=2))
