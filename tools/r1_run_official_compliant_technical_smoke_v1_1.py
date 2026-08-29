#!/usr/bin/env python3
"""Recover R1 Phase-B2.1 with the V3-bound official runtime only.

This versioned executor preserves every frozen scientific component from B2:
the 24-row roster, the two-arm schedule, smoke planner, generator, context,
mechanism, F-match, endpoint and safety evaluators.  Its sole implementation
change is to construct the complete V3-proven ``stage7c_environment(args)``
binding before the first official claim.  It never reads representation, BDD,
probes, checkpoints, or RBR artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from omegaconf import OmegaConf

from tools.r1_context_mechanism_core import build_canonical_context_record
from tools.r1_official_metric_canonicalizer import canonicalize_official_metrics, sha256_file
from tools.r1_official_technical_smoke_planner import R1OfficialTechnicalSmokePlanner
from tools.r1_run_official_compliant_technical_smoke_v1 import (
    CONTEXT_FIELDS,
    LEDGER_FIELDS,
    PAIR_FIELDS,
    RUN_CAP,
    SAFETY_FIELDS,
    _command_for as _legacy_command_for,
    _evaluate_pair,
    _git,
    _path_hash,
    _schedule as _legacy_schedule,
    _technical_summary,
    _write_csv,
)
from tools.r1_run_runtime_determinism_validation import read_json, write_json
from tools.stage7_m6_4b_run_locked_rollouts import stage7c_environment


ROOT = Path(__file__).resolve().parents[1]
R1_DIR = ROOT / "docs/stageR/r1"
MASTER_SEED = 2026082701
V3_DEFAULTS = {
    "nuplan_devkit_root": ROOT.parent / "nuplan-devkit",
    "tuplan_garage_root": ROOT.parent / "tuplan_garage",
    "nuplan_data_root": ROOT.parent / "nuplan/dataset/data",
    "nuplan_map_root": ROOT.parent / "nuplan/dataset/maps",
    "nuplan_exp_root": ROOT.parent / "nuplan/exp",
    "python_executable": Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9"),
    "command_timeout_s": 1200,
}
ENVIRONMENT_KEYS = (
    "NUPLAN_DEVKIT_ROOT", "NUPLAN_DATA_ROOT", "NUPLAN_MAPS_ROOT", "NUPLAN_MAP_ROOT", "NUPLAN_EXP_ROOT", "PYTHONPATH",
)
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
FROZEN_ROSTER_SHA256 = "0617e79b9f51d8b2ae8ac76b110e1dbcfaa77dad200a73b405eb2d6a54675e52"
FROZEN_SELECTOR_SALT_SHA256 = "617331678ef4573be11b5408a1dde2c910c8614177541dd51c650c08bc24baf9"

DEFAULT_ROSTER = R1_DIR / "r1_official_technical_smoke_roster_v1.0.json"
DEFAULT_SCOPE = R1_DIR / "r1_official_technical_smoke_scope_amendment_v1.0.json"
DEFAULT_SELECTOR = R1_DIR / "r1_future_compliant_smoke_selector_contract_v0.3.json"
DEFAULT_REPLAY = R1_DIR / "r1_official_nuplan_replay_contract_v1.0.json"
DEFAULT_HISTORICAL_MANIFEST = R1_DIR / "r1_official_technical_smoke_execution_manifest_v1.0.json"
DEFAULT_STATUS_CORRECTION = R1_DIR / "r1_official_technical_smoke_b2_status_correction_v1.0.json"
DEFAULT_ENVIRONMENT_BINDING = R1_DIR / "r1_official_smoke_runtime_environment_binding_v1.1.json"
DEFAULT_PREFLIGHT = R1_DIR / "r1_official_technical_smoke_recovery_preflight_v1.1.json"
DEFAULT_AUTHORIZATION = R1_DIR / "r1_official_compliant_technical_smoke_authorization_v1.1.json"
DEFAULT_AUTHORIZATION_REPORT = R1_DIR / "R1_Official_Compliant_Technical_Smoke_Authorization_v1.1.md"
DEFAULT_OUTPUT = ROOT / "outputs/r1_official_compliant_technical_smoke_v1_1"
DEFAULT_LEDGER = R1_DIR / "r1_official_technical_smoke_run_ledger_v1.1.csv"
DEFAULT_PAIR = R1_DIR / "r1_official_technical_smoke_pair_metrics_v1.1.csv"
DEFAULT_CONTEXT = R1_DIR / "r1_official_technical_smoke_context_identity_v1.1.csv"
DEFAULT_SAFETY = R1_DIR / "r1_official_technical_smoke_safety_v1.1.csv"
DEFAULT_FAMILY = R1_DIR / "r1_official_technical_smoke_family_summary_v1.1.csv"
DEFAULT_MANIFEST = R1_DIR / "r1_official_technical_smoke_execution_manifest_v1.1.json"


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _v3_env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        nuplan_devkit_root=args.nuplan_devkit_root,
        tuplan_garage_root=args.tuplan_garage_root,
        nuplan_data_root=args.nuplan_data_root,
        nuplan_map_root=args.nuplan_map_root,
        nuplan_exp_root=args.nuplan_exp_root,
    )


def build_runtime_environment(args: argparse.Namespace) -> Dict[str, str]:
    """Build exactly the V3 stage7c environment from explicit CLI roots."""
    return stage7c_environment(_v3_env_args(args))


def _v3_reference_environment() -> Dict[str, str]:
    return stage7c_environment(argparse.Namespace(**{key: value for key, value in V3_DEFAULTS.items() if key.endswith("_root")}))


def _nuplan_identity(args: argparse.Namespace, environment: Mapping[str, str]) -> Dict[str, str]:
    probe = "import importlib.metadata as m, nuplan; print(m.version('nuplan-devkit')); print(nuplan.__file__)"
    completed = subprocess.run(
        [str(args.python_executable.resolve()), "-c", probe], cwd=ROOT, env=dict(environment), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot query bound nuPlan version: {completed.stdout}")
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(values) != 2:
        raise RuntimeError(f"unexpected nuPlan version probe output: {completed.stdout}")
    return {"nuplan_devkit_version": values[0], "nuplan_module_path": str(Path(values[1]).resolve())}


def _validate_roster(args: argparse.Namespace) -> tuple[Dict[str, Any], List[Dict[str, str]]]:
    roster = read_json(args.roster)
    selector = read_json(args.selector_contract)
    if sha256_file(args.roster) != FROZEN_ROSTER_SHA256:
        raise ValueError("B2.1 roster SHA differs from the frozen 24-scenario roster")
    if selector.get("salt_sha256") != FROZEN_SELECTOR_SALT_SHA256:
        raise ValueError("B2.1 selector salt differs from the frozen value")
    entries = list(roster.get("entries", []))
    if len(entries) != 24 or len({str(row.get("scenario_token")) for row in entries}) != 24 or len({str(row.get("log_id")) for row in entries}) != 24:
        raise ValueError("B2.1 requires 24 entries with unique scenario tokens and log IDs")
    if sum(str(row.get("family")) == "R-HLC" for row in entries) != 12 or sum(str(row.get("family")) == "R-TSB" for row in entries) != 12:
        raise ValueError("B2.1 roster must retain 12 R-HLC and 12 R-TSB identities")
    legacy = _legacy_schedule(roster)
    schedule: List[Dict[str, str]] = []
    for row in legacy:
        arm_namespace = "B2R1_BASELINE" if "BASELINE" in row["smoke_arm"] else "B2R1_TREATMENT"
        schedule.append({**row, "run_id": f"{row['family']}__{row['scenario_token']}__{arm_namespace}"})
    if len(schedule) != RUN_CAP or len({row["run_id"] for row in schedule}) != RUN_CAP:
        raise ValueError("B2.1 must construct exactly 48 unique versioned run IDs")
    return roster, schedule


def _runtime_binding(args: argparse.Namespace) -> Dict[str, Any]:
    required_paths = [args.nuplan_devkit_root, args.tuplan_garage_root, args.nuplan_data_root, args.nuplan_map_root, args.nuplan_exp_root, args.python_executable]
    missing = [str(path) for path in required_paths if not path.exists()]
    simulation_script = args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py"
    if not simulation_script.is_file():
        missing.append(str(simulation_script))
    if missing:
        raise FileNotFoundError(f"V3-bound runtime path(s) missing: {missing}")
    roster, _ = _validate_roster(args)
    replay = read_json(args.replay_contract)
    base_env, v3_env = build_runtime_environment(args), _v3_reference_environment()
    observed = {key: base_env.get(key, "") for key in (*ENVIRONMENT_KEYS, *THREAD_KEYS)}
    expected = {key: v3_env.get(key, "") for key in (*ENVIRONMENT_KEYS, *THREAD_KEYS)}
    mismatches = {key: {"expected_v3": expected[key], "observed_b2_1": observed[key]} for key in observed if observed[key] != expected[key]}
    executable_match = str(args.python_executable.resolve()) == str(Path(V3_DEFAULTS["python_executable"]).resolve())
    if not executable_match:
        mismatches["python_executable"] = {"expected_v3": str(Path(V3_DEFAULTS["python_executable"]).resolve()), "observed_b2_1": str(args.python_executable.resolve())}
    binding = replay.get("binding", {})
    db_match = str(roster.get("db_fingerprint_sha256")) == str(binding.get("db_fingerprint_sha256"))
    map_match = str(roster.get("map_fingerprint_sha256")) == str(binding.get("map_fingerprint_sha256"))
    seed_match = int(roster.get("master_seed", -1)) == MASTER_SEED == int(binding.get("master_seed", -2))
    if not (db_match and map_match and seed_match):
        mismatches["frozen_fingerprints_or_seed"] = {"roster_db": roster.get("db_fingerprint_sha256"), "replay_db": binding.get("db_fingerprint_sha256"), "roster_map": roster.get("map_fingerprint_sha256"), "replay_map": binding.get("map_fingerprint_sha256"), "roster_seed": roster.get("master_seed"), "replay_seed": binding.get("master_seed")}
    identity = _nuplan_identity(args, base_env)
    return {
        "schema_version": "r1_official_smoke_runtime_environment_binding_v1.1",
        "status": "MATCHES_V3_BOUND_RUNTIME" if not mismatches else "DOES_NOT_MATCH_V3_BOUND_RUNTIME",
        "purpose": "B2_1_EXECUTION_ENVIRONMENT_BINDING_CORRECTION_ONLY",
        "v3_reference": {"runtime_validation_authorization": "docs/stageR/r1/r1_runtime_determinism_validation_v3_authorization_v1.0.json", "defaults": {key: str(value) for key, value in V3_DEFAULTS.items()}},
        "environment": observed,
        "python_executable": str(args.python_executable.resolve()),
        "nuplan": identity,
        "db_fingerprint_sha256": roster.get("db_fingerprint_sha256"),
        "map_fingerprint_sha256": roster.get("map_fingerprint_sha256"),
        "master_seed": MASTER_SEED,
        "checks": {"environment_and_threads_match_v3": not any(key in (*ENVIRONMENT_KEYS, *THREAD_KEYS) for key in mismatches), "python_executable_matches_v3": executable_match, "db_fingerprint_matches_replay": db_match, "map_fingerprint_matches_replay": map_match, "master_seed_matches_replay": seed_match, "mismatches": mismatches},
    }


def _preflight(args: argparse.Namespace) -> Dict[str, Any]:
    required = [args.roster, args.scope_amendment, args.selector_contract, args.replay_contract, args.environment_binding]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"recovery preflight required artifact(s) missing: {missing}")
    environment_binding = read_json(args.environment_binding)
    if environment_binding.get("status") != "MATCHES_V3_BOUND_RUNTIME":
        raise ValueError("B2.1 cannot preflight an environment that does not match V3")
    live_binding = _runtime_binding(args)
    if live_binding.get("status") != "MATCHES_V3_BOUND_RUNTIME":
        raise ValueError("live B2.1 environment no longer matches V3")
    roster, schedule = _validate_roster(args)
    scope, replay = read_json(args.scope_amendment), read_json(args.replay_contract)
    if scope.get("only_legal_scope", {}).get("total_official_closed_loop_runs") != RUN_CAP or replay.get("official_replay") != "READY_FOR_TECHNICAL_SMOKE_REVIEW":
        raise ValueError("frozen smoke scope or V3 replay contract is not executable")
    if not issubclass(R1OfficialTechnicalSmokePlanner, AbstractPlanner):
        raise TypeError("smoke planner is not an AbstractPlanner")
    base_env = build_runtime_environment(args)
    if any(not base_env.get(key) for key in (*ENVIRONMENT_KEYS, *THREAD_KEYS)):
        raise RuntimeError("complete stage7c environment assembly did not provide required bindings")
    planner_config_path = ROOT / "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke.yaml"
    planner_config = OmegaConf.load(planner_config_path)
    if planner_config.get("r1_official_technical_smoke", {}).get("_target_") != "tools.r1_official_technical_smoke_planner.R1OfficialTechnicalSmokePlanner":
        raise ValueError("Hydra smoke planner config no longer names the frozen planner")
    entry = roster["entries"][0]
    old_env = dict(os.environ)
    try:
        os.environ.update(base_env)
        planner = R1OfficialTechnicalSmokePlanner(type("FrozenScenario", (), {"token": entry["scenario_token"]})(), str(args.roster), str(entry["family"]), str(entry["arms"][0]), str(args.official_output.parent / "preflight_trace_not_executed"))
        if not planner.name():
            raise RuntimeError("smoke planner construction returned an empty name")
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    representative = {(row["family"], "BASELINE" if "BASELINE" in row["smoke_arm"] else "TREATMENT"): row for row in schedule}
    commands = {f"{family}_{arm}": _command_for(args, representative[(family, arm)], args.official_output / "preflight_commands" / f"{family}_{arm}") for family in ("R-HLC", "R-TSB") for arm in ("BASELINE", "TREATMENT")}
    if not all(command[1] == str((args.nuplan_devkit_root / "nuplan/planning/script/run_simulation.py").resolve()) for command in commands.values()):
        raise RuntimeError("preflight commands do not target bound run_simulation.py")
    synthetic_context = {"family": "R-TSB", "scenario_token": "synthetic", "map_version": "synthetic", "route_fingerprint": "synthetic", "initial_state_fingerprint": "synthetic", "map_location": "synthetic", "road_class": "synthetic", "log_id": "synthetic", "query_version": "recovery_preflight", "t_anchor_s": 1.0, "frames": [{"time_s": round(i * 0.1, 6), "ego_valid": True, "map_valid": True, "current_required_lane_valid": True, "speed_mps": 10.0, "lane_offset_m": 0.0, "legal_projected_dynamic_vehicle_count": 0, "slots": {}, "front": {"valid": False}} for i in range(10)], "hazard_multi_hot": ["NONE_OBSERVED"]}
    if not build_canonical_context_record(synthetic_context)["eligible"] or not callable(canonicalize_official_metrics) or not callable(_evaluate_pair):
        raise RuntimeError("frozen context, metric, mechanism/F-match or endpoint evaluator is not callable")
    cap_test = RecoveryBudget(schedule, args.official_output / "not_created_by_preflight.json", [dict(row) for row in schedule])
    cap = cap_test.assert_49th_rejected()
    return {
        "schema_version": "r1_official_technical_smoke_recovery_preflight_v1.1",
        "status": "PASS_COMPLETE_EXECUTION_PATH_NO_OFFICIAL_RUN_BUDGET_CONSUMED",
        "actual_official_run_count": 0,
        "official_run_budget_claimed": 0,
        "environment_binding_sha256": sha256_file(args.environment_binding),
        "environment_assembly": "stage7c_environment(argparse.Namespace(all_five_v3_roots))",
        "planner_abstract_planner_pass": True,
        "planner_construction_with_roster_config_and_environment_pass": True,
        "run_simulation_exists": True,
        "python_executable_exists": args.python_executable.is_file(),
        "hydra_planner_config": str(planner_config_path.resolve()),
        "hydra_planner_config_parse_pass": True,
        "command_generation": {key: command for key, command in commands.items()},
        "roster_checks": {"roster_sha256": sha256_file(args.roster), "entries": len(roster["entries"]), "unique_tokens": len({row["scenario_token"] for row in roster["entries"]}), "unique_logs": len({row["log_id"] for row in roster["entries"]}), "R-HLC": sum(row["family"] == "R-HLC" for row in roster["entries"]), "R-TSB": sum(row["family"] == "R-TSB" for row in roster["entries"]), "schedule": len(schedule), "unique_run_ids": len({row["run_id"] for row in schedule})},
        "frozen_contract_hashes": _contract_hashes(args),
        "evaluator_checks": {"official_metric_canonicalizer_callable": True, "context_callable": True, "mechanism_f_match_endpoint_callable": True},
        "49th_pre_run_claim": cap,
        "selector_rerun": "FORBIDDEN_NOT_PERFORMED",
        "official_simulation_started": False,
        "git": {"commit": _git(["rev-parse", "HEAD"]), "tree": _git(["rev-parse", "HEAD^{tree}"])}
    }


def _contract_hashes(args: argparse.Namespace) -> Dict[str, str]:
    paths = {
        "historical_b2_failure_manifest": args.historical_manifest, "scope_amendment": args.scope_amendment, "selector_contract": args.selector_contract,
        "roster": args.roster, "replay_contract": args.replay_contract, "environment_binding": args.environment_binding,
        "corrected_executor": Path(__file__), "legacy_frozen_science_executor": ROOT / "tools/r1_run_official_compliant_technical_smoke_v1.py",
        "smoke_planner": ROOT / "tools/r1_official_technical_smoke_planner.py", "hlc_generator": R1_DIR / "r1_hlc_generator_v2_contract_v1.0.json",
        "tsb_generator": R1_DIR / "r1_tsb_generator_v2_contract_v1.0.json", "hlc_endpoint": R1_DIR / "r1_hlc_generator_endpoint_validity_v1.0.json",
        "context_mechanism_contract": R1_DIR / "r1_context_contract_v1.0.json", "metric_canonicalizer": ROOT / "tools/r1_official_metric_canonicalizer.py",
        "simulation_config": ROOT / "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke.yaml",
    }
    return {key: _path_hash(path) for key, path in paths.items()}


@dataclass
class RecoveryBudget:
    schedule: Sequence[Mapping[str, str]]
    path: Path
    records: List[Dict[str, Any]]

    @classmethod
    def create(cls, schedule: Sequence[Mapping[str, str]], path: Path) -> "RecoveryBudget":
        budget = cls(schedule=schedule, path=path, records=[])
        budget.flush()
        return budget

    def flush(self) -> None:
        write_json(self.path, {"schema_version": "r1_official_technical_smoke_budget_v1.1", "unit": "OFFICIAL_CLOSED_LOOP_RUN", "authorized_cap": RUN_CAP, "claimed_count": len(self.records), "records": self.records})

    def claim(self, run: Mapping[str, str]) -> None:
        expected = {str(row["run_id"]) for row in self.schedule}
        if len(self.records) >= RUN_CAP:
            raise RuntimeError("OFFICIAL_CLOSED_LOOP_RUN cap 48 reached before simulation start")
        if str(run["run_id"]) not in expected or any(str(row.get("run_id")) == str(run["run_id"]) for row in self.records):
            raise RuntimeError(f"refusing unplanned or duplicate official run: {run['run_id']}")
        record = dict(run)
        record.update({"claim_status": "B2_1_CLAIMED_BEFORE_SIMULATION", "actual_run_number": len(self.records) + 1, "execution_status": "CLAIMED_NOT_STARTED", "official_command_return_code": "", "technical_failure_status": "", "technical_failure_reasons": "", "trace_sha256": "", "planner_binding_sha256": "", "canonical_metric_payload_sha256": "", "command_log_sha256": ""})
        self.records.append(record)
        self.flush()

    def complete(self, run_id: str, summary: Mapping[str, Any]) -> None:
        record = next(row for row in self.records if str(row["run_id"]) == run_id)
        record.update({"execution_status": "EXECUTED", "official_command_return_code": summary["official_command_return_code"], "technical_failure_status": summary["technical_failure_status"], "technical_failure_reasons": " | ".join(summary["technical_failure_reasons"]), "trace_sha256": summary.get("trace_sha256") or "", "planner_binding_sha256": summary.get("planner_binding_sha256") or "", "canonical_metric_payload_sha256": summary.get("metrics", {}).get("canonical_payload_sha256") or "", "command_log_sha256": summary.get("command_log_sha256") or ""})
        self.flush()

    def assert_49th_rejected(self) -> Dict[str, str]:
        saved = self.records
        self.records = [dict(row) for row in self.schedule]
        try:
            self.claim({"run_id": "R1_B2_1_FORBIDDEN_49TH_PRE_RUN_CLAIM"})
        except RuntimeError as exc:
            return {"status": "REJECTED_BEFORE_SIMULATION", "reason": str(exc)}
        finally:
            self.records = saved
        raise RuntimeError("49th B2.1 pre-run claim did not fail closed")


def _command_for(args: argparse.Namespace, run: Mapping[str, str], run_dir: Path) -> List[str]:
    command = _legacy_command_for(args, run, run_dir)
    return ["experiment_name=r1_official_compliant_technical_smoke_v1_1" if value == "experiment_name=r1_official_compliant_technical_smoke_v1" else value for value in command]


def _write_result_tables(args: argparse.Namespace, pair_results: Sequence[Mapping[str, Any]], budget: RecoveryBudget) -> List[Dict[str, Any]]:
    pair_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []
    for item in pair_results:
        identity = item["context"]["identity"]
        pair_rows.append({"scenario_token": item["scenario_token"], "log_id": item["log_id"], "family": item["family"], "baseline_run_id": item["baseline_run_id"], "treatment_run_id": item["treatment_run_id"], "technical_execution_pass": item["technical_execution_pass"], "context_identity_pass": identity["pair_context_identity_pass"], "mechanism_pair_status": item["mechanism"]["pair"]["status"], "mechanism_pair_pass": item["mechanism"]["pair"]["pass"], "primary_f_match_status": item["f_match"]["status"], "primary_f_match_pass": item["f_match"]["pass"], "secondary_heading_change_abs_total_delta": item["secondary_heading_change_abs_total_delta"], "endpoint_status": item["endpoint"]["status"], "endpoint_pass": item["endpoint"]["pass"], "engineering_status": item["engineering"]["status"], "engineering_pass": item["engineering"]["pass"], "pair_readiness": item["pair_readiness"]})
        context_rows.append({"scenario_token": item["scenario_token"], "log_id": item["log_id"], "family": item["family"], "baseline_run_id": item["baseline_run_id"], "treatment_run_id": item["treatment_run_id"], "raw_history_canonical_hash_equal": identity["fields"]["pre_context_raw_hash"], "canonical_context_json_hash_equal": identity["fields"]["canonical_context_json_hash"], "pair_context_identity_pass": identity["pair_context_identity_pass"], "baseline_raw_history_canonical_hash": item["context"]["baseline"]["pre_context_raw_hash"], "treatment_raw_history_canonical_hash": item["context"]["treatment"]["pre_context_raw_hash"], "baseline_canonical_context_json_hash": item["context"]["baseline"]["canonical_context_json_hash"], "treatment_canonical_context_json_hash": item["context"]["treatment"]["canonical_context_json_hash"]})
        base_safety, treatment_safety = item["baseline_safety"], item["treatment_safety"]
        safety_rows.append({"scenario_token": item["scenario_token"], "log_id": item["log_id"], "family": item["family"], "baseline_run_id": item["baseline_run_id"], "treatment_run_id": item["treatment_run_id"], "baseline_at_fault_collision_count": base_safety["collision"]["number_of_all_at_fault_collisions_stat_value"], "treatment_at_fault_collision_count": treatment_safety["collision"]["number_of_all_at_fault_collisions_stat_value"], "baseline_drivable_area_compliance": base_safety["drivable_area"]["drivable_area_compliance_stat_value"], "treatment_drivable_area_compliance": treatment_safety["drivable_area"]["drivable_area_compliance_stat_value"], "baseline_safety_pass": base_safety["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0 and base_safety["drivable_area"]["drivable_area_compliance_stat_value"], "treatment_safety_pass": treatment_safety["collision"]["number_of_all_at_fault_collisions_stat_value"] == 0 and treatment_safety["drivable_area"]["drivable_area_compliance_stat_value"], "pair_safety_pass": item["safety"]["pass"]})
    _write_csv(args.ledger, budget.records, LEDGER_FIELDS)
    _write_csv(args.pair_metrics, pair_rows, PAIR_FIELDS)
    _write_csv(args.context_identity, context_rows, CONTEXT_FIELDS)
    _write_csv(args.safety, safety_rows, SAFETY_FIELDS)
    family_rows: List[Dict[str, Any]] = []
    for family in ("R-HLC", "R-TSB"):
        rows = [row for row in pair_rows if row["family"] == family]
        all_pass = len(rows) == 12 and all(row["pair_readiness"] == "PAIR_ALL_REQUIRED_GATES_PASS" for row in rows)
        family_rows.append({"family": family, "required_pairs": 12, "completed_pairs": len(rows), "technical_execution_pass_pairs": sum(bool(row["technical_execution_pass"]) for row in rows), "context_identity_pass_pairs": sum(bool(row["context_identity_pass"]) for row in rows), "mechanism_pair_pass_pairs": sum(bool(row["mechanism_pair_pass"]) for row in rows), "primary_f_match_pass_pairs": sum(bool(row["primary_f_match_pass"]) for row in rows), "endpoint_pass_pairs": sum(bool(row["endpoint_pass"]) for row in rows) if family == "R-HLC" else "NOT_APPLICABLE", "engineering_pass_pairs": sum(bool(row["engineering_pass"]) for row in rows) if family == "R-HLC" else "DIAGNOSTIC_ONLY", "safety_pass_pairs": sum(bool(row["pair_safety_pass"]) for row in safety_rows if row["family"] == family), "readiness": "READY_FOR_FORMAL_DEVELOPMENT_ROSTER_REVIEW" if all_pass else "NOT_READY_FOR_FORMAL_DEVELOPMENT_ROSTER", "reason": "ALL_12_REQUIRED_GATES_PASS" if all_pass else "ONE_OR_MORE_FROZEN_REQUIRED_GATES_NOT_MET"})
    _write_csv(args.family_summary, family_rows, ("family", "required_pairs", "completed_pairs", "technical_execution_pass_pairs", "context_identity_pass_pairs", "mechanism_pair_pass_pairs", "primary_f_match_pass_pairs", "endpoint_pass_pairs", "engineering_pass_pairs", "safety_pass_pairs", "readiness", "reason"))
    return family_rows


def _authorize(args: argparse.Namespace) -> Dict[str, Any]:
    for path in (args.preflight, args.environment_binding, args.status_correction):
        if not path.is_file():
            raise FileNotFoundError(f"cannot authorize without required recovery artifact: {path}")
    preflight, environment, correction = read_json(args.preflight), read_json(args.environment_binding), read_json(args.status_correction)
    if preflight.get("status") != "PASS_COMPLETE_EXECUTION_PATH_NO_OFFICIAL_RUN_BUDGET_CONSUMED" or environment.get("status") != "MATCHES_V3_BOUND_RUNTIME":
        raise ValueError("B2.1 authorization requires a passing complete-path preflight and V3 environment binding")
    if correction.get("R1_RESIDUAL_BENCHMARK_ENABLEMENT") != "NOT_EVALUABLE_DUE_TO_PRE_SIMULATION_TECHNICAL_FAILURE":
        raise ValueError("historical B2 scientific status has not been corrected")
    roster, schedule = _validate_roster(args)
    binding = _contract_hashes(args)
    binding.update({"recovery_preflight": sha256_file(args.preflight), "historical_status_correction": sha256_file(args.status_correction), "authorization_source_local_git_commit_sha": _git(["rev-parse", "HEAD"]), "authorization_source_local_git_tree_sha": _git(["rev-parse", "HEAD^{tree}"]), "db_fingerprint_sha256": roster["db_fingerprint_sha256"], "map_fingerprint_sha256": roster["map_fingerprint_sha256"], "master_seed": MASTER_SEED, "run_cap": RUN_CAP})
    authority = {"schema_version": "r1_official_compliant_technical_smoke_authorization_v1.1", "status": "AUTHORIZED_ONCE_AFTER_ZERO_BUDGET_RECOVERY_PREFLIGHT", "authorization_scope": "R1_PHASE_B2_1_FRESH_RECOVERY_BATCH_ONLY", "owner_decision": "R1_OFFICIAL_COMPLIANT_TECHNICAL_SMOKE_V1_1=AUTHORIZED_ONCE_AFTER_ZERO_BUDGET_RECOVERY_PREFLIGHT", "precondition": {"preflight_path": str(args.preflight.relative_to(ROOT)), "preflight_sha256": sha256_file(args.preflight), "preflight_status": preflight["status"], "preflight_official_run_count": 0, "environment_binding_path": str(args.environment_binding.relative_to(ROOT)), "environment_binding_sha256": sha256_file(args.environment_binding), "environment_binding_status": environment["status"]}, "binding": binding, "historical_b2_record": {"manifest_sha256": sha256_file(args.historical_manifest), "claim_label": "HISTORICAL_B2_PRE_SIMULATION_TECHNICAL_CLAIM", "simulator_status": "SIMULATOR_NOT_STARTED", "b2_1_evidence": "NOT_PART_OF_B2_1_EVIDENCE", "raw_budget_modified": False}, "frozen_schedule": {"R-HLC": ["HLC_BASELINE_DECISIVE_MONOTONIC_LANE_CHANGE", "HLC_TREATMENT_HLC_GEN_V2_OPTION_B"], "R-TSB": ["TSB_BASELINE_SINGLE_CONTINUOUS_BRAKING", "TSB_TREATMENT_TSB_GEN_V2_OPTION_A"], "scenario_count": 24, "arms_per_scenario": 2, "official_closed_loop_runs": len(schedule), "run_id_namespace": "B2R1", "49th_plus_one_pre_run_claim": "MUST_REJECT_BEFORE_SIMULATION"}, "failure_policy": {"technical_failure": "STOP_ENTIRE_B2_1_BATCH_NO_REPLACEMENT_NO_RERUN", "scientific_or_generator_gate_failure": "RECORD_PAIR_AND_CONTINUE_FROZEN_FULL_SCHEDULE", "outcome_driven_threshold_or_generator_change": "FORBIDDEN"}, "scientific_protocol_deviation": "NO__EXECUTION_ENVIRONMENT_BINDING_CORRECTION_ONLY", "continued_prohibitions": ["selector_rerun", "scenario_replacement", "formal_development_rollout", "RBR_A_B_C_training", "representation_readout", "BDD", "probe", "new_planner_rollout"]}
    _write_new_json(args.authorization, authority)
    _write_new_text(args.authorization_report, "# R1 官方合规技术 Smoke 授权 v1.1\n\n本授权仅允许一次新的 B2.1 `48` run 官方闭环批次。它绑定通过的完整执行路径预检、V3 一致的显式运行时环境以及完全不变的 24-scenario frozen roster。历史 B2 的单条 pre-simulation claim 仅作为技术失败记录，不计入 B2.1 证据或新额度。\n\n`SCIENTIFIC_PROTOCOL_DEVIATION = NO`：本次仅修正 `stage7c_environment(args)` 的环境装配绑定；未改 roster、selector、planner、生成器、context、mechanism、F-match、endpoint 或 safety 规则。\n")
    return authority


def _validate_authorization(args: argparse.Namespace, roster: Mapping[str, Any]) -> Dict[str, Any]:
    authority = read_json(args.authorization)
    if authority.get("status") != "AUTHORIZED_ONCE_AFTER_ZERO_BUDGET_RECOVERY_PREFLIGHT":
        raise ValueError("B2.1 one-time recovery authorization is absent or invalid")
    expected = _contract_hashes(args)
    expected.update({"recovery_preflight": _path_hash(args.preflight), "historical_status_correction": _path_hash(args.status_correction), "db_fingerprint_sha256": roster["db_fingerprint_sha256"], "map_fingerprint_sha256": roster["map_fingerprint_sha256"], "master_seed": MASTER_SEED, "run_cap": RUN_CAP})
    binding = authority.get("binding", {})
    mismatches = {key: {"expected": value, "actual": binding.get(key)} for key, value in expected.items() if binding.get(key) != value}
    if mismatches:
        raise ValueError(f"B2.1 authorization binding mismatch: {mismatches}")
    if read_json(args.environment_binding).get("status") != "MATCHES_V3_BOUND_RUNTIME":
        raise ValueError("B2.1 execution environment binding is not V3-matched")
    if _runtime_binding(args).get("status") != "MATCHES_V3_BOUND_RUNTIME":
        raise ValueError("live environment binding changed after B2.1 authorization")
    return authority


def _execute(args: argparse.Namespace) -> Dict[str, Any]:
    result_paths = (args.ledger, args.pair_metrics, args.context_identity, args.safety, args.family_summary, args.manifest)
    if args.official_output.exists() or any(path.exists() for path in result_paths):
        raise FileExistsError("refusing to overwrite B2.1 official output or result artifact")
    roster, schedule = _validate_roster(args)
    authority = _validate_authorization(args, roster)
    args.official_output.mkdir(parents=True, exist_ok=False)
    budget = RecoveryBudget.create(schedule, args.official_output / "official_run_budget_v1.1.json")
    base_env = build_runtime_environment(args)
    summaries: Dict[str, Dict[str, Any]] = {}
    stopped_reason: str | None = None
    for run in schedule:
        budget.claim(run)
        run_dir = args.official_output / "runs" / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        environment = dict(base_env)
        environment.update({"R1_OFFICIAL_TECHNICAL_SMOKE_ROSTER": str(args.roster.resolve()), "R1_OFFICIAL_TECHNICAL_SMOKE_FAMILY": run["family"], "R1_OFFICIAL_TECHNICAL_SMOKE_ARM": run["smoke_arm"], "R1_OFFICIAL_TECHNICAL_SMOKE_TRACE_DIR": str((run_dir / "trace").resolve())})
        command = _command_for(args, run, run_dir)
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=args.command_timeout_s)
            command_output, return_code = completed.stdout, completed.returncode
        except subprocess.TimeoutExpired as exc:
            command_output, return_code = (exc.stdout or "") + "\nB2.1_COMMAND_TIMEOUT\n", 124
        (run_dir / "official_run.log").write_text(command_output, encoding="utf-8")
        summary = _technical_summary(run, return_code, run_dir)
        summaries[run["run_id"]] = summary
        budget.complete(run["run_id"], summary)
        if summary["technical_failure_status"] != "NO_TECHNICAL_FAILURE":
            stopped_reason = f"TECHNICAL_FAILURE:{run['run_id']}"
            break
    pair_results: List[Dict[str, Any]] = []
    if stopped_reason is None:
        entries = {str(row["scenario_token"]): row for row in roster["entries"]}
        for token, entry in entries.items():
            runs = [row for row in schedule if row["scenario_token"] == token]
            baseline = summaries[next(row["run_id"] for row in runs if "BASELINE" in row["smoke_arm"])]
            treatment = summaries[next(row["run_id"] for row in runs if "TREATMENT" in row["smoke_arm"])]
            pair_results.append(_evaluate_pair(entry, baseline, treatment))
    family_rows = _write_result_tables(args, pair_results, budget)
    manifest = {"schema_version": "r1_official_technical_smoke_execution_manifest_v1.1", "status": "COMPLETE" if stopped_reason is None else "STOPPED_TECHNICAL_FAILURE", "stopped_reason": stopped_reason, "actual_official_run_count": len(budget.records), "authorized_run_cap": RUN_CAP, "authorization_sha256": sha256_file(args.authorization), "roster_sha256": sha256_file(args.roster), "historical_b2_evidence": "NOT_PART_OF_B2_1_EVIDENCE", "run_budget": {"path": str((args.official_output / "official_run_budget_v1.1.json").resolve()), "49th_pre_run_claim": budget.assert_49th_rejected() if len(budget.records) == RUN_CAP else "NOT_REACHED_DUE_TO_TECHNICAL_FAILURE"}, "technical_failure_count": sum(row["technical_failure_status"] != "NO_TECHNICAL_FAILURE" for row in budget.records), "pair_result_count": len(pair_results), "family_readiness": family_rows, "raw_output_directory": str(args.official_output.resolve()), "raw_output_directory_committed": False, "scientific_protocol_deviation": "NO__EXECUTION_ENVIRONMENT_BINDING_CORRECTION_ONLY", "formal_development_rollout_authorized": False, "rbr_training_authorized": False, "git": {"commit": _git(["rev-parse", "HEAD"]), "tree": _git(["rev-parse", "HEAD^{tree}"])}}
    _write_new_json(args.manifest, manifest)
    return manifest


def _correct_historical_status(args: argparse.Namespace) -> Dict[str, Any]:
    historical = read_json(args.historical_manifest)
    expected = {"budget_claim_count": 1, "official_simulator_command_start_count": 0, "actual_official_closed_loop_run_count": 0}
    if any(historical.get(key) != value for key, value in expected.items()):
        raise ValueError("historical B2 manifest is not the declared one-claim, pre-simulation technical stop")
    correction = {"schema_version": "r1_official_technical_smoke_b2_status_correction_v1.0", "historical_b2_manifest_sha256": sha256_file(args.historical_manifest), "historical_b2": {**expected, "claim_record_label": "HISTORICAL_B2_PRE_SIMULATION_TECHNICAL_CLAIM", "simulator_status": "SIMULATOR_NOT_STARTED", "b2_1_evidence": "NOT_PART_OF_B2_1_EVIDENCE", "mechanism": "NOT_EVALUABLE", "F_match": "NOT_EVALUABLE", "endpoint": "NOT_EVALUABLE", "safety": "NOT_EVALUABLE"}, "R1_RESIDUAL_BENCHMARK_ENABLEMENT": "NOT_EVALUABLE_DUE_TO_PRE_SIMULATION_TECHNICAL_FAILURE", "RECOVERY_ACTION": "TECHNICAL_EXECUTION_PATH_CORRECTION_REQUIRED", "FORMAL_DEVELOPMENT_ROSTER": "NOT_READY", "scientific_protocol_deviation": "NO__EXECUTION_ENVIRONMENT_ASSEMBLY_DEFECT", "historical_v0_7_preserved": True}
    _write_new_json(args.status_correction, correction)
    return correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correct-b2-status", "environment-binding", "recovery-preflight", "authorize", "execute"), required=True)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--scope-amendment", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--selector-contract", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--replay-contract", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--historical-manifest", type=Path, default=DEFAULT_HISTORICAL_MANIFEST)
    parser.add_argument("--status-correction", type=Path, default=DEFAULT_STATUS_CORRECTION)
    parser.add_argument("--environment-binding", type=Path, default=DEFAULT_ENVIRONMENT_BINDING)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--authorization-report", type=Path, default=DEFAULT_AUTHORIZATION_REPORT)
    parser.add_argument("--official-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--pair-metrics", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--context-identity", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY)
    parser.add_argument("--family-summary", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--nuplan-devkit-root", type=Path, default=V3_DEFAULTS["nuplan_devkit_root"])
    parser.add_argument("--tuplan-garage-root", type=Path, default=V3_DEFAULTS["tuplan_garage_root"])
    parser.add_argument("--nuplan-data-root", type=Path, default=V3_DEFAULTS["nuplan_data_root"])
    parser.add_argument("--nuplan-map-root", type=Path, default=V3_DEFAULTS["nuplan_map_root"])
    parser.add_argument("--nuplan-exp-root", type=Path, default=V3_DEFAULTS["nuplan_exp_root"])
    parser.add_argument("--python-executable", type=Path, default=V3_DEFAULTS["python_executable"])
    parser.add_argument("--command-timeout-s", type=int, default=V3_DEFAULTS["command_timeout_s"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "correct-b2-status":
        result = _correct_historical_status(args)
    elif args.mode == "environment-binding":
        result = _runtime_binding(args)
        _write_new_json(args.environment_binding, result)
    elif args.mode == "recovery-preflight":
        result = _preflight(args)
        _write_new_json(args.preflight, result)
    elif args.mode == "authorize":
        result = _authorize(args)
    else:
        result = _execute(args)
    print(json.dumps({"status": result.get("status"), "actual_official_run_count": result.get("actual_official_run_count"), "stopped_reason": result.get("stopped_reason")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
