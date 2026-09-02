#!/usr/bin/env python3
"""B2.9-E executor sharing one full nuPlan lifecycle across canary and future official runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_b2_9_e_official_run_lifecycle import run_one_with_full_nuplan_lifecycle  # noqa: E402
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v3.1.json"
PAIR_BINDINGS = R1 / "r1_b2_9_e_frozen_pair_evaluation_bindings_v2.1.json"
FINAL_MANIFEST = R1 / "r1_b2_9_e_final_execution_binding_manifest_v2.1.json"
CANARY_ROSTER = R1 / "r1_b2_9_c_cross_family_engineering_canary_roster_v1.0.json"
CANARY_SOURCE_LEDGER = R1 / "r1_b2_9_c_cross_family_canary_run_ledger_v1.0.json"
ROSTER_SHA = "efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6"
SCHEDULE_SHA = "99f44095c27319b746921376d2549a00186303298b5266ff45dd008a98c08455"
PAIR_BINDING_SHA = "a606a87b01cd1fdd340070fca7e77170b6e0782aafa1e7c19ab6c91228cc9fa6"
CANARY_TOKENS = {"R-HLC": "b1be12bca092597a", "R-TSB": "b486f9cf33a85455"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any], *, allow_update: bool = False) -> None:
    if path.exists() and not allow_update:
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class FrozenBudgetLedger:
    cap: int = 48
    claimed: List[str] = field(default_factory=list)

    def claim(self, run_id: str) -> None:
        if len(self.claimed) >= self.cap:
            raise RuntimeError("HARD_FAIL_BEFORE_FULL_LIFECYCLE_CAP_48")
        if run_id in self.claimed:
            raise RuntimeError("HARD_FAIL_BEFORE_FULL_LIFECYCLE_DUPLICATE_CLAIM")
        self.claimed.append(run_id)

    def reject_49th(self) -> str:
        saved = list(self.claimed)
        self.claimed = [f"dry-claim-{index}" for index in range(48)]
        try:
            self.claim("forbidden-49th")
        except RuntimeError as exc:
            return str(exc)
        finally:
            self.claimed = saved
        raise RuntimeError("49TH_CLAIM_NOT_REJECTED")


def _load_frozen() -> Tuple[List[Mapping[str, Any]], Dict[Tuple[str, str], Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    if sha256(ROSTER) != ROSTER_SHA:
        raise ValueError("STOP_BEFORE_SIMULATION_IMMUTABLE_ROSTER_V3_SHA_MISMATCH")
    if sha256(SCHEDULE) != SCHEDULE_SHA or sha256(PAIR_BINDINGS) != PAIR_BINDING_SHA:
        raise ValueError("STOP_BEFORE_SIMULATION_VERSIONED_SCHEDULE_OR_PAIR_SHA_MISMATCH")
    roster, schedule, pairs = read_json(ROSTER), read_json(SCHEDULE), read_json(PAIR_BINDINGS)
    runs = sorted(schedule["runs"], key=lambda row: int(row["run_order"]))
    entries = {(row["scenario_token"], row["log_id"]): row for row in roster["entries"]}
    pair_by_id = {row["pair_id"]: row for row in pairs["pairs"]}
    if len(runs) != 48 or [int(row["run_order"]) for row in runs] != list(range(1, 49)):
        raise ValueError("STOP_BEFORE_SIMULATION_48_RUN_ORDER_MISMATCH")
    if len(entries) != 24 or len(pair_by_id) != 24:
        raise ValueError("STOP_BEFORE_SIMULATION_FROZEN_CARDINALITY_MISMATCH")
    if any((row["scenario_token"], row["log_id"]) not in entries or row["pair_id"] not in pair_by_id for row in runs):
        raise ValueError("STOP_BEFORE_SIMULATION_FROZEN_LOOKUP_MISMATCH")
    return runs, entries, pair_by_id


def build_planner_from_frozen_binding(
    roster_path: str, schedule_path: str, run_id: str, trace_dir: str
) -> R1OfficialTechnicalSmokePlannerV3_1:
    roster_file, schedule_file = Path(roster_path), Path(schedule_path)
    if sha256(roster_file) != ROSTER_SHA or sha256(schedule_file) != SCHEDULE_SHA:
        raise ValueError("PLANNER_FROZEN_ROSTER_OR_SCHEDULE_SHA_MISMATCH")
    roster, schedule = read_json(roster_file), read_json(schedule_file)
    runs = [row for row in schedule["runs"] if row["run_id"] == run_id]
    if len(runs) != 1:
        raise ValueError(f"FROZEN_RUN_ID_MATCH_COUNT_MUST_EQUAL_ONE:{run_id}:{len(runs)}")
    run = runs[0]
    entries = [
        row for row in roster["entries"]
        if row["scenario_token"] == run["scenario_token"] and row["log_id"] == run["log_id"]
    ]
    if len(entries) != 1:
        raise ValueError(f"FROZEN_ROSTER_IDENTITY_MATCH_COUNT_MUST_EQUAL_ONE:{run_id}:{len(entries)}")
    entry = entries[0]
    if run["family"] != entry["family"] or run["arm"] not in entry["arms"]:
        raise ValueError("FROZEN_SCHEDULE_ROSTER_FAMILY_OR_ARM_MISMATCH")
    return R1OfficialTechnicalSmokePlannerV3_1(entry, str(run["family"]), str(run["arm"]), trace_dir)


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path, experiment: str) -> List[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r1_official_technical_smoke_v3_1_b2_9_e",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
        "worker=sequential",
        "disable_callback_parallelization=true",
        "scenario_builder.max_workers=1",
        "max_callback_workers=1",
        "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0",
        "gpu=false",
        "seed=2026082701",
        "run_metric=true",
        "enable_simulation_progress_bar=false",
        f"experiment_name={experiment}",
        f"job_name={run['run_id']}",
        f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _trace(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _construct_and_optionally_execute(
    *,
    run: Mapping[str, Any],
    entry: Mapping[str, Any],
    root: Path,
    planner: R1OfficialTechnicalSmokePlannerV3_1,
    execute: bool,
    experiment: str,
) -> Dict[str, Any]:
    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1

    run_id = str(run["run_id"])
    run_root, trace_dir, trace_file, raw = root / run_id, root / run_id / "trace", root / run_id / "trace/realized_current_ego.jsonl", root / run_id / "raw"
    if run_root.exists() or trace_file.exists():
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_OUTPUT_OR_TRACE_REUSE:{run_id}")
    if official_count(str(entry["db_path"]), str(run["scenario_token"])) != 1:
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_EXACT_RESOLUTION_NOT_ONE:{run_id}")
    trace_dir.mkdir(parents=True)
    os.environ.update(
        {
            "R1_B2_9_E_ROSTER": str(ROSTER),
            "R1_B2_9_E_SCHEDULE": str(SCHEDULE),
            "R1_B2_9_E_RUN_ID": run_id,
            "R1_B2_9_E_TRACE_DIR": str(trace_dir),
        }
    )
    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    with initialize_config_dir(config_dir=str(config_root)):
        cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw, experiment))
    resolved = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if "${" in resolved:
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_UNRESOLVED_HYDRA:{run_id}")
    if planner.__class__ is not R1OfficialTechnicalSmokePlannerV3_1:
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_PLANNER_CLASS_MISMATCH:{run_id}:{planner.__class__.__name__}")
    common = set_up_common_builder(cfg, f"{experiment}_build")
    callback_worker = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
    runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
    if len(runners) != 1:
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_RUNNER_COUNT_NOT_ONE:{run_id}:{len(runners)}")
    controller = runners[0]._simulation._time_controller
    if controller.__class__ is not R1Primary80ScientificTimeControllerV1:
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_CONTROLLER_CLASS_MISMATCH:{run_id}:{controller.__class__.__name__}")
    controller_iterations = int(controller.number_of_iterations())
    if controller_iterations != 81:
        raise RuntimeError(f"STOP_BEFORE_SIMULATION_PRIMARY80_CONTROLLER_NOT_81:{run_id}:{controller_iterations}")
    lifecycle = None
    if execute:
        lifecycle = run_one_with_full_nuplan_lifecycle(
            runners=runners,
            common_builder=common,
            profiler_name=f"{experiment}_running",
            cfg=cfg,
            run_output_root=run_root,
        )
    audit = {
        "run_id": run_id,
        "pair_id": run["pair_id"],
        "family": run["family"],
        "scenario_token": run["scenario_token"],
        "log_id": run["log_id"],
        "arm": run["arm"],
        "run_order": run.get("run_order"),
        "exact_scenario_resolution": 1,
        "full_hydra_config_resolved": True,
        "planner_class": planner.__class__.__name__,
        "time_controller_class": controller.__class__.__name__,
        "controller_number_of_iterations": controller_iterations,
        "runner_count": 1,
        "runner_constructed": True,
        "trace_file_pre_run": "ABSENT",
        "full_lifecycle_executed": execute,
        "run_runners_called": bool(execute),
        "lifecycle": lifecycle,
    }
    if execute:
        audit.update({"run_root": str(run_root), "trace_file": str(trace_file)})
    return audit


def _authorize(manifest: Path, authorization: Optional[Path]) -> None:
    if authorization is None:
        raise PermissionError("STOP_BEFORE_SIMULATION_NEW_PACKAGE_OWNER_REAUTHORIZATION_REQUIRED")
    owner, bound = read_json(authorization), read_json(manifest)
    if owner.get("OFFICIAL_SMOKE_AUTHORIZED") is not True:
        raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_REAUTHORIZATION_FALSE")
    if owner.get("final_execution_manifest_sha256") != sha256(manifest):
        raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_MANIFEST_SHA_MISMATCH")
    if bound.get("status") != "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_REAUTHORIZATION":
        raise PermissionError("STOP_BEFORE_SIMULATION_FINAL_MANIFEST_NOT_READY_FOR_REAUTHORIZATION")
    if bound.get("authorization", {}).get("OFFICIAL_SMOKE_AUTHORIZED") is not False:
        raise PermissionError("STOP_BEFORE_SIMULATION_MANIFEST_AUTHORIZATION_STATE_INVALID")
    for path_text, expected in bound.get("complete_transitive_component_sha256", {}).items():
        path = Path(path_text) if Path(path_text).is_absolute() else ROOT / path_text
        if not path.is_file() or sha256(path) != expected:
            raise PermissionError(f"STOP_BEFORE_SIMULATION_COMPONENT_SHA_MISMATCH:{path_text}")


def run_official_package(
    *, execute: bool = False, output_root: Optional[Path] = None, authorization: Optional[Path] = None, manifest: Path = FINAL_MANIFEST
) -> Dict[str, Any]:
    runs, entries, pair_by_id = _load_frozen()
    if execute:
        _authorize(manifest, authorization)
        if output_root is None:
            raise ValueError("EXECUTE_REQUIRES_EXPLICIT_FRESH_OUTPUT_ROOT")
        if output_root.exists():
            raise FileExistsError(f"STOP_BEFORE_SIMULATION_OUTPUT_ROOT_REUSE:{output_root}")
    ledger, completed, audits = FrozenBudgetLedger(), set(), []
    temporary = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="r1_b2_9_e_zero_run_")
        root = Path(temporary.name)
    else:
        root = output_root
    try:
        from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair

        for run in runs:
            run_id = str(run["run_id"])
            entry = entries[(run["scenario_token"], run["log_id"])]
            pair = pair_by_id[str(run["pair_id"])]
            trace_dir = root / run_id / "trace"
            os.environ.update({"R1_B2_9_E_ROSTER": str(ROSTER), "R1_B2_9_E_SCHEDULE": str(SCHEDULE), "R1_B2_9_E_RUN_ID": run_id, "R1_B2_9_E_TRACE_DIR": str(trace_dir)})
            planner = build_planner_from_frozen_binding(str(ROSTER), str(SCHEDULE), run_id, str(trace_dir))
            ledger.claim(run_id)
            audit = _construct_and_optionally_execute(run=run, entry=entry, root=root, planner=planner, execute=execute, experiment="r1_b2_9_e_frozen_48run_smoke")
            if pair["scenario_token"] != run["scenario_token"] or pair["log_id"] != run["log_id"]:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_PAIR_BINDING_IDENTITY_MISMATCH:{run_id}")
            audit["pair_binding_lookup"] = pair["pair_id"]
            audit["budget_claimed_before_full_lifecycle"] = True
            audits.append(audit)
            if execute:
                completed.add(run_id)
                if {pair["baseline_run_id"], pair["treatment_run_id"]}.issubset(completed):
                    result = evaluate_frozen_pair(pair_binding=pair, baseline_run_dir=root / pair["baseline_run_id"], treatment_run_dir=root / pair["treatment_run_id"])
                    write_json(root / f"{pair['pair_id']}__evaluation.json", result)
        return {
            "schema_version": "r1_b2_9_e_zero_run_final_construction_audit_v1.0",
            "status": "48_OF_48_ZERO_RUN_CONSTRUCTION_PASS" if not execute else "48_RUN_NEW_PACKAGE_EXECUTION_COMPLETE",
            "runs": audits,
            "counts": {"exact_resolutions": len(audits), "planner_v3_1_bindings": len(audits), "Primary80_controller_bindings": len(audits), "runner_constructions": len(audits), "pair_binding_lookups": len(audits)},
            "claim_49": ledger.reject_49th(),
            "runner_run_calls": 0 if not execute else len(completed),
            "run_runners_calls": 0 if not execute else len(completed),
            "simulation_started": bool(execute),
            "official_runs": len(completed),
            "consumed_real_budget": len(completed),
            "selector_invoked": False,
            "scientific_identity_changed": False,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def _canary_binding_by_token() -> Dict[str, Dict[str, Any]]:
    from tools.r1_b2_9_c_cross_family_canary import _build_pair_bindings

    roster, ledger = read_json(CANARY_ROSTER), read_json(CANARY_SOURCE_LEDGER)
    bindings = _build_pair_bindings(roster, ledger)
    return {str(binding["scenario_token"]): binding for binding in bindings}


def run_exact_lifecycle_canary(*, output_root: Path, ledger_path: Path) -> Dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("EXACT_LIFECYCLE_CANARY_OUTPUT_REUSE_FORBIDDEN")
    roster = read_json(CANARY_ROSTER)
    entries = {str(entry["scenario_token"]): entry for entry in roster["entries"]}
    source_bindings = _canary_binding_by_token()
    selected = [("R-HLC", entries[CANARY_TOKENS["R-HLC"]]), ("R-TSB", entries[CANARY_TOKENS["R-TSB"]])]
    for family, entry in selected:
        if entry.get("SCIENTIFIC_USE_FORBIDDEN") is not True or entry.get("PERMANENT_FUTURE_SELECTOR_EXCLUSION") is not True or entry["family"] != family:
            raise PermissionError(f"CANARY_IDENTITY_NOT_PERMANENTLY_SCIENTIFIC_FORBIDDEN:{entry['scenario_token']}")
    output_root.mkdir(parents=True)
    recovered_pre_simulation_only = ledger_path.exists()
    if recovered_pre_simulation_only:
        ledger = read_json(ledger_path)
        if ledger.get("status") != "RUNNING_FAIL_CLOSED" or ledger.get("runs") or ledger.get("pairs"):
            raise PermissionError("CANARY_LEDGER_REUSE_AFTER_ANY_ACTUAL_RUN_FORBIDDEN")
        ledger["pre_simulation_construction_failures"] = [
            {
                "attempt": "A01",
                "output_root": "outputs/r1_b2_9_e_exact_lifecycle_canary_v1",
                "run_id": "R1B29E-CANARY-01-R-HLC-BASELINE-A01",
                "failure": "HYDRA_ENVIRONMENT_BINDING_MISSING_BEFORE_RUNNER_CONSTRUCTION",
                "simulation_started": False,
                "run_runners_called": False,
                "trace_rows": 0,
                "counts_as_canary_run": False,
            }
        ]
        ledger["actual_run_output_root"] = str(output_root)
        ledger["status"] = "PRE_SIMULATION_BINDING_REPAIRED_ACTUAL_CANARY_RUNNING"
        write_json(ledger_path, ledger, allow_update=True)
    else:
        ledger = {
            "schema_version": "r1_b2_9_e_exact_lifecycle_canary_run_ledger_v1.0",
            "status": "RUNNING_FAIL_CLOSED",
            "execution_primitive": "tools.r1_b2_9_e_official_run_lifecycle.run_one_with_full_nuplan_lifecycle",
            "future_official_executor": "tools/r1_b2_9_e_execute_frozen_48run_smoke.py",
            "runs": [], "pairs": [],
            "selector_invoked": False, "scientific_identity_changed": False,
            "scientific_use_forbidden": True, "scientific_outcomes_descriptive_only": True,
            "reruns": 0, "pre_simulation_construction_failures": [],
            "actual_run_output_root": str(output_root),
        }
        write_json(ledger_path, ledger)
    from tools.r1_b2_8_r3_1_official_safety_adapter import adapt_official_safety
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair

    for identity_index, (family, entry) in enumerate(selected, 1):
        pair_id = f"R1B29E-CANARY-{identity_index:02d}-{family}"
        arms: Sequence[str] = entry["arms"]
        pair_runs = []
        for arm_index, arm in enumerate(arms):
            arm_label = "BASELINE" if arm_index == 0 else "TREATMENT"
            attempt_label = "A02" if recovered_pre_simulation_only else "A01"
            run = {"run_id": f"{pair_id}-{arm_label}-{attempt_label}", "pair_id": pair_id, "family": family, "scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "arm": arm, "run_order": identity_index * 2 - 1 + arm_index}
            trace_dir = output_root / run["run_id"] / "trace"
            planner = R1OfficialTechnicalSmokePlannerV3_1(entry, family, str(arm), str(trace_dir))
            audit = _construct_and_optionally_execute(run=run, entry=entry, root=output_root, planner=planner, execute=True, experiment="r1_b2_9_e_exact_lifecycle_canary")
            rows = _trace(Path(audit["trace_file"]))
            indices = [int(row["iteration_index"]) for row in rows]
            sources = {row.get("primary_measurement_source") for row in rows}
            if indices != list(range(80)) or sources != {"REALIZED_CURRENT_EGO"}:
                raise RuntimeError(f"CANARY_PRIMARY80_TRACE_INVALID:{run['run_id']}")
            safety = adapt_official_safety(Path(audit["run_root"]))
            audit.update({"status": "TECHNICAL_COMPLETE", "primary_trace_rows": 80, "primary_80_complete": True, "secondary_planner_calls": 0, "metric_callback_complete": True, "safety_adapter_structural_complete": True, "safety_pass_descriptive_only": bool(safety["frozen_arm_safety_pass"]), "SCIENTIFIC_USE_FORBIDDEN": True})
            ledger["runs"].append(audit)
            pair_runs.append(audit)
            write_json(ledger_path, ledger, allow_update=True)
        binding = dict(source_bindings[str(entry["scenario_token"])])
        binding.update({"pair_id": pair_id, "baseline_run_id": pair_runs[0]["run_id"], "treatment_run_id": pair_runs[1]["run_id"]})
        result = evaluate_frozen_pair(pair_binding=binding, baseline_run_dir=Path(pair_runs[0]["run_root"]), treatment_run_dir=Path(pair_runs[1]["run_root"]))
        pair_audit = {"pair_id": pair_id, "family": family, "dispatcher_complete": result["dispatch_status"] == "EVALUATED_NO_POSTHOC_PAIR_DELETION", "dispatch_status": result["dispatch_status"], "evaluator_status_descriptive_only": result["evaluation"]["status"], "scientific_outcome_used": False}
        ledger["pairs"].append(pair_audit)
        write_json(output_root / f"{pair_id}__descriptive_evaluation.json", result)
        write_json(ledger_path, ledger, allow_update=True)
    ledger["counts"] = {"runs": len(ledger["runs"]), "HLC_technical_complete": sum(row["family"] == "R-HLC" and row["status"] == "TECHNICAL_COMPLETE" for row in ledger["runs"]), "TSB_technical_complete": sum(row["family"] == "R-TSB" and row["status"] == "TECHNICAL_COMPLETE" for row in ledger["runs"]), "exact_80_traces": sum(bool(row["primary_80_complete"]) for row in ledger["runs"]), "metric_lifecycle_complete": sum(bool(row["metric_callback_complete"]) for row in ledger["runs"]), "safety_adapter_complete": sum(bool(row["safety_adapter_structural_complete"]) for row in ledger["runs"]), "dispatcher_complete": sum(bool(row["dispatcher_complete"]) for row in ledger["pairs"])}
    ledger["status"] = "4_OF_4_EXACT_LIFECYCLE_CANARY_PASS" if ledger["counts"] == {"runs": 4, "HLC_technical_complete": 2, "TSB_technical_complete": 2, "exact_80_traces": 4, "metric_lifecycle_complete": 4, "safety_adapter_complete": 4, "dispatcher_complete": 2} else "FAIL_CLOSED"
    ledger["official_scientific_simulation"] = False
    ledger["actual_simulation_reruns"] = 0
    ledger["RBR_A/B/C"] = "NOT_AUTHORIZED"
    write_json(ledger_path, ledger, allow_update=True)
    if ledger["status"] != "4_OF_4_EXACT_LIFECYCLE_CANARY_PASS":
        raise RuntimeError("EXACT_LIFECYCLE_CANARY_CLOSURE_FAILED")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Future scientific execution; requires new-package owner authorization")
    parser.add_argument("--exact-lifecycle-canary", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=FINAL_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute and args.exact_lifecycle_canary:
        raise ValueError("SCIENTIFIC_EXECUTION_AND_CANARY_ARE_MUTUALLY_EXCLUSIVE")
    if args.exact_lifecycle_canary:
        if args.output_root is None or args.output is None:
            raise ValueError("CANARY_REQUIRES_FRESH_OUTPUT_ROOT_AND_VERSIONED_LEDGER_OUTPUT")
        result = run_exact_lifecycle_canary(output_root=args.output_root, ledger_path=args.output)
    else:
        result = run_official_package(execute=args.execute, output_root=args.output_root, authorization=args.authorization, manifest=args.manifest)
        if args.output:
            write_json(args.output, result)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "runner_run_calls": result.get("runner_run_calls", len(result["runs"]) if args.exact_lifecycle_canary else 0), "run_runners_calls": result.get("run_runners_calls", len(result["runs"]) if args.exact_lifecycle_canary else 0)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
