#!/usr/bin/env python3
"""Final B2.9-D 48-run executor; zero-run construction is the default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402
from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v3.0.json"
PAIR_BINDINGS = R1 / "r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json"
FINAL_MANIFEST = R1 / "r1_b2_9_d_final_execution_binding_manifest_v2.0.json"
ROSTER_SHA = "efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6"
SCHEDULE_SHA = "47b5512bc235eb533d44bf3c8106c97ea5467533fe62d0902a23316e5827b0cf"
PAIR_BINDING_SHA = "d3ae32f1f41ff4656f08c3cf95534c89e8d726604a40078e2fd3c816795cea11"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


@dataclass
class FrozenBudgetLedger:
    """No retry, no duplicate claim, and cap-before-runner semantics."""

    cap: int = 48
    claimed: List[str] = field(default_factory=list)

    def claim(self, run_id: str) -> None:
        if len(self.claimed) >= self.cap:
            raise RuntimeError("HARD_FAIL_BEFORE_RUNNER_RUN_CAP_48")
        if run_id in self.claimed:
            raise RuntimeError("HARD_FAIL_BEFORE_RUNNER_RUN_DUPLICATE_CLAIM")
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
    if sha256(ROSTER) != ROSTER_SHA or sha256(SCHEDULE) != SCHEDULE_SHA or sha256(PAIR_BINDINGS) != PAIR_BINDING_SHA:
        raise ValueError("STOP_BEFORE_SIMULATION_V3_ROSTER_SCHEDULE_OR_PAIR_SHA_MISMATCH")
    roster, schedule, pairs = read_json(ROSTER), read_json(SCHEDULE), read_json(PAIR_BINDINGS)
    runs = sorted(schedule["runs"], key=lambda row: int(row["run_order"]))
    if [int(row["run_order"]) for row in runs] != list(range(1, 49)):
        raise ValueError("STOP_BEFORE_SIMULATION_RUN_ORDER_MISMATCH")
    entries = {(row["scenario_token"], row["log_id"]): row for row in roster["entries"]}
    pair_by_id = {row["pair_id"]: row for row in pairs["pairs"]}
    if len(runs) != 48 or len(entries) != 24 or len(pair_by_id) != 24:
        raise ValueError("STOP_BEFORE_SIMULATION_FROZEN_CARDINALITY_MISMATCH")
    if any((row["scenario_token"], row["log_id"]) not in entries or row["pair_id"] not in pair_by_id for row in runs):
        raise ValueError("STOP_BEFORE_SIMULATION_FROZEN_IDENTITY_OR_PAIR_LOOKUP_MISMATCH")
    return runs, entries, pair_by_id


def build_planner_from_frozen_binding(
    roster_path: str, schedule_path: str, run_id: str, trace_dir: str
) -> R1OfficialTechnicalSmokePlannerV3_1:
    roster_file, schedule_file = Path(roster_path), Path(schedule_path)
    if sha256(roster_file) != ROSTER_SHA or sha256(schedule_file) != SCHEDULE_SHA:
        raise ValueError("PLANNER_FROZEN_ROSTER_OR_SCHEDULE_SHA_MISMATCH")
    roster, schedule = read_json(roster_file), read_json(schedule_file)
    run_rows = [row for row in schedule["runs"] if row["run_id"] == run_id]
    if len(run_rows) != 1:
        raise ValueError(f"FROZEN_RUN_ID_MATCH_COUNT_MUST_EQUAL_ONE:{run_id}:{len(run_rows)}")
    run = run_rows[0]
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


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> List[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r1_official_technical_smoke_v3_1_b2_9_d",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
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
        "experiment_name=r1_b2_9_d_frozen_48run_smoke",
        f"job_name={run['run_id']}",
        f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _authorize(manifest: Path, authorization: Optional[Path]) -> None:
    if authorization is None:
        raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_AUTHORIZATION_REQUIRED")
    owner = read_json(authorization)
    if owner.get("OFFICIAL_SMOKE_AUTHORIZED") is not True:
        raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_AUTHORIZATION_FALSE")
    if owner.get("final_execution_manifest_sha256") != sha256(manifest):
        raise PermissionError("STOP_BEFORE_SIMULATION_OWNER_MANIFEST_SHA_MISMATCH")
    bound = read_json(manifest)
    if bound.get("status") != "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION":
        raise PermissionError("STOP_BEFORE_SIMULATION_FINAL_MANIFEST_NOT_READY")
    if bound.get("authorization", {}).get("OFFICIAL_SMOKE_AUTHORIZED") is not False:
        raise PermissionError("STOP_BEFORE_SIMULATION_FROZEN_MANIFEST_AUTHORIZATION_STATE_INVALID")
    for relative_path, expected in bound.get("complete_transitive_component_sha256", {}).items():
        component = ROOT / relative_path
        if not component.is_file() or sha256(component) != expected:
            raise PermissionError(f"STOP_BEFORE_SIMULATION_COMPONENT_SHA_MISMATCH:{relative_path}")


def run(
    execute: bool = False,
    output_root: Optional[Path] = None,
    authorization: Optional[Path] = None,
    manifest: Path = FINAL_MANIFEST,
) -> Dict[str, Any]:
    runs, entries, pair_by_id = _load_frozen()
    if execute:
        _authorize(manifest, authorization)
        if output_root is None:
            raise ValueError("EXECUTE_REQUIRES_EXPLICIT_FRESH_OUTPUT_ROOT")
        if output_root.exists():
            raise FileExistsError(f"STOP_BEFORE_SIMULATION_OUTPUT_ROOT_REUSE:{output_root}")
    official_env()
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import (
        build_callbacks_worker,
        build_simulation_callbacks,
    )
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair
    from tools.r1_primary80_scientific_time_controller_v1 import R1Primary80ScientificTimeControllerV1

    ledger = FrozenBudgetLedger()
    completed = set()
    audit = []
    temporary = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="r1_b2_9_d_zero_run_")
        root = Path(temporary.name)
    else:
        root = output_root
    try:
        config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
        for run_row in runs:
            run_id = str(run_row["run_id"])
            entry = entries[(run_row["scenario_token"], run_row["log_id"])]
            run_root, trace_dir, trace_file, raw = (
                root / run_id,
                root / run_id / "trace",
                root / run_id / "trace" / "realized_current_ego.jsonl",
                root / run_id / "raw",
            )
            if run_root.exists() or trace_file.exists():
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_OUTPUT_OR_TRACE_REUSE:{run_id}")
            if official_count(str(entry["db_path"]), str(run_row["scenario_token"])) != 1:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_EXACT_RESOLUTION_NOT_ONE:{run_id}")
            trace_dir.mkdir(parents=True)
            os.environ.update(
                {
                    "R1_B2_9_D_ROSTER": str(ROSTER),
                    "R1_B2_9_D_SCHEDULE": str(SCHEDULE),
                    "R1_B2_9_D_RUN_ID": run_id,
                    "R1_B2_9_D_TRACE_DIR": str(trace_dir),
                }
            )
            with initialize_config_dir(config_dir=str(config_root)):
                cfg = compose(config_name="default_simulation", overrides=_overrides(run_row, entry, raw))
            resolved = json.dumps(
                OmegaConf.to_container(cfg, resolve=True), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if "${" in resolved:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_UNRESOLVED_HYDRA:{run_id}")
            planner_configs = list(cfg.planner.values())
            if len(planner_configs) != 1:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_EXPECTED_ONE_HYDRA_PLANNER:{run_id}")
            planner = instantiate(planner_configs[0])
            if planner.__class__ is not R1OfficialTechnicalSmokePlannerV3_1:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_PLANNER_CLASS_MISMATCH:{run_id}:{planner.__class__.__name__}")
            common = set_up_common_builder(cfg, "r1_b2_9_d_zero_run" if not execute else "r1_b2_9_d_execute")
            callback_worker = build_callbacks_worker(cfg)
            callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
            runners = build_simulations(
                cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner]
            )
            if len(runners) != 1:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_RUNNER_COUNT_NOT_ONE:{run_id}:{len(runners)}")
            controller = runners[0]._simulation._time_controller
            if controller.__class__ is not R1Primary80ScientificTimeControllerV1:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_CONTROLLER_CLASS_MISMATCH:{run_id}:{controller.__class__.__name__}")
            controller_iterations = int(controller.number_of_iterations())
            if controller_iterations != 81:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_PRIMARY80_CONTROLLER_NOT_81:{run_id}:{controller_iterations}")
            pair = pair_by_id[str(run_row["pair_id"])]
            if pair["scenario_token"] != run_row["scenario_token"] or pair["log_id"] != run_row["log_id"]:
                raise RuntimeError(f"STOP_BEFORE_SIMULATION_PAIR_BINDING_IDENTITY_MISMATCH:{run_id}")
            ledger.claim(run_id)
            runner_run_called = False
            if execute:
                runner_run_called = True
                try:
                    runners[0].run()
                except Exception as exc:
                    raise RuntimeError(
                        f"STOP_REMAINING_SCHEDULE_TECHNICAL_FAILURE_NO_RETRY_NO_REPLACEMENT:{run_id}:{type(exc).__name__}"
                    ) from exc
                completed.add(run_id)
                if {pair["baseline_run_id"], pair["treatment_run_id"]}.issubset(completed):
                    result = evaluate_frozen_pair(
                        pair_binding=pair,
                        baseline_run_dir=root / pair["baseline_run_id"],
                        treatment_run_dir=root / pair["treatment_run_id"],
                    )
                    result_path = root / f"{pair['pair_id']}__evaluation.json"
                    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            audit.append(
                {
                    "run_id": run_id,
                    "run_order": int(run_row["run_order"]),
                    "exact_scenario_resolution": 1,
                    "full_hydra_config_resolved": True,
                    "planner_class": planner.__class__.__name__,
                    "time_controller_class": controller.__class__.__name__,
                    "controller_number_of_iterations": controller_iterations,
                    "runner_count": 1,
                    "runner_constructed": True,
                    "trace_file_pre_run": "ABSENT",
                    "pair_binding_lookup": pair["pair_id"],
                    "budget_claimed_before_runner_run": True,
                    "runner_run_called": runner_run_called,
                    "simulation_started": runner_run_called,
                }
            )
        return {
            "schema_version": "r1_b2_9_d_zero_run_final_construction_audit_v1.0",
            "status": "48_OF_48_ZERO_RUN_CONSTRUCTION_PASS" if not execute else "48_RUN_EXECUTION_COMPLETE",
            "runs": audit,
            "counts": {
                "exact_resolutions": len(audit),
                "planner_v3_1_bindings": len(audit),
                "Primary80_controller_bindings": len(audit),
                "runner_constructions": len(audit),
                "pair_binding_lookups": len(audit),
            },
            "claim_49": ledger.reject_49th(),
            "simulation_started": bool(execute),
            "runner_run_calls": len(completed),
            "official_runs": len(completed),
            "consumed_real_budget": len(completed),
            "dry_claim_count": len(ledger.claimed) if not execute else None,
            "technical_failure_policy": "STOP_REMAINING_SCHEDULE_NO_RETRY_NO_REPLACEMENT",
            "scientific_gate_failure_policy": "RECORD_AND_CONTINUE_NO_RETRY_NO_REPLACEMENT",
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=FINAL_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        execute=args.execute,
        output_root=args.output_root,
        authorization=args.authorization,
        manifest=args.manifest,
    )
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{args.output}")
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "runs": len(result["runs"]),
                "runner_run_calls": result["runner_run_calls"],
                "simulation_started": result["simulation_started"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
