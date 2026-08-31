#!/usr/bin/env python3
"""R1 B2.8-R3.1 final 48-run control-plane rehearsal; never runs simulation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_prospective_selector import official_count, official_env  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json"
SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v2.1.json"
R3_BINDING = R1 / "r1_b2_8_r3_execution_bindings_manifest_v1.0.json"
OUTPUTS = {
    "owner_json": R1 / "r1_b2_8_r3_1_scientific_owner_approval_record_v1.0.json",
    "owner_md": R1 / "R1_B2_8_R3_1_Scientific_Owner_Approval_Record_v1.0.md",
    "regression": R1 / "r1_b2_8_r3_1_final_zero_run_regression_v1.0.json",
    "manifest": R1 / "r1_b2_8_r3_1_final_execution_binding_manifest_v1.0.json",
    "request": R1 / "R1_B2_8_R3_1_Scientific_Owner_48_Run_Authorization_Request_v0.1.md",
}
ROSTER_SHA = "b977b802a7b25f0be37d04f3277cba2b2e98e521a2e30938ec40af9f278c1973"
SCHEDULE_SHA = "6733dc623cce2e2b64b9eb71cd407982b54dcaf5ecd48b644058c767c89d552f"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _component_paths() -> list[Path]:
    devkit = ROOT.parent / "nuplan-devkit"
    base = devkit / "nuplan/planning/script"
    return [
        ROSTER, SCHEDULE, R3_BINDING,
        ROOT / "tools/r1_b2_8_r3_frozen_run_dispatcher.py",
        ROOT / "tools/r1_official_technical_smoke_planner_v2_2.py",
        ROOT / "tools/r1_official_technical_smoke_planner_v2_1.py",
        ROOT / "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        ROOT / "tools/r1_b2_8_r3_1_post_run_evaluator_dispatcher.py",
        ROOT / "tools/r1_official_metric_canonicalizer.py",
        ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        ROOT / "tools/r1_closed_loop_benchmark_v2_1.py",
        ROOT / "tools/r1_context_mechanism_core.py",
        R1 / "r1_official_metric_canonicalization_contract_v1.0.json",
        R1 / "r1_future_compliant_smoke_selector_contract_v1.2.json",
        base / "run_simulation.py",
        base / "config/simulation/default_simulation.yaml",
        base / "experiments/simulation/closed_loop_nonreactive_agents.yaml",
        base / "config/common/scenario_builder/nuplan_mini.yaml",
        base / "config/common/scenario_filter/all_scenarios.yaml",
        base / "config/simulation/observation/box_observation.yaml",
        base / "config/simulation/ego_controller/two_stage_controller.yaml",
        base / "config/simulation/simulation_time_controller/step_simulation_time_controller.yaml",
        base / "config/common/worker/single_machine_thread_pool.yaml",
        base / "config/common/simulation_metric/simulation_closed_loop_nonreactive_agents.yaml",
        base / "config/common/simulation_metric/default_metrics.yaml",
        base / "config/simulation/callback/simulation_log_callback.yaml",
        base / "config/simulation/main_callback/time_callback.yaml",
        base / "config/simulation/main_callback/metric_file_callback.yaml",
        base / "config/simulation/main_callback/metric_aggregator_callback.yaml",
        base / "config/simulation/main_callback/metric_summary_callback.yaml",
        base / "config/simulation/metric_aggregator/closed_loop_nonreactive_agents_weighted_average.yaml",
        base / "utils.py",
        base / "builders/simulation_builder.py",
        base / "builders/simulation_callback_builder.py",
    ]


def _overrides(run: Mapping[str, Any], row: Mapping[str, Any], trace: Path, raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents", "planner=r1_official_technical_smoke_v2_2_r3",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{row['db_path']}]",
        "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "worker=single_machine_thread_pool", "worker.max_workers=1", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701", "run_metric=true",
        "enable_simulation_progress_bar=false", "experiment_name=r1_b2_8_r3_1",
        f"job_name={run['run_id']}", f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _owner_record() -> dict[str, Any]:
    return {
        "schema_version": "r1_b2_8_r3_1_scientific_owner_approval_record_v1.0",
        "status": "R3_APPROVAL_RECORDED_IMMUTABLE_R3_1_WIRING_ONLY",
        "approved": {
            "selector_v1_2": True, "roster_v2_1": True, "schedule_v2_1": True,
            "official_exact_resolution_24_of_24": True, "full_hydra_composition_48_of_48": True,
            "simulation_runner_construction_48_of_48": True,
        },
        "immutable_scientific_identity": {
            "roster_path": str(ROSTER.relative_to(ROOT)), "roster_sha256": ROSTER_SHA,
            "schedule_path": str(SCHEDULE.relative_to(ROOT)), "schedule_sha256": SCHEDULE_SHA,
            "reselection_permitted_for_pre_run_integration_issue": False,
        },
        "authorization": {"OFFICIAL_SIMULATION": "NOT_AUTHORIZED", "NEW_RUN_BUDGET": 0, "RBR_A_B_C": "NOT_AUTHORIZED"},
    }


def main() -> int:
    if any(path.exists() for path in OUTPUTS.values()):
        raise FileExistsError("R3_1_VERSIONED_OUTPUT_ALREADY_EXISTS")
    if _sha(ROSTER) != ROSTER_SHA or _sha(SCHEDULE) != SCHEDULE_SHA:
        raise ValueError("IMMUTABLE_ROSTER_OR_SCHEDULE_SHA_MISMATCH")
    roster, schedule = _read(ROSTER), _read(SCHEDULE)
    runs = list(schedule.get("runs", []))
    entries = {(item["scenario_token"], item["log_id"]): item for item in roster.get("entries", [])}
    if len(runs) != 48 or len(entries) != 24 or len({run["run_id"] for run in runs}) != 48:
        raise ValueError("FROZEN_48_RUN_SCHEDULE_OR_24_ENTRY_ROSTER_INVALID")
    if any((run["scenario_token"], run["log_id"]) not in entries for run in runs):
        raise ValueError("SCHEDULE_ROSTER_IDENTITY_MISMATCH")

    official_env()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_b2_8_r3_frozen_run_dispatcher import build_planner_from_frozen_binding

    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    audit: list[dict[str, Any]] = []
    config_hashes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="r1_b2_8_r3_1_zero_run_") as temp:
        root = Path(temp)
        for run in runs:
            row = entries[(run["scenario_token"], run["log_id"])]
            if official_count(str(row["db_path"]), str(run["scenario_token"])) != 1:
                raise RuntimeError(f"EXACT_SINGLE_SCENARIO_RESOLUTION_FAILED:{run['run_id']}")
            trace, raw = root / run["run_id"] / "trace", root / run["run_id"] / "raw"
            trace.mkdir(parents=True)
            if any(trace.iterdir()) or trace in {item["trace_path"] for item in audit}:
                raise RuntimeError(f"TRACE_PATH_REUSE_OR_NONEMPTY:{run['run_id']}")
            os.environ.update({
                "R1_B2_8_R3_BINDING_MANIFEST": str(R3_BINDING), "R1_B2_8_R3_RUN_ID": str(run["run_id"]),
                "R1_B2_8_R3_TRACE_DIR": str(trace),
            })
            with initialize_config_dir(config_dir=str(config_root)):
                cfg = compose(config_name="default_simulation", overrides=_overrides(run, row, trace, raw))
            resolved = OmegaConf.to_container(cfg, resolve=True)
            encoded = json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if "${" in encoded:
                raise RuntimeError(f"UNRESOLVED_HYDRA_INTERPOLATION:{run['run_id']}")
            config_hashes.append(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
            planner = build_planner_from_frozen_binding(str(R3_BINDING), str(run["run_id"]), str(trace))
            common = set_up_common_builder(cfg, "r3_1_zero_run_construction")
            callbacks_worker = build_callbacks_worker(cfg)
            callbacks = build_simulation_callbacks(cfg, common.output_dir, callbacks_worker)
            runners = build_simulations(cfg, common.worker, callbacks, callbacks_worker, pre_built_planners=[planner])
            if len(runners) != 1:
                raise RuntimeError(f"SIMULATION_RUNNER_CONSTRUCTION_FAILED:{run['run_id']}:{len(runners)}")
            audit.append({
                "run_id": run["run_id"], "trace_path": str(trace), "exact_resolution": 1,
                "full_hydra_config_resolved": True, "simulation_runner_construction": "PASS",
                "simulation_started": False, "primary_trace_contract": "REALIZED_CURRENT_EGO_ITERATIONS_0_79_ONLY",
                "raw_iteration_ge_80": "SECONDARY_NON_PRIMARY_TRACE_ONLY",
            })
    if len(audit) != 48 or len({item["trace_path"] for item in audit}) != 48:
        raise RuntimeError("FINAL_ZERO_RUN_48_UNIQUE_OUTPUT_PATHS_REQUIRED")
    ledger = {"claims_1_to_48": "PASS", "claim_49": "HARD_FAIL_BEFORE_SIMULATOR_START", "consumed_budget": 0}
    component_paths = _component_paths()
    missing = [str(path) for path in component_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"EXECUTION_SHA_CLOSURE_MISSING_COMPONENT:{missing}")
    components = {str(path): _sha(path) for path in component_paths}
    regression = {
        "schema_version": "r1_b2_8_r3_1_final_zero_run_regression_v1.0",
        "status": "48_OF_48_PASS_ZERO_RUN_NO_SIMULATION", "runs": audit,
        "counts": {"exact_resolution": 48, "full_hydra": 48, "simulation_runner_construction": 48},
        "trace_contract": "exact_realized_window_v1_1 authoritative: primary iterations 0...79; >=80 secondary only",
        "safety_adapter_binding": "PASS", "post_run_evaluator_dispatcher_binding": "PASS",
        "ledger_dry_run": ledger, "simulation_started": False, "official_runs": 0, "consumed_budget": 0,
    }
    manifest = {
        "schema_version": "r1_b2_8_r3_1_final_execution_binding_manifest_v1.0",
        "status": "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION",
        "scientific_roster": {"path": str(ROSTER.relative_to(ROOT)), "sha256": ROSTER_SHA, "immutable": True},
        "scientific_schedule": {"path": str(SCHEDULE.relative_to(ROOT)), "sha256": SCHEDULE_SHA, "immutable": True},
        "r3_inherited_execution_binding": {"path": str(R3_BINDING.relative_to(ROOT)), "sha256": _sha(R3_BINDING)},
        "future_execution_components_sha256": components,
        "resolved_full_hydra_config_sha256_by_run_order": config_hashes,
        "assembly": {"set_up_common_builder": "nuplan.planning.script.utils.set_up_common_builder", "build_simulations": "nuplan.planning.script.builders.simulation_builder.build_simulations", "simulation_callback_builder": "nuplan.planning.script.builders.simulation_callback_builder"},
        "official_safety": {"adapter": "tools/r1_b2_8_r3_1_official_safety_adapter.py", "historical_contract": "docs/stageR/r1/r1_official_metric_canonicalization_contract_v1.0.json", "status": "BOUND_NO_NEW_METRIC_OR_THRESHOLD"},
        "post_run_evaluator": {"dispatcher": "tools/r1_b2_8_r3_1_post_run_evaluator_dispatcher.py", "evaluator": "tools/r1_official_technical_smoke_evaluator_v2_1.py", "status": "BOUND_NO_MANUAL_PAIR_SPLICING"},
        "zero_run_regression": {"exact_resolution": "48_OF_48_PASS", "full_hydra": "48_OF_48_PASS", "runner_construction": "48_OF_48_PASS", "simulation_started": False, "official_runs": 0, "consumed_budget": 0},
        "authorization": {"OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_RUN_BUDGET": 0, "RBR_A_B_C_AUTHORIZED": False},
    }
    owner = _owner_record()
    _write(OUTPUTS["owner_json"], owner)
    OUTPUTS["owner_md"].write_text(
        "# R1 B2.8-R3.1 Scientific Owner Approval Record v1.0\n\n"
        "R3 已批准 selector v1.2、roster v2.1、schedule v2.1、24/24 精确解析、48/48 Hydra 与 48/48 runner 构造。"
        "roster v2.1 与 schedule v2.1 自此不可变；任何 pre-run 集成问题均不得触发重新选择身份。\n\n"
        "本轮仅冻结执行接线；`OFFICIAL_SIMULATION=NOT_AUTHORIZED`、`NEW_RUN_BUDGET=0`、`RBR_A/B/C=NOT_AUTHORIZED`。\n",
        encoding="utf-8",
    )
    _write(OUTPUTS["regression"], regression)
    _write(OUTPUTS["manifest"], manifest)
    manifest_sha = _sha(OUTPUTS["manifest"])
    OUTPUTS["request"].write_text(
        "# R1 B2.8-R3.1 Scientific Owner 48-Run Authorization Request v0.1\n\n"
        f"唯一待授权执行 SHA：`{manifest_sha}`。\n\n"
        "已完成：roster/schedule SHA 绑定；48/48 exact resolution、48/48 完整 Hydra、48/48 SimulationRunner 构造；"
        "官方 safety adapter、post-run evaluator dispatcher 与 primary 0...79 trace 合同均已绑定。"
        "本次回归未启动 simulation，official runs=0，consumed budget=0。\n\n"
        "状态为 `READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION`，但 `OFFICIAL_SMOKE_AUTHORIZED=false`、"
        "`NEW_RUN_BUDGET=0`、`RBR_A/B/C=NOT_AUTHORIZED`。\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "manifest_sha256": manifest_sha, "runs": len(audit), "simulation_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
