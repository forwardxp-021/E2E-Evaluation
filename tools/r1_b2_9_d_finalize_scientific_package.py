#!/usr/bin/env python3
"""Finalize B2.9-D structural-dispatch and transitive-SHA closure without simulation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
R1 = ROOT / "docs/stageR/r1"
PAIR_BINDINGS = R1 / "r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json"
ZERO_RUN_AUDIT = R1 / "r1_b2_9_d_zero_run_final_construction_audit_v1.0.json"
STRUCTURAL_AUDIT = R1 / "r1_b2_9_d_dispatcher_structural_audit_v1.0.json"
FINAL_MANIFEST = R1 / "r1_b2_9_d_final_execution_binding_manifest_v2.0.json"
OWNER_REQUEST = R1 / "R1_B2_9_D_Scientific_Owner_48_Run_Authorization_Request_v0.1.md"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"

RUNTIME_CANDIDATE_SHA256 = "9af891c87c951494b382840154ef39036b019815cc8923875f41b0e24a632434"
PROTECTED_CSV_SHA256 = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"SHA_COMPONENT_MISSING:{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"number_of_all_at_fault_collisions_stat_value": [0]}
    ).to_parquet(path / "no_ego_at_fault_collisions.parquet")
    pd.DataFrame(
        {"drivable_area_compliance_stat_value": [True]}
    ).to_parquet(path / "drivable_area_compliance.parquet")


def _synthetic_realized_trace(path: Path, binding: Mapping[str, Any]) -> None:
    trace_dir = path / "trace"
    trace_dir.mkdir(parents=True)
    if binding["family"] == "R-HLC":
        source = np.asarray(binding["source_reference_xy"], dtype=float)
        target = np.asarray(binding["target_reference_xy"], dtype=float)
        start = source[0]
        end = target[min(len(target) - 1, max(1, len(target) // 2))]
    else:
        start = np.asarray([0.0, 0.0])
        end = np.asarray([80.0, 0.0])
    rows = []
    for iteration in range(80):
        xy = start + (end - start) * iteration / 79.0
        rows.append(
            {
                "primary_measurement_source": "REALIZED_CURRENT_EGO",
                "iteration_index": iteration,
                "current_ego": {
                    "time_us": 1_000_000 + iteration * 100_000,
                    "rear_axle": {
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                        "heading": 0.0,
                    },
                    "speed_mps": 5.0,
                },
            }
        )
    trace_file = trace_dir / "realized_current_ego.jsonl"
    trace_file.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def run_structural_dispatch() -> Mapping[str, Any]:
    pair_doc = read_json(PAIR_BINDINGS)
    pairs = pair_doc.get("pairs", [])
    if len(pairs) != 24:
        raise ValueError(f"PAIR_BINDING_COUNT_MUST_EQUAL_24:{len(pairs)}")
    audits: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="r1_b2_9_d_dispatch_structural_") as temporary:
        temp_root = Path(temporary)
        for index, binding in enumerate(pairs):
            baseline = temp_root / f"{index:02d}_baseline"
            treatment = temp_root / f"{index:02d}_treatment"
            for run_dir in (baseline, treatment):
                _metric_fixture(run_dir)
                _synthetic_realized_trace(run_dir, binding)
            result = evaluate_frozen_pair(
                pair_binding=binding,
                baseline_run_dir=baseline,
                treatment_run_dir=treatment,
            )
            if result.get("dispatch_status") != "EVALUATED_NO_POSTHOC_PAIR_DELETION":
                raise RuntimeError(f"STRUCTURAL_DISPATCH_FAILED:{binding['pair_id']}")
            audits.append(
                {
                    "pair_id": binding["pair_id"],
                    "family": binding["family"],
                    "dispatcher_invoked": True,
                    "dispatch_status": result["dispatch_status"],
                    "trace_rows_per_arm": 80,
                    "trace_source": "REALIZED_CURRENT_EGO",
                    "metric_input_format": "REAL_PARQUET_FORMAT_SYNTHETIC_VALUES",
                    "scientific_outcome_used_for_selection": False,
                }
            )
    return {
        "schema_version": "r1_b2_9_d_dispatcher_structural_audit_v1.0",
        "status": "24_OF_24_FROZEN_PAIR_DISPATCHER_STRUCTURAL_PASS",
        "pairs": audits,
        "counts": {
            "total": len(audits),
            "R-HLC": sum(row["family"] == "R-HLC" for row in audits),
            "R-TSB": sum(row["family"] == "R-TSB" for row in audits),
            "pass": len(audits),
        },
        "contract_valid_synthetic_primary80_trace": True,
        "real_format_temporary_metric_parquet": True,
        "future_realized_trace_used": False,
        "scientific_outcome_used_for_identity_selection": False,
        "runner_run_calls": 0,
        "simulation_started": False,
        "official_runs": 0,
        "consumed_real_budget": 0,
    }


def component_paths() -> List[Path]:
    """Closed list of every local and bound nuPlan component used by final execution."""
    repo = [
        "tools/r1_future_compliant_smoke_selector_v1_3.py",
        "docs/stageR/r1/r1_future_compliant_smoke_selector_contract_v1.3.json",
        "docs/stageR/r1/r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json",
        "docs/stageR/r1/r1_fresh_smoke_source_universe_v0.1.json",
        "docs/stageR/r1/r1_official_compliant_technical_smoke_roster_v3.0.json",
        "docs/stageR/r1/r1_official_compliant_technical_smoke_schedule_v3.0.json",
        "docs/stageR/r1/r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json",
        "docs/stageR/r1/r1_b2_9_d_selector_eligibility_audit_v1.0.json",
        "docs/stageR/r1/r1_b2_9_d_roster_v2_1_to_v3_0_comparison_v1.json",
        "docs/stageR/r1/r1_b2_9_d_zero_run_final_construction_audit_v1.0.json",
        "docs/stageR/r1/r1_b2_9_d_dispatcher_structural_audit_v1.0.json",
        "docs/stageR/r1/r1_b2_9_c_scientific_runtime_candidate_manifest_v1.0.json",
        "docs/stageR/r1/r1_primary80_scientific_runtime_horizon_contract_v1.0.json",
        "docs/stageR/r1/r1_hlc_route_progression_invariant_contract_v1.0.json",
        "docs/stageR/r1/r1_official_metric_canonicalization_contract_v1.0.json",
        "docs/stageR/r1/r1_closed_loop_context_implementation_contract_v2.1.json",
        "docs/stageR/r1/r1_hlc_pretreatment_dynamic_clearance_contract_v1.1.json",
        "docs/stageR/r1/r1_official_ego_footprint_binding_v1.0.json",
        "docs/stageR/r1/r1_hlc_generator_v2_contract_v1.0.json",
        "docs/stageR/r1/r1_tsb_generator_v2_contract_v1.0.json",
        "tools/r1_b2_9_d_execute_frozen_48run_smoke.py",
        "tools/r1_b2_9_d_freeze_pair_bindings.py",
        "tools/r1_b2_9_d_finalize_scientific_package.py",
        "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py",
        "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        "tools/r1_official_metric_canonicalizer.py",
        "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        "tools/r1_official_technical_smoke_planner_v3_1.py",
        "tools/r1_primary80_scientific_time_controller_v1.py",
        "tools/r1_closed_loop_benchmark_v2_3.py",
        "tools/r1_closed_loop_benchmark_v2_2.py",
        "tools/r1_closed_loop_benchmark_v2_1.py",
        "tools/r1_prospective_generator_contract_v2.py",
        "tools/r1_official_map_query_bridge_v2_1.py",
        "tools/r1_hlc_measurement_conformance_v1.py",
        "tools/r1_closed_loop_context_adapter_v2_1.py",
        "tools/r1_context_mechanism_core.py",
        "tools/r1_hlc_dynamic_clearance_v1_1.py",
        "tools/r1_official_ego_vehicle_binding_v1.py",
        "tools/r1_b2_8_r3_prospective_selector.py",
        "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v3_1_b2_9_d.yaml",
    ]
    nuplan = [
        "nuplan-devkit/nuplan/planning/script/run_simulation.py",
        "nuplan-devkit/nuplan/planning/script/utils.py",
        "nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py",
        "nuplan-devkit/nuplan/planning/script/builders/simulation_callback_builder.py",
        "nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py",
        "nuplan-devkit/nuplan/planning/simulation/simulation.py",
        "nuplan-devkit/nuplan/planning/simulation/simulation_time_controller/step_simulation_time_controller.py",
        "nuplan-devkit/nuplan/planning/simulation/controller/two_stage_controller.py",
        "nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py",
        "nuplan-devkit/nuplan/planning/simulation/observation/tracks_observation.py",
        "nuplan-devkit/nuplan/planning/metrics/metric_engine.py",
        "nuplan-devkit/nuplan/planning/script/config/simulation/default_simulation.yaml",
        "nuplan-devkit/nuplan/planning/script/experiments/simulation/closed_loop_nonreactive_agents.yaml",
        "nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_mini.yaml",
        "nuplan-devkit/nuplan/planning/script/config/common/scenario_filter/all_scenarios.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/observation/box_observation.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/simulation_time_controller/step_simulation_time_controller.yaml",
        "nuplan-devkit/nuplan/planning/script/config/common/worker/single_machine_thread_pool.yaml",
        "nuplan-devkit/nuplan/planning/script/config/common/simulation_metric/simulation_closed_loop_nonreactive_agents.yaml",
        "nuplan-devkit/nuplan/planning/script/config/common/simulation_metric/default_metrics.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/callback/simulation_log_callback.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/main_callback/time_callback.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/main_callback/metric_file_callback.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/main_callback/metric_aggregator_callback.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/main_callback/metric_summary_callback.yaml",
        "nuplan-devkit/nuplan/planning/script/config/simulation/metric_aggregator/closed_loop_nonreactive_agents_weighted_average.yaml",
    ]
    return [ROOT / path for path in repo] + [WORKSPACE / path for path in nuplan]


def build_manifest() -> Mapping[str, Any]:
    zero = read_json(ZERO_RUN_AUDIT)
    structural = read_json(STRUCTURAL_AUDIT)
    roster = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
    schedule = R1 / "r1_official_compliant_technical_smoke_schedule_v3.0.json"
    selector = R1 / "r1_future_compliant_smoke_selector_contract_v1.3.json"
    exclusion = R1 / "r1_b2_9_d_effective_scientific_exclusion_ledger_v1.0.json"
    source = R1 / "r1_fresh_smoke_source_universe_v0.1.json"
    executor = ROOT / "tools/r1_b2_9_d_execute_frozen_48run_smoke.py"
    candidate = R1 / "r1_b2_9_c_scientific_runtime_candidate_manifest_v1.0.json"
    if sha256(candidate) != RUNTIME_CANDIDATE_SHA256:
        raise ValueError("APPROVED_RUNTIME_CANDIDATE_SHA_MISMATCH")
    if sha256(PROTECTED_CSV) != PROTECTED_CSV_SHA256:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    if zero.get("status") != "48_OF_48_ZERO_RUN_CONSTRUCTION_PASS":
        raise ValueError("ZERO_RUN_CLOSURE_NOT_PASS")
    if structural.get("status") != "24_OF_24_FROZEN_PAIR_DISPATCHER_STRUCTURAL_PASS":
        raise ValueError("STRUCTURAL_DISPATCH_CLOSURE_NOT_PASS")
    components: Dict[str, str] = {}
    for component in component_paths():
        try:
            key = str(component.relative_to(ROOT))
        except ValueError:
            key = str(component)
        if key in components:
            raise ValueError(f"DUPLICATE_SHA_COMPONENT:{key}")
        components[key] = sha256(component)
    return {
        "schema_version": "r1_b2_9_d_final_execution_binding_manifest_v2.0",
        "status": "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_AUTHORIZATION",
        "selector_contract_sha256": sha256(selector),
        "selector_implementation_sha256": sha256(ROOT / "tools/r1_future_compliant_smoke_selector_v1_3.py"),
        "effective_exclusion_ledger_sha256": sha256(exclusion),
        "source_universe_sha256": sha256(source),
        "source_universe_reused": True,
        "scientific_roster_v3_sha256": sha256(roster),
        "scientific_schedule_v3_sha256": sha256(schedule),
        "pair_evaluation_bindings_v2_sha256": sha256(PAIR_BINDINGS),
        "runtime_candidate_sha256": sha256(candidate),
        "new_executor_sha256": sha256(executor),
        "zero_run_construction_audit_sha256": sha256(ZERO_RUN_AUDIT),
        "dispatcher_structural_audit_sha256": sha256(STRUCTURAL_AUDIT),
        "complete_transitive_component_sha256": components,
        "complete_transitive_sha_closure": "PASS",
        "closure_component_count": len(components),
        "construction": {
            "exact_single_scenario_resolutions": 48,
            "planner_v3_1_bindings": 48,
            "Primary80_controller_bindings": 48,
            "controller_number_of_iterations": 81,
            "runner_constructions": 48,
            "pair_binding_lookups": 48,
            "runner_run_calls": 0,
        },
        "pair_dispatcher_structural": {
            "pass": 24,
            "synthetic_primary80_only": True,
            "real_format_temporary_metric_parquet": True,
            "scientific_outcome_used": False,
        },
        "protected_csv_sha256": sha256(PROTECTED_CSV),
        "official_runs": 0,
        "consumed_real_budget": 0,
        "authorization": {
            "OFFICIAL_SMOKE_AUTHORIZED": False,
            "NEW_RUN_BUDGET": 0,
            "RBR_A/B/C": "NOT_AUTHORIZED",
        },
        "hard_restrictions": {
            "retry": "FORBIDDEN",
            "identity_replacement": "FORBIDDEN",
            "threshold_change": "FORBIDDEN",
            "run_id_reuse": "FORBIDDEN",
        },
    }


def owner_request(manifest_sha: str) -> str:
    comparison = read_json(R1 / "r1_b2_9_d_roster_v2_1_to_v3_0_comparison_v1.json")
    counts = comparison["counts"]
    replacements = comparison["replacements"]
    lines = [
        "# R1 B2.9-D Scientific Owner 48-Run Authorization Request v0.1",
        "",
        "## 请求事项",
        "",
        "B2.9-D 已完成 outcome-blind scientific roster rebuild、48-run schedule、24 pair pre-outcome binding、final executor 与完整 SHA 闭包。现仅请求 Scientific Owner 判断：是否授权下述 final manifest 对应的冻结 48-run official smoke 执行一次。",
        "",
        f"- final execution manifest SHA256：`{manifest_sha}`",
        "- 当前 `OFFICIAL_SMOKE_AUTHORIZED = false`",
        "- 当前 `NEW_RUN_BUDGET = 0`",
        "- 当前 `ACTUAL_OFFICIAL_RUNS = 0`",
        "- 当前 `RBR_A/B/C = NOT_AUTHORIZED`",
        "",
        "## 新 roster 摘要",
        "",
        "- roster v3.0：24 个 unique identities；R-HLC 12，R-TSB 12。",
        f"- HLC retained/replaced：{counts['HLC_retained']}/{counts['HLC_replaced']}。",
        f"- TSB retained/replaced：{counts['TSB_retained']}/{counts['TSB_replaced']}。",
        "- effective permanent exclusion：45 个 token/log identity；Attempt 1 的 `b1be12bca092597a` 保留 `OFFICIAL_ATTEMPT_CONSUMED = true`。",
        "- source universe 复用 `r1_fresh_smoke_source_universe_v0.1.json`；未因 Attempt 1 或 canary scientific outcome 重选。",
        "",
        "## 确定性 replacements",
        "",
    ]
    for row in replacements:
        old_identity = row["old_identity"]
        new_identity = row["new_identity"]
        lines.append(
            f"- {old_identity['family']}：旧 identity `{old_identity['scenario_token']}` → 新 identity `{new_identity['scenario_token']}`；new rank `{new_identity['selector_rank_sha256']}`；原因：{row['old_disposition']}。"
        )
    lines += [
        "",
        "## 执行前闭包",
        "",
        "- exact single scenario resolution：48/48 PASS。",
        "- full runner construction：48/48 PASS。",
        "- planner class：48/48 exact `R1OfficialTechnicalSmokePlannerV3_1`。",
        "- time controller：48/48 exact `R1Primary80ScientificTimeControllerV1`，`number_of_iterations() = 81`。",
        "- frozen pair binding lookup：48/48 PASS；24/24 binding 在 simulation 前完成。",
        "- dispatcher structural invocation：24/24 PASS（仅 contract-valid synthetic 80-row REALIZED trace 与 real-format temporary parquet）。",
        "- complete transitive SHA closure：PASS。",
        "- `runner.run()` 调用：0；official simulation：0；consumed budget：0。",
        "",
        "## 语义冻结",
        "",
        "HLC 的 planner reference 使用 `ROUTE_CONTINUOUS_V2_3`；measurement reference 继续使用 `FROZEN_NATIVE_SOURCE_TARGET_MEASUREMENT_CONTRACT`。本轮没有修改 measurement numerics、threshold、mechanism、F_match 或 safety contract。",
        "",
        "## Scientific Owner 唯一待决问题",
        "",
        f"是否对 final manifest SHA256 `{manifest_sha}` 授权一次冻结的 48-run official smoke？在收到匹配该 SHA 的显式授权前，executor 保持 fail-closed。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    structural = run_structural_dispatch()
    write_new_json(STRUCTURAL_AUDIT, structural)
    manifest = build_manifest()
    write_new_json(FINAL_MANIFEST, manifest)
    manifest_sha = sha256(FINAL_MANIFEST)
    if OWNER_REQUEST.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{OWNER_REQUEST}")
    OWNER_REQUEST.write_text(owner_request(manifest_sha), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dispatcher_structural_pass": structural["counts"]["pass"],
                "component_closure": manifest["complete_transitive_sha_closure"],
                "component_count": manifest["closure_component_count"],
                "final_manifest_sha256": manifest_sha,
                "runner_run_calls": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
