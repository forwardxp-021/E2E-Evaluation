#!/usr/bin/env python3
"""Finalize B2.9-E structural dispatch and complete lifecycle SHA closure without simulation."""

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
PAIR_BINDINGS = R1 / "r1_b2_9_e_frozen_pair_evaluation_bindings_v2.1.json"
ZERO_RUN_AUDIT = R1 / "r1_b2_9_e_zero_run_final_construction_audit_v1.0.json"
CANARY_LEDGER = R1 / "r1_b2_9_e_exact_lifecycle_canary_run_ledger_v1.0.json"
STRUCTURAL_AUDIT = R1 / "r1_b2_9_e_dispatcher_structural_audit_v1.0.json"
FINAL_MANIFEST = R1 / "r1_b2_9_e_final_execution_binding_manifest_v2.1.json"
OWNER_REQUEST = R1 / "R1_B2_9_E_Scientific_Owner_48_Run_Reauthorization_Request_v0.1.md"
PREDECESSOR_MANIFEST = R1 / "r1_b2_9_d_final_execution_binding_manifest_v2.0.json"
PREDECESSOR_STOP = R1 / "r1_b2_9_d_official_smoke_technical_stop_record_v1.0.json"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"

ROSTER_SHA256 = "efe8e9d680ca0bcacb367bc9b616610ca78c260195e53b8f025a7bd1d92c23e6"
SCHEDULE_SHA256 = "99f44095c27319b746921376d2549a00186303298b5266ff45dd008a98c08455"
PAIR_BINDING_SHA256 = "a606a87b01cd1fdd340070fca7e77170b6e0782aafa1e7c19ab6c91228cc9fa6"
PREDECESSOR_MANIFEST_SHA256 = "88d1d36ef721c43dda4ce5907d2a85968142279238a4c2f11309b7b2eebe2877"
PREDECESSOR_STOP_SHA256 = "01fd6b9a3fc17f2ba31635a58f83569345ecfd56c33eb0c0032993f0ecdcdbe2"
PROTECTED_CSV_SHA256 = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"


def sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"SHA_COMPONENT_MISSING:{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metric_fixture(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"number_of_all_at_fault_collisions_stat_value": [0]}).to_parquet(
        path / "no_ego_at_fault_collisions.parquet"
    )
    pd.DataFrame({"drivable_area_compliance_stat_value": [True]}).to_parquet(
        path / "drivable_area_compliance.parquet"
    )


def _synthetic_realized_trace(path: Path, binding: Mapping[str, Any]) -> None:
    trace_dir = path / "trace"
    trace_dir.mkdir(parents=True)
    if binding["family"] == "R-HLC":
        source = np.asarray(binding["source_reference_xy"], dtype=float)
        target = np.asarray(binding["target_reference_xy"], dtype=float)
        start = source[0]
        end = target[min(len(target) - 1, max(1, len(target) // 2))]
    else:
        start, end = np.asarray([0.0, 0.0]), np.asarray([80.0, 0.0])
    rows = []
    for iteration in range(80):
        xy = start + (end - start) * iteration / 79.0
        rows.append(
            {
                "primary_measurement_source": "REALIZED_CURRENT_EGO",
                "iteration_index": iteration,
                "current_ego": {
                    "time_us": 1_000_000 + iteration * 100_000,
                    "rear_axle": {"x": float(xy[0]), "y": float(xy[1]), "heading": 0.0},
                    "speed_mps": 5.0,
                },
            }
        )
    (trace_dir / "realized_current_ego.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8"
    )


def run_structural_dispatch() -> Dict[str, Any]:
    pairs = read_json(PAIR_BINDINGS)["pairs"]
    if len(pairs) != 24:
        raise ValueError(f"PAIR_BINDING_COUNT_MUST_EQUAL_24:{len(pairs)}")
    audits: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="r1_b2_9_e_dispatch_structural_") as temporary:
        temp_root = Path(temporary)
        for index, binding in enumerate(pairs):
            baseline = temp_root / f"{index:02d}_baseline"
            treatment = temp_root / f"{index:02d}_treatment"
            for run_dir in (baseline, treatment):
                _metric_fixture(run_dir)
                _synthetic_realized_trace(run_dir, binding)
            result = evaluate_frozen_pair(
                pair_binding=binding, baseline_run_dir=baseline, treatment_run_dir=treatment
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
                    "scientific_outcome_used": False,
                }
            )
    return {
        "schema_version": "r1_b2_9_e_dispatcher_structural_audit_v1.0",
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
        "runner_run_calls": 0,
        "run_runners_calls": 0,
        "simulation_started": False,
        "scientific_outcome_used": False,
    }


def component_paths() -> List[Path]:
    """Every local file in the future E runtime and its nuPlan post-run lifecycle."""
    predecessor = read_json(PREDECESSOR_MANIFEST)["complete_transitive_component_sha256"]
    replaced = {
        "docs/stageR/r1/r1_official_compliant_technical_smoke_schedule_v3.0.json",
        "docs/stageR/r1/r1_b2_9_d_frozen_pair_evaluation_bindings_v2.0.json",
        "docs/stageR/r1/r1_b2_9_d_zero_run_final_construction_audit_v1.0.json",
        "docs/stageR/r1/r1_b2_9_d_dispatcher_structural_audit_v1.0.json",
        "tools/r1_b2_9_d_execute_frozen_48run_smoke.py",
        "tools/r1_b2_9_d_freeze_pair_bindings.py",
        "tools/r1_b2_9_d_finalize_scientific_package.py",
        "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v3_1_b2_9_d.yaml",
    }
    paths = [Path(key) if Path(key).is_absolute() else ROOT / key for key in predecessor if key not in replaced]
    repo_additions = [
        "docs/stageR/r1/r1_official_compliant_technical_smoke_schedule_v3.1.json",
        "docs/stageR/r1/r1_b2_9_e_frozen_pair_evaluation_bindings_v2.1.json",
        "docs/stageR/r1/r1_b2_9_e_schedule_v3_0_to_v3_1_parity_audit_v1.json",
        "docs/stageR/r1/r1_b2_9_e_pair_binding_v2_0_to_v2_1_parity_audit_v1.json",
        "docs/stageR/r1/r1_b2_9_e_zero_run_final_construction_audit_v1.0.json",
        "docs/stageR/r1/r1_b2_9_e_exact_lifecycle_canary_run_ledger_v1.0.json",
        "docs/stageR/r1/r1_b2_9_e_dispatcher_structural_audit_v1.0.json",
        "docs/stageR/r1/R1_B2_9_E_Post_Run_Lifecycle_Forensic_v1.md",
        "docs/stageR/r1/r1_b2_9_d_final_execution_binding_manifest_v2.0.json",
        "docs/stageR/r1/r1_b2_9_d_official_smoke_technical_stop_record_v1.0.json",
        "tools/r1_b2_9_e_prepare_versioned_bindings.py",
        "tools/r1_b2_9_e_official_run_lifecycle.py",
        "tools/r1_b2_9_e_execute_frozen_48run_smoke.py",
        "tools/r1_b2_9_e_finalize_package.py",
        "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v3_1_b2_9_e.yaml",
    ]
    paths.extend(ROOT / path for path in repo_additions)
    nuplan_lifecycle = [
        "nuplan-devkit/nuplan/planning/simulation/runner/executor.py",
        "nuplan-devkit/nuplan/planning/simulation/runner/runner_report.py",
        "nuplan-devkit/nuplan/planning/simulation/main_callback/abstract_main_callback.py",
        "nuplan-devkit/nuplan/planning/simulation/main_callback/multi_main_callback.py",
        "nuplan-devkit/nuplan/planning/simulation/main_callback/time_callback.py",
        "nuplan-devkit/nuplan/planning/simulation/main_callback/metric_file_callback.py",
        "nuplan-devkit/nuplan/planning/simulation/main_callback/metric_aggregator_callback.py",
        "nuplan-devkit/nuplan/planning/simulation/main_callback/metric_summary_callback.py",
    ]
    paths.extend(WORKSPACE / path for path in nuplan_lifecycle)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def build_manifest() -> Dict[str, Any]:
    roster = R1 / "r1_official_compliant_technical_smoke_roster_v3.0.json"
    schedule = R1 / "r1_official_compliant_technical_smoke_schedule_v3.1.json"
    zero, canary, structural = read_json(ZERO_RUN_AUDIT), read_json(CANARY_LEDGER), read_json(STRUCTURAL_AUDIT)
    expected = {
        roster: ROSTER_SHA256,
        schedule: SCHEDULE_SHA256,
        PAIR_BINDINGS: PAIR_BINDING_SHA256,
        PREDECESSOR_MANIFEST: PREDECESSOR_MANIFEST_SHA256,
        PREDECESSOR_STOP: PREDECESSOR_STOP_SHA256,
        PROTECTED_CSV: PROTECTED_CSV_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"FROZEN_SHA_MISMATCH:{path}")
    if zero.get("status") != "48_OF_48_ZERO_RUN_CONSTRUCTION_PASS":
        raise ValueError("ZERO_RUN_CLOSURE_NOT_PASS")
    if canary.get("status") != "4_OF_4_EXACT_LIFECYCLE_CANARY_PASS":
        raise ValueError("EXACT_LIFECYCLE_CANARY_NOT_PASS")
    if structural.get("status") != "24_OF_24_FROZEN_PAIR_DISPATCHER_STRUCTURAL_PASS":
        raise ValueError("STRUCTURAL_DISPATCH_CLOSURE_NOT_PASS")
    if canary.get("counts") != {
        "runs": 4,
        "HLC_technical_complete": 2,
        "TSB_technical_complete": 2,
        "exact_80_traces": 4,
        "metric_lifecycle_complete": 4,
        "safety_adapter_complete": 4,
        "dispatcher_complete": 2,
    }:
        raise ValueError("CANARY_COUNT_CLOSURE_NOT_PASS")
    components: Dict[str, str] = {}
    for component in component_paths():
        try:
            key = str(component.relative_to(ROOT))
        except ValueError:
            key = str(component)
        components[key] = sha256(component)
    return {
        "schema_version": "r1_b2_9_e_final_execution_binding_manifest_v2.1",
        "status": "FROZEN_READY_FOR_SCIENTIFIC_OWNER_48_RUN_REAUTHORIZATION",
        "predecessor": {
            "PREDECESSOR_ATTEMPT": "B2_9_D_STOPPED_TECHNICAL_FAILURE",
            "final_manifest_sha256": sha256(PREDECESSOR_MANIFEST),
            "technical_stop_record_sha256": sha256(PREDECESSOR_STOP),
            "old_attempts_consumed": 2,
            "old_once_authorization_consumed": True,
            "old_outputs_reused": False,
            "old_scientific_result_reused": False,
            "old_trace_used_as_new_scientific_pair_input": False,
        },
        "scientific_roster_v3_sha256": sha256(roster),
        "scientific_roster_identity_changed": False,
        "selector_invoked": False,
        "source_universe_scanned": False,
        "scientific_schedule_v3_1_sha256": sha256(schedule),
        "pair_evaluation_bindings_v2_1_sha256": sha256(PAIR_BINDINGS),
        "shared_lifecycle_helper_sha256": sha256(ROOT / "tools/r1_b2_9_e_official_run_lifecycle.py"),
        "new_executor_sha256": sha256(ROOT / "tools/r1_b2_9_e_execute_frozen_48run_smoke.py"),
        "zero_run_construction_audit_sha256": sha256(ZERO_RUN_AUDIT),
        "exact_lifecycle_canary_ledger_sha256": sha256(CANARY_LEDGER),
        "dispatcher_structural_audit_sha256": sha256(STRUCTURAL_AUDIT),
        "exact_executor_canary": {**canary["counts"], "actual_simulation_reruns": 0, "scientific_use": False},
        "construction": {
            **zero["counts"],
            "controller_number_of_iterations": 81,
            "runner_run_calls": 0,
            "run_runners_calls": 0,
        },
        "pair_dispatcher_structural": {"pass": 24, "scientific_outcome_used": False},
        "complete_transitive_component_sha256": components,
        "complete_transitive_sha_closure": "PASS",
        "callback_transitive_sha_closure": "PASS",
        "closure_component_count": len(components),
        "protected_csv_sha256": sha256(PROTECTED_CSV),
        "official_scientific_simulations_this_round": 0,
        "engineering_canary_simulations_this_round": 4,
        "future_budget_semantics": {
            "package": "NEW_VERSIONED_EXECUTION_PACKAGE",
            "requires_new_owner_reauthorization": True,
            "NEW_RUN_BUDGET_IF_AUTHORIZED": 48,
            "remaining_old_budget_reused": False,
        },
        "authorization": {
            "OFFICIAL_SMOKE_AUTHORIZED": False,
            "NEW_RUN_BUDGET": 0,
            "RBR_A/B/C": "NOT_AUTHORIZED",
        },
        "hard_restrictions": {
            "retry": "FORBIDDEN",
            "identity_replacement": "FORBIDDEN",
            "threshold_change": "FORBIDDEN",
            "old_output_append_or_completion": "FORBIDDEN",
        },
    }


def owner_request(manifest_sha: str, manifest: Mapping[str, Any]) -> str:
    return f"""# R1 B2.9-E Scientific Owner 48-Run Reauthorization Request v0.1

## 请求事项

B2.9-E 已仅修复 post-run callback lifecycle，并将修复后的新版本执行包冻结。现请求 Scientific Owner 判断：是否针对下述新 manifest 授权一次全新的 48-run official scientific smoke。

- final execution manifest SHA256：`{manifest_sha}`
- 当前 `OFFICIAL_SMOKE_AUTHORIZED = false`
- 当前 `NEW_RUN_BUDGET = 0`
- 当前 `RBR_A/B/C = NOT_AUTHORIZED`

## 冻结资产与语义

- roster v3.0 SHA256：`{manifest['scientific_roster_v3_sha256']}`，identity 完全未变。
- schedule v3.1 SHA256：`{manifest['scientific_schedule_v3_1_sha256']}`；48/48 与 v3.0 scientific semantics 完全相同，仅使用全新 `R1B29E-...` run/pair references。
- pair bindings v2.1 SHA256：`{manifest['pair_evaluation_bindings_v2_1_sha256']}`；24/24 scientific semantics 完全相同，仅机械更新 run/pair references 和 package provenance。
- selector 未调用，source universe 未扫描，未重新 rank，未替换 identity。

## 生命周期修复与验证

- 新 shared lifecycle helper SHA256：`{manifest['shared_lifecycle_helper_sha256']}`。
- 新 executor SHA256：`{manifest['new_executor_sha256']}`；executor 不直接调用 `SimulationRunner.run()`，只经 shared helper 使用 nuPlan `run_runners(...)`。
- exact-executor engineering canary：HLC 2/2、TSB 2/2 technical complete；4/4 exact Primary80 trace、metric parquet、runner report 与 safety adapter complete；2/2 dispatcher complete；scientific outcome 仅 descriptive。
- 48/48 zero-run construction PASS；`runner.run = 0`、`run_runners = 0`。
- 24/24 frozen pair structural dispatcher PASS。
- callback 与完整 transitive SHA closure：PASS，共 {manifest['closure_component_count']} 个组件。

## 旧 Attempt 隔离

B2.9-D once authorization 已消费，两个旧 official attempts 永久保留为 `ATTEMPT_HISTORY_ONLY`。旧输出未删除、覆盖、append、补 callback、补 parquet或补 scientific evaluation，也未作为新 pair input。新授权若批准，其预算语义是新版本 package 的 `NEW_RUN_BUDGET = 48`，不是复用旧预算的 46。

## Scientific Owner 唯一待决问题

是否对 final manifest SHA256 `{manifest_sha}` 重新授权一次冻结的新版本 48-run official scientific smoke？在收到与该 SHA 精确匹配的显式授权前，executor 保持 fail-closed。
"""


def main() -> int:
    structural = run_structural_dispatch()
    write_new_json(STRUCTURAL_AUDIT, structural)
    manifest = build_manifest()
    write_new_json(FINAL_MANIFEST, manifest)
    manifest_sha = sha256(FINAL_MANIFEST)
    if OWNER_REQUEST.exists():
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{OWNER_REQUEST}")
    OWNER_REQUEST.write_text(owner_request(manifest_sha, manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "dispatcher_structural_pass": structural["counts"]["pass"],
                "component_count": manifest["closure_component_count"],
                "final_manifest_sha256": manifest_sha,
                "official_scientific_simulations": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
