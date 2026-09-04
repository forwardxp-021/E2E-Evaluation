#!/usr/bin/env python3
"""Write the non-circular BJ-B0 component SHA closure after all inputs are final."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/stageR/r2/r2_bj_b0_component_sha_binding_manifest_v1.0.json"
LOCAL = [
    "docs/stageR/r2/r2_bj_b0_preregistered_roster_selection_contract_v1.0.json",
    "docs/stageR/r2/r2_bj_a5_applicable_pool_provenance_manifest_v1.0.json",
    "docs/stageR/r2/r2_bj_a5_557_entry_eligibility_census_ledger_v1.0.json",
    "docs/stageR/r2/r2_bj_a5_native_generated_composite_component_audit_v1.0.json",
    "docs/stageR/r2/r2_bj_a5_finite_frame_census_envelope_v1.0.json",
    "docs/stageR/r2/r2_bj_a5_component_sha_binding_manifest_v1.0.json",
    "docs/stageR/r2/r2_bj_a_hlc_global_parameter_space_v4.0.json",
    "docs/stageR/r2/r2_bj_a_hlc_kinematic_architecture_contract_v4.0.json",
    "tools/r2_bj_a_hlc_morphology_feasible_generator_v4.py",
    "tools/r2_bj_a_hlc_morphology_feasible_planner_v4.py",
    "tools/r1_closed_loop_benchmark_v2_3.py",
    "tools/r1_official_technical_smoke_planner_v3_1.py",
    "tools/r1_primary80_scientific_time_controller_v1.py",
    "tools/r2_bj_b0_freeze_engineering_package.py",
    "tools/r2_bj_b0_hlc_v4_engineering_planner.py",
    "tools/r2_bj_b0_execute_frozen_hlc_v4_engineering.py",
    "configs/r1_official_technical_smoke_hydra/planner/r2_bj_b0_hlc_v4_engineering.yaml",
    "docs/stageR/r2/r2_bj_b0_hlc_v4_engineering_roster_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_permanent_engineering_exclusion_ledger_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_unselected_pool_disposition_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_hlc_v4_pair_schedule_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_exact_pair_binding_manifest_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_hlc_v4_execution_architecture_contract_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_online_failure_taxonomy_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_execution_authorization_gate_v1.0.json",
    "docs/stageR/r2/r2_bj_b0_zero_run_integration_preflight_audit_v1.0.json",
    "docs/stageR/r2/R2_BJ_B0_HLC_V4_Engineering_Freeze_Report_v1.md",
    "tests/test_r2_bj_b0_hlc_v4_engineering_freeze.py",
    "QUICK_REFERENCE.md",
]
EXTERNAL = [
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/default_simulation.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/experiments/simulation/closed_loop_nonreactive_agents.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_mini.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/common/scenario_filter/all_scenarios.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/observation/idm_agents_observation.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/common/simulation_metric/default_metrics.yaml",
    ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/common/worker/sequential.yaml",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"R2_BJ_B0_VERSIONED_OUTPUT_EXISTS:{OUT}")
    document = {
        "schema_version": "r2_bj_b0_component_sha_binding_manifest_v1.0",
        "status": "R2_BJ_B0_ZERO_RUN_EXECUTION_PACKAGE_FROZEN_READY_FOR_CANARY_OWNER_REVIEW",
        "baseline": {"remote_commit": "80afa7e4a1416c7e527fbd6ab6b2889ee9198be7", "local_commit": "c99551c022cbb595e1afb6e51ffb32003fc419c0", "tree": "0b22031bb4923b91b5b145e0658382f56ba0903f"},
        "components": [{"path": path, "sha256": sha(ROOT / path)} for path in LOCAL],
        "external_bound_nuplan_1_2_2_runtime_components": [{"absolute_path": str(path), "sha256": sha(path)} for path in EXTERNAL],
        "component_SHA_closure": "PASS",
        "A5_APPLICABLE_POOL": 34, "BJ_B_ROSTER": 8, "UNSELECTED_POOL": 26,
        "ROSTER_TOKEN_UNIQUE": "8/8", "ROSTER_LOG_UNIQUE": "8/8", "PAIR_BINDINGS": 8,
        "INTENDED_RUNS": 16, "full_Hydra_compositions": 16, "exact_scenario_resolutions": 16,
        "SimulationRunner_constructions": 16, "RUNNER_RUN": 0, "NEW_RUN_BUDGET": 0,
        "CANARY_AUTHORIZED": False, "R2_C_STARTED": False, "CONFIRMATORY_SMOKE_STARTED": False,
        "RBR_STARTED": False, "protected_CSV_sha256": "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8",
    }
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(sha(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
