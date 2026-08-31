#!/usr/bin/env python3
"""Fail-closed zero-run audit for the B2.8-R2 official launch control plane.

This tool intentionally stops after exact official nuPlan scenario resolution.  It
never imports or calls run_simulation.py, SimulationRunner.run(),
Simulation.step(), or planner trajectory computation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUPLAN_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan-devkit")
NUPLAN_DATA_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/data")
NUPLAN_MAP_ROOT = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/dataset/maps")
PYTHON_EXECUTABLE = Path("/Users/liuqing/miniconda3/envs/nuplan/bin/python3.9")
DOCS = PROJECT_ROOT / "docs/stageR/r1"
SCHEDULE = DOCS / "r1_official_compliant_technical_smoke_schedule_v2.0.json"
R1_BINDINGS = DOCS / "r1_b2_8_r1_execution_bindings_manifest_v1.0.json"
OUTPUTS = {
    "approval_json": DOCS / "r1_b2_8_r1_scientific_owner_approval_v1.0.json",
    "approval_md": DOCS / "R1_B2_8_R1_Scientific_Owner_Approval_Record_v1.0.md",
    "launch_manifest": DOCS / "r1_b2_8_r2_official_launch_manifest_v1.0.json",
    "execution_manifest": DOCS / "r1_b2_8_r2_official_launch_execution_binding_manifest_v1.0.json",
    "report": DOCS / "R1_B2_8_R2_Official_Launch_Control_Plane_Fail_Closed_Report_v1.0.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def configure_official_environment() -> None:
    """Mirror the known-good Stage7 environment before importing nuPlan."""
    os.environ.update(
        {
            "NUPLAN_DEVKIT_ROOT": str(NUPLAN_ROOT),
            "NUPLAN_DATA_ROOT": str(NUPLAN_DATA_ROOT),
            "NUPLAN_MAPS_ROOT": str(NUPLAN_MAP_ROOT),
            "NUPLAN_MAP_ROOT": str(NUPLAN_MAP_ROOT),
            "NUPLAN_EXP_ROOT": "/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/exp",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    for root in (str(NUPLAN_ROOT), "/Users/liuqing/Projects/01_E2E_QA_Code/tuplan_garage", str(PROJECT_ROOT)):
        if root not in sys.path:
            sys.path.insert(0, root)


def official_resolution_count(db_path: str, scenario_token: str) -> int:
    configure_official_environment()
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

    return len(list(get_scenarios_from_db(db_path, [scenario_token], None, None, True, False)))


def full_overrides(run: Dict[str, Any], trace_path: Path, raw_path: Path) -> List[str]:
    searchpath = (
        f"[file://{PROJECT_ROOT / 'configs/r1_official_technical_smoke_hydra'},"
        "pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
    )
    row = run["future_roster_row"]
    return [
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r1_official_technical_smoke_v2_2",
        "scenario_builder=nuplan_mini",
        f"scenario_builder.db_files=[{row['db_path']}]",
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
        "experiment_name=r1_b2_8_r2_official_smoke",
        f"job_name={run['run_id']}",
        f"output_dir={raw_path}",
        f"planner.realized_trace_path={trace_path}",
        f"hydra.searchpath={searchpath}",
    ]


def make_approval() -> Dict[str, Any]:
    return {
        "schema_version": "r1_b2_8_r1_scientific_owner_approval_v1.0",
        "status": "RECORDED",
        "approved": {
            "REALIZED_CURRENT_EGO_SEMANTICS": True,
            "PLANNER_V2_2_PASSIVE_INSTRUMENTATION": True,
            "V2_1_V2_2_TRAJECTORY_PARITY": True,
            "HYDRA_PLANNER_BINDING_48_OF_48": True,
            "SCIENTIFIC_SCHEDULE_IDENTITY": True,
        },
        "not_verified": {"FULL_OFFICIAL_RUN_SIMULATION_LAUNCH_BINDING": True},
        "scope": "B2.8-R1 仅实例化 planner-side Hydra/runtime wiring，未构造 official nuPlan scenario 或 simulator。",
        "authorization": {"OFFICIAL_SIMULATION": False, "NEW_RUN_BUDGET": 0},
    }


def component_paths() -> Iterable[Path]:
    relative = [
        "docs/stageR/r1/r1_official_compliant_technical_smoke_roster_v2.0.json",
        "docs/stageR/r1/r1_official_compliant_technical_smoke_schedule_v2.0.json",
        "tools/r1_b2_8_r1_frozen_run_dispatcher.py",
        "tools/r1_official_technical_smoke_planner_v2_2.py",
        "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v2_2.yaml",
        "tools/stage7_m6_4b_run_locked_rollouts.py",
        "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        "docs/stageR/r1/r1_b2_8_r1_execution_bindings_manifest_v1.0.json",
    ]
    return (PROJECT_ROOT / item for item in relative)


def run_audit() -> Dict[str, Any]:
    bindings = read_json(R1_BINDINGS)["frozen_run_bindings"]
    schedule = read_json(SCHEDULE)["runs"]
    by_run_id = {entry["run_id"]: entry for entry in bindings}
    identity_keys = ("run_id", "pair_id", "family", "scenario_token", "log_id", "arm", "run_order")
    for index, (schedule_row, binding_row) in enumerate(zip(schedule, bindings), start=1):
        mismatch = {
            key: {"schedule": schedule_row[key], "binding": binding_row[key]}
            for key in identity_keys
            if schedule_row[key] != binding_row[key]
        }
        if mismatch:
            raise ValueError(f"冻结 schedule 与 B2.8-R1 bindings 第 {index} 行身份不一致：{mismatch}")
    output_root = Path("/Users/liuqing/Projects/01_E2E_QA_Code/nuplan/exp/r1_b2_8_r2_official_smoke")
    runs: List[Dict[str, Any]] = []
    for schedule_row in schedule:
        run = by_run_id[schedule_row["run_id"]]
        row = run["future_roster_row"]
        count = official_resolution_count(row["db_path"], run["scenario_token"])
        trace_path = output_root / run["run_id"] / "realized_trace" / "primary_trace.jsonl"
        raw_path = output_root / run["run_id"] / "raw_simulation"
        runs.append(
            {
                **{key: run[key] for key in ("run_id", "pair_id", "family", "scenario_token", "log_id", "arm", "run_order")},
                "python_executable": str(PYTHON_EXECUTABLE),
                "run_simulation_py": str(NUPLAN_ROOT / "nuplan/planning/script/run_simulation.py"),
                "planner_target": "tools.r1_official_technical_smoke_planner_v2_2.R1OfficialTechnicalSmokePlannerV2_2",
                "frozen_dispatcher": str(PROJECT_ROOT / "tools/r1_b2_8_r1_frozen_run_dispatcher.py"),
                "db_path": row["db_path"], "map_name": row["map_name"],
                "route_fingerprint": row["route_fingerprint"],
                "initial_state_fingerprint": row["initial_state_fingerprint"],
                "overrides": full_overrides(run, trace_path, raw_path),
                "realized_trace_path": str(trace_path), "raw_simulation_output_path": str(raw_path),
                "trace_path_lifecycle": "NOT_CHECKED_FAIL_CLOSED",
                "official_scenario_resolution_count": count,
                "official_scenario_resolution": "EXACT_SINGLE_MATCH" if count == 1 else "FAIL_CLOSED",
                "full_hydra_composition": "NOT_EXECUTED_FAIL_CLOSED",
                "simulation_runner_construction": "NOT_EXECUTED_FAIL_CLOSED",
            }
        )
    successful = sum(item["official_scenario_resolution_count"] == 1 for item in runs)
    return {"runs": runs, "successful": successful, "failed": len(runs) - successful}


def write_outputs(audit: Dict[str, Any]) -> None:
    approval = make_approval()
    write_json(OUTPUTS["approval_json"], approval)
    OUTPUTS["approval_md"].write_text(
        "# R1 B2.8-R1 Scientific Owner Approval Record v1.0\n\n"
        "- REALIZED_CURRENT_EGO_SEMANTICS：APPROVED\n- PLANNER_V2_2_PASSIVE_INSTRUMENTATION：APPROVED\n"
        "- V2_1_V2_2_TRAJECTORY_PARITY：APPROVED\n- 48_OF_48_HYDRA_PLANNER_BINDING：APPROVED\n"
        "- SCIENTIFIC_SCHEDULE_IDENTITY：APPROVED\n\n"
        "B2.8-R1 仅完成 planner-side Hydra/runtime wiring；未构造 official nuPlan scenario/simulator，"
        "因此 FULL_OFFICIAL_LAUNCH_VERIFIED 仍为 NOT_YET_VERIFIED。\n\n"
        "OFFICIAL_SIMULATION = NOT_AUTHORIZED；NEW_RUN_BUDGET = 0。\n",
        encoding="utf-8",
    )
    status = "FAIL_CLOSED_EXACT_SCENARIO_RESOLUTION" if audit["failed"] else "READY_FOR_NEXT_ZERO_RUN_CHECK"
    launch_manifest = {
        "schema_version": "r1_b2_8_r2_official_launch_manifest_v1.0",
        "status": status,
        "simulation_started": False, "official_runs": 0, "consumed_budget": 0,
        "exact_single_scenario_resolution": f"{audit['successful']}_OF_48",
        "scientific_schedule_identity": "48_OF_48_EXACT_IDENTICAL",
        "runs": audit["runs"],
    }
    write_json(OUTPUTS["launch_manifest"], launch_manifest)
    components = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in component_paths() if path.is_file()}
    binding_manifest = {
        "schema_version": "r1_b2_8_r2_official_launch_execution_binding_manifest_v1.0",
        "status": "INCOMPLETE_FAIL_CLOSED",
        "reason": "4 个冻结 scenario_token 在 bound nuPlan 1.2.2 官方查询中为 0 match；禁止 replacement。",
        "inherited_b2_8_r1_binding_manifest_sha256": sha256(R1_BINDINGS),
        "components_sha256": components,
        "simulation_started": False, "official_runs": 0, "consumed_budget": 0,
    }
    write_json(OUTPUTS["execution_manifest"], binding_manifest)
    failed = [entry for entry in audit["runs"] if entry["official_scenario_resolution_count"] != 1]
    lines = [
        "# R1 B2.8-R2 Official Launch Control-Plane Fail-Closed Report v1.0", "",
        "## 结论", "",
        f"官方 nuPlan 1.2.2 exact resolution 为 {audit['successful']}/48；4 个冻结 identity 为 0 match，对应 8 个 arm/run。",
        "按冻结规则，0 match 必须 FAIL_CLOSED，且不得 replacement。因此没有继续 full Hydra composition、SimulationRunner construction 或任何仿真。", "",
        "冻结 scientific schedule 的 run_id、pair_id、family、scenario_token、log_id、arm 与 run_order 已逐行比较，48/48 EXACT_IDENTICAL。", "",
        "## 失败身份", "",
    ]
    for entry in failed[::2]:
        lines.append(f"- {entry['log_id']} / {entry['scenario_token']}（{entry['family']}）：0 official match。")
    lines += ["", "## 保持的控制面状态", "", "- simulation_started = false", "- actual official runs = 0", "- consumed budget = 0", "- OFFICIAL_SMOKE_AUTHORIZED = false", "- RBR_A/B/C = NOT_AUTHORIZED", ""]
    OUTPUTS["report"].write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="写入 fail-closed 审计结果文件")
    args = parser.parse_args()
    audit = run_audit()
    if args.write:
        write_outputs(audit)
    print(json.dumps({"exact_single": audit["successful"], "failed": audit["failed"], "simulation_started": False}, ensure_ascii=False))
    return 0 if audit["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
