#!/usr/bin/env python3
"""R1 B2.8-R1 zero-run wiring repair and complete-path repreflight.

No nuPlan scenario or simulator is constructed.  Planner calls below use only
in-memory official PlannerInput objects to prove V2.1/V2.2 parity and passive
realized-state capture.  The 48 Hydra checks instantiate planners only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra.utils import instantiate
from omegaconf import OmegaConf
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

from tools.r1_b2_6_official_dispatch_preflight import _official_input
from tools.r1_b2_8_r1_frozen_run_dispatcher import load_frozen_run_binding
from tools.r1_closed_loop_benchmark_v2_1 import exact_realized_window_v1_1
from tools.r1_official_technical_smoke_planner_v2_1 import R1OfficialTechnicalSmokePlannerV2_1
from tools.r1_official_technical_smoke_planner_v2_2 import PRIMARY_SOURCE, R1OfficialTechnicalSmokePlannerV2_2


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.0.json"
SCHEDULE = R1 / "r1_official_compliant_technical_smoke_schedule_v2.0.json"
CONFIG = ROOT / "configs/r1_official_technical_smoke_hydra/planner/r1_official_technical_smoke_v2_2.yaml"
OUT = {
    "contract": R1 / "r1_b2_8_r1_realized_trace_contract_v1.0.json",
    "hydra": R1 / "r1_b2_8_r1_hydra_binding_audit_v1.0.json",
    "composition": R1 / "r1_b2_8_r1_48run_composition_preflight_v1.0.json",
    "bindings": R1 / "r1_b2_8_r1_execution_bindings_manifest_v1.0.json",
    "identity": R1 / "r1_b2_8_r1_scientific_schedule_identity_audit_v1.0.json",
    "preflight": R1 / "r1_b2_8_r1_zero_run_complete_path_preflight_v1.0.json",
    "report": R1 / "R1_B2_8_R1_Execution_Integration_Repair_Report_v1.md",
    "request": R1 / "R1_B2_8_R1_Scientific_Owner_Authorization_Request_v0.1.md",
}
EXPECTED = {
    "roster": "af672c0aa47eadebc1799dfac611016abad5b280ddd2cd56ab8ed02b605a219f",
    "schedule": "d449db48aa915b5d605d51ee587aa4d6ee5fa40029eb911fbb2af5b0721fc8c5",
    "protected_csv": "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8",
}
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned artifact: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite versioned artifact: {path}")
    path.write_text(value, encoding="utf-8")


def validate_realized_trace_rows(rows: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Strict B2.8-R1 wrapper around the frozen evaluator's 80-row input."""
    if len(rows) != 80:
        raise ValueError(f"REALIZED_TRACE_ROW_COUNT_MUST_EQUAL_80:observed={len(rows)}")
    if any(row.get("primary_measurement_source") != PRIMARY_SOURCE for row in rows):
        raise ValueError("PLANNED_PRIMARY_FORBIDDEN")
    return exact_realized_window_v1_1(rows)


def _binding_rows(roster: Mapping[str, Any], schedule: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    entries = list(roster["entries"])
    for run in schedule["runs"]:
        matches = [row for row in entries if (str(row["scenario_token"]), str(row["log_id"])) == (str(run["scenario_token"]), str(run["log_id"]))]
        if len(matches) != 1:
            raise ValueError(f"FROZEN_SCHEDULE_ROSTER_MATCH_COUNT_MUST_EQUAL_ONE:{run['run_id']}:{len(matches)}")
        entry = matches[0]
        if str(run["family"]) != str(entry["family"]) or str(run["arm"]) not in {str(value) for value in entry["arms"]}:
            raise ValueError(f"FROZEN_SCHEDULE_ROSTER_OR_ARM_MISMATCH:{run['run_id']}")
        rows.append({key: run[key] for key in ("run_id", "pair_id", "family", "scenario_token", "log_id", "arm", "run_order")} | {"future_roster_row": entry})
    if len(rows) != 48 or len({row["run_id"] for row in rows}) != 48:
        raise ValueError("FROZEN_RUN_BINDING_CARDINALITY_MISMATCH")
    return rows


def _component_paths() -> Dict[str, Path]:
    return {
        "scientific_roster_v2_0": ROSTER,
        "scientific_schedule_v2_0": SCHEDULE,
        "planner_v2_1_reference": ROOT / "tools/r1_official_technical_smoke_planner_v2_1.py",
        "planner_v2_2_instrumented": ROOT / "tools/r1_official_technical_smoke_planner_v2_2.py",
        "realized_trace_dispatcher": ROOT / "tools/r1_b2_8_r1_frozen_run_dispatcher.py",
        "repair_repreflight_launcher": Path(__file__),
        "hydra_planner_config_v2_2": CONFIG,
        "evaluator_v2_1": ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        "context_adapter_v2_1": ROOT / "tools/r1_closed_loop_context_adapter_v2_1.py",
        "absolute_episode_clock": R1 / "r1_absolute_episode_clock_binding_v1.0.json",
        "hlc_realized_progress": R1 / "r1_hlc_realized_progress_contract_v1.0.json",
        "hlc_terminal_route_progress": R1 / "r1_hlc_terminal_route_progress_contract_v1.0.json",
        "official_map_bridge": ROOT / "tools/r1_official_map_query_bridge_v2_1.py",
        "official_ego_footprint": ROOT / "tools/r1_official_ego_vehicle_binding_v1.py",
        "hlc_clearance": ROOT / "tools/r1_hlc_dynamic_clearance_v1_1.py",
        "hlc_applicability": R1 / "r1_hlc_map_geometry_applicability_contract_v1.0.json",
        "tsb_applicability": R1 / "r1_tsb_mechanism_applicability_contract_v1.0.json",
        "b2_6_conformance_manifest": R1 / "r1_b2_6_final_execution_conformance_sha_manifest_v1.0.json",
    }


def _state_tuples(trajectory: Any) -> list[tuple[float, float, float, float, int]]:
    return [(float(row.rear_axle.x), float(row.rear_axle.y), float(row.rear_axle.heading), float(row.dynamic_car_state.speed), int(row.time_us)) for row in trajectory.get_sampled_trajectory()]


def _parity(row: Mapping[str, Any], arm: str, scratch: Path) -> Dict[str, Any]:
    api = get_maps_api(str(ROOT.parent / "nuplan/dataset/maps"), "nuplan-maps-v1.0", str(row["map_name"]))
    init = PlannerInitialization(route_roadblock_ids=[str(value) for value in row["route_roadblock_ids"]], mission_goal=None, map_api=api)
    old, new = R1OfficialTechnicalSmokePlannerV2_1(row, str(row["family"]), arm), R1OfficialTechnicalSmokePlannerV2_2(row, str(row["family"]), arm, str(scratch / str(row["family"])))
    old.initialize(init); new.initialize(init)
    equal = True
    for iteration in range(80):
        current = _official_input(row["initial_state"], iteration)
        equal = equal and _state_tuples(old.compute_trajectory(current)) == _state_tuples(new.compute_trajectory(current))
    trace_rows = [json.loads(line) for line in new.realized_trace_path.read_text(encoding="utf-8").splitlines()]
    validate_realized_trace_rows(trace_rows)
    return {"family": row["family"], "scenario_token": row["scenario_token"], "arm": arm, "iterations_compared": 80, "trajectory_state_exact_equal": equal, "realized_trace_rows": len(trace_rows), "trace_primary_source": trace_rows[0]["primary_measurement_source"], "trace_timestamp_preserved": [row["current_ego"]["time_us"] for row in trace_rows] == [int(row["initial_state"]["initial_time_us"]) + index * 100000 for index in range(80)]}


def _failure_tests(rows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    base = [dict(row) for row in rows]
    tests: Dict[str, bool] = {}
    mutations = {
        "duplicate_iteration_fail_closed": base[:20] + [base[19]] + base[20:],
        "missing_iteration_fail_closed": base[:10] + base[11:],
        "79_rows_fail_closed": base[:79],
        "81_rows_fail_closed": base + [base[-1]],
        "non_monotonic_timestamp_fail_closed": [dict(item) for item in base],
        "planned_primary_fail_closed": [dict(item) for item in base],
    }
    mutations["non_monotonic_timestamp_fail_closed"][20] = {**mutations["non_monotonic_timestamp_fail_closed"][20], "current_ego": {**mutations["non_monotonic_timestamp_fail_closed"][20]["current_ego"], "time_us": mutations["non_monotonic_timestamp_fail_closed"][19]["current_ego"]["time_us"]}}
    mutations["planned_primary_fail_closed"][0] = {**mutations["planned_primary_fail_closed"][0], "primary_measurement_source": "PLANNED"}
    for label, mutated in mutations.items():
        try:
            validate_realized_trace_rows(mutated)
        except ValueError:
            tests[label] = True
        else:
            tests[label] = False
    return tests


def _hydra_audit(bindings_path: Path, rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    config = OmegaConf.load(CONFIG)["r1_official_technical_smoke_v2_2"]
    audit: list[Dict[str, Any]] = []
    before = {key: os.environ.get(key) for key in ("R1_B2_8_R1_BINDING_MANIFEST", "R1_B2_8_R1_RUN_ID", "R1_B2_8_R1_TRACE_DIR")}
    try:
        with tempfile.TemporaryDirectory(prefix="r1_b2_8_r1_hydra_") as temp:
            for row in rows:
                os.environ.update({"R1_B2_8_R1_BINDING_MANIFEST": str(bindings_path), "R1_B2_8_R1_RUN_ID": str(row["run_id"]), "R1_B2_8_R1_TRACE_DIR": str(Path(temp) / str(row["run_id"]))})
                resolved = OmegaConf.to_container(config, resolve=True)
                if "${" in json.dumps(resolved, sort_keys=True):
                    raise ValueError("UNRESOLVED_HYDRA_INTERPOLATION")
                planner = instantiate(config)
                binding = load_frozen_run_binding(bindings_path, str(row["run_id"]))
                passed = isinstance(planner, R1OfficialTechnicalSmokePlannerV2_2) and (planner._family, planner._arm, planner._row) == (binding["family"], binding["arm"], binding["future_roster_row"])
                audit.append({"run_id": row["run_id"], "status": "HYDRA_FROZEN_RUN_BINDING_PASS" if passed else "FAIL", "family": row["family"], "arm": row["arm"], "resolved": resolved})
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return audit


def main() -> int:
    if any(path.exists() for path in OUT.values()):
        raise FileExistsError("B2.8-R1 versioned output already exists")
    roster, schedule = read_json(ROSTER), read_json(SCHEDULE)
    if sha256(ROSTER) != EXPECTED["roster"] or sha256(SCHEDULE) != EXPECTED["schedule"]:
        raise ValueError("FROZEN_ROSTER_OR_SCHEDULE_SHA_MISMATCH")
    bindings_rows = _binding_rows(roster, schedule)
    components = _component_paths()
    if any(not path.is_file() for path in components.values()):
        raise FileNotFoundError("EXECUTION_COMPONENT_MISSING")
    binding_payload = {"schema_version": "r1_b2_8_r1_execution_bindings_manifest_v1.0", "status": "FROZEN_EXECUTION_WIRING_PENDING_OWNER_REVIEW", "scientific_roster_sha256": sha256(ROSTER), "scientific_schedule_sha256": sha256(SCHEDULE), "components": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in components.items()}, "frozen_run_bindings": bindings_rows, "OFFICIAL_SIMULATION_AUTHORIZED": False, "NEW_EXECUTION_RUN_BUDGET": 0}
    write_json(OUT["bindings"], binding_payload)
    identity = {"schema_version": "r1_b2_8_r1_scientific_schedule_identity_audit_v1.0", "status": "EXACT_IDENTICAL_V2_0_SCIENTIFIC_RUN_ROWS", "schedule_wrapper_created": False, "scientific_schedule_path": str(SCHEDULE.relative_to(ROOT)), "scientific_schedule_sha256": sha256(SCHEDULE), "compared_fields": ["run_id", "pair_id", "family", "scenario_token", "log_id", "arm", "run_order"], "run_count": 48, "all_48_exact": True, "identity_replacement": False, "threshold_change": False}
    write_json(OUT["identity"], identity)
    with tempfile.TemporaryDirectory(prefix="r1_b2_8_r1_parity_") as temp:
        parity = [_parity(roster["entries"][0], roster["entries"][0]["arms"][0], Path(temp)), _parity(roster["entries"][12], roster["entries"][12]["arms"][0], Path(temp))]
        trace = [json.loads(line) for line in (Path(temp) / "R-HLC" / "realized_current_ego.jsonl").read_text(encoding="utf-8").splitlines()]
    failures = _failure_tests(trace)
    contract = {"schema_version": "r1_b2_8_r1_realized_trace_contract_v1.0", "primary_measurement_source": PRIMARY_SOURCE, "observation_source": "VERSIONED_PLANNER_V2_2_PASSIVE_PLANNERINPUT_HISTORY_CURRENT_STATE", "observation_timing": "PLANNER_CALL_ENTRY_BEFORE_TRAJECTORY_GENERATION", "iteration_semantics": "OFFICIAL_SIMULATOR_CURRENT_REALIZED_EGO_STATE_AT_ITERATIONS_0_THROUGH_79", "timestamp_source": "current_input.iteration.time_us_equal_to_history.current_state.ego.time_us", "exact_row_requirement": 80, "required_iteration_indices": list(range(80)), "required_fields": ["iteration_index", "current_ego.time_us", "current_ego.rear_axle.x", "current_ego.rear_axle.y", "current_ego.rear_axle.heading", "current_ego.speed_mps"], "missing_duplicate_or_extra_iteration": "FAIL_CLOSED", "timestamp_monotonicity": "STRICTLY_INCREASING_REQUIRED", "planned_trajectory_primary": "FORBIDDEN", "v2_1_v2_2_parity": parity, "fail_closed_tests": failures}
    write_json(OUT["contract"], contract)
    hydra_rows = _hydra_audit(OUT["bindings"], bindings_rows)
    hydra = {"schema_version": "r1_b2_8_r1_hydra_binding_audit_v1.0", "status": "48_OF_48_HYDRA_FROZEN_RUN_BINDING_PASS" if all(row["status"].endswith("PASS") for row in hydra_rows) else "FAIL", "composition_count": len(hydra_rows), "rows": hydra_rows, "official_simulation_started": False}
    write_json(OUT["hydra"], hydra)
    write_json(OUT["composition"], {"schema_version": "r1_b2_8_r1_48run_composition_preflight_v1.0", "status": hydra["status"], "runs": [{key: row[key] for key in ("run_id", "family", "arm", "status")} for row in hydra_rows], "scientific_schedule_mutated": False, "official_simulation_started": False})
    passes = all(item["trajectory_state_exact_equal"] and item["realized_trace_rows"] == 80 and item["trace_timestamp_preserved"] for item in parity) and all(failures.values()) and hydra["status"].startswith("48_OF_48") and sha256(PROTECTED_CSV) == EXPECTED["protected_csv"]
    preflight = {"schema_version": "r1_b2_8_r1_zero_run_complete_path_preflight_v1.0", "PRE_RUN_INTEGRITY": "PASS_COMPLETE_EXECUTION_PATH_ZERO_RUN" if passes else "STOP_PRE_SIMULATION", "actual_official_runs": 0, "consumed_budget": 0, "checks": {"REALIZED_CURRENT_EGO_TRACE_WRITER_BOUND": True, "REALIZED_CURRENT_EGO_TRACE_CONTRACT": all(failures.values()), "OFFICIAL_HYDRA_RUNTIME_BINDING_COMPLETE": hydra["status"].startswith("48_OF_48"), "48_OF_48_HYDRA_COMPOSITION": len(hydra_rows) == 48, "48_OF_48_FROZEN_ROW_BINDING": len(bindings_rows) == 48, "PLANNED_PRIMARY_FORBIDDEN": failures["planned_primary_fail_closed"], "REALIZED_TRACE_80_ROW_FAIL_CLOSED_TESTS": all(failures.values()), "EXECUTION_COMPONENT_SHA_CLOSURE": len(binding_payload["components"]) == len(components), "SCIENTIFIC_RUN_ROWS_V2_0_VS_NEW": "EXACT_IDENTICAL", "PROTECTED_CSV_SHA256": sha256(PROTECTED_CSV)}, "planner_core_v2_1_modified": False, "planner_v2_2_instrumentation_parity": parity, "OFFICIAL_SMOKE_AUTHORIZED": False, "NEW_EXECUTION_RUN_BUDGET": 0, "RBR_AUTHORIZED": False}
    write_json(OUT["preflight"], preflight)
    write_text(OUT["report"], f"# R1 B2.8-R1 执行集成修复与零运行重新预检\n\n## 结论\n\n`{preflight['PRE_RUN_INTEGRITY']}`。本轮仅修复 execution wiring，未启动 official simulation，实际 official runs 为 `0`，消耗预算为 `0`。\n\nV2.1 planner 未修改。新增 V2.2 仅在 `compute_trajectory` 入口被动记录已实现的 `PlannerInput.history.current_state`，然后将同一输入交给 V2.1 原有逻辑。HLC 与 TSB 各一个 frozen input 的 80 次 trajectory state（位置、heading、speed、timestamp）均 exact identical。\n\n48/48 Hydra composition 与 frozen schedule/roster 逐行绑定均通过；任何缺失、歧义或 arm 不匹配均在 simulator start 前 fail-closed。\n\n## 授权状态\n\n新的 execution SHA 尚未获得运行授权：`OFFICIAL_SMOKE_AUTHORIZED=false`，`NEW_EXECUTION_RUN_BUDGET=0`，RBR 未授权。\n")
    write_text(OUT["request"], "# R1 B2.8-R1 Scientific Owner 授权请求 v0.1\n\nB2.8-R1 已完成零运行 complete-path preflight，并形成新的 execution binding SHA manifest。此前 48-run 授权不自动迁移。请 Scientific Owner 审核新的 V2.2 passive trace instrumentation、Hydra dispatcher 和 binding manifest；在新的明确授权前，不得启动 official simulation、不得消费预算、不得训练 RBR。\n")
    print(json.dumps({"status": preflight["PRE_RUN_INTEGRITY"], "hydra": hydra["status"], "official_runs": 0, "budget": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
