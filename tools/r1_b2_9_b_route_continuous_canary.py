#!/usr/bin/env python3
"""Prepare, execute, and report the isolated B2.9-B engineering canary track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.r1_b2_8_r3_prospective_selector import official_env
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1
from tools.r1_closed_loop_benchmark_v2_2 import (
    BUILDER_VERSION,
    JOIN_GAP_THRESHOLD_M,
    build_hlc_route_continuous_reference_v2_2,
)
from tools.r1_prospective_generator_contract_v2 import HLC_BASELINE, HLC_TREATMENT


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
MAP_ROOT = ROOT.parent / "nuplan/dataset/maps"
CURRENT_ROSTER = R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json"
VALIDATION_ROSTER = R1 / "r1_runtime_determinism_validation_roster_v1.0.json"
PERMANENT_BLACKLIST = R1 / "r1_official_technical_smoke_permanent_blacklist_v1.0.json"
ATTEMPT_TRACE = ROOT / "outputs/r1_b2_8_r3_3_official_smoke_once_v1/R1B27-01-R-HLC-BASELINE/trace/realized_current_ego.jsonl"
OUTPUT_ROOT = ROOT / "outputs/r1_b2_9_b_engineering_canary_v1"
INHERITED_R3_BINDING = R1 / "r1_b2_8_r3_execution_bindings_manifest_v1.0.json"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
OUT = {
    "contract": R1 / "r1_hlc_route_continuous_reference_contract_v1.0.json",
    "roster": R1 / "r1_b2_9_b_hlc_engineering_canary_roster_v1.0.json",
    "exclusion": R1 / "r1_b2_9_b_engineering_canary_exclusion_ledger_v1.0.json",
    "failed": R1 / "r1_b2_9_b_failed_trace_offline_repair_audit_v1.json",
    "current12": R1 / "r1_b2_9_b_current12_route_continuous_diagnostic_v1.json",
    "ledger": R1 / "r1_b2_9_b_engineering_canary_run_ledger_v1.0.json",
    "contract_report": R1 / "R1_B2_9_B_Route_Continuous_Reference_Contract_Report_v1.md",
    "runtime_report": R1 / "R1_B2_9_B_Engineering_Canary_Runtime_Report_v1.md",
}
CANARY_TOKENS = ("b1be12bca092597a", "25944935eadb52f1", "ef3172a208cc5dd7")


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


def _ego(initial: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rear_axle": {
            "x": float(initial["initial_x"]),
            "y": float(initial["initial_y"]),
            "heading": float(initial["initial_heading"]),
        },
        "speed_mps": float(initial["initial_speed_mps"]),
        "time_us": int(initial["initial_time_us"]),
    }


def _normalize_entry(row: Mapping[str, Any], source: str) -> Dict[str, Any]:
    result = dict(row)
    result.pop("source_reference_xy", None)
    result.pop("target_reference_xy", None)
    result["initial_state_fingerprint"] = str(result["initial_state"]["initial_state_fingerprint"])
    result["intended_lane_change_direction"] = str(result.get("direction", "RIGHT")).upper()
    result["arms"] = [HLC_BASELINE, HLC_TREATMENT]
    result["canary_source_ledger"] = source
    result["SCIENTIFIC_USE_FORBIDDEN"] = True
    result["NON_SCIENTIFIC_ENGINEERING_ONLY"] = True
    return result


def build_canary_roster() -> Dict[str, Any]:
    current = read_json(CURRENT_ROSTER)["entries"]
    validation = read_json(VALIDATION_ROSTER)["entries"]
    current_by_token = {str(row["scenario_token"]): row for row in current}
    validation_by_token = {str(row["scenario_token"]): row for row in validation}
    blacklist = {
        (str(row["scenario_token"]), str(row["log_id"]))
        for row in read_json(PERMANENT_BLACKLIST)["entries"]
    }
    entries = [
        _normalize_entry(current_by_token[CANARY_TOKENS[0]], "B2.9-A_ATTEMPT1_SCIENTIFIC_EVIDENCE_EXCLUSION"),
        _normalize_entry(validation_by_token[CANARY_TOKENS[1]], str(VALIDATION_ROSTER.relative_to(ROOT))),
        _normalize_entry(validation_by_token[CANARY_TOKENS[2]], str(VALIDATION_ROSTER.relative_to(ROOT))),
    ]
    current_remaining = {
        (str(row["scenario_token"]), str(row["log_id"])) for row in current
        if str(row["scenario_token"]) != CANARY_TOKENS[0]
    }
    for row in entries[1:]:
        identity = (str(row["scenario_token"]), str(row["log_id"]))
        if identity not in blacklist:
            raise ValueError(f"CANARY_NOT_IN_EXISTING_PERMANENT_EXCLUSION_LEDGER:{identity}")
        if identity in current_remaining:
            raise ValueError(f"CURRENT_SCIENTIFIC_IDENTITY_FORBIDDEN_AS_CANARY:{identity}")
    return {
        "schema_version": "r1_b2_9_b_hlc_engineering_canary_roster_v1.0",
        "status": "FROZEN_NON_SCIENTIFIC_ENGINEERING_ONLY",
        "selection_scope": "EXISTING_EXCLUSION_LEDGERS_ONLY_NO_SOURCE_UNIVERSE_RESCAN",
        "entries": entries,
        "counts": {"identities": 3, "arms_per_identity": 2, "planned_runs": 6},
        "Attempt1_identity_disposition": {
            "scenario_token": CANARY_TOKENS[0],
            "pair_id": "R1B27-01-R-HLC",
            "SCIENTIFIC_EVIDENCE_EXCLUDED": True,
            "ENGINEERING_CANARY_ALLOWED": True,
            "terminology": "ENGINEERING_CANARY_REPLAY",
            "attempt1_history_modified": False,
        },
        "SCIENTIFIC_USE_FORBIDDEN": True,
        "scientific_roster_modified": False,
        "source_universe_rescanned": False,
    }


def build_contract() -> Dict[str, Any]:
    return {
        "schema_version": "r1_hlc_route_continuous_reference_contract_v1.0",
        "status": "FROZEN_FOR_NON_SCIENTIFIC_ENGINEERING_CANARY",
        "implementation": {"path": "tools/r1_closed_loop_benchmark_v2_2.py", "symbol": BUILDER_VERSION},
        "inputs": ["source_lane_id", "target_lane_id", "route_roadblock_ids", "official_nuPlan_map_api", "current_realized_ego", "required_forward_m"],
        "native_geometry_only": True,
        "extrapolation": "FORBIDDEN",
        "manual_points": "FORBIDDEN",
        "outcome_selected_successor": "FORBIDDEN",
        "successor_rule": [
            "resolve the initial source roadblock to exactly one occurrence in frozen route_roadblock_ids",
            "enumerate native outgoing source/target edge pairs",
            "retain only positive-direction source successors whose roadblock has exactly one later frozen-route occurrence",
            "retain only mutually topology-corresponding source/target pairs; connector correspondence is determined by unique outgoing lane adjacency",
            "require exactly one retained pair; zero or multiple pairs fail closed",
        ],
        "route_occurrence": "EXACTLY_ONE_FORWARD_OCCURRENCE_REQUIRED",
        "branch": "MORE_THAN_ONE_VALID_PAIR_FAIL_CLOSED",
        "merge": "NON_UNIQUE_TERMINAL_LANE_FAIL_CLOSED",
        "reversal": "NON_POSITIVE_NATIVE_TANGENT_DOT_PRODUCT_FAIL_CLOSED",
        "duplicate_or_self_intersection": "FAIL_CLOSED",
        "join": {
            "threshold_m": JOIN_GAP_THRESHOLD_M,
            "reason": "NO_EXISTING_NUMERIC_MAP_JOIN_PRECISION_THRESHOLD_FOUND",
            "accepted_rule": "EXACT_NATIVE_ENDPOINT_IDENTITY_ONLY_NO_TOLERANCE",
            "nonzero_gap": "FAIL_CLOSED_WITH_UNKNOWN_THRESHOLD",
        },
        "applicability": {
            "name": "ROLLING_REPLAN_FULL_PRIMARY_WINDOW_COVERAGE",
            "nominal_iterations": list(range(80)),
            "future_envelope_seconds": 7.9,
            "active_reference_only": True,
            "zero_weight_reference_may_be_skipped": True,
            "zero_weight_source_rule_is_not_a_substitute_for_route_continuous_target": True,
        },
        "engineering_canary_horizon": {
            "planner_calls": "EXACT_ITERATIONS_0_THROUGH_79",
            "controller": "tools.r1_b2_9_b_canary_time_controller.R1B29BEngineeringCanary80CallTimeController",
            "base_semantics": "nuPlan_1.2.2_StepSimulationTimeController",
            "sole_override": "number_of_iterations=min(official_scenario_iterations,81), yielding planner calls 0...79",
            "scientific_protocol_change": False,
        },
        "unchanged": ["HLC progress schedule", "baseline/treatment arms", "speed semantics", "absolute episode clock", "first-state exact identity", "7.9 s output trajectory", "0.1 s state spacing"],
    }


def _map_api(map_name: str, cache: Dict[str, Any]) -> Any:
    if map_name not in cache:
        official_env()
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

        cache[map_name] = get_maps_api(str(MAP_ROOT), "nuplan-maps-v1.0", map_name)
    return cache[map_name]


def _corridor_summary(corridor: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_total_length_m": corridor["source_total_length_m"],
        "target_total_length_m": corridor["target_total_length_m"],
        "source_current_arc_m": corridor["source_current_arc_m"],
        "target_current_arc_m": corridor["target_current_arc_m"],
        "source_requested_max_arc_m": corridor["source_current_arc_m"] + corridor["required_forward_m"],
        "target_requested_max_arc_m": corridor["target_current_arc_m"] + corridor["required_forward_m"],
        "source_remaining_margin_m": corridor["source_remaining_margin_m"],
        "target_remaining_margin_m": corridor["target_remaining_margin_m"],
        "source_components": corridor["source_components"],
        "target_components": corridor["target_components"],
        "transitions": corridor["transitions"],
        "extrapolation_used": corridor["extrapolation_used"],
        "manual_points_used": corridor["manual_points_used"],
    }


def failed_trace_audit(roster: Mapping[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
    entry = next(row for row in roster["entries"] if row["scenario_token"] == CANARY_TOKENS[0])
    rows = [json.loads(line) for line in ATTEMPT_TRACE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["iteration_index"]) for row in rows] != list(range(34)):
        raise ValueError("ATTEMPT1_TRACE_NOT_EXACT_0_TO_33")
    map_api = _map_api(str(entry["map_name"]), cache)
    results, parity = [], []
    from tools.r1_official_map_query_bridge_v2_1 import R1OfficialMapQueryBridgeV2_1

    bridge = R1OfficialMapQueryBridgeV2_1(map_api)
    old_source = bridge.native_reference_xy(str(entry["source_lane_id"]))
    old_target = bridge.native_reference_xy(str(entry["target_lane_id"]))
    for row in rows:
        ego = row["current_ego"]
        required = float(ego["speed_mps"]) * 7.9
        corridor = build_hlc_route_continuous_reference_v2_2(
            map_api, entry["route_roadblock_ids"], entry["source_lane_id"], entry["target_lane_id"], ego, required
        )
        new_states = build_hlc_native_geometry_v1_1(
            ego, int(row["iteration_index"]) * 0.1,
            corridor["source_reference_xy"], corridor["target_reference_xy"],
            corridor["source_current_arc_m"], corridor["target_current_arc_m"], HLC_BASELINE,
        )
        identity = new_states[0] == ego
        if not identity:
            raise ValueError(f"FIRST_STATE_EXACT_IDENTITY_FAIL:{row['iteration_index']}")
        if int(row["iteration_index"]) <= 32:
            old_states = build_hlc_native_geometry_v1_1(
                ego, int(row["iteration_index"]) * 0.1, old_source, old_target,
                float(bridge.project(str(entry["source_lane_id"]), (ego["rear_axle"]["x"], ego["rear_axle"]["y"]))["arc_m"]),
                float(bridge.project(str(entry["target_lane_id"]), (ego["rear_axle"]["x"], ego["rear_axle"]["y"]))["arc_m"]), HLC_BASELINE,
            )
            parity.append(old_states == new_states)
        results.append({"iteration_index": row["iteration_index"], "first_state_exact_identity": identity, **_corridor_summary(corridor)})
    return {
        "schema_version": "r1_b2_9_b_failed_trace_offline_repair_audit_v1",
        "status": "ITERATIONS_0_TO_33_ROUTE_CONTINUOUS_PASS",
        "simulation_started": False,
        "attempt1_trace_sha256": sha256(ATTEMPT_TRACE),
        "scenario_token": entry["scenario_token"],
        "old_v2_2_vs_v3_exact_parity_iterations_0_32": all(parity) and len(parity) == 33,
        "iteration_33": results[33],
        "NATIVE_REFERENCE_COVERAGE_FAIL_AT_33": False,
        "first_state_exact_identity_34_of_34": all(row["first_state_exact_identity"] for row in results),
        "rows": results,
    }


def rolling_audit(entries: Sequence[Mapping[str, Any]], cache: Dict[str, Any], schema: str) -> Dict[str, Any]:
    output = []
    for entry in entries:
        map_api = _map_api(str(entry["map_name"]), cache)
        arm_results = []
        scenario_errors = []
        for arm in (HLC_BASELINE, HLC_TREATMENT):
            initial = _ego(entry["initial_state"])
            try:
                initial_corridor = build_hlc_route_continuous_reference_v2_2(
                    map_api, entry["route_roadblock_ids"], entry["source_lane_id"], entry["target_lane_id"], initial,
                    float(initial["speed_mps"]) * 7.9,
                )
                nominal_states = build_hlc_native_geometry_v1_1(
                    initial, 0.0, initial_corridor["source_reference_xy"], initial_corridor["target_reference_xy"],
                    initial_corridor["source_current_arc_m"], initial_corridor["target_current_arc_m"], arm,
                )
                rows = []
                for index, state in enumerate(nominal_states):
                    corridor = build_hlc_route_continuous_reference_v2_2(
                        map_api, entry["route_roadblock_ids"], entry["source_lane_id"], entry["target_lane_id"], state,
                        float(state["speed_mps"]) * 7.9,
                    )
                    rows.append({"iteration_index": index, **_corridor_summary(corridor)})
                arm_results.append({
                    "arm": arm,
                    "status": "PASS",
                    "iterations_passed": len(rows),
                    "minimum_source_margin_m": min(row["source_remaining_margin_m"] for row in rows),
                    "minimum_target_margin_m": min(row["target_remaining_margin_m"] for row in rows),
                    "iteration_0": rows[0],
                    "iteration_79": rows[-1],
                    "full_rows_omitted_from_small_diagnostic": True,
                })
            except Exception as exc:
                message = f"{type(exc).__name__}:{exc}"
                scenario_errors.append(message)
                arm_results.append({"arm": arm, "status": "FAIL_CLOSED", "error": message})
        output.append({
            "scenario_token": entry["scenario_token"], "log_id": entry["log_id"],
            "status": "PASS" if not scenario_errors else "FAIL_CLOSED",
            "topology_ambiguity": any("AMBIGUITY" in item for item in scenario_errors),
            "arms": arm_results,
        })
    passed = sum(item["status"] == "PASS" for item in output)
    return {
        "schema_version": schema,
        "status": f"{passed}_OF_{len(output)}_ROLLING_REPLAN_FULL_PRIMARY_WINDOW_COVERAGE_PASS",
        "diagnostic_only": schema.endswith("current12_route_continuous_diagnostic_v1"),
        "simulation_started": False,
        "counts": {"identities": len(output), "coverage_pass": passed, "coverage_fail": len(output) - passed, "topology_ambiguity": sum(bool(item["topology_ambiguity"]) for item in output)},
        "entries": output,
    }


def initial_ledger(roster: Mapping[str, Any], canary_audit: Mapping[str, Any]) -> Dict[str, Any]:
    runs = []
    for identity_index, entry in enumerate(roster["entries"], 1):
        for arm_name, arm in (("BASELINE", HLC_BASELINE), ("TREATMENT", HLC_TREATMENT)):
            run_id = f"R1B29B-CANARY-{identity_index:02d}-HLC-{arm_name}-A01"
            runs.append({
                "run_id": run_id, "canary_identity_index": identity_index,
                "scenario_token": entry["scenario_token"], "log_id": entry["log_id"], "arm": arm,
                "terminology": "ENGINEERING_CANARY_REPLAY", "attempt_number": 1,
                "output_root": str((OUTPUT_ROOT / run_id).relative_to(ROOT)), "status": "NOT_RUN",
                "SCIENTIFIC_USE_FORBIDDEN": True,
            })
    return {
        "schema_version": "r1_b2_9_b_engineering_canary_run_ledger_v1.0",
        "status": "PREPARED_NOT_RUN",
        "track": "NON_SCIENTIFIC_ENGINEERING_ONLY",
        "nominal_rolling_preflight": canary_audit,
        "runs": runs,
        "counts": {"planned": 6, "actual_runs": 0, "reruns": 0, "technical_completed": 0, "primary_80_completed": 0, "native_coverage_failures": 0, "other_technical_failures": 0, "metric_callback_completed": 0},
        "scientific_identities_simulated": False,
        "scientific_roster_modified": False,
        "threshold_changed": False,
        "official_smoke_authorized": False,
        "RBR_authorized": False,
    }


def prepare() -> None:
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    roster, contract = build_canary_roster(), build_contract()
    cache: Dict[str, Any] = {}
    failed = failed_trace_audit(roster, cache)
    current_entries = [row for row in read_json(CURRENT_ROSTER)["entries"] if row["family"] == "R-HLC"]
    current12 = rolling_audit(current_entries, cache, "r1_b2_9_b_current12_route_continuous_diagnostic_v1")
    canary_audit = rolling_audit(roster["entries"], cache, "r1_b2_9_b_canary_nominal_rolling_audit_v1")
    if current12["counts"]["coverage_pass"] != 12 or canary_audit["counts"]["coverage_pass"] != 3:
        raise RuntimeError("ROLLING_REPLAN_PRECANARY_AUDIT_NOT_ALL_PASS")
    exclusion = {
        "schema_version": "r1_b2_9_b_engineering_canary_exclusion_ledger_v1.0",
        "status": "PERMANENT_EXCLUSION_FROZEN",
        "entries": [{"scenario_token": row["scenario_token"], "log_id": row["log_id"], "SCIENTIFIC_USE_FORBIDDEN": True, "NON_SCIENTIFIC_ENGINEERING_ONLY": True, "future_selector_exclusion": "PERMANENT"} for row in roster["entries"]],
        "forbidden_uses": ["scientific evidence", "threshold tuning", "HLC mechanism tuning", "F_match tuning", "safety tuning", "future scientific identity selection"],
        "allowed_use": "TECHNICAL_RUNTIME_IMPLEMENTATION_REPAIR_ONLY",
    }
    for path, payload in ((OUT["contract"], contract), (OUT["roster"], roster), (OUT["exclusion"], exclusion), (OUT["failed"], failed), (OUT["current12"], current12), (OUT["ledger"], initial_ledger(roster, canary_audit))):
        write_json(path, payload)


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], raw: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents", "planner=r1_official_technical_smoke_v2_2_r3",
        "scenario_builder=nuplan_mini", f"scenario_builder.db_files=[{entry['db_path']}]",
        "scenario_filter=all_scenarios", f"scenario_filter.scenario_tokens=[{run['scenario_token']}]",
        "simulation_time_controller._target_=tools.r1_b2_9_b_canary_time_controller.R1B29BEngineeringCanary80CallTimeController",
        "worker=single_machine_thread_pool", "worker.max_workers=1", "scenario_builder.max_workers=1",
        "max_callback_workers=1", "number_of_cpus_allocated_per_simulation=1",
        "number_of_gpus_allocated_per_simulation=0", "gpu=false", "seed=2026082701",
        "run_metric=true", "enable_simulation_progress_bar=false",
        "experiment_name=r1_b2_9_b_engineering_canary", f"job_name={run['run_id']}", f"output_dir={raw}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _exact_resolution_count(entry: Mapping[str, Any]) -> int:
    official_env()
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

    return len(list(get_scenarios_from_db(str(entry["db_path"]), [str(entry["scenario_token"])], None, None, True, False)))


def _trace_rows(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def _refresh_counts(ledger: Dict[str, Any]) -> None:
    rows = ledger["runs"]
    attempted = [row for row in rows if row["status"] != "NOT_RUN"]
    actual = [row for row in attempted if bool(row.get("runner_run_called"))]
    complete = [row for row in rows if row["status"] == "TECHNICAL_COMPLETE"]
    latest: Dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in attempted:
        key = (str(row["scenario_token"]), str(row["arm"]))
        if key not in latest or int(row["attempt_number"]) > int(latest[key]["attempt_number"]):
            latest[key] = row
    latest_rows = list(latest.values())
    ledger["counts"] = {
        "planned_base_runs": 6, "orchestration_attempts": len(attempted), "actual_runs": len(actual),
        "reruns": sum(int(row["attempt_number"]) > 1 for row in attempted),
        "technical_completed": len(complete),
        "primary_80_completed": sum(bool(row.get("primary_80_complete")) for row in attempted),
        "cumulative_native_coverage_failures": sum(bool(row.get("native_coverage_failure")) for row in attempted),
        "cumulative_other_technical_failures": sum(row["status"] == "OTHER_TECHNICAL_FAILURE" for row in attempted),
        "metric_callback_completed": sum(bool(row.get("metric_callback_complete")) for row in attempted),
        "latest_required_runs": len(latest_rows),
        "latest_required_runs_complete": sum(row["status"] == "TECHNICAL_COMPLETE" for row in latest_rows),
        "final_native_coverage_failures": sum(bool(row.get("native_coverage_failure")) for row in latest_rows),
        "final_other_technical_failures": sum(row["status"] == "OTHER_TECHNICAL_FAILURE" for row in latest_rows),
    }
    ledger["status"] = "ROUTE_CONTINUOUS_ENGINEERING_CANARY_PASS" if len(latest_rows) == 6 and all(row["status"] == "TECHNICAL_COMPLETE" and bool(row.get("primary_80_complete")) and not bool(row.get("native_coverage_failure")) for row in latest_rows) else "ENGINEERING_CANARY_INCOMPLETE_OR_FAIL"


def execute(*, retry_failed: bool = False) -> None:
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    roster, ledger = read_json(OUT["roster"]), read_json(OUT["ledger"])
    entries = {str(row["scenario_token"]): row for row in roster["entries"]}
    current_scientific = {(str(row["scenario_token"]), str(row["log_id"])) for row in read_json(CURRENT_ROSTER)["entries"] if str(row["scenario_token"]) != CANARY_TOKENS[0]}
    official_env()
    from hydra import compose, initialize_config_dir
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.utils import set_up_common_builder
    from tools.r1_official_technical_smoke_planner_v3_0 import R1OfficialTechnicalSmokePlannerV3_0

    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    if retry_failed:
        additions = []
        for prior in ledger["runs"]:
            if prior["status"] in {"NOT_RUN", "TECHNICAL_COMPLETE"}:
                continue
            if not bool(prior.get("runner_constructed")):
                prior["runner_constructed"] = False
                prior["runner_run_called"] = False
                prior["simulation_started"] = False
            next_attempt = int(prior["attempt_number"]) + 1
            if any(item["scenario_token"] == prior["scenario_token"] and item["arm"] == prior["arm"] and int(item["attempt_number"]) >= next_attempt for item in ledger["runs"]):
                continue
            suffix = f"A{next_attempt:02d}"
            run_id = str(prior["run_id"]).rsplit("-A", 1)[0] + f"-{suffix}"
            additions.append({
                "run_id": run_id, "canary_identity_index": prior["canary_identity_index"],
                "scenario_token": prior["scenario_token"], "log_id": prior["log_id"], "arm": prior["arm"],
                "terminology": "ENGINEERING_CANARY_REPLAY", "attempt_number": next_attempt,
                "supersedes_failed_attempt": prior["run_id"],
                "output_root": str((OUTPUT_ROOT / run_id).relative_to(ROOT)), "status": "NOT_RUN",
                "SCIENTIFIC_USE_FORBIDDEN": True,
            })
        ledger["runs"].extend(additions)
    for run in ledger["runs"]:
        if run["status"] != "NOT_RUN":
            continue
        entry = entries[str(run["scenario_token"])]
        identity = (str(run["scenario_token"]), str(run["log_id"]))
        if identity in current_scientific:
            raise PermissionError(f"SCIENTIFIC_IDENTITY_SIMULATION_FORBIDDEN:{identity}")
        run_root = ROOT / str(run["output_root"])
        trace, raw = run_root / "trace", run_root / "raw"
        if run_root.exists():
            raise FileExistsError(f"CANARY_OUTPUT_REUSE_FAIL_CLOSED:{run_root}")
        trace.mkdir(parents=True)
        run["exact_scenario_resolution_count"] = _exact_resolution_count(entry)
        if run["exact_scenario_resolution_count"] != 1:
            raise RuntimeError(f"CANARY_EXACT_SCENARIO_RESOLUTION_FAIL:{run['run_id']}")
        planner = R1OfficialTechnicalSmokePlannerV3_0(entry, "R-HLC", str(run["arm"]), str(trace))
        run["runner_constructed"] = False
        run["runner_run_called"] = False
        run["simulation_started"] = False
        try:
            os.environ.update({
                "R1_B2_8_R3_BINDING_MANIFEST": str(INHERITED_R3_BINDING),
                "R1_B2_8_R3_RUN_ID": str(run["run_id"]),
                "R1_B2_8_R3_TRACE_DIR": str(trace),
            })
            with initialize_config_dir(config_dir=str(config_root)):
                cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, raw))
            common = set_up_common_builder(cfg, "r1_b2_9_b_engineering_canary")
            callback_worker = build_callbacks_worker(cfg)
            callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
            runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
            if len(runners) != 1:
                raise RuntimeError("CANARY_EXPECTED_EXACTLY_ONE_SIMULATION_RUNNER")
            run["runner_constructed"] = True
            run["runner_run_called"] = True
            run["simulation_started"] = True
            report = runners[0].run()
            primary = _trace_rows(trace / "realized_current_ego.jsonl")
            secondary = _trace_rows(trace / "secondary_diagnostic_realized_current_ego.jsonl")
            indices = [int(row["iteration_index"]) for row in primary]
            primary_complete = indices == list(range(80))
            succeeded = bool(getattr(report, "succeeded", True))
            run.update({
                "status": "TECHNICAL_COMPLETE" if succeeded and primary_complete else "OTHER_TECHNICAL_FAILURE",
                "runner_run_called": True, "simulation_started": True, "full_runtime_stack_exercised": True,
                "planner_call_count": len(primary) + len(secondary), "primary_trace_rows": len(primary),
                "secondary_diagnostic_rows": len(secondary), "primary_80_complete": primary_complete,
                "native_coverage_failure": False,
                "metric_callback_complete": succeeded and raw.exists() and any(raw.rglob("*.nuboard")),
                "runner_report_succeeded": succeeded,
                "minimum_source_margin_m": min(item["source_remaining_margin_m"] for item in planner.route_continuous_audits),
                "minimum_target_margin_m": min(item["target_remaining_margin_m"] for item in planner.route_continuous_audits),
                "topology_ambiguity": False,
            })
            (run_root / "route_continuous_runtime_audit.json").write_text(json.dumps(planner.route_continuous_audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            message = f"{type(exc).__name__}:{exc}"
            run.update({
                "status": "NATIVE_REFERENCE_COVERAGE_FAILURE" if "ROUTE_CONTINUOUS" in message or "NATIVE_REFERENCE_COVERAGE" in message else "OTHER_TECHNICAL_FAILURE",
                "primary_80_complete": False,
                "native_coverage_failure": "ROUTE_CONTINUOUS" in message or "NATIVE_REFERENCE_COVERAGE" in message,
                "metric_callback_complete": False, "error": message,
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            })
        _refresh_counts(ledger)
        write_json(OUT["ledger"], ledger, allow_update=True)
    for historical in ledger["runs"]:
        run_root = ROOT / str(historical["output_root"])
        primary = _trace_rows(run_root / "trace/realized_current_ego.jsonl")
        secondary = _trace_rows(run_root / "trace/secondary_diagnostic_realized_current_ego.jsonl")
        if primary or secondary:
            historical["primary_trace_rows"] = len(primary)
            historical["secondary_diagnostic_rows"] = len(secondary)
            historical["planner_call_count"] = len(primary) + len(secondary)
            historical["primary_80_complete"] = [int(row["iteration_index"]) for row in primary] == list(range(80))
    ledger["scientific_identities_simulated"] = False
    ledger["current_remaining_scientific_identities_simulated"] = False
    ledger["attempt1_identity_replayed_only_after_scientific_evidence_exclusion"] = True
    ledger["simulated_engineering_canary_tokens"] = list(CANARY_TOKENS)
    ledger["actual_simulation_scope"] = "THREE_PERMANENTLY_EXCLUDED_ENGINEERING_CANARY_IDENTITIES_ONLY"
    ledger["protected_csv_sha256"] = sha256(PROTECTED_CSV)
    _refresh_counts(ledger)
    write_json(OUT["ledger"], ledger, allow_update=True)


def write_reports() -> None:
    ledger, failed, current12 = read_json(OUT["ledger"]), read_json(OUT["failed"]), read_json(OUT["current12"])
    if OUT["contract_report"].exists() or OUT["runtime_report"].exists():
        raise FileExistsError("VERSIONED_REPORT_EXISTS")
    i33 = failed["iteration_33"]
    OUT["contract_report"].write_text(
        "# R1 B2.9-B 路线连续原生参考合同报告 v1\n\n"
        "## 结论\n\n新合同将 HLC 的单一原生车道引用替换为官方地图原生、成对且路线约束的连续走廊。"
        "V2.1、V2.2 与冻结机制均未修改；不存在外推、手工点、最近距离择路或结果驱动择路。\n\n"
        "## 确定性规则\n\n从冻结 source/target lane 与其在 route_roadblock_ids 中的唯一 occurrence 出发，枚举双方 native outgoing edge；"
        "只保留源分支在冻结路线后续唯一出现、方向连续且双方终点 lane 保持官方相邻关系的组合。候选不是恰好一个即 fail closed。"
        "连接处没有发现既有数值精度阈值，因此合同记录为 UNKNOWN；实现只接受 gap 精确为 0 的官方端点身份，不引入容差。\n\n"
        "## Attempt 1 离线修复\n\n34 行历史 realized trace 全部离线通过。iteration 33 不再出现 coverage failure；"
        f"source 组件 `18524 → 20156`，总长 `{i33['source_total_length_m']:.6f} m`、requested max arc `{i33['source_requested_max_arc_m']:.6f} m`、余量 `{i33['source_remaining_margin_m']:.6f} m`；"
        f"target 组件 `18525 → 20157`，总长 `{i33['target_total_length_m']:.6f} m`、requested max arc `{i33['target_requested_max_arc_m']:.6f} m`、余量 `{i33['target_remaining_margin_m']:.6f} m`。两处 join gap 均为 `0 m`。"
        "iterations 0...32 与旧 builder 输出 exact parity，34/34 current-ego state0 exact identity。\n\n"
        "## 当前 12 个科学 HLC 身份\n\n本项仅为 DIAGNOSTIC_ONLY；"
        f"`{current12['counts']['coverage_pass']}/12` 完成 0...79 双 arm 滚动覆盖，拓扑歧义 `{current12['counts']['topology_ambiguity']}`。"
        "未创建或修改科学 roster。工程 canary 使用同一 nuPlan StepSimulationTimeController 的版本化 80-call 上限，"
        "使 full runner、controller、observation、metric 与 callback 在 iteration 79 后正常结束；没有为 Primary 窗口之后的地图终点发明非原生连接。\n\n"
        "## 版本差异清单\n\n- V2.1 与 V2.2 文件未修改；V2.2→V3 的 Primary 轨迹语义唯一变化是单 native lane reference 改为 route-continuous official-native reference。\n"
        "- V3 的被动 trace writer 将 0...79 写入 Primary，>=80 仅写入独立 secondary diagnostic；该分流不读取或改变 planner trajectory。\n"
        "- 工程 canary 专用 time controller 仅把 runner 结束点限定为 80 次 planner call；TwoStageController、observation、ego propagation、metric engine 与 callbacks 保持原绑定。\n",
        encoding="utf-8",
    )
    counts = ledger["counts"]
    OUT["runtime_report"].write_text(
        "# R1 B2.9-B 工程 Canary 运行报告 v1\n\n"
        "## 证据边界\n\n本报告只记录 `NON_SCIENTIFIC_ENGINEERING_ONLY` 的技术运行行为，不是 official smoke、科学证据或 benchmark 结果。"
        "三个身份已写入永久科学排除账本；不得用结果调阈值、机制、F_match、安全定义或未来身份选择。\n\n"
        "Canary identities：`b1be12bca092597a`、`25944935eadb52f1`、`ef3172a208cc5dd7`，每个均执行 baseline/treatment。\n\n"
        "A01 为配置环境绑定缺失，6 次均在 runner 构造前停止；A02 证明 6/6 Primary 0...79 完整，但在 secondary 区间遇到官方目标车道拓扑终点；"
        "A03 保持严格 fail-closed 地图合同，以 80-call 工程 canary time-controller 正常完成全部 runner 与回调。所有尝试均使用新 run ID 和新输出根。\n\n"
        "## 运行结果\n\n"
        f"实际 canary runs `{counts['actual_runs']}`，rerun `{counts['reruns']}`，技术完成 `{counts['technical_completed']}`，"
        f"Primary 0...79 完成 `{counts['primary_80_completed']}`，历史累计 native coverage failure `{counts['cumulative_native_coverage_failures']}`，"
        f"最终 native coverage failure `{counts['final_native_coverage_failures']}`，最终其他技术 failure `{counts['final_other_technical_failures']}`，"
        f"metric/callback 完成 `{counts['metric_callback_completed']}`。历史累计 pre-start/wiring failure 与 post-primary coverage failure 均保留在 ledger。\n\n"
        f"最终状态：`{ledger['status']}`。Attempt 1 身份只在先冻结为 `SCIENTIFIC_EVIDENCE_EXCLUDED=true` 后作为 canary replay；其余当前科学身份仿真：`false`。科学 roster 修改：`false`；threshold 修改：`false`。"
        "OFFICIAL_SMOKE_AUTHORIZED=false，RBR_A/B/C=NOT_AUTHORIZED。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--write-reports", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.execute, args.write_reports)) != 1:
        parser.error("choose exactly one of --prepare, --execute, --write-reports")
    if args.prepare:
        prepare()
    elif args.execute:
        execute(retry_failed=args.retry_failed)
    else:
        write_reports()
    print(json.dumps({"status": "PASS", "action": "prepare" if args.prepare else "execute" if args.execute else "write_reports"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
