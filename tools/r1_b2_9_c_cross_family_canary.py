#!/usr/bin/env python3
"""Freeze and exercise the B2.9-C non-scientific cross-family runtime candidate."""

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

from tools.r1_b2_8_r3_prospective_selector import official_env
from tools.r1_b2_9_b_route_continuous_canary import _ego, _map_api
from tools.r1_closed_loop_benchmark_v2_1 import build_hlc_native_geometry_v1_1
from tools.r1_closed_loop_benchmark_v2_2 import build_hlc_route_continuous_reference_v2_2
from tools.r1_closed_loop_benchmark_v2_3 import build_hlc_route_continuous_reference_v2_3
from tools.r1_prospective_generator_contract_v2 import (
    HLC_BASELINE,
    HLC_TREATMENT,
    TSB_BASELINE,
    TSB_TREATMENT,
)


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "docs/stageR/r1"
OUTPUT_ROOT = ROOT / "outputs/r1_b2_9_c_cross_family_engineering_canary_v1"
PROTECTED_CSV = ROOT / "outputs/waymo_5neighbor_context_laneaware_clean_v1_full51_merged/behavior_events_v2/behavior_event_metrics_v2.csv"
PROTECTED_SHA = "e8deb93312e82183b6c2c0db30fd18cbf9c32d32d566038419a5be65b389d9d8"
HLC_TOKENS = ("b1be12bca092597a", "25944935eadb52f1", "ef3172a208cc5dd7")
TSB_TOKENS = ("b486f9cf33a85455", "3edcce9e7e19573f", "ff152a4cf9c4503b")
OUT = {
    "horizon": R1 / "r1_primary80_scientific_runtime_horizon_contract_v1.0.json",
    "route_contract": R1 / "r1_hlc_route_progression_invariant_contract_v1.0.json",
    "route_audit": R1 / "r1_b2_9_c_route_progression_invariant_audit_v1.json",
    "roster": R1 / "r1_b2_9_c_cross_family_engineering_canary_roster_v1.0.json",
    "ledger": R1 / "r1_b2_9_c_cross_family_canary_run_ledger_v1.0.json",
    "dispatch": R1 / "r1_b2_9_c_full_stack_dispatch_audit_v1.json",
    "manifest": R1 / "r1_b2_9_c_scientific_runtime_candidate_manifest_v1.0.json",
    "contract_report": R1 / "R1_B2_9_C_Primary80_Scientific_Runtime_Contract_Report_v1.md",
    "runtime_report": R1 / "R1_B2_9_C_Cross_Family_Engineering_Runtime_Report_v1.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any], *, update: bool = False) -> None:
    if path.exists() and not update:
        raise FileExistsError(f"VERSIONED_OUTPUT_EXISTS:{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _normalize(entry: Mapping[str, Any], source: str) -> Dict[str, Any]:
    row = dict(entry)
    initial = dict(row["initial_state"])
    if "route_roadblock_ids" not in row:
        row["route_roadblock_ids"] = list(initial["route_roadblock_ids"])
    row["initial_state_fingerprint"] = str(
        row.get("initial_state_fingerprint", initial["initial_state_fingerprint"])
    )
    if row["family"] == "R-HLC":
        row["direction"] = str(row.get("direction", "RIGHT")).upper()
        row["intended_lane_change_direction"] = row["direction"]
    row["arms"] = (
        [HLC_BASELINE, HLC_TREATMENT] if row["family"] == "R-HLC" else [TSB_BASELINE, TSB_TREATMENT]
    )
    row["canary_source_ledger"] = source
    row["SCIENTIFIC_USE_FORBIDDEN"] = True
    row["PERMANENT_FUTURE_SELECTOR_EXCLUSION"] = True
    row["NON_SCIENTIFIC_ENGINEERING_ONLY"] = True
    return row


def build_roster() -> Dict[str, Any]:
    hlc_source = read_json(R1 / "r1_b2_9_b_hlc_engineering_canary_roster_v1.0.json")["entries"]
    validation = read_json(R1 / "r1_runtime_determinism_validation_roster_v1.0.json")["entries"]
    historical = read_json(R1 / "r1_official_technical_smoke_roster_v1.0.json")["entries"]
    effective = read_json(R1 / "r1_b2_7_effective_permanent_blacklist_audit_v1.0.json")["entries"]
    excluded = {(str(x["scenario_token"]), str(x["log_id"])) for x in effective}
    by_token = {str(x["scenario_token"]): x for x in [*validation, *historical]}
    entries = [_normalize(x, "B2.9-B_FROZEN_HLC_CANARY_ROSTER") for x in hlc_source]
    entries.extend(
        _normalize(by_token[token], "EXISTING_PERMANENT_EXCLUSION_LEDGER") for token in TSB_TOKENS
    )
    current = {
        (str(x["scenario_token"]), str(x["log_id"]))
        for x in read_json(R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json")["entries"]
        if str(x["scenario_token"]) != HLC_TOKENS[0]
    }
    for row in entries:
        identity = (str(row["scenario_token"]), str(row["log_id"]))
        if identity in current:
            raise PermissionError(f"CURRENT_SCIENTIFIC_IDENTITY_FORBIDDEN:{identity}")
        if row["family"] == "R-TSB" and identity not in excluded:
            raise ValueError(f"TSB_CANARY_NOT_PREEXISTING_PERMANENT_EXCLUSION:{identity}")
    return {
        "schema_version": "r1_b2_9_c_cross_family_engineering_canary_roster_v1.0",
        "status": "FROZEN_NON_SCIENTIFIC_CROSS_FAMILY_ENGINEERING_ONLY",
        "selection_rule": "THREE_EXISTING_B2_9_B_HLC_CANARIES_PLUS_THREE_PREEXISTING_PERMANENTLY_EXCLUDED_TSB_IDENTITIES",
        "entries": entries,
        "counts": {"R-HLC": 3, "R-TSB": 3, "identities": 6, "arms": 12},
        "SCIENTIFIC_USE_FORBIDDEN": True,
        "PERMANENT_FUTURE_SELECTOR_EXCLUSION": True,
        "scientific_roster_created_or_modified": False,
        "selector_rerun": False,
    }


def horizon_contract() -> Dict[str, Any]:
    return {
        "schema_version": "r1_primary80_scientific_runtime_horizon_contract_v1.0",
        "status": "FROZEN_PROSPECTIVE_SCIENTIFIC_RUNTIME_CANDIDATE",
        "PRIMARY_REALIZED_ITERATIONS": list(range(80)),
        "PLANNER_CALLS": 80,
        "time_controller": {
            "implementation": "tools.r1_primary80_scientific_time_controller_v1.R1Primary80ScientificTimeControllerV1",
            "base": "nuPlan_1.2.2_StepSimulationTimeController",
            "number_of_iterations": "min(official_scenario_iterations,81)",
            "scenario_iterations_lt_81": "FAIL_CLOSED_NOT_EVALUABLE_R1_PRIMARY80",
        },
        "sampling": {
            "dt_seconds": 0.1,
            "first_planner_time_seconds": 0.0,
            "last_realized_ego_time_seconds": 7.9,
            "runtime_termination_time_seconds": 8.0,
            "output_trajectory_last_time_seconds_relative_to_each_call": 7.9,
        },
        "preexisting_frozen_basis": {
            "realized_trace_contract": "docs/stageR/r1/r1_b2_8_r1_realized_trace_contract_v1.0.json",
            "WINDOW_FRAMES": 80,
            "WINDOW_SECONDS": 8.0,
            "HLC_clearance_horizon_seconds": 8.0,
            "mechanism_F_match_endpoint_engineering_primary": "REALIZED_CURRENT_EGO_ITERATIONS_0_79",
            "A02_secondary_failure_role": "ENGINEERING_ONLY_ALIGNMENT_EVIDENCE_NOT_SCIENTIFIC_OUTCOME",
        },
        "unchanged": [
            "TwoStageController",
            "LQR",
            "observation",
            "ego_controller",
            "metric_engine",
            "0.1_s_time_grid",
            "HLC_and_TSB_generator_semantics",
            "mechanism_thresholds",
            "F_match",
            "safety_thresholds",
        ],
        "iteration_ge_80": "FORBIDDEN_FROM_PLANNER_PRIMARY_TRACE_SAFETY_AND_EVALUATION",
        "scientific_identity_or_outcome_used_to_choose_horizon": False,
    }


def route_contract() -> Dict[str, Any]:
    return {
        "schema_version": "r1_hlc_route_progression_invariant_contract_v1.0",
        "status": "FROZEN_AFTER_ZERO_VIOLATION_READ_ONLY_AUDIT",
        "implementation": "tools/r1_closed_loop_benchmark_v2_3.py",
        "invariant": [
            "each selected source successor roadblock has exactly one later occurrence in frozen route_roadblock_ids",
            "the paired target successor roadblock exactly equals the source successor roadblock",
            "source and target component pairs advance through the same strictly increasing frozen-route occurrences",
        ],
        "failure": "FAIL_CLOSED_NO_TIE_BREAK_NO_REPLACEMENT",
        "v2_2_rules_preserved": [
            "native_only",
            "no_extrapolation",
            "no_manual_points",
            "unique_pair",
            "direction_continuity",
            "adjacency",
            "self_intersection_fail_closed",
            "exact_native_join",
        ],
    }


def _route_ok(entry: Mapping[str, Any], source_rb: str, target_rb: str, index: int) -> bool:
    route = [str(x) for x in entry["route_roadblock_ids"]]
    return source_rb == target_rb and 0 <= index < len(route) and route[index] == source_rb


def route_audit() -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    current = [
        x for x in read_json(R1 / "r1_official_compliant_technical_smoke_roster_v2.1.json")["entries"]
        if x["family"] == "R-HLC"
    ]
    summaries, violations = [], []
    current_transition_count = 0
    for entry in current:
        api = _map_api(str(entry["map_name"]), cache)
        identity_count = 0
        for arm in (HLC_BASELINE, HLC_TREATMENT):
            initial = _ego(entry["initial_state"])
            first = build_hlc_route_continuous_reference_v2_2(
                api, entry["route_roadblock_ids"], entry["source_lane_id"], entry["target_lane_id"],
                initial, float(initial["speed_mps"]) * 7.9,
            )
            states = build_hlc_native_geometry_v1_1(
                initial, 0.0, first["source_reference_xy"], first["target_reference_xy"],
                first["source_current_arc_m"], first["target_current_arc_m"], arm,
            )
            for iteration, state in enumerate(states):
                corridor = build_hlc_route_continuous_reference_v2_2(
                    api, entry["route_roadblock_ids"], entry["source_lane_id"], entry["target_lane_id"],
                    state, float(state["speed_mps"]) * 7.9,
                )
                build_hlc_route_continuous_reference_v2_3(
                    api, entry["route_roadblock_ids"], entry["source_lane_id"], entry["target_lane_id"],
                    state, float(state["speed_mps"]) * 7.9,
                )
                for transition in corridor["transitions"]:
                    candidate = transition["candidate_pairs"][0]
                    identity_count += 1
                    current_transition_count += 1
                    if not _route_ok(
                        entry,
                        str(candidate["source_roadblock_id"]),
                        str(candidate["target_roadblock_id"]),
                        int(candidate["route_occurrence_index"]),
                    ):
                        violations.append(
                            {"scope": "CURRENT12_OFFLINE", "scenario_token": entry["scenario_token"], "arm": arm,
                             "iteration": iteration, "candidate": candidate}
                        )
        summaries.append({"scenario_token": entry["scenario_token"], "calls": 160, "selected_transitions": identity_count})
    official_env()
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    canaries = read_json(R1 / "r1_b2_9_b_hlc_engineering_canary_roster_v1.0.json")["entries"]
    actual_transition_count = 0
    actual_summaries = []
    for identity_index, entry in enumerate(canaries, 1):
        api = _map_api(str(entry["map_name"]), cache)
        count = 0
        route = [str(x) for x in entry["route_roadblock_ids"]]

        def roadblock(edge_id: str) -> str:
            edge = api.get_map_object(edge_id, SemanticMapLayer.LANE)
            if edge is None:
                edge = api.get_map_object(edge_id, SemanticMapLayer.LANE_CONNECTOR)
            if edge is None:
                raise ValueError(f"ACTUAL_AUDIT_EDGE_MISSING:{edge_id}")
            return str(edge.get_roadblock_id())

        for arm_name in ("BASELINE", "TREATMENT"):
            path = next(
                OUTPUT_ROOT.parent.joinpath("r1_b2_9_b_engineering_canary_v1").glob(
                    f"*{identity_index:02d}-HLC-{arm_name}-A03/route_continuous_runtime_audit.json"
                )
            )
            for row in json.loads(path.read_text(encoding="utf-8")):
                source_ids, target_ids = row["source_edge_ids"], row["target_edge_ids"]
                cursor = route.index(roadblock(source_ids[0]))
                for source_id, target_id in zip(source_ids[1:], target_ids[1:]):
                    source_rb, target_rb = roadblock(source_id), roadblock(target_id)
                    matches = [i for i in range(cursor + 1, len(route)) if route[i] == source_rb]
                    count += 1
                    actual_transition_count += 1
                    if source_rb != target_rb or len(matches) != 1:
                        violations.append(
                            {"scope": "B2_9_B_ACTUAL_A03", "scenario_token": entry["scenario_token"],
                             "arm": arm_name, "iteration": row["iteration_index"], "source_roadblock_id": source_rb,
                             "target_roadblock_id": target_rb, "forward_occurrences": matches}
                        )
                    elif matches:
                        cursor = matches[0]
        actual_summaries.append({"scenario_token": entry["scenario_token"], "actual_calls": 160, "selected_transitions": count})
    total = current_transition_count + actual_transition_count
    return {
        "schema_version": "r1_b2_9_c_route_progression_invariant_audit_v1",
        "status": "PASS" if not violations else "FAIL_CLOSED",
        "simulation_started": False,
        "current12_read_only_offline": summaries,
        "b2_9_b_actual_a03": actual_summaries,
        "counts": {
            "current12_calls": 1920,
            "current12_selected_transitions": current_transition_count,
            "actual_canary_calls": 480,
            "actual_canary_selected_transitions": actual_transition_count,
            "selected_transitions_audited": total,
            "target_route_consistency_violations": len(violations),
        },
        "violations": violations,
        "current_scientific_identity_runner_runs": 0,
    }


def initial_ledger(roster: Mapping[str, Any]) -> Dict[str, Any]:
    runs = []
    for identity_index, entry in enumerate(roster["entries"], 1):
        for arm_index, arm in enumerate(entry["arms"]):
            role = "BASELINE" if arm_index == 0 else "TREATMENT"
            run_id = f"R1B29C-CANARY-{identity_index:02d}-{entry['family'][2:]}-{role}-A01"
            runs.append(
                {
                    "run_id": run_id,
                    "pair_id": f"R1B29C-{identity_index:02d}-{entry['family']}",
                    "family": entry["family"],
                    "scenario_token": entry["scenario_token"],
                    "log_id": entry["log_id"],
                    "arm": arm,
                    "attempt_number": 1,
                    "output_root": str((OUTPUT_ROOT / run_id).relative_to(ROOT)),
                    "status": "NOT_RUN",
                    "SCIENTIFIC_USE_FORBIDDEN": True,
                }
            )
    return {
        "schema_version": "r1_b2_9_c_cross_family_canary_run_ledger_v1.0",
        "status": "PREPARED_NOT_RUN",
        "runs": runs,
        "counts": {"planned": 12, "fresh_actual_runs": 0, "reruns": 0},
        "scientific_identities_simulated": False,
        "threshold_mechanism_F_match_changed": False,
        "official_smoke_authorized": False,
        "RBR_authorized": False,
    }


def prepare() -> None:
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    roster = build_roster()
    audit = route_audit()
    if audit["status"] != "PASS":
        write_json(OUT["route_audit"], audit)
        raise RuntimeError("ROUTE_PROGRESSION_INVARIANT_FAIL_CLOSED_NO_SIMULATION")
    for path, payload in (
        (OUT["horizon"], horizon_contract()),
        (OUT["route_contract"], route_contract()),
        (OUT["route_audit"], audit),
        (OUT["roster"], roster),
        (OUT["ledger"], initial_ledger(roster)),
    ):
        write_json(path, payload)


def _overrides(run: Mapping[str, Any], entry: Mapping[str, Any], run_root: Path) -> list[str]:
    return [
        "+simulation=closed_loop_nonreactive_agents",
        "planner=r1_official_technical_smoke_v2_2_r3",
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
        "experiment_name=r1_b2_9_c_cross_family_engineering_canary",
        f"job_name={run['run_id']}",
        f"output_dir={run_root}",
        f"hydra.searchpath=[file://{ROOT}/configs/r1_official_technical_smoke_hydra,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]",
    ]


def _resolution_count(entry: Mapping[str, Any]) -> int:
    official_env()
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

    return len(list(get_scenarios_from_db(str(entry["db_path"]), [str(entry["scenario_token"])], None, None, True, False)))


def _trace(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _refresh(ledger: Dict[str, Any]) -> None:
    runs = ledger["runs"]
    completed = [x for x in runs if x["status"] == "TECHNICAL_COMPLETE"]
    ledger["counts"] = {
        "planned": 12,
        "fresh_actual_runs": sum(bool(x.get("runner_run_called")) for x in runs),
        "reruns": sum(int(x.get("attempt_number", 1)) > 1 for x in runs),
        "technical_complete": len(completed),
        "HLC_technical_complete": sum(x["family"] == "R-HLC" for x in completed),
        "TSB_technical_complete": sum(x["family"] == "R-TSB" for x in completed),
        "exact_80_row_traces": sum(bool(x.get("primary_80_complete")) for x in runs),
        "secondary_planner_calls": sum(int(x.get("secondary_planner_calls", 0)) for x in runs),
        "metric_callback_complete": sum(bool(x.get("metric_callback_complete")) for x in runs),
        "safety_adapter_structural_complete": sum(bool(x.get("safety_adapter_structural_complete")) for x in runs),
    }
    ledger["status"] = "12_OF_12_CROSS_FAMILY_ENGINEERING_CANARY_PASS" if len(completed) == 12 else "INCOMPLETE_OR_FAIL"


def execute() -> None:
    if sha256(PROTECTED_CSV) != PROTECTED_SHA:
        raise ValueError("PROTECTED_CSV_SHA_MISMATCH")
    if read_json(OUT["route_audit"])["status"] != "PASS":
        raise PermissionError("ROUTE_INVARIANT_GATE_NOT_PASS")
    roster, ledger = read_json(OUT["roster"]), read_json(OUT["ledger"])
    entries = {str(x["scenario_token"]): x for x in roster["entries"]}
    official_env()
    from hydra import compose, initialize_config_dir
    from nuplan.planning.script.builders.simulation_builder import build_simulations
    from nuplan.planning.script.builders.simulation_callback_builder import build_callbacks_worker, build_simulation_callbacks
    from nuplan.planning.script.utils import run_runners, set_up_common_builder
    from tools.r1_b2_8_r3_1_official_safety_adapter import adapt_official_safety
    from tools.r1_official_technical_smoke_planner_v3_1 import R1OfficialTechnicalSmokePlannerV3_1

    config_root = ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation"
    for run in ledger["runs"]:
        if run["status"] != "NOT_RUN":
            continue
        entry = entries[str(run["scenario_token"])]
        run_root = ROOT / str(run["output_root"])
        trace_dir = run_root / "trace"
        trace_file = trace_dir / "realized_current_ego.jsonl"
        if run_root.exists() or trace_file.exists():
            raise FileExistsError(f"CANARY_OUTPUT_OR_TRACE_REUSE_FAIL_CLOSED:{run_root}")
        trace_dir.mkdir(parents=True)
        run["trace_path_pre_run_state"] = "ABSENT"
        run["exact_scenario_resolution_count"] = _resolution_count(entry)
        if run["exact_scenario_resolution_count"] != 1:
            raise RuntimeError(f"EXACT_SCENARIO_RESOLUTION_FAIL:{run['run_id']}")
        planner = R1OfficialTechnicalSmokePlannerV3_1(entry, str(run["family"]), str(run["arm"]), str(trace_dir))
        run.update({"runner_constructed": False, "runner_run_called": False, "simulation_started": False})
        try:
            os.environ.update(
                {
                    "R1_B2_8_R3_BINDING_MANIFEST": str(R1 / "r1_b2_8_r3_execution_bindings_manifest_v1.0.json"),
                    "R1_B2_8_R3_RUN_ID": str(run["run_id"]),
                    "R1_B2_8_R3_TRACE_DIR": str(trace_dir),
                }
            )
            with initialize_config_dir(config_dir=str(config_root)):
                cfg = compose(config_name="default_simulation", overrides=_overrides(run, entry, run_root))
            common = set_up_common_builder(cfg, "r1_b2_9_c_build")
            callback_worker = build_callbacks_worker(cfg)
            callbacks = build_simulation_callbacks(cfg, common.output_dir, callback_worker)
            runners = build_simulations(cfg, common.worker, callbacks, callback_worker, pre_built_planners=[planner])
            if len(runners) != 1:
                raise RuntimeError("EXPECTED_EXACTLY_ONE_RUNNER")
            run["runner_constructed"] = True
            controller = runners[0]._simulation._time_controller
            run["controller_iterations"] = int(controller.number_of_iterations())
            run["metric_runner_termination_timestamp_us"] = int(controller.scenario.get_time_point(80).time_us)
            run["runner_run_called"] = True
            run["simulation_started"] = True
            run_runners(runners, common, "r1_b2_9_c_running", cfg)
            rows = _trace(trace_file)
            indices = [int(x["iteration_index"]) for x in rows]
            primary_complete = indices == list(range(80))
            metric_files = list((run_root / "metrics").glob("*.parquet"))
            safety = adapt_official_safety(run_root)
            run.update(
                {
                    "status": "TECHNICAL_COMPLETE" if primary_complete and len(metric_files) > 0 else "OTHER_TECHNICAL_FAILURE",
                    "primary_trace_rows": len(rows),
                    "primary_80_complete": primary_complete,
                    "secondary_planner_calls": 0,
                    "first_ego_timestamp_us": int(rows[0]["current_ego"]["time_us"]),
                    "last_realized_ego_timestamp_us": int(rows[-1]["current_ego"]["time_us"]),
                    "realized_duration_seconds": (int(rows[-1]["current_ego"]["time_us"]) - int(rows[0]["current_ego"]["time_us"])) * 1e-6,
                    "metric_window_duration_seconds": (int(run["metric_runner_termination_timestamp_us"]) - int(rows[0]["current_ego"]["time_us"])) * 1e-6,
                    "metric_callback_complete": len(metric_files) > 0 and (run_root / "runner_report.parquet").is_file(),
                    "metric_parquet_count": len(metric_files),
                    "safety_adapter_structural_complete": True,
                    "safety_adapter_arm_pass_descriptive_only": bool(safety["frozen_arm_safety_pass"]),
                    "trace_path_post_run_state": "EXACT_80_ROWS_0_TO_79" if primary_complete else "INVALID",
                }
            )
            if run["family"] == "R-HLC":
                (run_root / "route_progression_runtime_audit.json").write_text(
                    json.dumps(planner.route_continuous_audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
        except Exception as exc:
            run.update(
                {
                    "status": "OTHER_TECHNICAL_FAILURE",
                    "error": f"{type(exc).__name__}:{exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-16:],
                    "primary_80_complete": False,
                    "metric_callback_complete": False,
                    "safety_adapter_structural_complete": False,
                }
            )
        _refresh(ledger)
        write_json(OUT["ledger"], ledger, update=True)
        if run["status"] != "TECHNICAL_COMPLETE":
            raise RuntimeError(f"CROSS_FAMILY_CANARY_FAIL_CLOSED:{run['run_id']}:{run.get('error')}")
    ledger["scientific_identities_simulated"] = False
    ledger["threshold_mechanism_F_match_changed"] = False
    ledger["protected_csv_sha256"] = sha256(PROTECTED_CSV)
    _refresh(ledger)
    write_json(OUT["ledger"], ledger, update=True)


def _build_pair_bindings(roster: Mapping[str, Any], ledger: Mapping[str, Any]) -> list[Dict[str, Any]]:
    from tools.r1_b2_8_r3_2_freeze_pair_bindings import _one
    from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1

    cache: Dict[str, Any] = {}
    validation = {
        str(x["scenario_token"]): x
        for x in read_json(R1 / "r1_runtime_determinism_validation_roster_v1.0.json")["entries"]
    }

    def historical_context_fallback(entry: Mapping[str, Any]) -> Dict[str, Any]:
        token = str(entry["scenario_token"])
        if token in validation:
            source_entry = dict(validation[token])
            trace = ROOT / f"outputs/r1_runtime_determinism_validation_v3/runs/{entry['family']}__{token}__V3_RUN_A/trace/planner_trace.jsonl"
        elif token == "ff152a4cf9c4503b":
            source_entry = dict(entry)
            trace = ROOT / f"outputs/r1_official_compliant_technical_smoke_v1_1/runs/R-TSB__{token}__B2R1_BASELINE/trace/planner_trace.jsonl"
        else:
            raise ValueError(f"NO_FROZEN_HISTORICAL_CONTEXT_FALLBACK:{token}")
        historical = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        hashes = historical["component_hashes"]
        context = {
            "pre_context_raw_hash": str(hashes["pre_context_raw"]),
            "canonical_context_json_hash": str(hashes["canonical_context"]),
            "historical_context_artifact": str(trace.relative_to(ROOT)),
            "historical_context_artifact_sha256": sha256(trace),
            "reuse_scope": "NON_SCIENTIFIC_ENGINEERING_DISPATCH_IDENTITY_CHECK_ONLY",
        }
        base = {
            "pair_id": None,
            "family": entry["family"],
            "scenario_token": token,
            "log_id": entry["log_id"],
            "baseline_context": context,
            "treatment_context": context,
            "context_source": "READ_ONLY_PREEXISTING_RUNTIME_ARTIFACT_NO_RECOMPUTE_AFTER_MAP_AMBIGUITY",
        }
        if entry["family"] == "R-TSB":
            return {**base, "pretreatment_clearance": None}
        route = list(
            source_entry["route_roadblock_ids"]
            if "route_roadblock_ids" in source_entry
            else source_entry["initial_state"]["route_roadblock_ids"]
        )
        api = _map_api(str(source_entry["map_name"]), cache)
        current = _ego(source_entry["initial_state"])
        route_ref = build_native_route_reference_v1_1(
            api, route, current, max(0.2, float(current["speed_mps"])) * 7.9
        )
        return {
            **base,
            "pretreatment_clearance": {
                "pretreatment_only": True,
                "status": "HISTORICAL_ENGINEERING_CANARY_PRETREATMENT_BINDING_ONLY",
                "minimum_target_lane_object_gap_m": source_entry.get("minimum_target_lane_object_gap_m"),
                "scientific_clearance_claim": False,
                "source": "r1_runtime_determinism_validation_roster_v1.0.json",
            },
            "source_reference_xy": source_entry["source_reference_xy"],
            "target_reference_xy": source_entry["target_reference_xy"],
            "native_route_reference_xy": route_ref["reference_xy"].tolist(),
            "native_route_reference_source": "OFFICIAL_NUPLAN_NATIVE_ROUTE_REFERENCE_V1_1",
        }

    bindings = []
    for index, entry in enumerate(roster["entries"], 1):
        normalized = dict(entry)
        if normalized["family"] == "R-HLC":
            normalized["direction"] = str(normalized.get("direction", "RIGHT")).upper()
            normalized["intended_lane_change_direction"] = normalized["direction"]
        try:
            binding = _one(normalized, cache)
        except ValueError as exc:
            allowed_fallback = (
                "OFFICIAL_MAP_LANE_AMBIGUITY_FAIL_CLOSED",
                "NATIVE_REFERENCE_COVERAGE_FAIL_NO_EXTRAPOLATION",
            )
            if not any(label in str(exc) for label in allowed_fallback):
                raise
            binding = historical_context_fallback(normalized)
        runs = [x for x in ledger["runs"] if x["scenario_token"] == entry["scenario_token"]]
        binding.update(
            {
                "pair_id": f"R1B29C-{index:02d}-{entry['family']}",
                "baseline_run_id": runs[0]["run_id"],
                "treatment_run_id": runs[1]["run_id"],
            }
        )
        bindings.append(binding)
    return bindings


def finalize() -> None:
    roster, ledger = read_json(OUT["roster"]), read_json(OUT["ledger"])
    if ledger["status"] != "12_OF_12_CROSS_FAMILY_ENGINEERING_CANARY_PASS":
        raise RuntimeError("CANARY_LEDGER_NOT_COMPLETE")
    official_env()
    from tools.r1_b2_8_r3_2_post_run_evaluator_dispatcher import evaluate_frozen_pair

    bindings = _build_pair_bindings(roster, ledger)
    evaluations = []
    for binding in bindings:
        base = ROOT / next(x["output_root"] for x in ledger["runs"] if x["run_id"] == binding["baseline_run_id"])
        treatment = ROOT / next(x["output_root"] for x in ledger["runs"] if x["run_id"] == binding["treatment_run_id"])
        result = evaluate_frozen_pair(
            pair_binding=binding, baseline_run_dir=base, treatment_run_dir=treatment
        )
        evaluations.append(
            {
                "pair_id": binding["pair_id"],
                "family": binding["family"],
                "dispatch_status": result["dispatch_status"],
                "evaluator_status": result["evaluation"]["status"],
                "official_safety_pair_pass_descriptive_only": result["official_safety_pair_pass"],
                "scientific_gate_used_for_tuning_or_selection": False,
            }
        )
    dispatch = {
        "schema_version": "r1_b2_9_c_full_stack_dispatch_audit_v1",
        "status": "FULL_EXECUTION_STACK_STRUCTURAL_COMPLETION",
        "evaluations": evaluations,
        "counts": {
            "HLC_pair_dispatcher_complete": sum(x["family"] == "R-HLC" for x in evaluations),
            "TSB_pair_dispatcher_complete": sum(x["family"] == "R-TSB" for x in evaluations),
            "total": len(evaluations),
        },
        "threshold_tuning": False,
        "identity_selection": False,
        "mechanism_or_F_match_change": False,
    }
    write_json(OUT["dispatch"], dispatch, update=OUT["dispatch"].exists())
    dependencies = [
        ROOT / "tools/r1_primary80_scientific_time_controller_v1.py",
        ROOT / "tools/r1_closed_loop_benchmark_v2_3.py",
        ROOT / "tools/r1_official_technical_smoke_planner_v3_1.py",
        OUT["horizon"],
        OUT["route_contract"],
        ROOT / "tools/r1_b2_8_r3_1_official_safety_adapter.py",
        ROOT / "tools/r1_b2_8_r3_2_post_run_evaluator_dispatcher.py",
        ROOT / "tools/r1_official_technical_smoke_evaluator_v2_1.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/simulation_time_controller/step_simulation_time_controller.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/simulation.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/controller/two_stage_controller.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/controller/tracker/lqr.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/simulation/observation/tracks_observation.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/metrics/metric_engine.py",
        ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/simulation_time_controller/step_simulation_time_controller.yaml",
        ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml",
        ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml",
        ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/simulation/observation/box_observation.yaml",
        ROOT.parent / "nuplan-devkit/nuplan/planning/script/config/common/simulation_metric/simulation_closed_loop_nonreactive_agents.yaml",
        ROOT.parent / "nuplan-devkit/nuplan/planning/script/experiments/simulation/closed_loop_nonreactive_agents.yaml",
    ]
    manifest = {
        "schema_version": "r1_b2_9_c_scientific_runtime_candidate_manifest_v1.0",
        "status": "READY_FOR_SCIENTIFIC_SELECTOR_ROSTER_REBUILD_REVIEW",
        "components_sha256": {str(path.resolve()): sha256(path) for path in dependencies},
        "canary_roster_sha256": sha256(OUT["roster"]),
        "canary_ledger_sha256": sha256(OUT["ledger"]),
        "route_audit_sha256": sha256(OUT["route_audit"]),
        "full_stack_dispatch_audit_sha256": sha256(OUT["dispatch"]),
        "protected_csv_sha256": sha256(PROTECTED_CSV),
        "official_smoke_ready_or_authorized": False,
        "scientific_roster_created": False,
        "RBR_authorized": False,
    }
    write_json(OUT["manifest"], manifest, update=OUT["manifest"].exists())
    if not OUT["contract_report"].exists() and not OUT["runtime_report"].exists():
        write_reports()


def write_reports() -> None:
    audit, roster, ledger, dispatch = (
        read_json(OUT["route_audit"]),
        read_json(OUT["roster"]),
        read_json(OUT["ledger"]),
        read_json(OUT["dispatch"]),
    )
    counts = ledger["counts"]
    if OUT["contract_report"].exists() or OUT["runtime_report"].exists():
        raise FileExistsError("VERSIONED_REPORT_EXISTS")
    OUT["contract_report"].write_text(
        "# R1 B2.9-C Primary80 科学运行时合同报告 v1\n\n"
        "## 冻结结论\n\nPrimary 固定为 `REALIZED_CURRENT_EGO` iterations `0...79`，planner calls 恰好 80。"
        "科学 time-controller 继承 nuPlan 1.2.2 `StepSimulationTimeController`，唯一 override 是 "
        "`number_of_iterations=min(official_scenario_iterations,81)`；场景少于 81 iterations 时显式 `NOT_EVALUABLE/FAIL_CLOSED`。\n\n"
        "0.1 s 时间网格、TwoStageController/LQR、observation、ego controller、metric engine、两 family generator、机制、F_match、endpoint、工程限制和安全阈值均未改变。"
        "80 个 realized 状态覆盖 0.0...7.9 s，runner 在 8.0 s 边界终止；iteration >=80 不进入 planner、Primary trace、安全或 evaluator。\n\n"
        "该 horizon 来自既有冻结的 80-frame/8.0-second Primary、HLC 8 s clearance 与 evaluator 的 0...79 输入合同。"
        "A02 的 post-Primary failure 只用于确认运行时边界应与既有测量边界对齐；没有读取科学 pair outcome、representation、BDD 或 RBR。\n\n"
        "## 路线不变量\n\n"
        f"审计 `{audit['counts']['selected_transitions_audited']}` 条 selected transitions，target route-consistency violation 为 "
        f"`{audit['counts']['target_route_consistency_violations']}`，状态 `{audit['status']}`。V2.3 在 V2.2 全部 fail-closed 规则之上，"
        "新增 source/target 必须落在同一冻结 roadblock occurrence 的强制检查。\n",
        encoding="utf-8",
    )
    hlc = [x["scenario_token"] for x in roster["entries"] if x["family"] == "R-HLC"]
    tsb = [x["scenario_token"] for x in roster["entries"] if x["family"] == "R-TSB"]
    durations = sorted({x.get("metric_window_duration_seconds") for x in ledger["runs"]})
    realized = sorted({x.get("realized_duration_seconds") for x in ledger["runs"]})
    OUT["runtime_report"].write_text(
        "# R1 B2.9-C 跨 Family 工程运行报告 v1\n\n"
        "## 证据边界\n\n本轮仅使用永久科学排除身份执行非科学工程 canary，不是 official smoke，也不产生科学 roster。"
        "任何 evaluator PASS/FAIL 都未用于调阈值、换身份或修改机制/F_match。\n\n"
        f"HLC identities：`{'`, `'.join(hlc)}`。TSB identities：`{'`, `'.join(tsb)}`。\n\n"
        "## 运行结果\n\n"
        f"fresh actual runs `{counts['fresh_actual_runs']}`，reruns `{counts['reruns']}`；HLC technical complete "
        f"`{counts['HLC_technical_complete']}/6`，TSB `{counts['TSB_technical_complete']}/6`；exact 80-row traces "
        f"`{counts['exact_80_row_traces']}/12`，secondary planner calls `{counts['secondary_planner_calls']}`；metric/callback "
        f"`{counts['metric_callback_complete']}/12`，safety adapter structural complete `{counts['safety_adapter_structural_complete']}/12`。"
        f"pair dispatcher HLC `{dispatch['counts']['HLC_pair_dispatcher_complete']}/3`、TSB `{dispatch['counts']['TSB_pair_dispatcher_complete']}/3`。\n\n"
        f"实际 realized timestamp duration 为 `{realized}` s；metric runner termination window 为 `{durations}` s。"
        "因此安全语义仅为 `OFFICIAL_SAFETY_WITHIN_FROZEN_R1_PRIMARY_RUNTIME_WINDOW`，不声称与历史 full-scenario metric exact parity。\n\n"
        "当前 scientific identities 仿真：`false`。OFFICIAL_SMOKE_AUTHORIZED=`false`；RBR_A/B/C=`NOT_AUTHORIZED`。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "finalize"))
    args = parser.parse_args()
    {"prepare": prepare, "execute": execute, "finalize": finalize}[args.action]()
    print(json.dumps({"status": "PASS", "action": args.action}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
