#!/usr/bin/env python3
"""Zero-rollout construction preflight for the frozen future V2 execution path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

from tools.r1_closed_loop_benchmark_v2_1 import build_native_route_reference_v1_1
from tools.r1_hlc_dynamic_clearance_v1_1 import evaluate_r1_hlc_dynamic_clearance_v1_1
from tools.r1_official_ego_vehicle_binding_v1 import official_ego_vehicle_binding_v1
from tools.r1_official_metric_canonicalizer import canonicalize_official_metrics
from tools.r1_official_technical_smoke_evaluator_v2 import R1OfficialTechnicalSmokeEvaluatorV2
from tools.r1_official_technical_smoke_planner_v2 import R1OfficialTechnicalSmokePlannerV2


FORBIDDEN_SELECTOR_FIELDS = {"outcome", "planner_outcome", "mechanism_outcome", "safety_outcome", "representation", "bdd", "probe", "checkpoint", "rbr"}


def validate_future_roster_row_schema(row: Mapping[str, Any]) -> None:
    required = {"scenario_token", "log_id", "family", "map_name", "route_roadblock_ids", "route_fingerprint", "initial_state_fingerprint"}
    missing = sorted(required - set(row))
    if missing or row.get("family") not in {"R-HLC", "R-TSB"} or not row.get("route_roadblock_ids"):
        raise ValueError(f"FUTURE_ROSTER_ROW_SCHEMA_INVALID:{missing}")


def validate_selector_inputs_outcome_blind(payload: Mapping[str, Any]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                lowered = str(key).lower()
                if lowered in FORBIDDEN_SELECTOR_FIELDS or lowered.endswith("_outcome"):
                    raise ValueError(f"SELECTOR_OUTCOME_FIELD_FORBIDDEN:{key}")
                walk(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                walk(nested)
    walk(payload)


def launch_official_simulation(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("B2_5_SIMULATION_LAUNCH_HARD_BLOCK")


def run_zero_rollout_preflight(*, future_roster_row: Mapping[str, Any], smoke_arm: str, map_api: Any, context_frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    validate_future_roster_row_schema(future_roster_row)
    validate_selector_inputs_outcome_blind(future_roster_row)
    planner = R1OfficialTechnicalSmokePlannerV2(future_roster_row, str(future_roster_row["family"]), smoke_arm)
    planner.initialize(SimpleNamespace(route_roadblock_ids=list(future_roster_row["route_roadblock_ids"]), map_api=map_api))
    context = planner.build_context_v2_1(context_frames)
    anchor = context_frames[10]["ego"]
    current = {"rear_axle": {"x": float(anchor["x"]), "y": float(anchor["y"]), "heading": float(anchor["heading"])}, "speed_mps": float(anchor["speed_mps"]), "time_us": int(context_frames[10]["time_us"])}
    route = build_native_route_reference_v1_1(map_api, future_roster_row["route_roadblock_ids"], current, max(0.2, current["speed_mps"]) * 7.9)
    footprint = official_ego_vehicle_binding_v1()
    x = [float(anchor["x"]) + index * 0.1 for index in range(80)]
    y = [float(anchor["y"])] * 80
    clearance = evaluate_r1_hlc_dynamic_clearance_v1_1(baseline_xy=list(zip(x, y)), treatment_xy=list(zip(x, y)), official_runtime_vehicle_parameters=footprint, original_replay_tracks={}, official_replay_observation_timestamps_s=[index * 0.1 for index in range(80)])
    evaluator = R1OfficialTechnicalSmokeEvaluatorV2()
    return {"status": "READY_TO_LAUNCH_OFFICIAL_SIMULATION", "simulation_launched": False, "launch_guard": "B2_5_SIMULATION_LAUNCH_HARD_BLOCK", "object_construction": {"planner": planner.__class__.__name__, "evaluator": evaluator.__class__.__name__, "metric_canonicalizer": f"{canonicalize_official_metrics.__module__}.{canonicalize_official_metrics.__name__}"}, "config_loading": "R1_OFFICIAL_TECHNICAL_SMOKE_V2", "map_binding": "OFFICIAL_MAP_BRIDGE_V2_1", "context_bridge_status": context["stage5d_slot_semantics"], "route_builder_status": route["builder_version"], "generator_binding": "FROZEN_HLC_OPTION_B_AND_TSB_OPTION_A", "clearance_eligibility": clearance["status"], "ledger": {"actual_candidates_enumerated": 0, "actual_roster_selected": False, "new_runs": 0}, "budget": {"authorized_new_runs": 0, "consumed_new_runs": 0}, "official_ego_footprint": footprint, "forbidden_historical_planner_dependency": True}


__all__ = ["launch_official_simulation", "run_zero_rollout_preflight", "validate_future_roster_row_schema", "validate_selector_inputs_outcome_blind"]
